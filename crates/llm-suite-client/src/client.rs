use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::stream::MessageStream;
use llm_suite_codec::error::{LlmError, LlmResult};
use llm_suite_codec::LlmCodec;
use llm_suite_common::exec::{execute_with_timeout, TimeoutError};
use llm_suite_common::parse_retry_after_ms;
use llm_suite_common::retry::{RetryAttemptDescriptor, RetryPolicy};
use llm_suite_types::llm::{LlmProfile, LlmRequest, LlmResult as LlmResponseType};
use reqwest::Client as ReqwestClient;
use tokio_util::sync::CancellationToken;

pub trait LlmClient: Send + Sync {
    fn generate(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> impl Future<Output = LlmResult<LlmResponseType>> + Send;
    fn generate_stream(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> impl Future<Output = LlmResult<Box<dyn MessageStream>>> + Send;
    fn count_tokens(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> impl Future<Output = LlmResult<llm_suite_types::llm::TokenCountResult>> + Send;
}

pub struct LlmClientImpl {
    client: ReqwestClient,
    codec: Arc<dyn LlmCodec>,
    profile: LlmProfile,
    /// Notified whenever a retry attempt is scheduled. The transport stays
    /// leaf: the observer is a plain callback the gateway wires to its own
    /// metrics sink.
    retry_observer: Option<RetryObserver>,
    /// Optional per-request fetch hook (Phase 4): rewrites the codec-built
    /// `reqwest::Request` before it is sent. A plain function pointer, so the
    /// hot path pays one branch and no dynamic dispatch. Runs after
    /// `extra_headers` so it observes the final headers; intended for
    /// tests and middleware that cannot be expressed as headers.
    fetch_hook: Option<FetchHook>,
}

/// Rewrites a codec-built request before it is sent (custom fetch escape
/// hatch). Plain function pointer: `Copy`, no dynamic dispatch.
pub type FetchHook = fn(reqwest::Request) -> reqwest::Request;

/// Callback invoked with the descriptor of every scheduled retry attempt.
pub type RetryObserver = Arc<dyn Fn(&RetryAttemptDescriptor) + Send + Sync>;

impl LlmClientImpl {
    pub fn new(client: ReqwestClient, codec: Arc<dyn LlmCodec>, profile: LlmProfile) -> Self {
        Self {
            client,
            codec,
            profile,
            retry_observer: None,
            fetch_hook: None,
        }
    }

    /// Attach an observer notified on every scheduled retry attempt.
    pub fn with_retry_observer(mut self, observer: RetryObserver) -> Self {
        self.retry_observer = Some(observer);
        self
    }

    /// Attach a fetch hook rewriting the codec-built request before send.
    pub fn with_fetch_hook(mut self, hook: FetchHook) -> Self {
        self.fetch_hook = Some(hook);
        self
    }

    /// Notify the observer about a scheduled retry attempt.
    fn notify_retry(&self, descriptor: &RetryAttemptDescriptor) {
        if let Some(observer) = &self.retry_observer {
            observer(descriptor);
        }
    }

    pub fn profile(&self) -> &LlmProfile {
        &self.profile
    }

    pub(crate) fn build_timeout(&self) -> Duration {
        Duration::from_secs(self.profile.timeout.unwrap_or(60))
    }

    /// Retry policy for one request: per-request overrides win over the
    /// profile defaults for that call only. A request without overrides
    /// behaves exactly like the legacy profile-only policy.
    fn retry_policy_for(&self, request: &LlmRequest) -> RetryPolicy {
        RetryPolicy {
            max_retries: request.max_retries.unwrap_or_else(|| self.max_retries()),
            base_delay_ms: request
                .retry_delay_ms
                .unwrap_or_else(|| self.profile.retry_delay.unwrap_or(1000)),
            exponential_backoff: true,
        }
    }

    /// Merge per-request `extra_headers` into a codec-built request, then
    /// apply the fetch hook. Ordering: codec auth/profile/identity headers
    /// first, request headers second (winning per name), fetch hook last.
    /// Invalid header names/values are skipped with a warning so one bad
    /// override cannot fail the whole request at build time.
    fn finalize_request(
        &self,
        mut http_request: reqwest::Request,
        request: &LlmRequest,
    ) -> reqwest::Request {
        if let Some(extra) = request.extra_headers.as_ref() {
            for (name, value) in extra {
                let rendered = value.as_str().unwrap_or("");
                match (
                    reqwest::header::HeaderName::from_bytes(name.as_bytes()),
                    reqwest::header::HeaderValue::from_str(rendered),
                ) {
                    (Ok(header_name), Ok(header_value)) => {
                        http_request.headers_mut().insert(header_name, header_value);
                    }
                    _ => {
                        tracing::warn!(
                            header_name = name.as_str(),
                            "dropping invalid per-request extra header"
                        );
                    }
                }
            }
        }
        if let Some(hook) = self.fetch_hook {
            hook(http_request)
        } else {
            http_request
        }
    }

    fn max_retries(&self) -> u32 {
        self.profile.max_retries.unwrap_or(3)
    }

    fn map_timeout_result<T>(
        result: Result<T, TimeoutError<reqwest::Error>>,
        timeout_ms: u64,
    ) -> LlmResult<T> {
        match result {
            Ok(v) => Ok(v),
            Err(TimeoutError::TimedOut(_)) => Err(LlmError::Timeout(timeout_ms)),
            Err(TimeoutError::Failed(e)) => {
                if e.is_timeout() {
                    Err(LlmError::Timeout(timeout_ms))
                } else {
                    Err(LlmError::HttpError(e))
                }
            }
        }
    }

    pub(crate) fn map_http_error(
        status: reqwest::StatusCode,
        body: &str,
        timeout_ms: u64,
        retry_after: Option<&str>,
    ) -> LlmError {
        if status.is_success() {
            return LlmError::InvalidResponse(format!(
                "Unexpected success status with body: {}",
                body
            ));
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return LlmError::RateLimited {
                retry_after_ms: parse_retry_after_ms(retry_after),
            };
        }
        let msg = format!("HTTP {}: {}", status.as_u16(), body);
        // Safety-net classification: a context-window rejection of the actual
        // request is surfaced as ContextLengthExceeded so callers can force a
        // compression event when the local estimate undercounted.
        let provisional = match status.as_u16() {
            401 | 403 => LlmError::AuthError(msg),
            408 | 504 => LlmError::Timeout(timeout_ms),
            _ => LlmError::ProviderError {
                status: Some(status.as_u16()),
                message: body.to_string(),
                retry_after_ms: None,
            },
        };
        if provisional.is_context_length_exceeded() {
            return LlmError::ContextLengthExceeded(body.to_string());
        }
        provisional
    }
}

/// Reads the `Retry-After` response header as an owned value so the body can
/// still be consumed afterwards.
fn retry_after_header(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

impl LlmClientImpl {
    async fn generate_inner(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> LlmResult<LlmResponseType> {
        let start = Instant::now();

        let http_request =
            self.finalize_request(self.codec.build_request(request, &self.profile)?, request);

        let timeout_dur = self.build_timeout();
        let timeout_ms = timeout_dur.as_millis() as u64;

        let response = if let Some(ref cancel) = cancel {
            let fut = execute_with_timeout(self.client.execute(http_request), Some(timeout_ms));
            tokio::select! {
                result = fut => Self::map_timeout_result(result, timeout_ms)?,
                _ = cancel.cancelled() => return Err(LlmError::Cancelled),
            }
        } else {
            Self::map_timeout_result(
                execute_with_timeout(self.client.execute(http_request), Some(timeout_ms)).await,
                timeout_ms,
            )?
        };

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = retry_after_header(&response);
            let body = response.text().await.unwrap_or_default();
            return Err(Self::map_http_error(
                status,
                &body,
                timeout_ms,
                retry_after.as_deref(),
            ));
        }

        let body = response.text().await?;
        let mut result = self.codec.parse_response(&body, request)?;

        result.duration = start.elapsed().as_millis() as i64;

        Ok(result)
    }

    async fn generate_stream_inner(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> LlmResult<Box<dyn MessageStream>> {
        let mut stream_request = request.clone();
        stream_request.stream = Some(true);

        let http_request = self.finalize_request(
            self.codec.build_request(&stream_request, &self.profile)?,
            &stream_request,
        );

        let timeout_dur = self.build_timeout();
        let timeout_ms = timeout_dur.as_millis() as u64;

        let response = if let Some(ref cancel) = cancel {
            let fut = execute_with_timeout(self.client.execute(http_request), Some(timeout_ms));
            tokio::select! {
                result = fut => Self::map_timeout_result(result, timeout_ms)?,
                _ = cancel.cancelled() => return Err(LlmError::Cancelled),
            }
        } else {
            Self::map_timeout_result(
                execute_with_timeout(self.client.execute(http_request), Some(timeout_ms)).await,
                timeout_ms,
            )?
        };

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = retry_after_header(&response);
            let body = response.text().await.unwrap_or_default();
            return Err(Self::map_http_error(
                status,
                &body,
                timeout_ms,
                retry_after.as_deref(),
            ));
        }

        let stream = eventsource_stream::EventStream::new(response.bytes_stream());

        Ok(Box::new(crate::stream::SseMessageStream::new(
            stream,
            self.codec.clone(),
            cancel,
            request.dead_loop_detection.as_ref(),
        )))
    }

    /// Execute token counting for a request: provider count-tokens API
    /// first, local estimation as fallback. Transport execution lives
    /// here with the rest of the HTTP paths; `token::count` only owns
    /// the pure estimation fallback.
    async fn count_tokens_inner(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> LlmResult<llm_suite_types::llm::TokenCountResult> {
        if let Some(http_request) = self
            .codec
            .build_count_tokens_request(request, &self.profile)?
        {
            let http_request = self.finalize_request(http_request, request);
            let timeout_dur = self.build_timeout();
            let timeout_ms = timeout_dur.as_millis() as u64;
            let response = if let Some(ref cancel) = cancel {
                let req = self.client.execute(http_request);
                tokio::select! {
                    result = req => result.map_err(LlmError::HttpError)?,
                    _ = cancel.cancelled() => return Err(LlmError::Cancelled),
                    _ = tokio::time::sleep(timeout_dur) => {
                        return Err(LlmError::Timeout(timeout_ms));
                    }
                }
            } else {
                tokio::time::timeout(timeout_dur, self.client.execute(http_request))
                    .await
                    .map_err(|_| LlmError::Timeout(timeout_ms))?
                    .map_err(|e| {
                        if e.is_timeout() {
                            LlmError::Timeout(timeout_ms)
                        } else {
                            LlmError::HttpError(e)
                        }
                    })?
            };

            if !response.status().is_success() {
                let status = response.status();
                let retry_after = retry_after_header(&response);
                let body = response.text().await.unwrap_or_default();
                return Err(Self::map_http_error(
                    status,
                    &body,
                    timeout_dur.as_millis() as u64,
                    retry_after.as_deref(),
                ));
            }

            let body = response.text().await?;
            let json: serde_json::Value = serde_json::from_str(&body)?;
            let input_tokens = self.codec.parse_count_tokens_response(&json)?;
            Ok(llm_suite_types::llm::TokenCountResult {
                input_tokens,
                raw: Some(json),
            })
        } else {
            // Fallback to local estimation
            let estimated = llm_suite_token::count::estimate_request_tokens(request);
            Ok(llm_suite_types::llm::TokenCountResult {
                input_tokens: estimated,
                raw: None,
            })
        }
    }
}

impl LlmClient for LlmClientImpl {
    async fn generate(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> LlmResult<LlmResponseType> {
        // Retry emission position: the `on_retry` callback below fires after
        // the budget/policy check and before the backoff delay, with the same
        // payload the async `LLM_RETRY_SCHEDULED` event carries (see
        // `RetryAttemptDescriptor::event_metadata`). Sync observation bridges
        // through hook handlers, gating through approval, post-processing
        // through trigger templates subscribed to the retry event. The log
        // line keeps attempts visible even with zero subscribers.
        // Per-request `max_retries` / `retry_delay_ms` win over the profile
        // defaults for this call only.
        let policy = self.retry_policy_for(request);
        let cancel_token = cancel.as_ref().map(|c| (c, LlmError::Cancelled));
        llm_suite_common::retry::execute_with_retry_observed(
            Some(&policy),
            |r| matches!(r, Err(e) if e.is_retryable()),
            |r| match r {
                Err(e) => e.to_string(),
                Ok(_) => String::new(),
            },
            Some(&|descriptor| {
                tracing::info!(
                    attempt = descriptor.attempt,
                    max_retries = descriptor.max_retries,
                    delay_ms = descriptor.delay_ms,
                    reason = %descriptor.reason,
                    "LLM request failed, scheduling retry"
                );
                self.notify_retry(descriptor);
            }),
            cancel_token,
            || self.generate_inner(request, cancel.clone()),
        )
        .await
    }

    async fn generate_stream(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> LlmResult<Box<dyn MessageStream>> {
        // Per-request `max_retries` / `retry_delay_ms` win over the profile
        // defaults for this call only.
        let policy = self.retry_policy_for(request);
        let cancel_token = cancel.as_ref().map(|c| (c, LlmError::Cancelled));
        llm_suite_common::retry::execute_with_retry_observed(
            Some(&policy),
            |r| matches!(r, Err(e) if e.is_retryable()),
            |r| match r {
                Err(e) => e.to_string(),
                Ok(_) => String::new(),
            },
            Some(&|descriptor| {
                tracing::info!(
                    attempt = descriptor.attempt,
                    max_retries = descriptor.max_retries,
                    delay_ms = descriptor.delay_ms,
                    reason = %descriptor.reason,
                    "LLM stream request failed, scheduling retry"
                );
                self.notify_retry(descriptor);
            }),
            cancel_token,
            || self.generate_stream_inner(request, cancel.clone()),
        )
        .await
    }

    async fn count_tokens(
        &self,
        request: &LlmRequest,
        cancel: Option<CancellationToken>,
    ) -> LlmResult<llm_suite_types::llm::TokenCountResult> {
        self.count_tokens_inner(request, cancel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::{LlmFormat, LlmProfile};

    fn profile(id: &str) -> LlmProfile {
        LlmProfile {
            id: id.to_string(),
            name: id.to_string(),
            format: LlmFormat::OpenaiChat,
            provider_id: None,
            model: "gpt-4o".to_string(),
            api_key: None,
            base_url: None,
            parameters: None,
            generation: None,
            timeout: None,
            max_retries: None,
            retry_delay: None,
            headers: None,
            metadata: None,
            tool_call_protocol: None,
            auth_type: None,
            custom_headers: None,
            custom_body: None,
            custom_body_enabled: None,
            query_params: None,
            stream_options: None,
            prompt_caching: None,
            context_window_size: None,
            max_output_tokens: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    #[test]
    fn map_http_error_classifies_auth_and_timeout() {
        assert!(matches!(
            LlmClientImpl::map_http_error(reqwest::StatusCode::UNAUTHORIZED, "denied", 5000, None),
            LlmError::AuthError(_)
        ));
        assert!(matches!(
            LlmClientImpl::map_http_error(reqwest::StatusCode::FORBIDDEN, "denied", 5000, None),
            LlmError::AuthError(_)
        ));
        assert!(matches!(
            LlmClientImpl::map_http_error(reqwest::StatusCode::REQUEST_TIMEOUT, "", 5000, None),
            LlmError::Timeout(5000)
        ));
        assert!(matches!(
            LlmClientImpl::map_http_error(reqwest::StatusCode::GATEWAY_TIMEOUT, "", 5000, None),
            LlmError::Timeout(5000)
        ));
    }

    #[test]
    fn map_http_error_classifies_provider_errors() {
        assert!(matches!(
            LlmClientImpl::map_http_error(reqwest::StatusCode::BAD_REQUEST, "bad", 5000, None),
            LlmError::ProviderError { .. }
        ));
        assert!(matches!(
            LlmClientImpl::map_http_error(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "slow down",
                5000,
                None
            ),
            LlmError::RateLimited { .. }
        ));
        assert!(matches!(
            LlmClientImpl::map_http_error(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "slow down",
                5000,
                Some("2")
            ),
            LlmError::RateLimited {
                retry_after_ms: Some(2000)
            }
        ));
        assert!(matches!(
            LlmClientImpl::map_http_error(
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                "boom",
                5000,
                None
            ),
            LlmError::ProviderError { .. }
        ));
    }

    #[test]
    fn map_http_error_upgrades_context_length_messages() {
        let err = LlmClientImpl::map_http_error(
            reqwest::StatusCode::BAD_REQUEST,
            "This model's maximum context length is 200000 tokens",
            5000,
            None,
        );
        assert!(
            matches!(err, LlmError::ContextLengthExceeded(_)),
            "provider context-length rejection must be surfaced: {err:?}"
        );
    }

    #[test]
    fn map_http_error_flags_success_status() {
        let err = LlmClientImpl::map_http_error(reqwest::StatusCode::OK, "unexpected", 5000, None);
        assert!(matches!(err, LlmError::InvalidResponse(_)));
    }

    #[test]
    fn retry_after_header_parses_seconds_and_dates() {
        assert_eq!(parse_retry_after_ms(None), None);
        assert_eq!(parse_retry_after_ms(Some("")), None);
        assert_eq!(parse_retry_after_ms(Some("2")), Some(2000));
        assert_eq!(parse_retry_after_ms(Some("not-a-date")), None);
        let future = (chrono::Utc::now() + chrono::Duration::seconds(120)).to_rfc2822();
        let parsed = parse_retry_after_ms(Some(&future)).expect("http date must parse");
        assert!((110_000..=130_000).contains(&parsed));
    }

    #[test]
    fn timeout_defaults_to_60_seconds() {
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            profile("p1"),
        );
        assert_eq!(client.build_timeout(), Duration::from_secs(60));

        let mut p = profile("p1");
        p.timeout = Some(5);
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            p,
        );
        assert_eq!(client.build_timeout(), Duration::from_secs(5));
    }

    #[test]
    fn retry_parameters_use_defaults_and_overrides() {
        let bare_request = test_request("p1");
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            profile("p1"),
        );
        assert_eq!(client.max_retries(), 3);
        let policy = client.retry_policy_for(&bare_request);
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay_ms, 1000);
        assert!(policy.exponential_backoff);

        let mut p = profile("p1");
        p.max_retries = Some(5);
        p.retry_delay = Some(250);
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            p,
        );
        assert_eq!(client.max_retries(), 5);
        let policy = client.retry_policy_for(&bare_request);
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.base_delay_ms, 250);
        assert!(policy.exponential_backoff);
    }

    #[test]
    fn per_request_retry_overrides_win_over_profile() {
        let mut p = profile("p1");
        p.max_retries = Some(5);
        p.retry_delay = Some(250);
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            p,
        );
        let mut request = test_request("p1");
        request.max_retries = Some(1);
        request.retry_delay_ms = Some(50);
        let policy = client.retry_policy_for(&request);
        assert_eq!(policy.max_retries, 1);
        assert_eq!(policy.base_delay_ms, 50);

        let bare = test_request("p1");
        let fallback = client.retry_policy_for(&bare);
        assert_eq!(fallback.max_retries, 5);
        assert_eq!(fallback.base_delay_ms, 250);
    }

    #[test]
    fn extra_headers_merge_and_fetch_hook_runs_last() {
        fn hook(mut req: reqwest::Request) -> reqwest::Request {
            req.headers_mut().insert(
                reqwest::header::HeaderName::from_static("x-hook"),
                reqwest::header::HeaderValue::from_static("ran"),
            );
            req
        }
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            profile("p1"),
        )
        .with_fetch_hook(hook);
        let mut request = test_request("p1");
        request.extra_headers = Some(
            [("x-extra".to_string(), serde_json::json!("v1"))]
                .into_iter()
                .collect(),
        );
        let base = reqwest::Client::new()
            .request(reqwest::Method::POST, "https://example.com/v1/chat/completions")
            .build()
            .unwrap();
        let finalized = client.finalize_request(base, &request);
        assert_eq!(finalized.headers().get("x-extra").unwrap(), "v1");
        assert_eq!(
            finalized.headers().get("x-hook").unwrap(),
            "ran",
            "the fetch hook runs after extra headers"
        );
    }

    #[test]
    fn invalid_extra_headers_are_dropped() {
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            profile("p1"),
        );
        let mut request = test_request("p1");
        request.extra_headers = Some(
            [("not a header name".to_string(), serde_json::json!("v1"))]
                .into_iter()
                .collect(),
        );
        let base = reqwest::Client::new()
            .request(reqwest::Method::POST, "https://example.com/v1/chat/completions")
            .build()
            .unwrap();
        let finalized = client.finalize_request(base, &request);
        assert!(
            finalized.headers().get("not a header name").is_none(),
            "invalid overrides must not fail the request"
        );
    }

    fn test_request(profile_id: &str) -> LlmRequest {
        LlmRequest {
            profile_id: profile_id.to_string(),
            messages: Vec::new(),
            parameters: None,
            generation: None,
            tools: None,
            tool_call_protocol: None,
            locked_tool_call_protocol: None,
            violation_policy: None,
            execution_id: None,
            stream: None,
            dead_loop_detection: None,
            protocol_auto_converted: None,
            timeout_ms: None,
            prompt_caching: None,
            cached_content: None,
            max_retries: None,
            retry_delay_ms: None,
            extra_headers: None,
        }
    }

    #[test]
    fn profile_accessor_returns_configured_profile() {
        let client = LlmClientImpl::new(
            reqwest::Client::new(),
            llm_suite_codec::create_codec(&LlmFormat::OpenaiChat).unwrap(),
            profile("p1"),
        );
        assert_eq!(client.profile().id, "p1");
        assert_eq!(client.profile().model, "gpt-4o");
    }
}
