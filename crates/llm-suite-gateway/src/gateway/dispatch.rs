//! The request entry points: generate, generate_stream and count_tokens.

use std::sync::Arc;

use llm_suite_client::client::LlmClient;
use llm_suite_client::token_stream::SharedTokenUsageSink;
use llm_suite_client::MessageStream;
use llm_suite_codec::error::LlmResult;
use llm_suite_types::llm::{LlmRequest, LlmResult as LlmResponseType};

use super::prepare::bound;
use super::LlmGateway;

impl LlmGateway {
    pub async fn generate(
        &self,
        request: &LlmRequest,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> LlmResult<LlmResponseType> {
        #[cfg(feature = "mock")]
        if let Some(client) = self.mock_client(&request.profile_id) {
            return client.generate(request, cancel).await;
        }

        let prepared = self.prepare(request).await?;
        self.admit(&prepared).await?;
        if let Some(on_request) = self.probe.on_request {
            on_request(&prepared.effective, &prepared.profile);
        }
        tracing::debug!(
            profile_id = prepared.profile.id.as_str(),
            model = prepared.profile.model.as_str(),
            message_count = prepared.effective.messages.len(),
            "llm request dispatched"
        );
        let start = std::time::Instant::now();
        let timeout_ms = prepared.effective.timeout_ms;
        let result = bound(timeout_ms, async {
            prepared.client.generate(&prepared.effective, cancel).await
        })
        .await;
        Self::record_breaker(prepared.breaker.as_ref(), &result);
        let duration_ms = start.elapsed().as_millis() as f64;
        if let Some(on_response) = self.probe.on_response {
            on_response(&prepared.effective, &result, duration_ms);
        }
        let context = self.call_context(&prepared);
        match &result {
            Ok(response) => {
                self.record_token_usage(response, &prepared, &context);
                self.record_request(duration_ms, true, None, &context);
            }
            Err(error) => {
                self.record_request(duration_ms, false, Some(error), &context);
            }
        }
        result
    }

    pub async fn generate_stream(
        &self,
        request: &LlmRequest,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> LlmResult<Box<dyn MessageStream>> {
        #[cfg(feature = "mock")]
        if let Some(client) = self.mock_client(&request.profile_id) {
            return client.generate_stream(request, cancel).await;
        }

        let prepared = self.prepare(request).await?;
        self.admit(&prepared).await?;
        if let Some(on_request) = self.probe.on_request {
            on_request(&prepared.effective, &prepared.profile);
        }
        tracing::debug!(
            profile_id = prepared.profile.id.as_str(),
            model = prepared.profile.model.as_str(),
            message_count = prepared.effective.messages.len(),
            "llm stream request dispatched"
        );
        let start = std::time::Instant::now();
        let timeout_ms = prepared.effective.timeout_ms;
        // The per-request bound covers stream establishment only (including
        // transport retries); consuming the returned stream is bounded by
        // the client-level and caller-level budgets, not here.
        let stream = bound(timeout_ms, async {
            prepared
                .client
                .generate_stream(&prepared.effective, cancel)
                .await
        })
        .await;
        Self::record_breaker(prepared.breaker.as_ref(), &stream);
        let duration_ms = start.elapsed().as_millis() as f64;
        if let Some(on_stream_start) = self.probe.on_stream_start {
            on_stream_start(&prepared.effective, stream.is_ok(), duration_ms);
        }
        let context = self.call_context(&prepared);
        match &stream {
            Ok(_) => {
                // The stream is not consumed yet: only the establishment
                // latency is known here. The end-to-end request duration is
                // recorded when the stream wrapper observes the final
                // message.
                if let Some(metrics) = self.token_metrics.as_ref() {
                    metrics.record_first_byte(duration_ms, &context);
                }
            }
            Err(error) => self.record_request(duration_ms, false, Some(error), &context),
        }
        let stream = stream?;
        let usage_sink: Option<SharedTokenUsageSink> = self.token_metrics.as_ref().map(|m| {
            self.cost_enriching_sink(Arc::clone(m), &prepared.profile) as SharedTokenUsageSink
        });
        let fallback_usage = self.estimated_usage(&prepared.effective);
        Ok(Box::new(
            llm_suite_client::TokenRecordingStream::new(
                stream,
                usage_sink,
                context,
                fallback_usage,
            )
            .with_metrics_sink(self.token_metrics.clone()),
        ))
    }

    pub async fn count_tokens(
        &self,
        request: &LlmRequest,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> LlmResult<llm_suite_types::llm::TokenCountResult> {
        #[cfg(feature = "mock")]
        if let Some(client) = self.mock_client(&request.profile_id) {
            return client.count_tokens(request, cancel).await;
        }

        let prepared = self.prepare(request).await?;
        self.admit(&prepared).await?;
        prepared
            .client
            .count_tokens(&prepared.effective, cancel)
            .await
    }
}
