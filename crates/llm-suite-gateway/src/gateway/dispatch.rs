//! The request entry points: generate, generate_stream and count_tokens.

use std::sync::Arc;

use llm_suite_client::client::LlmClient;
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

        let prepared = self.prepare(request)?;
        self.admit(&prepared).await?;
        let start = std::time::Instant::now();
        let timeout_ms = prepared.effective.timeout_ms;
        let result = bound(timeout_ms, async {
            prepared.client.generate(&prepared.effective, cancel).await
        })
        .await;
        Self::record_breaker(prepared.breaker.as_ref(), &result);
        let duration_ms = start.elapsed().as_millis() as f64;
        match &result {
            Ok(response) => {
                self.record_token_usage(response, &prepared.profile);
                self.record_request(duration_ms, true, None, &prepared.profile);
            }
            Err(error) => {
                self.record_request(duration_ms, false, Some(error), &prepared.profile);
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

        let prepared = self.prepare(request)?;
        self.admit(&prepared).await?;
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
        match &stream {
            Ok(_) => {
                self.record_request(duration_ms, true, None, &prepared.profile);
                if let Some(metrics) = self.token_metrics.as_ref() {
                    metrics.record_first_byte(duration_ms, Some(&prepared.profile.model));
                }
            }
            Err(error) => self.record_request(duration_ms, false, Some(error), &prepared.profile),
        }
        let stream = stream?;
        let usage_sink = self
            .token_metrics
            .as_ref()
            .map(|m| Arc::clone(m) as llm_suite_client::SharedTokenUsageSink);
        Ok(Box::new(
            llm_suite_client::TokenRecordingStream::new(
                stream,
                usage_sink,
                prepared.profile.model.clone(),
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

        let prepared = self.prepare(request)?;
        self.admit(&prepared).await?;
        prepared
            .client
            .count_tokens(&prepared.effective, cancel)
            .await
    }
}
