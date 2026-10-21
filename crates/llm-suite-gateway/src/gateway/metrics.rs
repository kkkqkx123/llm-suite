//! Token usage metrics and the low-cardinality error classifier.

use llm_suite_types::llm::{LlmProfile, LlmResult as LlmResponseType};
use llm_token::compute_cost;

use super::LlmGateway;

impl LlmGateway {
    pub(super) fn record_token_usage(&self, result: &LlmResponseType, profile: &LlmProfile) {
        let Some(metrics) = self.token_metrics.as_ref() else {
            return;
        };
        let Some(usage) = result.usage.as_ref() else {
            return;
        };
        // Enrich usage with a computed cost when a pricing resolver is wired.
        // `enriched` keeps the original fields; only `total_cost` (and the
        // per-kind cost fields) are filled in when a price is found.
        let mut enriched = usage.clone();
        if let Some(resolver) = self.pricing_resolver.as_ref() {
            if let Some(resolved) = resolver.resolve(profile.provider_id.as_deref(), &profile.model)
            {
                compute_cost(&enriched, &resolved.pricing, resolved.scope).apply_to(&mut enriched);
            }
        }
        metrics.record_token_usage(&enriched, Some(&profile.model));
    }

    pub(super) fn record_request(
        &self,
        duration_ms: f64,
        success: bool,
        error: Option<&llm_suite_codec::error::LlmError>,
        profile: &LlmProfile,
    ) {
        let Some(metrics) = self.token_metrics.as_ref() else {
            return;
        };
        metrics.record_request(
            duration_ms,
            success,
            error.map(classify_error),
            Some(&profile.model),
        );
        if !success && profile.max_retries.unwrap_or(0) > 0 {
            metrics.record_retry(Some(&profile.model));
        }
    }
}

/// Low-cardinality error classifier for LLM request metrics.
fn classify_error(error: &llm_suite_codec::error::LlmError) -> &'static str {
    match error {
        llm_suite_codec::error::LlmError::HttpError(_) => "http_error",
        llm_suite_codec::error::LlmError::SerializationError(_) => "serialization_error",
        llm_suite_codec::error::LlmError::ProviderError { .. } => "provider_error",
        llm_suite_codec::error::LlmError::ContextLengthExceeded(_) => "context_length_exceeded",
        llm_suite_codec::error::LlmError::ConfigError(_) => "config_error",
        llm_suite_codec::error::LlmError::StreamError(_) => "stream_error",
        llm_suite_codec::error::LlmError::ProfileNotFound(_) => "profile_not_found",
        llm_suite_codec::error::LlmError::UnsupportedFormat(_) => "unsupported_format",
        llm_suite_codec::error::LlmError::CodecNotFound(_) => "codec_not_found",
        llm_suite_codec::error::LlmError::Timeout(_) => "timeout",
        llm_suite_codec::error::LlmError::AuthError(_) => "auth_error",
        llm_suite_codec::error::LlmError::ToolNotFound(_) => "tool_not_found",
        llm_suite_codec::error::LlmError::InvalidResponse(_) => "invalid_response",
        llm_suite_codec::error::LlmError::Cancelled => "cancelled",
        llm_suite_codec::error::LlmError::RateLimited { .. } => "rate_limited",
        llm_suite_codec::error::LlmError::CircuitOpen => "circuit_open",
        llm_suite_codec::error::LlmError::ProxyError(_) => "proxy_error",
    }
}
