//! Token usage metrics: the call context of a request, the cost
//! enrichment of a usage report and the request outcome recording.

use std::sync::Arc;

use llm_suite_client::token_stream::TokenUsageSink;
use llm_suite_types::llm::{LlmProfile, LlmRequest, LlmResult as LlmResponseType, TokenUsageStats};
use llm_token::{
    compute_cache_savings, compute_cost, LlmCallContext, LlmUsageReport, PricingResolver,
};

use super::prepare::PreparedRequest;
use super::LlmGateway;

impl LlmGateway {
    /// Context of a call issued through a prepared request.
    ///
    /// The context carries the dimensions of the call (model, profile,
    /// provider, format, execution) and the accounting convention of the
    /// wire format, so the metrics sink never has to guess them.
    pub(super) fn call_context(&self, prepared: &PreparedRequest) -> LlmCallContext {
        LlmCallContext::from_profile(
            Some(prepared.profile.model.clone()),
            Some(prepared.profile.id.clone()),
            prepared.profile.provider_id.clone(),
            &prepared.profile.format,
            prepared.effective.execution_id.clone(),
        )
    }

    /// Local estimate of a request, used when the provider reports no
    /// usage. The estimate keeps the request visible in the metrics
    /// instead of silently dropping it.
    pub(super) fn estimated_usage(&self, request: &LlmRequest) -> Option<TokenUsageStats> {
        let input = llm_token::count::estimate_request_tokens(request);
        if input == 0 {
            return None;
        }
        Some(TokenUsageStats {
            prompt_tokens: input,
            completion_tokens: 0,
            total_tokens: input,
            reasoning_tokens: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: None,
            provider_total: None,
        })
    }

    /// Sink that adds the computed cost to a usage report before it
    /// reaches the metrics sink.
    ///
    /// The pricing resolver is optional: without it the report travels
    /// unchanged and carries no cost.
    pub(super) fn cost_enriching_sink(
        &self,
        inner: Arc<dyn llm_suite_client::LlmMetricsSink>,
        profile: &LlmProfile,
    ) -> Arc<CostEnrichingSink> {
        Arc::new(CostEnrichingSink {
            inner,
            pricing: self.pricing_resolver.clone(),
            provider_id: profile.provider_id.clone(),
            model: profile.model.clone(),
        })
    }

    pub(super) fn record_token_usage(
        &self,
        result: &LlmResponseType,
        prepared: &PreparedRequest,
        context: &LlmCallContext,
    ) {
        let Some(metrics) = self.token_metrics.as_ref() else {
            return;
        };
        let usage = match result.usage.as_ref() {
            Some(usage) => usage.clone(),
            None => match self.estimated_usage(&prepared.effective) {
                Some(estimate) => estimate,
                None => return,
            },
        };
        let estimated = result.usage.is_none();
        let mut report = LlmUsageReport::from_usage(context.clone(), usage);
        report.estimated = estimated;
        // Subscription-backed access is not metered, so no cost is attached.
        if prepared.is_subscription {
            metrics.record_token_usage(&report);
        } else {
            self.cost_enriching_sink(Arc::clone(metrics), &prepared.profile)
                .record_token_usage(&report);
        }
    }

    pub(super) fn record_request(
        &self,
        duration_ms: f64,
        success: bool,
        error: Option<&llm_suite_codec::error::LlmError>,
        context: &LlmCallContext,
    ) {
        let Some(metrics) = self.token_metrics.as_ref() else {
            return;
        };
        metrics.record_request(
            duration_ms,
            success,
            error.map(|e| e.metric_class()),
            context,
        );
    }
}

/// Adds the computed cost to a usage report before it reaches the
/// metrics sink.
pub(super) struct CostEnrichingSink {
    inner: Arc<dyn llm_suite_client::LlmMetricsSink>,
    pricing: Option<Arc<dyn PricingResolver>>,
    provider_id: Option<String>,
    model: String,
}

impl TokenUsageSink for CostEnrichingSink {
    fn record_token_usage(&self, report: &LlmUsageReport) {
        let mut enriched = report.clone();
        if let Some(resolver) = self.pricing.as_ref() {
            if let Some(resolved) = resolver.resolve(self.provider_id.as_deref(), &self.model) {
                compute_cost(&enriched.usage, &resolved.pricing, resolved.scope)
                    .apply_to(&mut enriched.usage);
                enriched.cache_savings = Some(compute_cache_savings(
                    &enriched.usage,
                    &resolved.pricing,
                    resolved.scope,
                ));
            }
        }
        self.inner.record_token_usage(&enriched);
    }
}
