//! Cost computation from real provider usage.
//!
//! Pure helper for host-side cost tracking: given a `TokenUsageStats`
//! (including the cache-read / cache-write breakdown) and per-model prices,
//! it computes the request cost with differential cache pricing. It never
//! invents usage: providers that reported no cache fields are billed at the
//! base rates for those tokens.

use llm_suite_types::llm::TokenUsageStats;
use serde::{Deserialize, Serialize};

/// Per-million-token prices for one model. Currency-agnostic: callers decide
/// whether the numbers are USD, CNY, and so on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenPricing {
    /// Price per million non-cached input tokens.
    pub input_per_mtok: f64,
    /// Price per million output (completion) tokens.
    pub output_per_mtok: f64,
    /// Price per million cache-hit (read) input tokens. `None` falls back to
    /// the full input price (no cache discount).
    pub cache_read_per_mtok: Option<f64>,
    /// Price per million cache-write input tokens at the 5-minute TTL
    /// (Anthropic bills cache writes above the base input rate). `None`
    /// falls back to the full input price.
    pub cache_write_5m_per_mtok: Option<f64>,
    /// Price per million cache-write input tokens at the 1-hour TTL.
    /// Note: `TokenUsageStats.cache_write_tokens` merges both TTL buckets
    /// (the Anthropic codec sums the breakdown), so this rate is only used
    /// by callers that track the TTL split themselves.
    pub cache_write_1h_per_mtok: Option<f64>,
}

/// Whether the provider's `prompt_tokens` already includes the cache-hit and
/// cache-write token counts or reports them on top of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTokenScope {
    /// `prompt_tokens` includes `cache_read_tokens` / `cache_write_tokens`
    /// (OpenAI Chat Completions and Responses semantics: `cached_tokens` is
    /// a subset of `prompt_tokens`).
    IncludedInPrompt,
    /// `prompt_tokens` excludes cache tokens; the provider reports them in
    /// addition (Anthropic Messages semantics: `input_tokens` does not
    /// contain `cache_read_input_tokens` / `cache_creation_input_tokens`).
    SeparateFromPrompt,
}

/// Computed cost breakdown for a single request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCost {
    pub prompt_tokens_cost: f64,
    pub completion_tokens_cost: f64,
    pub total_cost: f64,
}

impl TokenCost {
    /// Fill the cost fields of a `TokenUsageStats` (or of an audit copy of
    /// it) so the numbers travel with the usage they were derived from.
    pub fn apply_to(&self, usage: &mut TokenUsageStats) {
        usage.prompt_tokens_cost = Some(self.prompt_tokens_cost);
        usage.completion_tokens_cost = Some(self.completion_tokens_cost);
        usage.total_cost = Some(self.total_cost);
    }
}

/// Compute the cost of one request from real provider usage.
///
/// Under [`CacheTokenScope::IncludedInPrompt`] the cache token counts are
/// subtracted from `prompt_tokens` (saturating at zero, so malformed usage
/// payloads cannot produce a negative non-cached input); the remainder is
/// billed at the base input rate and the cache tokens at their own rates.
/// Under [`CacheTokenScope::SeparateFromPrompt`] the full `prompt_tokens` is
/// billed at the base input rate and cache tokens are billed additionally.
///
/// Cache writes are billed at the 5-minute rate because the unified usage
/// model does not carry the TTL split; callers tracking the split themselves
/// should compute the write term manually.
pub fn compute_cost(
    usage: &TokenUsageStats,
    pricing: &TokenPricing,
    scope: CacheTokenScope,
) -> TokenCost {
    let cache_read = usage.cache_read_tokens.unwrap_or(0) as f64;
    let cache_write = usage.cache_write_tokens.unwrap_or(0) as f64;

    let non_cached_input = match scope {
        CacheTokenScope::IncludedInPrompt => {
            (usage.prompt_tokens as f64 - cache_read - cache_write).max(0.0)
        }
        CacheTokenScope::SeparateFromPrompt => usage.prompt_tokens as f64,
    };

    let read_rate = pricing
        .cache_read_per_mtok
        .unwrap_or(pricing.input_per_mtok);
    let write_rate = pricing
        .cache_write_5m_per_mtok
        .unwrap_or(pricing.input_per_mtok);

    let per_mtok = 1_000_000.0;
    let prompt_tokens_cost = non_cached_input * pricing.input_per_mtok / per_mtok
        + cache_read * read_rate / per_mtok
        + cache_write * write_rate / per_mtok;
    let completion_tokens_cost =
        usage.completion_tokens as f64 * pricing.output_per_mtok / per_mtok;

    TokenCost {
        prompt_tokens_cost,
        completion_tokens_cost,
        total_cost: prompt_tokens_cost + completion_tokens_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(
        prompt: u32,
        completion: u32,
        cache_read: Option<u32>,
        cache_write: Option<u32>,
    ) -> TokenUsageStats {
        TokenUsageStats {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            reasoning_tokens: None,
            cache_read_tokens: cache_read,
            cache_write_tokens: cache_write,
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: None,
        }
    }

    #[test]
    fn openai_semantics_subtract_cache_tokens_from_prompt() {
        // OpenAI-style pricing: input $1.0/MTok, output $2.0/MTok, cached
        // read $0.1/MTok. prompt_tokens (1000) already includes the 800
        // cached tokens.
        let pricing = TokenPricing {
            input_per_mtok: 1.0,
            output_per_mtok: 2.0,
            cache_read_per_mtok: Some(0.1),
            cache_write_5m_per_mtok: None,
            cache_write_1h_per_mtok: None,
        };
        let cost = compute_cost(
            &usage(1000, 100, Some(800), None),
            &pricing,
            CacheTokenScope::IncludedInPrompt,
        );
        // 200 non-cached input + 800 cached read + 100 output.
        let expected = 200.0 * 1.0 / 1e6 + 800.0 * 0.1 / 1e6 + 100.0 * 2.0 / 1e6;
        assert!((cost.prompt_tokens_cost - (200.0 + 80.0) / 1e6).abs() < 1e-12);
        assert!((cost.completion_tokens_cost - 200.0 / 1e6).abs() < 1e-12);
        assert!((cost.total_cost - expected).abs() < 1e-12);
    }

    #[test]
    fn anthropic_semantics_bill_cache_tokens_on_top() {
        // Anthropic-style: input $3.0/MTok, read $0.3/MTok, 5m write
        // $3.75/MTok, output $15.0/MTok. prompt_tokens (10) excludes cache.
        let pricing = TokenPricing {
            input_per_mtok: 3.0,
            output_per_mtok: 15.0,
            cache_read_per_mtok: Some(0.3),
            cache_write_5m_per_mtok: Some(3.75),
            cache_write_1h_per_mtok: Some(6.0),
        };
        let cost = compute_cost(
            &usage(10, 20, Some(80), Some(15)),
            &pricing,
            CacheTokenScope::SeparateFromPrompt,
        );
        let expected_prompt = 10.0 * 3.0 / 1e6 + 80.0 * 0.3 / 1e6 + 15.0 * 3.75 / 1e6;
        assert!((cost.prompt_tokens_cost - expected_prompt).abs() < 1e-12);
        assert!((cost.completion_tokens_cost - 20.0 * 15.0 / 1e6).abs() < 1e-12);
        assert!((cost.total_cost - (expected_prompt + 300.0 / 1e6)).abs() < 1e-12);
    }

    #[test]
    fn malformed_usage_clamps_at_zero_non_cached_input() {
        let pricing = TokenPricing {
            input_per_mtok: 1.0,
            output_per_mtok: 1.0,
            cache_read_per_mtok: Some(0.1),
            cache_write_5m_per_mtok: None,
            cache_write_1h_per_mtok: None,
        };
        // Cache read (800) exceeds prompt_tokens (10): never negative.
        let cost = compute_cost(
            &usage(10, 0, Some(800), None),
            &pricing,
            CacheTokenScope::IncludedInPrompt,
        );
        assert!(cost.prompt_tokens_cost >= 0.0);
        assert!((cost.prompt_tokens_cost - 800.0 * 0.1 / 1e6).abs() < 1e-12);
    }

    #[test]
    fn missing_cache_rates_fall_back_to_input_price() {
        let pricing = TokenPricing {
            input_per_mtok: 1.0,
            output_per_mtok: 1.0,
            cache_read_per_mtok: None,
            cache_write_5m_per_mtok: None,
            cache_write_1h_per_mtok: None,
        };
        let cost = compute_cost(
            &usage(0, 0, Some(100), Some(50)),
            &pricing,
            CacheTokenScope::SeparateFromPrompt,
        );
        // All cache tokens billed at the base input rate.
        assert!((cost.prompt_tokens_cost - 150.0 / 1e6).abs() < 1e-12);
    }

    #[test]
    fn apply_to_fills_cost_fields() {
        let pricing = TokenPricing {
            input_per_mtok: 1.0,
            output_per_mtok: 1.0,
            cache_read_per_mtok: None,
            cache_write_5m_per_mtok: None,
            cache_write_1h_per_mtok: None,
        };
        let mut stats = usage(10, 5, None, None);
        compute_cost(&stats, &pricing, CacheTokenScope::IncludedInPrompt)
            .apply_to(&mut stats);
        assert_eq!(stats.prompt_tokens_cost, Some(10.0 / 1e6));
        assert_eq!(stats.completion_tokens_cost, Some(5.0 / 1e6));
        let total = stats.total_cost.expect("total_cost must be set");
        assert!((total - 15.0 / 1e6).abs() < 1e-15);
    }
}
