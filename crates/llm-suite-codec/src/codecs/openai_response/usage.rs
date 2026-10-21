//! Token usage accounting for the OpenAI Responses API.
//!
//! The streaming and the non-streaming path share this single reader: the
//! Responses usage object carries the same counters in both cases.

use llm_suite_types::llm::TokenUsageStats;

/// Parse the official Responses `usage` object into token usage stats.
pub fn parse_responses_usage(u: &serde_json::Value) -> TokenUsageStats {
    let input_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let output_tokens = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    TokenUsageStats {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        // The provider owns the total: the official usage object carries
        // it, and it may cover buckets the two counters do not describe.
        // The sum is only the fallback.
        total_tokens: u
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::from(input_tokens) + u64::from(output_tokens))
            as u32,
        reasoning_tokens: u
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .map(|r| r as u32),
        cache_read_tokens: u
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .map(|r| r as u32),
        cache_write_tokens: u
            .get("input_tokens_details")
            .and_then(|d| d.get("cache_write_tokens"))
            .and_then(|v| v.as_u64())
            .map(|r| r as u32),
        prompt_tokens_cost: None,
        completion_tokens_cost: None,
        total_cost: None,
    }
}
