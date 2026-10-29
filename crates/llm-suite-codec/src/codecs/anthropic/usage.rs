//! Token usage accounting for the Anthropic Messages API.
//!
//! The cache-write counter has a single implementation because the streaming
//! and the non-streaming path must agree on the precedence rule: the explicit
//! total wins, otherwise the TTL breakdown is summed.

use llm_suite_types::llm::{MessageStreamUsage, TokenUsageStats};

/// Cache-write input tokens: prefer the explicit total
/// (`cache_creation_input_tokens`), fall back to the sum of the TTL
/// breakdown `cache_creation.{ephemeral_5m_input_tokens,
/// ephemeral_1h_input_tokens}`. Returns `None` when neither is present.
pub fn cache_write_tokens(usage: &serde_json::Value) -> Option<u32> {
    if let Some(total) = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
    {
        return Some(total as u32);
    }
    let creation = usage.get("cache_creation")?;
    let sum: u64 = ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"]
        .iter()
        .filter_map(|k| creation.get(*k).and_then(|v| v.as_u64()))
        .sum();
    (sum > 0).then_some(sum as u32)
}

/// Build a usage event from an Anthropic usage object. Returns `None`
/// when the object carries no counters at all, so zero-usage events never
/// clobber accumulated values.
pub fn stream_usage(usage: &serde_json::Value) -> Option<MessageStreamUsage> {
    let prompt_tokens = usage
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let completion_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let reasoning_tokens = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("thinking_tokens"))
        .and_then(|v| v.as_u64())
        .map(|r| r as u32);
    let cache_read_tokens = usage
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .map(|r| r as u32);
    let cache_write_tokens = cache_write_tokens(usage);

    if prompt_tokens == 0
        && completion_tokens == 0
        && reasoning_tokens.is_none()
        && cache_read_tokens.is_none()
        && cache_write_tokens.is_none()
    {
        return None;
    }

    Some(MessageStreamUsage {
        usage: TokenUsageStats {
            prompt_tokens,
            completion_tokens,
            // The wire carries no total; the stream accumulator
            // recomputes it from the merged counters.
            total_tokens: 0,
            reasoning_tokens,
            cache_read_tokens,
            cache_write_tokens,
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: None,
            provider_total: crate::codecs::helpers::parse_provider_total(usage),
        },
    })
}
