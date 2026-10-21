//! Token usage accounting for the Gemini native API.

/// Parse the top-level `usageMetadata` object into token stats.
pub fn parse_usage_metadata(u: &serde_json::Value) -> llm_suite_types::llm::TokenUsageStats {
    // Cached content tokens: primary source `cachedContentTokenCount`;
    // when absent, fall back to summing the per-modality breakdown in
    // `cacheTokensDetails` (`[{modality, tokenCount}, ...]`).
    let cache_read_tokens = u
        .get("cachedContentTokenCount")
        .and_then(|v| v.as_u64())
        .map(|r| r as u32)
        .or_else(|| {
            let sum: u64 = u
                .get("cacheTokensDetails")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m.get("tokenCount").and_then(|t| t.as_u64()))
                        .sum()
                })
                .unwrap_or(0);
            (sum > 0).then_some(sum as u32)
        });
    let prompt_tokens = u
        .get("promptTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let completion_tokens = u
        .get("candidatesTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let reasoning_tokens = u
        .get("thoughtsTokenCount")
        .and_then(|v| v.as_u64())
        .map(|r| r as u32);
    llm_suite_types::llm::TokenUsageStats {
        prompt_tokens,
        completion_tokens,
        // The provider owns the total; without one it is the sum of the
        // counters it reports. The thoughts are billed on top of the
        // candidates, and the cached tokens are already part of the
        // prompt.
        total_tokens: u.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(
            u64::from(prompt_tokens)
                + u64::from(completion_tokens)
                + u64::from(reasoning_tokens.unwrap_or(0)),
        ) as u32,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens: None,
        prompt_tokens_cost: None,
        completion_tokens_cost: None,
        total_cost: None,
    }
}
