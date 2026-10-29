//! Token usage accounting for the OpenAI Chat Completions usage shape.

/// Parse an OpenAI-style usage object into token usage stats.
///
/// The cached input tokens are a subset of `prompt_tokens` in this family
/// (`CacheTokenScope::IncludedInPrompt`), so the total is the sum of the
/// prompt and the completion. The provider's own total wins when it is
/// present.
pub fn parse_openai_usage(u: &serde_json::Value) -> llm_suite_types::llm::TokenUsageStats {
    let reasoning_tokens = u
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|r| r.as_u64())
        .map(|r| r as u32);
    let cache_read_tokens = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|r| r.as_u64())
        .map(|r| r as u32);
    let cache_write_tokens = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cache_write_tokens"))
        .and_then(|r| r.as_u64())
        .map(|r| r as u32);
    let prompt_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let completion_tokens = u
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    llm_suite_types::llm::TokenUsageStats {
        prompt_tokens,
        completion_tokens,
        // The provider owns the total; without one it is the sum of the
        // two counters. The cached tokens are already part of the prompt.
        total_tokens: u
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::from(prompt_tokens) + u64::from(completion_tokens))
            as u32,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        prompt_tokens_cost: None,
        completion_tokens_cost: None,
        total_cost: None,
        provider_total: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_without_a_provider_total_is_summed() {
        // The cached tokens are already part of the prompt in this family,
        // so the total is the sum of the two counters.
        let usage = parse_openai_usage(&serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80}
        }));
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(usage.cache_read_tokens, Some(80));
    }

    #[test]
    fn usage_parses_official_cache_details() {
        // Official `CompletionUsage` shape: `prompt_tokens_details` carries
        // both `cached_tokens` and `cache_write_tokens`.
        let usage = parse_openai_usage(&serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 15},
            "completion_tokens_details": {"reasoning_tokens": 6}
        }));
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(usage.reasoning_tokens, Some(6));
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, Some(15));

        // Details objects absent → cache fields stay `None`.
        let bare = parse_openai_usage(&serde_json::json!({
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
        }));
        assert_eq!(bare.cache_read_tokens, None);
        assert_eq!(bare.cache_write_tokens, None);
    }
}
