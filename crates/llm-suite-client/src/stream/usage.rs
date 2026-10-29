//! Merging of the token usage reported across several stream events.

use llm_suite_types::llm::TokenUsageStats;

/// Merge the usage of a new event into the usage accumulated so far.
///
/// Field-wise merge: providers deliver usage across several events
/// (Anthropic's `message_start` carries input tokens, `message_delta`
/// carries cumulative output and may omit input tokens). Later non-zero /
/// `Some` values win; zero / `None` values never erase earlier data.
pub fn merge_usage(
    previous: Option<&TokenUsageStats>,
    incoming: &TokenUsageStats,
) -> TokenUsageStats {
    let Some(prev) = previous else {
        return incoming.clone();
    };
    let u = incoming;
    let prompt_tokens = if u.prompt_tokens > 0 {
        u.prompt_tokens
    } else {
        prev.prompt_tokens
    };
    let completion_tokens = if u.completion_tokens > 0 {
        u.completion_tokens
    } else {
        prev.completion_tokens
    };
    let cache_read_tokens = u.cache_read_tokens.or(prev.cache_read_tokens);
    let cache_write_tokens = u.cache_write_tokens.or(prev.cache_write_tokens);
    // A provider that reports a total of its own owns the number. Without
    // one the total is recomputed from the merged counters, and the cached
    // input tokens are added: a provider that reports them next to the
    // uncached ones (Anthropic) only reaches this path.
    let component_total = prompt_tokens
        + completion_tokens
        + cache_read_tokens.unwrap_or(0)
        + cache_write_tokens.unwrap_or(0);
    let total_tokens = if u.total_tokens > 0 {
        u.total_tokens
    } else if component_total > 0 {
        component_total
    } else {
        prev.total_tokens
    };
    TokenUsageStats {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        reasoning_tokens: u.reasoning_tokens.or(prev.reasoning_tokens),
        cache_read_tokens,
        cache_write_tokens,
        prompt_tokens_cost: u.prompt_tokens_cost.or(prev.prompt_tokens_cost),
        completion_tokens_cost: u.completion_tokens_cost.or(prev.completion_tokens_cost),
        total_cost: u.total_cost.or(prev.total_cost),
        provider_total: None,
    }
}
