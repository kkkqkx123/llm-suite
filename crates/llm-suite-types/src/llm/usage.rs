use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenUsageStats {
    /// Input tokens of the request.
    ///
    /// The scope follows the provider: the OpenAI family counts the cached
    /// input into the prompt (`cached_tokens` is a subset of
    /// `prompt_tokens`), Anthropic reports the uncached input next to the
    /// cache counters. `total_tokens` always covers the whole request, so
    /// the two scopes are resolved where the money is counted
    /// (`CacheTokenScope` in the pricing layer).
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Every token the request consumed, input and output.
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Cache-hit input tokens (provider `cached_tokens` / `cache_read_input_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    /// Cache-write input tokens (provider `cache_creation_input_tokens`).
    ///
    /// Note: `cache_write_tokens` is **not** part of the official OpenAI API
    /// (`PromptTokensDetails` only carries `cached_tokens`); the key is
    /// honored as an extension for OpenAI-compatible gateways. Official
    /// OpenAI endpoints never populate it, and the Gemini codec leaves it
    /// `None` (Gemini does not report cache-write counts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cost: Option<f64>,
}

/// Cost-track record of a single LLM request (real provider usage only).
///
/// The history is the accounting/audit trail: it never mixes in estimated
/// values. When a provider returns no usage the entry is recorded with zero
/// token fields and `estimated: Some(true)` as a marker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenUsageHistory {
    pub request_id: String,
    pub timestamp: super::super::Timestamp,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_id: Option<String>,
    /// True when the entry was filled with an estimation because the
    /// provider reported no usage (cost-track marker, never drives decisions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated: Option<bool>,
}

impl TokenUsageHistory {
    /// Build an audit entry from real provider usage. `estimated` stays `None`
    /// for real entries; the zero-token + `Some(true)` shape is reserved for
    /// entries recorded when the provider reported no usage at all.
    #[allow(clippy::too_many_arguments)]
    pub fn from_usage(
        request_id: impl Into<String>,
        timestamp: super::super::Timestamp,
        usage: &TokenUsageStats,
        model: Option<String>,
        profile: Option<String>,
        execution_id: Option<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            timestamp,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            cache_write_tokens: usage.cache_write_tokens,
            cost: usage.total_cost,
            model,
            profile,
            execution_id,
            estimated: None,
        }
    }
}

/// Result of a provider-side token counting API call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenCountResult {
    pub input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_usage_copies_full_breakdown_without_estimation_marker() {
        let usage = TokenUsageStats {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            reasoning_tokens: Some(6),
            cache_read_tokens: Some(80),
            cache_write_tokens: Some(15),
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: Some(0.42),
        };
        let history = TokenUsageHistory::from_usage(
            "req-1",
            1_700_000_000,
            &usage,
            Some("claude-sonnet".to_string()),
            Some("p1".to_string()),
            Some("exec-1".to_string()),
        );
        assert_eq!(history.request_id, "req-1");
        assert_eq!(history.prompt_tokens, 100);
        assert_eq!(history.completion_tokens, 20);
        assert_eq!(history.total_tokens, 120);
        assert_eq!(history.reasoning_tokens, Some(6));
        assert_eq!(history.cache_read_tokens, Some(80));
        assert_eq!(history.cache_write_tokens, Some(15));
        assert_eq!(history.cost, Some(0.42));
        assert_eq!(history.model.as_deref(), Some("claude-sonnet"));
        assert_eq!(history.profile.as_deref(), Some("p1"));
        assert_eq!(history.execution_id.as_deref(), Some("exec-1"));
        // Real entries never carry the estimation marker.
        assert_eq!(history.estimated, None);
    }
}
