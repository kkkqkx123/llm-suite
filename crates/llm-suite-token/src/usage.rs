//! Provider-independent token accounting.
//!
//! Providers report the same request under different accounting
//! conventions: some count the cached input inside the prompt, others
//! report it next to it; some count the reasoning tokens inside the
//! completion, others bill them on top. The structures here describe the
//! convention of a wire format and turn a provider usage report into one
//! set of numbers that mean the same thing everywhere.

use llm_suite_types::llm::{LlmFormat, TokenUsageStats};

use crate::pricing::CacheTokenScope;

/// Where a provider counts the reasoning tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningTokenScope {
    /// The reasoning tokens are part of the completion tokens
    /// (OpenAI Chat / OpenAI Responses / Anthropic).
    IncludedInCompletion,
    /// The reasoning tokens are billed on top of the completion tokens
    /// (Gemini counts the thoughts separately from the candidates).
    SeparateFromCompletion,
}

/// The accounting convention of a wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenAccounting {
    /// Placement of the cached input tokens.
    pub cache: CacheTokenScope,
    /// Placement of the reasoning tokens.
    pub reasoning: ReasoningTokenScope,
}

impl Default for TokenAccounting {
    /// The OpenAI-compatible convention: cached input inside the prompt,
    /// reasoning inside the completion.
    fn default() -> Self {
        Self {
            cache: CacheTokenScope::IncludedInPrompt,
            reasoning: ReasoningTokenScope::IncludedInCompletion,
        }
    }
}

/// Accounting convention of a wire format.
///
/// The built-in formats follow their provider protocol; a runtime-registered
/// format falls back to the OpenAI-compatible convention (cached input
/// inside the prompt, reasoning inside the completion), which is what an
/// OpenAI-compatible endpoint reports.
pub fn token_accounting(format: &LlmFormat) -> TokenAccounting {
    match format {
        LlmFormat::Anthropic => TokenAccounting {
            cache: CacheTokenScope::SeparateFromPrompt,
            reasoning: ReasoningTokenScope::IncludedInCompletion,
        },
        LlmFormat::OpenaiChat
        | LlmFormat::OpenaiResponse
        | LlmFormat::GeminiNative
        | LlmFormat::Custom(_) => TokenAccounting {
            cache: CacheTokenScope::IncludedInPrompt,
            reasoning: match format {
                LlmFormat::GeminiNative => ReasoningTokenScope::SeparateFromCompletion,
                _ => ReasoningTokenScope::IncludedInCompletion,
            },
        },
    }
}

/// Token breakdown that means the same thing across providers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NormalizedUsage {
    /// Input tokens that were not served from the cache.
    pub input_tokens: u32,
    /// Output tokens; contains the reasoning tokens when the provider counts
    /// them inside the completion.
    pub output_tokens: u32,
    /// Reasoning tokens, reported next to the completion.
    pub reasoning_tokens: u32,
    /// Input tokens served from the cache.
    pub cache_read_tokens: u32,
    /// Input tokens written into the cache.
    pub cache_write_tokens: u32,
    /// Every token the request consumed, independent of the provider
    /// accounting convention.
    pub total_tokens: u32,
}

/// Normalize a provider usage report.
///
/// The cached input is split off the prompt according to the accounting
/// convention, and the reasoning tokens are added to the total only when
/// the provider bills them on top of the completion. Malformed reports
/// (a cache count larger than the prompt) never produce a negative input:
/// the subtraction saturates at zero.
pub fn normalize_usage(usage: &TokenUsageStats, accounting: TokenAccounting) -> NormalizedUsage {
    let cache_read = u64::from(usage.cache_read_tokens.unwrap_or(0));
    let cache_write = u64::from(usage.cache_write_tokens.unwrap_or(0));
    let reasoning = u64::from(usage.reasoning_tokens.unwrap_or(0));
    let prompt = u64::from(usage.prompt_tokens);
    let completion = u64::from(usage.completion_tokens);

    let input = match accounting.cache {
        CacheTokenScope::IncludedInPrompt => prompt.saturating_sub(cache_read + cache_write),
        CacheTokenScope::SeparateFromPrompt => prompt,
    };

    let reasoning_extra = match accounting.reasoning {
        ReasoningTokenScope::IncludedInCompletion => 0,
        ReasoningTokenScope::SeparateFromCompletion => reasoning,
    };

    let total = input + cache_read + cache_write + completion + reasoning_extra;

    NormalizedUsage {
        input_tokens: clamp_u32(input),
        output_tokens: clamp_u32(completion),
        reasoning_tokens: clamp_u32(reasoning),
        cache_read_tokens: clamp_u32(cache_read),
        cache_write_tokens: clamp_u32(cache_write),
        total_tokens: clamp_u32(total),
    }
}

/// Saturating conversion of a token count into the 32-bit wire width.
fn clamp_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Context of one LLM call: everything about the call that is known
/// before the provider answers.
///
/// The gateway builds it once per request and hands it to the metrics
/// sink together with the usage report, so the sink never has to guess
/// which profile, provider or execution a request belonged to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlmCallContext {
    /// Model the profile points at.
    pub model: Option<String>,
    /// Profile the call was issued through.
    pub profile_id: Option<String>,
    /// Provider definition the profile references.
    pub provider_id: Option<String>,
    /// Wire format of the profile.
    pub format: Option<String>,
    /// Execution the call belongs to, when the caller knows it.
    pub execution_id: Option<String>,
    /// Accounting convention of the wire format.
    pub accounting: TokenAccounting,
}

impl LlmCallContext {
    /// Context of a call issued through a profile.
    pub fn from_profile(
        model: Option<String>,
        profile_id: Option<String>,
        provider_id: Option<String>,
        format: &LlmFormat,
        execution_id: Option<String>,
    ) -> Self {
        Self {
            model,
            profile_id,
            provider_id,
            format: Some(format.as_str().to_string()),
            execution_id,
            accounting: token_accounting(format),
        }
    }
}

/// One LLM call: the provider usage plus the call context.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmUsageReport {
    /// Context of the call.
    pub context: LlmCallContext,
    /// Usage as reported by the provider, with the computed costs filled in
    /// when a price is known. For an estimated request this holds the local
    /// estimate instead.
    pub usage: TokenUsageStats,
    /// True when the usage was produced locally because the provider
    /// reported none.
    pub estimated: bool,
    /// Amount the cache saved against pricing every input token at the
    /// base rate. `None` when no price is known.
    pub cache_savings: Option<f64>,
}

impl LlmUsageReport {
    /// Report of a call that carried a provider usage report.
    pub fn from_usage(context: LlmCallContext, usage: TokenUsageStats) -> Self {
        Self {
            context,
            usage,
            estimated: false,
            cache_savings: None,
        }
    }

    /// Normalized breakdown of the reported usage.
    pub fn normalized(&self) -> NormalizedUsage {
        normalize_usage(&self.usage, self.context.accounting)
    }

    /// True when the billing total came from the provider rather than the
    /// catalog estimate.
    pub fn total_from_provider(&self) -> bool {
        self.usage.total_from_provider()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(
        prompt: u32,
        completion: u32,
        reasoning: Option<u32>,
        cache_read: Option<u32>,
        cache_write: Option<u32>,
    ) -> TokenUsageStats {
        TokenUsageStats {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            reasoning_tokens: reasoning,
            cache_read_tokens: cache_read,
            cache_write_tokens: cache_write,
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: None,
            provider_total: None,
        }
    }

    #[test]
    fn openai_style_keeps_the_cache_inside_the_prompt() {
        let accounting = token_accounting(&LlmFormat::OpenaiChat);
        let normalized = normalize_usage(&usage(1000, 100, Some(30), Some(800), None), accounting);
        // 200 uncached input + 800 cached read + 100 output.
        assert_eq!(normalized.input_tokens, 200);
        assert_eq!(normalized.cache_read_tokens, 800);
        assert_eq!(normalized.output_tokens, 100);
        assert_eq!(normalized.reasoning_tokens, 30);
        // The reasoning tokens are already inside the completion.
        assert_eq!(normalized.total_tokens, 1100);
    }

    #[test]
    fn anthropic_style_bills_the_cache_next_to_the_prompt() {
        let accounting = token_accounting(&LlmFormat::Anthropic);
        let normalized = normalize_usage(&usage(10, 20, Some(8), Some(80), Some(15)), accounting);
        assert_eq!(normalized.input_tokens, 10);
        assert_eq!(normalized.cache_read_tokens, 80);
        assert_eq!(normalized.cache_write_tokens, 15);
        assert_eq!(normalized.total_tokens, 125);
    }

    #[test]
    fn gemini_style_adds_the_thoughts_on_top() {
        let accounting = token_accounting(&LlmFormat::GeminiNative);
        let normalized = normalize_usage(&usage(100, 50, Some(20), Some(40), None), accounting);
        assert_eq!(normalized.input_tokens, 60);
        assert_eq!(normalized.output_tokens, 50);
        assert_eq!(normalized.reasoning_tokens, 20);
        // 60 uncached input + 40 cached read + 50 output + 20 thoughts.
        assert_eq!(normalized.total_tokens, 170);
    }

    #[test]
    fn malformed_cache_count_never_produces_a_negative_input() {
        let accounting = token_accounting(&LlmFormat::OpenaiChat);
        let normalized = normalize_usage(&usage(10, 0, None, Some(800), None), accounting);
        assert_eq!(normalized.input_tokens, 0);
        assert_eq!(normalized.total_tokens, 800);
    }

    #[test]
    fn custom_format_falls_back_to_the_openai_convention() {
        let accounting = token_accounting(&LlmFormat::Custom("my-gateway".to_string()));
        let normalized = normalize_usage(&usage(100, 10, None, Some(60), None), accounting);
        assert_eq!(normalized.input_tokens, 40);
        assert_eq!(normalized.total_tokens, 110);
    }

    #[test]
    fn report_exposes_the_normalized_breakdown() {
        let context = LlmCallContext::from_profile(
            Some("claude-sonnet".to_string()),
            Some("default".to_string()),
            Some("anthropic".to_string()),
            &LlmFormat::Anthropic,
            Some("exec-1".to_string()),
        );
        let report = LlmUsageReport::from_usage(context, usage(10, 20, None, Some(80), Some(15)));
        assert!(!report.estimated);
        assert_eq!(report.context.format.as_deref(), Some("ANTHROPIC"));
        assert_eq!(report.normalized().total_tokens, 125);
    }
}
