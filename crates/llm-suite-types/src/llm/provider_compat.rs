use serde::{Deserialize, Serialize};

use super::generation::LlmMaxTokenField;

/// Narrow compatibility surface a codec actually branches on.
///
/// Every field is `Option`: `None` means "no opinion, use the codec
/// default". An explicit `Some` on the profile wins over the definition,
/// which wins over [`ProviderCompat::detect`]. New fields must be added
/// only together with a codec branch that reads them; dead metadata is
/// rejected. Current consumers: `supports_developer_role` (OpenAI Chat
/// message roles), `supports_strict_mode` (OpenAI Chat/Responses strict
/// markers), `supports_mid_convo_system_messages` (OpenAI Chat system
/// placement), `supports_long_cache_retention` (Anthropic cache ttl),
/// `max_tokens_field` (OpenAI Chat output-limit field via generation).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderCompat {
    /// Whether the OpenAI Chat wire format may send the `developer` role.
    /// Older gateways only know `system`; when `Some(false)` callers keep
    /// sending `system`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Whether the provider honors strict structured-output mode
    /// (`strict: true` on JSON schema tool/response formats).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Whether system messages may appear mid-conversation (after the
    /// first turn) instead of leading the history only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_mid_convo_system_messages: Option<bool>,
    /// Whether a long (1h) prompt-cache TTL is accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    /// Wire field used for the OpenAI Chat output limit. Falls back to
    /// the per-request `max_token_field` in `LlmGenerationParams` when
    /// set; compat only supplies the profile-wide default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<LlmMaxTokenField>,
}

impl ProviderCompat {
    /// Overlay `over` on top of `self`: `Some` values in `over` win.
    pub fn merge_over(&mut self, over: &ProviderCompat) {
        if over.supports_developer_role.is_some() {
            self.supports_developer_role = over.supports_developer_role;
        }
        if over.supports_strict_mode.is_some() {
            self.supports_strict_mode = over.supports_strict_mode;
        }
        if over.supports_mid_convo_system_messages.is_some() {
            self.supports_mid_convo_system_messages = over.supports_mid_convo_system_messages;
        }
        if over.supports_long_cache_retention.is_some() {
            self.supports_long_cache_retention = over.supports_long_cache_retention;
        }
        if over.max_tokens_field.is_some() {
            self.max_tokens_field = over.max_tokens_field;
        }
    }

    /// True when no compat opinion is expressed at all.
    pub fn is_empty(&self) -> bool {
        self.supports_developer_role.is_none()
            && self.supports_strict_mode.is_none()
            && self.supports_mid_convo_system_messages.is_none()
            && self.supports_long_cache_retention.is_none()
            && self.max_tokens_field.is_none()
    }

    /// Narrow heuristic for OpenAI-compatible endpoints.
    ///
    /// Only one signal is inferred today: local/dev gateways (localhost,
    /// `127.0.0.1`, Ollama, vLLM) historically accept only the legacy
    /// `max_tokens` field, so they default to `MaxTokens`. Every other
    /// endpoint yields no opinion and keeps the officially recommended
    /// `max_completion_tokens`. Explicit config always wins over this
    /// guess; callers log when the guess fires.
    pub fn detect(base_url: Option<&str>, format: &super::LlmFormat) -> Self {
        if !matches!(format, super::LlmFormat::OpenaiChat) {
            return Self::default();
        }
        let url = base_url.unwrap_or_default().to_ascii_lowercase();
        let looks_local = url.contains("localhost")
            || url.contains("127.0.0.1")
            || url.contains("0.0.0.0")
            || url.contains("ollama")
            || url.contains("vllm")
            || url.contains("llama.cpp")
            || url.contains("llama-cpp")
            || url.contains("localstack")
            || url.contains(".local")
            || url.contains("192.168.")
            || url.contains("10.");
        if looks_local {
            Self {
                max_tokens_field: Some(LlmMaxTokenField::MaxTokens),
                ..Default::default()
            }
        } else {
            Self::default()
        }
    }

    /// Effective output-limit field: explicit generation wins, then
    /// profile compat, then definition compat, then the codec default.
    pub fn effective_max_tokens_field(
        generation_field: Option<LlmMaxTokenField>,
        profile_compat: Option<&ProviderCompat>,
        definition_compat: Option<&ProviderCompat>,
    ) -> LlmMaxTokenField {
        generation_field
            .or_else(|| profile_compat.and_then(|c| c.max_tokens_field))
            .or_else(|| definition_compat.and_then(|c| c.max_tokens_field))
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmFormat;

    #[test]
    fn empty_by_default() {
        assert!(ProviderCompat::default().is_empty());
    }

    #[test]
    fn merge_prefers_overlay() {
        let mut base = ProviderCompat {
            supports_developer_role: Some(false),
            max_tokens_field: Some(LlmMaxTokenField::MaxCompletionTokens),
            ..Default::default()
        };
        let over = ProviderCompat {
            supports_developer_role: Some(true),
            ..Default::default()
        };
        base.merge_over(&over);
        assert_eq!(base.supports_developer_role, Some(true));
        assert_eq!(
            base.max_tokens_field,
            Some(LlmMaxTokenField::MaxCompletionTokens)
        );
    }

    #[test]
    fn local_urls_default_to_legacy_max_tokens() {
        for url in [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://ollama:11434/v1",
            "http://vllm-internal:8000/v1",
            "http://192.168.1.10:8000/v1",
        ] {
            let compat = ProviderCompat::detect(Some(url), &LlmFormat::OpenaiChat);
            assert_eq!(
                compat.max_tokens_field,
                Some(LlmMaxTokenField::MaxTokens),
                "local gateway must keep the legacy field: {url}"
            );
        }
    }

    #[test]
    fn hosted_openai_has_no_opinion() {
        let compat =
            ProviderCompat::detect(Some("https://api.openai.com/v1"), &LlmFormat::OpenaiChat);
        assert!(compat.is_empty());
    }

    #[test]
    fn hosted_providers_have_no_opinion() {
        for url in [
            "https://api.openai.com/v1",
            "https://openrouter.ai/api/v1",
            "https://api.anthropic.com/v1",
            "https://generativelanguage.googleapis.com/v1beta",
        ] {
            let compat = ProviderCompat::detect(Some(url), &LlmFormat::OpenaiChat);
            assert!(
                compat.is_empty(),
                "hosted endpoint must stay neutral: {url}"
            );
        }
    }

    #[test]
    fn detect_only_applies_to_openai_chat() {
        let compat =
            ProviderCompat::detect(Some("http://localhost:11434/v1"), &LlmFormat::Anthropic);
        assert!(compat.is_empty());
    }

    #[test]
    fn effective_field_prefers_generation_then_profile_then_definition() {
        let profile = ProviderCompat {
            max_tokens_field: Some(LlmMaxTokenField::MaxTokens),
            ..Default::default()
        };
        let definition = ProviderCompat {
            max_tokens_field: Some(LlmMaxTokenField::MaxCompletionTokens),
            ..Default::default()
        };
        assert_eq!(
            ProviderCompat::effective_max_tokens_field(
                Some(LlmMaxTokenField::MaxCompletionTokens),
                Some(&profile),
                Some(&definition)
            ),
            LlmMaxTokenField::MaxCompletionTokens
        );
        assert_eq!(
            ProviderCompat::effective_max_tokens_field(None, Some(&profile), Some(&definition)),
            LlmMaxTokenField::MaxTokens
        );
        assert_eq!(
            ProviderCompat::effective_max_tokens_field(None, None, Some(&definition)),
            LlmMaxTokenField::MaxCompletionTokens
        );
        assert_eq!(
            ProviderCompat::effective_max_tokens_field(None, None, None),
            LlmMaxTokenField::MaxCompletionTokens
        );
    }
}
