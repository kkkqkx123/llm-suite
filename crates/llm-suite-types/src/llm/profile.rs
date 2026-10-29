use serde::{Deserialize, Serialize};

/// Streaming options for LLM requests (e.g. whether to include usage in stream).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Prompt caching configuration for providers with explicit cache breakpoints
/// (Anthropic `cache_control`).
///
/// Presence of this field enables caching. Breakpoints are placed on the last
/// tool declaration, the system block and the last message content block
/// (at most 3 of the 4 API-allowed breakpoints).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmPromptCaching {
    /// Cache entry time-to-live: `"5m"` (provider default) or `"1h"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// Per-request prompt-caching override carried on `LlmRequest`.
///
/// When present it replaces the profile-level `prompt_caching` decision for
/// that single request — including explicitly disabling caching for a profile
/// that enables it. Currently honored by the Anthropic codec.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmRequestCaching {
    /// Enable or disable prompt caching for this request.
    pub enabled: bool,
    /// Cache entry time-to-live: `"5m"` (provider default) or `"1h"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmProfile {
    pub id: String,
    pub name: String,
    /// Wire protocol format used to encode requests for this profile.
    pub format: super::LlmFormat,
    /// Optional reference to an `LlmProviderDefinition` carrying the
    /// connection defaults (base URL, auth, headers, model discovery).
    /// Explicit fields on the profile win over the referenced definition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<super::generation::LlmGenerationParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_delay: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<crate::Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<crate::Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_protocol: Option<super::tool_call_protocol::ToolCallProtocolConfig>,
    /// Authentication type: "native" (format-specific headers) or "bearer"
    /// (Authorization: Bearer). Defaults to "native".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_type: Option<String>,
    /// Custom headers to add to every request (simple key-value map).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_headers: Option<crate::Metadata>,
    /// Custom body fields to deep-merge into the request body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_body: Option<serde_json::Value>,
    /// Whether custom body merging is enabled (default: true when custom_body present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_body_enabled: Option<bool>,
    /// Query parameters to append to the request URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_params: Option<crate::Metadata>,
    /// Streaming options (e.g. include_usage for OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<LlmStreamOptions>,
    /// Prompt caching configuration (enables Anthropic `cache_control`
    /// breakpoints when present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_caching: Option<LlmPromptCaching>,
    /// Model's maximum input context window size in tokens.
    ///
    /// This is a **model capability** property, distinct from the per-request
    /// `max_tokens` (output limit). It tells the runtime how many input tokens
    /// the model can accept before truncation or compression is needed.
    ///
    /// Common values: 128000 (GPT-4o, Gemini 2.5 Flash), 200000 (Claude 4.5),
    /// 256000 (Gemini 2.5 Pro), 1048576 (Gemini 2.5 Pro extended).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window_size: Option<u32>,
    /// Optional proxy URL (http/https/socks5) applied to the HTTP client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Hosts that bypass `proxy`: domain names (with or without a leading
    /// dot), CIDR blocks, or `*`. Wins over the provider definition list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_proxy: Option<Vec<String>>,
    /// Optional circuit breaker policy for this profile's endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub circuit_breaker: Option<super::CircuitBreakerConfig>,
    /// Optional authentication reference that outsources key resolution to
    /// the credential store. Absent keeps the static `api_key` behavior; an
    /// explicit `api_key` still wins over the referenced credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<super::ProfileAuthRef>,
    /// Optional provider compatibility opinions for this profile. Explicit
    /// fields win over the referenced provider definition, which wins over
    /// automatic detection. `None` keeps the codec defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<super::ProviderCompat>,
    /// Optional client identity headers (`User-Agent` / `HTTP-Referer` /
    /// `X-Title`). Explicit fields win per field over the referenced
    /// provider definition. `None` sends no identity header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<super::LlmIdentity>,
    /// Optional connection-level HTTP transport tuning (connect timeout,
    /// TLS opt-outs). Explicit fields win per field over the referenced
    /// provider definition. `None` keeps the default client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<super::LlmHttpTransport>,
    /// Optional per-profile environment overlay shadowing the process
    /// environment during auth resolution (keyed by variable name). Unioned
    /// with the definition map; profile keys win. Never carries secrets
    /// itself beyond what the host places there for tests or sandboxes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<std::collections::HashMap<String, String>>,
}

/// Reference from a profile to a stored credential.
///
/// The referenced credential owns the provider: once one is stored, ambient
/// environment keys are not consulted for that provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileAuthRef {
    /// Credential owner: the provider id keying the credential store.
    pub provider_id: String,
    /// Authentication method to use; absent infers from the stored
    /// credential type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<super::AuthMethod>,
}
