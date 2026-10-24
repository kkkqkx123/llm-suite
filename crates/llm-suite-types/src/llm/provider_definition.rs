use serde::{Deserialize, Serialize};

/// Declarative connection template for a class of LLM endpoints.
///
/// A provider definition carries connection defaults shared by multiple
/// profiles: base URL, authentication, default headers and model discovery.
/// It references one wire protocol format by name; profiles reference a
/// definition through `LlmProfile::provider_id` and override any default
/// with explicit fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmProviderDefinition {
    /// Provider id referenced by `LlmProfile::provider_id`.
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Default base URL; a profile-level `base_url` wins when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Default auth type (`native` / `bearer` / `x-api-key` /
    /// `x-goog-api-key`); a profile-level `auth_type` wins when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_type: Option<String>,
    /// Default headers; profile `custom_headers` override per key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_headers: Option<crate::Metadata>,
    /// Wire protocol format resolved through `CodecRegistry`
    /// (built-in variant or registered custom name).
    pub format: super::LlmFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_discovery: Option<super::ModelDiscovery>,
    /// Version segment used when joining the default models endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<crate::Metadata>,
    /// Default proxy URL (http/https/socks5); a profile-level `proxy` wins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Default hosts that bypass `proxy`; a profile-level `no_proxy` wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_proxy: Option<Vec<String>>,
    /// Provider-level rate limit shared by all profiles on this base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<super::RateLimitConfig>,
    /// Optional authentication declaration: marks this provider as needing a
    /// login flow and names the registered auth implementation. Absent means
    /// static key authentication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<super::ProviderAuthRef>,
}

/// Reference from a provider definition to a registered auth implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderAuthRef {
    /// Name of the registered auth implementation (built-in id or plugin
    /// contribution name).
    pub provider: String,
    /// Authentication method the declaration expects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<super::AuthMethod>,
    /// True when access through this provider is subscription-backed rather
    /// than metered; hosts suppress per-token cost accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_subscription: Option<bool>,
    /// Provider-specific auth parameters (endpoints, client id, scopes,
    /// flow). Never carries secrets; the shape is defined by the named
    /// auth implementation and passed to it verbatim at login time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<crate::Metadata>,
}
