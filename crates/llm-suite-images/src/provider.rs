//! Image model, context, options and provider port.
//!
//! `ImageModel` names a catalog entry; `ImagesContext` carries the resolved
//! endpoint plus credential for one provider; `ImagesOptions` carries the
//! per-call generation knobs. The provider trait is stateful like the
//! embedding provider: the context is bound at construction time and each
//! call supplies only the prompt plus options.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Lightweight descriptor of an image model in the catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageModel {
    /// Model id as sent to the provider (e.g. `dall-e-3`).
    pub id: String,
    /// Provider id owning the model (e.g. `builtin-openai`).
    pub provider_id: String,
}

impl ImageModel {
    /// Descriptor for a provider-owned model id.
    pub fn new(id: impl Into<String>, provider_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            provider_id: provider_id.into(),
        }
    }
}

/// Resolved endpoint plus credential for image generation.
///
/// Mirrors the embedding `EmbeddingConfig` shape under the design-doc name:
/// the base images URL, the bearer key (absent for keyless local servers),
/// the model served there and the transport knobs. Presets cover the
/// well-known endpoints; anything else uses [`ImagesContext::new`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesContext {
    /// Full images endpoint URL (e.g. `https://api.openai.com/v1/images/generations`).
    pub base_url: String,
    /// Bearer API key; absent for keyless local servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Image model served at the endpoint (e.g. `dall-e-3`).
    pub model: String,
    /// Request timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Proxy URL (http/https/socks5); absent for direct connections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Hosts that bypass `proxy`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub no_proxy: Vec<String>,
    /// Extra HTTP headers sent with every request.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub headers: std::collections::HashMap<String, String>,
    /// Extra query parameters appended to every request URL.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub query_params: std::collections::HashMap<String, String>,
    /// TCP/TLS connect timeout in seconds; `None` keeps the default.
    /// Mirrors the chat profile transport knob so images calls observe the
    /// same connection policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Accept invalid TLS certificates (self-signed local gateways).
    /// Explicit opt-in only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub danger_accept_invalid_certs: Option<bool>,
}

fn default_timeout_secs() -> u64 {
    60
}

impl ImagesContext {
    /// Context for an explicit endpoint URL and model.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            model: model.into(),
            timeout_secs: default_timeout_secs(),
            proxy: None,
            no_proxy: Vec::new(),
            headers: std::collections::HashMap::new(),
            query_params: std::collections::HashMap::new(),
            connect_timeout_secs: None,
            danger_accept_invalid_certs: None,
        }
    }

    /// OpenAI endpoint preset (`https://api.openai.com/v1/images/generations`).
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.openai.com/v1/images/generations", model)
            .with_api_key(api_key)
    }

    /// OpenRouter endpoint preset (`https://openrouter.ai/api/v1/images/generations`).
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(
            "https://openrouter.ai/api/v1/images/generations",
            model,
        )
        .with_api_key(api_key)
    }

    /// Sets the bearer API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the request timeout in seconds.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Sets the proxy URL.
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self
    }

    /// Sets the TCP/TLS connect timeout in seconds.
    pub fn with_connect_timeout(mut self, secs: u64) -> Self {
        self.connect_timeout_secs = Some(secs);
        self
    }

    /// Validates the wiring before any request is sent.
    pub fn validate(&self) -> Result<()> {
        if self.base_url.is_empty() {
            return Err(llm_suite_common::ProviderError::Config(
                "base_url is required".to_string(),
            ));
        }
        if self.model.is_empty() {
            return Err(llm_suite_common::ProviderError::Config(
                "model is required".to_string(),
            ));
        }
        url::Url::parse(&self.base_url).map_err(|err| {
            llm_suite_common::ProviderError::Config(format!("invalid base_url: {err}"))
        })?;
        Ok(())
    }
}

/// Per-call image generation knobs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImagesOptions {
    /// Image size (e.g. `1024x1024`); provider-specific values pass through.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Quality hint (e.g. `standard`, `hd`); provider-specific values pass through.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// Number of images to generate (defaults to 1 when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// Response format: `url` or `b64_json`. Defaults to `url` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

impl ImagesOptions {
    /// Options requesting one image with provider defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the image size.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the quality hint.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    /// Sets the image count.
    pub fn with_count(mut self, count: u32) -> Self {
        self.count = Some(count);
        self
    }
}

/// One generated image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageData {
    /// Base64-encoded image bytes (when `response_format = "b64_json"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// Image URL (when `response_format = "url"` or the provider returns one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Provider-rewritten prompt, when the provider reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// Result of an image generation call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImagesResult {
    /// Generated images in provider order; empty only for an empty request.
    pub images: Vec<ImageData>,
    /// Provider creation timestamp, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
}

/// Port for image generation capability.
#[async_trait]
pub trait ImageGenerationProvider: Send + Sync {
    /// Generates images for a text prompt.
    async fn generate_images(
        &self,
        prompt: &str,
        options: &ImagesOptions,
    ) -> Result<ImagesResult>;

    /// Model name served by this provider.
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_presets_point_at_expected_endpoints() {
        assert!(ImagesContext::openai("sk-test", "dall-e-3")
            .base_url
            .contains("api.openai.com"));
        assert!(ImagesContext::openrouter("sk-test", "openai/dall-e-3")
            .base_url
            .contains("openrouter.ai"));
    }

    #[test]
    fn context_validation_rejects_incomplete_wiring() {
        let missing_url = ImagesContext::new("", "dall-e-3");
        assert!(missing_url.validate().is_err());
        let missing_model = ImagesContext::new("http://example.test", "");
        assert!(missing_model.validate().is_err());
        let bad_url = ImagesContext::new("not-a-url", "dall-e-3");
        assert!(bad_url.validate().is_err());
    }

    #[test]
    fn options_default_is_empty() {
        let options = ImagesOptions::new();
        assert!(options.size.is_none());
        assert!(options.count.is_none());
    }

    #[test]
    fn image_model_carries_provider_ownership() {
        let model = ImageModel::new("dall-e-3", "builtin-openai");
        assert_eq!(model.id, "dall-e-3");
        assert_eq!(model.provider_id, "builtin-openai");
    }
}
