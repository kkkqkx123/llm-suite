//! Configuration for embedding providers.
//!
//! Aligned with linkrs `linkrs-embedding`: the API key is optional (local
//! servers such as Ollama need none), the expected vector dimension is part
//! of the config so misconfigured deployments fail fast, and a preprocessor
//! selects query/document prompt prefixes.
//!
//! Two distinct dimension concepts live in this config and must not be
//! conflated:
//! - [`EmbeddingConfig::dimension`] is a *local* value: the expected output
//!   size used for startup validation and response checking. It is never
//!   sent to the API.
//! - [`EmbeddingConfig::request_dimensions`] is a *wire* parameter: the
//!   OpenAI `dimensions` request field for models supporting output
//!   truncation. Fixed-dimension models (BAAI/bge-m3, all-minilm, ...) reject
//!   it, so it stays absent unless explicitly configured.
//!
//! General rule: any config field that is purely a local expectation must
//! never leak into a request body; wire-only parameters get their own field.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::EmbeddingError;
use crate::preprocessor::PreprocessorConfig;

/// Configuration for an OpenAI-compatible embedding endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Full embeddings endpoint URL (e.g. `https://api.openai.com/v1/embeddings`).
    pub base_url: String,
    /// Bearer API key; absent for keyless local servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Embedding model name (e.g. `text-embedding-3-small`).
    pub model: String,
    /// Request timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Expected vector dimension; required so startup validates the wiring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<usize>,
    /// `dimensions` request parameter sent to the API for models that
    /// support output truncation (e.g. OpenAI `text-embedding-3-*`).
    /// Absent for fixed-dimension models (e.g. BAAI/bge-m3), which reject
    /// the parameter with HTTP 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_dimensions: Option<usize>,
    /// Text preprocessing applied before sending the request.
    #[serde(default)]
    pub preprocessor: PreprocessorConfig,
    /// Proxy URL (http/https/socks4/socks4a/socks5/socks5h); absent for
    /// direct connections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Hosts that bypass `proxy`: domain names (with or without a leading
    /// dot), CIDR blocks, or `*`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub no_proxy: Vec<String>,
    /// Extra HTTP headers sent with every request.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: std::collections::HashMap<String, String>,
    /// Extra query parameters appended to every request URL.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub query_params: std::collections::HashMap<String, String>,
}

fn default_timeout_secs() -> u64 {
    30
}

impl EmbeddingConfig {
    /// Creates a config for an explicit endpoint URL and model.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            model: model.into(),
            timeout_secs: default_timeout_secs(),
            dimension: None,
            request_dimensions: None,
            preprocessor: PreprocessorConfig::default(),
            proxy: None,
            no_proxy: Vec::new(),
            headers: std::collections::HashMap::new(),
            query_params: std::collections::HashMap::new(),
        }
    }

    /// OpenAI endpoint preset (`https://api.openai.com/v1/embeddings`).
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.openai.com/v1/embeddings", model).with_api_key(api_key)
    }

    /// Ollama endpoint preset (`http://localhost:11434/api/embed`).
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new("http://localhost:11434/api/embed", model)
    }

    /// Azure OpenAI endpoint preset; `resource_url` already contains the
    /// deployment path and `api-version` query (e.g.
    /// `https://{resource}.openai.azure.com/openai/deployments/{id}/embeddings?api-version=2024-02-01`).
    pub fn azure(
        resource_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(resource_url, model).with_api_key(api_key)
    }

    /// Gemini OpenAI-compatibility endpoint preset
    /// (`https://generativelanguage.googleapis.com/v1beta/openai/embeddings`).
    pub fn gemini(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(
            "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
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

    /// Sets the expected vector dimension.
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Sets the `dimensions` request parameter sent to the API (only for
    /// models supporting output truncation).
    pub fn with_request_dimensions(mut self, dimensions: usize) -> Self {
        self.request_dimensions = Some(dimensions);
        self
    }

    /// Sets the text preprocessor.
    pub fn with_preprocessor(mut self, preprocessor: PreprocessorConfig) -> Self {
        self.preprocessor = preprocessor;
        self
    }

    /// Sets the proxy URL (http/https/socks5).
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self
    }

    /// Validates the wiring before any request is sent.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.base_url.is_empty() {
            return Err(EmbeddingError::Config("base_url is required".into()));
        }
        if self.model.is_empty() {
            return Err(EmbeddingError::Config("model is required".into()));
        }
        url::Url::parse(&self.base_url)
            .map_err(|err| EmbeddingError::Config(format!("invalid base_url: {err}")))?;
        match self.dimension {
            Some(dimension) if dimension > 0 => Ok(()),
            _ => Err(EmbeddingError::Config(
                "dimension is required for startup validation; set it with EmbeddingConfig::with_dimension()".into(),
            )),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/api/embed".into(),
            api_key: None,
            model: "all-minilm".into(),
            timeout_secs: default_timeout_secs(),
            dimension: None,
            request_dimensions: None,
            preprocessor: PreprocessorConfig::default(),
            proxy: None,
            no_proxy: Vec::new(),
            headers: std::collections::HashMap::new(),
            query_params: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::PreprocessorConfig;

    #[test]
    fn presets_point_at_expected_endpoints() {
        assert!(EmbeddingConfig::openai("sk-test", "text-embedding-3-small")
            .base_url
            .contains("api.openai.com"));
        assert!(EmbeddingConfig::ollama("nomic-embed-text")
            .base_url
            .contains("11434"));
        assert!(EmbeddingConfig::gemini("key", "text-embedding-004")
            .base_url
            .contains("generativelanguage"));
        let azure =
            EmbeddingConfig::azure("https://example.openai.azure.com/x", "key", "deployment");
        assert_eq!(azure.api_key.as_deref(), Some("key"));
    }

    #[test]
    fn validate_accepts_complete_config() {
        let config =
            EmbeddingConfig::new("http://localhost:11434/api/embed", "model").with_dimension(384);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_missing_fields() {
        let missing_url = EmbeddingConfig::new("", "model").with_dimension(384);
        assert!(missing_url.validate().is_err());

        let missing_model = EmbeddingConfig::new("http://example.com", "").with_dimension(384);
        assert!(missing_model.validate().is_err());

        let bad_url = EmbeddingConfig::new("not-a-url", "model").with_dimension(384);
        assert!(bad_url
            .validate()
            .unwrap_err()
            .to_string()
            .contains("base_url"));

        let missing_dimension = EmbeddingConfig::new("http://example.com", "model");
        assert!(missing_dimension
            .validate()
            .unwrap_err()
            .to_string()
            .contains("dimension"));
    }

    #[test]
    fn builder_sets_preprocessor() {
        let config = EmbeddingConfig::new("http://example.com", "model").with_preprocessor(
            PreprocessorConfig::Prefix {
                prefix: "search_document: ".into(),
            },
        );
        assert!(matches!(
            config.preprocessor,
            PreprocessorConfig::Prefix { .. }
        ));
    }
}
