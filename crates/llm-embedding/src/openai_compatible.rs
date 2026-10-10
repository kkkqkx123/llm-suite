//! OpenAI-compatible HTTP embedding provider.
//!
//! One transport covers every OpenAI-compatible endpoint (OpenAI, Gemini via
//! its compatibility layer, Azure OpenAI, Ollama, self-hosted servers):
//! provider differences are endpoint URLs plus optional bearer keys, selected
//! through [`EmbeddingConfig`] presets, not through transport branches.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::EmbeddingConfig;
use crate::error::{EmbeddingError, Result};
use crate::preprocessor::PreprocessorImpl;
use crate::provider::{EmbeddingInput, EmbeddingProvider, EmbeddingResult};

/// Provider that talks to any OpenAI-compatible `/embeddings` endpoint.
pub struct OpenAICompatibleProvider {
    config: EmbeddingConfig,
    preprocessor: PreprocessorImpl,
    client: reqwest::Client,
}

impl std::fmt::Debug for OpenAICompatibleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatibleProvider")
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .finish()
    }
}

/// Request body for the `/embeddings` endpoint.
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    /// Text strings or `input_image` content-part objects, per the OpenAI
    /// multimodal embedding input contract.
    input: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// Response body from the `/embeddings` endpoint.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<EmbeddingUsage>,
}

/// One embedding entry in the response.
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

/// Token usage reported by the provider.
#[derive(Debug, Deserialize, Default)]
struct EmbeddingUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

impl OpenAICompatibleProvider {
    /// Creates a provider; fails fast when the config is incomplete.
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        config.validate()?;
        let client = llm_proxy::build_http_client(
            config.timeout_secs,
            config.proxy.as_deref(),
            &config.no_proxy,
        )
        .map_err(|err| EmbeddingError::Config(err.to_string()))?;
        let preprocessor = PreprocessorImpl::from_config(&config.preprocessor);
        Ok(Self {
            config,
            preprocessor,
            client,
        })
    }

    /// Returns the provider configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Returns the active text preprocessor.
    pub fn preprocessor(&self) -> &PreprocessorImpl {
        &self.preprocessor
    }

    /// Builds the request body, applying preprocessing first.
    fn build_request(&self, texts: &[String]) -> EmbeddingRequest {
        let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
        EmbeddingRequest {
            model: self.config.model.clone(),
            input: self
                .preprocessor
                .process_batch(&borrowed)
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
            encoding_format: Some("float".into()),
            dimensions: self.config.request_dimensions,
        }
    }

    /// Builds the request body for a mixed text/image batch. Preprocessing
    /// applies to text inputs only; image inputs pass through untouched.
    fn build_multimodal_request(&self, inputs: &[EmbeddingInput]) -> EmbeddingRequest {
        EmbeddingRequest {
            model: self.config.model.clone(),
            input: inputs
                .iter()
                .map(|item| match item {
                    EmbeddingInput::Text(text) => {
                        serde_json::Value::String(self.preprocessor.preprocess(text))
                    }
                    image @ EmbeddingInput::Image { .. } => {
                        serde_json::to_value(image).unwrap_or(serde_json::Value::Null)
                    }
                })
                .collect(),
            encoding_format: Some("float".into()),
            dimensions: self.config.request_dimensions,
        }
    }

    /// Reassembles the response into input order and validates the contract.
    fn parse_response(&self, response: EmbeddingResponse) -> Result<EmbeddingResult> {
        let mut ordered = response.data;
        ordered.sort_by_key(|item| item.index);
        validate_embedding_data(&ordered, &self.config)?;

        let usage = response.usage.unwrap_or_default();
        Ok(EmbeddingResult {
            embeddings: ordered.into_iter().map(|item| item.embedding).collect(),
            prompt_tokens: usage.prompt_tokens,
            total_tokens: usage.total_tokens,
        })
    }

    /// Sends one request to the `/embeddings` endpoint and decodes the result.
    async fn send_request(&self, request: EmbeddingRequest) -> Result<EmbeddingResult> {
        let mut outgoing = self.client.post(&self.config.base_url).json(&request);
        if let Some(api_key) = &self.config.api_key {
            outgoing = outgoing.bearer_auth(api_key);
        }
        for (name, value) in &self.config.headers {
            outgoing = outgoing.header(name, value);
        }
        if !self.config.query_params.is_empty() {
            outgoing = outgoing.query(&self.config.query_params);
        }
        let response = outgoing.send().await?;

        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::Provider {
                status: status.as_u16(),
                message: body,
                retry_after_ms: llm_common::parse_retry_after_ms(retry_after.as_deref()),
            });
        }

        let decoded: EmbeddingResponse = response
            .json()
            .await
            .map_err(|err| EmbeddingError::Decode(err.to_string()))?;
        self.parse_response(decoded)
    }
}

/// Validates the provider response contract: exact count, sequential indexes,
/// non-empty finite vectors, and the configured dimension when set.
fn validate_embedding_data(data: &[EmbeddingData], config: &EmbeddingConfig) -> Result<()> {
    for (expected_index, item) in data.iter().enumerate() {
        if item.index != expected_index {
            return Err(EmbeddingError::Decode(format!(
                "embedding index mismatch: expected {expected_index}, received {}",
                item.index
            )));
        }
        if item.embedding.is_empty() {
            return Err(EmbeddingError::InvalidRequest(format!(
                "embedding at index {expected_index} is empty; check the dimension configuration"
            )));
        }
        if let Some(dimension) = config.dimension {
            if item.embedding.len() != dimension {
                return Err(EmbeddingError::InvalidRequest(format!(
                    "embedding dimension mismatch at index {expected_index}: expected {dimension}, received {}",
                    item.embedding.len()
                )));
            }
        }
        if item.embedding.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::InvalidRequest(format!(
                "embedding at index {expected_index} contains a non-finite value"
            )));
        }
    }
    Ok(())
}

#[async_trait]
impl EmbeddingProvider for OpenAICompatibleProvider {
    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResult> {
        if texts.is_empty() {
            return Ok(EmbeddingResult::default());
        }

        let request = self.build_request(texts);
        let result = self.send_request(request).await?;
        if result.embeddings.len() != texts.len() {
            return Err(EmbeddingError::Decode(format!(
                "embedding count mismatch: expected {}, received {}",
                texts.len(),
                result.embeddings.len()
            )));
        }
        Ok(result)
    }

    async fn embed_multimodal(&self, inputs: &[EmbeddingInput]) -> Result<EmbeddingResult> {
        if inputs.is_empty() {
            return Ok(EmbeddingResult::default());
        }

        let request = self.build_multimodal_request(inputs);
        let result = self.send_request(request).await?;
        if result.embeddings.len() != inputs.len() {
            return Err(EmbeddingError::Decode(format!(
                "embedding count mismatch: expected {}, received {}",
                inputs.len(),
                result.embeddings.len()
            )));
        }
        Ok(result)
    }

    fn dimension(&self) -> usize {
        self.config.dimension.unwrap_or(0)
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::PreprocessorConfig;

    fn test_config() -> EmbeddingConfig {
        EmbeddingConfig::new("http://example.com/embeddings", "test-model").with_dimension(2)
    }

    #[test]
    fn requires_complete_config() {
        let missing_dimension = EmbeddingConfig::new("http://example.com", "model");
        assert!(OpenAICompatibleProvider::new(missing_dimension).is_err());
    }

    #[test]
    fn accepts_a_proxy_with_a_bypass_list() {
        let config = test_config().with_proxy("http://127.0.0.1:7890");
        let config = EmbeddingConfig {
            no_proxy: vec!["localhost".to_string()],
            ..config
        };

        assert!(OpenAICompatibleProvider::new(config).is_ok());
    }

    #[test]
    fn rejects_an_unusable_proxy_without_echoing_the_url() {
        let config = test_config().with_proxy("ftp://user:secret@127.0.0.1:7890");

        let error = OpenAICompatibleProvider::new(config)
            .expect_err("unsupported scheme must fail")
            .to_string();

        assert!(error.contains("ftp"), "{error}");
        assert!(
            !error.contains("secret") && !error.contains("127.0.0.1:7890"),
            "credentials must not leak into the error: {error}"
        );
    }

    #[test]
    fn build_request_applies_preprocessor_and_dimensions() {
        let config = EmbeddingConfig::new("http://example.com", "nomic-embed")
            .with_dimension(768)
            .with_request_dimensions(512)
            .with_preprocessor(PreprocessorConfig::Prefix {
                prefix: "search_query: ".into(),
            });
        let provider = OpenAICompatibleProvider::new(config).expect("valid config");
        let request = provider.build_request(&["rust".to_string()]);
        assert_eq!(request.model, "nomic-embed");
        assert_eq!(request.input, vec!["search_query: rust".to_string()]);
        assert_eq!(request.encoding_format.as_deref(), Some("float"));
        assert_eq!(request.dimensions, Some(512));
    }

    #[test]
    fn build_multimodal_request_encodes_image_inputs() {
        let config =
            EmbeddingConfig::new("http://example.com", "bge-m3").with_dimension(1024);
        let provider = OpenAICompatibleProvider::new(config).expect("valid config");

        let request = provider.build_multimodal_request(&[
            EmbeddingInput::Text("hello".to_string()),
            EmbeddingInput::image("data:image/png;base64,AA"),
        ]);
        assert_eq!(
            request.input,
            vec![
                serde_json::json!("hello"),
                serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA"},
                }),
            ]
        );
    }

    #[test]
    fn build_request_omits_dimensions_for_fixed_dimension_models() {
        // Fixed-dimension models (e.g. BAAI/bge-m3) reject the `dimensions`
        // request parameter; only the validation dimension is configured.
        let config = EmbeddingConfig::new("http://example.com", "bge-m3").with_dimension(1024);
        let provider = OpenAICompatibleProvider::new(config).expect("valid config");
        let request = provider.build_request(&["rust".to_string()]);
        assert_eq!(request.dimensions, None);
    }

    #[test]
    fn parse_response_reorders_by_index() {
        let provider = OpenAICompatibleProvider::new(test_config()).expect("valid config");
        let response = EmbeddingResponse {
            data: vec![
                EmbeddingData {
                    index: 1,
                    embedding: vec![3.0, 4.0],
                },
                EmbeddingData {
                    index: 0,
                    embedding: vec![1.0, 2.0],
                },
            ],
            usage: None,
        };
        let result = provider.parse_response(response).expect("ordered parse");
        assert_eq!(result.embeddings[0], vec![1.0, 2.0]);
        assert_eq!(result.embeddings[1], vec![3.0, 4.0]);
    }

    #[test]
    fn parse_response_rejects_contract_violations() {
        let provider = OpenAICompatibleProvider::new(test_config()).expect("valid config");

        let duplicate = EmbeddingResponse {
            data: vec![
                EmbeddingData {
                    index: 0,
                    embedding: vec![1.0, 2.0],
                },
                EmbeddingData {
                    index: 0,
                    embedding: vec![3.0, 4.0],
                },
            ],
            usage: None,
        };
        assert!(provider.parse_response(duplicate).is_err());

        let wrong_dimension = EmbeddingResponse {
            data: vec![EmbeddingData {
                index: 0,
                embedding: vec![1.0],
            }],
            usage: None,
        };
        assert!(provider.parse_response(wrong_dimension).is_err());

        let non_finite = EmbeddingResponse {
            data: vec![EmbeddingData {
                index: 0,
                embedding: vec![f32::NAN, 2.0],
            }],
            usage: None,
        };
        assert!(provider.parse_response(non_finite).is_err());
    }

    #[test]
    fn accessors_expose_model_and_dimension() {
        let provider = OpenAICompatibleProvider::new(test_config()).expect("valid config");
        assert_eq!(provider.model_name(), "test-model");
        assert_eq!(provider.dimension(), 2);
        assert_eq!(provider.preprocessor().preprocess("x"), "x".to_string());
    }
}
