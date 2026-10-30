//! OpenAI-compatible HTTP image generation provider.
//!
//! One transport covers every OpenAI-compatible images endpoint: provider
//! differences are endpoint URLs plus optional bearer keys selected through
//! [`ImagesContext`] presets, not through transport branches.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::provider::{
    ImageData, ImageGenerationProvider, ImagesContext, ImagesOptions, ImagesResult,
};

/// Provider that talks to any OpenAI-compatible `/images/generations` endpoint.
pub struct OpenAiCompatibleImagesProvider {
    context: ImagesContext,
    client: reqwest::Client,
}

impl std::fmt::Debug for OpenAiCompatibleImagesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleImagesProvider")
            .field("model", &self.context.model)
            .field("base_url", &self.context.base_url)
            .finish()
    }
}

/// Request body for the `/images/generations` endpoint.
#[derive(Debug, Serialize)]
struct ImagesRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
}

/// Response body from the `/images/generations` endpoint.
#[derive(Debug, Deserialize)]
struct ImagesResponse {
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    data: Vec<ImagesResponseData>,
}

/// One image entry in the response.
#[derive(Debug, Deserialize)]
struct ImagesResponseData {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

impl OpenAiCompatibleImagesProvider {
    /// Creates a provider; fails fast when the context is incomplete.
    pub fn new(context: ImagesContext) -> Result<Self> {
        context.validate()?;
        let client = llm_suite_proxy::build_http_client_with_options(
            context.timeout_secs,
            context.proxy.as_deref(),
            &context.no_proxy,
            &llm_suite_proxy::HttpTransportOptions {
                connect_timeout_secs: context.connect_timeout_secs,
                danger_accept_invalid_certs: context.danger_accept_invalid_certs.unwrap_or(false),
            },
        )
        .map_err(|err| llm_suite_common::ProviderError::Config(err.to_string()))?;
        Ok(Self { context, client })
    }

    /// Returns the provider context.
    pub fn context(&self) -> &ImagesContext {
        &self.context
    }

    fn build_request(&self, prompt: &str, options: &ImagesOptions) -> ImagesRequest {
        ImagesRequest {
            model: self.context.model.clone(),
            prompt: prompt.to_string(),
            size: options.size.clone(),
            quality: options.quality.clone(),
            n: options.count,
            response_format: options.response_format.clone(),
        }
    }

    fn parse_response(&self, response: ImagesResponse) -> Result<ImagesResult> {
        let images = response
            .data
            .into_iter()
            .map(|item| ImageData {
                b64_json: item.b64_json,
                url: item.url,
                revised_prompt: item.revised_prompt,
            })
            .collect::<Vec<_>>();
        Ok(ImagesResult {
            images,
            created: response.created,
        })
    }
}

#[async_trait]
impl ImageGenerationProvider for OpenAiCompatibleImagesProvider {
    async fn generate_images(&self, prompt: &str, options: &ImagesOptions) -> Result<ImagesResult> {
        if prompt.is_empty() {
            return Err(llm_suite_common::ProviderError::InvalidRequest(
                "prompt is required".to_string(),
            ));
        }
        let request = self.build_request(prompt, options);
        let mut outgoing = self.client.post(&self.context.base_url).json(&request);
        if let Some(api_key) = &self.context.api_key {
            outgoing = outgoing.bearer_auth(api_key);
        }
        for (name, value) in &self.context.headers {
            outgoing = outgoing.header(name, value);
        }
        if !self.context.query_params.is_empty() {
            outgoing = outgoing.query(&self.context.query_params);
        }
        let response = outgoing
            .send()
            .await
            .map_err(|err| llm_suite_common::ProviderError::Transport(err.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            let body = response.text().await.unwrap_or_default();
            return Err(llm_suite_common::ProviderError::Provider {
                status: status.as_u16(),
                message: body,
                retry_after_ms: llm_suite_common::parse_retry_after_ms(retry_after.as_deref()),
            });
        }
        let decoded: ImagesResponse = response
            .json()
            .await
            .map_err(|err| llm_suite_common::ProviderError::Decode(err.to_string()))?;
        let result = self.parse_response(decoded)?;
        let expected = options.count.unwrap_or(1) as usize;
        if result.images.len() != expected
            && !(options.count.is_none() && !result.images.is_empty())
        {
            // Providers may return fewer images than requested under load;
            // surface the mismatch instead of silently accepting it.
            if result.images.len() > expected {
                return Err(llm_suite_common::ProviderError::Decode(format!(
                    "image count mismatch: expected {expected}, received {}",
                    result.images.len()
                )));
            }
        }
        Ok(result)
    }

    fn model_name(&self) -> &str {
        &self.context.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> ImagesContext {
        ImagesContext::new("http://example.test/images/generations", "dall-e-3")
    }

    #[test]
    fn requires_complete_context() {
        let missing = ImagesContext::new("", "dall-e-3");
        assert!(OpenAiCompatibleImagesProvider::new(missing).is_err());
    }

    #[test]
    fn rejects_an_unusable_proxy_without_echoing_the_url() {
        let context = ImagesContext {
            proxy: Some("ftp://user:secret@127.0.0.1:7890".to_string()),
            ..test_context()
        };
        let error = OpenAiCompatibleImagesProvider::new(context)
            .expect_err("unsupported scheme must fail")
            .to_string();
        assert!(error.contains("ftp"), "{error}");
        assert!(
            !error.contains("secret"),
            "credentials must not leak into the error: {error}"
        );
    }

    #[test]
    fn build_request_carries_prompt_and_options() {
        let provider = OpenAiCompatibleImagesProvider::new(test_context()).expect("valid context");
        let request = provider.build_request(
            "a cat",
            &ImagesOptions {
                size: Some("1024x1024".to_string()),
                quality: Some("hd".to_string()),
                count: Some(2),
                response_format: Some("url".to_string()),
            },
        );
        assert_eq!(request.model, "dall-e-3");
        assert_eq!(request.prompt, "a cat");
        assert_eq!(request.n, Some(2));
        assert_eq!(request.response_format.as_deref(), Some("url"));
    }

    #[test]
    fn parse_response_maps_data_entries() {
        let provider = OpenAiCompatibleImagesProvider::new(test_context()).expect("valid context");
        let result = provider
            .parse_response(ImagesResponse {
                created: Some(1),
                data: vec![ImagesResponseData {
                    b64_json: None,
                    url: Some("http://example.test/1.png".to_string()),
                    revised_prompt: Some("a cat".to_string()),
                }],
            })
            .expect("parse");
        assert_eq!(result.images.len(), 1);
        assert_eq!(
            result.images[0].url.as_deref(),
            Some("http://example.test/1.png")
        );
        assert_eq!(result.created, Some(1));
    }
}
