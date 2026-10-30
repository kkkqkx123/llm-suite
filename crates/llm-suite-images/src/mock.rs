//! Feature-gated mock image generation provider for tests.
//!
//! Deterministic: URLs are derived from the prompt hash, so tests can assert
//! ordering and equality without a real endpoint.

use std::collections::VecDeque;

use async_trait::async_trait;

use crate::error::Result;
use crate::provider::{ImageData, ImageGenerationProvider, ImagesOptions, ImagesResult};
use llm_suite_common::ProviderError;

/// One scripted step: either a successful generation or an injected error.
#[derive(Debug, Clone)]
pub enum MockImagesStep {
    /// Return generated images.
    Respond,
    /// Fail the call with an injected error.
    Fail(ProviderError),
}

type Shared<T> = std::sync::Mutex<T>;

/// Scripted, deterministic [`ImageGenerationProvider`] for tests.
pub struct MockImageGenerationProvider {
    model: String,
    steps: Shared<VecDeque<MockImagesStep>>,
    calls: Shared<Vec<String>>,
}

impl MockImageGenerationProvider {
    /// Provider serving `model` with default success behavior.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            steps: std::sync::Mutex::new(VecDeque::new()),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Queue scripted steps consumed in call order.
    pub fn with_steps(steps: Vec<MockImagesStep>) -> Self {
        Self {
            model: "mock-images".to_string(),
            steps: std::sync::Mutex::new(steps.into()),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Prompts observed so far, in call order.
    pub fn recorded_prompts(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }
}

#[async_trait]
impl ImageGenerationProvider for MockImageGenerationProvider {
    async fn generate_images(&self, prompt: &str, options: &ImagesOptions) -> Result<ImagesResult> {
        self.calls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(prompt.to_string());
        let step = self
            .steps
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .pop_front()
            .unwrap_or(MockImagesStep::Respond);
        match step {
            MockImagesStep::Fail(error) => Err(error),
            MockImagesStep::Respond => {
                let count = options.count.unwrap_or(1).max(1) as usize;
                let images = (0..count)
                    .map(|index| ImageData {
                        b64_json: None,
                        url: Some(format!("mock://{}/{index}", prompt.replace(' ', "-"))),
                        revised_prompt: None,
                    })
                    .collect();
                Ok(ImagesResult {
                    images,
                    created: None,
                })
            }
        }
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn responds_with_deterministic_urls() {
        let provider = MockImageGenerationProvider::new("mock-images");
        let result = provider
            .generate_images("a cat", &ImagesOptions::new().with_count(2))
            .await
            .expect("generate");
        assert_eq!(result.images.len(), 2);
        assert_eq!(provider.recorded_prompts(), vec!["a cat".to_string()]);
    }

    #[tokio::test]
    async fn scripted_failure_propagates() {
        let provider = MockImageGenerationProvider::with_steps(vec![MockImagesStep::Fail(
            ProviderError::InvalidRequest("boom".to_string()),
        )]);
        let error = provider
            .generate_images("a cat", &ImagesOptions::new())
            .await
            .expect_err("must fail");
        assert!(error.to_string().contains("boom"));
    }
}
