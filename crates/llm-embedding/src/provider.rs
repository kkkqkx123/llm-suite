//! Provider trait and result types for embeddings.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, Result};

/// Result of an embedding operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EmbeddingResult {
    /// One embedding vector per input text, in input order.
    pub embeddings: Vec<Vec<f32>>,
    /// Number of prompt tokens reported by the provider.
    pub prompt_tokens: u64,
    /// Total number of tokens reported by the provider.
    pub total_tokens: u64,
}

/// One embedding input item: text or an image reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Plain text input.
    Text(String),
    /// Image input, referenced by URL or data URI; serialized in the OpenAI
    /// `input_image` content-part form for multimodal embedding endpoints.
    Image {
        #[serde(rename = "type")]
        kind: String,
        image_url: ImageUrlPayload,
    },
}

/// The `image_url` payload of [`EmbeddingInput::Image`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrlPayload {
    pub url: String,
}

impl EmbeddingInput {
    /// Builds an image input from a URL or data URI.
    pub fn image(url: impl Into<String>) -> Self {
        EmbeddingInput::Image {
            kind: "image_url".to_string(),
            image_url: ImageUrlPayload { url: url.into() },
        }
    }

    /// Returns the text when this input is plain text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            EmbeddingInput::Text(text) => Some(text),
            EmbeddingInput::Image { .. } => None,
        }
    }
}

/// Port for text-to-vector embedding capability.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embeds a batch of texts, returning vectors in input order.
    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResult>;

    /// Embeds a mixed batch of text and image inputs, returning vectors in
    /// input order. The default implementation degrades to [`Self::embed`]
    /// when every input is text and rejects image inputs otherwise.
    async fn embed_multimodal(&self, inputs: &[EmbeddingInput]) -> Result<EmbeddingResult> {
        if inputs.iter().all(|item| item.as_text().is_some()) {
            let texts: Vec<String> = inputs
                .iter()
                .filter_map(|item| item.as_text())
                .map(str::to_string)
                .collect();
            return self.embed(&texts).await;
        }
        Err(EmbeddingError::InvalidRequest(
            "provider does not support multimodal (image) embedding inputs".into(),
        ))
    }

    /// Embeds a single text.
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let result = self.embed(&[text.to_string()]).await?;
        result
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::Decode("provider returned no embedding".into()))
    }

    /// Expected vector dimension of this provider.
    fn dimension(&self) -> usize;

    /// Model name served by this provider.
    fn model_name(&self) -> &str;
}
