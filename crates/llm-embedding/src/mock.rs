//! Feature-gated mock embedding provider for tests.
//!
//! Deterministic: vectors are derived from a stable hash of the input text,
//! so tests can assert ordering and equality without a real endpoint.

use std::collections::VecDeque;

use async_trait::async_trait;

use crate::error::{EmbeddingError, Result};
use crate::provider::{EmbeddingProvider, EmbeddingResult};

/// One scripted step: either a successful embedding or an injected error.
#[derive(Debug, Clone)]
pub enum MockEmbeddingStep {
    /// Return embeddings for the batch.
    Respond,
    /// Sleep before responding normally.
    Delayed {
        /// Delay before responding, in milliseconds.
        delay_ms: u64,
    },
    /// Fail the call with an injected error.
    Fail(EmbeddingError),
}

// A plain std Mutex is enough: no await is held across the lock.
type Shared<T> = std::sync::Mutex<T>;

/// Scripted, deterministic `EmbeddingProvider` for tests.
pub struct MockEmbeddingProvider {
    model: String,
    dimension: usize,
    /// Optional scripted step sequence, consumed in call order; when
    /// exhausted (or absent) every call responds normally.
    steps: Shared<VecDeque<MockEmbeddingStep>>,
    /// Recorded batch sizes, in call order.
    calls: Shared<Vec<usize>>,
    /// Texts of the most recent call.
    last_texts: Shared<Vec<String>>,
}

impl MockEmbeddingProvider {
    pub fn new(model: impl Into<String>, dimension: usize) -> Self {
        Self {
            model: model.into(),
            dimension,
            steps: std::sync::Mutex::new(VecDeque::new()),
            calls: std::sync::Mutex::new(Vec::new()),
            last_texts: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Queue scripted steps consumed in call order.
    pub fn with_steps(steps: Vec<MockEmbeddingStep>) -> Self {
        Self {
            model: "mock-embedding".to_string(),
            dimension: 8,
            steps: std::sync::Mutex::new(steps.into()),
            calls: std::sync::Mutex::new(Vec::new()),
            last_texts: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Overrides the vector dimension (chainable with [`Self::with_steps`]).
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Batch sizes observed so far, in call order.
    pub fn recorded_batch_sizes(&self) -> Vec<usize> {
        self.calls.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Texts of the most recent call.
    pub fn last_texts(&self) -> Vec<String> {
        self.last_texts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Normal scripted response for a batch.
    fn respond(&self, texts: &[String]) -> EmbeddingResult {
        let embeddings = texts.iter().map(|t| self.vector_for(t)).collect();
        let tokens = texts.iter().map(|t| t.len() as u64).sum::<u64>();
        EmbeddingResult {
            embeddings,
            prompt_tokens: tokens,
            total_tokens: tokens,
        }
    }

    /// Deterministic vector for one text: seeded by a stable hash of the
    /// input, normalized to unit length so similarity assertions behave.
    fn vector_for(&self, text: &str) -> Vec<f32> {        let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.as_bytes() {
            seed ^= u64::from(*byte);
            seed = seed.wrapping_mul(0x1000_0000_01b3);
        }
        let mut vec = Vec::with_capacity(self.dimension);
        let mut state = seed;
        for _ in 0..self.dimension {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map the top 32 bits to [-1.0, 1.0).
            let sample = ((state >> 32) as u32 as f64 / u32::MAX as f64) * 2.0 - 1.0;
            vec.push(sample as f32);
        }
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter().map(|v| v / norm).collect()
        } else {
            vec
        }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<EmbeddingResult> {
        self.calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(texts.len());
        *self.last_texts.lock().unwrap_or_else(|p| p.into_inner()) = texts.to_vec();
        let step = self
            .steps
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop_front();
        match step {
            Some(MockEmbeddingStep::Fail(err)) => Err(err),
            Some(MockEmbeddingStep::Delayed { delay_ms }) => {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                Ok(self.respond(texts))
            }
            _ => Ok(self.respond(texts)),
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_vectors_for_same_input() {
        let provider = MockEmbeddingProvider::new("m", 8);
        let a = provider.vector_for("hello world");
        let b = provider.vector_for("hello world");
        assert_eq!(a, b);
        let c = provider.vector_for("different");
        assert_ne!(a, c);
    }

    #[tokio::test]
    async fn scripted_failure_then_recover() {
        let provider = MockEmbeddingProvider::with_steps(vec![MockEmbeddingStep::Fail(
            EmbeddingError::Provider {
                status: 500,
                message: "boom".to_string(),
                retry_after_ms: None,
            },
        )]);
        assert!(provider.embed(&["x".to_string()]).await.is_err());
        assert!(provider.embed(&["x".to_string()]).await.is_ok());
    }

    #[tokio::test]
    async fn records_batch_sizes() {
        let provider = MockEmbeddingProvider::new("m", 4);
        provider
            .embed(&["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap_or_else(|e| panic!("embed must succeed: {e}"));
        provider.embed_one("d").await.unwrap_or_else(|e| panic!("embed_one must succeed: {e}"));
        assert_eq!(provider.recorded_batch_sizes(), vec![3, 1]);
    }
}
