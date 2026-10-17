//! Error type for the embedding crate.
//!
//! Re-exports the shared capability-layer [`ProviderError`] from `llm-suite-common`
//! under the embedding name; the enum previously lived here and was
//! structurally identical to the rerank one.

pub use llm_suite_common::ProviderError as EmbeddingError;

/// Result alias for embedding operations.
pub type Result<T> = std::result::Result<T, EmbeddingError>;
