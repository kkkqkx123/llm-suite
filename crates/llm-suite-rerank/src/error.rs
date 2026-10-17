//! Error type for the rerank crate.
//!
//! Re-exports the shared capability-layer [`ProviderError`] from `llm-suite-common`
//! under the rerank name; the enum previously lived here and was
//! structurally identical to the embedding one.

pub use llm_suite_common::ProviderError as RerankError;

/// Result alias for rerank operations.
pub type Result<T> = std::result::Result<T, RerankError>;
