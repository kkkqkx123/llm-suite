//! Error type for the decision crate.

pub use llm_suite_common::ProviderError as DecisionError;

/// Result alias for decision operations.
pub type Result<T> = std::result::Result<T, DecisionError>;
