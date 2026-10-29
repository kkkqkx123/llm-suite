//! Error type for the images crate.

pub use llm_suite_common::ProviderError as ImageGenerationError;

/// Result alias for image generation operations.
pub type Result<T> = std::result::Result<T, ImageGenerationError>;
