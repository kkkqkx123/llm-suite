//! Image generation capability for the llm-suite group.
//!
//! Mirrors the embedding/decision shape: a provider trait plus an
//! OpenAI-compatible HTTP transport covering every OpenAI-compatible
//! images endpoint (OpenAI, OpenRouter, Azure, self-hosted gateways).
//! The transport posts to `{base}/images/generations` with the OpenAI
//! images contract; provider differences are endpoint URLs plus optional
//! bearer keys selected through [`ImagesContext`] presets.

pub mod error;
#[cfg(feature = "mock")]
pub mod mock;
pub mod openai_compatible;
pub mod provider;

pub use error::{ImageGenerationError, Result};
pub use openai_compatible::OpenAiCompatibleImagesProvider;
pub use provider::{
    ImageData, ImageGenerationProvider, ImageModel, ImagesContext, ImagesOptions, ImagesResult,
};

#[cfg(feature = "mock")]
pub use mock::{MockImageGenerationProvider, MockImagesStep};
