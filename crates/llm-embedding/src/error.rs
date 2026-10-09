//! Error type for the embedding crate.

use thiserror::Error;

/// Embedding-specific error.
#[derive(Debug, Error, Clone)]
pub enum EmbeddingError {
    /// Local configuration is invalid before any request is sent.
    #[error("invalid embedding config: {0}")]
    Config(String),

    /// The request payload is invalid before it reaches the transport.
    #[error("invalid embedding request: {0}")]
    InvalidRequest(String),

    /// The remote provider returned a non-success response.
    #[error("embedding provider returned {status}: {message}")]
    Provider {
        /// HTTP status code reported by the provider.
        status: u16,
        /// Short error description from the provider response.
        message: String,
        /// Parsed `Retry-After` for 429 responses; `None` when absent.
        retry_after_ms: Option<u64>,
    },

    /// The provider response could not be decoded.
    #[error("failed to decode embedding response: {0}")]
    Decode(String),

    /// The HTTP client itself failed (network, TLS).
    #[error("embedding transport failure: {0}")]
    Transport(String),

    /// The embedding call exceeded its deadline.
    #[error("embedding request timed out")]
    Timeout,
}

/// Result alias for embedding operations.
pub type Result<T> = std::result::Result<T, EmbeddingError>;

impl From<reqwest::Error> for EmbeddingError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            return EmbeddingError::Timeout;
        }
        if let Some(status) = err.status() {
            EmbeddingError::Provider {
                status: status.as_u16(),
                message: err.to_string(),
                retry_after_ms: None,
            }
        } else {
            EmbeddingError::Transport(err.to_string())
        }
    }
}
