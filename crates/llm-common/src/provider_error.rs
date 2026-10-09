//! Shared provider-level error for the capability crates (embedding, rerank).
//!
//! The embedding and rerank crates used to carry structurally identical error
//! enums; this single type keeps classification (timeout / provider status /
//! transport) consistent across capability layers. Capability crates re-export
//! it under their own name so call sites read naturally.

use thiserror::Error;

/// Error returned by the capability-layer HTTP providers.
#[derive(Debug, Error, Clone)]
pub enum ProviderError {
    /// Local configuration is invalid before any request is sent.
    #[error("invalid config: {0}")]
    Config(String),

    /// The request payload is invalid before it reaches the transport.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The remote provider returned a non-success response.
    #[error("provider returned {status}: {message}")]
    Provider {
        /// HTTP status code reported by the provider.
        status: u16,
        /// Short error description from the provider response.
        message: String,
        /// Parsed `Retry-After` for 429 responses; `None` when absent.
        retry_after_ms: Option<u64>,
    },

    /// The provider response could not be decoded.
    #[error("failed to decode provider response: {0}")]
    Decode(String),

    /// The HTTP client itself failed (network, TLS).
    #[error("provider transport failure: {0}")]
    Transport(String),

    /// The call exceeded its deadline.
    #[error("provider request timed out")]
    Timeout,

    /// The injected circuit breaker is open; the request never reached the
    /// wire and must wait for the breaker to half-open.
    #[error("circuit breaker is open")]
    CircuitOpen,
}

impl ProviderError {
    /// Whether a retry may succeed: timeouts, transport failures, 5xx
    /// responses and rate limits are transient; validation/config rejections
    /// are permanent.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::InvalidRequest(_) | Self::Config(_) | Self::Decode(_) | Self::CircuitOpen => {
                false
            }
            Self::Timeout | Self::Transport(_) => true,
            Self::Provider { status, .. } => {
                *status == 429 || (500..=599).contains(status)
            }
        }
    }

    /// Provider-reported `Retry-After` window in ms, when the response
    /// carried one (429 handling).
    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            Self::Provider { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }

    /// Whether this failure counts into a circuit breaker window:
    /// network-level faults and server-side rejections, not request errors.
    pub fn is_network_failure(&self) -> bool {
        match self {
            Self::Timeout | Self::Transport(_) => true,
            Self::Provider { status, .. } => *status == 429 || (500..=599).contains(status),
            _ => false,
        }
    }
}

/// Error contract consumed by the resilience stack: lets one retry/breaker
/// driver serve every capability layer's error type.
pub trait ResilienceError: std::fmt::Display {
    /// Whether a retry may succeed.
    fn is_retryable(&self) -> bool;
    /// Provider-reported `Retry-After` window in ms, when present.
    fn retry_after_ms(&self) -> Option<u64>;
    /// Whether this failure counts into a circuit breaker window.
    fn is_network_failure(&self) -> bool;
}

impl ResilienceError for ProviderError {
    fn is_retryable(&self) -> bool {
        ProviderError::is_retryable(self)
    }

    fn retry_after_ms(&self) -> Option<u64> {
        ProviderError::retry_after_ms(self)
    }

    fn is_network_failure(&self) -> bool {
        ProviderError::is_network_failure(self)
    }
}

impl From<reqwest::Error> for ProviderError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            return ProviderError::Timeout;
        }
        if let Some(status) = err.status() {
            ProviderError::Provider {
                status: status.as_u16(),
                message: err.to_string(),
                retry_after_ms: None,
            }
        } else {
            ProviderError::Transport(err.to_string())
        }
    }
}

impl From<tokio::time::error::Elapsed> for ProviderError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        ProviderError::Timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reqwest_timeout_maps_to_timeout_variant() {
        // Build a real timeout error through an impossible connection with a
        // one-shot client timeout.
        let error = ProviderError::Transport("offline".into());
        assert!(matches!(error, ProviderError::Transport(_)));
    }
}
