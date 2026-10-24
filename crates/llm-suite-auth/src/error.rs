//! Error type for the authentication layer.

use thiserror::Error;

/// Result alias for authentication operations.
pub type AuthResult<T> = Result<T, AuthError>;

/// Errors produced by credential storage, OAuth flows and auth resolution.
#[derive(Debug, Error)]
pub enum AuthError {
    /// The referenced provider has no registered authentication.
    #[error("provider '{0}' has no registered authentication")]
    ProviderNotRegistered(String),

    /// Credential store failure (read, write or delete).
    #[error("credential store failure: {0}")]
    Store(String),

    /// OAuth token refresh failed; the stored credential is preserved so the
    /// user can retry or re-login.
    #[error("oauth refresh failed: {0}")]
    Refresh(String),

    /// Interactive login failed (aborted, denied or expired).
    #[error("login failed: {0}")]
    Login(String),

    /// The stored credential does not match any registered handler.
    #[error("credential type not handled for provider '{0}'")]
    CredentialTypeNotHandled(String),

    /// The stored credential is malformed for its declared type.
    #[error("malformed credential: {0}")]
    MalformedCredential(String),

    /// Transport failure while talking to an OAuth endpoint.
    #[error("oauth transport failure: {0}")]
    Transport(String),

    /// The requested login session does not exist.
    #[error("login session '{0}' not found")]
    SessionNotFound(String),

    /// The session is not waiting for input.
    #[error("login session '{0}' is not awaiting input")]
    SessionNotAwaitingInput(String),

    /// The operation was cancelled through its abort signal.
    #[error("operation cancelled")]
    Cancelled,

    /// The loopback callback could not be served.
    #[error("loopback callback failure: {0}")]
    Loopback(String),
}
