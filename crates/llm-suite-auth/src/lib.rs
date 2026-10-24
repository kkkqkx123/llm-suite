//! Credential storage, OAuth login/refresh primitives and provider-owned
//! authentication for the llm-suite crates.
//!
//! The crate is provider-agnostic and host-independent: endpoints, client
//! ids and scopes are supplied by the provider implementation that owns a
//! login. Resolution order for every request is: an explicit key wins, then
//! the stored credential (which owns its provider), then ambient sources.
//! OAuth refresh runs inside the store's serialized write path so
//! concurrent requests cannot double-refresh a rotated token, and a failed
//! refresh preserves the stored credential.
//!
//! The dynamic dispatch on [`ApiKeyAuth`] / [`OAuthAuth`] is the extension
//! point that lets hosts and plugins register provider-specific
//! authentication without this crate knowing any provider.

pub mod auth;
pub mod credential;
pub mod error;
pub mod interaction;
pub mod oauth;
pub mod registry;
pub mod resolve;
pub mod session;
pub mod store;

#[cfg(feature = "file-store")]
pub mod file_store;

pub use auth::{
    ApiKeyAuth, AuthContext, AuthSurfaceInfo, ModelAuth, OAuthAuth, ProviderAuth, ResolveInput,
    StdAuthContext,
};
pub use credential::{ApiKeyCredential, AuthMethod, Credential, CredentialInfo, OAuthCredential};
pub use error::{AuthError, AuthResult};
pub use interaction::{
    AuthEvent, AuthInfoLink, AuthInteraction, AuthPrompt, AuthPromptOption, NullInteraction,
};
pub use oauth::device_flow::{self, DeviceFlowConfig, DeviceFlowPending, DeviceFlowPoll};
pub use oauth::pkce::{self, LoopbackListener, PkceChallenge, PkceConfig, PkcePending};
pub use oauth::{credential_from_token, TokenResponse, EXPIRY_SAFETY_MARGIN_MS};
pub use registry::{AuthRegistry, ResolvedAuth};
pub use resolve::{
    resolve_provider_auth, ProviderEntry, ResolveOverrides, DEFAULT_MIN_OAUTH_VALIDITY_MS,
};
pub use session::{
    ExternalFlowDriver, ExternalPending, ExternalState, ExternalStep, LoginFlow, LoginSession,
    LoginState,
};
pub use store::{CredentialStore, MemoryCredentialStore, ModifyFn};

#[cfg(feature = "file-store")]
pub use file_store::FileCredentialStore;
