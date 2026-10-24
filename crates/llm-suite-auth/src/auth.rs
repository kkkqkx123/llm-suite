//! Provider-owned authentication: each provider declares how API keys and
//! OAuth logins resolve into per-request auth.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::credential::{AuthMethod, Credential, OAuthCredential};
use crate::error::AuthResult;
use crate::interaction::AuthPrompt;

/// Ambient context for key resolution (process environment, well-known
/// files). Injectable for tests and restricted hosts.
#[async_trait]
pub trait AuthContext: Send + Sync + std::fmt::Debug {
    /// Read an environment variable.
    async fn env(&self, name: &str) -> Option<String>;
    /// Check whether a file exists. Supports a leading `~`.
    async fn file_exists(&self, path: &str) -> bool;
}

/// Process-environment backed context.
#[derive(Debug, Default)]
pub struct StdAuthContext;

#[async_trait]
impl AuthContext for StdAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        let expanded = expand_tilde(path);
        std::path::Path::new(&expanded).exists()
    }
}

fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return format!("{}/{}", home_dir(), rest);
    }
    path.to_string()
}

fn home_dir() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default()
}

/// Resolved request auth for one call.
///
/// Anything that cannot be expressed as `api_key`, `headers` or `base_url`
/// is provider config, not auth.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelAuth {
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub base_url: Option<String>,
    /// True when access is backed by a subscription rather than metered
    /// billing; hosts use this to suppress per-token cost accounting.
    pub is_subscription: bool,
    /// Human-readable label for status UIs ("ANTHROPIC_API_KEY", "OAuth", ...).
    pub source: Option<String>,
}

/// Input shared by resolve implementations.
pub struct ResolveInput<'a> {
    pub ctx: &'a Arc<dyn AuthContext>,
    pub credential: Option<&'a Credential>,
}

/// API-key authentication: stored key plus ambient sources (environment
/// variables, credential files). Login is always an explicit paste of the
/// key; no provider-specific interactive handshake exists.
#[async_trait]
pub trait ApiKeyAuth: Send + Sync + std::fmt::Debug {
    /// Display name, e.g. "Anthropic API key".
    fn name(&self) -> String;

    /// The prompt to show when collecting a key interactively.
    fn login_prompt(&self) -> AuthPrompt {
        AuthPrompt::Secret {
            message: format!("Paste your {}:", self.name()),
            placeholder: None,
        }
    }

    /// Resolve auth from the stored credential and/or ambient sources.
    /// `None` means not configured.
    async fn resolve(&self, input: ResolveInput<'_>) -> AuthResult<Option<ModelAuth>>;
}

/// OAuth authentication backed by a login flow plus token refresh.
#[async_trait]
pub trait OAuthAuth: Send + Sync + std::fmt::Debug {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    fn name(&self) -> String;

    /// Whether access through this auth method is backed by a subscription.
    fn is_subscription(&self) -> bool {
        false
    }

    /// Selector label for the login option, e.g. "Sign in with Claude Pro/Max".
    fn login_label(&self) -> Option<String> {
        None
    }

    /// Begin the login flow. Returns the pending flow state the registry
    /// drives; implementations must not block on the user here.
    async fn begin_login(&self) -> AuthResult<crate::session::LoginFlow>;

    /// Exchange the refresh token. Network call; fails with
    /// [`AuthError::Refresh`] on invalid_grant and friends.
    async fn refresh(&self, credential: &OAuthCredential) -> AuthResult<OAuthCredential>;

    /// Side-effect-free derivation of request auth from a valid credential.
    /// Covers per-credential base URLs and derived headers.
    async fn to_auth(&self, credential: &OAuthCredential) -> AuthResult<ModelAuth>;
}

/// A provider's registered authentication surface. At least one of
/// `api_key` / `oauth` must be present.
#[derive(Default)]
pub struct ProviderAuth {
    pub api_key: Option<Arc<dyn ApiKeyAuth>>,
    pub oauth: Option<Arc<dyn OAuthAuth>>,
}

/// Non-secret metadata about one registered auth surface, for status UIs
/// and login option pickers. Never carries keys or tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthSurfaceInfo {
    /// Provider id.
    pub id: String,
    /// Display name of the active method, e.g. "Anthropic API key".
    pub display_name: String,
    /// Which method the surface registers.
    pub method: AuthMethod,
    /// Selector label when the method is OAuth, e.g. "Sign in with Claude".
    pub login_label: Option<String>,
    /// True when OAuth access is subscription-backed.
    pub is_subscription: bool,
}

impl ProviderAuth {
    pub fn api_key_only(auth: Arc<dyn ApiKeyAuth>) -> Self {
        Self {
            api_key: Some(auth),
            oauth: None,
        }
    }

    pub fn oauth_only(auth: Arc<dyn OAuthAuth>) -> Self {
        Self {
            api_key: None,
            oauth: Some(auth),
        }
    }

    /// True when at least one method is registered.
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some() || self.oauth.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::ApiKeyCredential;

    #[derive(Debug, Default)]
    struct EnvKeyAuth;

    #[async_trait]
    impl ApiKeyAuth for EnvKeyAuth {
        fn name(&self) -> String {
            "Acme API key".to_string()
        }

        async fn resolve(&self, input: ResolveInput<'_>) -> AuthResult<Option<ModelAuth>> {
            if let Some(cred) = input.credential {
                if let Some(key) = &cred.as_api_key().ok().and_then(|c| c.key.clone()) {
                    return Ok(Some(ModelAuth {
                        api_key: Some(key.clone()),
                        source: Some("stored credential".to_string()),
                        ..Default::default()
                    }));
                }
            }
            match input.ctx.env("ACME_API_KEY").await {
                Some(key) => Ok(Some(ModelAuth {
                    api_key: Some(key),
                    source: Some("ACME_API_KEY".to_string()),
                    ..Default::default()
                })),
                None => Ok(None),
            }
        }
    }

    #[derive(Debug, Default)]
    struct TestContext;

    #[async_trait]
    impl AuthContext for TestContext {
        async fn env(&self, name: &str) -> Option<String> {
            (name == "ACME_API_KEY").then(|| "sk-ambient".to_string())
        }
        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn ambient_env_resolution() {
        let auth = EnvKeyAuth;
        let ctx: Arc<dyn AuthContext> = Arc::new(TestContext);
        let resolved = auth
            .resolve(ResolveInput {
                ctx: &ctx,
                credential: None,
            })
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("sk-ambient"));
        assert_eq!(resolved.source.as_deref(), Some("ACME_API_KEY"));
        assert!(!resolved.is_subscription);
    }

    #[tokio::test]
    async fn stored_key_beats_ambient() {
        let auth = EnvKeyAuth;
        let ctx: Arc<dyn AuthContext> = Arc::new(TestContext);
        let stored = Credential::ApiKey(ApiKeyCredential {
            key: Some("sk-stored".to_string()),
            env: Default::default(),
        });
        let resolved = auth
            .resolve(ResolveInput {
                ctx: &ctx,
                credential: Some(&stored),
            })
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("sk-stored"));
    }

    #[test]
    fn api_key_login_is_paste_only_with_provider_prompt() {
        let auth = EnvKeyAuth;
        assert!(matches!(
            auth.login_prompt(),
            crate::interaction::AuthPrompt::Secret { .. }
        ));
    }
}
