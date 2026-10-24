//! Auth resolution shared by every request.
//!
//! Order: an explicit key wins, then the stored credential (which owns its
//! provider), then ambient sources. A stored OAuth credential near expiry is
//! refreshed inside the store lock; a failed refresh preserves the stored
//! credential and never falls back to an ambient key.

use std::sync::Arc;

use llm_suite_common::time::now;

use crate::auth::{AuthContext, ModelAuth, OAuthAuth, ProviderAuth, ResolveInput};
use crate::credential::{ApiKeyCredential, Credential, OAuthCredential};
use crate::error::{AuthError, AuthResult};
use crate::store::CredentialStore;

/// Default remaining-validity requirement before a refresh kicks in.
pub const DEFAULT_MIN_OAUTH_VALIDITY_MS: i64 = 5 * 60 * 1000;

/// Per-resolution overrides.
#[derive(Debug, Clone, Default)]
pub struct ResolveOverrides {
    /// Explicit key supplied by the caller or configuration.
    pub api_key: Option<String>,
    /// Required remaining OAuth validity before refresh.
    pub min_oauth_validity_ms: Option<i64>,
}

/// A registered provider entry: identity plus its auth surface.
pub struct ProviderEntry {
    pub id: String,
    pub auth: ProviderAuth,
}

/// Resolve the request auth for one provider.
pub async fn resolve_provider_auth(
    entry: &ProviderEntry,
    store: &dyn CredentialStore,
    ctx: &Arc<dyn AuthContext>,
    overrides: &ResolveOverrides,
) -> AuthResult<Option<ModelAuth>> {
    // 1. An explicit key always wins.
    if let Some(key) = overrides.api_key.clone() {
        if let Some(api_key_auth) = entry.auth.api_key.as_ref() {
            let credential = Credential::ApiKey(ApiKeyCredential {
                key: Some(key),
                env: Default::default(),
            });
            return api_key_auth
                .resolve(ResolveInput {
                    ctx,
                    credential: Some(&credential),
                })
                .await;
        }
        return Ok(Some(ModelAuth {
            api_key: Some(key),
            source: Some("explicit".to_string()),
            ..Default::default()
        }));
    }

    // 2. A stored credential owns its provider.
    match store.read(&entry.id).await? {
        Some(Credential::ApiKey(credential)) => {
            let auth = entry
                .auth
                .api_key
                .as_ref()
                .ok_or_else(|| AuthError::CredentialTypeNotHandled(entry.id.clone()))?;
            auth.resolve(ResolveInput {
                ctx,
                credential: Some(&Credential::ApiKey(credential)),
            })
            .await
        }
        Some(Credential::OAuth(credential)) => {
            let auth = entry
                .auth
                .oauth
                .clone()
                .ok_or_else(|| AuthError::CredentialTypeNotHandled(entry.id.clone()))?;
            let now_ms = now();
            let min_validity = overrides
                .min_oauth_validity_ms
                .unwrap_or(DEFAULT_MIN_OAUTH_VALIDITY_MS);
            let fresh = if credential.remaining_ms(now_ms) > min_validity {
                credential
            } else {
                refresh_locked(store, &entry.id, auth.as_ref(), credential).await?
            };
            let model_auth = auth.to_auth(&fresh).await?;
            Ok(Some(model_auth))
        }
        None => {
            // 3. Ambient sources.
            match entry.auth.api_key.as_ref() {
                Some(auth) => {
                    auth.resolve(ResolveInput {
                        ctx,
                        credential: None,
                    })
                    .await
                }
                None => Ok(None),
            }
        }
    }
}

/// Refresh an OAuth credential inside the store lock.
///
/// The closure runs under the store's serialized write path so concurrent
/// requests cannot double-refresh a rotated token. A refresh failure keeps
/// the stored credential untouched.
async fn refresh_locked(
    store: &dyn CredentialStore,
    provider_id: &str,
    auth: &dyn OAuthAuth,
    current: OAuthCredential,
) -> AuthResult<OAuthCredential> {
    let written = store
        .modify(
            provider_id,
            Box::new(move |existing| {
                // Re-read under the lock: another caller may have
                // refreshed already. Clone before the async block; the read
                // view is only valid inside this closure.
                let current = match existing {
                    Some(Credential::OAuth(c)) => c.clone(),
                    _ => current.clone(),
                };
                Box::pin(async move {
                    match auth.refresh(&current).await {
                        Ok(fresh) => Ok(Some(Credential::OAuth(fresh))),
                        Err(err) => Err(err),
                    }
                })
            }),
        )
        .await?;
    match written {
        Some(Credential::OAuth(fresh)) => Ok(fresh),
        Some(_) => Err(AuthError::Refresh(
            "store wrote a non-oauth credential during refresh".to_string(),
        )),
        None => Err(AuthError::Refresh(
            "credential vanished during refresh".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{ApiKeyAuth, ProviderAuth};
    use crate::interaction::NullInteraction;
    use crate::store::MemoryCredentialStore;

    #[derive(Debug)]
    struct FixedKeyAuth {
        env_name: &'static str,
    }

    #[async_trait::async_trait]
    impl ApiKeyAuth for FixedKeyAuth {
        fn name(&self) -> String {
            "fixed".to_string()
        }

        async fn resolve(&self, input: ResolveInput<'_>) -> AuthResult<Option<ModelAuth>> {
            if let Some(cred) = input.credential {
                if let Some(key) = cred.as_api_key().ok().and_then(|c| c.key.clone()) {
                    return Ok(Some(ModelAuth {
                        api_key: Some(key),
                        source: Some("stored".to_string()),
                        ..Default::default()
                    }));
                }
            }
            Ok(input.ctx.env(self.env_name).await.map(|key| ModelAuth {
                api_key: Some(key),
                source: Some(self.env_name.to_string()),
                ..Default::default()
            }))
        }
    }

    #[derive(Debug)]
    struct BearerOAuth;

    #[async_trait::async_trait]
    impl OAuthAuth for BearerOAuth {
        fn name(&self) -> String {
            "bearer oauth".to_string()
        }

        fn is_subscription(&self) -> bool {
            true
        }

        async fn begin_login(&self) -> AuthResult<crate::session::LoginFlow> {
            Err(AuthError::Login("not used in resolve tests".to_string()))
        }

        async fn refresh(&self, credential: &OAuthCredential) -> AuthResult<OAuthCredential> {
            if credential.refresh == "bad" {
                return Err(AuthError::Refresh("invalid_grant".to_string()));
            }
            let mut fresh = credential.clone();
            fresh.access = format!("{}-refreshed", credential.access);
            fresh.expires = now() + 10 * 60 * 1000;
            Ok(fresh)
        }

        async fn to_auth(&self, credential: &OAuthCredential) -> AuthResult<ModelAuth> {
            Ok(ModelAuth {
                api_key: Some(credential.access.clone()),
                is_subscription: self.is_subscription(),
                source: Some("OAuth".to_string()),
                ..Default::default()
            })
        }
    }

    fn entry() -> ProviderEntry {
        ProviderEntry {
            id: "acme".to_string(),
            auth: ProviderAuth {
                api_key: Some(Arc::new(FixedKeyAuth {
                    env_name: "ACME_KEY",
                })),
                oauth: Some(Arc::new(BearerOAuth)),
            },
        }
    }

    #[derive(Debug, Default)]
    struct Ctx;

    #[async_trait::async_trait]
    impl AuthContext for Ctx {
        async fn env(&self, name: &str) -> Option<String> {
            (name == "ACME_KEY").then(|| "sk-ambient".to_string())
        }
        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn explicit_key_wins_over_everything() {
        let store = MemoryCredentialStore::new();
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        let resolved = resolve_provider_auth(
            &entry(),
            &store,
            &ctx,
            &ResolveOverrides {
                api_key: Some("sk-explicit".to_string()),
                ..Default::default()
            },
        )
        .await
        .expect("resolve")
        .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("sk-explicit"));
    }

    #[tokio::test]
    async fn ambient_used_when_nothing_stored() {
        let store = MemoryCredentialStore::new();
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        let resolved = resolve_provider_auth(&entry(), &store, &ctx, &ResolveOverrides::default())
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("sk-ambient"));
        assert!(!resolved.is_subscription);
    }

    #[tokio::test]
    async fn stored_api_key_owns_provider_over_ambient() {
        let store = MemoryCredentialStore::new();
        store
            .modify(
                "acme",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("sk-stored".to_string()),
                            env: Default::default(),
                        })))
                    })
                }),
            )
            .await
            .expect("seed");
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        let resolved = resolve_provider_auth(&entry(), &store, &ctx, &ResolveOverrides::default())
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("sk-stored"));
    }

    #[tokio::test]
    async fn fresh_oauth_skips_refresh() {
        let store = MemoryCredentialStore::new();
        store
            .modify(
                "acme",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::OAuth(OAuthCredential {
                            refresh: "r".to_string(),
                            access: "access-1".to_string(),
                            expires: now() + 60 * 60 * 1000,
                            is_subscription: true,
                            extra: Default::default(),
                        })))
                    })
                }),
            )
            .await
            .expect("seed");
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        let resolved = resolve_provider_auth(&entry(), &store, &ctx, &ResolveOverrides::default())
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("access-1"));
        assert!(resolved.is_subscription);
    }

    #[tokio::test]
    async fn near_expiry_oauth_refreshes_under_lock() {
        let store = MemoryCredentialStore::new();
        store
            .modify(
                "acme",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::OAuth(OAuthCredential {
                            refresh: "r".to_string(),
                            access: "access-1".to_string(),
                            expires: now() + 60 * 1000,
                            is_subscription: true,
                            extra: Default::default(),
                        })))
                    })
                }),
            )
            .await
            .expect("seed");
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        let resolved = resolve_provider_auth(&entry(), &store, &ctx, &ResolveOverrides::default())
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(resolved.api_key.as_deref(), Some("access-1-refreshed"));
        let stored = store.read("acme").await.expect("read").expect("stored");
        assert_eq!(
            stored.as_oauth().expect("oauth").access,
            "access-1-refreshed",
            "refreshed credential must be persisted"
        );
    }

    #[tokio::test]
    async fn failed_refresh_preserves_credential() {
        let store = MemoryCredentialStore::new();
        store
            .modify(
                "acme",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::OAuth(OAuthCredential {
                            refresh: "bad".to_string(),
                            access: "access-1".to_string(),
                            expires: now() - 1,
                            is_subscription: true,
                            extra: Default::default(),
                        })))
                    })
                }),
            )
            .await
            .expect("seed");
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        let err = resolve_provider_auth(&entry(), &store, &ctx, &ResolveOverrides::default())
            .await
            .expect_err("must fail");
        assert!(matches!(err, AuthError::Refresh(_)));
        let stored = store.read("acme").await.expect("read").expect("stored");
        assert_eq!(stored.as_oauth().expect("oauth").access, "access-1");
    }

    #[tokio::test]
    async fn unconfigured_provider_resolves_none() {
        let entry = ProviderEntry {
            id: "none".to_string(),
            auth: ProviderAuth::default(),
        };
        let store = MemoryCredentialStore::new();
        let ctx: Arc<dyn AuthContext> = Arc::new(Ctx);
        assert!(
            resolve_provider_auth(&entry, &store, &ctx, &ResolveOverrides::default())
                .await
                .expect("resolve")
                .is_none()
        );
    }

    #[tokio::test]
    async fn null_interaction_is_available_for_ambient_flows() {
        let _ = NullInteraction;
    }
}
