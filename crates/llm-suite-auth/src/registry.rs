//! The auth registry: the single orchestration point hosts talk to.
//!
//! It owns the credential store, the registered provider auth surfaces and
//! the live login sessions, and turns a profile into a profile carrying
//! request-ready auth.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::Mutex;

use llm_suite_common::time::now;
use llm_suite_types::llm::LlmProfile;

use crate::auth::{AuthContext, ProviderAuth};
use crate::credential::{Credential, CredentialInfo};
use crate::error::{AuthError, AuthResult};
use crate::resolve::{resolve_provider_auth, ProviderEntry, ResolveOverrides};
use crate::session::{LoginSession, LoginState};
use crate::store::CredentialStore;

/// A profile plus the auth facts resolved for one request.
#[derive(Debug)]
pub struct ResolvedAuth {
    /// The profile to use for this request (auth fields injected).
    pub profile: LlmProfile,
    /// True when access is subscription-backed; hosts suppress per-token
    /// cost accounting for these requests.
    pub is_subscription: bool,
}

/// Registry of provider auth surfaces plus the credential store and the
/// live login sessions.
pub struct AuthRegistry {
    providers: DashMap<String, ProviderEntry>,
    store: Arc<dyn CredentialStore>,
    context: Arc<dyn AuthContext>,
    sessions: Mutex<HashMap<String, LoginSession>>,
    next_session: AtomicU64,
}

impl std::fmt::Debug for AuthRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthRegistry")
            .field("providers", &self.providers.len())
            .finish_non_exhaustive()
    }
}

impl AuthRegistry {
    /// Build a registry over a credential store and ambient context.
    pub fn new(store: Arc<dyn CredentialStore>, context: Arc<dyn AuthContext>) -> Self {
        Self {
            providers: DashMap::new(),
            store,
            context,
            sessions: Mutex::new(HashMap::new()),
            next_session: AtomicU64::new(1),
        }
    }

    /// The credential store backing this registry.
    pub fn store(&self) -> &Arc<dyn CredentialStore> {
        &self.store
    }

    /// Register (or replace) a provider's auth surface.
    pub fn register_provider(&self, id: impl Into<String>, auth: ProviderAuth) {
        let id = id.into();
        self.providers
            .insert(id.clone(), ProviderEntry { id, auth });
    }

    /// Remove a provider's auth surface.
    pub fn remove_provider(&self, id: &str) {
        self.providers.remove(id);
    }

    /// True when the provider has a registered auth surface.
    pub fn has_provider(&self, id: &str) -> bool {
        self.providers.contains_key(id)
    }

    /// Describe every registered auth surface (never secrets). Ordering is
    /// unspecified; frontends sort for display.
    pub fn describe_providers(&self) -> Vec<crate::auth::AuthSurfaceInfo> {
        self.providers
            .iter()
            .filter_map(|entry| {
                let auth = &entry.auth;
                if let Some(oauth) = auth.oauth.as_ref() {
                    Some(crate::auth::AuthSurfaceInfo {
                        id: entry.id.clone(),
                        display_name: oauth.name(),
                        method: crate::credential::AuthMethod::OAuth,
                        login_label: oauth.login_label(),
                        is_subscription: oauth.is_subscription(),
                    })
                } else {
                    auth.api_key
                        .as_ref()
                        .map(|api_key| crate::auth::AuthSurfaceInfo {
                            id: entry.id.clone(),
                            display_name: api_key.name(),
                            method: crate::credential::AuthMethod::ApiKey,
                            login_label: None,
                            is_subscription: false,
                        })
                }
            })
            .collect()
    }

    /// List stored credential metadata (never secrets).
    pub async fn list_credentials(&self) -> AuthResult<Vec<CredentialInfo>> {
        self.store.list().await
    }

    /// Remove a stored credential (logout).
    pub async fn logout(&self, provider_id: &str) -> AuthResult<()> {
        self.store.delete(provider_id).await
    }

    /// Resolve auth for a profile and return the effective profile.
    ///
    /// Returns `None` when the profile carries no auth reference (the caller
    /// keeps its original behavior) or when nothing is configured.
    pub async fn apply(&self, profile: &LlmProfile) -> AuthResult<Option<ResolvedAuth>> {
        let Some(auth_ref) = profile.auth.as_ref() else {
            return Ok(None);
        };
        let entry = self
            .providers
            .get(&auth_ref.provider_id)
            .ok_or_else(|| AuthError::ProviderNotRegistered(auth_ref.provider_id.clone()))?;
        let explicit = profile.api_key.clone();
        let overrides = ResolveOverrides {
            api_key: explicit,
            ..Default::default()
        };
        let Some(model_auth) =
            resolve_provider_auth(&entry, self.store.as_ref(), &self.context, &overrides).await?
        else {
            return Ok(None);
        };
        let is_subscription = model_auth.is_subscription;
        let mut effective = profile.clone();
        if let Some(key) = model_auth.api_key {
            effective.api_key = Some(key);
        }
        if let Some(base_url) = model_auth.base_url {
            effective.base_url = Some(base_url);
        }
        if !model_auth.headers.is_empty() {
            let mut headers = effective.custom_headers.clone().unwrap_or_default();
            for (key, value) in model_auth.headers {
                headers.insert(key, serde_json::Value::String(value));
            }
            effective.custom_headers = Some(headers);
        }
        Ok(Some(ResolvedAuth {
            profile: effective,
            is_subscription,
        }))
    }

    /// Start a login session for a provider.
    ///
    /// The returned session id drives `login_poll`, `login_submit` and
    /// `login_cancel`. The flow never blocks on a frontend.
    pub async fn login_start(&self, provider_id: &str) -> AuthResult<(String, LoginState)> {
        let entry = self
            .providers
            .get(provider_id)
            .ok_or_else(|| AuthError::ProviderNotRegistered(provider_id.to_string()))?;
        let flow = if let Some(oauth) = entry.auth.oauth.as_ref() {
            oauth.begin_login().await?
        } else if entry.auth.api_key.is_some() {
            crate::session::LoginFlow::ApiKey
        } else {
            return Err(AuthError::ProviderNotRegistered(provider_id.to_string()));
        };
        drop(entry);
        let mut session = LoginSession::new(provider_id.to_string(), flow);
        let state = session.initial_state().await?;
        let session_id = format!(
            "login-{}",
            self.next_session.fetch_add(1, Ordering::Relaxed)
        );
        self.sessions
            .lock()
            .await
            .insert(session_id.clone(), session);
        Ok((session_id, state))
    }

    /// Advance a login session's polling step.
    pub async fn login_poll(&self, session_id: &str) -> AuthResult<LoginState> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AuthError::SessionNotFound(session_id.to_string()))?;
        let state = session.poll(now()).await?;
        if let LoginState::Completed { provider_id, .. } = &state {
            let credential = session_credential(session)?;
            let provider_id = provider_id.clone();
            self.store
                .modify(
                    &provider_id,
                    Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
                )
                .await?;
            sessions.remove(session_id);
        }
        if matches!(state, LoginState::Failed { .. }) {
            sessions.remove(session_id);
        }
        Ok(state)
    }

    /// Submit user input for a session awaiting it. On success the credential
    /// is written through the store's serialized write path.
    pub async fn login_submit(&self, session_id: &str, input: String) -> AuthResult<LoginState> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AuthError::SessionNotFound(session_id.to_string()))?;
        let state = session.submit(input, &login_http_client(), now()).await?;
        if let LoginState::Completed {
            provider_id,
            method,
        } = &state
        {
            // The session produced a credential; persist it through the
            // serialized write path so concurrent refreshes stay exclusive.
            let credential = session_credential(session)?;
            let provider_id = provider_id.clone();
            let _ = method;
            self.store
                .modify(
                    &provider_id,
                    Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
                )
                .await?;
            sessions.remove(session_id);
        }
        if matches!(state, LoginState::Failed { .. }) {
            sessions.remove(session_id);
        }
        Ok(state)
    }

    /// Take the loopback listener of a session waiting for a callback.
    pub async fn login_take_loopback(
        &self,
        session_id: &str,
    ) -> AuthResult<Option<crate::oauth::pkce::LoopbackListener>> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AuthError::SessionNotFound(session_id.to_string()))?;
        Ok(session.takes_loopback())
    }

    /// Cancel and drop a login session.
    pub async fn login_cancel(&self, session_id: &str) -> AuthResult<()> {
        self.sessions.lock().await.remove(session_id);
        Ok(())
    }

    /// The polling interval a frontend should use between `login_poll` calls.
    pub async fn login_poll_interval(&self, session_id: &str) -> AuthResult<std::time::Duration> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| AuthError::SessionNotFound(session_id.to_string()))?;
        Ok(session.poll_interval())
    }
}

/// The credential a completed session produced.
///
/// Sessions produce their credential inside `submit`/`poll`; the registry
/// re-derives it here from the session's own bookkeeping so the write goes
/// through the store lock exactly once.
fn session_credential(session: &LoginSession) -> AuthResult<Credential> {
    session
        .produced_credential()
        .ok_or_else(|| AuthError::Login("completed session produced no credential".to_string()))
}

/// Shared HTTP client for OAuth token exchanges.
fn login_http_client() -> reqwest::Client {
    reqwest::Client::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::ApiKeyAuth;
    use crate::interaction::NullInteraction;
    use crate::store::MemoryCredentialStore;

    #[derive(Debug)]
    struct EnvKey;

    #[async_trait::async_trait]
    impl ApiKeyAuth for EnvKey {
        fn name(&self) -> String {
            "env key".to_string()
        }

        async fn resolve(
            &self,
            input: crate::auth::ResolveInput<'_>,
        ) -> AuthResult<Option<crate::auth::ModelAuth>> {
            Ok(input
                .ctx
                .env("TEST_KEY")
                .await
                .map(|key| crate::auth::ModelAuth {
                    api_key: Some(key),
                    source: Some("TEST_KEY".to_string()),
                    ..Default::default()
                }))
        }
    }

    fn registry() -> Arc<AuthRegistry> {
        Arc::new(AuthRegistry::new(
            Arc::new(MemoryCredentialStore::new()),
            Arc::new(crate::auth::StdAuthContext),
        ))
    }

    fn profile_with_auth() -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "p1".to_string(),
            format: llm_suite_types::llm::LlmFormat::OpenaiChat,
            provider_id: None,
            model: "m".to_string(),
            api_key: None,
            base_url: None,
            parameters: None,
            generation: None,
            timeout: None,
            max_retries: None,
            retry_delay: None,
            headers: None,
            metadata: None,
            tool_call_protocol: None,
            auth_type: None,
            custom_headers: None,
            custom_body: None,
            custom_body_enabled: None,
            query_params: None,
            stream_options: None,
            prompt_caching: None,
            context_window_size: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
            auth: Some(llm_suite_types::llm::ProfileAuthRef {
                provider_id: "acme".to_string(),
                method: None,
            }),
        }
    }

    #[tokio::test]
    async fn profile_without_auth_reference_is_untouched() {
        let registry = registry();
        let mut profile = profile_with_auth();
        profile.auth = None;
        assert!(registry.apply(&profile).await.expect("apply").is_none());
    }

    #[tokio::test]
    async fn unknown_provider_reference_fails_loudly() {
        let registry = registry();
        let err = registry
            .apply(&profile_with_auth())
            .await
            .expect_err("must fail");
        assert!(matches!(err, AuthError::ProviderNotRegistered(_)));
    }

    #[tokio::test]
    async fn api_key_login_flow_writes_credential() {
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::api_key_only(Arc::new(EnvKey)));
        let (session_id, state) = registry.login_start("acme").await.expect("start");
        assert!(matches!(state, LoginState::AwaitingInput { .. }));
        let state = registry
            .login_submit(&session_id, "sk-typed".to_string())
            .await
            .expect("submit");
        assert!(matches!(state, LoginState::Completed { .. }));
        let stored = registry
            .store()
            .read("acme")
            .await
            .expect("read")
            .expect("stored");
        assert_eq!(
            stored.as_api_key().expect("key").key.as_deref(),
            Some("sk-typed")
        );
    }

    #[tokio::test]
    async fn logout_removes_credential() {
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::api_key_only(Arc::new(EnvKey)));
        let (session_id, _) = registry.login_start("acme").await.expect("start");
        registry
            .login_submit(&session_id, "sk-typed".to_string())
            .await
            .expect("submit");
        registry.logout("acme").await.expect("logout");
        assert!(registry.store().read("acme").await.expect("read").is_none());
    }

    #[tokio::test]
    async fn ambient_only_provider_rejects_login_start() {
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::default());
        let err = registry.login_start("acme").await.expect_err("must fail");
        assert!(matches!(err, AuthError::ProviderNotRegistered(_)));
    }

    #[tokio::test]
    async fn unknown_session_fails_loudly() {
        let registry = registry();
        let err = registry.login_poll("nope").await.expect_err("must fail");
        assert!(matches!(err, AuthError::SessionNotFound(_)));
    }

    #[test]
    fn provider_entry_reexport_is_usable() {
        let _entry = ProviderEntry {
            id: "x".to_string(),
            auth: ProviderAuth::default(),
        };
        let _ = NullInteraction;
    }

    #[test]
    fn describe_providers_reports_surface_metadata() {
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::api_key_only(Arc::new(EnvKey)));
        let mut surfaces = registry.describe_providers();
        surfaces.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].id, "acme");
        assert_eq!(surfaces[0].display_name, "env key");
        assert_eq!(surfaces[0].method, crate::credential::AuthMethod::ApiKey);
        assert!(!surfaces[0].is_subscription);
    }
}
