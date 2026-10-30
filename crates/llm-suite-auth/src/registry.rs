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

    /// Describe every registered auth surface (never secrets). A provider
    /// with both methods registered yields two surfaces; ordering is
    /// unspecified and frontends sort for display.
    pub fn describe_providers(&self) -> Vec<crate::auth::AuthSurfaceInfo> {
        self.providers
            .iter()
            .flat_map(|entry| {
                let mut surfaces = Vec::new();
                if let Some(api_key) = entry.auth.api_key.as_ref() {
                    surfaces.push(crate::auth::AuthSurfaceInfo {
                        id: entry.id.clone(),
                        display_name: api_key.name(),
                        method: crate::credential::AuthMethod::ApiKey,
                        login_label: None,
                        is_subscription: false,
                    });
                }
                if let Some(oauth) = entry.auth.oauth.as_ref() {
                    surfaces.push(crate::auth::AuthSurfaceInfo {
                        id: entry.id.clone(),
                        display_name: oauth.name(),
                        method: crate::credential::AuthMethod::OAuth,
                        login_label: oauth.login_label(),
                        is_subscription: oauth.is_subscription(),
                    });
                }
                surfaces
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

    /// The ambient auth context backing resolution.
    pub fn context(&self) -> &Arc<dyn AuthContext> {
        &self.context
    }

    /// Resolve auth for a profile and return the effective profile.
    ///
    /// Returns `None` when the profile carries no auth reference (the caller
    /// keeps its original behavior) or when nothing is configured.
    pub async fn apply(&self, profile: &LlmProfile) -> AuthResult<Option<ResolvedAuth>> {
        self.apply_with_context(profile, &self.context).await
    }

    /// [`apply`] with an explicit ambient context. Hosts use this to thread
    /// a per-profile environment overlay ([`OverlayAuthContext`]) without
    /// replacing the registry-wide context: pass the registry context
    /// wrapped in the overlay when `profile.env` is present, the registry
    /// context itself otherwise.
    pub async fn apply_with_context(
        &self,
        profile: &LlmProfile,
        context: &Arc<dyn AuthContext>,
    ) -> AuthResult<Option<ResolvedAuth>> {
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
            resolve_provider_auth(&entry, self.store.as_ref(), context, &overrides).await?
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

    /// Advance a login session's polling step. The session is taken out of
    /// the map while polling so device polling and plugin calls never hold
    /// the global sessions lock across a network await.
    pub async fn login_poll(&self, session_id: &str) -> AuthResult<LoginState> {
        let mut session = self.take_session(session_id).await?;
        let poll_result = session.poll(now()).await;
        let state = match poll_result {
            Ok(state) => state,
            Err(err) => {
                self.sessions
                    .lock()
                    .await
                    .insert(session_id.to_string(), session);
                return Err(err);
            }
        };
        if let LoginState::Completed { provider_id, .. } = &state {
            let credential = session_credential(&session)?;
            let provider_id = provider_id.clone();
            self.store
                .modify(
                    &provider_id,
                    Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
                )
                .await?;
            return Ok(state);
        }
        if matches!(state, LoginState::Failed { .. }) {
            return Ok(state);
        }
        self.sessions
            .lock()
            .await
            .insert(session_id.to_string(), session);
        Ok(state)
    }

    /// Submit user input for a session awaiting it. On success the credential
    /// is written through the store's serialized write path. The session is
    /// taken out of the map while awaiting the token exchange so the global
    /// lock never spans a network await.
    pub async fn login_submit(&self, session_id: &str, input: String) -> AuthResult<LoginState> {
        let mut session = self.take_session(session_id).await?;
        let client = login_http_client();
        let submit_result = session.submit(input, &client, now()).await;
        let state = match submit_result {
            Ok(state) => state,
            Err(err) => {
                self.sessions
                    .lock()
                    .await
                    .insert(session_id.to_string(), session);
                return Err(err);
            }
        };
        if let LoginState::Completed {
            provider_id,
            method,
        } = &state
        {
            // The session produced a credential; persist it through the
            // serialized write path so concurrent refreshes stay exclusive.
            let credential = session_credential(&session)?;
            let provider_id = provider_id.clone();
            let _ = method;
            self.store
                .modify(
                    &provider_id,
                    Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
                )
                .await?;
            return Ok(state);
        }
        if matches!(state, LoginState::Failed { .. }) {
            return Ok(state);
        }
        self.sessions
            .lock()
            .await
            .insert(session_id.to_string(), session);
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

    /// Mark a login session as failed and drop it from the live set. Used by
    /// host-side drivers that own an external wait (loopback callback) and
    /// need to surface the failure through the normal polling state.
    pub async fn login_fail(&self, session_id: &str, error: String) -> AuthResult<LoginState> {
        let mut session = self.take_session(session_id).await?;
        let state = session.mark_failed(error);
        // Keep the terminal session in the map so a later frontend poll can
        // observe the failure (and its original cause) instead of a bare
        // "session not found". The first poll consumes and drops it.
        self.sessions
            .lock()
            .await
            .insert(session_id.to_string(), session);
        Ok(state)
    }

    /// The polling interval a frontend should use between `login_poll` calls.
    pub async fn login_poll_interval(&self, session_id: &str) -> AuthResult<std::time::Duration> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| AuthError::SessionNotFound(session_id.to_string()))?;
        Ok(session.poll_interval())
    }

    /// Remove a session from the map and hand ownership to the caller.
    /// Callers reinsert the session unless it completed or failed, so the
    /// global lock is never held across a network await.
    async fn take_session(&self, session_id: &str) -> AuthResult<LoginSession> {
        self.sessions
            .lock()
            .await
            .remove(session_id)
            .ok_or_else(|| AuthError::SessionNotFound(session_id.to_string()))
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
            // Mirror the production `EnvApiKeyAuth` order: an explicit or
            // stored credential wins, the ambient variable is the fallback.
            if let Some(credential) = input.credential {
                if let Some(key) = credential
                    .as_api_key()
                    .ok()
                    .and_then(|credential| credential.key.clone())
                {
                    return Ok(Some(crate::auth::ModelAuth {
                        api_key: Some(key),
                        source: Some("explicit credential".to_string()),
                        ..Default::default()
                    }));
                }
            }
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
            max_output_tokens: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
            auth: Some(llm_suite_types::llm::ProfileAuthRef {
                provider_id: "acme".to_string(),
                method: None,
            }),
            compat: None,
            identity: None,
            transport: None,
            env: None,
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

    #[test]
    fn describe_providers_reports_both_methods_for_dual_auth() {
        #[derive(Debug)]
        struct StubOAuth;
        #[async_trait::async_trait]
        impl crate::auth::OAuthAuth for StubOAuth {
            fn name(&self) -> String {
                "stub oauth".to_string()
            }
            async fn begin_login(&self) -> AuthResult<crate::session::LoginFlow> {
                Err(AuthError::Login("unused".to_string()))
            }
            async fn refresh(
                &self,
                credential: &crate::credential::OAuthCredential,
            ) -> AuthResult<crate::credential::OAuthCredential> {
                Ok(credential.clone())
            }
            async fn to_auth(
                &self,
                credential: &crate::credential::OAuthCredential,
            ) -> AuthResult<crate::auth::ModelAuth> {
                Ok(crate::auth::ModelAuth {
                    api_key: Some(credential.access.clone()),
                    ..Default::default()
                })
            }
        }
        let registry = registry();
        registry.register_provider(
            "acme",
            ProviderAuth {
                api_key: Some(Arc::new(EnvKey)),
                oauth: Some(Arc::new(StubOAuth)),
            },
        );
        let surfaces = registry.describe_providers();
        assert_eq!(surfaces.len(), 2, "dual auth must expose both surfaces");
        assert!(surfaces
            .iter()
            .any(|s| s.method == crate::credential::AuthMethod::ApiKey));
        assert!(surfaces
            .iter()
            .any(|s| s.method == crate::credential::AuthMethod::OAuth));
    }

    #[tokio::test]
    async fn env_overlay_shadows_the_inner_context() {
        use crate::auth::OverlayAuthContext;
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::api_key_only(Arc::new(EnvKey)));
        let overlay: Arc<dyn crate::auth::AuthContext> = Arc::new(OverlayAuthContext::new(
            registry.context().clone(),
            [("TEST_KEY".to_string(), "sk-overlay".to_string())]
                .into_iter()
                .collect(),
        ));
        let resolved = registry
            .apply_with_context(&profile_with_auth(), &overlay)
            .await
            .expect("apply")
            .expect("resolved");
        assert_eq!(
            resolved.profile.api_key.as_deref(),
            Some("sk-overlay"),
            "the overlay must shadow the process environment"
        );
    }

    #[tokio::test]
    async fn empty_overlay_value_hides_the_variable() {
        use crate::auth::OverlayAuthContext;
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::api_key_only(Arc::new(EnvKey)));
        let overlay: Arc<dyn crate::auth::AuthContext> = Arc::new(OverlayAuthContext::new(
            registry.context().clone(),
            [("TEST_KEY".to_string(), String::new())]
                .into_iter()
                .collect(),
        ));
        // The process environment almost surely lacks TEST_KEY; either way
        // the empty overlay must not resolve a key through the inner context.
        let resolved = registry
            .apply_with_context(&profile_with_auth(), &overlay)
            .await
            .expect("apply");
        if std::env::var("TEST_KEY").ok().filter(|v| !v.is_empty()).is_some() {
            assert!(
                resolved.is_none(),
                "an empty overlay entry must hide the ambient variable"
            );
        } else {
            assert!(resolved.is_none());
        }
    }

    #[tokio::test]
    async fn explicit_profile_key_still_wins_over_overlay() {
        use crate::auth::OverlayAuthContext;
        let registry = registry();
        registry.register_provider("acme", ProviderAuth::api_key_only(Arc::new(EnvKey)));
        let overlay: Arc<dyn crate::auth::AuthContext> = Arc::new(OverlayAuthContext::new(
            registry.context().clone(),
            [("TEST_KEY".to_string(), "sk-overlay".to_string())]
                .into_iter()
                .collect(),
        ));
        let mut profile = profile_with_auth();
        profile.api_key = Some("sk-explicit".to_string());
        let resolved = registry
            .apply_with_context(&profile, &overlay)
            .await
            .expect("apply")
            .expect("resolved");
        assert_eq!(
            resolved.profile.api_key.as_deref(),
            Some("sk-explicit"),
            "an explicit key always wins"
        );
    }
}
