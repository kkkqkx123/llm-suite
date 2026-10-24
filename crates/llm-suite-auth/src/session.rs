//! Host-driven login session state machine.
//!
//! One mechanism serves every frontend: CLI and TUI drive the session with
//! an [`AuthInteraction`], HTTP frontends drive it with user input plus
//! polling. The auth layer never blocks on a frontend.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::auth::ApiKeyAuth;
use crate::credential::{AuthMethod, Credential, OAuthCredential};
use crate::error::{AuthError, AuthResult};
use crate::interaction::{AuthEvent, AuthPrompt};
use crate::oauth::device_flow::DeviceFlowPending;
use crate::oauth::pkce::PkcePending;

/// Events emitted so far by a session, replayed to late-attaching frontends.
pub type EventLog = Vec<AuthEvent>;

/// Opaque state an externally-driven flow passes between host steps. The
/// driver owns the shape; the session only stores and replays it.
pub type ExternalState = serde_json::Value;

/// Flow-specific pending state owned by a session.
#[derive(Debug)]
pub enum LoginFlow {
    /// Device authorization grant; the registry polls the token endpoint.
    Device(DeviceFlowPending),
    /// PKCE paste flow; the user pastes the callback URL.
    PkcePaste(PkcePending),
    /// PKCE loopback flow; a local listener receives the callback.
    PkceLoopback(PkcePending),
    /// Interactive API-key entry.
    ApiKey,
    /// Flow whose state and transitions live outside this crate (e.g. a
    /// plugin-contributed auth implementation bridged by the host). The
    /// session stores the opaque state and calls the driver on the host's
    /// schedule; the driver never blocks on a frontend.
    External(ExternalPending),
}

/// One transition of an externally-driven login flow. `state` is the opaque
/// state the driver wants the session to hold for the next step.
#[derive(Debug, Clone)]
pub enum ExternalStep {
    /// Waiting on an external event; `events` join the session log.
    Pending {
        state: ExternalState,
        events: Vec<AuthEvent>,
    },
    /// The host must collect one input, then call `submit`.
    AwaitingInput {
        state: ExternalState,
        prompt: AuthPrompt,
        events: Vec<AuthEvent>,
    },
    /// The flow produced a credential; the session persists it.
    Completed { credential: Credential },
    /// The flow failed; any stored credential is preserved.
    Failed {
        error: String,
        events: Vec<AuthEvent>,
    },
}

/// Driver for a [`LoginFlow::External`] flow. Implementations translate the
/// host's step calls into their own protocol (typically a synchronous guest
/// contract) and map the outcome onto [`ExternalStep`].
#[async_trait]
pub trait ExternalFlowDriver: Send + Sync + std::fmt::Debug {
    /// The first step of the flow, from the driver's initial state.
    async fn initial(&self, state: &ExternalState) -> AuthResult<ExternalStep>;
    /// Advance a polling step.
    async fn poll(&self, state: &ExternalState) -> AuthResult<ExternalStep>;
    /// Accept user input for a step awaiting it.
    async fn submit(&self, state: &ExternalState, input: String) -> AuthResult<ExternalStep>;
    /// The interval the host should wait between `poll` calls.
    fn poll_interval(&self, state: &ExternalState) -> std::time::Duration {
        let _ = state;
        std::time::Duration::from_secs(1)
    }
}

/// Pending state of an externally-driven flow: the opaque state plus the
/// driver that advances it.
#[derive(Debug)]
pub struct ExternalPending {
    state: ExternalState,
    driver: Arc<dyn ExternalFlowDriver>,
}

impl ExternalPending {
    /// Build the pending state from the driver's initial state.
    pub fn new(state: ExternalState, driver: Arc<dyn ExternalFlowDriver>) -> Self {
        Self { state, driver }
    }

    /// The opaque state the driver last produced.
    pub fn state(&self) -> &ExternalState {
        &self.state
    }

    /// The driver advancing this flow.
    pub fn driver(&self) -> &Arc<dyn ExternalFlowDriver> {
        &self.driver
    }
}

/// One login session.
pub struct LoginSession {
    pub provider_id: String,
    flow: LoginFlow,
    events: EventLog,
    produced: Option<Credential>,
}

/// Observable state of a login session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoginState {
    /// The flow started; `events` carry the authorization URL or device code.
    Challenge {
        events: EventLog,
    },
    /// The host must collect one input, then call `login_submit`.
    AwaitingInput {
        prompt: AuthPrompt,
        events: EventLog,
    },
    /// Waiting on an external event (device poll window, loopback callback).
    Pending {
        events: EventLog,
    },
    /// The credential was written to the store.
    Completed {
        provider_id: String,
        method: AuthMethod,
    },
    /// The flow failed; any stored credential is preserved.
    Failed {
        error: String,
        events: EventLog,
    },
}

impl LoginSession {
    /// Build a session for a started flow, seeding the event log with the
    /// challenge events the flow already knows about. External flows seed
    /// their log in [`LoginSession::initial_state`], which asks the driver.
    pub fn new(provider_id: String, flow: LoginFlow) -> Self {
        let mut events = EventLog::new();
        match &flow {
            LoginFlow::Device(pending) => events.push(pending.challenge_event()),
            LoginFlow::PkcePaste(pending) => events.push(AuthEvent::AuthUrl {
                url: pending.authorize_url(),
                instructions: Some(
                    "Authorize in the browser, then paste the full callback URL back here."
                        .to_string(),
                ),
            }),
            LoginFlow::PkceLoopback(pending) => events.push(AuthEvent::AuthUrl {
                url: pending.authorize_url(),
                instructions: Some(
                    "Authorize in the browser; this window completes automatically.".to_string(),
                ),
            }),
            LoginFlow::ApiKey => {}
            LoginFlow::External(_) => {}
        }
        Self {
            provider_id,
            flow,
            events,
            produced: None,
        }
    }

    /// The initial observable state for this session. External flows ask
    /// their driver, so the first step can fail.
    pub async fn initial_state(&mut self) -> AuthResult<LoginState> {
        match &self.flow {
            LoginFlow::Device(_) => Ok(LoginState::Challenge {
                events: self.events.clone(),
            }),
            LoginFlow::PkcePaste(_) => Ok(LoginState::AwaitingInput {
                prompt: AuthPrompt::ManualCode {
                    message: "Paste the callback URL:".to_string(),
                    placeholder: Some("http://127.0.0.1:PORT/callback?code=...".to_string()),
                },
                events: self.events.clone(),
            }),
            LoginFlow::PkceLoopback(_) => Ok(LoginState::Pending {
                events: self.events.clone(),
            }),
            LoginFlow::ApiKey => Ok(LoginState::AwaitingInput {
                prompt: AuthPrompt::Secret {
                    message: "API key:".to_string(),
                    placeholder: None,
                },
                events: self.events.clone(),
            }),
            LoginFlow::External(pending) => {
                let driver = pending.driver().clone();
                let state = pending.state().clone();
                let step = driver.initial(&state).await?;
                Ok(self.map_external_step(step))
            }
        }
    }

    /// Advance a polling step (device flow poll, loopback callback wait).
    pub async fn poll(&mut self, now_ms: i64) -> AuthResult<LoginState> {
        match &mut self.flow {
            LoginFlow::Device(pending) => {
                use crate::oauth::device_flow::DeviceFlowPoll;
                match pending.poll_once(now_ms).await? {
                    DeviceFlowPoll::Pending => {
                        self.events.push(AuthEvent::Progress {
                            message: "waiting for authorization".to_string(),
                        });
                        Ok(LoginState::Pending {
                            events: self.events.clone(),
                        })
                    }
                    DeviceFlowPoll::SlowDown => {
                        pending.widen_interval();
                        self.events.push(AuthEvent::Progress {
                            message: "server asked to slow down".to_string(),
                        });
                        Ok(LoginState::Pending {
                            events: self.events.clone(),
                        })
                    }
                    DeviceFlowPoll::Denied(reason) => {
                        Ok(self.failed(format!("authorization denied: {reason}")))
                    }
                    DeviceFlowPoll::Expired => Ok(self.failed("device code expired".to_string())),
                    DeviceFlowPoll::Complete(credential) => {
                        Ok(self.completed(Credential::OAuth(credential)))
                    }
                }
            }
            LoginFlow::PkceLoopback(_) => Ok(LoginState::Pending {
                events: self.events.clone(),
            }),
            LoginFlow::PkcePaste(_) | LoginFlow::ApiKey => Ok(LoginState::Pending {
                events: self.events.clone(),
            }),
            LoginFlow::External(pending) => {
                let driver = pending.driver().clone();
                let state = pending.state().clone();
                let step = driver.poll(&state).await?;
                Ok(self.map_external_step(step))
            }
        }
    }

    /// Accept user input for flows that await it.
    pub async fn submit(
        &mut self,
        input: String,
        client: &reqwest::Client,
        now_ms: i64,
    ) -> AuthResult<LoginState> {
        match &mut self.flow {
            LoginFlow::PkcePaste(pending) => {
                let code = PkcePending::extract_code_from_paste(&input).ok_or_else(|| {
                    AuthError::Login("pasted value carried no authorization code".to_string())
                })?;
                let credential = pending.exchange_code(&code, client, now_ms).await?;
                Ok(self.completed(Credential::OAuth(credential)))
            }
            LoginFlow::ApiKey => {
                let credential = Credential::ApiKey(crate::credential::ApiKeyCredential {
                    key: Some(input),
                    env: Default::default(),
                });
                Ok(self.completed(credential))
            }
            LoginFlow::Device(_) => Err(AuthError::SessionNotAwaitingInput(
                "device flow is poll-driven".to_string(),
            )),
            LoginFlow::PkceLoopback(_) => Err(AuthError::SessionNotAwaitingInput(
                "loopback flow completes via the local callback".to_string(),
            )),
            LoginFlow::External(pending) => {
                let driver = pending.driver().clone();
                let state = pending.state().clone();
                let step = driver.submit(&state, input).await?;
                Ok(self.map_external_step(step))
            }
        }
    }

    /// The polling interval the host should wait between `poll` calls.
    pub fn poll_interval(&self) -> std::time::Duration {
        match &self.flow {
            LoginFlow::Device(pending) => pending.interval(),
            LoginFlow::External(pending) => pending.driver().poll_interval(pending.state()),
            _ => std::time::Duration::from_secs(1),
        }
    }

    /// Whether this flow still expects a loopback callback.
    pub fn takes_loopback(&mut self) -> Option<crate::oauth::pkce::LoopbackListener> {
        match &mut self.flow {
            LoginFlow::PkceLoopback(pending) => pending.take_listener(),
            _ => None,
        }
    }

    fn completed(&mut self, credential: Credential) -> LoginState {
        self.events.push(AuthEvent::Progress {
            message: "credential stored".to_string(),
        });
        self.produced = Some(credential.clone());
        LoginState::Completed {
            provider_id: self.provider_id.clone(),
            method: credential.method(),
        }
    }

    /// Fold one external step into the session: store the driver's next
    /// opaque state, append the events and map the outcome onto a
    /// [`LoginState`].
    fn map_external_step(&mut self, step: ExternalStep) -> LoginState {
        match step {
            ExternalStep::Pending { state, events } => {
                self.set_external_state(state);
                self.events.extend(events);
                LoginState::Pending {
                    events: self.events.clone(),
                }
            }
            ExternalStep::AwaitingInput {
                state,
                prompt,
                events,
            } => {
                self.set_external_state(state);
                self.events.extend(events);
                LoginState::AwaitingInput {
                    prompt,
                    events: self.events.clone(),
                }
            }
            ExternalStep::Completed { credential } => self.completed(credential),
            ExternalStep::Failed { error, events } => {
                self.events.extend(events);
                self.failed(error)
            }
        }
    }

    /// Replace the opaque state of an external flow; a no-op for the
    /// built-in flows.
    fn set_external_state(&mut self, state: ExternalState) {
        if let LoginFlow::External(pending) = &mut self.flow {
            pending.state = state;
        }
    }

    /// The credential a completed session produced, for the registry to
    /// persist through the store's serialized write path.
    pub fn produced_credential(&self) -> Option<Credential> {
        self.produced.clone()
    }

    fn failed(&mut self, error: String) -> LoginState {
        self.events.push(AuthEvent::Progress {
            message: format!("login failed: {error}"),
        });
        LoginState::Failed {
            error,
            events: self.events.clone(),
        }
    }
}

/// Build the prompt an API-key provider shows during login.
pub fn api_key_prompt(auth: &dyn ApiKeyAuth) -> AuthPrompt {
    auth.login_prompt()
}

/// Helper for tests and simple providers: a bearer-token OAuth credential
/// with no extra derivation.
pub fn bearer_to_auth(credential: &OAuthCredential) -> ModelAuthSeed {
    ModelAuthSeed {
        api_key: Some(credential.access.clone()),
        headers: Default::default(),
        base_url: None,
    }
}

/// Seed values for [`crate::auth::ModelAuth`] construction by providers.
#[derive(Debug, Clone, Default)]
pub struct ModelAuthSeed {
    pub api_key: Option<String>,
    pub headers: std::collections::BTreeMap<String, String>,
    pub base_url: Option<String>,
}

impl ModelAuthSeed {
    /// Convert into a full `ModelAuth` with the subscription flag and source.
    pub fn into_model_auth(self, is_subscription: bool, source: &str) -> crate::auth::ModelAuth {
        crate::auth::ModelAuth {
            api_key: self.api_key,
            headers: self.headers,
            base_url: self.base_url,
            is_subscription,
            source: Some(source.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn api_key_session_completes_on_submit() {
        let mut session = LoginSession::new("acme".to_string(), LoginFlow::ApiKey);
        let initial = session.initial_state().await.expect("initial state");
        assert!(matches!(initial, LoginState::AwaitingInput { .. }));
        let state = session
            .submit("sk-pasted".to_string(), &reqwest::Client::new(), 0)
            .await
            .expect("submit");
        match state {
            LoginState::Completed { provider_id, method } => {
                assert_eq!(provider_id, "acme");
                assert_eq!(method, AuthMethod::ApiKey);
            }
            other => panic!("unexpected state: {other:?}"),
        }
    }

    #[tokio::test]
    async fn device_session_rejects_submit() {
        let pending = crate::oauth::device_flow::DeviceFlowPending::from_parts(
            crate::oauth::device_flow::DeviceFlowConfig {
                device_authorization_url: "https://example.test/device".to_string(),
                token_url: "https://example.test/token".to_string(),
                client_id: "cid".to_string(),
                scope: None,
                is_subscription: true,
            },
            "dc".to_string(),
            "WDJB-MJHT".to_string(),
            "https://example.test/activate".to_string(),
            std::time::Duration::from_secs(5),
            10_000,
        );
        let mut session = LoginSession::new("acme".to_string(), LoginFlow::Device(pending));
        assert!(matches!(
            session.initial_state().await.expect("initial state"),
            LoginState::Challenge { .. }
        ));
        let err = session
            .submit("x".to_string(), &reqwest::Client::new(), 0)
            .await
            .expect_err("must fail");
        assert!(matches!(err, AuthError::SessionNotAwaitingInput(_)));
    }

    #[tokio::test]
    async fn paste_session_extracts_code_or_fails() {
        let pending = crate::oauth::pkce::start(crate::oauth::pkce::PkceConfig {
            authorize_url: "https://example.test/authorize".to_string(),
            token_url: "https://example.test/token".to_string(),
            client_id: "cid".to_string(),
            redirect_uri: "http://127.0.0.1:8080/callback".to_string(),
            scope: None,
            loopback: false,
            is_subscription: false,
        })
        .await
        .expect("start");
        let mut session = LoginSession::new("acme".to_string(), LoginFlow::PkcePaste(pending));
        // The token endpoint is unreachable here; the paste parsing itself
        // must fail before any network call for a code-less paste.
        let err = session
            .submit("http://127.0.0.1:8080/callback?state=only".to_string(), &reqwest::Client::new(), 0)
            .await
            .expect_err("must fail");
        assert!(matches!(err, AuthError::Login(_)));
    }

    #[test]
    fn model_auth_seed_carries_subscription_flag() {
        let auth = bearer_to_auth(&OAuthCredential {
            refresh: "r".to_string(),
            access: "a".to_string(),
            expires: 0,
            is_subscription: true,
            extra: Default::default(),
        })
        .into_model_auth(true, "OAuth");
        assert!(auth.is_subscription);
        assert_eq!(auth.api_key.as_deref(), Some("a"));
        assert_eq!(auth.source.as_deref(), Some("OAuth"));
    }

    /// Scripted external driver: `initial` asks for a paste, `submit`
    /// completes with the input as the access token, `poll` stays pending.
    #[derive(Debug)]
    struct ScriptedDriver;

    #[async_trait]
    impl ExternalFlowDriver for ScriptedDriver {
        async fn initial(&self, state: &ExternalState) -> AuthResult<ExternalStep> {
            assert_eq!(state, &serde_json::json!({"phase": "start"}));
            Ok(ExternalStep::AwaitingInput {
                state: serde_json::json!({"phase": "paste"}),
                prompt: AuthPrompt::ManualCode {
                    message: "Paste the code:".to_string(),
                    placeholder: None,
                },
                events: vec![AuthEvent::AuthUrl {
                    url: "https://example.test/authorize".to_string(),
                    instructions: None,
                }],
            })
        }

        async fn poll(&self, _state: &ExternalState) -> AuthResult<ExternalStep> {
            Ok(ExternalStep::Pending {
                state: serde_json::json!({"phase": "waiting"}),
                events: vec![AuthEvent::Progress {
                    message: "still waiting".to_string(),
                }],
            })
        }

        async fn submit(&self, state: &ExternalState, input: String) -> AuthResult<ExternalStep> {
            assert_eq!(state, &serde_json::json!({"phase": "paste"}));
            Ok(ExternalStep::Completed {
                credential: Credential::OAuth(OAuthCredential {
                    refresh: "r".to_string(),
                    access: input,
                    expires: 0,
                    is_subscription: true,
                    extra: Default::default(),
                }),
            })
        }
    }

    #[tokio::test]
    async fn external_flow_threads_state_and_completes() {
        let pending = ExternalPending::new(
            serde_json::json!({"phase": "start"}),
            Arc::new(ScriptedDriver),
        );
        let mut session = LoginSession::new("acme".to_string(), LoginFlow::External(pending));
        let initial = session.initial_state().await.expect("initial state");
        match initial {
            LoginState::AwaitingInput { prompt, events } => {
                assert!(matches!(prompt, AuthPrompt::ManualCode { .. }));
                assert_eq!(events.len(), 1);
            }
            other => panic!("unexpected state: {other:?}"),
        }
        // The driver's next state is threaded into the session.
        let state = session
            .submit("sk-pasted".to_string(), &reqwest::Client::new(), 0)
            .await
            .expect("submit");
        match state {
            LoginState::Completed { provider_id, .. } => assert_eq!(provider_id, "acme"),
            other => panic!("unexpected state: {other:?}"),
        }
        let produced = session.produced_credential().expect("credential");
        match produced {
            Credential::OAuth(cred) => assert_eq!(cred.access, "sk-pasted"),
            other => panic!("unexpected credential: {other:?}"),
        }
    }

    #[tokio::test]
    async fn external_flow_poll_stays_pending_with_threaded_state() {
        let pending = ExternalPending::new(
            serde_json::json!({"phase": "start"}),
            Arc::new(ScriptedDriver),
        );
        let mut session = LoginSession::new("acme".to_string(), LoginFlow::External(pending));
        let _ = session.initial_state().await;
        let state = session.poll(0).await.expect("poll");
        assert!(matches!(state, LoginState::Pending { .. }));
        if let LoginFlow::External(pending) = &session.flow {
            assert_eq!(pending.state(), &serde_json::json!({"phase": "waiting"}));
        } else {
            panic!("flow must stay external");
        }
    }
}
