//! UI-neutral login interaction protocol shared by CLI, TUI and HTTP
//! frontends: the auth layer emits events and asks for input; the host
//! decides how to render them.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AuthResult;

/// One link attached to an informational event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthInfoLink {
    pub url: String,
    pub label: Option<String>,
}

/// Events emitted during a login flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthEvent {
    /// Plain information with optional links.
    Info {
        message: String,
        links: Vec<AuthInfoLink>,
    },
    /// An authorization URL the user should open.
    AuthUrl {
        url: String,
        instructions: Option<String>,
    },
    /// A device-code pair to display while polling continues.
    DeviceCode {
        user_code: String,
        verification_uri: String,
        interval_seconds: Option<u64>,
        expires_in_seconds: Option<u64>,
    },
    /// Progress note (poll attempt, waiting for callback, ...).
    Progress {
        message: String,
    },
}

/// One option of a select prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthPromptOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

/// A prompt the user must answer before the flow can continue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthPrompt {
    Text {
        message: String,
        placeholder: Option<String>,
    },
    Secret {
        message: String,
        placeholder: Option<String>,
    },
    Select {
        message: String,
        options: Vec<AuthPromptOption>,
    },
    /// The user must paste a value received out of band (callback URL,
    /// verification code, ...).
    ManualCode {
        message: String,
        placeholder: Option<String>,
    },
}

/// Host-side interaction surface.
///
/// `prompt` returns the entered or selected value and rejects on cancel;
/// `notify` renders an event. Implementations must honor cancellation: a
/// pending prompt may be aborted when an out-of-band event resolves the step.
#[async_trait]
pub trait AuthInteraction: Send + Sync {
    async fn prompt(&self, prompt: AuthPrompt) -> AuthResult<String>;
    fn notify(&self, event: AuthEvent);
}

/// A no-op interaction for flows that never prompt (ambient-only providers,
/// tests). `prompt` always fails loudly.
#[derive(Debug, Default)]
pub struct NullInteraction;

#[async_trait]
impl AuthInteraction for NullInteraction {
    async fn prompt(&self, _prompt: AuthPrompt) -> AuthResult<String> {
        Err(crate::error::AuthError::Login(
            "no interactive frontend available".to_string(),
        ))
    }

    fn notify(&self, _event: AuthEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn null_interaction_fails_loudly() {
        let interaction = NullInteraction;
        let err = interaction
            .prompt(AuthPrompt::Text {
                message: "key".to_string(),
                placeholder: None,
            })
            .await
            .expect_err("must fail");
        assert!(matches!(err, crate::error::AuthError::Login(_)));
    }
}
