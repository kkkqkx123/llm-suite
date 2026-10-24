//! OAuth 2.0 device authorization grant (RFC 8628).
//!
//! Used by subscription providers whose login shows a short user code and a
//! verification URL while the client polls the token endpoint.

use std::time::Duration;

use serde::Deserialize;

use crate::credential::OAuthCredential;
use crate::error::{AuthError, AuthResult};
use crate::interaction::AuthEvent;
use crate::oauth::{credential_from_token, form_body, TokenResponse};

/// Default polling interval when the server omits one.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Configuration for one device-flow provider.
#[derive(Debug, Clone)]
pub struct DeviceFlowConfig {
    pub device_authorization_url: String,
    pub token_url: String,
    pub client_id: String,
    pub scope: Option<String>,
    /// Whether access through this flow is backed by a subscription.
    pub is_subscription: bool,
}

/// Device authorization endpoint response.
#[derive(Debug, Clone, Deserialize)]
struct DeviceAuthorizationResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    expires_in: Option<i64>,
    interval: Option<u64>,
}

/// A started device flow awaiting authorization.
#[derive(Debug)]
pub struct DeviceFlowPending {
    config: DeviceFlowConfig,
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    interval: Duration,
    expires_at_ms: i64,
    client: reqwest::Client,
}

/// Outcome of one poll attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceFlowPoll {
    /// Authorization still pending.
    Pending,
    /// Polling too fast; the caller should widen the interval.
    SlowDown,
    /// The user denied the request.
    Denied(String),
    /// The device code expired before authorization.
    Expired,
    /// Authorization completed.
    Complete(OAuthCredential),
}

/// Start the device flow: request a device code and build the pending state.
pub async fn start(
    config: DeviceFlowConfig,
    client: reqwest::Client,
    now_ms: i64,
) -> AuthResult<DeviceFlowPending> {
    let mut form: Vec<(&str, &str)> = vec![("client_id", config.client_id.as_str())];
    if let Some(scope) = config.scope.as_deref() {
        form.push(("scope", scope));
    }
    let response = client
        .post(&config.device_authorization_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form_body(&form))
        .send()
        .await
        .map_err(|e| AuthError::Transport(e.to_string()))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| AuthError::Transport(e.to_string()))?;
    if !status.is_success() {
        return Err(AuthError::Transport(format!(
            "device authorization failed: HTTP {} {}",
            status.as_u16(),
            body
        )));
    }
    let parsed: DeviceAuthorizationResponse = serde_json::from_str(&body)
        .map_err(|e| AuthError::Transport(format!("device authorization response: {e}")))?;
    let interval = Duration::from_secs(parsed.interval.unwrap_or_else(|| {
        u64::try_from(DEFAULT_POLL_INTERVAL.as_secs()).expect("default interval fits u64")
    }));
    let expires_at_ms = now_ms + parsed.expires_in.unwrap_or(600).saturating_mul(1000);
    Ok(DeviceFlowPending {
        config,
        device_code: parsed.device_code,
        user_code: parsed.user_code,
        verification_uri: parsed.verification_uri,
        verification_uri_complete: parsed.verification_uri_complete,
        interval,
        expires_at_ms,
        client,
    })
}

impl DeviceFlowPending {
    /// The event a frontend should render for this challenge. Prefers the
    /// server's pre-filled verification URI when it supplied one.
    pub fn challenge_event(&self) -> AuthEvent {
        AuthEvent::DeviceCode {
            user_code: self.user_code.clone(),
            verification_uri: self
                .verification_uri_complete
                .clone()
                .unwrap_or_else(|| self.verification_uri.clone()),
            interval_seconds: Some(self.interval.as_secs()),
            expires_in_seconds: Some(
                u64::try_from((self.expires_at_ms - oauth_now_ms()).max(0) / 1000).unwrap_or(0),
            ),
        }
    }

    /// Assemble a pending state from an already-obtained device code. Used
    /// by tests and by hosts that request the device code themselves.
    #[cfg(test)]
    pub(crate) fn from_parts(
        config: DeviceFlowConfig,
        device_code: String,
        user_code: String,
        verification_uri: String,
        interval: Duration,
        expires_at_ms: i64,
    ) -> Self {
        Self {
            config,
            device_code,
            user_code,
            verification_uri,
            verification_uri_complete: None,
            interval,
            expires_at_ms,
            client: reqwest::Client::new(),
        }
    }

    /// Current polling interval.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Widen the interval after a `SlowDown` response.
    pub fn widen_interval(&mut self) {
        self.interval = self.interval.saturating_mul(2);
    }

    /// True once the server-side expiry has passed.
    pub fn is_expired(&self, now_ms: i64) -> bool {
        now_ms >= self.expires_at_ms
    }

    /// Perform one token poll attempt.
    pub async fn poll_once(&self, now_ms: i64) -> AuthResult<DeviceFlowPoll> {
        if self.is_expired(now_ms) {
            return Ok(DeviceFlowPoll::Expired);
        }
        let form = [
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ("device_code", self.device_code.as_str()),
            ("client_id", self.config.client_id.as_str()),
        ];
        let response = self
            .client
            .post(&self.config.token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(form_body(
                &form.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            ))
            .send()
            .await
            .map_err(|e| AuthError::Transport(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| AuthError::Transport(e.to_string()))?;
        let token: TokenResponse = serde_json::from_str(&body).map_err(|e| {
            AuthError::Transport(format!("token response: {e} (HTTP {})", status.as_u16()))
        })?;
        match token.error.as_deref() {
            Some("authorization_pending") => Ok(DeviceFlowPoll::Pending),
            Some("slow_down") => Ok(DeviceFlowPoll::SlowDown),
            Some("access_denied") => Ok(DeviceFlowPoll::Denied(
                token
                    .error_description
                    .unwrap_or_else(|| "access denied".to_string()),
            )),
            Some("expired_token") => Ok(DeviceFlowPoll::Expired),
            Some(other) => Err(AuthError::Transport(format!(
                "device flow error '{other}': {}",
                token.error_description.unwrap_or_default()
            ))),
            None if token.access_token.is_some() => Ok(DeviceFlowPoll::Complete(
                credential_from_token(&token, self.config.is_subscription, now_ms)?,
            )),
            None => Err(AuthError::Transport(format!(
                "token response missing access_token (HTTP {})",
                status.as_u16()
            ))),
        }
    }
}

/// Current wall clock in Unix milliseconds.
fn oauth_now_ms() -> i64 {
    llm_suite_common::time::now()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_flags_and_expiry_math() {
        // Expiry is computed by the caller from `expires_in`; the state only
        // compares timestamps.
        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.test/device".to_string(),
            token_url: "https://example.test/token".to_string(),
            client_id: "cid".to_string(),
            scope: None,
            is_subscription: true,
        };
        let pending = DeviceFlowPending {
            config,
            device_code: "dc".to_string(),
            user_code: "WDJB-MJHT".to_string(),
            verification_uri: "https://example.test/activate".to_string(),
            verification_uri_complete: None,
            interval: Duration::from_secs(5),
            expires_at_ms: 10_000,
            client: reqwest::Client::new(),
        };
        assert!(!pending.is_expired(9_999));
        assert!(pending.is_expired(10_000));
        assert_eq!(pending.interval(), Duration::from_secs(5));
        let mut widened = pending;
        widened.widen_interval();
        assert_eq!(widened.interval(), Duration::from_secs(10));
        match widened.challenge_event() {
            AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                ..
            } => {
                assert_eq!(user_code, "WDJB-MJHT");
                assert_eq!(verification_uri, "https://example.test/activate");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn token_response_parses_pending_and_slow_down() {
        let pending: TokenResponse =
            serde_json::from_str(r#"{"error":"authorization_pending"}"#).expect("parse");
        assert_eq!(pending.error.as_deref(), Some("authorization_pending"));
        let slow: TokenResponse = serde_json::from_str(r#"{"error":"slow_down"}"#).expect("parse");
        assert_eq!(slow.error.as_deref(), Some("slow_down"));
    }
}
