//! OAuth flow primitives shared by the device and PKCE flows.
//!
//! The primitives are provider-agnostic: endpoints, client ids and scopes
//! are supplied by the provider implementation that owns the login.

pub mod device_flow;
pub mod pkce;

use serde::Deserialize;

use crate::credential::OAuthCredential;
use crate::error::{AuthError, AuthResult};

/// Safety margin subtracted from `expires_in` so a token is refreshed
/// slightly before the server considers it expired.
pub const EXPIRY_SAFETY_MARGIN_MS: i64 = 5 * 60 * 1000;

/// Token endpoint response covering both success and error shapes.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenResponse {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_in: Option<i64>,
    pub token_type: Option<String>,
    pub scope: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

/// Build a stored credential from a token endpoint response.
pub fn credential_from_token(
    token: &TokenResponse,
    is_subscription: bool,
    now_ms: i64,
) -> AuthResult<OAuthCredential> {
    let access = token.access_token.clone().ok_or_else(|| {
        AuthError::Transport(format!(
            "token response missing access_token: {}",
            token
                .error_description
                .clone()
                .or_else(|| token.error.clone())
                .unwrap_or_else(|| "unknown error".to_string())
        ))
    })?;
    let refresh = token
        .refresh_token
        .clone()
        .ok_or_else(|| AuthError::Transport("token response missing refresh_token".to_string()))?;
    let expires =
        now_ms + token.expires_in.unwrap_or(3600).saturating_mul(1000) - EXPIRY_SAFETY_MARGIN_MS;
    Ok(OAuthCredential {
        refresh,
        access,
        expires,
        is_subscription,
        extra: Default::default(),
    })
}

/// Standard URL-encoded form body helper.
pub fn form_body(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Minimal URL-encoding without pulling a form crate.
mod urlencoding {
    pub fn encode(input: &str) -> String {
        let mut out = String::with_capacity(input.len() * 3);
        for byte in input.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(byte as char)
                }
                _ => out.push_str(&format!("%{:02X}", byte)),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_body_encodes_reserved_characters() {
        let body = form_body(&[("grant_type", "authorization_code"), ("scope", "a b")]);
        assert_eq!(body, "grant_type=authorization_code&scope=a%20b");
    }

    #[test]
    fn credential_from_token_applies_safety_margin() {
        let token = TokenResponse {
            access_token: Some("a".to_string()),
            refresh_token: Some("r".to_string()),
            expires_in: Some(3600),
            token_type: None,
            scope: None,
            error: None,
            error_description: None,
        };
        let cred = credential_from_token(&token, true, 1_000_000).expect("credential");
        assert_eq!(cred.access, "a");
        assert_eq!(cred.refresh, "r");
        assert_eq!(
            cred.expires,
            1_000_000 + 3_600_000 - EXPIRY_SAFETY_MARGIN_MS
        );
        assert!(cred.is_subscription);
    }

    #[test]
    fn credential_from_token_reports_error_payload() {
        let token = TokenResponse {
            access_token: None,
            refresh_token: Some("r".to_string()),
            expires_in: None,
            token_type: None,
            scope: None,
            error: Some("invalid_grant".to_string()),
            error_description: Some("bad code".to_string()),
        };
        let err = credential_from_token(&token, false, 0).expect_err("must fail");
        assert!(err.to_string().contains("bad code"));
    }
}
