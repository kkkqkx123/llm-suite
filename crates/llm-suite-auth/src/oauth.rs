//! OAuth flow primitives shared by the device and PKCE flows.
//!
//! The primitives are provider-agnostic: endpoints, client ids and scopes
//! are supplied by the provider implementation that owns the login.

pub mod device_flow;
pub mod pkce;

use serde::{Deserialize, Serialize};

use crate::credential::OAuthCredential;
use crate::error::{AuthError, AuthResult};

/// Encoding of an OAuth token endpoint request body.
///
/// Most providers accept the RFC-standard `application/x-www-form-urlencoded`
/// body. A minority (notably Anthropic) require the same fields as a JSON
/// object. The field set is identical in both modes; only the encoding and
/// the `Content-Type` header differ, so one enum covers the whole
/// provider matrix without provider-specific branches in the flows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenRequestContentType {
    /// `application/x-www-form-urlencoded` (RFC 6749 default).
    #[default]
    Form,
    /// `application/json` with the same fields as a JSON object.
    Json,
}

impl TokenRequestContentType {
    /// The `Content-Type` header value for this encoding.
    pub fn header_value(self) -> &'static str {
        match self {
            Self::Form => "application/x-www-form-urlencoded",
            Self::Json => "application/json",
        }
    }

    /// Encode ordered field pairs into the request body.
    pub fn encode(self, pairs: &[(&str, &str)]) -> String {
        match self {
            Self::Form => form_body(pairs),
            Self::Json => {
                let mut map = serde_json::Map::with_capacity(pairs.len());
                for (k, v) in pairs {
                    map.insert(
                        (*k).to_string(),
                        serde_json::Value::String((*v).to_string()),
                    );
                }
                serde_json::Value::Object(map).to_string()
            }
        }
    }
}

/// Build a POST request for an OAuth token endpoint with the configured
/// body encoding.
pub fn token_request(
    client: &reqwest::Client,
    url: &str,
    content_type: TokenRequestContentType,
    pairs: &[(&str, &str)],
) -> reqwest::RequestBuilder {
    client
        .post(url)
        .header("Content-Type", content_type.header_value())
        .body(content_type.encode(pairs))
}

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
///
/// The stored `expires` is the server-reported expiry verbatim; the refresh
/// leeway lives at resolve time (`DEFAULT_MIN_OAUTH_VALIDITY_MS`) so there
/// is exactly one early-refresh policy.
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
    let expires = now_ms + token.expires_in.unwrap_or(3600).saturating_mul(1000);
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
    fn credential_from_token_stores_verbatim_expiry() {
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
        assert_eq!(cred.expires, 1_000_000 + 3_600_000);
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

    #[test]
    fn token_content_type_defaults_to_form_and_encodes_both_modes() {
        assert_eq!(
            TokenRequestContentType::default(),
            TokenRequestContentType::Form
        );
        assert_eq!(
            TokenRequestContentType::Form.header_value(),
            "application/x-www-form-urlencoded"
        );
        assert_eq!(
            TokenRequestContentType::Json.header_value(),
            "application/json"
        );
        let pairs = [("grant_type", "authorization_code"), ("scope", "a b")];
        assert_eq!(
            TokenRequestContentType::Form.encode(&pairs),
            "grant_type=authorization_code&scope=a%20b"
        );
        let json = TokenRequestContentType::Json.encode(&pairs);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("json body parses");
        assert_eq!(parsed["grant_type"], "authorization_code");
        assert_eq!(parsed["scope"], "a b");
    }
}
