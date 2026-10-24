//! Stored credential shapes: exactly one type-tagged credential per provider.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AuthError, AuthResult};

/// Authentication method selected for a provider.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    ApiKey,
    #[serde(rename = "oauth")]
    OAuth,
}

impl fmt::Display for AuthMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey => f.write_str("api_key"),
            Self::OAuth => f.write_str("oauth"),
        }
    }
}

/// Non-secret metadata for account and status enumeration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub method: AuthMethod,
    /// True when the credential grants subscription (non-metered) access.
    pub is_subscription: bool,
}

/// API-key credential. `env` carries provider-scoped config values such as
/// gateway account ids.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ApiKeyCredential {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
}

/// OAuth credential. `expires` is a Unix-milliseconds absolute timestamp;
/// a small safety margin is already subtracted by the flow that produced it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OAuthCredential {
    pub refresh: String,
    pub access: String,
    pub expires: i64,
    /// True when access through this credential is backed by a subscription.
    #[serde(default)]
    pub is_subscription: bool,
    /// Provider-specific extras (account id, scope, ...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}

impl OAuthCredential {
    /// Remaining validity in milliseconds; zero once expired.
    pub fn remaining_ms(&self, now_ms: i64) -> i64 {
        self.expires.saturating_sub(now_ms).max(0)
    }

    /// True when the credential is expired at `now_ms`.
    pub fn is_expired(&self, now_ms: i64) -> bool {
        self.remaining_ms(now_ms) <= 0
    }
}

/// One credential per provider, tagged by authentication method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey(ApiKeyCredential),
    #[serde(rename = "oauth")]
    OAuth(OAuthCredential),
}

impl Credential {
    /// The method this credential authenticates with.
    pub fn method(&self) -> AuthMethod {
        match self {
            Self::ApiKey(_) => AuthMethod::ApiKey,
            Self::OAuth(_) => AuthMethod::OAuth,
        }
    }

    /// Non-secret metadata for enumeration.
    pub fn info(&self, provider_id: impl Into<String>) -> CredentialInfo {
        CredentialInfo {
            provider_id: provider_id.into(),
            method: self.method(),
            is_subscription: matches!(self, Self::OAuth(c) if c.is_subscription),
        }
    }

    /// Borrow as an API-key credential, failing on type mismatch.
    pub fn as_api_key(&self) -> AuthResult<&ApiKeyCredential> {
        match self {
            Self::ApiKey(c) => Ok(c),
            Self::OAuth(_) => Err(AuthError::MalformedCredential(
                "expected api_key credential".to_string(),
            )),
        }
    }

    /// Borrow as an OAuth credential, failing on type mismatch.
    pub fn as_oauth(&self) -> AuthResult<&OAuthCredential> {
        match self {
            Self::OAuth(c) => Ok(c),
            Self::ApiKey(_) => Err(AuthError::MalformedCredential(
                "expected oauth credential".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oauth() -> OAuthCredential {
        OAuthCredential {
            refresh: "r-1".to_string(),
            access: "a-1".to_string(),
            expires: 1_000,
            is_subscription: true,
            extra: BTreeMap::from([("account".to_string(), Value::from("acct-1"))]),
        }
    }

    #[test]
    fn oauth_expiry_math() {
        let c = oauth();
        assert_eq!(c.remaining_ms(900), 100);
        assert!(!c.is_expired(900));
        assert!(c.is_expired(1_000));
        assert_eq!(c.remaining_ms(2_000), 0);
    }

    #[test]
    fn credential_serde_roundtrip_keeps_tag_and_extras() {
        let value = serde_json::to_value(Credential::OAuth(oauth())).expect("serialize");
        assert_eq!(value["type"], "oauth");
        assert_eq!(value["is_subscription"], true);
        assert_eq!(value["extra"]["account"], "acct-1");
        let back: Credential = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back.method(), AuthMethod::OAuth);
        assert!(back.as_oauth().expect("oauth").is_subscription);
    }

    #[test]
    fn api_key_credential_roundtrip_and_info() {
        let cred = Credential::ApiKey(ApiKeyCredential {
            key: Some("sk-test".to_string()),
            env: BTreeMap::from([("GW".to_string(), "g1".to_string())]),
        });
        let json = serde_json::to_string(&cred).expect("serialize");
        assert!(json.contains("\"type\":\"api_key\""));
        let back: Credential = serde_json::from_str(&json).expect("deserialize");
        let info = back.info("acme");
        assert_eq!(info.method, AuthMethod::ApiKey);
        assert!(!info.is_subscription);
        assert_eq!(info.provider_id, "acme");
        assert!(back.as_api_key().expect("api key").key.is_some());
        assert!(back.as_oauth().is_err());
    }

    #[test]
    fn auth_method_display() {
        assert_eq!(AuthMethod::ApiKey.to_string(), "api_key");
        assert_eq!(AuthMethod::OAuth.to_string(), "oauth");
    }
}
