use serde::{Deserialize, Serialize};

/// Client identity headers sent with every request (Phase 4).
///
/// Mirrors pi's `identity` (User-Agent / attribution): `user_agent` becomes
/// the `User-Agent` header, while `app_url` / `app_name` become the
/// OpenRouter-style attribution pair `HTTP-Referer` / `X-Title`. All fields
/// are optional; absent fields send no header so existing traffic is
/// byte-identical until a host opts in.
///
/// Merge rule (mirrors `ProviderCompat`): a profile-level value wins per
/// field over the referenced provider definition. The codec injects the
/// merged result; per-request `LlmRequest.extra_headers` still win over
/// both because the client applies them after the codec builds the request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmIdentity {
    /// Value for the `User-Agent` header (e.g. `"my-app/1.2.3"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// Application display name, sent as `X-Title` for gateway analytics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
    /// Application URL, sent as `HTTP-Referer` for gateway analytics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_url: Option<String>,
}

impl LlmIdentity {
    /// True when no identity header would be emitted.
    pub fn is_empty(&self) -> bool {
        self.user_agent.is_none() && self.app_name.is_none() && self.app_url.is_none()
    }

    /// Overlay `over` on top of `self` field by field; `Some` wins.
    pub fn merge_over(&mut self, over: &LlmIdentity) {
        if over.user_agent.is_some() {
            self.user_agent = over.user_agent.clone();
        }
        if over.app_name.is_some() {
            self.app_name = over.app_name.clone();
        }
        if over.app_url.is_some() {
            self.app_url = over.app_url.clone();
        }
    }
}

/// Connection-level HTTP transport tuning shared by every client of a
/// profile (Phase 4).
///
/// Only knobs the transport constructor consumes live here: adding a field
/// requires a branch in `llm-suite-proxy` plus a transport-cache-key entry
/// and a test. Per-request overrides (`LlmRequest.timeout_ms`,
/// `max_retries`, `extra_headers`) stay on the request; payload mutation
/// stays on `custom_body`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmHttpTransport {
    /// TCP/TLS connect timeout in seconds for the HTTP client. Absent keeps
    /// the reqwest default. Part of the transport cache key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    /// Accept invalid TLS certificates (self-signed local gateways). Must be
    /// an explicit opt-in; absent/`false` keeps verification. Part of the
    /// transport cache key; enabling it logs a warning at client build time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub danger_accept_invalid_certs: Option<bool>,
}

impl LlmHttpTransport {
    /// True when no transport knob deviates from the default client.
    pub fn is_empty(&self) -> bool {
        self.connect_timeout_secs.is_none() && self.danger_accept_invalid_certs.is_none()
    }

    /// Overlay `over` on top of `self` field by field; `Some` wins.
    pub fn merge_over(&mut self, over: &LlmHttpTransport) {
        if over.connect_timeout_secs.is_some() {
            self.connect_timeout_secs = over.connect_timeout_secs;
        }
        if over.danger_accept_invalid_certs.is_some() {
            self.danger_accept_invalid_certs = over.danger_accept_invalid_certs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_merge_prefers_profile_fields() {
        let mut base = LlmIdentity {
            user_agent: Some("base/1.0".to_string()),
            app_name: Some("base".to_string()),
            app_url: None,
        };
        base.merge_over(&LlmIdentity {
            user_agent: None,
            app_name: Some("override".to_string()),
            app_url: Some("https://example.com".to_string()),
        });
        assert_eq!(base.user_agent.as_deref(), Some("base/1.0"));
        assert_eq!(base.app_name.as_deref(), Some("override"));
        assert_eq!(base.app_url.as_deref(), Some("https://example.com"));
        assert!(!base.is_empty());
        assert!(LlmIdentity::default().is_empty());
    }

    #[test]
    fn transport_merge_prefers_profile_fields() {
        let mut base = LlmHttpTransport {
            connect_timeout_secs: Some(5),
            danger_accept_invalid_certs: None,
        };
        base.merge_over(&LlmHttpTransport {
            connect_timeout_secs: None,
            danger_accept_invalid_certs: Some(true),
        });
        assert_eq!(base.connect_timeout_secs, Some(5));
        assert_eq!(base.danger_accept_invalid_certs, Some(true));
        assert!(!base.is_empty());
        assert!(LlmHttpTransport::default().is_empty());
    }
}
