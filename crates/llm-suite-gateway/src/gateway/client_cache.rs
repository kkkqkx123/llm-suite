//! The cached transport clients, keyed by the profile they were built for.

use std::sync::Arc;

use llm_suite_client::client::LlmClientImpl;
use llm_suite_codec::error::{LlmError, LlmResult};
use llm_suite_types::llm::LlmProfile;

use super::LlmGateway;

/// The prefix every cache key of a profile starts with.
///
/// The key is built from the profile id, so the prefix identifies every
/// client that belongs to the profile and eviction drops all of them at
/// once.
pub fn client_cache_prefix(profile_id: &str) -> String {
    format!("{profile_id}::")
}

/// Cache key for the transport client. The proxy is part of the key so
/// switching proxies rebuilds the transport instead of reusing a stale
/// client. Auth-derived fields are folded in as a stable non-secret digest
/// so an OAuth refresh (or key rotation) rebuilds the transport instead of
/// reusing a client bound to the old credential. Connection-level transport
/// knobs (connect timeout, invalid-certificate opt-in) are part of the key
/// for the same reason.
///
/// Identity headers, the env overlay and per-request overrides are *not* in
/// the key: identity/env only affect request headers (already covered via
/// the auth digest when they change the resolved credential), and
/// per-request overrides apply after the cached client is fetched.
pub fn client_cache_key(profile: &LlmProfile) -> String {
    format!(
        "{}{}::{}::{}::{}::{}",
        client_cache_prefix(&profile.id),
        profile.model,
        profile.proxy.as_deref().unwrap_or(""),
        profile.no_proxy.as_deref().unwrap_or(&[]).join(","),
        auth_fingerprint(profile),
        transport_fingerprint(profile)
    )
}

/// Stable rendering of the connection-level transport knobs.
///
/// Only fields that change the built `reqwest::Client` belong here; the
/// gateway logs a warning when the invalid-certificate opt-in builds such
/// a client.
fn transport_fingerprint(profile: &LlmProfile) -> String {
    let transport = profile.transport.as_ref();
    format!(
        "ct={}::insecure={}",
        transport
            .and_then(|t| t.connect_timeout_secs)
            .map(|secs| secs.to_string())
            .unwrap_or_else(|| "-".to_string()),
        transport
            .and_then(|t| t.danger_accept_invalid_certs)
            .unwrap_or(false)
    )
}

/// Stable non-secret digest of the auth-derived profile fields.
///
/// The digest never contains the key itself so the cache key is safe to keep
/// in memory and to compare in tests; any credential rotation changes the
/// digest and forces a fresh transport client.
fn auth_fingerprint(profile: &LlmProfile) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for b in bytes {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    match profile.api_key.as_deref() {
        Some(key) => {
            mix(b"key");
            mix(key.as_bytes());
        }
        None => mix(b"no-key"),
    }
    match profile.base_url.as_deref() {
        Some(url) => {
            mix(b"base");
            mix(url.as_bytes());
        }
        None => mix(b"no-base"),
    }
    if let Some(headers) = profile.custom_headers.as_ref() {
        let mut keys: Vec<&String> = headers.keys().collect();
        keys.sort();
        mix(b"headers");
        for key in keys {
            mix(key.as_bytes());
            mix(b"=");
            if let Some(value) = headers.get(key) {
                mix(value.to_string().as_bytes());
            }
            mix(b";");
        }
    } else {
        mix(b"no-headers");
    }
    format!("{:016x}", hash)
}

impl LlmGateway {
    pub(super) fn get_or_create_client(
        &self,
        profile: &LlmProfile,
    ) -> LlmResult<Arc<LlmClientImpl>> {
        let key = client_cache_key(profile);

        if let Some(client) = self.clients.get(key.as_str()) {
            return Ok(client.clone());
        }

        let codec = self.codecs.get_by_format(&profile.format)?;
        let no_proxy = profile
            .no_proxy
            .clone()
            .or_else(|| {
                profile
                    .provider_id
                    .as_ref()
                    .and_then(|id| self.providers.get(id))
                    .and_then(|def| def.no_proxy.clone())
            })
            .unwrap_or_default();
        let transport = profile.transport.clone().unwrap_or_default();
        if transport.danger_accept_invalid_certs.unwrap_or(false) {
            tracing::warn!(
                profile_id = profile.id.as_str(),
                "building HTTP client with invalid TLS certificates accepted; \
                 enable only for local gateways"
            );
        }
        let client = llm_suite_proxy::build_http_client_with_options(
            profile.timeout.unwrap_or(60),
            profile.proxy.as_deref(),
            &no_proxy,
            &llm_suite_proxy::HttpTransportOptions {
                connect_timeout_secs: transport.connect_timeout_secs,
                danger_accept_invalid_certs: transport
                    .danger_accept_invalid_certs
                    .unwrap_or(false),
            },
        )
        .map_err(|err| LlmError::ProxyError(err.to_string()))?;

        let client_impl = Arc::new(self.build_client(client, codec, profile));
        self.clients.insert(key, client_impl.clone());
        Ok(client_impl)
    }

    /// Build the transport client for a profile and wire the retry
    /// observation into the metrics sink.
    ///
    /// The observer reports every scheduled retry attempt, so the retry
    /// count reflects what the transport actually did instead of a proxy
    /// derived from the final outcome.
    fn build_client(
        &self,
        client: reqwest::Client,
        codec: Arc<dyn llm_suite_codec::LlmCodec>,
        profile: &LlmProfile,
    ) -> LlmClientImpl {
        let mut impl_client = LlmClientImpl::new(client, codec, profile.clone());
        if let Some(hook) = self.fetch_hook {
            impl_client = impl_client.with_fetch_hook(hook);
        }
        let Some(metrics) = self.token_metrics.as_ref() else {
            return impl_client;
        };
        let metrics = Arc::clone(metrics);
        let model = profile.model.clone();
        let profile_id = profile.id.clone();
        let provider_id = profile.provider_id.clone();
        let format = profile.format.clone();
        let accounting = llm_token::token_accounting(&format);
        impl_client.with_retry_observer(Arc::new(move |_descriptor| {
            metrics.record_retry(&llm_token::LlmCallContext {
                model: Some(model.clone()),
                profile_id: Some(profile_id.clone()),
                provider_id: provider_id.clone(),
                format: Some(format.as_str().to_string()),
                execution_id: None,
                accounting,
            });
        }))
    }

    /// Drop every cached client that belongs to the given profile id.
    pub(super) fn evict_clients_for_profile(&self, profile_id: &str) {
        let prefix = client_cache_prefix(profile_id);
        self.clients.retain(|key, _| !key.starts_with(&prefix));
    }

    /// Public alias of `evict_clients_for_profile` for hosts that evict
    /// after a login, logout or profile update so the next request rebuilds
    /// the transport with the rotated credential.
    pub fn evict_client(&self, profile_id: &str) {
        self.evict_clients_for_profile(profile_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> LlmProfile {
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
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    #[test]
    fn cache_key_changes_when_auth_fields_rotate() {
        let base = test_profile();
        let mut rotated = base.clone();
        rotated.api_key = Some("sk-new".to_string());
        assert_ne!(
            client_cache_key(&base),
            client_cache_key(&rotated),
            "rotated api_key must rebuild the transport"
        );

        let mut moved_base = base.clone();
        moved_base.base_url = Some("https://new.example/v1".to_string());
        assert_ne!(
            client_cache_key(&base),
            client_cache_key(&moved_base),
            "rotated base_url must rebuild the transport"
        );
    }

    #[test]
    fn cache_key_hides_secrets() {
        let mut profile = test_profile();
        profile.api_key = Some("sk-secret-value".to_string());
        let key = client_cache_key(&profile);
        assert!(
            !key.contains("sk-secret-value"),
            "cache key must carry only a digest, never the secret: {key}"
        );
        assert!(
            key.starts_with(&client_cache_prefix("p1")),
            "prefix eviction must keep working: {key}"
        );
    }

    #[test]
    fn cache_key_changes_when_custom_headers_rotate() {
        let base = test_profile();
        let mut rotated = base.clone();
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-api-key".to_string(), serde_json::json!("old"));
        rotated.custom_headers = Some(headers.clone());
        let key_base = client_cache_key(&base);

        let mut changed = base.clone();
        let mut headers2 = headers;
        headers2.insert("x-api-key".to_string(), serde_json::json!("new"));
        changed.custom_headers = Some(headers2);
        assert_ne!(
            key_base,
            client_cache_key(&changed),
            "rotated custom header must rebuild the transport"
        );

        // Adding a header to an otherwise identical profile also changes the key.
        let mut added = base.clone();
        let mut h = std::collections::HashMap::new();
        h.insert("x-extra".to_string(), serde_json::json!(42));
        added.custom_headers = Some(h);
        assert_ne!(key_base, client_cache_key(&added));
    }

    #[test]
    fn cache_key_is_deterministic_for_reordered_headers() {
        let mut a = test_profile();
        let mut headers = std::collections::HashMap::new();
        headers.insert("b".to_string(), serde_json::json!("2"));
        headers.insert("a".to_string(), serde_json::json!("1"));
        a.custom_headers = Some(headers);

        let mut b = test_profile();
        let mut headers2 = std::collections::HashMap::new();
        headers2.insert("a".to_string(), serde_json::json!("1"));
        headers2.insert("b".to_string(), serde_json::json!("2"));
        b.custom_headers = Some(headers2);

        assert_eq!(client_cache_key(&a), client_cache_key(&b));
    }

    #[test]
    fn cache_key_stable_for_identical_profiles() {
        let a = test_profile();
        let b = test_profile();
        assert_eq!(client_cache_key(&a), client_cache_key(&b));
    }

    #[test]
    fn cache_key_changes_when_transport_knobs_rotate() {
        let base = test_profile();
        let mut with_timeout = base.clone();
        with_timeout.transport =
            Some(llm_suite_types::llm::LlmHttpTransport {
                connect_timeout_secs: Some(2),
                danger_accept_invalid_certs: None,
            });
        assert_ne!(
            client_cache_key(&base),
            client_cache_key(&with_timeout),
            "connect timeout must rebuild the transport"
        );

        let mut insecure = base.clone();
        insecure.transport = Some(llm_suite_types::llm::LlmHttpTransport {
            connect_timeout_secs: None,
            danger_accept_invalid_certs: Some(true),
        });
        assert_ne!(
            client_cache_key(&base),
            client_cache_key(&insecure),
            "the invalid-certificate opt-in must rebuild the transport"
        );
    }

    #[test]
    fn cache_key_ignores_identity_and_env() {
        let base = test_profile();
        let mut attributed = base.clone();
        attributed.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: Some("my-app/1.0".to_string()),
            app_name: Some("my app".to_string()),
            app_url: None,
        });
        attributed.env = Some(
            [("TEST_KEY".to_string(), "sk-overlay".to_string())]
                .into_iter()
                .collect(),
        );
        assert_eq!(
            client_cache_key(&base),
            client_cache_key(&attributed),
            "identity headers and the env overlay are request-scoped, \
             not connection-scoped"
        );
    }
}
