//! Gateway orchestration: mandatory profile resolution, request merging,
//! client caching, mock routing and metrics hooks.

use std::sync::Arc;

use dashmap::DashMap;
use llm_suite_client::SharedLlmMetricsSink;
use llm_suite_codec::CodecRegistry;
use llm_suite_config::profile::ProfileManager;
use llm_suite_config::provider::ProviderDefinitionRegistry;
use llm_suite_discovery::ModelCatalog;
use llm_token::PricingResolver;

pub use probe::RequestProbe;

pub(crate) mod client_cache;
pub(crate) mod dispatch;
pub(crate) mod images;
pub(crate) mod metrics;
pub(crate) mod migrate;
pub(crate) mod mock;
pub(crate) mod prepare;
pub(crate) mod probe;
pub(crate) mod registry;
pub(crate) mod resilience;

/// Single facade for all LLM calls.
///
/// Responsibilities:
/// - resolve the profile for a mandatory `profile_id` (no fallback branch)
/// - route to mock clients (test injection) or real clients
/// - resolve codecs through the registry (built-ins + runtime custom)
/// - record token usage metrics for both generate and stream paths
///
/// Request merging delegates to `llm_suite_config::merge_request`; metrics wrapping
/// delegates to `token::stream`.
#[derive(Clone)]
pub struct LlmGateway {
    clients: Arc<DashMap<String, Arc<llm_suite_client::client::LlmClientImpl>>>,
    profiles: ProfileManager,
    codecs: CodecRegistry,
    providers: ProviderDefinitionRegistry,
    model_catalog: ModelCatalog,
    /// Circuit breakers keyed by the resolved endpoint base URL, shared by
    /// all profiles pointing at the same provider.
    circuit_breakers: Arc<DashMap<String, Arc<llm_suite_client::CircuitBreaker>>>,
    /// Rate limiters keyed by the resolved endpoint base URL, shared by
    /// all profiles pointing at the same provider.
    rate_limiters: Arc<DashMap<String, Arc<llm_suite_common::ratelimit::RateLimiter>>>,
    #[cfg(feature = "mock")]
    mock_clients: Arc<DashMap<String, Arc<llm_suite_client::MockLlmClient>>>,
    token_metrics: Option<SharedLlmMetricsSink>,
    /// Optional pricing resolver that enriches recorded token usage with a
    /// computed cost. When present, `record_token_usage` resolves the
    /// (provider, model) price and attaches `total_cost` to the usage before
    /// handing it to the metrics sink. Zero overhead when absent.
    pricing_resolver: Option<Arc<dyn PricingResolver>>,
    /// Optional auth registry resolving stored credentials and OAuth tokens
    /// for profiles carrying an auth reference. Zero overhead when absent:
    /// profiles without a reference keep their static-key behavior.
    auth: Option<Arc<llm_suite_auth::AuthRegistry>>,
    /// Optional request observation hooks (concrete `fn` pointers, no
    /// dynamic dispatch). Empty by default; the hot path checks
    /// `is_empty` once per call.
    probe: probe::RequestProbe,
    /// Optional custom fetch hook rewriting the codec-built request before
    /// send (Phase 4). A plain function pointer wired into every transport
    /// client the gateway builds; `None` keeps the direct send path.
    fetch_hook: Option<llm_suite_client::client::FetchHook>,
}

impl LlmGateway {
    pub fn new() -> Self {
        Self::new_with_codec_registry(CodecRegistry::new())
    }

    /// Create a gateway with a caller-provided codec registry (custom
    /// formats must be registered on the registry before first use).
    pub fn new_with_codec_registry(codecs: CodecRegistry) -> Self {
        Self {
            clients: Arc::new(DashMap::new()),
            profiles: ProfileManager::new(),
            codecs,
            providers: ProviderDefinitionRegistry::new(),
            model_catalog: ModelCatalog::new(),
            circuit_breakers: Arc::new(DashMap::new()),
            rate_limiters: Arc::new(DashMap::new()),
            #[cfg(feature = "mock")]
            mock_clients: Arc::new(DashMap::new()),
            token_metrics: None,
            pricing_resolver: None,
            auth: None,
            probe: probe::RequestProbe::default(),
            fetch_hook: None,
        }
    }

    /// Attach an optional token usage collector (zero overhead when absent).
    pub fn with_token_metrics(mut self, token_metrics: SharedLlmMetricsSink) -> Self {
        self.token_metrics = Some(token_metrics);
        self
    }

    /// Attach an optional pricing resolver that enriches recorded token usage
    /// with a computed cost (see `record_token_usage`). Zero overhead when absent.
    pub fn with_pricing_resolver(mut self, resolver: Arc<dyn PricingResolver>) -> Self {
        self.pricing_resolver = Some(resolver);
        self
    }

    /// Attach request observation hooks (concrete `fn` pointers, no dynamic
    /// dispatch). The gateway invokes them around dispatch; `None` hooks
    /// cost a single branch check.
    pub fn with_request_probe(mut self, probe: probe::RequestProbe) -> Self {
        self.probe = probe;
        self
    }

    /// The currently attached request probe (empty by default).
    pub fn request_probe(&self) -> &probe::RequestProbe {
        &self.probe
    }

    /// Attach a custom fetch hook rewriting the codec-built request before
    /// send (tests, header-injecting middleware). Runs after per-request
    /// `extra_headers`; payload mutation stays on `custom_body`.
    pub fn with_fetch_hook(mut self, hook: llm_suite_client::client::FetchHook) -> Self {
        self.fetch_hook = Some(hook);
        self
    }

    /// The codec registry: register custom formats here before the
    /// first request that uses them.
    pub fn codec_registry(&self) -> &CodecRegistry {
        &self.codecs
    }
}

impl Default for LlmGateway {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_codec::error::{LlmError, LlmResult};
    use llm_suite_codec::LlmCodec;
    use llm_suite_types::llm::{LlmFormat, LlmProfile, LlmRequest};
    use llm_suite_types::tool::Tool;

    use crate::gateway::client_cache::client_cache_key;
    use llm_suite_auth::CredentialStore;
    use llm_suite_codec::CodecRegistry;

    /// A codec whose `build_request` fails with a distinctive error, used
    /// to prove the gateway resolved the *custom* codec from the registry.
    struct ProbeCodec;

    impl LlmCodec for ProbeCodec {
        fn build_request(
            &self,
            _request: &LlmRequest,
            _profile: &LlmProfile,
        ) -> LlmResult<reqwest::Request> {
            Err(LlmError::ConfigError("custom codec engaged".to_string()))
        }

        fn parse_response(
            &self,
            _body: &str,
            _request: &LlmRequest,
        ) -> LlmResult<llm_suite_types::llm::LlmResult> {
            Err(LlmError::ConfigError("custom codec engaged".to_string()))
        }

        fn convert_tools(&self, _tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
            Ok(Vec::new())
        }

        fn parse_tool_calls(
            &self,
            _result: &llm_suite_types::llm::LlmResult,
        ) -> Vec<llm_suite_types::message::LlmToolCall> {
            Vec::new()
        }
    }

    fn profile(id: &str, format: LlmFormat) -> LlmProfile {
        LlmProfile {
            id: id.to_string(),
            name: id.to_string(),
            format,
            provider_id: None,
            model: "custom-model".to_string(),
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
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    fn request(profile_id: &str) -> LlmRequest {
        LlmRequest {
            profile_id: profile_id.to_string(),
            messages: Vec::new(),
            parameters: None,
            generation: None,
            tools: None,
            tool_call_protocol: None,
            locked_tool_call_protocol: None,
            violation_policy: None,
            execution_id: None,
            stream: None,
            dead_loop_detection: None,
            protocol_auto_converted: None,
            timeout_ms: None,
            prompt_caching: None,
            cached_content: None,
            max_retries: None,
            retry_delay_ms: None,
            extra_headers: None,
        }
    }

    #[tokio::test]
    async fn custom_codec_resolved_via_registry() {
        let registry = CodecRegistry::new();
        registry
            .register("my_custom_provider", Arc::new(ProbeCodec))
            .expect("custom registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(registry);
        gateway
            .register_profile(profile(
                "p1",
                LlmFormat::Custom("my_custom_provider".to_string()),
            ))
            .unwrap();

        let err = gateway.generate(&request("p1"), None).await.unwrap_err();
        assert!(
            err.to_string().contains("custom codec engaged"),
            "custom codec must have been selected: {}",
            err
        );
    }

    #[tokio::test]
    async fn unregistered_custom_format_errors_before_http() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::Custom("nope".to_string())))
            .unwrap();

        let err = gateway.generate(&request("p1"), None).await.unwrap_err();
        assert!(matches!(err, LlmError::CodecNotFound(_)));
    }

    #[tokio::test]
    async fn empty_profile_id_resolves_to_default() {
        let registry = CodecRegistry::new();
        registry
            .register("probe", Arc::new(ProbeCodec))
            .expect("registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(registry);
        gateway
            .register_profile(profile("p1", LlmFormat::Custom("probe".to_string())))
            .unwrap();

        // The custom codec fails inside `build_request` before any HTTP;
        // reaching it proves the default profile was resolved.
        let err = gateway.generate(&request(""), None).await.unwrap_err();
        assert!(
            err.to_string().contains("custom codec engaged"),
            "empty id must resolve the default profile: {err}"
        );
    }

    #[tokio::test]
    async fn empty_profile_id_errors_without_default() {
        let gateway = LlmGateway::new();
        let err = gateway.generate(&request(""), None).await.unwrap_err();
        assert!(matches!(err, LlmError::ProfileNotFound(_)));
    }

    #[test]
    fn remove_profile_evicts_cached_clients() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();
        gateway
            .register_profile(profile("p2", LlmFormat::OpenaiChat))
            .unwrap();

        let client = gateway
            .get_or_create_client(&gateway.profiles.get("p1").unwrap())
            .unwrap();
        let key = client_cache_key(client.profile());
        assert!(gateway.clients.contains_key(key.as_str()));

        assert!(gateway.remove_profile("p1").is_some());
        assert!(
            !gateway.clients.contains_key(key.as_str()),
            "clients of the removed profile must be evicted"
        );
        assert!(!gateway.has_profile("p1"));
        assert!(gateway.has_profile("p2"));
    }

    #[test]
    fn remove_profile_preserves_other_clients() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();
        gateway
            .register_profile(profile("p2", LlmFormat::OpenaiChat))
            .unwrap();

        let client = gateway
            .get_or_create_client(&gateway.profiles.get("p2").unwrap())
            .unwrap();
        let key = client_cache_key(client.profile());

        gateway.remove_profile("p1");
        assert!(
            gateway.clients.contains_key(key.as_str()),
            "unrelated clients must survive"
        );
    }

    fn provider_definition() -> llm_suite_types::llm::LlmProviderDefinition {
        llm_suite_types::llm::LlmProviderDefinition {
            id: "acme".to_string(),
            name: None,
            description: None,
            base_url: Some("https://api.acme.test".to_string()),
            auth_type: Some("bearer".to_string()),
            default_headers: None,
            format: LlmFormat::OpenaiChat,
            model_discovery: None,
            api_version: None,
            metadata: None,
            proxy: None,
            no_proxy: None,
            rate_limit: None,
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    #[test]
    fn register_profile_merges_provider_defaults() {
        let gateway = LlmGateway::new();
        gateway
            .register_provider_definition(provider_definition())
            .unwrap();

        let mut pending = profile("p1", LlmFormat::OpenaiChat);
        pending.provider_id = Some("acme".to_string());
        gateway.register_profile(pending).unwrap();

        let stored = gateway.profiles.get("p1").unwrap();
        assert_eq!(stored.base_url.as_deref(), Some("https://api.acme.test"));
        assert_eq!(stored.auth_type.as_deref(), Some("bearer"));
    }

    #[test]
    fn register_profile_with_unknown_provider_fails() {
        let gateway = LlmGateway::new();
        let mut pending = profile("p1", LlmFormat::OpenaiChat);
        pending.provider_id = Some("nope".to_string());
        assert!(gateway.register_profile(pending).is_err());
    }

    #[test]
    fn provider_definition_change_evicts_clients() {
        let gateway = LlmGateway::new();
        gateway
            .register_provider_definition(provider_definition())
            .unwrap();
        let mut pending = profile("p1", LlmFormat::OpenaiChat);
        pending.provider_id = Some("acme".to_string());
        gateway.register_profile(pending).unwrap();

        let stored = gateway.profiles.get("p1").unwrap();
        let client = gateway.get_or_create_client(&stored).unwrap();
        let key = client_cache_key(client.profile());
        assert!(gateway.clients.contains_key(key.as_str()));

        gateway
            .register_provider_definition(provider_definition())
            .unwrap();
        assert!(
            !gateway.clients.contains_key(key.as_str()),
            "provider change must evict cached clients"
        );
    }

    #[test]
    fn proxy_change_rebuilds_cached_client() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();

        let stored = gateway.profiles.get("p1").unwrap();
        gateway.get_or_create_client(&stored).unwrap();
        let direct_key = client_cache_key(&stored);
        assert!(gateway.clients.contains_key(direct_key.as_str()));

        let mut proxied = stored.clone();
        proxied.proxy = Some("http://127.0.0.1:8080".to_string());
        gateway.get_or_create_client(&proxied).unwrap();
        let proxied_key = client_cache_key(&proxied);
        assert_ne!(direct_key, proxied_key);
        assert!(gateway.clients.contains_key(proxied_key.as_str()));
    }

    #[test]
    fn invalid_proxy_url_fails_as_proxy_error() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();

        let mut stored = gateway.profiles.get("p1").unwrap();
        stored.proxy = Some("://not a url".to_string());
        assert!(matches!(
            gateway.get_or_create_client(&stored),
            Err(LlmError::ProxyError(_))
        ));
    }

    #[test]
    fn proxy_error_hides_credentials() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();

        let mut stored = gateway.profiles.get("p1").unwrap();
        stored.proxy = Some("ftp://user:secret@127.0.0.1:2121".to_string());
        let error = match gateway.get_or_create_client(&stored) {
            Err(LlmError::ProxyError(message)) => message,
            Err(other) => panic!("unsupported scheme must fail as a proxy error, got {other}"),
            Ok(_) => panic!("unsupported scheme must fail as a proxy error"),
        };

        assert!(error.contains("ftp"), "{error}");
        assert!(
            !error.contains("secret") && !error.contains("127.0.0.1"),
            "credentials must stay out of the message: {error}"
        );
    }

    #[test]
    fn bypass_list_change_rebuilds_cached_client() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();
        let stored = gateway.profiles.get("p1").unwrap();
        gateway.get_or_create_client(&stored).unwrap();

        let mut bypassed = stored.clone();
        bypassed.proxy = Some("http://127.0.0.1:8080".to_string());
        bypassed.no_proxy = Some(vec!["localhost".to_string()]);
        let single_key = client_cache_key(&bypassed);
        gateway.get_or_create_client(&bypassed).unwrap();

        bypassed.no_proxy = Some(vec!["localhost".to_string(), "10.0.0.0/8".to_string()]);
        let wider_key = client_cache_key(&bypassed);
        assert_ne!(
            single_key, wider_key,
            "a different bypass list must rebuild the transport"
        );
    }

    /// OAuth surface whose stored credential is subscription-backed.
    #[derive(Debug)]
    struct SubscriptionOAuth;

    #[async_trait::async_trait]
    impl llm_suite_auth::OAuthAuth for SubscriptionOAuth {
        fn name(&self) -> String {
            "subscription oauth".to_string()
        }

        fn is_subscription(&self) -> bool {
            true
        }

        async fn begin_login(
            &self,
        ) -> llm_suite_auth::AuthResult<llm_suite_auth::session::LoginFlow> {
            Err(llm_suite_auth::AuthError::Login(
                "not used in gateway tests".to_string(),
            ))
        }

        async fn refresh(
            &self,
            credential: &llm_suite_auth::OAuthCredential,
        ) -> llm_suite_auth::AuthResult<llm_suite_auth::OAuthCredential> {
            Ok(credential.clone())
        }

        async fn to_auth(
            &self,
            credential: &llm_suite_auth::OAuthCredential,
        ) -> llm_suite_auth::AuthResult<llm_suite_auth::ModelAuth> {
            Ok(llm_suite_auth::ModelAuth {
                api_key: Some(credential.access.clone()),
                is_subscription: self.is_subscription(),
                source: Some("OAuth".to_string()),
                ..Default::default()
            })
        }
    }

    async fn auth_registry_with_stored_oauth() -> Arc<llm_suite_auth::AuthRegistry> {
        let store = Arc::new(llm_suite_auth::MemoryCredentialStore::new());
        let registry = llm_suite_auth::AuthRegistry::new(
            store.clone(),
            Arc::new(llm_suite_auth::StdAuthContext),
        );
        registry.register_provider(
            "acme",
            llm_suite_auth::ProviderAuth::oauth_only(Arc::new(SubscriptionOAuth)),
        );
        store
            .modify(
                "acme",
                Box::new(|_| {
                    Box::pin(async {
                        Ok::<_, llm_suite_auth::AuthError>(Some(llm_suite_auth::Credential::OAuth(
                            llm_suite_auth::OAuthCredential {
                                refresh: "r-1".to_string(),
                                access: "access-token".to_string(),
                                expires: llm_suite_common::time::now() + 60 * 60 * 1000,
                                is_subscription: true,
                                extra: Default::default(),
                            },
                        )))
                    })
                }),
            )
            .await
            .expect("seed");
        Arc::new(registry)
    }

    fn auth_profile(id: &str) -> LlmProfile {
        let mut pending = profile(id, LlmFormat::Custom("probe".to_string()));
        pending.auth = Some(llm_suite_types::llm::ProfileAuthRef {
            provider_id: "acme".to_string(),
            method: None,
        });
        pending
    }

    #[tokio::test]
    async fn auth_reference_injects_stored_credential_and_subscription_flag() {
        let codecs = CodecRegistry::new();
        codecs
            .register("probe", Arc::new(ProbeCodec))
            .expect("probe registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(codecs)
            .with_auth(auth_registry_with_stored_oauth().await);
        gateway.register_profile(auth_profile("p1")).unwrap();

        let prepared = gateway.prepare(&request("p1")).await.expect("prepare");
        assert_eq!(
            prepared.profile.api_key.as_deref(),
            Some("access-token"),
            "the stored OAuth access token must reach the effective profile"
        );
        assert!(
            prepared.is_subscription,
            "subscription-backed access must surface to cost accounting"
        );
    }

    #[tokio::test]
    async fn missing_auth_registry_keeps_static_profile_untouched() {
        let codecs = CodecRegistry::new();
        codecs
            .register("probe", Arc::new(ProbeCodec))
            .expect("probe registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(codecs);
        gateway.register_profile(auth_profile("p1")).unwrap();

        let prepared = gateway.prepare(&request("p1")).await.expect("prepare");
        assert_eq!(prepared.profile.api_key, None);
        assert!(!prepared.is_subscription);
    }

    #[tokio::test]
    async fn unregistered_auth_provider_fails_before_http() {
        let registry = llm_suite_auth::AuthRegistry::new(
            Arc::new(llm_suite_auth::MemoryCredentialStore::new()),
            Arc::new(llm_suite_auth::StdAuthContext),
        );
        let codecs = CodecRegistry::new();
        codecs
            .register("probe", Arc::new(ProbeCodec))
            .expect("probe registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(codecs).with_auth(Arc::new(registry));
        gateway.register_profile(auth_profile("p1")).unwrap();

        let err = match gateway.prepare(&request("p1")).await {
            Ok(_) => panic!("unregistered auth provider must fail before dispatch"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("acme"),
            "the provider id must stay in the message: {err}"
        );
    }

    #[tokio::test]
    async fn evict_client_drops_cached_clients_for_profile() {
        let codecs = CodecRegistry::new();
        codecs
            .register("probe", Arc::new(ProbeCodec))
            .expect("probe registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(codecs);
        gateway
            .register_profile(profile("p1", LlmFormat::Custom("probe".to_string())))
            .unwrap();

        // The probe codec fails, but the client is cached during prepare.
        let _ = gateway.generate(&request("p1"), None).await;
        let stored = gateway.profiles.get("p1").unwrap();
        let key = client_cache_key(&stored);
        assert!(gateway.clients.contains_key(key.as_str()));

        gateway.evict_client("p1");
        assert!(
            !gateway.clients.contains_key(key.as_str()),
            "evict_client must drop every cached client of the profile"
        );
    }

    #[tokio::test]
    async fn evict_clients_for_auth_drops_only_profiles_resolving_that_auth() {
        let codecs = CodecRegistry::new();
        codecs
            .register("probe", Arc::new(ProbeCodec))
            .expect("probe registration must succeed");
        let gateway = LlmGateway::new_with_codec_registry(codecs)
            .with_auth(auth_registry_with_stored_oauth().await);
        gateway.register_profile(auth_profile("p1")).unwrap();
        gateway
            .register_profile(profile("p2", LlmFormat::Custom("probe".to_string())))
            .unwrap();

        // Both probes fail at the codec, but each client is cached during
        // prepare; p1 resolves auth through "acme", p2 carries no auth.
        // Cache keys are computed from the effective (auth-resolved) profile
        // so a rotated credential rebuilds the transport.
        let _ = gateway.generate(&request("p1"), None).await;
        let _ = gateway.generate(&request("p2"), None).await;
        let auth_key = client_cache_key(
            &gateway
                .prepare(&request("p1"))
                .await
                .expect("prepare")
                .profile,
        );
        let plain_key = client_cache_key(&gateway.profiles.get("p2").unwrap());
        assert!(gateway.clients.contains_key(auth_key.as_str()));
        assert!(gateway.clients.contains_key(plain_key.as_str()));

        gateway.evict_clients_for_auth("acme");
        assert!(
            !gateway.clients.contains_key(auth_key.as_str()),
            "profiles resolving auth through the removed surface must be evicted"
        );
        assert!(
            gateway.clients.contains_key(plain_key.as_str()),
            "profiles without an auth reference must keep their cached clients"
        );

        // An unknown auth name is a no-op, not an error.
        gateway.evict_clients_for_auth("missing");
        assert!(gateway.clients.contains_key(plain_key.as_str()));
    }
}
