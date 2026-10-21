//! Gateway orchestration: mandatory profile resolution, request merging,
//! client caching, mock routing and metrics hooks.

use std::sync::Arc;

use dashmap::DashMap;
use llm_suite_client::SharedLlmMetricsSink;
use llm_suite_codec::CodecRegistry;
use llm_suite_config::catalog::ModelCatalog;
use llm_suite_config::profile::ProfileManager;
use llm_suite_config::provider::ProviderDefinitionRegistry;
use llm_token::PricingResolver;

pub(crate) mod client_cache;
pub(crate) mod dispatch;
pub(crate) mod metrics;
pub(crate) mod mock;
pub(crate) mod prepare;
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
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
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

    #[test]
    fn clear_all_drops_the_resilience_state() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("p1", LlmFormat::OpenaiChat))
            .unwrap();
        gateway.get_or_create_client(&gateway.profiles.get("p1").unwrap()).unwrap();
        gateway.shared_breaker("key", llm_suite_client::CircuitBreakerConfig::default());
        gateway.shared_limiter("key", 1.0, 1);

        gateway.clear_all();

        assert!(!gateway.has_profile("p1"));
        assert!(gateway.clients.is_empty());
        assert!(
            gateway.circuit_breakers.is_empty(),
            "a breaker that survived the clear would keep rejecting requests"
        );
        assert!(
            gateway.rate_limiters.is_empty(),
            "a token bucket that survived the clear would keep the old rate"
        );
    }
}
