use std::sync::Arc;

use dashmap::DashMap;
use llm_client::SharedLlmMetricsSink;
use llm_types::llm::{LlmProfile, LlmRequest, LlmResult as LlmResponseType};

use llm_client::client::{LlmClient, LlmClientImpl};
use llm_client::MessageStream;
use llm_codec::error::{LlmError, LlmResult};
use llm_codec::CodecRegistry;
use llm_config::catalog::ModelCatalog;
use llm_config::merge_request;
use llm_config::profile::ProfileManager;
use llm_config::provider::{apply_provider_defaults, ProviderDefinitionRegistry};

/// Assembled request ready for dispatch: the resolved profile snapshot,
/// the merged effective request and the cached client bound to them.
/// Produced only by `LlmGateway::prepare` so every entry point shares
/// a single assembly sequence.
struct PreparedRequest {
    profile: LlmProfile,
    effective: LlmRequest,
    client: Arc<LlmClientImpl>,
    breaker: Option<Arc<llm_client::CircuitBreaker>>,
    limiter: Option<Arc<llm_common::ratelimit::RateLimiter>>,
}

/// Apply the optional per-request wall-clock bound (milliseconds) to a
/// dispatch future. A request-level timeout covers the full client call
/// including the transport retry chain, so callers can guarantee a node
/// budget regardless of profile-level timeout/retry settings.
async fn bound<T>(
    timeout_ms: Option<u64>,
    fut: impl std::future::Future<Output = LlmResult<T>>,
) -> LlmResult<T> {
    match timeout_ms {
        Some(ms) if ms > 0 => tokio::time::timeout(std::time::Duration::from_millis(ms), fut)
            .await
            .map_err(|_| LlmError::Timeout(ms))?,
        _ => fut.await,
    }
}

/// Single facade for all LLM calls.
///
/// Responsibilities:
/// - resolve the profile for a mandatory `profile_id` (no fallback branch)
/// - route to mock clients (test injection) or real clients
/// - resolve codecs through the registry (built-ins + runtime custom)
/// - record token usage metrics for both generate and stream paths
///
/// Request merging delegates to `llm_config::merge_request`; metrics wrapping
/// delegates to `token::stream`.
#[derive(Clone)]
pub struct LlmGateway {
    clients: Arc<DashMap<String, Arc<LlmClientImpl>>>,
    profiles: ProfileManager,
    codecs: CodecRegistry,
    providers: ProviderDefinitionRegistry,
    model_catalog: ModelCatalog,
    /// Circuit breakers keyed by the resolved endpoint base URL, shared by
    /// all profiles pointing at the same provider.
    circuit_breakers: Arc<DashMap<String, Arc<llm_client::CircuitBreaker>>>,
    /// Rate limiters keyed by the resolved endpoint base URL, shared by
    /// all profiles pointing at the same provider.
    rate_limiters: Arc<DashMap<String, Arc<llm_common::ratelimit::RateLimiter>>>,
    #[cfg(feature = "mock")]
    mock_clients: Arc<DashMap<String, Arc<llm_client::MockLlmClient>>>,
    token_metrics: Option<SharedLlmMetricsSink>,
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
        }
    }

    /// Attach an optional token usage collector (zero overhead when absent).
    pub fn with_token_metrics(mut self, token_metrics: SharedLlmMetricsSink) -> Self {
        self.token_metrics = Some(token_metrics);
        self
    }

    pub fn register_profile(&self, profile: LlmProfile) -> LlmResult<()> {
        let effective = apply_provider_defaults(profile, &self.providers)?;
        let profile_id = effective.id.clone();
        self.profiles.register(effective)?;
        let prefix = format!("{profile_id}::");
        self.clients.retain(|key, _| !key.starts_with(&prefix));
        Ok(())
    }

    /// Register a provider definition. Only cached clients for profiles
    /// referencing the definition are evicted, so unrelated profiles keep
    /// their clients.
    pub fn register_provider_definition(
        &self,
        definition: llm_types::llm::LlmProviderDefinition,
    ) -> LlmResult<()> {
        let provider_id = definition.id.clone();
        self.providers.register(definition)?;
        self.evict_clients_for_provider(&provider_id);
        Ok(())
    }

    /// Remove a provider definition and evict cached clients for profiles
    /// referencing it. Those profiles keep their merged snapshot.
    pub fn remove_provider_definition(
        &self,
        id: &str,
    ) -> Option<llm_types::llm::LlmProviderDefinition> {
        let removed = self.providers.remove(id);
        if removed.is_some() {
            self.evict_clients_for_provider(id);
        }
        removed
    }

    /// Evict cached clients whose profile references the given provider id.
    fn evict_clients_for_provider(&self, provider_id: &str) {
        let affected: Vec<String> = self
            .profiles
            .list()
            .into_iter()
            .filter(|profile| profile.provider_id.as_deref() == Some(provider_id))
            .map(|profile| client_cache_key(&profile))
            .collect();
        for key in affected {
            self.clients.remove(&key);
        }
    }

    /// Circuit breaker for an arbitrary endpoint key (typically
    /// `base_url::provider_id`), shared with the chat path: embedding and
    /// rerank providers register against the same registry so one upstream
    /// shares one breaker across all three call paths.
    pub fn shared_breaker(
        &self,
        key: &str,
        config: llm_client::CircuitBreakerConfig,
    ) -> Arc<llm_client::CircuitBreaker> {
        self.circuit_breakers
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(llm_client::CircuitBreaker::new(config)))
            .clone()
    }

    /// Rate limiter for an arbitrary endpoint key (typically
    /// `base_url::provider_id`), shared with the chat path so the combined
    /// request rate across chat/embedding/rerank stays under the upstream
    /// budget.
    pub fn shared_limiter(
        &self,
        key: &str,
        requests_per_second: f64,
        burst: u32,
    ) -> Arc<llm_common::ratelimit::RateLimiter> {
        self.rate_limiters
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(llm_common::ratelimit::RateLimiter::new(requests_per_second, burst)))
            .clone()
    }

    /// Register a mock client under an arbitrary id (a real profile id can be
    /// reused, or a plain "mock" id). Mock hits take priority over profiles.
    #[cfg(feature = "mock")]
    pub fn register_mock(&self, id: impl Into<String>, client: Arc<llm_client::MockLlmClient>) {
        self.mock_clients.insert(id.into(), client);
    }

    #[cfg(feature = "mock")]
    fn mock_client(&self, id: &str) -> Option<Arc<llm_client::MockLlmClient>> {
        self.mock_clients.get(id).map(|c| c.clone())
    }

    /// Profile registry for assembly-time validation of profile references.
    pub fn profile_registry(&self) -> &ProfileManager {
        &self.profiles
    }

    /// The codec registry: register custom formats here before the
    /// first request that uses them.
    pub fn codec_registry(&self) -> &CodecRegistry {
        &self.codecs
    }

    /// The provider definition registry backing `LlmProfile::provider_id`.
    pub fn provider_registry(&self) -> &ProviderDefinitionRegistry {
        &self.providers
    }

    /// Off-hot-path model listing over a provider definition.
    pub fn model_catalog(&self) -> &ModelCatalog {
        &self.model_catalog
    }

    pub fn has_profile(&self, id: &str) -> bool {
        self.profiles.has(id)
    }

    /// Remove a profile and evict all cached clients bound to it. Returns the
    /// removed profile.
    pub fn remove_profile(&self, id: &str) -> Option<LlmProfile> {
        let removed = self.profiles.remove(id);
        if removed.is_some() {
            let prefix = format!("{id}::");
            self.clients.retain(|key, _| !key.starts_with(&prefix));
        }
        removed
    }

    /// Clear all profiles, provider definitions and the client cache.
    pub fn clear_all(&self) {
        self.profiles.clear();
        self.providers.clear();
        self.clients.clear();
    }

    pub async fn generate(
        &self,
        request: &LlmRequest,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> LlmResult<LlmResponseType> {
        #[cfg(feature = "mock")]
        if let Some(client) = self.mock_client(&request.profile_id) {
            return client.generate(request, cancel).await;
        }

        let prepared = self.prepare(request)?;
        // Rate limit after the breaker check: an open breaker rejects
        // without queueing, a closed one admits into the limiter.
        if let Some(breaker) = &prepared.breaker {
            if !breaker.check_allowed() {
                return Err(LlmError::CircuitOpen);
            }
        }
        if let Some(limiter) = &prepared.limiter {
            limiter.acquire().await;
        }
        let start = std::time::Instant::now();
        let timeout_ms = prepared.effective.timeout_ms;
        let result = bound(timeout_ms, async {
            prepared.client.generate(&prepared.effective, cancel).await
        })
        .await;
        Self::record_breaker(prepared.breaker.as_ref(), &result);
        let duration_ms = start.elapsed().as_millis() as f64;
        match &result {
            Ok(response) => {
                self.record_token_usage(response, &prepared.profile);
                self.record_request(duration_ms, true, None, &prepared.profile);
            }
            Err(error) => {
                self.record_request(duration_ms, false, Some(error), &prepared.profile);
            }
        }
        result
    }

    pub async fn generate_stream(
        &self,
        request: &LlmRequest,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> LlmResult<Box<dyn MessageStream>> {
        #[cfg(feature = "mock")]
        if let Some(client) = self.mock_client(&request.profile_id) {
            return client.generate_stream(request, cancel).await;
        }

        let prepared = self.prepare(request)?;
        if let Some(breaker) = &prepared.breaker {
            if !breaker.check_allowed() {
                return Err(LlmError::CircuitOpen);
            }
        }
        if let Some(limiter) = &prepared.limiter {
            limiter.acquire().await;
        }
        let start = std::time::Instant::now();
        let timeout_ms = prepared.effective.timeout_ms;
        // The per-request bound covers stream establishment only (including
        // transport retries); consuming the returned stream is bounded by
        // the client-level and caller-level budgets, not here.
        let stream = bound(timeout_ms, async {
            prepared
                .client
                .generate_stream(&prepared.effective, cancel)
                .await
        })
        .await;
        Self::record_breaker(prepared.breaker.as_ref(), &stream);
        let duration_ms = start.elapsed().as_millis() as f64;
        match &stream {
            Ok(_) => {
                self.record_request(duration_ms, true, None, &prepared.profile);
                if let Some(metrics) = self.token_metrics.as_ref() {
                    metrics.record_first_byte(duration_ms, Some(&prepared.profile.model));
                }
            }
            Err(error) => self.record_request(duration_ms, false, Some(error), &prepared.profile),
        }
        let stream = stream?;
        let usage_sink = self
            .token_metrics
            .as_ref()
            .map(|m| Arc::clone(m) as llm_client::SharedTokenUsageSink);
        Ok(Box::new(llm_client::TokenRecordingStream::new(
            stream,
            usage_sink,
            prepared.profile.model.clone(),
        )))
    }

    pub async fn count_tokens(
        &self,
        request: &LlmRequest,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> LlmResult<llm_types::llm::TokenCountResult> {
        #[cfg(feature = "mock")]
        if let Some(client) = self.mock_client(&request.profile_id) {
            return client.count_tokens(request, cancel).await;
        }

        let prepared = self.prepare(request)?;
        if let Some(breaker) = &prepared.breaker {
            if !breaker.check_allowed() {
                return Err(LlmError::CircuitOpen);
            }
        }
        if let Some(limiter) = &prepared.limiter {
            limiter.acquire().await;
        }
        prepared
            .client
            .count_tokens(&prepared.effective, cancel)
            .await
    }

    /// Single assembly preamble shared by all request entry points:
    /// resolve the profile, merge request overrides, then fetch the client.
    /// Mock routing and result post-processing stay in each caller.
    fn prepare(&self, request: &LlmRequest) -> LlmResult<PreparedRequest> {
        let profile = self.resolve_profile(&request.profile_id)?;
        let effective = merge_request(request, &profile)?;
        let client = self.get_or_create_client(&profile)?;
        let breaker = self.breaker_for(&profile);
        let limiter = self.limiter_for(&profile);
        Ok(PreparedRequest {
            profile,
            effective,
            client,
            breaker,
            limiter,
        })
    }

    /// Rate limiter for the profile's endpoint: explicit profile config
    /// wins, otherwise the referenced provider definition's `rate_limit`.
    /// Keyed by resolved base URL so profiles sharing a provider share one
    /// token bucket.
    fn limiter_for(&self, profile: &LlmProfile) -> Option<Arc<llm_common::ratelimit::RateLimiter>> {
        let base_url = profile.base_url.clone().unwrap_or_default();
        let key = format!(
            "{}::{}",
            base_url,
            profile.provider_id.clone().unwrap_or_default()
        );
        if let Some(entry) = self.rate_limiters.get(&key) {
            return Some(entry.clone());
        }
        let config = profile
            .metadata
            .as_ref()
            .and_then(|m| m.get("rate_limit"))
            .and_then(|v| serde_json::from_value::<llm_types::llm::RateLimitConfig>(v.clone()).ok())
            .or_else(|| {
                let provider_id = profile.provider_id.as_ref()?;
                let def = self.providers.get(provider_id)?;
                def.rate_limit
            })?;
        let limiter = Arc::new(llm_common::ratelimit::RateLimiter::new(
            config.requests_per_second,
            config.burst,
        ));
        self.rate_limiters.insert(key, limiter.clone());
        Some(limiter)
    }

    /// Circuit breaker for the profile's endpoint, keyed by resolved base
    /// URL so profiles sharing a provider share one breaker.
    fn breaker_for(&self, profile: &LlmProfile) -> Option<Arc<llm_client::CircuitBreaker>> {
        let config = profile.circuit_breaker.clone()?;
        let base_url = profile.base_url.clone().unwrap_or_default();
        let key = format!(
            "{}::{}",
            base_url,
            profile.provider_id.clone().unwrap_or_default()
        );
        Some(
            self.circuit_breakers
                .entry(key)
                .or_insert_with(|| {
                    Arc::new(llm_client::CircuitBreaker::new(
                        llm_client::CircuitBreakerConfig {
                            min_samples: config.min_samples,
                            failure_threshold: config.failure_threshold,
                            open_duration_ms: config.open_duration_ms,
                            half_open_max_probes: config.half_open_max_probes,
                        },
                    ))
                })
                .clone(),
        )
    }

    /// Count a network-classified failure (or success) into the breaker
    /// window. Semantic 4xx errors do not trip the breaker.
    fn record_breaker(
        breaker: Option<&Arc<llm_client::CircuitBreaker>>,
        result: &LlmResult<impl Sized>,
    ) {
        let Some(breaker) = breaker else {
            return;
        };
        match result {
            Ok(_) => breaker.record_success(),
            Err(error) => {
                let network_level = match error {
                    LlmError::HttpError(_) | LlmError::Timeout(_) | LlmError::StreamError(_) => {
                        true
                    }
                    LlmError::ProviderError { status, .. } => {
                        matches!(status, Some(500..=599))
                    }
                    _ => false,
                };
                if network_level {
                    breaker.record_failure();
                }
            }
        }
    }

    fn resolve_profile(&self, profile_id: &str) -> LlmResult<LlmProfile> {
        if profile_id.is_empty() {
            return self.profiles.get_default().ok_or_else(|| {
                LlmError::ProfileNotFound("(default profile not registered)".to_string())
            });
        }
        self.profiles
            .get(profile_id)
            .ok_or_else(|| LlmError::ProfileNotFound(profile_id.to_string()))
    }

    fn get_or_create_client(&self, profile: &LlmProfile) -> LlmResult<Arc<LlmClientImpl>> {
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
        let client = llm_proxy::build_http_client(
            profile.timeout.unwrap_or(60),
            profile.proxy.as_deref(),
            &no_proxy,
        )
        .map_err(|err| LlmError::ProxyError(err.to_string()))?;

        let client_impl = Arc::new(LlmClientImpl::new(client, codec, profile.clone()));
        self.clients.insert(key, client_impl.clone());
        Ok(client_impl)
    }

    fn record_token_usage(&self, result: &LlmResponseType, profile: &LlmProfile) {
        let Some(metrics) = self.token_metrics.as_ref() else {
            return;
        };
        let Some(usage) = result.usage.as_ref() else {
            return;
        };
        metrics.record_token_usage(
            usage.prompt_tokens as u64,
            usage.completion_tokens as u64,
            usage.total_cost,
            Some(&profile.model),
        );
    }

    fn record_request(
        &self,
        duration_ms: f64,
        success: bool,
        error: Option<&llm_codec::error::LlmError>,
        profile: &LlmProfile,
    ) {
        let Some(metrics) = self.token_metrics.as_ref() else {
            return;
        };
        metrics.record_request(
            duration_ms,
            success,
            error.map(classify_error),
            Some(&profile.model),
        );
        if !success && profile.max_retries.unwrap_or(0) > 0 {
            metrics.record_retry(Some(&profile.model));
        }
    }
}

/// Cache key for the transport client. The proxy is part of the key so
/// switching proxies rebuilds the transport instead of reusing a stale
/// client.
fn client_cache_key(profile: &LlmProfile) -> String {
    format!(
        "{}::{}::{}::{}",
        profile.id,
        profile.model,
        profile.proxy.as_deref().unwrap_or(""),
        profile.no_proxy.as_deref().unwrap_or(&[]).join(",")
    )
}

/// Low-cardinality error classifier for LLM request metrics.
fn classify_error(error: &llm_codec::error::LlmError) -> &'static str {
    match error {
        llm_codec::error::LlmError::HttpError(_) => "http_error",
        llm_codec::error::LlmError::SerializationError(_) => "serialization_error",
        llm_codec::error::LlmError::ProviderError { .. } => "provider_error",
        llm_codec::error::LlmError::ContextLengthExceeded(_) => "context_length_exceeded",
        llm_codec::error::LlmError::ConfigError(_) => "config_error",
        llm_codec::error::LlmError::StreamError(_) => "stream_error",
        llm_codec::error::LlmError::ProfileNotFound(_) => "profile_not_found",
        llm_codec::error::LlmError::UnsupportedFormat(_) => "unsupported_format",
        llm_codec::error::LlmError::CodecNotFound(_) => "codec_not_found",
        llm_codec::error::LlmError::Timeout(_) => "timeout",
        llm_codec::error::LlmError::AuthError(_) => "auth_error",
        llm_codec::error::LlmError::ToolNotFound(_) => "tool_not_found",
        llm_codec::error::LlmError::InvalidResponse(_) => "invalid_response",
        llm_codec::error::LlmError::Cancelled => "cancelled",
        llm_codec::error::LlmError::RateLimited { .. } => "rate_limited",
        llm_codec::error::LlmError::CircuitOpen => "circuit_open",
        llm_codec::error::LlmError::ProxyError(_) => "proxy_error",
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
    use llm_codec::error::LlmError;
    use llm_codec::CodecRegistry;
    use llm_codec::LlmCodec;
    use llm_types::llm::{LlmFormat, LlmProfile, LlmRequest, MessageStreamEvent};
    use llm_types::tool::Tool;

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

        fn parse_response(&self, _body: &str, _request: &LlmRequest) -> LlmResult<LlmResponseType> {
            Err(LlmError::ConfigError("custom codec engaged".to_string()))
        }

        fn parse_stream_chunk(&self, _data: &str) -> LlmResult<Option<MessageStreamEvent>> {
            Ok(None)
        }

        fn convert_tools(&self, _tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
            Ok(Vec::new())
        }

        fn parse_tool_calls(
            &self,
            _result: &LlmResponseType,
        ) -> Vec<llm_types::message::LlmToolCall> {
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
        let key = super::client_cache_key(client.profile());
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
        let key = super::client_cache_key(client.profile());

        gateway.remove_profile("p1");
        assert!(
            gateway.clients.contains_key(key.as_str()),
            "unrelated clients must survive"
        );
    }

    fn provider_definition() -> llm_types::llm::LlmProviderDefinition {
        llm_types::llm::LlmProviderDefinition {
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
        let key = super::client_cache_key(client.profile());
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
        let direct_key = super::client_cache_key(&stored);
        assert!(gateway.clients.contains_key(direct_key.as_str()));

        let mut proxied = stored.clone();
        proxied.proxy = Some("http://127.0.0.1:8080".to_string());
        gateway.get_or_create_client(&proxied).unwrap();
        let proxied_key = super::client_cache_key(&proxied);
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
        let single_key = super::client_cache_key(&bypassed);
        gateway.get_or_create_client(&bypassed).unwrap();

        bypassed.no_proxy = Some(vec!["localhost".to_string(), "10.0.0.0/8".to_string()]);
        let wider_key = super::client_cache_key(&bypassed);
        assert_ne!(
            single_key, wider_key,
            "a different bypass list must rebuild the transport"
        );
    }
}
