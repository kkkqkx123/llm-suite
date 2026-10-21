//! Circuit breaking and rate limiting.

use std::sync::Arc;

use llm_suite_codec::error::{LlmError, LlmResult};
use llm_suite_types::llm::LlmProfile;

use super::prepare::PreparedRequest;
use super::LlmGateway;

impl LlmGateway {
    /// Circuit breaker for an arbitrary endpoint key (typically
    /// `base_url::provider_id`), shared with the chat path: embedding and
    /// rerank providers register against the same registry so one upstream
    /// shares one breaker across all three call paths.
    pub fn shared_breaker(
        &self,
        key: &str,
        config: llm_suite_client::CircuitBreakerConfig,
    ) -> Arc<llm_suite_client::CircuitBreaker> {
        self.circuit_breakers
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(llm_suite_client::CircuitBreaker::new(config)))
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
    ) -> Arc<llm_suite_common::ratelimit::RateLimiter> {
        self.rate_limiters
            .entry(key.to_string())
            .or_insert_with(|| {
                Arc::new(llm_suite_common::ratelimit::RateLimiter::new(
                    requests_per_second,
                    burst,
                ))
            })
            .clone()
    }

    /// Rate limiter for the profile's endpoint: explicit profile config
    /// wins, otherwise the referenced provider definition's `rate_limit`.
    /// Keyed by resolved base URL so profiles sharing a provider share one
    /// token bucket.
    pub(super) fn limiter_for(
        &self,
        profile: &LlmProfile,
    ) -> Option<Arc<llm_suite_common::ratelimit::RateLimiter>> {
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
            .and_then(|v| {
                serde_json::from_value::<llm_suite_types::llm::RateLimitConfig>(v.clone()).ok()
            })
            .or_else(|| {
                let provider_id = profile.provider_id.as_ref()?;
                let def = self.providers.get(provider_id)?;
                def.rate_limit
            })?;
        let limiter = Arc::new(llm_suite_common::ratelimit::RateLimiter::new(
            config.requests_per_second,
            config.burst,
        ));
        self.rate_limiters.insert(key, limiter.clone());
        Some(limiter)
    }

    /// Circuit breaker for the profile's endpoint, keyed by resolved base
    /// URL so profiles sharing a provider share one breaker.
    pub(super) fn breaker_for(
        &self,
        profile: &LlmProfile,
    ) -> Option<Arc<llm_suite_client::CircuitBreaker>> {
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
                    Arc::new(llm_suite_client::CircuitBreaker::new(
                        llm_suite_client::CircuitBreakerConfig {
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

    /// Admit the request into the upstream.
    ///
    /// Rate limit after the breaker check: an open breaker rejects without
    /// queueing, a closed one admits into the limiter.
    pub(super) async fn admit(&self, prepared: &PreparedRequest) -> LlmResult<()> {
        if let Some(breaker) = &prepared.breaker {
            if !breaker.check_allowed() {
                return Err(LlmError::CircuitOpen);
            }
        }
        if let Some(limiter) = &prepared.limiter {
            limiter.acquire().await;
        }
        Ok(())
    }

    /// Count a network-classified failure (or success) into the breaker
    /// window. Semantic 4xx errors do not trip the breaker.
    pub(super) fn record_breaker(
        breaker: Option<&Arc<llm_suite_client::CircuitBreaker>>,
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
}
