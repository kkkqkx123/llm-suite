//! Assembly of a request into the pieces the dispatch paths need.

use std::sync::Arc;

use llm_suite_auth::AuthError;
use llm_suite_client::client::LlmClientImpl;
use llm_suite_codec::error::{LlmError, LlmResult};
use llm_suite_config::merge_request;
use llm_suite_types::llm::{LlmProfile, LlmRequest};

use super::LlmGateway;

/// Assembled request ready for dispatch: the resolved profile snapshot,
/// the merged effective request and the cached client bound to them.
/// Produced only by `LlmGateway::prepare` so every entry point shares
/// a single assembly sequence.
pub(super) struct PreparedRequest {
    pub profile: LlmProfile,
    pub effective: LlmRequest,
    pub client: Arc<LlmClientImpl>,
    pub breaker: Option<Arc<llm_suite_client::CircuitBreaker>>,
    pub limiter: Option<Arc<llm_suite_common::ratelimit::RateLimiter>>,
    /// True when the resolved auth is subscription-backed; suppresses
    /// per-token cost accounting for this request.
    pub is_subscription: bool,
}

/// Apply the optional per-request wall-clock bound (milliseconds) to a
/// dispatch future. A request-level timeout covers the full client call
/// including the transport retry chain, so callers can guarantee a node
/// budget regardless of profile-level timeout/retry settings.
pub(super) async fn bound<T>(
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

impl LlmGateway {
    /// Single assembly preamble shared by all request entry points:
    /// resolve the profile, merge request overrides, then fetch the client.
    /// Mock routing and result post-processing stay in each caller.
    pub(super) async fn prepare(&self, request: &LlmRequest) -> LlmResult<PreparedRequest> {
        let stored_profile = self.resolve_profile(&request.profile_id)?;
        // Resolve auth first: the effective profile carries the injected
        // key/headers/base_url, and the client cache key is computed from
        // it so a rotated credential rebuilds the transport.
        let (profile, is_subscription) = match self.auth.as_ref() {
            Some(registry) => match registry.apply(&stored_profile).await.map_err(auth_error)? {
                Some(resolved) => (resolved.profile, resolved.is_subscription),
                None => (stored_profile, false),
            },
            None => (stored_profile, false),
        };
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
            is_subscription,
        })
    }

    pub(super) fn resolve_profile(&self, profile_id: &str) -> LlmResult<LlmProfile> {
        if profile_id.is_empty() {
            return self.profiles.get_default().ok_or_else(|| {
                LlmError::ProfileNotFound("(default profile not registered)".to_string())
            });
        }
        self.profiles
            .get(profile_id)
            .ok_or_else(|| LlmError::ProfileNotFound(profile_id.to_string()))
    }
}

/// Map an auth-layer failure onto the gateway error surface. The gateway
/// stays provider-agnostic; the message keeps the provider id so hosts can
/// point the user at `login`.
fn auth_error(error: AuthError) -> LlmError {
    LlmError::ConfigError(error.to_string())
}
