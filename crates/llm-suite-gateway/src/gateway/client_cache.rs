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
/// client.
pub fn client_cache_key(profile: &LlmProfile) -> String {
    format!(
        "{}{}::{}::{}",
        client_cache_prefix(&profile.id),
        profile.model,
        profile.proxy.as_deref().unwrap_or(""),
        profile.no_proxy.as_deref().unwrap_or(&[]).join(",")
    )
}

impl LlmGateway {
    pub(super) fn get_or_create_client(&self, profile: &LlmProfile) -> LlmResult<Arc<LlmClientImpl>> {
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
        let client = llm_suite_proxy::build_http_client(
            profile.timeout.unwrap_or(60),
            profile.proxy.as_deref(),
            &no_proxy,
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
        let impl_client = LlmClientImpl::new(client, codec, profile.clone());
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
}
