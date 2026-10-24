//! The profile and provider definition registries.

use llm_suite_codec::error::LlmResult;
use llm_suite_config::catalog::ModelCatalog;
use llm_suite_config::profile::ProfileManager;
use llm_suite_config::provider::{apply_provider_defaults, ProviderDefinitionRegistry};
use llm_suite_auth::AuthRegistry;
use llm_suite_types::llm::LlmProfile;
use std::sync::Arc;

use super::LlmGateway;

impl LlmGateway {
    pub fn register_profile(&self, profile: LlmProfile) -> LlmResult<()> {
        let effective = apply_provider_defaults(profile, &self.providers)?;
        let profile_id = effective.id.clone();
        self.profiles.register(effective)?;
        self.evict_clients_for_profile(&profile_id);
        Ok(())
    }

    /// Register a provider definition. Only cached clients for profiles
    /// referencing the definition are evicted, so unrelated profiles keep
    /// their clients.
    pub fn register_provider_definition(
        &self,
        definition: llm_suite_types::llm::LlmProviderDefinition,
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
    ) -> Option<llm_suite_types::llm::LlmProviderDefinition> {
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
            .map(|profile| profile.id.clone())
            .collect();
        for profile_id in affected {
            self.evict_clients_for_profile(&profile_id);
        }
    }

    /// Profile registry for assembly-time validation of profile references.
    pub fn profile_registry(&self) -> &ProfileManager {
        &self.profiles
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
            self.evict_clients_for_profile(id);
        }
        removed
    }

    /// Attach an optional auth registry. Profiles carrying an auth reference
    /// then resolve their credential (stored key, OAuth token, ambient
    /// environment) through the registry on every request; OAuth refresh
    /// happens inside the registry's serialized write path. Callers must
    /// evict cached clients after login/logout so a rotated credential
    /// reaches the transport (see `evict_clients_for_auth`).
    pub fn with_auth(mut self, registry: Arc<AuthRegistry>) -> Self {
        self.auth = Some(registry);
        self
    }

    /// The attached auth registry, when one is wired.
    pub fn auth_registry(&self) -> Option<&Arc<AuthRegistry>> {
        self.auth.as_ref()
    }

    /// Evict cached clients whose profile resolves auth through the given
    /// auth name (the credential owner). Call after a login, logout or auth
    /// surface removal so the next request re-resolves the credential.
    pub fn evict_clients_for_auth(&self, auth_name: &str) {
        let affected: Vec<String> = self
            .profiles
            .list()
            .into_iter()
            .filter(|profile| {
                profile
                    .auth
                    .as_ref()
                    .map(|auth| auth.provider_id == auth_name)
                    .unwrap_or(false)
            })
            .map(|profile| profile.id.clone())
            .collect();
        for profile_id in affected {
            self.evict_clients_for_profile(&profile_id);
        }
    }

    /// Clear every piece of state the gateway holds: the profiles, the
    /// provider definitions, the cached clients, the circuit breakers and
    /// the rate limiters. A breaker that survived the clear would keep
    /// rejecting requests for a profile that no longer exists.
    pub fn clear_all(&self) {
        self.profiles.clear();
        self.providers.clear();
        self.clients.clear();
        self.circuit_breakers.clear();
        self.rate_limiters.clear();
        #[cfg(feature = "mock")]
        self.mock_clients.clear();
    }
}
