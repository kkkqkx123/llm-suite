//! Proxy credential resolution for gateway request assembly.
//!
//! Persisted profiles carry a route-only proxy address. The password lives
//! in the credential store under a key derived from that route, so one
//! enterprise egress shares one credential across every profile pointing
//! at it. Resolution composes the transient in-memory address just before
//! transport construction and never writes it back to the registry.

use llm_suite_auth::{AuthRegistry, ProxyCredential};
use llm_suite_codec::error::{LlmError, LlmResult};
use llm_suite_types::llm::LlmProfile;

/// Resolve the proxy credential for a route address, when one is stored.
///
/// Returns `None` for anonymous routes. Fails only on storage failure, not
/// on a missing entry.
pub async fn resolve_proxy_credential(
    registry: &AuthRegistry,
    route: &str,
) -> LlmResult<Option<ProxyCredential>> {
    let key = llm_suite_proxy::credential_key(route);
    let stored = registry
        .store()
        .read(&key)
        .await
        .map_err(|err| LlmError::ConfigError(err.to_string()))?;
    let Some(credential) = stored else {
        return Ok(None);
    };
    match credential.as_proxy() {
        Ok(proxy) => Ok(Some(proxy.clone())),
        Err(_) => Err(LlmError::ConfigError(
            "proxy credential for route is not a proxy credential".to_string(),
        )),
    }
}

/// Compose the transient effective proxy address for a profile.
///
/// Route-only addresses stay as-is when no credential is stored. The result
/// lives only in the effective request profile, never in the registry.
pub async fn effective_proxy_url(
    registry: Option<&AuthRegistry>,
    profile: &LlmProfile,
) -> LlmResult<Option<String>> {
    let Some(route) = profile.proxy.as_deref() else {
        return Ok(None);
    };
    let Some(registry) = registry else {
        return Ok(Some(route.to_string()));
    };
    let Some(credential) = resolve_proxy_credential(registry, route).await? else {
        return Ok(Some(route.to_string()));
    };
    llm_suite_proxy::compose_proxy_url(route, &credential.username, &credential.password)
        .map(Some)
        .map_err(|err| LlmError::ProxyError(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_auth::CredentialStore;
    use std::sync::Arc;

    fn route_profile(route: &str) -> LlmProfile {
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
            max_output_tokens: None,
            proxy: Some(route.to_string()),
            no_proxy: None,
            circuit_breaker: None,
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    #[tokio::test]
    async fn anonymous_route_passes_through() {
        let store = Arc::new(llm_suite_auth::MemoryCredentialStore::new());
        let registry = AuthRegistry::new(store, Arc::new(llm_suite_auth::StdAuthContext));
        let profile = route_profile("http://127.0.0.1:8080");
        let url = effective_proxy_url(Some(&registry), &profile)
            .await
            .expect("resolve");
        assert_eq!(url.as_deref(), Some("http://127.0.0.1:8080"));
    }

    #[tokio::test]
    async fn stored_credential_composes_transient_address() {
        let store = Arc::new(llm_suite_auth::MemoryCredentialStore::new());
        let registry = AuthRegistry::new(store.clone(), Arc::new(llm_suite_auth::StdAuthContext));
        let route = "http://127.0.0.1:8080";
        let key = llm_suite_proxy::credential_key(route);
        store
            .modify(
                &key,
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(llm_suite_auth::Credential::Proxy(
                            llm_suite_auth::ProxyCredential {
                                username: "corp-user".to_string(),
                                password: "s3cret".to_string(),
                            },
                        )))
                    })
                }),
            )
            .await
            .expect("seed");
        let profile = route_profile(route);
        let url = effective_proxy_url(Some(&registry), &profile)
            .await
            .expect("resolve");
        assert_eq!(
            url.as_deref(),
            Some("http://corp-user:s3cret@127.0.0.1:8080")
        );
    }
}
