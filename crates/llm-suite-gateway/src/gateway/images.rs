//! Image generation through the gateway.
//!
//! The chat paths (`generate` / `generate_stream`) resolve a codec and a
//! cached transport client. Image generation has no codec: every
//! OpenAI-compatible images endpoint speaks the same
//! `POST {base}/images/generations` contract, so the gateway builds a
//! short-lived [`OpenAiCompatibleImagesProvider`] from the resolved profile
//! on each call instead of caching a client. Profiles resolve exactly as
//! chat does (provider defaults merged at registration, auth applied per
//! call), so a rotated credential reaches the next images call without an
//! explicit eviction.

use llm_suite_codec::error::{LlmError, LlmResult};
use llm_suite_images::ImageGenerationProvider;
use llm_suite_types::llm::LlmProfile;

use super::LlmGateway;

impl LlmGateway {
    /// Generate images with the model bound to `profile_id`.
    ///
    /// Resolution mirrors [`LlmGateway::prepare`]: the stored profile is
    /// loaded, the attached auth registry (when any) injects the effective
    /// key, and the endpoint plus transport knobs come from the effective
    /// profile. The provider transport is constructed per call; there is
    /// no cached images client to evict.
    pub async fn generate_images(
        &self,
        profile_id: &str,
        prompt: &str,
        options: &llm_suite_images::ImagesOptions,
    ) -> LlmResult<llm_suite_images::ImagesResult> {
        if prompt.is_empty() {
            return Err(LlmError::ConfigError(
                "image prompt must not be empty".to_string(),
            ));
        }
        let stored = self.resolve_profile(profile_id)?;
        // Mirror `prepare`: a per-profile env overlay shadows the registry
        // context for this resolution only.
        let (profile, _) = match self.auth.as_ref() {
            Some(registry) => {
                let overlay: Option<std::sync::Arc<dyn llm_suite_auth::AuthContext>> =
                    stored.env.as_ref().filter(|map| !map.is_empty()).map(|map| {
                        std::sync::Arc::new(llm_suite_auth::OverlayAuthContext::new(
                            registry.context().clone(),
                            map.clone(),
                        ))
                            as std::sync::Arc<dyn llm_suite_auth::AuthContext>
                    });
                let resolved = match overlay.as_ref() {
                    Some(context) => {
                        registry.apply_with_context(&stored, context).await.map_err(|e| {
                            LlmError::ConfigError(e.to_string())
                        })?
                    }
                    None => registry
                        .apply(&stored)
                        .await
                        .map_err(|e| LlmError::ConfigError(e.to_string()))?,
                };
                match resolved {
                    Some(resolved) => (resolved.profile, resolved.is_subscription),
                    None => (stored, false),
                }
            }
            None => (stored, false),
        };
        let context = images_context_for_profile(&profile)?;
        tracing::debug!(
            profile_id = profile.id.as_str(),
            model = profile.model.as_str(),
            endpoint = context.base_url.as_str(),
            "llm images request dispatched"
        );
        let provider =
            llm_suite_images::OpenAiCompatibleImagesProvider::new(context).map_err(images_error)?;
        provider
            .generate_images(prompt, options)
            .await
            .map_err(images_error)
    }
}

/// Build the images endpoint context from an effective (auth-resolved) profile.
///
/// The profile `base_url` names the API root (e.g. `https://api.openai.com/v1`);
/// the images path is appended unless the URL already carries it. Timeout,
/// proxy, bypass list, extra headers, query params, client identity and
/// connection-level transport knobs travel verbatim so an images call
/// observes the same transport policy as chat. Explicit `custom_headers` win
/// over identity defaults per header name.
pub fn images_context_for_profile(
    profile: &LlmProfile,
) -> LlmResult<llm_suite_images::ImagesContext> {
    let base_url = profile.base_url.clone().ok_or_else(|| {
        LlmError::ConfigError(format!(
            "profile '{}' has no base_url for image generation",
            profile.id
        ))
    })?;
    let mut context =
        llm_suite_images::ImagesContext::new(join_images_url(&base_url), profile.model.clone());
    if let Some(api_key) = profile.api_key.clone() {
        context = context.with_api_key(api_key);
    }
    context.timeout_secs = profile.timeout.unwrap_or(60);
    if let Some(proxy) = profile.proxy.clone() {
        context = context.with_proxy(proxy);
    }
    if let Some(no_proxy) = profile.no_proxy.clone() {
        context.no_proxy = no_proxy;
    }
    if let Some(transport) = profile.transport.as_ref() {
        context.connect_timeout_secs = transport.connect_timeout_secs;
        context.danger_accept_invalid_certs = transport.danger_accept_invalid_certs;
    }
    // Identity defaults first so explicit headers win per name below.
    if let Some(identity) = profile.identity.as_ref() {
        if let Some(user_agent) = identity.user_agent.as_deref() {
            context
                .headers
                .entry("User-Agent".to_string())
                .or_insert_with(|| user_agent.to_string());
        }
        if let Some(app_url) = identity.app_url.as_deref() {
            context
                .headers
                .entry("HTTP-Referer".to_string())
                .or_insert_with(|| app_url.to_string());
        }
        if let Some(app_name) = identity.app_name.as_deref() {
            context
                .headers
                .entry("X-Title".to_string())
                .or_insert_with(|| app_name.to_string());
        }
    }
    for (name, value) in metadata_headers(profile.headers.as_ref()) {
        context.headers.insert(name, value);
    }
    for (name, value) in metadata_headers(profile.custom_headers.as_ref()) {
        context.headers.insert(name, value);
    }
    for (name, value) in metadata_headers(profile.query_params.as_ref()) {
        context.query_params.insert(name, value);
    }
    context.validate().map_err(images_error)?;
    Ok(context)
}

/// Join `{base_url}/images/generations` unless the URL already carries the
/// images path. Trailing slashes are tolerated.
pub fn join_images_url(base_url: &str) -> String {
    const SUFFIX: &str = "images/generations";
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with(SUFFIX) {
        trimmed.to_string()
    } else {
        format!("{trimmed}/{SUFFIX}")
    }
}

fn metadata_headers(
    metadata: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Vec<(String, String)> {
    let Some(metadata) = metadata else {
        return Vec::new();
    };
    metadata
        .iter()
        .map(|(name, value)| {
            let rendered = match value {
                serde_json::Value::String(text) => text.clone(),
                _ => value.to_string(),
            };
            (name.clone(), rendered)
        })
        .collect()
}

fn images_error(error: llm_suite_common::ProviderError) -> LlmError {
    match error {
        llm_suite_common::ProviderError::Config(message)
        | llm_suite_common::ProviderError::InvalidRequest(message) => {
            LlmError::ConfigError(message)
        }
        llm_suite_common::ProviderError::Provider {
            status,
            message,
            retry_after_ms,
        } => LlmError::ProviderError {
            status: Some(status),
            message,
            retry_after_ms,
        },
        llm_suite_common::ProviderError::Decode(message) => LlmError::InvalidResponse(message),
        llm_suite_common::ProviderError::Transport(message) => LlmError::StreamError(message),
        llm_suite_common::ProviderError::Timeout => LlmError::Timeout(0),
        llm_suite_common::ProviderError::CircuitOpen => LlmError::CircuitOpen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::LlmFormat;

    fn profile() -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "p1".to_string(),
            format: LlmFormat::OpenaiChat,
            provider_id: None,
            model: "dall-e-3".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: Some("https://api.openai.com/v1".to_string()),
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

    #[test]
    fn joins_images_endpoint_without_duplication() {
        assert_eq!(
            join_images_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/images/generations"
        );
        assert_eq!(
            join_images_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/images/generations"
        );
        assert_eq!(
            join_images_url("https://api.openai.com/v1/images/generations"),
            "https://api.openai.com/v1/images/generations"
        );
    }

    #[test]
    fn context_carries_profile_transport_policy() {
        let mut pending = profile();
        pending.proxy = Some("http://127.0.0.1:8080".to_string());
        pending.no_proxy = Some(vec!["localhost".to_string()]);
        let context = images_context_for_profile(&pending).expect("context");
        assert_eq!(
            context.base_url,
            "https://api.openai.com/v1/images/generations"
        );
        assert_eq!(context.model, "dall-e-3");
        assert_eq!(context.api_key.as_deref(), Some("sk-test"));
        assert_eq!(context.proxy.as_deref(), Some("http://127.0.0.1:8080"));
        assert_eq!(context.no_proxy, vec!["localhost".to_string()]);
    }

    #[test]
    fn context_carries_identity_and_transport() {
        let mut pending = profile();
        pending.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: Some("my-app/1.0".to_string()),
            app_name: Some("my app".to_string()),
            app_url: Some("https://example.com".to_string()),
        });
        pending.transport = Some(llm_suite_types::llm::LlmHttpTransport {
            connect_timeout_secs: Some(2),
            danger_accept_invalid_certs: Some(true),
        });
        let context = images_context_for_profile(&pending).expect("context");
        assert_eq!(
            context.headers.get("User-Agent").map(String::as_str),
            Some("my-app/1.0")
        );
        assert_eq!(
            context.headers.get("X-Title").map(String::as_str),
            Some("my app")
        );
        assert_eq!(
            context.headers.get("HTTP-Referer").map(String::as_str),
            Some("https://example.com")
        );
        assert_eq!(context.connect_timeout_secs, Some(2));
        assert_eq!(context.danger_accept_invalid_certs, Some(true));
    }

    #[test]
    fn explicit_headers_win_over_identity_in_images_context() {
        let mut pending = profile();
        pending.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: Some("identity/1.0".to_string()),
            app_name: None,
            app_url: None,
        });
        pending.custom_headers = Some(
            [("User-Agent".to_string(), serde_json::json!("explicit/2.0"))]
                .into_iter()
                .collect(),
        );
        let context = images_context_for_profile(&pending).expect("context");
        assert_eq!(
            context.headers.get("User-Agent").map(String::as_str),
            Some("explicit/2.0"),
            "explicit headers must win over identity defaults"
        );
    }

    #[test]
    fn missing_base_url_fails_before_http() {
        let mut pending = profile();
        pending.base_url = None;
        assert!(images_context_for_profile(&pending).is_err());
    }

    #[tokio::test]
    async fn empty_prompt_fails_without_profile_lookup() {
        let gateway = LlmGateway::new();
        let err = gateway
            .generate_images("missing", "", &llm_suite_images::ImagesOptions::new())
            .await
            .expect_err("empty prompt must fail");
        assert!(matches!(err, LlmError::ConfigError(_)));
    }

    #[tokio::test]
    async fn unknown_profile_fails_before_http() {
        let gateway = LlmGateway::new();
        let err = gateway
            .generate_images("missing", "a cat", &llm_suite_images::ImagesOptions::new())
            .await
            .expect_err("unknown profile must fail");
        assert!(matches!(err, LlmError::ProfileNotFound(_)));
    }
}
