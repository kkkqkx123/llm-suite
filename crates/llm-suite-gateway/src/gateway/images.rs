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
        let (profile, _) = match self.auth.as_ref() {
            Some(registry) => match registry
                .apply(&stored)
                .await
                .map_err(|e| LlmError::ConfigError(e.to_string()))?
            {
                Some(resolved) => (resolved.profile, resolved.is_subscription),
                None => (stored, false),
            },
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
/// proxy, bypass list, extra headers and query params travel verbatim so an
/// images call observes the same transport policy as chat.
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
    for (name, value) in metadata_headers(profile.custom_headers.as_ref()) {
        context.headers.insert(name, value);
    }
    for (name, value) in metadata_headers(profile.headers.as_ref()) {
        context.headers.entry(name).or_insert(value);
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
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
            auth: None,
            compat: None,
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
