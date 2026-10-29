use std::sync::Arc;

use dashmap::DashMap;
use llm_suite_types::llm::{LlmFormat, LlmProfile, LlmProviderDefinition};

use llm_suite_codec::error::{LlmError, LlmResult};

/// Registry of declarative provider definitions (connection templates).
///
/// A definition carries the connection defaults shared by multiple profiles.
/// Profiles reference a definition through `LlmProfile::provider_id`; the
/// defaults are merged into the profile once at registration time, so the
/// request hot path never touches this registry.
#[derive(Clone, Default)]
pub struct ProviderDefinitionRegistry {
    providers: Arc<DashMap<String, LlmProviderDefinition>>,
}

impl ProviderDefinitionRegistry {
    pub fn new() -> Self {
        Self {
            providers: Arc::new(DashMap::new()),
        }
    }

    /// Register a definition, replacing any previous entry under the same id.
    pub fn register(&self, definition: LlmProviderDefinition) -> LlmResult<()> {
        if definition.id.trim().is_empty() {
            return Err(LlmError::ConfigError(
                "Provider definition validation failed: 'id' is required".to_string(),
            ));
        }
        if let LlmFormat::Custom(name) = &definition.format {
            if name.trim().is_empty() {
                return Err(LlmError::ConfigError(format!(
                    "Provider definition '{}' is missing 'format'",
                    definition.id
                )));
            }
        }
        self.providers.insert(definition.id.clone(), definition);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<LlmProviderDefinition> {
        self.providers.get(id).map(|entry| entry.clone())
    }

    pub fn has(&self, id: &str) -> bool {
        self.providers.contains_key(id)
    }

    pub fn remove(&self, id: &str) -> Option<LlmProviderDefinition> {
        self.providers.remove(id).map(|(_, definition)| definition)
    }

    pub fn list(&self) -> Vec<LlmProviderDefinition> {
        self.providers.iter().map(|entry| entry.clone()).collect()
    }

    pub fn clear(&self) {
        self.providers.clear();
    }

    pub fn size(&self) -> usize {
        self.providers.len()
    }
}

/// Merge a provider definition's connection defaults into a profile.
///
/// Explicit profile fields always win; only missing fields are filled from
/// the definition. Fails when the referenced definition is unknown or when
/// its wire format disagrees with the profile format.
pub fn apply_provider_defaults(
    mut profile: LlmProfile,
    providers: &ProviderDefinitionRegistry,
) -> LlmResult<LlmProfile> {
    let Some(provider_id) = profile.provider_id.clone() else {
        // Standalone profiles carry no definition to merge, but the narrow
        // auto-detection still applies so a directly configured local
        // gateway keeps working without an explicit compat block.
        if profile.compat.is_none() {
            let detected = llm_suite_types::llm::ProviderCompat::detect(
                profile.base_url.as_deref(),
                &profile.format,
            );
            if !detected.is_empty() {
                tracing::debug!(
                    profile_id = profile.id.as_str(),
                    base_url = profile.base_url.as_deref().unwrap_or(""),
                    "auto-detected provider compat; explicit config wins"
                );
                profile.compat = Some(detected);
            }
        }
        return Ok(profile);
    };
    let definition = providers.get(&provider_id).ok_or_else(|| {
        LlmError::ConfigError(format!(
            "Profile '{}' references unknown provider '{}'",
            profile.id, provider_id
        ))
    })?;

    if !formats_match(&profile.format, &definition.format) {
        return Err(LlmError::ConfigError(format!(
            "Profile '{}' format '{}' disagrees with provider '{}' format '{}'",
            profile.id,
            profile.format.as_str(),
            provider_id,
            definition.format.as_str()
        )));
    }

    if profile.base_url.is_none() {
        profile.base_url = definition.base_url.clone();
    }
    if profile.auth_type.is_none() {
        profile.auth_type = definition.auth_type.clone();
    }
    if profile.auth.is_none() {
        if let Some(auth_ref) = definition.auth.as_ref() {
            // The definition names the auth implementation owning the
            // credential (a built-in auth id or a plugin-contributed auth
            // name); the profile references it as its credential owner.
            profile.auth = Some(llm_suite_types::llm::ProfileAuthRef {
                provider_id: auth_ref.provider.clone(),
                method: auth_ref.method,
            });
        }
    }
    if profile.proxy.is_none() {
        profile.proxy = definition.proxy.clone();
    }
    if let Some(default_headers) = definition.default_headers.clone() {
        let custom_headers = profile.custom_headers.get_or_insert_with(Default::default);
        for (key, value) in default_headers {
            custom_headers.entry(key).or_insert(value);
        }
    }
    if profile.compat.is_none() {
        profile.compat = definition.compat.clone();
    } else if let (Some(profile_compat), Some(definition_compat)) =
        (profile.compat.as_mut(), definition.compat.as_ref())
    {
        let mut merged = definition_compat.clone();
        merged.merge_over(profile_compat);
        *profile_compat = merged;
    }
    if profile.identity.is_none() {
        profile.identity = definition.identity.clone();
    } else if let (Some(profile_identity), Some(definition_identity)) =
        (profile.identity.as_mut(), definition.identity.as_ref())
    {
        let mut merged = definition_identity.clone();
        merged.merge_over(profile_identity);
        *profile_identity = merged;
    }
    if profile.transport.is_none() {
        profile.transport = definition.transport.clone();
    } else if let (Some(profile_transport), Some(definition_transport)) =
        (profile.transport.as_mut(), definition.transport.as_ref())
    {
        let mut merged = definition_transport.clone();
        merged.merge_over(profile_transport);
        *profile_transport = merged;
    }
    if let Some(definition_env) = definition.env.clone() {
        let env = profile.env.get_or_insert_with(Default::default);
        for (key, value) in definition_env {
            env.entry(key).or_insert(value);
        }
    }
    if profile.compat.is_none() {
        let detected = llm_suite_types::llm::ProviderCompat::detect(
            profile
                .base_url
                .as_deref()
                .or(definition.base_url.as_deref()),
            &profile.format,
        );
        if !detected.is_empty() {
            tracing::debug!(
                profile_id = profile.id.as_str(),
                base_url = profile
                    .base_url
                    .as_deref()
                    .or(definition.base_url.as_deref())
                    .unwrap_or(""),
                "auto-detected provider compat; explicit config wins"
            );
            profile.compat = Some(detected);
        }
    }
    Ok(profile)
}

fn formats_match(profile: &LlmFormat, provider: &LlmFormat) -> bool {
    match (profile, provider) {
        (LlmFormat::Custom(a), LlmFormat::Custom(b)) => a.eq_ignore_ascii_case(b),
        (a, b) => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::LlmFormat;

    fn profile(id: &str) -> LlmProfile {
        LlmProfile {
            id: id.to_string(),
            name: id.to_string(),
            format: LlmFormat::OpenaiChat,
            provider_id: Some("acme".to_string()),
            model: "acme-7b".to_string(),
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
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    fn definition() -> LlmProviderDefinition {
        LlmProviderDefinition {
            id: "acme".to_string(),
            name: None,
            description: None,
            base_url: Some("https://api.acme.test".to_string()),
            auth_type: Some("bearer".to_string()),
            default_headers: Some(
                [("x-tenant".to_string(), serde_json::json!("t1"))]
                    .into_iter()
                    .collect(),
            ),
            format: LlmFormat::OpenaiChat,
            model_discovery: None,
            api_version: None,
            metadata: None,
            proxy: Some("http://proxy.acme.test:8080".to_string()),
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
    fn register_rejects_blank_id_or_format() {
        let registry = ProviderDefinitionRegistry::new();
        let mut bad = definition();
        bad.id = "  ".to_string();
        assert!(registry.register(bad).is_err());
        let mut bad = definition();
        bad.format = LlmFormat::Custom("  ".to_string());
        assert!(registry.register(bad).is_err());
    }

    #[test]
    fn defaults_fill_only_missing_fields() {
        let registry = ProviderDefinitionRegistry::new();
        registry.register(definition()).unwrap();

        let mut profile = profile("p1");
        profile.base_url = Some("https://override.test".to_string());
        let merged = apply_provider_defaults(profile, &registry).unwrap();
        assert_eq!(
            merged.base_url.as_deref(),
            Some("https://override.test"),
            "explicit profile values win"
        );
        assert_eq!(merged.auth_type.as_deref(), Some("bearer"));
        assert_eq!(merged.proxy.as_deref(), Some("http://proxy.acme.test:8080"));
        assert_eq!(
            merged.custom_headers.as_ref().unwrap().get("x-tenant"),
            Some(&serde_json::json!("t1"))
        );
    }

    #[test]
    fn definition_auth_reference_fills_missing_profile_auth() {
        let registry = ProviderDefinitionRegistry::new();
        let mut declared = definition();
        declared.auth = Some(llm_suite_types::llm::ProviderAuthRef {
            provider: "claude-subscription".to_string(),
            method: Some(llm_suite_types::llm::AuthMethod::OAuth),
            is_subscription: Some(true),
            params: None,
        });
        registry.register(declared).unwrap();

        let merged = apply_provider_defaults(profile("p1"), &registry).unwrap();
        let auth = merged.auth.expect("auth inherited from definition");
        assert_eq!(auth.provider_id, "claude-subscription");
        assert_eq!(auth.method, Some(llm_suite_types::llm::AuthMethod::OAuth));

        // An explicit profile auth reference always wins.
        let mut explicit = profile("p2");
        explicit.auth = Some(llm_suite_types::llm::ProfileAuthRef {
            provider_id: "own-key".to_string(),
            method: None,
        });
        let merged = apply_provider_defaults(explicit, &registry).unwrap();
        assert_eq!(merged.auth.unwrap().provider_id, "own-key");
    }

    #[test]
    fn explicit_proxy_wins_over_definition() {
        let registry = ProviderDefinitionRegistry::new();
        registry.register(definition()).unwrap();
        let mut profile = profile("p1");
        profile.proxy = Some("http://override.test:3128".to_string());
        let merged = apply_provider_defaults(profile, &registry).unwrap();
        assert_eq!(merged.proxy.as_deref(), Some("http://override.test:3128"));
    }

    #[test]
    fn identity_merges_per_field_with_profile_winning() {
        let registry = ProviderDefinitionRegistry::new();
        let mut declared = definition();
        declared.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: Some("acme/1.0".to_string()),
            app_name: Some("acme app".to_string()),
            app_url: None,
        });
        registry.register(declared).unwrap();

        let mut pending = profile("p1");
        pending.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: None,
            app_name: Some("profile app".to_string()),
            app_url: Some("https://profile.example".to_string()),
        });
        let merged = apply_provider_defaults(pending, &registry).unwrap();
        let identity = merged.identity.expect("identity must merge");
        assert_eq!(identity.user_agent.as_deref(), Some("acme/1.0"));
        assert_eq!(identity.app_name.as_deref(), Some("profile app"));
        assert_eq!(identity.app_url.as_deref(), Some("https://profile.example"));
    }

    #[test]
    fn transport_merges_per_field_with_profile_winning() {
        let registry = ProviderDefinitionRegistry::new();
        let mut declared = definition();
        declared.transport = Some(llm_suite_types::llm::LlmHttpTransport {
            connect_timeout_secs: Some(3),
            danger_accept_invalid_certs: None,
        });
        registry.register(declared).unwrap();

        let mut pending = profile("p1");
        pending.transport = Some(llm_suite_types::llm::LlmHttpTransport {
            connect_timeout_secs: None,
            danger_accept_invalid_certs: Some(true),
        });
        let merged = apply_provider_defaults(pending, &registry).unwrap();
        let transport = merged.transport.expect("transport must merge");
        assert_eq!(transport.connect_timeout_secs, Some(3));
        assert_eq!(transport.danger_accept_invalid_certs, Some(true));
    }

    #[test]
    fn env_unions_with_profile_keys_winning() {
        let registry = ProviderDefinitionRegistry::new();
        let mut declared = definition();
        declared.env = Some(
            [
                ("SHARED".to_string(), "from-definition".to_string()),
                ("DEF_ONLY".to_string(), "d".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        registry.register(declared).unwrap();

        let mut pending = profile("p1");
        pending.env = Some(
            [("SHARED".to_string(), "from-profile".to_string())]
                .into_iter()
                .collect(),
        );
        let merged = apply_provider_defaults(pending, &registry).unwrap();
        let env = merged.env.expect("env must merge");
        assert_eq!(env.get("SHARED").map(String::as_str), Some("from-profile"));
        assert_eq!(env.get("DEF_ONLY").map(String::as_str), Some("d"));
    }

    #[test]
    fn unknown_provider_id_fails() {
        let registry = ProviderDefinitionRegistry::new();
        assert!(apply_provider_defaults(profile("p1"), &registry).is_err());
    }

    #[test]
    fn profile_without_provider_id_passes_through() {
        let registry = ProviderDefinitionRegistry::new();
        let mut profile = profile("p1");
        profile.provider_id = None;
        let merged = apply_provider_defaults(profile.clone(), &registry).unwrap();
        assert_eq!(merged, profile);
    }

    #[test]
    fn format_mismatch_fails() {
        let registry = ProviderDefinitionRegistry::new();
        registry.register(definition()).unwrap();
        let mut mismatched = profile("p1");
        mismatched.format = LlmFormat::Anthropic;
        assert!(apply_provider_defaults(mismatched, &registry).is_err());
    }

    #[test]
    fn custom_format_matches_case_insensitively() {
        let registry = ProviderDefinitionRegistry::new();
        let mut custom_definition = definition();
        custom_definition.id = "custom".to_string();
        custom_definition.format = LlmFormat::Custom("ACME_CHAT".to_string());
        registry.register(custom_definition).unwrap();
        let mut custom_profile = profile("p1");
        custom_profile.provider_id = Some("custom".to_string());
        custom_profile.format = LlmFormat::Custom("acme_chat".to_string());
        assert!(apply_provider_defaults(custom_profile, &registry).is_ok());
    }

    #[test]
    fn compat_merges_per_field_with_profile_winning() {
        use llm_suite_types::llm::generation::LlmMaxTokenField;
        use llm_suite_types::llm::ProviderCompat;
        let registry = ProviderDefinitionRegistry::new();
        let mut declared = definition();
        declared.compat = Some(ProviderCompat {
            supports_developer_role: Some(false),
            supports_strict_mode: Some(true),
            max_tokens_field: Some(LlmMaxTokenField::MaxCompletionTokens),
            ..Default::default()
        });
        registry.register(declared).unwrap();

        let mut pending = profile("p1");
        pending.compat = Some(ProviderCompat {
            supports_developer_role: Some(true),
            ..Default::default()
        });
        let merged = apply_provider_defaults(pending, &registry).unwrap();
        let compat = merged.compat.expect("compat must merge");
        assert_eq!(compat.supports_developer_role, Some(true));
        assert_eq!(compat.supports_strict_mode, Some(true));
        assert_eq!(
            compat.max_tokens_field,
            Some(LlmMaxTokenField::MaxCompletionTokens)
        );
    }

    #[test]
    fn local_gateway_detects_legacy_field_without_a_definition() {
        use llm_suite_types::llm::generation::LlmMaxTokenField;
        let registry = ProviderDefinitionRegistry::new();
        let mut standalone = profile("local");
        standalone.provider_id = None;
        standalone.base_url = Some("http://localhost:11434/v1".to_string());
        let merged = apply_provider_defaults(standalone, &registry).unwrap();
        assert_eq!(
            merged.compat.and_then(|c| c.max_tokens_field),
            Some(LlmMaxTokenField::MaxTokens)
        );
    }

    #[test]
    fn explicit_compat_wins_over_detection() {
        use llm_suite_types::llm::generation::LlmMaxTokenField;
        use llm_suite_types::llm::ProviderCompat;
        let registry = ProviderDefinitionRegistry::new();
        let mut standalone = profile("local");
        standalone.provider_id = None;
        standalone.base_url = Some("http://localhost:11434/v1".to_string());
        standalone.compat = Some(ProviderCompat {
            max_tokens_field: Some(LlmMaxTokenField::MaxCompletionTokens),
            ..Default::default()
        });
        let merged = apply_provider_defaults(standalone, &registry).unwrap();
        assert_eq!(
            merged.compat.and_then(|c| c.max_tokens_field),
            Some(LlmMaxTokenField::MaxCompletionTokens)
        );
    }
}
