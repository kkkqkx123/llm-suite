//! Profile-side catalog helpers (no network).
//!
//! The network half of discovery (`ModelCatalog`, endpoint joining,
//! extraction, credential filtering) lives in `llm-suite-discovery`.
//! What stays here is pure and dependency-free: bridging a caller-supplied
//! model catalog slice into profile defaults. The caller builds the slice
//! from its own pricing table or discovery results, so this crate never
//! carries concrete provider or model data.

use llm_suite_types::llm::{LlmProfile, ModelMetadata};

/// Fill profile gaps from a model catalog slice.
///
/// Only fills `context_window_size` when the profile leaves it empty and
/// an entry matches `profile.model` exactly (optionally scoped by
/// `profile.provider_id`). Never overwrites explicit config.
pub fn apply_catalog_metadata(mut profile: LlmProfile, catalog: &[ModelMetadata]) -> LlmProfile {
    if profile.context_window_size.is_some() {
        return profile;
    }
    if let Some(entry) = catalog_entry_for_profile(&profile, catalog) {
        if let Some(window) = entry.context_window {
            profile.context_window_size = Some(window);
        }
    }
    profile
}

/// Look up the catalog entry a profile resolves to, if any.
pub fn catalog_entry_for_profile<'a>(
    profile: &LlmProfile,
    catalog: &'a [ModelMetadata],
) -> Option<&'a ModelMetadata> {
    catalog.iter().find(|m| {
        m.id == profile.model
            && profile
                .provider_id
                .as_ref()
                .map(|id| id == &m.provider_id)
                .unwrap_or(true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::{InputModality, LlmFormat, ModelType};

    fn test_profile(model: &str, provider_id: Option<&str>) -> LlmProfile {
        LlmProfile {
            id: "p".to_string(),
            name: "p".to_string(),
            format: LlmFormat::OpenaiChat,
            provider_id: provider_id.map(str::to_string),
            model: model.to_string(),
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

    fn test_entry(id: &str, provider_id: &str, window: Option<u32>) -> ModelMetadata {
        ModelMetadata {
            id: id.to_string(),
            provider_id: provider_id.to_string(),
            model_type: ModelType::Chat,
            format: None,
            context_window: window,
            max_output_tokens: None,
            input_modalities: vec![InputModality::Text],
            reasoning: false,
            thinking_levels: Vec::new(),
            cost: None,
        }
    }

    #[test]
    fn fill_only_fills_missing_window() {
        let catalog = vec![test_entry("m1", "p1", Some(128_000))];
        let filled = apply_catalog_metadata(test_profile("m1", Some("p1")), &catalog);
        assert_eq!(filled.context_window_size, Some(128_000));

        let mut explicit = filled.clone();
        explicit.context_window_size = Some(8_000);
        let kept = apply_catalog_metadata(explicit, &catalog);
        assert_eq!(kept.context_window_size, Some(8_000));

        let untouched = apply_catalog_metadata(test_profile("missing", Some("p1")), &catalog);
        assert_eq!(untouched.context_window_size, None);
    }

    #[test]
    fn provider_scoping_disambiguates_shared_model_names() {
        let catalog = vec![
            test_entry("shared", "a", Some(10)),
            test_entry("shared", "b", Some(20)),
        ];
        let filled = apply_catalog_metadata(test_profile("shared", Some("b")), &catalog);
        assert_eq!(filled.context_window_size, Some(20));
        // Without a provider scope the first matching entry wins.
        let unscoped = apply_catalog_metadata(test_profile("shared", None), &catalog);
        assert_eq!(unscoped.context_window_size, Some(10));
    }
}
