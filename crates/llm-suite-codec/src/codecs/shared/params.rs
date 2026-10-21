//! Request parameter merging and typed generation resolution.

use llm_suite_types::llm::{LlmProfile, LlmRequest};

/// Merge profile-level and request-level parameters into the body.
///
/// Untyped passthrough keys flow as-is (matching the deprecated TS
/// `Object.assign(body, otherParams)` behavior) except `stream`, which is
/// controlled by the caller, and the typed generation keys, which are
/// emitted through the canonical `generation` resolver instead. This keeps a
/// single source of truth for thinking/verbosity/limits while preserving an
/// escape hatch for provider-specific options like `logit_bias`.
pub fn merge_and_apply_params(
    body: &mut serde_json::Value,
    profile: &LlmProfile,
    request_params: &Option<serde_json::Value>,
) {
    let merged_params = crate::codecs::helpers::merge_parameters(profile, request_params);
    for (key, value) in merged_params {
        if key == "stream" || crate::generation::is_typed_param_key(&key) {
            continue;
        }
        body[key] = value;
    }
}

/// Resolve the typed generation params for a request, merging the profile
/// defaults with request overrides and legacy `parameters` keys.
pub fn resolve_generation(
    request: &LlmRequest,
    profile: &LlmProfile,
) -> crate::error::LlmResult<llm_suite_types::llm::generation::LlmGenerationParams> {
    crate::generation::resolve_generation(profile, request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::shared::test_support::profile_with;

    #[test]
    fn params_pass_through_untouched() {
        let mut profile = profile_with(None);
        profile.parameters = Some(serde_json::json!({
            "logit_bias": {"123": -100},
            "logprobs": true,
        }));
        let mut body = serde_json::json!({"model": "m", "messages": []});
        merge_and_apply_params(
            &mut body,
            &profile,
            &Some(serde_json::json!({
                "top_logprobs": 3,
                "stream": true,
            })),
        );

        assert_eq!(
            body["logit_bias"]["123"],
            serde_json::json!(-100),
            "untyped provider options must pass through"
        );
        assert_eq!(body["logprobs"], serde_json::json!(true));
        assert_eq!(body["top_logprobs"], serde_json::json!(3));
        assert!(
            body.get("stream").is_none(),
            "stream must be controlled by the caller"
        );
    }

    #[test]
    fn typed_generation_keys_are_not_duplicated_by_passthrough() {
        let mut profile = profile_with(None);
        profile.parameters = Some(serde_json::json!({
            "temperature": 0.7,
            "reasoning_effort": "high",
            "maxTokens": 100,
        }));
        let mut body = serde_json::json!({"model": "m", "messages": []});
        merge_and_apply_params(&mut body, &profile, &None);

        assert!(
            body.get("temperature").is_none(),
            "typed keys must flow through the generation resolver, not passthrough"
        );
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("maxTokens").is_none());
    }
}
