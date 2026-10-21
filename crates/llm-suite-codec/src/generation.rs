use crate::error::LlmResult;
use llm_suite_types::llm::generation::LlmGenerationParams;
use llm_suite_types::llm::{LlmProfile, LlmRequest};

pub mod emit;
pub mod parse;
pub mod validate;

pub use emit::{
    apply_anthropic, apply_gemini_generation_config, apply_openai_chat, apply_openai_responses,
    gemini_schema_value, gemini_tool_config,
};
pub use parse::{is_typed_param_key, parse_legacy_generation};
pub use validate::{validate_generation, ANTHROPIC_DEFAULT_MAX_TOKENS};

pub fn resolve_generation(
    profile: &LlmProfile,
    request: &LlmRequest,
) -> LlmResult<LlmGenerationParams> {
    let mut resolved = LlmGenerationParams::default();
    if let Some(ref legacy) = profile.parameters {
        resolved.merge_over(&parse::parse_legacy_generation(legacy));
    }
    if let Some(ref typed) = profile.generation {
        resolved.merge_over(typed);
    }
    if let Some(ref legacy) = request.parameters {
        resolved.merge_over(&parse::parse_legacy_generation(legacy));
    }
    if let Some(ref typed) = request.generation {
        resolved.merge_over(typed);
    }
    validate::validate_generation(&resolved, &profile.format)?;
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::generation::{
        LlmMaxTokenField, LlmThinkingConfig, ThinkingLevel, Verbosity,
    };
    use llm_suite_types::llm::LlmFormat;

    fn profile_with(
        format: LlmFormat,
        generation: Option<LlmGenerationParams>,
        parameters: Option<serde_json::Value>,
    ) -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format,
            provider_id: None,
            model: "test-model".to_string(),
            api_key: None,
            base_url: None,
            parameters,
            generation,
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
        }
    }

    fn request_with(
        generation: Option<LlmGenerationParams>,
        parameters: Option<serde_json::Value>,
    ) -> LlmRequest {
        LlmRequest {
            profile_id: "p1".to_string(),
            messages: Vec::new(),
            parameters,
            generation,
            tools: None,
            tool_call_protocol: None,
            locked_tool_call_protocol: None,
            violation_policy: None,
            execution_id: None,
            stream: None,
            dead_loop_detection: None,
            protocol_auto_converted: None,
            timeout_ms: None,
            prompt_caching: None,
            cached_content: None,
        }
    }

    #[test]
    fn legacy_camel_case_keys_are_normalized() {
        let legacy = serde_json::json!({
            "maxTokens": 512,
            "topP": 0.9,
            "reasoningEffort": "high",
            "verbosity": "low",
        });
        let parsed = parse_legacy_generation(&legacy);
        assert_eq!(parsed.max_tokens, Some(512));
        assert_eq!(parsed.top_p, Some(0.9));
        assert_eq!(parsed.thinking_level(), Some(ThinkingLevel::High));
        assert_eq!(parsed.verbosity, Some(Verbosity::Low));
    }

    #[test]
    fn typed_generation_wins_over_legacy() {
        let profile = profile_with(
            LlmFormat::OpenaiChat,
            Some(LlmGenerationParams {
                temperature: Some(0.2),
                ..Default::default()
            }),
            Some(serde_json::json!({"temperature": 0.9, "maxTokens": 100})),
        );
        let resolved = resolve_generation(&profile, &request_with(None, None)).unwrap();
        assert_eq!(resolved.temperature, Some(0.2));
        assert_eq!(resolved.max_tokens, Some(100));
    }

    #[test]
    fn anthropic_thinking_budget_is_validated() {
        let profile = profile_with(LlmFormat::Anthropic, None, None);
        let request = request_with(
            Some(LlmGenerationParams {
                max_tokens: Some(1000),
                thinking: Some(LlmThinkingConfig {
                    budget_tokens: Some(2000),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            None,
        );
        assert!(resolve_generation(&profile, &request).is_err());
    }

    #[test]
    fn gemini_thinking_level_and_budget_conflict() {
        let params = LlmGenerationParams {
            thinking: Some(LlmThinkingConfig {
                level: Some(ThinkingLevel::High),
                budget_tokens: Some(1024),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut config = serde_json::json!({});
        assert!(apply_gemini_generation_config(&mut config, &params).is_err());
    }

    #[test]
    fn openai_chat_max_token_field_is_explicit_not_inferred() {
        let mut body = serde_json::json!({"model": "gpt-5"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                max_tokens: Some(512),
                max_token_field: Some(LlmMaxTokenField::MaxCompletionTokens),
                thinking: Some(LlmThinkingConfig {
                    level: Some(ThinkingLevel::Xhigh),
                    ..Default::default()
                }),
                verbosity: Some(Verbosity::Low),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        assert_eq!(body["max_completion_tokens"], serde_json::json!(512));
        assert_eq!(body["reasoning_effort"], serde_json::json!("xhigh"));
        assert_eq!(body["verbosity"], serde_json::json!("low"));
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn openai_chat_reasoning_model_name_does_not_change_field() {
        // The model name must never drive emission: without an explicit
        // `max_token_field` the officially recommended field is used.
        let mut body = serde_json::json!({"model": "gpt-5"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                max_tokens: Some(512),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        assert_eq!(body["max_completion_tokens"], serde_json::json!(512));
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn openai_chat_legacy_max_tokens_field_is_opt_in() {
        let mut body = serde_json::json!({"model": "local-model"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                max_tokens: Some(512),
                max_token_field: Some(LlmMaxTokenField::MaxTokens),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        assert_eq!(body["max_tokens"], serde_json::json!(512));
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn openai_chat_non_reasoning_uses_the_recommended_field() {
        let mut body = serde_json::json!({"model": "gpt-4o"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                max_tokens: Some(512),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        assert_eq!(body["max_completion_tokens"], serde_json::json!(512));
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn chat_rejects_the_responses_only_service_tier() {
        let mut body = serde_json::json!({"model": "gpt-4o"});
        let err = apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                service_tier: Some(llm_suite_types::llm::generation::LlmServiceTier::Ultrafast),
                ..Default::default()
            },
        )
        .expect_err("`ultrafast` is not a Chat Completions tier");
        assert!(format!("{err}").contains("ultrafast"), "{err}");
    }

    #[test]
    fn responses_accepts_the_ultrafast_service_tier() {
        let mut body = serde_json::json!({"model": "gpt-4o"});
        apply_openai_responses(
            &mut body,
            &LlmGenerationParams {
                service_tier: Some(llm_suite_types::llm::generation::LlmServiceTier::Ultrafast),
                ..Default::default()
            },
        )
        .expect("`ultrafast` is a Responses tier");
        assert_eq!(body["service_tier"], serde_json::json!("ultrafast"));
    }

    #[test]
    fn the_end_user_identifier_uses_the_safety_identifier_field() {
        let mut body = serde_json::json!({"model": "gpt-4o"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                user_id: Some("user-1".to_string()),
                ..Default::default()
            },
        )
        .expect("must apply");
        assert_eq!(body["safety_identifier"], serde_json::json!("user-1"));
        assert!(body.get("user").is_none(), "`user` is superseded");
    }

    #[test]
    fn anthropic_adaptive_thinking_emits_effort() {
        let mut body = serde_json::json!({"model": "claude"});
        apply_anthropic(
            &mut body,
            &LlmGenerationParams {
                max_tokens: Some(16000),
                thinking: Some(LlmThinkingConfig {
                    level: Some(ThinkingLevel::High),
                    adaptive: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(body["thinking"]["type"], serde_json::json!("adaptive"));
        assert_eq!(body["output_config"]["effort"], serde_json::json!("high"));
        assert_eq!(body["max_tokens"], serde_json::json!(16000));
    }

    #[test]
    fn openai_chat_json_schema_response_format_defaults_strict_false() {
        use llm_suite_types::llm::generation::LlmResponseFormat;
        use llm_suite_types::llm::generation::ResponseFormatKind;
        let mut body = serde_json::json!({"model": "gpt-4o"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                response_format: Some(LlmResponseFormat {
                    kind: Some(ResponseFormatKind::JsonSchema),
                    name: Some("resp".to_string()),
                    schema: Some(serde_json::json!({"type": "object"})),
                    strict: None,
                }),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        let rf = body["response_format"]["json_schema"]
            .as_object()
            .expect("json_schema present");
        assert_eq!(rf.get("strict"), Some(&serde_json::json!(false)));
    }

    #[test]
    fn openai_chat_json_schema_without_a_schema_stays_valid() {
        use llm_suite_types::llm::generation::LlmResponseFormat;
        use llm_suite_types::llm::generation::ResponseFormatKind;
        let mut body = serde_json::json!({"model": "gpt-4o"});
        apply_openai_chat(
            &mut body,
            &LlmGenerationParams {
                response_format: Some(LlmResponseFormat {
                    kind: Some(ResponseFormatKind::JsonSchema),
                    name: Some("resp".to_string()),
                    schema: None,
                    strict: None,
                }),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        // The definition lives under `json_schema`; a bare `name` next to
        // the type is not read by the API.
        assert_eq!(
            body["response_format"]["type"],
            serde_json::json!("json_schema")
        );
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            serde_json::json!("resp")
        );
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            serde_json::json!({"type": "object"})
        );
        assert!(body["response_format"].get("name").is_none());
    }

    #[test]
    fn the_anthropic_sampling_parameters_reach_the_request_body() {
        // The official SDK no longer types them, but the API still accepts
        // them for the models that use them.
        let mut body = serde_json::json!({"model": "claude"});
        apply_anthropic(
            &mut body,
            &LlmGenerationParams {
                temperature: Some(0.2),
                top_p: Some(0.9),
                top_k: Some(40),
                ..Default::default()
            },
        )
        .expect("generation must apply");
        assert_eq!(body["temperature"], serde_json::json!(0.2));
        assert_eq!(body["top_p"], serde_json::json!(0.9));
        assert_eq!(body["top_k"], serde_json::json!(40));
    }

    #[test]
    fn the_gemini_generation_config_block_is_read() {
        let legacy = serde_json::json!({
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.8,
                "topK": 20,
                "maxOutputTokens": 256,
                "stopSequences": ["END"],
                "thinkingConfig": {"thinkingBudget": 1024, "includeThoughts": true},
                "responseMimeType": "application/json",
                "responseSchema": {"type": "object"}
            }
        });
        let parsed = parse_legacy_generation(&legacy);
        assert_eq!(parsed.temperature, Some(0.4));
        assert_eq!(parsed.top_p, Some(0.8));
        assert_eq!(parsed.top_k, Some(20));
        assert_eq!(parsed.max_tokens, Some(256));
        assert_eq!(parsed.stop, Some(vec!["END".to_string()]));
        let thinking = parsed.thinking.expect("thinking must be read");
        assert_eq!(thinking.budget_tokens, Some(1024));
        assert_eq!(thinking.include_thoughts, Some(true));
        let format = parsed.response_format.expect("format must be read");
        assert_eq!(
            format.kind,
            Some(llm_suite_types::llm::generation::ResponseFormatKind::JsonSchema)
        );
        assert_eq!(format.schema, Some(serde_json::json!({"type": "object"})));
    }

    #[test]
    fn the_outer_keys_win_over_the_gemini_generation_config_block() {
        let legacy = serde_json::json!({
            "temperature": 0.9,
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 256}
        });
        let parsed = parse_legacy_generation(&legacy);
        assert_eq!(parsed.temperature, Some(0.9));
        assert_eq!(parsed.max_tokens, Some(256));
    }
}
