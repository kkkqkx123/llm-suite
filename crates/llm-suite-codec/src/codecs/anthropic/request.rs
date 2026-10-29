//! Assembly of the Anthropic Messages request body.

use crate::error::LlmResult;
use llm_suite_types::llm::{LlmProfile, LlmPromptCaching, LlmRequest};

use super::{caching, messages, tools};

/// Build the request body (separated from the HTTP layer for testability).
pub fn build_body(request: &LlmRequest, profile: &LlmProfile) -> LlmResult<serde_json::Value> {
    let use_text_mode = super::super::shared::is_text_mode(request);

    // `convert_messages` skips system messages; the system content is sent
    // in the dedicated `system` field. Text mode injects the original
    // system + tool usage instructions + declarations; native mode keeps
    // the original system message.
    let system_content = super::super::shared::system_content(request)?;

    let history = if use_text_mode {
        super::super::shared::convert_history_for_text_mode(&request.messages, request)
    } else {
        request.messages.clone()
    };
    let wire_messages = messages::convert_messages(&history)?;

    let mut body = serde_json::json!({
        "model": profile.model,
        "messages": wire_messages,
    });

    if use_text_mode {
        let system = super::super::shared::text_mode_system_content(request)?;
        if !system.is_empty() {
            body["system"] = serde_json::json!(system);
        }
    } else if !system_content.is_empty() {
        body["system"] = serde_json::json!(system_content);
    }

    let generation = super::super::shared::resolve_generation(request, profile)?;
    crate::generation::apply_anthropic(&mut body, &generation)?;
    super::super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

    if !use_text_mode {
        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::json!(tools::build_tool_defs(tools));
        }
    }

    if request.stream == Some(true) {
        body["stream"] = serde_json::json!(true);
    }

    super::super::shared::apply_custom_body(&mut body, profile);

    // Per-request override wins over the profile-level decision,
    // including explicitly disabling caching for a caching profile.
    let request_caching = request
        .prompt_caching
        .as_ref()
        .map(|c| LlmPromptCaching { ttl: c.ttl.clone() });
    let caching_config: Option<&LlmPromptCaching> = match &request.prompt_caching {
        Some(req) if !req.enabled => None,
        Some(_) => request_caching.as_ref(),
        None => profile.prompt_caching.as_ref(),
    };
    if let Some(caching_config) = caching_config {
        caching::apply_prompt_caching_with_compat(
            &mut body,
            caching_config,
            profile.compat.as_ref(),
        );
    }

    Ok(body)
}

/// Build the count-tokens request body (system included so the count
/// matches what `build_request` sends).
pub fn build_count_tokens_body(
    request: &LlmRequest,
    profile: &LlmProfile,
) -> LlmResult<serde_json::Value> {
    let use_text_mode = super::super::shared::is_text_mode(request);
    let history = if use_text_mode {
        super::super::shared::convert_history_for_text_mode(&request.messages, request)
    } else {
        request.messages.clone()
    };
    let wire_messages = messages::convert_messages(&history)?;

    let mut body = serde_json::json!({
        "model": profile.model,
        "messages": wire_messages,
    });

    if use_text_mode {
        let system = super::super::shared::text_mode_system_content(request)?;
        if !system.is_empty() {
            body["system"] = serde_json::json!(system);
        }
    } else {
        let system = super::super::shared::system_content(request)?;
        if !system.is_empty() {
            body["system"] = serde_json::json!(system);
        }
    }
    if !use_text_mode {
        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::json!(tools::build_tool_defs(tools));
        }
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::anthropic::test_support::{msg, profile, request, rich_msg};
    use llm_suite_types::llm::{LlmRequestCaching, ToolCallProtocol};
    use llm_suite_types::message::{MessageContent, MessageRole};

    #[test]
    fn native_mode_sends_system_field() {
        let body = build_body(
            &request(
                vec![
                    msg(MessageRole::System, "You are a helper"),
                    msg(MessageRole::User, "Hello"),
                ],
                None,
            ),
            &profile(),
        )
        .expect("build must succeed");

        assert_eq!(body["system"], serde_json::json!("You are a helper"));
        let roles: Vec<&str> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["role"].as_str())
            .collect();
        assert_eq!(roles, vec!["user"], "system must not appear in messages");
    }

    #[test]
    fn text_mode_system_field_includes_tool_instructions() {
        let mut req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hello"),
            ],
            None,
        );
        req.tool_call_protocol = Some(ToolCallProtocol::Xml);
        let body = build_body(&req, &profile()).expect("must build");

        let system = body["system"].as_str().unwrap();
        assert!(system.contains("You are a helper"));
        assert!(system.contains("<tool_use>"), "must carry instructions");
    }

    #[test]
    fn params_pass_through_except_stream_and_stop() {
        let req = request(
            vec![msg(MessageRole::User, "Hello")],
            Some(serde_json::json!({
                "temperature": 0.2,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "tool_choice": {"type": "auto"},
                "stop": ["END"],
                "stream": true,
            })),
        );
        let body = build_body(&req, &profile()).expect("must build");

        assert_eq!(body["thinking"]["budget_tokens"], serde_json::json!(1024));
        assert_eq!(body["tool_choice"]["type"], serde_json::json!("auto"));
        assert_eq!(body["stop_sequences"], serde_json::json!(["END"]));
        assert!(body.get("stop").is_none(), "stop must be mapped");
        assert!(body.get("stream").is_none(), "stream controlled by caller");
        assert_eq!(
            body["temperature"],
            serde_json::json!(0.2),
            "the sampling parameters are still accepted by the API"
        );
    }

    #[test]
    fn tool_choice_none_uses_the_official_shape() {
        use llm_suite_types::llm::generation::{LlmToolChoice, ToolChoiceMode};

        let mut body = serde_json::json!({"model": "claude"});
        crate::generation::apply_anthropic(
            &mut body,
            &llm_suite_types::llm::generation::LlmGenerationParams {
                tool_choice: Some(LlmToolChoice {
                    mode: Some(ToolChoiceMode::None),
                    tool_name: None,
                }),
                parallel_tool_calls: Some(false),
                ..Default::default()
            },
        )
        .expect("tool choice must be accepted");
        assert_eq!(body["tool_choice"]["type"], serde_json::json!("none"));
        assert!(
            body["tool_choice"]
                .get("disable_parallel_tool_use")
                .is_none(),
            "the none variant carries no parallel-tool flag"
        );
    }

    #[test]
    fn count_tokens_includes_system() {
        let req = request(vec![msg(MessageRole::System, "You are a helper")], None);
        let body = build_count_tokens_body(&req, &profile()).expect("must build");
        assert_eq!(body["system"], serde_json::json!("You are a helper"));
    }

    #[test]
    fn tool_without_parameters_gets_empty_object_schema() {
        let mut req = request(vec![msg(MessageRole::User, "hi")], None);
        req.tools = Some(vec![llm_suite_types::tool::Tool {
            name: "get_time".to_string(),
            description: "Returns the current time".to_string(),
            parameters: None,
        }]);
        let body = build_body(&req, &profile()).expect("must build");
        let schema = &body["tools"][0]["input_schema"];
        assert_eq!(schema["type"], serde_json::json!("object"));
        assert!(
            schema.get("properties").is_some(),
            "input_schema must not be null"
        );
    }

    #[test]
    fn prompt_caching_breakpoints_are_applied() {
        let mut req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hello"),
                msg(MessageRole::Assistant, "Hi there"),
                msg(MessageRole::User, "Bye"),
            ],
            None,
        );
        req.tools = Some(vec![llm_suite_types::tool::Tool {
            name: "get_weather".to_string(),
            description: "weather lookup".to_string(),
            parameters: None,
        }]);

        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching { ttl: None });
        let body = build_body(&req, &caching_profile).expect("must build");

        // Default TTL shape is the bare ephemeral marker (provider default 5m).
        let control = serde_json::json!({"type": "ephemeral"});

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["cache_control"], control);

        // System is upgraded to the block-array form carrying the marker.
        assert_eq!(
            body["system"],
            serde_json::json!([{
                "type": "text",
                "text": "You are a helper",
                "cache_control": control,
            }])
        );

        // The first and the last message are marked: the stable prefix and
        // the current turn. The API allows four breakpoints.
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        let last_blocks = messages.last().unwrap()["content"].as_array().unwrap();
        assert_eq!(last_blocks.last().unwrap()["cache_control"], control);
        let first_blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(first_blocks.last().unwrap()["cache_control"], control);
        assert_eq!(messages[1]["content"][0].get("cache_control"), None);
    }

    #[test]
    fn prompt_caching_ttl_is_forwarded() {
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        let body = build_body(
            &request(
                vec![
                    msg(MessageRole::System, "You are a helper"),
                    msg(MessageRole::User, "Hi"),
                ],
                None,
            ),
            &caching_profile,
        )
        .expect("must build");
        let control = serde_json::json!({"type": "ephemeral", "ttl": "1h"});
        assert_eq!(
            body["messages"].as_array().unwrap().last().unwrap()["content"][0]["cache_control"],
            control
        );
        assert_eq!(body["system"][0]["cache_control"], control);
    }

    #[test]
    fn without_prompt_caching_no_cache_control_is_emitted() {
        let body = build_body(
            &request(vec![msg(MessageRole::User, "Hi")], None),
            &profile(),
        )
        .expect("must build");
        assert_eq!(body["system"], serde_json::json!(null));
        assert!(body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
    }

    #[test]
    fn request_override_enables_caching_without_profile_setting() {
        let mut req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hi"),
            ],
            None,
        );
        req.prompt_caching = Some(LlmRequestCaching {
            enabled: true,
            ttl: Some("1h".to_string()),
        });
        let body = build_body(&req, &profile()).expect("must build");
        let control = serde_json::json!({"type": "ephemeral", "ttl": "1h"});
        assert_eq!(body["system"][0]["cache_control"], control);
        assert_eq!(
            body["messages"].as_array().unwrap().last().unwrap()["content"][0]["cache_control"],
            control
        );
    }

    #[test]
    fn request_override_disables_profile_caching() {
        let mut req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hi"),
            ],
            None,
        );
        req.prompt_caching = Some(LlmRequestCaching {
            enabled: false,
            ttl: None,
        });
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching { ttl: None });
        let body = build_body(&req, &caching_profile).expect("must build");
        // System stays in the plain-string form (no block-array upgrade).
        assert_eq!(body["system"], serde_json::json!("You are a helper"));
        assert!(body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
    }

    #[test]
    fn request_override_ttl_wins_over_profile() {
        let mut req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hi"),
            ],
            None,
        );
        req.prompt_caching = Some(LlmRequestCaching {
            enabled: true,
            ttl: Some("5m".to_string()),
        });
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        let body = build_body(&req, &caching_profile).expect("must build");
        let control = serde_json::json!({"type": "ephemeral", "ttl": "5m"});
        assert_eq!(body["system"][0]["cache_control"], control);
    }

    #[test]
    fn the_cache_breakpoint_skips_a_thinking_block() {
        // The thinking blocks carry no `cache_control`; the breakpoint goes
        // on the last block that accepts one.
        let assistant = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Text {
                    text: "the answer".to_string(),
                },
                MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
                },
            ],
        );
        let req = request(vec![msg(MessageRole::User, "Hi"), assistant], None);
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        let body = build_body(&req, &caching_profile).expect("must build");

        let messages = body["messages"].as_array().unwrap();
        let last_blocks = messages.last().unwrap()["content"].as_array().unwrap();
        let marked = last_blocks
            .iter()
            .find(|block| block.get("cache_control").is_some())
            .expect("the breakpoint must be placed");
        assert_eq!(marked["type"], serde_json::json!("text"));
    }

    #[test]
    fn long_retention_falls_back_when_the_endpoint_rejects_it() {
        let req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hi"),
            ],
            None,
        );
        let mut rejecting = profile();
        rejecting.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        rejecting.compat = Some(llm_suite_types::llm::ProviderCompat {
            supports_long_cache_retention: Some(false),
            ..Default::default()
        });
        let body = build_body(&req, &rejecting).expect("must build");
        assert_eq!(
            body["system"][0]["cache_control"],
            serde_json::json!({"type": "ephemeral"}),
            "the ttl marker must be omitted, leaving the provider default"
        );
    }

    #[test]
    fn long_retention_is_kept_without_compat_objection() {
        let req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hi"),
            ],
            None,
        );
        let mut keeping = profile();
        keeping.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        keeping.compat = Some(llm_suite_types::llm::ProviderCompat {
            supports_long_cache_retention: Some(true),
            ..Default::default()
        });
        let body = build_body(&req, &keeping).expect("must build");
        assert_eq!(
            body["system"][0]["cache_control"],
            serde_json::json!({"type": "ephemeral", "ttl": "1h"})
        );
    }
}
