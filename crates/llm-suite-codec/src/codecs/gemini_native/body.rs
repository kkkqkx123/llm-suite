//! Assembly of the Gemini native request body.

use crate::error::LlmResult;
use llm_suite_types::llm::{LlmProfile, LlmRequest};

use super::{contents, tools};

/// Build the request body (separated from the HTTP layer for testability).
pub fn build_body(request: &LlmRequest, profile: &LlmProfile) -> LlmResult<serde_json::Value> {
    let generation = super::super::shared::resolve_generation(request, profile)?;

    // Only user-set fields are emitted; unset fields stay out of the
    // payload so the server-side model defaults apply. Defaults such as
    // `topK: 40` or `maxOutputTokens: 4096` must not silently clamp
    // modern Gemini models.
    let mut generation_config = serde_json::json!({});
    crate::generation::apply_gemini_generation_config(&mut generation_config, &generation)?;

    let use_text_mode = super::super::shared::is_text_mode(request);

    // System instructions are sent in the dedicated `systemInstruction`
    // field in both modes. Text mode injects the original system + tool
    // usage instructions + declarations; native mode keeps the original
    // system message.
    let (system_content, _) =
        llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);

    let history = if use_text_mode {
        super::super::shared::convert_history_for_text_mode(&request.messages, request)
    } else {
        request.messages.clone()
    };
    let messages = contents::convert_messages(&history)?;

    let mut body = serde_json::json!({
        "contents": messages,
    });

    // The generation config is only sent when it carries something;
    // an empty object would override the server-side model defaults.
    if !generation_config.is_null() && generation_config.as_object().is_some_and(|c| !c.is_empty())
    {
        body["generationConfig"] = generation_config;
    }

    // The tool-calling mode lives next to `generationConfig`, not inside
    // it; the official request envelope declares it at the top level.
    if let Some(tool_config) = crate::generation::gemini_tool_config(&generation) {
        body["toolConfig"] = tool_config;
    }

    if use_text_mode {
        let system = super::super::shared::text_mode_system_content(request);
        if !system.is_empty() {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": system}],
                "role": "user",
            });
        }
    } else {
        if let Some(system) = system_content {
            if !system.is_empty() {
                body["systemInstruction"] = serde_json::json!({
                    "parts": [{"text": system}],
                    "role": "user",
                });
            }
        }
        if let Some(tools) = &request.tools {
            // Reuse the single `convert_tools` implementation so the body
            // shape and the trait method can never drift apart.
            body["tools"] = serde_json::json!(tools::convert_tools(tools)?);
        }
    }

    // Explicit context-cache reference (Gemini `cachedContent` resource
    // created via the caches API). Sent as-is at the top level.
    if let Some(cached_content) = &request.cached_content {
        body["cachedContent"] = serde_json::json!(cached_content);
    }

    // Untyped provider options (`safetySettings`, `labels`, …) reach the
    // request body like they do in the other formats.
    super::super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

    super::super::shared::apply_custom_body(&mut body, profile);

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::gemini_native::test_support::{msg, profile, request};
    use llm_suite_types::message::{Message, MessageContentValue, MessageRole};

    #[test]
    fn empty_generation_config_is_not_sent() {
        let body = build_body(&request(vec![], None), &profile()).expect("must build");
        assert!(
            body.get("generationConfig").is_none(),
            "an empty generation config would override the model defaults"
        );
    }

    #[test]
    fn cached_content_reference_is_forwarded() {
        let mut req = request(vec![msg(MessageRole::User, "Hi")], None);
        req.cached_content = Some("cachedContents/abc123".to_string());
        let body = build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["cachedContent"],
            serde_json::json!("cachedContents/abc123")
        );

        // Absent by default: no key is emitted.
        let plain = build_body(&request(vec![], None), &profile()).expect("must build");
        assert!(plain.get("cachedContent").is_none());
    }

    #[test]
    fn the_tool_mode_is_a_top_level_field_of_the_request() {
        let mut req = request(vec![msg(MessageRole::User, "Hi")], None);
        req.generation = Some(llm_suite_types::llm::generation::LlmGenerationParams {
            tool_choice: Some(llm_suite_types::llm::generation::LlmToolChoice {
                mode: Some(llm_suite_types::llm::generation::ToolChoiceMode::None),
                tool_name: None,
            }),
            ..Default::default()
        });
        let body = build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["mode"],
            serde_json::json!("NONE")
        );
        assert!(
            body["generationConfig"].get("toolConfig").is_none(),
            "the tool mode is not a member of the generation config"
        );
    }

    #[test]
    fn native_mode_sends_system_instruction() {
        let req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hello"),
            ],
            None,
        );
        let body = build_body(&req, &profile()).expect("must build");

        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("You are a helper")
        );
        let roles: Vec<&str> = body["contents"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["role"].as_str())
            .collect();
        assert_eq!(roles, vec!["user"], "system must not appear in contents");
    }

    #[test]
    fn tool_messages_map_to_user_role_with_named_function_response() {
        let tool_msg = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::ToolResult {
                    tool_result: llm_suite_types::message::ToolResultContent {
                        tool_use_id: "call_1".to_string(),
                        content: r#"{"temp":25}"#.to_string(),
                        is_error: None,
                    },
                },
            ]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: Some("get_weather".to_string()),
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let req = request(
            vec![tool_msg, msg(MessageRole::User, "next")],
            None,
        );
        let body = build_body(&req, &profile()).expect("must build");
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(
            contents[0]["role"],
            serde_json::json!("user"),
            "Gemini only accepts user/model roles"
        );
        assert_eq!(
            contents[0]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather"),
            "functionResponse must carry the tool name"
        );
        assert_eq!(
            contents[0]["parts"][0]["functionResponse"]["response"],
            serde_json::json!({"temp": 25}),
            "a JSON tool output is passed through as the response object"
        );
    }

    #[test]
    fn native_mode_sends_tools_alongside_system() {
        let mut req = request(
            vec![
                msg(MessageRole::System, "You are a helper"),
                msg(MessageRole::User, "Hello"),
            ],
            None,
        );
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()]);
        let body = build_body(&req, &profile()).expect("must build");

        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("You are a helper")
        );
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            serde_json::json!("get_weather")
        );
        // The trait method and the body must agree on the shape.
        let trait_shape = super::super::tools::convert_tools(req.tools.as_deref().unwrap()).unwrap();
        assert_eq!(trait_shape, body["tools"].as_array().unwrap().clone());
    }

    #[test]
    fn generation_config_only_emits_user_set_fields() {
        let body = build_body(&request(vec![], None), &profile()).expect("must build");

        let config = &body["generationConfig"];
        assert!(
            config.get("temperature").is_none(),
            "unset fields must not be forced"
        );
        assert!(config.get("topP").is_none());
        assert!(config.get("topK").is_none());
        assert!(config.get("maxOutputTokens").is_none());

        let req = request(
            vec![],
            Some(serde_json::json!({"temperature": 0.2, "max_tokens": 128})),
        );
        let body = build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["generationConfig"]["temperature"],
            serde_json::json!(0.2)
        );
        assert_eq!(
            body["generationConfig"]["maxOutputTokens"],
            serde_json::json!(128)
        );
    }

    #[test]
    fn the_system_instruction_carries_a_role() {
        let body = build_body(
            &request(vec![msg(MessageRole::System, "You are a helper")], None),
            &profile(),
        )
        .expect("must build");
        assert_eq!(
            body["systemInstruction"]["role"],
            serde_json::json!("user"),
            "the official shape wraps the text in a content object"
        );
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("You are a helper")
        );
    }

    #[test]
    fn untyped_provider_options_reach_the_request_body() {
        let mut req = request(vec![], None);
        req.parameters = Some(serde_json::json!({
            "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH"}]
        }));
        let body = build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["safetySettings"][0]["category"],
            serde_json::json!("HARM_CATEGORY_HATE_SPEECH")
        );
    }

    #[test]
    fn the_official_generation_config_block_is_parsed_not_overwritten() {
        let mut req = request(vec![], None);
        req.parameters = Some(serde_json::json!({
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 256,
                "stopSequences": ["END"]
            }
        }));
        let body = build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["generationConfig"]["temperature"],
            serde_json::json!(0.4)
        );
        assert_eq!(
            body["generationConfig"]["maxOutputTokens"],
            serde_json::json!(256)
        );
        assert_eq!(
            body["generationConfig"]["stopSequences"],
            serde_json::json!(["END"])
        );
        assert!(
            body.get("generationConfig").is_none() || body["generationConfig"].is_object(),
            "the resolved configuration must survive"
        );
    }

    #[test]
    fn the_generation_members_never_reach_the_request_envelope() {
        // The generation members belong inside `generationConfig`; at the
        // top level the API rejects them as unknown fields.
        let mut req = request(vec![], None);
        req.parameters = Some(serde_json::json!({
            "responseMimeType": "application/json",
            "responseSchema": {"type": "object"},
            "candidateCount": 2
        }));
        let body = build_body(&req, &profile()).expect("must build");
        for key in ["responseMimeType", "responseSchema", "candidateCount"] {
            assert!(body.get(key).is_none(), "`{key}` must not leak: {body}");
        }
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
    }
}
