//! Shared fixtures and codec-level tests for the OpenAI Chat codec.
//!
//! Unit tests for the individual stages live next to those stages
//! (`messages.rs`, `stream.rs`, …); everything here exercises the codec
//! facade end to end through its public `LlmCodec` methods.

use super::OpenaiChatCodec;
use crate::codecs::LlmCodec;
use llm_suite_types::llm::{LlmProfile, LlmRequest, LlmStreamOptions, ToolCallProtocol};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

pub(crate) fn text_msg(role: MessageRole, text: &str) -> Message {
    Message {
        id: llm_suite_types::Id::new(),
        role,
        content: MessageContentValue::Text(text.to_string()),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    }
}

pub(crate) fn profile() -> LlmProfile {
    LlmProfile {
        id: "p1".to_string(),
        name: "test".to_string(),
        format: llm_suite_types::llm::LlmFormat::OpenaiChat,
        provider_id: None,
        model: "gpt-4o".to_string(),
        api_key: Some("sk-test".to_string()),
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
    }
}

pub(crate) fn request_with_format(format: ToolCallProtocol) -> LlmRequest {
    LlmRequest {
        profile_id: "p1".to_string(),
        messages: vec![
            text_msg(MessageRole::System, "You are a helper"),
            text_msg(MessageRole::User, "What is the weather?"),
        ],
        parameters: None,
        generation: None,
        tools: Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"name": "city", "value": null, "type": "string"}},
                "required": ["city"]
            }
        }))
        .unwrap()]),
        tool_call_protocol: Some(format),
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

pub(crate) fn rich_tool_msg() -> Message {
    Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::Assistant,
        content: MessageContentValue::Rich(vec![
            MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Beijing"}),
                    thought_signature: None,
                },
            },
            MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    blocks: None,
                    is_error: Some(true),
                },
            },
        ]),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_options_are_only_sent_when_the_profile_asks_for_them() {
        let codec = OpenaiChatCodec::new();
        let mut req = request_with_format(ToolCallProtocol::Native);
        req.stream = Some(true);

        let body = codec.build_body(&req, &profile()).unwrap();
        assert!(
            body.get("stream_options").is_none(),
            "the usage block must be opt-in"
        );

        let mut p = profile();
        p.stream_options = Some(LlmStreamOptions {
            include_usage: Some(false),
        });
        let body = codec.build_body(&req, &p).unwrap();
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn text_mode_injects_templates_and_skips_native_tools() {
        let codec = OpenaiChatCodec::new();
        let body = codec
            .build_body(&request_with_format(ToolCallProtocol::Xml), &profile())
            .unwrap();

        assert!(body.get("tools").is_none(), "text mode must not send tools");
        let system = body["messages"][0]["content"].as_str().unwrap();
        assert!(
            system.contains("get_weather"),
            "system must declare tools: {system}"
        );
        assert!(
            system.contains("<tool_use>"),
            "system must carry usage instructions"
        );
        assert!(system.contains("You are a helper"), "original system kept");
    }

    #[test]
    fn native_mode_sends_tools_and_keeps_original_system() {
        let codec = OpenaiChatCodec::new();
        let body = codec
            .build_body(&request_with_format(ToolCallProtocol::Native), &profile())
            .unwrap();

        assert!(body.get("tools").is_some());
        let system = body["messages"][0]["content"].as_str().unwrap();
        assert_eq!(system, "You are a helper");
    }

    #[test]
    fn text_mode_parse_extracts_xml_tool_calls() {
        let codec = OpenaiChatCodec::new();
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"<tool_use>\n  <tool_name>get_weather</tool_name>\n  <parameters>\n    <city>Beijing</city>\n  </parameters>\n</tool_use>"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

        let result = codec
            .parse_response(body, &request_with_format(ToolCallProtocol::Xml))
            .unwrap();
        let calls = result.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Beijing"));
    }

    #[test]
    fn stream_options_from_profile() {
        let mut p = profile();
        p.stream_options = Some(LlmStreamOptions {
            include_usage: Some(false),
        });
        let mut req = request_with_format(ToolCallProtocol::Native);
        req.stream = Some(true);
        let codec = OpenaiChatCodec::new();
        let body = codec.build_body(&req, &p).unwrap();
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn custom_body_is_deep_merged() {
        let mut p = profile();
        p.custom_body = Some(serde_json::json!({"custom_field": {"nested": 1}}));
        let codec = OpenaiChatCodec::new();
        let body = codec
            .build_body(&request_with_format(ToolCallProtocol::Native), &p)
            .unwrap();
        assert_eq!(body["custom_field"]["nested"], serde_json::json!(1));
    }

    #[test]
    fn prompt_cache_key_passthrough_via_request_parameters() {
        // OpenAI prompt caching is automatic; `prompt_cache_key` (cache
        // routing hint) reaches the wire through the generic request
        // `parameters` passthrough — no dedicated field needed.
        let mut req = request_with_format(ToolCallProtocol::Native);
        req.parameters = Some(serde_json::json!({"prompt_cache_key": "conv-42"}));
        let codec = OpenaiChatCodec::new();
        let body = codec.build_body(&req, &profile()).unwrap();
        assert_eq!(body["prompt_cache_key"], serde_json::json!("conv-42"));
    }

    #[test]
    fn text_mode_converts_tool_call_history_to_blocks() {
        let mut req = request_with_format(ToolCallProtocol::Xml);
        let mut assistant = text_msg(MessageRole::Assistant, "checking weather");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Beijing"}"#.to_string(),
            },
            thought_signature: None,
        }]);
        let mut tool_result = text_msg(MessageRole::Tool, "sunny");
        tool_result.tool_call_id = Some("call_1".to_string());
        req.messages.push(assistant);
        req.messages.push(tool_result);

        let codec = OpenaiChatCodec::new();
        let body = codec.build_body(&req, &profile()).unwrap();

        let messages = body["messages"].as_array().unwrap();
        let assistant_msg = messages
            .iter()
            .find(|m| m["role"] == serde_json::json!("assistant"))
            .expect("assistant message present");
        assert!(
            assistant_msg.get("tool_calls").is_none(),
            "native tool_calls must be dropped in text mode"
        );
        let content = assistant_msg["content"].as_str().unwrap();
        assert!(content.contains("<tool_use>"), "{content}");
        assert!(content.contains("get_weather"), "{content}");

        let result_msg = messages
            .iter()
            .find(|m| {
                m["role"] == serde_json::json!("user")
                    && m["content"]
                        .as_str()
                        .is_some_and(|c| c.contains("<tool_result>"))
            })
            .expect("converted tool result present");
        assert!(
            result_msg.get("tool_call_id").is_none(),
            "tool_call_id must be dropped after conversion"
        );
    }

    #[test]
    fn native_mode_rejects_tool_message_without_tool_call_id() {
        let codec = OpenaiChatCodec::new();
        let mut req = request_with_format(ToolCallProtocol::Native);
        // A `tool` role message without `tool_call_id` is invalid for OpenAI Chat.
        req.messages.push(text_msg(MessageRole::Tool, "sunny"));
        let err = codec
            .build_body(&req, &profile())
            .expect_err("missing tool_call_id must fail fast");
        assert!(
            format!("{err}").contains("tool_call_id"),
            "error must mention tool_call_id: {err}"
        );
    }

    #[test]
    fn rich_tool_block_becomes_a_tool_call_with_json_arguments() {
        let codec = OpenaiChatCodec::new();
        let mut req = request_with_format(ToolCallProtocol::Native);
        req.messages.push(rich_tool_msg());
        let body = codec.build_body(&req, &profile()).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let assistant = messages
            .iter()
            .find(|m| m["role"] == serde_json::json!("assistant"))
            .expect("assistant message present");
        assert_eq!(
            assistant["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            assistant["tool_calls"][0]["function"]["arguments"],
            serde_json::json!(r#"{"city":"Beijing"}"#),
            "arguments must be a JSON string"
        );
    }

    #[test]
    fn tool_result_block_becomes_a_tool_message_without_error_flag() {
        let codec = OpenaiChatCodec::new();
        let mut req = request_with_format(ToolCallProtocol::Native);
        req.messages.push(rich_tool_msg());
        let body = codec.build_body(&req, &profile()).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let tool_msg = messages
            .iter()
            .find(|m| m["role"] == serde_json::json!("tool"))
            .expect("tool message present");
        assert_eq!(tool_msg["tool_call_id"], "call_1");
        assert_eq!(tool_msg["content"], "sunny");
        assert!(
            tool_msg.get("is_error").is_none(),
            "Chat Completions tool messages carry no error flag"
        );
    }

    #[test]
    fn a_tool_without_parameters_declares_an_empty_object_schema() {
        let codec = OpenaiChatCodec::new();
        let mut req = request_with_format(ToolCallProtocol::Native);
        req.tools = Some(vec![llm_suite_types::tool::Tool {
            name: "get_time".to_string(),
            description: "Returns the current time".to_string(),
            parameters: None,
        }]);
        let body = codec.build_body(&req, &profile()).expect("must build");
        let parameters = &body["tools"][0]["function"]["parameters"];
        assert!(
            parameters.is_object(),
            "the schema must be an object, never null: {body:?}"
        );
        assert_eq!(parameters["type"], serde_json::json!("object"));
    }

    #[test]
    fn a_custom_body_replaces_the_built_message_list() {
        let mut p = profile();
        p.custom_body =
            Some(serde_json::json!({"messages": [{"role": "user", "content": "only"}]}));
        let codec = OpenaiChatCodec::new();
        let body = codec
            .build_body(&request_with_format(ToolCallProtocol::Native), &p)
            .unwrap();
        let messages = body["messages"].as_array().expect("messages");
        assert_eq!(
            messages.len(),
            1,
            "a custom body must not be concatenated with the built messages"
        );
        assert_eq!(messages[0]["content"], serde_json::json!("only"));
    }

    #[test]
    fn annotations_are_reported_as_metadata() {
        let codec = OpenaiChatCodec::new();
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"42","annotations":[{"type":"url_citation","url_citation":{"start_index":0,"end_index":2,"title":"T","url":"https://example.com"}}]},"finish_reason":"stop"}]}"#;
        let result = codec
            .parse_response(body, &request_with_format(ToolCallProtocol::Native))
            .expect("must parse");
        let metadata = result.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["annotations"][0]["url_citation"]["url"],
            serde_json::json!("https://example.com")
        );
    }

    #[test]
    fn full_endpoint_with_query_string_is_used_as_is() {
        let codec = OpenaiChatCodec::new();
        let mut p = profile();
        p.base_url = Some(
            "https://example.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-01"
                .to_string(),
        );
        let req = codec
            .build_request(&request_with_format(ToolCallProtocol::Native), &p)
            .expect("request must build");
        assert_eq!(
            req.url().as_str(),
            "https://example.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-01"
        );
    }
}
