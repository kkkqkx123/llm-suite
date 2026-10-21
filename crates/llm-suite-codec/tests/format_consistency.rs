//! Cross-format consistency checks.
//!
//! Every codec encodes the same conversation differently, but the semantics
//! the host observes have to match: the system content reaches the wire
//! through the channel the provider reserves for it, the tool call and its
//! result both survive the encoding, and the response envelope reports the
//! reasoning and its token count the same way everywhere.

use llm_suite_codec::{
    AnthropicCodec, GeminiNativeCodec, LlmCodec, OpenaiChatCodec, OpenaiResponseCodec,
};
use llm_suite_types::llm::{LlmFormat, LlmProfile, LlmRequest, LlmResult as LlmResponseType};
use llm_suite_types::message::{
    LlmFunctionCall, LlmToolCall, Message, MessageContentValue, MessageRole,
};
use std::sync::Arc;

const SYSTEM: &str = "You are a helper";

fn profile(format: LlmFormat, model: &str) -> LlmProfile {
    LlmProfile {
        id: "p1".to_string(),
        name: "test".to_string(),
        format,
        provider_id: None,
        model: model.to_string(),
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
    }
}

fn msg(role: MessageRole, text: &str) -> Message {
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

/// A conversation with a system prompt, a user turn, an assistant tool call
/// and the matching tool result.
fn conversation() -> LlmRequest {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.tool_calls = Some(vec![LlmToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        call_id: Some("call_1".to_string()),
        function: LlmFunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"city":"Beijing"}"#.to_string(),
        },
    }]);
    let mut result = msg(MessageRole::Tool, "sunny");
    result.tool_call_id = Some("call_1".to_string());

    LlmRequest {
        profile_id: "p1".to_string(),
        messages: vec![
            msg(MessageRole::System, SYSTEM),
            msg(MessageRole::User, "weather?"),
            assistant,
            result,
        ],
        parameters: None,
        generation: None,
        tools: Some(vec![llm_suite_types::tool::Tool {
            name: "get_weather".to_string(),
            description: "Get the weather".to_string(),
            parameters: None,
        }]),
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

fn body_of(request: &reqwest::Request) -> serde_json::Value {
    serde_json::from_slice(request.body().expect("body").as_bytes().unwrap())
        .expect("the body must be JSON")
}

fn codecs() -> Vec<(Arc<dyn LlmCodec>, LlmProfile)> {
    vec![
        (
            Arc::new(OpenaiChatCodec::new()),
            profile(LlmFormat::OpenaiChat, "gpt-4o"),
        ),
        (
            Arc::new(OpenaiResponseCodec::new()),
            profile(LlmFormat::OpenaiResponse, "gpt-4o"),
        ),
        (
            Arc::new(AnthropicCodec::new()),
            profile(LlmFormat::Anthropic, "claude-sonnet"),
        ),
        (
            Arc::new(GeminiNativeCodec::new()),
            profile(LlmFormat::GeminiNative, "gemini-2.5-pro"),
        ),
    ]
}

/// The system content reaches the wire exactly once, through the channel the
/// provider reserves for it.
#[test]
fn the_system_content_is_carried_once() {
    for (codec, profile) in codecs() {
        let request = codec
            .build_request(&conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);
        let text = serde_json::to_string(&body).expect("serializable");

        let occurrences = text.matches(SYSTEM).count();
        assert_eq!(
            occurrences, 1,
            "the system content must appear exactly once in {:?}: {body}",
            profile.format
        );

        match profile.format {
            LlmFormat::Anthropic => {
                assert_eq!(body["system"], serde_json::json!(SYSTEM));
                let roles: Vec<&str> = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .filter_map(|m| m["role"].as_str())
                    .collect();
                assert!(
                    !roles.contains(&"system"),
                    "the system message must not stay in the message list: {body}"
                );
            }
            LlmFormat::GeminiNative => {
                assert_eq!(
                    body["systemInstruction"]["parts"][0]["text"],
                    serde_json::json!(SYSTEM)
                );
                let roles: Vec<&str> = body["contents"]
                    .as_array()
                    .expect("contents")
                    .iter()
                    .filter_map(|c| c["role"].as_str())
                    .collect();
                assert!(
                    !roles.contains(&"system"),
                    "the system message must not stay in the contents: {body}"
                );
            }
            LlmFormat::OpenaiChat => {
                let systems: Vec<&serde_json::Value> = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .filter(|m| m["role"] == serde_json::json!("system"))
                    .collect();
                assert_eq!(systems.len(), 1, "{body}");
                assert_eq!(systems[0]["content"], serde_json::json!(SYSTEM));
            }
            LlmFormat::OpenaiResponse => {
                let systems: Vec<&serde_json::Value> = body["input"]
                    .as_array()
                    .expect("input")
                    .iter()
                    .filter(|m| m["role"] == serde_json::json!("system"))
                    .collect();
                assert_eq!(systems.len(), 1, "{body}");
                assert_eq!(systems[0]["content"], serde_json::json!(SYSTEM));
            }
            LlmFormat::Custom(name) => panic!("unexpected format {name}"),
        }
    }
}

/// The tool call and the tool result both reach the wire, correlated by the
/// identity the provider uses.
#[test]
fn the_tool_call_and_its_result_reach_the_wire() {
    for (codec, profile) in codecs() {
        let request = codec
            .build_request(&conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);
        let text = serde_json::to_string(&body).expect("serializable");

        assert!(
            text.contains("get_weather"),
            "the tool name must reach the wire for {:?}: {body}",
            profile.format
        );
        assert!(
            text.contains("sunny"),
            "the tool result must reach the wire for {:?}: {body}",
            profile.format
        );

        match profile.format {
            LlmFormat::OpenaiChat => {
                let call = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .find(|m| m.get("tool_calls").is_some())
                    .expect("the tool call must be present");
                assert_eq!(call["tool_calls"][0]["id"], serde_json::json!("call_1"));
                let result = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .find(|m| m["role"] == serde_json::json!("tool"))
                    .expect("the tool result must be present");
                assert_eq!(result["tool_call_id"], serde_json::json!("call_1"));
            }
            LlmFormat::OpenaiResponse => {
                let input = body["input"].as_array().expect("input");
                let call = input
                    .iter()
                    .find(|i| i["type"] == serde_json::json!("function_call"))
                    .expect("the function call must be present");
                assert_eq!(call["call_id"], serde_json::json!("call_1"));
                let result = input
                    .iter()
                    .find(|i| i["type"] == serde_json::json!("function_call_output"))
                    .expect("the function output must be present");
                assert_eq!(result["call_id"], serde_json::json!("call_1"));
                assert_eq!(result["output"], serde_json::json!("sunny"));
            }
            LlmFormat::Anthropic => {
                let messages = body["messages"].as_array().expect("messages");
                let call = messages
                    .iter()
                    .find(|m| {
                        m["content"]
                            .as_array()
                            .is_some_and(|blocks| blocks.iter().any(|b| b["type"] == "tool_use"))
                    })
                    .expect("the tool use block must be present");
                assert_eq!(call["content"][0]["id"], serde_json::json!("call_1"));
                let result = messages
                    .iter()
                    .find(|m| {
                        m["content"]
                            .as_array()
                            .is_some_and(|blocks| blocks.iter().any(|b| b["type"] == "tool_result"))
                    })
                    .expect("the tool result block must be present");
                assert_eq!(
                    result["content"][0]["tool_use_id"],
                    serde_json::json!("call_1")
                );
            }
            LlmFormat::GeminiNative => {
                let contents = body["contents"].as_array().expect("contents");
                let call = contents
                    .iter()
                    .find(|c| {
                        c["parts"].as_array().is_some_and(|parts| {
                            parts.iter().any(|p| p.get("functionCall").is_some())
                        })
                    })
                    .expect("the function call part must be present");
                assert_eq!(
                    call["parts"][0]["functionCall"]["name"],
                    serde_json::json!("get_weather")
                );
                let result = contents
                    .iter()
                    .find(|c| {
                        c["parts"].as_array().is_some_and(|parts| {
                            parts.iter().any(|p| p.get("functionResponse").is_some())
                        })
                    })
                    .expect("the function response part must be present");
                assert_eq!(
                    result["parts"][0]["functionResponse"]["name"],
                    serde_json::json!("get_weather")
                );
                assert!(
                    result["parts"][0]["functionResponse"]["response"].is_object(),
                    "the function response payload is an object: {body}"
                );
            }
            LlmFormat::Custom(name) => panic!("unexpected format {name}"),
        }
    }
}

/// A tool without a parameter schema is declared with an empty object schema
/// everywhere; no codec sends a null schema.
#[test]
fn a_tool_without_parameters_declares_an_object_schema() {
    for (codec, profile) in codecs() {
        let request = codec
            .build_request(&conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);
        let text = serde_json::to_string(&body).expect("serializable");
        assert!(
            !text.contains("\"parameters\":null"),
            "a null schema is rejected by the providers: {body}"
        );
    }
}

/// The response envelope reports the reasoning and its token count the same
/// way for every provider.
#[test]
fn the_response_envelope_reports_the_reasoning_the_same_way() {
    let request = LlmRequest {
        profile_id: "p1".to_string(),
        messages: Vec::new(),
        parameters: None,
        generation: None,
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
    };

    let bodies = [
        (
            LlmFormat::OpenaiChat,
            r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi","reasoning_content":"pondering"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"completion_tokens_details":{"reasoning_tokens":4}}}"#,
        ),
        (
            LlmFormat::OpenaiResponse,
            r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"pondering"}]},{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3,"output_tokens_details":{"reasoning_tokens":4}}}"#,
        ),
        (
            LlmFormat::Anthropic,
            r#"{"id":"msg_1","model":"claude","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"pondering","signature":"sig"},{"type":"text","text":"hi"}],"usage":{"input_tokens":1,"output_tokens":2,"output_tokens_details":{"thinking_tokens":4}}}"#,
        ),
        (
            LlmFormat::GeminiNative,
            r#"{"responseId":"resp_1","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"text":"pondering","thought":true},{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3,"thoughtsTokenCount":4}}"#,
        ),
    ];

    for (format, body) in bodies {
        let codec = match format {
            LlmFormat::OpenaiChat => Arc::new(OpenaiChatCodec::new()) as Arc<dyn LlmCodec>,
            LlmFormat::OpenaiResponse => Arc::new(OpenaiResponseCodec::new()),
            LlmFormat::Anthropic => Arc::new(AnthropicCodec::new()),
            LlmFormat::GeminiNative => Arc::new(GeminiNativeCodec::new()),
            LlmFormat::Custom(name) => panic!("unexpected format {name}"),
        };
        let result: LlmResponseType = codec.parse_response(body, &request).expect("must parse");

        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("pondering"),
            "{format:?} must report the reasoning"
        );
        assert_eq!(
            result.message.thinking.as_deref(),
            Some("pondering"),
            "{format:?} must carry the reasoning on the message as well"
        );
        assert_eq!(
            result.reasoning_tokens,
            Some(4),
            "{format:?} must report the reasoning tokens"
        );
        assert_eq!(
            result.usage.as_ref().and_then(|u| u.reasoning_tokens),
            Some(4),
            "{format:?} must report the reasoning tokens in the usage as well"
        );
    }
}
