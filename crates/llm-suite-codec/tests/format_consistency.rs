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
    FileContent, FileSource, ImageUrlContent, LlmFunctionCall, LlmToolCall, Message,
    MessageContent, MessageContentValue, MessageRole, ToolResultContent, ToolUseContent,
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
        thought_signature: None,
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
                // The Responses API reserves `instructions` for the system
                // prompt; it is not an input item.
                assert_eq!(body["instructions"], serde_json::json!(SYSTEM));
                let roles: Vec<&str> = body["input"]
                    .as_array()
                    .expect("input")
                    .iter()
                    .filter_map(|m| m["role"].as_str())
                    .collect();
                assert!(
                    !roles.contains(&"system"),
                    "the system message must not stay in the input: {body}"
                );
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

/// A request without messages; used by the checks that only look at the
/// response side.
fn empty_request() -> LlmRequest {
    LlmRequest {
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
    }
}

/// The codec of a format, without a profile.
fn codec_of(format: &LlmFormat) -> Arc<dyn LlmCodec> {
    match format {
        LlmFormat::OpenaiChat => Arc::new(OpenaiChatCodec::new()),
        LlmFormat::OpenaiResponse => Arc::new(OpenaiResponseCodec::new()),
        LlmFormat::Anthropic => Arc::new(AnthropicCodec::new()),
        LlmFormat::GeminiNative => Arc::new(GeminiNativeCodec::new()),
        LlmFormat::Custom(name) => panic!("unexpected format {name}"),
    }
}

/// The OpenAI formats and Gemini report the same stop vocabulary; Anthropic
/// reports the official `stop_reason` set of its own.
#[test]
fn the_stop_vocabulary_is_the_same_for_every_format() {
    let bodies = [
        (
            LlmFormat::OpenaiChat,
            r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
            Some("tool_calls"),
        ),
        (
            LlmFormat::OpenaiResponse,
            r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":"{}"}]}"#,
            Some("tool_calls"),
        ),
        (
            LlmFormat::GeminiNative,
            r#"{"responseId":"resp_1","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}"#,
            Some("tool_calls"),
        ),
        // The content filter is the same value everywhere.
        (
            LlmFormat::OpenaiChat,
            r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null},"finish_reason":"content_filter"}]}"#,
            Some("content_filter"),
        ),
        (
            LlmFormat::OpenaiResponse,
            r#"{"id":"resp_1","model":"gpt-4o","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}"#,
            Some("content_filter"),
        ),
        (
            LlmFormat::GeminiNative,
            r#"{"responseId":"resp_1","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"IMAGE_SAFETY"}]}"#,
            Some("content_filter"),
        ),
    ];

    for (format, body, expected) in bodies {
        let result = codec_of(&format)
            .parse_response(body, &empty_request())
            .expect("must parse");
        assert_eq!(
            result.finish_reason.as_deref(),
            expected,
            "{format:?} must report {expected:?}"
        );
    }
}

/// A refusal is reported on the same channel by every format that has one.
#[test]
fn the_refusal_is_reported_on_the_same_channel() {
    let bodies = [
        (
            LlmFormat::OpenaiChat,
            r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null,"refusal":"I cannot help with that."},"finish_reason":"stop"}]}"#,
        ),
        (
            LlmFormat::OpenaiResponse,
            r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"I cannot help with that."}]}]}"#,
        ),
        (
            LlmFormat::Anthropic,
            r#"{"id":"msg_1","model":"claude","stop_reason":"refusal","content":[],"stop_details":{"type":"refusal","category":"general_harms","explanation":"I cannot help with that."}}"#,
        ),
    ];

    for (format, body) in bodies {
        let result = codec_of(&format)
            .parse_response(body, &empty_request())
            .expect("must parse");
        assert_eq!(
            result.refusal.as_deref(),
            Some("I cannot help with that."),
            "{format:?} must report the refusal"
        );
    }
}

/// The sources of a grounded answer are reported under the same metadata key
/// by every format that has them.
#[test]
fn the_sources_of_a_grounded_answer_share_one_metadata_key() {
    let bodies = [
        (
            LlmFormat::OpenaiChat,
            r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"grounded","annotations":[{"type":"url_citation","url":"https://example.com"}]},"finish_reason":"stop"}]}"#,
        ),
        (
            LlmFormat::OpenaiResponse,
            r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"grounded","annotations":[{"type":"url_citation","url":"https://example.com"}]}]}]}"#,
        ),
        (
            LlmFormat::Anthropic,
            r#"{"id":"msg_1","model":"claude","stop_reason":"end_turn","content":[{"type":"text","text":"grounded","citations":[{"type":"char_location","cited_text":"x"}]}]}"#,
        ),
    ];

    for (format, body) in bodies {
        let result = codec_of(&format)
            .parse_response(body, &empty_request())
            .expect("must parse");
        let metadata = result.metadata.expect("metadata must be present");
        let annotations = metadata
            .get("annotations")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("{format:?} must report the sources"));
        assert_eq!(annotations.len(), 1, "{format:?}");
    }
}

/// The token counters carry the same meaning for every format, read from the
/// official usage object of each provider.
///
/// The fixtures describe the same request: 100 input tokens of which 80 were
/// served from the cache, 20 output tokens of which 6 were spent on
/// reasoning, 120 tokens in total.
#[test]
fn the_token_counters_carry_the_same_meaning() {
    let bodies = [
        (
            LlmFormat::OpenaiChat,
            r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"completion_tokens_details":{"reasoning_tokens":6},"prompt_tokens_details":{"cached_tokens":80}}}"#,
        ),
        (
            LlmFormat::OpenaiResponse,
            r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,"output_tokens_details":{"reasoning_tokens":6},"input_tokens_details":{"cached_tokens":80}}}"#,
        ),
        (
            LlmFormat::Anthropic,
            r#"{"id":"msg_1","model":"claude","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":20,"output_tokens":20,"output_tokens_details":{"thinking_tokens":6},"cache_read_input_tokens":80}}"#,
        ),
        (
            LlmFormat::GeminiNative,
            r#"{"responseId":"resp_1","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"totalTokenCount":120,"thoughtsTokenCount":6,"cachedContentTokenCount":80}}"#,
        ),
    ];

    for (format, body) in bodies {
        let result = codec_of(&format)
            .parse_response(body, &empty_request())
            .expect("must parse");
        let usage = result.usage.expect("usage must be present");
        assert_eq!(usage.completion_tokens, 20, "{format:?}");
        assert_eq!(usage.total_tokens, 120, "{format:?}");
        assert_eq!(usage.reasoning_tokens, Some(6), "{format:?}");
        assert_eq!(usage.cache_read_tokens, Some(80), "{format:?}");
        // `prompt_tokens` keeps the scope of the provider: the OpenAI
        // family counts the cached input into the prompt, Anthropic
        // reports it next to it. The pricing layer resolves the
        // difference through `CacheTokenScope`.
        let prompt = if matches!(format, LlmFormat::Anthropic) {
            20
        } else {
            100
        };
        assert_eq!(usage.prompt_tokens, prompt, "{format:?}");
    }
}

/// Every format names a function tool with the shape its API defines.
#[test]
fn the_tool_choice_uses_the_shape_of_its_api() {
    use llm_suite_types::llm::generation::{LlmGenerationParams, LlmToolChoice};

    let generation = LlmGenerationParams {
        tool_choice: Some(LlmToolChoice {
            mode: None,
            tool_name: Some("get_weather".to_string()),
        }),
        ..Default::default()
    };

    for (codec, mut profile) in codecs() {
        let format = profile.format.clone();
        profile.generation = Some(generation.clone());
        let request = codec
            .build_request(&conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);

        match format {
            LlmFormat::OpenaiChat => {
                // The Chat API nests the name under `function`.
                assert_eq!(
                    body["tool_choice"]["type"],
                    serde_json::json!("function"),
                    "{body}"
                );
                assert_eq!(
                    body["tool_choice"]["function"]["name"],
                    serde_json::json!("get_weather"),
                    "{body}"
                );
            }
            LlmFormat::OpenaiResponse => {
                // The Responses API names the function flat.
                assert_eq!(
                    body["tool_choice"]["type"],
                    serde_json::json!("function"),
                    "{body}"
                );
                assert_eq!(
                    body["tool_choice"]["name"],
                    serde_json::json!("get_weather"),
                    "{body}"
                );
                assert!(body["tool_choice"].get("function").is_none(), "{body}");
            }
            LlmFormat::Anthropic => {
                assert_eq!(
                    body["tool_choice"]["type"],
                    serde_json::json!("tool"),
                    "{body}"
                );
                assert_eq!(
                    body["tool_choice"]["name"],
                    serde_json::json!("get_weather"),
                    "{body}"
                );
            }
            LlmFormat::GeminiNative => {
                let config = &body["toolConfig"]["functionCallingConfig"];
                assert_eq!(config["mode"], serde_json::json!("ANY"), "{body}");
                assert_eq!(
                    config["allowedFunctionNames"],
                    serde_json::json!(["get_weather"]),
                    "{body}"
                );
            }
            LlmFormat::Custom(name) => panic!("unexpected format {name}"),
        }
    }
}

/// Every format that offers a counting endpoint uses the official one and
/// reads the official response field.
#[test]
fn the_count_tokens_endpoints_match_the_official_apis() {
    for (codec, profile) in codecs() {
        let format = profile.format.clone();
        let request = codec
            .build_count_tokens_request(&conversation(), &profile)
            .expect("must not fail");
        if matches!(format, LlmFormat::OpenaiChat) {
            assert!(
                request.is_none(),
                "the Chat Completions API has no counting endpoint"
            );
            continue;
        }
        let request = request.expect("the format has a counting endpoint");
        let url = request.url().to_string();
        match format {
            LlmFormat::Anthropic => assert!(url.ends_with("/messages/count_tokens"), "{url}"),
            LlmFormat::OpenaiResponse => assert!(url.ends_with("/responses/input_tokens"), "{url}"),
            LlmFormat::GeminiNative => assert!(url.ends_with(":countTokens"), "{url}"),
            _ => {}
        }
    }
}

/// The counting response is read from the field the provider uses.
#[test]
fn the_count_tokens_response_is_read_from_the_official_field() {
    let anthropic = AnthropicCodec::new();
    let body: serde_json::Value = serde_json::from_str(r#"{"input_tokens":13}"#).unwrap();
    assert_eq!(anthropic.parse_count_tokens_response(&body).unwrap(), 13);

    let responses = OpenaiResponseCodec::new();
    let body: serde_json::Value =
        serde_json::from_str(r#"{"object":"response.input_tokens","input_tokens":13}"#).unwrap();
    assert_eq!(responses.parse_count_tokens_response(&body).unwrap(), 13);

    let gemini = GeminiNativeCodec::new();
    let body: serde_json::Value = serde_json::from_str(r#"{"totalTokens":13}"#).unwrap();
    assert_eq!(gemini.parse_count_tokens_response(&body).unwrap(), 13);
}

/// Untyped provider options reach the request body in every format.
#[test]
fn the_untyped_options_reach_the_request_body() {
    let cases = [
        (LlmFormat::OpenaiChat, "logprobs"),
        (LlmFormat::OpenaiResponse, "truncation"),
        (LlmFormat::Anthropic, "container"),
        (LlmFormat::GeminiNative, "safetySettings"),
    ];

    for (format, key) in cases {
        let mut profile = profile(format.clone(), "model-x");
        profile.parameters = Some(serde_json::json!({ key: "value" }));
        let request = codec_of(&format)
            .build_request(&conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);
        assert!(
            body.get(key).is_some(),
            "{format:?} must forward the untyped option `{key}`: {body}"
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

/// A conversation whose messages carry every kind of content the internal
/// model knows: text, an image, a document, a tool call and the rich result
/// that answers it.
fn multimodal_conversation() -> LlmRequest {
    let mut assistant = msg(MessageRole::Assistant, "");
    assistant.content = MessageContentValue::Rich(vec![
        MessageContent::Text {
            text: "let me look".to_string(),
        },
        MessageContent::ToolUse {
            tool_use: ToolUseContent {
                id: "call_1".to_string(),
                name: "screenshot".to_string(),
                input: serde_json::json!({}),
                thought_signature: None,
            },
        },
    ]);
    let mut result = msg(MessageRole::Tool, "attached");
    result.tool_call_id = Some("call_1".to_string());
    result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
        tool_result: ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "attached".to_string(),
            blocks: Some(vec![MessageContent::ImageUrl {
                image_url: ImageUrlContent {
                    url: "data:image/png;base64,QUJD".to_string(),
                    detail: None,
                    file_id: None,
                },
            }]),
            is_error: None,
        },
    }]);
    let user = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::User,
        content: MessageContentValue::Rich(vec![
            MessageContent::Text {
                text: "what is this".to_string(),
            },
            MessageContent::ImageUrl {
                image_url: ImageUrlContent {
                    url: "data:image/png;base64,QUJD".to_string(),
                    detail: None,
                    file_id: None,
                },
            },
            MessageContent::FileInput {
                file: FileContent {
                    source: FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "REVG".to_string(),
                    },
                    filename: Some("report.pdf".to_string()),
                    detail: None,
                    citations_enabled: None,
                    context: None,
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
    };

    let mut request = empty_request();
    request.messages = vec![user, assistant, result];
    request
}

/// Every format keeps the picture of the user turn and of the tool result.
#[test]
fn the_multimodal_content_survives_every_format() {
    for (codec, profile) in codecs() {
        let request = codec
            .build_request(&multimodal_conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);
        let text = serde_json::to_string(&body).expect("serializable");

        assert!(
            text.contains("QUJD"),
            "the image payload must reach the wire for {:?}: {body}",
            profile.format
        );
        assert!(
            text.contains("REVG"),
            "the document must reach the wire for {:?}: {body}",
            profile.format
        );
        assert!(
            text.contains("screenshot"),
            "the tool call must reach the wire for {:?}: {body}",
            profile.format
        );
        assert!(
            text.contains("attached"),
            "the tool result must reach the wire for {:?}: {body}",
            profile.format
        );
    }
}

/// The tool call stays a structured call in every format; none of them
/// degrades it into prose.
#[test]
fn the_tool_call_stays_structured_in_every_format() {
    for (codec, profile) in codecs() {
        let request = codec
            .build_request(&multimodal_conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);

        match profile.format {
            LlmFormat::OpenaiChat => {
                let call = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .find(|m| m.get("tool_calls").is_some())
                    .expect("the tool call must be present");
                assert_eq!(call["tool_calls"][0]["function"]["name"], "screenshot");
            }
            LlmFormat::OpenaiResponse => {
                let call = body["input"]
                    .as_array()
                    .expect("input")
                    .iter()
                    .find(|i| i["type"] == serde_json::json!("function_call"))
                    .expect("the function call must be present");
                assert_eq!(call["name"], "screenshot");
                assert_eq!(call["call_id"], serde_json::json!("call_1"));
            }
            LlmFormat::Anthropic => {
                let call = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .find_map(|m| {
                        m["content"].as_array().and_then(|blocks| {
                            blocks
                                .iter()
                                .find(|b| b["type"] == serde_json::json!("tool_use"))
                                .cloned()
                        })
                    })
                    .expect("the tool use block must be present");
                assert_eq!(call["name"], "screenshot");
            }
            LlmFormat::GeminiNative => {
                let call = body["contents"]
                    .as_array()
                    .expect("contents")
                    .iter()
                    .find_map(|c| {
                        c["parts"].as_array().and_then(|parts| {
                            parts
                                .iter()
                                .find(|p| p.get("functionCall").is_some())
                                .cloned()
                        })
                    })
                    .expect("the function call part must be present");
                assert_eq!(call["functionCall"]["name"], "screenshot");
            }
            LlmFormat::Custom(name) => panic!("unexpected format {name}"),
        }
    }
}

/// The tool result stays correlated with the call it answers in every
/// format.
#[test]
fn the_tool_result_stays_correlated_in_every_format() {
    for (codec, profile) in codecs() {
        let request = codec
            .build_request(&multimodal_conversation(), &profile)
            .expect("must build");
        let body = body_of(&request);

        match profile.format {
            LlmFormat::OpenaiChat => {
                let result = body["messages"]
                    .as_array()
                    .expect("messages")
                    .iter()
                    .find(|m| m["role"] == serde_json::json!("tool"))
                    .expect("the tool message must be present");
                assert_eq!(result["tool_call_id"], serde_json::json!("call_1"));
            }
            LlmFormat::OpenaiResponse => {
                let result = body["input"]
                    .as_array()
                    .expect("input")
                    .iter()
                    .find(|i| i["type"] == serde_json::json!("function_call_output"))
                    .expect("the function call output must be present");
                assert_eq!(result["call_id"], serde_json::json!("call_1"));
            }
            LlmFormat::Anthropic => {
                let result = body["messages"]
                    .as_array()
                    .expect("messages")
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
                let result = body["contents"]
                    .as_array()
                    .expect("contents")
                    .iter()
                    .find(|c| {
                        c["parts"].as_array().is_some_and(|parts| {
                            parts.iter().any(|p| p.get("functionResponse").is_some())
                        })
                    })
                    .expect("the function response part must be present");
                assert_eq!(
                    result["parts"][0]["functionResponse"]["name"],
                    serde_json::json!("screenshot")
                );
            }
            LlmFormat::Custom(name) => panic!("unexpected format {name}"),
        }
    }
}

/// A system prompt that carries more than text is refused by every format;
/// none of them drops the content silently.
#[test]
fn a_rich_system_prompt_is_refused_by_every_format() {
    for (codec, profile) in codecs() {
        let mut request = empty_request();
        request.messages = vec![Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::System,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: ImageUrlContent {
                    url: "data:image/png;base64,QUJD".to_string(),
                    detail: None,
                    file_id: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }];
        let err = codec
            .build_request(&request, &profile)
            .expect_err("the system prompt is text only");
        assert!(
            format!("{err}").contains("system"),
            "{:?} must report the system prompt: {err}",
            profile.format
        );
    }
}

/// The text of a tool result reaches the wire exactly once, whichever
/// format carries it.
#[test]
fn the_text_of_a_tool_result_reaches_the_wire_once() {
    for (codec, profile) in codecs() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.content = MessageContentValue::Rich(vec![MessageContent::ToolUse {
            tool_use: ToolUseContent {
                id: "call_1".to_string(),
                name: "screenshot".to_string(),
                input: serde_json::json!({}),
                thought_signature: None,
            },
        }]);
        let mut result = msg(MessageRole::Tool, "");
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "attached".to_string(),
                blocks: Some(vec![MessageContent::Text {
                    text: "attached".to_string(),
                }]),
                is_error: None,
            },
        }]);

        let mut request = empty_request();
        request.messages = vec![assistant, result];
        let built = codec
            .build_request(&request, &profile)
            .expect("must build");
        let body = body_of(&built);
        let text = serde_json::to_string(&body).expect("serializable");
        let occurrences = text.matches("attached").count();
        assert_eq!(
            occurrences, 1,
            "the result text must not be repeated for {:?}: {body}",
            profile.format
        );
    }
}

/// A tool result that carries a block the format cannot hold is reported
/// by every format; none of them drops it silently.
#[test]
fn a_tool_result_with_an_unsupported_block_is_refused_by_every_format() {
    for (codec, profile) in codecs() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.content = MessageContentValue::Rich(vec![MessageContent::ToolUse {
            tool_use: ToolUseContent {
                id: "call_1".to_string(),
                name: "screenshot".to_string(),
                input: serde_json::json!({}),
                thought_signature: None,
            },
        }]);
        let mut result = msg(MessageRole::Tool, "");
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "attached".to_string(),
                blocks: Some(vec![MessageContent::Thinking {
                    thinking: "let me verify".to_string(),
                    signature: Some("sig".to_string()),
                }]),
                is_error: None,
            },
        }]);

        let mut request = empty_request();
        request.messages = vec![assistant, result];
        let err = codec
            .build_request(&request, &profile)
            .expect_err("a block the format cannot hold must be reported");
        assert!(
            format!("{err}").contains("thinking"),
            "{:?} must name the block it cannot carry: {err}",
            profile.format
        );
    }
}
