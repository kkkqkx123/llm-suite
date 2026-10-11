use crate::error::LlmResult;
use llm_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent, ToolCallProtocol,
};
use llm_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// Whether the request runs in text-based tool mode (non-native tool protocol).
pub fn is_text_mode(request: &LlmRequest) -> bool {
    matches!(
        request.tool_call_protocol,
        Some(ref f) if *f != ToolCallProtocol::Native
    )
}

/// Effective tool call protocol: the locked protocol wins, otherwise the
/// request protocol, otherwise native.
pub fn effective_tool_call_protocol(request: &LlmRequest) -> ToolCallProtocol {
    request
        .locked_tool_call_protocol
        .as_ref()
        .map(|c| c.format.clone())
        .or_else(|| request.tool_call_protocol.clone())
        .unwrap_or(ToolCallProtocol::Native)
}

/// Convert message history to the text-based tool protocol when the request
/// runs in text mode (native tool calls / results become XML or JSON blocks
/// in the message content). Messages are returned unchanged in native mode.
pub fn convert_history_for_text_mode(messages: &[Message], request: &LlmRequest) -> Vec<Message> {
    if !is_text_mode(request) {
        return messages.to_vec();
    }
    let protocol = effective_tool_call_protocol(request);
    let markers = request
        .locked_tool_call_protocol
        .as_ref()
        .and_then(|c| c.markers.clone());
    llm_message::history_converter::convert_to_text_mode(messages, &protocol, markers.as_ref())
}

/// Build the system prompt content for text-based tool mode: existing system
/// message + tool usage instructions + tool declarations.
pub fn text_mode_system_content(request: &LlmRequest) -> String {
    use llm_tool_call::tool::protocol::{build_text_mode_system_content, extract_system_message};
    let protocol = effective_tool_call_protocol(request);
    let (system, _) = extract_system_message(&request.messages);
    let tools = request.tools.as_deref().unwrap_or(&[]);
    build_text_mode_system_content(system.as_deref().unwrap_or(""), tools, protocol, false)
}

/// Parse tool calls from text content in text-based tool mode.
pub fn parse_text_tool_calls(
    request: &LlmRequest,
    content: &str,
) -> Vec<llm_types::message::LlmToolCall> {
    use llm_tool_call::tool::parser::parse_from_text;
    use llm_tool_call::tool::protocol::get_tool_call_parser_options;
    let protocol = effective_tool_call_protocol(request);
    if protocol == ToolCallProtocol::Native {
        return Vec::new();
    }
    let markers = request
        .locked_tool_call_protocol
        .as_ref()
        .and_then(|c| c.markers.clone());
    let options = get_tool_call_parser_options(protocol, markers.as_ref());
    parse_from_text(content, &options)
}

/// Convert internal messages into OpenAI Chat Completions `messages`.
///
/// OpenAI Chat has no `tool_use` / `tool_result` content-part types (those are
/// Anthropic's). Tool calls therefore flow through the message-level
/// `tool_calls` field, and tool results must be standalone `tool` role
/// messages. This converter folds `ToolUse` rich blocks into `tool_calls` and
/// emits `ToolResult` rich blocks as separate `tool` messages appended right
/// after their source message (the canonical OpenAI ordering).
pub fn convert_openai_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();

    for msg in messages {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };

        // Native tool calls already carried on the message (canonical form).
        let mut tool_calls: Vec<serde_json::Value> = Vec::new();
        if let Some(ref tcs) = msg.tool_calls {
            for tc in tcs {
                tool_calls.push(serde_json::json!({
                    "id": tc.id,
                    "type": tc.r#type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }));
            }
        }

        // `ToolResult` rich blocks cannot be content parts; they become
        // standalone `tool` role messages emitted after this entry.
        let mut tool_result_entries: Vec<serde_json::Value> = Vec::new();
        // Tracks whether the source message carries any non-tool-result content.
        let mut has_content = false;

        let mut entry = match &msg.content {
            MessageContentValue::Text(text) => {
                has_content = true;
                serde_json::json!({"role": role, "content": text})
            }
            MessageContentValue::Rich(blocks) => {
                let mut content: Vec<serde_json::Value> = Vec::new();
                for block in blocks {
                    match block {
                        MessageContent::Text { text } => {
                            has_content = true;
                            content.push(serde_json::json!({"type": "text", "text": text}));
                        }
                        MessageContent::ImageUrl { image_url } => {
                            has_content = true;
                            let mut img = serde_json::json!({
                                "type": "image_url",
                                "image_url": {"url": image_url.url}
                            });
                            if let Some(ref detail) = image_url.detail {
                                img["image_url"]["detail"] = serde_json::json!(detail);
                            }
                            content.push(img);
                        }
                        // Fold ToolUse blocks into the message's `tool_calls`.
                        MessageContent::ToolUse { tool_use } => {
                            tool_calls.push(serde_json::json!({
                                "id": tool_use.id,
                                "type": "function",
                                "function": {
                                    "name": tool_use.name,
                                    "input": tool_use.input,
                                }
                            }));
                        }
                        // Emit ToolResult blocks as standalone `tool` messages.
                        MessageContent::ToolResult { tool_result } => {
                            let mut tmsg = serde_json::json!({
                                "role": "tool",
                                "content": tool_result.content,
                                "tool_call_id": tool_result.tool_use_id,
                            });
                            if let Some(is_err) = tool_result.is_error {
                                tmsg["is_error"] = serde_json::json!(is_err);
                            }
                            tool_result_entries.push(tmsg);
                        }
                        MessageContent::Thinking { thinking, .. } => {
                            has_content = true;
                            content.push(serde_json::json!({"type": "text", "text": thinking}));
                        }
                    }
                }
                serde_json::json!({"role": role, "content": content})
            }
        };

        if !tool_calls.is_empty() {
            entry["tool_calls"] = serde_json::json!(tool_calls);
        }
        if let Some(ref tool_call_id) = msg.tool_call_id {
            entry["tool_call_id"] = serde_json::json!(tool_call_id);
        }

        // A `tool` role message whose only content was a ToolResult block has
        // no valid standalone body of its own; skip it and rely on the derived
        // `tool` messages above.
        if !(msg.role == MessageRole::Tool && !has_content && !tool_result_entries.is_empty()) {
            out.push(entry);
        }
        out.extend(tool_result_entries);
    }

    out
}

/// OpenAI Chat requires every `tool` role message to carry `tool_call_id` (and
/// to follow the assistant message that produced the matching `tool_calls`).
/// Validate the former so an invalid request fails fast locally instead of
/// round-tripping a `400` from the server. Text mode re-renders tool results as
/// inline blocks, so this only applies to native (non-text) requests.
pub fn validate_openai_chat_messages(request: &LlmRequest) -> LlmResult<()> {
    for msg in &request.messages {
        if msg.role == llm_types::message::MessageRole::Tool && msg.tool_call_id.is_none() {
            return Err(crate::error::LlmError::ConfigError(
                "OpenAI Chat requires every `tool` role message to carry `tool_call_id`"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

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
) -> crate::error::LlmResult<llm_types::llm::generation::LlmGenerationParams> {
    crate::generation::resolve_generation(profile, request)
}

/// Parse an OpenAI-style usage object into token usage stats.
pub fn parse_openai_usage(u: &serde_json::Value) -> llm_types::llm::TokenUsageStats {
    let reasoning_tokens = u
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|r| r.as_u64())
        .map(|r| r as u32);
    let cache_read_tokens = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|r| r.as_u64())
        .map(|r| r as u32);
    llm_types::llm::TokenUsageStats {
        prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        completion_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens: None,
        prompt_tokens_cost: None,
        completion_tokens_cost: None,
        total_cost: None,
    }
}

pub fn parse_openai_chat_response(body: &str) -> LlmResult<LlmResponseType> {
    let json: serde_json::Value = serde_json::from_str(body)?;

    let id = json.get("id").and_then(|v| v.as_str()).map(String::from);
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let empty_choices = vec![];
    let choices = json
        .get("choices")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_choices);
    let first_choice = choices.first();
    let message = first_choice.and_then(|c| c.get("message"));

    let content = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(String::from);

    // Provider-native value passed through untouched (`stop` / `tool_calls`
    // / `length` / `content_filter`); never normalized here.
    let finish_reason = first_choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str())
        .map(String::from);

    // NOTE: `reasoning_content` is NOT part of the official OpenAI Chat
    // Completions schema (it is absent from `openai-python`). It is a
    // non-standard extension used by DeepSeek and several Chinese
    // OpenAI-compatible gateways to surface intermediate reasoning. Official
    // OpenAI reasoning models do not expose it here, so strict OpenAI
    // responses will leave this empty.
    let reasoning_content = message
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|r| r.as_str())
        .map(String::from);

    let tool_calls = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let id = tc.get("id")?.as_str()?.to_string();
                    let function = tc.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    let arguments = function.get("arguments")?.as_str()?.to_string();
                    Some(llm_types::message::LlmToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: llm_types::message::LlmFunctionCall { name, arguments },
                    })
                })
                .collect()
        });

    let usage = json.get("usage").map(parse_openai_usage);

    let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

    let msg = llm_types::message::Message {
        id: llm_types::Id::new(),
        role: MessageRole::Assistant,
        content: MessageContentValue::Text(content.clone().unwrap_or_default()),
        timestamp: llm_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: tool_calls.clone(),
        thinking: reasoning_content.clone(),
        thinking_signature: None,
        metadata: None,
    };

    let mut metadata = std::collections::HashMap::new();
    if let Some(created) = json.get("created").and_then(|v| v.as_i64()) {
        metadata.insert("created".to_string(), serde_json::json!(created));
    }
    if let Some(fp) = json.get("system_fingerprint").and_then(|v| v.as_str()) {
        metadata.insert("system_fingerprint".to_string(), serde_json::json!(fp));
    }
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };

    Ok(LlmResponseType {
        id,
        model,
        content,
        message: msg,
        tool_calls,
        usage,
        finish_reason,
        duration: 0,
        reasoning_content,
        reasoning_signature: None,
        reasoning_tokens,
        metadata,
        stream_stats: None,
    })
}

/// Parse one OpenAI Chat streaming SSE chunk into zero or more logical events.
///
/// Returns a `Vec` (not a single `Option`) because a chunk's
/// `choices[].delta.tool_calls` array may carry several tool calls at once
/// (parallel tool calls). Each entry yields its own `ToolCallDelta` so none
/// are dropped; the accumulator keys them by `index`.
pub fn parse_openai_stream_chunk(data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
    if data == "[DONE]" {
        return Ok(vec![MessageStreamEvent::End(
            llm_types::llm::MessageStreamEnd {},
        )]);
    }

    let json: serde_json::Value = serde_json::from_str(data)?;

    let mut events = Vec::new();

    // Provider stream identity rides on every chunk; surface it first so the
    // accumulator records it before any terminal event in the same chunk.
    let stream_id = json.get("id").and_then(|v| v.as_str()).map(String::from);
    let stream_model = json.get("model").and_then(|v| v.as_str()).map(String::from);
    if stream_id.is_some() || stream_model.is_some() {
        events.push(MessageStreamEvent::Start(
            llm_types::llm::MessageStreamStart {
                id: stream_id,
                model: stream_model,
            },
        ));
    }

    let empty_choices = vec![];
    let choices = json
        .get("choices")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_choices);

    // A chunk without choices carries only usage (stream_options.include_usage).
    if choices.is_empty() {
        if let Some(usage) = json.get("usage") {
            events.push(MessageStreamEvent::Usage(
                llm_types::llm::MessageStreamUsage {
                    usage: parse_openai_usage(usage),
                },
            ));
        }
        return Ok(events);
    }

    if let Some(choice) = choices.first() {
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                events.push(MessageStreamEvent::Text(
                    llm_types::llm::MessageStreamText {
                        snapshot: String::new(),
                        text: content.to_string(),
                    },
                ));
            }
            // `reasoning_content` is a non-standard extension (DeepSeek /
            // Chinese gateways), not an official OpenAI Chat field; see the
            // note in `parse_openai_chat_response`.
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                events.push(MessageStreamEvent::ReasoningText(
                    llm_types::llm::MessageStreamReasoning {
                        snapshot: String::new(),
                        reasoning: reasoning.to_string(),
                        signature: None,
                        signature_snapshot: None,
                    },
                ));
            }
            // Emit one ToolCallDelta per tool call so parallel calls packed
            // into a single delta array are all forwarded (previously only the
            // first was kept).
            if let Some(trait_tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for call in trait_tool_calls {
                    let index = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let id = call.get("id").and_then(|v| v.as_str()).map(String::from);
                    let function = call.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let arguments = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    events.push(MessageStreamEvent::ToolCallDelta(
                        llm_types::llm::MessageStreamToolCallDelta {
                            index,
                            id,
                            name,
                            arguments,
                            is_snapshot: false,
                        },
                    ));
                }
            }
        }
        // Provider-native terminal values passed through untouched; any other
        // value (including a missing one) leaves the stream open.
        if let Some(finish_reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            let is_done = matches!(
                finish_reason,
                "stop" | "tool_calls" | "length" | "content_filter"
            );
            if is_done {
                events.push(MessageStreamEvent::End(llm_types::llm::MessageStreamEnd {}));
            }
        }
    }

    Ok(events)
}

pub fn convert_openai_tools(tools: &[llm_types::tool::Tool]) -> LlmResult<Vec<serde_json::Value>> {
    let tool_defs: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect();
    Ok(tool_defs)
}

/// Apply authentication, custom headers and query parameters to the request.
///
/// Auth type resolution: `profile.auth_type` wins when set ("native" /
/// "bearer" / "x-api-key" / "x-goog-api-key"), otherwise `default_auth` is
/// used. "native" maps to the format-specific header (x-api-key for Anthropic,
/// x-goog-api-key for Gemini native, Bearer otherwise).
pub fn apply_auth_and_headers(
    req_builder: reqwest::RequestBuilder,
    profile: &LlmProfile,
    default_auth: &str,
) -> reqwest::RequestBuilder {
    let mut builder = req_builder;

    if let Some(api_key) = &profile.api_key {
        let auth_type = profile.auth_type.as_deref().unwrap_or(default_auth);
        match auth_type {
            "native" => match profile.format {
                llm_types::llm::LlmFormat::Anthropic => {
                    builder = builder.header("x-api-key", api_key);
                }
                llm_types::llm::LlmFormat::GeminiNative => {
                    builder = builder.header("x-goog-api-key", api_key);
                }
                _ => {
                    builder = builder.header("Authorization", format!("Bearer {}", api_key));
                }
            },
            "x-api-key" => {
                builder = builder.header("x-api-key", api_key);
            }
            "x-goog-api-key" => {
                builder = builder.header("x-goog-api-key", api_key);
            }
            _ => {
                builder = builder.header("Authorization", format!("Bearer {}", api_key));
            }
        }
    }

    if let Some(headers) = &profile.headers {
        for (key, value) in headers.iter() {
            builder = builder.header(key, value.as_str().unwrap_or(""));
        }
    }
    if let Some(custom_headers) = &profile.custom_headers {
        for (key, value) in custom_headers.iter() {
            builder = builder.header(key, value.as_str().unwrap_or(""));
        }
    }
    if let Some(params) = &profile.query_params {
        let pairs: Vec<(&str, &str)> = params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str().unwrap_or("")))
            .collect();
        builder = builder.query(&pairs);
    }

    builder
}

/// Deep-merge `profile.custom_body` into the request body (custom body wins),
/// governed by `custom_body_enabled`.
pub fn apply_custom_body(body: &mut serde_json::Value, profile: &LlmProfile) {
    if profile.custom_body_enabled.unwrap_or(true) {
        if let Some(custom) = &profile.custom_body {
            *body = crate::codecs::helpers::deep_merge(body, custom);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_types::llm::MessageStreamEvent;

    #[test]
    fn stream_usage_chunk_is_extracted() {
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        match events
            .into_iter()
            .find(|e| matches!(e, MessageStreamEvent::Usage(_)))
        {
            Some(MessageStreamEvent::Usage(usage)) => {
                assert_eq!(usage.usage.prompt_tokens, 10);
                assert_eq!(usage.usage.completion_tokens, 5);
                assert_eq!(usage.usage.total_tokens, 15);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }
    }

    #[test]
    fn stream_start_carries_provider_identity() {
        let chunk = r#"{"id":"chatcmpl-1","model":"gpt-4o","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        match events.into_iter().next() {
            Some(MessageStreamEvent::Start(start)) => {
                assert_eq!(start.id.as_deref(), Some("chatcmpl-1"));
                assert_eq!(start.model.as_deref(), Some("gpt-4o"));
            }
            other => panic!("expected Start first, got {:?}", other),
        }
    }

    #[test]
    fn empty_choices_without_usage_yields_only_start() {
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        assert_eq!(events.len(), 1, "only the Start identity event is emitted");
        assert!(matches!(events[0], MessageStreamEvent::Start(_)));
    }

    #[test]
    fn chunk_without_identity_and_content_is_ignored() {
        let chunk = r#"{"object":"chat.completion.chunk","choices":[]}"#;
        assert!(parse_openai_stream_chunk(chunk)
            .expect("chunk must parse")
            .is_empty());
    }

    #[test]
    fn stream_tool_call_delta_is_extracted() {
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Bei"}}]}}]}"#;
        match parse_openai_stream_chunk(chunk)
            .expect("chunk must parse")
            .into_iter()
            .find(|e| matches!(e, MessageStreamEvent::ToolCallDelta(_)))
        {
            Some(MessageStreamEvent::ToolCallDelta(delta)) => {
                assert_eq!(delta.index, 0);
                assert_eq!(delta.id.as_deref(), Some("call_1"));
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                assert_eq!(delta.arguments.as_deref(), Some(r#"{"city":"Bei"#));
                assert!(!delta.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn stream_chunk_with_parallel_tool_calls_yields_all_deltas() {
        // A single SSE chunk carrying two tool calls must produce two
        // ToolCallDelta events (previously only the first was kept).
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}},{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{}"}}]}}]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        let deltas: Vec<_> = events
            .into_iter()
            .filter_map(|e| match e {
                MessageStreamEvent::ToolCallDelta(d) => Some(d),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 2, "both parallel tool calls must be emitted");
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(deltas[1].index, 1);
        assert_eq!(deltas[1].id.as_deref(), Some("call_2"));
    }

    #[test]
    fn openai_chat_tool_use_blocks_become_tool_calls_and_tool_results_become_tool_messages() {
        use llm_types::message::MessageContent;
        use llm_types::message::{MessageContentValue, MessageRole};

        let assistant = Message {
            id: llm_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![
                MessageContent::Text {
                    text: "let me check".to_string(),
                },
                MessageContent::ToolUse {
                    tool_use: llm_types::message::ToolUseContent {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        input: serde_json::json!({"city": "Beijing"}),
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

        // A `tool` role message whose content is a ToolResult rich block.
        let tool = Message {
            id: llm_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let out = convert_openai_messages(&[assistant, tool]);

        // Assistant message must carry tool_calls and must NOT contain any
        // illegal `tool_use` / `tool_result` content part.
        let asst = &out[0];
        assert_eq!(asst["role"], "assistant");
        assert_eq!(asst["tool_calls"][0]["id"], "call_1");
        assert_eq!(asst["tool_calls"][0]["function"]["name"], "get_weather");
        if let Some(arr) = asst["content"].as_array() {
            for part in arr {
                assert_ne!(part["type"], "tool_use");
                assert_ne!(part["type"], "tool_result");
            }
        }

        // The tool result must surface as a standalone `tool` role message.
        let tool_msg = out
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool message present");
        assert_eq!(tool_msg["tool_call_id"], "call_1");
        assert_eq!(tool_msg["content"], "sunny");
    }
}

#[cfg(test)]
mod enhancement_tests {
    use super::*;
    use llm_types::llm::LlmFormat;

    fn profile_with(auth_type: Option<&str>) -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format: LlmFormat::Anthropic,
            provider_id: None,
            model: "claude-3-5-sonnet".to_string(),
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
            auth_type: auth_type.map(String::from),
            custom_headers: Some(
                [("x-custom".to_string(), serde_json::json!("v1"))]
                    .into_iter()
                    .collect(),
            ),
            custom_body: None,
            custom_body_enabled: None,
            query_params: Some(
                [("api-version".to_string(), serde_json::json!("2024-01"))]
                    .into_iter()
                    .collect(),
            ),
            stream_options: None,
            context_window_size: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
        }
    }

    #[test]
    fn native_auth_uses_provider_header() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(Some("native")), "bearer")
            .build()
            .unwrap();
        assert_eq!(
            req.headers().get("x-api-key").unwrap(),
            "sk-test",
            "anthropic native auth must use x-api-key"
        );
        assert!(req.headers().get("Authorization").is_none());
    }

    #[test]
    fn bearer_auth_overrides_default() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(Some("bearer")), "x-api-key")
            .build()
            .unwrap();
        assert!(req.headers().get("x-api-key").is_none());
        assert_eq!(
            req.headers().get("Authorization").unwrap(),
            "Bearer sk-test"
        );
    }

    #[test]
    fn custom_headers_and_query_params_are_applied() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(None), "bearer")
            .build()
            .unwrap();
        assert_eq!(req.headers().get("x-custom").unwrap(), "v1");
        assert_eq!(req.url().query().unwrap(), "api-version=2024-01");
    }

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
