use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent, ToolCallProtocol,
};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// Whether the request runs in text-based tool mode (non-native tool protocol).
///
/// The locked protocol wins over the requested one, so the mode branch, the
/// protocol used for parsing and the protocol used for rendering always
/// agree with each other.
pub fn is_text_mode(request: &LlmRequest) -> bool {
    effective_tool_call_protocol(request) != ToolCallProtocol::Native
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
    llm_suite_message::history_converter::convert_to_text_mode(
        messages,
        &protocol,
        markers.as_ref(),
    )
}

/// Build the system prompt content for text-based tool mode: existing system
/// message + tool usage instructions + tool declarations.
pub fn text_mode_system_content(request: &LlmRequest) -> String {
    use llm_suite_tool_call::tool::protocol::{
        build_text_mode_system_content, extract_system_message,
    };
    let protocol = effective_tool_call_protocol(request);
    let (system, _) = extract_system_message(&request.messages);
    let tools = request.tools.as_deref().unwrap_or(&[]);
    build_text_mode_system_content(system.as_deref().unwrap_or(""), tools, protocol, false)
}

/// Parse tool calls from text content in text-based tool mode.
pub fn parse_text_tool_calls(
    request: &LlmRequest,
    content: &str,
) -> Vec<llm_suite_types::message::LlmToolCall> {
    use llm_suite_tool_call::tool::parser::parse_from_text;
    use llm_suite_tool_call::tool::protocol::get_tool_call_parser_options;
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
                            // Chat Completions carries the call arguments as a
                            // JSON string under `function.arguments`.
                            let arguments = serde_json::to_string(&tool_use.input)
                                .unwrap_or_else(|_| "{}".into());
                            tool_calls.push(serde_json::json!({
                                "id": tool_use.id,
                                "type": "function",
                                "function": {
                                    "name": tool_use.name,
                                    "arguments": arguments,
                                }
                            }));
                        }
                        // Emit ToolResult blocks as standalone `tool` messages.
                        MessageContent::ToolResult { tool_result } => {
                            tool_result_entries.push(serde_json::json!({
                                "role": "tool",
                                "content": tool_result.content,
                                "tool_call_id": tool_result.tool_use_id,
                            }));
                        }
                        MessageContent::Thinking { thinking, .. } => {
                            has_content = true;
                            content.push(serde_json::json!({"type": "text", "text": thinking}));
                        }
                        MessageContent::RedactedThinking { data } => {
                            // Chat Completions has no redacted-thinking part;
                            // the opaque payload is kept as text so the block
                            // is not silently dropped from the history.
                            has_content = true;
                            content.push(serde_json::json!({"type": "text", "text": data}));
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

/// OpenAI Chat requires every `tool` role message to carry `tool_call_id`.
/// Validate it so an invalid request fails fast locally instead of
/// round-tripping a `400` from the server. Text mode re-renders tool results
/// as inline blocks, so this only applies to native (non-text) requests.
///
/// A message whose result lives in a rich `ToolResult` block is not checked:
/// `convert_openai_messages` turns that block into a standalone `tool`
/// message that carries the id, and the source message itself never reaches
/// the wire as a `tool` entry.
pub fn validate_openai_chat_messages(request: &LlmRequest) -> LlmResult<()> {
    for msg in &request.messages {
        if msg.role != llm_suite_types::message::MessageRole::Tool || msg.tool_call_id.is_some() {
            continue;
        }
        let carries_tool_result = matches!(
            &msg.content,
            MessageContentValue::Rich(blocks)
                if blocks.iter().any(|b| matches!(b, MessageContent::ToolResult { .. }))
        );
        if carries_tool_result {
            continue;
        }
        return Err(crate::error::LlmError::ConfigError(
            "OpenAI Chat requires every `tool` role message to carry `tool_call_id`".to_string(),
        ));
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
) -> crate::error::LlmResult<llm_suite_types::llm::generation::LlmGenerationParams> {
    crate::generation::resolve_generation(profile, request)
}

/// Parse an OpenAI-style usage object into token usage stats.
///
/// The cached input tokens are a subset of `prompt_tokens` in this family
/// (`CacheTokenScope::IncludedInPrompt`), so the total is the sum of the
/// prompt and the completion. The provider's own total wins when it is
/// present.
pub fn parse_openai_usage(u: &serde_json::Value) -> llm_suite_types::llm::TokenUsageStats {
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
    let cache_write_tokens = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cache_write_tokens"))
        .and_then(|r| r.as_u64())
        .map(|r| r as u32);
    let prompt_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let completion_tokens = u
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    llm_suite_types::llm::TokenUsageStats {
        prompt_tokens,
        completion_tokens,
        // The provider owns the total; without one it is the sum of the
        // two counters. The cached tokens are already part of the prompt.
        total_tokens: u
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::from(prompt_tokens) + u64::from(completion_tokens))
            as u32,
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
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
                    let function = tc.get("function")?;
                    // The function name identifies the call and is required;
                    // a missing id or argument string is tolerated so a
                    // compatible gateway cannot make a whole call disappear.
                    let name = function.get("name")?.as_str()?.to_string();
                    let arguments = function
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| format!("call_{}", llm_suite_common::generate_id()));
                    Some(llm_suite_types::message::LlmToolCall {
                        id,
                        r#type: "function".to_string(),
                        call_id: None,
                        function: llm_suite_types::message::LlmFunctionCall { name, arguments },
                    })
                })
                .collect()
        });

    let usage = json.get("usage").map(parse_openai_usage);

    // A refusal is the provider declining to answer; it is kept apart from
    // the answer text so an empty answer can be told from a refusal.
    let refusal = message
        .and_then(|m| m.get("refusal"))
        .and_then(|r| r.as_str())
        .map(String::from);

    let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

    let msg = llm_suite_types::message::Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::Assistant,
        content: MessageContentValue::Text(content.clone().unwrap_or_default()),
        timestamp: llm_suite_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: tool_calls.clone(),
        // The reasoning text also lives on the message so that every codec
        // reports it the same way.
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
    // Web-search citations ride on `message.annotations`; they are the only
    // place the sources of a grounded answer are reported.
    if let Some(annotations) = message
        .and_then(|m| m.get("annotations"))
        .and_then(|a| a.as_array())
        .filter(|a| !a.is_empty())
    {
        metadata.insert(
            llm_suite_types::llm::ANNOTATIONS_KEY.to_string(),
            serde_json::json!(annotations),
        );
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
        refusal,
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
            llm_suite_types::llm::MessageStreamEnd {},
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
            llm_suite_types::llm::MessageStreamStart {
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
    let choice = choices.first();

    if let Some(choice) = choice {
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                events.push(MessageStreamEvent::Text(
                    llm_suite_types::llm::MessageStreamText {
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
                    llm_suite_types::llm::MessageStreamReasoning {
                        snapshot: String::new(),
                        reasoning: reasoning.to_string(),
                        signature: None,
                        signature_snapshot: None,
                    },
                ));
            }
            // A refusal is the provider declining to answer; it travels on
            // its own channel instead of mixing into the answer text.
            if let Some(refusal) = delta.get("refusal").and_then(|r| r.as_str()) {
                events.push(MessageStreamEvent::Refusal(
                    llm_suite_types::llm::MessageStreamRefusal {
                        refusal: refusal.to_string(),
                        snapshot: String::new(),
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
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id,
                            call_id: None,
                            name,
                            arguments,
                            thought_signature: None,
                            is_snapshot: false,
                        },
                    ));
                }
            }
        }
    }

    // The usage block may ride on any chunk. OpenAI sends it in a chunk
    // without choices, but compatible gateways attach it to the last chunk
    // that also carries the finish reason; reading it only from the
    // choice-less chunk would lose the counters there. It is emitted before
    // the terminal event so the accumulator sees the numbers in time.
    if let Some(usage) = json.get("usage") {
        events.push(MessageStreamEvent::Usage(
            llm_suite_types::llm::MessageStreamUsage {
                usage: parse_openai_usage(usage),
            },
        ));
    }

    if let Some(choice) = choice {
        // Provider-native terminal values passed through untouched; any other
        // value (including a missing one) leaves the stream open. The set is
        // the official `FinishReason` enumeration.
        if let Some(finish_reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            let is_done = matches!(
                finish_reason,
                "stop" | "tool_calls" | "length" | "content_filter" | "function_call"
            );
            if is_done {
                events.push(MessageStreamEvent::End(
                    llm_suite_types::llm::MessageStreamEnd {},
                ));
            }
        }
    }

    Ok(events)
}

pub fn convert_openai_tools(
    tools: &[llm_suite_types::tool::Tool],
) -> LlmResult<Vec<serde_json::Value>> {
    let tool_defs: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            // The official `function.parameters` is a JSON Schema object; a
            // tool without a schema must not serialize `null` (the API
            // rejects it). An empty object schema is the accepted
            // no-argument declaration.
            let parameters =
                t.parameters
                    .clone()
                    .unwrap_or(llm_suite_types::tool::ToolParameterSchema {
                        r#type: "object".to_string(),
                        properties: Default::default(),
                        required: Vec::new(),
                        additional_properties: None,
                    });
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                }
            })
        })
        .collect();
    Ok(tool_defs)
}

/// Build the endpoint URL for a provider path.
///
/// A base URL that already ends with the path is used as-is, so a profile can
/// carry a full endpoint including a query string (deployment-style
/// endpoints). Otherwise the path is appended. A trailing slash on the base
/// URL never produces a double slash.
///
/// The path may carry its own query string (Gemini's streaming endpoint uses
/// `?alt=sse`). The base query and the path query are merged instead of
/// concatenated, so a base URL that already carries a query (Azure's
/// `?api-version=...`) still yields a well-formed URL.
pub fn endpoint_url(base_url: &str, path: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let (base_without_query, base_query) = match trimmed.split_once('?') {
        Some((base, query)) => (base, Some(query)),
        None => (trimmed, None),
    };
    let (path_without_query, path_query) = match path.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (path, None),
    };
    let url = if base_without_query.ends_with(&format!("/{path_without_query}")) {
        base_without_query.to_string()
    } else {
        format!("{base_without_query}/{path_without_query}")
    };
    let mut query_parts: Vec<&str> = Vec::new();
    if let Some(query) = base_query {
        query_parts.push(query);
    }
    if let Some(query) = path_query {
        query_parts.push(query);
    }
    if query_parts.is_empty() {
        url
    } else {
        format!("{url}?{}", query_parts.join("&"))
    }
}

/// Apply authentication, custom headers and query parameters to the request.
///
/// Auth type resolution: `profile.auth_type` wins when set ("native" /
/// "bearer" / "x-api-key" / "x-goog-api-key"), otherwise `default_auth` is
/// used. "native" maps to the format-specific header (x-api-key for Anthropic,
/// x-goog-api-key for Gemini native, Bearer otherwise).
///
/// Headers are written once per name: `reqwest` appends duplicate header
/// values instead of replacing them, so the profile headers are merged into a
/// single table (`custom_headers` wins over `headers`) before they are set.
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
                llm_suite_types::llm::LlmFormat::Anthropic => {
                    builder = builder.header("x-api-key", api_key);
                }
                llm_suite_types::llm::LlmFormat::GeminiNative => {
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

    for (key, value) in merged_profile_headers(profile) {
        builder = builder.header(key, value);
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

/// The profile headers as a single name → value table.
///
/// `custom_headers` overrides `headers` for the same name: the request must
/// carry exactly one value per header name.
fn merged_profile_headers(profile: &LlmProfile) -> Vec<(String, String)> {
    let mut merged: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    if let Some(headers) = &profile.headers {
        for (key, value) in headers {
            merged.insert(key.clone(), value.as_str().unwrap_or("").to_string());
        }
    }
    if let Some(custom_headers) = &profile.custom_headers {
        for (key, value) in custom_headers {
            merged.insert(key.clone(), value.as_str().unwrap_or("").to_string());
        }
    }
    merged.into_iter().collect()
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
    use llm_suite_types::llm::MessageStreamEvent;

    #[test]
    fn endpoint_url_appends_the_path_only_once() {
        assert_eq!(
            super::endpoint_url("https://api.openai.com/v1", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            super::endpoint_url("https://api.openai.com/v1/", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn endpoint_url_keeps_a_full_endpoint() {
        assert_eq!(
            super::endpoint_url(
                "https://example.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-01",
                "chat/completions"
            ),
            "https://example.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-01"
        );
    }

    #[test]
    fn endpoint_url_merges_the_base_query_with_the_path_query() {
        // A path that carries its own query (the Gemini streaming endpoint)
        // and a base that carries one as well must not be concatenated with a
        // second question mark.
        assert_eq!(
            super::endpoint_url(
                "https://generativelanguage.googleapis.com/v1beta",
                "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            super::endpoint_url(
                "https://gateway.example.com/v1?api-version=2024-01",
                "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
            ),
            "https://gateway.example.com/v1/models/gemini-2.5-pro:streamGenerateContent?api-version=2024-01&alt=sse"
        );
    }

    #[test]
    fn refusal_is_parsed_and_kept_out_of_the_content() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null,"refusal":"I cannot help with that."},"finish_reason":"stop"}]}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        assert_eq!(result.content, None);
        assert_eq!(result.refusal.as_deref(), Some("I cannot help with that."));
    }

    #[test]
    fn tool_call_without_arguments_keeps_the_call() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather"}}]},"finish_reason":"tool_calls"}]}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        let calls = result.tool_calls.expect("tool calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{}");
    }

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
    fn stream_usage_next_to_the_choices_is_not_dropped() {
        // Compatible gateways attach the usage to the last chunk that also
        // carries the finish reason; the counters must survive that.
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        let usage = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::Usage(u) => Some(&u.usage),
                _ => None,
            })
            .expect("the usage must not be dropped");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert!(
            matches!(events.last(), Some(MessageStreamEvent::End(_))),
            "the usage must arrive before the terminal event"
        );
    }

    #[test]
    fn usage_without_a_provider_total_is_summed() {
        // The cached tokens are already part of the prompt in this family,
        // so the total is the sum of the two counters.
        let usage = parse_openai_usage(&serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80}
        }));
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(usage.cache_read_tokens, Some(80));
    }

    #[test]
    fn usage_parses_official_cache_details() {
        // Official `CompletionUsage` shape: `prompt_tokens_details` carries
        // both `cached_tokens` and `cache_write_tokens`.
        let usage = parse_openai_usage(&serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 15},
            "completion_tokens_details": {"reasoning_tokens": 6}
        }));
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(usage.reasoning_tokens, Some(6));
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, Some(15));

        // Details objects absent → cache fields stay `None`.
        let bare = parse_openai_usage(&serde_json::json!({
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
        }));
        assert_eq!(bare.cache_read_tokens, None);
        assert_eq!(bare.cache_write_tokens, None);
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
    fn the_deprecated_function_call_reason_ends_the_stream() {
        let chunk =
            r#"{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        assert!(
            matches!(events.last(), Some(MessageStreamEvent::End(_))),
            "every official finish reason terminates the turn"
        );
    }

    #[test]
    fn a_tool_without_parameters_declares_an_empty_object_schema() {
        let tools = vec![llm_suite_types::tool::Tool {
            name: "get_time".to_string(),
            description: "Returns the current time".to_string(),
            parameters: None,
        }];
        let shape = convert_openai_tools(&tools).expect("must convert");
        let parameters = &shape[0]["function"]["parameters"];
        assert!(
            parameters.is_object(),
            "the schema must be an object, never null: {shape:?}"
        );
        assert_eq!(parameters["type"], serde_json::json!("object"));
    }

    #[test]
    fn a_tool_result_in_a_rich_block_needs_no_message_level_tool_call_id() {
        use llm_suite_types::message::{MessageContent, MessageContentValue, MessageRole};

        let mut request = LlmRequest {
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
        request.messages.push(Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
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
        });

        // The rich block carries the correlation id, so the message itself
        // does not have to.
        validate_openai_chat_messages(&request).expect("a rich tool result must not be rejected");

        let converted = convert_openai_messages(&request.messages);
        let tool_message = converted
            .iter()
            .find(|m| m["role"] == serde_json::json!("tool"))
            .expect("the tool message must be present");
        assert_eq!(tool_message["tool_call_id"], serde_json::json!("call_1"));
    }

    #[test]
    fn a_tool_message_without_any_result_is_rejected() {
        use llm_suite_types::message::{MessageContentValue, MessageRole};

        let request = LlmRequest {
            profile_id: "p1".to_string(),
            messages: vec![Message {
                id: llm_suite_types::Id::new(),
                role: MessageRole::Tool,
                content: MessageContentValue::Text("sunny".to_string()),
                timestamp: 0,
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                thinking: None,
                thinking_signature: None,
                metadata: None,
            }],
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
        let err = validate_openai_chat_messages(&request)
            .expect_err("a tool message without a result id must be rejected");
        assert!(format!("{err}").contains("tool_call_id"), "{err}");
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
        use llm_suite_types::message::MessageContent;
        use llm_suite_types::message::{MessageContentValue, MessageRole};

        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![
                MessageContent::Text {
                    text: "let me check".to_string(),
                },
                MessageContent::ToolUse {
                    tool_use: llm_suite_types::message::ToolUseContent {
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
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
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
    use llm_suite_types::llm::LlmFormat;

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
            prompt_caching: None,
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
    fn a_header_name_is_written_once() {
        let mut profile = profile_with(None);
        profile.headers = Some(
            [("x-dup".to_string(), serde_json::json!("from-headers"))]
                .into_iter()
                .collect(),
        );
        profile.custom_headers = Some(
            [(
                "x-dup".to_string(),
                serde_json::json!("from-custom-headers"),
            )]
            .into_iter()
            .collect(),
        );
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile, "bearer")
            .build()
            .unwrap();
        let values: Vec<&str> = req
            .headers()
            .get_all("x-dup")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        assert_eq!(
            values,
            vec!["from-custom-headers"],
            "a header name must carry exactly one value"
        );
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
