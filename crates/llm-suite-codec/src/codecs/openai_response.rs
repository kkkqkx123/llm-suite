use super::LlmCodec;
use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent, MessageStreamStart,
};
use llm_suite_types::tool::Tool;
use reqwest::Method;
use std::collections::HashMap;

pub struct OpenaiResponseCodec {
    base_url: String,
    /// Reused across requests; constructing a `reqwest::Client` per call would
    /// discard the connection pool and TLS session cache every time.
    client: reqwest::Client,
}

impl Default for OpenaiResponseCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenaiResponseCodec {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    fn convert_messages(
        &self,
        messages: &[llm_suite_types::message::Message],
    ) -> Vec<serde_json::Value> {
        let mut items: Vec<serde_json::Value> = Vec::new();

        // A function call is keyed by its provider call id. A tool result
        // references that same id, so every identity a result may carry (the
        // item id or the call id) is mapped onto the id the call item uses.
        let mut call_ids: HashMap<&str, String> = HashMap::new();
        for msg in messages {
            if let Some(ref calls) = msg.tool_calls {
                for tc in calls {
                    let call_id = tc.call_id.clone().unwrap_or_else(|| tc.id.clone());
                    call_ids.insert(tc.id.as_str(), call_id.clone());
                    if let Some(ref provider_call_id) = tc.call_id {
                        call_ids.insert(provider_call_id.as_str(), call_id.clone());
                    }
                }
            }
        }

        for msg in messages {
            let role = match msg.role {
                // The system prompt travels in the dedicated `instructions`
                // field, not as an input item.
                llm_suite_types::message::MessageRole::System => continue,
                llm_suite_types::message::MessageRole::User => "user",
                llm_suite_types::message::MessageRole::Assistant => "assistant",
                llm_suite_types::message::MessageRole::Tool => "tool",
            };

            // Map the message body to a Responses `input` content value
            // (string shorthand or `input_text` / `input_image` parts).
            let content: serde_json::Value = match &msg.content {
                llm_suite_types::message::MessageContentValue::Text(text) => {
                    serde_json::Value::String(text.clone())
                }
                // The Responses API uses `input_text` / `input_image`
                // content part types (not Chat's `text` / `image_url`),
                // so Rich blocks are mapped explicitly.
                llm_suite_types::message::MessageContentValue::Rich(blocks) => {
                    serde_json::Value::Array(
                        blocks
                            .iter()
                            .map(|block| match block {
                                llm_suite_types::message::MessageContent::Text { text } => {
                                    serde_json::json!({"type": "input_text", "text": text})
                                }
                                llm_suite_types::message::MessageContent::ImageUrl {
                                    image_url,
                                } => {
                                    let mut part = serde_json::json!({
                                        "type": "input_image",
                                        "image_url": image_url.url,
                                    });
                                    if let Some(ref detail) = image_url.detail {
                                        part["detail"] = serde_json::json!(detail);
                                    }
                                    part
                                }
                                llm_suite_types::message::MessageContent::Thinking {
                                    thinking,
                                    ..
                                } => {
                                    serde_json::json!({"type": "input_text", "text": thinking})
                                }
                                llm_suite_types::message::MessageContent::RedactedThinking {
                                    data,
                                } => {
                                    // The Responses API has no redacted-thinking
                                    // part; the opaque payload is kept as text
                                    // so the block is not dropped from history.
                                    serde_json::json!({"type": "input_text", "text": data})
                                }
                                // Tool blocks are not valid user content in
                                // the Responses API; render them as text so
                                // history is not silently dropped.
                                llm_suite_types::message::MessageContent::ToolUse { tool_use } => {
                                    serde_json::json!({
                                        "type": "input_text",
                                        "text": tool_use.input.to_string(),
                                    })
                                }
                                llm_suite_types::message::MessageContent::ToolResult {
                                    tool_result,
                                } => {
                                    serde_json::json!({
                                        "type": "input_text",
                                        "text": tool_result.content,
                                    })
                                }
                            })
                            .collect(),
                    )
                }
            };

            match msg.role {
                // Tool results are `function_call_output` items in the
                // Responses API, correlated to the originating function call by
                // `call_id` (the message's `tool_call_id`). Chat's
                // `tool_calls`/`tool_call_id` shape is rejected by the API.
                llm_suite_types::message::MessageRole::Tool => {
                    if let Some(ref tool_call_id) = msg.tool_call_id {
                        // The output item has to carry the same id the call
                        // item was keyed by, whichever identity the history
                        // recorded the result under.
                        let call_id = call_ids
                            .get(tool_call_id.as_str())
                            .cloned()
                            .unwrap_or_else(|| tool_call_id.clone());
                        let output = match &msg.content {
                            llm_suite_types::message::MessageContentValue::Text(text) => {
                                text.clone()
                            }
                            llm_suite_types::message::MessageContentValue::Rich(blocks) => blocks
                                .iter()
                                .map(|b| match b {
                                    llm_suite_types::message::MessageContent::Text { text } => {
                                        text.clone()
                                    }
                                    _ => String::new(),
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        items.push(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output,
                        }));
                    } else {
                        // No correlation id available: fall back to a plain
                        // input item so the tool result is not dropped.
                        items.push(serde_json::json!({
                            "role": "user",
                            "content": content,
                        }));
                    }
                }
                // Assistant turns with tool calls emit one `function_call`
                // item per call (keyed by `call_id`), plus a text message item
                // when the assistant also produced prose.
                llm_suite_types::message::MessageRole::Assistant => {
                    let has_text = match &content {
                        serde_json::Value::String(s) => !s.is_empty(),
                        serde_json::Value::Array(a) => !a.is_empty(),
                        _ => false,
                    };
                    if has_text {
                        items.push(serde_json::json!({
                            "role": "assistant",
                            "content": content,
                        }));
                    }
                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            let call_id = tc.call_id.clone().unwrap_or_else(|| tc.id.clone());
                            items.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }));
                        }
                    }
                }
                _ => {
                    items.push(serde_json::json!({
                        "role": role,
                        "content": content,
                    }));
                }
            }
        }

        items
    }

    /// Wire tool declarations. The Responses API declares a function tool
    /// flat: `name`, `description` and `parameters` sit next to `type`
    /// instead of being wrapped in a `function` object.
    fn response_tool_defs(tools: &[Tool]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                // The schema is a JSON Schema object; a tool without one must
                // not serialize `null` (the API rejects it). An empty object
                // schema is the accepted no-argument declaration.
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
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                })
            })
            .collect()
    }

    /// Split a request into the system instructions and the input history.
    ///
    /// The system prompt travels in the dedicated `instructions` field, the
    /// channel the official API reserves for it; it is not an input item.
    /// In text mode the instructions carry the tool declarations as well.
    fn split_system_and_history(
        &self,
        request: &LlmRequest,
    ) -> (String, Vec<llm_suite_types::message::Message>) {
        let (system_content, filtered) =
            llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);
        if super::shared::is_text_mode(request) {
            let history = super::shared::convert_history_for_text_mode(&filtered, request);
            (super::shared::text_mode_system_content(request), history)
        } else {
            (system_content.unwrap_or_default(), filtered)
        }
    }

    /// Build the request body (separated from the HTTP layer for testability).
    fn build_body(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let use_text_mode = super::shared::is_text_mode(request);
        let (instructions, history) = self.split_system_and_history(request);

        let mut body = serde_json::json!({
            "model": profile.model,
            "input": self.convert_messages(&history),
        });

        if !instructions.is_empty() {
            body["instructions"] = serde_json::json!(instructions);
        }

        if request.stream == Some(true) {
            body["stream"] = serde_json::json!(true);
        }

        let generation = super::shared::resolve_generation(request, profile)?;
        crate::generation::apply_openai_responses(&mut body, &generation)?;
        super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

        if !use_text_mode {
            if let Some(tools) = &request.tools {
                body["tools"] = serde_json::json!(Self::response_tool_defs(tools));
            }
        }

        super::shared::apply_custom_body(&mut body, profile);

        Ok(body)
    }

    /// Build the count-tokens body: the same `input` (plus native `tools`)
    /// that `build_request` sends, without stream flags or generation
    /// parameters (they do not affect the input count).
    fn build_count_tokens_body(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let use_text_mode = super::shared::is_text_mode(request);
        let (instructions, history) = self.split_system_and_history(request);

        let mut body = serde_json::json!({
            "model": profile.model,
            "input": self.convert_messages(&history),
        });

        if !instructions.is_empty() {
            body["instructions"] = serde_json::json!(instructions);
        }
        if !use_text_mode {
            if let Some(tools) = &request.tools {
                body["tools"] = serde_json::json!(Self::response_tool_defs(tools));
            }
        }

        Ok(body)
    }

    fn parse_tool_calls_from_output(
        &self,
        output: &[serde_json::Value],
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
        output
            .iter()
            .filter_map(|item| {
                let item_type = item.get("type").and_then(|v| v.as_str())?;
                if item_type != "function_call" {
                    return None;
                }
                let id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = match item.get("arguments").and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => {
                        tracing::debug!(
                            tool_id = id.as_str(),
                            tool_name = name.as_str(),
                            "function call arguments missing or not a string; falling back to empty object",
                        );
                        "{}".to_string()
                    }
                };
                Some(llm_suite_types::message::LlmToolCall {
                    id,
                    r#type: "function".to_string(),
                    call_id: Some(call_id),
                    function: llm_suite_types::message::LlmFunctionCall { name, arguments },
                })
            })
            .collect()
    }

    /// Parse one Responses SSE chunk into zero or more stream events.
    ///
    /// The terminal event carries the final usage and closes the turn, so a
    /// chunk can produce both a usage and a terminal event. The usage event
    /// comes first: the stream accumulator builds the final message when the
    /// terminal event arrives.
    fn parse_stream_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // The Responses SSE stream is terminated by a `[DONE]` sentinel that
        // is not JSON; it closes the turn like the Chat Completions stream
        // does.
        if data.trim() == "[DONE]" {
            return Ok(vec![MessageStreamEvent::End(
                llm_suite_types::llm::MessageStreamEnd {},
            )]);
        }

        let json: serde_json::Value = serde_json::from_str(data)?;
        let event_type = json.get("type").and_then(|v| v.as_str());
        let mut events = Vec::new();

        match event_type {
            Some("response.created") => {
                // Stream identity; the accumulator records it for the final
                // message and forwards it once.
                let response = json.get("response");
                let id = response
                    .and_then(|r| r.get("id"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let model = response
                    .and_then(|r| r.get("model"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                if id.is_some() || model.is_some() {
                    events.push(MessageStreamEvent::Start(MessageStreamStart { id, model }));
                }
            }
            Some("response.output_item.added") => {
                // The function name and the call identity ride on the output
                // item; the argument events that follow only carry fragments.
                let item = json.get("item");
                if item.and_then(|i| i.get("type")).and_then(|v| v.as_str())
                    == Some("function_call")
                {
                    let index = json
                        .get("output_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let id = item
                        .and_then(|i| i.get("id"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let call_id = item
                        .and_then(|i| i.get("call_id"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let name = item
                        .and_then(|i| i.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    events.push(MessageStreamEvent::ToolCallDelta(
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id,
                            call_id,
                            name,
                            arguments: None,
                            thought_signature: None,
                            is_snapshot: false,
                        },
                    ));
                }
            }
            Some("response.output_item.done") => {
                // The finished output item is the authoritative view of a
                // function call: it carries the call id and the complete
                // argument string next to the item id.
                let item = json.get("item");
                if item.and_then(|i| i.get("type")).and_then(|v| v.as_str())
                    == Some("function_call")
                {
                    let index = json
                        .get("output_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let id = item
                        .and_then(|i| i.get("id"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let call_id = item
                        .and_then(|i| i.get("call_id"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let name = item
                        .and_then(|i| i.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let arguments = item
                        .and_then(|i| i.get("arguments"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    events.push(MessageStreamEvent::ToolCallDelta(
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id,
                            call_id,
                            name,
                            arguments,
                            thought_signature: None,
                            is_snapshot: true,
                        },
                    ));
                }
            }
            Some("response.output_text.delta") => {
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    events.push(MessageStreamEvent::Text(
                        llm_suite_types::llm::MessageStreamText {
                            snapshot: String::new(),
                            text: delta.to_string(),
                        },
                    ));
                }
            }
            Some("response.output_text.annotation.added") => {
                // The sources of a grounded answer travel on their own
                // channel so they never mix into the answer text.
                if let Some(annotation) = json.get("annotation") {
                    events.push(MessageStreamEvent::Annotation(
                        llm_suite_types::llm::MessageStreamAnnotation {
                            annotation: annotation.clone(),
                        },
                    ));
                }
            }
            Some("response.reasoning_text.delta")
            | Some("response.reasoning_summary_text.delta") => {
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    events.push(MessageStreamEvent::ReasoningText(
                        llm_suite_types::llm::MessageStreamReasoning {
                            reasoning: delta.to_string(),
                            snapshot: String::new(),
                            signature: None,
                            signature_snapshot: None,
                        },
                    ));
                }
            }
            Some("response.refusal.delta") => {
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    events.push(MessageStreamEvent::Refusal(
                        llm_suite_types::llm::MessageStreamRefusal {
                            refusal: delta.to_string(),
                            snapshot: String::new(),
                        },
                    ));
                }
            }
            Some("response.function_call_arguments.delta") => {
                // Incremental arguments fragment. The output index is the
                // position of the call in the response output array and is
                // stable for the whole stream, so fragments of one call share
                // it and parallel calls stay apart.
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    let index = json
                        .get("output_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    events.push(MessageStreamEvent::ToolCallDelta(
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id: None,
                            call_id: None,
                            name: None,
                            arguments: Some(delta.to_string()),
                            thought_signature: None,
                            is_snapshot: false,
                        },
                    ));
                }
            }
            Some("response.function_call_arguments.done") => {
                // The event carries the complete argument string and the call
                // identity at its top level; the name was announced by the
                // output item event.
                if let Some(arguments) = json.get("arguments").and_then(|v| v.as_str()) {
                    let index = json
                        .get("output_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let id = json
                        .get("item_id")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    events.push(MessageStreamEvent::ToolCallDelta(
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id,
                            call_id: None,
                            name: None,
                            arguments: Some(arguments.to_string()),
                            thought_signature: None,
                            is_snapshot: true,
                        },
                    ));
                }
            }
            Some("response.failed") => {
                // The failure payload lives in the completed response object:
                // `{"type":"response.failed","response":{...,"error":{...}}}`.
                let message = json
                    .pointer("/response/error/message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("response failed")
                    .to_string();
                tracing::warn!("Responses stream ended with a failed response: {message}");
                events.push(MessageStreamEvent::Error(
                    llm_suite_types::llm::MessageStreamError { error: message },
                ));
                events.push(MessageStreamEvent::End(
                    llm_suite_types::llm::MessageStreamEnd {},
                ));
            }
            Some("error") => {
                // Standalone stream error event: `{type, code, message, param}`.
                let message = json
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("stream error")
                    .to_string();
                tracing::warn!("Responses stream ended with an error event: {message}");
                events.push(MessageStreamEvent::Error(
                    llm_suite_types::llm::MessageStreamError { error: message },
                ));
                events.push(MessageStreamEvent::End(
                    llm_suite_types::llm::MessageStreamEnd {},
                ));
            }
            Some("response.completed") | Some("response.incomplete") => {
                // The terminal event carries the final usage summary inside
                // the response object: `{"type":..,"response":{..,"usage":..}}`.
                if let Some(usage) = json.pointer("/response/usage") {
                    let prompt_tokens = usage
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let completion_tokens = usage
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    events.push(MessageStreamEvent::Usage(
                        llm_suite_types::llm::MessageStreamUsage {
                            usage: llm_suite_types::llm::TokenUsageStats {
                                prompt_tokens,
                                completion_tokens,
                                // The provider owns the total; without one it
                                // is the sum of the two counters.
                                total_tokens: usage
                                    .get("total_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(
                                        u64::from(prompt_tokens) + u64::from(completion_tokens),
                                    ) as u32,
                                reasoning_tokens: usage
                                    .get("output_tokens_details")
                                    .and_then(|d| d.get("reasoning_tokens"))
                                    .and_then(|v| v.as_u64())
                                    .map(|r| r as u32),
                                cache_read_tokens: usage
                                    .get("input_tokens_details")
                                    .and_then(|d| d.get("cached_tokens"))
                                    .and_then(|v| v.as_u64())
                                    .map(|r| r as u32),
                                cache_write_tokens: usage
                                    .get("input_tokens_details")
                                    .and_then(|d| d.get("cache_write_tokens"))
                                    .and_then(|v| v.as_u64())
                                    .map(|r| r as u32),
                                prompt_tokens_cost: None,
                                completion_tokens_cost: None,
                                total_cost: None,
                            },
                        },
                    ));
                }
                events.push(MessageStreamEvent::End(
                    llm_suite_types::llm::MessageStreamEnd {},
                ));
            }
            other => {
                tracing::debug!(
                    event_type = other.unwrap_or(""),
                    "unhandled Responses stream event; ignoring",
                );
            }
        }

        Ok(events)
    }
}

impl LlmCodec for OpenaiResponseCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        let url = super::shared::endpoint_url(
            profile.base_url.as_deref().unwrap_or(&self.base_url),
            "responses",
        );

        let body = self.build_body(request, profile)?;

        let mut req_builder = self
            .client
            .request(Method::POST, &url)
            .header("Content-Type", "application/json")
            .json(&body);

        req_builder = super::shared::apply_auth_and_headers(req_builder, profile, "bearer");

        req_builder
            .build()
            .map_err(crate::error::LlmError::HttpError)
    }

    fn build_count_tokens_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<Option<reqwest::Request>> {
        let url = super::shared::endpoint_url(
            profile.base_url.as_deref().unwrap_or(&self.base_url),
            "responses/input_tokens",
        );

        let body = self.build_count_tokens_body(request, profile)?;

        let mut req_builder = self
            .client
            .request(Method::POST, &url)
            .header("Content-Type", "application/json")
            .json(&body);

        req_builder = super::shared::apply_auth_and_headers(req_builder, profile, "bearer");

        req_builder
            .build()
            .map(Some)
            .map_err(crate::error::LlmError::HttpError)
    }

    fn parse_response(&self, body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
        let mut result = self.parse_response_inner(body)?;
        if super::shared::is_text_mode(request) {
            if let Some(content) = result.content.clone() {
                let calls = super::shared::parse_text_tool_calls(request, &content);
                if !calls.is_empty() {
                    result.tool_calls = Some(calls.clone());
                    result.message.tool_calls = Some(calls);
                }
            }
        }
        Ok(result)
    }

    fn parse_stream_chunk_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        self.parse_stream_events(data)
    }

    fn convert_tools(&self, tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
        Ok(Self::response_tool_defs(tools))
    }

    fn parse_tool_calls(
        &self,
        result: &LlmResponseType,
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
        result.tool_calls.clone().unwrap_or_default()
    }
}

impl OpenaiResponseCodec {
    fn parse_response_inner(&self, body: &str) -> LlmResult<LlmResponseType> {
        let json: serde_json::Value = serde_json::from_str(body)?;

        let id = json.get("id").and_then(|v| v.as_str()).map(String::from);
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let status = json
            .get("status")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Surface failed responses or explicit error objects as errors rather
        // than returning a silently-empty result. `error` is omitted (or null)
        // on success, so it is only acted on when present and non-null.
        if status.as_deref() == Some("failed") {
            let msg = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| "response generation failed".to_string());
            return Err(crate::error::LlmError::InvalidResponse(format!(
                "OpenAI Responses: {msg}"
            )));
        }
        if let Some(err) = json.get("error").filter(|e| !e.is_null()) {
            let msg = err
                .get("message")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| err.as_str().map(String::from))
                .unwrap_or_else(|| "response error".to_string());
            return Err(crate::error::LlmError::InvalidResponse(format!(
                "OpenAI Responses: {msg}"
            )));
        }

        // Aggregate text from every `output_text` block across all `message`
        // output items (mirrors the official `Response.output_text` property),
        // capturing refusals and reasoning text into their own channels.
        let mut text_parts: Vec<String> = Vec::new();
        let mut refusal_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        // The sources of a grounded answer ride on the `output_text` blocks.
        let mut annotations: Vec<serde_json::Value> = Vec::new();
        if let Some(output) = json.get("output").and_then(|v| v.as_array()) {
            for item in output {
                match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "message" => {
                        if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                            for block in blocks {
                                match block.get("type").and_then(|t| t.as_str()) {
                                    Some("output_text") => {
                                        if let Some(t) = block.get("text").and_then(|v| v.as_str())
                                        {
                                            text_parts.push(t.to_string());
                                        }
                                        if let Some(list) =
                                            block.get("annotations").and_then(|v| v.as_array())
                                        {
                                            annotations.extend(list.iter().cloned());
                                        }
                                    }
                                    // A refusal is a content block inside the
                                    // message item, not a separate output item.
                                    Some("refusal") => {
                                        if let Some(t) =
                                            block.get("refusal").and_then(|v| v.as_str())
                                        {
                                            refusal_parts.push(t.to_string());
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    "reasoning" => {
                        // The summary list and the reasoning text list are
                        // siblings of the item, both optional.
                        if let Some(blocks) = item.get("summary").and_then(|c| c.as_array()) {
                            for block in blocks {
                                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                    reasoning_parts.push(t.to_string());
                                }
                            }
                        }
                        if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                            for block in blocks {
                                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                    reasoning_parts.push(t.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let content = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.concat())
        };
        let refusal = if refusal_parts.is_empty() {
            None
        } else {
            Some(refusal_parts.concat())
        };
        let reasoning_content = if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.concat())
        };

        let mut tool_calls: Option<Vec<llm_suite_types::message::LlmToolCall>> = None;
        if let Some(output_arr) = json.get("output").and_then(|v| v.as_array()) {
            let calls = self.parse_tool_calls_from_output(output_arr);
            if !calls.is_empty() {
                tool_calls = Some(calls);
            }
        }

        let usage = json.get("usage").map(|u| {
            let input_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let output_tokens = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            llm_suite_types::llm::TokenUsageStats {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                // The provider owns the total: the official usage object
                // carries it, and it may cover buckets the two counters do
                // not describe. The sum is only the fallback.
                total_tokens: u
                    .get("total_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::from(input_tokens) + u64::from(output_tokens))
                    as u32,
                reasoning_tokens: u
                    .get("output_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
                    .and_then(|v| v.as_u64())
                    .map(|r| r as u32),
                cache_read_tokens: u
                    .get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .map(|r| r as u32),
                cache_write_tokens: u
                    .get("input_tokens_details")
                    .and_then(|d| d.get("cache_write_tokens"))
                    .and_then(|v| v.as_u64())
                    .map(|r| r as u32),
                prompt_tokens_cost: None,
                completion_tokens_cost: None,
                total_cost: None,
            }
        });

        let message = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Assistant,
            content: llm_suite_types::message::MessageContentValue::Text(
                content.clone().unwrap_or_default(),
            ),
            timestamp: llm_suite_common::time::now(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: tool_calls.clone(),
            // The reasoning text also lives on the message so that replaying
            // the message in a follow-up turn keeps it, the same way the
            // Anthropic codec does.
            thinking: reasoning_content.clone(),
            thinking_signature: None,
            metadata: None,
        };

        // The raw lifecycle state of the response; the stop value below is
        // derived from it.
        let raw_status = status;

        // The stop value uses the same vocabulary as the Chat Completions
        // codec: the Responses `status` is a lifecycle state, not a stop
        // reason. The reason an incomplete response stopped lives in
        // `incomplete_details.reason`.
        let finish_reason = Self::stop_reason(
            raw_status.as_deref(),
            json.pointer("/incomplete_details/reason")
                .and_then(|v| v.as_str()),
            tool_calls.as_ref().is_some_and(|calls| !calls.is_empty()),
        );

        // Why an incomplete response stopped: `max_output_tokens`,
        // `max_messages`, `content_filter` or `steered`.
        let mut metadata = std::collections::HashMap::new();
        if let Some(reason) = json
            .pointer("/incomplete_details/reason")
            .and_then(|v| v.as_str())
        {
            metadata.insert("incomplete_reason".to_string(), serde_json::json!(reason));
        }
        if let Some(ref status) = raw_status {
            metadata.insert("status".to_string(), serde_json::json!(status));
        }
        if !annotations.is_empty() {
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

        let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

        Ok(LlmResponseType {
            id,
            model,
            content,
            message,
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

    /// Map the Responses lifecycle state onto the stop vocabulary the Chat
    /// Completions codec reports.
    ///
    /// A finished response stopped because the model answered (`stop`), ran
    /// out of output budget (`length`), was cut by a content filter
    /// (`content_filter`) or asked for a tool (`tool_calls`). The official
    /// `incomplete_details.reason` values (`max_output_tokens`,
    /// `max_messages`, `content_filter`, `steered`) decide the outcome.
    ///
    /// The Responses API has no tool-call stop reason — a completed response
    /// that carries function calls is reported as `tool_calls`, the value the
    /// Chat Completions codec reports for the same situation.
    fn stop_reason(
        status: Option<&str>,
        incomplete_reason: Option<&str>,
        has_tool_calls: bool,
    ) -> Option<String> {
        match status {
            Some("failed") => Some("error".to_string()),
            Some("incomplete") => Some(match incomplete_reason {
                Some("max_output_tokens") | Some("max_messages") => "length".to_string(),
                Some("content_filter") => "content_filter".to_string(),
                Some(other) => other.to_string(),
                None => "incomplete".to_string(),
            }),
            Some("completed") => Some(if has_tool_calls {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            }),
            Some("cancelled") => Some("cancelled".to_string()),
            Some("queued") | Some("in_progress") => None,
            Some(other) => Some(other.to_string()),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call_and_output_share_the_provider_call_id() {
        let codec = OpenaiResponseCodec::new();
        let mut assistant = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Assistant,
            content: llm_suite_types::message::MessageContentValue::Text(String::new()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: Some(vec![llm_suite_types::message::LlmToolCall {
                id: "fc_1".to_string(),
                r#type: "function".to_string(),
                call_id: Some("call_1".to_string()),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        // The result references the call by its item id.
        let mut result = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Tool,
            content: llm_suite_types::message::MessageContentValue::Text("sunny".to_string()),
            timestamp: 0,
            tool_call_id: Some("fc_1".to_string()),
            tool_name: Some("get_weather".to_string()),
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        assistant.timestamp = 0;
        result.timestamp = 0;

        let items = codec.convert_messages(&[assistant, result]);
        let call = items
            .iter()
            .find(|item| item["type"] == serde_json::json!("function_call"))
            .expect("function call item present");
        let output = items
            .iter()
            .find(|item| item["type"] == serde_json::json!("function_call_output"))
            .expect("function call output item present");
        assert_eq!(call["call_id"], serde_json::json!("call_1"));
        assert_eq!(
            output["call_id"], call["call_id"],
            "the output must reference the same call"
        );
    }

    #[test]
    fn streamed_tool_call_carries_the_provider_call_id() {
        let codec = OpenaiResponseCodec::new();
        let added = r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#;
        match codec
            .parse_stream_chunk_events(added)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.call_id.as_deref(), Some("call_1"));
                assert_eq!(d.name.as_deref(), Some("get_weather"));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn tool_declaration_is_flat() {
        let codec = OpenaiResponseCodec::new();
        let tools = vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()];
        let shape = codec.convert_tools(&tools).unwrap();
        assert_eq!(shape[0]["type"], serde_json::json!("function"));
        assert_eq!(shape[0]["name"], serde_json::json!("get_weather"));
        assert_eq!(shape[0]["description"], serde_json::json!("Get weather"));
        assert!(
            shape[0].get("function").is_none(),
            "Responses declares function tools flat"
        );
    }

    #[test]
    fn named_tool_choice_is_flat() {
        use llm_suite_types::llm::generation::{LlmToolChoice, ToolChoiceMode};

        let mut body = serde_json::json!({"model": "gpt-4o"});
        crate::generation::apply_openai_responses(
            &mut body,
            &llm_suite_types::llm::generation::LlmGenerationParams {
                tool_choice: Some(LlmToolChoice {
                    mode: Some(ToolChoiceMode::Required),
                    tool_name: Some("get_weather".to_string()),
                }),
                ..Default::default()
            },
        )
        .expect("tool choice must be accepted");
        assert_eq!(body["tool_choice"]["type"], serde_json::json!("function"));
        assert_eq!(
            body["tool_choice"]["name"],
            serde_json::json!("get_weather")
        );
        assert!(
            body["tool_choice"].get("function").is_none(),
            "Responses names a function flat"
        );
    }

    #[test]
    fn function_call_arguments_delta_and_done_share_index() {
        let codec = OpenaiResponseCodec::new();

        let delta = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":"}"#;
        match codec
            .parse_stream_chunk_events(delta)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 0);
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"#));
                assert!(!d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }

        let done = r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"city\":\"Beijing\"}"}"#;
        match codec
            .parse_stream_chunk_events(done)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 0, "fragments must land on the same call");
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
                assert!(d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn output_item_added_announces_the_function_name() {
        let codec = OpenaiResponseCodec::new();
        let added = r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#;
        match codec
            .parse_stream_chunk_events(added)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 1);
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.name.as_deref(), Some("get_weather"));
                assert_eq!(d.arguments, None);
                assert!(!d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn terminal_event_emits_usage_and_ends_the_turn() {
        let codec = OpenaiResponseCodec::new();

        let completed = r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"output_tokens_details":{"reasoning_tokens":2},"input_tokens_details":{"cached_tokens":3,"cache_write_tokens":4}}}}"#;
        let events = codec
            .parse_stream_chunk_events(completed)
            .expect("completed chunk parses");
        assert_eq!(events.len(), 2, "usage and terminal event");
        match &events[0] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.prompt_tokens, 10);
                assert_eq!(u.usage.completion_tokens, 5);
                assert_eq!(u.usage.total_tokens, 15);
                assert_eq!(u.usage.reasoning_tokens, Some(2));
                assert_eq!(u.usage.cache_read_tokens, Some(3));
                assert_eq!(u.usage.cache_write_tokens, Some(4));
            }
            other => panic!("expected Usage, got {other:?}"),
        }
        assert!(
            matches!(events[1], MessageStreamEvent::End(_)),
            "the turn must end even when usage is present"
        );

        let without_usage = codec
            .parse_stream_chunk_events(r#"{"type":"response.completed"}"#)
            .expect("completed without usage parses");
        assert!(matches!(without_usage[0], MessageStreamEvent::End(_)));
    }

    #[test]
    fn created_event_emits_stream_identity() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk_events(
                r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-4o"}}"#,
            )
            .expect("created chunk parses")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::Start(s)) => {
                assert_eq!(s.id.as_deref(), Some("resp_1"));
                assert_eq!(s.model.as_deref(), Some("gpt-4o"));
            }
            other => panic!("expected Start, got {other:?}"),
        }
    }

    #[test]
    fn multiple_call_deltas_get_distinct_indices() {
        let codec = OpenaiResponseCodec::new();
        let chunk = |index: usize| {
            format!(
                r#"{{"type":"response.function_call_arguments.delta","item_id":"fc_{index}","output_index":{index},"delta":"x"}}"#
            )
        };
        let i1 = match codec
            .parse_stream_chunk_events(&chunk(0))
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        let i2 = match codec
            .parse_stream_chunk_events(&chunk(1))
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_ne!(i1, i2);
        // The same output index always identifies the same call.
        let again = match codec
            .parse_stream_chunk_events(&chunk(0))
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_eq!(again, i1);
    }

    #[test]
    fn the_done_sentinel_ends_the_stream() {
        // The official Responses stream closes with a `[DONE]` sentinel that
        // is not JSON.
        let codec = OpenaiResponseCodec::new();
        let events = codec
            .parse_stream_chunk_events("[DONE]")
            .expect("the sentinel must not be a parse error");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], MessageStreamEvent::End(_)));
    }

    #[test]
    fn parse_response_extracts_text_output_and_usage() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id": "resp_1",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "the answer"}]
            }],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 2}
            }
        }"#;
        let result = codec
            .parse_response(
                body,
                &LlmRequest {
                    profile_id: "p".to_string(),
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
                },
            )
            .unwrap();
        assert_eq!(result.content.as_deref(), Some("the answer"));
        assert_eq!(
            result.finish_reason.as_deref(),
            Some("stop"),
            "the stop value uses the same vocabulary as the Chat Completions codec"
        );
        assert_eq!(
            result.metadata.as_ref().unwrap()["status"],
            serde_json::json!("completed"),
            "the raw response status stays available"
        );
        assert_eq!(result.usage.as_ref().unwrap().total_tokens, 20);
        assert_eq!(result.usage.as_ref().unwrap().cache_read_tokens, Some(5));
        assert_eq!(result.usage.as_ref().unwrap().cache_write_tokens, Some(2));
        assert_eq!(result.model, "gpt-4o");
    }

    #[test]
    fn parse_response_extracts_function_calls() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id": "resp_2",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": ""}]},
                {"type": "function_call", "id": "fc_7", "call_id": "call_7", "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
            ]
        }"#;
        let result = codec
            .parse_response(
                body,
                &LlmRequest {
                    profile_id: "p".to_string(),
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
                },
            )
            .unwrap();
        let calls = result.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "fc_7");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn parse_response_keeps_message_text_when_function_call_follows() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id": "resp_4",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "thinking aloud"}]},
                {"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
            ]
        }"#;
        let result = codec
            .parse_response(
                body,
                &LlmRequest {
                    profile_id: "p".to_string(),
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
                },
            )
            .unwrap();
        assert_eq!(result.content.as_deref(), Some("thinking aloud"));
        assert_eq!(
            result.tool_calls.as_ref().unwrap()[0].function.name,
            "get_weather"
        );
    }

    #[test]
    fn parse_response_tolerates_empty_output() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{"id":"resp_3","object":"response","model":"gpt-4o","status":"incomplete"}"#;
        let result = codec
            .parse_response(
                body,
                &LlmRequest {
                    profile_id: "p".to_string(),
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
                },
            )
            .unwrap();
        assert_eq!(result.content, None);
        assert_eq!(result.tool_calls, None);
        assert_eq!(result.usage, None);
    }

    fn count_profile() -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format: llm_suite_types::llm::LlmFormat::OpenaiResponse,
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
        }
    }

    fn count_request() -> LlmRequest {
        LlmRequest {
            profile_id: "p1".to_string(),
            messages: vec![llm_suite_types::message::Message {
                id: llm_suite_types::Id::new(),
                role: llm_suite_types::message::MessageRole::User,
                content: llm_suite_types::message::MessageContentValue::Text("hello".to_string()),
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
        }
    }

    fn request_body(req: &reqwest::Request) -> serde_json::Value {
        req.body()
            .unwrap()
            .as_bytes()
            .map(|b| serde_json::from_slice(b).unwrap())
            .unwrap()
    }

    #[test]
    fn the_request_carries_no_parameter_the_api_does_not_know() {
        // The Responses API has no `seed`; only Chat Completions does.
        let codec = OpenaiResponseCodec::new();
        let mut request = count_request();
        request.generation = Some(llm_suite_types::llm::generation::LlmGenerationParams {
            seed: Some(42),
            ..Default::default()
        });
        let built = codec
            .build_request(&request, &count_profile())
            .expect("must build");
        let body = request_body(&built);
        assert!(
            body.get("seed").is_none(),
            "the Responses API rejects unknown request fields: {body:?}"
        );
    }

    #[test]
    fn the_json_schema_format_is_flat() {
        // The Responses API keeps name, schema and strict next to the type,
        // while Chat Completions nests them under `json_schema`.
        let codec = OpenaiResponseCodec::new();
        let mut request = count_request();
        request.generation = Some(llm_suite_types::llm::generation::LlmGenerationParams {
            response_format: Some(llm_suite_types::llm::generation::LlmResponseFormat {
                kind: Some(llm_suite_types::llm::generation::ResponseFormatKind::JsonSchema),
                name: Some("answer".to_string()),
                schema: Some(serde_json::json!({"type": "object"})),
                strict: Some(true),
            }),
            ..Default::default()
        });
        let built = codec
            .build_request(&request, &count_profile())
            .expect("must build");
        let body = request_body(&built);
        let format = &body["text"]["format"];
        assert_eq!(format["type"], serde_json::json!("json_schema"));
        assert_eq!(format["name"], serde_json::json!("answer"));
        assert_eq!(format["strict"], serde_json::json!(true));
        assert!(format.get("json_schema").is_none(), "{body:?}");
    }

    #[test]
    fn rich_image_blocks_map_to_input_image_parts() {
        let codec = OpenaiResponseCodec::new();
        let req = LlmRequest {
            profile_id: "p1".to_string(),
            messages: vec![llm_suite_types::message::Message {
                id: llm_suite_types::Id::new(),
                role: llm_suite_types::message::MessageRole::User,
                content: llm_suite_types::message::MessageContentValue::Rich(vec![
                    llm_suite_types::message::MessageContent::Text {
                        text: "describe".to_string(),
                    },
                    llm_suite_types::message::MessageContent::ImageUrl {
                        image_url: llm_suite_types::message::ImageUrlContent {
                            url: "https://example.com/a.png".to_string(),
                            detail: Some("auto".to_string()),
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
        let entries = codec.convert_messages(&req.messages);
        let parts = entries[0]["content"].as_array().expect("content array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["image_url"], "https://example.com/a.png");
        assert_eq!(parts[1]["detail"], "auto");
    }

    #[test]
    fn count_tokens_request_targets_input_tokens_endpoint() {
        let codec = OpenaiResponseCodec::new();
        let req = codec
            .build_count_tokens_request(&count_request(), &count_profile())
            .expect("count request must build")
            .expect("responses provider supports counting");
        assert_eq!(
            req.url().as_str(),
            "https://api.openai.com/v1/responses/input_tokens"
        );
        assert_eq!(
            req.headers().get("Authorization").unwrap(),
            "Bearer sk-test"
        );
        let body = request_body(&req);
        assert_eq!(body["model"], serde_json::json!("gpt-4o"));
        assert_eq!(body["input"][0]["content"], serde_json::json!("hello"));
        assert!(
            body.get("stream").is_none(),
            "count body must not carry stream flags"
        );
    }

    #[test]
    fn count_tokens_body_carries_native_tools() {
        let codec = OpenaiResponseCodec::new();
        let mut req = count_request();
        req.tool_call_protocol = Some(llm_suite_types::llm::ToolCallProtocol::Native);
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()]);
        let body = codec
            .build_count_tokens_body(&req, &count_profile())
            .unwrap();
        assert_eq!(body["tools"][0]["name"], serde_json::json!("get_weather"));
        assert_eq!(body["tools"][0]["type"], serde_json::json!("function"));
    }

    #[test]
    fn count_tokens_body_text_mode_inlines_tools_into_system() {
        let codec = OpenaiResponseCodec::new();
        let mut req = count_request();
        req.messages.insert(
            0,
            llm_suite_types::message::Message {
                id: llm_suite_types::Id::new(),
                role: llm_suite_types::message::MessageRole::System,
                content: llm_suite_types::message::MessageContentValue::Text(
                    "You are a helper".to_string(),
                ),
                timestamp: 0,
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                thinking: None,
                thinking_signature: None,
                metadata: None,
            },
        );
        req.tool_call_protocol = Some(llm_suite_types::llm::ToolCallProtocol::Xml);
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()]);
        let body = codec
            .build_count_tokens_body(&req, &count_profile())
            .unwrap();
        assert!(
            body.get("tools").is_none(),
            "text mode must not send native tools"
        );
        let system = body["instructions"].as_str().unwrap();
        assert!(system.contains("get_weather"), "system must declare tools");
    }

    #[test]
    fn count_tokens_response_parses_input_tokens() {
        let codec = OpenaiResponseCodec::new();
        let body: serde_json::Value =
            serde_json::from_str(r#"{"object":"response.input_tokens","input_tokens":13}"#)
                .unwrap();
        assert_eq!(codec.parse_count_tokens_response(&body).unwrap(), 13);
        let empty: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(codec.parse_count_tokens_response(&empty).unwrap(), 0);
    }

    fn minimal_request() -> LlmRequest {
        LlmRequest {
            profile_id: "p".to_string(),
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

    #[test]
    fn convert_messages_emits_function_call_and_output_items() {
        let codec = OpenaiResponseCodec::new();
        let assistant = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Assistant,
            content: llm_suite_types::message::MessageContentValue::Text(
                "let me check".to_string(),
            ),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: Some(vec![llm_suite_types::message::LlmToolCall {
                id: "fc_1".to_string(),
                r#type: "function".to_string(),
                call_id: Some("call_abc".to_string()),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "get_weather".to_string(),
                    arguments: "{\"city\":\"Beijing\"}".to_string(),
                },
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let tool_result = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Tool,
            content: llm_suite_types::message::MessageContentValue::Text("sunny".to_string()),
            timestamp: 0,
            tool_call_id: Some("call_abc".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let items = codec.convert_messages(&[assistant, tool_result]);

        // assistant text item + function_call item + function_call_output item
        assert_eq!(items.len(), 3);

        let func_call = items
            .iter()
            .find(|i| i.get("type").and_then(|v| v.as_str()) == Some("function_call"))
            .expect("function_call item present");
        assert_eq!(func_call["call_id"], serde_json::json!("call_abc"));
        assert_eq!(func_call["name"], serde_json::json!("get_weather"));
        assert_eq!(
            func_call["arguments"],
            serde_json::json!("{\"city\":\"Beijing\"}")
        );
        assert!(
            func_call.get("tool_calls").is_none(),
            "must not emit Chat-style tool_calls"
        );

        let func_output = items
            .iter()
            .find(|i| i.get("type").and_then(|v| v.as_str()) == Some("function_call_output"))
            .expect("function_call_output item present");
        assert_eq!(func_output["call_id"], serde_json::json!("call_abc"));
        assert_eq!(func_output["output"], serde_json::json!("sunny"));
    }

    #[test]
    fn parse_response_captures_call_id() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id": "resp_2", "object":"response", "model":"gpt-4o", "status":"completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":""}]},
                {"type":"function_call","id":"fc_7","call_id":"call_7","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}
            ]
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        let calls = result.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id.as_deref(), Some("call_7"));
    }

    #[test]
    fn a_tool_calling_turn_reports_the_tool_call_stop_reason() {
        // The Responses API has no tool-call stop reason; the unified
        // vocabulary the Chat codec reports must still be used.
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}
            ]
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("tool_calls"));

        let plain = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hi"}]}
            ]
        }"#;
        let result = codec.parse_response(plain, &minimal_request()).unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn the_provider_total_token_count_is_preferred() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
            "usage":{"input_tokens":10,"output_tokens":5,"total_tokens":21}
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        let usage = result.usage.expect("usage must be present");
        assert_eq!(usage.total_tokens, 21, "the provider owns the total");

        // Without the provider total the sum is the fallback.
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
            "usage":{"input_tokens":10,"output_tokens":5}
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        let usage = result.usage.expect("usage must be present");
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn the_sources_of_a_grounded_answer_are_reported_as_metadata() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"grounded","annotations":[
                {"type":"url_citation","url":"https://example.com","title":"Example"}
            ]}]}]
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        let metadata = result.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["annotations"][0]["url"],
            serde_json::json!("https://example.com")
        );
    }

    #[test]
    fn the_system_prompt_travels_in_the_instructions_field() {
        let codec = OpenaiResponseCodec::new();
        let mut request = minimal_request();
        request.messages.push(llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::System,
            content: llm_suite_types::message::MessageContentValue::Text(
                "You are a helper".to_string(),
            ),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        });
        request.messages.push(llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::User,
            content: llm_suite_types::message::MessageContentValue::Text("hi".to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        });

        let body = codec
            .build_body(&request, &count_profile())
            .expect("must build");
        assert_eq!(body["instructions"], serde_json::json!("You are a helper"));
        let input = body["input"].as_array().expect("input");
        assert!(
            input
                .iter()
                .all(|item| item["role"] != serde_json::json!("system")),
            "the system prompt must not be an input item: {body}"
        );
        assert_eq!(input.len(), 1);

        // The counting request carries the same instructions.
        let counted = codec
            .build_count_tokens_body(&request, &count_profile())
            .expect("must build");
        assert_eq!(
            counted["instructions"],
            serde_json::json!("You are a helper")
        );
    }

    #[test]
    fn the_stream_reports_the_sources_of_a_grounded_answer() {
        let codec = OpenaiResponseCodec::new();
        let chunk = r#"{"type":"response.output_text.annotation.added","output_index":0,"content_index":0,"item_id":"msg_1","annotation_index":0,"annotation":{"type":"url_citation","url":"https://example.com"}}"#;
        match codec
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::Annotation(annotation)) => {
                assert_eq!(
                    annotation.annotation["url"],
                    serde_json::json!("https://example.com")
                );
            }
            other => panic!("expected Annotation, got {other:?}"),
        }
    }

    #[test]
    fn the_finished_output_item_completes_the_tool_call() {
        let codec = OpenaiResponseCodec::new();
        let chunk = r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}"#;
        match codec
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(delta)) => {
                assert_eq!(delta.index, 0);
                assert_eq!(delta.id.as_deref(), Some("fc_1"));
                assert_eq!(delta.call_id.as_deref(), Some("call_1"));
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                assert_eq!(delta.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
                assert!(delta.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_aggregates_multiple_text_blocks() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"first part"},{"type":"output_text","text":" second part"}]},
                {"type":"message","content":[{"type":"output_text","text":" and more"}]}
            ]
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        assert_eq!(
            result.content.as_deref(),
            Some("first part second part and more")
        );
    }

    #[test]
    fn parse_response_captures_refusal_and_reasoning() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"reasoning","summary":[{"type":"summary_text","text":"I am thinking..."}]},
                {"type":"message","content":[
                    {"type":"output_text","text":"hello"},
                    {"type":"refusal","refusal":"I cannot help with that."}
                ]}
            ]
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.content.as_deref(), Some("hello"));
        assert_eq!(result.refusal.as_deref(), Some("I cannot help with that."));
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("I am thinking...")
        );
        assert_eq!(
            result.message.thinking.as_deref(),
            Some("I am thinking..."),
            "the message carries the reasoning so a follow-up turn keeps it"
        );
    }

    #[test]
    fn parse_response_reports_the_reasoning_tokens_of_the_usage() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 12}
            }
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.reasoning_tokens, Some(12));
        assert_eq!(result.usage.as_ref().unwrap().reasoning_tokens, Some(12));
    }

    #[test]
    fn an_incomplete_response_reports_why_it_stopped() {
        let codec = OpenaiResponseCodec::new();
        let cases = [
            ("max_output_tokens", "length"),
            ("max_messages", "length"),
            ("content_filter", "content_filter"),
        ];
        for (reason, expected) in cases {
            let body = format!(
                r#"{{"id":"resp_x","object":"response","model":"gpt-4o","status":"incomplete",
                    "incomplete_details":{{"reason":"{reason}"}},
                    "output": []}}"#
            );
            let result = codec.parse_response(&body, &minimal_request()).unwrap();
            assert_eq!(
                result.finish_reason.as_deref(),
                Some(expected),
                "{reason} must map to {expected}"
            );
            assert_eq!(
                result.metadata.as_ref().unwrap()["incomplete_reason"],
                serde_json::json!(reason)
            );
            assert_eq!(
                result.metadata.as_ref().unwrap()["status"],
                serde_json::json!("incomplete")
            );
        }
    }

    #[test]
    fn parse_response_failed_status_returns_error() {
        let codec = OpenaiResponseCodec::new();
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"failed",
            "error":{"message":"rate limit exceeded"},
            "output": []
        }"#;
        let err = codec
            .parse_response(body, &minimal_request())
            .expect_err("failed response must surface an error");
        assert!(format!("{err:?}").contains("rate limit exceeded"));
    }

    #[test]
    fn stream_text_delta_emits_the_increment() {
        let codec = OpenaiResponseCodec::new();
        // The running snapshot is owned by the stream accumulator, not by the
        // codec: the codec reports the raw increment only.
        match codec
            .parse_stream_chunk_events(r#"{"type":"response.output_text.delta","delta":"Hello "}"#)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::Text(t) => {
                assert_eq!(t.text, "Hello ");
                assert_eq!(t.snapshot, "");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn stream_reasoning_delta_emits_reasoning_text() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk_events(
                r#"{"type":"response.reasoning_text.delta","delta":"thinking"}"#,
            )
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ReasoningText(r) => {
                assert_eq!(r.reasoning, "thinking");
                assert_eq!(r.snapshot, "");
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }

    #[test]
    fn stream_refusal_delta_emits_refusal() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk_events(r#"{"type":"response.refusal.delta","delta":"I cannot."}"#)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::Refusal(r) => assert_eq!(r.refusal, "I cannot."),
            other => panic!("expected Refusal, got {other:?}"),
        }
    }

    #[test]
    fn stream_function_call_arguments_done_emits_snapshot() {
        let codec = OpenaiResponseCodec::new();
        let delta = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":"}"#;
        codec.parse_stream_chunk_events(delta).unwrap();
        let done = r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"city\":\"Beijing\"}"}"#;
        match codec
            .parse_stream_chunk_events(done)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => {
                assert!(d.is_snapshot, "done must emit a full snapshot");
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn stream_failed_emits_error_event() {
        let codec = OpenaiResponseCodec::new();
        let chunk = r#"{"type":"response.failed","response":{"status":"failed","error":{"message":"boom","code":"server_error"}}}"#;
        let events = codec
            .parse_stream_chunk_events(chunk)
            .expect("failed chunk parses");
        assert_eq!(events.len(), 2, "the error and the terminal event");
        match &events[0] {
            MessageStreamEvent::Error(e) => assert_eq!(e.error, "boom"),
            other => panic!("expected Error, got {other:?}"),
        }
        assert!(
            matches!(events[1], MessageStreamEvent::End(_)),
            "a failed turn must end the stream"
        );
    }
}
