use super::LlmCodec;
use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent,
};
use llm_suite_types::tool::Tool;
use reqwest::Method;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct OpenaiResponseCodec {
    base_url: String,
    /// Reused across requests; constructing a `reqwest::Client` per call would
    /// discard the connection pool and TLS session cache every time.
    client: reqwest::Client,
    /// Maps streaming `item_id` to the accumulated tool call index so
    /// `function_call_arguments.delta` fragments land on the same call as the
    /// final `function_call_arguments.done` snapshot.
    call_indices: Mutex<HashMap<String, usize>>,
    next_index: Mutex<usize>,
    /// Accumulated assistant text across `output_text.delta` chunks so the
    /// streamed `Text` event carries a running snapshot (matching the official
    /// SDK's accumulator). Reset on `response.created`.
    text_snapshot: Mutex<String>,
    /// Accumulated reasoning text across `reasoning_text.delta` chunks.
    reasoning_snapshot: Mutex<String>,
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
            call_indices: Mutex::new(HashMap::new()),
            next_index: Mutex::new(0),
            text_snapshot: Mutex::new(String::new()),
            reasoning_snapshot: Mutex::new(String::new()),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
            call_indices: Mutex::new(HashMap::new()),
            next_index: Mutex::new(0),
            text_snapshot: Mutex::new(String::new()),
            reasoning_snapshot: Mutex::new(String::new()),
        }
    }

    fn call_index_for(&self, item_id: &str) -> usize {
        let mut indices = llm_suite_common::lock::lock_ok(self.call_indices.lock());
        if let Some(idx) = indices.get(item_id) {
            return *idx;
        }
        let mut next = llm_suite_common::lock::lock_ok(self.next_index.lock());
        let idx = *next;
        *next += 1;
        indices.insert(item_id.to_string(), idx);
        idx
    }

    fn convert_messages(
        &self,
        messages: &[llm_suite_types::message::Message],
    ) -> Vec<serde_json::Value> {
        let mut items: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            let role = match msg.role {
                llm_suite_types::message::MessageRole::System => "system",
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
                    if let Some(ref call_id) = msg.tool_call_id {
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

    fn response_tool_defs(tools: &[Tool]) -> Vec<serde_json::Value> {
        tools
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
            .collect()
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

        let mut body = serde_json::json!({
            "model": profile.model,
            "input": self.convert_messages(&request.messages),
        });

        if use_text_mode {
            let history = super::shared::convert_history_for_text_mode(&request.messages, request);
            body["input"] = serde_json::json!(self.convert_messages(&history));
            let system = super::shared::text_mode_system_content(request);
            if !system.is_empty() {
                let mut input = body["input"].take();
                if let Some(arr) = input.as_array_mut() {
                    arr.insert(0, serde_json::json!({"role": "system", "content": system}));
                }
                body["input"] = input;
            }
        } else if let Some(tools) = &request.tools {
            body["tools"] = serde_json::json!(Self::response_tool_defs(tools));
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
}

impl LlmCodec for OpenaiResponseCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        let url = format!(
            "{}/responses",
            profile.base_url.as_deref().unwrap_or(&self.base_url)
        );

        let use_text_mode = super::shared::is_text_mode(request);

        let mut body = serde_json::json!({
            "model": profile.model,
            "input": self.convert_messages(&request.messages),
        });

        if use_text_mode {
            let history = super::shared::convert_history_for_text_mode(&request.messages, request);
            body["input"] = serde_json::json!(self.convert_messages(&history));
            let system = super::shared::text_mode_system_content(request);
            if !system.is_empty() {
                let mut input = body["input"].take();
                if let Some(arr) = input.as_array_mut() {
                    arr.insert(0, serde_json::json!({"role": "system", "content": system}));
                }
                body["input"] = input;
            }
        }

        if request.stream == Some(true) {
            body["stream"] = serde_json::json!(true);
        }

        let generation = super::shared::resolve_generation(request, profile)?;
        crate::generation::apply_openai_responses(&mut body, &generation);
        super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

        if !use_text_mode {
            if let Some(tools) = &request.tools {
                body["tools"] = serde_json::json!(Self::response_tool_defs(tools));
            }
        }

        super::shared::apply_custom_body(&mut body, profile);

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
        let url = format!(
            "{}/responses/input_tokens",
            profile.base_url.as_deref().unwrap_or(&self.base_url)
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

    fn parse_stream_chunk(&self, data: &str) -> LlmResult<Option<MessageStreamEvent>> {
        if data.is_empty() {
            return Ok(None);
        }

        let json: serde_json::Value = serde_json::from_str(data)?;

        let event_type = json.get("type").and_then(|v| v.as_str());
        match event_type {
            Some("response.created") => {
                // Stream identity; the accumulator records it for the final
                // message and forwards it once. Reset accumulated snapshots so
                // a reused codec doesn't carry text from a previous request.
                *llm_suite_common::lock::lock_ok(self.text_snapshot.lock()) = String::new();
                *llm_suite_common::lock::lock_ok(self.reasoning_snapshot.lock()) = String::new();
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
                    return Ok(Some(MessageStreamEvent::Start(
                        llm_suite_types::llm::MessageStreamStart { id, model },
                    )));
                }
            }
            Some("response.output_text.delta") => {
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    let mut snap = llm_suite_common::lock::lock_ok(self.text_snapshot.lock());
                    snap.push_str(delta);
                    return Ok(Some(MessageStreamEvent::Text(
                        llm_suite_types::llm::MessageStreamText {
                            snapshot: snap.clone(),
                            text: delta.to_string(),
                        },
                    )));
                }
            }
            Some("response.reasoning_text.delta")
            | Some("response.reasoning_summary_text.delta") => {
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    let mut snap = llm_suite_common::lock::lock_ok(self.reasoning_snapshot.lock());
                    snap.push_str(delta);
                    return Ok(Some(MessageStreamEvent::ReasoningText(
                        llm_suite_types::llm::MessageStreamReasoning {
                            reasoning: delta.to_string(),
                            snapshot: snap.clone(),
                            signature: None,
                            signature_snapshot: None,
                        },
                    )));
                }
            }
            Some("response.refusal.delta") => {
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    let mut snap = llm_suite_common::lock::lock_ok(self.text_snapshot.lock());
                    snap.push_str(delta);
                    return Ok(Some(MessageStreamEvent::Text(
                        llm_suite_types::llm::MessageStreamText {
                            snapshot: snap.clone(),
                            text: delta.to_string(),
                        },
                    )));
                }
            }
            Some("response.function_call_arguments.delta") => {
                // Incremental arguments fragment; grouped by item_id onto a
                // stable index (mirrors the toolCallsDelta accumulation).
                let item_id = json
                    .get("item_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                    let index = self.call_index_for(&item_id);
                    return Ok(Some(MessageStreamEvent::ToolCallDelta(
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id: None,
                            name: None,
                            arguments: Some(delta.to_string()),
                            is_snapshot: false,
                        },
                    )));
                }
            }
            Some("response.function_call_arguments.done") => {
                // Complete snapshot: name + full arguments, replacing the
                // accumulated fragments on the same index. The official event
                // name is `function_call_arguments.done` (not
                // `function_call.done`).
                let item = json.get("item");
                let item_id = item
                    .and_then(|i| i.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let index = self.call_index_for(&item_id);
                let name = item
                    .and_then(|i| i.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let arguments = item
                    .and_then(|i| i.get("arguments"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                return Ok(Some(MessageStreamEvent::ToolCallDelta(
                    llm_suite_types::llm::MessageStreamToolCallDelta {
                        index,
                        id: Some(item_id),
                        name,
                        arguments,
                        is_snapshot: true,
                    },
                )));
            }
            Some("response.failed") | Some("response.error") => {
                let message = json
                    .get("error")
                    .and_then(|e| match e {
                        serde_json::Value::String(s) => Some(s.clone()),
                        _ => e.get("message").and_then(|v| v.as_str()).map(String::from),
                    })
                    .unwrap_or_else(|| "stream error".to_string());
                tracing::warn!(
                    event = event_type.unwrap_or(""),
                    "Responses stream ended with failure/error event",
                );
                return Ok(Some(MessageStreamEvent::Error(
                    llm_suite_types::llm::MessageStreamError { error: message },
                )));
            }
            Some("response.completed") | Some("response.incomplete") => {
                // The completed event carries the final usage summary.
                if let Some(usage) = json.get("usage") {
                    return Ok(Some(MessageStreamEvent::Usage(
                        llm_suite_types::llm::MessageStreamUsage {
                            usage: llm_suite_types::llm::TokenUsageStats {
                                prompt_tokens: usage
                                    .get("input_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as u32,
                                completion_tokens: usage
                                    .get("output_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as u32,
                                total_tokens: usage
                                    .get("total_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as u32,
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
                    )));
                }
                return Ok(Some(MessageStreamEvent::End(
                    llm_suite_types::llm::MessageStreamEnd {},
                )));
            }
            other => {
                tracing::debug!(
                    event_type = other.unwrap_or(""),
                    "unhandled Responses stream event; ignoring",
                );
            }
        }

        Ok(None)
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
        if let Some(output) = json.get("output").and_then(|v| v.as_array()) {
            for item in output {
                match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "message" => {
                        if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                            for block in blocks {
                                if block.get("type").and_then(|t| t.as_str()) == Some("output_text")
                                {
                                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                        text_parts.push(t.to_string());
                                    }
                                }
                            }
                        }
                    }
                    "refusal" => {
                        if let Some(t) = item.get("refusal").and_then(|v| v.as_str()) {
                            refusal_parts.push(t.to_string());
                        }
                    }
                    "reasoning" => {
                        if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                            for block in blocks {
                                match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                                    "text" | "summary_text" => {
                                        if let Some(t) = block.get("text").and_then(|v| v.as_str())
                                        {
                                            reasoning_parts.push(t.to_string());
                                        }
                                    }
                                    _ => {}
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
                total_tokens: input_tokens + output_tokens,
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
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        // Provider-native status passed through untouched (`completed` /
        // `incomplete` / `failed`); never normalized here.
        let finish_reason = status;

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
            reasoning_tokens: None,
            metadata: None,
            refusal,
            stream_stats: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call_arguments_delta_and_done_share_index() {
        let codec = OpenaiResponseCodec::new();

        let delta = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"city\":"}"#;
        match codec.parse_stream_chunk(delta).expect("chunk must parse") {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 0);
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"#));
                assert!(!d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }

        let done = r#"{"type":"response.function_call_arguments.done","item":{"type":"function_call","id":"fc_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}"#;
        match codec.parse_stream_chunk(done).expect("chunk must parse") {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 0, "fragments must land on the same call");
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.name.as_deref(), Some("get_weather"));
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
                assert!(d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn stream_chunks_cover_text_completion_and_usage() {
        let codec = OpenaiResponseCodec::new();

        match codec
            .parse_stream_chunk(r#"{"type":"response.output_text.delta","delta":"Hello "}"#)
            .expect("text chunk parses")
        {
            Some(MessageStreamEvent::Text(t)) => assert_eq!(t.text, "Hello "),
            other => panic!("expected Text, got {other:?}"),
        }

        match codec
            .parse_stream_chunk(
                r#"{"type":"response.completed","response":{"status":"completed"},"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"output_tokens_details":{"reasoning_tokens":2},"input_tokens_details":{"cached_tokens":3,"cache_write_tokens":4}}}"#,
            )
            .expect("completed chunk parses")
        {
            Some(MessageStreamEvent::Usage(u)) => {
                assert_eq!(u.usage.prompt_tokens, 10);
                assert_eq!(u.usage.completion_tokens, 5);
                assert_eq!(u.usage.total_tokens, 15);
                assert_eq!(u.usage.reasoning_tokens, Some(2));
                assert_eq!(u.usage.cache_read_tokens, Some(3));
                assert_eq!(u.usage.cache_write_tokens, Some(4));
            }
            other => panic!("expected Usage, got {other:?}"),
        }

        match codec
            .parse_stream_chunk(r#"{"type":"response.completed"}"#)
            .expect("completed without usage parses")
        {
            Some(MessageStreamEvent::End(_)) => {}
            other => panic!("expected End, got {other:?}"),
        }

        assert!(codec.parse_stream_chunk("").unwrap().is_none());
        assert!(codec
            .parse_stream_chunk(r#"{"type":"unknown"}"#)
            .unwrap()
            .is_none());
    }

    #[test]
    fn created_event_emits_stream_identity() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk(
                r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-4o"}}"#,
            )
            .expect("created chunk parses")
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
        let chunk = |id: &str| {
            format!(
                r#"{{"type":"response.function_call_arguments.delta","item_id":"{id}","delta":"x"}}"#
            )
        };
        let i1 = match codec.parse_stream_chunk(&chunk("fc_a")).unwrap().unwrap() {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        let i2 = match codec.parse_stream_chunk(&chunk("fc_b")).unwrap().unwrap() {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_ne!(i1, i2);
        // Repeating the same item_id keeps its index.
        let again = match codec.parse_stream_chunk(&chunk("fc_a")).unwrap().unwrap() {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_eq!(again, i1);
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
        assert_eq!(result.finish_reason.as_deref(), Some("completed"));
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
        assert_eq!(
            body["tools"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
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
        let system = body["input"][0]["content"].as_str().unwrap();
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
                {"type":"reasoning","content":[{"type":"text","text":"I am thinking..."}]},
                {"type":"message","content":[{"type":"output_text","text":"hello"}]},
                {"type":"refusal","refusal":"I cannot help with that."}
            ]
        }"#;
        let result = codec.parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.content.as_deref(), Some("hello"));
        assert_eq!(result.refusal.as_deref(), Some("I cannot help with that."));
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("I am thinking...")
        );
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
    fn stream_text_delta_carries_accumulated_snapshot() {
        let codec = OpenaiResponseCodec::new();
        // created resets the snapshot (no identity present -> None, which is fine)
        let _ = codec.parse_stream_chunk(r#"{"type":"response.created"}"#);
        match codec
            .parse_stream_chunk(r#"{"type":"response.output_text.delta","delta":"Hello "}"#)
            .unwrap()
            .unwrap()
        {
            MessageStreamEvent::Text(t) => {
                assert_eq!(t.text, "Hello ");
                assert_eq!(t.snapshot, "Hello ");
            }
            other => panic!("expected Text, got {other:?}"),
        }
        match codec
            .parse_stream_chunk(r#"{"type":"response.output_text.delta","delta":"world"}"#)
            .unwrap()
            .unwrap()
        {
            MessageStreamEvent::Text(t) => {
                assert_eq!(t.text, "world");
                assert_eq!(t.snapshot, "Hello world");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn stream_reasoning_delta_emits_reasoning_text() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk(r#"{"type":"response.reasoning_text.delta","delta":"thinking"}"#)
            .unwrap()
            .unwrap()
        {
            MessageStreamEvent::ReasoningText(r) => {
                assert_eq!(r.reasoning, "thinking");
                assert_eq!(r.snapshot, "thinking");
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }

    #[test]
    fn stream_refusal_delta_emits_text() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk(r#"{"type":"response.refusal.delta","delta":"I cannot."}"#)
            .unwrap()
            .unwrap()
        {
            MessageStreamEvent::Text(t) => assert_eq!(t.text, "I cannot."),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn stream_function_call_arguments_done_emits_snapshot() {
        let codec = OpenaiResponseCodec::new();
        let delta = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"city\":"}"#;
        codec.parse_stream_chunk(delta).unwrap();
        let done = r#"{"type":"response.function_call_arguments.done","item":{"type":"function_call","id":"fc_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}"#;
        match codec.parse_stream_chunk(done).unwrap().unwrap() {
            MessageStreamEvent::ToolCallDelta(d) => {
                assert!(d.is_snapshot, "done must emit a full snapshot");
                assert_eq!(d.name.as_deref(), Some("get_weather"));
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn stream_failed_emits_error_event() {
        let codec = OpenaiResponseCodec::new();
        match codec
            .parse_stream_chunk(r#"{"type":"response.failed","error":{"message":"boom"}}"#)
            .unwrap()
            .unwrap()
        {
            MessageStreamEvent::Error(e) => assert_eq!(e.error, "boom"),
            other => panic!("expected Error, got {other:?}"),
        }
    }
}
