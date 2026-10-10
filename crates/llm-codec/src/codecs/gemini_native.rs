use super::LlmCodec;
use crate::error::LlmResult;
use llm_types::llm::{LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent};
use llm_types::tool::Tool;
use reqwest::Method;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Globally unique tool call indices for Gemini streams (each `functionCall`
/// part is a complete snapshot; unique indices keep separate calls apart in
/// the accumulator).
static GEMINI_CALL_INDEX: AtomicUsize = AtomicUsize::new(0);

pub struct GeminiNativeCodec {
    base_url: String,
}

impl Default for GeminiNativeCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiNativeCodec {
    pub fn new() -> Self {
        Self {
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self { base_url }
    }

    /// Effective API endpoint base.
    fn endpoint_base<'a>(&'a self, profile: &'a LlmProfile) -> &'a str {
        profile.base_url.as_deref().unwrap_or(&self.base_url)
    }

    /// Build a POST request against `url` with a JSON body, then apply the
    /// profile auth/headers. Auth is header-only (`x-goog-api-key` for
    /// native Gemini auth); the API key is never placed in the URL query.
    /// A process-wide shared client is used purely to assemble the request;
    /// execution happens on the caller's client with connection pooling.
    fn build_json_request(
        &self,
        url: String,
        body: &serde_json::Value,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        static BUILD_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
        let build_client = BUILD_CLIENT.get_or_init(reqwest::Client::new);

        let payload = serde_json::to_vec(body).map_err(|e| {
            crate::error::LlmError::ConfigError(format!("failed to serialize request body: {}", e))
        })?;

        let builder = build_client
            .request(Method::POST, &url)
            .header("Content-Type", "application/json")
            .body(payload);

        let builder = super::shared::apply_auth_and_headers(builder, profile, "native");

        builder
            .build()
            .map_err(crate::error::LlmError::HttpError)
    }

    fn convert_messages(&self, messages: &[llm_types::message::Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .filter_map(|msg| {
                // Gemini only accepts "user" and "model" roles; tool results
                // ride in a user message as `functionResponse` parts.
                let role = match msg.role {
                    llm_types::message::MessageRole::System => return None,
                    llm_types::message::MessageRole::User => "user",
                    llm_types::message::MessageRole::Assistant => "model",
                    llm_types::message::MessageRole::Tool => "user",
                };

                let parts = match &msg.content {
                    llm_types::message::MessageContentValue::Text(text) => {
                        if text.is_empty() && msg.tool_calls.is_none() {
                            return None;
                        }
                        let mut p = Vec::new();
                        if !text.is_empty() {
                            p.push(serde_json::json!({"text": text}));
                        }
                        if let Some(ref tool_calls) = msg.tool_calls {
                            for tc in tool_calls {
                                let args: serde_json::Value = serde_json::from_str(
                                    &tc.function.arguments,
                                )
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                                p.push(serde_json::json!({
                                    "function_call": {
                                        "name": tc.function.name,
                                        "args": args,
                                    }
                                }));
                            }
                        }
                        p
                    }
                    llm_types::message::MessageContentValue::Rich(blocks) => blocks
                        .iter()
                        .filter_map(|block| match block {
                            llm_types::message::MessageContent::Text { text } => {
                                Some(serde_json::json!({"text": text}))
                            }
                            llm_types::message::MessageContent::ImageUrl { image_url } => {
                                // Inline base64 images become `inline_data`;
                                // remote URLs become `file_data` references.
                                // Unresolvable references are skipped rather
                                // than silently corrupted.
                                match super::helpers::resolve_image_payload(&image_url.url) {
                                    Ok(super::helpers::ResolvedImage::Base64 {
                                        media_type,
                                        data,
                                    }) => Some(serde_json::json!({
                                        "inline_data": {
                                            "mime_type": media_type,
                                            "data": data,
                                        }
                                    })),
                                    Ok(super::helpers::ResolvedImage::Url(url)) => {
                                        Some(serde_json::json!({
                                            "file_data": {
                                                "mime_type":
                                                    super::helpers::guess_image_media_type(&url),
                                                "file_uri": url,
                                            }
                                        }))
                                    }
                                    Err(_) => None,
                                }
                            }
                            llm_types::message::MessageContent::ToolResult { tool_result } => {
                                let content_val: serde_json::Value =
                                    serde_json::from_str(&tool_result.content).unwrap_or_else(
                                        |_| serde_json::Value::String(tool_result.content.clone()),
                                    );
                                // Gemini requires the tool name so the model can
                                // match the response back to its `functionCall`.
                                // `Message.tool_name` carries it; fall back to
                                // an empty name only when it is genuinely absent.
                                let name = msg.tool_name.clone().unwrap_or_default();
                                Some(serde_json::json!({
                                    "function_response": {
                                        "name": name,
                                        "response": content_val,
                                    }
                                }))
                            }
                            _ => None,
                        })
                        .collect(),
                };

                Some(serde_json::json!({"role": role, "parts": parts}))
            })
            .collect()
    }

    fn convert_generation_config(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let generation = super::shared::resolve_generation(request, profile)?;

        // Only user-set fields are emitted; unset fields stay out of the
        // payload so the server-side model defaults apply. Defaults such as
        // `topK: 40` or `maxOutputTokens: 4096` must not silently clamp
        // modern Gemini models.
        let mut config = serde_json::json!({});

        crate::generation::apply_gemini_generation_config(&mut config, &generation)?;

        Ok(config)
    }

    /// Map a raw Gemini `finishReason` onto the unified finish-reason
    /// vocabulary used across codecs.
    fn normalize_finish_reason(raw: &str) -> String {
        match raw {
            "STOP" => "stop".to_string(),
            "MAX_TOKENS" => "length".to_string(),
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
                "content_filter".to_string()
            }
            other => other.to_ascii_lowercase(),
        }
    }

    /// Parse the top-level `usageMetadata` object into token stats.
    fn parse_usage_metadata(u: &serde_json::Value) -> llm_types::llm::TokenUsageStats {
        llm_types::llm::TokenUsageStats {
            prompt_tokens: u
                .get("promptTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u
                .get("totalTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            reasoning_tokens: u
                .get("thoughtsTokenCount")
                .and_then(|v| v.as_u64())
                .map(|r| r as u32),
            cache_read_tokens: u
                .get("cachedContentTokenCount")
                .and_then(|v| v.as_u64())
                .map(|r| r as u32),
            cache_write_tokens: None,
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: None,
        }
    }
}

impl LlmCodec for GeminiNativeCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        let streaming = request.stream == Some(true);

        // Streaming uses the SSE endpoint; the non-streaming endpoint returns
        // a single JSON object that no SSE parser can consume.
        let action = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        let url = format!("{}/models/{}:{}", self.endpoint_base(profile), profile.model, action);

        let body = self.build_body(request, profile)?;

        self.build_json_request(url, &body, profile)
    }

    fn build_count_tokens_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<Option<reqwest::Request>> {
        let url = format!(
            "{}/models/{}:countTokens",
            self.endpoint_base(profile),
            profile.model
        );

        // Reuse the inference body so the count covers exactly what would
        // be sent, minus `generationConfig` (output steering does not
        // affect the input count and is not part of the count schema).
        let mut body = self.build_body(request, profile)?;
        if let Some(map) = body.as_object_mut() {
            map.remove("generationConfig");
        }

        Ok(Some(self.build_json_request(url, &body, profile)?))
    }

    fn parse_count_tokens_response(&self, body: &serde_json::Value) -> LlmResult<u32> {
        // The Gemini count-tokens API only ever returns `totalTokens`.
        Ok(body
            .get("totalTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32)
    }

    fn parse_response(&self, body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
        let mut result = self.parse_gemini_response(body)?;
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
        Ok(self.parse_stream_chunk_events(data)?.into_iter().next())
    }

    fn parse_stream_chunk_events(
        &self,
        data: &str,
    ) -> LlmResult<Vec<MessageStreamEvent>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let json: serde_json::Value = serde_json::from_str(data)?;
        let mut events = Vec::new();

        if let Some(candidates) = json.get("candidates").and_then(|v| v.as_array()) {
            if let Some(candidate) = candidates.first() {
                // Every part is emitted — a single chunk can pack text,
                // reasoning and function-call parts together.
                if let Some(parts) = candidate
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|v| v.as_array())
                {
                    for part in parts {
                        // `thought` is a boolean flag on the part (official
                        // schema: Part.thought: bool); a part like
                        // {"text": "...", "thought": true} is reasoning.
                        let is_thought =
                            part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            if is_thought {
                                events.push(MessageStreamEvent::ReasoningText(
                                    llm_types::llm::MessageStreamReasoning {
                                        snapshot: String::new(),
                                        reasoning: text.to_string(),
                                    },
                                ));
                            } else {
                                events.push(MessageStreamEvent::Text(
                                    llm_types::llm::MessageStreamText {
                                        snapshot: String::new(),
                                        text: text.to_string(),
                                    },
                                ));
                            }
                            continue;
                        }
                        if let Some(thought) = part.get("thought").and_then(|v| v.as_str()) {
                            // Backwards compatibility for string-valued thought.
                            events.push(MessageStreamEvent::ReasoningText(
                                llm_types::llm::MessageStreamReasoning {
                                    snapshot: String::new(),
                                    reasoning: thought.to_string(),
                                },
                            ));
                            continue;
                        }
                        if let Some(func_call) = part.get("functionCall") {
                            let name = func_call
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let args = func_call
                                .get("args")
                                .cloned()
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                            let arguments = serde_json::to_string(&args).unwrap_or_default();
                            events.push(MessageStreamEvent::ToolCallDelta(
                                llm_types::llm::MessageStreamToolCallDelta {
                                    index: GEMINI_CALL_INDEX.fetch_add(1, Ordering::Relaxed),
                                    id: None,
                                    name: Some(name),
                                    arguments: Some(arguments),
                                    is_snapshot: true,
                                },
                            ));
                        }
                    }
                }

                // `finishReason` rides on the final chunk, often together
                // with content parts; it must not be swallowed by the part
                // handling above.
                if candidate.get("finishReason").is_some() {
                    events.push(MessageStreamEvent::End(
                        llm_types::llm::MessageStreamEnd {},
                    ));
                }
            }
        }

        // Streaming usage arrives in the final chunk's `usageMetadata`.
        if let Some(u) = json.get("usageMetadata") {
            events.push(MessageStreamEvent::Usage(llm_types::llm::MessageStreamUsage {
                usage: Self::parse_usage_metadata(u),
            }));
        }

        Ok(events)
    }

    fn convert_tools(&self, tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
        // Full `tools` array shape: `[{"functionDeclarations": [...]}]`, ready
        // to be assigned to the request body `tools` field.
        let function_declarations: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        Ok(vec![serde_json::json!({
            "functionDeclarations": function_declarations,
        })])
    }

    fn parse_tool_calls(&self, result: &LlmResponseType) -> Vec<llm_types::message::LlmToolCall> {
        result.tool_calls.clone().unwrap_or_default()
    }
}

impl GeminiNativeCodec {
    /// Build the request body (separated from the HTTP layer for testability).
    fn build_body(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let generation_config = self.convert_generation_config(request, profile)?;

        let use_text_mode = super::shared::is_text_mode(request);

        // System instructions are sent in the dedicated `systemInstruction`
        // field in both modes. Text mode injects the original system + tool
        // usage instructions + declarations; native mode keeps the original
        // system message.
        let (system_content, _) =
            llm_tool_call::tool::protocol::extract_system_message(&request.messages);

        let history = if use_text_mode {
            super::shared::convert_history_for_text_mode(&request.messages, request)
        } else {
            request.messages.clone()
        };
        let messages = self.convert_messages(&history);

        let mut body = serde_json::json!({
            "contents": messages,
            "generationConfig": generation_config,
        });

        if use_text_mode {
            let system = super::shared::text_mode_system_content(request);
            if !system.is_empty() {
                body["systemInstruction"] = serde_json::json!({"parts": [{"text": system}]});
            }
        } else {
            if let Some(system) = system_content {
                if !system.is_empty() {
                    body["systemInstruction"] = serde_json::json!({"parts": [{"text": system}]});
                }
            }
            if let Some(tools) = &request.tools {
                // Reuse the single `convert_tools` implementation so the body
                // shape and the trait method can never drift apart.
                body["tools"] = serde_json::json!(self.convert_tools(tools)?);
            }
        }

        super::shared::apply_custom_body(&mut body, profile);

        Ok(body)
    }

    fn parse_gemini_response(&self, body: &str) -> LlmResult<LlmResponseType> {
        let json: serde_json::Value = serde_json::from_str(body)?;

        let candidates = json.get("candidates").and_then(|v| v.as_array());
        let first_candidate = candidates.and_then(|c| c.first());

        let mut text_content = String::new();
        let mut reasoning_content: Option<String> = None;
        let mut tool_calls = Vec::new();
        let mut thought_signatures: Vec<String> = Vec::new();

        if let Some(candidate) = first_candidate {
            let content = candidate.get("content");
            if let Some(c) = content {
                let parts = c.get("parts").and_then(|v| v.as_array());
                if let Some(parts) = parts {
                    for part in parts {
                        // `thought` is a boolean flag on the part (official
                        // schema: Part.thought: bool); thinking-model parts
                        // look like {"text": "...", "thought": true}.
                        let is_thought =
                            part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            if is_thought {
                                reasoning_content
                                    .get_or_insert_with(String::new)
                                    .push_str(text);
                            } else {
                                text_content.push_str(text);
                            }
                        }
                        if let Some(thought) = part.get("thought").and_then(|v| v.as_str()) {
                            // Backwards compatibility for string-valued thought.
                            reasoning_content
                                .get_or_insert_with(String::new)
                                .push_str(thought);
                        }
                        if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str()) {
                            thought_signatures.push(sig.to_string());
                        }
                        if let Some(func_call) = part.get("functionCall") {
                            let name = func_call
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let args = func_call
                                .get("args")
                                .cloned()
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                            let arguments = serde_json::to_string(&args).unwrap_or_default();
                            tool_calls.push(llm_types::message::LlmToolCall {
                                id: format!("gemini_call_{}", llm_common::generate_id()),
                                r#type: "function".to_string(),
                                function: llm_types::message::LlmFunctionCall { name, arguments },
                            });
                        }
                    }
                }
            }
        }

        let usage = json
            .get("usageMetadata")
            .map(Self::parse_usage_metadata);

        let message = llm_types::message::Message {
            id: llm_types::Id::new(),
            role: llm_types::message::MessageRole::Assistant,
            content: llm_types::message::MessageContentValue::Text(text_content.clone()),
            timestamp: llm_common::time::now(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.clone())
            },
            thinking: None,
            metadata: None,
        };

        let finish_reason = first_candidate
            .and_then(|c| c.get("finishReason"))
            .and_then(|v| v.as_str())
            .map(Self::normalize_finish_reason);

        let mut metadata = std::collections::HashMap::new();
        if let Some(fr) = first_candidate
            .and_then(|c| c.get("finishReason"))
            .and_then(|v| v.as_str())
        {
            metadata.insert("finish_reason_raw".to_string(), serde_json::json!(fr));
        }
        if !thought_signatures.is_empty() {
            metadata.insert(
                "thought_signatures".to_string(),
                serde_json::json!(thought_signatures),
            );
        }
        let metadata = if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        };

        let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

        Ok(LlmResponseType {
            // Prefer the server-assigned id; fall back to a local id.
            id: json
                .get("responseId")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| Some(llm_common::generate_id())),
            model: json
                .get("modelVersion")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            content: if text_content.is_empty() {
                None
            } else {
                Some(text_content)
            },
            message,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            usage,
            finish_reason,
            duration: 0,
            reasoning_content: reasoning_content.clone(),
            reasoning_tokens,
            metadata,
            stream_stats: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call_part_emits_snapshot_delta() {
        let chunk = r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Beijing"}}}]},"finishReason":"STOP"}]}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse");
        assert!(
            events.len() >= 2,
            "tool call part and End must both be emitted, got {:?}",
            events
        );
        match &events[0] {
            MessageStreamEvent::ToolCallDelta(delta) => {
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                let args: serde_json::Value =
                    serde_json::from_str(delta.arguments.as_deref().unwrap()).unwrap();
                assert_eq!(args["city"], "Beijing");
                assert!(delta.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
        assert!(
            matches!(events.last(), Some(MessageStreamEvent::End(_))),
            "finishReason in the same chunk must still yield End"
        );
    }

    #[test]
    fn thought_bool_parts_stream_as_reasoning() {
        let codec = GeminiNativeCodec::new();
        let chunk = r#"{"candidates":[{"content":{"parts":[{"text":"secret plan","thought":true},{"text":"answer"}]}}]}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        assert_eq!(events.len(), 2, "both parts must be emitted, got {:?}", events);
        assert!(matches!(
            &events[0],
            MessageStreamEvent::ReasoningText(_)
        ));
        assert!(matches!(&events[1], MessageStreamEvent::Text(_)));
    }

    #[test]
    fn usage_metadata_chunk_emits_usage_event() {
        let codec = GeminiNativeCodec::new();
        let chunk = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.prompt_tokens, 10);
                assert_eq!(u.usage.completion_tokens, 5);
                assert_eq!(u.usage.total_tokens, 15);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }
    }

    fn profile() -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format: llm_types::llm::LlmFormat::GeminiNative,
            provider_id: None,
            model: "gemini-1.5-pro".to_string(),
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
            context_window_size: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
        }
    }

    fn msg(role: llm_types::message::MessageRole, text: &str) -> llm_types::message::Message {
        llm_types::message::Message {
            id: llm_types::Id::new(),
            role,
            content: llm_types::message::MessageContentValue::Text(text.to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            metadata: None,
        }
    }

    fn request(
        messages: Vec<llm_types::message::Message>,
        params: Option<serde_json::Value>,
    ) -> LlmRequest {
        LlmRequest {
            profile_id: "p1".to_string(),
            messages,
            parameters: params,
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
        }
    }

    #[test]
    fn rich_image_blocks_become_inline_data_or_file_data() {
        let codec = GeminiNativeCodec::new();
        let image_msg = llm_types::message::Message {
            id: llm_types::Id::new(),
            role: llm_types::message::MessageRole::User,
            content: llm_types::message::MessageContentValue::Rich(vec![
                llm_types::message::MessageContent::Text {
                    text: "what is this".to_string(),
                },
                llm_types::message::MessageContent::ImageUrl {
                    image_url: llm_types::message::ImageUrlContent {
                        url: "data:image/png;base64,AAAA".to_string(),
                        detail: None,
                    },
                },
                llm_types::message::MessageContent::ImageUrl {
                    image_url: llm_types::message::ImageUrlContent {
                        url: "https://example.com/a.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            metadata: None,
        };
        let entries = codec.convert_messages(&[image_msg]);
        assert_eq!(entries.len(), 1);
        let parts = entries[0]["parts"].as_array().expect("parts array");
        // Regression guard: none of the three blocks may be silently dropped.
        assert_eq!(parts.len(), 3, "text + inline_data + file_data");
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["inline_data"]["mime_type"], "image/png");
        assert_eq!(parts[1]["inline_data"]["data"], "AAAA");
        assert_eq!(parts[2]["file_data"]["file_uri"], "https://example.com/a.jpg");
        assert_eq!(parts[2]["file_data"]["mime_type"], "image/jpeg");
    }

    #[test]
    fn native_mode_sends_system_instruction() {
        let codec = GeminiNativeCodec::new();
        let req = request(
            vec![
                msg(llm_types::message::MessageRole::System, "You are a helper"),
                msg(llm_types::message::MessageRole::User, "Hello"),
            ],
            None,
        );
        let body = codec.build_body(&req, &profile()).expect("must build");

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
        let codec = GeminiNativeCodec::new();
        let tool_msg = llm_types::message::Message {
            id: llm_types::Id::new(),
            role: llm_types::message::MessageRole::Tool,
            content: llm_types::message::MessageContentValue::Rich(vec![
                llm_types::message::MessageContent::ToolResult {
                    tool_result: llm_types::message::ToolResultContent {
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
            metadata: None,
        };
        let req = request(vec![tool_msg, msg(llm_types::message::MessageRole::User, "next")], None);
        let body = codec.build_body(&req, &profile()).expect("must build");
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[0]["role"], serde_json::json!("user"),
            "Gemini only accepts user/model roles");
        assert_eq!(
            contents[0]["parts"][0]["function_response"]["name"],
            serde_json::json!("get_weather"),
            "function_response must carry the tool name"
        );
    }

    #[test]
    fn native_mode_sends_tools_alongside_system() {
        let codec = GeminiNativeCodec::new();
        let req = request(
            vec![
                msg(llm_types::message::MessageRole::System, "You are a helper"),
                msg(llm_types::message::MessageRole::User, "Hello"),
            ],
            None,
        );
        let mut req = req;
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()]);
        let body = codec.build_body(&req, &profile()).expect("must build");

        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("You are a helper")
        );
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            serde_json::json!("get_weather")
        );
        // The trait method and the body must agree on the shape.
        let trait_shape = codec.convert_tools(req.tools.as_deref().unwrap()).unwrap();
        assert_eq!(trait_shape, body["tools"].as_array().unwrap().clone());
    }

    #[test]
    fn generation_config_only_emits_user_set_fields() {
        let codec = GeminiNativeCodec::new();
        let body = codec
            .build_body(&request(vec![], None), &profile())
            .expect("must build");

        let config = &body["generationConfig"];
        assert!(config.get("temperature").is_none(), "unset fields must not be forced");
        assert!(config.get("topP").is_none());
        assert!(config.get("topK").is_none());
        assert!(config.get("maxOutputTokens").is_none());

        let req = request(
            vec![],
            Some(serde_json::json!({"temperature": 0.2, "max_tokens": 128})),
        );
        let body = codec.build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["generationConfig"]["temperature"],
            serde_json::json!(0.2)
        );
        assert_eq!(
            body["generationConfig"]["maxOutputTokens"],
            serde_json::json!(128)
        );
    }

    fn count_request() -> LlmRequest {
        request(
            vec![
                msg(llm_types::message::MessageRole::System, "You are a helper"),
                msg(llm_types::message::MessageRole::User, "Hello"),
            ],
            None,
        )
    }

    #[test]
    fn count_tokens_request_targets_count_tokens_endpoint_without_key_in_url() {
        let codec = GeminiNativeCodec::new();
        let req = codec
            .build_count_tokens_request(&count_request(), &profile())
            .expect("count request must build")
            .expect("gemini native supports counting");
        let url = req.url().as_str();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:countTokens",
            "api key must not appear in the URL"
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "sk-test",
            "auth must ride the x-goog-api-key header"
        );
        let body: serde_json::Value = req
            .body()
            .unwrap()
            .as_bytes()
            .map(|b| serde_json::from_slice(b).unwrap())
            .unwrap();
        assert_eq!(body["contents"][0]["role"], serde_json::json!("user"));
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("You are a helper")
        );
        assert!(
            body.get("generationConfig").is_none(),
            "count body must not carry output steering"
        );
    }

    #[test]
    fn stream_request_targets_stream_generate_content_sse() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(vec![msg(llm_types::message::MessageRole::User, "Hi")], None);
        req.stream = Some(true);
        let req = codec.build_request(&req, &profile()).expect("must build");
        let url = req.url().as_str();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "sk-test",
            "auth must ride the x-goog-api-key header"
        );
    }

    #[test]
    fn non_stream_request_targets_generate_content() {
        let codec = GeminiNativeCodec::new();
        let req = codec
            .build_request(&request(vec![msg(llm_types::message::MessageRole::User, "Hi")], None), &profile())
            .expect("must build");
        assert_eq!(
            req.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn count_tokens_body_covers_same_contents_as_inference() {
        let codec = GeminiNativeCodec::new();
        let inference = codec
            .build_body(&count_request(), &profile())
            .expect("inference body must build");
        let req = codec
            .build_count_tokens_request(&count_request(), &profile())
            .expect("count request must build")
            .unwrap();
        let count_body: serde_json::Value = req
            .body()
            .unwrap()
            .as_bytes()
            .map(|b| serde_json::from_slice(b).unwrap())
            .unwrap();
        assert_eq!(count_body["contents"], inference["contents"]);
        assert_eq!(
            count_body["systemInstruction"],
            inference["systemInstruction"]
        );
    }

    #[test]
    fn count_tokens_response_parses_total_tokens() {
        let codec = GeminiNativeCodec::new();
        let body: serde_json::Value = serde_json::from_str(r#"{"totalTokens": 42}"#).unwrap();
        assert_eq!(codec.parse_count_tokens_response(&body).unwrap(), 42);
        let empty: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(codec.parse_count_tokens_response(&empty).unwrap(), 0);
    }

    #[test]
    fn parse_response_prefers_response_id_and_normalizes_finish_reason() {
        let codec = GeminiNativeCodec::new();
        let body = r#"{
            "responseId": "resp-abc",
            "modelVersion": "gemini-2.5-flash",
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
        }"#;
        let req = request(vec![], None);
        let result = codec.parse_response(body, &req).expect("must parse");
        assert_eq!(result.id.as_deref(), Some("resp-abc"));
        assert_eq!(result.finish_reason.as_deref(), Some("length"));
        assert_eq!(
            result.metadata.as_ref().unwrap()["finish_reason_raw"],
            serde_json::json!("MAX_TOKENS")
        );
        assert_eq!(result.usage.as_ref().unwrap().total_tokens, 5);
    }

    #[test]
    fn parse_response_routes_thought_bool_into_reasoning() {
        let codec = GeminiNativeCodec::new();
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"text": "plan", "thought": true},
                {"text": "answer"}
            ]}}]
        }"#;
        let req = request(vec![], None);
        let result = codec.parse_response(body, &req).expect("must parse");
        assert_eq!(result.content.as_deref(), Some("answer"));
        assert_eq!(result.reasoning_content.as_deref(), Some("plan"));
    }

    #[test]
    fn parse_response_captures_thought_signature() {
        let codec = GeminiNativeCodec::new();
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "sig-1"}
            ]}}]
        }"#;
        let req = request(vec![], None);
        let result = codec.parse_response(body, &req).expect("must parse");
        assert_eq!(
            result.metadata.as_ref().unwrap()["thought_signatures"],
            serde_json::json!(["sig-1"])
        );
    }
}
