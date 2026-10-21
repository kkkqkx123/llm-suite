use super::LlmCodec;
use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent,
    THOUGHT_SIGNATURES_KEY,
};
use llm_suite_types::message::Message;
use llm_suite_types::tool::Tool;
use reqwest::Method;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The thought signature that belongs to the tool call at `index`, when the
/// provider signed it.
fn thought_signature_for(message: &Message, index: usize) -> Option<String> {
    message
        .metadata
        .as_ref()?
        .get(THOUGHT_SIGNATURES_KEY)?
        .as_array()?
        .get(index)?
        .as_str()
        .map(String::from)
}

pub struct GeminiNativeCodec {
    base_url: String,
    /// Tool-call index source for streamed function calls. Every streamed
    /// `functionCall` part is a complete snapshot, so each call needs its own
    /// index to stay apart in the stream accumulator. The counter is reset
    /// when a chunk carries the finish reason, so the numbering describes the
    /// position of the call inside its own response.
    call_index: AtomicUsize,
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
            call_index: AtomicUsize::new(0),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            call_index: AtomicUsize::new(0),
        }
    }

    /// Effective API endpoint base.
    fn endpoint_base<'a>(&'a self, profile: &'a LlmProfile) -> &'a str {
        profile.base_url.as_deref().unwrap_or(&self.base_url)
    }

    /// The model segment of the endpoint path.
    ///
    /// The model lives in the URL, not in the body, and the path segment is
    /// always `models/{model}`. A model that already carries the prefix is
    /// not prefixed twice.
    fn model_segment(model: &str) -> String {
        let model = model.trim();
        if model.starts_with("models/") || model.starts_with("tunedModels/") {
            model.to_string()
        } else {
            format!("models/{model}")
        }
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

        builder.build().map_err(crate::error::LlmError::HttpError)
    }

    /// Convert internal messages into Gemini `contents`.
    ///
    /// Gemini only accepts the `user` and `model` roles; tool results ride
    /// in a user message as `functionResponse` parts. The function name and
    /// the call id a result refers to are taken from the tool call that
    /// produced it, because Gemini matches the response back to the call.
    fn convert_messages(
        &self,
        messages: &[llm_suite_types::message::Message],
    ) -> LlmResult<Vec<serde_json::Value>> {
        // The function name of every tool call in the history, keyed by the
        // call identity the result refers to.
        let mut function_names: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        // The call id the provider assigned, keyed by the internal call
        // identity. Only a provider-assigned id is echoed back in the
        // function response; a locally generated one would not match any
        // call the provider knows.
        let mut provider_call_ids: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for msg in messages {
            if let Some(calls) = &msg.tool_calls {
                for call in calls {
                    function_names.insert(call.id.clone(), call.function.name.clone());
                    if let Some(ref call_id) = call.call_id {
                        provider_call_ids.insert(call.id.clone(), call_id.clone());
                    }
                }
            }
            if let llm_suite_types::message::MessageContentValue::Rich(blocks) = &msg.content {
                for block in blocks {
                    if let llm_suite_types::message::MessageContent::ToolUse { tool_use } = block {
                        function_names.insert(tool_use.id.clone(), tool_use.name.clone());
                    }
                }
            }
        }

        let mut contents: Vec<serde_json::Value> = Vec::new();
        for msg in messages {
            let role = match msg.role {
                llm_suite_types::message::MessageRole::System => continue,
                llm_suite_types::message::MessageRole::User => "user",
                llm_suite_types::message::MessageRole::Assistant => "model",
                llm_suite_types::message::MessageRole::Tool => "user",
            };

            let parts = self.convert_parts(msg, &function_names, &provider_call_ids)?;
            // A message without any part is not a valid Gemini content.
            if parts.is_empty() {
                continue;
            }
            contents.push(serde_json::json!({"role": role, "parts": parts}));
        }
        Ok(contents)
    }

    /// Convert one message into Gemini parts.
    fn convert_parts(
        &self,
        msg: &llm_suite_types::message::Message,
        function_names: &std::collections::HashMap<String, String>,
        provider_call_ids: &std::collections::HashMap<String, String>,
    ) -> LlmResult<Vec<serde_json::Value>> {
        let mut parts: Vec<serde_json::Value> = Vec::new();
        // Whether the content already carried the function calls; the flat
        // call list is only used when it did not.
        let mut saw_tool_use = false;

        match &msg.content {
            llm_suite_types::message::MessageContentValue::Text(text) => {
                if msg.role == llm_suite_types::message::MessageRole::Tool {
                    // A tool result is a function response; plain text would
                    // hide the output from the model.
                    let call_id = msg.tool_call_id.clone().unwrap_or_default();
                    let name = self.resolve_function_name(msg, &call_id, function_names)?;
                    let provider_id = provider_call_ids.get(&call_id).map(String::as_str);
                    parts.push(Self::function_response_part(&name, provider_id, text));
                } else if !text.is_empty() {
                    parts.push(serde_json::json!({"text": text}));
                }
            }
            llm_suite_types::message::MessageContentValue::Rich(blocks) => {
                let mut call_ordinal = 0usize;
                for block in blocks {
                    match block {
                        llm_suite_types::message::MessageContent::Text { text } => {
                            parts.push(serde_json::json!({"text": text}));
                        }
                        llm_suite_types::message::MessageContent::ImageUrl { image_url } => {
                            // Inline base64 images become `inlineData`;
                            // remote URLs become `fileData` references.
                            // Unresolvable references are skipped rather
                            // than silently corrupted.
                            match super::helpers::resolve_image_payload(&image_url.url) {
                                Ok(super::helpers::ResolvedImage::Base64 { media_type, data }) => {
                                    parts.push(serde_json::json!({
                                        "inlineData": {
                                            "mimeType": media_type,
                                            "data": data,
                                        }
                                    }))
                                }
                                Ok(super::helpers::ResolvedImage::Url(url)) => {
                                    parts.push(serde_json::json!({
                                        "fileData": {
                                            "mimeType":
                                                super::helpers::guess_image_media_type(&url),
                                            "fileUri": url,
                                        }
                                    }))
                                }
                                Err(_) => {}
                            }
                        }
                        llm_suite_types::message::MessageContent::ToolUse { tool_use } => {
                            let mut part = serde_json::json!({
                                "functionCall": {
                                    "name": tool_use.name,
                                    "args": tool_use.input,
                                }
                            });
                            // The provider signs a function call with an
                            // opaque thought signature; the signature has
                            // to travel back with the call in the next turn.
                            if let Some(signature) = thought_signature_for(msg, call_ordinal) {
                                part["thoughtSignature"] = serde_json::json!(signature);
                            }
                            parts.push(part);
                            call_ordinal += 1;
                            saw_tool_use = true;
                        }
                        llm_suite_types::message::MessageContent::ToolResult { tool_result } => {
                            let name = self.resolve_function_name(
                                msg,
                                &tool_result.tool_use_id,
                                function_names,
                            )?;
                            let provider_id = provider_call_ids
                                .get(&tool_result.tool_use_id)
                                .map(String::as_str);
                            parts.push(Self::function_response_part(
                                &name,
                                provider_id,
                                &tool_result.content,
                            ));
                        }
                        // The reasoning text itself is not replayed; only the
                        // thought signature on a function call is.
                        llm_suite_types::message::MessageContent::Thinking { .. }
                        | llm_suite_types::message::MessageContent::RedactedThinking { .. } => {}
                    }
                }
            }
        }

        // Tool calls carried on the message itself become function call
        // parts; the rich blocks already hold them otherwise.
        if !saw_tool_use {
            if let Some(calls) = &msg.tool_calls {
                for (ordinal, call) in calls.iter().enumerate() {
                    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    let mut part = serde_json::json!({
                        "functionCall": {
                            "name": call.function.name,
                            "args": args,
                        }
                    });
                    // The provider signs a function call with an opaque
                    // thought signature; it has to travel back with the call
                    // in the next turn.
                    if let Some(signature) = thought_signature_for(msg, ordinal) {
                        part["thoughtSignature"] = serde_json::json!(signature);
                    }
                    parts.push(part);
                }
            }
        }

        Ok(parts)
    }

    /// The function name a tool result answers.
    ///
    /// The name is taken from the result message itself when it carries one,
    /// otherwise from the tool call the result refers to. A result whose
    /// function cannot be named is rejected: Gemini matches the response
    /// back to the call by name, so an unnamed response is meaningless.
    fn resolve_function_name(
        &self,
        msg: &llm_suite_types::message::Message,
        call_id: &str,
        function_names: &std::collections::HashMap<String, String>,
    ) -> LlmResult<String> {
        if let Some(name) = msg.tool_name.as_ref().filter(|name| !name.is_empty()) {
            return Ok(name.clone());
        }
        if let Some(name) = function_names.get(call_id) {
            return Ok(name.clone());
        }
        Err(crate::error::LlmError::ConfigError(format!(
            "Gemini tool result for call `{call_id}` carries no function name; keep the assistant tool call in the history or name the result"
        )))
    }

    /// Build a `functionResponse` part.
    ///
    /// The official `response` field is a JSON object, not a bare value: the
    /// provider matches the response to the call by name and reads the
    /// result from the `output` / `error` keys. A tool output that is not
    /// already an object is therefore wrapped in `output`.
    ///
    /// The call id is echoed back when the provider assigned one, so the
    /// response can be matched to the exact call it answers.
    fn function_response_part(
        name: &str,
        call_id: Option<&str>,
        content: &str,
    ) -> serde_json::Value {
        let response_value: serde_json::Value = match serde_json::from_str(content) {
            Ok(serde_json::Value::Object(object)) => serde_json::Value::Object(object),
            _ => serde_json::json!({ "output": content }),
        };
        let mut response = serde_json::json!({
            "name": name,
            "response": response_value,
        });
        if let Some(call_id) = call_id.filter(|id| !id.is_empty()) {
            response["id"] = serde_json::json!(call_id);
        }
        serde_json::json!({"functionResponse": response})
    }

    /// Map a raw Gemini `finishReason` onto the unified finish-reason
    /// vocabulary used across codecs.
    ///
    /// The whole safety family maps to `content_filter`: Gemini splits it
    /// over many members (`SAFETY`, `IMAGE_SAFETY`, `NO_IMAGE`, …) while the
    /// OpenAI formats report a single value for the same situation.
    fn normalize_finish_reason(raw: &str) -> String {
        match raw {
            "STOP" => "stop".to_string(),
            "MAX_TOKENS" => "length".to_string(),
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "NO_IMAGE"
            | "IMAGE_RECITATION"
            | "IMAGE_OTHER" => "content_filter".to_string(),
            other => other.to_ascii_lowercase(),
        }
    }

    /// Parse the top-level `usageMetadata` object into token stats.
    fn parse_usage_metadata(u: &serde_json::Value) -> llm_suite_types::llm::TokenUsageStats {
        // Cached content tokens: primary source `cachedContentTokenCount`;
        // when absent, fall back to summing the per-modality breakdown in
        // `cacheTokensDetails` (`[{modality, tokenCount}, ...]`).
        let cache_read_tokens = u
            .get("cachedContentTokenCount")
            .and_then(|v| v.as_u64())
            .map(|r| r as u32)
            .or_else(|| {
                let sum: u64 = u
                    .get("cacheTokensDetails")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|m| m.get("tokenCount").and_then(|t| t.as_u64()))
                            .sum()
                    })
                    .unwrap_or(0);
                (sum > 0).then_some(sum as u32)
            });
        let prompt_tokens = u
            .get("promptTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion_tokens = u
            .get("candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let reasoning_tokens = u
            .get("thoughtsTokenCount")
            .and_then(|v| v.as_u64())
            .map(|r| r as u32);
        llm_suite_types::llm::TokenUsageStats {
            prompt_tokens,
            completion_tokens,
            // The provider owns the total; without one it is the sum of the
            // counters it reports. The thoughts are billed on top of the
            // candidates, and the cached tokens are already part of the
            // prompt.
            total_tokens: u.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(
                u64::from(prompt_tokens)
                    + u64::from(completion_tokens)
                    + u64::from(reasoning_tokens.unwrap_or(0)),
            ) as u32,
            reasoning_tokens,
            cache_read_tokens,
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
        let url = super::shared::endpoint_url(
            self.endpoint_base(profile),
            &format!("{}:{}", Self::model_segment(&profile.model), action),
        );

        let body = self.build_body(request, profile)?;

        self.build_json_request(url, &body, profile)
    }

    fn build_count_tokens_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<Option<reqwest::Request>> {
        let url = super::shared::endpoint_url(
            self.endpoint_base(profile),
            &format!("{}:countTokens", Self::model_segment(&profile.model)),
        );

        // The counting endpoint takes the request contents only: system
        // instructions, tool declarations and the generation config are not
        // part of its contract.
        let use_text_mode = super::shared::is_text_mode(request);
        let history = if use_text_mode {
            super::shared::convert_history_for_text_mode(&request.messages, request)
        } else {
            request.messages.clone()
        };
        let body = serde_json::json!({
            "contents": self.convert_messages(&history)?,
        });

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

    fn parse_stream_chunk_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let json: serde_json::Value = serde_json::from_str(data)?;
        let mut events = Vec::new();

        // Streaming usage arrives in the final chunk's `usageMetadata`. It
        // must be surfaced before the terminal event: the accumulator builds
        // the final message when the terminal event arrives.
        if let Some(u) = json.get("usageMetadata") {
            events.push(MessageStreamEvent::Usage(
                llm_suite_types::llm::MessageStreamUsage {
                    usage: Self::parse_usage_metadata(u),
                },
            ));
        }

        if let Some(candidates) = json.get("candidates").and_then(|v| v.as_array()) {
            if let Some(candidate) = candidates.first() {
                // Every part is emitted — a single chunk can pack text,
                // reasoning and function-call parts together.
                if let Some(parts) = candidate
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|v| v.as_array())
                {
                    // The thought signature may ride on the function call
                    // itself or on the thought part in front of it.
                    let mut pending_signature: Option<String> = None;
                    for part in parts {
                        if let Some(signature) =
                            part.get("thoughtSignature").and_then(|v| v.as_str())
                        {
                            pending_signature = Some(signature.to_string());
                        }
                        // `thought` is a boolean flag on the part (official
                        // schema: Part.thought: bool); a part like
                        // {"text": "...", "thought": true} is reasoning.
                        let is_thought = part
                            .get("thought")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            if is_thought {
                                events.push(MessageStreamEvent::ReasoningText(
                                    llm_suite_types::llm::MessageStreamReasoning {
                                        snapshot: String::new(),
                                        reasoning: text.to_string(),
                                        signature: None,
                                        signature_snapshot: None,
                                    },
                                ));
                            } else {
                                events.push(MessageStreamEvent::Text(
                                    llm_suite_types::llm::MessageStreamText {
                                        snapshot: String::new(),
                                        text: text.to_string(),
                                    },
                                ));
                            }
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
                            // The provider assigns the call id; it has to be
                            // echoed back in the matching function response.
                            let id = func_call
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            // The thought signature may ride on the function
                            // call itself or on the thought part in front of
                            // it.
                            let signature = part
                                .get("thoughtSignature")
                                .and_then(|v| v.as_str())
                                .map(String::from)
                                .or_else(|| pending_signature.clone());
                            events.push(MessageStreamEvent::ToolCallDelta(
                                llm_suite_types::llm::MessageStreamToolCallDelta {
                                    index: self.call_index.fetch_add(1, Ordering::Relaxed),
                                    // The provider-assigned call id is also
                                    // the correlation id: it is the id the
                                    // matching function response has to
                                    // echo back.
                                    call_id: id.clone(),
                                    id,
                                    name: Some(name),
                                    arguments: Some(arguments),
                                    thought_signature: signature,
                                    is_snapshot: true,
                                },
                            ));
                        }
                    }
                }

                // `finishReason` rides on the final chunk, often together
                // with content parts; it must not be swallowed by the part
                // handling above. It also marks the end of the response, so
                // the tool-call numbering starts over for the next one.
                if candidate.get("finishReason").is_some() {
                    self.call_index.store(0, Ordering::Relaxed);
                    events.push(MessageStreamEvent::End(
                        llm_suite_types::llm::MessageStreamEnd {},
                    ));
                }
            }
        }

        Ok(events)
    }

    fn convert_tools(&self, tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
        // Full `tools` array shape: `[{"functionDeclarations": [...]}]`, ready
        // to be assigned to the request body `tools` field. The parameter
        // schema is rewritten into Gemini's dialect, whose type names are
        // uppercase enum members.
        let function_declarations: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                // The parameter schema is optional; a tool without one leaves
                // the field out instead of sending a null schema.
                let mut declaration = serde_json::Map::new();
                declaration.insert("name".to_string(), serde_json::json!(t.name));
                declaration.insert("description".to_string(), serde_json::json!(t.description));
                if let Some(schema) = &t.parameters {
                    declaration.insert(
                        "parameters".to_string(),
                        crate::generation::gemini_schema_value(
                            &serde_json::to_value(schema).unwrap_or(serde_json::Value::Null),
                        ),
                    );
                }
                serde_json::Value::Object(declaration)
            })
            .collect();
        Ok(vec![serde_json::json!({
            "functionDeclarations": function_declarations,
        })])
    }

    fn parse_tool_calls(
        &self,
        result: &LlmResponseType,
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
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
        let generation = super::shared::resolve_generation(request, profile)?;

        // Only user-set fields are emitted; unset fields stay out of the
        // payload so the server-side model defaults apply. Defaults such as
        // `topK: 40` or `maxOutputTokens: 4096` must not silently clamp
        // modern Gemini models.
        let mut generation_config = serde_json::json!({});
        crate::generation::apply_gemini_generation_config(&mut generation_config, &generation)?;

        let use_text_mode = super::shared::is_text_mode(request);

        // System instructions are sent in the dedicated `systemInstruction`
        // field in both modes. Text mode injects the original system + tool
        // usage instructions + declarations; native mode keeps the original
        // system message.
        let (system_content, _) =
            llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);

        let history = if use_text_mode {
            super::shared::convert_history_for_text_mode(&request.messages, request)
        } else {
            request.messages.clone()
        };
        let messages = self.convert_messages(&history)?;

        let mut body = serde_json::json!({
            "contents": messages,
        });

        // The generation config is only sent when it carries something;
        // an empty object would override the server-side model defaults.
        if !generation_config.is_null()
            && generation_config.as_object().is_some_and(|c| !c.is_empty())
        {
            body["generationConfig"] = generation_config;
        }

        // The tool-calling mode lives next to `generationConfig`, not inside
        // it; the official request envelope declares it at the top level.
        if let Some(tool_config) = crate::generation::gemini_tool_config(&generation) {
            body["toolConfig"] = tool_config;
        }

        if use_text_mode {
            let system = super::shared::text_mode_system_content(request);
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
                body["tools"] = serde_json::json!(self.convert_tools(tools)?);
            }
        }

        // Explicit context-cache reference (Gemini `cachedContent` resource
        // created via the caches API). Sent as-is at the top level.
        if let Some(cached_content) = &request.cached_content {
            body["cachedContent"] = serde_json::json!(cached_content);
        }

        // Untyped provider options (`safetySettings`, `labels`, …) reach the
        // request body like they do in the other formats.
        super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

        super::shared::apply_custom_body(&mut body, profile);

        Ok(body)
    }

    fn parse_gemini_response(&self, body: &str) -> LlmResult<LlmResponseType> {
        let json: serde_json::Value = serde_json::from_str(body)?;

        // A blocked prompt carries no candidates; the reason lives in the
        // prompt feedback and has to surface as an error instead of an empty
        // answer.
        if let Some(block_reason) = json
            .pointer("/promptFeedback/blockReason")
            .and_then(|v| v.as_str())
        {
            let detail = json
                .pointer("/promptFeedback/blockReasonMessage")
                .and_then(|v| v.as_str())
                .unwrap_or(block_reason);
            return Err(crate::error::LlmError::InvalidResponse(format!(
                "Gemini blocked the prompt ({block_reason}): {detail}"
            )));
        }

        let candidates = json.get("candidates").and_then(|v| v.as_array());
        let first_candidate = candidates.and_then(|c| c.first());

        let mut text_content = String::new();
        let mut reasoning_content: Option<String> = None;
        let mut tool_calls = Vec::new();
        // One entry per tool call; the signature the provider attached to the
        // function call has to travel back with that call.
        let mut thought_signatures: Vec<Option<String>> = Vec::new();

        if let Some(candidate) = first_candidate {
            let content = candidate.get("content");
            if let Some(c) = content {
                let parts = c.get("parts").and_then(|v| v.as_array());
                if let Some(parts) = parts {
                    // The thought signature may ride on the function call
                    // itself or on the thought part in front of it; the
                    // pending signature covers the second case.
                    let mut pending_signature: Option<String> = None;
                    for part in parts {
                        let signature = part
                            .get("thoughtSignature")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        if let Some(signature) = &signature {
                            pending_signature = Some(signature.clone());
                        }
                        // `thought` is a boolean flag on the part (official
                        // schema: Part.thought: bool); thinking-model parts
                        // look like {"text": "...", "thought": true}.
                        let is_thought = part
                            .get("thought")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            if is_thought {
                                reasoning_content
                                    .get_or_insert_with(String::new)
                                    .push_str(text);
                            } else {
                                text_content.push_str(text);
                            }
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
                            // The provider assigns the call id; it has to be
                            // echoed back in the matching function response.
                            // A locally generated id stands in for the missing
                            // one but must not be echoed back as if the
                            // provider had issued it.
                            let provider_id = func_call
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            let id = provider_id.clone().unwrap_or_else(|| {
                                format!("gemini_call_{}", llm_suite_common::generate_id())
                            });
                            tool_calls.push(llm_suite_types::message::LlmToolCall {
                                id,
                                r#type: "function".to_string(),
                                call_id: provider_id,
                                function: llm_suite_types::message::LlmFunctionCall {
                                    name,
                                    arguments,
                                },
                            });
                            thought_signatures.push(signature.or(pending_signature.clone()));
                        }
                    }
                }
            }
        }

        let usage = json.get("usageMetadata").map(Self::parse_usage_metadata);

        let mut message_metadata = std::collections::HashMap::new();
        if !thought_signatures.is_empty() {
            message_metadata.insert(
                THOUGHT_SIGNATURES_KEY.to_string(),
                serde_json::json!(thought_signatures),
            );
        }
        let message_metadata = if message_metadata.is_empty() {
            None
        } else {
            Some(message_metadata)
        };

        let message = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Assistant,
            content: llm_suite_types::message::MessageContentValue::Text(text_content.clone()),
            timestamp: llm_suite_common::time::now(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.clone())
            },
            // The reasoning text also lives on the message so that replaying
            // the message in a follow-up turn keeps it.
            thinking: reasoning_content.clone(),
            thinking_signature: None,
            metadata: message_metadata,
        };

        // The envelope carries the normalized stop value; the raw
        // `finishReason` stays available in the metadata. Every codec
        // documents which spelling it reports, so consumers match per family.
        let finish_reason = first_candidate
            .and_then(|c| c.get("finishReason"))
            .and_then(|v| v.as_str())
            .map(Self::normalize_finish_reason)
            .map(|reason| {
                // Gemini has no tool-call finish reason: a turn that ends
                // with a function call is reported as `tool_calls`, the
                // value the OpenAI formats report for the same situation.
                if reason == "stop" && !tool_calls.is_empty() {
                    "tool_calls".to_string()
                } else {
                    reason
                }
            });

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
                .or_else(|| Some(llm_suite_common::generate_id())),
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
            reasoning_signature: None,
            reasoning_tokens,
            metadata,
            refusal: None,
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
        assert_eq!(
            events.len(),
            2,
            "both parts must be emitted, got {:?}",
            events
        );
        assert!(matches!(&events[0], MessageStreamEvent::ReasoningText(_)));
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

    #[test]
    fn empty_generation_config_is_not_sent() {
        let codec = GeminiNativeCodec::new();
        let body = codec
            .build_body(&request(vec![], None), &profile())
            .expect("must build");
        assert!(
            body.get("generationConfig").is_none(),
            "an empty generation config would override the model defaults"
        );
    }

    #[test]
    fn tool_parameter_schema_uses_gemini_enum_members() {
        let codec = GeminiNativeCodec::new();
        let tools = vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["city"]
            }
        }))
        .unwrap()];
        let shape = codec.convert_tools(&tools).unwrap();
        let parameters = &shape[0]["functionDeclarations"][0]["parameters"];
        assert_eq!(parameters["type"], serde_json::json!("OBJECT"));
        assert_eq!(
            parameters["properties"]["city"]["type"],
            serde_json::json!("STRING")
        );
        assert_eq!(
            parameters["properties"]["days"]["type"],
            serde_json::json!("INTEGER")
        );
        assert_eq!(
            parameters["properties"]["tags"]["type"],
            serde_json::json!("ARRAY")
        );
        assert_eq!(
            parameters["properties"]["tags"]["items"]["type"],
            serde_json::json!("STRING")
        );
    }

    #[test]
    fn thinking_level_is_emitted_as_gemini_enum_member() {
        let mut config = serde_json::json!({});
        crate::generation::apply_gemini_generation_config(
            &mut config,
            &llm_suite_types::llm::generation::LlmGenerationParams {
                thinking: Some(llm_suite_types::llm::generation::LlmThinkingConfig {
                    level: Some(llm_suite_types::llm::generation::ThinkingLevel::High),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect("thinking level must be accepted");
        assert_eq!(
            config["thinkingConfig"]["thinkingLevel"],
            serde_json::json!("HIGH")
        );
    }

    #[test]
    fn thought_signature_travels_back_with_the_function_call() {
        let codec = GeminiNativeCodec::new();
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "sig-1"}
            ]}}]
        }"#;
        let result = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        let message = result.message.clone();
        assert_eq!(
            message.metadata.as_ref().unwrap()["thought_signatures"],
            serde_json::json!(["sig-1"])
        );

        let contents = codec
            .convert_messages(std::slice::from_ref(&message))
            .expect("must convert");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["thoughtSignature"], serde_json::json!("sig-1"));
    }

    #[test]
    fn named_tool_choice_restricts_the_allowed_functions() {
        let tool_config = crate::generation::gemini_tool_config(
            &llm_suite_types::llm::generation::LlmGenerationParams {
                tool_choice: Some(llm_suite_types::llm::generation::LlmToolChoice {
                    mode: None,
                    tool_name: Some("get_weather".to_string()),
                }),
                ..Default::default()
            },
        )
        .expect("tool choice must be accepted");
        assert_eq!(
            tool_config["functionCallingConfig"]["mode"],
            serde_json::json!("ANY")
        );
        assert_eq!(
            tool_config["functionCallingConfig"]["allowedFunctionNames"],
            serde_json::json!(["get_weather"])
        );
    }

    #[test]
    fn blocked_prompt_surfaces_as_an_error() {
        let codec = GeminiNativeCodec::new();
        let body =
            r#"{"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "unsafe"}}"#;
        let err = codec
            .parse_response(body, &request(vec![], None))
            .expect_err("a blocked prompt must not look like an empty answer");
        assert!(format!("{err}").contains("SAFETY"), "{err}");
    }

    #[test]
    fn usage_metadata_parses_cached_content_tokens() {
        let codec = GeminiNativeCodec::new();
        // Primary source: `cachedContentTokenCount` (python-genai
        // `GenerateContentResponseUsageMetadata.cached_content_token_count`).
        let chunk = r#"{"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"totalTokenCount":105,"thoughtsTokenCount":3,"cachedContentTokenCount":40}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.reasoning_tokens, Some(3));
                assert_eq!(u.usage.cache_read_tokens, Some(40));
                assert_eq!(u.usage.cache_write_tokens, None);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }

        // Fallback: `cacheTokensDetails` per-modality breakdown when the
        // primary count is absent.
        let chunk = r#"{"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"totalTokenCount":105,"cacheTokensDetails":[{"modality":"TEXT","tokenCount":20},{"modality":"IMAGE","tokenCount":8}]}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.cache_read_tokens, Some(28));
            }
            other => panic!("expected Usage event, got {:?}", other),
        }

        // Neither present → `None`.
        let chunk = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.cache_read_tokens, None);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }
    }

    fn profile() -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format: llm_suite_types::llm::LlmFormat::GeminiNative,
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
            prompt_caching: None,
            context_window_size: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
        }
    }

    fn msg(
        role: llm_suite_types::message::MessageRole,
        text: &str,
    ) -> llm_suite_types::message::Message {
        llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role,
            content: llm_suite_types::message::MessageContentValue::Text(text.to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    fn request(
        messages: Vec<llm_suite_types::message::Message>,
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
            prompt_caching: None,
            cached_content: None,
        }
    }

    #[test]
    fn cached_content_reference_is_forwarded() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(
            vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
            None,
        );
        req.cached_content = Some("cachedContents/abc123".to_string());
        let body = codec.build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["cachedContent"],
            serde_json::json!("cachedContents/abc123")
        );

        // Absent by default: no key is emitted.
        let plain = codec
            .build_body(&request(vec![], None), &profile())
            .expect("must build");
        assert!(plain.get("cachedContent").is_none());
    }

    #[test]
    fn rich_image_blocks_become_inline_data_or_file_data() {
        let codec = GeminiNativeCodec::new();
        let image_msg = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::User,
            content: llm_suite_types::message::MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::Text {
                    text: "what is this".to_string(),
                },
                llm_suite_types::message::MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "data:image/png;base64,AAAA".to_string(),
                        detail: None,
                    },
                },
                llm_suite_types::message::MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
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
            thinking_signature: None,
            metadata: None,
        };
        let entries = codec.convert_messages(&[image_msg]).expect("must convert");
        assert_eq!(entries.len(), 1);
        let parts = entries[0]["parts"].as_array().expect("parts array");
        // Regression guard: none of the three blocks may be silently dropped.
        assert_eq!(parts.len(), 3, "text + inlineData + fileData");
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
        assert_eq!(parts[2]["fileData"]["fileUri"], "https://example.com/a.jpg");
        assert_eq!(parts[2]["fileData"]["mimeType"], "image/jpeg");
    }

    #[test]
    fn the_tool_mode_is_a_top_level_field_of_the_request() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(
            vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
            None,
        );
        req.generation = Some(llm_suite_types::llm::generation::LlmGenerationParams {
            tool_choice: Some(llm_suite_types::llm::generation::LlmToolChoice {
                mode: Some(llm_suite_types::llm::generation::ToolChoiceMode::None),
                tool_name: None,
            }),
            ..Default::default()
        });
        let body = codec.build_body(&req, &profile()).expect("must build");
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
    fn a_plain_text_tool_result_becomes_a_function_response() {
        let codec = GeminiNativeCodec::new();
        let mut assistant = msg(llm_suite_types::message::MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        let mut result = msg(llm_suite_types::message::MessageRole::Tool, "sunny");
        result.tool_call_id = Some("call_1".to_string());

        let contents = codec
            .convert_messages(&[assistant, result])
            .expect("must convert");
        assert_eq!(contents[0]["role"], serde_json::json!("model"));
        assert_eq!(
            contents[0]["parts"][0]["functionCall"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(contents[1]["role"], serde_json::json!("user"));
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather"),
            "the function name comes from the call the result answers"
        );
        assert!(
            contents[1]["parts"][0]["functionResponse"]
                .get("id")
                .is_none(),
            "a call id the provider never issued must not be echoed back"
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["response"],
            serde_json::json!({"output": "sunny"}),
            "the response payload is an object, the text lands in `output`"
        );
    }

    #[test]
    fn a_tool_result_without_a_known_function_is_rejected() {
        let codec = GeminiNativeCodec::new();
        let mut result = msg(llm_suite_types::message::MessageRole::Tool, "sunny");
        result.tool_call_id = Some("call_unknown".to_string());
        let err = codec
            .convert_messages(&[result])
            .expect_err("an unnamed function response is meaningless");
        assert!(format!("{err}").contains("call_unknown"), "{err}");
    }

    #[test]
    fn the_provider_call_id_is_kept_and_echoed_back() {
        let codec = GeminiNativeCodec::new();
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"functionCall": {"id": "fc-1", "name": "get_weather", "args": {}}}
            ]}}]
        }"#;
        let result = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        let calls = result.tool_calls.expect("tool calls present");
        assert_eq!(calls[0].id, "fc-1", "the provider call id must survive");
        assert_eq!(
            calls[0].call_id.as_deref(),
            Some("fc-1"),
            "the provider call id is the correlation id"
        );

        let mut tool_result = msg(llm_suite_types::message::MessageRole::Tool, "sunny");
        tool_result.tool_call_id = Some("fc-1".to_string());
        let contents = codec
            .convert_messages(&[result.message.clone(), tool_result])
            .expect("must convert");
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["id"],
            serde_json::json!("fc-1")
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather")
        );
    }

    #[test]
    fn a_rich_tool_use_block_becomes_a_function_call_part() {
        let codec = GeminiNativeCodec::new();
        let mut assistant = msg(llm_suite_types::message::MessageRole::Assistant, "checking");
        assistant.content = llm_suite_types::message::MessageContentValue::Rich(vec![
            llm_suite_types::message::MessageContent::Text {
                text: "checking".to_string(),
            },
            llm_suite_types::message::MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Beijing"}),
                },
            },
        ]);
        let contents = codec.convert_messages(&[assistant]).expect("must convert");
        let parts = contents[0]["parts"].as_array().expect("parts");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], serde_json::json!("checking"));
        assert_eq!(
            parts[1]["functionCall"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            parts[1]["functionCall"]["args"]["city"],
            serde_json::json!("Beijing")
        );
    }

    #[test]
    fn a_rich_tool_result_names_the_function_of_its_tool_use_block() {
        let codec = GeminiNativeCodec::new();
        let mut assistant = msg(llm_suite_types::message::MessageRole::Assistant, "");
        assistant.content = llm_suite_types::message::MessageContentValue::Rich(vec![
            llm_suite_types::message::MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({}),
                },
            },
        ]);
        let mut result = msg(llm_suite_types::message::MessageRole::Tool, "");
        result.content = llm_suite_types::message::MessageContentValue::Rich(vec![
            llm_suite_types::message::MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    is_error: None,
                },
            },
        ]);

        let contents = codec
            .convert_messages(&[assistant, result])
            .expect("must convert");
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather")
        );
        assert!(
            contents[1]["parts"][0]["functionResponse"]
                .get("id")
                .is_none(),
            "a call id the provider never issued must not be echoed back"
        );
    }

    #[test]
    fn a_message_without_parts_is_skipped() {
        let codec = GeminiNativeCodec::new();
        let empty = msg(llm_suite_types::message::MessageRole::User, "");
        let contents = codec
            .convert_messages(&[
                empty,
                msg(llm_suite_types::message::MessageRole::User, "hi"),
            ])
            .expect("must convert");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["parts"][0]["text"], serde_json::json!("hi"));
    }

    #[test]
    fn a_thought_signature_on_the_thought_part_is_kept() {
        let codec = GeminiNativeCodec::new();
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"text": "planning", "thought": true, "thoughtSignature": "sig-1"},
                {"functionCall": {"name": "f", "args": {}}}
            ]}}]
        }"#;
        let result = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(
            result.message.metadata.as_ref().unwrap()["thought_signatures"],
            serde_json::json!(["sig-1"])
        );
    }

    #[test]
    fn native_mode_sends_system_instruction() {
        let codec = GeminiNativeCodec::new();
        let req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hello"),
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
        let tool_msg = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Tool,
            content: llm_suite_types::message::MessageContentValue::Rich(vec![
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
            vec![
                tool_msg,
                msg(llm_suite_types::message::MessageRole::User, "next"),
            ],
            None,
        );
        let body = codec.build_body(&req, &profile()).expect("must build");
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
        let codec = GeminiNativeCodec::new();
        let req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hello"),
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
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hello"),
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
        assert!(
            body.get("systemInstruction").is_none(),
            "the counting endpoint does not take a system instruction"
        );
        assert!(
            body.get("generationConfig").is_none(),
            "count body must not carry output steering"
        );
    }

    #[test]
    fn stream_request_targets_stream_generate_content_sse() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(
            vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
            None,
        );
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
            .build_request(
                &request(
                    vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
                    None,
                ),
                &profile(),
            )
            .expect("must build");
        assert_eq!(
            req.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn a_model_that_already_carries_the_prefix_is_not_prefixed_twice() {
        let codec = GeminiNativeCodec::new();
        let mut p = profile();
        p.model = "models/gemini-1.5-pro".to_string();
        let req = codec
            .build_request(
                &request(
                    vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
                    None,
                ),
                &p,
            )
            .expect("must build");
        assert_eq!(
            req.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn a_base_url_with_a_query_still_yields_a_valid_streaming_url() {
        let codec = GeminiNativeCodec::new();
        let mut p = profile();
        p.base_url = Some("https://gateway.example.com/v1?api-version=2024-01".to_string());
        let mut req = request(
            vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
            None,
        );
        req.stream = Some(true);
        let built = codec.build_request(&req, &p).expect("must build");
        assert_eq!(
            built.url().as_str(),
            "https://gateway.example.com/v1/models/gemini-1.5-pro:streamGenerateContent?api-version=2024-01&alt=sse"
        );
    }

    #[test]
    fn the_system_instruction_carries_a_role() {
        let codec = GeminiNativeCodec::new();
        let body = codec
            .build_body(
                &request(
                    vec![msg(
                        llm_suite_types::message::MessageRole::System,
                        "You are a helper",
                    )],
                    None,
                ),
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
    fn the_tool_call_numbering_restarts_with_every_response() {
        let codec = GeminiNativeCodec::new();
        let call_chunk =
            r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}"#;
        let end_chunk = r#"{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}"#;

        let first = codec
            .parse_stream_chunk_events(call_chunk)
            .expect("must parse");
        let index = match &first[0] {
            MessageStreamEvent::ToolCallDelta(delta) => delta.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_eq!(index, 0, "the first call of a response is number zero");

        // The response ends; the next response starts the numbering again.
        codec
            .parse_stream_chunk_events(end_chunk)
            .expect("must parse");
        let second = codec
            .parse_stream_chunk_events(call_chunk)
            .expect("must parse");
        match &second[0] {
            MessageStreamEvent::ToolCallDelta(delta) => {
                assert_eq!(
                    delta.index, 0,
                    "the numbering describes the position inside the response"
                );
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn the_part_fields_use_the_official_camel_case_names() {
        let codec = GeminiNativeCodec::new();
        let mut assistant = msg(llm_suite_types::message::MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        let mut result = msg(llm_suite_types::message::MessageRole::Tool, "sunny");
        result.tool_call_id = Some("call_1".to_string());

        let contents = codec
            .convert_messages(&[assistant, result])
            .expect("must convert");
        let call = &contents[0]["parts"][0];
        assert!(
            call.get("functionCall").is_some(),
            "the call part uses the official name: {contents:?}"
        );
        assert!(call.get("function_call").is_none());
        let response = &contents[1]["parts"][0];
        assert!(
            response.get("functionResponse").is_some(),
            "the response part uses the official name: {contents:?}"
        );
        assert!(response.get("function_response").is_none());
    }

    #[test]
    fn count_tokens_body_covers_the_same_contents_as_the_inference_request() {
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
        assert!(
            count_body.get("systemInstruction").is_none(),
            "the counting endpoint does not take a system instruction"
        );
        assert!(count_body.get("tools").is_none());
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

    #[test]
    fn the_whole_safety_family_reports_the_content_filter() {
        let codec = GeminiNativeCodec::new();
        for raw in [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "NO_IMAGE",
            "IMAGE_RECITATION",
            "IMAGE_OTHER",
        ] {
            let body = format!(
                r#"{{"candidates":[{{"content":{{"parts":[{{"text":"x"}}]}},"finishReason":"{raw}"}}]}}"#
            );
            let result = codec
                .parse_response(&body, &request(vec![], None))
                .expect("must parse");
            assert_eq!(
                result.finish_reason.as_deref(),
                Some("content_filter"),
                "{raw} must report the content filter"
            );
        }

        // The other official reasons keep their own meaning.
        let body =
            r#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"MAX_TOKENS"}]}"#;
        let result = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(result.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn untyped_provider_options_reach_the_request_body() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(vec![], None);
        req.parameters = Some(serde_json::json!({
            "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH"}]
        }));
        let body = codec.build_body(&req, &profile()).expect("must build");
        assert_eq!(
            body["safetySettings"][0]["category"],
            serde_json::json!("HARM_CATEGORY_HATE_SPEECH")
        );
    }

    #[test]
    fn the_official_generation_config_block_is_parsed_not_overwritten() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(vec![], None);
        req.parameters = Some(serde_json::json!({
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 256,
                "stopSequences": ["END"]
            }
        }));
        let body = codec.build_body(&req, &profile()).expect("must build");
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
        let codec = GeminiNativeCodec::new();
        let mut req = request(vec![], None);
        req.parameters = Some(serde_json::json!({
            "responseMimeType": "application/json",
            "responseSchema": {"type": "object"},
            "candidateCount": 2
        }));
        let body = codec.build_body(&req, &profile()).expect("must build");
        for key in ["responseMimeType", "responseSchema", "candidateCount"] {
            assert!(body.get(key).is_none(), "`{key}` must not leak: {body}");
        }
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
    }
}
