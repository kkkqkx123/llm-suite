use super::LlmCodec;
use crate::codecs::helpers::{resolve_image_payload, ResolvedImage};
use crate::error::{LlmError, LlmResult};
use llm_suite_types::llm::{
    LlmProfile, LlmPromptCaching, LlmRequest, LlmResult as LlmResponseType, MessageStreamEnd,
    MessageStreamError, MessageStreamEvent, MessageStreamInputJson, MessageStreamReasoning,
    MessageStreamRedactedThinking, MessageStreamRefusal, MessageStreamStart, MessageStreamText,
    MessageStreamToolCallDelta, MessageStreamUsage, TokenUsageStats,
};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};
use llm_suite_types::tool::{Tool, ToolParameterSchema};
use reqwest::Method;

/// Media types accepted by the Anthropic Messages API for image blocks.
const ANTHROPIC_IMAGE_MEDIA_TYPES: [&str; 4] =
    ["image/jpeg", "image/png", "image/gif", "image/webp"];

pub struct AnthropicCodec {
    base_url: String,
    /// Reused HTTP client: constructing a `reqwest::Client` per request would
    /// discard the connection pool and TLS session cache on every call.
    client: reqwest::Client,
}

impl Default for AnthropicCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicCodec {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.anthropic.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    /// Build the request body (separated from the HTTP layer for testability).
    fn build_body(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let use_text_mode = super::shared::is_text_mode(request);

        // `convert_messages` skips system messages; the system content is sent
        // in the dedicated `system` field. Text mode injects the original
        // system + tool usage instructions + declarations; native mode keeps
        // the original system message.
        let (system_content, _) =
            llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);

        let history = if use_text_mode {
            super::shared::convert_history_for_text_mode(&request.messages, request)
        } else {
            request.messages.clone()
        };
        let messages = self.convert_messages(&history)?;

        let mut body = serde_json::json!({
            "model": profile.model,
            "messages": messages,
        });

        if use_text_mode {
            let system = super::shared::text_mode_system_content(request);
            if !system.is_empty() {
                body["system"] = serde_json::json!(system);
            }
        } else if let Some(system) = system_content {
            if !system.is_empty() {
                body["system"] = serde_json::json!(system);
            }
        }

        let generation = super::shared::resolve_generation(request, profile)?;
        crate::generation::apply_anthropic(&mut body, &generation)?;
        super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

        if !use_text_mode {
            if let Some(tools) = &request.tools {
                body["tools"] = serde_json::json!(Self::build_tool_defs(tools));
            }
        }

        if request.stream == Some(true) {
            body["stream"] = serde_json::json!(true);
        }

        super::shared::apply_custom_body(&mut body, profile);

        // Per-request override wins over the profile-level decision,
        // including explicitly disabling caching for a caching profile.
        let request_caching = request
            .prompt_caching
            .as_ref()
            .map(|c| LlmPromptCaching { ttl: c.ttl.clone() });
        let caching: Option<&LlmPromptCaching> = match &request.prompt_caching {
            Some(req) if !req.enabled => None,
            Some(_) => request_caching.as_ref(),
            None => profile.prompt_caching.as_ref(),
        };
        if let Some(caching) = caching {
            Self::apply_prompt_caching(&mut body, caching);
        }

        Ok(body)
    }

    /// Build the count-tokens request body (system included so the count
    /// matches what `build_request` sends).
    fn build_count_tokens_body(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let use_text_mode = super::shared::is_text_mode(request);
        let history = if use_text_mode {
            super::shared::convert_history_for_text_mode(&request.messages, request)
        } else {
            request.messages.clone()
        };
        let messages = self.convert_messages(&history)?;

        let mut body = serde_json::json!({
            "model": profile.model,
            "messages": messages,
        });

        if use_text_mode {
            let system = super::shared::text_mode_system_content(request);
            if !system.is_empty() {
                body["system"] = serde_json::json!(system);
            }
        } else {
            let (system, _) =
                llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);
            if let Some(system) = system {
                if !system.is_empty() {
                    body["system"] = serde_json::json!(system);
                }
            }
        }
        if !use_text_mode {
            if let Some(tools) = &request.tools {
                body["tools"] = serde_json::json!(Self::build_tool_defs(tools));
            }
        }

        Ok(body)
    }

    /// Apply prompt-caching breakpoints (official `CacheControlEphemeralParam`
    /// shape: `{"type": "ephemeral", "ttl": "5m" | "1h"}`) on the last tool
    /// declaration, the system block, the first message and the last message
    /// — the four breakpoints the API allows.
    fn apply_prompt_caching(body: &mut serde_json::Value, caching: &LlmPromptCaching) {
        let mut control = serde_json::json!({"type": "ephemeral"});
        if let Some(ttl) = &caching.ttl {
            control["ttl"] = serde_json::json!(ttl);
        }

        // Last tool declaration.
        if let Some(tools) = body.get_mut("tools").and_then(|v| v.as_array_mut()) {
            if let Some(last) = tools.last_mut() {
                last["cache_control"] = control.clone();
            }
        }

        // System block: sent as a plain string elsewhere; the block-array form
        // is required to carry `cache_control`.
        if let Some(system) = body.get("system").and_then(|v| v.as_str()) {
            body["system"] = serde_json::json!([{
                "type": "text",
                "text": system,
                "cache_control": control.clone(),
            }]);
        }

        // First and last message: the stable prefix and the current turn.
        let message_count = body
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|messages| messages.len())
            .unwrap_or(0);
        for index in [0, message_count.saturating_sub(1)] {
            if message_count == 0 {
                break;
            }
            if let Some(last_block) = body
                .get_mut("messages")
                .and_then(|v| v.as_array_mut())
                .and_then(|m| m.get_mut(index))
                .and_then(|m| m.get_mut("content"))
                .and_then(|v| v.as_array_mut())
                .and_then(|b| b.last_mut())
            {
                last_block["cache_control"] = control.clone();
            }
        }
    }

    /// Cache-write input tokens: prefer the explicit total
    /// (`cache_creation_input_tokens`), fall back to the sum of the TTL
    /// breakdown `cache_creation.{ephemeral_5m_input_tokens,
    /// ephemeral_1h_input_tokens}`. Returns `None` when neither is present.
    fn cache_write_tokens(usage: &serde_json::Value) -> Option<u32> {
        if let Some(total) = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
        {
            return Some(total as u32);
        }
        let creation = usage.get("cache_creation")?;
        let sum: u64 = ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"]
            .iter()
            .filter_map(|k| creation.get(*k).and_then(|v| v.as_u64()))
            .sum();
        (sum > 0).then_some(sum as u32)
    }

    /// Wire tool declarations. `input_schema` is mandatory for Anthropic and
    /// must be a JSON Schema object; a missing schema falls back to an empty
    /// object schema instead of serializing `null` (which the API rejects).
    fn build_tool_defs(tools: &[Tool]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                let input_schema = t.parameters.clone().unwrap_or(ToolParameterSchema {
                    r#type: "object".to_string(),
                    properties: Default::default(),
                    required: Vec::new(),
                    additional_properties: None,
                });
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": input_schema,
                })
            })
            .collect()
    }

    /// Convert internal messages into Anthropic `messages`.
    ///
    /// Enforces the Anthropic contract:
    /// - roles strictly alternate between `user` and `assistant`; consecutive
    ///   same-role messages (parallel tool results, follow-up user texts) are
    ///   folded into a single message with combined content blocks;
    /// - `tool_result` blocks live in `user` messages and are placed first;
    /// - content blocks are non-empty (`text` requires at least one char);
    /// - images use `base64` or `url` sources with a supported media type;
    /// - `thinking` blocks are only replayed when they carry a `signature`
    ///   (the API rejects unsigned thinking blocks).
    fn convert_messages(&self, messages: &[Message]) -> LlmResult<Vec<serde_json::Value>> {
        // First pass: map every message to (role, blocks), dropping messages
        // that produce no blocks at all.
        let mut entries: Vec<(String, Vec<serde_json::Value>)> = Vec::new();

        for msg in messages {
            let role = match msg.role {
                MessageRole::System => continue,
                MessageRole::User | MessageRole::Tool => "user",
                MessageRole::Assistant => "assistant",
            };

            let mut tool_result_blocks: Vec<serde_json::Value> = Vec::new();
            let mut blocks: Vec<serde_json::Value> = Vec::new();

            let rich_has_tool_result = matches!(
                &msg.content,
                MessageContentValue::Rich(parts)
                    if parts.iter().any(|b| matches!(b, MessageContent::ToolResult { .. }))
            );

            // A message that carries thinking blocks keeps them, each with its
            // own signature; the flat summary fields are only a fallback for
            // messages that have none.
            let mut saw_thinking_block = false;
            match &msg.content {
                MessageContentValue::Text(text) => {
                    // On a tool-result message the text IS the tool output; it
                    // is consumed by the `tool_result` synthesis below instead
                    // of becoming a separate (duplicate) text block.
                    if msg.tool_call_id.is_none() && !text.trim().is_empty() {
                        blocks.push(serde_json::json!({"type": "text", "text": text}));
                    }
                }
                MessageContentValue::Rich(parts) => {
                    for block in parts {
                        match block {
                            MessageContent::Text { text } => {
                                if !text.trim().is_empty() {
                                    blocks.push(serde_json::json!({"type": "text", "text": text}));
                                }
                            }
                            MessageContent::ImageUrl { image_url } => {
                                blocks.push(Self::convert_image_block(&image_url.url)?);
                            }
                            MessageContent::ToolUse { tool_use } => {
                                blocks.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": tool_use.id,
                                    "name": tool_use.name,
                                    "input": tool_use.input,
                                }));
                            }
                            MessageContent::ToolResult { tool_result } => {
                                tool_result_blocks.push(Self::tool_result_block(
                                    &tool_result.tool_use_id,
                                    &tool_result.content,
                                    tool_result.is_error,
                                ));
                            }
                            MessageContent::Thinking {
                                thinking,
                                signature,
                            } => {
                                saw_thinking_block = true;
                                // Only thinking blocks with a verifiable
                                // signature may be replayed; unsigned ones are
                                // rejected by the API and must not be faked.
                                if let Some(signature) = signature {
                                    if !signature.is_empty() {
                                        blocks.push(serde_json::json!({
                                            "type": "thinking",
                                            "thinking": thinking,
                                            "signature": signature,
                                        }));
                                    }
                                }
                            }
                            MessageContent::RedactedThinking { data } => {
                                // The redacted payload is opaque and must be
                                // sent back verbatim.
                                blocks.push(serde_json::json!({
                                    "type": "redacted_thinking",
                                    "data": data,
                                }));
                            }
                        }
                    }
                }
            }

            // Assistant thinking captured as plain text plus a provider
            // signature (retained from a previous reply) is re-emitted as a
            // signed thinking block so multi-turn extended thinking keeps
            // working. Unsigned thinking is never synthesized: the API
            // rejects it. Messages that already carry thinking blocks keep
            // their own blocks, each with its own signature.
            //
            // The block goes to the front of the content: the API requires
            // the thinking blocks to precede the text and the tool calls of
            // the assistant turn.
            if role == "assistant" && !saw_thinking_block {
                if let Some(thinking) = msg.thinking.as_deref() {
                    if !thinking.is_empty() {
                        if let Some(signature) = msg.thinking_signature.as_deref() {
                            if !signature.is_empty() {
                                blocks.insert(
                                    0,
                                    serde_json::json!({
                                        "type": "thinking",
                                        "thinking": thinking,
                                        "signature": signature,
                                    }),
                                );
                            }
                        }
                    }
                }
            }

            // OpenAI-style tool_calls carried on the message itself.
            // Unparseable arguments fall back to `{}`: the API requires
            // `input` to be an object, and dropping the whole call would lose
            // more than a degraded empty input.
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments)
                    {
                        Ok(value) => value,
                        Err(_) => {
                            tracing::debug!(
                                tool_id = tc.id.as_str(),
                                tool_name = tc.function.name.as_str(),
                                arguments_preview =
                                    tc.function.arguments.chars().take(200).collect::<String>(),
                                "assistant tool call arguments are not valid JSON; falling back to empty object",
                            );
                            serde_json::Value::Object(serde_json::Map::new())
                        }
                    };
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": args,
                    }));
                }
            }

            // A message addressed to one tool call via `tool_call_id` with
            // plain-text content becomes a `tool_result` block, unless the
            // rich content already carries `ToolResult` blocks.
            if msg.tool_call_id.is_some() && !rich_has_tool_result {
                let content_text = match &msg.content {
                    MessageContentValue::Text(text) => text.clone(),
                    MessageContentValue::Rich(parts) => parts
                        .iter()
                        .filter_map(|b| match b {
                            MessageContent::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                let is_error = msg
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("is_error"))
                    .and_then(|v| v.as_bool());
                tool_result_blocks.push(Self::tool_result_block(
                    msg.tool_call_id.as_deref().unwrap_or_default(),
                    &content_text,
                    is_error,
                ));
            }

            // `tool_result` blocks must come first inside their message.
            let mut message_blocks = tool_result_blocks;
            message_blocks.extend(blocks);

            // The thinking blocks must lead the assistant turn: the API
            // rejects a thinking block that follows the text or the tool
            // calls of its turn.
            if role == "assistant" {
                message_blocks = Self::thinking_blocks_first(message_blocks);
            }

            if message_blocks.is_empty() {
                continue;
            }

            // A `tool_result` block inside an assistant message would be
            // rejected by the API; route those blocks into a user turn.
            if role == "assistant" {
                let has_tool_result = message_blocks
                    .iter()
                    .any(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"));
                if has_tool_result {
                    let assistant_blocks: Vec<serde_json::Value> = message_blocks
                        .iter()
                        .filter(|b| b.get("type").and_then(|v| v.as_str()) != Some("tool_result"))
                        .cloned()
                        .collect();
                    if !assistant_blocks.is_empty() {
                        entries.push(("assistant".to_string(), assistant_blocks));
                    }
                    let tool_results: Vec<serde_json::Value> = message_blocks
                        .into_iter()
                        .filter(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
                        .collect();
                    entries.push(("user".to_string(), tool_results));
                    continue;
                }
            }

            entries.push((role.to_string(), message_blocks));
        }

        // Second pass: merge consecutive same-role entries so the final list
        // strictly alternates (Anthropic rejects consecutive same-role
        // messages, e.g. parallel tool results).
        let mut result: Vec<serde_json::Value> = Vec::with_capacity(entries.len());
        for (role, blocks) in entries {
            match result.last_mut() {
                Some(last) if last["role"] == serde_json::json!(role) => {
                    if let Some(arr) = last["content"].as_array_mut() {
                        arr.extend(blocks);
                    }
                }
                _ => result.push(serde_json::json!({"role": role, "content": blocks})),
            }
        }

        Ok(result)
    }

    /// Move the thinking blocks of an assistant turn in front of the other
    /// blocks.
    ///
    /// The Messages API requires the thinking blocks to be the first blocks of
    /// the assistant turn; a history that stores the text before the thinking
    /// would otherwise be rejected.
    fn thinking_blocks_first(blocks: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        let is_thinking = |block: &serde_json::Value| {
            matches!(
                block.get("type").and_then(|v| v.as_str()),
                Some("thinking") | Some("redacted_thinking")
            )
        };
        let mut ordered: Vec<serde_json::Value> = Vec::with_capacity(blocks.len());
        ordered.extend(blocks.iter().filter(|b| is_thinking(b)).cloned());
        ordered.extend(blocks.iter().filter(|b| !is_thinking(b)).cloned());
        ordered
    }

    fn tool_result_block(
        tool_use_id: &str,
        content: &str,
        is_error: Option<bool>,
    ) -> serde_json::Value {
        let mut block = serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        });
        // `is_error` is optional; only set it when true to keep the wire
        // format minimal.
        if is_error == Some(true) {
            block["is_error"] = serde_json::json!(true);
        }
        block
    }

    /// Map an image reference to an Anthropic image block. Data URIs become
    /// `base64` sources, http(s) URLs use the native `url` source, anything
    /// else is rejected up front.
    fn convert_image_block(url: &str) -> LlmResult<serde_json::Value> {
        match resolve_image_payload(url)? {
            ResolvedImage::Base64 { media_type, data } => {
                if !ANTHROPIC_IMAGE_MEDIA_TYPES.contains(&media_type.as_str()) {
                    return Err(LlmError::ConfigError(format!(
                        "Anthropic image media type `{media_type}` is not supported (expected one of jpeg/png/gif/webp)"
                    )));
                }
                Ok(serde_json::json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }))
            }
            ResolvedImage::Url(url) => Ok(serde_json::json!({
                "type": "image",
                "source": {"type": "url", "url": url},
            })),
        }
    }

    /// Parse one Anthropic SSE chunk into zero or more stream events.
    ///
    /// One wire event can map to several internal events: `message_delta`
    /// carries both cumulative usage and the final stop reason, so it emits
    /// `Usage` followed by `End` (the accumulator tolerates the duplicate
    /// `End` from the subsequent `message_stop`).
    fn parse_stream_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let json: serde_json::Value = serde_json::from_str(data)?;
        let event_type = json.get("type").and_then(|v| v.as_str());

        match event_type {
            Some("message_start") => {
                let mut events = Vec::new();
                // Surface the provider message identity first so the
                // accumulator records it before any terminal event.
                let stream_id = json
                    .pointer("/message/id")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let stream_model = json
                    .pointer("/message/model")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                if stream_id.is_some() || stream_model.is_some() {
                    events.push(MessageStreamEvent::Start(MessageStreamStart {
                        id: stream_id,
                        model: stream_model,
                    }));
                }
                // Initial usage: `message_delta` may omit `input_tokens`
                // (MessageDeltaUsage.input_tokens is optional), so it must be
                // captured here or stream-side prompt tokens are lost.
                if let Some(usage) = json.pointer("/message/usage") {
                    if let Some(event) = Self::stream_usage(usage) {
                        events.push(MessageStreamEvent::Usage(event));
                    }
                }
                Ok(events)
            }
            Some("content_block_start") => {
                if let Some(b) = json.get("content_block") {
                    match b.get("type").and_then(|v| v.as_str()) {
                        Some("tool_use") => {
                            let index =
                                json.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let id = b.get("id").and_then(|v| v.as_str()).map(String::from);
                            let name = b.get("name").and_then(|v| v.as_str()).map(String::from);
                            return Ok(vec![MessageStreamEvent::ToolCallDelta(
                                MessageStreamToolCallDelta {
                                    index,
                                    id,
                                    call_id: None,
                                    name,
                                    arguments: None,
                                    thought_signature: None,
                                    is_snapshot: false,
                                },
                            )]);
                        }
                        // The redacted payload only appears in the start
                        // frame; no delta frame carries it. It has to reach
                        // the consumer so the reply can be replayed in the
                        // next turn.
                        Some("redacted_thinking") => {
                            if let Some(data) = b.get("data").and_then(|v| v.as_str()) {
                                return Ok(vec![MessageStreamEvent::RedactedThinking(
                                    MessageStreamRedactedThinking {
                                        data: data.to_string(),
                                    },
                                )]);
                            }
                        }
                        // A thinking block may open with the first part of
                        // its text; the signature arrives later, in a
                        // `signature_delta`.
                        Some("thinking") => {
                            let thinking = b.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                            if !thinking.is_empty() {
                                return Ok(vec![MessageStreamEvent::ReasoningText(
                                    MessageStreamReasoning {
                                        reasoning: thinking.to_string(),
                                        snapshot: String::new(),
                                        signature: None,
                                        signature_snapshot: None,
                                    },
                                )]);
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Vec::new())
            }
            Some("content_block_delta") => {
                if let Some(d) = json.get("delta") {
                    match d.get("type").and_then(|v| v.as_str()) {
                        Some("text_delta") => {
                            if let Some(text) = d.get("text").and_then(|v| v.as_str()) {
                                // `snapshot` is rebuilt by the stream
                                // accumulator; keep the placeholder empty.
                                return Ok(vec![MessageStreamEvent::Text(MessageStreamText {
                                    text: text.to_string(),
                                    snapshot: String::new(),
                                })]);
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(reasoning) = d.get("thinking").and_then(|v| v.as_str()) {
                                return Ok(vec![MessageStreamEvent::ReasoningText(
                                    MessageStreamReasoning {
                                        reasoning: reasoning.to_string(),
                                        snapshot: String::new(),
                                        signature: None,
                                        signature_snapshot: None,
                                    },
                                )]);
                            }
                        }
                        Some("signature_delta") => {
                            if let Some(signature) = d.get("signature").and_then(|v| v.as_str()) {
                                return Ok(vec![MessageStreamEvent::ReasoningText(
                                    MessageStreamReasoning {
                                        reasoning: String::new(),
                                        snapshot: String::new(),
                                        signature: Some(signature.to_string()),
                                        signature_snapshot: None,
                                    },
                                )]);
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(partial) = d.get("partial_json").and_then(|v| v.as_str()) {
                                let index = json.get("index").and_then(|v| v.as_u64()).unwrap_or(0)
                                    as usize;
                                // The incremental JSON parser of the stream
                                // accumulator works on this event: it
                                // refreshes the parsed snapshot of the
                                // arguments buffer while it is still
                                // incomplete.
                                return Ok(vec![MessageStreamEvent::InputJson(
                                    MessageStreamInputJson {
                                        partial_json: partial.to_string(),
                                        index: Some(index),
                                        parsed_snapshot: None,
                                    },
                                )]);
                            }
                        }
                        // The sources of a grounded answer travel on their
                        // own channel so they never mix into the answer text.
                        Some("citations_delta") => {
                            if let Some(citation) = d.get("citation") {
                                return Ok(vec![MessageStreamEvent::Annotation(
                                    llm_suite_types::llm::MessageStreamAnnotation {
                                        annotation: citation.clone(),
                                    },
                                )]);
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Vec::new())
            }
            Some("message_delta") => {
                let mut events = Vec::new();
                if let Some(usage) = json.get("usage") {
                    if let Some(event) = Self::stream_usage(usage) {
                        events.push(MessageStreamEvent::Usage(event));
                    }
                }
                // A refusal reports the reason of the refusal in the
                // official `stop_details` object; it travels on the refusal
                // channel, like the Chat Completions refusal delta.
                if json.pointer("/delta/stop_reason").and_then(|v| v.as_str()) == Some("refusal") {
                    let refusal = json
                        .pointer("/delta/stop_details/explanation")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .or_else(|| {
                            json.pointer("/delta/stop_details/category")
                                .and_then(|v| v.as_str())
                                .map(|category| format!("refused: {category}"))
                        });
                    if let Some(refusal) = refusal {
                        events.push(MessageStreamEvent::Refusal(MessageStreamRefusal {
                            refusal,
                            snapshot: String::new(),
                        }));
                    }
                }
                if let Some(stop_reason) =
                    json.pointer("/delta/stop_reason").and_then(|v| v.as_str())
                {
                    // Everything except `pause_turn` terminates the turn
                    // (the official StopReason set also includes `refusal`
                    // and `model_context_window_exceeded`).
                    if stop_reason != "pause_turn" {
                        events.push(MessageStreamEvent::End(MessageStreamEnd {}));
                    }
                }
                Ok(events)
            }
            Some("message_stop") => Ok(vec![MessageStreamEvent::End(MessageStreamEnd {})]),
            Some("error") => {
                // Mid-stream provider errors must be surfaced, not swallowed.
                let err = &json["error"];
                let err_type = err
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let message = err
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                Ok(vec![MessageStreamEvent::Error(MessageStreamError {
                    error: format!("anthropic {err_type}: {message}"),
                })])
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Build a usage event from an Anthropic usage object. Returns `None`
    /// when the object carries no counters at all, so zero-usage events never
    /// clobber accumulated values.
    fn stream_usage(usage: &serde_json::Value) -> Option<MessageStreamUsage> {
        let prompt_tokens = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion_tokens = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let reasoning_tokens = usage
            .get("output_tokens_details")
            .and_then(|d| d.get("thinking_tokens"))
            .and_then(|v| v.as_u64())
            .map(|r| r as u32);
        let cache_read_tokens = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .map(|r| r as u32);
        let cache_write_tokens = Self::cache_write_tokens(usage);

        if prompt_tokens == 0
            && completion_tokens == 0
            && reasoning_tokens.is_none()
            && cache_read_tokens.is_none()
            && cache_write_tokens.is_none()
        {
            return None;
        }

        Some(MessageStreamUsage {
            usage: TokenUsageStats {
                prompt_tokens,
                completion_tokens,
                // The wire carries no total; the stream accumulator
                // recomputes it from the merged counters.
                total_tokens: 0,
                reasoning_tokens,
                cache_read_tokens,
                cache_write_tokens,
                prompt_tokens_cost: None,
                completion_tokens_cost: None,
                total_cost: None,
            },
        })
    }
}

impl LlmCodec for AnthropicCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        let url = super::shared::endpoint_url(
            profile.base_url.as_deref().unwrap_or(&self.base_url),
            "messages",
        );

        let body = self.build_body(request, profile)?;

        let mut req_builder = self
            .client
            .request(Method::POST, &url)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&body);

        req_builder = super::shared::apply_auth_and_headers(req_builder, profile, "x-api-key");

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
            "messages/count_tokens",
        );

        let body = self.build_count_tokens_body(request, profile)?;

        let mut req_builder = self
            .client
            .request(Method::POST, &url)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&body);

        // Same auth handling as `build_request` so bearer/custom headers and
        // query params are not silently dropped for count_tokens.
        req_builder = super::shared::apply_auth_and_headers(req_builder, profile, "x-api-key");

        req_builder
            .build()
            .map(Some)
            .map_err(crate::error::LlmError::HttpError)
    }

    fn parse_response(&self, body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
        let mut result = self.parse_anthropic_response(body)?;
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
        Ok(Self::build_tool_defs(tools))
    }

    fn parse_tool_calls(
        &self,
        result: &LlmResponseType,
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
        result.tool_calls.clone().unwrap_or_default()
    }
}

impl AnthropicCodec {
    fn parse_anthropic_response(&self, body: &str) -> LlmResult<LlmResponseType> {
        let json: serde_json::Value = serde_json::from_str(body)?;

        // Error payloads ({"type":"error",...}) must not be parsed into an
        // empty message; surface the provider error instead.
        if json.get("type").and_then(|v| v.as_str()) == Some("error") {
            let err = &json["error"];
            let err_type = err
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(LlmError::InvalidResponse(format!(
                "anthropic error ({err_type}): {message}"
            )));
        }

        let id = json.get("id").and_then(|v| v.as_str()).map(String::from);
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // Provider-native value passed through untouched (`end_turn` /
        // `tool_use` / `max_tokens` / `stop_sequence` / `pause_turn` /
        // `refusal` / `model_context_window_exceeded`); never normalized here.
        let stop_reason = json
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(String::from);

        let content_arr = json.get("content").and_then(|v| v.as_array());
        let mut text_content = String::new();
        let mut thinking_content = String::new();
        let mut thinking_signature: Option<String> = None;
        let mut tool_calls = Vec::new();
        // Content blocks are kept per block so every thinking block keeps its
        // own signature when the message is replayed in a later turn.
        let mut content_blocks: Vec<MessageContent> = Vec::new();
        // Sources of a grounded answer; the official text block carries them
        // on `citations`.
        let mut citations: Vec<serde_json::Value> = Vec::new();

        if let Some(blocks) = content_arr {
            for block in blocks {
                let block_type = block.get("type").and_then(|v| v.as_str());
                match block_type {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                            text_content.push_str(text);
                            content_blocks.push(MessageContent::Text {
                                text: text.to_string(),
                            });
                        }
                        if let Some(block_citations) =
                            block.get("citations").and_then(|v| v.as_array())
                        {
                            citations.extend(block_citations.iter().cloned());
                        }
                    }
                    Some("thinking") => {
                        // Extended thinking output; surfaced through
                        // `reasoning_content` / `message.thinking`. The first
                        // non-empty block signature is retained so the reply
                        // can be replayed in later turns.
                        if let Some(text) = block.get("thinking").and_then(|v| v.as_str()) {
                            thinking_content.push_str(text);
                            let signature = block
                                .get("signature")
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            if thinking_signature.is_none() {
                                if let Some(signature) = signature.as_ref() {
                                    if !signature.is_empty() {
                                        thinking_signature = Some(signature.clone());
                                    }
                                }
                            }
                            content_blocks.push(MessageContent::Thinking {
                                thinking: text.to_string(),
                                signature,
                            });
                        }
                    }
                    Some("redacted_thinking") => {
                        // The provider redacted the reasoning; the payload is
                        // opaque and has to be sent back verbatim.
                        if let Some(data) = block.get("data").and_then(|v| v.as_str()) {
                            content_blocks.push(MessageContent::RedactedThinking {
                                data: data.to_string(),
                            });
                        }
                    }
                    Some("tool_use") => {
                        let tc_id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        let arguments = serde_json::to_string(&input).unwrap_or_default();
                        tool_calls.push(llm_suite_types::message::LlmToolCall {
                            id: tc_id,
                            r#type: "function".to_string(),
                            call_id: None,
                            function: llm_suite_types::message::LlmFunctionCall { name, arguments },
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = json.get("usage").map(|u| {
            let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let cache_read = u
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let cache_write = Self::cache_write_tokens(u).unwrap_or(0);
            llm_suite_types::llm::TokenUsageStats {
                // Anthropic reports the cached input tokens next to the
                // uncached ones; `prompt_tokens` stays the uncached count so
                // the cache breakdown remains a separate, additive signal.
                prompt_tokens: input,
                completion_tokens: output,
                // The official total input is `input_tokens` plus the cache
                // read and cache creation tokens.
                total_tokens: input + cache_read + cache_write + output,
                // Anthropic-native breakdown (not the OpenAI-shaped
                // `completion_tokens_details.reasoning_tokens`).
                reasoning_tokens: u
                    .get("output_tokens_details")
                    .and_then(|d| d.get("thinking_tokens"))
                    .and_then(|v| v.as_u64())
                    .map(|r| r as u32),
                cache_read_tokens: u
                    .get("cache_read_input_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|r| r as u32),
                cache_write_tokens: Self::cache_write_tokens(u),
                prompt_tokens_cost: None,
                completion_tokens_cost: None,
                total_cost: None,
            }
        });

        let message = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Assistant,
            // Thinking blocks carry a per-block signature and have to survive
            // into the next turn, so the message keeps the block structure
            // whenever the reply contains one.
            content: if content_blocks
                .iter()
                .any(|block| !matches!(block, MessageContent::Text { .. }))
            {
                MessageContentValue::Rich(content_blocks)
            } else {
                MessageContentValue::Text(text_content.clone())
            },
            timestamp: llm_suite_common::time::now(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.clone())
            },
            thinking: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content.clone())
            },
            thinking_signature: thinking_signature.clone(),
            metadata: None,
        };

        let mut metadata = std::collections::HashMap::new();
        if let Some(t) = json.get("type").and_then(|v| v.as_str()) {
            metadata.insert("type".to_string(), serde_json::json!(t));
        }
        if let Some(sr) = &stop_reason {
            metadata.insert("stop_reason".to_string(), serde_json::json!(sr));
        }
        if let Some(seq) = json.get("stop_sequence").and_then(|v| v.as_str()) {
            metadata.insert("stop_sequence".to_string(), serde_json::json!(seq));
        }
        if !citations.is_empty() {
            metadata.insert(
                llm_suite_types::llm::ANNOTATIONS_KEY.to_string(),
                serde_json::json!(citations),
            );
        }
        // A refusal carries the reason of the refusal in the official
        // `stop_details` object; it is the Anthropic counterpart of the
        // Chat Completions `message.refusal` field.
        let stop_details = json.get("stop_details").filter(|v| !v.is_null());
        if let Some(details) = stop_details {
            metadata.insert("stop_details".to_string(), details.clone());
        }
        let refusal = match stop_reason.as_deref() {
            Some("refusal") => stop_details
                .and_then(|d| d.get("explanation"))
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| {
                    // The refusal has no explanation; the category names the
                    // policy that triggered it.
                    stop_details
                        .and_then(|d| d.get("category"))
                        .and_then(|v| v.as_str())
                        .map(|category| format!("refused: {category}"))
                }),
            _ => None,
        };
        let metadata = if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        };

        let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

        Ok(LlmResponseType {
            id,
            model,
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
            finish_reason: stop_reason,
            duration: 0,
            reasoning_content: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content)
            },
            reasoning_signature: thinking_signature,
            reasoning_tokens,
            metadata,
            refusal,
            stream_stats: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::LlmRequestCaching;

    #[test]
    fn content_block_start_tool_use_emits_name_and_id() {
        let chunk = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#;
        match AnthropicCodec::new()
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(delta)) => {
                assert_eq!(delta.index, 1);
                assert_eq!(delta.id.as_deref(), Some("toolu_1"));
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                assert_eq!(delta.arguments, None);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn input_json_delta_emits_arguments_fragment() {
        let chunk = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Beijing\"}"}}"#;
        match AnthropicCodec::new()
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            // The incremental JSON parser of the stream accumulator works on
            // this event.
            Some(MessageStreamEvent::InputJson(input)) => {
                assert_eq!(input.index, Some(1));
                assert_eq!(input.partial_json, r#"{"city":"Beijing"}"#);
            }
            other => panic!("expected InputJson, got {:?}", other),
        }
    }

    fn profile() -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format: llm_suite_types::llm::LlmFormat::Anthropic,
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
    fn a_redacted_thinking_block_is_not_dropped() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"c2VjcmV0"}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        match &events[0] {
            MessageStreamEvent::RedactedThinking(redacted) => {
                assert_eq!(redacted.data, "c2VjcmV0");
            }
            other => panic!("expected RedactedThinking, got {other:?}"),
        }
    }

    #[test]
    fn citations_are_reported_as_metadata() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"m","model":"claude","stop_reason":"end_turn","content":[{"type":"text","text":"grounded","citations":[{"type":"char_location","cited_text":"x","document_index":0,"start_char_index":0,"end_char_index":1}]}]}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        let metadata = out.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["annotations"][0]["type"],
            serde_json::json!("char_location")
        );
    }

    #[test]
    fn the_replayed_thinking_block_comes_first() {
        // The API requires the thinking blocks to precede the text and the
        // tool calls of the assistant turn.
        let codec = AnthropicCodec::new();
        let mut assistant = msg(MessageRole::Assistant, "the answer");
        assistant.thinking = Some("Let me verify".to_string());
        assistant.thinking_signature = Some("sig-abc".to_string());

        let body = codec
            .build_body(
                &request(vec![msg(MessageRole::User, "hi"), assistant], None),
                &profile(),
            )
            .expect("must build");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));
        assert_eq!(content[1]["text"], serde_json::json!("the answer"));
    }

    #[test]
    fn native_mode_sends_system_field() {
        let codec = AnthropicCodec::new();
        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(
                            llm_suite_types::message::MessageRole::System,
                            "You are a helper",
                        ),
                        msg(llm_suite_types::message::MessageRole::User, "Hello"),
                    ],
                    None,
                ),
                &profile(),
            )
            .expect("build must succeed");

        assert_eq!(body["system"], serde_json::json!("You are a helper"));
        let roles: Vec<&str> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["role"].as_str())
            .collect();
        assert_eq!(roles, vec!["user"], "system must not appear in messages");
    }

    #[test]
    fn text_mode_system_field_includes_tool_instructions() {
        let codec = AnthropicCodec::new();
        let mut req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hello"),
            ],
            None,
        );
        req.tool_call_protocol = Some(llm_suite_types::llm::ToolCallProtocol::Xml);
        let body = codec.build_body(&req, &profile()).expect("must build");

        let system = body["system"].as_str().unwrap();
        assert!(system.contains("You are a helper"));
        assert!(system.contains("<tool_use>"), "must carry instructions");
    }

    #[test]
    fn params_pass_through_except_stream_and_stop() {
        let codec = AnthropicCodec::new();
        let req = request(
            vec![msg(llm_suite_types::message::MessageRole::User, "Hello")],
            Some(serde_json::json!({
                "temperature": 0.2,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "tool_choice": {"type": "auto"},
                "stop": ["END"],
                "stream": true,
            })),
        );
        let body = codec.build_body(&req, &profile()).expect("must build");

        assert_eq!(body["thinking"]["budget_tokens"], serde_json::json!(1024));
        assert_eq!(body["tool_choice"]["type"], serde_json::json!("auto"));
        assert_eq!(body["stop_sequences"], serde_json::json!(["END"]));
        assert!(body.get("stop").is_none(), "stop must be mapped");
        assert!(body.get("stream").is_none(), "stream controlled by caller");
        assert_eq!(
            body["temperature"],
            serde_json::json!(0.2),
            "the sampling parameters are still accepted by the API"
        );
    }

    #[test]
    fn tool_choice_none_uses_the_official_shape() {
        use llm_suite_types::llm::generation::{LlmToolChoice, ToolChoiceMode};

        let mut body = serde_json::json!({"model": "claude"});
        crate::generation::apply_anthropic(
            &mut body,
            &llm_suite_types::llm::generation::LlmGenerationParams {
                tool_choice: Some(LlmToolChoice {
                    mode: Some(ToolChoiceMode::None),
                    tool_name: None,
                }),
                parallel_tool_calls: Some(false),
                ..Default::default()
            },
        )
        .expect("tool choice must be accepted");
        assert_eq!(body["tool_choice"]["type"], serde_json::json!("none"));
        assert!(
            body["tool_choice"]
                .get("disable_parallel_tool_use")
                .is_none(),
            "the none variant carries no parallel-tool flag"
        );
    }

    #[test]
    fn count_tokens_includes_system() {
        let codec = AnthropicCodec::new();
        let req = request(
            vec![msg(
                llm_suite_types::message::MessageRole::System,
                "You are a helper",
            )],
            None,
        );
        let body = codec
            .build_count_tokens_body(&req, &profile())
            .expect("must build");
        assert_eq!(body["system"], serde_json::json!("You are a helper"));
    }

    // ---------- helpers for richer messages ----------

    fn rich_msg(
        role: llm_suite_types::message::MessageRole,
        blocks: Vec<llm_suite_types::message::MessageContent>,
    ) -> llm_suite_types::message::Message {
        llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role,
            content: llm_suite_types::message::MessageContentValue::Rich(blocks),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    fn tool_msg(
        tool_use_id: &str,
        text: &str,
        is_error: bool,
    ) -> llm_suite_types::message::Message {
        let mut m = msg(llm_suite_types::message::MessageRole::Tool, text);
        m.tool_call_id = Some(tool_use_id.to_string());
        m.metadata = Some(std::collections::HashMap::from([(
            "is_error".to_string(),
            serde_json::json!(is_error),
        )]));
        m
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> llm_suite_types::message::LlmToolCall {
        llm_suite_types::message::LlmToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    // ---------------- conversion: alternation & tool results ----------------

    #[test]
    fn parallel_tool_results_merge_into_one_user_message() {
        let codec = AnthropicCodec::new();
        let mut assistant = msg(
            llm_suite_types::message::MessageRole::Assistant,
            "Checking weather and time",
        );
        assistant.tool_calls = Some(vec![
            tool_call("call_1", "get_weather", r#"{"city":"Beijing"}"#),
            tool_call("call_2", "get_time", "{}"),
        ]);

        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(
                            llm_suite_types::message::MessageRole::User,
                            "weather and time?",
                        ),
                        assistant,
                        tool_msg("call_1", "sunny", false),
                        tool_msg("call_2", "rainy", false),
                    ],
                    None,
                ),
                &profile(),
            )
            .expect("build must succeed");

        let messages = body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages.iter().filter_map(|m| m["role"].as_str()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "user"],
            "roles must alternate"
        );

        let assistant_blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(assistant_blocks.len(), 3);
        assert_eq!(assistant_blocks[0]["type"], serde_json::json!("text"));
        assert_eq!(assistant_blocks[1]["id"], serde_json::json!("call_1"));
        assert_eq!(assistant_blocks[2]["id"], serde_json::json!("call_2"));

        let user_blocks = messages[2]["content"].as_array().unwrap();
        assert_eq!(
            user_blocks.len(),
            2,
            "both tool results live in one user message"
        );
        assert_eq!(user_blocks[0]["type"], serde_json::json!("tool_result"));
        assert_eq!(user_blocks[0]["tool_use_id"], serde_json::json!("call_1"));
        assert_eq!(user_blocks[0]["content"], serde_json::json!("sunny"));
        assert_eq!(user_blocks[1]["tool_use_id"], serde_json::json!("call_2"));
        assert!(
            user_blocks[0].get("is_error").is_none(),
            "is_error only set when true"
        );
    }

    #[test]
    fn tool_result_is_error_is_preserved() {
        let codec = AnthropicCodec::new();
        let mut assistant = msg(llm_suite_types::message::MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![tool_call("call_1", "boom", "{}")]);

        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(llm_suite_types::message::MessageRole::User, "run"),
                        assistant,
                        tool_msg("call_1", "exploded", true),
                    ],
                    None,
                ),
                &profile(),
            )
            .expect("build must succeed");

        let block = &body["messages"][2]["content"][0];
        assert_eq!(block["is_error"], serde_json::json!(true));
    }

    #[test]
    fn thinking_block_replayed_only_with_signature() {
        let codec = AnthropicCodec::new();

        let with_signature = rich_msg(
            llm_suite_types::message::MessageRole::Assistant,
            vec![
                llm_suite_types::message::MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
                },
                llm_suite_types::message::MessageContent::Text {
                    text: "42".to_string(),
                },
            ],
        );
        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(llm_suite_types::message::MessageRole::User, "hi"),
                        with_signature,
                    ],
                    None,
                ),
                &profile(),
            )
            .expect("build must succeed");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["thinking"], serde_json::json!("Let me verify"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));

        let without_signature = rich_msg(
            llm_suite_types::message::MessageRole::Assistant,
            vec![
                llm_suite_types::message::MessageContent::Thinking {
                    thinking: "unsigned".to_string(),
                    signature: None,
                },
                llm_suite_types::message::MessageContent::Text {
                    text: "42".to_string(),
                },
            ],
        );
        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(llm_suite_types::message::MessageRole::User, "hi"),
                        without_signature,
                    ],
                    None,
                ),
                &profile(),
            )
            .expect("build must succeed");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(
            content.len(),
            1,
            "unsigned thinking must be dropped, not sent"
        );
        assert_eq!(content[0]["type"], serde_json::json!("text"));
    }

    #[test]
    fn https_image_uses_url_source() {
        let codec = AnthropicCodec::new();
        let user = rich_msg(
            llm_suite_types::message::MessageRole::User,
            vec![llm_suite_types::message::MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/cat.png".to_string(),
                    detail: None,
                },
            }],
        );
        let body = codec
            .build_body(&request(vec![user], None), &profile())
            .expect("build must succeed");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], serde_json::json!("image"));
        assert_eq!(block["source"]["type"], serde_json::json!("url"));
        assert_eq!(
            block["source"]["url"],
            serde_json::json!("https://example.com/cat.png")
        );
    }

    #[test]
    fn data_uri_image_uses_base64_source_with_validated_media_type() {
        let codec = AnthropicCodec::new();
        let user = rich_msg(
            llm_suite_types::message::MessageRole::User,
            vec![llm_suite_types::message::MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/webp;base64,QUJD".to_string(),
                    detail: None,
                },
            }],
        );
        let body = codec
            .build_body(&request(vec![user], None), &profile())
            .expect("build must succeed");
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["source"]["type"], serde_json::json!("base64"));
        assert_eq!(
            block["source"]["media_type"],
            serde_json::json!("image/webp")
        );
        assert_eq!(block["source"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn unsupported_image_media_type_is_rejected() {
        let codec = AnthropicCodec::new();
        let user = rich_msg(
            llm_suite_types::message::MessageRole::User,
            vec![llm_suite_types::message::MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/bmp;base64,QUJD".to_string(),
                    detail: None,
                },
            }],
        );
        let err = codec
            .build_body(&request(vec![user], None), &profile())
            .expect_err("bmp must be rejected");
        assert!(matches!(err, LlmError::ConfigError(_)));
    }

    #[test]
    fn tool_without_parameters_gets_empty_object_schema() {
        let codec = AnthropicCodec::new();
        let mut req = request(
            vec![msg(llm_suite_types::message::MessageRole::User, "hi")],
            None,
        );
        req.tools = Some(vec![llm_suite_types::tool::Tool {
            name: "get_time".to_string(),
            description: "Returns the current time".to_string(),
            parameters: None,
        }]);
        let body = codec.build_body(&req, &profile()).expect("must build");
        let schema = &body["tools"][0]["input_schema"];
        assert_eq!(schema["type"], serde_json::json!("object"));
        assert!(
            schema.get("properties").is_some(),
            "input_schema must not be null"
        );
    }

    #[test]
    fn empty_text_blocks_and_empty_messages_are_skipped() {
        let codec = AnthropicCodec::new();
        let user = rich_msg(
            llm_suite_types::message::MessageRole::User,
            vec![
                llm_suite_types::message::MessageContent::Text {
                    text: String::new(),
                },
                llm_suite_types::message::MessageContent::Text {
                    text: "hi".to_string(),
                },
            ],
        );
        let body = codec
            .build_body(&request(vec![user], None), &profile())
            .expect("build must succeed");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], serde_json::json!("hi"));

        // A message that only carries empty text produces no message at all.
        let ghost = rich_msg(
            llm_suite_types::message::MessageRole::User,
            vec![llm_suite_types::message::MessageContent::Text {
                text: "  ".to_string(),
            }],
        );
        let body = codec
            .build_body(&request(vec![ghost], None), &profile())
            .expect("build must succeed");
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    // ---------------- response parsing ----------------

    #[test]
    fn parse_response_extracts_thinking_and_native_reasoning_tokens() {
        let codec = AnthropicCodec::new();
        let body = r#"{
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "Let me verify", "signature": "sig"},
                {"type": "text", "text": "42"}
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "output_tokens_details": {"thinking_tokens": 8},
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2
            }
        }"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(out.content.as_deref(), Some("42"));
        assert_eq!(out.reasoning_content.as_deref(), Some("Let me verify"));
        assert_eq!(out.message.thinking.as_deref(), Some("Let me verify"));
        assert_eq!(out.reasoning_tokens, Some(8));
        let usage = out.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(
            usage.total_tokens, 37,
            "the official total counts the cached input tokens"
        );
        assert_eq!(usage.cache_read_tokens, Some(5));
        assert_eq!(usage.cache_write_tokens, Some(2));
    }

    #[test]
    fn the_matched_stop_sequence_is_reported_as_metadata() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"m","model":"claude","stop_reason":"stop_sequence","stop_sequence":"END","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(
            out.metadata.expect("metadata must be present")["stop_sequence"],
            serde_json::json!("END")
        );
    }

    #[test]
    fn error_payload_is_surfaced_as_invalid_response() {
        let codec = AnthropicCodec::new();
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let err = codec
            .parse_response(body, &request(vec![], None))
            .expect_err("error payload must not parse into a message");
        match err {
            LlmError::InvalidResponse(msg) => {
                assert!(msg.contains("overloaded_error"));
                assert!(msg.contains("Overloaded"));
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }

    // ---------------- streaming ----------------

    #[test]
    fn message_start_emits_identity_and_initial_usage() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet","usage":{"input_tokens":25,"output_tokens":1}}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        assert_eq!(events.len(), 2);
        match &events[0] {
            MessageStreamEvent::Start(s) => {
                assert_eq!(s.id.as_deref(), Some("msg_1"));
                assert_eq!(s.model.as_deref(), Some("claude-sonnet"));
            }
            other => panic!("expected Start first, got {other:?}"),
        }
        match &events[1] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.prompt_tokens, 25);
                assert_eq!(u.usage.completion_tokens, 1);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn signature_delta_emits_reasoning_signature_fragment() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-part"}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        match &events[0] {
            MessageStreamEvent::ReasoningText(r) => {
                assert!(r.reasoning.is_empty());
                assert_eq!(r.signature.as_deref(), Some("sig-part"));
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }

    #[test]
    fn response_signature_is_retained_and_replayed() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"msg_2","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"Let me verify","signature":"sig-abc"},{"type":"text","text":"done"}],"usage":{"input_tokens":3,"output_tokens":5}}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = codec.parse_response(body, &request).expect("must parse");
        assert_eq!(out.message.thinking.as_deref(), Some("Let me verify"));
        assert_eq!(out.message.thinking_signature.as_deref(), Some("sig-abc"));
        assert_eq!(out.reasoning_signature.as_deref(), Some("sig-abc"));

        let wire = codec
            .convert_messages(std::slice::from_ref(&out.message))
            .expect("must convert");
        let thinking = wire[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "thinking")
            .expect("signed thinking must be replayed");
        assert_eq!(thinking["signature"], "sig-abc");
        assert_eq!(thinking["thinking"], "Let me verify");

        // A thinking block without a signature is never replayed.
        let mut unsigned = out.message.clone();
        unsigned.content = llm_suite_types::message::MessageContentValue::Rich(vec![
            llm_suite_types::message::MessageContent::Thinking {
                thinking: "Let me verify".to_string(),
                signature: None,
            },
        ]);
        let wire = codec
            .convert_messages(std::slice::from_ref(&unsigned))
            .expect("must convert");
        let replayed_thinking = wire.iter().any(|message| {
            message["content"]
                .as_array()
                .map(|blocks| blocks.iter().any(|block| block["type"] == "thinking"))
                .unwrap_or(false)
        });
        assert!(!replayed_thinking, "unsigned thinking must not be replayed");
    }

    #[test]
    fn every_thinking_block_keeps_its_own_signature() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"msg_3","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"first","signature":"sig-1"},{"type":"thinking","thinking":"second","signature":"sig-2"},{"type":"text","text":"done"}]}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = codec.parse_response(body, &request).expect("must parse");

        let wire = codec
            .convert_messages(std::slice::from_ref(&out.message))
            .expect("must convert");
        let blocks = wire[0]["content"].as_array().unwrap();
        let signatures: Vec<&str> = blocks
            .iter()
            .filter(|b| b["type"] == "thinking")
            .map(|b| b["signature"].as_str().unwrap())
            .collect();
        assert_eq!(signatures, vec!["sig-1", "sig-2"]);
    }

    #[test]
    fn redacted_thinking_is_kept_and_replayed() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"msg_4","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"c2VjcmV0"}]}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = codec.parse_response(body, &request).expect("must parse");

        let wire = codec
            .convert_messages(std::slice::from_ref(&out.message))
            .expect("must convert");
        let blocks = wire[0]["content"].as_array().unwrap();
        let redacted = blocks
            .iter()
            .find(|b| b["type"] == "redacted_thinking")
            .expect("redacted thinking must be replayed");
        assert_eq!(redacted["data"], "c2VjcmV0");
    }

    #[test]
    fn message_delta_emits_usage_then_end() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":25,"output_tokens":7}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        assert_eq!(events.len(), 2, "usage and end must both be emitted");
        match &events[0] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.prompt_tokens, 25);
                assert_eq!(u.usage.completion_tokens, 7);
            }
            other => panic!("expected Usage first, got {other:?}"),
        }
        assert!(matches!(events[1], MessageStreamEvent::End(_)));
    }

    #[test]
    fn pause_turn_keeps_stream_open_until_message_stop() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":5}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events[0], MessageStreamEvent::Usage(_)),
            "no End on pause_turn"
        );

        let stop = codec
            .parse_stream_chunk_events(r#"{"type":"message_stop"}"#)
            .expect("must parse");
        assert!(matches!(stop[0], MessageStreamEvent::End(_)));
    }

    #[test]
    fn refusal_and_context_window_stop_reasons_are_terminal() {
        let codec = AnthropicCodec::new();
        for reason in ["refusal", "model_context_window_exceeded", "tool_use"] {
            let chunk =
                format!(r#"{{"type":"message_delta","delta":{{"stop_reason":"{reason}"}}}}"#);
            let events = codec.parse_stream_chunk_events(&chunk).expect("must parse");
            assert!(
                matches!(events.last(), Some(MessageStreamEvent::End(_))),
                "{reason} must terminate the stream"
            );
        }
    }

    // ---------------- usage / prompt caching ----------------

    #[test]
    fn cache_creation_ttl_breakdown_is_summed() {
        // Official `Usage.cache_creation` object without the legacy total.
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"m","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":4,"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":3}}}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        let usage = out.usage.expect("usage present");
        assert_eq!(usage.cache_read_tokens, Some(4));
        assert_eq!(
            usage.cache_write_tokens,
            Some(5),
            "5m + 1h breakdown summed"
        );

        // The explicit legacy total wins when both shapes are present.
        let body = r#"{"id":"m","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":7,"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":3}}}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(
            out.usage.unwrap().cache_write_tokens,
            Some(7),
            "explicit cache_creation_input_tokens takes precedence"
        );

        // Neither shape present → `None`.
        let body = r#"{"id":"m","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":5}}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(out.usage.unwrap().cache_write_tokens, None);
    }

    #[test]
    fn stream_message_delta_summarizes_ttl_breakdown() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"cache_creation":{"ephemeral_5m_input_tokens":11}}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        match &events[0] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.completion_tokens, 7);
                assert_eq!(u.usage.cache_write_tokens, Some(11));
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn prompt_caching_breakpoints_are_applied() {
        let codec = AnthropicCodec::new();
        let mut req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hello"),
                msg(llm_suite_types::message::MessageRole::Assistant, "Hi there"),
                msg(llm_suite_types::message::MessageRole::User, "Bye"),
            ],
            None,
        );
        req.tools = Some(vec![llm_suite_types::tool::Tool {
            name: "get_weather".to_string(),
            description: "weather lookup".to_string(),
            parameters: None,
        }]);

        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching { ttl: None });
        let body = codec
            .build_body(&req, &caching_profile)
            .expect("must build");

        // Default TTL shape is the bare ephemeral marker (provider default 5m).
        let control = serde_json::json!({"type": "ephemeral"});

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["cache_control"], control);

        // System is upgraded to the block-array form carrying the marker.
        assert_eq!(
            body["system"],
            serde_json::json!([{
                "type": "text",
                "text": "You are a helper",
                "cache_control": control,
            }])
        );

        // The first and the last message are marked: the stable prefix and
        // the current turn. The API allows four breakpoints.
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        let last_blocks = messages.last().unwrap()["content"].as_array().unwrap();
        assert_eq!(last_blocks.last().unwrap()["cache_control"], control);
        let first_blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(first_blocks.last().unwrap()["cache_control"], control);
        assert_eq!(messages[1]["content"][0].get("cache_control"), None);
    }

    #[test]
    fn prompt_caching_ttl_is_forwarded() {
        let codec = AnthropicCodec::new();
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(
                            llm_suite_types::message::MessageRole::System,
                            "You are a helper",
                        ),
                        msg(llm_suite_types::message::MessageRole::User, "Hi"),
                    ],
                    None,
                ),
                &caching_profile,
            )
            .expect("must build");
        let control = serde_json::json!({"type": "ephemeral", "ttl": "1h"});
        assert_eq!(
            body["messages"].as_array().unwrap().last().unwrap()["content"][0]["cache_control"],
            control
        );
        assert_eq!(body["system"][0]["cache_control"], control);
    }

    #[test]
    fn without_prompt_caching_no_cache_control_is_emitted() {
        let codec = AnthropicCodec::new();
        let body = codec
            .build_body(
                &request(
                    vec![msg(llm_suite_types::message::MessageRole::User, "Hi")],
                    None,
                ),
                &profile(),
            )
            .expect("must build");
        assert_eq!(body["system"], serde_json::json!(null));
        assert!(body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
    }

    #[test]
    fn request_override_enables_caching_without_profile_setting() {
        let codec = AnthropicCodec::new();
        let mut req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hi"),
            ],
            None,
        );
        req.prompt_caching = Some(LlmRequestCaching {
            enabled: true,
            ttl: Some("1h".to_string()),
        });
        let body = codec.build_body(&req, &profile()).expect("must build");
        let control = serde_json::json!({"type": "ephemeral", "ttl": "1h"});
        assert_eq!(body["system"][0]["cache_control"], control);
        assert_eq!(
            body["messages"].as_array().unwrap().last().unwrap()["content"][0]["cache_control"],
            control
        );
    }

    #[test]
    fn request_override_disables_profile_caching() {
        let codec = AnthropicCodec::new();
        let mut req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hi"),
            ],
            None,
        );
        req.prompt_caching = Some(LlmRequestCaching {
            enabled: false,
            ttl: None,
        });
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching { ttl: None });
        let body = codec
            .build_body(&req, &caching_profile)
            .expect("must build");
        // System stays in the plain-string form (no block-array upgrade).
        assert_eq!(body["system"], serde_json::json!("You are a helper"));
        assert!(body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
    }

    #[test]
    fn request_override_ttl_wins_over_profile() {
        let codec = AnthropicCodec::new();
        let mut req = request(
            vec![
                msg(
                    llm_suite_types::message::MessageRole::System,
                    "You are a helper",
                ),
                msg(llm_suite_types::message::MessageRole::User, "Hi"),
            ],
            None,
        );
        req.prompt_caching = Some(LlmRequestCaching {
            enabled: true,
            ttl: Some("5m".to_string()),
        });
        let mut caching_profile = profile();
        caching_profile.prompt_caching = Some(LlmPromptCaching {
            ttl: Some("1h".to_string()),
        });
        let body = codec
            .build_body(&req, &caching_profile)
            .expect("must build");
        let control = serde_json::json!({"type": "ephemeral", "ttl": "5m"});
        assert_eq!(body["system"][0]["cache_control"], control);
    }

    #[test]
    fn error_event_is_surfaced() {
        let codec = AnthropicCodec::new();
        let chunk =
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        match &events[0] {
            MessageStreamEvent::Error(e) => {
                assert!(e.error.contains("overloaded_error"));
            }
            other => panic!("expected Error event, got {other:?}"),
        }
    }

    #[test]
    fn a_stored_thinking_block_comes_before_the_text() {
        // The API requires the thinking blocks to lead the assistant turn,
        // no matter where the history stored them.
        let codec = AnthropicCodec::new();
        let assistant = llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role: llm_suite_types::message::MessageRole::Assistant,
            content: llm_suite_types::message::MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::Text {
                    text: "the answer".to_string(),
                },
                llm_suite_types::message::MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
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

        let body = codec
            .build_body(
                &request(
                    vec![
                        msg(llm_suite_types::message::MessageRole::User, "hi"),
                        assistant,
                    ],
                    None,
                ),
                &profile(),
            )
            .expect("must build");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));
        assert_eq!(content[1]["text"], serde_json::json!("the answer"));
    }

    #[test]
    fn a_refusal_is_reported_on_the_refusal_channel() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"m","model":"claude","stop_reason":"refusal","content":[{"type":"text","text":"I cannot help with that."}],"stop_details":{"type":"refusal","category":"general_harms","explanation":"policy"}}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(out.refusal.as_deref(), Some("policy"));
        let metadata = out.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["stop_details"]["category"],
            serde_json::json!("general_harms")
        );
    }

    #[test]
    fn a_refusal_without_an_explanation_names_the_category() {
        let codec = AnthropicCodec::new();
        let body = r#"{"id":"m","model":"claude","stop_reason":"refusal","content":[],"stop_details":{"type":"refusal","category":"cyber"}}"#;
        let out = codec
            .parse_response(body, &request(vec![], None))
            .expect("must parse");
        assert_eq!(out.refusal.as_deref(), Some("refused: cyber"));
    }

    #[test]
    fn the_stream_reports_the_sources_of_a_grounded_answer() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"x","document_index":0}}}"#;
        match codec
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::Annotation(annotation)) => {
                assert_eq!(
                    annotation.annotation["type"],
                    serde_json::json!("char_location")
                );
            }
            other => panic!("expected Annotation, got {other:?}"),
        }
    }

    #[test]
    fn the_stream_reports_a_refusal() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","explanation":"policy"}},"usage":{"output_tokens":3}}"#;
        let events = codec.parse_stream_chunk_events(chunk).expect("must parse");
        let refusal = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::Refusal(r) => Some(r.refusal.clone()),
                _ => None,
            })
            .expect("the refusal must be reported");
        assert_eq!(refusal, "policy");
    }

    #[test]
    fn the_thinking_block_start_text_is_not_dropped() {
        let codec = AnthropicCodec::new();
        let chunk = r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"Let me verify"}}"#;
        match codec
            .parse_stream_chunk_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ReasoningText(reasoning)) => {
                assert_eq!(reasoning.reasoning, "Let me verify");
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }
}
