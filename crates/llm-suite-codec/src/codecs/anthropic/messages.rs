//! Conversion of internal messages into Anthropic `messages`.

use crate::codecs::helpers::{block_name, native_block, resolve_image_payload, ResolvedImage};
use crate::error::{LlmError, LlmResult};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// Media types accepted by the Anthropic Messages API for image blocks.
const ANTHROPIC_IMAGE_MEDIA_TYPES: [&str; 4] =
    ["image/jpeg", "image/png", "image/gif", "image/webp"];

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
pub fn convert_messages(messages: &[Message]) -> LlmResult<Vec<serde_json::Value>> {
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
                            if let Some(file_id) = &image_url.file_id {
                                // Files-API reference: Anthropic carries it
                                // as a `file` image source.
                                blocks.push(serde_json::json!({
                                    "type": "image",
                                    "source": {"type": "file", "file_id": file_id},
                                }));
                            } else {
                                blocks.push(convert_image_block(&image_url.url)?);
                            }
                        }
                        // The Messages API has no audio input part.
                        MessageContent::InputAudio { input_audio } => {
                            return Err(LlmError::ConfigError(format!(
                                "anthropic messages do not accept audio input (format `{}`)",
                                input_audio.format
                            )));
                        }
                        MessageContent::FileInput { file } => {
                            blocks.push(convert_document_block(file)?);
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
                            tool_result_blocks.push(tool_result_block(tool_result)?);
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
                        MessageContent::Native {
                            provider,
                            kind,
                            payload,
                        } => {
                            // The blocks only Anthropic has (search
                            // results, browsing state, server tool
                            // results) travel in their own shape and are
                            // replayed verbatim.
                            blocks.push(native_block(provider, "anthropic", kind, payload)?);
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
                let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
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
            tool_result_blocks.push(tool_result_block(
                &llm_suite_types::message::ToolResultContent {
                    tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                    content: content_text,
                    blocks: None,
                    is_error,
                },
            )?);
        }

        // `tool_result` blocks must come first inside their message.
        let mut message_blocks = tool_result_blocks;
        message_blocks.extend(blocks);

        // The thinking blocks must lead the assistant turn: the API
        // rejects a thinking block that follows the text or the tool
        // calls of its turn.
        if role == "assistant" {
            message_blocks = thinking_blocks_first(message_blocks);
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
pub fn thinking_blocks_first(blocks: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
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

/// Build a `tool_result` block.
///
/// The rich blocks (images, documents, …) ride in the `content` array; the
/// plain-text form stays a string. The API accepts text, image, document and
/// the provider's own blocks inside the array; anything else is reported
/// instead of being dropped.
pub fn tool_result_block(
    tool_result: &llm_suite_types::message::ToolResultContent,
) -> LlmResult<serde_json::Value> {
    let content = match &tool_result.blocks {
        Some(blocks) if !blocks.is_empty() => {
            let mut parts: Vec<serde_json::Value> = Vec::new();
            for block in blocks {
                match block {
                    llm_suite_types::message::MessageContent::Text { text } => {
                        parts.push(serde_json::json!({"type": "text", "text": text}));
                    }
                    llm_suite_types::message::MessageContent::ImageUrl { image_url } => {
                        if let Some(file_id) = &image_url.file_id {
                            parts.push(serde_json::json!({
                                "type": "image",
                                "source": {"type": "file", "file_id": file_id},
                            }));
                        } else {
                            parts.push(convert_image_block(&image_url.url)?);
                        }
                    }
                    llm_suite_types::message::MessageContent::FileInput { file } => {
                        parts.push(convert_document_block(file)?);
                    }
                    llm_suite_types::message::MessageContent::Native {
                        provider,
                        kind,
                        payload,
                    } => {
                        parts.push(native_block(provider, "anthropic", kind, payload)?);
                    }
                    other => {
                        return Err(LlmError::ConfigError(format!(
                            "an Anthropic tool result carries text, image and document blocks, \
                             not a `{}` block",
                            block_name(other)
                        )));
                    }
                }
            }
            // The plain-text content of the result leads the array unless
            // the rich blocks carry text of their own.
            if let Some(text) = tool_result.leading_text() {
                parts.insert(0, serde_json::json!({"type": "text", "text": text}));
            }
            serde_json::json!(parts)
        }
        _ => serde_json::json!(tool_result.fallback_text()),
    };
    let mut block = serde_json::json!({
        "type": "tool_result",
        "tool_use_id": tool_result.tool_use_id,
        "content": content,
    });
    // `is_error` is optional; only set it when true to keep the wire
    // format minimal.
    if tool_result.is_error == Some(true) {
        block["is_error"] = serde_json::json!(true);
    }
    Ok(block)
}

/// Media types accepted by the Anthropic Messages API for document blocks.
const ANTHROPIC_DOCUMENT_MEDIA_TYPES: [&str; 2] = ["application/pdf", "text/plain"];

/// Map a file/document input to an Anthropic `document` block.
///
/// Base64 PDFs become `base64` sources, plain text becomes a `text`
/// source, remote URLs a `url` source and Files-API uploads a `file`
/// source; a document built from blocks becomes a `content` source.
/// Anything else is rejected up front.
pub fn convert_document_block(
    file: &llm_suite_types::message::FileContent,
) -> LlmResult<serde_json::Value> {
    use llm_suite_types::message::FileSource;

    let source = match &file.source {
        FileSource::Base64 { media_type, data } => {
            if !ANTHROPIC_DOCUMENT_MEDIA_TYPES.contains(&media_type.as_str()) {
                return Err(LlmError::ConfigError(format!(
                    "Anthropic document media type `{media_type}` is not supported (expected application/pdf or text/plain)"
                )));
            }
            if media_type == "text/plain" {
                serde_json::json!({
                    "type": "text",
                    "media_type": "text/plain",
                    "data": data,
                })
            } else {
                serde_json::json!({
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                })
            }
        }
        FileSource::Url { url } => serde_json::json!({
            "type": "url",
            "url": url,
        }),
        FileSource::FileId { file_id } => serde_json::json!({
            "type": "file",
            "file_id": file_id,
        }),
        FileSource::Content { blocks } => {
            // The `content` source takes a list of text and image blocks.
            let mut content: Vec<serde_json::Value> = Vec::new();
            for block in blocks {
                match block {
                    MessageContent::Text { text } => {
                        content.push(serde_json::json!({"type": "text", "text": text}));
                    }
                    MessageContent::ImageUrl { image_url } => {
                        if let Some(file_id) = &image_url.file_id {
                            content.push(serde_json::json!({
                                "type": "image",
                                "source": {"type": "file", "file_id": file_id},
                            }));
                        } else {
                            content.push(convert_image_block(&image_url.url)?);
                        }
                    }
                    other => {
                        return Err(LlmError::ConfigError(format!(
                            "an Anthropic document source takes text and image blocks, not a \
                             `{}` block",
                            block_name(other)
                        )));
                    }
                }
            }
            serde_json::json!({
                "type": "content",
                "content": content,
            })
        }
    };
    let mut block = serde_json::json!({
        "type": "document",
        "source": source,
    });
    if let Some(title) = &file.filename {
        block["title"] = serde_json::json!(title);
    }
    if let Some(context) = &file.context {
        block["context"] = serde_json::json!(context);
    }
    if let Some(enabled) = file.citations_enabled {
        block["citations"] = serde_json::json!({"enabled": enabled});
    }
    Ok(block)
}

/// Map an image reference to an Anthropic image block. Data URIs become
/// `base64` sources, http(s) URLs use the native `url` source, anything
/// else is rejected up front.
pub fn convert_image_block(url: &str) -> LlmResult<serde_json::Value> {
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


#[cfg(test)]
#[path = "messages/tests.rs"]
mod tests;
