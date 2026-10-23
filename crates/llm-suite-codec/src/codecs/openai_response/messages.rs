//! Conversion of internal messages into OpenAI Responses `input` items.

use std::collections::HashMap;

use crate::error::LlmResult;
use llm_suite_types::message::{
    Message, MessageContent, MessageContentValue, MessageRole, ToolResultContent,
};

/// Build a `function_call_output` value from a tool result: plain text stays
/// the string shorthand, rich blocks become the content-item list the
/// Responses API accepts (`input_text` / `input_image` / `input_file`).
fn tool_output_value(content: &MessageContentValue) -> crate::error::LlmResult<serde_json::Value> {
    match content {
        MessageContentValue::Text(text) => Ok(serde_json::Value::String(text.clone())),
        MessageContentValue::Rich(blocks) => {
            // A tool message wraps its payload in a `ToolResult` block; the
            // rich blocks to publish live inside it.
            if let Some(MessageContent::ToolResult { tool_result }) = blocks.first() {
                return tool_output_from_result(tool_result);
            }
            let parts = content_parts(blocks)?;
            Ok(output_value(parts))
        }
    }
}

/// The content parts of a list of blocks.
fn content_parts(blocks: &[MessageContent]) -> crate::error::LlmResult<Vec<serde_json::Value>> {
    blocks.iter().map(content_part).collect()
}

/// The output value of a content-item list.
///
/// The API accepts a plain string as well; the shorthand is used when the
/// list carries nothing but text, so a media part never gets flattened
/// away.
fn output_value(parts: Vec<serde_json::Value>) -> serde_json::Value {
    let text_only = parts
        .iter()
        .all(|part| part["type"] == serde_json::json!("input_text"));
    if text_only {
        let joined = parts
            .iter()
            .map(|part| part["text"].as_str().unwrap_or("").to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return serde_json::Value::String(joined);
    }
    serde_json::Value::Array(parts)
}

/// Build the `output` value of a `function_call_output` item from a tool
/// result block.
fn tool_output_from_result(
    tool_result: &ToolResultContent,
) -> crate::error::LlmResult<serde_json::Value> {
    let Some(blocks) = tool_result.blocks.as_deref().filter(|b| !b.is_empty()) else {
        return Ok(serde_json::Value::String(tool_result.fallback_text()));
    };
    let mut items: Vec<serde_json::Value> = Vec::new();
    // The plain-text content of the result leads the list; when the rich
    // blocks carry text of their own it would repeat them.
    if let Some(text) = tool_result.leading_text() {
        items.push(serde_json::json!({"type": "input_text", "text": text}));
    }
    for block in blocks {
        items.push(content_part(block)?);
    }
    Ok(serde_json::Value::Array(items))
}

/// Map a content block onto an input content part.
///
/// The API accepts text, image and file parts; a block it has no part for
/// is reported instead of being dropped.
fn content_part(block: &MessageContent) -> crate::error::LlmResult<serde_json::Value> {
    match block {
        MessageContent::Text { text } => Ok(serde_json::json!({
            "type": "input_text",
            "text": text,
        })),
        MessageContent::ImageUrl { image_url } => {
            let mut part = serde_json::json!({"type": "input_image"});
            if let Some(file_id) = &image_url.file_id {
                // Files-API reference: `file_id`, not `image_url`.
                part["file_id"] = serde_json::json!(file_id);
            } else {
                part["image_url"] = serde_json::json!(image_url.url);
            }
            // The official image part carries a detail level; `auto` lets
            // the provider pick.
            part["detail"] = serde_json::json!(image_url
                .detail
                .unwrap_or(llm_suite_types::message::ImageDetail::Auto));
            Ok(part)
        }
        MessageContent::FileInput { file } => {
            let mut part = serde_json::json!({"type": "input_file"});
            match &file.source {
                llm_suite_types::message::FileSource::Base64 { data, .. } => {
                    part["file_data"] = serde_json::json!(data);
                }
                llm_suite_types::message::FileSource::Url { url } => {
                    part["file_url"] = serde_json::json!(url);
                }
                llm_suite_types::message::FileSource::FileId { file_id } => {
                    part["file_id"] = serde_json::json!(file_id);
                }
                llm_suite_types::message::FileSource::Content { .. } => {
                    return Err(crate::error::LlmError::ConfigError(
                        "openai responses has no document built from content blocks; send the \
                         document as file data, as a file url or as a file id"
                            .to_string(),
                    ));
                }
            }
            if let Some(filename) = &file.filename {
                part["filename"] = serde_json::json!(filename);
            }
            if let Some(detail) = file.detail {
                part["detail"] = serde_json::json!(detail);
            }
            Ok(part)
        }
        other => Err(crate::error::LlmError::ConfigError(format!(
            "an OpenAI Responses content item carries text, image and file parts, not a `{}` \
             block",
            crate::codecs::helpers::block_name(other)
        ))),
    }
}

/// The call id a tool result is correlated with.
///
/// The result references the call by the id the call was keyed by; the
/// history may hold either identity, so both are mapped onto the id the
/// call item uses.
fn resolve_call_id(call_ids: &HashMap<&str, String>, id: &str) -> String {
    call_ids.get(id).cloned().unwrap_or_else(|| id.to_string())
}

pub fn convert_messages(messages: &[Message]) -> LlmResult<Vec<serde_json::Value>> {
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
        if let MessageContentValue::Rich(blocks) = &msg.content {
            for block in blocks {
                if let MessageContent::ToolUse { tool_use } = block {
                    call_ids.insert(tool_use.id.as_str(), tool_use.id.clone());
                }
            }
        }
    }

    for msg in messages {
        let role = match msg.role {
            // The system prompt travels in the dedicated `instructions`
            // field, not as an input item.
            MessageRole::System => continue,
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };

        // The tool calls of the turn: the canonical flat list and the rich
        // blocks that carry the same thing.
        let mut calls: Vec<(String, String, String)> = Vec::new();
        for tc in msg.tool_calls.iter().flatten() {
            let call_id = tc.call_id.clone().unwrap_or_else(|| tc.id.clone());
            calls.push((
                call_id,
                tc.function.name.clone(),
                tc.function.arguments.clone(),
            ));
        }
        // The tool results of the turn, as (call id, output) pairs.
        let mut outputs: Vec<(String, serde_json::Value)> = Vec::new();

        // Map the message body to a Responses `input` content value
        // (string shorthand or `input_text` / `input_image` parts).
        let mut parts: Vec<serde_json::Value> = Vec::new();
        let mut plain_text: Option<String> = None;
        match &msg.content {
            MessageContentValue::Text(text) => plain_text = Some(text.clone()),
            // The Responses API uses `input_text` / `input_image`
            // content part types (not Chat's `text` / `image_url`),
            // so Rich blocks are mapped explicitly.
            MessageContentValue::Rich(blocks) => {
                for block in blocks {
                    match block {
                        MessageContent::Text { text } => {
                            parts.push(serde_json::json!({"type": "input_text", "text": text}))
                        }
                        MessageContent::ImageUrl { .. } => parts.push(content_part(block)?),
                        MessageContent::Thinking { thinking, .. } => {
                            parts.push(serde_json::json!({"type": "input_text", "text": thinking}))
                        }
                        MessageContent::RedactedThinking { data } => {
                            // The Responses API has no redacted-thinking
                            // part; the opaque payload is kept as text
                            // so the block is not dropped from history.
                            parts.push(serde_json::json!({"type": "input_text", "text": data}))
                        }
                        MessageContent::InputAudio { input_audio } => {
                            // `input_audio` is not part of the Responses
                            // message-content union; refuse instead of
                            // corrupting the item.
                            return Err(crate::error::LlmError::ConfigError(format!(
                                "openai responses input items do not accept audio input (format `{}`)",
                                input_audio.format
                            )));
                        }
                        MessageContent::FileInput { .. } => parts.push(content_part(block)?),
                        // A tool call is a `function_call` item, never a
                        // content part: the call has to stay structured so
                        // its result can be correlated with it.
                        MessageContent::ToolUse { tool_use } => {
                            let arguments = serde_json::to_string(&tool_use.input)
                                .unwrap_or_else(|_| "{}".into());
                            calls.push((tool_use.id.clone(), tool_use.name.clone(), arguments));
                        }
                        // A tool result is a `function_call_output` item.
                        MessageContent::ToolResult { tool_result } => {
                            let call_id = tool_result.tool_use_id.clone();
                            outputs.push((call_id, tool_output_from_result(tool_result)?));
                        }
                        MessageContent::Native {
                            provider,
                            kind,
                            payload,
                        } => parts.push(crate::codecs::helpers::native_block(
                            provider, "openai", kind, payload,
                        )?),
                    }
                }
            }
        }

        // The content of the turn: the plain text stays the string
        // shorthand, the rich blocks become the content-item list.
        let content = match plain_text {
            Some(text) => serde_json::Value::String(text),
            None => serde_json::Value::Array(parts),
        };

        match msg.role {
            // Tool results are `function_call_output` items in the
            // Responses API, correlated to the originating function call by
            // `call_id` (the message's `tool_call_id`). Chat's
            // `tool_calls`/`tool_call_id` shape is rejected by the API.
            MessageRole::Tool => {
                if outputs.is_empty() {
                    // The plain-text result of the turn.
                    let call_id = msg.tool_call_id.clone().unwrap_or_default();
                    outputs.push((call_id, tool_output_value(&msg.content)?));
                } else if let Some(ref id) = msg.tool_call_id {
                    // The message knows the call it answers; the result
                    // block may not carry the id.
                    for (block_id, _) in outputs.iter_mut() {
                        if block_id.is_empty() {
                            *block_id = id.clone();
                        }
                    }
                }
                for (call_id, output) in outputs {
                    if call_id.is_empty() {
                        // No correlation id available: fall back to a plain
                        // input item so the tool result is not dropped.
                        items.push(serde_json::json!({
                            "role": "user",
                            "content": output,
                        }));
                        continue;
                    }
                    // The output item has to carry the same id the call
                    // item was keyed by, whichever identity the history
                    // recorded the result under.
                    items.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": resolve_call_id(&call_ids, &call_id),
                        "output": output,
                    }));
                }
            }
            // Assistant turns with tool calls emit one `function_call`
            // item per call (keyed by `call_id`), plus a text message item
            // when the assistant also produced prose.
            MessageRole::Assistant => {
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
                for (call_id, name, arguments) in calls {
                    items.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }));
                }
                for (call_id, output) in outputs {
                    items.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": resolve_call_id(&call_ids, &call_id),
                        "output": output,
                    }));
                }
            }
            _ => {
                let has_text = match &content {
                    serde_json::Value::String(s) => !s.is_empty(),
                    serde_json::Value::Array(a) => !a.is_empty(),
                    _ => false,
                };
                if has_text {
                    items.push(serde_json::json!({
                        "role": role,
                        "content": content,
                    }));
                }
                for (call_id, output) in outputs {
                    items.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": resolve_call_id(&call_ids, &call_id),
                        "output": output,
                    }));
                }
            }
        }
    }

    Ok(items)
}

#[cfg(test)]
#[path = "messages/tests.rs"]
mod tests;
