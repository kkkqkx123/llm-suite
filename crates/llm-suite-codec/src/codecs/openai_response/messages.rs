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
                        MessageContent::ImageUrl { .. } => {
                            parts.push(content_part(block)?)
                        }
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
                        MessageContent::FileInput { .. } => {
                            parts.push(content_part(block)?)
                        }
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
mod tests {
    use super::*;

    #[test]
    fn function_call_and_output_share_the_provider_call_id() {
        let mut assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text(String::new()),
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
                thought_signature: None,
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        // The result references the call by its item id.
        let mut result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Text("sunny".to_string()),
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

        let items = convert_messages(&[assistant, result]).expect("must convert");
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
    fn rich_image_blocks_map_to_input_image_parts() {
        let req_messages = vec![Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![
                MessageContent::Text {
                    text: "describe".to_string(),
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://example.com/a.png".to_string(),
                        detail: Some(llm_suite_types::message::ImageDetail::Auto),
                        file_id: None,
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
        }];
        let entries = convert_messages(&req_messages).expect("must convert");
        let parts = entries[0]["content"].as_array().expect("content array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["image_url"], "https://example.com/a.png");
        assert_eq!(parts[1]["detail"], "auto");
    }

    #[test]
    fn an_image_without_a_detail_carries_the_provider_default() {
        let req_messages = vec![Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
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
        let entries = convert_messages(&req_messages).expect("must convert");
        assert_eq!(
            entries[0]["content"][0]["detail"],
            serde_json::json!("auto"),
            "the official image part carries a detail level"
        );
    }

    #[test]
    fn convert_messages_emits_function_call_and_output_items() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text("let me check".to_string()),
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
                thought_signature: None,
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let tool_result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Text("sunny".to_string()),
            timestamp: 0,
            tool_call_id: Some("call_abc".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let items = convert_messages(&[assistant, tool_result]).expect("must convert");

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
    fn a_rich_tool_use_block_becomes_a_function_call_item() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Beijing"}),
                    thought_signature: None,
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
        let items = convert_messages(&[assistant]).expect("must convert");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], serde_json::json!("function_call"));
        assert_eq!(items[0]["call_id"], serde_json::json!("call_1"));
        assert_eq!(items[0]["name"], serde_json::json!("get_weather"));
        assert_eq!(
            items[0]["arguments"],
            serde_json::json!("{\"city\":\"Beijing\"}")
        );
    }

    #[test]
    fn a_rich_tool_result_block_becomes_a_function_call_output_item() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
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
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    blocks: None,
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
        let items = convert_messages(&[assistant, result]).expect("must convert");
        let call = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call"))
            .expect("the call must be present");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("the output must be present");
        assert_eq!(call["call_id"], serde_json::json!("call_1"));
        assert_eq!(
            output["call_id"], call["call_id"],
            "the result must reference the call it answers"
        );
        assert_eq!(output["output"], serde_json::json!("sunny"));
    }

    #[test]
    fn a_file_input_becomes_an_input_file_part_with_the_right_reference() {
        let msg = |source| Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source,
                    filename: Some("doc.pdf".to_string()),
                    detail: None,
                    citations_enabled: None,
                    context: None,
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
        let items = convert_messages(&[msg(llm_suite_types::message::FileSource::Base64 {
            media_type: "application/pdf".to_string(),
            data: "QUJD".to_string(),
        })])
        .expect("must convert");
        let part = &items[0]["content"][0];
        assert_eq!(part["type"], serde_json::json!("input_file"));
        assert_eq!(part["file_data"], serde_json::json!("QUJD"));
        assert_eq!(part["filename"], serde_json::json!("doc.pdf"));

        let items = convert_messages(&[msg(llm_suite_types::message::FileSource::Url {
            url: "https://example.com/a.pdf".to_string(),
        })])
        .expect("must convert");
        assert_eq!(
            items[0]["content"][0]["file_url"],
            serde_json::json!("https://example.com/a.pdf")
        );

        let items = convert_messages(&[msg(llm_suite_types::message::FileSource::FileId {
            file_id: "file_1".to_string(),
        })])
        .expect("must convert");
        assert_eq!(
            items[0]["content"][0]["file_id"],
            serde_json::json!("file_1")
        );
    }

    #[test]
    fn a_file_id_image_uses_file_id_not_image_url() {
        let msg = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: String::new(),
                    detail: None,
                    file_id: Some("file_9".to_string()),
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
        let items = convert_messages(&[msg]).expect("must convert");
        let part = &items[0]["content"][0];
        assert_eq!(part["type"], serde_json::json!("input_image"));
        assert_eq!(part["file_id"], serde_json::json!("file_9"));
        assert!(part.get("image_url").is_none());
    }

    #[test]
    fn a_rich_tool_result_becomes_a_multimodal_function_call_output() {
        let mut assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text(String::new()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: Some(vec![llm_suite_types::message::LlmToolCall {
                id: "fc_1".to_string(),
                r#type: "function".to_string(),
                call_id: Some("call_1".to_string()),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "screenshot".to_string(),
                    arguments: "{}".to_string(),
                },
                thought_signature: None,
            }]),
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        assistant.timestamp = 0;
        let mut result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "attached".to_string(),
                    blocks: Some(vec![MessageContent::ImageUrl {
                        image_url: llm_suite_types::message::ImageUrlContent {
                            url: "data:image/png;base64,QUJD".to_string(),
                            detail: None,
                            file_id: None,
                        },
                    }]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        result.timestamp = 0;

        let items = convert_messages(&[assistant, result]).expect("must convert");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("output item present");
        let parts = output["output"].as_array().expect("rich output array");
        assert_eq!(parts[0]["type"], serde_json::json!("input_text"));
        assert_eq!(parts[1]["type"], serde_json::json!("input_image"));
        assert_eq!(
            parts[1]["image_url"],
            serde_json::json!("data:image/png;base64,QUJD")
        );
    }

    #[test]
    fn a_tool_result_does_not_repeat_the_text_its_blocks_carry() {
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "the report".to_string(),
                    blocks: Some(vec![MessageContent::Text {
                        text: "the report".to_string(),
                    }]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let items = convert_messages(&[result]).expect("must convert");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("output item present");
        let parts = output["output"].as_array().expect("rich output array");
        assert_eq!(
            parts.len(),
            1,
            "the plain text would repeat the text of the block: {parts:?}"
        );
        assert_eq!(parts[0]["type"], serde_json::json!("input_text"));
    }

    #[test]
    fn a_tool_result_with_media_keeps_the_media_next_to_its_text() {
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "see attached".to_string(),
                    blocks: Some(vec![
                        MessageContent::Text {
                            text: "see attached".to_string(),
                        },
                        MessageContent::ImageUrl {
                            image_url: llm_suite_types::message::ImageUrlContent {
                                url: "data:image/png;base64,QUJD".to_string(),
                                detail: None,
                                file_id: None,
                            },
                        },
                    ]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let items = convert_messages(&[result]).expect("must convert");
        let output = items
            .iter()
            .find(|i| i["type"] == serde_json::json!("function_call_output"))
            .expect("output item present");
        let parts = output["output"].as_array().expect("rich output array");
        assert_eq!(parts[0]["type"], serde_json::json!("input_text"));
        assert_eq!(
            parts[1]["type"],
            serde_json::json!("input_image"),
            "the media must not be flattened away: {parts:?}"
        );
    }

    #[test]
    fn a_tool_result_with_an_unsupported_block_is_reported() {
        let result = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "see attached".to_string(),
                    blocks: Some(vec![MessageContent::Thinking {
                        thinking: "let me verify".to_string(),
                        signature: Some("sig".to_string()),
                    }]),
                    is_error: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: Some("call_1".to_string()),
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };

        let err = convert_messages(&[result])
            .expect_err("a block the format cannot carry must be reported");
        assert!(format!("{err}").contains("thinking"), "{err}");
    }
}
