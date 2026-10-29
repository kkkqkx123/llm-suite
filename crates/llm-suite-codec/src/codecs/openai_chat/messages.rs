//! Internal message conversion into OpenAI Chat Completions `messages`.

use crate::codecs::helpers::native_block;
use crate::error::{LlmError, LlmResult};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// The audio containers the Chat Completions API accepts.
const CHAT_AUDIO_FORMATS: [&str; 2] = ["wav", "mp3"];

/// Convert internal messages into OpenAI Chat Completions `messages`.
///
/// OpenAI Chat has no `tool_use` / `tool_result` content-part types (those are
/// Anthropic's). Tool calls therefore flow through the message-level
/// `tool_calls` field, and tool results must be standalone `tool` role
/// messages. This converter folds `ToolUse` rich blocks into `tool_calls` and
/// emits `ToolResult` rich blocks as separate `tool` messages appended right
/// after their source message (the canonical OpenAI ordering).
pub fn convert_openai_messages(messages: &[Message]) -> LlmResult<Vec<serde_json::Value>> {
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
                            // A Chat image part carries the picture itself:
                            // a remote URL or a data URI. The Files API has
                            // no image form here, so an id cannot be sent as
                            // an image; it belongs to a `file` part.
                            if image_url.file_id.is_some() {
                                return Err(LlmError::ConfigError(
                                    "OpenAI Chat images are referenced by `image_url.url` only; \
                                     a Files-API id cannot be sent as an image part, use a `file` \
                                     part with the file id instead"
                                        .to_string(),
                                ));
                            }
                            let mut img = serde_json::json!({
                                "type": "image_url",
                                "image_url": {"url": image_url.url}
                            });
                            if let Some(detail) = image_url.detail {
                                img["image_url"]["detail"] = serde_json::json!(detail);
                            }
                            content.push(img);
                        }
                        MessageContent::InputAudio { input_audio } => {
                            has_content = true;
                            if !CHAT_AUDIO_FORMATS.contains(&input_audio.format.as_str()) {
                                return Err(LlmError::ConfigError(format!(
                                    "OpenAI Chat input audio accepts `wav` or `mp3`, not `{}`",
                                    input_audio.format
                                )));
                            }
                            content.push(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": input_audio.data,
                                    "format": input_audio.format,
                                }
                            }));
                        }
                        MessageContent::FileInput { file } => {
                            // Chat `file` parts carry a file_id or base64
                            // file_data; a remote URL has no wire form here.
                            // The API has no media type field, so the media
                            // type of the source is not part of the request.
                            has_content = true;
                            let mut entry = match &file.source {
                                llm_suite_types::message::FileSource::Base64 { data, .. } => {
                                    serde_json::json!({"file_data": data})
                                }
                                llm_suite_types::message::FileSource::FileId { file_id } => {
                                    serde_json::json!({"file_id": file_id})
                                }
                                llm_suite_types::message::FileSource::Url { url } => {
                                    return Err(LlmError::ConfigError(format!(
                                        "openai chat file parts accept only file_id or base64 \
                                         file_data, not remote URLs; inline the file instead: {url}"
                                    )));
                                }
                                llm_suite_types::message::FileSource::Content { .. } => {
                                    return Err(LlmError::ConfigError(
                                        "openai chat has no document built from content blocks; \
                                         send the document as file data or as a file id"
                                            .to_string(),
                                    ));
                                }
                            };
                            if let Some(filename) = &file.filename {
                                entry["filename"] = serde_json::json!(filename);
                            }
                            content.push(serde_json::json!({"type": "file", "file": entry}));
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
                        // Chat tool messages carry text-only content; rich
                        // blocks flatten to their text (SDK semantics).
                        MessageContent::ToolResult { tool_result } => {
                            // A tool message carries text only; the media a
                            // tool returned degrades to that text. A block
                            // that has no text form at all is reported
                            // instead of being dropped.
                            if let Some(blocks) = &tool_result.blocks {
                                for block in blocks {
                                    if !matches!(
                                        block,
                                        MessageContent::Text { .. }
                                            | MessageContent::ImageUrl { .. }
                                            | MessageContent::FileInput { .. }
                                    ) {
                                        return Err(LlmError::ConfigError(format!(
                                            "an OpenAI Chat tool message carries text only; its \
                                             result cannot hold a `{}` block",
                                            crate::codecs::helpers::block_name(block)
                                        )));
                                    }
                                }
                            }
                            // The result is correlated with the call it
                            // answers: the id the block carries, or the id
                            // of the message the block sits on.
                            let call_id = if tool_result.tool_use_id.is_empty() {
                                msg.tool_call_id.clone().unwrap_or_default()
                            } else {
                                tool_result.tool_use_id.clone()
                            };
                            if call_id.is_empty() {
                                return Err(LlmError::ConfigError(
                                    "an OpenAI Chat tool message must name the call it answers"
                                        .to_string(),
                                ));
                            }
                            tool_result_entries.push(serde_json::json!({
                                "role": "tool",
                                "content": tool_result.fallback_text(),
                                "tool_call_id": call_id,
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
                        MessageContent::Native {
                            provider,
                            kind,
                            payload,
                        } => {
                            has_content = true;
                            content.push(native_block(provider, "openai", kind, payload)?);
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

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::message::{MessageContentValue, MessageRole};

    #[test]
    fn a_tool_result_in_a_rich_block_needs_no_message_level_tool_call_id() {
        let mut request = llm_suite_types::llm::LlmRequest {
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
            max_retries: None,
            retry_delay_ms: None,
            extra_headers: None,
        };
        request.messages.push(Message {
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
        });

        // The rich block carries the correlation id, so the message itself
        // does not have to.
        super::super::validate::validate_openai_chat_messages(&request)
            .expect("a rich tool result must not be rejected");

        let converted =
            convert_openai_messages(&request.messages).expect("conversion must succeed");
        let tool_message = converted
            .iter()
            .find(|m| m["role"] == serde_json::json!("tool"))
            .expect("the tool message must be present");
        assert_eq!(tool_message["tool_call_id"], serde_json::json!("call_1"));
    }

    #[test]
    fn a_tool_result_without_an_id_falls_back_to_the_message_id() {
        let tool = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: String::new(),
                    content: "sunny".to_string(),
                    blocks: None,
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

        let converted = convert_openai_messages(&[tool]).expect("conversion must succeed");
        let tool_message = converted
            .iter()
            .find(|m| m["role"] == serde_json::json!("tool"))
            .expect("the tool message must be present");
        assert_eq!(
            tool_message["tool_call_id"],
            serde_json::json!("call_1"),
            "the message names the call the result answers"
        );
    }

    #[test]
    fn a_tool_result_without_any_id_is_refused() {
        let tool = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Tool,
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: String::new(),
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

        let err = convert_openai_messages(&[tool])
            .expect_err("a tool message without a call id cannot be correlated");
        assert!(format!("{err}").contains("call"), "{err}");
    }

    #[test]
    fn openai_chat_tool_use_blocks_become_tool_calls_and_tool_results_become_tool_messages() {
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
                        thought_signature: None,
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

        let out = convert_openai_messages(&[assistant, tool]).expect("conversion must succeed");

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

    #[test]
    fn an_input_audio_block_becomes_an_input_audio_part() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::InputAudio {
                input_audio: llm_suite_types::message::InputAudioContent {
                    data: "QUJD".to_string(),
                    format: "wav".to_string(),
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
        let out = convert_openai_messages(&[user]).expect("must convert");
        let part = &out[0]["content"][0];
        assert_eq!(part["type"], serde_json::json!("input_audio"));
        assert_eq!(part["input_audio"]["format"], serde_json::json!("wav"));
        assert_eq!(part["input_audio"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn an_unsupported_audio_format_is_rejected() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::InputAudio {
                input_audio: llm_suite_types::message::InputAudioContent {
                    data: "QUJD".to_string(),
                    format: "ogg".to_string(),
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
        let err = convert_openai_messages(&[user]).expect_err("only wav and mp3 are accepted");
        assert!(format!("{err}").contains("mp3"), "{err}");
    }

    #[test]
    fn a_files_api_image_is_rejected() {
        let user = Message {
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
        let err = convert_openai_messages(&[user]).expect_err("chat has no image file id");
        assert!(format!("{err}").contains("file"), "{err}");
    }

    #[test]
    fn a_file_input_maps_to_file_data_or_file_id() {
        let base64_user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "QUJD".to_string(),
                    },
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
        let out = convert_openai_messages(&[base64_user]).expect("must convert");
        let part = &out[0]["content"][0];
        assert_eq!(part["type"], serde_json::json!("file"));
        assert_eq!(part["file"]["file_data"], serde_json::json!("QUJD"));
        assert_eq!(part["file"]["filename"], serde_json::json!("doc.pdf"));

        let id_user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::FileId {
                        file_id: "file_1".to_string(),
                    },
                    filename: None,
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
        let out = convert_openai_messages(&[id_user]).expect("must convert");
        assert_eq!(
            out[0]["content"][0]["file"]["file_id"],
            serde_json::json!("file_1")
        );
    }

    #[test]
    fn a_remote_url_file_input_is_rejected() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Url {
                        url: "https://example.com/a.pdf".to_string(),
                    },
                    filename: None,
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
        let err = convert_openai_messages(&[user]).expect_err("no URL form on chat file parts");
        assert!(format!("{err}").contains("file_id"), "{err}");
    }

    #[test]
    fn an_image_detail_is_sent_in_the_official_shape() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: Some(llm_suite_types::message::ImageDetail::High),
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
        };
        let out = convert_openai_messages(&[user]).expect("must convert");
        assert_eq!(
            out[0]["content"][0]["image_url"]["detail"],
            serde_json::json!("high")
        );
    }
}
