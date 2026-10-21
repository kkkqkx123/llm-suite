//! Internal message conversion into OpenAI Chat Completions `messages`.

use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

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
        super::super::validate::validate_openai_chat_messages(&request)
            .expect("a rich tool result must not be rejected");

        let converted = convert_openai_messages(&request.messages);
        let tool_message = converted
            .iter()
            .find(|m| m["role"] == serde_json::json!("tool"))
            .expect("the tool message must be present");
        assert_eq!(tool_message["tool_call_id"], serde_json::json!("call_1"));
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
