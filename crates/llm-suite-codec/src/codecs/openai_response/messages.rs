//! Conversion of internal messages into OpenAI Responses `input` items.

use std::collections::HashMap;

use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

pub fn convert_messages(messages: &[Message]) -> Vec<serde_json::Value> {
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
            MessageRole::System => continue,
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };

        // Map the message body to a Responses `input` content value
        // (string shorthand or `input_text` / `input_image` parts).
        let content: serde_json::Value = match &msg.content {
            MessageContentValue::Text(text) => serde_json::Value::String(text.clone()),
            // The Responses API uses `input_text` / `input_image`
            // content part types (not Chat's `text` / `image_url`),
            // so Rich blocks are mapped explicitly.
            MessageContentValue::Rich(blocks) => serde_json::Value::Array(
                blocks
                    .iter()
                    .map(|block| match block {
                        MessageContent::Text { text } => {
                            serde_json::json!({"type": "input_text", "text": text})
                        }
                        MessageContent::ImageUrl { image_url } => {
                            let mut part = serde_json::json!({
                                "type": "input_image",
                                "image_url": image_url.url,
                            });
                            if let Some(ref detail) = image_url.detail {
                                part["detail"] = serde_json::json!(detail);
                            }
                            part
                        }
                        MessageContent::Thinking { thinking, .. } => {
                            serde_json::json!({"type": "input_text", "text": thinking})
                        }
                        MessageContent::RedactedThinking { data } => {
                            // The Responses API has no redacted-thinking
                            // part; the opaque payload is kept as text
                            // so the block is not dropped from history.
                            serde_json::json!({"type": "input_text", "text": data})
                        }
                        // Tool blocks are not valid user content in
                        // the Responses API; render them as text so
                        // history is not silently dropped.
                        MessageContent::ToolUse { tool_use } => {
                            serde_json::json!({
                                "type": "input_text",
                                "text": tool_use.input.to_string(),
                            })
                        }
                        MessageContent::ToolResult { tool_result } => {
                            serde_json::json!({
                                "type": "input_text",
                                "text": tool_result.content,
                            })
                        }
                    })
                    .collect(),
            ),
        };

        match msg.role {
            // Tool results are `function_call_output` items in the
            // Responses API, correlated to the originating function call by
            // `call_id` (the message's `tool_call_id`). Chat's
            // `tool_calls`/`tool_call_id` shape is rejected by the API.
            MessageRole::Tool => {
                if let Some(ref tool_call_id) = msg.tool_call_id {
                    // The output item has to carry the same id the call
                    // item was keyed by, whichever identity the history
                    // recorded the result under.
                    let call_id = call_ids
                        .get(tool_call_id.as_str())
                        .cloned()
                        .unwrap_or_else(|| tool_call_id.clone());
                    let output = match &msg.content {
                        MessageContentValue::Text(text) => text.clone(),
                        MessageContentValue::Rich(blocks) => blocks
                            .iter()
                            .map(|b| match b {
                                MessageContent::Text { text } => text.clone(),
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

        let items = convert_messages(&[assistant, result]);
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
        }];
        let entries = convert_messages(&req_messages);
        let parts = entries[0]["content"].as_array().expect("content array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["image_url"], "https://example.com/a.png");
        assert_eq!(parts[1]["detail"], "auto");
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
        let items = convert_messages(&[assistant, tool_result]);

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
}
