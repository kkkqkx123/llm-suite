use llm_suite_types::llm::{
    ToolCallMarkers, ToolCallProtocol, DEFAULT_JSON_MARKER_END, DEFAULT_JSON_MARKER_START,
};
use llm_suite_types::message::{
    FileContent, FileSource, LlmToolCall, Message, MessageContent, MessageContentValue, MessageRole,
};

/// Default XML tags used for text-mode tool call/result rendering.
const XML_TOOL_CALL: &str = "tool_use";
const XML_TOOL_NAME: &str = "tool_name";
const XML_TOOL_ARGS: &str = "parameters";
const XML_TOOL_RESULT: &str = "tool_result";
const XML_TOOL_CALL_ID: &str = "tool_call_id";
const XML_TOOL_OUTPUT: &str = "tool_output";

/// Convert an entire message history from native function-calling format to a
/// text-based tool format (XML/JSON):
/// - assistant messages with `tool_calls` get the calls rendered into their
///   content and the `tool_calls` field dropped
/// - tool-result messages become user messages with the result rendered as text
///
/// Returns messages unchanged for `ToolCallProtocol::Native`.
pub fn convert_to_text_mode(
    messages: &[Message],
    format: &ToolCallProtocol,
    markers: Option<&ToolCallMarkers>,
) -> Vec<Message> {
    if *format == ToolCallProtocol::Native {
        return messages.to_vec();
    }

    messages
        .iter()
        .map(|msg| {
            if msg.role == MessageRole::Assistant
                && msg
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
            {
                convert_assistant_message(msg, format, markers)
            } else if msg.role == MessageRole::Tool && msg.tool_call_id.is_some() {
                convert_tool_result_message(msg, format, markers)
            } else if let MessageContentValue::Rich(blocks) = &msg.content {
                if blocks.iter().any(|block| {
                    matches!(
                        block,
                        llm_suite_types::message::MessageContent::ToolUse { .. }
                    )
                }) {
                    convert_tool_use_message(msg, format, markers)
                } else if blocks.iter().any(is_media_block) {
                    convert_media_message(msg)
                } else {
                    msg.clone()
                }
            } else {
                msg.clone()
            }
        })
        .collect()
}

/// Whether a block carries media the text protocol cannot carry inline.
fn is_media_block(block: &MessageContent) -> bool {
    matches!(
        block,
        MessageContent::ImageUrl { .. }
            | MessageContent::InputAudio { .. }
            | MessageContent::FileInput { .. }
    )
}

/// Convert a message whose rich content carries tool use blocks into a
/// text-only message: the calls are rendered in the text protocol and the
/// blocks are dropped.
fn convert_tool_use_message(
    message: &Message,
    format: &ToolCallProtocol,
    markers: Option<&ToolCallMarkers>,
) -> Message {
    let mut converted = message.clone();
    let calls: Vec<LlmToolCall> = match &message.content {
        MessageContentValue::Rich(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                llm_suite_types::message::MessageContent::ToolUse { tool_use } => {
                    Some(LlmToolCall {
                        id: tool_use.id.clone(),
                        r#type: "function".to_string(),
                        call_id: None,
                        function: llm_suite_types::message::LlmFunctionCall {
                            name: tool_use.name.clone(),
                            arguments: tool_use.input.to_string(),
                        },
                        thought_signature: tool_use.thought_signature.clone(),
                    })
                }
                _ => None,
            })
            .collect(),
        MessageContentValue::Text(_) => Vec::new(),
    };

    let tool_call_text = render_tool_calls(&calls, format, markers);
    let existing = extract_text(message);
    let content = if existing.is_empty() {
        tool_call_text
    } else if tool_call_text.is_empty() {
        existing
    } else {
        format!("{existing}\n\n{tool_call_text}")
    };
    converted.content = MessageContentValue::Text(with_placeholders(content, message));
    converted.tool_calls = None;
    converted
}

/// Text-mode protocols cannot carry media inline; render every media block
/// as a placeholder line instead of dropping them.
///
/// The placeholders name the payload, they never carry it: a data URI is
/// reduced to its media type and size so the base64 payload stays out of
/// the prompt.
fn convert_media_message(message: &Message) -> Message {
    let mut converted = message.clone();
    converted.content =
        MessageContentValue::Text(with_placeholders(message.text_content(), message));
    converted
}

/// The placeholder lines of the media blocks of a message.
fn media_placeholders(message: &Message) -> Vec<String> {
    let MessageContentValue::Rich(blocks) = &message.content else {
        return Vec::new();
    };
    blocks
        .iter()
        .filter_map(|block| match block {
            MessageContent::ImageUrl { image_url } => {
                Some(format!("[image: {}]", media_reference(&image_url.url)))
            }
            MessageContent::InputAudio { input_audio } => {
                Some(format!("[audio: {}]", input_audio.format))
            }
            MessageContent::FileInput { file } => {
                Some(format!("[file: {}]", file_reference(file)))
            }
            _ => None,
        })
        .collect()
}

/// The text of a message followed by the placeholders of its media blocks.
///
/// No path through the text protocol loses a media block: the placeholders
/// stand in for the payload the protocol cannot carry.
fn with_placeholders(text: String, message: &Message) -> String {
    let placeholders = media_placeholders(message).join("\n");
    if placeholders.is_empty() {
        text
    } else if text.is_empty() {
        placeholders
    } else {
        format!("{text}\n{placeholders}")
    }
}

/// A short reference to a media payload.
///
/// A remote reference stays as it is; an inline payload is named by its
/// media type and size, never by its content.
fn media_reference(url: &str) -> String {
    let Some(rest) = url.strip_prefix("data:") else {
        return url.to_string();
    };
    let (meta, payload) = rest.split_once(',').unwrap_or((rest, ""));
    let media_type = meta.trim_end_matches(";base64");
    let media_type = if media_type.is_empty() {
        "application/octet-stream"
    } else {
        media_type
    };
    format!("{media_type} ({} base64 chars)", payload.len())
}

/// A short reference to a document payload.
fn file_reference(file: &FileContent) -> String {
    match &file.source {
        FileSource::Base64 { media_type, data } => {
            format!("{media_type} ({} base64 chars)", data.len())
        }
        FileSource::Url { url } => url.clone(),
        FileSource::FileId { file_id } => file_id.clone(),
        FileSource::Content { blocks } => format!("{} content blocks", blocks.len()),
    }
}

/// Convert an assistant message with tool calls into a text-only message.
pub fn convert_assistant_message(
    message: &Message,
    format: &ToolCallProtocol,
    markers: Option<&ToolCallMarkers>,
) -> Message {
    let mut converted = message.clone();
    let calls = message.tool_calls.as_deref().unwrap_or_default();

    let tool_call_text = render_tool_calls(calls, format, markers);
    let existing = extract_text(message);
    let content = if existing.is_empty() {
        tool_call_text
    } else {
        format!("{existing}\n\n{tool_call_text}")
    };
    converted.content = MessageContentValue::Text(with_placeholders(content, message));
    converted.tool_calls = None;
    converted
}

/// Convert a tool-result message into a text-mode user message.
pub fn convert_tool_result_message(
    message: &Message,
    format: &ToolCallProtocol,
    markers: Option<&ToolCallMarkers>,
) -> Message {
    let mut converted = message.clone();
    let id = message.tool_call_id.as_deref().unwrap_or("");
    let rendered = render_tool_result(id, &extract_text(message), format, markers);
    converted.role = MessageRole::User;
    converted.content = MessageContentValue::Text(with_placeholders(rendered, message));
    converted.tool_call_id = None;
    converted
}

fn extract_text(message: &Message) -> String {
    message.text_content()
}

/// Render tool calls into the text representation of the given format.
pub fn render_tool_calls(
    calls: &[LlmToolCall],
    format: &ToolCallProtocol,
    markers: Option<&ToolCallMarkers>,
) -> String {
    match format {
        ToolCallProtocol::Xml => calls
            .iter()
            .map(|tc| {
                format!(
                    "<{XML_TOOL_CALL}>\n<{XML_TOOL_NAME}>{}</{XML_TOOL_NAME}>\n<{XML_TOOL_ARGS}>{}</{XML_TOOL_ARGS}>\n</{XML_TOOL_CALL}>",
                    tc.function.name, tc.function.arguments
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        ToolCallProtocol::JsonWrapped | ToolCallProtocol::JsonRaw => {
            let start = markers.map_or(DEFAULT_JSON_MARKER_START, ToolCallMarkers::effective_start);
            let end = markers.map_or(DEFAULT_JSON_MARKER_END, ToolCallMarkers::effective_end);
            calls
                .iter()
                .map(|tc| {
                    let parsed = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        .unwrap_or_else(|_| serde_json::json!(tc.function.arguments));
                    format!(
                        "{start}\n{}\n{end}",
                        serde_json::json!({
                            "tool": tc.function.name,
                            "parameters": parsed,
                        })
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        ToolCallProtocol::Native => String::new(),
    }
}

/// Render a single tool result into the text representation of the given format.
pub fn render_tool_result(
    tool_call_id: &str,
    output: &str,
    format: &ToolCallProtocol,
    markers: Option<&ToolCallMarkers>,
) -> String {
    match format {
        ToolCallProtocol::Xml => format!(
            "<{XML_TOOL_RESULT}>\n<{XML_TOOL_CALL_ID}>{tool_call_id}</{XML_TOOL_CALL_ID}>\n<{XML_TOOL_OUTPUT}>{output}</{XML_TOOL_OUTPUT}>\n</{XML_TOOL_RESULT}>"
        ),
        ToolCallProtocol::JsonWrapped | ToolCallProtocol::JsonRaw => {
            let start = markers.map_or(DEFAULT_JSON_MARKER_START, ToolCallMarkers::effective_start);
            let end = markers.map_or(DEFAULT_JSON_MARKER_END, ToolCallMarkers::effective_end);
            format!(
                "{start}\n{}\n{end}",
                serde_json::json!({
                    "tool_call_id": tool_call_id,
                    "output": output,
                })
            )
        }
        ToolCallProtocol::Native => output.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(role: MessageRole, text: &str) -> Message {
        Message {
            id: llm_suite_types::Id::new(),
            role,
            content: MessageContentValue::Text(text.to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    fn make_tool_call(id: &str, name: &str, args: &str) -> LlmToolCall {
        LlmToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
            thought_signature: None,
        }
    }

    fn make_assistant_with_calls(calls: Vec<LlmToolCall>) -> Message {
        let mut msg = make_msg(MessageRole::Assistant, "checking");
        msg.tool_calls = Some(calls);
        msg
    }

    fn make_tool_result(id: &str, output: &str) -> Message {
        let mut msg = make_msg(MessageRole::Tool, output);
        msg.tool_call_id = Some(id.to_string());
        msg
    }

    fn make_rich_image_message() -> Message {
        Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::Text {
                    text: "look at this".to_string(),
                },
                llm_suite_types::message::MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://e.com/a.png".to_string(),
                        detail: None,
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
        }
    }

    #[test]
    fn image_blocks_become_placeholders_in_text_mode() {
        for format in [ToolCallProtocol::Xml, ToolCallProtocol::JsonWrapped] {
            let converted = convert_to_text_mode(&[make_rich_image_message()], &format, None);
            match &converted[0].content {
                MessageContentValue::Text(text) => {
                    assert!(text.contains("look at this"), "text kept: {text}");
                    assert!(
                        text.contains("[image: https://e.com/a.png]"),
                        "placeholder kept: {text}"
                    );
                }
                other => panic!("expected Text content, got {:?}", other),
            }
        }
    }

    #[test]
    fn a_media_message_keeps_every_media_block() {
        let message = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::Text {
                    text: "look at this".to_string(),
                },
                llm_suite_types::message::MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://e.com/a.png".to_string(),
                        detail: None,
                        file_id: None,
                    },
                },
                llm_suite_types::message::MessageContent::FileInput {
                    file: llm_suite_types::message::FileContent {
                        source: llm_suite_types::message::FileSource::Url {
                            url: "https://e.com/report.pdf".to_string(),
                        },
                        filename: Some("report.pdf".to_string()),
                        detail: None,
                        citations_enabled: None,
                        context: None,
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
        let converted = convert_to_text_mode(&[message], &ToolCallProtocol::Xml, None);
        let MessageContentValue::Text(text) = &converted[0].content else {
            panic!("expected text");
        };
        assert!(text.contains("[image: https://e.com/a.png]"), "{text}");
        assert!(
            text.contains("[file: https://e.com/report.pdf]"),
            "the document must not be dropped: {text}"
        );
    }

    #[test]
    fn an_inline_payload_never_reaches_the_prompt() {
        let message = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "data:image/png;base64,QUJDREVGRw==".to_string(),
                        detail: None,
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
        };
        let converted = convert_to_text_mode(&[message], &ToolCallProtocol::Xml, None);
        let MessageContentValue::Text(text) = &converted[0].content else {
            panic!("expected text");
        };
        assert!(text.contains("image/png"), "{text}");
        assert!(
            !text.contains("QUJDREVGRw"),
            "the payload stays out: {text}"
        );
    }

    #[test]
    fn rich_tool_use_blocks_are_rendered_in_text_mode() {
        let message = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![
                llm_suite_types::message::MessageContent::Text {
                    text: "checking".to_string(),
                },
                llm_suite_types::message::MessageContent::ToolUse {
                    tool_use: llm_suite_types::message::ToolUseContent {
                        id: "c1".to_string(),
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
        let converted = convert_to_text_mode(&[message], &ToolCallProtocol::Xml, None);
        let MessageContentValue::Text(text) = &converted[0].content else {
            panic!("expected text");
        };
        assert!(text.contains("checking"), "{text}");
        assert!(text.contains("<tool_use>"), "{text}");
        assert!(
            text.contains("<tool_name>get_weather</tool_name>"),
            "{text}"
        );
    }

    #[test]
    fn native_format_returns_messages_unchanged() {
        let messages = vec![
            make_assistant_with_calls(vec![make_tool_call("c1", "get_weather", "{}")]),
            make_tool_result("c1", "sunny"),
        ];
        let converted = convert_to_text_mode(&messages, &ToolCallProtocol::Native, None);
        assert_eq!(converted.len(), 2);
        assert!(converted[0].tool_calls.is_some(), "native keeps tool_calls");
    }

    #[test]
    fn xml_conversion_embeds_tool_calls_and_results() {
        let messages = vec![
            make_assistant_with_calls(vec![make_tool_call(
                "c1",
                "get_weather",
                r#"{"city":"Beijing"}"#,
            )]),
            make_tool_result("c1", "sunny"),
        ];
        let converted = convert_to_text_mode(&messages, &ToolCallProtocol::Xml, None);

        let assistant = &converted[0];
        assert!(assistant.tool_calls.is_none(), "tool_calls dropped");
        let MessageContentValue::Text(text) = &assistant.content else {
            panic!("expected text");
        };
        assert!(text.contains("<tool_use>"), "{text}");
        assert!(
            text.contains("<tool_name>get_weather</tool_name>"),
            "{text}"
        );
        assert!(text.contains(r#"{"city":"Beijing"}"#), "{text}");

        let result = &converted[1];
        assert_eq!(result.role, MessageRole::User, "tool result -> user");
        let MessageContentValue::Text(result_text) = &result.content else {
            panic!("expected text");
        };
        assert!(result_text.contains("<tool_result>"), "{result_text}");
        assert!(
            result_text.contains("<tool_call_id>c1</tool_call_id>"),
            "{result_text}"
        );
        assert!(result_text.contains("sunny"), "{result_text}");
    }

    #[test]
    fn json_conversion_uses_markers() {
        let messages = vec![
            make_assistant_with_calls(vec![make_tool_call("c1", "get_weather", "{}")]),
            make_tool_result("c1", "sunny"),
        ];
        let converted = convert_to_text_mode(&messages, &ToolCallProtocol::JsonWrapped, None);

        let MessageContentValue::Text(text) = &converted[0].content else {
            panic!("expected text");
        };
        assert!(text.contains("<<<TOOL_CALL>>>"), "{text}");
        let parsed: serde_json::Value = serde_json::from_str(
            text.split("<<<TOOL_CALL>>>")
                .nth(1)
                .unwrap()
                .split("<<<END_TOOL_CALL>>>")
                .next()
                .unwrap()
                .trim(),
        )
        .expect("marker payload must be JSON");
        assert_eq!(parsed["tool"], serde_json::json!("get_weather"));
        assert_eq!(parsed["parameters"], serde_json::json!({}));

        let MessageContentValue::Text(result_text) = &converted[1].content else {
            panic!("expected text");
        };
        assert!(
            result_text.contains("\"tool_call_id\":\"c1\""),
            "{result_text}"
        );
        assert!(
            result_text.contains("\"output\":\"sunny\""),
            "{result_text}"
        );
    }

    #[test]
    fn assistant_without_calls_is_untouched() {
        let messages = vec![make_msg(MessageRole::Assistant, "plain answer")];
        let converted = convert_to_text_mode(&messages, &ToolCallProtocol::Xml, None);
        assert_eq!(converted[0].content, messages[0].content);
        assert!(converted[0].tool_calls.is_none());
    }

    #[test]
    fn a_tool_call_message_keeps_its_media_as_a_placeholder() {
        let mut assistant = make_msg(MessageRole::Assistant, "");
        assistant.content = MessageContentValue::Rich(vec![
            MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "c1".to_string(),
                    name: "screenshot".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                },
            },
            MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/png;base64,QUJDREVGRw==".to_string(),
                    detail: None,
                    file_id: None,
                },
            },
        ]);
        let converted = convert_to_text_mode(&[assistant], &ToolCallProtocol::Xml, None);
        let MessageContentValue::Text(text) = &converted[0].content else {
            panic!("expected text");
        };
        assert!(text.contains("<tool_use>"), "{text}");
        assert!(
            text.contains("[image: image/png"),
            "the picture must not be dropped: {text}"
        );
        assert!(
            !text.contains("QUJDREVGRw"),
            "the payload stays out of the prompt: {text}"
        );
    }

    #[test]
    fn a_tool_result_message_keeps_its_media_as_a_placeholder() {
        let mut result = make_msg(MessageRole::Tool, "attached");
        result.tool_call_id = Some("c1".to_string());
        result.content = MessageContentValue::Rich(vec![
            MessageContent::Text {
                text: "attached".to_string(),
            },
            MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "QUJDREVGRw==".to_string(),
                    },
                    filename: Some("report.pdf".to_string()),
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            },
        ]);
        let converted = convert_to_text_mode(&[result], &ToolCallProtocol::Xml, None);
        let MessageContentValue::Text(text) = &converted[0].content else {
            panic!("expected text");
        };
        assert!(text.contains("<tool_result>"), "{text}");
        assert!(
            text.contains("[file: application/pdf"),
            "the document must not be dropped: {text}"
        );
        assert!(
            !text.contains("QUJDREVGRw"),
            "the payload stays out of the prompt: {text}"
        );
    }
}
