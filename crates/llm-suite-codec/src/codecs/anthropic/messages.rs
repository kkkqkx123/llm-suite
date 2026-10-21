//! Conversion of internal messages into Anthropic `messages`.

use crate::codecs::helpers::{resolve_image_payload, ResolvedImage};
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
                            blocks.push(convert_image_block(&image_url.url)?);
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
                            tool_result_blocks.push(tool_result_block(
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

pub fn tool_result_block(
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
mod tests {
    use super::*;
    use crate::codecs::anthropic::request::build_body;
    use crate::codecs::anthropic::test_support::{msg, profile, request, rich_msg, tool_call, tool_msg};

    #[test]
    fn the_replayed_thinking_block_comes_first() {
        // The API requires the thinking blocks to precede the text and the
        // tool calls of the assistant turn.
        let mut assistant = msg(MessageRole::Assistant, "the answer");
        assistant.thinking = Some("Let me verify".to_string());
        assistant.thinking_signature = Some("sig-abc".to_string());

        let body = build_body(
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
    fn parallel_tool_results_merge_into_one_user_message() {
        let mut assistant = msg(MessageRole::Assistant, "Checking weather and time");
        assistant.tool_calls = Some(vec![
            tool_call("call_1", "get_weather", r#"{"city":"Beijing"}"#),
            tool_call("call_2", "get_time", "{}"),
        ]);

        let body = build_body(
            &request(
                vec![
                    msg(MessageRole::User, "weather and time?"),
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
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![tool_call("call_1", "boom", "{}")]);

        let body = build_body(
            &request(
                vec![
                    msg(MessageRole::User, "run"),
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
        let with_signature = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
                },
                MessageContent::Text {
                    text: "42".to_string(),
                },
            ],
        );
        let body = build_body(
            &request(vec![msg(MessageRole::User, "hi"), with_signature], None),
            &profile(),
        )
        .expect("build must succeed");
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], serde_json::json!("thinking"));
        assert_eq!(content[0]["thinking"], serde_json::json!("Let me verify"));
        assert_eq!(content[0]["signature"], serde_json::json!("sig-abc"));
        assert_eq!(content[1]["type"], serde_json::json!("text"));

        let without_signature = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Thinking {
                    thinking: "unsigned".to_string(),
                    signature: None,
                },
                MessageContent::Text {
                    text: "42".to_string(),
                },
            ],
        );
        let body = build_body(
            &request(vec![msg(MessageRole::User, "hi"), without_signature], None),
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
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/cat.png".to_string(),
                    detail: None,
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("build must succeed");
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
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/webp;base64,QUJD".to_string(),
                    detail: None,
                },
            }],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("build must succeed");
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
        let user = rich_msg(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "data:image/bmp;base64,QUJD".to_string(),
                    detail: None,
                },
            }],
        );
        let err = build_body(&request(vec![user], None), &profile())
            .expect_err("bmp must be rejected");
        assert!(matches!(err, LlmError::ConfigError(_)));
    }

    #[test]
    fn empty_text_blocks_and_empty_messages_are_skipped() {
        let user = rich_msg(
            MessageRole::User,
            vec![
                MessageContent::Text {
                    text: String::new(),
                },
                MessageContent::Text {
                    text: "hi".to_string(),
                },
            ],
        );
        let body = build_body(&request(vec![user], None), &profile()).expect("build must succeed");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], serde_json::json!("hi"));

        // A message that only carries empty text produces no message at all.
        let ghost = rich_msg(
            MessageRole::User,
            vec![MessageContent::Text {
                text: "  ".to_string(),
            }],
        );
        let body = build_body(&request(vec![ghost], None), &profile()).expect("build must succeed");
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn a_stored_thinking_block_comes_before_the_text() {
        // The API requires the thinking blocks to lead the assistant turn,
        // no matter where the history stored them.
        let assistant = rich_msg(
            MessageRole::Assistant,
            vec![
                MessageContent::Text {
                    text: "the answer".to_string(),
                },
                MessageContent::Thinking {
                    thinking: "Let me verify".to_string(),
                    signature: Some("sig-abc".to_string()),
                },
            ],
        );

        let body = build_body(
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
}
