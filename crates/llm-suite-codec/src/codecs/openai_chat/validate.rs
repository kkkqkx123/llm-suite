//! Local validation of OpenAI Chat request messages.

use crate::codecs::helpers::block_name;
use crate::error::LlmResult;
use llm_suite_types::llm::LlmRequest;
use llm_suite_types::message::{MessageContent, MessageContentValue, MessageRole};

/// OpenAI Chat requires every `tool` role message to carry `tool_call_id`.
/// Validate it so an invalid request fails fast locally instead of
/// round-tripping a `400` from the server. Text mode re-renders tool results
/// as inline blocks, so this only applies to native (non-text) requests.
///
/// A message whose result lives in a rich `ToolResult` block is not checked:
/// `convert_openai_messages` turns that block into a standalone `tool`
/// message that carries the id, and the source message itself never reaches
/// the wire as a `tool` entry.
pub fn validate_openai_chat_messages(request: &LlmRequest) -> LlmResult<()> {
    for msg in &request.messages {
        if msg.role != MessageRole::Tool || msg.tool_call_id.is_some() {
            continue;
        }
        let carries_tool_result = matches!(
            &msg.content,
            MessageContentValue::Rich(blocks)
                if blocks.iter().any(|b| matches!(b, MessageContent::ToolResult { .. }))
        );
        if carries_tool_result {
            continue;
        }
        return Err(crate::error::LlmError::ConfigError(
            "OpenAI Chat requires every `tool` role message to carry `tool_call_id`".to_string(),
        ));
    }
    validate_content_parts(request)
}

/// The content parts a role may carry.
///
/// Only the user role accepts the full multimodal union; every other role
/// takes text parts (the assistant next to its tool calls).
#[derive(Clone, Copy, PartialEq, Eq)]
enum AllowedParts {
    Multimodal,
    Text,
}

/// Check that every content block fits the role it sits in.
///
/// The API accepts the full content-part union on a user message only; a
/// developer, system or tool message takes text parts, and an assistant
/// message takes text parts next to its tool calls. A block that breaks the
/// rule is rejected here instead of being sent to the server.
fn validate_content_parts(request: &LlmRequest) -> LlmResult<()> {
    for msg in &request.messages {
        let allowed = match msg.role {
            MessageRole::User => AllowedParts::Multimodal,
            MessageRole::System | MessageRole::Assistant | MessageRole::Tool => AllowedParts::Text,
        };
        let MessageContentValue::Rich(blocks) = &msg.content else {
            continue;
        };
        for block in blocks {
            let name = block_name(block);
            let fits = match block {
                MessageContent::Text { .. }
                | MessageContent::Thinking { .. }
                | MessageContent::RedactedThinking { .. } => true,
                MessageContent::ImageUrl { .. }
                | MessageContent::InputAudio { .. }
                | MessageContent::FileInput { .. } => allowed == AllowedParts::Multimodal,
                // A tool call becomes the message-level `tool_calls`; only
                // the assistant turn has them.
                MessageContent::ToolUse { .. } => msg.role == MessageRole::Assistant,
                // A tool result becomes a standalone `tool` message that
                // answers the assistant turn that made the call.
                MessageContent::ToolResult { .. } => {
                    matches!(msg.role, MessageRole::Assistant | MessageRole::Tool)
                }
                // A native block is a provider block of the assistant turn.
                MessageContent::Native { .. } => msg.role == MessageRole::Assistant,
            };
            if !fits {
                return Err(crate::error::LlmError::ConfigError(format!(
                    "OpenAI Chat does not accept a `{name}` content part on a `{}` message; \
                     only a user message carries images, audio and files",
                    role_name(&msg.role)
                )));
            }
        }
    }
    Ok(())
}

/// The name of a role, used in the error message.
fn role_name(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::message::{MessageContentValue, MessageRole};

    fn request_with(messages: Vec<llm_suite_types::message::Message>) -> LlmRequest {
        LlmRequest {
            profile_id: "p1".to_string(),
            messages,
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
        }
    }

    fn text_message(role: MessageRole, text: &str) -> llm_suite_types::message::Message {
        llm_suite_types::message::Message {
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

    fn rich_message(
        role: MessageRole,
        blocks: Vec<MessageContent>,
    ) -> llm_suite_types::message::Message {
        llm_suite_types::message::Message {
            id: llm_suite_types::Id::new(),
            role,
            content: MessageContentValue::Rich(blocks),
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
    fn a_tool_message_without_any_result_is_rejected() {
        let request = request_with(vec![text_message(MessageRole::Tool, "sunny")]);
        let err = validate_openai_chat_messages(&request)
            .expect_err("a tool message without a result id must be rejected");
        assert!(format!("{err}").contains("tool_call_id"), "{err}");
    }

    #[test]
    fn an_image_on_a_user_message_is_accepted() {
        let request = request_with(vec![rich_message(
            MessageRole::User,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                    file_id: None,
                },
            }],
        )]);
        validate_openai_chat_messages(&request).expect("a user message carries images");
    }

    #[test]
    fn an_image_on_an_assistant_message_is_rejected() {
        let request = request_with(vec![rich_message(
            MessageRole::Assistant,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                    file_id: None,
                },
            }],
        )]);
        let err = validate_openai_chat_messages(&request)
            .expect_err("an assistant message takes text parts only");
        assert!(format!("{err}").contains("image_url"), "{err}");
        assert!(format!("{err}").contains("assistant"), "{err}");
    }

    #[test]
    fn an_image_on_a_system_message_is_rejected() {
        let request = request_with(vec![rich_message(
            MessageRole::System,
            vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                    file_id: None,
                },
            }],
        )]);
        let err = validate_openai_chat_messages(&request)
            .expect_err("a system message takes text parts only");
        assert!(format!("{err}").contains("system"), "{err}");
    }

    #[test]
    fn a_tool_call_on_a_user_message_is_rejected() {
        let request = request_with(vec![rich_message(
            MessageRole::User,
            vec![MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                },
            }],
        )]);
        let err = validate_openai_chat_messages(&request)
            .expect_err("a tool call belongs to the assistant turn");
        assert!(format!("{err}").contains("tool_use"), "{err}");
    }

    #[test]
    fn a_tool_result_on_a_user_message_is_rejected() {
        let request = request_with(vec![rich_message(
            MessageRole::User,
            vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    blocks: None,
                    is_error: None,
                },
            }],
        )]);
        let err = validate_openai_chat_messages(&request)
            .expect_err("a tool result belongs to the tool turn");
        assert!(format!("{err}").contains("tool_result"), "{err}");
    }

    #[test]
    fn a_tool_result_on_an_assistant_message_is_accepted() {
        // The converter turns the block into a standalone `tool` message
        // that answers the assistant turn.
        let request = request_with(vec![rich_message(
            MessageRole::Assistant,
            vec![MessageContent::ToolResult {
                tool_result: llm_suite_types::message::ToolResultContent {
                    tool_use_id: "call_1".to_string(),
                    content: "sunny".to_string(),
                    blocks: None,
                    is_error: None,
                },
            }],
        )]);
        validate_openai_chat_messages(&request).expect("the tool turn answers the call");
    }
}
