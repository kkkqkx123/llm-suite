//! Local validation of OpenAI Chat request messages.

use crate::error::LlmResult;
use llm_suite_types::llm::LlmRequest;
use llm_suite_types::message::{MessageContent, MessageContentValue};

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
        if msg.role != llm_suite_types::message::MessageRole::Tool || msg.tool_call_id.is_some() {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::message::{MessageContentValue, MessageRole};

    #[test]
    fn a_tool_message_without_any_result_is_rejected() {
        let request = LlmRequest {
            profile_id: "p1".to_string(),
            messages: vec![llm_suite_types::message::Message {
                id: llm_suite_types::Id::new(),
                role: MessageRole::Tool,
                content: MessageContentValue::Text("sunny".to_string()),
                timestamp: 0,
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                thinking: None,
                thinking_signature: None,
                metadata: None,
            }],
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
        let err = validate_openai_chat_messages(&request)
            .expect_err("a tool message without a result id must be rejected");
        assert!(format!("{err}").contains("tool_call_id"), "{err}");
    }
}
