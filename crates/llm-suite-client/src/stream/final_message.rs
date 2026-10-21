//! Assembly of the final message of a streamed turn.

use llm_suite_types::llm::{MessageStreamFinal, TokenUsageStats, ANNOTATIONS_KEY};
use llm_suite_types::message::{
    LlmToolCall, Message, MessageContent, MessageContentValue, MessageRole,
};

/// The accumulated state of a turn, as the final message builder sees it.
pub struct TurnState<'a> {
    pub text: &'a str,
    pub reasoning: &'a str,
    pub reasoning_signature: &'a str,
    pub refusal: &'a str,
    pub stream_id: Option<&'a str>,
    pub stream_model: Option<&'a str>,
    pub content_blocks: &'a [MessageContent],
    pub annotations: &'a [serde_json::Value],
    pub usage: Option<&'a TokenUsageStats>,
    pub tool_calls: &'a [LlmToolCall],
    pub call_signatures: &'a [Option<String>],
}

/// Build the final message of the turn.
pub fn build_final_message(state: TurnState<'_>) -> MessageStreamFinal {
    // The rich blocks lead the content: a provider that sends them
    // (Anthropic's redacted thinking) requires them before the text of the
    // turn.
    let content = if state.content_blocks.is_empty() {
        MessageContentValue::Text(state.text.to_string())
    } else {
        let mut blocks = state.content_blocks.to_vec();
        if !state.text.is_empty() {
            blocks.push(MessageContent::Text {
                text: state.text.to_string(),
            });
        }
        MessageContentValue::Rich(blocks)
    };
    let metadata = message_metadata(state.call_signatures, state.annotations);

    let message = Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::Assistant,
        content,
        timestamp: llm_suite_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: if state.tool_calls.is_empty() {
            None
        } else {
            Some(state.tool_calls.to_vec())
        },
        thinking: if state.reasoning.is_empty() {
            None
        } else {
            Some(state.reasoning.to_string())
        },
        thinking_signature: if state.reasoning_signature.is_empty() {
            None
        } else {
            Some(state.reasoning_signature.to_string())
        },
        metadata,
    };

    MessageStreamFinal {
        message,
        usage: state.usage.cloned(),
        stream_stats: None, // Filled by the gateway stream wrapper
        id: state.stream_id.map(String::from),
        model: state.stream_model.map(String::from),
        refusal: if state.refusal.is_empty() {
            None
        } else {
            Some(state.refusal.to_string())
        },
    }
}

/// The metadata of the final message: the provider signatures of the
/// tool calls and the sources of a grounded answer. `None` when the
/// turn carries neither.
fn message_metadata(
    signatures: &[Option<String>],
    annotations: &[serde_json::Value],
) -> Option<std::collections::HashMap<String, serde_json::Value>> {
    let mut metadata = tool_call_signatures(signatures);
    if annotations.is_empty() {
        return metadata;
    }
    metadata.get_or_insert_with(Default::default).insert(
        ANNOTATIONS_KEY.to_string(),
        serde_json::json!(annotations),
    );
    metadata
}

/// The provider signatures of the assembled tool calls, one entry per
/// call in the order the calls are reported. `None` when no call carries
/// a signature.
fn tool_call_signatures(
    signatures: &[Option<String>],
) -> Option<std::collections::HashMap<String, serde_json::Value>> {
    if signatures.iter().all(|signature| signature.is_none()) {
        return None;
    }
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        llm_suite_types::llm::THOUGHT_SIGNATURES_KEY.to_string(),
        serde_json::json!(signatures),
    );
    Some(metadata)
}
