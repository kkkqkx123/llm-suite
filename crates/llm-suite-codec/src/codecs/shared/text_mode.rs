//! Text-based tool mode: protocol negotiation, history rendering and the
//! extraction of tool calls that were carried as plain text.

use crate::error::{LlmError, LlmResult};
use llm_suite_types::llm::{LlmRequest, ToolCallProtocol};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// Whether the request runs in text-based tool mode (non-native tool protocol).
///
/// The locked protocol wins over the requested one, so the mode branch, the
/// protocol used for parsing and the protocol used for rendering always
/// agree with each other.
pub fn is_text_mode(request: &LlmRequest) -> bool {
    effective_tool_call_protocol(request) != ToolCallProtocol::Native
}

/// Effective tool call protocol: the locked protocol wins, otherwise the
/// request protocol, otherwise native.
pub fn effective_tool_call_protocol(request: &LlmRequest) -> ToolCallProtocol {
    request
        .locked_tool_call_protocol
        .as_ref()
        .map(|c| c.format.clone())
        .or_else(|| request.tool_call_protocol.clone())
        .unwrap_or(ToolCallProtocol::Native)
}

/// Convert message history to the text-based tool protocol when the request
/// runs in text mode (native tool calls / results become XML or JSON blocks
/// in the message content). Messages are returned unchanged in native mode.
pub fn convert_history_for_text_mode(messages: &[Message], request: &LlmRequest) -> Vec<Message> {
    if !is_text_mode(request) {
        return messages.to_vec();
    }
    let protocol = effective_tool_call_protocol(request);
    let markers = request
        .locked_tool_call_protocol
        .as_ref()
        .and_then(|c| c.markers.clone());
    llm_suite_message::history_converter::convert_to_text_mode(
        messages,
        &protocol,
        markers.as_ref(),
    )
}

/// The system content of the request.
///
/// Every provider reserves a text-only channel for the system prompt, so a
/// system message that carries anything else is rejected instead of having
/// its content dropped silently.
pub fn system_content(request: &LlmRequest) -> LlmResult<String> {
    for msg in &request.messages {
        if msg.role != MessageRole::System {
            continue;
        }
        let MessageContentValue::Rich(blocks) = &msg.content else {
            continue;
        };
        for block in blocks {
            if !matches!(block, MessageContent::Text { .. }) {
                return Err(LlmError::ConfigError(format!(
                    "the system prompt is text only; it cannot carry a `{}` content block",
                    crate::codecs::helpers::block_name(block)
                )));
            }
        }
    }
    use llm_suite_tool_call::tool::protocol::extract_system_message;
    let (system, _) = extract_system_message(&request.messages);
    Ok(system.unwrap_or_default())
}

/// The request history with the system messages replaced by their merged
/// text content.
///
/// The Chat Completions format keeps the system prompt in the message list;
/// the merged text takes the place of the first system message so the
/// position of the prompt is preserved.
pub fn history_with_system_text(request: &LlmRequest) -> LlmResult<Vec<Message>> {
    let system = system_content(request)?;
    let mut history: Vec<Message> = Vec::with_capacity(request.messages.len());
    let mut placed = false;
    for msg in &request.messages {
        if msg.role != MessageRole::System {
            history.push(msg.clone());
            continue;
        }
        if !placed && !system.is_empty() {
            history.push(Message::system_text(system.clone()));
            placed = true;
        }
    }
    if !placed && !system.is_empty() {
        history.insert(0, Message::system_text(system));
    }
    Ok(history)
}

/// Build the system prompt content for text-based tool mode: existing system
/// message + tool usage instructions + tool declarations.
pub fn text_mode_system_content(request: &LlmRequest) -> LlmResult<String> {
    use llm_suite_tool_call::tool::protocol::build_text_mode_system_content;
    let protocol = effective_tool_call_protocol(request);
    let system = system_content(request)?;
    let tools = request.tools.as_deref().unwrap_or(&[]);
    Ok(build_text_mode_system_content(
        system.as_str(),
        tools,
        protocol,
        false,
    ))
}

/// Parse tool calls from text content in text-based tool mode.
pub fn parse_text_tool_calls(
    request: &LlmRequest,
    content: &str,
) -> Vec<llm_suite_types::message::LlmToolCall> {
    use llm_suite_tool_call::tool::parser::parse_from_text;
    use llm_suite_tool_call::tool::protocol::get_tool_call_parser_options;
    let protocol = effective_tool_call_protocol(request);
    if protocol == ToolCallProtocol::Native {
        return Vec::new();
    }
    let markers = request
        .locked_tool_call_protocol
        .as_ref()
        .and_then(|c| c.markers.clone());
    let options = get_tool_call_parser_options(protocol, markers.as_ref());
    parse_from_text(content, &options)
}
