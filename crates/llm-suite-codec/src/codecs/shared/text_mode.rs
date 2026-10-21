//! Text-based tool mode: protocol negotiation, history rendering and the
//! extraction of tool calls that were carried as plain text.

use llm_suite_types::llm::{LlmRequest, ToolCallProtocol};
use llm_suite_types::message::Message;

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

/// Build the system prompt content for text-based tool mode: existing system
/// message + tool usage instructions + tool declarations.
pub fn text_mode_system_content(request: &LlmRequest) -> String {
    use llm_suite_tool_call::tool::protocol::{
        build_text_mode_system_content, extract_system_message,
    };
    let protocol = effective_tool_call_protocol(request);
    let (system, _) = extract_system_message(&request.messages);
    let tools = request.tools.as_deref().unwrap_or(&[]);
    build_text_mode_system_content(system.as_deref().unwrap_or(""), tools, protocol, false)
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
