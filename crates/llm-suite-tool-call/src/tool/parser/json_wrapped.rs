//! The marker-wrapped JSON text protocol: the tool call is a JSON object
//! between a start and an end marker.

use llm_suite_types::llm::ToolCallMarkers;
use llm_suite_types::message::LlmToolCall;

use super::json_raw::convert_to_standard_tool_call;

/// Parse tool calls from JSON wrapped with custom markers.
pub fn parse_json_tool_calls(text: &str, markers: &ToolCallMarkers) -> Vec<LlmToolCall> {
    let mut results = Vec::new();
    let start_marker = markers.effective_start();
    let end_marker = markers.effective_end();

    let mut remaining = text;

    while let Some(start_pos) = remaining.find(start_marker) {
        let after_start = &remaining[start_pos + start_marker.len()..];
        if let Some(end_pos) = after_start.find(end_marker) {
            let json_str = after_start[..end_pos].trim();
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(call) = convert_to_standard_tool_call(&value) {
                    results.push(call);
                }
            }
            remaining = &after_start[end_pos + end_marker.len()..];
        } else {
            break;
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_tool_calls() {
        let markers = ToolCallMarkers::default_json();
        let text = r#"
<<<TOOL_CALL>>>
{"tool": "search", "parameters": {"query": "test"}}
<<<END_TOOL_CALL>>>
"#;
        let calls = parse_json_tool_calls(text, &markers);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn custom_json_markers_are_respected() {
        let markers = ToolCallMarkers {
            start: Some("<<<CALL>>>".to_string()),
            end: Some("<<<DONE>>>".to_string()),
        };
        let calls = parse_json_tool_calls("<<<CALL>>>{\"tool\": \"custom\"}<<<DONE>>>", &markers);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "custom");

        // Default markers do not match custom text.
        let calls = parse_json_tool_calls(
            "<<<CALL>>>{\"tool\": \"custom\"}<<<DONE>>>",
            &ToolCallMarkers::default_json(),
        );
        assert!(calls.is_empty());
    }
}
