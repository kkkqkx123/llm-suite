//! Detection of the text protocol a message carries.

use llm_suite_types::llm::ToolCallMarkers;

use super::xml::TOOL_USE_OPEN_TAG;

/// Check if text contains XML tool calls.
pub fn has_xml_tool_calls(text: &str) -> bool {
    text.contains(TOOL_USE_OPEN_TAG)
}

/// Check if text contains wrapped JSON tool calls.
pub fn has_json_tool_calls(text: &str, markers: &ToolCallMarkers) -> bool {
    text.contains(markers.effective_start())
}

/// Check if text contains raw JSON tool calls.
pub fn has_raw_json_tool_calls(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with('{') || trimmed.starts_with('[')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_helpers_detect_format() {
        assert!(has_xml_tool_calls("<tool_use>..."));
        assert!(!has_xml_tool_calls("no tags"));
        assert!(has_json_tool_calls(
            "<<<TOOL_CALL>>>{}",
            &ToolCallMarkers::default_json()
        ));
        assert!(!has_json_tool_calls(
            "plain",
            &ToolCallMarkers::default_json()
        ));
        assert!(has_raw_json_tool_calls(r#"{"tool": "x"}"#));
        assert!(has_raw_json_tool_calls("[1,2]"));
        assert!(!has_raw_json_tool_calls("text"));
        assert!(!has_raw_json_tool_calls("  text  "));
    }
}
