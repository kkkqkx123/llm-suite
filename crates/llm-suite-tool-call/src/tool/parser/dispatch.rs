//! Dispatch across the text protocols: the format order the caller asked
//! for, and the best-effort read of a partially streamed message.

use llm_suite_types::llm::ToolCallMarkers;
use llm_suite_types::message::LlmToolCall;

use super::json_raw::parse_raw_json_tool_calls;
use super::json_wrapped::parse_json_tool_calls;
use super::options::{ParseFormat, ToolCallParseOptions};
use super::xml::{parse_xml_tool_calls, TOOL_USE_OPEN_TAG};

/// Auto-detect and parse tool calls from text.
pub fn parse_from_text(text: &str, options: &ToolCallParseOptions) -> Vec<LlmToolCall> {
    if text.is_empty() {
        return Vec::new();
    }

    for format in &options.preferred_formats {
        let calls = match format {
            ParseFormat::Xml => parse_xml_tool_calls(text),
            ParseFormat::Json => {
                let markers = options
                    .markers
                    .clone()
                    .unwrap_or_else(ToolCallMarkers::default_json);
                parse_json_tool_calls(text, &markers)
            }
            ParseFormat::Raw => parse_raw_json_tool_calls(text),
        };

        if !calls.is_empty() {
            return calls;
        }
    }

    // Fallback for partial streaming content
    if options.allow_partial {
        parse_partial(text, options)
    } else {
        Vec::new()
    }
}

/// Parse partial tool calls (for streaming chunks).
pub fn parse_partial(text: &str, options: &ToolCallParseOptions) -> Vec<LlmToolCall> {
    if text.is_empty() {
        return Vec::new();
    }

    // Try XML first
    if text.contains(TOOL_USE_OPEN_TAG) {
        return parse_xml_tool_calls(text);
    }

    // Try wrapped JSON
    let markers = options
        .markers
        .clone()
        .unwrap_or_else(ToolCallMarkers::default_json);
    let start_marker = markers.effective_start();
    if text.contains(start_marker) {
        return parse_json_tool_calls(text, &markers);
    }

    // Try raw JSON best-effort
    parse_raw_json_tool_calls(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_from_text_tries_formats_in_order() {
        let opts = ToolCallParseOptions {
            preferred_formats: vec![ParseFormat::Xml],
            markers: None,
            allow_partial: false,
        };
        let xml = parse_from_text(
            "<tool_use><tool_name>t1</tool_name><parameters><q>1</q></parameters></tool_use>",
            &opts,
        );
        assert_eq!(xml.len(), 1);

        // Json preferred: xml in the text is ignored, json is parsed.
        let json_opts = ToolCallParseOptions {
            preferred_formats: vec![ParseFormat::Json],
            markers: None,
            allow_partial: false,
        };
        let calls = parse_from_text(
            "<<<TOOL_CALL>>>{\"tool\": \"t2\"}<<<END_TOOL_CALL>>>",
            &json_opts,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "t2");

        // No match and no partial: empty.
        assert!(parse_from_text("plain text", &json_opts).is_empty());
        assert!(parse_from_text("", &json_opts).is_empty());
    }

    #[test]
    fn parse_partial_handles_incomplete_content() {
        let opts = ToolCallParseOptions {
            preferred_formats: vec![ParseFormat::Xml],
            markers: None,
            allow_partial: false,
        };
        // A complete XML block embedded in still-streaming text is parsed.
        let xml = parse_partial(
            "<tool_use><tool_name>t1</tool_name><parameters><q>1</q></parameters></tool_use>more",
            &opts,
        );
        assert_eq!(xml.len(), 1);
        assert_eq!(xml[0].function.name, "t1");

        // A genuinely truncated XML block (no closing tag) yields nothing.
        let truncated = parse_partial(
            "<tool_use><tool_name>t1</tool_name><parameters><q>1</q></parameters>",
            &opts,
        );
        assert!(truncated.is_empty());

        // Wrapped json parses when the end marker is present.
        let calls = parse_partial(
            "<<<TOOL_CALL>>>{\"tool\": \"t2\"}<<<END_TOOL_CALL>>>",
            &opts,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "t2");

        // Missing end marker: no call can be extracted.
        let truncated = parse_partial("<<<TOOL_CALL>>>{\"tool\": \"t2\"}", &opts);
        assert!(truncated.is_empty());

        // Raw json best effort.
        let raw = parse_partial(r#"{"name": "t3", "arguments"#, &opts);
        assert!(raw.is_empty());

        assert!(parse_partial("", &opts).is_empty());
    }
}
