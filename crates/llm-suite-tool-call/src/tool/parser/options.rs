//! The options and the format vocabulary shared by the parsers.

use llm_suite_types::llm::ToolCallMarkers;

/// Options for parsing tool calls from text.
#[derive(Debug, Clone)]
pub struct ToolCallParseOptions {
    pub preferred_formats: Vec<ParseFormat>,
    pub markers: Option<ToolCallMarkers>,
    pub allow_partial: bool,
}

impl Default for ToolCallParseOptions {
    fn default() -> Self {
        Self {
            preferred_formats: vec![ParseFormat::Xml, ParseFormat::Json, ParseFormat::Raw],
            markers: None,
            allow_partial: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ParseFormat {
    Xml,
    Json,
    Raw,
}
