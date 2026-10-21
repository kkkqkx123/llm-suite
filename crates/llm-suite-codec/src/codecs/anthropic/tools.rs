//! Tool declaration encoding for the Anthropic Messages API.

use llm_suite_types::tool::{Tool, ToolParameterSchema};

/// Wire tool declarations. `input_schema` is mandatory for Anthropic and
/// must be a JSON Schema object; a missing schema falls back to an empty
/// object schema instead of serializing `null` (which the API rejects).
pub fn build_tool_defs(tools: &[Tool]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            let input_schema = t.parameters.clone().unwrap_or(ToolParameterSchema {
                r#type: "object".to_string(),
                properties: Default::default(),
                required: Vec::new(),
                additional_properties: None,
            });
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": input_schema,
            })
        })
        .collect()
}
