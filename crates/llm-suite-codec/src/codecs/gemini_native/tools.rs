//! Tool declaration encoding for the Gemini native API.

use crate::error::LlmResult;
use llm_suite_types::tool::Tool;

/// Full `tools` array shape: `[{"functionDeclarations": [...]}]`, ready
/// to be assigned to the request body `tools` field. The parameter
/// schema is rewritten into Gemini's dialect, whose type names are
/// uppercase enum members.
pub fn convert_tools(tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
    let function_declarations: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            // The parameter schema is optional; a tool without one leaves
            // the field out instead of sending a null schema.
            let mut declaration = serde_json::Map::new();
            declaration.insert("name".to_string(), serde_json::json!(t.name));
            declaration.insert("description".to_string(), serde_json::json!(t.description));
            if let Some(schema) = &t.parameters {
                declaration.insert(
                    "parameters".to_string(),
                    crate::generation::gemini_schema_value(
                        &serde_json::to_value(schema).unwrap_or(serde_json::Value::Null),
                    ),
                );
            }
            serde_json::Value::Object(declaration)
        })
        .collect();
    Ok(vec![serde_json::json!({
        "functionDeclarations": function_declarations,
    })])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_parameter_schema_uses_gemini_enum_members() {
        let tools = vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["city"]
            }
        }))
        .unwrap()];
        let shape = convert_tools(&tools).unwrap();
        let parameters = &shape[0]["functionDeclarations"][0]["parameters"];
        assert_eq!(parameters["type"], serde_json::json!("OBJECT"));
        assert_eq!(
            parameters["properties"]["city"]["type"],
            serde_json::json!("STRING")
        );
        assert_eq!(
            parameters["properties"]["days"]["type"],
            serde_json::json!("INTEGER")
        );
        assert_eq!(
            parameters["properties"]["tags"]["type"],
            serde_json::json!("ARRAY")
        );
        assert_eq!(
            parameters["properties"]["tags"]["items"]["type"],
            serde_json::json!("STRING")
        );
    }
}
