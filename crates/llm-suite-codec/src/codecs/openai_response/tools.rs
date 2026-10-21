//! Tool declaration encoding for the OpenAI Responses API.

use llm_suite_types::tool::Tool;

/// Wire tool declarations. The Responses API declares a function tool
/// flat: `name`, `description` and `parameters` sit next to `type`
/// instead of being wrapped in a `function` object.
pub fn response_tool_defs(tools: &[Tool]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            // The schema is a JSON Schema object; a tool without one must
            // not serialize `null` (the API rejects it). An empty object
            // schema is the accepted no-argument declaration.
            let parameters =
                t.parameters
                    .clone()
                    .unwrap_or(llm_suite_types::tool::ToolParameterSchema {
                        r#type: "object".to_string(),
                        properties: Default::default(),
                        required: Vec::new(),
                        additional_properties: None,
                    });
            serde_json::json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": parameters,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_declaration_is_flat() {
        let tools = vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()];
        let shape = response_tool_defs(&tools);
        assert_eq!(shape[0]["type"], serde_json::json!("function"));
        assert_eq!(shape[0]["name"], serde_json::json!("get_weather"));
        assert_eq!(shape[0]["description"], serde_json::json!("Get weather"));
        assert!(
            shape[0].get("function").is_none(),
            "Responses declares function tools flat"
        );
    }
}
