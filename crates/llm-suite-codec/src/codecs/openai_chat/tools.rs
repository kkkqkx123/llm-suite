//! Tool schema conversion for the OpenAI Chat Completions `tools` field.

use crate::error::LlmResult;

pub fn convert_openai_tools(
    tools: &[llm_suite_types::tool::Tool],
) -> LlmResult<Vec<serde_json::Value>> {
    let tool_defs: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            // The official `function.parameters` is a JSON Schema object; a
            // tool without a schema must not serialize `null` (the API
            // rejects it). An empty object schema is the accepted
            // no-argument declaration.
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
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                }
            })
        })
        .collect();
    Ok(tool_defs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_without_parameters_declares_an_empty_object_schema() {
        let tools = vec![llm_suite_types::tool::Tool {
            name: "get_time".to_string(),
            description: "Returns the current time".to_string(),
            parameters: None,
        }];
        let shape = convert_openai_tools(&tools).expect("must convert");
        let parameters = &shape[0]["function"]["parameters"];
        assert!(
            parameters.is_object(),
            "the schema must be an object, never null: {shape:?}"
        );
        assert_eq!(parameters["type"], serde_json::json!("object"));
    }
}
