//! The raw JSON text protocol: the tool call is a JSON object, with or
//! without a code fence around it.

use llm_suite_types::message::{LlmFunctionCall, LlmToolCall};

use super::call_id::generate_tool_call_id;

/// Parse raw JSON tool calls (no markers).
pub fn parse_raw_json_tool_calls(text: &str) -> Vec<LlmToolCall> {
    let cleaned = text
        .replace("```json", "")
        .replace("```", "")
        .trim()
        .to_string();

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&cleaned) {
        let array = match value {
            serde_json::Value::Array(arr) => arr,
            other => vec![other],
        };

        array
            .into_iter()
            .filter_map(|v| convert_to_standard_tool_call(&v))
            .collect()
    } else {
        Vec::new()
    }
}

/// Convert any known tool call format to standard LlmToolCall.
pub(super) fn convert_to_standard_tool_call(value: &serde_json::Value) -> Option<LlmToolCall> {
    let obj = value.as_object()?;

    // Format 1: { tool: "name", parameters: {} }
    if let Some(tool) = obj.get("tool").and_then(|v| v.as_str()) {
        let params = obj
            .get("parameters")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        return Some(LlmToolCall {
            id: generate_tool_call_id(),
            r#type: "function".to_string(),
            call_id: None,
            function: LlmFunctionCall {
                name: tool.to_string(),
                arguments: serde_json::to_string(&params).unwrap_or_default(),
            },
            thought_signature: None,
        });
    }

    // Format 2: { name: "name", arguments: "{}" | {} }
    if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
        let args = match obj.get("arguments") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => serde_json::to_string(v).unwrap_or_default(),
            None => "{}".to_string(),
        };
        return Some(LlmToolCall {
            id: generate_tool_call_id(),
            r#type: "function".to_string(),
            call_id: None,
            function: LlmFunctionCall {
                name: name.to_string(),
                arguments: args,
            },
            thought_signature: None,
        });
    }

    // Format 3: OpenAI native { id, function: { name, arguments } }
    if let Some(function) = obj.get("function").and_then(|v| v.as_object()) {
        if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
            let args = match function.get("arguments") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => serde_json::to_string(v).unwrap_or_default(),
                None => "{}".to_string(),
            };
            return Some(LlmToolCall {
                id: obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(generate_tool_call_id),
                r#type: "function".to_string(),
                call_id: None,
                function: LlmFunctionCall {
                    name: name.to_string(),
                    arguments: args,
                },
                thought_signature: None,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_raw_json_tool_calls() {
        let json = r#"[{"name": "search", "arguments": {"query": "test"}}]"#;
        let calls = parse_raw_json_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn parse_raw_json_handles_fenced_and_single_objects() {
        let fenced = r#"```json
{"name": "search", "arguments": "{}"}
```"#;
        let calls = parse_raw_json_tool_calls(fenced);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let single = parse_raw_json_tool_calls(r#"{"tool": "lookup", "parameters": {"id": 1}}"#);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].function.name, "lookup");

        let invalid = parse_raw_json_tool_calls("this is not json");
        assert!(invalid.is_empty());
    }

    #[test]
    fn convert_supports_all_three_formats() {
        // Format 1: { tool, parameters }
        let f1 = parse_raw_json_tool_calls(r#"{"tool": "a", "parameters": {"x": 1}}"#);
        assert_eq!(f1[0].function.name, "a");

        // Format 2: { name, arguments: object }
        let f2 = parse_raw_json_tool_calls(r#"{"name": "b", "arguments": {"y": 2}}"#);
        assert_eq!(f2[0].function.name, "b");
        assert!(
            f2[0].function.arguments.contains("\"y\": 2")
                || f2[0].function.arguments.contains("\"y\":2")
        );

        // Format 3: OpenAI native with id + function
        let f3 = parse_raw_json_tool_calls(
            r#"[{"id": "call_9", "type": "function", "function": {"name": "c", "arguments": "{\"z\":3}"}}]"#,
        );
        assert_eq!(f3[0].function.name, "c");
        assert_eq!(f3[0].id, "call_9");

        // Unsupported shape -> filtered out
        let none = parse_raw_json_tool_calls(r#"{"foo": "bar"}"#);
        assert!(none.is_empty());
    }
}
