//! The `general` tool's invoke protocol: the request body of the outer
//! tool carries the inner tool calls as JSON.

use llm_suite_types::message::{LlmFunctionCall, LlmToolCall};

use super::call_id::generate_tool_call_id;

/// Parse inner tool invocations from the `general` tool's JSON request body.
///
/// Supported formats (JSON):
///
/// Single call: `{"tool": "web_search", "parameters": {"query": "rust"}}`
/// Multiple calls: `[{"tool": "a", "parameters": {}}, {"tool": "b", "parameters": {}}]`
///
/// Parameters support full JSON types: strings, numbers, booleans, null,
/// arrays, and objects. This eliminates the XML escaping and type inference
/// limitations of the previous XML-based invoke protocol.
pub fn parse_invoke_json_calls(json_text: &str) -> Vec<LlmToolCall> {
    match parse_invoke_json_calls_detailed(json_text) {
        Ok(items) => items.into_iter().filter_map(|r| r.ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// Per-item parse failure inside a `general` request body. `index` is the
/// position inside the request array (`None` means the whole body failed to
/// parse, e.g. empty text or invalid JSON). Callers that need audit-grade
/// error positions (history conversion, per-call result aggregation) must
/// use [`parse_invoke_json_calls_detailed`]; the plain
/// [`parse_invoke_json_calls`] wrapper keeps the historical silent-drop
/// behavior for compatibility.
#[derive(Debug, Clone, PartialEq)]
pub struct InvokeParseError {
    pub index: Option<usize>,
    pub reason: String,
}

impl std::fmt::Display for InvokeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.index {
            Some(idx) => write!(f, "invoke item {idx} is invalid: {}", self.reason),
            None => write!(f, "invoke body is invalid: {}", self.reason),
        }
    }
}

/// Parse inner tool invocations with explicit per-item errors.
///
/// Returns `Err` when the whole body is unusable (empty text, invalid JSON,
/// or a JSON shape that is neither object nor array); otherwise returns one
/// entry per array element (or a single entry for the object form), where
/// `Err` entries pinpoint the offending position instead of silently
/// dropping it.
pub fn parse_invoke_json_calls_detailed(
    json_text: &str,
) -> Result<Vec<Result<LlmToolCall, InvokeParseError>>, InvokeParseError> {
    let trimmed = json_text.trim();
    if trimmed.is_empty() {
        return Err(InvokeParseError {
            index: None,
            reason: "empty request body".to_string(),
        });
    }

    let value: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            return Err(InvokeParseError {
                index: None,
                reason: format!("invalid JSON: {e}"),
            });
        }
    };

    match value {
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                return Err(InvokeParseError {
                    index: None,
                    reason: "empty invoke array".to_string(),
                });
            }
            Ok(items
                .into_iter()
                .enumerate()
                .map(|(index, item)| {
                    convert_invoke_object(item).map_err(|reason| InvokeParseError {
                        index: Some(index),
                        reason,
                    })
                })
                .collect())
        }
        obj @ serde_json::Value::Object(_) => {
            Ok(vec![convert_invoke_object(obj).map_err(|reason| InvokeParseError {
                index: Some(0),
                reason,
            })])
        }
        _ => Err(InvokeParseError {
            index: None,
            reason: "expected a JSON object or an array of objects".to_string(),
        }),
    }
}

/// Convert a single JSON invoke object into an LlmToolCall.
/// Expects `{"tool": "...", "parameters": {...}}`. Returns the reason when
/// invalid.
///
/// The assigned id is a placeholder only: the `general` invoker overwrites it
/// immediately with the stable `general_history::derive_inner_call_id`
/// derivation (`"{outer}#{index}#{tool}"`) so checkpoint replay keys survive
/// re-execution. Callers must not persist or correlate on this id.
fn convert_invoke_object(value: serde_json::Value) -> Result<LlmToolCall, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "expected a JSON object".to_string())?;
    let tool_name = obj
        .get("tool")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing required string field \"tool\"".to_string())?;
    if tool_name.is_empty() {
        return Err("field \"tool\" must not be empty".to_string());
    }
    let parameters = match obj.get("parameters") {
        Some(serde_json::Value::Object(map)) => map.clone(),
        Some(_) => return Err("field \"parameters\" must be a JSON object".to_string()),
        None => serde_json::Map::new(),
    };
    Ok(LlmToolCall {
        id: generate_tool_call_id(),
        r#type: "function".to_string(),
        call_id: None,
        function: LlmFunctionCall {
            name: tool_name.to_string(),
            arguments: serde_json::to_string(&parameters).unwrap_or_default(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invoke_json_parses_single_call_with_typed_parameters() {
        let json = r#"{"tool": "web_search", "parameters": {"query": "rust 异步", "limit": 5, "cached": true}}"#;
        let calls = parse_invoke_json_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "web_search");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], serde_json::json!("rust 异步"));
        assert_eq!(args["limit"], serde_json::json!(5));
        assert_eq!(args["cached"], serde_json::json!(true));
    }

    #[test]
    fn invoke_json_supports_nested_array_and_object_parameters() {
        let json = r#"{"tool": "list_files", "parameters": {"patterns": ["*.rs", "*.toml"], "config": {"recursive": true}}}"#;
        let calls = parse_invoke_json_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_files");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["patterns"], serde_json::json!(["*.rs", "*.toml"]));
        assert_eq!(args["config"], serde_json::json!({"recursive": true}));
    }

    #[test]
    fn invoke_json_handles_multiple_calls_and_malformed_input() {
        let json = r#"[
  {"tool": "write_file", "parameters": {"path": "a.txt", "content": "hi"}},
  {"tool": "edit_file", "parameters": {"path": "b.txt"}}
]"#;
        let calls = parse_invoke_json_calls(json);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "write_file");
        assert_eq!(calls[1].function.name, "edit_file");

        // Missing tool / no parameters -> skipped.
        assert!(parse_invoke_json_calls(r#"{"parameters": {"a": 1}}"#).is_empty());
        assert!(parse_invoke_json_calls(r#"{"tool": "x", "parameters": []}"#).is_empty());
        assert!(parse_invoke_json_calls("plain text").is_empty());
        assert!(parse_invoke_json_calls("").is_empty());
        assert!(parse_invoke_json_calls("null").is_empty());
    }

    #[test]
    fn general_request_body_is_passed_through_verbatim() {
        let xml = r#"<tool_use>
  <tool_name>general</tool_name>
  <parameters>
    <request>{"tool": "web_search", "parameters": {"query": "rust", "limit": 5}}</request>
  </parameters>
</tool_use>"#;
        let calls = super::super::xml::parse_xml_tool_calls(xml);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "general");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let request = args["request"].as_str().unwrap();
        assert_eq!(
            request,
            r#"{"tool": "web_search", "parameters": {"query": "rust", "limit": 5}}"#
        );

        // The verbatim body is re-parsed by parse_invoke_json_calls: the
        // inner call resolves and keeps its typed values.
        let inner = parse_invoke_json_calls(request);
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].function.name, "web_search");
        let inner_args: serde_json::Value =
            serde_json::from_str(&inner[0].function.arguments).unwrap();
        assert_eq!(inner_args["query"], serde_json::json!("rust"));
        assert_eq!(inner_args["limit"], serde_json::json!(5));
    }

    #[test]
    fn general_request_with_whitespace_is_trimmed_but_kept_raw() {
        let xml = "<tool_use><tool_name>general</tool_name><parameters><request>\n  \
                   {\"tool\": \"write_file\", \"parameters\": {\"path\": \"a.txt\"}}\n\
                   </request></parameters></tool_use>";
        let calls = super::super::xml::parse_xml_tool_calls(xml);
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args["request"].as_str().unwrap(),
            "{\"tool\": \"write_file\", \"parameters\": {\"path\": \"a.txt\"}}"
        );
        let inner = parse_invoke_json_calls(args["request"].as_str().unwrap());
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].function.name, "write_file");
    }
}
