//! The XML text protocol: a tool call is a `tool_use` block whose
//! `parameters` child carries the arguments.

use llm_suite_types::message::{LlmFunctionCall, LlmToolCall};

use super::call_id::generate_tool_call_id;

/// The tag that opens an XML tool call block.
pub(super) const TOOL_USE_OPEN_TAG: &str = "<tool_use>";
/// The tag that closes an XML tool call block.
pub(super) const TOOL_USE_CLOSE_TAG: &str = "</tool_use>";

/// Parse tool calls from XML format text.
///
/// Supported format:
/// ```xml
/// <tool_use>
///   <tool_name>tool_name</tool_name>
///   <parameters>
///     <param1>value1</param1>
///   </parameters>
/// </tool_use>
/// ```
pub fn parse_xml_tool_calls(xml_text: &str) -> Vec<LlmToolCall> {
    let mut results = Vec::new();
    let mut remaining = xml_text;

    while let Some(start_pos) = find_tag_start(remaining, "tool_use") {
        let after_start = &remaining[start_pos + TOOL_USE_OPEN_TAG.len()..];
        if let Some(end_pos) = find_tag_end(after_start, "tool_use") {
            let content = &after_start[..end_pos];
            if let Some(call) = parse_xml_tool_call_block(content) {
                results.push(call);
            }
            remaining = &after_start[end_pos + TOOL_USE_CLOSE_TAG.len()..];
        } else {
            break;
        }
    }

    results
}

fn find_tag_start(text: &str, tag_name: &str) -> Option<usize> {
    let pattern = format!("<{}>", tag_name);
    text.find(&pattern)
}

fn find_tag_end(text: &str, tag_name: &str) -> Option<usize> {
    let pattern = format!("</{}>", tag_name);
    text.find(&pattern)
}

fn parse_xml_tool_call_block(content: &str) -> Option<LlmToolCall> {
    let tool_name = extract_xml_element(content, "tool_name")?
        .trim()
        .to_string();
    let params_content = extract_xml_element(content, "parameters")?;
    let args = parse_xml_parameters(&params_content);

    Some(LlmToolCall {
        id: generate_tool_call_id(),
        r#type: "function".to_string(),
        call_id: None,
        function: LlmFunctionCall {
            name: tool_name,
            arguments: serde_json::to_string(&args).unwrap_or_default(),
        },
    })
}

fn extract_xml_element(content: &str, tag_name: &str) -> Option<String> {
    let open_tag = format!("<{}>", tag_name);
    let close_tag = format!("</{}>", tag_name);

    let start = content.find(&open_tag)? + open_tag.len();
    let end = content[start..].find(&close_tag)? + start;

    Some(content[start..end].to_string())
}

fn parse_xml_parameters(params_content: &str) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    let mut remaining = params_content.trim();

    while !remaining.is_empty() {
        remaining = remaining.trim_start();
        if remaining.is_empty() || !remaining.starts_with('<') {
            break;
        }

        if let Some((tag_name, inner, rest)) = extract_next_xml_element(remaining) {
            let value = if contains_nested_element(&inner) {
                if tag_name == "item" || inner.contains("<item>") {
                    serde_json::Value::Array(parse_xml_array(&inner))
                } else {
                    parse_xml_parameters(&inner)
                }
            } else {
                parse_xml_value(&inner)
            };
            result.insert(tag_name, value);
            remaining = rest;
        } else {
            break;
        }
    }

    serde_json::Value::Object(result)
}

fn extract_next_xml_element(text: &str) -> Option<(String, String, &str)> {
    if !text.starts_with('<') {
        return None;
    }

    let tag_end = text.find('>')?;
    let tag_name = text[1..tag_end].trim().to_string();
    let close_tag = format!("</{}>", tag_name);

    let after_open = &text[tag_end + 1..];
    let close_pos = after_open.find(&close_tag)?;
    let inner = after_open[..close_pos].to_string();
    let rest = &after_open[close_pos + close_tag.len()..];

    Some((tag_name, inner, rest))
}

fn contains_nested_element(text: &str) -> bool {
    text.trim().starts_with('<') && text.contains('>')
}

fn parse_xml_array(array_content: &str) -> Vec<serde_json::Value> {
    let mut items = Vec::new();
    let mut remaining = array_content.trim();

    while let Some(start) = find_tag_start(remaining, "item") {
        let after_start = &remaining[start + 6..];
        if let Some(end) = find_tag_end(after_start, "item") {
            let inner = &after_start[..end];
            let value = if contains_nested_element(inner) {
                parse_xml_parameters(inner)
            } else {
                parse_xml_value(inner)
            };
            items.push(value);
            remaining = &after_start[end + 7..];
        } else {
            break;
        }
    }

    items
}

fn parse_xml_value(value: &str) -> serde_json::Value {
    let trimmed = value.trim();

    if trimmed == "true" {
        serde_json::Value::Bool(true)
    } else if trimmed == "false" {
        serde_json::Value::Bool(false)
    } else if trimmed == "null" {
        serde_json::Value::Null
    } else if let Ok(int_val) = trimmed.parse::<i64>() {
        serde_json::Value::Number(int_val.into())
    } else if let Ok(float_val) = trimmed.parse::<f64>() {
        serde_json::Number::from_f64(float_val)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::String(trimmed.to_string()))
    } else {
        serde_json::Value::String(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xml_tool_calls() {
        let xml = r#"
<tool_use>
  <tool_name>search</tool_name>
  <parameters>
    <query>test</query>
  </parameters>
</tool_use>
"#;
        let calls = parse_xml_tool_calls(xml);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn xml_values_are_typed() {
        let xml = r#"
<tool_use>
  <tool_name>search</tool_name>
  <parameters>
    <count>3</count>
    <ratio>0.5</ratio>
    <flag>true</flag>
    <nothing>null</nothing>
    <list><item>a</item><item>2</item></list>
  </parameters>
</tool_use>"#;
        let calls = parse_xml_tool_calls(xml);
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["count"], serde_json::json!(3));
        assert_eq!(args["ratio"], serde_json::json!(0.5));
        assert_eq!(args["flag"], serde_json::json!(true));
        assert_eq!(args["nothing"], serde_json::Value::Null);
        assert_eq!(args["list"], serde_json::json!(["a", 2]));
    }

    #[test]
    fn xml_without_parameters_is_skipped() {
        let calls = parse_xml_tool_calls("<tool_use><tool_name>only</tool_name></tool_use>");
        assert!(calls.is_empty(), "missing <parameters> block is skipped");
        assert!(parse_xml_tool_calls("no tool use here").is_empty());
    }

    #[test]
    fn nested_non_invoke_xml_still_recurses() {
        let xml = "<tool_use><tool_name>t</tool_name><parameters>\
                   <meta><inner>v</inner></meta>\
                   </parameters></tool_use>";
        let calls = parse_xml_tool_calls(xml);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["meta"]["inner"], serde_json::json!("v"));
    }
}
