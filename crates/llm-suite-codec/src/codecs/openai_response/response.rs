//! Parsing of a non-streaming OpenAI Responses response body.

use crate::error::LlmResult;
use llm_suite_types::llm::{LlmRequest, LlmResult as LlmResponseType};

use super::usage::parse_responses_usage;

/// Parse an OpenAI Responses body into the common response shape.
///
/// In text mode the tool calls carried as plain text are extracted from the
/// answer and attached to the result.
pub fn parse_response(body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
    let mut result = parse_response_inner(body)?;
    if super::super::shared::is_text_mode(request) {
        if let Some(content) = result.content.clone() {
            let calls = super::super::shared::parse_text_tool_calls(request, &content);
            if !calls.is_empty() {
                result.tool_calls = Some(calls.clone());
                result.message.tool_calls = Some(calls);
            }
        }
    }
    Ok(result)
}

pub fn parse_tool_calls_from_output(
    output: &[serde_json::Value],
) -> Vec<llm_suite_types::message::LlmToolCall> {
    output
        .iter()
        .filter_map(|item| {
            let item_type = item.get("type").and_then(|v| v.as_str())?;
            if item_type != "function_call" {
                return None;
            }
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = match item.get("arguments").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => {
                    tracing::debug!(
                        tool_id = id.as_str(),
                        tool_name = name.as_str(),
                        "function call arguments missing or not a string; falling back to empty object",
                    );
                    "{}".to_string()
                }
            };
            Some(llm_suite_types::message::LlmToolCall {
                id,
                r#type: "function".to_string(),
                call_id: Some(call_id),
                function: llm_suite_types::message::LlmFunctionCall { name, arguments },
            })
        })
        .collect()
}

pub fn parse_response_inner(body: &str) -> LlmResult<LlmResponseType> {
    let json: serde_json::Value = serde_json::from_str(body)?;

    let id = json.get("id").and_then(|v| v.as_str()).map(String::from);
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let status = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Surface failed responses or explicit error objects as errors rather
    // than returning a silently-empty result. `error` is omitted (or null)
    // on success, so it is only acted on when present and non-null.
    if status.as_deref() == Some("failed") {
        let msg = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| "response generation failed".to_string());
        return Err(crate::error::LlmError::InvalidResponse(format!(
            "OpenAI Responses: {msg}"
        )));
    }
    if let Some(err) = json.get("error").filter(|e| !e.is_null()) {
        let msg = err
            .get("message")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| err.as_str().map(String::from))
            .unwrap_or_else(|| "response error".to_string());
        return Err(crate::error::LlmError::InvalidResponse(format!(
            "OpenAI Responses: {msg}"
        )));
    }

    // Aggregate text from every `output_text` block across all `message`
    // output items (mirrors the official `Response.output_text` property),
    // capturing refusals and reasoning text into their own channels.
    let mut text_parts: Vec<String> = Vec::new();
    let mut refusal_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    // The sources of a grounded answer ride on the `output_text` blocks.
    let mut annotations: Vec<serde_json::Value> = Vec::new();
    if let Some(output) = json.get("output").and_then(|v| v.as_array()) {
        for item in output {
            match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "message" => {
                    if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                        for block in blocks {
                            match block.get("type").and_then(|t| t.as_str()) {
                                Some("output_text") => {
                                    if let Some(t) = block.get("text").and_then(|v| v.as_str())
                                    {
                                        text_parts.push(t.to_string());
                                    }
                                    if let Some(list) =
                                        block.get("annotations").and_then(|v| v.as_array())
                                    {
                                        annotations.extend(list.iter().cloned());
                                    }
                                }
                                // A refusal is a content block inside the
                                // message item, not a separate output item.
                                Some("refusal") => {
                                    if let Some(t) =
                                        block.get("refusal").and_then(|v| v.as_str())
                                    {
                                        refusal_parts.push(t.to_string());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "reasoning" => {
                    // The summary list and the reasoning text list are
                    // siblings of the item, both optional.
                    if let Some(blocks) = item.get("summary").and_then(|c| c.as_array()) {
                        for block in blocks {
                            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                reasoning_parts.push(t.to_string());
                            }
                        }
                    }
                    if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                        for block in blocks {
                            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                reasoning_parts.push(t.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.concat())
    };
    let refusal = if refusal_parts.is_empty() {
        None
    } else {
        Some(refusal_parts.concat())
    };
    let reasoning_content = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.concat())
    };

    let mut tool_calls: Option<Vec<llm_suite_types::message::LlmToolCall>> = None;
    if let Some(output_arr) = json.get("output").and_then(|v| v.as_array()) {
        let calls = parse_tool_calls_from_output(output_arr);
        if !calls.is_empty() {
            tool_calls = Some(calls);
        }
    }

    let usage = json.get("usage").map(parse_responses_usage);

    let message = llm_suite_types::message::Message {
        id: llm_suite_types::Id::new(),
        role: llm_suite_types::message::MessageRole::Assistant,
        content: llm_suite_types::message::MessageContentValue::Text(
            content.clone().unwrap_or_default(),
        ),
        timestamp: llm_suite_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: tool_calls.clone(),
        // The reasoning text also lives on the message so that replaying
        // the message in a follow-up turn keeps it, the same way the
        // Anthropic codec does.
        thinking: reasoning_content.clone(),
        thinking_signature: None,
        metadata: None,
    };

    // The raw lifecycle state of the response; the stop value below is
    // derived from it.
    let raw_status = status;

    // The stop value uses the same vocabulary as the Chat Completions
    // codec: the Responses `status` is a lifecycle state, not a stop
    // reason. The reason an incomplete response stopped lives in
    // `incomplete_details.reason`.
    let finish_reason = stop_reason(
        raw_status.as_deref(),
        json.pointer("/incomplete_details/reason")
            .and_then(|v| v.as_str()),
        tool_calls.as_ref().is_some_and(|calls| !calls.is_empty()),
    );

    // Why an incomplete response stopped: `max_output_tokens`,
    // `max_messages`, `content_filter` or `steered`.
    let mut metadata = std::collections::HashMap::new();
    if let Some(reason) = json
        .pointer("/incomplete_details/reason")
        .and_then(|v| v.as_str())
    {
        metadata.insert("incomplete_reason".to_string(), serde_json::json!(reason));
    }
    if let Some(ref status) = raw_status {
        metadata.insert("status".to_string(), serde_json::json!(status));
    }
    if !annotations.is_empty() {
        metadata.insert(
            llm_suite_types::llm::ANNOTATIONS_KEY.to_string(),
            serde_json::json!(annotations),
        );
    }
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };

    let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

    Ok(LlmResponseType {
        id,
        model,
        content,
        message,
        tool_calls,
        usage,
        finish_reason,
        duration: 0,
        reasoning_content,
        reasoning_signature: None,
        reasoning_tokens,
        metadata,
        refusal,
        stream_stats: None,
    })
}

/// Map the Responses lifecycle state onto the stop vocabulary the Chat
/// Completions codec reports.
///
/// A finished response stopped because the model answered (`stop`), ran
/// out of output budget (`length`), was cut by a content filter
/// (`content_filter`) or asked for a tool (`tool_calls`). The official
/// `incomplete_details.reason` values (`max_output_tokens`,
/// `max_messages`, `content_filter`, `steered`) decide the outcome.
///
/// The Responses API has no tool-call stop reason — a completed response
/// that carries function calls is reported as `tool_calls`, the value the
/// Chat Completions codec reports for the same situation.
pub fn stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    has_tool_calls: bool,
) -> Option<String> {
    match status {
        Some("failed") => Some("error".to_string()),
        Some("incomplete") => Some(match incomplete_reason {
            Some("max_output_tokens") | Some("max_messages") => "length".to_string(),
            Some("content_filter") => "content_filter".to_string(),
            Some(other) => other.to_string(),
            None => "incomplete".to_string(),
        }),
        Some("completed") => Some(if has_tool_calls {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        }),
        Some("cancelled") => Some("cancelled".to_string()),
        Some("queued") | Some("in_progress") => None,
        Some(other) => Some(other.to_string()),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::openai_response::test_support::minimal_request;

    #[test]
    fn parse_response_extracts_text_output_and_usage() {
        let body = r#"{
            "id": "resp_1",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "the answer"}]
            }],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 2}
            }
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.content.as_deref(), Some("the answer"));
        assert_eq!(
            result.finish_reason.as_deref(),
            Some("stop"),
            "the stop value uses the same vocabulary as the Chat Completions codec"
        );
        assert_eq!(
            result.metadata.as_ref().unwrap()["status"],
            serde_json::json!("completed"),
            "the raw response status stays available"
        );
        assert_eq!(result.usage.as_ref().unwrap().total_tokens, 20);
        assert_eq!(result.usage.as_ref().unwrap().cache_read_tokens, Some(5));
        assert_eq!(result.usage.as_ref().unwrap().cache_write_tokens, Some(2));
        assert_eq!(result.model, "gpt-4o");
    }

    #[test]
    fn parse_response_extracts_function_calls() {
        let body = r#"{
            "id": "resp_2",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": ""}]},
                {"type": "function_call", "id": "fc_7", "call_id": "call_7", "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
            ]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        let calls = result.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "fc_7");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn parse_response_keeps_message_text_when_function_call_follows() {
        let body = r#"{
            "id": "resp_4",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "thinking aloud"}]},
                {"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
            ]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.content.as_deref(), Some("thinking aloud"));
        assert_eq!(
            result.tool_calls.as_ref().unwrap()[0].function.name,
            "get_weather"
        );
    }

    #[test]
    fn parse_response_tolerates_empty_output() {
        let body = r#"{"id":"resp_3","object":"response","model":"gpt-4o","status":"incomplete"}"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.content, None);
        assert_eq!(result.tool_calls, None);
        assert_eq!(result.usage, None);
    }

    #[test]
    fn parse_response_captures_call_id() {
        let body = r#"{
            "id": "resp_2", "object":"response", "model":"gpt-4o", "status":"completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":""}]},
                {"type":"function_call","id":"fc_7","call_id":"call_7","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}
            ]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        let calls = result.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id.as_deref(), Some("call_7"));
    }

    #[test]
    fn a_tool_calling_turn_reports_the_tool_call_stop_reason() {
        // The Responses API has no tool-call stop reason; the unified
        // vocabulary the Chat codec reports must still be used.
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}
            ]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("tool_calls"));

        let plain = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hi"}]}
            ]
        }"#;
        let result = parse_response(plain, &minimal_request()).unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn the_provider_total_token_count_is_preferred() {
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
            "usage":{"input_tokens":10,"output_tokens":5,"total_tokens":21}
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        let usage = result.usage.expect("usage must be present");
        assert_eq!(usage.total_tokens, 21, "the provider owns the total");

        // Without the provider total the sum is the fallback.
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
            "usage":{"input_tokens":10,"output_tokens":5}
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        let usage = result.usage.expect("usage must be present");
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn the_sources_of_a_grounded_answer_are_reported_as_metadata() {
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"grounded","annotations":[
                {"type":"url_citation","url":"https://example.com","title":"Example"}
            ]}]}]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        let metadata = result.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["annotations"][0]["url"],
            serde_json::json!("https://example.com")
        );
    }

    #[test]
    fn parse_response_aggregates_multiple_text_blocks() {
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"first part"},{"type":"output_text","text":" second part"}]},
                {"type":"message","content":[{"type":"output_text","text":" and more"}]}
            ]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(
            result.content.as_deref(),
            Some("first part second part and more")
        );
    }

    #[test]
    fn parse_response_captures_refusal_and_reasoning() {
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [
                {"type":"reasoning","summary":[{"type":"summary_text","text":"I am thinking..."}]},
                {"type":"message","content":[
                    {"type":"output_text","text":"hello"},
                    {"type":"refusal","refusal":"I cannot help with that."}
                ]}
            ]
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.content.as_deref(), Some("hello"));
        assert_eq!(result.refusal.as_deref(), Some("I cannot help with that."));
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("I am thinking...")
        );
        assert_eq!(
            result.message.thinking.as_deref(),
            Some("I am thinking..."),
            "the message carries the reasoning so a follow-up turn keeps it"
        );
    }

    #[test]
    fn parse_response_reports_the_reasoning_tokens_of_the_usage() {
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"completed",
            "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 12}
            }
        }"#;
        let result = parse_response(body, &minimal_request()).unwrap();
        assert_eq!(result.reasoning_tokens, Some(12));
        assert_eq!(result.usage.as_ref().unwrap().reasoning_tokens, Some(12));
    }

    #[test]
    fn an_incomplete_response_reports_why_it_stopped() {
        let cases = [
            ("max_output_tokens", "length"),
            ("max_messages", "length"),
            ("content_filter", "content_filter"),
        ];
        for (reason, expected) in cases {
            let body = format!(
                r#"{{"id":"resp_x","object":"response","model":"gpt-4o","status":"incomplete",
                    "incomplete_details":{{"reason":"{reason}"}},
                    "output": []}}"#
            );
            let result = parse_response(&body, &minimal_request()).unwrap();
            assert_eq!(
                result.finish_reason.as_deref(),
                Some(expected),
                "{reason} must map to {expected}"
            );
            assert_eq!(
                result.metadata.as_ref().unwrap()["incomplete_reason"],
                serde_json::json!(reason)
            );
            assert_eq!(
                result.metadata.as_ref().unwrap()["status"],
                serde_json::json!("incomplete")
            );
        }
    }

    #[test]
    fn parse_response_failed_status_returns_error() {
        let body = r#"{
            "id":"resp_x","object":"response","model":"gpt-4o","status":"failed",
            "error":{"message":"rate limit exceeded"},
            "output": []
        }"#;
        let err = parse_response(body, &minimal_request())
            .expect_err("failed response must surface an error");
        assert!(format!("{err:?}").contains("rate limit exceeded"));
    }
}
