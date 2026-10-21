//! Parsing of a non-streaming Gemini native response body.

use crate::error::LlmResult;
use llm_suite_types::llm::{LlmRequest, LlmResult as LlmResponseType, THOUGHT_SIGNATURES_KEY};

use super::finish_reason::normalize_finish_reason;
use super::part::{read_parts, Part};
use super::usage::parse_usage_metadata;

/// Parse a Gemini native response body into the common response shape.
///
/// In text mode the tool calls carried as plain text are extracted from the
/// answer and attached to the result.
pub fn parse_response(body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
    let mut result = parse_gemini_response(body)?;
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

pub fn parse_gemini_response(body: &str) -> LlmResult<LlmResponseType> {
    let json: serde_json::Value = serde_json::from_str(body)?;

    // A blocked prompt carries no candidates; the reason lives in the
    // prompt feedback and has to surface as an error instead of an empty
    // answer.
    if let Some(block_reason) = json
        .pointer("/promptFeedback/blockReason")
        .and_then(|v| v.as_str())
    {
        let detail = json
            .pointer("/promptFeedback/blockReasonMessage")
            .and_then(|v| v.as_str())
            .unwrap_or(block_reason);
        return Err(crate::error::LlmError::InvalidResponse(format!(
            "Gemini blocked the prompt ({block_reason}): {detail}"
        )));
    }

    let candidates = json.get("candidates").and_then(|v| v.as_array());
    let first_candidate = candidates.and_then(|c| c.first());

    let mut text_content = String::new();
    let mut reasoning_content: Option<String> = None;
    let mut tool_calls = Vec::new();
    // One entry per tool call; the signature the provider attached to the
    // function call has to travel back with that call.
    let mut thought_signatures: Vec<Option<String>> = Vec::new();

    if let Some(candidate) = first_candidate {
        let content = candidate.get("content");
        if let Some(c) = content {
            let parts = c.get("parts").and_then(|v| v.as_array());
            if let Some(parts) = parts {
                for part in read_parts(parts) {
                    match part {
                        Part::Thought { text } => {
                            reasoning_content.get_or_insert_with(String::new).push_str(&text);
                        }
                        Part::Text { text } => {
                            text_content.push_str(&text);
                        }
                        Part::FunctionCall {
                            name,
                            arguments,
                            provider_id,
                            signature,
                        } => {
                            // The provider assigns the call id; it has to be
                            // echoed back in the matching function response.
                            // A locally generated id stands in for the missing
                            // one but must not be echoed back as if the
                            // provider had issued it.
                            let id = provider_id.clone().unwrap_or_else(|| {
                                format!("gemini_call_{}", llm_suite_common::generate_id())
                            });
                            tool_calls.push(llm_suite_types::message::LlmToolCall {
                                id,
                                r#type: "function".to_string(),
                                call_id: provider_id,
                                function: llm_suite_types::message::LlmFunctionCall {
                                    name,
                                    arguments,
                                },
                            });
                            thought_signatures.push(signature);
                        }
                    }
                }
            }
        }
    }

    let usage = json.get("usageMetadata").map(parse_usage_metadata);

    let mut message_metadata = std::collections::HashMap::new();
    if !thought_signatures.is_empty() {
        message_metadata.insert(
            THOUGHT_SIGNATURES_KEY.to_string(),
            serde_json::json!(thought_signatures),
        );
    }
    let message_metadata = if message_metadata.is_empty() {
        None
    } else {
        Some(message_metadata)
    };

    let message = llm_suite_types::message::Message {
        id: llm_suite_types::Id::new(),
        role: llm_suite_types::message::MessageRole::Assistant,
        content: llm_suite_types::message::MessageContentValue::Text(text_content.clone()),
        timestamp: llm_suite_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls.clone())
        },
        // The reasoning text also lives on the message so that replaying
        // the message in a follow-up turn keeps it.
        thinking: reasoning_content.clone(),
        thinking_signature: None,
        metadata: message_metadata,
    };

    // The envelope carries the normalized stop value; the raw
    // `finishReason` stays available in the metadata. Every codec
    // documents which spelling it reports, so consumers match per family.
    let finish_reason = first_candidate
        .and_then(|c| c.get("finishReason"))
        .and_then(|v| v.as_str())
        .map(normalize_finish_reason)
        .map(|reason| {
            // Gemini has no tool-call finish reason: a turn that ends
            // with a function call is reported as `tool_calls`, the
            // value the OpenAI formats report for the same situation.
            if reason == "stop" && !tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                reason
            }
        });

    let mut metadata = std::collections::HashMap::new();
    if let Some(fr) = first_candidate
        .and_then(|c| c.get("finishReason"))
        .and_then(|v| v.as_str())
    {
        metadata.insert("finish_reason_raw".to_string(), serde_json::json!(fr));
    }
    if !thought_signatures.is_empty() {
        metadata.insert(
            "thought_signatures".to_string(),
            serde_json::json!(thought_signatures),
        );
    }
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };

    let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

    Ok(LlmResponseType {
        // Prefer the server-assigned id; fall back to a local id.
        id: json
            .get("responseId")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| Some(llm_suite_common::generate_id())),
        model: json
            .get("modelVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        content: if text_content.is_empty() {
            None
        } else {
            Some(text_content)
        },
        message,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        usage,
        finish_reason,
        duration: 0,
        reasoning_content: reasoning_content.clone(),
        reasoning_signature: None,
        reasoning_tokens,
        metadata,
        refusal: None,
        stream_stats: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::gemini_native::contents::convert_messages;
    use crate::codecs::gemini_native::test_support::{msg, request};
    use llm_suite_types::message::MessageRole;

    #[test]
    fn blocked_prompt_surfaces_as_an_error() {
        let body =
            r#"{"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "unsafe"}}"#;
        let err = parse_response(body, &request(vec![], None))
            .expect_err("a blocked prompt must not look like an empty answer");
        assert!(format!("{err}").contains("SAFETY"), "{err}");
    }

    #[test]
    fn thought_signature_travels_back_with_the_function_call() {
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "sig-1"}
            ]}}]
        }"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        let message = result.message.clone();
        assert_eq!(
            message.metadata.as_ref().unwrap()["thought_signatures"],
            serde_json::json!(["sig-1"])
        );

        let contents =
            convert_messages(std::slice::from_ref(&message)).expect("must convert");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["thoughtSignature"], serde_json::json!("sig-1"));
    }

    #[test]
    fn a_thought_signature_on_the_thought_part_is_kept() {
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"text": "planning", "thought": true, "thoughtSignature": "sig-1"},
                {"functionCall": {"name": "f", "args": {}}}
            ]}}]
        }"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(
            result.message.metadata.as_ref().unwrap()["thought_signatures"],
            serde_json::json!(["sig-1"])
        );
    }

    #[test]
    fn parse_response_prefers_response_id_and_normalizes_finish_reason() {
        let body = r#"{
            "responseId": "resp-abc",
            "modelVersion": "gemini-2.5-flash",
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
        }"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(result.id.as_deref(), Some("resp-abc"));
        assert_eq!(result.finish_reason.as_deref(), Some("length"));
        assert_eq!(
            result.metadata.as_ref().unwrap()["finish_reason_raw"],
            serde_json::json!("MAX_TOKENS")
        );
        assert_eq!(result.usage.as_ref().unwrap().total_tokens, 5);
    }

    #[test]
    fn parse_response_routes_thought_bool_into_reasoning() {
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"text": "plan", "thought": true},
                {"text": "answer"}
            ]}}]
        }"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(result.content.as_deref(), Some("answer"));
        assert_eq!(result.reasoning_content.as_deref(), Some("plan"));
    }

    #[test]
    fn parse_response_captures_thought_signature() {
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "sig-1"}
            ]}}]
        }"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(
            result.metadata.as_ref().unwrap()["thought_signatures"],
            serde_json::json!(["sig-1"])
        );
    }

    #[test]
    fn the_whole_safety_family_reports_the_content_filter() {
        for raw in [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "NO_IMAGE",
            "IMAGE_RECITATION",
            "IMAGE_OTHER",
        ] {
            let body = format!(
                r#"{{"candidates":[{{"content":{{"parts":[{{"text":"x"}}]}},"finishReason":"{raw}"}}]}}"#
            );
            let result = parse_response(&body, &request(vec![], None)).expect("must parse");
            assert_eq!(
                result.finish_reason.as_deref(),
                Some("content_filter"),
                "{raw} must report the content filter"
            );
        }

        // The other official reasons keep their own meaning.
        let body =
            r#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"MAX_TOKENS"}]}"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(result.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn the_provider_call_id_is_kept_and_echoed_back() {
        let body = r#"{
            "candidates": [{"content": {"parts": [
                {"functionCall": {"id": "fc-1", "name": "get_weather", "args": {}}}
            ]}}]
        }"#;
        let result = parse_response(body, &request(vec![], None)).expect("must parse");
        let calls = result.tool_calls.expect("tool calls present");
        assert_eq!(calls[0].id, "fc-1", "the provider call id must survive");
        assert_eq!(
            calls[0].call_id.as_deref(),
            Some("fc-1"),
            "the provider call id is the correlation id"
        );

        let mut tool_result = msg(MessageRole::Tool, "sunny");
        tool_result.tool_call_id = Some("fc-1".to_string());
        let contents =
            convert_messages(&[result.message.clone(), tool_result]).expect("must convert");
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["id"],
            serde_json::json!("fc-1")
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather")
        );
    }
}
