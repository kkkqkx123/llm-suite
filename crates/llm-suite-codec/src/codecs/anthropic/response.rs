//! Parsing of a non-streaming Anthropic Messages response body.

use crate::error::{LlmError, LlmResult};
use llm_suite_types::llm::{LlmRequest, LlmResult as LlmResponseType};
use llm_suite_types::message::{MessageContent, MessageContentValue};

use super::usage::cache_write_tokens;

/// Parse an Anthropic response body into the common response shape.
///
/// In text mode the tool calls carried as plain text are extracted from the
/// answer and attached to the result.
pub fn parse_response(body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
    let mut result = parse_anthropic_response(body)?;
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

pub fn parse_anthropic_response(body: &str) -> LlmResult<LlmResponseType> {
    let json: serde_json::Value = serde_json::from_str(body)?;

    // Error payloads ({"type":"error",...}) must not be parsed into an
    // empty message; surface the provider error instead.
    if json.get("type").and_then(|v| v.as_str()) == Some("error") {
        let err = &json["error"];
        let err_type = err
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let message = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(LlmError::InvalidResponse(format!(
            "anthropic error ({err_type}): {message}"
        )));
    }

    let id = json.get("id").and_then(|v| v.as_str()).map(String::from);
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // Provider-native value passed through untouched (`end_turn` /
    // `tool_use` / `max_tokens` / `stop_sequence` / `pause_turn` /
    // `refusal` / `model_context_window_exceeded`); never normalized here.
    let stop_reason = json
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .map(String::from);

    let content_arr = json.get("content").and_then(|v| v.as_array());
    let mut text_content = String::new();
    let mut thinking_content = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut tool_calls = Vec::new();
    // Content blocks are kept per block so every thinking block keeps its
    // own signature when the message is replayed in a later turn.
    let mut content_blocks: Vec<MessageContent> = Vec::new();
    // Sources of a grounded answer; the official text block carries them
    // on `citations`.
    let mut citations: Vec<serde_json::Value> = Vec::new();

    if let Some(blocks) = content_arr {
        for block in blocks {
            let block_type = block.get("type").and_then(|v| v.as_str());
            match block_type {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        text_content.push_str(text);
                        content_blocks.push(MessageContent::Text {
                            text: text.to_string(),
                        });
                    }
                    if let Some(block_citations) =
                        block.get("citations").and_then(|v| v.as_array())
                    {
                        citations.extend(block_citations.iter().cloned());
                    }
                }
                Some("thinking") => {
                    // Extended thinking output; surfaced through
                    // `reasoning_content` / `message.thinking`. The first
                    // non-empty block signature is retained so the reply
                    // can be replayed in later turns.
                    if let Some(text) = block.get("thinking").and_then(|v| v.as_str()) {
                        thinking_content.push_str(text);
                        let signature = block
                            .get("signature")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        if thinking_signature.is_none() {
                            if let Some(signature) = signature.as_ref() {
                                if !signature.is_empty() {
                                    thinking_signature = Some(signature.clone());
                                }
                            }
                        }
                        content_blocks.push(MessageContent::Thinking {
                            thinking: text.to_string(),
                            signature,
                        });
                    }
                }
                Some("redacted_thinking") => {
                    // The provider redacted the reasoning; the payload is
                    // opaque and has to be sent back verbatim.
                    if let Some(data) = block.get("data").and_then(|v| v.as_str()) {
                        content_blocks.push(MessageContent::RedactedThinking {
                            data: data.to_string(),
                        });
                    }
                }
                Some("tool_use") => {
                    let tc_id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input = block
                        .get("input")
                        .cloned()
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    let arguments = serde_json::to_string(&input).unwrap_or_default();
                    tool_calls.push(llm_suite_types::message::LlmToolCall {
                        id: tc_id,
                        r#type: "function".to_string(),
                        call_id: None,
                        function: llm_suite_types::message::LlmFunctionCall { name, arguments },
                        thought_signature: None,
                    });
                }
                Some(other) => {
                    // A block the codec does not model (a server tool
                    // result, a container upload, a search result) travels
                    // in the provider's own shape so the reply can be
                    // replayed in the next turn.
                    content_blocks.push(MessageContent::Native {
                        provider: "anthropic".to_string(),
                        kind: other.to_string(),
                        payload: block.clone(),
                    });
                }
                // A block without a type carries no name to replay it
                // under; there is nothing to keep.
                None => {}
            }
        }
    }

    let usage = json.get("usage").map(|u| {
        let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let cache_read = u
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let cache_write = cache_write_tokens(u).unwrap_or(0);
        llm_suite_types::llm::TokenUsageStats {
            // Anthropic reports the cached input tokens next to the
            // uncached ones; `prompt_tokens` stays the uncached count so
            // the cache breakdown remains a separate, additive signal.
            prompt_tokens: input,
            completion_tokens: output,
            // The official total input is `input_tokens` plus the cache
            // read and cache creation tokens.
            total_tokens: input + cache_read + cache_write + output,
            // Anthropic-native breakdown (not the OpenAI-shaped
            // `completion_tokens_details.reasoning_tokens`).
            reasoning_tokens: u
                .get("output_tokens_details")
                .and_then(|d| d.get("thinking_tokens"))
                .and_then(|v| v.as_u64())
                .map(|r| r as u32),
            cache_read_tokens: u
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
                .map(|r| r as u32),
            cache_write_tokens: cache_write_tokens(u),
            prompt_tokens_cost: None,
            completion_tokens_cost: None,
            total_cost: None,
        }
    });

    let message = llm_suite_types::message::Message {
        id: llm_suite_types::Id::new(),
        role: llm_suite_types::message::MessageRole::Assistant,
        // Thinking blocks carry a per-block signature and have to survive
        // into the next turn, so the message keeps the block structure
        // whenever the reply contains one.
        content: if content_blocks
            .iter()
            .any(|block| !matches!(block, MessageContent::Text { .. }))
        {
            MessageContentValue::Rich(content_blocks)
        } else {
            MessageContentValue::Text(text_content.clone())
        },
        timestamp: llm_suite_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls.clone())
        },
        thinking: if thinking_content.is_empty() {
            None
        } else {
            Some(thinking_content.clone())
        },
        thinking_signature: thinking_signature.clone(),
        metadata: None,
    };

    let mut metadata = std::collections::HashMap::new();
    if let Some(t) = json.get("type").and_then(|v| v.as_str()) {
        metadata.insert("type".to_string(), serde_json::json!(t));
    }
    if let Some(sr) = &stop_reason {
        metadata.insert("stop_reason".to_string(), serde_json::json!(sr));
    }
    if let Some(seq) = json.get("stop_sequence").and_then(|v| v.as_str()) {
        metadata.insert("stop_sequence".to_string(), serde_json::json!(seq));
    }
    if !citations.is_empty() {
        metadata.insert(
            llm_suite_types::llm::ANNOTATIONS_KEY.to_string(),
            serde_json::json!(citations),
        );
    }
    // A refusal carries the reason of the refusal in the official
    // `stop_details` object; it is the Anthropic counterpart of the
    // Chat Completions `message.refusal` field.
    let stop_details = json.get("stop_details").filter(|v| !v.is_null());
    if let Some(details) = stop_details {
        metadata.insert("stop_details".to_string(), details.clone());
    }
    let refusal = match stop_reason.as_deref() {
        Some("refusal") => stop_details
            .and_then(|d| d.get("explanation"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                // The refusal has no explanation; the category names the
                // policy that triggered it.
                stop_details
                    .and_then(|d| d.get("category"))
                    .and_then(|v| v.as_str())
                    .map(|category| format!("refused: {category}"))
            }),
        _ => None,
    };
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };

    let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

    Ok(LlmResponseType {
        id,
        model,
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
        finish_reason: stop_reason,
        duration: 0,
        reasoning_content: if thinking_content.is_empty() {
            None
        } else {
            Some(thinking_content)
        },
        reasoning_signature: thinking_signature,
        reasoning_tokens,
        metadata,
        refusal,
        stream_stats: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::anthropic::messages::convert_messages;
    use crate::codecs::anthropic::test_support::{msg, request};
    use llm_suite_types::message::MessageRole;

    #[test]
    fn citations_are_reported_as_metadata() {
        let body = r#"{"id":"m","model":"claude","stop_reason":"end_turn","content":[{"type":"text","text":"grounded","citations":[{"type":"char_location","cited_text":"x","document_index":0,"start_char_index":0,"end_char_index":1}]}]}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        let metadata = out.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["annotations"][0]["type"],
            serde_json::json!("char_location")
        );
    }

    #[test]
    fn parse_response_extracts_thinking_and_native_reasoning_tokens() {
        let body = r#"{
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "Let me verify", "signature": "sig"},
                {"type": "text", "text": "42"}
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "output_tokens_details": {"thinking_tokens": 8},
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2
            }
        }"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(out.content.as_deref(), Some("42"));
        assert_eq!(out.reasoning_content.as_deref(), Some("Let me verify"));
        assert_eq!(out.message.thinking.as_deref(), Some("Let me verify"));
        assert_eq!(out.reasoning_tokens, Some(8));
        let usage = out.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(
            usage.total_tokens, 37,
            "the official total counts the cached input tokens"
        );
        assert_eq!(usage.cache_read_tokens, Some(5));
        assert_eq!(usage.cache_write_tokens, Some(2));
    }

    #[test]
    fn the_matched_stop_sequence_is_reported_as_metadata() {
        let body = r#"{"id":"m","model":"claude","stop_reason":"stop_sequence","stop_sequence":"END","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(
            out.metadata.expect("metadata must be present")["stop_sequence"],
            serde_json::json!("END")
        );
    }

    #[test]
    fn error_payload_is_surfaced_as_invalid_response() {
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let err =
            parse_response(body, &request(vec![], None)).expect_err("error payload must not parse into a message");
        match err {
            LlmError::InvalidResponse(msg) => {
                assert!(msg.contains("overloaded_error"));
                assert!(msg.contains("Overloaded"));
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }

    #[test]
    fn cache_creation_ttl_breakdown_is_summed() {
        // Official `Usage.cache_creation` object without the legacy total.
        let body = r#"{"id":"m","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":4,"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":3}}}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        let usage = out.usage.expect("usage present");
        assert_eq!(usage.cache_read_tokens, Some(4));
        assert_eq!(
            usage.cache_write_tokens,
            Some(5),
            "5m + 1h breakdown summed"
        );

        // The explicit legacy total wins when both shapes are present.
        let body = r#"{"id":"m","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":7,"cache_creation":{"ephemeral_5m_input_tokens":2,"ephemeral_1h_input_tokens":3}}}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(
            out.usage.unwrap().cache_write_tokens,
            Some(7),
            "explicit cache_creation_input_tokens takes precedence"
        );

        // Neither shape present → `None`.
        let body = r#"{"id":"m","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":5}}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(out.usage.unwrap().cache_write_tokens, None);
    }

    #[test]
    fn a_refusal_is_reported_on_the_refusal_channel() {
        let body = r#"{"id":"m","model":"claude","stop_reason":"refusal","content":[{"type":"text","text":"I cannot help with that."}],"stop_details":{"type":"refusal","category":"general_harms","explanation":"policy"}}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(out.refusal.as_deref(), Some("policy"));
        let metadata = out.metadata.expect("metadata must be present");
        assert_eq!(
            metadata["stop_details"]["category"],
            serde_json::json!("general_harms")
        );
    }

    #[test]
    fn a_refusal_without_an_explanation_names_the_category() {
        let body = r#"{"id":"m","model":"claude","stop_reason":"refusal","content":[],"stop_details":{"type":"refusal","category":"cyber"}}"#;
        let out = parse_response(body, &request(vec![], None)).expect("must parse");
        assert_eq!(out.refusal.as_deref(), Some("refused: cyber"));
    }

    #[test]
    fn response_signature_is_retained_and_replayed() {
        let body = r#"{"id":"msg_2","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"Let me verify","signature":"sig-abc"},{"type":"text","text":"done"}],"usage":{"input_tokens":3,"output_tokens":5}}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = parse_response(body, &request).expect("must parse");
        assert_eq!(out.message.thinking.as_deref(), Some("Let me verify"));
        assert_eq!(out.message.thinking_signature.as_deref(), Some("sig-abc"));
        assert_eq!(out.reasoning_signature.as_deref(), Some("sig-abc"));

        let wire = convert_messages(std::slice::from_ref(&out.message)).expect("must convert");
        let thinking = wire[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "thinking")
            .expect("signed thinking must be replayed");
        assert_eq!(thinking["signature"], "sig-abc");
        assert_eq!(thinking["thinking"], "Let me verify");

        // A thinking block without a signature is never replayed.
        let mut unsigned = out.message.clone();
        unsigned.content = MessageContentValue::Rich(vec![MessageContent::Thinking {
            thinking: "Let me verify".to_string(),
            signature: None,
        }]);
        let wire = convert_messages(std::slice::from_ref(&unsigned)).expect("must convert");
        let replayed_thinking = wire.iter().any(|message| {
            message["content"]
                .as_array()
                .map(|blocks| blocks.iter().any(|block| block["type"] == "thinking"))
                .unwrap_or(false)
        });
        assert!(!replayed_thinking, "unsigned thinking must not be replayed");
    }

    #[test]
    fn every_thinking_block_keeps_its_own_signature() {
        let body = r#"{"id":"msg_3","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"first","signature":"sig-1"},{"type":"thinking","thinking":"second","signature":"sig-2"},{"type":"text","text":"done"}]}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = parse_response(body, &request).expect("must parse");

        let wire = convert_messages(std::slice::from_ref(&out.message)).expect("must convert");
        let blocks = wire[0]["content"].as_array().unwrap();
        let signatures: Vec<&str> = blocks
            .iter()
            .filter(|b| b["type"] == "thinking")
            .map(|b| b["signature"].as_str().unwrap())
            .collect();
        assert_eq!(signatures, vec!["sig-1", "sig-2"]);
    }

    #[test]
    fn redacted_thinking_is_kept_and_replayed() {
        let body = r#"{"id":"msg_4","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"c2VjcmV0"}]}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = parse_response(body, &request).expect("must parse");

        let wire = convert_messages(std::slice::from_ref(&out.message)).expect("must convert");
        let blocks = wire[0]["content"].as_array().unwrap();
        let redacted = blocks
            .iter()
            .find(|b| b["type"] == "redacted_thinking")
            .expect("redacted thinking must be replayed");
        assert_eq!(redacted["data"], "c2VjcmV0");
    }

    #[test]
    fn a_server_tool_block_is_kept_and_replayed() {
        // The server tool blocks are the provider's own; they travel in
        // their own shape so the search result does not vanish.
        let body = r#"{"id":"msg_5","model":"claude-sonnet","stop_reason":"end_turn","content":[{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}},{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://example.com","title":"Example","encrypted_content":"enc","page_age":"one day"}]}]}"#;
        let request = request(vec![msg(MessageRole::User, "hi")], None);
        let out = parse_response(body, &request).expect("must parse");

        let MessageContentValue::Rich(blocks) = &out.message.content else {
            panic!("the server tool blocks must be kept as rich content");
        };
        let native: Vec<&MessageContent> = blocks
            .iter()
            .filter(|block| matches!(block, MessageContent::Native { .. }))
            .collect();
        assert_eq!(
            native.len(),
            2,
            "both server tool blocks must be kept: {blocks:?}"
        );

        // The blocks are replayed verbatim in the next turn.
        let wire = convert_messages(std::slice::from_ref(&out.message)).expect("must convert");
        let replayed = wire[0]["content"].as_array().unwrap();
        let server_tool = replayed
            .iter()
            .find(|b| b["type"] == "server_tool_use")
            .expect("the server tool use must be replayed");
        assert_eq!(server_tool["name"], "web_search");
        let search_result = replayed
            .iter()
            .find(|b| b["type"] == "web_search_tool_result")
            .expect("the search result must be replayed");
        assert_eq!(
            search_result["content"][0]["title"],
            "Example",
            "the result travels verbatim"
        );
    }
}
