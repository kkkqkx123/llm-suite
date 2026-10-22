//! Parsing of one Anthropic Messages SSE chunk into stream events.

use crate::error::LlmResult;
use llm_suite_types::llm::{
    MessageStreamEnd, MessageStreamError, MessageStreamEvent, MessageStreamInputJson,
    MessageStreamNativeBlock, MessageStreamReasoning, MessageStreamRedactedThinking,
    MessageStreamRefusal, MessageStreamStart, MessageStreamText, MessageStreamToolCallDelta,
};

use super::usage::stream_usage;

/// Parse one Anthropic SSE chunk into zero or more stream events.
///
/// One wire event can map to several internal events: `message_delta`
/// carries both cumulative usage and the final stop reason, so it emits
/// `Usage` followed by `End` (the accumulator tolerates the duplicate
/// `End` from the subsequent `message_stop`).
pub fn parse_stream_events(data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let json: serde_json::Value = serde_json::from_str(data)?;
    let event_type = json.get("type").and_then(|v| v.as_str());

    match event_type {
        Some("message_start") => {
            let mut events = Vec::new();
            // Surface the provider message identity first so the
            // accumulator records it before any terminal event.
            let stream_id = json
                .pointer("/message/id")
                .and_then(|v| v.as_str())
                .map(String::from);
            let stream_model = json
                .pointer("/message/model")
                .and_then(|v| v.as_str())
                .map(String::from);
            if stream_id.is_some() || stream_model.is_some() {
                events.push(MessageStreamEvent::Start(MessageStreamStart {
                    id: stream_id,
                    model: stream_model,
                }));
            }
            // Initial usage: `message_delta` may omit `input_tokens`
            // (MessageDeltaUsage.input_tokens is optional), so it must be
            // captured here or stream-side prompt tokens are lost.
            if let Some(usage) = json.pointer("/message/usage") {
                if let Some(event) = stream_usage(usage) {
                    events.push(MessageStreamEvent::Usage(event));
                }
            }
            Ok(events)
        }
        Some("content_block_start") => {
            if let Some(b) = json.get("content_block") {
                match b.get("type").and_then(|v| v.as_str()) {
                    Some("tool_use") => {
                        let index = json.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let id = b.get("id").and_then(|v| v.as_str()).map(String::from);
                        let name = b.get("name").and_then(|v| v.as_str()).map(String::from);
                        return Ok(vec![MessageStreamEvent::ToolCallDelta(
                            MessageStreamToolCallDelta {
                                index,
                                id,
                                call_id: None,
                                name,
                                arguments: None,
                                thought_signature: None,
                                is_snapshot: false,
                            },
                        )]);
                    }
                    // The redacted payload only appears in the start
                    // frame; no delta frame carries it. It has to reach
                    // the consumer so the reply can be replayed in the
                    // next turn.
                    Some("redacted_thinking") => {
                        if let Some(data) = b.get("data").and_then(|v| v.as_str()) {
                            return Ok(vec![MessageStreamEvent::RedactedThinking(
                                MessageStreamRedactedThinking {
                                    data: data.to_string(),
                                },
                            )]);
                        }
                    }
                    // A thinking block may open with the first part of
                    // its text; the signature arrives later, in a
                    // `signature_delta`.
                    Some("thinking") => {
                        let thinking = b.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                        if !thinking.is_empty() {
                            return Ok(vec![MessageStreamEvent::ReasoningText(
                                MessageStreamReasoning {
                                    reasoning: thinking.to_string(),
                                    snapshot: String::new(),
                                    signature: None,
                                    signature_snapshot: None,
                                },
                            )]);
                        }
                    }
                    // The text of a block arrives in the delta frames; the
                    // start frame only announces it.
                    Some("text") => {}
                    Some(other) => {
                        // A block the codec does not model (a server tool
                        // result, a container upload, a search result)
                        // travels in the provider's own shape so the
                        // reply can be replayed in the next turn.
                        return Ok(vec![MessageStreamEvent::NativeBlock(
                            MessageStreamNativeBlock {
                                provider: "anthropic".to_string(),
                                kind: other.to_string(),
                                payload: b.clone(),
                            },
                        )]);
                    }
                    // A block without a type carries no name to replay it
                    // under; there is nothing to keep.
                    None => {}
                }
            }
            Ok(Vec::new())
        }
        Some("content_block_delta") => {
            if let Some(d) = json.get("delta") {
                match d.get("type").and_then(|v| v.as_str()) {
                    Some("text_delta") => {
                        if let Some(text) = d.get("text").and_then(|v| v.as_str()) {
                            // `snapshot` is rebuilt by the stream
                            // accumulator; keep the placeholder empty.
                            return Ok(vec![MessageStreamEvent::Text(MessageStreamText {
                                text: text.to_string(),
                                snapshot: String::new(),
                            })]);
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(reasoning) = d.get("thinking").and_then(|v| v.as_str()) {
                            return Ok(vec![MessageStreamEvent::ReasoningText(
                                MessageStreamReasoning {
                                    reasoning: reasoning.to_string(),
                                    snapshot: String::new(),
                                    signature: None,
                                    signature_snapshot: None,
                                },
                            )]);
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(signature) = d.get("signature").and_then(|v| v.as_str()) {
                            return Ok(vec![MessageStreamEvent::ReasoningText(
                                MessageStreamReasoning {
                                    reasoning: String::new(),
                                    snapshot: String::new(),
                                    signature: Some(signature.to_string()),
                                    signature_snapshot: None,
                                },
                            )]);
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = d.get("partial_json").and_then(|v| v.as_str()) {
                            let index =
                                json.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            // The incremental JSON parser of the stream
                            // accumulator works on this event: it
                            // refreshes the parsed snapshot of the
                            // arguments buffer while it is still
                            // incomplete.
                            return Ok(vec![MessageStreamEvent::InputJson(
                                MessageStreamInputJson {
                                    partial_json: partial.to_string(),
                                    index: Some(index),
                                    parsed_snapshot: None,
                                },
                            )]);
                        }
                    }
                    // The sources of a grounded answer travel on their
                    // own channel so they never mix into the answer text.
                    Some("citations_delta") => {
                        if let Some(citation) = d.get("citation") {
                            return Ok(vec![MessageStreamEvent::Annotation(
                                llm_suite_types::llm::MessageStreamAnnotation {
                                    annotation: citation.clone(),
                                },
                            )]);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Vec::new())
        }
        Some("message_delta") => {
            let mut events = Vec::new();
            if let Some(usage) = json.get("usage") {
                if let Some(event) = stream_usage(usage) {
                    events.push(MessageStreamEvent::Usage(event));
                }
            }
            // A refusal reports the reason of the refusal in the
            // official `stop_details` object; it travels on the refusal
            // channel, like the Chat Completions refusal delta.
            if json.pointer("/delta/stop_reason").and_then(|v| v.as_str()) == Some("refusal") {
                let refusal = json
                    .pointer("/delta/stop_details/explanation")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or_else(|| {
                        json.pointer("/delta/stop_details/category")
                            .and_then(|v| v.as_str())
                            .map(|category| format!("refused: {category}"))
                    });
                if let Some(refusal) = refusal {
                    events.push(MessageStreamEvent::Refusal(MessageStreamRefusal {
                        refusal,
                        snapshot: String::new(),
                    }));
                }
            }
            if let Some(stop_reason) =
                json.pointer("/delta/stop_reason").and_then(|v| v.as_str())
            {
                // Everything except `pause_turn` terminates the turn
                // (the official StopReason set also includes `refusal`
                // and `model_context_window_exceeded`).
                if stop_reason != "pause_turn" {
                    events.push(MessageStreamEvent::End(MessageStreamEnd {}));
                }
            }
            Ok(events)
        }
        Some("message_stop") => Ok(vec![MessageStreamEvent::End(MessageStreamEnd {})]),
        Some("error") => {
            // Mid-stream provider errors must be surfaced, not swallowed.
            let err = &json["error"];
            let err_type = err
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            Ok(vec![MessageStreamEvent::Error(MessageStreamError {
                error: format!("anthropic {err_type}: {message}"),
            })])
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_block_start_tool_use_emits_name_and_id() {
        let chunk = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#;
        match parse_stream_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(delta)) => {
                assert_eq!(delta.index, 1);
                assert_eq!(delta.id.as_deref(), Some("toolu_1"));
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                assert_eq!(delta.arguments, None);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn input_json_delta_emits_arguments_fragment() {
        let chunk = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Beijing\"}"}}"#;
        match parse_stream_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            // The incremental JSON parser of the stream accumulator works on
            // this event.
            Some(MessageStreamEvent::InputJson(input)) => {
                assert_eq!(input.index, Some(1));
                assert_eq!(input.partial_json, r#"{"city":"Beijing"}"#);
            }
            other => panic!("expected InputJson, got {:?}", other),
        }
    }

    #[test]
    fn a_redacted_thinking_block_is_not_dropped() {
        let chunk = r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"c2VjcmV0"}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        match &events[0] {
            MessageStreamEvent::RedactedThinking(redacted) => {
                assert_eq!(redacted.data, "c2VjcmV0");
            }
            other => panic!("expected RedactedThinking, got {other:?}"),
        }
    }

    #[test]
    fn a_server_tool_block_is_not_dropped() {
        let chunk = r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        match &events[0] {
            MessageStreamEvent::NativeBlock(native) => {
                assert_eq!(native.provider, "anthropic");
                assert_eq!(native.kind, "server_tool_use");
                assert_eq!(native.payload["name"], "web_search");
            }
            other => panic!("expected NativeBlock, got {other:?}"),
        }
    }

    #[test]
    fn a_text_block_start_carries_no_block_of_its_own() {
        // The text of a block arrives in the delta frames.
        let chunk = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert!(events.is_empty(), "the text block must not be reported: {events:?}");
    }

    #[test]
    fn message_start_emits_identity_and_initial_usage() {
        let chunk = r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet","usage":{"input_tokens":25,"output_tokens":1}}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert_eq!(events.len(), 2);
        match &events[0] {
            MessageStreamEvent::Start(s) => {
                assert_eq!(s.id.as_deref(), Some("msg_1"));
                assert_eq!(s.model.as_deref(), Some("claude-sonnet"));
            }
            other => panic!("expected Start first, got {other:?}"),
        }
        match &events[1] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.prompt_tokens, 25);
                assert_eq!(u.usage.completion_tokens, 1);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn signature_delta_emits_reasoning_signature_fragment() {
        let chunk = r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-part"}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        match &events[0] {
            MessageStreamEvent::ReasoningText(r) => {
                assert!(r.reasoning.is_empty());
                assert_eq!(r.signature.as_deref(), Some("sig-part"));
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }

    #[test]
    fn message_delta_emits_usage_then_end() {
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":25,"output_tokens":7}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert_eq!(events.len(), 2, "usage and end must both be emitted");
        match &events[0] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.prompt_tokens, 25);
                assert_eq!(u.usage.completion_tokens, 7);
            }
            other => panic!("expected Usage first, got {other:?}"),
        }
        assert!(matches!(events[1], MessageStreamEvent::End(_)));
    }

    #[test]
    fn pause_turn_keeps_stream_open_until_message_stop() {
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":5}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events[0], MessageStreamEvent::Usage(_)),
            "no End on pause_turn"
        );

        let stop = parse_stream_events(r#"{"type":"message_stop"}"#).expect("must parse");
        assert!(matches!(stop[0], MessageStreamEvent::End(_)));
    }

    #[test]
    fn refusal_and_context_window_stop_reasons_are_terminal() {
        for reason in ["refusal", "model_context_window_exceeded", "tool_use"] {
            let chunk =
                format!(r#"{{"type":"message_delta","delta":{{"stop_reason":"{reason}"}}}}"#);
            let events = parse_stream_events(&chunk).expect("must parse");
            assert!(
                matches!(events.last(), Some(MessageStreamEvent::End(_))),
                "{reason} must terminate the stream"
            );
        }
    }

    #[test]
    fn stream_message_delta_summarizes_ttl_breakdown() {
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"cache_creation":{"ephemeral_5m_input_tokens":11}}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        match &events[0] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.completion_tokens, 7);
                assert_eq!(u.usage.cache_write_tokens, Some(11));
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn error_event_is_surfaced() {
        let chunk =
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        match &events[0] {
            MessageStreamEvent::Error(e) => {
                assert!(e.error.contains("overloaded_error"));
            }
            other => panic!("expected Error event, got {other:?}"),
        }
    }

    #[test]
    fn the_stream_reports_the_sources_of_a_grounded_answer() {
        let chunk = r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"x","document_index":0}}}"#;
        match parse_stream_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::Annotation(annotation)) => {
                assert_eq!(
                    annotation.annotation["type"],
                    serde_json::json!("char_location")
                );
            }
            other => panic!("expected Annotation, got {other:?}"),
        }
    }

    #[test]
    fn the_stream_reports_a_refusal() {
        let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","explanation":"policy"}},"usage":{"output_tokens":3}}"#;
        let events = parse_stream_events(chunk).expect("must parse");
        let refusal = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::Refusal(r) => Some(r.refusal.clone()),
                _ => None,
            })
            .expect("the refusal must be reported");
        assert_eq!(refusal, "policy");
    }

    #[test]
    fn the_thinking_block_start_text_is_not_dropped() {
        let chunk = r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"Let me verify"}}"#;
        match parse_stream_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ReasoningText(reasoning)) => {
                assert_eq!(reasoning.reasoning, "Let me verify");
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }
}
