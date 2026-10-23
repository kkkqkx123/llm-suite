//! Parsing of one OpenAI Responses SSE chunk into stream events.

use crate::error::LlmResult;
use llm_suite_types::llm::{MessageStreamEvent, MessageStreamStart};

use super::usage::parse_responses_usage;

/// Parse one Responses SSE chunk into zero or more stream events.
///
/// The terminal event carries the final usage and closes the turn, so a
/// chunk can produce both a usage and a terminal event. The usage event
/// comes first: the stream accumulator builds the final message when the
/// terminal event arrives.
pub fn parse_stream_events(data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    // The Responses SSE stream is terminated by a `[DONE]` sentinel that
    // is not JSON; it closes the turn like the Chat Completions stream
    // does.
    if data.trim() == "[DONE]" {
        return Ok(vec![MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        )]);
    }

    let json: serde_json::Value = serde_json::from_str(data)?;
    let event_type = json.get("type").and_then(|v| v.as_str());
    let mut events = Vec::new();

    match event_type {
        Some("response.created") => {
            // Stream identity; the accumulator records it for the final
            // message and forwards it once.
            let response = json.get("response");
            let id = response
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
                .map(String::from);
            let model = response
                .and_then(|r| r.get("model"))
                .and_then(|v| v.as_str())
                .map(String::from);
            if id.is_some() || model.is_some() {
                events.push(MessageStreamEvent::Start(MessageStreamStart { id, model }));
            }
        }
        Some("response.output_item.added") => {
            // The function name and the call identity ride on the output
            // item; the argument events that follow only carry fragments.
            let item = json.get("item");
            if item.and_then(|i| i.get("type")).and_then(|v| v.as_str()) == Some("function_call") {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let id = item
                    .and_then(|i| i.get("id"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let call_id = item
                    .and_then(|i| i.get("call_id"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let name = item
                    .and_then(|i| i.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                events.push(MessageStreamEvent::ToolCallDelta(
                    llm_suite_types::llm::MessageStreamToolCallDelta {
                        index,
                        id,
                        call_id,
                        name,
                        arguments: None,
                        thought_signature: None,
                        is_snapshot: false,
                    },
                ));
            }
        }
        Some("response.output_item.done") => {
            // The finished output item is the authoritative view of a
            // function call: it carries the call id and the complete
            // argument string next to the item id.
            let item = json.get("item");
            if item.and_then(|i| i.get("type")).and_then(|v| v.as_str()) == Some("function_call") {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let id = item
                    .and_then(|i| i.get("id"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let call_id = item
                    .and_then(|i| i.get("call_id"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let name = item
                    .and_then(|i| i.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let arguments = item
                    .and_then(|i| i.get("arguments"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                events.push(MessageStreamEvent::ToolCallDelta(
                    llm_suite_types::llm::MessageStreamToolCallDelta {
                        index,
                        id,
                        call_id,
                        name,
                        arguments,
                        thought_signature: None,
                        is_snapshot: true,
                    },
                ));
            }
        }
        Some("response.output_text.delta") => {
            if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                events.push(MessageStreamEvent::Text(
                    llm_suite_types::llm::MessageStreamText {
                        snapshot: String::new(),
                        text: delta.to_string(),
                    },
                ));
            }
        }
        Some("response.output_text.annotation.added") => {
            // The sources of a grounded answer travel on their own
            // channel so they never mix into the answer text.
            if let Some(annotation) = json.get("annotation") {
                events.push(MessageStreamEvent::Annotation(
                    llm_suite_types::llm::MessageStreamAnnotation {
                        annotation: annotation.clone(),
                    },
                ));
            }
        }
        Some("response.reasoning_text.delta") | Some("response.reasoning_summary_text.delta") => {
            if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                events.push(MessageStreamEvent::ReasoningText(
                    llm_suite_types::llm::MessageStreamReasoning {
                        reasoning: delta.to_string(),
                        snapshot: String::new(),
                        signature: None,
                        signature_snapshot: None,
                    },
                ));
            }
        }
        Some("response.refusal.delta") => {
            if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                events.push(MessageStreamEvent::Refusal(
                    llm_suite_types::llm::MessageStreamRefusal {
                        refusal: delta.to_string(),
                        snapshot: String::new(),
                    },
                ));
            }
        }
        Some("response.function_call_arguments.delta") => {
            // Incremental arguments fragment. The output index is the
            // position of the call in the response output array and is
            // stable for the whole stream, so fragments of one call share
            // it and parallel calls stay apart.
            if let Some(delta) = json.get("delta").and_then(|v| v.as_str()) {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                events.push(MessageStreamEvent::ToolCallDelta(
                    llm_suite_types::llm::MessageStreamToolCallDelta {
                        index,
                        id: None,
                        call_id: None,
                        name: None,
                        arguments: Some(delta.to_string()),
                        thought_signature: None,
                        is_snapshot: false,
                    },
                ));
            }
        }
        Some("response.function_call_arguments.done") => {
            // The event carries the complete argument string and the call
            // identity at its top level; the name was announced by the
            // output item event.
            if let Some(arguments) = json.get("arguments").and_then(|v| v.as_str()) {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let id = json
                    .get("item_id")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                events.push(MessageStreamEvent::ToolCallDelta(
                    llm_suite_types::llm::MessageStreamToolCallDelta {
                        index,
                        id,
                        call_id: None,
                        name: None,
                        arguments: Some(arguments.to_string()),
                        thought_signature: None,
                        is_snapshot: true,
                    },
                ));
            }
        }
        Some("response.failed") => {
            // The failure payload lives in the completed response object:
            // `{"type":"response.failed","response":{...,"error":{...}}}`.
            let message = json
                .pointer("/response/error/message")
                .and_then(|v| v.as_str())
                .unwrap_or("response failed")
                .to_string();
            tracing::warn!("Responses stream ended with a failed response: {message}");
            events.push(MessageStreamEvent::Error(
                llm_suite_types::llm::MessageStreamError { error: message },
            ));
            events.push(MessageStreamEvent::End(
                llm_suite_types::llm::MessageStreamEnd {},
            ));
        }
        Some("error") => {
            // Standalone stream error event: `{type, code, message, param}`.
            let message = json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("stream error")
                .to_string();
            tracing::warn!("Responses stream ended with an error event: {message}");
            events.push(MessageStreamEvent::Error(
                llm_suite_types::llm::MessageStreamError { error: message },
            ));
            events.push(MessageStreamEvent::End(
                llm_suite_types::llm::MessageStreamEnd {},
            ));
        }
        Some("response.completed") | Some("response.incomplete") => {
            // The terminal event carries the final usage summary inside
            // the response object: `{"type":..,"response":{..,"usage":..}}`.
            if let Some(usage) = json.pointer("/response/usage") {
                events.push(MessageStreamEvent::Usage(
                    llm_suite_types::llm::MessageStreamUsage {
                        usage: parse_responses_usage(usage),
                    },
                ));
            }
            events.push(MessageStreamEvent::End(
                llm_suite_types::llm::MessageStreamEnd {},
            ));
        }
        other => {
            tracing::debug!(
                event_type = other.unwrap_or(""),
                "unhandled Responses stream event; ignoring",
            );
        }
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_tool_call_carries_the_provider_call_id() {
        let added = r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#;
        match parse_stream_events(added)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.call_id.as_deref(), Some("call_1"));
                assert_eq!(d.name.as_deref(), Some("get_weather"));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn function_call_arguments_delta_and_done_share_index() {
        let delta = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":"}"#;
        match parse_stream_events(delta)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 0);
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"#));
                assert!(!d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }

        let done = r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"city\":\"Beijing\"}"}"#;
        match parse_stream_events(done)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 0, "fragments must land on the same call");
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
                assert!(d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn output_item_added_announces_the_function_name() {
        let added = r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#;
        match parse_stream_events(added)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(d)) => {
                assert_eq!(d.index, 1);
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.name.as_deref(), Some("get_weather"));
                assert_eq!(d.arguments, None);
                assert!(!d.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn terminal_event_emits_usage_and_ends_the_turn() {
        let completed = r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"output_tokens_details":{"reasoning_tokens":2},"input_tokens_details":{"cached_tokens":3,"cache_write_tokens":4}}}}"#;
        let events = parse_stream_events(completed).expect("completed chunk parses");
        assert_eq!(events.len(), 2, "usage and terminal event");
        match &events[0] {
            MessageStreamEvent::Usage(u) => {
                assert_eq!(u.usage.prompt_tokens, 10);
                assert_eq!(u.usage.completion_tokens, 5);
                assert_eq!(u.usage.total_tokens, 15);
                assert_eq!(u.usage.reasoning_tokens, Some(2));
                assert_eq!(u.usage.cache_read_tokens, Some(3));
                assert_eq!(u.usage.cache_write_tokens, Some(4));
            }
            other => panic!("expected Usage, got {other:?}"),
        }
        assert!(
            matches!(events[1], MessageStreamEvent::End(_)),
            "the turn must end even when usage is present"
        );

        let without_usage = parse_stream_events(r#"{"type":"response.completed"}"#)
            .expect("completed without usage parses");
        assert!(matches!(without_usage[0], MessageStreamEvent::End(_)));
    }

    #[test]
    fn created_event_emits_stream_identity() {
        match parse_stream_events(
            r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-4o"}}"#,
        )
        .expect("created chunk parses")
        .into_iter()
        .next()
        {
            Some(MessageStreamEvent::Start(s)) => {
                assert_eq!(s.id.as_deref(), Some("resp_1"));
                assert_eq!(s.model.as_deref(), Some("gpt-4o"));
            }
            other => panic!("expected Start, got {other:?}"),
        }
    }

    #[test]
    fn multiple_call_deltas_get_distinct_indices() {
        let chunk = |index: usize| {
            format!(
                r#"{{"type":"response.function_call_arguments.delta","item_id":"fc_{index}","output_index":{index},"delta":"x"}}"#
            )
        };
        let i1 = match parse_stream_events(&chunk(0))
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        let i2 = match parse_stream_events(&chunk(1))
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_ne!(i1, i2);
        // The same output index always identifies the same call.
        let again = match parse_stream_events(&chunk(0))
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => d.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_eq!(again, i1);
    }

    #[test]
    fn the_done_sentinel_ends_the_stream() {
        // The official Responses stream closes with a `[DONE]` sentinel that
        // is not JSON.
        let events = parse_stream_events("[DONE]").expect("the sentinel must not be a parse error");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], MessageStreamEvent::End(_)));
    }

    #[test]
    fn the_stream_reports_the_sources_of_a_grounded_answer() {
        let chunk = r#"{"type":"response.output_text.annotation.added","output_index":0,"content_index":0,"item_id":"msg_1","annotation_index":0,"annotation":{"type":"url_citation","url":"https://example.com"}}"#;
        match parse_stream_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::Annotation(annotation)) => {
                assert_eq!(
                    annotation.annotation["url"],
                    serde_json::json!("https://example.com")
                );
            }
            other => panic!("expected Annotation, got {other:?}"),
        }
    }

    #[test]
    fn the_finished_output_item_completes_the_tool_call() {
        let chunk = r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}"#;
        match parse_stream_events(chunk)
            .expect("chunk must parse")
            .into_iter()
            .next()
        {
            Some(MessageStreamEvent::ToolCallDelta(delta)) => {
                assert_eq!(delta.index, 0);
                assert_eq!(delta.id.as_deref(), Some("fc_1"));
                assert_eq!(delta.call_id.as_deref(), Some("call_1"));
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                assert_eq!(delta.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
                assert!(delta.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn stream_text_delta_emits_the_increment() {
        // The running snapshot is owned by the stream accumulator, not by the
        // codec: the codec reports the raw increment only.
        match parse_stream_events(r#"{"type":"response.output_text.delta","delta":"Hello "}"#)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::Text(t) => {
                assert_eq!(t.text, "Hello ");
                assert_eq!(t.snapshot, "");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn stream_reasoning_delta_emits_reasoning_text() {
        match parse_stream_events(r#"{"type":"response.reasoning_text.delta","delta":"thinking"}"#)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ReasoningText(r) => {
                assert_eq!(r.reasoning, "thinking");
                assert_eq!(r.snapshot, "");
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }
    }

    #[test]
    fn stream_refusal_delta_emits_refusal() {
        match parse_stream_events(r#"{"type":"response.refusal.delta","delta":"I cannot."}"#)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::Refusal(r) => assert_eq!(r.refusal, "I cannot."),
            other => panic!("expected Refusal, got {other:?}"),
        }
    }

    #[test]
    fn stream_function_call_arguments_done_emits_snapshot() {
        let delta = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":"}"#;
        parse_stream_events(delta).unwrap();
        let done = r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"city\":\"Beijing\"}"}"#;
        match parse_stream_events(done)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
        {
            MessageStreamEvent::ToolCallDelta(d) => {
                assert!(d.is_snapshot, "done must emit a full snapshot");
                assert_eq!(d.id.as_deref(), Some("fc_1"));
                assert_eq!(d.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }

    #[test]
    fn stream_failed_emits_error_event() {
        let chunk = r#"{"type":"response.failed","response":{"status":"failed","error":{"message":"boom","code":"server_error"}}}"#;
        let events = parse_stream_events(chunk).expect("failed chunk parses");
        assert_eq!(events.len(), 2, "the error and the terminal event");
        match &events[0] {
            MessageStreamEvent::Error(e) => assert_eq!(e.error, "boom"),
            other => panic!("expected Error, got {other:?}"),
        }
        assert!(
            matches!(events[1], MessageStreamEvent::End(_)),
            "a failed turn must end the stream"
        );
    }
}
