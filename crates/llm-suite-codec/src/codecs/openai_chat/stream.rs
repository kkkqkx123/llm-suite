//! Parsing of one OpenAI Chat Completions streaming SSE chunk.

use crate::error::LlmResult;
use llm_suite_types::llm::MessageStreamEvent;

use super::usage::parse_openai_usage;

/// Parse one OpenAI Chat streaming SSE chunk into zero or more logical events.
///
/// Returns a `Vec` (not a single `Option`) because a chunk's
/// `choices[].delta.tool_calls` array may carry several tool calls at once
/// (parallel tool calls). Each entry yields its own `ToolCallDelta` so none
/// are dropped; the accumulator keys them by `index`.
pub fn parse_openai_stream_chunk(data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
    if data == "[DONE]" {
        return Ok(vec![MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        )]);
    }

    let json: serde_json::Value = serde_json::from_str(data)?;

    let mut events = Vec::new();

    // Provider stream identity rides on every chunk; surface it first so the
    // accumulator records it before any terminal event in the same chunk.
    let stream_id = json.get("id").and_then(|v| v.as_str()).map(String::from);
    let stream_model = json.get("model").and_then(|v| v.as_str()).map(String::from);
    if stream_id.is_some() || stream_model.is_some() {
        events.push(MessageStreamEvent::Start(
            llm_suite_types::llm::MessageStreamStart {
                id: stream_id,
                model: stream_model,
            },
        ));
    }

    let empty_choices = vec![];
    let choices = json
        .get("choices")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_choices);
    let choice = choices.first();

    if let Some(choice) = choice {
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                events.push(MessageStreamEvent::Text(
                    llm_suite_types::llm::MessageStreamText {
                        snapshot: String::new(),
                        text: content.to_string(),
                    },
                ));
            }
            // `reasoning_content` is a non-standard extension (DeepSeek /
            // Chinese gateways), not an official OpenAI Chat field; see the
            // note in `parse_openai_chat_response`.
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                events.push(MessageStreamEvent::ReasoningText(
                    llm_suite_types::llm::MessageStreamReasoning {
                        snapshot: String::new(),
                        reasoning: reasoning.to_string(),
                        signature: None,
                        signature_snapshot: None,
                    },
                ));
            }
            // A refusal is the provider declining to answer; it travels on
            // its own channel instead of mixing into the answer text.
            if let Some(refusal) = delta.get("refusal").and_then(|r| r.as_str()) {
                events.push(MessageStreamEvent::Refusal(
                    llm_suite_types::llm::MessageStreamRefusal {
                        refusal: refusal.to_string(),
                        snapshot: String::new(),
                    },
                ));
            }
            // Emit one ToolCallDelta per tool call so parallel calls packed
            // into a single delta array are all forwarded (previously only the
            // first was kept).
            if let Some(trait_tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for call in trait_tool_calls {
                    let index = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let id = call.get("id").and_then(|v| v.as_str()).map(String::from);
                    let function = call.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let arguments = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    events.push(MessageStreamEvent::ToolCallDelta(
                        llm_suite_types::llm::MessageStreamToolCallDelta {
                            index,
                            id,
                            call_id: None,
                            name,
                            arguments,
                            thought_signature: None,
                            is_snapshot: false,
                        },
                    ));
                }
            }
        }
    }

    // The usage block may ride on any chunk. OpenAI sends it in a chunk
    // without choices, but compatible gateways attach it to the last chunk
    // that also carries the finish reason; reading it only from the
    // choice-less chunk would lose the counters there. It is emitted before
    // the terminal event so the accumulator sees the numbers in time.
    if let Some(usage) = json.get("usage") {
        events.push(MessageStreamEvent::Usage(
            llm_suite_types::llm::MessageStreamUsage {
                usage: parse_openai_usage(usage),
            },
        ));
    }

    if let Some(choice) = choice {
        // Provider-native terminal values passed through untouched; any other
        // value (including a missing one) leaves the stream open. The set is
        // the official `FinishReason` enumeration.
        if let Some(finish_reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            let is_done = matches!(
                finish_reason,
                "stop" | "tool_calls" | "length" | "content_filter" | "function_call"
            );
            if is_done {
                events.push(MessageStreamEvent::End(
                    llm_suite_types::llm::MessageStreamEnd {},
                ));
            }
        }
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_usage_chunk_is_extracted() {
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        match events
            .into_iter()
            .find(|e| matches!(e, MessageStreamEvent::Usage(_)))
        {
            Some(MessageStreamEvent::Usage(usage)) => {
                assert_eq!(usage.usage.prompt_tokens, 10);
                assert_eq!(usage.usage.completion_tokens, 5);
                assert_eq!(usage.usage.total_tokens, 15);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }
    }

    #[test]
    fn stream_usage_next_to_the_choices_is_not_dropped() {
        // Compatible gateways attach the usage to the last chunk that also
        // carries the finish reason; the counters must survive that.
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        let usage = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::Usage(u) => Some(&u.usage),
                _ => None,
            })
            .expect("the usage must not be dropped");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert!(
            matches!(events.last(), Some(MessageStreamEvent::End(_))),
            "the usage must arrive before the terminal event"
        );
    }

    #[test]
    fn stream_start_carries_provider_identity() {
        let chunk = r#"{"id":"chatcmpl-1","model":"gpt-4o","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        match events.into_iter().next() {
            Some(MessageStreamEvent::Start(start)) => {
                assert_eq!(start.id.as_deref(), Some("chatcmpl-1"));
                assert_eq!(start.model.as_deref(), Some("gpt-4o"));
            }
            other => panic!("expected Start first, got {:?}", other),
        }
    }

    #[test]
    fn empty_choices_without_usage_yields_only_start() {
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        assert_eq!(events.len(), 1, "only the Start identity event is emitted");
        assert!(matches!(events[0], MessageStreamEvent::Start(_)));
    }

    #[test]
    fn chunk_without_identity_and_content_is_ignored() {
        let chunk = r#"{"object":"chat.completion.chunk","choices":[]}"#;
        assert!(parse_openai_stream_chunk(chunk)
            .expect("chunk must parse")
            .is_empty());
    }

    #[test]
    fn the_deprecated_function_call_reason_ends_the_stream() {
        let chunk =
            r#"{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        assert!(
            matches!(events.last(), Some(MessageStreamEvent::End(_))),
            "every official finish reason terminates the turn"
        );
    }

    #[test]
    fn stream_tool_call_delta_is_extracted() {
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Bei"}}]}}]}"#;
        match parse_openai_stream_chunk(chunk)
            .expect("chunk must parse")
            .into_iter()
            .find(|e| matches!(e, MessageStreamEvent::ToolCallDelta(_)))
        {
            Some(MessageStreamEvent::ToolCallDelta(delta)) => {
                assert_eq!(delta.index, 0);
                assert_eq!(delta.id.as_deref(), Some("call_1"));
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                assert_eq!(delta.arguments.as_deref(), Some(r#"{"city":"Bei"#));
                assert!(!delta.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
    }

    #[test]
    fn stream_chunk_with_parallel_tool_calls_yields_all_deltas() {
        // A single SSE chunk carrying two tool calls must produce two
        // ToolCallDelta events (previously only the first was kept).
        let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}},{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{}"}}]}}]}"#;
        let events = parse_openai_stream_chunk(chunk).expect("chunk must parse");
        let deltas: Vec<_> = events
            .into_iter()
            .filter_map(|e| match e {
                MessageStreamEvent::ToolCallDelta(d) => Some(d),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 2, "both parallel tool calls must be emitted");
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(deltas[1].index, 1);
        assert_eq!(deltas[1].id.as_deref(), Some("call_2"));
    }
}
