//! Parsing of one Gemini native SSE chunk into stream events.

use std::sync::atomic::Ordering;

use crate::error::LlmResult;
use llm_suite_types::llm::MessageStreamEvent;

use super::part::{read_parts, Part};
use super::usage::parse_usage_metadata;
use super::GeminiNativeCodec;

impl GeminiNativeCodec {
    /// Parse one Gemini native SSE chunk into zero or more stream events.
    pub(super) fn parse_stream_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let json: serde_json::Value = serde_json::from_str(data)?;
        let mut events = Vec::new();

        // Streaming usage arrives in the final chunk's `usageMetadata`. It
        // must be surfaced before the terminal event: the accumulator builds
        // the final message when the terminal event arrives.
        if let Some(u) = json.get("usageMetadata") {
            events.push(MessageStreamEvent::Usage(
                llm_suite_types::llm::MessageStreamUsage {
                    usage: parse_usage_metadata(u),
                },
            ));
        }

        if let Some(candidates) = json.get("candidates").and_then(|v| v.as_array()) {
            if let Some(candidate) = candidates.first() {
                // Every part is emitted — a single chunk can pack text,
                // reasoning and function-call parts together.
                if let Some(parts) = candidate
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|v| v.as_array())
                {
                    for part in read_parts(parts) {
                        match part {
                            Part::Thought { text } => {
                                events.push(MessageStreamEvent::ReasoningText(
                                    llm_suite_types::llm::MessageStreamReasoning {
                                        snapshot: String::new(),
                                        reasoning: text,
                                        signature: None,
                                        signature_snapshot: None,
                                    },
                                ));
                            }
                            Part::Text { text } => {
                                events.push(MessageStreamEvent::Text(
                                    llm_suite_types::llm::MessageStreamText {
                                        snapshot: String::new(),
                                        text,
                                    },
                                ));
                            }
                            Part::FunctionCall {
                                name,
                                arguments,
                                provider_id,
                                signature,
                            } => {
                                events.push(MessageStreamEvent::ToolCallDelta(
                                    llm_suite_types::llm::MessageStreamToolCallDelta {
                                        index: self.call_index.fetch_add(1, Ordering::Relaxed),
                                        // The provider-assigned call id is also
                                        // the correlation id: it is the id the
                                        // matching function response has to
                                        // echo back.
                                        call_id: provider_id.clone(),
                                        id: provider_id,
                                        name: Some(name),
                                        arguments: Some(arguments),
                                        thought_signature: signature,
                                        is_snapshot: true,
                                    },
                                ));
                            }
                        }
                    }
                }

                // `finishReason` rides on the final chunk, often together
                // with content parts; it must not be swallowed by the part
                // handling above. It also marks the end of the response, so
                // the tool-call numbering starts over for the next one.
                if candidate.get("finishReason").is_some() {
                    self.call_index.store(0, Ordering::Relaxed);
                    events.push(MessageStreamEvent::End(
                        llm_suite_types::llm::MessageStreamEnd {},
                    ));
                }
            }
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::MessageStreamEvent;

    #[test]
    fn function_call_part_emits_snapshot_delta() {
        let chunk = r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Beijing"}}}]},"finishReason":"STOP"}]}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_events(chunk)
            .expect("chunk must parse");
        assert!(
            events.len() >= 2,
            "tool call part and End must both be emitted, got {:?}",
            events
        );
        match &events[0] {
            MessageStreamEvent::ToolCallDelta(delta) => {
                assert_eq!(delta.name.as_deref(), Some("get_weather"));
                let args: serde_json::Value =
                    serde_json::from_str(delta.arguments.as_deref().unwrap()).unwrap();
                assert_eq!(args["city"], "Beijing");
                assert!(delta.is_snapshot);
            }
            other => panic!("expected ToolCallDelta, got {:?}", other),
        }
        assert!(
            matches!(events.last(), Some(MessageStreamEvent::End(_))),
            "finishReason in the same chunk must still yield End"
        );
    }

    #[test]
    fn thought_bool_parts_stream_as_reasoning() {
        let chunk = r#"{"candidates":[{"content":{"parts":[{"text":"secret plan","thought":true},{"text":"answer"}]}}]}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_events(chunk)
            .expect("must parse");
        assert_eq!(
            events.len(),
            2,
            "both parts must be emitted, got {:?}",
            events
        );
        assert!(matches!(&events[0], MessageStreamEvent::ReasoningText(_)));
        assert!(matches!(&events[1], MessageStreamEvent::Text(_)));
    }

    #[test]
    fn usage_metadata_chunk_emits_usage_event() {
        let chunk = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_events(chunk)
            .expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.prompt_tokens, 10);
                assert_eq!(u.usage.completion_tokens, 5);
                assert_eq!(u.usage.total_tokens, 15);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }
    }

    #[test]
    fn usage_metadata_parses_cached_content_tokens() {
        // Primary source: `cachedContentTokenCount` (python-genai
        // `GenerateContentResponseUsageMetadata.cached_content_token_count`).
        let chunk = r#"{"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"totalTokenCount":105,"thoughtsTokenCount":3,"cachedContentTokenCount":40}}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_events(chunk)
            .expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.reasoning_tokens, Some(3));
                assert_eq!(u.usage.cache_read_tokens, Some(40));
                assert_eq!(u.usage.cache_write_tokens, None);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }

        // Fallback: `cacheTokensDetails` per-modality breakdown when the
        // primary count is absent.
        let chunk = r#"{"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"totalTokenCount":105,"cacheTokensDetails":[{"modality":"TEXT","tokenCount":20},{"modality":"IMAGE","tokenCount":8}]}}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_events(chunk)
            .expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.cache_read_tokens, Some(28));
            }
            other => panic!("expected Usage event, got {:?}", other),
        }

        // Neither present → `None`.
        let chunk = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#;
        let events = GeminiNativeCodec::new()
            .parse_stream_events(chunk)
            .expect("must parse");
        match events.as_slice() {
            [MessageStreamEvent::Usage(u)] => {
                assert_eq!(u.usage.cache_read_tokens, None);
            }
            other => panic!("expected Usage event, got {:?}", other),
        }
    }

    #[test]
    fn the_tool_call_numbering_restarts_with_every_response() {
        let codec = GeminiNativeCodec::new();
        let call_chunk =
            r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}"#;
        let end_chunk = r#"{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}"#;

        let first = codec.parse_stream_events(call_chunk).expect("must parse");
        let index = match &first[0] {
            MessageStreamEvent::ToolCallDelta(delta) => delta.index,
            other => panic!("expected ToolCallDelta, got {other:?}"),
        };
        assert_eq!(index, 0, "the first call of a response is number zero");

        // The response ends; the next response starts the numbering again.
        codec.parse_stream_events(end_chunk).expect("must parse");
        let second = codec.parse_stream_events(call_chunk).expect("must parse");
        match &second[0] {
            MessageStreamEvent::ToolCallDelta(delta) => {
                assert_eq!(
                    delta.index, 0,
                    "the numbering describes the position inside the response"
                );
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
    }
}
