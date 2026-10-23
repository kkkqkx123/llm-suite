//! Streaming: the public stream contract, the SSE transport and the
//! accumulator that turns stream events into a final message.

use async_trait::async_trait;
use llm_suite_codec::error::LlmError;
use llm_suite_types::llm::MessageStreamEvent;

pub(crate) mod accumulator;
pub(crate) mod final_message;
pub(crate) mod sse;
pub(crate) mod tool_call;
pub(crate) mod usage;

pub use accumulator::{Accumulated, MessageAccumulator};
pub use sse::SseMessageStream;

#[async_trait]
pub trait MessageStream: Send {
    async fn next(&mut self) -> Option<Result<MessageStreamEvent, LlmError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use eventsource_stream::EventStream;
    use llm_suite_codec::LlmCodec;
    use llm_suite_types::llm::{
        MessageStreamReasoning, MessageStreamToolCallDelta, MessageStreamUsage,
    };
    use llm_suite_types::message::{MessageContent, MessageContentValue};

    use crate::stream::accumulator::Accumulated;

    /// Feed a real OpenAI-style SSE byte stream through `SseMessageStream` with
    /// the real `OpenaiChatCodec`, proving that the `FinalMessage` is
    /// assembled and delivered before the `End` event (regression test for the
    /// `End` short-circuit that skipped the accumulator).
    #[tokio::test]
    async fn real_sse_path_emits_final_message_before_end() {
        use futures::stream;

        let sse = concat!(
            "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Bei\"}}]}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"jing\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );

        let inner = stream::iter(vec![Ok::<_, std::io::Error>(sse.as_bytes().to_vec())]);
        let sse_stream = EventStream::new(inner);
        let codec: Arc<dyn LlmCodec> = Arc::new(llm_suite_codec::OpenaiChatCodec::new());
        let mut stream = SseMessageStream::new(sse_stream, codec, None, None);

        let mut events: Vec<MessageStreamEvent> = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.expect("no stream error"));
        }

        let text: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                MessageStreamEvent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, vec!["Hello", " world"]);

        let final_msg = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::FinalMessage(f) => Some(f),
                _ => None,
            })
            .expect("FinalMessage must be emitted");
        assert_eq!(
            llm_suite_message::helper::extract_text_content(&final_msg.message),
            "Hello world"
        );
        let calls = final_msg
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls must be assembled");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);
        assert_eq!(final_msg.usage.as_ref().unwrap().total_tokens, 15);

        let final_index = events
            .iter()
            .position(|e| matches!(e, MessageStreamEvent::FinalMessage(_)))
            .unwrap();
        let end_index = events
            .iter()
            .position(|e| matches!(e, MessageStreamEvent::End(_)))
            .unwrap();
        assert!(final_index < end_index, "FinalMessage must precede End");
        assert!(matches!(events.last(), Some(MessageStreamEvent::End(_))));
    }

    #[test]
    fn openai_style_deltas_accumulate_into_full_tool_call() {
        let mut acc = MessageAccumulator::new(None);

        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                call_id: None,
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Bei"#.to_string()),
                thought_signature: None,
                is_snapshot: false,
            },
        ));
        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 0,
                id: None,
                call_id: None,
                name: None,
                arguments: Some(r#"jing"}"#.to_string()),
                thought_signature: None,
                is_snapshot: false,
            },
        ));
        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 1,
                id: Some("call_2".to_string()),
                call_id: None,
                name: Some("get_time".to_string()),
                arguments: Some("{}".to_string()),
                thought_signature: None,
                is_snapshot: false,
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        let calls = final_msg.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn snapshot_delta_replaces_fragments() {
        let mut acc = MessageAccumulator::new(None);

        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 0,
                id: Some("fc_1".to_string()),
                call_id: None,
                name: None,
                arguments: Some(r#"{"a":"#.to_string()),
                thought_signature: None,
                is_snapshot: false,
            },
        ));
        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 0,
                id: Some("fc_1".to_string()),
                call_id: None,
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
                thought_signature: None,
                is_snapshot: true,
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        let calls = final_msg.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn tool_call_signatures_reach_the_final_message() {
        let mut acc = MessageAccumulator::new(None);

        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                call_id: None,
                name: Some("get_weather".to_string()),
                arguments: Some("{}".to_string()),
                thought_signature: Some("sig-1".to_string()),
                is_snapshot: true,
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        let calls = final_msg.message.tool_calls.expect("tool calls present");
        assert_eq!(
            calls[0].thought_signature.as_deref(),
            Some("sig-1"),
            "the provider signature rides on the call itself"
        );
    }

    #[test]
    fn text_snapshot_and_usage_merge_on_final_message() {
        let mut acc = MessageAccumulator::new(None);

        acc.push(MessageStreamEvent::Text(
            llm_suite_types::llm::MessageStreamText {
                text: "hello".to_string(),
                snapshot: String::new(),
            },
        ));
        acc.push(MessageStreamEvent::Text(
            llm_suite_types::llm::MessageStreamText {
                text: " world".to_string(),
                snapshot: String::new(),
            },
        ));
        acc.push(MessageStreamEvent::Usage(MessageStreamUsage {
            usage: llm_suite_types::llm::TokenUsageStats {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                reasoning_tokens: None,
                cache_read_tokens: None,
                cache_write_tokens: None,
                prompt_tokens_cost: None,
                completion_tokens_cost: None,
                total_cost: None,
            },
        }));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        assert_eq!(
            final_msg.message.content,
            MessageContentValue::Text("hello world".to_string())
        );
        let usage = final_msg.usage.unwrap();
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn usage_merge_combines_split_anthropic_usage_events() {
        let mut acc = MessageAccumulator::new(None);

        // Anthropic `message_start`: input tokens (output not yet known).
        acc.push(MessageStreamEvent::Usage(MessageStreamUsage {
            usage: llm_suite_types::llm::TokenUsageStats {
                prompt_tokens: 25,
                completion_tokens: 0,
                total_tokens: 0,
                reasoning_tokens: None,
                cache_read_tokens: Some(3),
                cache_write_tokens: None,
                prompt_tokens_cost: None,
                completion_tokens_cost: None,
                total_cost: None,
            },
        }));
        // Anthropic `message_delta`: cumulative output, input may be omitted.
        acc.push(MessageStreamEvent::Usage(MessageStreamUsage {
            usage: llm_suite_types::llm::TokenUsageStats {
                prompt_tokens: 0,
                completion_tokens: 7,
                total_tokens: 0,
                reasoning_tokens: Some(2),
                cache_read_tokens: None,
                cache_write_tokens: None,
                prompt_tokens_cost: None,
                completion_tokens_cost: None,
                total_cost: None,
            },
        }));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        let usage = final_msg.usage.expect("usage must be merged");
        assert_eq!(
            usage.prompt_tokens, 25,
            "input from message_start must survive"
        );
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(
            usage.total_tokens, 35,
            "the official Anthropic total counts the cached input tokens"
        );
        assert_eq!(usage.cache_read_tokens, Some(3));
        assert_eq!(usage.reasoning_tokens, Some(2));
    }

    #[test]
    fn the_sources_of_a_grounded_answer_land_on_the_final_message() {
        use llm_suite_types::llm::MessageStreamAnnotation;

        let mut acc = MessageAccumulator::new(None);
        acc.push(MessageStreamEvent::Annotation(MessageStreamAnnotation {
            annotation: serde_json::json!({"type": "url_citation", "url": "https://a.example"}),
        }));
        acc.push(MessageStreamEvent::Annotation(MessageStreamAnnotation {
            annotation: serde_json::json!({"type": "url_citation", "url": "https://b.example"}),
        }));
        acc.push(MessageStreamEvent::Text(
            llm_suite_types::llm::MessageStreamText {
                text: "grounded".to_string(),
                snapshot: String::new(),
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        let metadata = final_msg
            .message
            .metadata
            .expect("the sources must reach the final message");
        let annotations = metadata["annotations"]
            .as_array()
            .expect("the sources must be a list");
        assert_eq!(annotations.len(), 2);
        assert_eq!(
            annotations[0]["url"],
            serde_json::json!("https://a.example")
        );
    }

    #[test]
    fn start_event_is_forwarded_once_and_lands_on_final_message() {
        use llm_suite_types::llm::MessageStreamStart;

        let mut acc = MessageAccumulator::new(None);

        let first = acc.push(MessageStreamEvent::Start(MessageStreamStart {
            id: Some("msg_1".to_string()),
            model: Some("claude-sonnet".to_string()),
        }));
        let Accumulated::Forwarded(MessageStreamEvent::Start(s)) = first else {
            panic!("first Start must be forwarded");
        };
        assert_eq!(s.id.as_deref(), Some("msg_1"));
        assert_eq!(s.model.as_deref(), Some("claude-sonnet"));

        let second = acc.push(MessageStreamEvent::Start(MessageStreamStart {
            id: Some("msg_2".to_string()),
            model: None,
        }));
        assert!(
            matches!(second, Accumulated::Consumed),
            "repeat Start events must be swallowed"
        );

        acc.push(MessageStreamEvent::Text(
            llm_suite_types::llm::MessageStreamText {
                text: "hi".to_string(),
                snapshot: String::new(),
            },
        ));
        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        assert_eq!(final_msg.id.as_deref(), Some("msg_1"), "first id wins");
        assert_eq!(final_msg.model.as_deref(), Some("claude-sonnet"));
    }

    #[test]
    fn reasoning_signature_accumulates_into_final_message() {
        let mut acc = MessageAccumulator::new(None);

        acc.push(MessageStreamEvent::ReasoningText(MessageStreamReasoning {
            reasoning: "let me think".to_string(),
            snapshot: String::new(),
            signature: Some("sig-a".to_string()),
            signature_snapshot: None,
        }));
        let emitted = acc.push(MessageStreamEvent::ReasoningText(MessageStreamReasoning {
            reasoning: " further".to_string(),
            snapshot: String::new(),
            signature: Some("sig-b".to_string()),
            signature_snapshot: None,
        }));
        let Accumulated::Forwarded(MessageStreamEvent::ReasoningText(r)) = emitted else {
            panic!("reasoning must be forwarded");
        };
        assert_eq!(r.snapshot, "let me think further");
        assert_eq!(
            r.signature_snapshot.as_deref(),
            Some("sig-asig-b"),
            "signature fragments must accumulate"
        );

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        assert_eq!(
            final_msg.message.thinking.as_deref(),
            Some("let me think further")
        );
        assert_eq!(
            final_msg.message.thinking_signature.as_deref(),
            Some("sig-asig-b")
        );
    }

    #[test]
    fn input_json_fragments_build_the_tool_call_and_its_parsed_snapshot() {
        use llm_suite_types::llm::MessageStreamInputJson;

        let mut acc = MessageAccumulator::new(None);

        // The call is announced first, the arguments follow in fragments.
        acc.push(MessageStreamEvent::ToolCallDelta(
            MessageStreamToolCallDelta {
                index: 0,
                id: Some("toolu_1".to_string()),
                call_id: None,
                name: Some("get_weather".to_string()),
                arguments: None,
                thought_signature: None,
                is_snapshot: false,
            },
        ));
        let emitted = acc.push(MessageStreamEvent::InputJson(MessageStreamInputJson {
            partial_json: "{\"city\":\"Bei".to_string(),
            index: Some(0),
            parsed_snapshot: None,
        }));
        let Accumulated::Forwarded(MessageStreamEvent::InputJson(input)) = emitted else {
            panic!("the fragment must be forwarded");
        };
        assert_eq!(input.index, Some(0));
        assert_eq!(input.partial_json, "{\"city\":\"Bei");
        assert_eq!(
            input.parsed_snapshot.as_ref().and_then(|v| v.get("city")),
            Some(&serde_json::json!("Bei")),
            "the incomplete arguments are parsed on the fly"
        );

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        let calls = final_msg.message.tool_calls.expect("tool calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Bei");
    }

    #[test]
    fn a_redacted_thinking_payload_reaches_the_final_message() {
        use llm_suite_types::llm::MessageStreamRedactedThinking;

        let mut acc = MessageAccumulator::new(None);
        acc.push(MessageStreamEvent::RedactedThinking(
            MessageStreamRedactedThinking {
                data: "c2VjcmV0".to_string(),
            },
        ));
        acc.push(MessageStreamEvent::Text(
            llm_suite_types::llm::MessageStreamText {
                text: "answer".to_string(),
                snapshot: String::new(),
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_msg)) = final_event
        else {
            panic!("expected FinalMessage");
        };
        match &final_msg.message.content {
            MessageContentValue::Rich(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(
                    &blocks[0],
                    MessageContent::RedactedThinking { data } if data == "c2VjcmV0"
                ));
                assert!(matches!(
                    &blocks[1],
                    MessageContent::Text { text } if text == "answer"
                ));
            }
            other => panic!("expected rich content, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_anthropic_stream_assembles_the_tool_call_from_the_json_fragments() {
        use futures::stream;

        let sse = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet\",\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Bei\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"jing\\\"}\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let inner = stream::iter(vec![Ok::<_, std::io::Error>(sse.as_bytes().to_vec())]);
        let sse_stream = EventStream::new(inner);
        let codec: Arc<dyn LlmCodec> = Arc::new(llm_suite_codec::AnthropicCodec::new());
        let mut stream = SseMessageStream::new(sse_stream, codec, None, None);

        let mut events: Vec<MessageStreamEvent> = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.expect("no stream error"));
        }

        let final_msg = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::FinalMessage(f) => Some(f),
                _ => None,
            })
            .expect("FinalMessage must be emitted");
        let calls = final_msg
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls must be assembled");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);

        // The incremental parser reports the arguments while they are still
        // incomplete.
        let fragments: Vec<&MessageStreamEvent> = events
            .iter()
            .filter(|e| matches!(e, MessageStreamEvent::InputJson(_)))
            .collect();
        assert_eq!(fragments.len(), 2, "both fragments must be forwarded");
        assert!(matches!(events.last(), Some(MessageStreamEvent::End(_))));
    }

    #[tokio::test]
    async fn the_responses_stream_ends_on_the_done_sentinel() {
        use futures::stream;

        // The official Responses stream closes with `data: [DONE]`.
        let sse = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n",
            "data: [DONE]\n\n",
        );

        let inner = stream::iter(vec![Ok::<_, std::io::Error>(sse.as_bytes().to_vec())]);
        let sse_stream = EventStream::new(inner);
        let codec: Arc<dyn LlmCodec> = Arc::new(llm_suite_codec::OpenaiResponseCodec::new());
        let mut stream = SseMessageStream::new(sse_stream, codec, None, None);

        let mut events: Vec<MessageStreamEvent> = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.expect("no stream error"));
        }

        let final_msg = events
            .iter()
            .find_map(|e| match e {
                MessageStreamEvent::FinalMessage(f) => Some(f),
                _ => None,
            })
            .expect("FinalMessage must be emitted");
        assert_eq!(
            llm_suite_message::helper::extract_text_content(&final_msg.message),
            "Hello"
        );
        assert_eq!(final_msg.usage.as_ref().unwrap().total_tokens, 5);
        assert!(matches!(events.last(), Some(MessageStreamEvent::End(_))));
    }
}
