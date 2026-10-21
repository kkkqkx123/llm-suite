use crate::dead_loop_detector::DeadLoopDetector;
use async_trait::async_trait;
use eventsource_stream::EventStream;
use futures::StreamExt;
use llm_suite_codec::error::LlmError;
use llm_suite_codec::LlmCodec;
use llm_suite_types::llm::{
    MessageStreamEvent, MessageStreamFinal, MessageStreamInputJson, MessageStreamReasoning,
    MessageStreamStart, MessageStreamText, TokenUsageStats,
};
use llm_suite_types::message::{
    LlmFunctionCall, LlmToolCall, Message, MessageContent, MessageContentValue, MessageRole,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait MessageStream: Send {
    async fn next(&mut self) -> Option<Result<MessageStreamEvent, LlmError>>;
}

pub struct SseMessageStream<S> {
    stream: EventStream<S>,
    codec: Arc<dyn LlmCodec>,
    cancel: Option<CancellationToken>,
    done: bool,
    /// The codec emitted `End`; the accumulator built the `FinalMessage`
    /// and the `End` event itself is pending on the next `next()` call.
    pending_end: bool,
    /// Raw events decoded from a chunk that have not yet been routed
    /// through the accumulator (codecs may emit several events per chunk).
    pending_raw: std::collections::VecDeque<MessageStreamEvent>,
    accumulator: MessageAccumulator,
}

impl<S> SseMessageStream<S> {
    pub fn new(
        stream: EventStream<S>,
        codec: Arc<dyn LlmCodec>,
        cancel: Option<CancellationToken>,
        dead_loop_config: Option<&llm_suite_types::llm::DeadLoopDetectionConfig>,
    ) -> Self {
        Self {
            stream,
            codec,
            cancel,
            done: false,
            pending_end: false,
            pending_raw: std::collections::VecDeque::new(),
            accumulator: MessageAccumulator::new(dead_loop_config),
        }
    }
}

#[async_trait]
impl<S, B, E> MessageStream for SseMessageStream<S>
where
    S: futures::Stream<Item = Result<B, E>> + Unpin + Send + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + Send + 'static,
{
    async fn next(&mut self) -> Option<Result<MessageStreamEvent, LlmError>> {
        if self.done {
            return None;
        }

        // The final message was already delivered; terminate with the
        // pending `End` event before the stream is exhausted.
        if self.pending_end {
            self.pending_end = false;
            self.done = true;
            return Some(Ok(MessageStreamEvent::End(
                llm_suite_types::llm::MessageStreamEnd {},
            )));
        }

        if let Some(ref cancel) = self.cancel {
            if cancel.is_cancelled() {
                self.done = true;
                return Some(Err(LlmError::Cancelled));
            }
        }

        loop {
            // Take the next raw event either from a previous multi-event
            // chunk or from the wire.
            let raw_event = if let Some(ev) = self.pending_raw.pop_front() {
                ev
            } else {
                match self.stream.next().await {
                    Some(Ok(event)) => {
                        let data = event.data.trim().to_string();

                        if data.is_empty() {
                            continue;
                        }

                        match self.codec.parse_stream_chunk_events(&data) {
                            Ok(events) => {
                                if events.is_empty() {
                                    continue;
                                }
                                let mut iter = events.into_iter();
                                let first = iter.next().expect("non-empty checked above");
                                self.pending_raw.extend(iter);
                                first
                            }
                            Err(e) => return Some(Err(e)),
                        }
                    }
                    Some(Err(e)) => {
                        self.done = true;
                        return Some(Err(LlmError::StreamError(e.to_string())));
                    }
                    None => {
                        self.done = true;
                        return None;
                    }
                }
            };

            // Always route `End` through the accumulator: it
            // assembles the `FinalMessage` (content, tool calls,
            // usage) before the stream terminates. The `End`
            // event itself is emitted on the next call.
            if let Some(emitted) = self.accumulator.push(raw_event) {
                match emitted {
                    MessageStreamEvent::Abort(_) => {
                        self.done = true;
                    }
                    MessageStreamEvent::End(_) => {
                        self.done = true;
                    }
                    MessageStreamEvent::FinalMessage(_) => {
                        self.pending_end = true;
                    }
                    _ => {}
                }
                return Some(Ok(emitted));
            }
        }
    }
}

/// Accumulates streaming events into message snapshots.
pub struct MessageAccumulator {
    text_snapshot: String,
    reasoning_snapshot: String,
    reasoning_signature_snapshot: String,
    refusal_snapshot: String,
    stream_id: Option<String>,
    stream_model: Option<String>,
    start_emitted: bool,
    tool_calls: BTreeMap<usize, PartialToolCall>,
    content_blocks: Vec<MessageContent>,
    usage: Option<TokenUsageStats>,
    dead_loop_detector: Option<DeadLoopDetector>,
    final_emitted: bool,
}

#[derive(Debug, Clone)]
struct PartialToolCall {
    id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    index: usize,
}

impl MessageAccumulator {
    pub fn new(dead_loop_config: Option<&llm_suite_types::llm::DeadLoopDetectionConfig>) -> Self {
        let dead_loop_detector = dead_loop_config.map(|c| DeadLoopDetector::new(c.into()));
        Self {
            text_snapshot: String::new(),
            reasoning_snapshot: String::new(),
            reasoning_signature_snapshot: String::new(),
            refusal_snapshot: String::new(),
            stream_id: None,
            stream_model: None,
            start_emitted: false,
            tool_calls: BTreeMap::new(),
            content_blocks: Vec::new(),
            usage: None,
            dead_loop_detector,
            final_emitted: false,
        }
    }

    /// Push a raw event and return the semantically-enriched event to emit.
    pub fn push(&mut self, event: MessageStreamEvent) -> Option<MessageStreamEvent> {
        match event {
            MessageStreamEvent::Start(start) => {
                if self.stream_id.is_none() {
                    self.stream_id = start.id.clone();
                }
                if self.stream_model.is_none() {
                    self.stream_model = start.model.clone();
                }
                if self.start_emitted {
                    return None;
                }
                self.start_emitted = true;
                Some(MessageStreamEvent::Start(MessageStreamStart {
                    id: self.stream_id.clone(),
                    model: self.stream_model.clone(),
                }))
            }
            MessageStreamEvent::Text(text_event) => {
                self.text_snapshot.push_str(&text_event.text);
                Some(MessageStreamEvent::Text(MessageStreamText {
                    text: text_event.text,
                    snapshot: self.text_snapshot.clone(),
                }))
            }
            MessageStreamEvent::ReasoningText(reasoning_event) => {
                self.reasoning_snapshot.push_str(&reasoning_event.reasoning);
                if let Some(signature) = reasoning_event.signature.as_deref() {
                    self.reasoning_signature_snapshot.push_str(signature);
                }

                // Dead loop detection
                if let Some(ref mut detector) = self.dead_loop_detector {
                    let result = detector.detect(&self.reasoning_snapshot);
                    if result.detected {
                        return Some(MessageStreamEvent::Abort(
                            llm_suite_types::llm::MessageStreamAbort {
                                reason: result
                                    .details
                                    .unwrap_or_else(|| "Dead loop detected".to_string()),
                            },
                        ));
                    }
                }

                Some(MessageStreamEvent::ReasoningText(MessageStreamReasoning {
                    reasoning: reasoning_event.reasoning,
                    snapshot: self.reasoning_snapshot.clone(),
                    signature: reasoning_event.signature,
                    signature_snapshot: if self.reasoning_signature_snapshot.is_empty() {
                        None
                    } else {
                        Some(self.reasoning_signature_snapshot.clone())
                    },
                }))
            }
            MessageStreamEvent::InputJson(input_json) => {
                // For Anthropic-style input_json_delta, accumulate by index
                let index = input_json.index.unwrap_or(0);
                let partial_tool_call =
                    self.tool_calls
                        .entry(index)
                        .or_insert_with(|| PartialToolCall {
                            id: None,
                            call_id: None,
                            name: None,
                            arguments: String::new(),
                            index,
                        });
                partial_tool_call
                    .arguments
                    .push_str(&input_json.partial_json);

                // Try to parse the accumulated JSON
                let parsed_snapshot = llm_suite_tool_call::partial_json_parser::parse_partial_json(
                    &partial_tool_call.arguments,
                )
                .as_complete()
                .cloned();

                Some(MessageStreamEvent::InputJson(MessageStreamInputJson {
                    partial_json: input_json.partial_json,
                    index: Some(index),
                    parsed_snapshot,
                }))
            }
            MessageStreamEvent::ToolCallDelta(delta) => {
                // Accumulate tool call fragments
                let partial =
                    self.tool_calls
                        .entry(delta.index)
                        .or_insert_with(|| PartialToolCall {
                            id: None,
                            call_id: None,
                            name: None,
                            arguments: String::new(),
                            index: delta.index,
                        });

                if let Some(call_id) = delta.call_id.clone() {
                    partial.call_id = Some(call_id);
                }

                if delta.is_snapshot {
                    // Complete snapshot (Gemini functionCall, OpenAI response
                    // function_call.done): replace accumulated fragments.
                    if let Some(id) = delta.id.clone() {
                        partial.id = Some(id);
                    }
                    if let Some(name) = delta.name.clone() {
                        partial.name = Some(name);
                    }
                    if let Some(ref args) = delta.arguments {
                        partial.arguments = args.clone();
                    }
                } else {
                    if let Some(id) = delta.id.clone() {
                        partial.id = Some(id);
                    }
                    if let Some(name) = delta.name.clone() {
                        partial.name = Some(name);
                    }
                    if let Some(ref args) = delta.arguments {
                        partial.arguments.push_str(args);
                    }
                }

                // Return the delta for consumers that want incremental updates
                Some(MessageStreamEvent::ToolCallDelta(delta))
            }
            MessageStreamEvent::Refusal(refusal_event) => {
                self.refusal_snapshot.push_str(&refusal_event.refusal);
                Some(MessageStreamEvent::Refusal(
                    llm_suite_types::llm::MessageStreamRefusal {
                        refusal: refusal_event.refusal.clone(),
                        snapshot: self.refusal_snapshot.clone(),
                    },
                ))
            }
            MessageStreamEvent::Usage(usage_event) => {
                // Field-wise merge: providers deliver usage across several
                // events (Anthropic's `message_start` carries input tokens,
                // `message_delta` carries cumulative output and may omit
                // input tokens). Later non-zero / `Some` values win; zero /
                // `None` values never erase earlier data. `total_tokens` is
                // recomputed from prompt + completion when the wire only
                // carries partial counters.
                let u = &usage_event.usage;
                let merged = match self.usage.take() {
                    Some(prev) => {
                        let prompt_tokens = if u.prompt_tokens > 0 {
                            u.prompt_tokens
                        } else {
                            prev.prompt_tokens
                        };
                        let completion_tokens = if u.completion_tokens > 0 {
                            u.completion_tokens
                        } else {
                            prev.completion_tokens
                        };
                        let total_tokens = if u.total_tokens > 0 {
                            u.total_tokens
                        } else if u.prompt_tokens > 0 || u.completion_tokens > 0 {
                            prompt_tokens + completion_tokens
                        } else {
                            prev.total_tokens
                        };
                        TokenUsageStats {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            reasoning_tokens: u.reasoning_tokens.or(prev.reasoning_tokens),
                            cache_read_tokens: u.cache_read_tokens.or(prev.cache_read_tokens),
                            cache_write_tokens: u.cache_write_tokens.or(prev.cache_write_tokens),
                            prompt_tokens_cost: u.prompt_tokens_cost.or(prev.prompt_tokens_cost),
                            completion_tokens_cost: u
                                .completion_tokens_cost
                                .or(prev.completion_tokens_cost),
                            total_cost: u.total_cost.or(prev.total_cost),
                        }
                    }
                    None => u.clone(),
                };
                self.usage = Some(merged);
                Some(MessageStreamEvent::Usage(usage_event))
            }
            MessageStreamEvent::End(_) => {
                if self.final_emitted {
                    return None;
                }
                self.final_emitted = true;

                // Build final message
                let tool_calls = self.assemble_tool_calls();
                let content = if self.content_blocks.is_empty() {
                    MessageContentValue::Text(self.text_snapshot.clone())
                } else {
                    MessageContentValue::Rich(self.content_blocks.clone())
                };

                let message = Message {
                    id: llm_suite_types::Id::new(),
                    role: MessageRole::Assistant,
                    content,
                    timestamp: llm_suite_common::time::now(),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    thinking: if self.reasoning_snapshot.is_empty() {
                        None
                    } else {
                        Some(self.reasoning_snapshot.clone())
                    },
                    thinking_signature: if self.reasoning_signature_snapshot.is_empty() {
                        None
                    } else {
                        Some(self.reasoning_signature_snapshot.clone())
                    },
                    metadata: None,
                };

                Some(MessageStreamEvent::FinalMessage(MessageStreamFinal {
                    message,
                    usage: self.usage.clone(),
                    stream_stats: None, // Filled by the gateway stream wrapper
                    id: self.stream_id.clone(),
                    model: self.stream_model.clone(),
                    refusal: if self.refusal_snapshot.is_empty() {
                        None
                    } else {
                        Some(self.refusal_snapshot.clone())
                    },
                }))
            }
            // Pass through other events unchanged
            _ => Some(event),
        }
    }

    fn assemble_tool_calls(&self) -> Vec<LlmToolCall> {
        self.tool_calls
            .values()
            .filter_map(|partial| {
                let name = partial.name.clone()?;
                let id = partial.id.clone().unwrap_or_else(|| {
                    format!("call_{}_{}", llm_suite_common::time::now(), partial.index)
                });
                let arguments = if partial.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    partial.arguments.clone()
                };
                Some(LlmToolCall {
                    id,
                    r#type: "function".to_string(),
                    call_id: partial.call_id.clone(),
                    function: LlmFunctionCall { name, arguments },
                })
            })
            .collect()
    }
}

impl PartialToolCall {}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::{MessageStreamToolCallDelta, MessageStreamUsage};

    /// Feed a real OpenAI-style SSE byte stream through `SseMessageStream` with
    /// the real `OpenaiChatCodec`, proving that the `FinalMessage` is
    /// assembled and delivered before the `End` event (regression test for the
    /// `End` short-circuit that skipped the accumulator).
    #[tokio::test]
    async fn real_sse_path_emits_final_message_before_end() {
        use futures::stream;
        use llm_suite_types::llm::MessageStreamEvent;

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
                is_snapshot: false,
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let MessageStreamEvent::FinalMessage(final_msg) = final_event.unwrap() else {
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
                is_snapshot: true,
            },
        ));

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let MessageStreamEvent::FinalMessage(final_msg) = final_event.unwrap() else {
            panic!("expected FinalMessage");
        };
        let calls = final_msg.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn text_snapshot_and_usage_merge_on_final_message() {
        let mut acc = MessageAccumulator::new(None);

        acc.push(MessageStreamEvent::Text(MessageStreamText {
            text: "hello".to_string(),
            snapshot: String::new(),
        }));
        acc.push(MessageStreamEvent::Text(MessageStreamText {
            text: " world".to_string(),
            snapshot: String::new(),
        }));
        acc.push(MessageStreamEvent::Usage(MessageStreamUsage {
            usage: TokenUsageStats {
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
        let MessageStreamEvent::FinalMessage(final_msg) = final_event.unwrap() else {
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
            usage: TokenUsageStats {
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
            usage: TokenUsageStats {
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
        let MessageStreamEvent::FinalMessage(final_msg) = final_event.unwrap() else {
            panic!("expected FinalMessage");
        };
        let usage = final_msg.usage.expect("usage must be merged");
        assert_eq!(
            usage.prompt_tokens, 25,
            "input from message_start must survive"
        );
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(
            usage.total_tokens, 32,
            "total recomputed from prompt + completion"
        );
        assert_eq!(usage.cache_read_tokens, Some(3));
        assert_eq!(usage.reasoning_tokens, Some(2));
    }

    #[test]
    fn start_event_is_forwarded_once_and_lands_on_final_message() {
        use llm_suite_types::llm::MessageStreamStart;

        let mut acc = MessageAccumulator::new(None);

        let first = acc.push(MessageStreamEvent::Start(MessageStreamStart {
            id: Some("msg_1".to_string()),
            model: Some("claude-sonnet".to_string()),
        }));
        match first.expect("first Start must be forwarded") {
            MessageStreamEvent::Start(s) => {
                assert_eq!(s.id.as_deref(), Some("msg_1"));
                assert_eq!(s.model.as_deref(), Some("claude-sonnet"));
            }
            other => panic!("expected Start, got {other:?}"),
        }

        let second = acc.push(MessageStreamEvent::Start(MessageStreamStart {
            id: Some("msg_2".to_string()),
            model: None,
        }));
        assert!(second.is_none(), "repeat Start events must be swallowed");

        acc.push(MessageStreamEvent::Text(MessageStreamText {
            text: "hi".to_string(),
            snapshot: String::new(),
        }));
        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let MessageStreamEvent::FinalMessage(final_msg) = final_event.unwrap() else {
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
        match emitted.expect("reasoning must be forwarded") {
            MessageStreamEvent::ReasoningText(r) => {
                assert_eq!(r.snapshot, "let me think further");
                assert_eq!(
                    r.signature_snapshot.as_deref(),
                    Some("sig-asig-b"),
                    "signature fragments must accumulate"
                );
            }
            other => panic!("expected ReasoningText, got {other:?}"),
        }

        let final_event = acc.push(MessageStreamEvent::End(
            llm_suite_types::llm::MessageStreamEnd {},
        ));
        let MessageStreamEvent::FinalMessage(final_msg) = final_event.unwrap() else {
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
}
