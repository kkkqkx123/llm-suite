//! Accumulation of stream events into the snapshots of a turn.

use crate::dead_loop_detector::DeadLoopDetector;
use llm_suite_types::llm::{
    MessageStreamEvent, MessageStreamInputJson, MessageStreamReasoning, MessageStreamStart,
    MessageStreamText, TokenUsageStats,
};
use llm_suite_types::message::MessageContent;

use super::final_message::{build_final_message, TurnState};
use super::tool_call::ToolCallState;
use super::usage::merge_usage;

/// What the accumulator did with a raw event.
///
/// The distinction matters to the transport: a finalized turn still owes
/// the consumer its terminal `End` event, an aborted turn is over, and a
/// consumed event produces nothing at all.
#[derive(Debug)]
pub enum Accumulated {
    /// The event carried nothing new; nothing is forwarded.
    Consumed,
    /// The event was enriched and is forwarded to the consumer.
    Forwarded(MessageStreamEvent),
    /// The turn ended: the final message was built. The terminal `End`
    /// event is delivered separately.
    Finalized(MessageStreamEvent),
    /// The turn was aborted by the dead-loop guard.
    Aborted(MessageStreamEvent),
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
    tool_calls: ToolCallState,
    content_blocks: Vec<MessageContent>,
    annotations: Vec<serde_json::Value>,
    usage: Option<TokenUsageStats>,
    dead_loop_detector: Option<DeadLoopDetector>,
    final_emitted: bool,
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
            tool_calls: ToolCallState::default(),
            content_blocks: Vec::new(),
            annotations: Vec::new(),
            usage: None,
            dead_loop_detector,
            final_emitted: false,
        }
    }

    /// Push a raw event and report what the accumulator did with it.
    pub fn push(&mut self, event: MessageStreamEvent) -> Accumulated {
        match event {
            MessageStreamEvent::Start(start) => {
                if self.stream_id.is_none() {
                    self.stream_id = start.id.clone();
                }
                if self.stream_model.is_none() {
                    self.stream_model = start.model.clone();
                }
                if self.start_emitted {
                    return Accumulated::Consumed;
                }
                self.start_emitted = true;
                Accumulated::Forwarded(MessageStreamEvent::Start(MessageStreamStart {
                    id: self.stream_id.clone(),
                    model: self.stream_model.clone(),
                }))
            }
            MessageStreamEvent::Text(text_event) => {
                self.text_snapshot.push_str(&text_event.text);
                Accumulated::Forwarded(MessageStreamEvent::Text(MessageStreamText {
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
                        return Accumulated::Aborted(MessageStreamEvent::Abort(
                            llm_suite_types::llm::MessageStreamAbort {
                                reason: result
                                    .details
                                    .unwrap_or_else(|| "Dead loop detected".to_string()),
                            },
                        ));
                    }
                }

                Accumulated::Forwarded(MessageStreamEvent::ReasoningText(MessageStreamReasoning {
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
                let arguments = self
                    .tool_calls
                    .append_arguments(index, &input_json.partial_json);

                // Try to parse the accumulated JSON
                let parsed_snapshot =
                    llm_suite_tool_call::partial_json_parser::parse_partial_json(&arguments)
                        .as_complete()
                        .cloned();

                Accumulated::Forwarded(MessageStreamEvent::InputJson(MessageStreamInputJson {
                    partial_json: input_json.partial_json,
                    index: Some(index),
                    parsed_snapshot,
                }))
            }
            MessageStreamEvent::ToolCallDelta(delta) => {
                // Accumulate tool call fragments
                self.tool_calls.apply_delta(&delta);

                // Return the delta for consumers that want incremental updates
                Accumulated::Forwarded(MessageStreamEvent::ToolCallDelta(delta))
            }
            MessageStreamEvent::Refusal(refusal_event) => {
                self.refusal_snapshot.push_str(&refusal_event.refusal);
                Accumulated::Forwarded(MessageStreamEvent::Refusal(
                    llm_suite_types::llm::MessageStreamRefusal {
                        refusal: refusal_event.refusal.clone(),
                        snapshot: self.refusal_snapshot.clone(),
                    },
                ))
            }
            MessageStreamEvent::RedactedThinking(redacted) => {
                // The payload is opaque and has to survive into the final
                // message so the reply can be replayed in the next turn.
                self.content_blocks.push(MessageContent::RedactedThinking {
                    data: redacted.data.clone(),
                });
                Accumulated::Forwarded(MessageStreamEvent::RedactedThinking(redacted))
            }
            MessageStreamEvent::NativeBlock(native) => {
                // The provider block travels verbatim so the reply can be
                // replayed in the next turn.
                self.content_blocks.push(MessageContent::Native {
                    provider: native.provider.clone(),
                    kind: native.kind.clone(),
                    payload: native.payload.clone(),
                });
                Accumulated::Forwarded(MessageStreamEvent::NativeBlock(native))
            }
            MessageStreamEvent::Annotation(annotation) => {
                // The sources of a grounded answer reach the final message
                // under the same key the non-streaming path uses.
                self.annotations.push(annotation.annotation.clone());
                Accumulated::Forwarded(MessageStreamEvent::Annotation(annotation))
            }
            MessageStreamEvent::Usage(usage_event) => {
                self.usage = Some(merge_usage(self.usage.as_ref(), &usage_event.usage));
                Accumulated::Forwarded(MessageStreamEvent::Usage(usage_event))
            }
            MessageStreamEvent::End(_) => {
                if self.final_emitted {
                    return Accumulated::Consumed;
                }
                self.final_emitted = true;

                let tool_calls = self.tool_calls.assemble();
                let final_message = build_final_message(TurnState {
                    text: &self.text_snapshot,
                    reasoning: &self.reasoning_snapshot,
                    reasoning_signature: &self.reasoning_signature_snapshot,
                    refusal: &self.refusal_snapshot,
                    stream_id: self.stream_id.as_deref(),
                    stream_model: self.stream_model.as_deref(),
                    content_blocks: &self.content_blocks,
                    annotations: &self.annotations,
                    usage: self.usage.as_ref(),
                    tool_calls: &tool_calls,
                });

                Accumulated::Finalized(MessageStreamEvent::FinalMessage(final_message))
            }
            // Pass through other events unchanged
            other => Accumulated::Forwarded(other),
        }
    }
}
