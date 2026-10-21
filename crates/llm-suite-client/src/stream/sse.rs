//! The SSE transport: wire frames in, stream events out.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use eventsource_stream::EventStream;
use futures::StreamExt;
use llm_suite_codec::error::LlmError;
use llm_suite_codec::LlmCodec;
use llm_suite_types::llm::MessageStreamEvent;
use tokio_util::sync::CancellationToken;

use super::accumulator::{Accumulated, MessageAccumulator};
use super::MessageStream;

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
    pending_raw: VecDeque<MessageStreamEvent>,
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
            pending_raw: VecDeque::new(),
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
            match self.accumulator.push(raw_event) {
                Accumulated::Consumed => continue,
                Accumulated::Forwarded(event) => return Some(Ok(event)),
                Accumulated::Finalized(event) => {
                    self.pending_end = true;
                    return Some(Ok(event));
                }
                Accumulated::Aborted(event) => {
                    self.done = true;
                    return Some(Ok(event));
                }
            }
        }
    }
}
