use std::sync::Arc;

use llm_suite_token::{LlmCallContext, LlmUsageReport};
use llm_suite_types::llm::{MessageStreamEvent, StreamStats, TokenUsageStats};

use crate::stream::MessageStream;
use llm_suite_codec::error::LlmError;

/// Generic token-usage sink decoupled from any concrete metrics collector.
/// Implementations receive the full provider usage breakdown (including
/// reasoning and cache hit/write tokens), the cost and the call context;
/// the facade layer wires a project-specific adapter (e.g. the metrics
/// crate collector) into this trait.
pub trait TokenUsageSink: Send + Sync {
    fn record_token_usage(&self, report: &LlmUsageReport);
}

/// Observation points for gateway request metrics, decoupled from any
/// concrete metrics collector. The facade layer wires a project-specific
/// adapter into this trait; the transport and gateway layers stay leaf.
pub trait LlmMetricsSink: TokenUsageSink {
    fn record_first_byte(&self, duration_ms: f64, context: &LlmCallContext);
    /// Per-stream generation statistics emitted once when the wrapped
    /// stream produces its final message: real time-to-first-token,
    /// generation window, end-to-end duration, chunk count, reasoning text
    /// length and output size. Default no-op so existing implementors keep
    /// compiling; the facade adapter overrides it.
    fn record_stream_stats(
        &self,
        stats: &StreamStats,
        completion_tokens: u32,
        context: &LlmCallContext,
    ) {
        let _ = (stats, completion_tokens, context);
    }
    fn record_request(
        &self,
        duration_ms: f64,
        success: bool,
        error_kind: Option<&str>,
        context: &LlmCallContext,
    );
    fn record_retry(&self, context: &LlmCallContext);
}

/// Convenience alias for shared token-usage sinks.
pub type SharedTokenUsageSink = Arc<dyn TokenUsageSink>;

/// Convenience alias for shared gateway metrics sinks.
pub type SharedLlmMetricsSink = Arc<dyn LlmMetricsSink>;

/// Stream wrapper that records token usage from mid-stream usage events
/// (OpenAI `include_usage` chunk, Anthropic `message_delta`) or, as a
/// fallback, from the final message once the stream is exhausted. It also
/// collects stream statistics (chunk count / first-chunk latency / first
/// reasoning chunk / stream and total durations / reasoning text length)
/// and attaches them to the `FinalMessage` event.
pub struct TokenRecordingStream {
    inner: Box<dyn MessageStream>,
    sink: Option<SharedTokenUsageSink>,
    metrics: Option<SharedLlmMetricsSink>,
    context: LlmCallContext,
    last_usage: Option<TokenUsageStats>,
    /// Output size reported with `record_stream_stats`: tracked separately
    /// from `last_usage` because a final message without usage must not
    /// erase a mid-stream usage observation.
    completion_tokens: u32,
    recorded: bool,
    stats_recorded: bool,
    start_time: i64,
    first_chunk_time: Option<i64>,
    first_reasoning_time: Option<i64>,
    last_chunk_time: Option<i64>,
    chunk_count: u32,
    /// Accumulated length of the reasoning text streamed for the turn.
    reasoning_chars: u32,
    /// Local estimate used when the provider reports no usage at all.
    fallback_usage: Option<TokenUsageStats>,
    /// Whether the request outcome was already reported to the metrics sink.
    request_recorded: bool,
}

impl TokenRecordingStream {
    pub fn new(
        inner: Box<dyn MessageStream>,
        sink: Option<SharedTokenUsageSink>,
        context: LlmCallContext,
        fallback_usage: Option<TokenUsageStats>,
    ) -> Self {
        Self {
            inner,
            sink,
            metrics: None,
            context,
            last_usage: None,
            completion_tokens: 0,
            recorded: false,
            stats_recorded: false,
            start_time: llm_suite_common::time::now(),
            first_chunk_time: None,
            first_reasoning_time: None,
            last_chunk_time: None,
            chunk_count: 0,
            reasoning_chars: 0,
            fallback_usage,
            request_recorded: false,
        }
    }

    /// Attach a metrics sink that receives the per-stream generation
    /// statistics once the final message arrives. Zero overhead when absent.
    pub fn with_metrics_sink(mut self, metrics: Option<SharedLlmMetricsSink>) -> Self {
        self.metrics = metrics;
        self
    }

    fn record(&mut self, usage: &TokenUsageStats, estimated: bool) {
        let Some(sink) = &self.sink else {
            return;
        };
        self.recorded = true;
        let mut report = LlmUsageReport::from_usage(self.context.clone(), usage.clone());
        report.estimated = estimated;
        sink.record_token_usage(&report);
    }

    /// Report the request outcome to the metrics sink exactly once.
    fn record_request(&mut self, success: bool, error_kind: Option<&str>, end_time: i64) {
        if self.request_recorded {
            return;
        }
        self.request_recorded = true;
        let Some(metrics) = &self.metrics else {
            return;
        };
        let duration_ms = end_time.saturating_sub(self.start_time).max(0) as f64;
        metrics.record_request(duration_ms, success, error_kind, &self.context);
    }

    fn build_stats(&self, end_time: i64) -> StreamStats {
        let first = self.first_chunk_time.unwrap_or(end_time);
        let last = self.last_chunk_time.unwrap_or(first);
        StreamStats {
            chunk_count: self.chunk_count,
            time_to_first_chunk: first - self.start_time,
            stream_duration: last.saturating_sub(first),
            total_duration: end_time.saturating_sub(self.start_time),
            reasoning_chars: self.reasoning_chars,
            time_to_first_reasoning: self
                .first_reasoning_time
                .map(|t| t.saturating_sub(self.start_time)),
        }
    }
}

#[async_trait::async_trait]
impl MessageStream for TokenRecordingStream {
    async fn next(&mut self) -> Option<Result<MessageStreamEvent, LlmError>> {
        let mut event = self.inner.next().await;

        let now = llm_suite_common::time::now();
        if event.is_some() {
            if self.first_chunk_time.is_none() {
                self.first_chunk_time = Some(now);
            }
            self.last_chunk_time = Some(now);
            self.chunk_count = self.chunk_count.saturating_add(1);
        }

        if let Some(Ok(MessageStreamEvent::ReasoningText(reasoning))) = &event {
            if self.first_reasoning_time.is_none() {
                self.first_reasoning_time = Some(now);
            }
            self.reasoning_chars = self
                .reasoning_chars
                .saturating_add(reasoning.reasoning.chars().count() as u32);
        }

        if let Some(Ok(MessageStreamEvent::Usage(usage))) = &event {
            self.last_usage = Some(usage.usage.clone());
            self.completion_tokens = usage.usage.completion_tokens;
            if !self.recorded {
                self.record(&usage.usage, false);
            }
        }

        if let Some(Ok(MessageStreamEvent::FinalMessage(msg))) = &mut event {
            if let Some(usage) = &msg.usage {
                self.completion_tokens = usage.completion_tokens;
            }
            self.last_usage = msg.usage.clone();
            if msg.stream_stats.is_none() {
                msg.stream_stats = Some(self.build_stats(now));
            }
            if !self.stats_recorded {
                self.stats_recorded = true;
                if let (Some(metrics), Some(stats)) = (&self.metrics, msg.stream_stats.as_ref()) {
                    metrics.record_stream_stats(
                        stats,
                        self.completion_tokens,
                        &self.context,
                    );
                }
            }
            // The stream completed: the end-to-end duration is the request
            // duration for a streaming call.
            let total_duration = msg
                .stream_stats
                .as_ref()
                .map(|s| s.total_duration)
                .unwrap_or_else(|| now.saturating_sub(self.start_time));
            self.record_request(true, None, self.start_time + total_duration);
        }

        if let Some(Err(error)) = &event {
            self.record_request(false, Some(error.metric_class()), now);
        }

        if event.is_none() && !self.recorded {
            if let Some(usage) = self.last_usage.clone() {
                self.record(&usage, false);
            } else if let Some(fallback) = self.fallback_usage.clone() {
                self.record(&fallback, true);
            }
        }

        if event.is_none() {
            // The stream ended without a final message: the request never
            // completed normally.
            self.record_request(false, Some("stream_incomplete"), now);
        }

        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::message::{Message, MessageContentValue, MessageRole};

    fn context() -> LlmCallContext {
        LlmCallContext {
            model: Some("test-model".to_string()),
            profile_id: Some("default".to_string()),
            provider_id: Some("openai".to_string()),
            format: Some("OPENAI_CHAT".to_string()),
            execution_id: Some("exec-1".to_string()),
            accounting: llm_suite_token::token_accounting(&llm_suite_types::llm::LlmFormat::OpenaiChat),
        }
    }

    struct FakeStream {
        events: Vec<MessageStreamEvent>,
    }

    #[async_trait::async_trait]
    impl MessageStream for FakeStream {
        async fn next(&mut self) -> Option<Result<MessageStreamEvent, LlmError>> {
            if self.events.is_empty() {
                return None;
            }
            Some(Ok(self.events.remove(0)))
        }
    }

    fn text_event(text: &str) -> MessageStreamEvent {
        MessageStreamEvent::Text(llm_suite_types::llm::MessageStreamText {
            text: text.to_string(),
            snapshot: text.to_string(),
        })
    }

    fn reasoning_event(text: &str) -> MessageStreamEvent {
        MessageStreamEvent::ReasoningText(llm_suite_types::llm::MessageStreamReasoning {
            reasoning: text.to_string(),
            snapshot: text.to_string(),
            signature: None,
            signature_snapshot: None,
        })
    }

    fn assistant_message(content: &str) -> Message {
        Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Text(content.to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn final_message_carries_stream_stats() {
        let inner = FakeStream {
            events: vec![
                text_event("hello"),
                text_event(" world"),
                MessageStreamEvent::FinalMessage(llm_suite_types::llm::MessageStreamFinal {
                    message: assistant_message("hello world"),
                    usage: None,
                    stream_stats: None,
                    id: None,
                    model: None,
                    refusal: None,
                }),
            ],
        };
        let mut stream = TokenRecordingStream::new(Box::new(inner), None, context(), None);

        let mut final_stats = None;
        while let Some(event) = stream.next().await {
            if let Ok(MessageStreamEvent::FinalMessage(msg)) = event {
                final_stats = msg.stream_stats;
            }
        }

        let stats = final_stats.expect("FinalMessage must carry stream_stats");
        assert!(stats.chunk_count >= 2, "chunk_count: {}", stats.chunk_count);
        assert!(
            stats.time_to_first_chunk >= 0,
            "time_to_first_chunk: {}",
            stats.time_to_first_chunk
        );
        assert!(stats.total_duration >= 0);
    }

    #[tokio::test]
    async fn reasoning_chunks_are_measured() {
        let inner = FakeStream {
            events: vec![
                reasoning_event("think"),
                reasoning_event("ing"),
                MessageStreamEvent::FinalMessage(llm_suite_types::llm::MessageStreamFinal {
                    message: assistant_message("done"),
                    usage: None,
                    stream_stats: None,
                    id: None,
                    model: None,
                    refusal: None,
                }),
            ],
        };
        let mut stream = TokenRecordingStream::new(Box::new(inner), None, context(), None);

        let mut final_stats = None;
        while let Some(event) = stream.next().await {
            if let Ok(MessageStreamEvent::FinalMessage(msg)) = event {
                final_stats = msg.stream_stats;
            }
        }

        let stats = final_stats.expect("FinalMessage must carry stream_stats");
        assert_eq!(stats.reasoning_chars, 8);
        assert!(stats.time_to_first_reasoning.is_some());
    }

    /// Captures the full payload forwarded to `TokenUsageSink` so tests can
    /// assert that cache/reasoning breakdowns reach the metrics boundary.
    struct RecordingSink {
        calls: std::sync::Mutex<Vec<LlmUsageReport>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn last(&self) -> LlmUsageReport {
            self.calls
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("sink must have been called")
        }
    }

    impl TokenUsageSink for RecordingSink {
        fn record_token_usage(&self, report: &LlmUsageReport) {
            self.calls.lock().unwrap().push(report.clone());
        }
    }

    #[tokio::test]
    async fn sink_receives_full_usage_breakdown() {
        let inner = FakeStream {
            events: vec![
                text_event("done"),
                MessageStreamEvent::Usage(llm_suite_types::llm::MessageStreamUsage {
                    usage: TokenUsageStats {
                        prompt_tokens: 100,
                        completion_tokens: 20,
                        total_tokens: 120,
                        reasoning_tokens: Some(6),
                        cache_read_tokens: Some(80),
                        cache_write_tokens: Some(15),
                        prompt_tokens_cost: None,
                        completion_tokens_cost: None,
                        total_cost: None,
                    },
                }),
                MessageStreamEvent::FinalMessage(llm_suite_types::llm::MessageStreamFinal {
                    message: assistant_message("done"),
                    usage: None,
                    stream_stats: None,
                    id: None,
                    model: None,
                    refusal: None,
                }),
            ],
        };
        let sink = std::sync::Arc::new(RecordingSink::new());
        let mut stream =
            TokenRecordingStream::new(Box::new(inner), Some(sink.clone()), context(), None);
        while stream.next().await.is_some() {}

        let report = sink.last();
        assert_eq!(report.context.model.as_deref(), Some("test-model"));
        assert_eq!(report.context.profile_id.as_deref(), Some("default"));
        assert_eq!(report.context.execution_id.as_deref(), Some("exec-1"));
        assert_eq!(report.usage.prompt_tokens, 100);
        assert_eq!(report.usage.completion_tokens, 20);
        assert_eq!(report.usage.reasoning_tokens, Some(6));
        assert_eq!(report.usage.cache_read_tokens, Some(80));
        assert_eq!(report.usage.cache_write_tokens, Some(15));
    }

    /// Captures the per-stream generation statistics forwarded to
    /// `LlmMetricsSink::record_stream_stats`.
    struct RecordingMetricsSink {
        stats: std::sync::Mutex<Vec<(StreamStats, u32)>>,
    }

    impl RecordingMetricsSink {
        fn new() -> Self {
            Self {
                stats: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn last(&self) -> (StreamStats, u32) {
            self.stats
                .lock()
                .expect("stats lock poisoned")
                .last()
                .cloned()
                .expect("metrics sink must have been called")
        }
    }

    impl TokenUsageSink for RecordingMetricsSink {
        fn record_token_usage(&self, _report: &LlmUsageReport) {}
    }

    impl LlmMetricsSink for RecordingMetricsSink {
        fn record_first_byte(&self, _duration_ms: f64, _context: &LlmCallContext) {}

        fn record_stream_stats(
            &self,
            stats: &StreamStats,
            completion_tokens: u32,
            _context: &LlmCallContext,
        ) {
            self.stats
                .lock()
                .expect("stats lock poisoned")
                .push((stats.clone(), completion_tokens));
        }

        fn record_request(
            &self,
            _duration_ms: f64,
            _success: bool,
            _error_kind: Option<&str>,
            _context: &LlmCallContext,
        ) {
        }

        fn record_retry(&self, _context: &LlmCallContext) {}
    }

    #[tokio::test]
    async fn metrics_sink_receives_stream_stats_once() {
        let inner = FakeStream {
            events: vec![
                text_event("done"),
                MessageStreamEvent::Usage(llm_suite_types::llm::MessageStreamUsage {
                    usage: TokenUsageStats {
                        prompt_tokens: 100,
                        completion_tokens: 20,
                        total_tokens: 120,
                        reasoning_tokens: None,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                        prompt_tokens_cost: None,
                        completion_tokens_cost: None,
                        total_cost: None,
                    },
                }),
                MessageStreamEvent::FinalMessage(llm_suite_types::llm::MessageStreamFinal {
                    message: assistant_message("done"),
                    usage: None,
                    stream_stats: None,
                    id: None,
                    model: None,
                    refusal: None,
                }),
            ],
        };
        let metrics = std::sync::Arc::new(RecordingMetricsSink::new());
        let mut stream =
            TokenRecordingStream::new(Box::new(inner), None, context(), None)
                .with_metrics_sink(Some(metrics.clone()));
        while stream.next().await.is_some() {}

        let (stats, completion_tokens) = metrics.last();
        assert_eq!(completion_tokens, 20);
        assert!(stats.chunk_count >= 2, "chunk_count: {}", stats.chunk_count);
        assert!(
            stats.time_to_first_chunk >= 0,
            "time_to_first_chunk: {}",
            stats.time_to_first_chunk
        );
        assert!(stats.total_duration >= 0);
        assert_eq!(metrics.stats.lock().expect("stats lock poisoned").len(), 1);
    }
}
