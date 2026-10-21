//! Server-sent-events streaming for chat completions.
//!
//! Reads `POST {base_url}/chat/completions` with `"stream": true` through
//! `eventsource-stream` and surfaces content deltas. Multi-protocol codec
//! streaming stays in `wf-llm`; this module only covers the plain
//! OpenAI-compatible delta shape.

use std::collections::VecDeque;
use std::pin::Pin;

use eventsource_stream::{Event, EventStream, EventStreamError};
use futures::StreamExt;
use serde_json::Value;

use crate::error::{ChatError, Result};

/// One event from a streaming chat response.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamEvent {
    /// Incremental content fragment.
    Delta(String),
    /// Incremental tool call fragment. The `index` groups the fragments of
    /// one logical call.
    ToolCall {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    /// Provider-declared finish reason (e.g. `stop`, `length`).
    Finished(String),
}

/// SSE stream of [`ChatStreamEvent`] backed by a chat HTTP response.
///
/// The inner byte stream is boxed so callers name one concrete type.
pub struct ChatEventStream {
    inner: Pin<Box<dyn futures::Stream<Item = EventStreamItem> + Send>>,
    done: bool,
    /// Events decoded from a chunk that have not been handed out yet; one
    /// chunk can carry several tool call fragments.
    pending: VecDeque<ChatStreamEvent>,
}

type EventStreamItem = std::result::Result<Event, EventStreamError<reqwest::Error>>;

impl ChatEventStream {
    /// Wraps a streaming chat HTTP response.
    pub fn new(response: reqwest::Response) -> Self {
        let stream = EventStream::new(response.bytes_stream());
        Self {
            inner: Box::pin(stream),
            done: false,
            pending: VecDeque::new(),
        }
    }

    /// Returns the next stream event, or `None` when the stream ended.
    pub async fn next(&mut self) -> Option<Result<ChatStreamEvent>> {
        if self.done {
            return None;
        }
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match self.inner.next().await {
                None => {
                    self.done = true;
                    return None;
                }
                Some(Err(err)) => {
                    self.done = true;
                    return Some(Err(ChatError::Transport(err.to_string())));
                }
                Some(Ok(event)) => {
                    let data = event.data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" {
                        self.done = true;
                        return None;
                    }
                    match parse_stream_chunk(data) {
                        Ok(events) => {
                            if events.is_empty() {
                                continue;
                            }
                            let mut events = events.into_iter();
                            let first = events.next().expect("non-empty checked above");
                            self.pending.extend(events);
                            return Some(Ok(first));
                        }
                        Err(err) => {
                            self.done = true;
                            return Some(Err(err));
                        }
                    }
                }
            }
        }
    }
}

/// Parses one SSE data payload into the stream events it carries.
///
/// A chunk can pack several tool call fragments, so the payload maps to a
/// list of events.
pub fn parse_stream_chunk(data: &str) -> Result<Vec<ChatStreamEvent>> {
    let value: Value =
        serde_json::from_str(data).map_err(|err| ChatError::Decode(err.to_string()))?;
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());

    let mut events = Vec::new();

    if let Some(calls) = choice
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for call in calls {
            let function = call.get("function");
            events.push(ChatStreamEvent::ToolCall {
                index: call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize,
                id: call.get("id").and_then(Value::as_str).map(String::from),
                name: function
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .map(String::from),
                arguments: function
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .map(String::from),
            });
        }
    }

    if let Some(reason) = choice
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
    {
        events.push(ChatStreamEvent::Finished(reason.to_string()));
    }

    let delta = choice
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !delta.is_empty() {
        events.push(ChatStreamEvent::Delta(delta.to_string()));
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_delta() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        assert_eq!(
            parse_stream_chunk(data).expect("parsed"),
            vec![ChatStreamEvent::Delta("Hello".into())]
        );
    }

    #[test]
    fn skips_role_only_chunks() {
        let data = r#"{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        assert!(parse_stream_chunk(data).expect("parsed").is_empty());
    }

    #[test]
    fn surfaces_finish_reason() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(
            parse_stream_chunk(data).expect("parsed"),
            vec![ChatStreamEvent::Finished("stop".into())]
        );
    }

    #[test]
    fn surfaces_tool_call_fragments() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}"#;
        assert_eq!(
            parse_stream_chunk(data).expect("parsed"),
            vec![ChatStreamEvent::ToolCall {
                index: 0,
                id: Some("call_1".to_string()),
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"#.to_string()),
            }]
        );
    }

    #[test]
    fn rejects_malformed_payloads() {
        assert!(parse_stream_chunk("not json").is_err());
    }
}
