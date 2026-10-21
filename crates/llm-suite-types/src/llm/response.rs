use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamStats {
    pub chunk_count: u32,
    pub time_to_first_chunk: i64,
    pub stream_duration: i64,
    pub total_duration: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmResult {
    pub id: Option<String>,
    pub model: String,
    pub content: Option<String>,
    pub message: super::super::message::Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<super::super::message::LlmToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::TokenUsageStats>,
    /// Stop value of the turn.
    ///
    /// The two OpenAI formats and Gemini report the same vocabulary
    /// (`stop` / `length` / `content_filter` / `tool_calls`); Anthropic
    /// reports its own `stop_reason` set (`end_turn` / `tool_use` /
    /// `max_tokens` / `stop_sequence` / `pause_turn` / `refusal` /
    /// `model_context_window_exceeded`). The raw lifecycle state of a
    /// Responses call stays available in `metadata` under `status`.
    pub finish_reason: Option<String>,
    pub duration: i64,
    /// Reasoning text the provider produced. The same text is carried on
    /// `message.thinking`, so replaying the message in a follow-up turn
    /// keeps it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Provider signature for the reasoning block (Anthropic extended
    /// thinking). Mirrors the assistant message signature for callers that
    /// only read the envelope.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Model refusal text when the provider declines to answer. Distinct from
    /// `content` so callers can tell an empty answer apart from a refusal.
    /// Populated from the Chat Completions `message.refusal` field and from
    /// Responses API `refusal` output items (non-streaming).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<super::super::Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_stats: Option<StreamStats>,
}
