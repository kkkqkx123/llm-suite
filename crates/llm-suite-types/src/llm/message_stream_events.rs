use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageStreamEventType {
    Connect,
    Start,
    Stream,
    Text,
    InputJson,
    Message,
    FinalMessage,
    Error,
    Abort,
    End,
    ReasoningText,
    RedactedThinking,
    Refusal,
    Usage,
    ToolCallDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum MessageStreamEvent {
    Connect(MessageStreamConnect),
    Start(MessageStreamStart),
    Stream(MessageStreamChunk),
    Text(MessageStreamText),
    InputJson(MessageStreamInputJson),
    Message(MessageStreamMsg),
    FinalMessage(MessageStreamFinal),
    Error(MessageStreamError),
    Abort(MessageStreamAbort),
    End(MessageStreamEnd),
    ReasoningText(MessageStreamReasoning),
    RedactedThinking(MessageStreamRedactedThinking),
    Refusal(MessageStreamRefusal),
    Usage(MessageStreamUsage),
    ToolCallDelta(MessageStreamToolCallDelta),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamConnect {
    pub session_id: String,
}

/// Provider stream identity observed on the first chunk carrying it.
///
/// OpenAI Chat repeats `id`/`model` on every chunk, Anthropic sends them
/// once in `message_start`, Responses sends them in `response.created`.
/// Either field may be absent depending on the provider; the accumulator
/// keeps the first non-empty values and forwards only the first event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamStart {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamChunk {
    pub content: String,
}

/// Text increment with the accumulated snapshot of the whole streamed text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamText {
    pub text: String,
    #[serde(default)]
    pub snapshot: String,
}

/// Incremental tool_use JSON input. `index` refers to the tool call block the
/// fragment belongs to (Anthropic-style); `parsed_snapshot` is the best-effort
/// parse of the accumulated JSON buffer, refreshed by the message accumulator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamInputJson {
    pub partial_json: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parsed_snapshot: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamMsg {
    pub message: super::super::message::Message,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamFinal {
    pub message: super::super::message::Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::TokenUsageStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_stats: Option<super::StreamStats>,
    /// Provider message id observed mid-stream (`None` when the provider
    /// never sent one). Mirrors the non-streaming result envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider model observed mid-stream (`None` when never sent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Refusal text the provider produced instead of an answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamError {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamAbort {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamEnd {}

/// Reasoning increment with the accumulated snapshot.
///
/// `signature` carries an Anthropic `signature_delta` fragment for this
/// event (`None` for pure text deltas and for providers without signed
/// thinking); `signature_snapshot` is the accumulated signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamReasoning {
    pub reasoning: String,
    #[serde(default)]
    pub snapshot: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature_snapshot: Option<String>,
}

/// Refusal increment with the accumulated snapshot.
///
/// A refusal is the provider declining to answer. It travels on its own
/// channel so it never mixes into the answer text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamRefusal {
    pub refusal: String,
    #[serde(default)]
    pub snapshot: String,
}

/// Opaque provider payload that has to be replayed verbatim.
///
/// Anthropic redacts a thinking block when a safety filter intervenes; the
/// redacted payload travels in the block and the API expects it back in the
/// next turn. The provider never sends it as text, so it gets its own
/// channel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamRedactedThinking {
    pub data: String,
}

/// Token usage reported mid-stream (e.g. OpenAI's `include_usage` chunk or
/// Anthropic's `message_delta` usage).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamUsage {
    pub usage: super::TokenUsageStats,
}

/// Incremental streaming tool call data.
///
/// Providers deliver tool calls across multiple chunks. `index` groups the
/// fragments of one logical call; `id`/`name` are typically present only on
/// the first fragment (OpenAI) or on `content_block_start` (Anthropic).
/// `arguments` is an increment to be accumulated by the consumer. When
/// `is_snapshot` is true the fragment carries a complete tool call (e.g.
/// Gemini `functionCall` parts), and `arguments` is the fully serialized
/// JSON arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStreamToolCallDelta {
    pub index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-native call correlation id. The Responses API uses it to
    /// pair a function call with its output in a follow-up turn; formats
    /// without such an id leave it unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    /// Provider-signed opaque payload attached to the call. Gemini signs a
    /// function call with a thought signature that has to travel back with
    /// the call in the next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(default)]
    pub is_snapshot: bool,
}

/// Metadata key holding the provider signatures attached to the tool calls
/// of a reply, one entry per tool call in the order the calls are reported.
pub const THOUGHT_SIGNATURES_KEY: &str = "thought_signatures";
