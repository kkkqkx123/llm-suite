use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// The detail level an image input is rendered with.
///
/// The OpenAI wire formats accept these four values; the Anthropic and the
/// Gemini formats have no equivalent and ignore the hint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
    Original,
}

/// The detail level a file input is rendered with.
///
/// A file accepts three values while an image accepts a fourth one
/// (`original`), so the two value sets stay apart instead of sharing a
/// single loose check.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileDetail {
    Auto,
    Low,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmToolCall {
    pub id: String,
    pub r#type: String,
    pub function: LlmFunctionCall,
    /// Provider-native tool call correlation id. OpenAI Responses API uses
    /// `call_id` to associate a `function_call` output item with its
    /// `function_call_output` in follow-up turns; Chat Completions has no
    /// equivalent and leaves this `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// The provider signature of the call (the Gemini thought signature).
    /// It has to travel back with the call in the next turn, so it rides on
    /// the call itself instead of a positional side channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUseContent {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
    /// The provider signature of the call (the Gemini thought signature).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultContent {
    pub tool_use_id: String,
    pub content: String,
    /// Rich tool-result content (images, documents, …). When present it
    /// takes precedence over the plain-text `content`; codecs that cannot
    /// carry rich blocks fall back to `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocks: Option<Vec<MessageContent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolResultContent {
    /// Plain-text tool result (the canonical shape).
    pub fn text(tool_use_id: String, content: String) -> Self {
        Self {
            tool_use_id,
            content,
            blocks: None,
            is_error: None,
        }
    }

    /// The text a codec should fall back to: the rich blocks flattened to
    /// text, or the plain-text content.
    ///
    /// A result whose blocks carry no text at all (a tool that returned an
    /// image only) still has its plain-text content; the fallback never
    /// degrades a result into an empty string.
    pub fn fallback_text(&self) -> String {
        match &self.blocks {
            None => self.content.clone(),
            Some(blocks) => {
                let text = blocks
                    .iter()
                    .filter_map(|b| match b {
                        MessageContent::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if text.is_empty() {
                    self.content.clone()
                } else {
                    text
                }
            }
        }
    }

    /// The plain-text content of the result, unless the rich blocks carry
    /// text of their own.
    ///
    /// The plain text leads the rich blocks on the wire. When the blocks
    /// already carry text, the plain text would repeat it, so it is left
    /// out and the blocks speak for themselves.
    pub fn leading_text(&self) -> Option<&str> {
        let blocks_carry_text = self.blocks.as_ref().is_some_and(|blocks| {
            blocks
                .iter()
                .any(|block| matches!(block, MessageContent::Text { .. }))
        });
        if blocks_carry_text || self.content.is_empty() {
            None
        } else {
            Some(self.content.as_str())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrlContent {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
    /// Files-API reference (Anthropic `source.type=file`, OpenAI Responses
    /// `input_image.file_id`). Mutually exclusive with `url`; codecs that
    /// cannot carry a file reference reject the block explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

impl ImageUrlContent {
    /// A remote-URL (or data URI) image reference.
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
            file_id: None,
        }
    }

    /// A Files-API image reference.
    pub fn file_id(file_id: impl Into<String>) -> Self {
        Self {
            url: String::new(),
            detail: None,
            file_id: Some(file_id.into()),
        }
    }
}

/// Audio carried inline in a user message (OpenAI Chat `input_audio`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputAudioContent {
    /// Base64-encoded audio payload.
    pub data: String,
    /// Container format: `wav` or `mp3`.
    pub format: String,
}

/// How a file/document input references its bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FileSource {
    /// Raw bytes, base64-encoded, with an IANA media type.
    Base64 { media_type: String, data: String },
    /// A remote URL (OpenAI Responses `input_file.file_url`).
    Url { url: String },
    /// A Files-API upload id.
    FileId { file_id: String },
    /// A document built from content blocks (the Anthropic document source
    /// `content`).
    Content { blocks: Vec<MessageContent> },
}

/// A document/file input block (OpenAI `file` / `input_file`, Anthropic
/// `document`, Gemini `inlineData`/`fileData` for non-image media).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileContent {
    pub source: FileSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// The detail level the Responses API renders the file with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<FileDetail>,
    /// The Anthropic document blocks ask for citations with this flag; it
    /// only has an effect on a document block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations_enabled: Option<bool>,
    /// The context the Anthropic document block shows to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrlContent,
    },
    /// Inline audio input (OpenAI Chat `input_audio`).
    InputAudio {
        input_audio: InputAudioContent,
    },
    /// A document / file input block.
    FileInput {
        file: FileContent,
    },
    ToolUse {
        tool_use: ToolUseContent,
    },
    ToolResult {
        tool_result: ToolResultContent,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Thinking block the provider redacted. The payload is opaque and must
    /// be sent back verbatim in later turns.
    RedactedThinking {
        data: String,
    },
    /// A provider-native content block.
    ///
    /// The blocks every provider shares have their own variant; the blocks
    /// only one provider has (server tool results, browsing state, container
    /// uploads, video metadata) travel as the provider's own JSON so a reply
    /// can be replayed verbatim. A codec that does not serve the provider
    /// rejects the block instead of dropping it.
    Native {
        /// The provider family the block belongs to: `openai`, `anthropic`
        /// or `gemini`.
        provider: String,
        /// The provider-native block name: the content block type for the
        /// OpenAI and the Anthropic formats, the part field for Gemini.
        kind: String,
        /// The provider-native block body.
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub id: crate::Id,
    pub role: MessageRole,
    pub content: MessageContentValue,
    pub timestamp: crate::Timestamp,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<LlmToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Provider signature for the thinking block (Anthropic extended
    /// thinking). Required to replay the thinking block in later turns;
    /// unsigned thinking is never replayed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<crate::Metadata>,
}

impl Message {
    fn fresh_id() -> crate::Id {
        uuid::Uuid::now_v7().to_string()
    }

    fn fresh_timestamp() -> crate::Timestamp {
        chrono::Utc::now().timestamp_millis()
    }

    /// Canonical plain-text system message.
    pub fn system_text(text: String) -> Self {
        Self {
            id: Self::fresh_id(),
            role: MessageRole::System,
            content: MessageContentValue::Text(text),
            timestamp: Self::fresh_timestamp(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    /// Canonical plain-text user message.
    pub fn user_text(text: String) -> Self {
        Self {
            id: Self::fresh_id(),
            role: MessageRole::User,
            content: MessageContentValue::Text(text),
            timestamp: Self::fresh_timestamp(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    /// Canonical tool result message carrying the call id, optional tool
    /// name and error flag.
    pub fn tool_result(
        tool_call_id: String,
        tool_name: Option<String>,
        content: String,
        is_error: bool,
    ) -> Self {
        let metadata = is_error.then(|| {
            std::collections::HashMap::from([(
                "is_error".to_string(),
                serde_json::Value::Bool(true),
            )])
        });
        Self {
            id: Self::fresh_id(),
            role: MessageRole::Tool,
            content: MessageContentValue::Text(content),
            timestamp: Self::fresh_timestamp(),
            tool_call_id: Some(tool_call_id),
            tool_name,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata,
        }
    }

    /// Canonical plain-text message with an explicit role.
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            id: Self::fresh_id(),
            role,
            content: MessageContentValue::Text(text.into()),
            timestamp: Self::fresh_timestamp(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    /// Plain-text view of the content, joining rich text parts.
    pub fn text_content(&self) -> String {
        match &self.content {
            MessageContentValue::Text(text) => text.clone(),
            MessageContentValue::Rich(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    MessageContent::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContentValue {
    Text(String),
    Rich(Vec<MessageContent>),
}

// `#[serde(untagged)]` cannot deserialize a variant that is itself an
// internally tagged enum (`#[serde(tag = "type")]`): the buffered replay
// machinery rejects it with "data did not match any variant". A custom
// Deserialize over `serde_json::Value` keeps the round trip working and is
// lenient towards the reference formats: a bare string (plain text), an
// array of typed content parts, or a single content part object.
impl<'de> Deserialize<'de> for MessageContentValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(text) => Ok(MessageContentValue::Text(text)),
            serde_json::Value::Array(items) => {
                serde_json::from_value(serde_json::Value::Array(items))
                    .map(MessageContentValue::Rich)
                    .map_err(D::Error::custom)
            }
            serde_json::Value::Object(_) => serde_json::from_value(value)
                .map(|content: MessageContent| MessageContentValue::Rich(vec![content]))
                .map_err(D::Error::custom),
            other => Err(D::Error::custom(format!(
                "invalid message content value: {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rich content of a message survives the JSON round trip.
    #[test]
    fn the_rich_content_survives_the_json_round_trip() {
        let blocks = vec![
            MessageContent::Text {
                text: "look".to_string(),
            },
            MessageContent::ImageUrl {
                image_url: ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: Some(ImageDetail::High),
                    file_id: None,
                },
            },
            MessageContent::FileInput {
                file: FileContent {
                    source: FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "QUJD".to_string(),
                    },
                    filename: Some("report.pdf".to_string()),
                    detail: Some(FileDetail::Low),
                    citations_enabled: Some(true),
                    context: Some("a report".to_string()),
                },
            },
            MessageContent::ToolUse {
                tool_use: ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Beijing"}),
                    thought_signature: Some("sig-1".to_string()),
                },
            },
            MessageContent::Native {
                provider: "anthropic".to_string(),
                kind: "search_result".to_string(),
                payload: serde_json::json!({"type": "search_result", "title": "Example"}),
            },
        ];
        let value = MessageContentValue::Rich(blocks.clone());
        let json = serde_json::to_string(&value).expect("must serialize");
        let parsed: MessageContentValue = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(parsed, value);
    }

    /// A document built from blocks keeps its blocks through the round trip.
    #[test]
    fn a_document_built_from_blocks_survives_the_round_trip() {
        let value = MessageContentValue::Rich(vec![MessageContent::FileInput {
            file: FileContent {
                source: FileSource::Content {
                    blocks: vec![MessageContent::Text {
                        text: "Chapter 1".to_string(),
                    }],
                },
                filename: None,
                detail: None,
                citations_enabled: None,
                context: None,
            },
        }]);
        let json = serde_json::to_string(&value).expect("must serialize");
        let parsed: MessageContentValue = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(parsed, value);
    }

    /// A tool result whose blocks carry no text still has its content.
    #[test]
    fn a_media_only_tool_result_keeps_its_text() {
        let result = ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "attached".to_string(),
            blocks: Some(vec![MessageContent::ImageUrl {
                image_url: ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                    file_id: None,
                },
            }]),
            is_error: None,
        };
        assert_eq!(result.fallback_text(), "attached");
    }

    /// The plain text of a result leads the rich blocks, unless the blocks
    /// carry text of their own.
    #[test]
    fn the_leading_text_of_a_result_follows_its_blocks() {
        let media_only = ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "attached".to_string(),
            blocks: Some(vec![MessageContent::ImageUrl {
                image_url: ImageUrlContent {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                    file_id: None,
                },
            }]),
            is_error: None,
        };
        assert_eq!(media_only.leading_text(), Some("attached"));

        let with_text = ToolResultContent {
            tool_use_id: "call_1".to_string(),
            content: "attached".to_string(),
            blocks: Some(vec![MessageContent::Text {
                text: "the report".to_string(),
            }]),
            is_error: None,
        };
        assert_eq!(
            with_text.leading_text(),
            None,
            "the blocks speak for themselves; the plain text would repeat them"
        );

        let plain = ToolResultContent::text("call_1".to_string(), "sunny".to_string());
        assert_eq!(
            plain.leading_text(),
            Some("sunny"),
            "a plain result is its own leading text"
        );
    }
}
