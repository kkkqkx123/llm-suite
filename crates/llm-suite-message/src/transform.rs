use llm_suite_types::message::{
    FileSource, LlmToolCall, Message, MessageContent, MessageContentValue, MessageRole,
};

/// Capabilities of the target model a history is moved to.
///
/// `same_model` means the history stays on the model that produced it:
/// thinking signatures and provider-native blocks are kept verbatim.
/// A cross-model move drops them, because a signature minted by one
/// provider is rejected (or worse, misread) by another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformTarget {
    pub supports_image: bool,
    pub same_model: bool,
}

impl TransformTarget {
    pub fn same_model(supports_image: bool) -> Self {
        Self {
            supports_image,
            same_model: true,
        }
    }

    pub fn cross_model(supports_image: bool) -> Self {
        Self {
            supports_image,
            same_model: false,
        }
    }

    /// Capabilities derived from the wire format only. Every built-in
    /// format carries images, so this returns image-capable; a catalog
    /// lookup on [`llm_suite_types::llm::ModelMetadata`] refines it when
    /// the caller knows the concrete model.
    pub fn for_format(format: &llm_suite_types::llm::LlmFormat, same_model: bool) -> Self {
        let _ = format;
        Self {
            supports_image: true,
            same_model,
        }
    }
}

/// Maximum tool-call id length accepted across providers (Anthropic
/// truncates past 64 characters). Longer ids are shortened deterministically
/// so the call/result correlation survives the move.
pub const MAX_TOOL_CALL_ID_LEN: usize = 64;

/// Move a history to the target model.
///
/// Rules (all `O(n)`, allocation proportional to the history):
/// - same model: the history is returned unchanged (clone);
/// - cross model: thinking blocks (`Thinking`, `RedactedThinking`,
///   `thinking`/`thinking_signature` fields, `thought_signature` on calls)
///   are dropped, because a signature minted by one provider is opaque to
///   another; provider-native blocks are dropped for the same reason;
/// - image/file blocks are downgraded to a placeholder text line when the
///   target cannot carry images, never silently dropped;
/// - tool-call ids longer than [`MAX_TOOL_CALL_ID_LEN`] are shortened, and
///   the matching tool results follow the same mapping;
/// - an assistant turn with tool calls and no following result gets a
///   synthesized empty tool result so the next provider never sees an
///   orphan call.
///
/// System messages are kept in place: every codec already routes them to
/// its own channel (Anthropic `system`, OpenAI `system`/`developer`,
/// Gemini `systemInstruction`), so no reordering is needed here.
pub fn transform_for_target(messages: &[Message], target: &TransformTarget) -> Vec<Message> {
    if target.same_model && target.supports_image {
        return messages.to_vec();
    }

    let mut id_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut normalize_id =
        |id: &str, map: &mut std::collections::HashMap<String, String>| -> String {
            if id.len() <= MAX_TOOL_CALL_ID_LEN {
                return id.to_string();
            }
            if let Some(mapped) = map.get(id) {
                return mapped.clone();
            }
            let short = shorten_tool_id(id);
            map.insert(id.to_string(), short.clone());
            short
        };

    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 2);
    for msg in messages {
        let mut converted = msg.clone();

        if !target.same_model {
            converted.thinking = None;
            converted.thinking_signature = None;
            if let Some(calls) = converted.tool_calls.as_mut() {
                remap_calls(calls, &mut id_map, &mut normalize_id);
                for call in calls.iter_mut() {
                    call.thought_signature = None;
                }
            }
            converted.content = strip_cross_model_blocks(&converted.content, target);
        } else {
            if let Some(calls) = converted.tool_calls.as_mut() {
                remap_calls(calls, &mut id_map, &mut normalize_id);
            }
            if !target.supports_image {
                converted.content = downgrade_images(&converted.content);
            }
        }

        if let Some(tool_call_id) = converted.tool_call_id.as_ref() {
            if tool_call_id.len() > MAX_TOOL_CALL_ID_LEN {
                converted.tool_call_id = Some(normalize_id(tool_call_id, &mut id_map));
            }
        }
        if let MessageContentValue::Rich(blocks) = &converted.content {
            let mut remapped = blocks.clone();
            remap_block_tool_ids(&mut remapped, &mut id_map, &mut normalize_id);
            converted.content = MessageContentValue::Rich(remapped);
        }

        out.push(converted);
    }

    synthesize_missing_tool_results(&mut out);
    out
}

fn remap_calls(
    calls: &mut [LlmToolCall],
    id_map: &mut std::collections::HashMap<String, String>,
    normalize: &mut impl FnMut(&str, &mut std::collections::HashMap<String, String>) -> String,
) {
    for call in calls.iter_mut() {
        if call.id.len() > MAX_TOOL_CALL_ID_LEN {
            call.id = normalize(&call.id.clone(), id_map);
        }
    }
}

fn remap_block_tool_ids(
    blocks: &mut [MessageContent],
    id_map: &mut std::collections::HashMap<String, String>,
    normalize: &mut impl FnMut(&str, &mut std::collections::HashMap<String, String>) -> String,
) {
    for block in blocks.iter_mut() {
        match block {
            MessageContent::ToolUse { tool_use } => {
                if tool_use.id.len() > MAX_TOOL_CALL_ID_LEN {
                    tool_use.id = normalize(&tool_use.id.clone(), id_map);
                }
                tool_use.thought_signature = None;
            }
            MessageContent::ToolResult { tool_result } => {
                if tool_result.tool_use_id.len() > MAX_TOOL_CALL_ID_LEN {
                    tool_result.tool_use_id = normalize(&tool_result.tool_use_id.clone(), id_map);
                }
            }
            _ => {}
        }
    }
}

/// Deterministic shortening: keep a readable head and a hash tail so two
/// distinct long ids never collapse onto one short id.
fn shorten_tool_id(id: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    let digest = format!("{:016x}", hasher.finish());
    let head_len = MAX_TOOL_CALL_ID_LEN.saturating_sub(digest.len() + 1).max(8);
    let head: String = id.chars().take(head_len).collect();
    format!("{head}-{digest}")
}

/// Drop cross-model blocks: thinking payloads and provider-native blocks.
/// Image blocks survive here; the image downgrade below handles them.
fn strip_cross_model_blocks(
    content: &MessageContentValue,
    target: &TransformTarget,
) -> MessageContentValue {
    match content {
        MessageContentValue::Text(_) => content.clone(),
        MessageContentValue::Rich(blocks) => {
            let mut kept: Vec<MessageContent> = Vec::with_capacity(blocks.len());
            let mut pending_text: Vec<String> = Vec::new();
            for block in blocks {
                match block {
                    MessageContent::Thinking { .. } | MessageContent::RedactedThinking { .. } => {}
                    MessageContent::Native { .. } => {}
                    MessageContent::ImageUrl { image_url } if !target.supports_image => {
                        pending_text.push(format!("[image: {}]", image_reference(&image_url.url)));
                    }
                    MessageContent::FileInput { file } if !target.supports_image => {
                        pending_text.push(format!("[file: {}]", file_reference(file)));
                    }
                    MessageContent::Text { text } => {
                        if !text.trim().is_empty() || kept.is_empty() {
                            kept.push(block.clone());
                        }
                    }
                    other => kept.push(other.clone()),
                }
            }
            if !pending_text.is_empty() {
                kept.push(MessageContent::Text {
                    text: pending_text.join("\n"),
                });
            }
            if kept.is_empty() {
                MessageContentValue::Text(String::new())
            } else {
                MessageContentValue::Rich(kept)
            }
        }
    }
}

/// Downgrade image blocks of a same-model move whose target cannot carry
/// images. Cross-model moves already pass through
/// `strip_cross_model_blocks`; this covers the remaining case.
fn downgrade_images(content: &MessageContentValue) -> MessageContentValue {
    match content {
        MessageContentValue::Text(_) => content.clone(),
        MessageContentValue::Rich(blocks) => {
            if !blocks.iter().any(|b| {
                matches!(
                    b,
                    MessageContent::ImageUrl { .. } | MessageContent::FileInput { .. }
                )
            }) {
                return content.clone();
            }
            let mut kept: Vec<MessageContent> = Vec::new();
            let mut placeholders: Vec<String> = Vec::new();
            for block in blocks {
                match block {
                    MessageContent::ImageUrl { image_url } => {
                        placeholders.push(format!("[image: {}]", image_reference(&image_url.url)));
                    }
                    MessageContent::FileInput { file } => {
                        placeholders.push(format!("[file: {}]", file_reference(file)));
                    }
                    other => kept.push(other.clone()),
                }
            }
            if !placeholders.is_empty() {
                kept.push(MessageContent::Text {
                    text: placeholders.join("\n"),
                });
            }
            MessageContentValue::Rich(kept)
        }
    }
}

fn image_reference(url: &str) -> String {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, payload) = rest.split_once(',').unwrap_or((rest, ""));
        let media_type = meta.trim_end_matches(";base64");
        let media_type = if media_type.is_empty() {
            "application/octet-stream"
        } else {
            media_type
        };
        format!("{media_type} ({} base64 chars)", payload.len())
    } else {
        url.to_string()
    }
}

fn file_reference(file: &llm_suite_types::message::FileContent) -> String {
    match &file.source {
        FileSource::Base64 { media_type, data } => {
            format!("{media_type} ({} base64 chars)", data.len())
        }
        FileSource::Url { url } => url.clone(),
        FileSource::FileId { file_id } => file_id.clone(),
        FileSource::Content { blocks } => format!("{} content blocks", blocks.len()),
    }
}

/// Append a synthesized empty tool result after every assistant turn whose
/// calls have no matching result later in the history.
fn synthesize_missing_tool_results(messages: &mut Vec<Message>) {
    let mut answered: std::collections::HashSet<String> = std::collections::HashSet::new();
    for msg in messages.iter() {
        match &msg.content {
            MessageContentValue::Rich(blocks) => {
                for block in blocks {
                    if let MessageContent::ToolResult { tool_result } = block {
                        answered.insert(tool_result.tool_use_id.clone());
                    }
                }
            }
            _ => {}
        }
        if msg.role == MessageRole::Tool {
            if let Some(id) = msg.tool_call_id.as_ref() {
                answered.insert(id.clone());
            }
        }
    }

    let mut insertions: Vec<(usize, Message)> = Vec::new();
    for (index, msg) in messages.iter().enumerate() {
        let calls: Vec<String> = match &msg.tool_calls {
            Some(calls) => calls.iter().map(|c| c.id.clone()).collect(),
            None => match &msg.content {
                MessageContentValue::Rich(blocks) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        MessageContent::ToolUse { tool_use } => Some(tool_use.id.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            },
        };
        for id in calls {
            if !answered.contains(&id) {
                answered.insert(id.clone());
                insertions.push((
                    index + 1 + insertions.len(),
                    Message::tool_result(id, None, String::new(), false),
                ));
            }
        }
    }
    for (index, msg) in insertions.into_iter() {
        if index <= messages.len() {
            messages.insert(index, msg);
        } else {
            messages.push(msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::message::{ImageUrlContent, ToolResultContent, ToolUseContent};

    fn text(role: MessageRole, text: &str) -> Message {
        Message::text(role, text)
    }

    #[test]
    fn same_model_with_images_is_a_noop_clone() {
        let history = vec![text(MessageRole::User, "hi")];
        let target = TransformTarget::same_model(true);
        let moved = transform_for_target(&history, &target);
        assert_eq!(moved, history);
    }

    #[test]
    fn cross_model_drops_thinking_and_signatures() {
        let assistant = Message {
            thinking: Some("hmm".to_string()),
            thinking_signature: Some("sig".to_string()),
            tool_calls: Some(vec![LlmToolCall {
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "w".to_string(),
                    arguments: "{}".to_string(),
                },
                call_id: None,
                thought_signature: Some("gemini-sig".to_string()),
            }]),
            ..text(MessageRole::Assistant, "working")
        };
        let target = TransformTarget::cross_model(true);
        let moved = transform_for_target(&[assistant], &target);
        assert_eq!(moved.len(), 2, "an orphan call gains a synthesized result");
        assert_eq!(moved[0].thinking, None);
        assert_eq!(moved[0].thinking_signature, None);
        let calls = moved[0].tool_calls.as_ref().expect("calls kept");
        assert_eq!(calls[0].thought_signature, None);
    }

    #[test]
    fn images_downgrade_to_placeholders_without_payload() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: ImageUrlContent::url("data:image/png;base64,QUJDREFGRw==".to_string()),
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let target = TransformTarget::cross_model(false);
        let moved = transform_for_target(&[user], &target);
        assert_eq!(moved.len(), 1);
        let content = moved[0].text_content();
        assert!(content.contains("[image:"), "{content}");
        assert!(!content.contains("QUJDREFGRw"), "payload must stay out");
    }

    #[test]
    fn long_tool_ids_are_shortened_on_both_sides() {
        let long_id = "a".repeat(100);
        let assistant = Message {
            tool_calls: Some(vec![LlmToolCall {
                id: long_id.clone(),
                r#type: "function".to_string(),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "w".to_string(),
                    arguments: "{}".to_string(),
                },
                call_id: None,
                thought_signature: None,
            }]),
            ..text(MessageRole::Assistant, "call")
        };
        let tool = Message {
            content: MessageContentValue::Rich(vec![MessageContent::ToolResult {
                tool_result: ToolResultContent::text(long_id.clone(), "ok".to_string()),
            }]),
            ..text(MessageRole::Tool, "ok")
        };
        let target = TransformTarget::cross_model(true);
        let moved = transform_for_target(&[assistant, tool], &target);
        let call_id = moved[0].tool_calls.as_ref().expect("calls")[0].id.clone();
        assert!(call_id.len() <= MAX_TOOL_CALL_ID_LEN, "{call_id}");
        let result_id = match &moved[1].content {
            MessageContentValue::Rich(blocks) => match &blocks[0] {
                MessageContent::ToolResult { tool_result } => tool_result.tool_use_id.clone(),
                _ => panic!("expected tool result"),
            },
            _ => panic!("expected rich"),
        };
        assert_eq!(call_id, result_id, "call and result stay correlated");
    }

    #[test]
    fn orphan_calls_gain_synthesized_results() {
        let assistant = Message {
            tool_calls: Some(vec![LlmToolCall {
                id: "orphan".to_string(),
                r#type: "function".to_string(),
                function: llm_suite_types::message::LlmFunctionCall {
                    name: "w".to_string(),
                    arguments: "{}".to_string(),
                },
                call_id: None,
                thought_signature: None,
            }]),
            ..text(MessageRole::Assistant, "call")
        };
        let target = TransformTarget::cross_model(true);
        let moved = transform_for_target(&[assistant], &target);
        assert_eq!(moved.len(), 2);
        assert_eq!(moved[1].role, MessageRole::Tool);
        assert_eq!(moved[1].tool_call_id.as_deref(), Some("orphan"));
    }

    #[test]
    fn native_blocks_do_not_leak_across_models() {
        let assistant = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![
                MessageContent::Text {
                    text: "hi".to_string(),
                },
                MessageContent::Native {
                    provider: "anthropic".to_string(),
                    kind: "search_result".to_string(),
                    payload: serde_json::json!({"title": "x"}),
                },
                MessageContent::ToolUse {
                    tool_use: ToolUseContent {
                        id: "c1".to_string(),
                        name: "w".to_string(),
                        input: serde_json::json!({}),
                        thought_signature: None,
                    },
                },
            ]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let target = TransformTarget::cross_model(true);
        let moved = transform_for_target(&[assistant], &target);
        let has_native = match &moved[0].content {
            MessageContentValue::Rich(blocks) => blocks
                .iter()
                .any(|b| matches!(b, MessageContent::Native { .. })),
            _ => false,
        };
        assert!(!has_native, "provider-native blocks must not cross models");
    }
}
