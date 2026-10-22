//! Conversion of internal messages into Gemini `contents`.

use std::collections::HashMap;

use crate::codecs::helpers::{block_name, native_block};
use crate::error::LlmResult;
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// The media type a document falls back to when its name says nothing.
const DOCUMENT_MEDIA_TYPE: &str = "application/pdf";

/// The media type an image falls back to when its name says nothing.
const IMAGE_MEDIA_TYPE: &str = "image/png";

/// Convert internal messages into Gemini `contents`.
///
/// Gemini only accepts the `user` and `model` roles; tool results ride
/// in a user message as `functionResponse` parts. The function name and
/// the call id a result refers to are taken from the tool call that
/// produced it, because Gemini matches the response back to the call.
pub fn convert_messages(messages: &[Message]) -> LlmResult<Vec<serde_json::Value>> {
    // The function name of every tool call in the history, keyed by the
    // call identity the result refers to.
    let mut function_names: HashMap<String, String> = HashMap::new();
    // The call id the provider assigned, keyed by the internal call
    // identity. Only a provider-assigned id is echoed back in the
    // function response; a locally generated one would not match any
    // call the provider knows.
    let mut provider_call_ids: HashMap<String, String> = HashMap::new();
    for msg in messages {
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                function_names.insert(call.id.clone(), call.function.name.clone());
                if let Some(ref call_id) = call.call_id {
                    provider_call_ids.insert(call.id.clone(), call_id.clone());
                }
            }
        }
        if let MessageContentValue::Rich(blocks) = &msg.content {
            for block in blocks {
                if let MessageContent::ToolUse { tool_use } = block {
                    function_names.insert(tool_use.id.clone(), tool_use.name.clone());
                }
            }
        }
    }

    let mut contents: Vec<serde_json::Value> = Vec::new();
    for msg in messages {
        let role = match msg.role {
            MessageRole::System => continue,
            MessageRole::User => "user",
            MessageRole::Assistant => "model",
            MessageRole::Tool => "user",
        };

        let parts = convert_parts(msg, &function_names, &provider_call_ids)?;
        // A message without any part is not a valid Gemini content.
        if parts.is_empty() {
            continue;
        }
        contents.push(serde_json::json!({"role": role, "parts": parts}));
    }
    Ok(contents)
}

/// Convert one message into Gemini parts.
fn convert_parts(
    msg: &Message,
    function_names: &HashMap<String, String>,
    provider_call_ids: &HashMap<String, String>,
) -> LlmResult<Vec<serde_json::Value>> {
    let mut parts: Vec<serde_json::Value> = Vec::new();
    // Whether the content already carried the function calls; the flat
    // call list is only used when it did not.
    let mut saw_tool_use = false;

    match &msg.content {
        MessageContentValue::Text(text) => {
            if msg.role == MessageRole::Tool {
                // A tool result is a function response; plain text would
                // hide the output from the model.
                let call_id = msg.tool_call_id.clone().unwrap_or_default();
                let name = resolve_function_name(msg, &call_id, function_names)?;
                let provider_id = provider_call_ids.get(&call_id).map(String::as_str);
                let is_error = msg
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("is_error"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                parts.push(function_response_part(
                    &name,
                    provider_id,
                    text,
                    None,
                    is_error,
                )?);
            } else if !text.is_empty() {
                parts.push(serde_json::json!({"text": text}));
            }
        }
        MessageContentValue::Rich(blocks) => {
            for block in blocks {
                match block {
                    MessageContent::Text { text } => {
                        parts.push(serde_json::json!({"text": text}));
                    }
                    MessageContent::ImageUrl { image_url } => {
                        // Inline base64 images become `inlineData`. Remote
                        // references only ride on `fileData` when they are
                        // URIs Gemini actually accepts (gs://, Files API,
                        // YouTube); anything else is an explicit error —
                        // an arbitrary https URL would be rejected by the
                        // server, and dropping the block silently would
                        // corrupt the conversation.
                        let url = image_url.url.as_str();
                        let part = if url.starts_with("data:") {
                            match super::super::helpers::resolve_image_payload(url)? {
                                super::super::helpers::ResolvedImage::Base64 {
                                    media_type,
                                    data,
                                } => serde_json::json!({
                                    "inlineData": {
                                        "mimeType": media_type,
                                        "data": data,
                                    }
                                }),
                                super::super::helpers::ResolvedImage::Url(_) => {
                                    unreachable!("a data URI never resolves to a remote URL")
                                }
                            }
                        } else if super::super::helpers::is_gemini_file_uri(url) {
                            serde_json::json!({
                                "fileData": {
                                    "mimeType": super::super::helpers::guess_media_type(
                                        url,
                                        IMAGE_MEDIA_TYPE,
                                    ),
                                    "fileUri": url,
                                }
                            })
                        } else {
                            return Err(crate::error::LlmError::ConfigError(format!(
                                "gemini fileData.fileUri only accepts gs://, Files API \
                                 or YouTube URIs; inline other remote media as a data \
                                 URI instead: {url}"
                            )));
                        };
                        parts.push(part);
                    }
                    MessageContent::InputAudio { input_audio } => {
                        // Audio rides on `inlineData` like any other media.
                        let mime = match input_audio.format.as_str() {
                            "wav" => "audio/wav",
                            "mp3" => "audio/mp3",
                            other => {
                                return Err(crate::error::LlmError::ConfigError(format!(
                                    "unsupported audio format `{other}` (expected wav|mp3)"
                                )));
                            }
                        };
                        parts.push(serde_json::json!({
                            "inlineData": {
                                "mimeType": mime,
                                "data": input_audio.data,
                            }
                        }));
                    }
                    MessageContent::FileInput { file } => {
                        // Documents carry on `inlineData` (base64) or
                        // `fileData` (gs://, Files API, YouTube only).
                        let part = match &file.source {
                            llm_suite_types::message::FileSource::Base64 { media_type, data } => {
                                let mut blob = serde_json::json!({
                                    "mimeType": media_type,
                                    "data": data,
                                });
                                if let Some(filename) = &file.filename {
                                    blob["displayName"] = serde_json::json!(filename);
                                }
                                serde_json::json!({"inlineData": blob})
                            }
                            llm_suite_types::message::FileSource::Url { url } => {
                                if super::super::helpers::is_gemini_file_uri(url) {
                                    let mut file_data = serde_json::json!({
                                        "mimeType": super::super::helpers::guess_media_type(
                                            url,
                                            DOCUMENT_MEDIA_TYPE,
                                        ),
                                        "fileUri": url,
                                    });
                                    if let Some(filename) = &file.filename {
                                        file_data["displayName"] = serde_json::json!(filename);
                                    }
                                    serde_json::json!({"fileData": file_data})
                                } else {
                                    return Err(crate::error::LlmError::ConfigError(format!(
                                        "gemini fileData.fileUri only accepts gs://, Files API \
                                         or YouTube URIs; inline other remote media as a data \
                                         URI instead: {url}"
                                    )));
                                }
                            }
                            llm_suite_types::message::FileSource::FileId { file_id } => {
                                return Err(crate::error::LlmError::ConfigError(format!(
                                    "gemini does not accept Files-API ids directly; upload the \
                                     file and pass its Files API URI instead: {file_id}"
                                )));
                            }
                            llm_suite_types::message::FileSource::Content { .. } => {
                                return Err(crate::error::LlmError::ConfigError(
                                    "gemini has no document built from content blocks; send the \
                                     document as inline data or as a file URI"
                                        .to_string(),
                                ));
                            }
                        };
                        parts.push(part);
                    }
                    MessageContent::ToolUse { tool_use } => {
                        let mut part = serde_json::json!({
                            "functionCall": {
                                "name": tool_use.name,
                                "args": tool_use.input,
                            }
                        });
                        // The provider signs a function call with an
                        // opaque thought signature; the signature has
                        // to travel back with the call in the next turn.
                        if let Some(signature) = &tool_use.thought_signature {
                            part["thoughtSignature"] = serde_json::json!(signature);
                        }
                        parts.push(part);
                        saw_tool_use = true;
                    }
                    MessageContent::ToolResult { tool_result } => {
                        let name =
                            resolve_function_name(msg, &tool_result.tool_use_id, function_names)?;
                        let provider_id = provider_call_ids
                            .get(&tool_result.tool_use_id)
                            .map(String::as_str);
                        // A function response has one text channel (the
                        // response object) and a media channel (the
                        // parts); the text of the blocks joins the text.
                        let text = tool_result.fallback_text();
                        let media = tool_result
                            .blocks
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|block| !matches!(block, MessageContent::Text { .. }))
                            .cloned()
                            .collect::<Vec<_>>();
                        parts.push(function_response_part(
                            &name,
                            provider_id,
                            &text,
                            Some(&media),
                            tool_result.is_error == Some(true),
                        )?);
                    }
                    // The reasoning text itself is not replayed; only the
                    // thought signature on a function call is.
                    MessageContent::Thinking { .. } | MessageContent::RedactedThinking { .. } => {}
                    MessageContent::Native {
                        provider,
                        kind,
                        payload,
                    } => {
                        parts.push(native_block(provider, "gemini", kind, payload)?);
                    }
                }
            }
        }
    }

    // Tool calls carried on the message itself become function call
    // parts; the rich blocks already hold them otherwise.
    if !saw_tool_use {
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                let mut part = serde_json::json!({
                    "functionCall": {
                        "name": call.function.name,
                        "args": args,
                    }
                });
                // The provider signs a function call with an opaque
                // thought signature; it has to travel back with the call
                // in the next turn.
                if let Some(signature) = &call.thought_signature {
                    part["thoughtSignature"] = serde_json::json!(signature);
                }
                parts.push(part);
            }
        }
    }

    Ok(parts)
}

/// The function name a tool result answers.
///
/// The name is taken from the result message itself when it carries one,
/// otherwise from the tool call the result refers to. A result whose
/// function cannot be named is rejected: Gemini matches the response
/// back to the call by name, so an unnamed response is meaningless.
fn resolve_function_name(
    msg: &Message,
    call_id: &str,
    function_names: &HashMap<String, String>,
) -> LlmResult<String> {
    if let Some(name) = msg.tool_name.as_ref().filter(|name| !name.is_empty()) {
        return Ok(name.clone());
    }
    if let Some(name) = function_names.get(call_id) {
        return Ok(name.clone());
    }
    Err(crate::error::LlmError::ConfigError(format!(
        "Gemini tool result for call `{call_id}` carries no function name; keep the assistant tool call in the history or name the result"
    )))
}

/// Build a `functionResponse` part.
///
/// The official `response` field is a JSON object, not a bare value: the
/// provider matches the response to the call by name and reads the
/// result from the `output` / `error` keys. A tool output that is not
/// already an object is therefore wrapped in `output`; a failed call is
/// reported under `error`.
///
/// The call id is echoed back when the provider assigned one, so the
/// response can be matched to the exact call it answers.
fn function_response_part(
    name: &str,
    call_id: Option<&str>,
    content: &str,
    blocks: Option<&[llm_suite_types::message::MessageContent]>,
    is_error: bool,
) -> LlmResult<serde_json::Value> {
    let response_value: serde_json::Value = if is_error {
        serde_json::json!({ "error": content })
    } else {
        match serde_json::from_str(content) {
            Ok(serde_json::Value::Object(object)) => serde_json::Value::Object(object),
            _ => serde_json::json!({ "output": content }),
        }
    };
    let mut response = serde_json::json!({
        "name": name,
        "response": response_value,
    });
    // Rich tool output rides on `functionResponse.parts` as inline blobs;
    // the `fileData` form is Vertex-only, so only inlineData is emitted.
    if let Some(blocks) = blocks {
        let mut parts: Vec<serde_json::Value> = Vec::new();
        for block in blocks {
            match block {
                llm_suite_types::message::MessageContent::ImageUrl { image_url } => {
                    if image_url.file_id.is_some() {
                        return Err(crate::error::LlmError::ConfigError(
                            "gemini does not accept a Files-API id in a function response; \
                             inline the image instead"
                                .to_string(),
                        ));
                    }
                    match super::super::helpers::resolve_image_payload(&image_url.url) {
                        Ok(super::super::helpers::ResolvedImage::Base64 { media_type, data }) => {
                            parts.push(serde_json::json!({
                                "inlineData": {
                                    "mimeType": media_type,
                                    "data": data,
                                }
                            }))
                        }
                        Ok(super::super::helpers::ResolvedImage::Url(_)) => {
                            return Err(crate::error::LlmError::ConfigError(
                                "a function response carries inline data only; inline the \
                                 image instead of referencing it by URL"
                                    .to_string(),
                            ));
                        }
                        Err(err) => return Err(err),
                    }
                }
                llm_suite_types::message::MessageContent::FileInput { file } => {
                    let llm_suite_types::message::FileSource::Base64 { media_type, data } =
                        &file.source
                    else {
                        return Err(crate::error::LlmError::ConfigError(
                            "a function response carries inline data only; inline the document \
                             instead of referencing it by id or URL"
                                .to_string(),
                        ));
                    };
                    parts.push(serde_json::json!({
                        "inlineData": {
                            "mimeType": media_type,
                            "data": data,
                        }
                    }));
                }
                other => {
                    return Err(crate::error::LlmError::ConfigError(format!(
                        "a gemini function response carries images and documents in its parts \
                         and its text in the response object, not a `{}` block",
                        block_name(other)
                    )));
                }
            }
        }
        if !parts.is_empty() {
            response["parts"] = serde_json::json!(parts);
        }
    }
    if let Some(call_id) = call_id.filter(|id| !id.is_empty()) {
        response["id"] = serde_json::json!(call_id);
    }
    Ok(serde_json::json!({"functionResponse": response}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::gemini_native::test_support::msg;
    use llm_suite_types::message::MessageRole;

    #[test]
    fn rich_image_blocks_become_inline_data_or_file_data() {
        let image_msg = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![
                MessageContent::Text {
                    text: "what is this".to_string(),
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "data:image/png;base64,AAAA".to_string(),
                        detail: None,
                        file_id: None,
                    },
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "gs://bucket/a.jpg".to_string(),
                        detail: None,
                        file_id: None,
                    },
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://generativelanguage.googleapis.com/v1beta/files/abc"
                            .to_string(),
                        detail: None,
                        file_id: None,
                    },
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://youtube.com/watch?v=abc".to_string(),
                        detail: None,
                        file_id: None,
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
        let entries = convert_messages(&[image_msg]).expect("must convert");
        assert_eq!(entries.len(), 1);
        let parts = entries[0]["parts"].as_array().expect("parts array");
        // Regression guard: none of the blocks may be silently dropped.
        assert_eq!(parts.len(), 5, "text + inlineData + 3 fileData");
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
        assert_eq!(parts[2]["fileData"]["fileUri"], "gs://bucket/a.jpg");
        assert_eq!(parts[2]["fileData"]["mimeType"], "image/jpeg");
        assert_eq!(
            parts[3]["fileData"]["fileUri"],
            "https://generativelanguage.googleapis.com/v1beta/files/abc"
        );
        assert_eq!(
            parts[4]["fileData"]["fileUri"],
            "https://youtube.com/watch?v=abc"
        );
    }

    #[test]
    fn an_arbitrary_https_image_url_is_rejected_not_mapped_to_file_data() {
        let image_msg = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::ImageUrl {
                image_url: llm_suite_types::message::ImageUrlContent {
                    url: "https://example.com/a.jpg".to_string(),
                    detail: None,
                    file_id: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let err = convert_messages(&[image_msg])
            .expect_err("a plain https URL must not become a fileUri");
        assert!(format!("{err}").contains("fileData.fileUri"), "{err}");
    }

    #[test]
    fn a_plain_text_tool_result_becomes_a_function_response() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]);
        let mut result = msg(MessageRole::Tool, "sunny");
        result.tool_call_id = Some("call_1".to_string());

        let contents = convert_messages(&[assistant, result]).expect("must convert");
        assert_eq!(contents[0]["role"], serde_json::json!("model"));
        assert_eq!(
            contents[0]["parts"][0]["functionCall"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(contents[1]["role"], serde_json::json!("user"));
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather"),
            "the function name comes from the call the result answers"
        );
        assert!(
            contents[1]["parts"][0]["functionResponse"]
                .get("id")
                .is_none(),
            "a call id the provider never issued must not be echoed back"
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["response"],
            serde_json::json!({"output": "sunny"}),
            "the response payload is an object, the text lands in `output`"
        );
    }

    #[test]
    fn a_tool_result_without_a_known_function_is_rejected() {
        let mut result = msg(MessageRole::Tool, "sunny");
        result.tool_call_id = Some("call_unknown".to_string());
        let err =
            convert_messages(&[result]).expect_err("an unnamed function response is meaningless");
        assert!(format!("{err}").contains("call_unknown"), "{err}");
    }

    #[test]
    fn a_rich_tool_use_block_becomes_a_function_call_part() {
        let mut assistant = msg(MessageRole::Assistant, "checking");
        assistant.content = MessageContentValue::Rich(vec![
            MessageContent::Text {
                text: "checking".to_string(),
            },
            MessageContent::ToolUse {
                tool_use: llm_suite_types::message::ToolUseContent {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "Beijing"}),
                    thought_signature: None,
                },
            },
        ]);
        let contents = convert_messages(&[assistant]).expect("must convert");
        let parts = contents[0]["parts"].as_array().expect("parts");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], serde_json::json!("checking"));
        assert_eq!(
            parts[1]["functionCall"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            parts[1]["functionCall"]["args"]["city"],
            serde_json::json!("Beijing")
        );
    }

    #[test]
    fn a_rich_tool_result_names_the_function_of_its_tool_use_block() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.content = MessageContentValue::Rich(vec![MessageContent::ToolUse {
            tool_use: llm_suite_types::message::ToolUseContent {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({}),
                thought_signature: None,
            },
        }]);
        let mut result = msg(MessageRole::Tool, "");
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "sunny".to_string(),
                blocks: None,
                is_error: None,
            },
        }]);

        let contents = convert_messages(&[assistant, result]).expect("must convert");
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            serde_json::json!("get_weather")
        );
        assert!(
            contents[1]["parts"][0]["functionResponse"]
                .get("id")
                .is_none(),
            "a call id the provider never issued must not be echoed back"
        );
    }

    #[test]
    fn a_message_without_parts_is_skipped() {
        let empty = msg(MessageRole::User, "");
        let contents =
            convert_messages(&[empty, msg(MessageRole::User, "hi")]).expect("must convert");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["parts"][0]["text"], serde_json::json!("hi"));
    }

    #[test]
    fn the_part_fields_use_the_official_camel_case_names() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]);
        let mut result = msg(MessageRole::Tool, "sunny");
        result.tool_call_id = Some("call_1".to_string());

        let contents = convert_messages(&[assistant, result]).expect("must convert");
        let call = &contents[0]["parts"][0];
        assert!(
            call.get("functionCall").is_some(),
            "the call part uses the official name: {contents:?}"
        );
        assert!(call.get("function_call").is_none());
        let response = &contents[1]["parts"][0];
        assert!(
            response.get("functionResponse").is_some(),
            "the response part uses the official name: {contents:?}"
        );
        assert!(response.get("function_response").is_none());
    }

    #[test]
    fn an_input_audio_block_becomes_inline_data() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::InputAudio {
                input_audio: llm_suite_types::message::InputAudioContent {
                    data: "QUJD".to_string(),
                    format: "wav".to_string(),
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let contents = convert_messages(&[user]).expect("must convert");
        let part = &contents[0]["parts"][0];
        assert_eq!(
            part["inlineData"]["mimeType"],
            serde_json::json!("audio/wav")
        );
        assert_eq!(part["inlineData"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn a_base64_file_input_becomes_inline_data() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Base64 {
                        media_type: "application/pdf".to_string(),
                        data: "QUJD".to_string(),
                    },
                    filename: None,
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let contents = convert_messages(&[user]).expect("must convert");
        let part = &contents[0]["parts"][0];
        assert_eq!(
            part["inlineData"]["mimeType"],
            serde_json::json!("application/pdf")
        );
        assert_eq!(part["inlineData"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn a_file_id_input_is_rejected() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::FileId {
                        file_id: "file_1".to_string(),
                    },
                    filename: None,
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let err = convert_messages(&[user]).expect_err("gemini needs a URI, not a Files-API id");
        assert!(format!("{err}").contains("Files API"), "{err}");
    }

    #[test]
    fn rich_tool_result_blocks_become_function_response_parts() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "screenshot".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]);
        let mut result = msg(MessageRole::Tool, "attached");
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "attached".to_string(),
                blocks: Some(vec![MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "data:image/png;base64,QUJD".to_string(),
                        detail: None,
                        file_id: None,
                    },
                }]),
                is_error: None,
            },
        }]);

        let contents = convert_messages(&[assistant, result]).expect("must convert");
        let part = &contents[1]["parts"][0]["functionResponse"];
        assert_eq!(part["response"]["output"], serde_json::json!("attached"));
        let parts = part["parts"].as_array().expect("rich parts array");
        assert_eq!(
            parts[0]["inlineData"]["mimeType"],
            serde_json::json!("image/png")
        );
        assert_eq!(parts[0]["inlineData"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn a_document_uri_is_not_mislabelled_as_an_image() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::FileInput {
                file: llm_suite_types::message::FileContent {
                    source: llm_suite_types::message::FileSource::Url {
                        url: "gs://bucket/report.pdf".to_string(),
                    },
                    filename: None,
                    detail: None,
                    citations_enabled: None,
                    context: None,
                },
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let contents = convert_messages(&[user]).expect("must convert");
        let part = &contents[0]["parts"][0];
        assert_eq!(
            part["fileData"]["mimeType"],
            serde_json::json!("application/pdf")
        );
        assert_eq!(
            part["fileData"]["fileUri"],
            serde_json::json!("gs://bucket/report.pdf")
        );
    }

    #[test]
    fn a_document_in_a_tool_result_becomes_inline_data() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "report".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]);
        let mut result = msg(MessageRole::Tool, "attached");
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "attached".to_string(),
                blocks: Some(vec![MessageContent::FileInput {
                    file: llm_suite_types::message::FileContent {
                        source: llm_suite_types::message::FileSource::Base64 {
                            media_type: "application/pdf".to_string(),
                            data: "QUJD".to_string(),
                        },
                        filename: None,
                        detail: None,
                        citations_enabled: None,
                        context: None,
                    },
                }]),
                is_error: None,
            },
        }]);

        let contents = convert_messages(&[assistant, result]).expect("must convert");
        let parts = contents[1]["parts"][0]["functionResponse"]["parts"]
            .as_array()
            .expect("rich parts array");
        assert_eq!(
            parts[0]["inlineData"]["mimeType"],
            serde_json::json!("application/pdf")
        );
        assert_eq!(parts[0]["inlineData"]["data"], serde_json::json!("QUJD"));
    }

    #[test]
    fn a_failed_tool_result_is_reported_under_the_error_key() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![llm_suite_types::message::LlmToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            call_id: None,
            function: llm_suite_types::message::LlmFunctionCall {
                name: "boom".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]);
        let mut result = msg(MessageRole::Tool, "it exploded");
        result.tool_call_id = Some("call_1".to_string());
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "it exploded".to_string(),
                blocks: None,
                is_error: Some(true),
            },
        }]);

        let contents = convert_messages(&[assistant, result]).expect("must convert");
        let response = &contents[1]["parts"][0]["functionResponse"]["response"];
        assert_eq!(response["error"], serde_json::json!("it exploded"));
        assert!(response.get("output").is_none());
    }

    #[test]
    fn a_native_part_is_replayed_verbatim() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::Native {
                provider: "gemini".to_string(),
                kind: "video_metadata".to_string(),
                payload: serde_json::json!({
                    "inlineData": {"mimeType": "video/mp4", "data": "QUJD"},
                    "videoMetadata": {"startOffset": "0s", "endOffset": "10s"},
                }),
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let contents = convert_messages(&[user]).expect("must convert");
        let part = &contents[0]["parts"][0];
        assert_eq!(part["videoMetadata"]["endOffset"], serde_json::json!("10s"));
        assert_eq!(
            part["inlineData"]["mimeType"],
            serde_json::json!("video/mp4")
        );
    }

    #[test]
    fn a_native_part_of_another_provider_is_rejected() {
        let user = Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Rich(vec![MessageContent::Native {
                provider: "anthropic".to_string(),
                kind: "search_result".to_string(),
                payload: serde_json::json!({"type": "search_result"}),
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        };
        let err = convert_messages(&[user]).expect_err("a block of another provider is refused");
        assert!(format!("{err}").contains("anthropic"), "{err}");
    }
}
