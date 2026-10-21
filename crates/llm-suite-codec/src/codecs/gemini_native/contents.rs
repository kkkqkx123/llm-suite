//! Conversion of internal messages into Gemini `contents`.

use std::collections::HashMap;

use crate::error::LlmResult;
use llm_suite_types::llm::THOUGHT_SIGNATURES_KEY;
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

/// The thought signature that belongs to the tool call at `index`, when the
/// provider signed it.
pub fn thought_signature_for(message: &Message, index: usize) -> Option<String> {
    message
        .metadata
        .as_ref()?
        .get(THOUGHT_SIGNATURES_KEY)?
        .as_array()?
        .get(index)?
        .as_str()
        .map(String::from)
}

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
                parts.push(function_response_part(&name, provider_id, text));
            } else if !text.is_empty() {
                parts.push(serde_json::json!({"text": text}));
            }
        }
        MessageContentValue::Rich(blocks) => {
            let mut call_ordinal = 0usize;
            for block in blocks {
                match block {
                    MessageContent::Text { text } => {
                        parts.push(serde_json::json!({"text": text}));
                    }
                    MessageContent::ImageUrl { image_url } => {
                        // Inline base64 images become `inlineData`;
                        // remote URLs become `fileData` references.
                        // Unresolvable references are skipped rather
                        // than silently corrupted.
                        match super::super::helpers::resolve_image_payload(&image_url.url) {
                            Ok(super::super::helpers::ResolvedImage::Base64 {
                                media_type,
                                data,
                            }) => {
                                parts.push(serde_json::json!({
                                    "inlineData": {
                                        "mimeType": media_type,
                                        "data": data,
                                    }
                                }))
                            }
                            Ok(super::super::helpers::ResolvedImage::Url(url)) => {
                                parts.push(serde_json::json!({
                                    "fileData": {
                                        "mimeType":
                                            super::super::helpers::guess_image_media_type(&url),
                                        "fileUri": url,
                                    }
                                }))
                            }
                            Err(_) => {}
                        }
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
                        if let Some(signature) = thought_signature_for(msg, call_ordinal) {
                            part["thoughtSignature"] = serde_json::json!(signature);
                        }
                        parts.push(part);
                        call_ordinal += 1;
                        saw_tool_use = true;
                    }
                    MessageContent::ToolResult { tool_result } => {
                        let name = resolve_function_name(
                            msg,
                            &tool_result.tool_use_id,
                            function_names,
                        )?;
                        let provider_id = provider_call_ids
                            .get(&tool_result.tool_use_id)
                            .map(String::as_str);
                        parts.push(function_response_part(
                            &name,
                            provider_id,
                            &tool_result.content,
                        ));
                    }
                    // The reasoning text itself is not replayed; only the
                    // thought signature on a function call is.
                    MessageContent::Thinking { .. } | MessageContent::RedactedThinking { .. } => {}
                }
            }
        }
    }

    // Tool calls carried on the message itself become function call
    // parts; the rich blocks already hold them otherwise.
    if !saw_tool_use {
        if let Some(calls) = &msg.tool_calls {
            for (ordinal, call) in calls.iter().enumerate() {
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
                if let Some(signature) = thought_signature_for(msg, ordinal) {
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
/// already an object is therefore wrapped in `output`.
///
/// The call id is echoed back when the provider assigned one, so the
/// response can be matched to the exact call it answers.
fn function_response_part(
    name: &str,
    call_id: Option<&str>,
    content: &str,
) -> serde_json::Value {
    let response_value: serde_json::Value = match serde_json::from_str(content) {
        Ok(serde_json::Value::Object(object)) => serde_json::Value::Object(object),
        _ => serde_json::json!({ "output": content }),
    };
    let mut response = serde_json::json!({
        "name": name,
        "response": response_value,
    });
    if let Some(call_id) = call_id.filter(|id| !id.is_empty()) {
        response["id"] = serde_json::json!(call_id);
    }
    serde_json::json!({"functionResponse": response})
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
                    },
                },
                MessageContent::ImageUrl {
                    image_url: llm_suite_types::message::ImageUrlContent {
                        url: "https://example.com/a.jpg".to_string(),
                        detail: None,
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
        // Regression guard: none of the three blocks may be silently dropped.
        assert_eq!(parts.len(), 3, "text + inlineData + fileData");
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
        assert_eq!(parts[2]["fileData"]["fileUri"], "https://example.com/a.jpg");
        assert_eq!(parts[2]["fileData"]["mimeType"], "image/jpeg");
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
        let err = convert_messages(&[result])
            .expect_err("an unnamed function response is meaningless");
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
            },
        }]);
        let mut result = msg(MessageRole::Tool, "");
        result.content = MessageContentValue::Rich(vec![MessageContent::ToolResult {
            tool_result: llm_suite_types::message::ToolResultContent {
                tool_use_id: "call_1".to_string(),
                content: "sunny".to_string(),
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
        let contents = convert_messages(&[empty, msg(MessageRole::User, "hi")])
            .expect("must convert");
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
}
