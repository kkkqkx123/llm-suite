//! Assembly of the OpenAI Responses request body.

use crate::error::LlmResult;
use llm_suite_types::llm::{LlmProfile, LlmRequest};
use llm_suite_types::message::Message;

use super::{messages, tools};

/// Split a request into the system instructions and the input history.
///
/// The system prompt travels in the dedicated `instructions` field, the
/// channel the official API reserves for it; it is not an input item.
/// In text mode the instructions carry the tool declarations as well.
pub fn split_system_and_history(request: &LlmRequest) -> LlmResult<(String, Vec<Message>)> {
    let (_, filtered) =
        llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);
    if super::super::shared::is_text_mode(request) {
        let history = super::super::shared::convert_history_for_text_mode(&filtered, request);
        let instructions = super::super::shared::text_mode_system_content(request)?;
        Ok((instructions, history))
    } else {
        let instructions = super::super::shared::system_content(request)?;
        Ok((instructions, filtered))
    }
}

/// Build the request body (separated from the HTTP layer for testability).
pub fn build_body(request: &LlmRequest, profile: &LlmProfile) -> LlmResult<serde_json::Value> {
    let use_text_mode = super::super::shared::is_text_mode(request);
    let (instructions, history) = split_system_and_history(request)?;

    let mut body = serde_json::json!({
        "model": profile.model,
        "input": messages::convert_messages(&history)?,
    });

    if !instructions.is_empty() {
        body["instructions"] = serde_json::json!(instructions);
    }

    if request.stream == Some(true) {
        body["stream"] = serde_json::json!(true);
    }

    let generation = super::super::shared::resolve_generation(request, profile)?;
    crate::generation::apply_openai_responses(&mut body, &generation)?;
    super::super::shared::merge_and_apply_params(&mut body, profile, &request.parameters);

    if !use_text_mode {
        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::json!(tools::response_tool_defs(tools));
        }
    }

    super::super::shared::apply_custom_body(&mut body, profile);

    Ok(body)
}

/// Build the count-tokens body: the same `input` (plus native `tools`)
/// that `build_request` sends, without stream flags or generation
/// parameters (they do not affect the input count).
pub fn build_count_tokens_body(
    request: &LlmRequest,
    profile: &LlmProfile,
) -> LlmResult<serde_json::Value> {
    let use_text_mode = super::super::shared::is_text_mode(request);
    let (instructions, history) = split_system_and_history(request)?;

    let mut body = serde_json::json!({
        "model": profile.model,
        "input": messages::convert_messages(&history)?,
    });

    if !instructions.is_empty() {
        body["instructions"] = serde_json::json!(instructions);
    }
    if !use_text_mode {
        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::json!(tools::response_tool_defs(tools));
        }
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::openai_response::test_support::{
        count_profile, count_request, minimal_request, request_body,
    };
    use crate::codecs::openai_response::OpenaiResponseCodec;
    use crate::codecs::LlmCodec;
    use llm_suite_types::message::{Message, MessageContentValue, MessageRole};

    #[test]
    fn named_tool_choice_is_flat() {
        use llm_suite_types::llm::generation::{LlmToolChoice, ToolChoiceMode};

        let mut body = serde_json::json!({"model": "gpt-4o"});
        crate::generation::apply_openai_responses(
            &mut body,
            &llm_suite_types::llm::generation::LlmGenerationParams {
                tool_choice: Some(LlmToolChoice {
                    mode: Some(ToolChoiceMode::Required),
                    tool_name: Some("get_weather".to_string()),
                }),
                ..Default::default()
            },
        )
        .expect("tool choice must be accepted");
        assert_eq!(body["tool_choice"]["type"], serde_json::json!("function"));
        assert_eq!(
            body["tool_choice"]["name"],
            serde_json::json!("get_weather")
        );
        assert!(
            body["tool_choice"].get("function").is_none(),
            "Responses names a function flat"
        );
    }

    #[test]
    fn the_request_carries_no_parameter_the_api_does_not_know() {
        // The Responses API has no `seed`; only Chat Completions does.
        let codec = OpenaiResponseCodec::new();
        let mut request = count_request();
        request.generation = Some(llm_suite_types::llm::generation::LlmGenerationParams {
            seed: Some(42),
            ..Default::default()
        });
        let built = codec
            .build_request(&request, &count_profile())
            .expect("must build");
        let body = request_body(&built);
        assert!(
            body.get("seed").is_none(),
            "the Responses API rejects unknown request fields: {body:?}"
        );
    }

    #[test]
    fn the_json_schema_format_is_flat() {
        // The Responses API keeps name, schema and strict next to the type,
        // while Chat Completions nests them under `json_schema`.
        let codec = OpenaiResponseCodec::new();
        let mut request = count_request();
        request.generation = Some(llm_suite_types::llm::generation::LlmGenerationParams {
            response_format: Some(llm_suite_types::llm::generation::LlmResponseFormat {
                kind: Some(llm_suite_types::llm::generation::ResponseFormatKind::JsonSchema),
                name: Some("answer".to_string()),
                schema: Some(serde_json::json!({"type": "object"})),
                strict: Some(true),
            }),
            ..Default::default()
        });
        let built = codec
            .build_request(&request, &count_profile())
            .expect("must build");
        let body = request_body(&built);
        let format = &body["text"]["format"];
        assert_eq!(format["type"], serde_json::json!("json_schema"));
        assert_eq!(format["name"], serde_json::json!("answer"));
        assert_eq!(format["strict"], serde_json::json!(true));
        assert!(format.get("json_schema").is_none(), "{body:?}");
    }

    #[test]
    fn count_tokens_request_targets_input_tokens_endpoint() {
        let codec = OpenaiResponseCodec::new();
        let req = codec
            .build_count_tokens_request(&count_request(), &count_profile())
            .expect("count request must build")
            .expect("responses provider supports counting");
        assert_eq!(
            req.url().as_str(),
            "https://api.openai.com/v1/responses/input_tokens"
        );
        assert_eq!(
            req.headers().get("Authorization").unwrap(),
            "Bearer sk-test"
        );
        let body = request_body(&req);
        assert_eq!(body["model"], serde_json::json!("gpt-4o"));
        assert_eq!(body["input"][0]["content"], serde_json::json!("hello"));
        assert!(
            body.get("stream").is_none(),
            "count body must not carry stream flags"
        );
    }

    #[test]
    fn count_tokens_body_carries_native_tools() {
        let mut req = count_request();
        req.tool_call_protocol = Some(llm_suite_types::llm::ToolCallProtocol::Native);
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()]);
        let body = build_count_tokens_body(&req, &count_profile()).unwrap();
        assert_eq!(body["tools"][0]["name"], serde_json::json!("get_weather"));
        assert_eq!(body["tools"][0]["type"], serde_json::json!("function"));
    }

    #[test]
    fn count_tokens_body_text_mode_inlines_tools_into_system() {
        let mut req = count_request();
        req.messages.insert(
            0,
            Message {
                id: llm_suite_types::Id::new(),
                role: MessageRole::System,
                content: MessageContentValue::Text("You are a helper".to_string()),
                timestamp: 0,
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
                thinking: None,
                thinking_signature: None,
                metadata: None,
            },
        );
        req.tool_call_protocol = Some(llm_suite_types::llm::ToolCallProtocol::Xml);
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }))
        .unwrap()]);
        let body = build_count_tokens_body(&req, &count_profile()).unwrap();
        assert!(
            body.get("tools").is_none(),
            "text mode must not send native tools"
        );
        let system = body["instructions"].as_str().unwrap();
        assert!(system.contains("get_weather"), "system must declare tools");
    }

    #[test]
    fn count_tokens_response_parses_input_tokens() {
        let codec = OpenaiResponseCodec::new();
        let body: serde_json::Value =
            serde_json::from_str(r#"{"object":"response.input_tokens","input_tokens":13}"#)
                .unwrap();
        assert_eq!(codec.parse_count_tokens_response(&body).unwrap(), 13);
        let empty: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(codec.parse_count_tokens_response(&empty).unwrap(), 0);
    }

    #[test]
    fn the_system_prompt_travels_in_the_instructions_field() {
        let mut request = minimal_request();
        request.messages.push(Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::System,
            content: MessageContentValue::Text("You are a helper".to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        });
        request.messages.push(Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::User,
            content: MessageContentValue::Text("hi".to_string()),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        });

        let body = build_body(&request, &count_profile()).expect("must build");
        assert_eq!(body["instructions"], serde_json::json!("You are a helper"));
        let input = body["input"].as_array().expect("input");
        assert!(
            input
                .iter()
                .all(|item| item["role"] != serde_json::json!("system")),
            "the system prompt must not be an input item: {body}"
        );
        assert_eq!(input.len(), 1);

        // The counting request carries the same instructions.
        let counted = build_count_tokens_body(&request, &count_profile()).expect("must build");
        assert_eq!(
            counted["instructions"],
            serde_json::json!("You are a helper")
        );
    }
}
