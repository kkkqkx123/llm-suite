//! Parsing of a non-streaming OpenAI Chat Completions response body.

use crate::error::LlmResult;
use llm_suite_types::llm::LlmResult as LlmResponseType;
use llm_suite_types::message::{MessageContentValue, MessageRole};

use super::usage::parse_openai_usage;

pub fn parse_openai_chat_response(body: &str) -> LlmResult<LlmResponseType> {
    let json: serde_json::Value = serde_json::from_str(body)?;

    let id = json.get("id").and_then(|v| v.as_str()).map(String::from);
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let empty_choices = vec![];
    let choices = json
        .get("choices")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_choices);
    let first_choice = choices.first();
    let message = first_choice.and_then(|c| c.get("message"));

    let content = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(String::from);

    // Provider-native value passed through untouched (`stop` / `tool_calls`
    // / `length` / `content_filter`); never normalized here.
    let finish_reason = first_choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str())
        .map(String::from);

    // NOTE: `reasoning_content` is NOT part of the official OpenAI Chat
    // Completions schema (it is absent from `openai-python`). It is a
    // non-standard extension used by DeepSeek and several Chinese
    // OpenAI-compatible gateways to surface intermediate reasoning. Official
    // OpenAI reasoning models do not expose it here, so strict OpenAI
    // responses will leave this empty.
    let reasoning_content = message
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|r| r.as_str())
        .map(String::from);

    let tool_calls = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let function = tc.get("function")?;
                    // The function name identifies the call and is required;
                    // a missing id or argument string is tolerated so a
                    // compatible gateway cannot make a whole call disappear.
                    let name = function.get("name")?.as_str()?.to_string();
                    let arguments = function
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| format!("call_{}", llm_suite_common::generate_id()));
                    Some(llm_suite_types::message::LlmToolCall {
                        id,
                        r#type: "function".to_string(),
                        call_id: None,
                        function: llm_suite_types::message::LlmFunctionCall { name, arguments },
                        thought_signature: None,
                    })
                })
                .collect()
        });

    let mut usage = json.get("usage").map(parse_openai_usage);
    // Aggregating gateways report the billed total next to the usage
    // object instead of inside it; the usage-level value wins when both
    // are present.
    if usage.as_ref().and_then(|u| u.provider_total).is_none() {
        if let Some(total) = crate::codecs::helpers::parse_provider_total(&json) {
            if let Some(stats) = usage.as_mut() {
                stats.provider_total = Some(total);
            }
        }
    }

    // A refusal is the provider declining to answer; it is kept apart from
    // the answer text so an empty answer can be told from a refusal.
    let refusal = message
        .and_then(|m| m.get("refusal"))
        .and_then(|r| r.as_str())
        .map(String::from);

    let reasoning_tokens = usage.as_ref().and_then(|u| u.reasoning_tokens);

    let msg = llm_suite_types::message::Message {
        id: llm_suite_types::Id::new(),
        role: MessageRole::Assistant,
        content: MessageContentValue::Text(content.clone().unwrap_or_default()),
        timestamp: llm_suite_common::time::now(),
        tool_call_id: None,
        tool_name: None,
        tool_calls: tool_calls.clone(),
        // The reasoning text also lives on the message so that every codec
        // reports it the same way.
        thinking: reasoning_content.clone(),
        thinking_signature: None,
        metadata: None,
    };

    let mut metadata = std::collections::HashMap::new();
    if let Some(created) = json.get("created").and_then(|v| v.as_i64()) {
        metadata.insert("created".to_string(), serde_json::json!(created));
    }
    if let Some(fp) = json.get("system_fingerprint").and_then(|v| v.as_str()) {
        metadata.insert("system_fingerprint".to_string(), serde_json::json!(fp));
    }
    // Web-search citations ride on `message.annotations`; they are the only
    // place the sources of a grounded answer are reported.
    if let Some(annotations) = message
        .and_then(|m| m.get("annotations"))
        .and_then(|a| a.as_array())
        .filter(|a| !a.is_empty())
    {
        metadata.insert(
            llm_suite_types::llm::ANNOTATIONS_KEY.to_string(),
            serde_json::json!(annotations),
        );
    }
    let metadata = if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    };

    Ok(LlmResponseType {
        id,
        model,
        content,
        message: msg,
        tool_calls,
        usage,
        finish_reason,
        duration: 0,
        reasoning_content,
        reasoning_signature: None,
        reasoning_tokens,
        metadata,
        refusal,
        stream_stats: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_is_parsed_and_kept_out_of_the_content() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null,"refusal":"I cannot help with that."},"finish_reason":"stop"}]}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        assert_eq!(result.content, None);
        assert_eq!(result.refusal.as_deref(), Some("I cannot help with that."));
    }

    #[test]
    fn tool_call_without_arguments_keeps_the_call() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather"}}]},"finish_reason":"tool_calls"}]}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        let calls = result.tool_calls.expect("tool calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn gateway_cost_extension_becomes_the_provider_total() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00012}}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        let usage = result.usage.expect("usage present");
        assert_eq!(usage.provider_total, Some(0.00012));
        assert!(usage.total_from_provider());
    }

    #[test]
    fn top_level_cost_backfills_a_usage_without_one() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"cost":0.00034}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        let usage = result.usage.expect("usage present");
        assert_eq!(usage.provider_total, Some(0.00034));
    }

    #[test]
    fn usage_level_cost_wins_over_the_top_level_value() {
        let body = r#"{"id":"1","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.5},"cost":0.9}"#;
        let result = parse_openai_chat_response(body).expect("must parse");
        let usage = result.usage.expect("usage present");
        assert_eq!(usage.provider_total, Some(0.5));
    }
}
