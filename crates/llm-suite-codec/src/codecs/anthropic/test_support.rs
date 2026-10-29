//! Shared fixtures for the Anthropic codec tests.

use llm_suite_types::llm::{LlmFormat, LlmProfile, LlmRequest};
use llm_suite_types::message::{Message, MessageContent, MessageContentValue, MessageRole};

pub fn profile() -> LlmProfile {
    LlmProfile {
        id: "p1".to_string(),
        name: "test".to_string(),
        format: LlmFormat::Anthropic,
        provider_id: None,
        model: "claude-3-5-sonnet".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: None,
        parameters: None,
        generation: None,
        timeout: None,
        max_retries: None,
        retry_delay: None,
        headers: None,
        metadata: None,
        tool_call_protocol: None,
        auth_type: None,
        custom_headers: None,
        custom_body: None,
        custom_body_enabled: None,
        query_params: None,
        stream_options: None,
        prompt_caching: None,
        context_window_size: None,
        proxy: None,
        no_proxy: None,
        circuit_breaker: None,
        auth: None,
        compat: None,
    }
}

pub fn msg(role: MessageRole, text: &str) -> Message {
    Message {
        id: llm_suite_types::Id::new(),
        role,
        content: MessageContentValue::Text(text.to_string()),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    }
}

pub fn request(messages: Vec<Message>, params: Option<serde_json::Value>) -> LlmRequest {
    LlmRequest {
        profile_id: "p1".to_string(),
        messages,
        parameters: params,
        generation: None,
        tools: None,
        tool_call_protocol: None,
        locked_tool_call_protocol: None,
        violation_policy: None,
        execution_id: None,
        stream: None,
        dead_loop_detection: None,
        protocol_auto_converted: None,
        timeout_ms: None,
        prompt_caching: None,
        cached_content: None,
    }
}

pub fn rich_msg(role: MessageRole, blocks: Vec<MessageContent>) -> Message {
    Message {
        id: llm_suite_types::Id::new(),
        role,
        content: MessageContentValue::Rich(blocks),
        timestamp: 0,
        tool_call_id: None,
        tool_name: None,
        tool_calls: None,
        thinking: None,
        thinking_signature: None,
        metadata: None,
    }
}

pub fn tool_msg(tool_use_id: &str, text: &str, is_error: bool) -> Message {
    let mut m = msg(MessageRole::Tool, text);
    m.tool_call_id = Some(tool_use_id.to_string());
    m.metadata = Some(std::collections::HashMap::from([(
        "is_error".to_string(),
        serde_json::json!(is_error),
    )]));
    m
}

pub fn tool_call(id: &str, name: &str, arguments: &str) -> llm_suite_types::message::LlmToolCall {
    llm_suite_types::message::LlmToolCall {
        id: id.to_string(),
        r#type: "function".to_string(),
        call_id: None,
        function: llm_suite_types::message::LlmFunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
        thought_signature: None,
    }
}
