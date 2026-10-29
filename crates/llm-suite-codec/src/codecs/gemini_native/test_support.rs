//! Shared fixtures for the Gemini native codec tests.

use llm_suite_types::llm::{LlmFormat, LlmProfile, LlmRequest};
use llm_suite_types::message::{Message, MessageContentValue, MessageRole};

pub fn profile() -> LlmProfile {
    LlmProfile {
        id: "p1".to_string(),
        name: "test".to_string(),
        format: LlmFormat::GeminiNative,
        provider_id: None,
        model: "gemini-1.5-pro".to_string(),
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
        identity: None,
        transport: None,
        env: None,
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
        max_retries: None,
        retry_delay_ms: None,
        extra_headers: None,
    }
}

pub fn count_request() -> LlmRequest {
    request(
        vec![
            msg(MessageRole::System, "You are a helper"),
            msg(MessageRole::User, "Hello"),
        ],
        None,
    )
}
