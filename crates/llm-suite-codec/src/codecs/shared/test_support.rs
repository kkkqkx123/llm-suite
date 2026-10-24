//! Shared fixtures for the crate-internal codec tests.

use llm_suite_types::llm::{LlmFormat, LlmProfile};

/// A profile carrying an API key, custom headers and query parameters, used
/// by the header and parameter merging tests.
pub fn profile_with(auth_type: Option<&str>) -> LlmProfile {
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
        auth_type: auth_type.map(String::from),
        custom_headers: Some(
            [("x-custom".to_string(), serde_json::json!("v1"))]
                .into_iter()
                .collect(),
        ),
        custom_body: None,
        custom_body_enabled: None,
        query_params: Some(
            [("api-version".to_string(), serde_json::json!("2024-01"))]
                .into_iter()
                .collect(),
        ),
        stream_options: None,
        prompt_caching: None,
        context_window_size: None,
        proxy: None,
        no_proxy: None,
        circuit_breaker: None,
        auth: None,
    }
}
