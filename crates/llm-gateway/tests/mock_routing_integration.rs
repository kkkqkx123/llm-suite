//! Integration tests through the gateway's mock routing: profile
//! resolution + request merging + scripted mock client, verifying the
//! orchestration path without any HTTP.

use std::sync::Arc;

use llm_client::{LlmResponseSpec, MockLlmClient};
use llm_gateway::LlmGateway;
use llm_message::message_builder::user_text;
use llm_types::llm::{LlmFormat, LlmRequest};
use llm_types::llm::profile::LlmProfile;

fn profile(id: &str) -> LlmProfile {
    profile_with_retries(id, None)
}

/// A profile whose transport-level retries are disabled, for tests that
/// intentionally fall through to the real transport path and must fail fast
/// instead of burning the default exponential backoff chain.
fn profile_with_retries(id: &str, max_retries: Option<u32>) -> LlmProfile {
    LlmProfile {
        id: id.to_string(),
        name: id.to_string(),
        format: LlmFormat::OpenaiChat,
        provider_id: None,
        model: "mock-model".to_string(),
        api_key: None,
        base_url: None,
        parameters: None,
        generation: None,
        timeout: None,
        max_retries,
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
        context_window_size: None,
        proxy: None,
        circuit_breaker: None,
    }
}

fn chat_request(profile_id: &str, text: &str) -> LlmRequest {
    LlmRequest {
        profile_id: profile_id.to_string(),
        messages: vec![user_text(text)],
        parameters: None,
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
    }
}

#[tokio::test]
async fn mock_routed_generate_returns_scripted_response() {
    let gateway = LlmGateway::new();
    let mock = Arc::new(MockLlmClient::new());
    mock.script(LlmResponseSpec::text("scripted answer").with_usage(7, 3));
    gateway.register_profile(profile("mocked")).unwrap();
    gateway.register_mock("mocked", mock.clone());

    let response = gateway
        .generate(&chat_request("mocked", "hi"), None)
        .await
        .expect("mock-routed generate must succeed");

    assert_eq!(response.content.as_deref(), Some("scripted answer"));
    let usage = response.usage.expect("scripted usage must pass through");
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(usage.completion_tokens, 3);

    // The mock must have seen the exact request the caller issued.
    let recorded = mock.recorded_requests();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].messages.len(), 1);
    assert_eq!(recorded[0].messages[0].text_content(), "hi");
}

#[tokio::test]
async fn mock_routed_stream_yields_scripted_events() {
    let gateway = LlmGateway::new();
    let mock = Arc::new(MockLlmClient::new());
    mock.default(LlmResponseSpec::text("streamed answer"));
    gateway.register_profile(profile("mock-stream")).unwrap();
    gateway.register_mock("mock-stream", mock);

    let mut stream = gateway
        .generate_stream(&chat_request("mock-stream", "go"), None)
        .await
        .expect("mock-routed streaming must succeed");

    let mut text = String::new();
    let mut saw_end = false;
    while let Some(event) = stream.next().await {
        match event.expect("stream must not error") {
            llm_types::llm::MessageStreamEvent::Text(t) => text.push_str(&t.text),
            llm_types::llm::MessageStreamEvent::End(_) => saw_end = true,
            _ => {}
        }
    }
    assert_eq!(text, "streamed answer");
    assert!(saw_end);
}

#[tokio::test]
async fn mock_routing_scopes_by_profile_id() {
    let gateway = LlmGateway::new();
    let mock = Arc::new(MockLlmClient::new());
    mock.script(LlmResponseSpec::text("from mock"));
    gateway.register_profile(profile("with-mock")).unwrap();
    let mut real_profile = profile_with_retries("without-mock", Some(0));
    // Unreachable local port: connection fails immediately (no DNS, no
    // proxy); the 1s timeout only bounds pathological environments where a
    // closed-port connect hangs instead of being refused.
    real_profile.base_url = Some("http://127.0.0.1:1".to_string());
    real_profile.timeout = Some(1);
    gateway.register_profile(real_profile).unwrap();
    gateway.register_mock("with-mock", mock);

    let mocked = gateway
        .generate(&chat_request("with-mock", "q"), None)
        .await
        .expect("mock-registered profile must be routed to the mock");
    assert_eq!(mocked.content.as_deref(), Some("from mock"));

    // The other profile has no registered mock, so it falls through to the
    // real transport path and fails on the missing base URL before any HTTP.
    let real = gateway
        .generate(&chat_request("without-mock", "q"), None)
        .await;
    assert!(real.is_err(), "unmocked profile must not hit the mock");
}

#[tokio::test]
async fn mock_scripted_error_propagates_through_gateway() {
    let gateway = LlmGateway::new();
    let mock = Arc::new(MockLlmClient::new());
    mock.script_error(llm_codec::error::LlmError::Timeout(123));
    gateway.register_profile(profile("mock-err")).unwrap();
    gateway.register_mock("mock-err", mock);

    let error = gateway
        .generate(&chat_request("mock-err", "q"), None)
        .await
        .expect_err("scripted error must propagate");
    assert!(matches!(error, llm_codec::error::LlmError::Timeout { .. }));
}

#[tokio::test]
async fn count_tokens_routes_to_mock() {
    let gateway = LlmGateway::new();
    let mock = Arc::new(MockLlmClient::new());
    gateway.register_profile(profile("mock-count")).unwrap();
    gateway.register_mock("mock-count", mock.clone());

    let request = chat_request("mock-count", "count me");
    let result = gateway
        .count_tokens(&request, None)
        .await
        .expect("mock count_tokens must succeed");
    // The mock counts via local estimation, so a non-zero count proves the
    // request reached the mock's estimator.
    assert!(result.input_tokens > 0);
}
