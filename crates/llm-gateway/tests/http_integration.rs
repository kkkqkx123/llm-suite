//! End-to-end integration tests over the real HTTP path: gateway ->
//! codec -> reqwest transport -> bare-TCP mock server (no network access).
//!
//! Covers plain text generation, SSE streaming, transport retry on
//! transient server errors, retry exhaustion and request-timeout failure.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use llm_client::LlmClient;
use llm_codec::codecs::OpenaiChatCodec;
use llm_client::http_mock::{MockResponse, MockServer};
use llm_client::client::LlmClientImpl;
use llm_gateway::LlmGateway;
use llm_message::message_builder::user_text;
use llm_types::llm::{LlmFormat, LlmRequest};
use llm_types::llm::profile::LlmProfile;

fn http_profile(id: &str, base_url: &str, max_retries: u32) -> LlmProfile {
    LlmProfile {
        id: id.to_string(),
        name: id.to_string(),
        format: LlmFormat::OpenaiChat,
        provider_id: None,
        model: "mock-model".to_string(),
        api_key: Some("test-key".to_string()),
        base_url: Some(base_url.to_string()),
        parameters: None,
        generation: None,
        timeout: None,
        max_retries: Some(max_retries),
        retry_delay: Some(10),
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

fn chat_response_json(content: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-integration-1",
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
    .to_string()
}

#[tokio::test]
async fn gateway_end_to_end_text_generation_over_http() {
    let server = MockServer::spawn(move |req| {
        assert_eq!(req.method, "POST");
        assert!(req.path.ends_with("/chat/completions"));
        assert!(req.body.contains("mock-model"));
        assert!(req.body.contains("hello gateway"));
        MockResponse::ok_json(chat_response_json("Hello from mock"))
    })
    .await;

    let gateway = LlmGateway::new();
    gateway
        .register_profile(http_profile("http-e2e", &server.url(""), 0))
        .unwrap();

    let response = gateway
        .generate(&chat_request("http-e2e", "hello gateway"), None)
        .await
        .expect("generation must succeed");

    assert_eq!(response.content.as_deref(), Some("Hello from mock"));
    let usage = response.usage.expect("usage must be parsed");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(server.call_count(), 1);
}

#[tokio::test]
async fn gateway_end_to_end_streaming_over_http() {
    let server = MockServer::spawn(move |_| {
        let delta = |text: &str| {
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": text}}]
            })
            .to_string()
        };
        MockResponse::Sse {
            status: 200,
            events: vec![delta("Hel"), delta("lo"), "[DONE]".to_string()],
        }
    })
    .await;

    let gateway = LlmGateway::new();
    gateway
        .register_profile(http_profile("http-stream", &server.url(""), 0))
        .unwrap();

    let mut stream = gateway
        .generate_stream(&chat_request("http-stream", "stream it"), None)
        .await
        .expect("stream establishment must succeed");

    let mut text = String::new();
    let mut saw_end = false;
    while let Some(event) = stream.next().await {
        match event.expect("stream must not error") {
            llm_types::llm::MessageStreamEvent::Text(t) => text.push_str(&t.text),
            llm_types::llm::MessageStreamEvent::End(_) => saw_end = true,
            _ => {}
        }
    }

    assert_eq!(text, "Hello");
    assert!(saw_end, "the [DONE] sentinel must surface as an End event");
}

#[tokio::test]
async fn transport_retries_transient_server_error_then_succeeds() {
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_handler = hits.clone();
    let server = MockServer::spawn(move |_| {
        if hits_handler.fetch_add(1, Ordering::SeqCst) == 0 {
            MockResponse::status(500, r#"{"error": "transient"}"#)
        } else {
            MockResponse::ok_json(chat_response_json("recovered"))
        }
    })
    .await;

    let gateway = LlmGateway::new();
    gateway
        .register_profile(http_profile("http-retry", &server.url(""), 2))
        .unwrap();

    let response = gateway
        .generate(&chat_request("http-retry", "retry me"), None)
        .await
        .expect("second attempt must succeed");

    assert_eq!(response.content.as_deref(), Some("recovered"));
    assert_eq!(server.call_count(), 2);
}

#[tokio::test]
async fn retry_exhaustion_surfaces_provider_error() {
    let server = MockServer::spawn(move |_| {
        MockResponse::status(429, r#"{"error": "rate limited"}"#)
    })
    .await;

    let gateway = LlmGateway::new();
    gateway
        .register_profile(http_profile("http-exhaust", &server.url(""), 1))
        .unwrap();

    let error = gateway
        .generate(&chat_request("http-exhaust", "still limited"), None)
        .await
        .expect_err("exhausted retries must fail");

    // 1 initial attempt + 1 retry = 2 calls, then the provider error surfaces.
    assert_eq!(server.call_count(), 2);
    assert!(error.is_retryable(), "429 must classify as retryable: {error}");
}

#[tokio::test]
async fn request_timeout_fails_with_timeout_error() {
    let server = MockServer::spawn(move |_| {
        MockResponse::delayed_json(
            200,
            chat_response_json("too late"),
            std::time::Duration::from_millis(1500),
        )
    })
    .await;

    // Direct client-level test: the profile attempt timeout (1s) is shorter
    // than the delayed response (1.5s).
    let mut profile = http_profile("http-timeout", &server.url(""), 0);
    profile.timeout = Some(1);
    let client = LlmClientImpl::new(
        reqwest::Client::new(),
        Arc::new(OpenaiChatCodec::new()),
        profile,
    );

    let error = client
        .generate(&chat_request("http-timeout", "slow server"), None)
        .await
        .expect_err("delayed response must time out");

    assert!(
        matches!(error, llm_codec::error::LlmError::Timeout(_)),
        "must surface as a timeout error: {error}"
    );
}
