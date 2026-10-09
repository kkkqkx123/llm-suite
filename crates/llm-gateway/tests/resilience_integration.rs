//! Resilience integration tests over the real HTTP path: gateway circuit
//! breaker opening/recovery and per-base-URL rate limiting, both driven by
//! a bare-TCP mock server (no network access).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use llm_client::http_mock::{MockResponse, MockServer};
use llm_gateway::LlmGateway;
use llm_message::message_builder::user_text;
use llm_types::llm::profile::LlmProfile;
use llm_types::llm::{LlmFormat, LlmRequest};

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
        no_proxy: None,
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
        "id": "chatcmpl-resilience",
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
    .to_string()
}

#[tokio::test]
async fn circuit_breaker_opens_after_consecutive_5xx() {
    let server = MockServer::spawn(move |_| MockResponse::status(500, r#"{"error":"down"}"#)).await;

    let gateway = LlmGateway::new();
    let mut profile = http_profile("cb-open", &server.url(""), 0);
    profile.circuit_breaker = Some(llm_types::llm::CircuitBreakerConfig {
        min_samples: 3,
        failure_threshold: 0.5,
        open_duration_ms: 60_000,
        half_open_max_probes: 1,
    });
    gateway.register_profile(profile).unwrap();

    // 3 failures fill the window and trip the breaker.
    for _ in 0..3 {
        gateway
            .generate(&chat_request("cb-open", "probe"), None)
            .await
            .expect_err("5xx must fail");
    }
    assert_eq!(server.call_count(), 3);

    // The breaker is open: the next request is rejected locally without
    // touching the server.
    let error = gateway
        .generate(&chat_request("cb-open", "blocked"), None)
        .await
        .expect_err("open breaker must reject");
    assert!(
        matches!(error, llm_codec::error::LlmError::CircuitOpen),
        "must surface as CircuitOpen: {error}"
    );
    assert_eq!(
        server.call_count(),
        3,
        "open breaker must not hit the server"
    );
}

#[tokio::test]
async fn circuit_breaker_ignores_semantic_4xx() {
    let server = MockServer::spawn(move |_| MockResponse::status(400, r#"{"error":"bad"}"#)).await;

    let gateway = LlmGateway::new();
    let mut profile = http_profile("cb-4xx", &server.url(""), 0);
    profile.circuit_breaker = Some(llm_types::llm::CircuitBreakerConfig {
        min_samples: 2,
        failure_threshold: 0.5,
        open_duration_ms: 60_000,
        half_open_max_probes: 1,
    });
    gateway.register_profile(profile).unwrap();

    // Repeated 4xx must never open the breaker: every request reaches the
    // server (initial + no retries configured).
    for _ in 0..6 {
        gateway
            .generate(&chat_request("cb-4xx", "still bad"), None)
            .await
            .expect_err("4xx must fail");
    }
    assert_eq!(
        server.call_count(),
        6,
        "semantic 4xx must not trip the breaker"
    );
}

#[tokio::test]
async fn circuit_breaker_half_open_probe_closes_breaker() {
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_handler = hits.clone();
    let server = MockServer::spawn(move |_| {
        // First 3 calls fail (open the breaker); the probe afterwards
        // succeeds (close it again).
        if hits_handler.fetch_add(1, Ordering::SeqCst) < 3 {
            MockResponse::status(500, r#"{"error":"down"}"#)
        } else {
            MockResponse::ok_json(chat_response_json("recovered"))
        }
    })
    .await;

    let gateway = LlmGateway::new();
    let mut profile = http_profile("cb-half", &server.url(""), 0);
    profile.circuit_breaker = Some(llm_types::llm::CircuitBreakerConfig {
        min_samples: 3,
        failure_threshold: 0.5,
        open_duration_ms: 20,
        half_open_max_probes: 1,
    });
    gateway.register_profile(profile).unwrap();

    for _ in 0..3 {
        gateway
            .generate(&chat_request("cb-half", "probe"), None)
            .await
            .expect_err("5xx must fail");
    }

    // Wait past the open duration, then the probe goes through and closes
    // the breaker on success.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let response = gateway
        .generate(&chat_request("cb-half", "probe"), None)
        .await
        .expect("half-open probe must be admitted");
    assert_eq!(response.content.as_deref(), Some("recovered"));

    // The breaker is closed again: the next request flows immediately.
    let _ = gateway
        .generate(&chat_request("cb-half", "still fine"), None)
        .await
        .expect("closed breaker must admit requests");
    assert_eq!(server.call_count(), 5);
}

#[tokio::test]
async fn rate_limiter_throttles_shared_base_url() {
    // Server answers instantly; the limiter is what paces the calls.
    let server = MockServer::spawn(move |_| MockResponse::ok_json(chat_response_json("ok"))).await;

    let gateway = LlmGateway::new();
    let mut profile = http_profile("rl-a", &server.url(""), 0);
    // Provider-level rate limit via profile metadata override path: 10 rps,
    // burst 1, so 4 calls cost ~300ms of waiting.
    profile.metadata = Some(
        [(
            "rate_limit".to_string(),
            serde_json::json!({ "requests_per_second": 10.0, "burst": 1 }),
        )]
        .into_iter()
        .collect(),
    );
    gateway.register_profile(profile).unwrap();

    // A second profile pointing at the same base URL shares the bucket.
    let mut profile_b = http_profile("rl-b", &server.url(""), 0);
    profile_b.metadata = None;
    gateway.register_profile(profile_b).unwrap();

    let start = Instant::now();
    for (i, id) in ["rl-a", "rl-a", "rl-b", "rl-b"].iter().enumerate() {
        let _ = gateway
            .generate(&chat_request(id, "paced"), None)
            .await
            .unwrap_or_else(|e| panic!("call {i} must succeed: {e}"));
    }
    let elapsed = start.elapsed();

    // 4 calls at 10 rps with burst 1: 3 waits of ~100ms.
    assert!(
        elapsed >= std::time::Duration::from_millis(250),
        "shared limiter must pace calls past the burst: {elapsed:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_millis(2000),
        "throttling must stay near the configured rate: {elapsed:?}"
    );
    assert_eq!(server.call_count(), 4);
}

#[tokio::test]
async fn rate_limiter_not_configured_admits_immediately() {
    let server = MockServer::spawn(move |_| MockResponse::ok_json(chat_response_json("ok"))).await;

    let gateway = LlmGateway::new();
    gateway
        .register_profile(http_profile("rl-off", &server.url(""), 0))
        .unwrap();

    let start = Instant::now();
    for _ in 0..4 {
        let _ = gateway
            .generate(&chat_request("rl-off", "fast"), None)
            .await
            .expect("call must succeed");
    }
    assert!(
        start.elapsed() < std::time::Duration::from_millis(1000),
        "without rate_limit config calls must not be paced (only cold-start latency allowed)"
    );
}
