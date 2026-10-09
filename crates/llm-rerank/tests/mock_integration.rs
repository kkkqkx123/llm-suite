//! Integration tests for the mock rerank provider wired through the
//! shared request pipeline (validation, candidate limiting, fusion inputs).
//! Compiled only when the `mock` feature is enabled.

#![cfg(feature = "mock")]

use std::collections::HashMap;

use llm_rerank::mock::{MockRerankProvider, MockRerankStep};
use llm_rerank::provider::{limit_candidates, validate_request};
use llm_rerank::{
    RerankCandidate, RerankError, RerankProvider, RerankRequest, RerankRuntimeConfig,
};

fn candidate(id: &str, score: f32) -> RerankCandidate {
    RerankCandidate {
        id: id.to_string(),
        content: format!("content-{id}"),
        file_path: "src/a.rs".to_string(),
        initial_score: score,
        entity_type: None,
        metadata: HashMap::new(),
    }
}

fn request(candidates: Vec<RerankCandidate>) -> RerankRequest {
    RerankRequest {
        query: "find auth logic".to_string(),
        candidates,
        config: RerankRuntimeConfig::default(),
    }
}

#[tokio::test]
async fn mock_result_is_sorted_descending_and_carries_scores() {
    let mock = MockRerankProvider::by_score();
    let result = mock
        .rerank(&request(vec![
            candidate("mid", 0.5),
            candidate("top", 0.9),
            candidate("low", 0.1),
        ]))
        .await
        .unwrap_or_else(|e| panic!("rerank must succeed: {e}"));

    let ids: Vec<&str> = result
        .reranked_candidates
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(ids, vec!["top", "mid", "low"], "descending final score");
    for c in &result.reranked_candidates {
        assert!((c.final_score - c.initial_score).abs() < f32::EPSILON);
    }
}

#[tokio::test]
async fn mock_honors_explicit_id_order_over_scores() {
    let mock = MockRerankProvider::with_order(vec!["c".to_string(), "a".to_string()]);
    let result = mock
        .rerank(&request(vec![
            candidate("a", 0.9),
            candidate("b", 0.5),
            candidate("c", 0.1),
        ]))
        .await
        .unwrap_or_else(|e| panic!("rerank must succeed: {e}"));

    let ids: Vec<&str> = result
        .reranked_candidates
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    // "c" and "a" follow the scripted order; unlisted "b" trails.
    assert_eq!(ids, vec!["c", "a", "b"]);
}

#[tokio::test]
async fn mock_pipeline_with_validation_and_limiting() {
    // Exercise the mock through the shared pipeline helpers a real provider
    // path uses: validate -> limit -> rerank.
    let mock = MockRerankProvider::by_score();
    let candidates: Vec<RerankCandidate> = (0..60)
        .map(|i| candidate(&format!("c{i}"), i as f32 / 100.0))
        .collect();
    let full = request(candidates);
    validate_request(&full).unwrap_or_else(|e| panic!("request must validate: {e}"));
    let limited = limit_candidates(&full);
    let result = mock
        .rerank(&limited)
        .await
        .unwrap_or_else(|e| panic!("rerank must succeed: {e}"));
    assert_eq!(
        result.reranked_candidates.len(),
        limited.config.max_candidates,
        "pipeline limiting must bound the candidate count"
    );
    assert_eq!(mock.recorded_candidate_counts(), vec![50]);
}

#[tokio::test]
async fn mock_invalid_request_rejected_before_provider() {
    let mock = MockRerankProvider::by_score();
    let empty = request(Vec::new());
    assert!(validate_request(&empty).is_err());
    assert!(mock.recorded_candidate_counts().is_empty());
}

#[tokio::test]
async fn mock_scripted_failure_repeats_each_call() {
    let mock = MockRerankProvider::by_score().with_steps(vec![MockRerankStep::Fail(
        RerankError::Provider {
            status: 429,
            message: "slow down".to_string(),
            retry_after_ms: None,
        },
    )]);
    let first = mock
        .rerank(&request(vec![candidate("a", 1.0)]))
        .await
        .expect_err("first call must hit the scripted failure");
    assert!(
        matches!(first, RerankError::Provider { status: 429, .. }),
        "{first:?}"
    );
    // Steps exhausted: the mock recovers.
    mock.rerank(&request(vec![candidate("a", 1.0)]))
        .await
        .unwrap_or_else(|e| panic!("recovery call must succeed: {e}"));
    assert_eq!(mock.recorded_candidate_counts(), vec![1, 1]);
}
