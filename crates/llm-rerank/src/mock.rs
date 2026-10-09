//! Feature-gated mock rerank provider for tests.
//!
//! Deterministic: candidates are reordered by caller-supplied id order (or
//! by descending initial score when no order is given), so assertions never
//! depend on a remote service.

use std::collections::{HashMap, VecDeque};

use async_trait::async_trait;

use crate::error::{RerankError, Result};
use crate::provider::{RerankCandidate, RerankProvider, RerankRequest, RerankResult};

/// One scripted step consumed in call order.
#[derive(Debug, Clone)]
pub enum MockRerankStep {
    /// Rerank normally.
    Respond,
    /// Fail the call with an injected error.
    Fail(RerankError),
}

// A plain std Mutex is enough: no await is held across the lock.
type Shared<T> = std::sync::Mutex<T>;

/// Scripted, deterministic `RerankProvider` for tests.
pub struct MockRerankProvider {
    /// Target order by candidate id; ids absent from the list keep their
    /// relative order after the listed ones.
    id_order: Vec<String>,
    steps: Shared<VecDeque<MockRerankStep>>,
    /// Recorded candidate counts, in call order.
    calls: Shared<Vec<usize>>,
}

impl MockRerankProvider {
    /// Mock that orders candidates by descending initial score.
    pub fn by_score() -> Self {
        Self {
            id_order: Vec::new(),
            steps: std::sync::Mutex::new(VecDeque::new()),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Mock that orders candidates by the given id sequence.
    pub fn with_order(id_order: Vec<String>) -> Self {
        Self {
            id_order,
            steps: std::sync::Mutex::new(VecDeque::new()),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Queue scripted steps consumed in call order.
    pub fn with_steps(self, steps: Vec<MockRerankStep>) -> Self {
        *self.steps.lock().unwrap_or_else(|p| p.into_inner()) = steps.into();
        self
    }

    /// Candidate counts observed so far, in call order.
    pub fn recorded_candidate_counts(&self) -> Vec<usize> {
        self.calls.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn position(&self, id: &str) -> usize {
        self.id_order
            .iter()
            .position(|x| x == id)
            .unwrap_or(usize::MAX)
    }

    fn rank(&self, request: &RerankRequest) -> Vec<RerankCandidate> {
        let mut candidates = request.candidates.clone();
        if self.id_order.is_empty() {
            candidates.sort_by(|a, b| {
                b.initial_score
                    .partial_cmp(&a.initial_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            candidates.sort_by_key(|c| self.position(&c.id));
        }
        candidates
    }
}

#[async_trait]
impl RerankProvider for MockRerankProvider {
    async fn rerank(&self, request: &RerankRequest) -> Result<RerankResult> {
        self.calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(request.candidates.len());
        let step = self
            .steps
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop_front();
        match step {
            Some(MockRerankStep::Fail(err)) => Err(err),
            _ => {
                let ranked = self.rank(request);
                let reranked = ranked
                    .into_iter()
                    .map(|c| crate::provider::RerankedCandidate {
                        id: c.id,
                        rerank_score: c.initial_score,
                        initial_score: c.initial_score,
                        final_score: c.initial_score,
                        rank_change: 0,
                        reasoning: None,
                    })
                    .collect();
                Ok(RerankResult::new(reranked))
            }
        }
    }

    fn provider_name(&self) -> &str {
        "mock-rerank"
    }

    fn is_available(&self) -> bool {
        true
    }
}

// Keep the HashMap import referenced on older toolchains where the
// candidate metadata type would otherwise be unused in this module.
#[allow(dead_code)]
type UnusedMetadata = HashMap<String, String>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::RerankRuntimeConfig;

    fn candidate(id: &str, score: f32) -> RerankCandidate {
        RerankCandidate {
            id: id.to_string(),
            content: format!("content-{id}"),
            file_path: "a.rs".to_string(),
            initial_score: score,
            entity_type: None,
            metadata: HashMap::new(),
        }
    }

    fn request(candidates: Vec<RerankCandidate>) -> RerankRequest {
        RerankRequest {
            query: "q".to_string(),
            candidates,
            config: RerankRuntimeConfig::default(),
        }
    }

    #[tokio::test]
    async fn orders_by_initial_score() {
        let mock = MockRerankProvider::by_score();
        let result = mock
            .rerank(&request(vec![
                candidate("low", 0.1),
                candidate("high", 0.9),
            ]))
            .await
            .unwrap_or_else(|e| panic!("rerank must succeed: {e}"));
        assert_eq!(result.reranked_candidates[0].id, "high");
    }

    #[tokio::test]
    async fn orders_by_explicit_id_list() {
        let mock = MockRerankProvider::with_order(vec!["b".to_string(), "a".to_string()]);
        let result = mock
            .rerank(&request(vec![candidate("a", 0.9), candidate("b", 0.1)]))
            .await
            .unwrap_or_else(|e| panic!("rerank must succeed: {e}"));
        assert_eq!(result.reranked_candidates[0].id, "b");
    }

    #[tokio::test]
    async fn scripted_failure_then_recover() {
        let mock = MockRerankProvider::by_score().with_steps(vec![MockRerankStep::Fail(
            RerankError::Provider {
                status: 500,
                message: "boom".to_string(),
                retry_after_ms: None,
            },
        )]);
        assert!(mock
            .rerank(&request(vec![candidate("a", 1.0)]))
            .await
            .is_err());
        assert!(mock
            .rerank(&request(vec![candidate("a", 1.0)]))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn records_candidate_counts() {
        let mock = MockRerankProvider::by_score();
        let _ = mock
            .rerank(&request(vec![candidate("a", 1.0), candidate("b", 0.5)]))
            .await;
        assert_eq!(mock.recorded_candidate_counts(), vec![2]);
    }
}
