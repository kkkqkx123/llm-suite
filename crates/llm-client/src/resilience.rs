//! Shared resilience stack (limiter + breaker + retry) for capability
//! providers.
//!
//! Chat traffic gets these protections inside the gateway; embedding and
//! rerank providers inject the same components here so all three call paths
//! share one implementation and, via the gateway's per-endpoint registries,
//! one set of breaker/limiter instances per upstream.

use std::sync::Arc;

use llm_common::provider_error::ResilienceError;
use llm_common::retry::{RetryPolicy, execute_with_retry_floor};
use tokio_util::sync::CancellationToken;

use crate::CircuitBreaker;
pub use llm_common::ratelimit::RateLimiter;

/// Resilience components for one provider endpoint. Every field is optional:
/// an absent component disables that protection.
#[derive(Clone, Default)]
pub struct Resilience {
    breaker: Option<Arc<CircuitBreaker>>,
    limiter: Option<Arc<RateLimiter>>,
    retry: Option<RetryPolicy>,
}

impl std::fmt::Debug for Resilience {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resilience")
            .field("breaker", &self.breaker.is_some())
            .field("limiter", &self.limiter.is_some())
            .field("retry", &self.retry.is_some())
            .finish()
    }
}

impl Resilience {
    /// A stack without any protection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a circuit breaker shared per upstream endpoint.
    pub fn with_breaker(mut self, breaker: Arc<CircuitBreaker>) -> Self {
        self.breaker = Some(breaker);
        self
    }

    /// Attaches a token-bucket rate limiter shared per upstream endpoint.
    pub fn with_limiter(mut self, limiter: Arc<RateLimiter>) -> Self {
        self.limiter = Some(limiter);
        self
    }

    /// Attaches the retry budget applied to transient failures.
    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }

    /// Whether a circuit breaker is attached and currently rejects requests.
    pub fn is_breaker_open(&self) -> bool {
        self.breaker
            .as_ref()
            .is_some_and(|breaker| !breaker.check_allowed())
    }

    /// Runs `operation` under the attached limiter, breaker and retry policy.
    ///
    /// `open_error` builds the error surfaced when the breaker rejects the
    /// request before it reaches the wire. A rate-limit error raises the
    /// retry delay to at least the provider's `Retry-After` window. Breaker
    /// accounting counts successes and network-level failures only.
    pub async fn execute<E, F, Fut, T>(&self, open_error: impl Fn() -> E, operation: F) -> Result<T, E>
    where
        E: ResilienceError,
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        if let Some(limiter) = &self.limiter {
            limiter.acquire().await;
        }
        if let Some(breaker) = &self.breaker {
            if !breaker.check_allowed() {
                return Err(open_error());
            }
        }

        let result = execute_with_retry_floor(
            self.retry.as_ref(),
            |r| matches!(r, Err(e) if e.is_retryable()),
            |r| match r {
                Err(e) => e.to_string(),
                Ok(_) => String::new(),
            },
            None,
            None::<(&CancellationToken, E)>,
            |r| match r {
                Err(e) => e.retry_after_ms().unwrap_or(0),
                Ok(_) => 0,
            },
            operation,
        )
        .await;

        if let Some(breaker) = &self.breaker {
            match &result {
                Ok(_) => breaker.record_success(),
                Err(e) if e.is_network_failure() => breaker.record_failure(),
                Err(_) => {}
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug)]
    enum TestError {
        Transient,
        Permanent,
        RateLimited(u64),
        BreakerOpen,
    }

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Transient => write!(f, "transient"),
                Self::Permanent => write!(f, "permanent"),
                Self::RateLimited(_) => write!(f, "rate limited"),
                Self::BreakerOpen => write!(f, "breaker open"),
            }
        }
    }

    impl ResilienceError for TestError {
        fn is_retryable(&self) -> bool {
            matches!(self, Self::Transient | Self::RateLimited(_))
        }

        fn retry_after_ms(&self) -> Option<u64> {
            match self {
                Self::RateLimited(ms) => Some(*ms),
                _ => None,
            }
        }

        fn is_network_failure(&self) -> bool {
            matches!(self, Self::Transient | Self::RateLimited(_))
        }
    }

    #[tokio::test]
    async fn retries_transient_until_success_without_breaker() {
        let calls = Arc::new(AtomicU32::new(0));
        let resilience = Resilience::new().with_retry(RetryPolicy {
            max_retries: 3,
            base_delay_ms: 1,
            exponential_backoff: false,
        });
        let seen = calls.clone();
        let result = resilience
            .execute(
                || TestError::BreakerOpen,
                || {
                    let seen = seen.clone();
                    async move {
                        let call = seen.fetch_add(1, Ordering::SeqCst);
                        if call < 2 {
                            Err(TestError::Transient)
                        } else {
                            Ok(call)
                        }
                    }
                },
            )
            .await;
        assert_eq!(result.expect("must succeed"), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn permanent_failure_is_not_retried() {
        let calls = Arc::new(AtomicU32::new(0));
        let resilience = Resilience::new().with_retry(RetryPolicy {
            max_retries: 5,
            base_delay_ms: 1,
            exponential_backoff: false,
        });
        let seen = calls.clone();
        let result: Result<(), TestError> = resilience
            .execute(
                || TestError::BreakerOpen,
                || {
                    let seen = seen.clone();
                    async move {
                        seen.fetch_add(1, Ordering::SeqCst);
                        Err(TestError::Permanent)
                    }
                },
            )
            .await;
        assert!(matches!(result, Err(TestError::Permanent)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn breaker_open_rejects_before_the_wire() {
        let breaker = Arc::new(CircuitBreaker::new(crate::CircuitBreakerConfig {
            min_samples: 1,
            failure_threshold: 1.0,
            open_duration_ms: 60_000,
            half_open_max_probes: 1,
        }));
        breaker.record_failure();
        let resilience = Resilience::new().with_breaker(breaker);
        let result: Result<(), TestError> = resilience
            .execute(|| TestError::BreakerOpen, || async { Ok(()) })
            .await;
        assert!(matches!(result, Err(TestError::BreakerOpen)));
    }

    #[tokio::test]
    async fn failures_feed_the_breaker_window() {
        let breaker = Arc::new(CircuitBreaker::new(crate::CircuitBreakerConfig {
            min_samples: 2,
            failure_threshold: 1.0,
            open_duration_ms: 60_000,
            half_open_max_probes: 1,
        }));
        let resilience = Resilience::new()
            .with_breaker(breaker.clone())
            .with_retry(RetryPolicy {
                max_retries: 0,
                base_delay_ms: 1,
                exponential_backoff: false,
            });
        let _: Result<(), TestError> = resilience
            .execute(|| TestError::BreakerOpen, || async { Err(TestError::Transient) })
            .await;
        // Two counted failures (window is fed once per execute call, but the
        // retry loop inside one call feeds nothing extra) — after the second
        // execute the breaker must be open.
        let _: Result<(), TestError> = resilience
            .execute(|| TestError::BreakerOpen, || async { Err(TestError::Transient) })
            .await;
        assert!(
            !breaker.check_allowed(),
            "breaker must open after repeated network failures"
        );
    }
}
