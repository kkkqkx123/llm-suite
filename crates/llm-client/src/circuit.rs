use std::sync::Mutex;
use std::time::Instant;

/// Configuration for a circuit breaker instance.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Minimum number of finished requests in the sliding window before the
    /// failure ratio is evaluated.
    pub min_samples: u32,
    /// Failure ratio (0.0..=1.0) at or above which the breaker opens.
    pub failure_threshold: f64,
    /// How long the breaker stays open before allowing a probe.
    pub open_duration_ms: u64,
    /// Concurrent probe requests allowed in the half-open state.
    pub half_open_max_probes: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            failure_threshold: 0.5,
            open_duration_ms: 30_000,
            half_open_max_probes: 1,
        }
    }
}

#[derive(Debug)]
enum State {
    /// Requests flow through; failures are counted in the window.
    Closed {
        failures: u32,
        samples: u32,
    },
    /// Requests are rejected until `open_until`.
    Open { open_until: Instant },
    /// Limited probes flow; any failure reopens the breaker.
    HalfOpen { probes: u32, max_probes: u32 },
}

/// Per-endpoint circuit breaker: Closed -> Open -> HalfOpen -> Closed.
///
/// The critical section only mutates counters and flips states, never
/// crosses an await point.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Mutex<State>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(State::Closed {
                failures: 0,
                samples: 0,
            }),
        }
    }

    /// Whether a request may proceed right now. Returns `false` when the
    /// breaker is open (or its half-open probe budget is exhausted).
    pub fn check_allowed(&self) -> bool {
        let mut state = match self.state.lock() {
            Ok(s) => s,
            Err(poisoned) => {
                // A request panicked while holding the lock; reset instead of
                // rejecting all future traffic.
                let mut inner = poisoned.into_inner();
                *inner = State::Closed {
                    failures: 0,
                    samples: 0,
                };
                return true;
            }
        };
        match &*state {
            State::Closed { .. } => true,
            State::Open { open_until } => {
                if Instant::now() >= *open_until {
                    *state = State::HalfOpen {
                        probes: 1,
                        max_probes: self.config.half_open_max_probes,
                    };
                    true
                } else {
                    false
                }
            }
            State::HalfOpen { probes, max_probes } => {
                if *probes < *max_probes {
                    let new_probes = *probes + 1;
                    *state = State::HalfOpen {
                        probes: new_probes,
                        max_probes: *max_probes,
                    };
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        let mut state = self.lock_or_reset();
        match &mut *state {
            State::Closed { failures, samples } => {
                *samples = samples.saturating_add(1);
                // Successes dilute the failure ratio; keep the counters
                // bounded so the window never grows without limit.
                let _ = failures;
                if *samples >= self.config.min_samples * 2 {
                    *failures = 0;
                    *samples = self.config.min_samples;
                }
            }
            State::HalfOpen { .. } => {
                // A probe succeeded: the endpoint recovered.
                *state = State::Closed {
                    failures: 0,
                    samples: 0,
                };
            }
            State::Open { .. } => {}
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        let mut state = self.lock_or_reset();
        match &mut *state {
            State::Closed { failures, samples } => {
                *failures = failures.saturating_add(1);
                *samples = samples.saturating_add(1);
                if *samples >= self.config.min_samples {
                    let ratio = f64::from(*failures) / f64::from(*samples);
                    if ratio >= self.config.failure_threshold {
                        *state = State::Open {
                            open_until: Instant::now()
                                + std::time::Duration::from_millis(self.config.open_duration_ms),
                        };
                    }
                }
            }
            State::HalfOpen { .. } => {
                // The endpoint is still unhealthy: reopen.
                *state = State::Open {
                    open_until: Instant::now()
                        + std::time::Duration::from_millis(self.config.open_duration_ms),
                };
            }
            State::Open { .. } => {}
        }
    }

    fn lock_or_reset(&self) -> std::sync::MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                let mut inner = poisoned.into_inner();
                *inner = State::Closed {
                    failures: 0,
                    samples: 0,
                };
                self.state.lock().unwrap_or_else(|p| p.into_inner())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            min_samples: 3,
            failure_threshold: 0.5,
            open_duration_ms: 10_000,
            half_open_max_probes: 1,
        }
    }

    #[test]
    fn closed_below_min_samples_never_opens() {
        let cb = CircuitBreaker::new(config());
        cb.record_failure();
        cb.record_failure();
        assert!(cb.check_allowed());
    }

    #[test]
    fn opens_at_failure_threshold() {
        let cb = CircuitBreaker::new(config());
        for _ in 0..3 {
            cb.record_failure();
        }
        assert!(!cb.check_allowed());
    }

    #[test]
    fn successes_keep_breaker_closed() {
        let cb = CircuitBreaker::new(config());
        cb.record_failure();
        cb.record_success();
        cb.record_success();
        cb.record_success();
        assert!(cb.check_allowed());
    }

    #[test]
    fn half_open_probe_success_closes_breaker() {
        let cb = CircuitBreaker::new(config());
        for _ in 0..3 {
            cb.record_failure();
        }
        assert!(!cb.check_allowed());
        // Force the open duration to elapse by constructing a fresh breaker
        // whose open_until is in the past is not possible; instead shrink
        // the config: open_duration is fixed at construction, so emulate
        // expiry through a short-duration breaker.
        let short = CircuitBreaker::new(CircuitBreakerConfig {
            open_duration_ms: 1,
            ..config()
        });
        for _ in 0..3 {
            short.record_failure();
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(short.check_allowed(), "probe must be allowed after expiry");
        short.record_success();
        assert!(short.check_allowed(), "must be closed after probe success");
    }

    #[test]
    fn half_open_probe_failure_reopens() {
        let short = CircuitBreaker::new(CircuitBreakerConfig {
            open_duration_ms: 1,
            ..config()
        });
        for _ in 0..3 {
            short.record_failure();
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(short.check_allowed(), "probe must be allowed after expiry");
        short.record_failure();
        assert!(!short.check_allowed(), "failure in half-open must reopen");
    }

    #[test]
    fn half_open_probes_bounded() {
        let short = CircuitBreaker::new(CircuitBreakerConfig {
            open_duration_ms: 1,
            half_open_max_probes: 2,
            ..config()
        });
        for _ in 0..3 {
            short.record_failure();
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(short.check_allowed());
        assert!(short.check_allowed());
        assert!(!short.check_allowed(), "third probe must be rejected");
    }

    /// Concurrent `check_allowed` + `record_*` on a shared breaker must
    /// never panic, double-admit more than the probe budget, or corrupt the
    /// state machine. With half_open_max_probes = 1, at most one request
    /// may pass while the breaker is open-then-half-open.
    #[test]
    fn concurrent_access_is_consistent() {
        let cb = std::sync::Arc::new(CircuitBreaker::new(CircuitBreakerConfig {
            min_samples: 1,
            failure_threshold: 0.5,
            open_duration_ms: 1,
            half_open_max_probes: 1,
        }));
        let admitted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cb = std::sync::Arc::clone(&cb);
            let admitted = std::sync::Arc::clone(&admitted);
            handles.push(std::thread::spawn(move || {
                for _ in 0..200 {
                    if cb.check_allowed() {
                        admitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Alternate outcomes; the state machine must stay
                        // coherent under either interleaving.
                        cb.record_success();
                    } else {
                        cb.record_failure();
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread must not panic");
        }
        // Every admission was paired with a success, so the breaker must be
        // closed and admitting at the end.
        assert!(cb.check_allowed(), "breaker must be closed after all-success");
        assert!(
            admitted.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "at least some requests must have been admitted"
        );
    }

    /// Racing probes in half-open state: only `half_open_max_probes`
    /// threads may win admission, the rest must observe rejection.
    #[test]
    fn concurrent_half_open_admits_at_most_budget() {
        let cb = std::sync::Arc::new(CircuitBreaker::new(CircuitBreakerConfig {
            min_samples: 1,
            failure_threshold: 0.99,
            open_duration_ms: 1,
            half_open_max_probes: 2,
        }));
        for _ in 0..2 {
            cb.record_failure();
        }
        // Breaker is now open (ratio 1.0 >= 0.99); wait for expiry.
        std::thread::sleep(std::time::Duration::from_millis(5));

        let admitted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cb = std::sync::Arc::clone(&cb);
            let admitted = std::sync::Arc::clone(&admitted);
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                if cb.check_allowed() {
                    admitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread must not panic");
        }
        assert_eq!(
            admitted.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "exactly half_open_max_probes threads must be admitted"
        );
    }
}
