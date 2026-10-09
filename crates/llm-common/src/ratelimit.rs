use std::sync::Mutex;

use tokio::time::{Duration, Instant};

/// Token-bucket rate limiter: `acquire()` waits until a token is available.
///
/// The bucket refills continuously at `requests_per_second` and may hold up
/// to `burst` tokens, allowing short bursts above the steady rate.
#[derive(Debug)]
pub struct RateLimiter {
    requests_per_second: f64,
    burst: f64,
    state: Mutex<BucketState>,
}

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    /// Create a limiter allowing `requests_per_second` steady-state with an
    /// initial/full-burst capacity of `burst` tokens.
    pub fn new(requests_per_second: f64, burst: u32) -> Self {
        Self {
            requests_per_second: requests_per_second.max(0.001),
            burst: f64::from(burst.max(1)),
            state: Mutex::new(BucketState {
                tokens: f64::from(burst.max(1)),
                last_refill: Instant::now(),
            }),
        }
    }

    /// Wait until a token is available, then consume it.
    pub async fn acquire(&self) {
        loop {
            let wait = {
                let mut state = match self.state.lock() {
                    Ok(s) => s,
                    Err(poisoned) => {
                        let mut inner = poisoned.into_inner();
                        inner.tokens = self.burst;
                        inner.last_refill = Instant::now();
                        self.state.lock().unwrap_or_else(|p| p.into_inner())
                    }
                };
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_refill);
                state.tokens = (state.tokens + elapsed.as_secs_f64() * self.requests_per_second)
                    .min(self.burst);
                state.last_refill = now;
                if state.tokens >= 1.0 {
                    state.tokens -= 1.0;
                    None
                } else {
                    // Time until one token accumulates.
                    Some(Duration::from_secs_f64(
                        (1.0 - state.tokens) / self.requests_per_second,
                    ))
                }
            };
            match wait {
                None => return,
                Some(d) => tokio::time::sleep(d).await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn burst_is_immediate() {
        let limiter = RateLimiter::new(1.0, 3);
        let start = Instant::now();
        for _ in 0..3 {
            limiter.acquire().await;
        }
        assert!(start.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn steady_rate_throttles() {
        let limiter = RateLimiter::new(20.0, 1);
        let start = Instant::now();
        for _ in 0..3 {
            limiter.acquire().await;
        }
        // Two extra tokens at 20 rps ≈ 100ms total.
        assert!(start.elapsed() >= Duration::from_millis(80));
    }
}
