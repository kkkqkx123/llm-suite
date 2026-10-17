use std::time::Duration;

use tokio_util::sync::CancellationToken;

/// Retry policy: maximum retry count, base delay and whether the delay grows
/// exponentially (`base * 2^(attempt-1)`).
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub exponential_backoff: bool,
}

impl RetryPolicy {
    /// Delay before `attempt` (1-based) in ms. `floor_ms` (e.g. a provider
    /// `Retry-After` value) raises the computed delay so rate-limit hints
    /// are honored even when the backoff would fire earlier.
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        self.delay_for_attempt_with_floor(attempt, 0)
    }

    /// Delay before `attempt` (1-based) in ms, bounded below by `floor_ms`.
    pub fn delay_for_attempt_with_floor(&self, attempt: u32, floor_ms: u64) -> u64 {
        let computed = if self.exponential_backoff {
            self.base_delay_ms * 2u64.pow(attempt.saturating_sub(1))
        } else {
            self.base_delay_ms
        };
        computed.max(floor_ms)
    }
}

/// One scheduled retry attempt: the emission payload shared by the sync
/// `on_retry` callback and any async retry event metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryAttemptDescriptor {
    /// 1-based attempt number about to run (first retry is 1).
    pub attempt: u32,
    /// Policy cap the attempt counts against.
    pub max_retries: u32,
    /// Delay before the attempt in ms.
    pub delay_ms: u64,
    /// Why the previous attempt is retryable (caller-provided label).
    pub reason: String,
}

/// Execute `operation` with retries governed by `policy`. `should_retry`
/// decides whether the latest result warrants another attempt. When `policy`
/// is `None` the operation runs exactly once (fail fast). An optional cancel
/// token interrupts the retry delay, returning the paired error value.
/// `reason` labels the failure for the `on_retry` observation callback.
/// `delay_floor` optionally extracts a minimum wait from the latest result
/// (e.g. a provider `Retry-After` hint on a rate-limit error), raising the
/// scheduled delay above the plain backoff value.
pub async fn execute_with_retry_observed<F, Fut, T, E>(
    policy: Option<&RetryPolicy>,
    should_retry: impl Fn(&Result<T, E>) -> bool,
    reason: impl Fn(&Result<T, E>) -> String,
    on_retry: Option<&(dyn Fn(&RetryAttemptDescriptor) + Send + Sync)>,
    cancel: Option<(&CancellationToken, E)>,
    operation: F,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    execute_with_retry_floor(
        policy,
        should_retry,
        reason,
        on_retry,
        cancel,
        |_| 0,
        operation,
    )
    .await
}

/// Like [`execute_with_retry_observed`] with a per-result delay floor.
pub async fn execute_with_retry_floor<F, Fut, T, E>(
    policy: Option<&RetryPolicy>,
    should_retry: impl Fn(&Result<T, E>) -> bool,
    reason: impl Fn(&Result<T, E>) -> String,
    on_retry: Option<&(dyn Fn(&RetryAttemptDescriptor) + Send + Sync)>,
    cancel: Option<(&CancellationToken, E)>,
    delay_floor: impl Fn(&Result<T, E>) -> u64,
    operation: F,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let Some(policy) = policy else {
        return operation().await;
    };
    let (mut cancel_err, cancel_token) = match cancel {
        Some((token, err)) => (Some(err), Some(token)),
        None => (None, None),
    };
    let mut attempt = 0u32;
    loop {
        let result = operation().await;
        if !should_retry(&result) || attempt >= policy.max_retries {
            return result;
        }
        attempt += 1;
        let delay_ms = policy.delay_for_attempt_with_floor(attempt, delay_floor(&result));
        if let Some(callback) = on_retry {
            callback(&RetryAttemptDescriptor {
                attempt,
                max_retries: policy.max_retries,
                delay_ms,
                reason: reason(&result),
            });
        }
        if let Some(token) = &cancel_token {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
                _ = token.cancelled() => {
                    return Err(cancel_err.take().expect("cancel token implies error"));
                }
            }
        } else {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
    }
}
