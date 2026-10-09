use serde::{Deserialize, Serialize};

/// Declarative circuit breaker policy attached to a profile or provider
/// definition. The runtime state machine lives in `llm_client::CircuitBreaker`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CircuitBreakerConfig {
    /// Minimum finished requests in the sliding window before the failure
    /// ratio is evaluated.
    #[serde(default = "default_min_samples")]
    pub min_samples: u32,
    /// Failure ratio (0.0..=1.0) at or above which the breaker opens.
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: f64,
    /// How long the breaker stays open before allowing a probe (ms).
    #[serde(default = "default_open_duration_ms")]
    pub open_duration_ms: u64,
    /// Concurrent probe requests allowed in the half-open state.
    #[serde(default = "default_half_open_max_probes")]
    pub half_open_max_probes: u32,
}

fn default_min_samples() -> u32 {
    10
}

fn default_failure_threshold() -> f64 {
    0.5
}

fn default_open_duration_ms() -> u64 {
    30_000
}

fn default_half_open_max_probes() -> u32 {
    1
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            min_samples: default_min_samples(),
            failure_threshold: default_failure_threshold(),
            open_duration_ms: default_open_duration_ms(),
            half_open_max_probes: default_half_open_max_probes(),
        }
    }
}

/// Provider-level rate limit shared by all profiles pointing at the same
/// base URL. Tokens refill continuously at `requests_per_second` with a
/// maximum instantaneous `burst`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RateLimitConfig {
    /// Steady-state requests per second.
    pub requests_per_second: f64,
    /// Maximum burst size (bucket capacity).
    #[serde(default = "default_burst")]
    pub burst: u32,
}

fn default_burst() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_defaults_on_partial_json() {
        let config: CircuitBreakerConfig = serde_json::from_str("{}").unwrap_or_else(|e| {
            panic!("empty object must use serde defaults: {e}");
        });
        assert_eq!(config.min_samples, 10);
        assert!((config.failure_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.open_duration_ms, 30_000);
        assert_eq!(config.half_open_max_probes, 1);
    }

    #[test]
    fn rate_limit_burst_defaults_when_missing() {
        let config: RateLimitConfig = serde_json::from_str(r#"{"requests_per_second": 5.0}"#)
            .unwrap_or_else(|e| {
                panic!("missing burst must default: {e}");
            });
        assert!((config.requests_per_second - 5.0).abs() < f64::EPSILON);
        assert_eq!(config.burst, 1);
    }

    #[test]
    fn circuit_breaker_round_trips() {
        let config = CircuitBreakerConfig {
            min_samples: 3,
            failure_threshold: 0.8,
            open_duration_ms: 1_000,
            half_open_max_probes: 2,
        };
        let json = serde_json::to_string(&config).unwrap_or_else(|e| panic!("serialize: {e}"));
        let back: CircuitBreakerConfig =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize: {e}"));
        assert_eq!(back, config);
    }

    #[test]
    fn profile_omits_absent_proxy_and_circuit_breaker() {
        let json = serde_json::json!({
            "id": "p", "name": "p", "format": "openai-chat", "model": "m"
        });
        let text = serde_json::to_string(&json).unwrap_or_else(|e| panic!("serialize: {e}"));
        assert!(!text.contains("proxy"), "absent proxy must be omitted");
        assert!(
            !text.contains("circuit_breaker"),
            "absent circuit_breaker must be omitted"
        );
    }

    #[test]
    fn profile_deserializes_proxy_and_circuit_breaker() {
        let json = serde_json::json!({
            "id": "p", "name": "p", "format": "openai-chat", "model": "m",
            "proxy": "socks5://127.0.0.1:1080",
            "circuit_breaker": { "min_samples": 3 }
        });
        let profile: crate::llm::LlmProfile =
            serde_json::from_value(json).unwrap_or_else(|e| panic!("deserialize: {e}"));
        assert_eq!(profile.proxy.as_deref(), Some("socks5://127.0.0.1:1080"));
        let cb = profile
            .circuit_breaker
            .unwrap_or_else(|| panic!("circuit_breaker must parse"));
        assert_eq!(cb.min_samples, 3);
        assert_eq!(
            cb.open_duration_ms, 30_000,
            "unspecified fields must default"
        );
    }
}
