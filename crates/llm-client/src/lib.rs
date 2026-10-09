// Transport layer: client trait and HTTP implementation, stream accumulation,
// dead-loop guard, generic usage sink and feature-gated mocks.
pub mod circuit;
pub mod client;
pub mod dead_loop_detector;
pub mod resilience;
pub mod stream;
pub mod token_stream;

#[cfg(feature = "mock")]
pub mod http_mock;
#[cfg(feature = "mock")]
pub mod mock;

pub use circuit::{CircuitBreaker, CircuitBreakerConfig};
pub use client::LlmClient;
pub use dead_loop_detector::{DeadLoopDetectionResult, DeadLoopDetector, DeadLoopDetectorConfig};
pub use resilience::{RateLimiter, Resilience};
#[cfg(feature = "mock")]
pub use mock::{LlmResponseSpec, MockLlmClient, MockMessageStream};
pub use stream::MessageStream;
pub use token_stream::{
    LlmMetricsSink, SharedLlmMetricsSink, SharedTokenUsageSink, TokenRecordingStream,
    TokenUsageSink,
};
