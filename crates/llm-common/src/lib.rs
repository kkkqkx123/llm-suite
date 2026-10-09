//! Shared runtime helpers for the llm-suite crates.
//!
//! Leaf utilities that the provider crates need but that are independent of
//! any host project: poisoned-lock recovery, wall-clock time, id generation,
//! retry execution and timeout wrapping.

pub mod exec;
pub mod http;
pub mod id;
pub mod lock;
pub mod ratelimit;
pub mod retry;
pub mod time;

pub use http::parse_retry_after_ms;
pub use id::generate_id;
pub use lock::lock_ok;
pub use time::now;
