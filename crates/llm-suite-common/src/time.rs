use chrono::Utc;

pub use llm_suite_types::Timestamp;

pub fn now() -> Timestamp {
    Utc::now().timestamp_millis()
}
