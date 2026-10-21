//! The identifier assigned to a tool call that the text protocol produced.
//!
//! The text protocol carries no call id, so one is generated per call. It
//! only has to be unique inside the turn.

/// Generate the identifier of a tool call.
pub(super) fn generate_tool_call_id() -> String {
    format!(
        "call_{}_{}",
        llm_suite_common::time::now(),
        llm_suite_common::generate_id()
    )
}
