//! Request observation without dynamic dispatch.
//!
//! The probe is a concrete value of plain `fn` pointers. `None` is the hot-path default and costs a
//! single branch. When set, the gateway invokes the hooks around dispatch
//! so debuggers and middleware can observe the typed request and the typed
//! outcome. Wire-payload inspection stays on `tracing` spans; mutation or
//! header injection stays on `metadata` / `custom_body`, which already
//! travel through the codec.

use llm_suite_codec::error::LlmError;
use llm_suite_types::llm::{LlmProfile, LlmRequest, LlmResult as LlmResponseType};

/// Observation hooks for one gateway call.
///
/// All hooks are plain function pointers so the probe stays `Copy` and
/// never introduces dynamic dispatch. Hosts needing captured state should
/// funnel through `tracing` spans or the metrics sink instead.
#[derive(Clone, Copy, Default, Debug)]
pub struct RequestProbe {
    /// Invoked after profile resolution and request merging, before dispatch.
    pub on_request: Option<fn(&LlmRequest, &LlmProfile)>,
    /// Invoked after a `generate` attempt finishes (success or failure).
    /// Carries the elapsed dispatch time in milliseconds.
    pub on_response: Option<fn(&LlmRequest, &Result<LlmResponseType, LlmError>, f64)>,
    /// Invoked when a `generate_stream` establishment finishes (`Ok` when
    /// the stream was established). Stream consumption itself is observed
    /// through the metrics sink.
    pub on_stream_start: Option<fn(&LlmRequest, bool, f64)>,
}

impl RequestProbe {
    /// True when no hook is set; the gateway skips every probe branch.
    pub fn is_empty(&self) -> bool {
        self.on_request.is_none() && self.on_response.is_none() && self.on_stream_start.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_request(_req: &LlmRequest, _profile: &LlmProfile) {}
    fn on_response(_req: &LlmRequest, _res: &Result<LlmResponseType, LlmError>, _ms: f64) {}

    #[test]
    fn empty_by_default() {
        assert!(RequestProbe::default().is_empty());
    }

    #[test]
    fn non_empty_when_any_hook_set() {
        let probe = RequestProbe {
            on_request: Some(on_request),
            on_response: None,
            on_stream_start: None,
        };
        assert!(!probe.is_empty());
        let probe = RequestProbe {
            on_request: None,
            on_response: Some(on_response),
            on_stream_start: None,
        };
        assert!(!probe.is_empty());
    }

    #[test]
    fn probe_is_copy() {
        let probe = RequestProbe {
            on_request: Some(on_request),
            on_response: None,
            on_stream_start: None,
        };
        let copied = probe;
        assert!(!copied.is_empty());
        assert!(!probe.is_empty());
    }
}
