//! Mapping of the Gemini finish reasons onto the unified vocabulary.

/// Map a raw Gemini `finishReason` onto the unified finish-reason
/// vocabulary used across codecs.
///
/// The whole safety family maps to `content_filter`: Gemini splits it
/// over many members (`SAFETY`, `IMAGE_SAFETY`, `NO_IMAGE`, …) while the
/// OpenAI formats report a single value for the same situation.
pub fn normalize_finish_reason(raw: &str) -> String {
    match raw {
        "STOP" => "stop".to_string(),
        "MAX_TOKENS" => "length".to_string(),
        "SAFETY"
        | "RECITATION"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "NO_IMAGE"
        | "IMAGE_RECITATION"
        | "IMAGE_OTHER" => "content_filter".to_string(),
        other => other.to_ascii_lowercase(),
    }
}
