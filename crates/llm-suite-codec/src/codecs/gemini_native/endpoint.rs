//! Endpoint construction for the Gemini native API.

use llm_suite_types::llm::LlmProfile;

/// Effective API endpoint base.
pub fn endpoint_base<'a>(base_url: &'a str, profile: &'a LlmProfile) -> &'a str {
    profile.base_url.as_deref().unwrap_or(base_url)
}

/// The model segment of the endpoint path.
///
/// The model lives in the URL, not in the body, and the path segment is
/// always `models/{model}`. A model that already carries the prefix is
/// not prefixed twice.
pub fn model_segment(model: &str) -> String {
    let model = model.trim();
    if model.starts_with("models/") || model.starts_with("tunedModels/") {
        model.to_string()
    } else {
        format!("models/{model}")
    }
}
