//! HTTP request assembly for the Gemini native API.

use crate::error::LlmResult;
use llm_suite_types::llm::LlmProfile;
use reqwest::Method;

/// Build a POST request against `url` with a JSON body, then apply the
/// profile auth/headers. Auth is header-only (`x-goog-api-key` for
/// native Gemini auth); the API key is never placed in the URL query.
/// A process-wide shared client is used purely to assemble the request;
/// execution happens on the caller's client with connection pooling.
pub fn build_json_request(
    url: String,
    body: &serde_json::Value,
    profile: &LlmProfile,
) -> LlmResult<reqwest::Request> {
    static BUILD_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    let build_client = BUILD_CLIENT.get_or_init(reqwest::Client::new);

    let payload = serde_json::to_vec(body).map_err(|e| {
        crate::error::LlmError::ConfigError(format!("failed to serialize request body: {}", e))
    })?;

    let builder = build_client
        .request(Method::POST, &url)
        .header("Content-Type", "application/json")
        .body(payload);

    let builder = super::super::shared::apply_auth_and_headers(builder, profile, "native");

    builder.build().map_err(crate::error::LlmError::HttpError)
}
