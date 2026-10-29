//! Provider-agnostic HTTP plumbing: endpoint URL composition, authentication
//! headers, query parameters and the custom body merge.

use crate::error::LlmResult;
use llm_suite_types::llm::LlmProfile;

/// Build the endpoint URL for a provider path.
///
/// A base URL that already ends with the path is used as-is, so a profile can
/// carry a full endpoint including a query string (deployment-style
/// endpoints). Otherwise the path is appended. A trailing slash on the base
/// URL never produces a double slash.
///
/// The path may carry its own query string (Gemini's streaming endpoint uses
/// `?alt=sse`). The base query and the path query are merged instead of
/// concatenated, so a base URL that already carries a query (Azure's
/// `?api-version=...`) still yields a well-formed URL.
pub fn endpoint_url(base_url: &str, path: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let (base_without_query, base_query) = match trimmed.split_once('?') {
        Some((base, query)) => (base, Some(query)),
        None => (trimmed, None),
    };
    let (path_without_query, path_query) = match path.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (path, None),
    };
    let url = if base_without_query.ends_with(&format!("/{path_without_query}")) {
        base_without_query.to_string()
    } else {
        format!("{base_without_query}/{path_without_query}")
    };
    let mut query_parts: Vec<&str> = Vec::new();
    if let Some(query) = base_query {
        query_parts.push(query);
    }
    if let Some(query) = path_query {
        query_parts.push(query);
    }
    if query_parts.is_empty() {
        url
    } else {
        format!("{url}?{}", query_parts.join("&"))
    }
}

/// Assemble the POST JSON request shared by a codec's inference and
/// count-tokens endpoints, so header handling cannot drift apart between
/// the two. `fixed_headers` carries format-specific constants (e.g. the
/// Anthropic `anthropic-version` header); auth, profile headers and query
/// parameters go through `apply_auth_and_headers`.
pub fn build_post_json_request(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    profile: &LlmProfile,
    default_auth: &str,
    fixed_headers: &[(&str, &str)],
) -> LlmResult<reqwest::Request> {
    let mut req_builder = client
        .request(reqwest::Method::POST, url)
        .header("Content-Type", "application/json");
    for (name, value) in fixed_headers {
        req_builder = req_builder.header(*name, *value);
    }
    let req_builder = req_builder.json(body);

    apply_auth_and_headers(req_builder, profile, default_auth)
        .build()
        .map_err(crate::error::LlmError::HttpError)
}

/// Apply authentication, identity, custom headers and query parameters to
/// the request.
///
/// Auth type resolution: `profile.auth_type` wins when set ("native" /
/// "bearer" / "x-api-key" / "x-goog-api-key"), otherwise `default_auth` is
/// used. "native" maps to the format-specific header (x-api-key for Anthropic,
/// x-goog-api-key for Gemini native, Bearer otherwise).
///
/// Header precedence per name: identity defaults (`User-Agent` /
/// `HTTP-Referer` / `X-Title`) first, then `headers`, then `custom_headers`.
/// Headers are written once per name: `reqwest` appends duplicate header
/// values instead of replacing them, so every layer merges into a single
/// table before it is set. Per-request `LlmRequest.extra_headers` are
/// applied later by the client and win over all of these.
pub fn apply_auth_and_headers(
    req_builder: reqwest::RequestBuilder,
    profile: &LlmProfile,
    default_auth: &str,
) -> reqwest::RequestBuilder {
    let mut builder = req_builder;

    if let Some(api_key) = &profile.api_key {
        let auth_type = profile.auth_type.as_deref().unwrap_or(default_auth);
        match auth_type {
            "native" => match profile.format {
                llm_suite_types::llm::LlmFormat::Anthropic => {
                    builder = builder.header("x-api-key", api_key);
                }
                llm_suite_types::llm::LlmFormat::GeminiNative => {
                    builder = builder.header("x-goog-api-key", api_key);
                }
                _ => {
                    builder = builder.header("Authorization", format!("Bearer {}", api_key));
                }
            },
            "x-api-key" => {
                builder = builder.header("x-api-key", api_key);
            }
            "x-goog-api-key" => {
                builder = builder.header("x-goog-api-key", api_key);
            }
            _ => {
                builder = builder.header("Authorization", format!("Bearer {}", api_key));
            }
        }
    }

    for (key, value) in merged_profile_headers(profile) {
        builder = builder.header(key, value);
    }
    if let Some(params) = &profile.query_params {
        let pairs: Vec<(&str, &str)> = params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str().unwrap_or("")))
            .collect();
        builder = builder.query(&pairs);
    }

    builder
}

/// The profile headers as a single name → value table.
///
/// Layering per name: identity defaults first, then `headers`, then
/// `custom_headers` (most explicit wins). The request must carry exactly one
/// value per header name.
fn merged_profile_headers(profile: &LlmProfile) -> Vec<(String, String)> {
    let mut merged: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    if let Some(identity) = profile.identity.as_ref() {
        if let Some(user_agent) = identity.user_agent.as_deref() {
            merged.insert("User-Agent".to_string(), user_agent.to_string());
        }
        if let Some(app_url) = identity.app_url.as_deref() {
            merged.insert("HTTP-Referer".to_string(), app_url.to_string());
        }
        if let Some(app_name) = identity.app_name.as_deref() {
            merged.insert("X-Title".to_string(), app_name.to_string());
        }
    }
    if let Some(headers) = &profile.headers {
        for (key, value) in headers {
            merged.insert(key.clone(), value.as_str().unwrap_or("").to_string());
        }
    }
    if let Some(custom_headers) = &profile.custom_headers {
        for (key, value) in custom_headers {
            merged.insert(key.clone(), value.as_str().unwrap_or("").to_string());
        }
    }
    merged.into_iter().collect()
}

/// Deep-merge `profile.custom_body` into the request body (custom body wins),
/// governed by `custom_body_enabled`.
pub fn apply_custom_body(body: &mut serde_json::Value, profile: &LlmProfile) {
    if profile.custom_body_enabled.unwrap_or(true) {
        if let Some(custom) = &profile.custom_body {
            *body = crate::codecs::helpers::deep_merge(body, custom);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::shared::test_support::profile_with;

    #[test]
    fn endpoint_url_appends_the_path_only_once() {
        assert_eq!(
            endpoint_url("https://api.openai.com/v1", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            endpoint_url("https://api.openai.com/v1/", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn endpoint_url_keeps_a_full_endpoint() {
        assert_eq!(
            endpoint_url(
                "https://example.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-01",
                "chat/completions"
            ),
            "https://example.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-01"
        );
    }

    #[test]
    fn endpoint_url_merges_the_base_query_with_the_path_query() {
        // A path that carries its own query (the Gemini streaming endpoint)
        // and a base that carries one as well must not be concatenated with a
        // second question mark.
        assert_eq!(
            endpoint_url(
                "https://generativelanguage.googleapis.com/v1beta",
                "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            endpoint_url(
                "https://gateway.example.com/v1?api-version=2024-01",
                "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
            ),
            "https://gateway.example.com/v1/models/gemini-2.5-pro:streamGenerateContent?api-version=2024-01&alt=sse"
        );
    }

    #[test]
    fn native_auth_uses_provider_header() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(Some("native")), "bearer")
            .build()
            .unwrap();
        assert_eq!(
            req.headers().get("x-api-key").unwrap(),
            "sk-test",
            "anthropic native auth must use x-api-key"
        );
        assert!(req.headers().get("Authorization").is_none());
    }

    #[test]
    fn bearer_auth_overrides_default() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(Some("bearer")), "x-api-key")
            .build()
            .unwrap();
        assert!(req.headers().get("x-api-key").is_none());
        assert_eq!(
            req.headers().get("Authorization").unwrap(),
            "Bearer sk-test"
        );
    }

    #[test]
    fn custom_headers_and_query_params_are_applied() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(None), "bearer")
            .build()
            .unwrap();
        assert_eq!(req.headers().get("x-custom").unwrap(), "v1");
        assert_eq!(req.url().query().unwrap(), "api-version=2024-01");
    }

    #[test]
    fn identity_headers_are_injected_once() {
        let mut profile = profile_with(None);
        profile.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: Some("my-app/1.0".to_string()),
            app_name: Some("my app".to_string()),
            app_url: Some("https://example.com".to_string()),
        });
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile, "bearer")
            .build()
            .unwrap();
        assert_eq!(req.headers().get("User-Agent").unwrap(), "my-app/1.0");
        assert_eq!(req.headers().get("X-Title").unwrap(), "my app");
        assert_eq!(
            req.headers().get("HTTP-Referer").unwrap(),
            "https://example.com"
        );
    }

    #[test]
    fn custom_headers_win_over_identity_defaults() {
        let mut profile = profile_with(None);
        profile.identity = Some(llm_suite_types::llm::LlmIdentity {
            user_agent: Some("identity/1.0".to_string()),
            app_name: None,
            app_url: None,
        });
        profile.custom_headers = Some(
            [("User-Agent".to_string(), serde_json::json!("explicit/2.0"))]
                .into_iter()
                .collect(),
        );
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile, "bearer")
            .build()
            .unwrap();
        let values: Vec<&str> = req
            .headers()
            .get_all("User-Agent")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        assert_eq!(
            values,
            vec!["explicit/2.0"],
            "explicit headers must win over identity defaults"
        );
    }

    #[test]
    fn absent_identity_sends_no_identity_header() {
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile_with(None), "bearer")
            .build()
            .unwrap();
        assert!(req.headers().get("User-Agent").is_none());
        assert!(req.headers().get("X-Title").is_none());
        assert!(req.headers().get("HTTP-Referer").is_none());
    }

    #[test]
    fn a_header_name_is_written_once() {
        let mut profile = profile_with(None);
        profile.headers = Some(
            [("x-dup".to_string(), serde_json::json!("from-headers"))]
                .into_iter()
                .collect(),
        );
        profile.custom_headers = Some(
            [(
                "x-dup".to_string(),
                serde_json::json!("from-custom-headers"),
            )]
            .into_iter()
            .collect(),
        );
        let req =
            reqwest::Client::new().request(reqwest::Method::POST, "https://example.com/messages");
        let req = apply_auth_and_headers(req, &profile, "bearer")
            .build()
            .unwrap();
        let values: Vec<&str> = req
            .headers()
            .get_all("x-dup")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        assert_eq!(
            values,
            vec!["from-custom-headers"],
            "a header name must carry exactly one value"
        );
    }
}
