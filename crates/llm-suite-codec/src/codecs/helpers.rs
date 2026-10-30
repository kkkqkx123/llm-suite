use crate::error::{LlmError, LlmResult};
use llm_suite_types::llm::LlmProfile;
use llm_suite_types::message::MessageContent;
use std::collections::HashMap;

/// An image reference resolved into a form a provider can carry on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedImage {
    /// A `data:` URI: media type plus base64 payload (without the `data:...;base64,` prefix).
    Base64 { media_type: String, data: String },
    /// A remote http(s) URL, for providers that accept URL references directly.
    Url(String),
}

/// Parse an image reference (data URI or remote URL).
///
/// Never fabricates data: a malformed `data:` URI is an error, not a guess.
/// No network I/O happens here — remote URLs are returned as-is and providers
/// that cannot carry them must reject them explicitly.
pub fn resolve_image_payload(url: &str) -> LlmResult<ResolvedImage> {
    if let Some(rest) = url.strip_prefix("data:") {
        let Some(meta) = rest.split_once(',').map(|(m, _)| m) else {
            return Err(LlmError::ConfigError(format!(
                "malformed data URI (missing ','): {}",
                truncate(url)
            )));
        };
        let Some(media_type) = meta.strip_suffix(";base64") else {
            return Err(LlmError::ConfigError(format!(
                "data URI is not base64-encoded: {}",
                truncate(url)
            )));
        };
        let data = rest.split_once(',').map(|(_, d)| d).unwrap_or_default();
        if data.is_empty() {
            return Err(LlmError::ConfigError(format!(
                "data URI has empty payload: {}",
                truncate(url)
            )));
        }
        let media_type = if media_type.is_empty() {
            "application/octet-stream".to_string()
        } else {
            media_type.to_string()
        };
        Ok(ResolvedImage::Base64 {
            media_type,
            data: data.to_string(),
        })
    } else if url.starts_with("http://") || url.starts_with("https://") {
        Ok(ResolvedImage::Url(url.to_string()))
    } else {
        Err(LlmError::ConfigError(format!(
            "unsupported image reference scheme: {}",
            truncate(url)
        )))
    }
}

/// The name of a content block, used in the error messages.
pub fn block_name(block: &MessageContent) -> &'static str {
    match block {
        MessageContent::Text { .. } => "text",
        MessageContent::ImageUrl { .. } => "image_url",
        MessageContent::InputAudio { .. } => "input_audio",
        MessageContent::FileInput { .. } => "file_input",
        MessageContent::ToolUse { .. } => "tool_use",
        MessageContent::ToolResult { .. } => "tool_result",
        MessageContent::Thinking { .. } => "thinking",
        MessageContent::RedactedThinking { .. } => "redacted_thinking",
        MessageContent::Native { .. } => "native",
    }
}

/// The native blocks the Gemini format recognizes.
///
/// A Gemini part is identified by the field it carries, not by a `type`
/// tag, so the list names the part field of every native block the format
/// accepts.
pub(crate) const GEMINI_NATIVE_PART_FIELDS: [(&str, &str); 4] = [
    ("video_metadata", "videoMetadata"),
    ("executable_code", "executableCode"),
    ("code_execution_result", "codeExecutionResult"),
    ("media_resolution", "mediaResolution"),
];

/// Validate a native block and return the payload to put on the wire.
///
/// The payload is the provider's own block and is used verbatim; the kind
/// says what the block is. Two rules hold for every format: the block must
/// belong to the family the format serves, and the payload must carry the
/// field that identifies it — the `type` tag for the OpenAI and the
/// Anthropic formats, the part field for Gemini. A block that breaks
/// either rule is rejected instead of being sent as something the
/// provider never asked for.
pub fn native_block(
    provider: &str,
    format: &str,
    kind: &str,
    payload: &serde_json::Value,
) -> LlmResult<serde_json::Value> {
    if provider != format {
        return Err(LlmError::ConfigError(format!(
            "the native block `{kind}` belongs to the `{provider}` format; the {format} format cannot carry it"
        )));
    }
    if format == "gemini" {
        let Some((_, field)) = GEMINI_NATIVE_PART_FIELDS
            .iter()
            .find(|(name, _)| *name == kind)
        else {
            return Err(LlmError::ConfigError(format!(
                "gemini has no native part `{kind}`"
            )));
        };
        if payload.get(*field).is_none() {
            return Err(LlmError::ConfigError(format!(
                "the native block `{kind}` carries no `{field}` field"
            )));
        }
        return Ok(payload.clone());
    }
    match payload.get("type").and_then(|v| v.as_str()) {
        Some(block_type) if block_type == kind => Ok(payload.clone()),
        Some(block_type) => Err(LlmError::ConfigError(format!(
            "the native block `{kind}` carries the type `{block_type}`"
        ))),
        None => Err(LlmError::ConfigError(format!(
            "the native block `{kind}` has no `type` field"
        ))),
    }
}

/// Whether a remote reference can ride on Gemini's `fileData.fileUri`.
///
/// The Gemini API only accepts `gs://` URIs, Files API URIs and YouTube
/// URLs here — an arbitrary public https URL is rejected server-side and
/// must be inlined as a data URI (or uploaded via the Files API) instead.
pub fn is_gemini_file_uri(url: &str) -> bool {
    if url.starts_with("gs://") {
        return true;
    }
    if url.starts_with("https://") {
        let host_path_ok =
            url.contains("generativelanguage.googleapis.com") && url.contains("/files/");
        let youtube_ok = url.starts_with("https://youtube.com/")
            || url.starts_with("https://www.youtube.com/")
            || url.starts_with("https://youtu.be/");
        return host_path_ok || youtube_ok;
    }
    false
}

/// The media type of a reference, inferred from the data URI prefix or the
/// file extension.
///
/// The inference covers the media the providers accept (images, documents,
/// audio, video); an extension nobody knows falls back to the given
/// default, which the caller picks per kind of content.
pub fn guess_media_type(url: &str, fallback: &str) -> String {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some(meta) = rest.split(',').next() {
            let meta = meta.trim_end_matches(";base64");
            if !meta.is_empty() {
                return meta.to_string();
            }
        }
    }
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "pdf" => "application/pdf",
        "txt" | "text" | "md" => "text/plain",
        "csv" => "text/csv",
        "json" => "application/json",
        "html" | "htm" => "text/html",
        "mp3" => "audio/mp3",
        "wav" => "audio/wav",
        "m4a" => "audio/mp4",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        _ => fallback,
    }
    .to_string()
}

fn truncate(s: &str) -> String {
    if s.len() <= 64 {
        s.to_string()
    } else {
        // llm-suite stays free of core dependencies; keep the same floor
        // semantics as cce_utils::truncate_to_byte_len locally.
        let mut end = 64.min(s.len());
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Deep merge two JSON values:
/// - Objects are recursively merged (source wins on key conflicts)
/// - Arrays are replaced by the source
/// - Scalars are replaced by source
///
/// Arrays are replaced rather than concatenated: the wire body carries
/// list-valued fields (`messages`, `tools`, `stop`) whose content is built by
/// the codec, and concatenating a custom-body array with the built one would
/// produce a malformed request.
/// Billing total reported by the provider itself, when present.
///
/// The official usage shapes carry no such field; OpenAI-compatible
/// gateways (OpenRouter and aggregators) attach it as a numeric `cost`,
/// `total_cost` or `provider_cost` extension next to the counters. A
/// non-numeric or negative value is ignored so a malformed extension can
/// never poison cost accounting. String-encoded numbers are accepted
/// because some gateways serialize the decimal as a string.
fn parse_cost_value(value: &serde_json::Value) -> Option<f64> {
    let parsed = if let Some(number) = value.as_f64() {
        Some(number)
    } else if let Some(text) = value.as_str() {
        text.parse::<f64>().ok()
    } else {
        None
    };
    parsed.filter(|total| total.is_finite() && *total >= 0.0)
}

/// Billing total reported by the provider itself, when present.
///
/// The official usage shapes carry no such field; OpenAI-compatible
/// gateways (OpenRouter and aggregators) attach it as a numeric `cost`,
/// `total_cost` or `provider_cost` extension next to the counters. A
/// non-numeric or negative value is ignored so a malformed extension can
/// never poison cost accounting. String-encoded numbers are accepted
/// because some gateways serialize the decimal as a string.
///
/// OpenRouter additionally reports BYOK upstream costs under
/// `cost_details`; in that case the billed total is `cost` plus the
/// upstream inference cost, mirroring pi's `applyOpenRouterReportedCost`.
pub fn parse_provider_total(usage: &serde_json::Value) -> Option<f64> {
    // OpenRouter BYOK: the charged total is the gateway cost plus the
    // upstream provider cost. Handle this before the generic single-field
    // fallback so a BYOK response is not under-counted.
    if let Some(cost) = usage.get("cost").and_then(parse_cost_value) {
        let is_byok = usage.get("is_byok").and_then(|value| value.as_bool());
        if is_byok == Some(true) {
            if let Some(details) = usage.get("cost_details") {
                let upstream = details
                    .get("upstream_inference_cost")
                    .and_then(parse_cost_value)
                    .or_else(|| {
                        let prompt = details
                            .get("upstream_inference_prompt_cost")
                            .and_then(parse_cost_value)?;
                        let completion = details
                            .get("upstream_inference_completions_cost")
                            .and_then(parse_cost_value)?;
                        Some(prompt + completion)
                    });
                if let Some(upstream) = upstream {
                    return Some(cost + upstream);
                }
            }
        }
        if is_byok != Some(true) {
            return Some(cost);
        }
    }

    for key in [
        "cost",
        "total_cost",
        "provider_cost",
        "billed_cost",
        "total_price",
    ] {
        if let Some(value) = usage.get(key) {
            if let Some(total) = parse_cost_value(value) {
                return Some(total);
            }
        }
    }
    None
}

pub fn deep_merge(target: &serde_json::Value, source: &serde_json::Value) -> serde_json::Value {
    match (target, source) {
        (serde_json::Value::Object(t), serde_json::Value::Object(s)) => {
            let mut result = t.clone();
            for (k, v) in s {
                if let Some(existing) = result.get(k) {
                    result.insert(k.clone(), deep_merge(existing, v));
                } else {
                    result.insert(k.clone(), v.clone());
                }
            }
            serde_json::Value::Object(result)
        }
        _ => source.clone(),
    }
}

pub fn merge_parameters(
    profile: &LlmProfile,
    request_params: &Option<serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut merged = HashMap::new();

    if let Some(ref profile_params) = profile.parameters {
        if let Some(obj) = profile_params.as_object() {
            for (key, value) in obj {
                merged.insert(key.clone(), value.clone());
            }
        }
    }

    if let Some(ref params) = request_params {
        if let Some(obj) = params.as_object() {
            for (key, value) in obj {
                merged.insert(key.clone(), value.clone());
            }
        }
    }

    merged
}

pub fn build_auth_header(
    api_key: &Option<String>,
    native_header: &str,
) -> Option<(String, String)> {
    api_key
        .as_ref()
        .map(|key| (native_header.to_string(), key.clone()))
}

pub fn build_bearer_header(api_key: &Option<String>) -> Option<(String, String)> {
    api_key
        .as_ref()
        .map(|key| ("Authorization".to_string(), format!("Bearer {}", key)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_suite_types::llm::{LlmFormat, LlmProfile};

    fn profile_with_params(params: Option<serde_json::Value>) -> LlmProfile {
        LlmProfile {
            id: "p1".to_string(),
            name: "test".to_string(),
            format: LlmFormat::OpenaiChat,
            provider_id: None,
            model: "gpt-4o".to_string(),
            api_key: None,
            base_url: None,
            parameters: params,
            generation: None,
            timeout: None,
            max_retries: None,
            retry_delay: None,
            headers: None,
            metadata: None,
            tool_call_protocol: None,
            auth_type: None,
            custom_headers: None,
            custom_body: None,
            custom_body_enabled: None,
            query_params: None,
            stream_options: None,
            prompt_caching: None,
            context_window_size: None,
            max_output_tokens: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    #[test]
    fn deep_merge_merges_objects_recursively() {
        let target = serde_json::json!({
            "a": 1,
            "nested": {"x": 1, "keep": "me"},
            "list": [1, 2],
        });
        let source = serde_json::json!({
            "b": 2,
            "nested": {"x": 99},
            "list": [3],
        });
        let merged = deep_merge(&target, &source);
        assert_eq!(merged["a"], serde_json::json!(1));
        assert_eq!(merged["b"], serde_json::json!(2));
        assert_eq!(merged["nested"]["x"], serde_json::json!(99), "source wins");
        assert_eq!(merged["nested"]["keep"], serde_json::json!("me"));
        assert_eq!(
            merged["list"],
            serde_json::json!([3]),
            "arrays are replaced, never concatenated"
        );
    }

    #[test]
    fn deep_merge_scalars_are_replaced_by_source() {
        assert_eq!(
            deep_merge(&serde_json::json!(1), &serde_json::json!(2)),
            serde_json::json!(2)
        );
        assert_eq!(
            deep_merge(&serde_json::json!({"a": 1}), &serde_json::json!("flat")),
            serde_json::json!("flat")
        );
        assert_eq!(
            deep_merge(&serde_json::json!([1]), &serde_json::json!({"o": 1})),
            serde_json::json!({"o": 1})
        );
    }

    #[test]
    fn merge_parameters_request_wins_on_conflicts() {
        let profile = profile_with_params(Some(serde_json::json!({
            "temperature": 0.7,
            "max_tokens": 100,
        })));
        let merged = merge_parameters(
            &profile,
            &Some(serde_json::json!({
                "temperature": 0.2,
                "extra": true,
            })),
        );
        assert_eq!(merged["temperature"], serde_json::json!(0.2));
        assert_eq!(merged["max_tokens"], serde_json::json!(100));
        assert_eq!(merged["extra"], serde_json::json!(true));
    }

    #[test]
    fn merge_parameters_handles_missing_and_non_object() {
        let profile = profile_with_params(None);
        assert!(merge_parameters(&profile, &None).is_empty());
        assert!(merge_parameters(&profile, &Some(serde_json::json!("nope"))).is_empty());

        let profile = profile_with_params(Some(serde_json::json!({"a": 1})));
        let merged = merge_parameters(&profile, &Some(serde_json::json!("nope")));
        assert_eq!(merged["a"], serde_json::json!(1));
    }

    #[test]
    fn auth_headers_are_built_correctly() {
        assert_eq!(
            build_auth_header(&Some("sk-1".to_string()), "x-api-key"),
            Some(("x-api-key".to_string(), "sk-1".to_string()))
        );
        assert_eq!(build_auth_header(&None, "x-api-key"), None);
        assert_eq!(
            build_bearer_header(&Some("sk-1".to_string())),
            Some(("Authorization".to_string(), "Bearer sk-1".to_string()))
        );
        assert_eq!(build_bearer_header(&None), None);
    }

    #[test]
    fn resolve_data_uri_parses_media_type_and_payload() {
        let resolved = resolve_image_payload("data:image/png;base64,AAAA").expect("must resolve");
        assert_eq!(
            resolved,
            ResolvedImage::Base64 {
                media_type: "image/png".to_string(),
                data: "AAAA".to_string(),
            }
        );
    }

    #[test]
    fn resolve_rejects_malformed_and_non_base64_data_uris() {
        assert!(resolve_image_payload("data:image/png;base64").is_err());
        assert!(resolve_image_payload("data:image/png,AAAA").is_err());
        assert!(resolve_image_payload("data:;base64,").is_err());
    }

    #[test]
    fn resolve_passes_remote_url_through() {
        let resolved = resolve_image_payload("https://example.com/a.png").expect("must resolve");
        assert_eq!(
            resolved,
            ResolvedImage::Url("https://example.com/a.png".to_string())
        );
        assert!(resolve_image_payload("ftp://example.com/a.png").is_err());
    }

    #[test]
    fn a_native_block_of_the_serving_family_is_passed_through() {
        let payload = serde_json::json!({"type": "search_result", "title": "Example"});
        let out = native_block("anthropic", "anthropic", "search_result", &payload)
            .expect("the block belongs to the family");
        assert_eq!(out, payload, "the payload travels verbatim");
    }

    #[test]
    fn a_native_block_of_another_family_is_rejected() {
        let payload = serde_json::json!({"type": "search_result"});
        let err = native_block("anthropic", "openai", "search_result", &payload)
            .expect_err("a foreign block cannot be carried");
        assert!(format!("{err}").contains("anthropic"), "{err}");
    }

    #[test]
    fn a_native_block_must_carry_the_field_that_identifies_it() {
        let wrong_type = serde_json::json!({"type": "tool_use"});
        let err = native_block("anthropic", "anthropic", "search_result", &wrong_type)
            .expect_err("the type must match the kind");
        assert!(format!("{err}").contains("tool_use"), "{err}");

        let no_type = serde_json::json!({"title": "Example"});
        assert!(native_block("anthropic", "anthropic", "search_result", &no_type).is_err());

        // A Gemini part is identified by the field it carries.
        let part = serde_json::json!({"videoMetadata": {"endOffset": "10s"}});
        assert!(native_block("gemini", "gemini", "video_metadata", &part).is_ok());
        assert!(native_block("gemini", "gemini", "media_resolution", &part).is_err());
    }

    #[test]
    fn guess_media_type_prefers_data_uri_then_extension() {
        assert_eq!(
            guess_media_type("data:image/webp;base64,AA", "image/png"),
            "image/webp"
        );
        assert_eq!(
            guess_media_type("https://e.com/x.jpg?a=1", "image/png"),
            "image/jpeg"
        );
        assert_eq!(
            guess_media_type("https://e.com/unknown", "image/png"),
            "image/png"
        );
        assert_eq!(
            guess_media_type("gs://bucket/report.pdf", "image/png"),
            "application/pdf",
            "a document must not be labelled as an image"
        );
        assert_eq!(
            guess_media_type("gs://bucket/clip.mp4", "image/png"),
            "video/mp4"
        );
    }

    #[test]
    fn provider_total_reads_gateway_cost_extensions() {
        assert_eq!(
            parse_provider_total(&serde_json::json!({"cost": 0.5})),
            Some(0.5)
        );
        assert_eq!(
            parse_provider_total(&serde_json::json!({"total_cost": 1.25})),
            Some(1.25)
        );
        assert_eq!(
            parse_provider_total(&serde_json::json!({"provider_cost": "0.75"})),
            Some(0.75),
            "string-encoded decimals are accepted"
        );
        assert_eq!(parse_provider_total(&serde_json::json!({})), None);
        assert_eq!(
            parse_provider_total(&serde_json::json!({"cost": -1.0})),
            None,
            "a negative total must never poison accounting"
        );
        assert_eq!(
            parse_provider_total(&serde_json::json!({"cost": "n/a"})),
            None
        );
    }
}
