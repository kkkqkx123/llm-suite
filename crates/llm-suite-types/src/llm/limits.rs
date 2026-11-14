use crate::message::{Message, MessageContent, MessageContentValue};

/// Reserved key carrying request limits inside profile or request
/// `parameters`. The key is consumed by gateway admission and never reaches
/// the wire: the parameter passthrough skips it like `stream`.
pub const REQUEST_LIMITS_KEY: &str = "request_limits";

/// Admission limits for one LLM request.
///
/// These guard the request before encoding, not the provider behavior after
/// it. Oversized images are rejected instead of silently downscaled; callers
/// resize before sending. Counts cover image blocks only, matching the Pi
/// `inputLimits.images` semantics; byte size covers the serialized messages
/// plus tool declarations as a pre-codec approximation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlmRequestLimits {
    /// Maximum serialized size of messages plus tool declarations in bytes.
    pub max_request_bytes: Option<u64>,
    /// Maximum image blocks accepted in one message.
    pub max_images_per_message: Option<u32>,
    /// Maximum image blocks accepted across one request.
    pub max_images_per_request: Option<u32>,
}

/// Parse request limits from merged parameters. Unknown keys and mistyped
/// values fail so a typoed limit cannot silently run unbounded.
pub fn parse_request_limits(
    parameters: &Option<serde_json::Value>,
) -> Result<Option<LlmRequestLimits>, String> {
    let Some(limits_value) = parameters.as_ref().and_then(|v| v.get(REQUEST_LIMITS_KEY)) else {
        return Ok(None);
    };
    if limits_value.is_null() {
        return Ok(None);
    }
    let obj = limits_value
        .as_object()
        .ok_or_else(|| format!("'{REQUEST_LIMITS_KEY}' must be an object with limit fields"))?;
    let mut limits = LlmRequestLimits::default();
    for (key, value) in obj {
        match key.as_str() {
            "max_request_bytes" => {
                limits.max_request_bytes = Some(expect_u64(key, value)?);
            }
            "max_images_per_message" => {
                limits.max_images_per_message = Some(expect_u32(key, value)?);
            }
            "max_images_per_request" => {
                limits.max_images_per_request = Some(expect_u32(key, value)?);
            }
            other => {
                return Err(format!(
                    "'{REQUEST_LIMITS_KEY}' has no '{other}' field; supported fields are max_request_bytes, max_images_per_message and max_images_per_request"
                ));
            }
        }
    }
    Ok(Some(limits))
}

fn expect_u64(key: &str, value: &serde_json::Value) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("'{REQUEST_LIMITS_KEY}.{key}' must be a non-negative integer"))
}

fn expect_u32(key: &str, value: &serde_json::Value) -> Result<u32, String> {
    let raw = expect_u64(key, value)?;
    u32::try_from(raw).map_err(|_| format!("'{REQUEST_LIMITS_KEY}.{key}' exceeds the u32 range"))
}

/// Enforce parsed limits against the effective request. Returns the first
/// violation; callers map it onto their own error surface.
pub fn check_request_limits(
    limits: &LlmRequestLimits,
    messages: &[Message],
    tools: &Option<Vec<crate::tool::Tool>>,
) -> Result<(), String> {
    if let Some(max_bytes) = limits.max_request_bytes {
        let size = request_body_size(messages, tools)
            .map_err(|e| format!("failed to size request for '{REQUEST_LIMITS_KEY}': {e}"))?;
        if size > max_bytes {
            return Err(format!(
                "request size {size} bytes exceeds '{REQUEST_LIMITS_KEY}.max_request_bytes' ({max_bytes})"
            ));
        }
    }
    if limits.max_images_per_message.is_some() || limits.max_images_per_request.is_some() {
        let mut total: u32 = 0;
        for (index, message) in messages.iter().enumerate() {
            let count = message_image_count(message);
            if let Some(max_per_message) = limits.max_images_per_message {
                if count > max_per_message {
                    return Err(format!(
                        "message {index} carries {count} images, exceeding '{REQUEST_LIMITS_KEY}.max_images_per_message' ({max_per_message})"
                    ));
                }
            }
            total = total.saturating_add(count);
        }
        if let Some(max_per_request) = limits.max_images_per_request {
            if total > max_per_request {
                return Err(format!(
                    "request carries {total} images, exceeding '{REQUEST_LIMITS_KEY}.max_images_per_request' ({max_per_request})"
                ));
            }
        }
    }
    Ok(())
}

fn message_image_count(message: &Message) -> u32 {
    match &message.content {
        MessageContentValue::Text(_) => 0,
        MessageContentValue::Rich(blocks) => blocks
            .iter()
            .filter(|block| matches!(block, MessageContent::ImageUrl { .. }))
            .count() as u32,
    }
}

fn request_body_size(
    messages: &[Message],
    tools: &Option<Vec<crate::tool::Tool>>,
) -> Result<u64, serde_json::Error> {
    let body = serde_json::json!({
        "messages": messages,
        "tools": tools,
    });
    serde_json::to_string(&body).map(|rendered| rendered.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(value: serde_json::Value) -> Option<serde_json::Value> {
        Some(value)
    }

    #[test]
    fn absent_key_parses_to_no_limits() {
        assert_eq!(parse_request_limits(&None).unwrap(), None);
        assert_eq!(
            parse_request_limits(&params(serde_json::json!({"temperature": 0.5}))).unwrap(),
            None
        );
        assert_eq!(
            parse_request_limits(&params(serde_json::json!({"request_limits": null}))).unwrap(),
            None
        );
    }

    #[test]
    fn known_fields_parse() {
        let parsed = parse_request_limits(&params(serde_json::json!({
            "request_limits": {
                "max_request_bytes": 1024,
                "max_images_per_message": 2,
                "max_images_per_request": 4,
            }
        })))
        .unwrap()
        .expect("limits must parse");
        assert_eq!(parsed.max_request_bytes, Some(1024));
        assert_eq!(parsed.max_images_per_message, Some(2));
        assert_eq!(parsed.max_images_per_request, Some(4));
    }

    #[test]
    fn unknown_fields_and_mistyped_values_fail() {
        assert!(parse_request_limits(&params(serde_json::json!({
            "request_limits": {"max_images": 2}
        })))
        .is_err());
        assert!(parse_request_limits(&params(serde_json::json!({
            "request_limits": {"max_images_per_message": "two"}
        })))
        .is_err());
        assert!(parse_request_limits(&params(serde_json::json!({
            "request_limits": [1]
        })))
        .is_err());
    }
}
