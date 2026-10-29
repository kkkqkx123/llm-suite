//! Prompt-caching breakpoints on the Anthropic request body.

use llm_suite_types::llm::{LlmPromptCaching, ProviderCompat};

/// Apply prompt-caching breakpoints (official `CacheControlEphemeralParam`
/// shape: `{"type": "ephemeral", "ttl": "5m" | "1h"}`) on the last tool
/// declaration, the system block, the first message and the last message
/// — the four breakpoints the API allows — honoring the endpoint compat.
///
/// Endpoints that reject the long retention (`supports_long_cache_retention
/// == Some(false)`) get the provider-default retention instead of `1h`:
/// the `ttl` marker is omitted, which the API reads as its default. Any
/// other ttl value, and any other compat, passes through untouched.
pub fn apply_prompt_caching_with_compat(
    body: &mut serde_json::Value,
    caching: &LlmPromptCaching,
    compat: Option<&ProviderCompat>,
) {
    let long_retention_rejected =
        compat.and_then(|c| c.supports_long_cache_retention) == Some(false);
    let effective_ttl = match (&caching.ttl, long_retention_rejected) {
        (Some(ttl), true) if ttl == "1h" => {
            tracing::debug!("endpoint rejects long cache retention; using default retention");
            None
        }
        (ttl, _) => ttl.clone(),
    };
    let effective = LlmPromptCaching { ttl: effective_ttl };
    apply_with_ttl(body, &effective);
}

fn apply_with_ttl(body: &mut serde_json::Value, caching: &LlmPromptCaching) {
    let mut control = serde_json::json!({"type": "ephemeral"});
    if let Some(ttl) = &caching.ttl {
        control["ttl"] = serde_json::json!(ttl);
    }

    // Last tool declaration.
    if let Some(tools) = body.get_mut("tools").and_then(|v| v.as_array_mut()) {
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = control.clone();
        }
    }

    // System block: sent as a plain string elsewhere; the block-array form
    // is required to carry `cache_control`.
    if let Some(system) = body.get("system").and_then(|v| v.as_str()) {
        body["system"] = serde_json::json!([{
            "type": "text",
            "text": system,
            "cache_control": control.clone(),
        }]);
    }

    // First and last message: the stable prefix and the current turn.
    let message_count = body
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|messages| messages.len())
        .unwrap_or(0);
    for index in [0, message_count.saturating_sub(1)] {
        if message_count == 0 {
            break;
        }
        let Some(blocks) = body
            .get_mut("messages")
            .and_then(|v| v.as_array_mut())
            .and_then(|m| m.get_mut(index))
            .and_then(|m| m.get_mut("content"))
            .and_then(|v| v.as_array_mut())
        else {
            continue;
        };
        // The thinking blocks carry no `cache_control`; the breakpoint goes
        // on the last block that accepts one.
        if let Some(block) = blocks
            .iter_mut()
            .rev()
            .find(|block| !is_thinking_block(block))
        {
            block["cache_control"] = control.clone();
        }
    }
}

/// Whether a block is a thinking block; those carry no cache control.
fn is_thinking_block(block: &serde_json::Value) -> bool {
    matches!(
        block.get("type").and_then(|v| v.as_str()),
        Some("thinking") | Some("redacted_thinking")
    )
}
