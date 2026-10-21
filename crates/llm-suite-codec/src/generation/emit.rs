use super::validate::ANTHROPIC_DEFAULT_MAX_TOKENS;
use crate::error::{LlmError, LlmResult};
use llm_suite_types::llm::generation::{
    LlmGenerationParams, LlmServiceTier, LlmThinkingConfig, LlmToolChoice, ResponseFormatKind,
    ThinkingLevel, ToolChoiceMode,
};

/// Chat Completions names a function through a nested `function` object.
fn emit_tool_choice_openai_chat(choice: &LlmToolChoice) -> serde_json::Value {
    if let Some(ref name) = choice.tool_name {
        return serde_json::json!({
            "type": "function",
            "function": {"name": name}
        });
    }
    match choice.mode {
        Some(ToolChoiceMode::Auto) => serde_json::json!("auto"),
        Some(ToolChoiceMode::Any) | Some(ToolChoiceMode::Required) => serde_json::json!("required"),
        Some(ToolChoiceMode::None) => serde_json::json!("none"),
        None => serde_json::json!("auto"),
    }
}

/// Responses names a function through a flat `name` field next to `type`.
fn emit_tool_choice_openai_responses(choice: &LlmToolChoice) -> serde_json::Value {
    if let Some(ref name) = choice.tool_name {
        return serde_json::json!({
            "type": "function",
            "name": name
        });
    }
    match choice.mode {
        Some(ToolChoiceMode::Auto) => serde_json::json!("auto"),
        Some(ToolChoiceMode::Any) | Some(ToolChoiceMode::Required) => serde_json::json!("required"),
        Some(ToolChoiceMode::None) => serde_json::json!("none"),
        None => serde_json::json!("auto"),
    }
}

/// Anthropic accepts `auto`, `any`, `tool` (named) and `none`. The `none`
/// variant carries no parallel-tool flag.
fn emit_tool_choice_anthropic(
    choice: &LlmToolChoice,
    parallel_tool_calls: Option<bool>,
) -> serde_json::Value {
    let disable_parallel = parallel_tool_calls.is_some_and(|v| !v);
    if let Some(ref name) = choice.tool_name {
        let mut obj = serde_json::json!({"type": "tool", "name": name});
        if disable_parallel {
            obj["disable_parallel_tool_use"] = serde_json::json!(true);
        }
        return obj;
    }
    let kind = match choice.mode {
        Some(ToolChoiceMode::Any) | Some(ToolChoiceMode::Required) => "any",
        Some(ToolChoiceMode::None) => "none",
        _ => "auto",
    };
    let mut obj = serde_json::json!({"type": kind});
    if kind != "none" && disable_parallel {
        obj["disable_parallel_tool_use"] = serde_json::json!(true);
    }
    obj
}

/// Service tiers the OpenAI APIs accept. A tier outside the set is a
/// configuration mistake and fails fast instead of being sent to the API.
fn openai_service_tier(tier: LlmServiceTier) -> LlmResult<&'static str> {
    match tier {
        LlmServiceTier::Auto => Ok("auto"),
        LlmServiceTier::Default => Ok("default"),
        LlmServiceTier::Flex => Ok("flex"),
        LlmServiceTier::Scale => Ok("scale"),
        LlmServiceTier::Priority => Ok("priority"),
        LlmServiceTier::Fast => Ok("fast"),
        LlmServiceTier::Ultrafast => Ok("ultrafast"),
        LlmServiceTier::StandardOnly => Err(LlmError::ConfigError(
            "service tier `standard_only` is an Anthropic tier; OpenAI accepts auto, default, flex, scale, priority and fast"
                .to_string(),
        )),
    }
}

/// Service tiers the Anthropic Messages API accepts.
fn anthropic_service_tier(tier: LlmServiceTier) -> LlmResult<&'static str> {
    match tier {
        LlmServiceTier::Auto => Ok("auto"),
        LlmServiceTier::StandardOnly => Ok("standard_only"),
        other => Err(LlmError::ConfigError(format!(
            "service tier `{}` is not supported by Anthropic; use auto or standard_only",
            other.as_str()
        ))),
    }
}

/// Emit the OpenAI Chat generation parameters.
///
/// The `max_tokens` value is emitted as `max_tokens` or
/// `max_completion_tokens` according to the explicit `max_token_field`
/// configuration (defaulting to `max_tokens`). The model name is never
/// inspected: which field a backend accepts is a deployment decision that
/// belongs to profile configuration.
pub fn apply_openai_chat(body: &mut serde_json::Value, gen: &LlmGenerationParams) -> LlmResult<()> {
    if let Some(max_tokens) = gen.max_tokens {
        let key = gen.max_token_field.unwrap_or_default().as_str();
        body[key] = serde_json::json!(max_tokens);
    }
    if let Some(temperature) = gen.temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = gen.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if let Some(ref stop) = gen.stop {
        body["stop"] = serde_json::json!(stop);
    }
    if let Some(seed) = gen.seed {
        body["seed"] = serde_json::json!(seed);
    }
    if let Some(penalty) = gen.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(penalty);
    }
    if let Some(penalty) = gen.presence_penalty {
        body["presence_penalty"] = serde_json::json!(penalty);
    }
    if let Some(ref tier) = gen.service_tier {
        body["service_tier"] = serde_json::json!(openai_service_tier(*tier)?);
    }
    if let Some(store) = gen.store {
        body["store"] = serde_json::json!(store);
    }
    if let Some(ref user_id) = gen.user_id {
        body["user"] = serde_json::json!(user_id);
    }
    if let Some(ref thinking) = gen.thinking {
        if let Some(level) = thinking.level {
            body["reasoning_effort"] = serde_json::json!(level.as_str());
        }
    }
    if let Some(verbosity) = gen.verbosity {
        body["verbosity"] = serde_json::json!(verbosity.as_str());
    }
    if let Some(ref choice) = gen.tool_choice {
        body["tool_choice"] = emit_tool_choice_openai_chat(choice);
    }
    if let Some(parallel) = gen.parallel_tool_calls {
        body["parallel_tool_calls"] = serde_json::json!(parallel);
    }
    if let Some(ref format) = gen.response_format {
        match format.kind {
            Some(ResponseFormatKind::Text) => {
                body["response_format"] = serde_json::json!({"type": "text"});
            }
            Some(ResponseFormatKind::JsonObject) => {
                body["response_format"] = serde_json::json!({"type": "json_object"});
            }
            Some(ResponseFormatKind::JsonSchema) | None => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".to_string(), serde_json::json!("json_schema"));
                if let Some(ref schema) = format.schema {
                    obj.insert(
                        "json_schema".to_string(),
                        serde_json::json!({
                            "name": format.name.clone().unwrap_or_else(|| "response".to_string()),
                            "schema": schema,
                            "strict": format.strict.unwrap_or(false),
                        }),
                    );
                } else if let Some(ref name) = format.name {
                    obj.insert("name".to_string(), serde_json::json!(name));
                }
                body["response_format"] = serde_json::Value::Object(obj);
            }
        }
    }
    if gen.top_k.is_some() {
        tracing::warn!("top_k is not supported by the OpenAI Chat API and was ignored");
    }
    Ok(())
}

pub fn apply_openai_responses(
    body: &mut serde_json::Value,
    gen: &LlmGenerationParams,
) -> LlmResult<()> {
    if let Some(max_tokens) = gen.max_tokens {
        body["max_output_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(temperature) = gen.temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = gen.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if let Some(seed) = gen.seed {
        body["seed"] = serde_json::json!(seed);
    }
    if let Some(ref tier) = gen.service_tier {
        body["service_tier"] = serde_json::json!(openai_service_tier(*tier)?);
    }
    if let Some(store) = gen.store {
        body["store"] = serde_json::json!(store);
    }
    if let Some(ref user_id) = gen.user_id {
        body["safety_identifier"] = serde_json::json!(user_id);
    }
    if let Some(ref thinking) = gen.thinking {
        if thinking.level.is_some() || thinking.summary.is_some() {
            let mut reasoning = serde_json::Map::new();
            if let Some(level) = thinking.level {
                reasoning.insert("effort".to_string(), serde_json::json!(level.as_str()));
            }
            if let Some(summary) = thinking.summary {
                reasoning.insert("summary".to_string(), serde_json::json!(summary.as_str()));
            }
            body["reasoning"] = serde_json::Value::Object(reasoning);
        }
    }
    let mut text = body
        .get("text")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let mut has_text = false;
    if let Some(verbosity) = gen.verbosity {
        text.insert(
            "verbosity".to_string(),
            serde_json::json!(verbosity.as_str()),
        );
        has_text = true;
    }
    if let Some(ref format) = gen.response_format {
        match format.kind {
            Some(ResponseFormatKind::Text) => {
                text.insert("format".to_string(), serde_json::json!({"type": "text"}));
                has_text = true;
            }
            Some(ResponseFormatKind::JsonObject) => {
                text.insert(
                    "format".to_string(),
                    serde_json::json!({"type": "json_object"}),
                );
                has_text = true;
            }
            Some(ResponseFormatKind::JsonSchema) | None => {
                if let Some(ref schema) = format.schema {
                    text.insert(
                        "format".to_string(),
                        serde_json::json!({
                            "type": "json_schema",
                            "name": format.name.clone().unwrap_or_else(|| "response".to_string()),
                            "schema": schema,
                            "strict": format.strict.unwrap_or(false),
                        }),
                    );
                    has_text = true;
                }
            }
        }
    }
    if has_text {
        body["text"] = serde_json::Value::Object(text);
    }
    if let Some(ref choice) = gen.tool_choice {
        body["tool_choice"] = emit_tool_choice_openai_responses(choice);
    }
    if let Some(parallel) = gen.parallel_tool_calls {
        body["parallel_tool_calls"] = serde_json::json!(parallel);
    }
    if gen.stop.is_some() {
        tracing::warn!(
            "stop sequences are not supported by the OpenAI Responses API mapping and were ignored"
        );
    }
    if gen.top_k.is_some() {
        tracing::warn!("top_k is not supported by the OpenAI Responses API and was ignored");
    }
    if gen.frequency_penalty.is_some() || gen.presence_penalty.is_some() {
        tracing::warn!("frequency/presence penalties are not supported by the OpenAI Responses API mapping and were ignored");
    }
    Ok(())
}

pub fn apply_anthropic(body: &mut serde_json::Value, gen: &LlmGenerationParams) -> LlmResult<()> {
    let max_tokens = gen.max_tokens.unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS);
    body["max_tokens"] = serde_json::json!(max_tokens);
    if let Some(ref stop) = gen.stop {
        body["stop_sequences"] = serde_json::json!(stop);
    }
    if let Some(ref tier) = gen.service_tier {
        body["service_tier"] = serde_json::json!(anthropic_service_tier(*tier)?);
    }
    if let Some(ref user_id) = gen.user_id {
        body["metadata"] = serde_json::json!({"user_id": user_id});
    }
    if let Some(ref thinking) = gen.thinking {
        apply_anthropic_thinking(body, thinking, max_tokens)?;
    }
    let mut output_config = body
        .get("output_config")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let mut has_output_config = !output_config.is_empty();
    if let Some(ref thinking) = gen.thinking {
        if thinking.adaptive.unwrap_or(false)
            || (thinking.level.is_some() && thinking.budget_tokens.is_none())
        {
            if let Some(level) = thinking.level {
                let effort = match level {
                    ThinkingLevel::None => None,
                    ThinkingLevel::Minimal => {
                        tracing::warn!("Anthropic effort has no minimal level; mapped to low");
                        Some("low")
                    }
                    ThinkingLevel::Low => Some("low"),
                    ThinkingLevel::Medium => Some("medium"),
                    ThinkingLevel::High => Some("high"),
                    ThinkingLevel::Xhigh => Some("xhigh"),
                    ThinkingLevel::Max => Some("max"),
                };
                if let Some(effort) = effort {
                    output_config.insert("effort".to_string(), serde_json::json!(effort));
                    has_output_config = true;
                }
            }
        }
    }
    if let Some(ref format) = gen.response_format {
        if let Some(ref schema) = format.schema {
            output_config.insert(
                "format".to_string(),
                serde_json::json!({"type": "json_schema", "schema": schema}),
            );
            has_output_config = true;
        } else if format.kind == Some(ResponseFormatKind::JsonObject) {
            output_config.insert(
                "format".to_string(),
                serde_json::json!({"type": "json_schema", "schema": {"type": "object"}}),
            );
            has_output_config = true;
        }
    }
    if has_output_config {
        body["output_config"] = serde_json::Value::Object(output_config);
    }
    if let Some(ref choice) = gen.tool_choice {
        body["tool_choice"] = emit_tool_choice_anthropic(choice, gen.parallel_tool_calls);
    } else if gen.parallel_tool_calls.is_some_and(|v| !v) {
        body["tool_choice"] =
            serde_json::json!({"type": "auto", "disable_parallel_tool_use": true});
    }
    if gen.verbosity.is_some() {
        tracing::warn!("verbosity is not supported by Anthropic and was ignored");
    }
    if gen.seed.is_some() {
        tracing::warn!("seed is not supported by Anthropic and was ignored");
    }
    if gen.temperature.is_some() || gen.top_p.is_some() || gen.top_k.is_some() {
        tracing::warn!(
            "sampling parameters are not supported by the Anthropic Messages API and were ignored"
        );
    }
    if gen.frequency_penalty.is_some() || gen.presence_penalty.is_some() {
        tracing::warn!(
            "frequency/presence penalties are not supported by Anthropic and were ignored"
        );
    }
    if gen.store.is_some() {
        tracing::warn!("store is not supported by Anthropic and was ignored");
    }
    Ok(())
}

fn apply_anthropic_thinking(
    body: &mut serde_json::Value,
    thinking: &LlmThinkingConfig,
    max_tokens: u32,
) -> LlmResult<()> {
    // Thinking is switched off explicitly: the official API has a `disabled`
    // variant, and omitting the field leaves the model default in place.
    if thinking.level == Some(ThinkingLevel::None)
        && thinking.budget_tokens.is_none()
        && !thinking.adaptive.unwrap_or(false)
    {
        body["thinking"] = serde_json::json!({"type": "disabled"});
        return Ok(());
    }
    if let Some(budget) = thinking.budget_tokens {
        if budget < 1024 {
            return Err(LlmError::ConfigError(
                "Anthropic thinking budget_tokens must be at least 1024".to_string(),
            ));
        }
        if budget >= max_tokens {
            return Err(LlmError::ConfigError(format!(
                "Anthropic thinking budget_tokens ({budget}) must be less than max_tokens ({max_tokens})"
            )));
        }
        let mut obj = serde_json::json!({"type": "enabled", "budget_tokens": budget});
        if let Some(display) = thinking.display {
            obj["display"] = serde_json::json!(display.as_str());
        }
        body["thinking"] = obj;
        return Ok(());
    }
    if thinking.adaptive.unwrap_or(false) || thinking.level.is_some() {
        let mut obj = serde_json::json!({"type": "adaptive"});
        if let Some(display) = thinking.display {
            obj["display"] = serde_json::json!(display.as_str());
        }
        body["thinking"] = obj;
        return Ok(());
    }
    if thinking.display.is_some() || thinking.include_thoughts.is_some() {
        tracing::warn!(
            "Anthropic thinking display/include options need a level or budget to take effect"
        );
    }
    Ok(())
}

/// Gemini's `Schema.type` is an uppercase enum; the internal JSON-Schema
/// model carries lowercase type names. Map the known type names and leave
/// every other value untouched.
fn gemini_schema_type(value: &str) -> Option<&'static str> {
    match value.to_ascii_lowercase().as_str() {
        "string" => Some("STRING"),
        "number" => Some("NUMBER"),
        "integer" => Some("INTEGER"),
        "boolean" => Some("BOOLEAN"),
        "array" => Some("ARRAY"),
        "object" => Some("OBJECT"),
        "null" => Some("NULL"),
        _ => None,
    }
}

/// Rewrite a JSON Schema value into Gemini's dialect: every `type` field
/// holding a known type name becomes the uppercase enum member. Nested
/// schemas (`items`, `properties`, `anyOf`, …) are rewritten recursively.
pub fn gemini_schema_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (key, item) in map {
                let rewritten = if key == "type" {
                    match item.as_str().and_then(gemini_schema_type) {
                        Some(mapped) => serde_json::json!(mapped),
                        None => gemini_schema_value(item),
                    }
                } else {
                    gemini_schema_value(item)
                };
                out.insert(key.clone(), rewritten);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(gemini_schema_value).collect())
        }
        other => other.clone(),
    }
}

/// Gemini's `thinkingLevel` enum. `None` has no member: thinking is switched
/// off through the thinking budget instead.
fn gemini_thinking_level(level: ThinkingLevel) -> LlmResult<&'static str> {
    match level {
        ThinkingLevel::Minimal => Ok("MINIMAL"),
        ThinkingLevel::Low => Ok("LOW"),
        ThinkingLevel::Medium => Ok("MEDIUM"),
        ThinkingLevel::High => Ok("HIGH"),
        ThinkingLevel::Xhigh | ThinkingLevel::Max => Err(LlmError::ConfigError(format!(
            "Gemini thinkingLevel has no `{}` member; supported levels are minimal, low, medium and high",
            level.as_str()
        ))),
        ThinkingLevel::None => Err(LlmError::ConfigError(
            "Gemini thinkingLevel has no `none` member; switch thinking off through the thinking budget"
                .to_string(),
        )),
    }
}

pub fn apply_gemini_generation_config(
    generation_config: &mut serde_json::Value,
    gen: &LlmGenerationParams,
) -> LlmResult<()> {
    if let Some(temperature) = gen.temperature {
        generation_config["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = gen.top_p {
        generation_config["topP"] = serde_json::json!(top_p);
    }
    if let Some(top_k) = gen.top_k {
        generation_config["topK"] = serde_json::json!(top_k);
    }
    if let Some(max_tokens) = gen.max_tokens {
        generation_config["maxOutputTokens"] = serde_json::json!(max_tokens);
    }
    if let Some(ref stop) = gen.stop {
        generation_config["stopSequences"] = serde_json::json!(stop);
    }
    if let Some(ref thinking) = gen.thinking {
        if thinking.level.is_some() && thinking.budget_tokens.is_some() {
            return Err(LlmError::ConfigError(
                "Gemini thinkingLevel and thinkingBudget are mutually exclusive; set only one"
                    .to_string(),
            ));
        }
        let mut thinking_config = serde_json::Map::new();
        if let Some(level) = thinking.level {
            if level == ThinkingLevel::None {
                // Gemini switches thinking off through the budget.
                thinking_config.insert("thinkingBudget".to_string(), serde_json::json!(0));
            } else {
                thinking_config.insert(
                    "thinkingLevel".to_string(),
                    serde_json::json!(gemini_thinking_level(level)?),
                );
            }
        }
        if let Some(budget) = thinking.budget_tokens {
            thinking_config.insert("thinkingBudget".to_string(), serde_json::json!(budget));
        }
        if let Some(include) = thinking.include_thoughts {
            thinking_config.insert("includeThoughts".to_string(), serde_json::json!(include));
        }
        if !thinking_config.is_empty() {
            generation_config["thinkingConfig"] = serde_json::Value::Object(thinking_config);
        }
        if thinking.display.is_some() || thinking.summary.is_some() {
            tracing::warn!(
                "thinking display/summary options are not supported by Gemini and were ignored"
            );
        }
    }
    if let Some(ref format) = gen.response_format {
        match format.kind {
            Some(ResponseFormatKind::JsonObject) => {
                generation_config["responseMimeType"] = serde_json::json!("application/json");
                if let Some(ref schema) = format.schema {
                    generation_config["responseSchema"] = gemini_schema_value(schema);
                }
            }
            Some(ResponseFormatKind::JsonSchema) => {
                generation_config["responseMimeType"] = serde_json::json!("application/json");
                if let Some(ref schema) = format.schema {
                    generation_config["responseSchema"] = gemini_schema_value(schema);
                }
            }
            _ => {}
        }
    }
    if let Some(ref choice) = gen.tool_choice {
        let mut function_calling_config = serde_json::Map::new();
        if let Some(ref name) = choice.tool_name {
            // A named function restricts the model to that function; the
            // mode alone would still allow any tool.
            function_calling_config.insert("mode".to_string(), serde_json::json!("ANY"));
            function_calling_config.insert(
                "allowedFunctionNames".to_string(),
                serde_json::json!([name]),
            );
        } else {
            let mode = match choice.mode {
                Some(ToolChoiceMode::Auto) => "AUTO",
                Some(ToolChoiceMode::Any) | Some(ToolChoiceMode::Required) => "ANY",
                Some(ToolChoiceMode::None) => "NONE",
                None => "AUTO",
            };
            function_calling_config.insert("mode".to_string(), serde_json::json!(mode));
        }
        generation_config["toolConfig"] = serde_json::json!({
            "functionCallingConfig": serde_json::Value::Object(function_calling_config),
        });
    }
    if gen.verbosity.is_some() {
        tracing::warn!(
            "verbosity is not supported by Gemini native generationConfig and was ignored"
        );
    }
    if gen.service_tier.is_some() || gen.store.is_some() || gen.seed.is_some() {
        tracing::warn!("service_tier/store/seed are not supported by Gemini native generationConfig and were ignored");
    }
    if gen.frequency_penalty.is_some() || gen.presence_penalty.is_some() {
        tracing::warn!("frequency/presence penalties are not mapped for Gemini native generationConfig and were ignored");
    }
    if gen.user_id.is_some() {
        tracing::warn!(
            "user_id is not supported by Gemini native generationConfig and was ignored"
        );
    }
    Ok(())
}
