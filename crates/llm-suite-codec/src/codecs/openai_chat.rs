
use super::shared;
use super::LlmCodec;
use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent,
};
use llm_suite_types::tool::Tool;
use reqwest::Method;

pub(crate) mod messages;
pub(crate) mod response;
pub(crate) mod stream;
pub(crate) mod tools;
pub(crate) mod usage;
pub(crate) mod validate;

#[cfg(test)]
pub(crate) mod test_support;

pub struct OpenaiChatCodec {
    base_url: String,
    // Reused across requests; constructing a `reqwest::Client` per call would
    // discard the connection pool and TLS session cache every time.
    client: reqwest::Client,
}

impl Default for OpenaiChatCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenaiChatCodec {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    /// Build the request body (separated from the HTTP layer for testability).
    fn build_body(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<serde_json::Value> {
        let use_text_mode = shared::is_text_mode(request);

        let messages = if use_text_mode {
            let (_, filtered) =
                llm_suite_tool_call::tool::protocol::extract_system_message(&request.messages);
            let content = shared::text_mode_system_content(request);
            let history = shared::convert_history_for_text_mode(&filtered, request);
            let mut converted = messages::convert_openai_messages(&history);
            if !content.is_empty() {
                converted.insert(0, serde_json::json!({"role": "system", "content": content}));
            }
            converted
        } else {
            messages::convert_openai_messages(&request.messages)
        };

        let mut body = serde_json::json!({
            "model": profile.model,
            "messages": messages,
        });

        let generation = shared::resolve_generation(request, profile)?;
        crate::generation::apply_openai_chat(&mut body, &generation)?;
        shared::merge_and_apply_params(&mut body, profile, &request.parameters);

        if !use_text_mode {
            // `tool` role messages must carry `tool_call_id` for OpenAI Chat.
            validate::validate_openai_chat_messages(request)?;
            if let Some(tools) = &request.tools {
                body["tools"] = serde_json::json!(tools::convert_openai_tools(tools)?);
            }
        }

        if request.stream == Some(true) {
            body["stream"] = serde_json::json!(true);
            // The usage block of a stream is opt-in: sending `stream_options`
            // to a server that does not know the field makes it reject the
            // whole request.
            if let Some(options) = &profile.stream_options {
                if let Some(include_usage) = options.include_usage {
                    body["stream_options"] = serde_json::json!({"include_usage": include_usage});
                }
            }
        }

        shared::apply_custom_body(&mut body, profile);

        Ok(body)
    }
}

impl LlmCodec for OpenaiChatCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        // The profile may carry a full endpoint URL (custom path or query
        // string such as Azure `.../chat/completions?api-version=...`); in
        // that case it is used as-is, otherwise the built-in path applies.
        let profile_url = profile.base_url.as_deref().unwrap_or(&self.base_url);
        let url = shared::endpoint_url(profile_url, "chat/completions");

        let body = self.build_body(request, profile)?;

        let mut req_builder = self
            .client
            .request(Method::POST, &url)
            .header("Content-Type", "application/json")
            .json(&body);

        req_builder = shared::apply_auth_and_headers(req_builder, profile, "bearer");

        req_builder
            .build()
            .map_err(crate::error::LlmError::HttpError)
    }

    // Chat Completions has no token counting endpoint (unlike the Responses
    // API `/responses/input_tokens`): keep the default `None` so the caller
    // falls back to local estimation. Covers all OpenAI-compatible third
    // parties routed through this codec as well.
    fn build_count_tokens_request(
        &self,
        _request: &LlmRequest,
        _profile: &LlmProfile,
    ) -> LlmResult<Option<reqwest::Request>> {
        Ok(None)
    }

    fn parse_response(&self, body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
        let mut result = response::parse_openai_chat_response(body)?;
        if shared::is_text_mode(request) {
            if let Some(content) = result.content.clone() {
                let calls = shared::parse_text_tool_calls(request, &content);
                if !calls.is_empty() {
                    result.tool_calls = Some(calls.clone());
                    result.message.tool_calls = Some(calls);
                }
            }
        }
        Ok(result)
    }

    fn parse_stream_chunk_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        stream::parse_openai_stream_chunk(data)
    }

    fn convert_tools(&self, tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
        tools::convert_openai_tools(tools)
    }

    fn parse_tool_calls(
        &self,
        result: &LlmResponseType,
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
        result.tool_calls.clone().unwrap_or_default()
    }
}
