//! Wire protocol codec for the OpenAI Responses API.

use super::LlmCodec;
use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent,
};
use llm_suite_types::tool::Tool;

pub(crate) mod messages;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod stream;
pub(crate) mod tools;
pub(crate) mod usage;

#[cfg(test)]
pub(crate) mod test_support;

pub struct OpenaiResponseCodec {
    base_url: String,
    /// Reused across requests; constructing a `reqwest::Client` per call would
    /// discard the connection pool and TLS session cache on every time.
    client: reqwest::Client,
}

impl Default for OpenaiResponseCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenaiResponseCodec {
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
}

impl LlmCodec for OpenaiResponseCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        let url = super::shared::endpoint_url(
            profile.base_url.as_deref().unwrap_or(&self.base_url),
            "responses",
        );
        super::shared::build_post_json_request(
            &self.client,
            &url,
            &request::build_body(request, profile)?,
            profile,
            "bearer",
            &[],
        )
    }

    fn build_count_tokens_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<Option<reqwest::Request>> {
        let url = super::shared::endpoint_url(
            profile.base_url.as_deref().unwrap_or(&self.base_url),
            "responses/input_tokens",
        );
        super::shared::build_post_json_request(
            &self.client,
            &url,
            &request::build_count_tokens_body(request, profile)?,
            profile,
            "bearer",
            &[],
        )
        .map(Some)
    }

    fn parse_response(&self, body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
        response::parse_response(body, request)
    }

    fn parse_stream_chunk_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        stream::parse_stream_events(data)
    }

    fn convert_tools(&self, tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
        Ok(tools::response_tool_defs(tools))
    }

    fn parse_tool_calls(
        &self,
        result: &LlmResponseType,
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
        result.tool_calls.clone().unwrap_or_default()
    }
}
