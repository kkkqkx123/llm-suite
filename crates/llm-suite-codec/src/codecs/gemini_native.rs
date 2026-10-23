//! Wire protocol codec for the Gemini native API.

use std::sync::atomic::AtomicUsize;

use super::LlmCodec;
use crate::error::LlmResult;
use llm_suite_types::llm::{
    LlmProfile, LlmRequest, LlmResult as LlmResponseType, MessageStreamEvent,
};
use llm_suite_types::tool::Tool;

pub(crate) mod body;
pub(crate) mod contents;
pub(crate) mod endpoint;
pub(crate) mod finish_reason;
pub(crate) mod part;
pub(crate) mod response;
pub(crate) mod stream;
pub(crate) mod tools;
pub(crate) mod transport;
pub(crate) mod usage;

#[cfg(test)]
pub(crate) mod test_support;

pub struct GeminiNativeCodec {
    base_url: String,
    /// Tool-call index source for streamed function calls. Every streamed
    /// `functionCall` part is a complete snapshot, so each call needs its own
    /// index to stay apart in the stream accumulator. The counter is reset
    /// when a chunk carries the finish reason, so the numbering describes the
    /// position of the call inside its own response.
    call_index: AtomicUsize,
}

impl Default for GeminiNativeCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiNativeCodec {
    pub fn new() -> Self {
        Self {
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            call_index: AtomicUsize::new(0),
        }
    }

    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            call_index: AtomicUsize::new(0),
        }
    }
}

impl LlmCodec for GeminiNativeCodec {
    fn build_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<reqwest::Request> {
        let streaming = request.stream == Some(true);

        // Streaming uses the SSE endpoint; the non-streaming endpoint returns
        // a single JSON object that no SSE parser can consume.
        let action = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        let url = super::shared::endpoint_url(
            endpoint::endpoint_base(&self.base_url, profile),
            &format!("{}:{}", endpoint::model_segment(&profile.model), action),
        );

        let body = body::build_body(request, profile)?;

        transport::build_json_request(url, &body, profile)
    }

    fn build_count_tokens_request(
        &self,
        request: &LlmRequest,
        profile: &LlmProfile,
    ) -> LlmResult<Option<reqwest::Request>> {
        let url = super::shared::endpoint_url(
            endpoint::endpoint_base(&self.base_url, profile),
            &format!("{}:countTokens", endpoint::model_segment(&profile.model)),
        );

        // The counting endpoint takes the request contents only: system
        // instructions, tool declarations and the generation config are not
        // part of its contract.
        let use_text_mode = super::shared::is_text_mode(request);
        let history = if use_text_mode {
            super::shared::convert_history_for_text_mode(&request.messages, request)
        } else {
            request.messages.clone()
        };
        let body = serde_json::json!({
            "contents": contents::convert_messages(&history)?,
        });

        Ok(Some(transport::build_json_request(url, &body, profile)?))
    }

    fn parse_count_tokens_response(&self, body: &serde_json::Value) -> LlmResult<u32> {
        // The Gemini count-tokens API only ever returns `totalTokens`.
        Ok(body
            .get("totalTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32)
    }

    fn parse_response(&self, body: &str, request: &LlmRequest) -> LlmResult<LlmResponseType> {
        response::parse_response(body, request)
    }

    fn parse_stream_chunk_events(&self, data: &str) -> LlmResult<Vec<MessageStreamEvent>> {
        self.parse_stream_events(data)
    }

    fn convert_tools(&self, tools: &[Tool]) -> LlmResult<Vec<serde_json::Value>> {
        tools::convert_tools(tools)
    }

    fn parse_tool_calls(
        &self,
        result: &LlmResponseType,
    ) -> Vec<llm_suite_types::message::LlmToolCall> {
        result.tool_calls.clone().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::gemini_native::test_support::{count_request, msg, profile, request};
    use llm_suite_types::message::MessageRole;

    #[test]
    fn thinking_level_is_emitted_as_gemini_enum_member() {
        let mut config = serde_json::json!({});
        crate::generation::apply_gemini_generation_config(
            &mut config,
            &llm_suite_types::llm::generation::LlmGenerationParams {
                thinking: Some(llm_suite_types::llm::generation::LlmThinkingConfig {
                    level: Some(llm_suite_types::llm::generation::ThinkingLevel::High),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect("thinking level must be accepted");
        assert_eq!(
            config["thinkingConfig"]["thinkingLevel"],
            serde_json::json!("HIGH")
        );
    }

    #[test]
    fn named_tool_choice_restricts_the_allowed_functions() {
        let tool_config = crate::generation::gemini_tool_config(
            &llm_suite_types::llm::generation::LlmGenerationParams {
                tool_choice: Some(llm_suite_types::llm::generation::LlmToolChoice {
                    mode: None,
                    tool_name: Some("get_weather".to_string()),
                }),
                ..Default::default()
            },
        )
        .expect("tool choice must be accepted");
        assert_eq!(
            tool_config["functionCallingConfig"]["mode"],
            serde_json::json!("ANY")
        );
        assert_eq!(
            tool_config["functionCallingConfig"]["allowedFunctionNames"],
            serde_json::json!(["get_weather"])
        );
    }

    #[test]
    fn count_tokens_request_targets_count_tokens_endpoint_without_key_in_url() {
        let codec = GeminiNativeCodec::new();
        let req = codec
            .build_count_tokens_request(&count_request(), &profile())
            .expect("count request must build")
            .expect("gemini native supports counting");
        let url = req.url().as_str();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:countTokens",
            "api key must not appear in the URL"
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "sk-test",
            "auth must ride the x-goog-api-key header"
        );
        let body: serde_json::Value = req
            .body()
            .unwrap()
            .as_bytes()
            .map(|b| serde_json::from_slice(b).unwrap())
            .unwrap();
        assert_eq!(body["contents"][0]["role"], serde_json::json!("user"));
        assert!(
            body.get("systemInstruction").is_none(),
            "the counting endpoint does not take a system instruction"
        );
        assert!(
            body.get("generationConfig").is_none(),
            "count body must not carry output steering"
        );
    }

    #[test]
    fn stream_request_targets_stream_generate_content_sse() {
        let codec = GeminiNativeCodec::new();
        let mut req = request(vec![msg(MessageRole::User, "Hi")], None);
        req.stream = Some(true);
        let req = codec.build_request(&req, &profile()).expect("must build");
        let url = req.url().as_str();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "sk-test",
            "auth must ride the x-goog-api-key header"
        );
    }

    #[test]
    fn non_stream_request_targets_generate_content() {
        let codec = GeminiNativeCodec::new();
        let req = codec
            .build_request(
                &request(vec![msg(MessageRole::User, "Hi")], None),
                &profile(),
            )
            .expect("must build");
        assert_eq!(
            req.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn a_model_that_already_carries_the_prefix_is_not_prefixed_twice() {
        let codec = GeminiNativeCodec::new();
        let mut p = profile();
        p.model = "models/gemini-1.5-pro".to_string();
        let req = codec
            .build_request(&request(vec![msg(MessageRole::User, "Hi")], None), &p)
            .expect("must build");
        assert_eq!(
            req.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn a_base_url_with_a_query_still_yields_a_valid_streaming_url() {
        let codec = GeminiNativeCodec::new();
        let mut p = profile();
        p.base_url = Some("https://gateway.example.com/v1?api-version=2024-01".to_string());
        let mut req = request(vec![msg(MessageRole::User, "Hi")], None);
        req.stream = Some(true);
        let built = codec.build_request(&req, &p).expect("must build");
        assert_eq!(
            built.url().as_str(),
            "https://gateway.example.com/v1/models/gemini-1.5-pro:streamGenerateContent?api-version=2024-01&alt=sse"
        );
    }

    #[test]
    fn count_tokens_body_covers_the_same_contents_as_the_inference_request() {
        let codec = GeminiNativeCodec::new();
        let inference =
            body::build_body(&count_request(), &profile()).expect("inference body must build");
        let req = codec
            .build_count_tokens_request(&count_request(), &profile())
            .expect("count request must build")
            .unwrap();
        let count_body: serde_json::Value = req
            .body()
            .unwrap()
            .as_bytes()
            .map(|b| serde_json::from_slice(b).unwrap())
            .unwrap();
        assert_eq!(count_body["contents"], inference["contents"]);
        assert!(
            count_body.get("systemInstruction").is_none(),
            "the counting endpoint does not take a system instruction"
        );
        assert!(count_body.get("tools").is_none());
    }

    #[test]
    fn count_tokens_response_parses_total_tokens() {
        let codec = GeminiNativeCodec::new();
        let body: serde_json::Value = serde_json::from_str(r#"{"totalTokens": 42}"#).unwrap();
        assert_eq!(codec.parse_count_tokens_response(&body).unwrap(), 42);
        let empty: serde_json::Value = serde_json::from_str("{}").unwrap();
        assert_eq!(codec.parse_count_tokens_response(&empty).unwrap(), 0);
    }
}
