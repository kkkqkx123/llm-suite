//! OpenAI-style `/decisions` provider.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::DecisionConfig;
use crate::error::{DecisionError, Result};
use crate::provider::DecisionProvider;
use crate::types::{
    ChoiceProbability, ChoiceValue, DecisionAnswer, DecisionQuestion, DecisionRequest,
    DecisionResponse, DecisionUsage, InputTokensDetails, NamedAnswer, OutputTokensDetails,
    ScoreProbability, validate_request, DecisionState,
};

pub struct OpenAIDecisionProvider {
    config: DecisionConfig,
    client: reqwest::Client,
}

impl std::fmt::Debug for OpenAIDecisionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIDecisionProvider")
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    model: String,
    answers: Vec<OpenAIAnswer>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIAnswer {
    Predicate {
        #[serde(default)]
        name: Option<String>,
        probability: f64,
    },
    Choice {
        #[serde(default)]
        name: Option<String>,
        choice: ChoiceValue,
        confidence: f64,
        probabilities: Vec<OpenAIChoiceProbability>,
    },
    Score {
        #[serde(default)]
        name: Option<String>,
        score: f64,
        confidence: f64,
        probabilities: Vec<OpenAIScoreProbability>,
    },
    Refusal {
        #[serde(default)]
        name: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct OpenAIChoiceProbability {
    value: ChoiceValue,
    probability: f64,
}

#[derive(Debug, Deserialize)]
struct OpenAIScoreProbability {
    label: String,
    probability: f64,
    value: i64,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAIUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<OpenAIInputTokensDetails>,
    #[serde(default)]
    output_tokens_details: Option<OpenAIOutputTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAIInputTokensDetails {
    #[serde(default)]
    cache_write_tokens: u64,
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAIOutputTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

impl OpenAIDecisionProvider {
    pub fn new(config: DecisionConfig) -> Result<Self> {
        config.validate()?;
        let client = llm_suite_proxy::build_http_client(
            config.timeout_secs,
            config.proxy.as_deref(),
            &config.no_proxy,
        )
        .map_err(|err| DecisionError::Config(err.to_string()))?;
        Ok(Self { config, client })
    }

    pub fn config(&self) -> &DecisionConfig {
        &self.config
    }

    fn build_body(&self, request: &DecisionRequest) -> Result<Value> {
        validate_request(request)?;
        if request.questions.is_empty() || request.questions.len() > 200 {
            return Err(DecisionError::InvalidRequest(
                "OpenAI decision questions must contain 1 to 200 items".into(),
            ));
        }
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());

        let image_count = count_images(&request.state);
        if image_count > 128 {
            return Err(DecisionError::InvalidRequest(
                "OpenAI decision supports at most 128 images".into(),
            ));
        }
        let input = state_to_input(&request.state)?;
        let questions = questions_to_openai(&request.questions)?;

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), Value::String(model));
        body.insert("input".to_string(), input);
        body.insert("questions".to_string(), Value::Array(questions));

        if let Some(value) = &request.safety_identifier {
            body.insert("safety_identifier".to_string(), Value::String(value.clone()));
        }

        Ok(Value::Object(body))
    }

    fn parse_response(&self, response: OpenAIResponse) -> DecisionResponse {
        let answers = response
            .answers
            .into_iter()
            .map(|answer| match answer {
                OpenAIAnswer::Predicate { name, probability } => NamedAnswer {
                    name: name.clone(),
                    answer: DecisionAnswer::Predicate { name, probability },
                },
                OpenAIAnswer::Choice {
                    name,
                    choice,
                    confidence,
                    probabilities,
                } => NamedAnswer {
                    name: name.clone(),
                    answer: DecisionAnswer::Choice {
                        name,
                        choice,
                        confidence,
                        probabilities: probabilities
                            .into_iter()
                            .map(|item| ChoiceProbability {
                                value: item.value,
                                probability: item.probability,
                            })
                            .collect(),
                    },
                },
                OpenAIAnswer::Score {
                    name,
                    score,
                    confidence,
                    probabilities,
                } => NamedAnswer {
                    name: name.clone(),
                    answer: DecisionAnswer::Score {
                        name,
                        score,
                        confidence,
                        probabilities: probabilities
                            .into_iter()
                            .map(|item| ScoreProbability {
                                level: item.value,
                                label: Some(item.label),
                                probability: item.probability,
                            })
                            .collect(),
                        legend: None,
                    },
                },
                OpenAIAnswer::Refusal { name } => NamedAnswer {
                    name: name.clone(),
                    answer: DecisionAnswer::Refusal { name },
                },
            })
            .collect();

        DecisionResponse {
            model: response.model,
            answers,
            usage: DecisionUsage {
                input_tokens: Some(response.usage.input_tokens),
                output_tokens: Some(response.usage.output_tokens),
                total_tokens: Some(response.usage.total_tokens),
                input_tokens_details: response.usage.input_tokens_details.map(|details| {
                    InputTokensDetails {
                        cache_write_tokens: details.cache_write_tokens,
                        cached_tokens: details.cached_tokens,
                    }
                }),
                output_tokens_details: response.usage.output_tokens_details.map(|details| {
                    OutputTokensDetails {
                        reasoning_tokens: details.reasoning_tokens,
                    }
                }),
            },
        }
    }
}

fn count_images(state: &DecisionState) -> usize {
    match state {
        DecisionState::Parts(parts) => parts
            .iter()
            .filter(|part| matches!(part, crate::types::ContentPart::InputImage { .. }))
            .count(),
        DecisionState::Messages(messages) => messages
            .iter()
            .filter_map(|message| match &message.content {
                crate::types::MessageContent::Parts(parts) => Some(
                    parts
                        .iter()
                        .filter(|part| matches!(part, crate::types::ContentPart::InputImage { .. }))
                        .count(),
                ),
                _ => None,
            })
            .sum(),
        _ => 0,
    }
}

fn state_to_input(state: &DecisionState) -> Result<Value> {
    match state {
        DecisionState::Text(text) => Ok(Value::String(text.clone())),
        DecisionState::Json(_) => Err(DecisionError::InvalidRequest(
            "OpenAI decision protocol does not support JSON state".into(),
        )),
        DecisionState::Parts(parts) => {
            for part in parts {
                match part {
                    crate::types::ContentPart::InputText { .. }
                    | crate::types::ContentPart::InputImage { .. } => {}
                    crate::types::ContentPart::InputJson { .. } => {
                        return Err(DecisionError::InvalidRequest(
                            "OpenAI decision protocol does not support JSON parts".into(),
                        ));
                    }
                }
            }
            let parts = serde_json::to_value(parts)
                .map_err(|err| DecisionError::Decode(err.to_string()))?;
            Ok(json!([{ "role": "user", "type": "message", "content": parts }]))
        }
        DecisionState::Messages(messages) => {
            for message in messages {
                if !matches!(message.role, crate::types::MessageRole::User) {
                    return Err(DecisionError::InvalidRequest(
                        "OpenAI decision protocol only supports user messages".into(),
                    ));
                }
            }
            serde_json::to_value(messages).map_err(|err| DecisionError::Decode(err.to_string()))
        }
    }
}

fn questions_to_openai(questions: &[crate::types::NamedQuestion]) -> Result<Vec<Value>> {
    let mut output = Vec::new();
    for named in questions {
        let question = named.question.clone();
        if let DecisionQuestion::Predicate {
            instructions, ..
        } = &question
        {
            match instructions {
                Some(value) if value_non_empty(value) => {}
                _ => {
                    return Err(DecisionError::InvalidRequest(
                        "predicate instructions must not be empty".into(),
                    ));
                }
            }
        }
        if let DecisionQuestion::Choice {
            instructions: Some(instructions),
            ..
        } = &question
        {
            if !value_non_empty(instructions) {
                return Err(DecisionError::InvalidRequest(
                    "choice instructions must not be empty".into(),
                ));
            }
        }
        if let DecisionQuestion::Choice { choices, .. } = &question {
            if choices.len() < 2 {
                return Err(DecisionError::InvalidRequest(
                    "OpenAI decision choice requires 2 to 255 options".into(),
                ));
            }
        }
        if let DecisionQuestion::Choice {
            instructions: None, ..
        } = &question
        {
            return Err(DecisionError::InvalidRequest(
                "choice instructions must not be empty".into(),
            ));
        }
        if let DecisionQuestion::Score {
            instructions: Some(instructions),
            ..
        } = &question
        {
            if !value_non_empty(instructions) {
                return Err(DecisionError::InvalidRequest(
                    "score instructions must not be empty".into(),
                ));
            }
        }
        if let DecisionQuestion::Score {
            instructions: None, ..
        } = &question
        {
            return Err(DecisionError::InvalidRequest(
                "score instructions must not be empty".into(),
            ));
        }
        if let DecisionQuestion::Score { levels, .. } = &question {
            for level in levels {
                match &level.label {
                    Some(label) if !label.trim().is_empty() => {}
                    _ => {
                        return Err(DecisionError::InvalidRequest(
                            "score level label must not be empty".into(),
                        ));
                    }
                }
            }
        }
        if let DecisionQuestion::Box { .. } = &question {
            return Err(DecisionError::InvalidRequest(
                "OpenAI protocol does not support box questions".into(),
            ));
        }

        let mut value = serde_json::to_value(question)
            .map_err(|err| DecisionError::Decode(err.to_string()))?;
        if let Some(name) = &named.name {
            if let Value::Object(map) = &mut value {
                map.insert("name".to_string(), Value::String(name.clone()));
            }
        }
        // Ensure question name from NamedQuestion is present when available.
        if named.name.is_some() {
            if let Value::Object(map) = &mut value {
                if !map.contains_key("name") {
                    map.insert(
                        "name".to_string(),
                        Value::String(named.name.clone().unwrap()),
                    );
                }
            }
        }
        output.push(value);
    }
    Ok(output)
}

fn value_non_empty(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
        _ => true,
    }
}

#[async_trait]
impl DecisionProvider for OpenAIDecisionProvider {
    async fn create(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let body = self.build_body(request)?;

        let mut outgoing = self.client.post(&self.config.base_url).json(&body);
        if let Some(api_key) = &self.config.api_key {
            outgoing = outgoing.bearer_auth(api_key);
        }
        for (name, value) in &self.config.headers {
            outgoing = outgoing.header(name, value);
        }
        if !self.config.query_params.is_empty() {
            outgoing = outgoing.query(&self.config.query_params);
        }

        let response = outgoing
            .send()
            .await
            .map_err(DecisionError::from)?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            let body = response.text().await.unwrap_or_default();
            return Err(DecisionError::Provider {
                status: status.as_u16(),
                message: body,
                retry_after_ms: llm_suite_common::parse_retry_after_ms(retry_after.as_deref()),
            });
        }

        let decoded: OpenAIResponse = response
            .json()
            .await
            .map_err(|err| DecisionError::Decode(err.to_string()))?;
        Ok(self.parse_response(decoded))
    }

    fn protocol_name(&self) -> &'static str {
        "openai"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NamedQuestion;

    #[test]
    fn build_body_openai_predicate() {
        let config = DecisionConfig::default();
        let provider = OpenAIDecisionProvider::new(config).unwrap();
        let request = DecisionRequest {
            model: None,
            state: DecisionState::Text("hello".into()),
            questions: vec![NamedQuestion {
                name: Some("q1".into()),
                question: DecisionQuestion::Predicate {
                    name: None,
                    instructions: Some(serde_json::json!("is it?")),
                    true_description: None,
                    false_description: None,
                },
            }],
            safety_identifier: None,
            assets: None,
        };
        let body = provider.build_body(&request).unwrap();
        assert_eq!(body["model"], "gpt-6-luna");
        assert_eq!(body["input"], "hello");
        assert_eq!(body["questions"][0]["type"], "predicate");
    }

    #[test]
    fn build_body_rejects_box() {
        let config = DecisionConfig::default();
        let provider = OpenAIDecisionProvider::new(config).unwrap();
        let request = DecisionRequest {
            model: None,
            state: DecisionState::Text("hi".into()),
            questions: vec![NamedQuestion {
                name: Some("q".into()),
                question: DecisionQuestion::Box { name: None, instructions: Some(serde_json::json!("loc")) },
            }],
            safety_identifier: None,
            assets: None,
        };
        assert!(provider.build_body(&request).is_err());
    }

    #[test]
    fn parse_openai_response() {
        let config = DecisionConfig::default();
        let provider = OpenAIDecisionProvider::new(config).unwrap();
        let raw = r#"{"model":"m","answers":[{"type":"predicate","name":"q","probability":0.9},{"type":"refusal","name":"r"}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}"#;
        let decoded: OpenAIResponse = serde_json::from_str(raw).unwrap();
        let resp = provider.parse_response(decoded);
        assert_eq!(resp.answers.len(), 2);
        assert!(matches!(resp.answers[1].answer, DecisionAnswer::Refusal { .. }));
    }
}
