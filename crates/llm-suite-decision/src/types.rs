//! Common decision types shared by OpenAI and Jev/TypeSafe protocols.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::DecisionError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChoiceValue {
    Bool(bool),
    Text(String),
}

impl ChoiceValue {
    pub fn as_text(&self) -> String {
        match self {
            ChoiceValue::Text(value) => value.clone(),
            ChoiceValue::Bool(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    InputJson {
        data: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Developer,
    System,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionMessage {
    pub role: MessageRole,
    pub content: MessageContent,
    #[serde(rename = "type", default = "default_message_type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

fn default_message_type() -> String {
    "message".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionState {
    Text(String),
    Json(Value),
    Parts(Vec<ContentPart>),
    Messages(Vec<DecisionMessage>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub value: ChoiceValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreLevel {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionQuestion {
    Predicate {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        true_description: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        false_description: Option<Value>,
    },
    Choice {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
        choices: Vec<ChoiceOption>,
    },
    Score {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
        levels: Vec<ScoreLevel>,
    },
    Box {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedQuestion {
    pub name: Option<String>,
    pub question: DecisionQuestion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceProbability {
    pub value: ChoiceValue,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreProbability {
    pub level: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxCoordinates {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Predicate {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        probability: f64,
    },
    Choice {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        choice: ChoiceValue,
        confidence: f64,
        probabilities: Vec<ChoiceProbability>,
    },
    Score {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        score: f64,
        confidence: f64,
        probabilities: Vec<ScoreProbability>,
        #[serde(skip_serializing_if = "Option::is_none")]
        legend: Option<BTreeMap<i64, Value>>,
    },
    Box {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(rename = "box")]
        coordinates: BoxCoordinates,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
    },
    Refusal {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedAnswer {
    pub name: Option<String>,
    pub answer: DecisionAnswer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cache_write_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DecisionUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Vec<NamedAnswer>,
    pub usage: DecisionUsage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub state: DecisionState,
    pub questions: Vec<NamedQuestion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(default, skip)]
    pub assets: Option<std::collections::BTreeMap<String, crate::assets::ImageAsset>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionModelMetadata {
    pub name: String,
    pub description: String,
    pub release_date: String,
}

pub fn validate_request(request: &DecisionRequest) -> Result<(), DecisionError> {
    if request.questions.is_empty() || request.questions.len() > 400 {
        return Err(DecisionError::InvalidRequest(
            "questions must contain 1 to 400 items".into(),
        ));
    }
    for named in &request.questions {
        let question = &named.question;
        match question {
            DecisionQuestion::Predicate { instructions, .. } => {
                if let Some(instructions) = instructions {
                    if !value_non_empty(instructions) {
                        return Err(DecisionError::InvalidRequest(
                            "predicate instructions must not be empty".into(),
                        ));
                    }
                }
            }
            DecisionQuestion::Choice {
                instructions,
                choices,
                ..
            } => {
                if choices.is_empty() || choices.len() > 255 {
                    return Err(DecisionError::InvalidRequest(
                        "choice requires 2 to 255 options".into(),
                    ));
                }
                let mut seen = std::collections::HashSet::new();
                for choice in choices {
                    if !seen.insert(choice.value.as_text()) {
                        return Err(DecisionError::InvalidRequest(
                            "choice values must be unique".into(),
                        ));
                    }
                }
                if let Some(instructions) = instructions {
                    if !value_non_empty(instructions) {
                        return Err(DecisionError::InvalidRequest(
                            "choice instructions must not be empty".into(),
                        ));
                    }
                }
            }
            DecisionQuestion::Score {
                instructions,
                levels,
                ..
            } => {
                if levels.is_empty() {
                    return Err(DecisionError::InvalidRequest(
                        "score requires at least one level".into(),
                    ));
                }
                if let Some(instructions) = instructions {
                    if !value_non_empty(instructions) {
                        return Err(DecisionError::InvalidRequest(
                            "score instructions must not be empty".into(),
                        ));
                    }
                }
            }
            DecisionQuestion::Box { instructions, .. } => {
                if let Some(instructions) = instructions {
                    if !value_non_empty(instructions) {
                        return Err(DecisionError::InvalidRequest(
                            "box instructions must not be empty".into(),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_value_distinguishes_bool_and_text() {
        let bool_value: ChoiceValue = serde_json::from_str("true").unwrap();
        let text_value: ChoiceValue = serde_json::from_str("\"true\"").unwrap();
        assert_ne!(bool_value, text_value);
        assert_eq!(serde_json::to_string(&bool_value).unwrap(), "true");
        assert_eq!(serde_json::to_string(&text_value).unwrap(), "\"true\"");
    }

    #[test]
    fn validate_request_rejects_empty_questions() {
        let request = DecisionRequest {
            model: None,
            state: DecisionState::Text("hello".to_string()),
            questions: Vec::new(),
            safety_identifier: None,
            assets: None,
        };
        assert!(validate_request(&request).is_err());
    }
}
