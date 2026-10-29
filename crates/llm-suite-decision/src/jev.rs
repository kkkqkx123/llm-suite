//! Jev/TypeSafe-style `/v1/systemone` provider.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::DecisionConfig;
use crate::error::{DecisionError, Result};
use crate::provider::DecisionProvider;
use crate::types::{
    BoxCoordinates, ChoiceOption, ChoiceProbability, ChoiceValue, DecisionAnswer, DecisionQuestion,
    DecisionRequest, DecisionResponse, DecisionState, DecisionUsage, NamedAnswer, NamedQuestion,
    ScoreProbability, validate_request,
};

pub struct JevDecisionProvider {
    config: DecisionConfig,
    client: reqwest::Client,
}

impl std::fmt::Debug for JevDecisionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevDecisionProvider")
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct JevResponse {
    model: String,
    answers: BTreeMap<String, JevAnswer>,
    usage: Option<JevUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum JevAnswer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: BTreeMap<String, f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: BTreeMap<String, Value>,
        probabilities: BTreeMap<String, f64>,
    },
    Box {
        #[serde(rename = "box")]
        coordinates: BoxCoordinates,
        confidence: Option<f64>,
    },
}

#[derive(Debug, Deserialize, Default)]
struct JevUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
}

impl JevDecisionProvider {
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
        // Jev requires names for every question.
        for named in &request.questions {
            if named.name.is_none() && named.question_name().is_none() {
                return Err(DecisionError::InvalidRequest(
                    "Jev protocol requires a name for every question".into(),
                ));
            }
        }

        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());

        let state = state_to_jev(&request.state);

        let mut questions = serde_json::Map::new();
        for named in &request.questions {
            let name = named
                .name
                .clone()
                .or_else(|| named.question_name())
                .ok_or_else(|| {
                    DecisionError::InvalidRequest(
                        "Jev protocol requires a name for every question".into(),
                    )
                })?;
            let question = question_to_jev(&named.question)?;
            questions.insert(name, question);
        }

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), Value::String(model));
        body.insert("state".to_string(), state);
        body.insert("questions".to_string(), Value::Object(questions));

        Ok(Value::Object(body))
    }

    fn parse_response(&self, response: JevResponse) -> DecisionResponse {
        let mut answers = Vec::new();
        for (name, answer) in response.answers {
            let named = match answer {
                JevAnswer::Noul { noul } => NamedAnswer {
                    name: Some(name.clone()),
                    answer: DecisionAnswer::Predicate {
                        name: Some(name),
                        probability: noul,
                    },
                },
                JevAnswer::Choice {
                    choice,
                    confidence,
                    probabilities,
                } => NamedAnswer {
                    name: Some(name.clone()),
                    answer: DecisionAnswer::Choice {
                        name: Some(name),
                        choice: ChoiceValue::Text(choice),
                        confidence,
                        probabilities: probabilities
                            .into_iter()
                            .map(|(key, value)| ChoiceProbability {
                                value: ChoiceValue::Text(key),
                                probability: value,
                            })
                            .collect(),
                    },
                },
                JevAnswer::Score {
                    score,
                    confidence,
                    legend,
                    probabilities,
                } => {
                    let mut legend_out = BTreeMap::new();
                    for (key, value) in legend {
                        if let Ok(level) = key.parse::<i64>() {
                            legend_out.insert(level, value);
                        }
                    }
                    let probability_list = probabilities
                        .into_iter()
                        .filter_map(|(key, value)| {
                            key.parse::<i64>().ok().map(|level| ScoreProbability {
                                level,
                                label: legend_out.get(&level).and_then(value_as_text),
                                probability: value,
                            })
                        })
                        .collect();
                    NamedAnswer {
                        name: Some(name.clone()),
                        answer: DecisionAnswer::Score {
                            name: Some(name),
                            score,
                            confidence,
                            probabilities: probability_list,
                            legend: Some(legend_out),
                        },
                    }
                }
                JevAnswer::Box {
                    coordinates,
                    confidence,
                } => NamedAnswer {
                    name: Some(name.clone()),
                    answer: DecisionAnswer::Box {
                        name: Some(name),
                        coordinates,
                        confidence,
                    },
                },
            };
            answers.push(named);
        }

        let usage = response.usage.unwrap_or_default();
        DecisionResponse {
            model: response.model,
            answers,
            usage: DecisionUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: None,
                input_tokens_details: None,
                output_tokens_details: None,
            },
        }
    }
}

trait QuestionName {
    fn question_name(&self) -> Option<String>;
}

impl QuestionName for NamedQuestion {
    fn question_name(&self) -> Option<String> {
        match &self.question {
            DecisionQuestion::Predicate { name, .. }
            | DecisionQuestion::Choice { name, .. }
            | DecisionQuestion::Score { name, .. }
            | DecisionQuestion::Box { name, .. } => name.clone(),
        }
    }
}

fn state_to_jev(state: &DecisionState) -> Value {
    match state {
        DecisionState::Text(text) => Value::String(text.clone()),
        DecisionState::Json(value) => value.clone(),
        DecisionState::Parts(parts) => {
            let mut output = Vec::new();
            for part in parts {
                output.push(serde_json::to_value(part).unwrap_or(Value::Null));
            }
            Value::Array(output)
        }
        DecisionState::Messages(messages) => {
            serde_json::to_value(messages).unwrap_or(Value::Null)
        }
    }
}

fn question_to_jev(question: &DecisionQuestion) -> Result<Value> {
    match question {
        DecisionQuestion::Predicate {
            instructions,
            true_description,
            false_description,
            ..
        } => {
            let mut criteria = serde_json::Map::new();
            if let Some(value) = true_description {
                criteria.insert("true".to_string(), value.clone());
            }
            if let Some(value) = false_description {
                criteria.insert("false".to_string(), value.clone());
            }
            let mut object = serde_json::Map::new();
            object.insert("type".to_string(), Value::String("noul".to_string()));
            if let Some(instructions) = instructions_if_non_empty(instructions) {
                object.insert("instructions".to_string(), instructions);
            }
            if !criteria.is_empty() {
                object.insert("criteria".to_string(), Value::Object(criteria));
            }
            Ok(Value::Object(object))
        }
        DecisionQuestion::Choice {
            instructions,
            choices,
            ..
        } => {
            let mut criteria = serde_json::Map::new();
            for choice in choices {
                criteria.insert(choice.value.as_text(), choice.description.clone().unwrap_or(Value::Null));
            }
            let mut object = serde_json::Map::new();
            object.insert("type".to_string(), Value::String("choice".to_string()));
            if let Some(instructions) = instructions.as_ref().and_then(instructions_if_non_empty) {
                object.insert("instructions".to_string(), instructions);
            }
            object.insert("criteria".to_string(), Value::Object(criteria));
            Ok(Value::Object(object))
        }
        DecisionQuestion::Score {
            instructions,
            levels,
            ..
        } => {
            let criteria: Vec<Value> = levels
                .iter()
                .map(|level| {
                    level
                        .description
                        .clone()
                        .or_else(|| level.label.clone().map(Value::String))
                        .unwrap_or(Value::Null)
                })
                .collect();
            let mut object = serde_json::Map::new();
            object.insert("type".to_string(), Value::String("score".to_string()));
            if let Some(instructions) = instructions.as_ref().and_then(instructions_if_non_empty) {
                object.insert("instructions".to_string(), instructions);
            }
            object.insert("criteria".to_string(), Value::Array(criteria));
            Ok(Value::Object(object))
        }
        DecisionQuestion::Box { instructions, .. } => {
            let mut object = serde_json::Map::new();
            object.insert("type".to_string(), Value::String("box".to_string()));
            if let Some(instructions) = instructions.as_ref().and_then(instructions_if_non_empty) {
                object.insert("instructions".to_string(), instructions);
            }
            Ok(Value::Object(object))
        }
    }
}

fn instructions_if_non_empty(value: &Value) -> Option<Value> {
    match value {
        Value::String(text) if text.trim().is_empty() => None,
        Value::Array(items) if items.is_empty() => None,
        Value::Object(map) if map.is_empty() => None,
        Value::Null => None,
        _ => Some(value.clone()),
    }
}

fn value_as_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        _ => None,
    }
}

#[async_trait]
impl DecisionProvider for JevDecisionProvider {
    async fn create(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let body = self.build_body(request)?;

        // Determine pending assets. Explicit assets are keyed by alias; references
        // embedded in the serialized body are matched against known assets.
        let explicit = request.assets.clone().unwrap_or_default();
        let mut known: HashMap<String, crate::assets::ImageAsset> = HashMap::new();
        for (alias, asset) in &explicit {
            known.insert(alias.clone(), asset.clone());
            known.insert(asset.reference().to_string(), asset.clone());
        }

        let base_bytes = serde_json::to_vec(&body)
            .map_err(|err| DecisionError::Decode(err.to_string()))?;

        // Collect pending assets in insertion order: explicit first, then
        // references found in the body in order of appearance.
        let mut pending: Vec<(String, crate::assets::ImageAsset)> = Vec::new();
        let mut pending_keys: HashSet<String> = HashSet::new();
        for (alias, asset) in &explicit {
            if pending_keys.insert(alias.clone()) {
                pending.push((alias.clone(), asset.clone()));
            }
        }
        if !known.is_empty() {
            for reference in crate::assets::scan_references(&base_bytes) {
                if let Some(asset) = known.get(&reference) {
                    if pending_keys.insert(reference.clone()) {
                        pending.push((reference, asset.clone()));
                    }
                }
            }
        }

        let (content_type, body_bytes) = if pending.is_empty() {
            ("application/json".to_string(), base_bytes)
        } else {
            // Resolve all assets.
            let mut resolved_map: HashMap<String, crate::assets::ResolvedImage> = HashMap::new();
            for (placeholder, asset) in &pending {
                let resolved = asset.resolve_async().await.map_err(|err| {
                    DecisionError::InvalidRequest(format!("Asset {placeholder:?}: {err}"))
                })?;
                resolved_map.insert(placeholder.clone(), resolved);
            }

            // Build asset entries.
            let mut asset_entries = serde_json::Map::new();
            for (i, (placeholder, _)) in pending.iter().enumerate() {
                let part_name = format!("asset-{}", i);
                let mut entry = serde_json::Map::new();
                entry.insert("type".to_string(), Value::String("image".to_string()));
                entry.insert("part".to_string(), Value::String(part_name));
                asset_entries.insert(placeholder.clone(), Value::Object(entry));
            }

            // Re-serialize body with assets.
            let mut body_with_assets = body.clone();
            if let Value::Object(map) = &mut body_with_assets {
                map.insert("assets".to_string(), Value::Object(asset_entries));
            }
            let request_bytes = serde_json::to_vec(&body_with_assets)
                .map_err(|err| DecisionError::Decode(err.to_string()))?;

            // Build multipart.
            let mut parts = Vec::new();
            parts.push(crate::assets::Part {
                name: "request".to_string(),
                content_type: "application/json".to_string(),
                content: request_bytes,
                filename: None,
            });
            for (i, (placeholder, _)) in pending.iter().enumerate() {
                let resolved = resolved_map.get(placeholder).unwrap();
                parts.push(crate::assets::Part {
                    name: format!("asset-{}", i),
                    content_type: resolved.content_type.clone(),
                    content: resolved.data.clone(),
                    filename: Some(format!("asset-{}", i)),
                });
            }
            let (ct, bytes) = crate::assets::encode_multipart(&parts);
            (ct, bytes)
        };

        let mut outgoing = self
            .client
            .post(&self.config.base_url)
            .body(body_bytes);
        // set content type
        outgoing = outgoing.header("content-type", &content_type);

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

        let decoded: JevResponse = response
            .json()
            .await
            .map_err(|err| DecisionError::Decode(err.to_string()))?;
        Ok(self.parse_response(decoded))
    }

    fn protocol_name(&self) -> &'static str {
        "jev"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_jev_requires_names() {
        let config = DecisionConfig::default();
        let provider = JevDecisionProvider::new(config).unwrap();
        let request = DecisionRequest {
            model: None,
            state: DecisionState::Text("hi".into()),
            questions: vec![NamedQuestion {
                name: None,
                question: DecisionQuestion::Predicate {
                    name: None,
                    instructions: serde_json::json!("q"),
                    true_description: None,
                    false_description: None,
                },
            }],
            prompt_cache_key: None,
            prompt_cache_options: None,
            safety_identifier: None,
            assets: None,
        };
        // name missing both NamedQuestion and question -> error
        assert!(provider.build_body(&request).is_err());
    }

    #[test]
    fn build_body_jev_maps_questions() {
        let config = DecisionConfig::default();
        let provider = JevDecisionProvider::new(config).unwrap();
        let request = DecisionRequest {
            model: Some("m".into()),
            state: DecisionState::Text("state".into()),
            questions: vec![NamedQuestion {
                name: Some("q1".into()),
                question: DecisionQuestion::Choice {
                    name: None,
                    instructions: Some(serde_json::json!("choose")),
                    choices: vec![
                        ChoiceOption { value: ChoiceValue::Text("a".into()), description: Some(serde_json::json!("desc a")) },
                        ChoiceOption { value: ChoiceValue::Text("b".into()), description: None },
                    ],
                },
            }],
            prompt_cache_key: None,
            prompt_cache_options: None,
            safety_identifier: None,
            assets: None,
        };
        let body = provider.build_body(&request).unwrap();
        assert_eq!(body["model"], "m");
        assert_eq!(body["state"], "state");
        assert!(body["questions"]["q1"]["criteria"].is_object());
    }

    #[test]
    fn parse_jev_response() {
        let config = DecisionConfig::default();
        let provider = JevDecisionProvider::new(config).unwrap();
        let raw = r#"{"model":"m","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":2}}"#;
        let decoded: JevResponse = serde_json::from_str(raw).unwrap();
        let resp = provider.parse_response(decoded);
        assert_eq!(resp.answers.len(), 1);
        assert_eq!(resp.answers[0].name.as_deref(), Some("q"));
    }
}
