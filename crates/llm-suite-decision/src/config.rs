//! Configuration for decision providers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::error::DecisionError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionConfig {
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub model: String,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub no_proxy: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub query_params: HashMap<String, String>,
}

fn default_timeout_secs() -> u64 {
    30
}

impl DecisionConfig {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            model: model.into(),
            timeout_secs: default_timeout_secs(),
            proxy: None,
            no_proxy: Vec::new(),
            headers: HashMap::new(),
            query_params: HashMap::new(),
        }
    }

    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.openai.com/v1/decisions", model).with_api_key(api_key)
    }

    pub fn jev(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.typesafe.ai/v1/systemone", model).with_api_key(api_key)
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self
    }

    pub fn with_no_proxy(mut self, no_proxy: Vec<String>) -> Self {
        self.no_proxy = no_proxy;
        self
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.base_url.is_empty() {
            return Err(DecisionError::Config("base_url is required".into()));
        }
        if self.model.is_empty() {
            return Err(DecisionError::Config("model is required".into()));
        }
        Url::parse(&self.base_url)
            .map_err(|err| DecisionError::Config(format!("invalid base_url: {err}")))?;
        Ok(())
    }
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1/decisions".into(),
            api_key: None,
            model: "gpt-6-luna".into(),
            timeout_secs: default_timeout_secs(),
            proxy: None,
            no_proxy: Vec::new(),
            headers: HashMap::new(),
            query_params: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_requires_base_url_and_model() {
        let mut cfg = DecisionConfig::new("", "m");
        assert!(cfg.validate().is_err());
        cfg.base_url = "https://example.com".into();
        cfg.model = "".into();
        assert!(cfg.validate().is_err());
        cfg.model = "m".into();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn presets_use_full_endpoints() {
        let o = DecisionConfig::openai("key", "m");
        assert_eq!(o.base_url, "https://api.openai.com/v1/decisions");
        let j = DecisionConfig::jev("key", "m");
        assert_eq!(j.base_url, "https://api.typesafe.ai/v1/systemone");
    }
}
