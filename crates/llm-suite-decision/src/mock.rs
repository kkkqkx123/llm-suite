//! Feature-gated mock decision provider for tests.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::{DecisionError, Result};
use crate::provider::DecisionProvider;
use crate::types::{DecisionRequest, DecisionResponse};

pub struct MockDecisionProvider {
    model: String,
    response: Option<DecisionResponse>,
    error: Option<DecisionError>,
    calls: Mutex<usize>,
    last_request: Mutex<Option<DecisionRequest>>,
}

impl MockDecisionProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            response: None,
            error: None,
            calls: Mutex::new(0),
            last_request: Mutex::new(None),
        }
    }

    pub fn with_response(mut self, response: DecisionResponse) -> Self {
        self.response = Some(response);
        self
    }

    pub fn with_error(mut self, error: DecisionError) -> Self {
        self.error = Some(error);
        self
    }

    pub fn recorded_calls(&self) -> usize {
        *self.calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn last_request(&self) -> Option<DecisionRequest> {
        self.last_request
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[async_trait]
impl DecisionProvider for MockDecisionProvider {
    async fn create(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        *self.calls.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) += 1;
        *self.last_request.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(request.clone());
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        self.response
            .clone()
            .ok_or_else(|| DecisionError::Decode("mock response not configured".into()))
    }

    fn protocol_name(&self) -> &'static str {
        "mock"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
