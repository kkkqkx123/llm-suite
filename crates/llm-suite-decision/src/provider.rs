//! Decision provider port.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{DecisionRequest, DecisionResponse};

#[async_trait]
pub trait DecisionProvider: Send + Sync {
    async fn create(&self, request: &DecisionRequest) -> Result<DecisionResponse>;

    fn protocol_name(&self) -> &'static str;

    fn model_name(&self) -> &str;
}
