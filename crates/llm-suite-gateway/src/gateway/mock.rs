//! The mock clients used for test injection.

use super::LlmGateway;

impl LlmGateway {
    /// Register a mock client under an arbitrary id (a real profile id can be
    /// reused, or a plain "mock" id). Mock hits take priority over profiles.
    #[cfg(feature = "mock")]
    pub fn register_mock(
        &self,
        id: impl Into<String>,
        client: std::sync::Arc<llm_suite_client::MockLlmClient>,
    ) {
        self.mock_clients.insert(id.into(), client);
    }

    #[cfg(feature = "mock")]
    pub(super) fn mock_client(
        &self,
        id: &str,
    ) -> Option<std::sync::Arc<llm_suite_client::MockLlmClient>> {
        self.mock_clients.get(id).map(|c| c.clone())
    }
}
