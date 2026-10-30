//! Explicit cross-model history migration.
//!
//! The gateway never rewrites histories on its own: the request hot path
//! sends exactly the messages the caller assembled. When a session moves
//! to another profile (model switch, provider move, checkpoint resume on a
//! different profile), the owner calls [`LlmGateway::migrate_history_for_profile`]
//! once and continues the session with the returned history.

use llm_suite_message::transform::{transform_for_target, TransformTarget};
use llm_suite_types::message::Message;

use super::LlmGateway;

impl LlmGateway {
    /// Rewrite a history for the target profile, once, before continuing
    /// the session on it.
    ///
    /// Same model and format is a clone-level no-op, so unconditional calls
    /// are safe; a genuine move drops foreign thinking signatures and
    /// native blocks, downgrades images the target cannot carry, shortens
    /// overlong tool-call ids and synthesizes results for orphan calls. An
    /// unknown target profile (or unknown source) never fails: unknown
    /// targets return the history unchanged, unknown sources assume a
    /// cross-model move so no foreign signature leaks.
    pub fn migrate_history_for_profile(
        &self,
        messages: &[Message],
        from_profile_id: Option<&str>,
        to_profile_id: &str,
    ) -> Vec<Message> {
        self.migrate_history_for_profile_with_image_support(
            messages,
            from_profile_id,
            to_profile_id,
            true,
        )
    }

    /// Variant of [`Self::migrate_history_for_profile`] that lets the caller
    /// refine the target's image capability from a model catalog instead of
    /// assuming every wire format carries images.
    pub fn migrate_history_for_profile_with_image_support(
        &self,
        messages: &[Message],
        from_profile_id: Option<&str>,
        to_profile_id: &str,
        supports_image: bool,
    ) -> Vec<Message> {
        let Some(target) = self.profiles.get(to_profile_id) else {
            return messages.to_vec();
        };
        let same_model = from_profile_id
            .and_then(|id| self.profiles.get(id))
            .map(|source| source.model == target.model && source.format == target.format)
            .unwrap_or(false);
        let wire_target = TransformTarget {
            supports_image,
            same_model,
        };
        transform_for_target(messages, &wire_target)
    }
}

#[cfg(test)]
mod tests {
    use llm_suite_types::llm::{LlmFormat, LlmProfile};
    use llm_suite_types::message::{MessageContent, MessageContentValue, MessageRole};

    use super::*;

    fn profile(id: &str, model: &str) -> LlmProfile {
        LlmProfile {
            id: id.to_string(),
            name: id.to_string(),
            format: LlmFormat::OpenaiChat,
            provider_id: None,
            model: model.to_string(),
            api_key: None,
            base_url: None,
            parameters: None,
            generation: None,
            timeout: None,
            max_retries: None,
            retry_delay: None,
            headers: None,
            metadata: None,
            tool_call_protocol: None,
            auth_type: None,
            custom_headers: None,
            custom_body: None,
            custom_body_enabled: None,
            query_params: None,
            stream_options: None,
            prompt_caching: None,
            context_window_size: None,
            max_output_tokens: None,
            proxy: None,
            no_proxy: None,
            circuit_breaker: None,
            auth: None,
            compat: None,
            identity: None,
            transport: None,
            env: None,
        }
    }

    fn thinking_message() -> Message {
        Message {
            id: llm_suite_types::Id::new(),
            role: MessageRole::Assistant,
            content: MessageContentValue::Rich(vec![MessageContent::Thinking {
                thinking: "hmm".to_string(),
                signature: Some("foreign-sig".to_string()),
            }]),
            timestamp: 0,
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
            thinking: None,
            thinking_signature: None,
            metadata: None,
        }
    }

    #[test]
    fn same_model_migration_is_a_noop_clone() {
        let gateway = LlmGateway::new();
        gateway.register_profile(profile("a", "gpt-4o")).unwrap();
        gateway.register_profile(profile("b", "gpt-4o")).unwrap();
        let history = vec![thinking_message()];
        let moved = gateway.migrate_history_for_profile(&history, Some("a"), "b");
        assert_eq!(moved, history);
    }

    #[test]
    fn cross_model_migration_drops_foreign_thinking() {
        let gateway = LlmGateway::new();
        gateway.register_profile(profile("a", "gpt-4o")).unwrap();
        gateway
            .register_profile(profile("b", "claude-sonnet"))
            .unwrap();
        let moved = gateway.migrate_history_for_profile(&[thinking_message()], Some("a"), "b");
        assert_eq!(moved.len(), 1);
        match &moved[0].content {
            MessageContentValue::Rich(blocks) => assert!(
                blocks
                    .iter()
                    .all(|b| !matches!(b, MessageContent::Thinking { .. })),
                "foreign thinking must not leak"
            ),
            MessageContentValue::Text(text) => assert!(
                !text.contains("hmm"),
                "foreign thinking must not leak into text"
            ),
        }
    }

    #[test]
    fn unknown_target_keeps_history_unchanged() {
        let gateway = LlmGateway::new();
        gateway.register_profile(profile("a", "gpt-4o")).unwrap();
        let history = vec![thinking_message()];
        let moved = gateway.migrate_history_for_profile(&history, Some("a"), "missing");
        assert_eq!(moved, history);
    }

    #[test]
    fn unknown_source_assumes_a_cross_model_move() {
        let gateway = LlmGateway::new();
        gateway
            .register_profile(profile("b", "claude-sonnet"))
            .unwrap();
        let moved = gateway.migrate_history_for_profile(&[thinking_message()], None, "b");
        assert_eq!(moved.len(), 1);
        match &moved[0].content {
            MessageContentValue::Rich(blocks) => assert!(
                blocks
                    .iter()
                    .all(|b| !matches!(b, MessageContent::Thinking { .. })),
                "foreign thinking must not leak"
            ),
            MessageContentValue::Text(_) => {}
        }
    }
}
