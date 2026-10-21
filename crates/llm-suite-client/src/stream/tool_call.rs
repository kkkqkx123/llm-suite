//! The tool calls of a turn, accumulated across the stream events.

use std::collections::BTreeMap;

use llm_suite_types::llm::MessageStreamToolCallDelta;
use llm_suite_types::message::{LlmFunctionCall, LlmToolCall};

/// A tool call that is still being assembled.
#[derive(Debug, Clone)]
pub struct PartialToolCall {
    pub id: Option<String>,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
    pub index: usize,
    /// Provider-signed payload attached to the call (Gemini thought
    /// signature); it has to travel back with the call in the next turn.
    pub thought_signature: Option<String>,
}

/// The tool calls of a turn, keyed by the index the provider used.
///
/// The Anthropic path announces a call with a `ToolCallDelta` and delivers
/// its arguments as `InputJson` fragments; both land in the same entry, so
/// the two event kinds share this state.
#[derive(Debug, Default)]
pub struct ToolCallState {
    calls: BTreeMap<usize, PartialToolCall>,
}

impl ToolCallState {
    /// Append an argument fragment to the call at `index` and return the
    /// accumulated arguments.
    pub fn append_arguments(&mut self, index: usize, fragment: &str) -> String {
        let partial = self.entry(index);
        partial.arguments.push_str(fragment);
        partial.arguments.clone()
    }

    /// Apply a tool-call delta to the call at its index.
    pub fn apply_delta(&mut self, delta: &MessageStreamToolCallDelta) {
        let partial = self.entry(delta.index);
        if let Some(call_id) = &delta.call_id {
            partial.call_id = Some(call_id.clone());
        }
        if let Some(signature) = &delta.thought_signature {
            partial.thought_signature = Some(signature.clone());
        }
        if let Some(id) = &delta.id {
            partial.id = Some(id.clone());
        }
        if let Some(name) = &delta.name {
            partial.name = Some(name.clone());
        }
        match &delta.arguments {
            // A complete snapshot (Gemini functionCall, OpenAI response
            // function_call.done) replaces the accumulated fragments.
            Some(args) if delta.is_snapshot => partial.arguments = args.clone(),
            Some(args) => partial.arguments.push_str(args),
            None => {}
        }
    }

    /// Assemble the tool calls of the turn together with the provider
    /// signature of every call, in the order the calls are reported.
    pub fn assemble(&self) -> (Vec<LlmToolCall>, Vec<Option<String>>) {
        let mut calls = Vec::new();
        let mut signatures = Vec::new();
        for partial in self.calls.values() {
            let Some(name) = partial.name.clone() else {
                continue;
            };
            let id = partial.id.clone().unwrap_or_else(|| {
                format!("call_{}_{}", llm_suite_common::time::now(), partial.index)
            });
            let arguments = if partial.arguments.is_empty() {
                "{}".to_string()
            } else {
                partial.arguments.clone()
            };
            calls.push(LlmToolCall {
                id,
                r#type: "function".to_string(),
                call_id: partial.call_id.clone(),
                function: LlmFunctionCall { name, arguments },
            });
            signatures.push(partial.thought_signature.clone());
        }
        (calls, signatures)
    }

    fn entry(&mut self, index: usize) -> &mut PartialToolCall {
        self.calls.entry(index).or_insert_with(|| PartialToolCall {
            id: None,
            call_id: None,
            name: None,
            arguments: String::new(),
            index,
            thought_signature: None,
        })
    }
}
