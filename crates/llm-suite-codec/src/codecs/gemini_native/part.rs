//! Reading of Gemini `Part` objects.
//!
//! The streaming and the non-streaming path read the same part schema, so
//! the wire knowledge lives here once: the `thought` flag, the text, the
//! function call and the thought signature that belongs to it.

/// A Gemini content part, normalized into the shape the codec reports.
#[derive(Debug, Clone)]
pub enum Part {
    /// A reasoning part: the official `thought` flag was set on it.
    Thought { text: String },
    /// A plain text part.
    Text { text: String },
    /// A function call part.
    FunctionCall {
        name: String,
        /// The call arguments as a JSON string.
        arguments: String,
        /// The call id the provider assigned, when it assigned one.
        provider_id: Option<String>,
        /// The thought signature that belongs to this call.
        signature: Option<String>,
    },
}

/// Read the parts of a candidate.
///
/// The thought signature may ride on the function call itself or on the
/// thought part in front of it; the pending signature covers the second
/// case.
pub fn read_parts(parts: &[serde_json::Value]) -> Vec<Part> {
    let mut pending_signature: Option<String> = None;
    let mut result: Vec<Part> = Vec::new();

    for part in parts {
        let signature = part
            .get("thoughtSignature")
            .and_then(|v| v.as_str())
            .map(String::from);
        if let Some(signature) = &signature {
            pending_signature = Some(signature.clone());
        }
        // `thought` is a boolean flag on the part (official schema:
        // Part.thought: bool); a part like {"text": "...", "thought": true}
        // is reasoning.
        let is_thought = part
            .get("thought")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
            if is_thought {
                result.push(Part::Thought {
                    text: text.to_string(),
                });
            } else {
                result.push(Part::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(func_call) = part.get("functionCall") {
            let name = func_call
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = func_call
                .get("args")
                .cloned()
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            let arguments = serde_json::to_string(&args).unwrap_or_default();
            // The provider assigns the call id; it has to be echoed back in
            // the matching function response.
            let provider_id = func_call
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from);
            result.push(Part::FunctionCall {
                name,
                arguments,
                provider_id,
                signature: signature.or_else(|| pending_signature.clone()),
            });
        }
    }

    result
}
