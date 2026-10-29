use serde::{Deserialize, Serialize};

/// What a model is built for. Only `Chat` travels the gateway hot path
/// today; `Image` / `Classifier` reserve the catalog shape for the later
/// image-generation phase without widening the current request types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    #[default]
    Chat,
    Image,
    Classifier,
}

/// Input modalities a model accepts. Used by the cross-provider message
/// transform to decide whether image blocks survive the move.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InputModality {
    Text,
    Image,
}

/// One pricing tier: when the request input volume falls at or below
/// `up_to_tokens`, the tier rates apply to the whole request. `None` is
/// the catch-all tier above every bounded tier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCostTier {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up_to_tokens: Option<u64>,
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

/// Per-million-token prices of one model entry in the catalog.
///
/// Currency-agnostic: callers decide whether the numbers are USD, CNY or
/// any other unit. `tiers` is empty for flat pricing; when present the
/// first tier whose `up_to_tokens` covers the request input wins, and a
/// `None` tier is the catch-all. Tier selection math lives in the token
/// crate (`TokenPricing::rates_for` via `TokenPricing::from_model_cost`);
/// this struct carries data only so the leaf stays dependency-free.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCost {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

/// Static catalog entry describing one model.
///
/// Structural metadata only (identity, shape, capabilities, price table).
/// Provider compat that affects request encoding lives in
/// [`super::ProviderCompat`], not here, so this struct stays a pure
/// data leaf with no codec dependency beyond the format marker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    pub id: String,
    pub provider_id: String,
    #[serde(default)]
    pub model_type: ModelType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<super::LlmFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub input_modalities: Vec<InputModality>,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub thinking_levels: Vec<super::generation::ThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
}

impl ModelMetadata {
    /// Whether the model accepts image input.
    pub fn supports_image(&self) -> bool {
        self.input_modalities.contains(&InputModality::Image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_support_follows_modalities() {
        let mut meta = ModelMetadata {
            id: "m".to_string(),
            provider_id: "p".to_string(),
            model_type: ModelType::Chat,
            format: None,
            context_window: None,
            max_output_tokens: None,
            input_modalities: vec![InputModality::Text],
            reasoning: false,
            thinking_levels: Vec::new(),
            cost: None,
        };
        assert!(!meta.supports_image());
        meta.input_modalities.push(InputModality::Image);
        assert!(meta.supports_image());
    }
}
