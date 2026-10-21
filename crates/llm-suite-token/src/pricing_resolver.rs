//! Host-side pricing resolution for token cost computation.
//!
//! The gateway records token usage and needs a `(provider, model) -> price`
//! lookup to turn raw usage into cost. That lookup is owned by the host crate
//! (a static LiteLLM table plus a storage-backed custom table), so the trait
//! is defined here in `llm-suite-token`. This keeps the gateway dependent only
//! on this crate rather than on any host workspace.

use crate::pricing::{CacheTokenScope, TokenPricing};

/// A resolved price for one model, ready to feed [`compute_cost`](crate::pricing::compute_cost).
#[derive(Debug, Clone)]
pub struct ResolvedPricing {
    pub pricing: TokenPricing,
    pub scope: CacheTokenScope,
}

/// Maps a `(provider, model)` pair to its pricing.
///
/// Implemented by host-side resolvers. The gateway depends only on this trait
/// (and on `llm-suite-token` in general), so the actual table wiring stays in
/// the host workspace and the submodule remains reusable on its own.
pub trait PricingResolver: Send + Sync {
    /// Resolve pricing for the given provider/model.
    ///
    /// Returns `None` when no price is known, in which case the gateway records
    /// usage without attaching a cost. `provider` may be `None` when the caller
    /// has no provider context.
    fn resolve(&self, provider: Option<&str>, model: &str) -> Option<ResolvedPricing>;
}
