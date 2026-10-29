// Configuration layer: profile and provider registries plus request
// merging and profile-side catalog helpers.
//
// Network discovery (`ModelCatalog`, endpoint joining, extraction,
// credential filtering) lives in `llm-suite-discovery` so hosts that only
// validate profiles and merge requests never pay for the TLS stack.
pub mod catalog;
pub mod merge;
pub mod profile;
pub mod provider;

pub use catalog::{apply_catalog_metadata, catalog_entry_for_profile};
pub use merge::merge_request;
pub use profile::ProfileManager;
pub use provider::{apply_provider_defaults, ProviderDefinitionRegistry};
