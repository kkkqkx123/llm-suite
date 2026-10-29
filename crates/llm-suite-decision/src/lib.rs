//! Decision capability for the llm-suite group.

pub mod assets;
pub mod config;
pub mod error;
#[cfg(feature = "mock")]
pub mod mock;
#[cfg(feature = "runner")]
pub mod runner;
pub mod openai;
pub mod jev;
pub mod provider;
pub mod types;

pub use assets::{DecisionAsset, ImageAsset, Part, ResolvedImage, content_type_from_bytes, encode_multipart, reference_pattern, resolve_assets, resolve_assets_async, scan_references};
pub use config::DecisionConfig;
pub use error::{DecisionError, Result};
pub use openai::OpenAIDecisionProvider;
pub use jev::JevDecisionProvider;
pub use provider::DecisionProvider;
pub use types::{
    BoxCoordinates, ChoiceProbability, ChoiceValue, ContentPart, DecisionAnswer, DecisionMessage,
    DecisionModelMetadata, DecisionQuestion, DecisionRequest, DecisionResponse, DecisionState,
    DecisionUsage, MessageContent, MessageRole, NamedAnswer, NamedQuestion, ScoreLevel,
    ScoreProbability, validate_request,
};

#[cfg(feature = "mock")]
pub use mock::MockDecisionProvider;
#[cfg(feature = "runner")]
pub use runner::{
    AsyncDecisionCandidate, AsyncDecisionRunner, AsyncExecute, DecisionDiagnostic, DecisionRunResult,
    DecisionRunner, DecisionRunnerCancelledError, DecisionRunnerError, DecisionRunnerParams,
    DecisionRunnerReason, DiagnosticObserver, DiagnosticUsage, SyncDecisionCandidate, SyncExecute,
    SyncExecutorOutput,
};
