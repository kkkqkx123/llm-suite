//! Decision capability for the llm-suite group.

pub mod assets;
pub mod config;
pub mod error;
pub mod jev;
#[cfg(feature = "mock")]
pub mod mock;
pub mod openai;
pub mod provider;
#[cfg(feature = "runner")]
pub mod runner;
pub mod types;

pub use assets::{
    content_type_from_bytes, encode_multipart, reference_pattern, resolve_assets,
    resolve_assets_async, scan_references, DecisionAsset, ImageAsset, Part, ResolvedImage,
};
pub use config::DecisionConfig;
pub use error::{DecisionError, Result};
pub use jev::JevDecisionProvider;
pub use openai::OpenAIDecisionProvider;
pub use provider::DecisionProvider;
pub use types::{
    validate_request, BoxCoordinates, ChoiceProbability, ChoiceValue, ContentPart, DecisionAnswer,
    DecisionMessage, DecisionModelMetadata, DecisionQuestion, DecisionRequest, DecisionResponse,
    DecisionState, DecisionUsage, MessageContent, MessageRole, NamedAnswer, NamedQuestion,
    ScoreLevel, ScoreProbability,
};

#[cfg(feature = "mock")]
pub use mock::MockDecisionProvider;
#[cfg(feature = "runner")]
pub use runner::{
    AsyncDecisionCandidate, AsyncDecisionRunner, AsyncExecute, DecisionDiagnostic,
    DecisionRunResult, DecisionRunner, DecisionRunnerCancelledError, DecisionRunnerError,
    DecisionRunnerParams, DecisionRunnerReason, DiagnosticObserver, DiagnosticUsage,
    SyncDecisionCandidate, SyncExecute, SyncExecutorOutput,
};
