//! Synchronous and asynchronous Decision runners with diagnostics.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use crate::provider::DecisionProvider;
use crate::types::{ChoiceValue, DecisionAnswer, DecisionRequest, DecisionResponse, DecisionState, MessageContent, MessageRole, NamedQuestion};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionRunnerReason {
    InvalidParams,
    InvalidCandidates,
    InvalidAnswer,
    AsyncExecutor,
    InvalidVerification,
    VerificationFailed,
    StepLimit,
    Failed,
    Cancelled,
}

impl std::fmt::Display for DecisionRunnerReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            DecisionRunnerReason::InvalidParams => "invalid_params",
            DecisionRunnerReason::InvalidCandidates => "invalid_candidates",
            DecisionRunnerReason::InvalidAnswer => "invalid_answer",
            DecisionRunnerReason::AsyncExecutor => "async_executor",
            DecisionRunnerReason::InvalidVerification => "invalid_verification",
            DecisionRunnerReason::VerificationFailed => "verification_failed",
            DecisionRunnerReason::StepLimit => "step_limit",
            DecisionRunnerReason::Failed => "failed",
            DecisionRunnerReason::Cancelled => "cancelled",
        };
        write!(f, "{}", text)
    }
}

#[derive(Debug)]
pub struct DecisionRunnerError {
    pub reason: DecisionRunnerReason,
    pub attempted: Vec<String>,
    pub completed: Vec<String>,
}

impl std::fmt::Display for DecisionRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for DecisionRunnerError {}

#[derive(Debug)]
pub struct DecisionRunnerCancelledError {
    pub attempted: Vec<String>,
    pub completed: Vec<String>,
}

impl std::fmt::Display for DecisionRunnerCancelledError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cancelled")
    }
}

impl std::error::Error for DecisionRunnerCancelledError {}

#[derive(Debug, Clone)]
pub struct DiagnosticUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct DecisionDiagnostic {
    pub api: &'static str,
    pub stage: &'static str,
    pub reason: &'static str,
    pub call_id: Option<String>,
    pub member: Option<String>,
    pub duration_ms: f64,
    pub usage: Option<DiagnosticUsage>,
}

pub type DiagnosticObserver = Arc<dyn Fn(DecisionDiagnostic) + Send + Sync>;

#[derive(Debug, Clone)]
pub struct DecisionRunnerParams {
    pub model: String,
    pub input: String,
    pub safety_identifier: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DecisionRunResult {
    pub status: String,
    pub decisions: Vec<DecisionResponse>,
    pub attempted_candidate_ids: Vec<String>,
    pub completed_candidate_ids: Vec<String>,
}

pub enum SyncExecutorOutput {
    Completed,
    Awaitable(Pin<Box<dyn Future<Output = ()> + Send>>),
}

pub type SyncExecute = Arc<dyn Fn() -> SyncExecutorOutput + Send + Sync>;
pub type AsyncExecute = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

#[derive(Clone)]
pub enum SyncDecisionCandidate {
    Action { id: String, description: String, execute: SyncExecute },
    Done { id: String, description: String },
}

#[derive(Clone)]
pub enum AsyncDecisionCandidate {
    Action { id: String, description: String, execute: AsyncExecute },
    Done { id: String, description: String },
}

#[derive(Clone)]
struct Candidate {
    id: String,
    description: String,
    done: bool,
    sync_execute: Option<SyncExecute>,
    async_execute: Option<AsyncExecute>,
}

struct RunState {
    attempted: Vec<String>,
    completed: Vec<String>,
    decisions: Vec<DecisionResponse>,
    cancelled: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    params: DecisionRunnerParams,
    max_steps: usize,
}

impl RunState {
    fn new(params: DecisionRunnerParams, max_steps: usize, cancelled: Option<Arc<dyn Fn() -> bool + Send + Sync>>) -> Result<Self, DecisionRunnerError> {
        if max_steps < 1 {
            return Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidParams, attempted: vec![], completed: vec![] });
        }
        Ok(Self { attempted: Vec::new(), completed: Vec::new(), decisions: Vec::new(), cancelled, params, max_steps })
    }
    fn check_cancelled(&self) -> Result<(), DecisionRunnerError> {
        if let Some(cancelled) = &self.cancelled {
            if cancelled() {
                return Err(DecisionRunnerError { reason: DecisionRunnerReason::Cancelled, attempted: self.attempted.clone(), completed: self.completed.clone() });
            }
        }
        Ok(())
    }
    fn validate_candidates_sync(&self, values: Vec<SyncDecisionCandidate>) -> Result<Vec<Candidate>, DecisionRunnerError> {
        if !(1..=255).contains(&values.len()) { return Err(self.err_invalid_candidates()); }
        let mut snapshot = Vec::new();
        let mut ids = std::collections::HashSet::new();
        for value in values {
            match value {
                SyncDecisionCandidate::Action { id, description, execute } => {
                    if id.trim().is_empty() || description.trim().is_empty() { return Err(self.err_invalid_candidates()); }
                    if !ids.insert(id.clone()) { return Err(self.err_invalid_candidates()); }
                    snapshot.push(Candidate { id, description, done: false, sync_execute: Some(execute), async_execute: None });
                }
                SyncDecisionCandidate::Done { id, description } => {
                    if id.trim().is_empty() || description.trim().is_empty() { return Err(self.err_invalid_candidates()); }
                    if !ids.insert(id.clone()) { return Err(self.err_invalid_candidates()); }
                    snapshot.push(Candidate { id, description, done: true, sync_execute: None, async_execute: None });
                }
            }
        }
        if snapshot.iter().filter(|c| c.done).count() != 1 { return Err(self.err_invalid_candidates()); }
        Ok(snapshot)
    }
    fn validate_candidates_async(&self, values: Vec<AsyncDecisionCandidate>) -> Result<Vec<Candidate>, DecisionRunnerError> {
        if !(1..=255).contains(&values.len()) { return Err(self.err_invalid_candidates()); }
        let mut snapshot = Vec::new();
        let mut ids = std::collections::HashSet::new();
        for value in values {
            match value {
                AsyncDecisionCandidate::Action { id, description, execute } => {
                    if id.trim().is_empty() || description.trim().is_empty() { return Err(self.err_invalid_candidates()); }
                    if !ids.insert(id.clone()) { return Err(self.err_invalid_candidates()); }
                    snapshot.push(Candidate { id, description, done: false, sync_execute: None, async_execute: Some(execute) });
                }
                AsyncDecisionCandidate::Done { id, description } => {
                    if id.trim().is_empty() || description.trim().is_empty() { return Err(self.err_invalid_candidates()); }
                    if !ids.insert(id.clone()) { return Err(self.err_invalid_candidates()); }
                    snapshot.push(Candidate { id, description, done: true, sync_execute: None, async_execute: None });
                }
            }
        }
        if snapshot.iter().filter(|c| c.done).count() != 1 { return Err(self.err_invalid_candidates()); }
        Ok(snapshot)
    }
    fn err_invalid_candidates(&self) -> DecisionRunnerError {
        DecisionRunnerError { reason: DecisionRunnerReason::InvalidCandidates, attempted: self.attempted.clone(), completed: self.completed.clone() }
    }
    fn build_request(&self, observation: &str, candidates: &[Candidate]) -> DecisionRequest {
        let messages = vec![
            crate::types::DecisionMessage { role: MessageRole::User, content: MessageContent::Text(self.params.input.clone()), kind: "message".to_string(), phase: None },
            crate::types::DecisionMessage { role: MessageRole::User, content: MessageContent::Text(observation.to_string()), kind: "message".to_string(), phase: None },
        ];
        let question = crate::types::DecisionQuestion::Choice {
            name: Some("next_action".to_string()),
            instructions: Some(serde_json::Value::String("Select the supplied action that best advances the task, or done if the task is complete.".to_string())),
            choices: candidates.iter().map(|c| crate::types::ChoiceOption { value: ChoiceValue::Text(c.id.clone()), description: Some(serde_json::Value::String(c.description.clone())) }).collect(),
        };
        let named = NamedQuestion { name: Some("next_action".to_string()), question };
        DecisionRequest {
            model: Some(self.params.model.clone()),
            state: DecisionState::Messages(messages),
            questions: vec![named],
            prompt_cache_key: None,
            prompt_cache_options: None,
            safety_identifier: self.params.safety_identifier.clone(),
            assets: None,
        }
    }
}


fn emit(observer: &Option<DiagnosticObserver>, diag: DecisionDiagnostic) {
    if let Some(obs) = observer { obs(diag); }
}

pub struct DecisionRunner {
    pub observe: Arc<dyn Fn() -> String + Send + Sync>,
    pub candidates: Arc<dyn Fn(String) -> Vec<SyncDecisionCandidate> + Send + Sync>,
    pub verify: Option<Arc<dyn Fn(String) -> bool + Send + Sync>>,
}

impl DecisionRunner {
    pub fn new(
        observe: impl Fn() -> String + Send + Sync + 'static,
        candidates: impl Fn(String) -> Vec<SyncDecisionCandidate> + Send + Sync + 'static,
        verify: Option<impl Fn(String) -> bool + Send + Sync + 'static>,
    ) -> Self {
        Self {
            observe: Arc::new(observe),
            candidates: Arc::new(candidates),
            verify: verify.map(|v| Arc::new(v) as Arc<dyn Fn(String) -> bool + Send + Sync>),
        }
    }

    pub async fn run(
        &self,
        provider: &dyn DecisionProvider,
        params: DecisionRunnerParams,
        max_steps: usize,
        cancelled: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
        on_diagnostic: Option<DiagnosticObserver>,
    ) -> Result<DecisionRunResult, DecisionRunnerError> {
        let mut run = RunState::new(params, max_steps, cancelled)?;
        let observe = self.observe.clone();
        let candidate_provider = self.candidates.clone();
        let verify = self.verify.clone();

        for step in 0..run.max_steps {
            run.check_cancelled()?;
            let observation = (observe)();
            let candidates_input = (candidate_provider)(observation.clone());
            let candidates = match run.validate_candidates_sync(candidates_input) { Ok(c) => c, Err(e) => return Err(e) };
            let only_done = candidates.len() == 1;
            let mut choice: Option<Candidate> = None;
            if only_done {
                choice = Some(candidates[0].clone());
            } else {
                let started = Instant::now();
                let request = run.build_request(&observation, &candidates);
                match provider.create(&request).await {
                    Ok(decision) => {
                        match select_candidate(&mut run, &decision, &candidates) {
                            Ok((c, usage)) => {
                                let diag_usage = usage.map(|u| DiagnosticUsage { input_tokens: u.0, output_tokens: u.1, total_tokens: u.2 });
                                emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "request", reason: "completed", call_id: None, member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: diag_usage });
                                choice = c;
                            }
                            Err(e) => {
                                emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "request", reason: "api_failed", call_id: None, member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                                return Err(e);
                            }
                        }
                    }
                    Err(_) => {
                        emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "request", reason: "api_failed", call_id: None, member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                        return Err(DecisionRunnerError { reason: DecisionRunnerReason::Failed, attempted: run.attempted.clone(), completed: run.completed.clone() });
                    }
                }
                run.check_cancelled()?;
            }

            emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "selection", reason: if choice.is_none() { "refused" } else { "completed" }, call_id: choice.as_ref().map(|c| c.id.clone()), member: None, duration_ms: 0.0, usage: None });

            if choice.is_none() {
                return Ok(DecisionRunResult { status: "refused".to_string(), decisions: run.decisions.clone(), attempted_candidate_ids: run.attempted.clone(), completed_candidate_ids: run.completed.clone() });
            }
            let choice = choice.unwrap();
            if choice.done {
                if verify.is_none() {
                    return Ok(DecisionRunResult { status: "unverified".to_string(), decisions: run.decisions.clone(), attempted_candidate_ids: run.attempted.clone(), completed_candidate_ids: run.completed.clone() });
                }
                run.check_cancelled()?;
                let fresh_obs = (observe)();
                let started = Instant::now();
                let verification = (verify.as_ref().unwrap())(fresh_obs);
                emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "verification", reason: if verification { "completed" } else { "refused" }, call_id: Some(choice.id.clone()), member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                if verification {
                    return Ok(DecisionRunResult { status: "verified".to_string(), decisions: run.decisions.clone(), attempted_candidate_ids: run.attempted.clone(), completed_candidate_ids: run.completed.clone() });
                }
                if only_done {
                    return Err(DecisionRunnerError { reason: DecisionRunnerReason::VerificationFailed, attempted: run.attempted.clone(), completed: run.completed.clone() });
                }
                if step == run.max_steps - 1 {
                    return Err(DecisionRunnerError { reason: DecisionRunnerReason::StepLimit, attempted: run.attempted.clone(), completed: run.completed.clone() });
                }
                continue;
            }

            if step == run.max_steps - 1 {
                return Err(DecisionRunnerError { reason: DecisionRunnerReason::StepLimit, attempted: run.attempted.clone(), completed: run.completed.clone() });
            }
            if choice.sync_execute.is_none() {
                return Err(DecisionRunnerError { reason: DecisionRunnerReason::Failed, attempted: run.attempted.clone(), completed: run.completed.clone() });
            }
            run.check_cancelled()?;
            run.attempted.push(choice.id.clone());
            let started = Instant::now();
            let exec = choice.sync_execute.clone().unwrap();
            match exec() {
                SyncExecutorOutput::Awaitable(_) => {
                    emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "action", reason: "driver_failed", call_id: Some(choice.id.clone()), member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                    return Err(DecisionRunnerError { reason: DecisionRunnerReason::AsyncExecutor, attempted: run.attempted.clone(), completed: run.completed.clone() });
                }
                SyncExecutorOutput::Completed => {}
            }
            emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "action", reason: "completed", call_id: Some(choice.id.clone()), member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
            run.completed.push(choice.id.clone());
            run.check_cancelled()?;
        }

        Err(DecisionRunnerError { reason: DecisionRunnerReason::StepLimit, attempted: run.attempted.clone(), completed: run.completed.clone() })
    }
}

pub struct AsyncDecisionRunner {
    pub observe: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>,
    pub candidates: Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Vec<AsyncDecisionCandidate>> + Send>> + Send + Sync>,
    pub verify: Option<Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>>,
}

impl AsyncDecisionRunner {
    pub fn new(
        observe: impl Fn() -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync + 'static,
        candidates: impl Fn(String) -> Pin<Box<dyn Future<Output = Vec<AsyncDecisionCandidate>> + Send>> + Send + Sync + 'static,
        verify: Option<impl Fn(String) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync + 'static>,
    ) -> Self {
        Self {
            observe: Arc::new(observe),
            candidates: Arc::new(candidates),
            verify: verify.map(|v| Arc::new(v) as Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>),
        }
    }

    pub async fn run(
        &self,
        provider: &dyn DecisionProvider,
        params: DecisionRunnerParams,
        max_steps: usize,
        cancelled: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
        on_diagnostic: Option<DiagnosticObserver>,
    ) -> Result<DecisionRunResult, DecisionRunnerError> {
        let mut run = RunState::new(params, max_steps, cancelled)?;
        let observe = self.observe.clone();
        let candidate_provider = self.candidates.clone();
        let verify = self.verify.clone();

        for step in 0..run.max_steps {
            run.check_cancelled()?;
            let observation = (observe)().await;
            let candidates_input = (candidate_provider)(observation.clone()).await;
            let candidates = match run.validate_candidates_async(candidates_input) { Ok(c) => c, Err(e) => return Err(e) };
            let only_done = candidates.len() == 1;
            let mut choice: Option<Candidate> = None;
            if only_done {
                choice = Some(candidates[0].clone());
            } else {
                let started = Instant::now();
                let request = run.build_request(&observation, &candidates);
                match provider.create(&request).await {
                    Ok(decision) => {
                        match select_candidate(&mut run, &decision, &candidates) {
                            Ok((c, usage)) => {
                                let diag_usage = usage.map(|u| DiagnosticUsage { input_tokens: u.0, output_tokens: u.1, total_tokens: u.2 });
                                emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "request", reason: "completed", call_id: None, member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: diag_usage });
                                choice = c;
                            }
                            Err(e) => {
                                emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "request", reason: "api_failed", call_id: None, member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                                return Err(e);
                            }
                        }
                    }
                    Err(_) => {
                        emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "request", reason: "api_failed", call_id: None, member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                        return Err(DecisionRunnerError { reason: DecisionRunnerReason::Failed, attempted: run.attempted.clone(), completed: run.completed.clone() });
                    }
                }
                run.check_cancelled()?;
            }

            emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "selection", reason: if choice.is_none() { "refused" } else { "completed" }, call_id: choice.as_ref().map(|c| c.id.clone()), member: None, duration_ms: 0.0, usage: None });

            if choice.is_none() {
                return Ok(DecisionRunResult { status: "refused".to_string(), decisions: run.decisions.clone(), attempted_candidate_ids: run.attempted.clone(), completed_candidate_ids: run.completed.clone() });
            }
            let choice = choice.unwrap();
            if choice.done {
                if verify.is_none() {
                    return Ok(DecisionRunResult { status: "unverified".to_string(), decisions: run.decisions.clone(), attempted_candidate_ids: run.attempted.clone(), completed_candidate_ids: run.completed.clone() });
                }
                run.check_cancelled()?;
                let fresh_obs = (observe)().await;
                let started = Instant::now();
                let verification = (verify.as_ref().unwrap())(fresh_obs).await;
                emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "verification", reason: if verification { "completed" } else { "refused" }, call_id: Some(choice.id.clone()), member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
                if verification {
                    return Ok(DecisionRunResult { status: "verified".to_string(), decisions: run.decisions.clone(), attempted_candidate_ids: run.attempted.clone(), completed_candidate_ids: run.completed.clone() });
                }
                if only_done {
                    return Err(DecisionRunnerError { reason: DecisionRunnerReason::VerificationFailed, attempted: run.attempted.clone(), completed: run.completed.clone() });
                }
                if step == run.max_steps - 1 {
                    return Err(DecisionRunnerError { reason: DecisionRunnerReason::StepLimit, attempted: run.attempted.clone(), completed: run.completed.clone() });
                }
                continue;
            }

            if step == run.max_steps - 1 {
                return Err(DecisionRunnerError { reason: DecisionRunnerReason::StepLimit, attempted: run.attempted.clone(), completed: run.completed.clone() });
            }
            if choice.async_execute.is_none() {
                return Err(DecisionRunnerError { reason: DecisionRunnerReason::Failed, attempted: run.attempted.clone(), completed: run.completed.clone() });
            }
            run.check_cancelled()?;
            run.attempted.push(choice.id.clone());
            let started = Instant::now();
            let exec = choice.async_execute.clone().unwrap();
            (exec)().await;
            emit(&on_diagnostic, DecisionDiagnostic { api: "decisions", stage: "action", reason: "completed", call_id: Some(choice.id.clone()), member: None, duration_ms: started.elapsed().as_secs_f64() * 1000.0, usage: None });
            run.completed.push(choice.id.clone());
            run.check_cancelled()?;
        }

        Err(DecisionRunnerError { reason: DecisionRunnerReason::StepLimit, attempted: run.attempted.clone(), completed: run.completed.clone() })
    }
}

fn select_candidate(
    run: &mut RunState,
    decision: &DecisionResponse,
    candidates: &[Candidate],
) -> Result<(Option<Candidate>, Option<(u64, u64, u64)>), DecisionRunnerError> {
    run.decisions.push(decision.clone());
    if decision.answers.len() != 1 {
        return Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidAnswer, attempted: run.attempted.clone(), completed: run.completed.clone() });
    }
    let answer = &decision.answers[0];
    if answer.name.as_deref() != Some("next_action") {
        return Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidAnswer, attempted: run.attempted.clone(), completed: run.completed.clone() });
    }
    let usage_tuple = (decision.usage.input_tokens.unwrap_or(0), decision.usage.output_tokens.unwrap_or(0), decision.usage.total_tokens.unwrap_or(0));
    match &answer.answer {
        DecisionAnswer::Refusal { .. } => Ok((None, Some(usage_tuple))),
        DecisionAnswer::Choice { choice, confidence, probabilities, .. } => {
            if !confidence.is_finite() { return Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidAnswer, attempted: run.attempted.clone(), completed: run.completed.clone() }); }
            for p in probabilities { if !p.probability.is_finite() { return Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidAnswer, attempted: run.attempted.clone(), completed: run.completed.clone() }); } }
            let choice_text = match choice { ChoiceValue::Text(t) => t.clone(), ChoiceValue::Bool(b) => b.to_string() };
            for c in candidates {
                if c.id == choice_text { return Ok((Some(c.clone()), Some(usage_tuple))); }
            }
            Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidAnswer, attempted: run.attempted.clone(), completed: run.completed.clone() })
        }
        _ => Err(DecisionRunnerError { reason: DecisionRunnerReason::InvalidAnswer, attempted: run.attempted.clone(), completed: run.completed.clone() }),
    }
}
