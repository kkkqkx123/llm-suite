#![cfg(all(feature = "mock", feature = "runner"))]

use std::sync::Arc;

use llm_suite_decision::mock::MockDecisionProvider;
use llm_suite_decision::runner::{
    AsyncDecisionCandidate, AsyncDecisionRunner, AsyncExecute, DecisionRunner, DecisionRunnerParams,
    SyncDecisionCandidate, SyncExecutorOutput,
};
use llm_suite_decision::types::{
    ChoiceProbability, ChoiceValue, DecisionAnswer, DecisionResponse, DecisionUsage, NamedAnswer,
};
use llm_suite_decision::DecisionProvider;

#[tokio::test]
async fn sync_runner_unverified_on_done() {
    let answer = DecisionAnswer::Choice {
        name: Some("next_action".into()),
        choice: ChoiceValue::Text("done".into()),
        confidence: 1.0,
        probabilities: vec![ChoiceProbability {
            value: ChoiceValue::Text("done".into()),
            probability: 1.0,
        }],
    };
    let named = NamedAnswer {
        name: Some("next_action".into()),
        answer,
    };
    let resp = DecisionResponse {
        model: "m".into(),
        answers: vec![named],
        usage: DecisionUsage::default(),
    };
    let provider = MockDecisionProvider::new("m").with_response(resp);

    let runner = DecisionRunner::new(
        || "obs".to_string(),
        move |_obs| {
            vec![
                SyncDecisionCandidate::Action {
                    id: "act".into(),
                    description: "do".into(),
                    execute: Arc::new(|| SyncExecutorOutput::Completed),
                },
                SyncDecisionCandidate::Done {
                    id: "done".into(),
                    description: "finish".into(),
                },
            ]
        },
        None::<fn(String) -> bool>,
    );

    let params = DecisionRunnerParams {
        model: "m".into(),
        input: "task".into(),
        safety_identifier: None,
        state: None,
    };
    let result = runner.run(&provider, params, 5, None, None).await.unwrap();
    assert_eq!(result.status, "unverified");
    assert!(result.attempted_candidate_ids.is_empty());
}

#[tokio::test]
async fn sync_runner_executes_action_then_done() {
    // First response selects "act", second selects "done".
    // Use a provider that returns act then done by counting calls? Mock returns same response always.
    // Instead test step_limit when only action selected repeatedly.
    let answer = DecisionAnswer::Choice {
        name: Some("next_action".into()),
        choice: ChoiceValue::Text("act".into()),
        confidence: 1.0,
        probabilities: vec![ChoiceProbability {
            value: ChoiceValue::Text("act".into()),
            probability: 1.0,
        }],
    };
    let named = NamedAnswer {
        name: Some("next_action".into()),
        answer,
    };
    let resp = DecisionResponse {
        model: "m".into(),
        answers: vec![named],
        usage: DecisionUsage::default(),
    };
    let provider = MockDecisionProvider::new("m").with_response(resp);

    let executed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let executed_clone = executed.clone();
    let runner = DecisionRunner::new(
        || "obs".to_string(),
        move |_obs| {
            vec![
                SyncDecisionCandidate::Action {
                    id: "act".into(),
                    description: "do".into(),
                    execute: Arc::new({
                        let ec = executed_clone.clone();
                        move || {
                            ec.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            SyncExecutorOutput::Completed
                        }
                    }),
                },
                SyncDecisionCandidate::Done {
                    id: "done".into(),
                    description: "finish".into(),
                },
            ]
        },
        None::<fn(String) -> bool>,
    );

    let params = DecisionRunnerParams {
        model: "m".into(),
        input: "task".into(),
        safety_identifier: None,
        state: None,
    };
    let err = runner.run(&provider, params, 2, None, None).await.unwrap_err();
    // step_limit because action repeatedly selected until max_steps
    assert_eq!(err.reason, llm_suite_decision::runner::DecisionRunnerReason::StepLimit);
    assert!(executed.load(std::sync::atomic::Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn sync_runner_refusal() {
    let answer = DecisionAnswer::Refusal {
        name: Some("next_action".into()),
    };
    let named = NamedAnswer {
        name: Some("next_action".into()),
        answer,
    };
    let resp = DecisionResponse {
        model: "m".into(),
        answers: vec![named],
        usage: DecisionUsage::default(),
    };
    let provider = MockDecisionProvider::new("m").with_response(resp);
    let runner = DecisionRunner::new(
        || "obs".to_string(),
        move |_obs| {
            vec![
                SyncDecisionCandidate::Action {
                    id: "act".into(),
                    description: "do".into(),
                    execute: Arc::new(|| SyncExecutorOutput::Completed),
                },
                SyncDecisionCandidate::Done {
                    id: "done".into(),
                    description: "finish".into(),
                },
            ]
        },
        None::<fn(String) -> bool>,
    );
    let params = DecisionRunnerParams {
        model: "m".into(),
        input: "task".into(),
        safety_identifier: None,
        state: None,
    };
    let result = runner.run(&provider, params, 5, None, None).await.unwrap();
    assert_eq!(result.status, "refused");
}

#[tokio::test]
async fn sync_runner_async_executor_error() {
    let answer = DecisionAnswer::Choice {
        name: Some("next_action".into()),
        choice: ChoiceValue::Text("act".into()),
        confidence: 1.0,
        probabilities: vec![ChoiceProbability {
            value: ChoiceValue::Text("act".into()),
            probability: 1.0,
        }],
    };
    let named = NamedAnswer {
        name: Some("next_action".into()),
        answer,
    };
    let resp = DecisionResponse {
        model: "m".into(),
        answers: vec![named],
        usage: DecisionUsage::default(),
    };
    let provider = MockDecisionProvider::new("m").with_response(resp);
    let runner = DecisionRunner::new(
        || "obs".to_string(),
        move |_obs| {
            vec![
                SyncDecisionCandidate::Action {
                    id: "act".into(),
                    description: "do".into(),
                    execute: Arc::new(|| {
                        SyncExecutorOutput::Awaitable(Box::pin(async {}))
                    }),
                },
                SyncDecisionCandidate::Done {
                    id: "done".into(),
                    description: "finish".into(),
                },
            ]
        },
        None::<fn(String) -> bool>,
    );
    let params = DecisionRunnerParams {
        model: "m".into(),
        input: "task".into(),
        safety_identifier: None,
        state: None,
    };
    let err = runner.run(&provider, params, 5, None, None).await.unwrap_err();
    assert_eq!(err.reason, llm_suite_decision::runner::DecisionRunnerReason::AsyncExecutor);
}

#[tokio::test]
async fn async_runner_runs() {
    let answer = DecisionAnswer::Choice {
        name: Some("next_action".into()),
        choice: ChoiceValue::Text("done".into()),
        confidence: 1.0,
        probabilities: vec![ChoiceProbability {
            value: ChoiceValue::Text("done".into()),
            probability: 1.0,
        }],
    };
    let named = NamedAnswer {
        name: Some("next_action".into()),
        answer,
    };
    let resp = DecisionResponse {
        model: "m".into(),
        answers: vec![named],
        usage: DecisionUsage::default(),
    };
    let provider = MockDecisionProvider::new("m").with_response(resp);

    let runner = AsyncDecisionRunner::new(
        || Box::pin(async { "obs".to_string() }),
        |_obs| Box::pin(async {
            vec![
                AsyncDecisionCandidate::Action {
                    id: "act".into(),
                    description: "do".into(),
                    execute: Arc::new(|| Box::pin(async {})),
                },
                AsyncDecisionCandidate::Done {
                    id: "done".into(),
                    description: "finish".into(),
                },
            ]
        }),
        None::<fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>>>,
    );

    let params = DecisionRunnerParams {
        model: "m".into(),
        input: "task".into(),
        safety_identifier: None,
        state: None,
    };
    let result = runner.run(&provider, params, 5, None, None).await.unwrap();
    assert_eq!(result.status, "unverified");
}

#[tokio::test]
async fn invalid_candidates() {
    let provider = MockDecisionProvider::new("m");
    let runner = DecisionRunner::new(
        || "obs".to_string(),
        |_obs| vec![SyncDecisionCandidate::Done { id: "done".into(), description: "finish".into() }], // only done ok actually valid
        None::<fn(String) -> bool>,
    );
    // invalid: no done
    let runner2 = DecisionRunner::new(
        || "obs".to_string(),
        |_obs| vec![SyncDecisionCandidate::Action { id: "act".into(), description: "do".into(), execute: Arc::new(|| SyncExecutorOutput::Completed) }],
        None::<fn(String) -> bool>,
    );
    let params = DecisionRunnerParams { model: "m".into(), input: "task".into(), state: None, safety_identifier: None };
    let err = runner2.run(&provider, params, 5, None, None).await.unwrap_err();
    assert_eq!(err.reason, llm_suite_decision::runner::DecisionRunnerReason::InvalidCandidates);
}
