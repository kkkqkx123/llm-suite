# llm-suite

A modular Rust toolkit that turns "call an LLM" from scattered ad-hoc HTTP
code into a layered, provider-agnostic stack: protocol types, wire codecs,
transport, capability clients, configuration and a single orchestration
gateway.

It is developed as a workspace of 12 small crates under `crates/`, consumed
by host projects (e.g. `wf-agent`) via a git submodule.

## Architecture

The crates form a strict DAG from protocol leaves up to the gateway:

```
                ┌──────────────┐
                │ llm-suite-gateway  │   orchestration entry point
                └──────┬───────┘
        ┌──────────────┼────────────────┬───────────────┐
┌───────┴──────┐ ┌─────┴───────┐ ┌──────┴──────┐ ┌──────┴───────┐
│ llm-suite-chat-basic│ │ llm-suite-embedding│ │ llm-suite-rerank  │ │  llm-suite-client  │
│ (chat client) │ │ (embeddings) │ │  (rerank)   │ │  (transport) │
└───────┬──────┘ └─────────────┘ └─────────────┘ └──────┬───────┘
        │                       ┌────────────────────────┤
┌───────┴──────┐        ┌───────┴──────┐          ┌──────┴──────┐
│  llm-suite-codec   │        │ llm-suite-tool-call│          │ llm-suite-config  │
│ (wire codecs)│        │ (tool parsing)│         │(profiles...)│
└───────┬──────┘        └──────┬───────┘          └──────┬──────┘
        │          ┌───────────┴──────────┐              │
┌───────┴──────┐   │                      │              │
│ llm-suite-message  │───┤ llm-suite-types  llm-suite-token │──────────────┤
│ (transforms) │   │ (protocol leaves)    │              │
└──────────────┘   └──────────────────────┘   ┌──────────┴────┐
                                              │  llm-suite-common   │
                                              │(runtime utils)│
                                              └───────────────┘
```

## Crates

| Crate | Responsibility |
|---|---|
| `llm-suite-common` | Shared runtime utilities independent of any host: poisoned-lock recovery, wall-clock time, id generation, retry helpers, exec helpers. |
| `llm-suite-types` | Dependency-free protocol leaf: chat messages, LLM request/response envelopes, wire tool declarations, parameter schemas and usage models shared by every layer. |
| `llm-suite-message` | Pure message transforms: fluent message builder, conversation-history conversion (e.g. to provider wire format), text extraction and history text helpers. |
| `llm-suite-token` | Token sizing: pure text estimation plus provider-shaped request counting (messages, tool declarations, images) for budgeting and context-window checks. |
| `llm-suite-codec` | Wire protocol layer: the `LlmCodec` trait, built-in codecs (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Gemini native), generation-parameter mapping and a runtime codec registry. Adding a new provider protocol means implementing one trait and registering it. |
| `llm-suite-tool-call` | Tool-call protocol parsing, gated behind feature flags so callers only compile what they use. Includes a streaming partial-JSON parser for incremental tool-call argument assembly. |
| `llm-suite-client` | Transport layer: an async client trait plus a reqwest-backed HTTP implementation (SSE streaming supported), stream accumulation, a dead-loop guard for pathological streams, a generic usage sink, and feature-gated scripted/TCP mocks for testing. |
| `llm-suite-chat-basic` | A minimal OpenAI-compatible `/chat/completions` client: request building, retry with backoff, client-side rate limiting, and both blocking and streaming calls. Intentionally codec-free for simple use cases. |
| `llm-suite-embedding` | Embedding capability: a provider trait plus an OpenAI-compatible HTTP implementation covering OpenAI, Gemini, Azure and Ollama endpoints, with a text preprocessing pipeline. |
| `llm-suite-decision` | Decision capability: OpenAI `/decisions` and Jev/TypeSafe `/v1/systemone` clients, shared question/answer types, config, mock provider. |
| `llm-suite-config` | Configuration layer: provider definition registry, named profiles, model catalog (loaded from JSON) and request merging that applies profile/provider/model defaults onto outgoing requests. |
| `llm-suite-gateway` | The orchestration entry point (`LlmGateway`): mandatory profile resolution, request merging, client caching, mock routing for tests, and metrics hooks. Host code talks only to this crate. |

## How it works

A request flows through four stages:

1. **Configure** — the host registers provider definitions (endpoint, auth,
   default model) and named profiles; a model catalog maps model ids to
   capabilities and limits.
2. **Resolve** — callers ask the `LlmGateway` for a client by profile name.
   The gateway resolves the profile, merges defaults onto the request
   (`llm-suite-config::merge_request`), selects the wire codec for the provider
   (`llm-suite-codec` registry) and returns a cached client.
3. **Encode & transport** — the codec serializes the `llm-suite-types` request
   envelope into the provider's wire format; `llm-suite-client` performs the HTTP
   call (or SSE stream), applying retry/rate-limit policy, and parses the
   response back through the codec.
4. **Post-process** — tool calls are incrementally reassembled by
   `llm-suite-tool-call`'s partial-JSON parser; usage is fed into the token sink
   (`llm-suite-token`); embeddings and reranks go through their capability crates
   with the same profile/config machinery.

Design principles:

- **Strict DAG, no cycles**: protocol leaves (`llm-suite-types`, `llm-suite-common`) know
  nothing about transport or providers; the gateway is the only crate that
  wires everything together.
- **Provider-agnostic core**: provider differences live in codecs and
  provider trait implementations, not in business logic.
- **Feature-gated extras**: tool-call protocol parsers and test mocks are
  behind cargo features so production builds stay lean.
- **Host-independent**: no crate depends on the host application; everything
  is reusable outside `wf-agent`.

## Usage

Add the crates you need from the workspace (paths are workspace-relative):

```toml
[dependencies]
llm-suite-gateway = { path = "crates/llm-suite-gateway" }
```

Build the project:

```shell
cargo build --workspace
cargo clippy --workspace --all-targets
```

## License

Private project; see the host repository for licensing details.
