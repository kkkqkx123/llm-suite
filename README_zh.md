# llm-kit

一个模块化的 Rust 工具集，将"调用大模型"从分散的临时 HTTP 代码转变为分层的、
与具体厂商无关的协议栈：协议类型、线上编解码器（codec）、传输层、能力客户端、
配置层，以及统一的编排网关。

以 workspace 形式开发，`crates/` 下共 12 个小 crate，宿主项目（如 `wf-agent`）
通过 git submodule 引入使用。

## 架构

各 crate 自底向上构成严格无环依赖图（DAG），顶层为网关：

```
                ┌──────────────┐
                │ llm-gateway  │   编排入口
                └──────┬───────┘
        ┌──────────────┼────────────────┬───────────────┐
┌───────┴──────┐ ┌─────┴───────┐ ┌──────┴──────┐ ┌──────┴───────┐
│llm-chat-basic│ │llm-embedding│ │ llm-rerank  │ │  llm-client  │
│ (基础对话客户端)│ │  (向量嵌入)  │ │  (重排序)    │ │  (传输层)     │
└───────┬──────┘ └─────────────┘ └─────────────┘ └──────┬───────┘
        │                       ┌────────────────────────┤
┌───────┴──────┐        ┌───────┴──────┐          ┌──────┴──────┐
│  llm-codec   │        │ llm-tool-call│          │ llm-config  │
│ (线上协议编解码)│        │ (工具调用解析) │         │(配置/档案/目录)│
└───────┬──────┘        └──────┬───────┘          └──────┬──────┘
        │          ┌───────────┴──────────┐              │
┌───────┴──────┐   │                      │              │
│ llm-message  │───┤ llm-types  llm-token │──────────────┤
│ (消息变换)     │   │ (协议叶子:类型/Token) │              │
└──────────────┘   └──────────────────────┘   ┌──────────┴────┐
                                              │  llm-common   │
                                              │  (运行时工具)   │
                                              └───────────────┘
```

## 各 Crate 职责

| Crate | 职责 |
|---|---|
| `llm-common` | 与宿主无关的共享运行时工具：毒锁（poisoned lock）恢复、墙钟时间、ID 生成、重试辅助、exec 辅助。 |
| `llm-types` | 零依赖协议叶子：聊天消息、LLM 请求/响应信封、线上工具声明、参数 Schema 与用量模型，被所有层共享。 |
| `llm-message` | 纯消息变换：流式消息构建器、会话历史转换（如转为厂商线上格式）、文本提取及历史文本辅助。 |
| `llm-token` | Token 估算：纯文本估算 + 按厂商请求形态计数（消息、工具声明、图片），用于预算控制与上下文窗口校验。 |
| `llm-codec` | 线上协议层：`LlmCodec` trait、内置编解码器（OpenAI Chat Completions、OpenAI Responses、Anthropic Messages、Gemini 原生）、生成参数映射与运行时 codec 注册表。接入新厂商协议只需实现一个 trait 并注册。 |
| `llm-tool-call` | 工具调用协议解析，按 feature flag 门控，调用方只为用到的协议付出编译成本。内含流式部分 JSON 解析器，支持增量拼装工具调用参数。 |
| `llm-client` | 传输层：异步 client trait + 基于 reqwest 的 HTTP 实现（支持 SSE 流式）、流累积、针对异常流的死循环守卫、通用用量收集器，以及 feature 门控的脚本化/TCP mock 用于测试。 |
| `llm-chat-basic` | 最小化的 OpenAI 兼容 `/chat/completions` 客户端：请求构建、指数退避重试、客户端限流，支持阻塞与流式调用。刻意不依赖 codec，面向简单场景。 |
| `llm-embedding` | 向量嵌入能力：provider trait + OpenAI 兼容 HTTP 实现，覆盖 OpenAI、Gemini、Azure、Ollama 端点，并带文本预处理管线。 |
| `llm-rerank` | 重排序能力：Cohere 兼容的专用 `/rerank` provider 与生成式（基于 LLM 的）重排序器，附结果融合策略。 |
| `llm-config` | 配置层：厂商定义注册表、命名 profile、模型目录（从 JSON 加载），以及将 profile/厂商/模型默认值合并到外发请求的请求合并逻辑。 |
| `llm-gateway` | 编排入口（`LlmGateway`）：强制 profile 解析、请求合并、客户端缓存、面向测试的 mock 路由、指标钩子。宿主代码只与本 crate 交互。 |

## 工作原理

一次请求经过四个阶段：

1. **配置**：宿主注册厂商定义（端点、鉴权、默认模型）与命名 profile；
   模型目录将模型 id 映射到能力与限制。
2. **解析**：调用方通过 profile 名向 `LlmGateway` 请求客户端。网关解析
   profile，将默认值合并到请求（`llm-config::merge_request`），按厂商选择
   线上编解码器（`llm-codec` 注册表），返回缓存的客户端。
3. **编码与传输**：codec 将 `llm-types` 的请求信封序列化为厂商线上格式；
   `llm-client` 执行 HTTP 调用（或 SSE 流式），应用重试/限流策略，并将
   响应经 codec 解析回协议类型。
4. **后处理**：工具调用由 `llm-tool-call` 的部分 JSON 解析器增量重组；
   用量数据汇入 token 收集器（`llm-token`）；向量嵌入与重排序经由各自
   的能力 crate，复用同一套 profile/配置机制。

设计原则：

- **严格 DAG、无循环依赖**：协议叶子（`llm-types`、`llm-common`）不感知
  传输与厂商；只有网关将所有部件装配在一起。
- **核心与厂商无关**：厂商差异全部封装在 codec 与 provider trait 实现中，
  不污染业务逻辑。
- **按需编译**：工具调用协议解析器与测试 mock 均置于 cargo feature 之后，
  生产构建保持精简。
- **不依赖宿主**：任何 crate 都不依赖宿主应用，可完全脱离 `wf-agent` 复用。

## 使用

从 workspace 中按需引入所需 crate（路径相对 workspace 根）：

```toml
[dependencies]
llm-gateway = { path = "crates/llm-gateway" }
```

构建项目：

```shell
cargo build --workspace
cargo clippy --workspace --all-targets
```

## 许可

私有项目，许可信息见宿主仓库。
