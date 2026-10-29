# Decision API 能力接入设计方案

## 1. 背景与目标

`ref/open-python`（OpenAI Python SDK）提供 `/decisions` 能力：用一个模型对同一输入批量回答 1–400 个分类/打分问题，返回结构化答案与 token 用量。`ref/typesafe-sdk-python`（TypeSafe/Jev SDK）提供 `/v1/systemone` 能力，同样是“对同一状态回答命名问题”的决策模型，但请求/响应字段、问题原语与资源形态不同。

当前 Rust 工作区没有对应能力。新增 crate 时，应把 Jev/TypeSafe 风格视为决策模型的通用规范之一，而不是只支持 OpenAI 风格；同一能力 crate 需要同时支持 OpenAI 风格与 Jev 风格 API。

本方案只描述设计与接入路径，不包含完整实现代码。

## 2. 参考实现分析

### 2.1 OpenAI 风格（`ref/open-python`）

- 资源入口：`client.decisions.create(...)`。
- HTTP：`POST /decisions`。
- 请求体：
  - `model`：必填。
  - `input`：必填，字符串或消息/工具调用数组。
  - `questions`：必填，1–400 个问题，类型为 predicate / choice / score。
  - `prompt_cache_key`、`prompt_cache_options`、`safety_identifier`：可选。
- 响应体：`Decision`，包含按问题顺序排列的 `answers`、`model`、`usage`。
- 问题形态：
  - `predicate`：判断陈述为真的概率。
  - `choice`：从 2–255 个唯一选项中选一个，选项值可为字符串或布尔。
  - `score`：按有序等级打分，等级带标签与描述。
- 答案形态：
  - predicate：`probability`。
  - choice：`choice`、`confidence`、每个选项的概率。
  - score：`score`、`confidence`、每个等级的概率。
  - refusal：模型可拒绝单个问题。
- 选择值类型敏感：字符串 `"true"` 与布尔值 `true` 是不同值。

### 2.2 Jev/TypeSafe 风格（`ref/typesafe-sdk-python`）

- 资源入口：`client.system_one(state, questions, ...)`。
- HTTP：`POST /v1/systemone`。
- 请求体：
  - `model`：必填。
  - `state`：必填，文本、JSON 对象/数组或图片资产。
  - `questions`：必填，非空命名映射，键为调用方自定义答案名。
  - `assets`：可选，图片资产映射，使用引用占位符。
- 响应体：`{ model, answers, usage }`，其中 `answers` 是按问题名键控的映射。
- 问题原语：
  - `noul`：yes/no 问题，返回 0–1 概率；可带 `true`/`false` 结果描述。
  - `choice`：按标签选择，criteria 是标签到描述/`None` 的映射。
  - `score`：按有序等级打分，criteria 是描述列表，分数从 0 开始。
  - `box`：在图片中定位目标，返回坐标与置信度。
- 答案形态：
  - noul：`noul` 概率。
  - choice：`choice` 标签、`confidence`、按标签分组的概率。
  - score：`score`、`confidence`、`legend`、按等级分组的概率。
  - box：坐标与可空置信度。
- 用量比 OpenAI 简化，但 Rust 侧应保留统一用量结构的可选字段。

### 2.3 高层决策执行器（可选）

`ref/open-python` 还提供 `DecisionsToolRunner`，基于 Decision API 做动作选择循环。它属于 Agent 编排层，不属于 HTTP 能力本身。当前工作区没有对应编排抽象，第一阶段不迁移；如 host 需要，可在 host 或后续编排 crate 中实现。

## 3. 当前项目现状与差距

- 已有 `llm-suite-embedding`、`llm-suite-rerank` 两个能力 crate，模式成熟。
- `llm-suite-common::ProviderError` 已统一能力层错误，可直接复用。
- `llm-suite-proxy::build_http_client` 已统一代理、超时、TLS 处理，应复用。
- 缺少 Decision 类型、配置、provider trait、HTTP 实现、mock 与测试。
- 网关只编排 chat；embedding/rerank 由 host 直接构造 provider，Decision 也应遵循同样边界。

## 4. 设计目标与非目标

目标：

- 提供类型安全的决策客户端，同时支持 OpenAI `/decisions` 与 Jev `/v1/systemone`。
- 以一组通用决策类型承载两种协议，协议差异收敛在 adapter/provider 内部。
- 与 embedding/rerank 保持一致的 crate 结构、错误语义、配置预设、校验风格。
- 通过 serde 正确表达判别联合、类型敏感值、命名答案与可选字段。

非目标：

- 不在客户端强制模型白名单。
- 不把 Decision 接入网关 chat 编排；仅复用共享 breaker/limiter 注册表。
- 不提供向后兼容层或旧数据迁移。

Jev 图片资产上传与 Decision runner 编排均纳入正式设计，分别在 5.12 与 5.13 描述，不作为延后项。

## 5. 新增 crate：`llm-suite-decision`

### 5.1 模块布局

```
crates/llm-suite-decision/
  Cargo.toml
  src/
    lib.rs
    config.rs
    error.rs
    types.rs
    assets.rs
    provider.rs
    openai.rs
    jev.rs
    runner.rs
    mock.rs          # feature = "mock"
```

`lib.rs` 导出通用类型、provider trait、两个协议 provider、mock（feature-gated）。

### 5.2 Cargo 依赖

与 embedding 对齐：`llm-suite-common`、`llm-suite-proxy`、`reqwest`、`serde`、`serde_json`、`thiserror`、`async-trait`、`tokio`、`tracing`、`url`。feature 仅 `mock`。dev-dependencies 包含 `tokio`、`async-trait`。

工作区 `Cargo.toml` 的 `members` 需加入新 crate。

### 5.3 配置

通用 `DecisionConfig` 字段与 `RerankConfig`/`EmbeddingConfig` 一致：`base_url`、`api_key`、`model`、`timeout_secs`、`proxy`、`no_proxy`、`headers`、`query_params`。

预设：

- `DecisionConfig::openai(api_key, model)`：`https://api.openai.com/v1/decisions`
- `DecisionConfig::jev(api_key, model)`：`https://api.typesafe.ai/v1/systemone`

两个预设的 `base_url` 都是完整 endpoint URL，与 embedding/rerank 的配置风格一致；provider 直接 POST 到 `base_url`，不在 provider 内部拼接 `/v1/...`。

校验：`base_url` 非空且可解析为 URL，`model` 非空。API key 可选。

### 5.4 错误

`DecisionError` 直接复用 `llm_suite_common::ProviderError`，别名为 `DecisionError`。这样 retry/breaker 判定与其它能力层一致。

### 5.5 通用类型模型

通用请求：

- `DecisionRequest`：
  - `model: Option<String>`，缺省时使用 provider 默认模型；
  - `state: DecisionState`，表示被评估内容；
  - `questions: Vec<NamedQuestion>`，每项包含可选 `name` 与 `question`；
  - 可选 OpenAI 专用字段：`prompt_cache_key`、`prompt_cache_options`、`safety_identifier`；
  - 可选 Jev 专用字段：`assets`。
- `DecisionState`：
  - `Text(String)`；
  - `Json(serde_json::Value)`；
  - `Parts(Vec<ContentPart>)`；
  - `Messages(Vec<DecisionMessage>)`；
  - `AssetRef(String)` 或 `Assets(Map<String, Asset>)`（Jev 引用）。
- `NamedQuestion { name: Option<String>, question: DecisionQuestion }`。

通用问题：

- `Predicate`：判断陈述为真的概率，对应 OpenAI predicate 与 Jev noul。字段：`instructions`、可选 `true_description`、可选 `false_description`。
- `Choice`：从命名选项中选择。字段：`instructions`、`options: Vec<ChoiceOption { value: ChoiceValue, description: Option<Content> }>`。
- `Score`：按有序等级打分。字段：`instructions`、`levels: Vec<ScoreLevel { label: Option<String>, description: Option<Content> }>`。
- `Box`：在图片中定位目标，字段：`instructions`。仅 Jev 协议支持。

通用答案：

- `Predicate { probability }`，对应 OpenAI predicate probability 与 Jev noul。
- `Choice { choice: ChoiceValue, confidence, probabilities: Vec<(ChoiceValue, f64)> }`。
- `Score { score, confidence, probabilities: Vec<(i64, f64)>, legend: Option<Map<i64, Content>> }`。
- `Box { coordinates, confidence: Option<f64> }`。
- `Refusal { name: Option<String> }`。

通用响应：

- `DecisionResponse`：
  - `model: String`；
  - `answers: Vec<NamedAnswer>`，每项包含可选 `name` 与 `answer`；
  - `usage: DecisionUsage`。

通用用量：

- `DecisionUsage` 保留 OpenAI 详细字段，但全部允许缺失/默认：输入/输出/总 token、缓存写入、缓存命中、推理 token。

类型敏感值：

- `ChoiceValue` 使用 untagged 枚举，区分字符串与布尔。

内容类型：

- `Content` 为 JSON-like 递归类型，允许文本、数字、布尔、对象、数组、空值；图片资产在 Jev 路径通过引用占位符表达。

### 5.6 协议映射

OpenAI adapter：

- `DecisionState::Text` -> `input` 字符串；
- `Messages` -> `input` 消息数组；
- 其它状态 -> 仅当能表达为消息/JSON 时支持，否则返回 `InvalidRequest`；
- `Predicate` -> OpenAI predicate，instructions 直接发送，忽略 true/false 描述或将其折叠进 instructions（实现时二选一并保持一致）；
- `Choice` -> OpenAI choice，options 映射为 `choices: [{ value, description }]`；
- `Score` -> OpenAI score，levels 映射为 `levels: [{ label, description }]`；
- `Box` -> 返回 `InvalidRequest`；
- `NamedQuestion.name` -> OpenAI question `name`；
- 响应 answers 数组 -> `Vec<NamedAnswer>`，name 取 answer.name；
- refusal 按 `type = "refusal"` 解析。

Jev adapter：

- `DecisionState` -> `state`，文本/JSON/数组直接序列化，资产引用按 Jev 规则处理；
- `Predicate` -> Jev noul，instructions 与 criteria `{ true, false }`；
- `Choice` -> Jev choice，options 映射为 criteria map：标签为 `ChoiceValue` 的字符串形式，描述为 option.description；
- `Score` -> Jev score，levels 映射为 criteria 描述列表；label 可放入 legend 或忽略；
- `Box` -> Jev box；
- `NamedQuestion.name` 必填，缺失则 `InvalidRequest`；
- questions Vec -> Jev questions map；
- 响应 answers map -> `Vec<NamedAnswer>`，按键名排序；
- Jev noul answer -> `Predicate { probability: noul }`；
- Jev score answer 的 legend -> `Score.legend`。

### 5.7 Provider trait

```rust
#[async_trait]
trait DecisionProvider: Send + Sync {
    async fn create(&self, request: &DecisionRequest) -> Result<DecisionResponse>;
    fn protocol_name(&self) -> &'static str;
    fn model_name(&self) -> &str;
}
```

MVP 只提供单次调用。批量问题已在单请求内表达。

### 5.8 OpenAI provider

`OpenAIDecisionProvider`：

- 构造时调用 `config.validate()`；
- 使用 `llm_suite_proxy::build_http_client`；
- `create` 校验请求、通过 OpenAI adapter 序列化、POST 到 `base_url`；
- 附加 bearer、headers、query params；
- 非 2xx 映射为 `ProviderError::Provider`，解析 `Retry-After`；
- 成功响应通过 adapter 反序列化为 `DecisionResponse`。

### 5.9 Jev provider

`JevDecisionProvider`：

- 构造时调用 `config.validate()`；
- 使用 `llm_suite_proxy::build_http_client`；
- `create` 校验请求、通过 Jev adapter 序列化、POST 到 `base_url`；
- 附加 bearer、headers；
- 非 2xx 映射为 `ProviderError::Provider`；
- 成功响应通过 adapter 反序列化为 `DecisionResponse`；
- 第一阶段仅支持文本/JSON/消息状态与引用占位；完整资产 multipart 上传作为后续阶段。

### 5.10 Mock

feature-gated `MockDecisionProvider`：

- 可返回预设 `DecisionResponse`；
- 可注入 `DecisionError`；
- 记录调用次数与最近请求；
- 不生成默认答案。

### 5.11 校验与 misuse-resistance

- `ChoiceValue` 的类型区分必须在序列化层保证。
- 任何 answer 可为 refusal，调用方必须处理 refusal 分支；客户端不把 refusal 转换为成功答案。
- 不对 model 做白名单。
- 不吞掉解析错误；字段缺失、数量不匹配、非有限数值都返回 `Decode`/`InvalidRequest`。
- 协议不支持的问题类型（如 OpenAI 不支持 Box）必须在发送前失败，而不是静默忽略。
- Jev 命名问题缺少 name 时必须失败。
- 不提供 fallback 默认答案。

### 5.12 Jev 图片资产上传

Jev/TypeSafe 使用引用占位符与 multipart 上传图片，Rust 侧应完整支持，而不是延后。

资产类型：

- `DecisionAsset`（当前仅 `ImageAsset`）：
  - 来源：文件路径、字节缓冲、base64 文本；
  - 字段：内容类型、字节数据、唯一引用字符串 `<image:{hex32}>`。
- `ImageAsset` 构造时只保存来源，不立即读取；`resolve` / `resolve_async` 时读取并校验非空。
- 内容类型检测：按魔术字节识别 PNG/JPEG，其它返回 `application/octet-stream`，与参考实现一致。

请求编码流程：

1. 调用方在 `DecisionState`、问题 instructions/criteria 中使用 `asset.reference()` 字符串占位。
2. `DecisionRequest.assets` 为 `Map<String, DecisionAsset>`，键为非空别名。
3. provider 先解析所有资产，同对象只解析一次；失败时错误带别名。
4. 将资产映射转换为 Jev wire 的 `assets` 字段：`{ alias: { type: "image", part: "asset-{i}" } }`。
5. 序列化 JSON body，扫描其中所有 `<image:...>` 引用，把未显式登记的资产也加入待上传集合。
6. 若无资产，发送普通 JSON 请求。
7. 若有资产，构造 multipart：
   - 第一个 part 名为 `request`，内容为 JSON body，content type 为 `application/json`；
   - 后续每个资产一个 part，name 为 `asset-{i}`，filename 可与 name 相同，content type 为检测结果；
   - boundary 使用随机 UUID。
8. multipart body 替换 JSON body，Content-Type 改为 `multipart/form-data; boundary=...`。

校验：

- 别名必须是非空字符串；
- 资产必须是支持的图片类型；
- 图片字节不能为空；
- 文件读取失败时错误必须包含别名与路径信息；
- 资产引用扫描只识别合法的 32 位十六进制引用，避免误判普通文本。

### 5.13 Decision runner 编排

OpenAI Python 的 `DecisionsToolRunner` 应作为正式编排能力提供，但放在 `runner` feature 或独立模块中，不污染基础 HTTP 能力。

模块：`runner.rs`，feature `runner`，依赖 `DecisionProvider` 与标准库/async-trait。

核心类型：

- `DecisionCandidate`：
  - `Action { id, description, execute }`
  - `Done { id, description }`
- `DecisionRunnerParams`：`model`、`input`/`state`、可选 `safety_identifier`。
- `DecisionRunResult`：`status`（verified/unverified/refused）、`decisions`、`attempted_candidate_ids`、`completed_candidate_ids`。
- `DecisionRunnerError`：reason、attempted、completed；reason 覆盖 invalid_params、invalid_candidates、invalid_answer、async_executor、invalid_verification、verification_failed、step_limit、failed、cancelled。
- `DecisionRunnerCancelledError`：继承取消语义，携带 attempted/completed。

同步与异步两套实现：

- `DecisionRunner::run(provider, params, max_steps, cancelled, on_diagnostic)`；
- `AsyncDecisionRunner::run(provider, params, max_steps, cancelled, on_diagnostic)`。

执行循环：

1. 校验 params 与 max_steps。
2. 每轮：
   - 检查取消；
   - 调用 observe，深拷贝观测值；
   - 调用 candidates，校验 1–255 个、id 唯一、描述非空、恰好一个 done、action 有可调用 executor；
   - 若只有 done 候选，直接选择它；
   - 否则构造 `DecisionRequest`，问题为 choice，选项为候选 id/description，调用 provider；
   - 严格解析 answer：字段集合、类型、choice 值、有限 confidence、概率结构、id 匹配；
   - refusal 则返回 refused；
   - done 则执行 verify（如提供），成功返回 verified，失败在非最后一步继续；
   - action 则记录 attempted，执行 executor，同步版本拒绝 awaitable 结果，记录 completed。
3. 步数超限返回 step_limit；取消返回 cancelled；其它异常包装为 failed。

诊断：

- `on_diagnostic` 回调记录 request、selection、verification、action 的开始/完成/失败，与参考实现一致。
- 不在 runner 中重试 HTTP；重试由 provider/弹性层处理。

与 provider 的关系：

- runner 接受 `Arc<dyn DecisionProvider>` 或泛型 `P: DecisionProvider`；
- 通过通用 `DecisionRequest`/`DecisionResponse` 调用，不感知 OpenAI/Jev 差异；
- Jev 路径下 questions 必须有 name，runner 使用固定 `next_action` 作为 name，两个协议都可表达。

### 5.12 与 5.13 的测试策略补充

- 资产：文件读取、内容类型检测、引用扫描、multipart 编码、异步解析、别名错误。
- runner：候选校验、refusal、verified/unverified、step_limit、取消、同步 executor 返回 awaitable、异步执行、诊断计数。

## 6. 与现有 crate 的集成

- `Cargo.toml` workspace members 增加新 crate。
- `README.md` crate 表格增加一行。
- host 构造 provider 的方式与 embedding/rerank 一致。
- 共享弹性组件：provider 只返回 `DecisionError`；host 可用 `llm-suite-client::Resilience` 包装调用，或使用网关共享 breaker/limiter。
- 错误类型实现 `ResilienceError` 已由 `ProviderError` 提供。

## 7. 测试策略

- 单元测试：配置校验、两种协议的请求序列化、响应反序列化、判别联合、ChoiceValue 类型区分、数量/唯一性校验、命名答案映射、协议不支持类型失败。
- 资产测试：文件读取、内容类型检测、引用扫描、multipart 编码、异步解析、别名错误。
- Runner 测试：候选校验、refusal、verified/unverified、step_limit、取消、同步 executor 返回 awaitable、异步执行、诊断计数。
- Mock 测试：成功、refusal、provider 错误。
- HTTP 层：验证路径、headers、query params、错误映射。
- 不做真实网络测试。

## 8. 实施顺序

1. 建立 crate 骨架与 workspace 注册。
2. 实现 config 与 error。
3. 实现通用 types 与 ChoiceValue。
4. 实现 OpenAI adapter 与 provider。
5. 实现 Jev adapter 与 provider（文本/JSON 状态）。
6. 实现 Jev 图片资产上传（assets、resolve、multipart）。
7. 实现 mock provider 与基础测试。
8. 实现 Decision runner（同步/异步）与诊断。
9. 更新 README 与示例。
10. 补充资产与 runner 的专项测试。

## 9. 风险与开放问题

- OpenAI 与 Jev 的问题原语不完全同构，通用类型需要保留可选字段，避免强行统一导致信息丢失。
- Jev 的命名问题与 OpenAI 的顺序答案需要在响应层做稳定映射。
- Jev 图片资产上传涉及 multipart 与引用解析，复杂度高，但已有参考实现，应按 5.12 正式实现。
- OpenAI predicate 与 Jev noul 的 criteria 描述无法完全互转，需要定义 adapter 的降级规则。
- Rust 的 untagged ChoiceValue 需要仔细处理 bool 与 string 的反序列化优先级。
- Decision runner 的回调与生命周期管理应放在编排层，但已有参考实现，应按 5.13 正式实现。
