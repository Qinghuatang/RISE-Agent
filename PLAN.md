# Implementation Plan

本计划按照人工审核后的 contract 修正。M0–M11 已完成；M12 工程实现和真实单论文 pilot
已通过人工审核。M12.1 Design-Rationale Preservation Hardening 与 M12.2 Scientific
Controller Inspiration Boundary 已完成；当前进入 M13 Batch Evaluation and Research Agent
Evaluation Framework。每个 milestone 完成后都必须
运行对应测试，并确认没有改变方法定义。

## 计划原则和依赖关系

实现顺序不直接照搬方法文档的 1→10 顺序：先固定可序列化契约、离线依赖和 State/Memory 边界，再实现模块，最后才实现 Controller 和完整 trajectory。这样 deterministic merge、action mask、budget 和 Commit Gate 都可以在 Controller 出现前独立测试。

```text
M0 Foundation + LLM infrastructure
  -> M1 Schemas/contracts
  -> M2 Markdown -> SeedPaperContext
  -> M3 ResearchState + ResearchMemory + trajectory
  -> M4 Gap + Solution
  -> M5 Related-work Verification + Semantic Scholar adapter
  -> M6 Inspiration + arXiv adapter
  -> M7 Targeted Revision
  -> M8 Experiment Design
  -> M9 Scientific Controller
  -> M10 Commit + Finalizer
  -> M11 Offline End-to-End Trajectory
  -> M12 Real Single-Paper Integration Pilot
  -> M12.1 Design-Rationale Preservation Hardening
  -> M12.2 Scientific Controller Inspiration Boundary
  -> M13 Batch / Experiment Support
```

M4 依赖 SeedPaperContext 和 State/Memory；M5 依赖 retrieval interfaces/cache 并加入真实 Semantic Scholar adapter；M6 依赖 M5 的 verification context 并加入真实 arXiv adapter；M7/M8 依赖验证结果和 Memory；M9 必须等模块输出契约稳定，避免把 Controller 变成隐藏 planner；M10 依赖稳定 State、Verification 和 Experiment schema；M12 必须在 M11 离线 E2E 稳定后执行；M12.1 只在 M12 通过后做通用 rationale-preservation hardening；M12.2 只复用现有 Controller inputs 和七个 Action，M13 必须等待 M12.2 通过。

每个 milestone 的 unit test 都使用 fake LLM/provider 或纯 deterministic fixture。除非特别写明，均“不需要真实 LLM / 网络”。

## M0 — Project Foundation + LLM Infrastructure

### Objective

建立最小 Python 项目骨架、配置读取、`LLMClient` abstraction、DeepSeek adapter、Fake LLM、serialization、日志/错误边界和离线 CLI；不实现 SeedPaperContext、Research State 或任何科研模块。

### Inputs / dependencies

- 当前 `docs/`、`data/` 只读。
- `AGENTS.md`、`ARCHITECTURE.md` 和本计划。
- 已确认的人工 contract decisions；本 M0 不再保留这些事项的 unresolved 分支。

### Files/modules to implement

- `pyproject.toml`、`src/research_agent/` 最小 package 和 `python -m research_agent` entry point。
- `src/research_agent/llm/config.py`：读取 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 timeout/retry 配置。
- `src/research_agent/llm/client.py`：面向 structured output 的 `LLMClient` Protocol、request/response metadata 和明确的 malformed-output errors。
- `src/research_agent/llm/deepseek.py`：`DeepSeekClient` infrastructure adapter；本轮不发起网络请求。
- `src/research_agent/llm/fake.py`：后续离线测试使用的 deterministic `FakeLLMClient`。
- `src/research_agent/serialization/`：JSON/JSONL、schema/version envelope 和明确 serialization errors。
- `src/research_agent/errors.py`、`src/research_agent/logging_utils.py`：最小错误层和 logging 配置。
- `tests/unit/test_config.py`、`test_llm_client.py`、`test_deepseek.py`、`test_fake_llm.py`、`test_serialization.py`、`tests/integration/test_cli_smoke.py`。

### Acceptance criteria

- package 可安装/导入，默认配置不要求 API key；没有 `DEEPSEEK_API_KEY` 时所有 offline tests 正常运行。
- `LLMClient` 面向 structured output；DeepSeek-specific logic 只存在于 `DeepSeekClient`，scientific modules 未来只能依赖 Protocol。
- `DeepSeekClient` 能从配置构造，缺少请求所需配置时明确报错；structured response 解析失败时明确报错，不静默吞掉 malformed output。
- `FakeLLMClient` 可按队列/handler 返回结构化 fixture，并记录调用。
- JSON/JSONL 序列化具有 schema/version envelope 和明确错误。
- `python -m research_agent --help`（以及 offline smoke command，如提供）成功运行且不触网。
- `data/` 和 `docs/` 在测试中被验证为只读输入路径，不被测试写入。

### Unit tests

- 环境变量覆盖 `DEEPSEEK_*` 配置；缺少 key 时 offline config/client construction 可用，实际 request 明确失败。
- `LLMRequest`/structured response metadata 和 Protocol contract。
- Fake LLM 的成功、耗尽、handler failure 和 malformed fixture 行为。
- DeepSeek client 的配置构造、request payload/response parsing（使用 injected fake transport，不联网）和 malformed response errors。
- JSON/JSONL round-trip、schema/version mismatch、invalid JSON 和不可序列化 payload。

### Integration tests

- `myrag` 环境中运行 `python -m unittest discover -s tests -v` 或 `python -m pytest -q`，默认不触网。
- 运行 CLI smoke 和测试后检查 docs/data 内容 hash 或 git diff 不变（若仓库已启用 git）。

### Real LLM / network

不需要；禁止真实 DeepSeek API 和任何论文网络 API。可以保留显式 opt-in 的真实 smoke test，但本 milestone 默认 skip/不运行。

### Minimal runnable example

运行 `python -m research_agent --help`，用 `FakeLLMClient` 完成一次结构化 fixture 调用，并写入/读回一条 JSONL schema/version envelope。

## M1 — Core Schemas and Contracts

### Objective

把方法文档已确定的 Research State、Verification、Inspiration、Revision、Memory、Controller 和 Commit 数据结构转成可校验模型；不实现模块逻辑。

### Inputs / dependencies

- M0 serialization/config。
- `ARCHITECTURE.md` 第 5 节及 `docs/implementation_spec.md`。
- 使用已确认的 Component、Experiment、CurrentSolution、VerificationFocus、RefineGap、all-rejected-candidate 和 latest-verification contracts；不得新增未确认状态。

### Files/modules to implement

- `src/research_agent/models/seed.py`
- `src/research_agent/models/base.py`、`state.py`、`gap.py`、`verification.py`、`inspiration.py`、`revision.py`、`memory.py`、`controller.py`、`commit.py`、`trajectory.py`
- M1 contract repair：`CurrentSolution.novelty_summary` 可为 `None`；`VerificationRecord` task/result/focus 一致性；`InspirationResult` 1–3/empty 数量约束；`RevisionResult` revised/unable patch 一致性。
- `src/research_agent/models/__init__.py`
- `src/research_agent/serialization/codec.py`：兼容 Pydantic v2 `model_dump(mode="json")`，保留 M0 envelope。
- `tests/test_models.py`：valid/invalid contract tests；不引入网络 fixture。

### Acceptance criteria

- 所有 canonical schemas 使用 Pydantic v2、`ConfigDict(extra="forbid")`，required 字段、枚举、嵌套结构和 2–5 Components 约束可校验。
- `GapAgenda` 独立于 Research State/Research Memory，最多保存五个 Gap；`VerificationFocus` 严格区分 `full_solution` 和 `changed_units`。
- `novelty_status` 只有文档规定的四个值；Action 只有七个固定值；`Experiment` 不接受 `design` 字段。
- `ResearchState` 可以表达空状态、无 Current Solution、`unchecked` novelty 等中间态；不为此增加科研状态。
- Targeted Revision 只产生 typed `ComponentRevisionPatch`，不携带 Component/Experiment/novelty/evidence 状态字段。
- 结构化错误不会被静默转换成空文本，Pydantic 模型能通过 M0 JSON/JSONL envelope round-trip。

### Unit tests

- 每个 core schema 至少有 valid example；覆盖 extra field、非法状态、`CurrentSolution`/`GapAgenda` 尺寸和 intermediate ResearchState。
- `Component` 状态、`CurrentSolution.status`、`novelty_status`、canonical Experiment 和 `VerificationFocus` 的非法组合。
- Revision patch 的 `status`/`novelty_status`/`evidence_refs` 等额外字段被拒绝；验证结果不接受 `unchecked`。
- CommitSnapshot 包含 Gap/Solution/experiments/最新验证引用；ResearchMemory 保持六个顶层字段。

### Integration tests

- 从 Pydantic JSON fixture 反序列化完整 State + Memory + Snapshot 并重新序列化；JSONL envelope 保持 schema/version。

### Real LLM / network

不需要；禁止调用。

### Minimal runnable example

从 fixture 构造一个 Active Gap、Current Solution 和一条 distinct Verification Result，输出 Pydantic JSON envelope；M1 完成后停止，不实现 Markdown 抽取或科研模块。

## M2 — Markdown → SeedPaperContext

### Objective

在不修改原 Markdown 的前提下，完成 deterministic Markdown parsing、完整 main-paper body 构建、一次受约束的 LLM structured extraction 和独立 Extraction Audit。

### Inputs / dependencies

- M0 配置/序列化，M1 SeedPaperContext schema。
- `data/` 中的 MinerU Markdown。
- Fake `LLMClient` fixture；真实 DeepSeek 不属于默认路径。
- 从 M2 开始所有 DeepSeek structured-output prompt 必须明确 JSON 输出要求、expected schema/structure 和合理 `max_tokens`；本 M1 不新增 prompt abstraction。

### Files/modules to implement

- `src/research_agent/ingestion/markdown.py`：heading hierarchy、section spans、references/appendix boundary、HTML table/equation/image blocks，以及只删除最早尾部边界后的 full-body reading context。
- `src/research_agent/models/seed_extraction.py`：`SeedContextExtractionDraft`、`SourceAnchor`、`SeedExtractionAudit`。
- `src/research_agent/modules/seed_context.py`：只依赖 `LLMClient` 的 one-call full-body pipeline、dynamic Pydantic JSON schema prompt 和 deterministic materialization。
- `src/research_agent/prompts/seed_context.md`：JSON-only、section-grounded extraction prompt。
- `src/research_agent/__main__.py`：显式 `extract-seed-context` DeepSeek entry point；默认 help/offline smoke 不触网。
- `tests/test_seed_context.py` 及现有 M0/M1 tests。

### Acceptance criteria

- 能从抽样文件定位 title、abstract、heading hierarchy/spans、HTML tables、equation blocks、references boundary、appendix 和 image placeholders。
- title/abstract 由 parser deterministic 提取；LLM 只归纳科学字段，`paper_id` 始终由文件 stem 决定。
- 抽取只返回 schema 字段；字段缺失为 `None`/`[]`，不能凭常识补全；References 不进入 LLM context。
- 每个非空 LLM 归纳字段有真实 source section anchor；不存在或被排除的 source id、无 evidence 的非空值会失败。
- 所有正文 section（包括 Related Work、Theory、Background、Analysis 等）一次性发送；不按字符数或 section 分块、不 synthesis、不静默截断。provider context limit 不足时显式失败，交由人工调整。
- 输出分离为 `seed_context.json` 与 `seed_extraction_audit.json`，原始文件仍未修改。

### Unit tests

- `## ABSTRACT` 定位、大小写/编号 heading、appendix/reference 最早边界和 full-body context。
- 多行 HTML table、`$$`、`<eq>` 和图片占位符完整保留；Related Work、Method、Experiments 等正文均进入请求。
- ByteFlow/Cautious 中断词和 `# param ...` 伪算法 heading 只作为文本 artifact，不误判为科研 section。
- fake LLM 返回缺失字段、额外字段、无 evidence、无效 source id 和 References/Appendix evidence 的校验；prompt 必须包含由 `SeedContextExtractionDraft.model_json_schema()` 生成的实际 JSON schema。

### Integration tests

- 对三篇抽样 Markdown 运行 deterministic parser-only smoke，检查 title、abstract、section 数、完整正文 section、references/appendix boundary 和 warnings。
- 用 FakeLLM 完成单篇 extraction、audit round-trip 和超长正文 one-call；检查 request 不包含 References/Appendix、包含全部正文且不存在 synthesis second call。

### Real LLM / network

offline acceptance 默认不需要真实 LLM/网络；但本轮人工要求在两组离线测试全部通过后，显式 opt-in 对指定 ByteFlow Markdown 执行且仅执行一次 DeepSeek smoke。credentials 缺失时安全停止并记录配置缺失；不批量运行、不调用论文检索 API。

### Minimal runnable example

通过 FakeLLM 给定 ByteFlow Markdown，输出 `seed_context.json` 和 `seed_extraction_audit.json`；离线测试通过后，使用显式 `extract-seed-context` 命令对同一篇 ByteFlow 运行一次 real smoke（若 credentials 可用），并核对 request count、input_chars、token usage、context 和 audit。

## M3 — Research State, Research Memory and Trajectory

### Objective

实现 trajectory-local State/Memory store、相关查询、State diff 和完整事件记录，为所有科研模块提供可测试的确定性基础。

### Inputs / dependencies

- M1 schemas；M2 的 SeedPaperContext 作为 trajectory 起点。
- 方法文档 §1、§9 和 §10.5。

### Files/modules to implement

- `src/research_agent/runtime/state_store.py`
- `src/research_agent/memory/store.py`
- `src/research_agent/trajectory/diff.py`、`trajectory/recorder.py`、`trajectory/context.py`
- 现有 `serialization/codec.py` 的 JSON/JSONL envelope（不建立新的 persistence framework）
- `tests/test_runtime.py`

### Acceptance criteria

- Memory 分为 paper/verification/inspiration/revision/failure/decision log，且支持 relevant-memory 过滤。
- 一篇 Seed Paper 一个 Memory，默认不跨 paper 读取。
- 每个模块事件可记录 input summary、result、State diff、Memory writes、budget 和 error。
- Memory 写入为 deterministic store operation，不需要 Memory Agent。
- Research State 只保存 `active_gap`、`candidate_solutions`、`current_solution`、`experiments`，不增加 runtime 状态字段。
- GapAgenda 只作为 `gap_agenda_updated` trajectory payload；RefineGap 只通过 `gap_refined` event 和 `has_refined_gap()` 查询约束一次。
- State、Memory 和 append-only Trajectory 均可 JSON/JSONL round-trip；拒绝非法 envelope、非法 payload 和倒序 step。
- M3 不实现 event-sourcing、reverse/apply patch、数据库或 replay engine；完整 reconstruction 延后到 M11。

### Unit tests

- 各类 record 保存/查询、按已有 schema 字段和 recency 过滤；不假装支持 schema 未提供的 gap/solution 关联。
- 失败方向只在明确 rejected/failed 条件下进入 Failure Memory。
- State diff 对局部 Component/Experiment 修改稳定且可序列化。

### Integration tests

- 构造一条 seed → gap → verification 的假 trajectory，检查 State、Memory、GapAgenda、events 的 persistence 与一致性；不要求回放事件重建 State/Memory。

### Real LLM / network

不需要；禁止调用。

### Minimal runnable example

创建一篇空 trajectory，写入一个 related paper、一个 verification record 和一个 controller decision，再导出 JSONL。

## M4 — Gap Discovery and Solution Ideation（已完成）

### Objective

实现 Seed-paper-grounded Gap agenda 和验证后 Solution 的 generate–select–develop 契约；Gap Discovery 同时执行 method-level actionability、evidence sufficiency、whole-method consistency 和 confidence calibration；先覆盖模块自身，不接真实检索。

### Inputs / dependencies

- M1 schemas、M2 SeedPaperContext、M3 Memory/State。
- `LLMClient` 抽象和 fake structured responses。

### Files/modules to implement

- `src/research_agent/modules/gap_discovery.py`：one-call discovery、证据引用、active selection、RefineGap、SwitchGap。
- `src/research_agent/modules/solution_ideation.py`：FormSolution、Select + Develop、candidate pool reuse。
- `src/research_agent/models/gap.py`、`models/solution.py`：strict LLM drafts 和 M4 module results。
- `src/research_agent/prompts/gap_discovery.md`、`gap_refine.md`、`solution_form.md`、`solution_select_develop.md`。
- `src/research_agent/__main__.py`：显式 `discover-gaps` smoke CLI。
- `tests/test_m4.py`：Gap、Refine、Switch、Solution 和 trajectory integration tests。

### Acceptance criteria

- 初始 Gap 最多保留五个，不使用数值评分公式；Gap 必须是能够通过方法/机制实质修改、扩展、替换或重设计解决的 method-level problem，选择后只把 Active Gap 写入 State。
- Gap evidence refs 只允许 `research_objective`、`core_method`、`main_components[*]`、`main_findings[*]` 和 `limitations[*]`；`experiment_context.*` 只供后续 Experiment Design，不能用于初始 Gap generation。
- 纯 experimental/evaluation gap（缺少实验、dataset、baseline、metric、ablation、scale test 或 reporting）必须在 generation boundary 被拒绝；正常成功的 Gap Discovery Draft 必须返回 1–5 个 Candidate Gaps。
- Gap 来源按 explicit limitation/future direction → observed method weakness → method trade-off/constraint → grounded method opportunity 排序。Evidence Sufficiency 不要求已证明 failure，但必须证明机会与当前方法真实相关；fallback 必须回答 method constraint、remaining capability、substantive change、whole-method interaction 和 evidence 五个问题。
- Prompt 必须区分 design fact 与 method limitation，保留 Whole-Method Consistency；明确 contradiction 时不返回 Gap，partial mitigation 时只描述剩余问题。Confidence：high 主要用于 author-stated limitation/unresolved problem 或 direct failure，medium 可用于 author-stated future direction/clear trade-off/strong opportunity，low 用于 grounded opportunity without direct failure；priority 不得抬高 confidence。
- Prompt 要求模型按 method relevance、scientific value、actionability、evidence strength、priority 和 researchability 排序；deterministic active selection 直接选择 `gaps[0]`，不跳过 low-confidence candidate，不增加 numeric score。
- `partially_resolved` 只允许一次 RefineGap；Refine 只更新 statement，保持 ID、confidence、priority、reason 和 seed evidence；`resolved`/`rejected` 只能引起 deterministic Switch Gap，不生成 Solution。
- Solution 只能在 Memory 中该 Active Gap 的 latest Gap Verification 为 `valid` 时调用；FormSolution 一次产生三个候选，第二次完成 Select + Develop，形成 2–5 Components；新 Components 状态为 `ready`，Current Solution novelty 为 `unchecked`。
- Current Solution 为空且所有 Candidate Solutions 都 rejected 时，按最大既有 `S<number>` 生成下一组候选并替换旧池；仍有 candidate 时只执行 Select + Develop，不重新 Form。
- Solution 形成后输出确定性后置条件 `VERIFY_RELATED_WORK`；M4 不实现 Verification 或任何 M5 retrieval/provider。
- 所有 LLM draft 使用 `extra="forbid"` 和运行时注入的 Pydantic JSON schema；LLM 不生成 IDs/statuses，State 和 trajectory 由 deterministic materialization 更新。

### Unit tests

- confidence、candidate order、first-candidate active selection（包括 low-confidence candidate）。
- candidate/active/resolved/rejected 状态和 switch/refine 分支、一次性 Refine 保护。
- method-only evidence refs、拒绝 `experiment_context.*` refs、0/1–5/>5 gaps、strict extra fields、deterministic IDs。
- method-level prompt boundary、纯 evaluation gap 示例、observed method weakness 与 missing experiment 的区分。
- generic synthetic reasoning boundaries：explicit limitation、author future method direction、trade-off、constraint without demonstrated failure、grounded opportunity fallback、pure experimental future work rejection、1–5 count、0 candidate Pydantic rejection、low-confidence active selection、whole-method contradiction，以及限制 Prompt paper specialization。
- 三候选、2–5 components、剩余候选复用、全 rejected 后的 sequential IDs。
- latest valid Gap Verification 前置条件和 Failure Memory 类型过滤。
- 不允许 Solution Ideation 自行调用 Inspiration。

### Integration tests

- fake LLM 驱动 Seed Context → Gap Agenda → valid Gap → Current Solution，检查 State 和 Memory。

### Real LLM / network

默认不需要；所有 offline tests 使用 FakeLLM。离线 suite 全部通过后，先对原始 Markdown 执行一次新的 full-body、one-call SeedPaperContext extraction，再对新的 context 执行一次 Gap Discovery call；不做 Gap Verification，不执行 Solution Ideation，不调用论文检索 API。真实论文只作为 evaluation case，不反向触发 paper-specific Prompt 调整。

### Minimal runnable example

用一个 fixture SeedPaperContext 离线生成最多五个 Gap 和 active gap；写入 GapAgenda trajectory。提供 valid Gap Verification fixture 后，用 FakeLLM 完成三个候选和一个 2–5 Component Current Solution，返回 `VERIFY_RELATED_WORK`。

## M5 — Related-work Verification（已完成）

### Objective

实现论文池抽象、trajectory-local cache、独立的全局 Semantic Scholar JSON cache、第一版 `SemanticScholarProvider`、Gap/Solution 两种 verification、最多五篇筛选、结构化比较和 focused re-verification。每个 verification task 使用一次 structured LLM call；论文比较证据固定为标题+摘要。M5 工程补齐还包括 provider 限流/重试、科学搜索计数、Solution ComparisonRecord 审计路径和 core-overlap Failure Memory。

### Inputs / dependencies

- M1 State/Verification schemas、M3 Memory。
- `RelatedPaperProvider`、PaperMetadata 和 cache interface。
- M4 的 Active Gap/Current Solution fixture。

### Files/modules to implement

- `src/research_agent/retrieval/interfaces.py`、`fake.py`、`semantic_scholar.py`
- `src/research_agent/retrieval/semantic_scholar_cache.py`：跨 trajectory/运行的持久化 provider cache
- `src/research_agent/retrieval/semantic_scholar.py`：第一版真实 Semantic Scholar adapter
- `src/research_agent/modules/related_work_verification.py`
- `src/research_agent/prompts/verify_gap.md`、`verify_solution.md`
- `src/research_agent/models/memory.py`、`memory/store.py`：轻量 SearchRecord 和保存接口
- `src/research_agent/__main__.py`：显式 `verify-gap` smoke CLI
- `tests/test_m5.py`

### Acceptance criteria

- 每个 Seed Paper 只建立一次 Related Paper Pool；cache 命中不增加 search budget。
- 只实现 `VERIFY_GAP`/`VERIFY_SOLUTION`；结果分别为文档规定的三种 Gap 状态或三种 novelty 状态。
- `distinct` 允许普通 shared elements；只有 novelty-relevant overlaps 影响 partial/core。
- Focused re-verification 接收 changed units，但仍以完整 Solution 为上下文。
- 证据粒度默认 `abstract_only`，不虚构全文证据。
- Related pool 由官方 Semantic Scholar Academic Graph title resolution + Recommendations API 获取，最多 20 篇；local paper id 与 provider paper id 分离。
- provider 从 `config/semantic_scholar.json` 读取 API key、base URL、`auth_mode`、timeout/retry 和可选 `min_request_interval_seconds`；CLI 可用 `--semantic-scholar-config` 指定 JSON 文件。当 JSON 的 `api_key` 为空、空白或缺失时，从 `SEMANTIC_SCHOLAR_API_KEY` 回退读取；非空 JSON 值优先，两者都没有时在真实请求前显式失败。`auth_mode=bearer` 发送 `Authorization: Bearer <token>`，用于兼容中转站示例；`auth_mode=x-api-key` 保留官方 header。当前配置默认指向中转站基础路由 `https://s2api.ominiai.cn/s2`。每次真实 HTTP request 前 pacing；clock/sleep 可注入，离线测试不真实等待。HTTP 429/5xx/timeout/connection 有限重试，429 遵循 `Retry-After`，400/401/403/404 与 malformed response 不重试，并以结构化 retrieval error 暴露 status/retry metadata。
- provider 内部 HTTP request count 与 scientific search count 分开审计；cache miss 的一次 related-pool acquisition 即使失败也恰计一次 scientific search，cache hit 计零，retry 和多 endpoint request 不重复计 scientific search；provider 失败不伪造 valid 结果。
- 全局 cache 按 operation + request parameters 内容寻址，保存成功的标题搜索和推荐论文 JSON 响应；新 trajectory 或新 provider instance 命中完整 cache 时不发 HTTP 请求，partial hit 只请求未缓存 endpoint；失败/malformed response 不落成功缓存，cache 文件不含 API key。
- `VERIFY_SOLUTION` 仍为一次 structured call，但内部 draft 同时返回不超过五个合法、去重的 `ComparisonRecord` 与嵌套 `SolutionVerificationResult`；比较记录只写入 `solution_verified` trajectory/audit，不进入 State 或 canonical VerificationRecord.result。最终 related-paper refs 与 comparisons 严格一致。
- 成功完成验证校验后，只有 `core_overlap` 写入一个 `rejected_solution` FailureRecord；distinct/partial/shared elements 不写 Failure Memory。
- 真实 smoke 只执行显式 opt-in 的 provider/verification 流程；历史官方匿名请求返回 HTTP 429，当前中转站已完成 Academic Graph、Recommendations 和一次真实 VERIFY_GAP；本轮未执行 VERIFY_SOLUTION real run。
- 真实 provider 的网络调用只在显式 opt-in test 中执行；默认 unit/integration tests 使用 fake provider。

### Unit tests

- provider 调用一次、cache 调用零次新 search。
- 全局 cache 跨 provider instance/trajectory 复用两类成功响应；cache hit 的 provider request count 和 scientific search count 均为 0。
- provider 失败不创建 cache entry，未知/损坏 cache metadata 显式报错。
- Gap `valid/partially_resolved/resolved` 的 state application。
- distinct/partial/core 边界和 component overlap 定位。
- full_solution 与 changed_units 的筛选输入不同但 output schema 相同。

### Integration tests

- fake provider + fake LLM 完成 Gap Verification 和 Solution Verification，检查 Memory record、State novelty、Failure Memory。

### Real LLM / network

默认不需要；默认禁止真实网络。可做 provider contract tests；真实 Semantic Scholar test 必须显式 opt-in。

### Minimal runnable example

给定 fixture paper pool，输出一个 GapVerificationResult 和一个 `partial_overlap` SolutionVerificationResult，检查 comparison/audit 与 core-overlap FailureRecord；真实 smoke 使用 `config/semantic_scholar.json` 中的 API key，通过 `verify-gap --semantic-scholar-config ...` 显式运行。若 key 缺失，必须停止在 provider acquisition，不伪造科学结果。

## M6 — Inspiration Retrieval（已完成）

### Objective

实现 Inspiration Need、Dynamic Entity Bridging、第一版 `ArxivInspirationSearchProvider`、联合查询、grounded/fallback 结果和 Inspiration Memory。

### Inputs / dependencies

- M3 Memory、M5 Verification context。
- `InspirationSearchProvider` interface、M5 state-store target preconditions。

### Files/modules to implement

- `src/research_agent/modules/inspiration_retrieval.py`
- `src/research_agent/retrieval/arxiv.py`：第一版真实 arXiv adapter
- `src/research_agent/prompts/inspiration_entities.md`、`inspiration_extract.md`
- `tests/test_m6.py`

### Acceptance criteria

- 从 Research Need 生成五个 Bridge Entity，并选择最多三个；实体类型受文档约束。
- 选中实体形成一次 combined query；真实 search 成功时 grounding=`arxiv`，不可用时 grounding=`none`。
- 最多三个 Inspiration Candidates，输出 transferable principle/adaptation hint，不直接改 State。
- 无 useful inspiration 时返回 `no_useful_inspiration`，不自动执行其他 Action。
- Call 1 一次完成 Need/实体生成/选择，Call 2 一次完成 grounded 或 fallback transfer；不增加第三次调用。
- arXiv search 最多一次、最多 20 篇；查询采用 trajectory-local exact normalized cache，cache hit 不增加 scientific search count。
- provider failure 不写成功 SearchRecord；successful empty search 可写空 SearchRecord；fallback 候选不得伪造 paper attribution。
- M6 只写 PaperMemory/InspirationMemory/trajectory，`ResearchState` before/after 相同。

### Unit tests

- entity selection 的 structural relevance/novel direction/searchability fixture。
- query 生成、search budget、cache/fallback。
- grounded 与 LLM-only candidate 的字段和 grounding 标记。
- rejected mechanism/previous entities 的相关 Memory 过滤。

### Integration tests

- fake provider + fake LLM 产出 InspirationResult，确认后置动作为 Targeted Revision，而不是 State 直接更新。

### Real LLM / network

默认不需要；默认禁止真实网络。本轮仅使用 FakeLLM/Fake provider 和注入式 Atom transport，未执行真实 arXiv 或 DeepSeek。

### Minimal runnable example

用 generic active-gap/current-solution fixture 生成三个可 grounding 的 Inspiration Candidate，并导出 InspirationRecord；fallback fixture 验证 provider 不可用时的 LLM-only 结果。

## M7 — Targeted Revision（已完成）

### Objective

实现 Revision Need、一次 LLM patch、deterministic merge、Component/Experiment stale 传播和 focused re-verification contract。

### Inputs / dependencies

- M1 Revision/State schemas、M3 Memory、M5 Verification、M6 Inspiration。
- 遵守已确认的 Component 状态：新组件 `ready`，修改后 `modified`，重新验证成功后 `ready`。

### Files/modules to implement

- `src/research_agent/modules/targeted_revision.py`
- `src/research_agent/models/revision.py`：`TargetedRevisionDraft` 与严格 patch 字段
- `src/research_agent/prompts/targeted_revision.md`
- `tests/test_m7.py`：generic precondition、patch、merge、stale、focus、Memory 和
  trajectory contract

### Acceptance criteria

- State preconditions、显式 gap/solution ID、unfinished Component 和 evidence gate
  在 LLM 前 deterministic 校验；失败不产生 LLM、State、Memory 或 trajectory mutation。
- Partial overlap 与 useful Inspiration 都可触发 revision；每次 revision 最多选择一个
  runtime Inspiration candidate，并使用 canonical `<record>:<index>` ref。
- 历史 Inspiration 不会自动成为当前 revision 输入；只有显式
  `inspiration_record_id` 指定的单个 found record 才可提供 candidates。证据 refs 只
  包含当前 verification refs 和实际 selected grounded candidate 的 paper ref。
- `TrajectoryRecorder.validate_next_step()` 在 Revision 写 Memory/State/event 前执行，
  非法 step 不调用 LLM 且不产生任何 mutation。
- 一次 structured LLM call 返回内部 `TargetedRevisionDraft`；输入只包含 active gap、
  完整 Current Solution、Revision Need、最新 verification、可用 inspirations 和相关
  failure memory。
- patch 外字段和 evidence refs 原值不变；未知 ID、重复/新增/删除/重命名、no-op、
  仅 rationale、changed_units 不一致的 patch 被拒绝。
- 成功 merge 后变化 Component 为 `modified`，core change 使未修改 Component 为
  `stale`，novelty 为 `unchecked`；core/component 变化按确定性规则传播 Experiment
  stale，并返回 focused/full re-verification focus。
- `unable_to_revise` 不产生半成品 State，写入 RevisionRecord 和 `failed_revision`
  FailureRecord；成功只写 revision memory，并固定下一动作
  `VERIFY_RELATED_WORK`，不在 M7 内调用验证。

### Unit tests

- 仅修改单个 Component、两个 Component、Component+rationale 和 core idea 的 patch。
- experiment 按 `what_it_validates` 的 Component ID 局部 stale，以及 core change 全量
  stale 的规则。
- deterministic merge、unknown/duplicate/no-op/rationale-only 拒绝、失败原子性、
  State diff、inspiration selection 和 strict output schema。

### Integration tests

- partial verification 或 useful Inspiration → revision patch → merge → changed_units/
  full focus fixture；另测 unable-to-revise 的 state-unchanged failure path。

### Real LLM / network

不需要；fake LLM 和 trajectory-local fake Memory，禁止真实网络。

### Minimal runnable example

对 generic Component overlap fixture 应用一个 patch，打印新 Solution、focus 和 stale
experiments；另运行一个 unable-to-revise fixture 检查 State 未变化。

## M8 — Experiment Design（已完成；本轮停止于 M8）

### Objective

实现基于 Seed Paper experiment context 的 claim-oriented 2–4 个实验设计、coverage check 和 revision-aware 更新。

### Inputs / dependencies

- M2 SeedPaperContext、M5 verification、M7 changed_units/stale state。
- M1 已确认的 canonical Experiment schema；不创建独立 `design` 字段。

### Files/modules implemented

- `src/research_agent/modules/experiment_design.py`
- `src/research_agent/models/experiment_design.py`：内部 `KeyClaimDraft`、
  `ExperimentProposalDraft`、`ExperimentDesignDraft`
- `src/research_agent/prompts/experiment_design.md`
- `tests/test_m8.py`

### Acceptance criteria

- LLM 前 deterministic 检查 State、latest Gap/Solution Verification、Components、
  explicit ID 和 trajectory step；失败不调用 LLM、不写 State/Memory/trajectory。
- 一次 structured LLM call 生成 internal Key Claims 与 initial/stale proposals；schema
  从 `ExperimentDesignDraft.model_json_schema()` 注入。
- 初始实验为 2–4 个并确定性分配 `E1...En`；stale redesign 只返回每个 stale ID
  恰好一次，复用其 ID；ready 实验原样保留，最终数量保持 2–4。
- Claim target 只允许 `core_solution` 或已有 Component；至少覆盖整体 solution 和一个
  Component，Claim/Experiment 双向 coverage 闭合，`what_it_validates` 具有 exact unit
  anchor。
- 至少一个 core effectiveness Experiment 使用精确 `Seed Method` baseline；必填质量、
  重复实验、unknown refs、extra fields 和非法 redesign 均拒绝。
- 成功只更新 `ResearchState.experiments`，记录 `experiments_designed` audit，Memory
  不写入；全 ready 时 deterministic no-op，LLM 和 trajectory 均为 0。
- 不执行任何 external search，不预测实验结果，不实现 Controller/Commit/Finalizer。

### Unit tests

- 缺少整体 Solution/机制 coverage、invalid Claim anchor、unknown claim ref 时失败。
- State/verification/step preconditions、initial E IDs、stale-only redesign、ready preservation
  和 final count contract。
- 必填字段、Seed Method baseline、重复实验、extra fields、no-op 和 State-only update。

### Integration tests

- generic distinct Solution + valid verifications + Seed experiment context → ready
  experiments；另覆盖 M7 revision 后 stale-only redesign。

### Real LLM / network

不需要；fake LLM；禁止 network。

### Minimal runnable example

从 generic SeedPaperContext 和 fixture Solution 生成两个实验设计及 coverage summary，
或对一个 stale Experiment 保持 ID 进行 redesign。

## M9 — Scientific Controller（已完成；本轮停止于 M9）

### Objective

在所有专门模块契约稳定后，实现 rule-guided routing、allowed action mask、budget-aware control 和必要时的 LLM Controller。

### Inputs / dependencies

- M3 State/Memory/trajectory。
- M4–M8 所有 module result。
- 固定 Action Space、Commit Gate 输入和方法文档 §2。

### Files/modules implemented

- `src/research_agent/controller/budget.py`
- `src/research_agent/controller/scientific_controller.py`
- `src/research_agent/prompts/controller.md`
- `src/research_agent/models/controller.py`：execution-only Context/Result contracts
- `src/research_agent/errors.py`：`ControllerRoutingError`
- `tests/test_m9.py`

### Acceptance criteria

- 唯一动作由规则直接返回，不调用 LLM。
- 多个合法动作才调用 LLM，且输出 action 必须在 mask 内。
- 固定 Action Space 只有 `SOLUTION_IDEATION`、`SEEK_INSPIRATION`、
  `VERIFY_RELATED_WORK`、`TARGETED_REVISION`、`DESIGN_EXPERIMENT`、`SWITCH_GAP`、
  `COMMIT`；RefineGap 是 Gap 模块内部动作，不新增 Component Verification。
- 搜索预算耗尽时禁止主动 Inspiration；已有 trajectory-local RelatedPaperPool 时
  `VERIFY_RELATED_WORK` 仍可在零 search budget 下执行。
- Step/search 由实际 action execution 更新；Controller decision 本身不消耗预算，cache
  hit 的 scientific search count 为 0，provider HTTP request count 不参与 budget。
- Solution Ideation、Targeted Revision、Switch Gap 都预留后续 mandatory
  `VERIFY_RELATED_WORK` 的一个 step；预算不足时不会启动这些动作。
- 主动 `SEEK_INSPIRATION` 还必须满足 `remaining_searches >= 1` 且
  `remaining_steps >= 3`，为 Inspiration found → Targeted Revision → Verification
  保留最坏路径；实际预算仍按每个已执行 Action 逐步扣除。
- budget exhausted 或没有合法动作时返回 `stopped` 与 deterministic unresolved issues，
  不包装为完整 proposal。
- Controller 只读取并返回 execution-only routing result，不修改 State/Memory/trajectory；
  `DecisionRecord` 只由 action 执行完成后的 helper 写入实际 result summary。

### Unit tests

- New Gap/no verification、valid Gap/no Solution、resolved Gap、unchecked/revised
  Solution 和 Inspiration-found 的 deterministic routes。
- partial overlap、distinct/stale/ready experiments、commit allowed 的 action mask。
- LLM 越权 action、dynamic schema、预算边界、mandatory verification reserve、
  zero-search verification 和 stopped/unresolved behavior。
- raw partial resolution 与 later `gap_refined` boundary，以及 no-refine/no-component-action
  contract。

### Integration tests

- 用 generic fake module results 跑多个 decision turns，验证 deterministic decisions、
  one-call multi-action routing、State unchanged、post-execution DecisionRecord 和 budget。

### Real LLM / network

不需要；Controller 使用 fake LLM 测试。

### Minimal runnable example

给定 partial-overlap State，fake Controller 在 revision/inspiration 两个允许动作中返回一个合法决策；
给定新 Solution 则无 LLM 直接返回 `VERIFY_RELATED_WORK`。给定成熟且 commit-eligible
State，在 search budget exhausted 时只返回 `COMMIT`。

### M9 method-boundary notes

- `ControllerContext` 只携带 active gap/current solution 的紧凑摘要、Component/Experiment
  status、最新 action/result 摘要、相关 failure memory、commit input、budget 和 action mask；
  不发送完整 Seed Paper、完整 Memory、完整 trajectory 或 paper-pool abstracts。
- `ControllerDecision` 的 JSON schema 由 `model_json_schema()` 注入 controller prompt；
  LLM 只能在 action mask 中选择一个动作，非法动作显式失败，不自动替换或重试。
- `ControllerRoutingResult` 是 execution-only artifact，不是 Research State。Controller
  不实现 Commit Gate、CommitSnapshot、Finalizer、runner 或任何隐式 planner/reflection loop。

## M10 — Commit + Finalizer（已完成）

### Objective

实现 deterministic Commit Gate、State freeze、CommitSnapshot 和 read-only Finalizer。

### Inputs / dependencies

- M1 Commit/Proposal schemas、M3 Memory、M5 Verification、M8 Experiments、M9 Controller。
- 使用已确认的 `get_latest_gap_verification(active_gap.id)` Memory query，以及 canonical Experiment schema。

### Files/modules implemented

- `src/research_agent/modules/commit.py`
- `src/research_agent/modules/finalizer.py`
- `src/research_agent/prompts/finalizer.md`
- `src/research_agent/models/commit.py`：`FinalizerDraft`
- `tests/test_m10.py`

### Acceptance criteria

- Gate 五项条件全部满足才允许 commit，并按固定顺序生成 deterministic
  `unmet_conditions`；State/最新 Verification 冲突显式抛出 precondition error。
- Gap/Solution 都使用实际 matching latest Verification；`partial_overlap`、`core_overlap`、
  unchecked、stale、modified 均不能 commit。
- Snapshot 必须自己重跑 Gate，保存 exact State units、latest Verification 和稳定去重 evidence
  refs，并与 State/Memory 深隔离。
- `CommitModule.commit()` 只记录 `research_committed`/`COMMIT` 空 diff、空 Memory writes 事件。
- Finalizer 一次 structured LLM call 只生成 narrative-only `FinalizerDraft`；Python exact
  materialize committed Gap、Solution、Rationale、Components/Experiments，Proposal 的
  novelty positioning 遵守 evidence level，贡献使用 expected/potential 语气，不报告实验结果。
- Snapshot 和 FinalResearchProposal 使用既有 schema/version envelope 持久化 helpers。

### Unit tests

- 每一个 Gate 条件单独失败、latest verification、冲突状态及全部通过。
- Snapshot 与 State/Memory/自身后续变化隔离，evidence refs 稳定去重，artifact round-trip。
- Commit step preflight、Gate failure、successful trajectory event 和空 diff/writes。
- Finalizer fake LLM 的动态 schema、2–3 contributions、read-only、compact input 和禁止新增
  Component/Experiment 校验。

### Integration tests

- distinct + ready experiments + valid gap → COMMIT → snapshot → final proposal；未满足条件则不得生成 Snapshot/Proposal。

### Real LLM / network

不需要；fake LLM；禁止 network。

### Minimal runnable example

对一个全通过 fixture 输出 CommitSnapshot JSON 和 FinalResearchProposal JSON。

### M10 method-boundary notes

- Commit Gate 不承担 maturity scoring，不调用 LLM，不修改 State/Memory/Verification。
- Finalizer 不能重新做 Related-work Verification；`abstract_only` novelty 只能使用限定性
  表述，`distinct` 不等于 zero overlap。
- Finalizer 输入不包含完整 Markdown、完整 Memory、trajectory 或 paper pool abstracts；
  M10 不实现 Runner、replay、real pilot 或 batch runner。

## M11 — Offline End-to-End Trajectory（已完成；本轮停止于 M11）

### Objective

把 Seed Context、Gap、Verification、Solution、Revision/Inspiration、Experiment、Controller、Commit 串成一条可回放 trajectory；不扩展方法 Action。

### Inputs / dependencies

- M0–M10 全部。
- 全部 fake LLM/provider fixtures。

### Files/modules implemented

- `src/research_agent/runner/trajectory_runner.py`：显式注入的 module bundle、Controller loop、固定 action dispatch。
- `src/research_agent/serialization/trajectory.py`：独立 `trajectory_replay.jsonl` journal、加载和无模块 replay。
- `src/research_agent/models/runner.py`：Replay/Manifest/RunResult 等 execution-only contracts。
- `src/research_agent/memory/store.py`：validated whole-memory replace，仅供 replay reconstruction。
- `tests/test_m11.py`：generic valid、partial-revision、inspiration-revision、gap-switch、refine、budget、failure、replay fixtures。

### Acceptance criteria

- Runner 只 bootstrap 一次 Gap Discovery，之后每一轮重新读取 State/Memory、确定性计算 Commit Gate、调用 Controller 并 dispatch 一个已有 Scientific Action；没有新增 Action、planner 或科学判断。
- `trajectory event step` 与 `ControllerBudget.steps_used` 分离；bootstrap、Controller audit、budget audit、RefineGap 和 Finalizer 不消耗 scientific step。
- 每次 Action 按 related/inspiration module cumulative counter 的 delta 计 scientific search；cache hit 为 0，provider HTTP request 不进入 scientific budget；失败的已开始 Action 仍消耗 step 和已经发生的 search delta。
- `VERIFY_RELATED_WORK` target、focused verification、InspirationRecord trigger 和 partial Gap Refine 均由 deterministic runner 解析；RefineGap 不写 DecisionRecord，事件 action 为 `None`。
- `trajectory.jsonl` 继续是 compact scientific audit；新增的 `trajectory_replay.jsonl` 才保存完整 `state_after` 和六类顶层 Memory canonical after-values，不伪装成 event sourcing。
- valid commit、partial-overlap revision、inspiration-guided revision、resolved-gap switch、partial refine、budget stop 和 action failure 均有 generic offline coverage；失败或预算耗尽不生成 fake proposal。

### Unit tests

- runner 状态机边界、固定 dispatch、bootstrap/action semantics、target/focus/trigger resolution、异常隔离、step/search 计数、DecisionRecord timing 和 replay schema validation。

### Integration tests

- fake LLM/provider 串联完整 commit 与未解决路径；committed/stopped/failed trajectory 均从 replay journal 重建出相同 State/Memory，replay 不调用 LLM/provider/controller。

### Real LLM / network

不需要；默认完全离线。

### Minimal runnable example

运行一个 fake trajectory，生成 `seed_context.json`、`research_state.json`、`research_memory.json`、`trajectory.jsonl`、`trajectory_replay.jsonl`、`run_manifest.json`；仅 Commit 成功时生成 Snapshot 和 Proposal。

## M12 — Real Single-Paper Integration Pilot

### Objective

使用一篇真实 `data/` MinerU Markdown，显式 opt-in 地运行真实 DeepSeek、真实 Semantic Scholar related-paper retrieval，以及 trajectory 需要时的真实 arXiv inspiration search，验证单篇完整链路。

### Inputs / dependencies

- M0–M11 全部，尤其是 M5/M6 的真实 adapters 和 M11 离线 trajectory。
- `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 及真实 provider 配置。
- 人工明确开启 real pilot；不得作为默认测试。

### Files/modules to implement

- `tests/real/test_single_paper_pilot.py` 或等价显式 opt-in runner。
- `src/research_agent/runner/pilot.py`：调用、预算、错误和 metadata 记录。
- `outputs/pilots/`：不覆盖原始 data 的 pilot outputs。

### Acceptance criteria

- 完整运行 `Markdown → SeedPaperContext → Agent trajectory → Commit/unfinished result`。
- 记录 LLM calls、search calls、token/cost metadata（可用时）、trajectory length、errors、final state，以及 final proposal 或 unresolved issues。
- 真实调用失败时保留可诊断输出，不伪造完成状态；pilot 明确标记是否通过 Commit Gate。
- 只有单篇 pilot 稳定后，M13 才可开始。

### Unit tests

- 不在默认 unit suite 中运行真实服务；只测试 pilot manifest、metadata/error serialization 和 opt-in guard。

### Integration tests

- 显式设置 real-pilot flag 且 credentials 存在时运行一篇指定 paper；没有 flag/key 时必须 skip，不得触网。

### Real LLM / network

需要真实 DeepSeek；需要真实 Semantic Scholar；只有 trajectory 选择 `SEEK_INSPIRATION` 时才需要真实 arXiv。全部显式 opt-in。

### Minimal runnable example

对一篇指定 Markdown 运行 pilot，生成 `outputs/pilots/<paper_id>/` 下的 manifest、trajectory、final state 和 proposal/unresolved result。

### Actual M12 attempts

- 第一次 ByteFlow real pilot 使用 `outputs/m12_real_byteflow/`，在 Gap Discovery 的
  structured output 非 JSON 后失败；该历史 artifact 保留，不覆盖。
- 按人工要求移除 pilot 对所有 DeepSeek 请求的本地 `max_tokens` 转发限制后，第二次
  使用 `outputs/m12_real_byteflow_unbounded/` 完成完整运行；固定 `max_steps=12`、
  `max_searches=2`，并移除 HTTP/HTTPS proxy 环境变量。
- 第二次运行 `committed`：10 次 DeepSeek call 全部成功，累计 tokens 为
  `49950/34733/84683`；trajectory events=22、replay records=6。
- Semantic Scholar 的 `paper_search` 和 `recommendations` 均从 global cache 命中，
  HTTP requests=0、related pool=20；arXiv 未被选择，HTTP requests=0。
- Decision sequence 为 `VERIFY_RELATED_WORK → SOLUTION_IDEATION →
  VERIFY_RELATED_WORK → DESIGN_EXPERIMENT → COMMIT`；Gap Verification=`valid`、
  Solution Verification=`distinct`，Commit Gate 和 Finalizer 均成功。
- replay 对最终 State、Memory 和 trajectory 的只读重建全部一致，M12 通过；本轮停止，
  M13 不启动，等待人工审核。

## M12.1 — Design-Rationale Preservation Hardening

### Objective

保留 Seed Paper 中作者明确的 design rationale、method constraints、trade-offs 和
rejected alternatives，使后续 Gap Discovery、RefineGap、FormSolution 与 Select+Develop
不会把已经说明理由的设计选择误当作未考虑过的方案。只做 paper-agnostic 的 evidence
preservation 和 constraint-consistency hardening，不改变科研模块、Action、状态、评分或
workflow。

### Inputs / dependencies

- M12 已通过的 SeedPaperContext、Gap、Solution、trajectory contracts。
- 完整 main-paper body 的 deterministic parser 和 one-call Seed extraction。
- M4/M5 已有 Gap/Verification/Failure Memory 输入边界；不读取 References/Appendix
  作为 evidence。

### Files/modules to implement

- `src/research_agent/models/seed.py`、`models/seed_extraction.py`：增加
  `method_constraints`；不增加 rejected-alternatives 等平行字段。
- `src/research_agent/modules/seed_context.py`、`gap_discovery.py`：materialize anchors
  并开放 `seed:method_constraints[i]`，继续排除 `seed:experiment_context.*`。
- `src/research_agent/modules/gap_discovery.py`：RefineGap 读取同一 Seed constraint
  context，不增加第二次 critic call。
- `src/research_agent/prompts/seed_context.md`、`gap_discovery.md`、`gap_refine.md`、
  `solution_form.md`、`solution_select_develop.md`：加入通用 rationale/constraint
  preservation rules。
- `tests/test_m12_1.py`：只使用机制 A/B、通用 stability constraint 等 synthetic fixtures。

### Acceptance criteria

- `SeedPaperContext.method_constraints` 只保存 author-explicit constraints/rationales/
  trade-offs/rejected alternatives and reasons；architecture fact alone 不生成字段。
- 每个 constraint evidence item 必须有 main-body `source_section_ids`；References/Appendix
  不能成为 anchor；final SeedPaperContext 不包含 audit。
- Gap evidence vocabulary 新增 indexed `seed:method_constraints`，不重新加入
  `experiment_context`；Gap/Refine prompt 包含 design-rationale consistency gate。
- Solution prompts 要求不直接重提 rejected alternative 或违反 constraint；若挑战选择，
  必须处理 rejection reason；仍严格为 FormSolution 3 candidates + Select/Develop 1 call。
- 不增加 `consistency_check`、critic/reviewer/reflection agent、Action、State、Memory
  或新的 verification loop；不写 paper-specific prompt/code/test。
- 真实 smoke 只作为 evaluation case，不能反向 special-case prompt；一次真实 regression
  pilot 采用新的输出目录，不覆盖 M12 artifacts。

### Unit tests

- Seed rationale/rejected alternative extraction、source-anchor legality、invalid source
  rejection、round-trip 和 canonical prompt text。
- Gap method-constraint evidence ref 合法、experiment ref 仍非法、whole-method/design-
  rationale prompt boundary、Refine constraint propagation。
- Solution exactly-three/exactly-two-call contract、constraint context injection 和
  rejection-reason preservation wording。

### Integration tests

- 完整 offline suite 串联 Seed → Gap → Refine/Verified Gap → Solution 的 synthetic
  trajectory；不调用真实服务。
- 一次真实 ByteFlow regression pilot：新的 Seed extraction、Gap Discovery 及后续完整
  M12 trajectory，仅用于评估约束 preservation/consistency。

### Real LLM / network

Unit/integration tests 默认不需要网络。M12.1 acceptance 需要恰好一次 fresh real
single-paper pilot；DeepSeek、Semantic Scholar/arXiv 是否请求由既有 Controller trajectory
决定。真实 LLM 首次调用后不得再改 prompt 或 rerun tuning。

### Minimal runnable example

离线运行 `pytest`/`unittest` 后，以新的 output directory 对既有 ByteFlow Markdown
执行 M12 pilot，检查 `method_constraints`、GapAgenda、Candidate Solutions、verification、
experiments、Commit/Finalizer、token metadata 和 replay；完成后停止，不进入 M13。

### Actual M12.1 regression result

- Fresh ByteFlow run: `outputs/m12_real_design_rationale_regression/`；status=`committed`，
  commit gate passed，10 DeepSeek calls，5 scientific actions，22 trajectory events，6 replay records。
- Seed extraction produced 6 method constraints in one call; Gap Discovery produced 3 gaps and
  selected G1. Solution Form/Select+Develop kept the existing 3-candidate/2-call contract and
  selected S1 with constraint-preserving rationale. Verification, experiments, Commit and Finalizer
  completed; no Inspiration/Revision was selected.
- Semantic Scholar `paper_search` and `recommendations` were global-cache hits with
  HTTP request count 0; arXiv request count 0. Token totals were prompt/completion/total =
  `52362/45966/98328`; replay equality passed.
- Offline acceptance was pytest `229 passed`、unittest `229 tests OK`、compileall passed。
  No docs method file or data file was modified. M13 remains pending and is not entered.

## M12.2 — Scientific Controller Inspiration Boundary

### Objective

在不增加 Action、Agent、reflection/critic loop、numeric score 或任何 State/Memory/科学
schema 字段的前提下，补足 Controller 对“已有可行 solution 仍值得探索外部机制”的通用
判断。`SEEK_INSPIRATION` 保持可选，不再因为 solution 存在就默认跳过，也不因 distinct
或 high-priority 单独出现就强制触发。

### Inputs / dependencies

- M9 Scientific Controller 的现有 deterministic action mask、budget 和七个 Action。
- 现有 Active Gap、CurrentSolution、Component 和 latest Solution Verification 字段。
- M12.1 已确认的 paper-agnostic anti-overfitting 约束和 M12 real pilot artifact。

### Files/modules to implement

- `src/research_agent/controller/scientific_controller.py`：增加 execution-only、
  paper-agnostic Inspiration opportunity gate；只在 gate 为真且 budget 允许时把
  `SEEK_INSPIRATION` 加入已有合法 Action 集合；明确新机制时不开放该可选分支。
- `src/research_agent/prompts/controller.md`：解释 gate 的 optional semantics、distinct/
  priority 不足以强制 Inspiration，以及新机制可直接进入 experiment/commit。
- `tests/test_m9.py` 或独立 Controller tests：generic synthetic scenarios A–D 和 prompt
  anti-specialization 检查。
- `ARCHITECTURE.md`、`PLAN.md`、`IMPLEMENTATION_STATUS.md`：记录边界、验证和 real
  evaluation；不修改 `docs/` 与 `data/`。

### Acceptance criteria

- Scenario A：local/incremental solution 时 Inspiration 可作为合法可选 Action。
- Scenario B：clear new mechanism 时 Inspiration 不被要求，若只有 experiment route 则
  deterministic 选择 `DESIGN_EXPERIMENT`。
- Scenario C：high-priority gap + incremental solution 时 Inspiration 成为候选。
- Scenario D：distinct solution 本身不自动触发 Inspiration。
- 多个合法 Action 时最多调用一次 Controller LLM；单一合法 Action 不调用 LLM。
- 不改变七个 Action、State/Memory/Gap/Solution/Verification schema；无 paper-specific
  prompt/code/test specialization。
- 离线 suite 全通过后只对既有 real pilot 做一次 evaluation regression，报告 Inspiration
  trigger、Action sequence 和 Commit success；不进入 M13。

### Unit tests

- deterministic gate 的四个 generic scenarios、budget/action mask 和 no-LLM single-action
  paths。
- prompt 包含 gate semantics，且不包含真实论文名称、专有机制或 special-case wording。

### Integration tests

- 完整 offline pytest 与 unittest suite。
- 使用现有单篇 real pilot input 做一次回归；不重新调参、不运行 M13 或其他论文检索流程。

### Real LLM / network

离线 tests 不需要真实 LLM 或网络。Acceptance 的 real regression 需要 DeepSeek；若本次
Controller 选择 Inspiration，既有 arXiv provider 才按现有 workflow 发生请求；Semantic
Scholar 继续使用已有缓存策略。真实运行只作 evaluation case。

### Minimal runnable example

离线测试后，对已有 ByteFlow Markdown 运行 `real-pilot` 到新的 M12.2 output directory，
比较 M12.1 的 action sequence、Inspiration trigger、Commit Gate 和 replay artifact。

### Actual M12.2 result

- Offline acceptance completed before the real run: pytest `234 passed`、unittest `234 tests
  OK`、`compileall -q src tests` passed。
- Fresh real output: `outputs/m12_2_real_controller_inspiration_regression/`; no previous
  M12/M12.1 artifact was overwritten. Seed extraction, Gap Discovery and Gap Verification
  completed; Semantic Scholar used the global cache (`paper_search` + `recommendations`)
  with HTTP requests `0` and arXiv requests `0`.
- The run then failed during the second `VERIFY_RELATED_WORK` Action, which performs
  Solution Verification: `The read operation timed out`. `llm_request_count=6` includes five
  successful calls and this failed sixth call; its `max_tokens` was omitted as required by
  the unbounded-output pilot. The failed call had `input_chars=47741`; token metadata was
  unavailable. Successful-call subtotal was prompt/completion/total `32091/28867/60958`;
  overall pilot totals remain unavailable because the run did not complete.
- Partial Action sequence was
  `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK`; the failed final action
  was another `VERIFY_RELATED_WORK`. No `SEEK_INSPIRATION` decision was reached, so this
  real run does not provide evidence that the trigger changed. Commit Gate was not reached
  (`commit_gate_passed=false`), and no CommitSnapshot/Finalizer artifact was created.
- The real result is retained as evaluation evidence only. No further prompt tuning, rerun,
  paper-specific rule, or M13 work was performed.
- A subsequent requested rerun used command-level `DEEPSEEK_TIMEOUT_SECONDS=180` and
  `DEEPSEEK_MAX_RETRIES=0`, still omitted `max_tokens`, and saved to the fresh
  `outputs/m12_2_real_controller_inspiration_timeout180/` directory. It completed
  `committed` with 10 successful DeepSeek calls, 5 scientific Actions, prompt/completion/total
  tokens `52764/31474/84238`, Semantic Scholar global-cache HTTP requests `0`, and arXiv
  requests `0`.
- In this successful evaluation, the new gate changed the legal action masks: at Experiment
  Design the Controller saw `DESIGN_EXPERIMENT + SEEK_INSPIRATION`, and at the mature state it
  saw `COMMIT + SEEK_INSPIRATION`. The Controller selected `DESIGN_EXPERIMENT` and then
  `COMMIT`; no Inspiration search was forced. The final sequence remained
  `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK → DESIGN_EXPERIMENT → COMMIT`.
  Commit Gate, Finalizer and replay reconstruction all succeeded; State, Memory and trajectory
  replay were equal. M13 remains unstarted.

## M13 — Batch Evaluation and Research Agent Evaluation Framework

### Objective

建立多篇 Seed Paper 的隔离评估与人工复核接口，验证现有单篇 workflow 在不同论文上的
执行表现；不修改任何科研 workflow 定义、Action、schema 或 Controller 边界。

### Inputs / dependencies

- M12 已稳定的真实单篇 pilot 和 M11 可回放的单篇 runner。
- `data/` 原始 Markdown 只读。

### Files/modules to implement

- `src/research_agent/models/evaluation.py`：Execution-only `EvaluationReport`、
  `PaperEvaluation` 和每 Gap 的可审计摘要。
- `src/research_agent/runner/batch_evaluation.py`：论文发现、随机子集、逐篇输出隔离、
  对既有 `RealSinglePaperPilot` 的注入和 metadata aggregation；当前实现保持串行。
- `src/research_agent/retrieval/semantic_scholar_cache.py`：cache miss 与请求写入的
  文件锁，避免未来并行时重复付费请求。
- `src/research_agent/retrieval/arxiv_cache.py` 与 `retrieval/arxiv.py`：Inspiration
  Search 的全局持久响应缓存及安全原子写入。
- `human_review_template.json`：人工填写的 Gap/Solution/Final Proposal 质量字段，
  不进入 workflow。
- `scripts/run_m13_batch_5_serial.sh`：5 篇随机子集真实执行脚本。
- `tests/test_m13.py` 及 M6 arXiv cache regression tests。

### Acceptance criteria

- 每篇 paper 有独立 State/Memory/trajectory/output；不跨 paper 共享 Memory。
- `evaluation_report.json` 记录选中的 paper IDs、每篇 action/status/gap/solution/
  inspiration/cost/replay metrics 及 failure type counts。
- 每篇 paper 的 State、Memory、trajectory 和 pilot artifacts 互相隔离；单篇失败不终止
  后续论文，也不污染其他 paper。
- 真实 batch 继续复用 `TrajectoryRunner`，不复制 workflow logic，不绕过 Controller。
- 单篇失败不会静默污染其他 paper；汇总包含失败原因和 unresolved issues。

### Unit tests

- paper id/output path 隔离、重复运行保护、EvaluationReport closed contract。
- one trajectory failure 不影响其他 fixture。
- arXiv global cache hit 不再触发 provider request；Semantic Scholar cache 保留两阶段
  provider 请求的全局复用。

### Integration tests

- 对 3 篇抽样 Markdown 执行 fake batch，生成汇总 JSON/JSONL；检查 data 原文未修改。

### Real LLM / network

默认不需要；真实 batch 另行通过显式 `--allow-real-services` 开启并记录成本，不属于
默认 acceptance；不得绕过 M12。当前批准的 5 篇脚本为串行版本；并行调度留待后续工程
优化，缓存锁已提前保证未来并行的 miss-check 安全。

### Minimal runnable example

离线运行三篇抽样论文，得到每篇 trajectory 和 batch summary，包含成功/失败/未解决条件；
真实运行由人工执行 `scripts/run_m13_batch_5_serial.sh` 后再检查。

### M13 current implementation result

- `BatchEvaluationRunner` 当前按固定 seed 从 `data/` 发现并选择 Markdown，逐篇调用现有
  `RealSinglePaperPilot`，输出至 `outputs/batch_eval/paper_id/`，不共享 State/Memory。
- CLI 为 `batch-eval`；真实服务必须显式传入 `--allow-real-services`，默认无网络路径。
- `EvaluationReport` 只保存 count/categorical metrics 和安全成本 metadata，不计算不可解释
  的总分、不引入 ranking 或新的科研状态。
- 5 篇执行脚本固定 `max_papers=5`、`selection_seed=1301`、DeepSeek timeout 默认 180 秒，
  输出目录可通过 `M13_OUTPUT_DIR` 覆盖；真实 rerun 已生成完整 report，结果为 3 篇 committed、
  2 篇 contract failure。
- 真实 batch 暴露的两个通用工程问题已修复：full-solution distinct verification 会确定性
  收敛所有已验证的 `modified/stale` Components；initial Experiment Design 会为 core
  effectiveness proposal materialize canonical `Seed Method` baseline，空 baseline 仍拒绝。
- 本轮 offline suite 为 `pytest 242 passed`、`unittest 242 tests OK`；未修改 method docs、
  原始 data 或真实历史 outputs。
- 本轮尚未进入并行实现；未来并行化必须保持每篇 executor/State/Memory/output 隔离，
  并继续使用 provider global cache lock。

## Correctness hardening before M13 continuation（非新 milestone）

本轮只修复通用 correctness 与 Controller routing contract，不扩展 M13 workflow，也不
重新运行五篇 batch。

### Scope and dependencies

- 依赖现有 M2 Markdown parser、M5 verification、M7 Targeted Revision、M9 Controller
  和 M11 trajectory runner。
- `docs/` 与 `data/` 保持只读；真实论文只能作为 evaluation case，不能成为 prompt 或
  scientific code 的特调目标。

### Required implementation

- `ingestion/markdown.py` 采用保守 Appendix parser：只识别显式 Appendix/Appendices/
  Supplementary/Supplemental markers，删除通用单字母 heading 触发；References 与
  Appendix 取最早 cutoff，未发现边界则保留完整正文。
- `Component.status` 表示 freshness（`ready`、`modified`、`stale`），与
  `CurrentSolution.novelty_status`（`unchecked`、`distinct`、`partial_overlap`、
  `core_overlap`）正交。`distinct`/`partial_overlap` 的 full verification 收敛所有
  pending modified/stale Components；changed-units 只能覆盖全部 pending modified，且
  stale 必须走 full-solution verification。
- Targeted Revision 继续使用 lightweight patch + deterministic merge，并将已有
  `SeedPaperContext.method_constraints` 作为结构化输入传入 prompt，保留作者明确的
  rationale、trade-off 和 rejected-alternative reason。
- 删除 Inspiration lexical gate。合法 Inspiration 只由既有 State、Verification、
  action mask 和 productive reserve 决定；多个合法 Action 才调用一次 Controller LLM，
  不增加 Action、Agent、score 或 schema 字段。

### Acceptance and tests

- Generic tests 覆盖 parser false positives、explicit tail boundaries、freshness 收敛、
  changed-units/stale invariants、重新验证后第二次 Revision、method constraint wiring、
  wording-invariant Inspiration mask、reserve 与 illegal action rejection。
- 运行 pytest、unittest 和 compileall；随后对全部 `data/*.md` 只读生成
  `outputs/parser_audit.json`，并报告 37 篇的 cutoff 与 suspicious counts。
- 离线通过后只对 AtlasKV 执行一次 fresh real single-paper regression，复用既有 cache，
  不强制 Inspiration，不进入 M13 batch。

### Completion gate

AtlasKV 结果必须是 committed 或 controlled stopped，且不能再次出现
`Targeted Revision cannot run before re-verification`。记录 action sequence、LLM/provider
metadata、cache/request count 和 replay equality 后停止，等待人工审核；M13 batch 是否
恢复由后续人工决定。

### Actual correctness-hardening result

- Offline acceptance passed with pytest `253 passed`, unittest `253 tests OK`, and compileall.
- `outputs/parser_audit.json` covers all 37 Markdown files: 37 References-first cutoffs,
  5 explicit Appendix detections after References, 0 Appendix-first cutoffs, 0 structural
  inferences, 0 no-cut papers, 0 suspiciously short bodies, and 0 title false positives.
- The canonical fresh AtlasKV regression is
  `outputs/real_atlaskv_correctness_regression_final2/`: `committed`, 10 successful DeepSeek
  calls, prompt/completion/total tokens `58684/33035/91719`, Semantic Scholar global-cache
  HTTP `0`, arXiv HTTP `0`, and replay equality for State/Memory/trajectory. Its high-level
  actions were `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK →
  DESIGN_EXPERIMENT → COMMIT`; both optional Inspiration masks were offered but not selected.
- Two additional transient real-run artifacts from tool-session recovery are retained for
  audit (`.../network` committed; `.../final` failed on a related-paper reference contract;
  the first no-proxy attempt failed at DNS). No further real run or M13 batch is started.

## Milestone 完成后的共同检查

- 运行完整 offline test suite。
- 检查 `docs/` 与 `data/` 没有修改。
- 检查所有模块仍使用结构化输入/输出，未直接 import `DeepSeekClient`。
- 检查新增 Action、状态、评分公式、检索图或复杂 Memory 机制；如有，停止并记录 open issue。
- 更新 `IMPLEMENTATION_STATUS.md`：完成项、测试结果、剩余风险和下一 milestone。
