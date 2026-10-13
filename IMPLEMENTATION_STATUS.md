# Implementation Status

## 当前阶段

M13 — Batch Evaluation and Research Agent Evaluation Framework 正在实现；当前已完成
离线 BatchEvaluationRunner、EvaluationReport、人工复核模板、provider cache hardening、
5 篇串行真实执行脚本，以及一次真实 5-paper batch artifact 检查。该 batch 中 3 篇
committed、2 篇因通用工程 contract 失败；本轮已完成对应修复和离线回归验证。M12.2
及之前的真实 pilot 均已保留历史/回归 artifact。

### M13 correctness hardening（本轮）

已修复上一轮 development regression 暴露的通用工程边界：Gap 单次 Refine 后的
second-partial 安全 reject/switch、嵌套 Inspiration `result.status` 解析、found →
TARGETED_REVISION deterministic handoff、Inspiration statistics 语义区分，以及
commit-ready 状态无新 evidence 时不重新开放普通 proactive Inspiration。RefineGap、
Action space、State/Memory schema 和 Controller Prompt 未被扩展；仅加强了 Gap
residual wording、second verification 语义和 Inspiration title/abstract grounding。

离线验证：pytest `263 passed`，unittest `263 tests OK`，compileall 通过。一次受限
网络 regression 的 5 篇均因 DeepSeek DNS 解析失败而在 Seed extraction 阶段失败；
随后主机网络尝试完成第 1 篇 Seed Context，但后续 DeepSeek 请求长期无响应，已停止，
因此没有可用于 correctness 判断的完整 5-paper regression report。未运行或修改正式
P001–P040 evaluation workflow。
第一次真实尝试曾因 Gap Discovery 返回非 JSON 而失败；随后按人工要求移除 pilot 的
LLM `max_tokens` 转发限制，第二次运行成功 `committed`。M12 本轮通过，停止于 M12，
未进入 M13。此前 M9 Scientific Controller 已通过人工审核，本轮先修复了
`SEEK_INSPIRATION` 的 productive reserve。M8 stale-redesign Claim semantics、M7 的
explicit Inspiration trigger、selected evidence provenance 和 trajectory-step preflight
contract 也已补齐。

## 已完成内容

- 完整阅读 `docs/method_overview.md`。
- 完整阅读 `docs/implementation_spec.md`，覆盖 Research State、Controller、Gap、Solution、Verification、Inspiration、Revision、Experiment、Memory、Commit + Finalizer 的字段、流程和接口草案。
- 检查 `data/`：当前实际为 37 个 MinerU Markdown 文件；抽样阅读：
  - `MinerU_markdown_10107_ByteFlow_Language_Modeli_2090055877820440591.md`
  - `MinerU_markdown_13056_Bayesian_Attention_Mecha_2090055813601447937.md`
  - `MinerU_markdown_14690_Cautious_Optimizers_Impr_2090055813601447940.md`
- 确认数据中的标题/摘要/章节、HTML 表格、混合 LaTeX/HTML 公式、图片占位、references 和 MinerU artifact 情况。
- 创建架构草案、长期工程约束和分 milestone 实施计划。
- 应用人工确认的 7 项 method/contract decisions：Gap/Verification 边界、Component 状态、canonical Experiment、CurrentSolution.status、RefineGap、all-rejected candidates、latest Gap Verification query。
- 按人工裁决将计划调整为 M0–M13，并加入 Semantic Scholar adapter、arXiv adapter 和 Real Single-Paper Integration Pilot。
- 完成 `pyproject.toml`、最小 `src/research_agent/` package、环境配置、`LLMClient`、`DeepSeekClient`、`FakeLLMClient`、JSON/JSONL serialization、基础 errors/logging 和 offline CLI。
- 完成 16 个 M0 测试：configuration、structured LLM contract、fake LLM、DeepSeek adapter 的 injected-transport/error behavior、serialization 和 CLI smoke。
- 完成 M1 Pydantic v2 canonical schemas：SeedPaperContext、ResearchState、Gap/GapAgenda、Solution/Component/Experiment、Verification、VerificationFocus、Inspiration、Revision patch、ResearchMemory records、ControllerDecision、Commit 和最小 Trajectory contract。
- M1 所有科学 schemas 继承 `StrictModel`，使用 `ConfigDict(extra="forbid")`；保留 `GapAgenda` 独立于 State/Memory、`CurrentSolution` 组件 2–5 个、canonical Experiment 无 `design`、typed revision patch 等人工裁决。
- 修复四个 M1 contract：`novelty_summary` 允许 `None`；`VerificationRecord` 强制 task/result/focus 一致；`InspirationResult` 强制 found 为 1–3 个、no-useful 为 0 个；`RevisionResult` 强制 revised/unable patch 自洽。
- 扩展 JSON/JSONL codec，使用 Pydantic v2 `model_dump(mode="json")` 并保留 M0 schema/version envelope。
- 完成 M1 contract/serialization tests；与 M0 合计 24 个测试通过后再进入 M2。
- 完成 deterministic MinerU Markdown parser：title/abstract、heading hierarchy、section spans、references/appendix boundary、HTML table/equation/image blocks 和 full main-paper body context。
- 删除 Markdown → SeedPaperContext 的 relevance selection、`SeedContextChunk`、`build_seed_context_chunks`、`max_chars`、自动分块、chunk extraction 和 synthesis 分支。
- 完成只依赖 `LLMClient` 的 one-call full-body extraction pipeline；只删除 References/Bibliography 与 Appendix/Appendices/Supplementary Material 中最早出现的尾部边界之后内容，不做 silent truncation。
- 修复 `seed_context.md`：运行时从 `SeedContextExtractionDraft.model_json_schema()` 注入 canonical JSON schema；CLI 不再提供 `--max-chars`。
- 修复真实 structured-output 稳定性：默认 output budget 调整为 8,000 tokens；prompt 要求 deterministic `paper_id/title/abstract` 返回 `null`，并要求科学归纳简洁、不复制长段落，降低 JSON response 截断风险；未改变 schema 或 one-call 方法。
- 完成 M2 offline tests：FakeLLM extraction、完整正文保留、tail 排除、one-call/无 synthesis、超长正文不切分、source evidence 校验、dynamic prompt schema、audit/output round-trip 和 MinerU artifact。
- 完成 `ResearchStateStore`：只保存最新经过 Pydantic validation 的 `ResearchState`，提供 `get_state()`、`replace_state()`、`snapshot_state()` 和 JSON persistence；不实现 scientific transition rules。
- 完成 trajectory-local `ResearchMemoryStore`：严格使用现有 `ResearchMemory` categories，提供 paper upsert、related pool seed 校验、各类 save 接口、latest verification、latest gap verification 和简单 recency/type/ref queries；不引入 GapAgenda、vector store 或 memory agent。
- 完成 `compute_state_diff()`：输出 deterministic JSON-serializable leaf field paths，支持 Component description、novelty status 和 Experiment status 变化审计，不支持 reverse/apply。
- 完成 append-only `TrajectoryRecorder` 和 `TrajectoryContext`：顺序校验、GapAgenda/RefineGap helpers、memory writes/state diff/budget/error metadata、JSONL persistence；不实现 event-sourcing 或 replay engine。
- 完成 M3 离线测试：M0/M1/M2 原有 36 tests 加 M3 runtime tests，共 `50 passed`；Pydantic output validation 与 state/memory/trajectory round-trip 通过。
- 完成 M4 Gap Discovery and Management：SeedPaper-only one-call candidate generation、严格 evidence reference 校验、GapAgenda、deterministic active selection、bounded RefineGap 和 deterministic SwitchGap。
- 完成 M4 Solution Ideation：valid Gap Verification 前置条件、FormSolution exactly 3、Select + Develop、2–5 Components、deterministic `S<number>`/`C<number>` IDs、candidate pool reuse 和 all-rejected replacement。
- 新增 M4 strict draft/result models：`GapCandidateDraft`、`GapDiscoveryDraft`、`GapRefinementDraft`、`GapDiscoveryResult`、`CandidateSolutionDraft`、`FormSolutionsDraft`、`SolutionDevelopmentDraft`、`SolutionIdeationResult`；所有 LLM draft `extra="forbid"`。
- 新增四份 M4 task-specific prompts，并在运行时从各 Pydantic draft 的 `model_json_schema()` 注入实际 JSON schema；Gap、Solution prompt 均明确 JSON-only 和科研边界。
- 新增显式 `discover-gaps` CLI：读取已存在的 ByteFlow `SeedPaperContext`，输出 Gap result、Research State、Research Memory、trajectory 和安全的 LLM metadata audit；不重新执行 M2。
- 完成 M4 离线测试：总计 `67 passed`，覆盖 Gap、Refine、Switch、Solution、Failure Memory 过滤、前置条件和 trajectory integration。
- 完成一次 ByteFlow Gap Discovery real DeepSeek smoke：只调用 Gap Discovery，未执行 Gap Verification、Solution Ideation 或论文检索。
- 根据真实 ByteFlow 输出修正 Gap Discovery boundary：Gap 必须能够自然导向方法/机制的实质修改、扩展、替换或重设计；纯 experimental/evaluation gap 不再是合法 Candidate Gap。
- 修改 Gap evidence vocabulary：初始 Gap 只接受 `seed:research_objective`、`seed:core_method`、`seed:main_components[*]`、`seed:main_findings[*]` 和 `seed:limitations[*]`；`experiment_context.*` 仍保留在 SeedPaperContext，但只供后续 Experiment Design。
- 此前更新 Gap Discovery 与 RefineGap prompts，明确 method-level requirement、missing experiment/evaluation rejection、observed method weakness 与 missing experiment 的区别；本轮已将 Evidence Sufficiency 改为支持 grounded opportunity，不再把 0 Gap 作为正常成功结果。
- 新增 M4 boundary tests：experiment refs rejection、method evidence acceptance、pure evaluation prompt semantics、RefineGap method-level wording；offline suite 共 `70 passed`。
- 按新 boundary 重新运行一次 ByteFlow Gap Discovery real smoke；仍只执行一次真实 DeepSeek request。
- 从真实运行中抽象出通用 failure pattern：模型可能把 design fact 当成 confirmed limitation、只沿局部组件推断系统级失败、忽略其他路径的 compensation/mitigation/bypass，或在没有 direct failure evidence 时使用过强因果措辞。
- 更新 `gap_discovery.md`：加入 explicit limitation/future direction、observed weakness、trade-off/constraint、grounded opportunity 的来源层级和 Conservative Method Opportunity fallback；保留 Evidence Sufficiency、Whole-Method Consistency、Confidence Calibration，并要求 1–5 个候选。
- 更新 `gap_refine.md`：要求 evidence-consistent、whole-method-consistent、无 unsupported causal claim，partial mitigation 只能保留可辩护的 residual issue。
- 未增加 `consistency_check` 字段、Gap Critic、Reviewer、Reflection Agent、second-pass verifier 或其他新科研模块；Gap Discovery 仍为 one-call structured LLM operation。
- 在 `tests/test_m4.py` 新增 generic synthetic boundary tests：explicit future direction、constraint without demonstrated failure、compensating alternative path、low-confidence/high-priority independence、1–5 count 和 0 candidate rejection；offline suite 当前共 `77 passed`。
- 在 `AGENTS.md` 增加长期 anti-overfitting 约束：real-paper smoke 只能是 evaluation case，Prompt 改进必须先抽象为 paper-agnostic principle。
- 按新的人工裁决扩展 `seed_context.md`：`limitations` 现在从完整 main-paper body 的任意正文位置抽取作者明确写出的 limitation、shortcoming、trade-off、boundary、future work 和 improvement/extension direction；不增加 SeedPaperContext 字段，不允许模型自行发明 limitation。
- 将 `GapDiscoveryDraft.gaps` 改为 Pydantic `min_length=1, max_length=5`；正常初始 Gap Discovery 不再接受空候选。
- 将 active/switch selection 改为直接选择模型按 worth-exploring 顺序给出的第一个 candidate，low confidence 不再是 admission gate。
- 实现 `RelatedPaperProvider`、`FakeRelatedPaperProvider` 和官方 Semantic Scholar `SemanticScholarProvider`；通过 Academic Graph 标题解析 Seed Paper，再使用 Recommendations API 获取最多 20 篇 related papers。
- 实现 provider 配置、timeout/retry、注入式 GET transport、normalized-title exact/conservative match、ambiguity/no-match、malformed/429/timeout 和 missing-abstract 的显式处理；不提供其他检索 fallback。
- 实现 trajectory-local related-paper pool cache；PaperRecord 去重/upsert，SearchRecord 记录 provider、local seed ref、resolved provider paper id、operation、返回 refs、step、cache 和 provider HTTP request count。
- 实现 `RelatedWorkVerificationModule` 的 `VERIFY_GAP`/`VERIFY_SOLUTION`，每个 task 一次 structured LLM call；Gap 只写 VerificationRecord，Solution 在结果校验后做 distinct/partial/core 的 deterministic State transition，支持 focused changed_units。
- 新增 `verify_gap.md`、`verify_solution.md`，运行时从现有 result models 注入 canonical JSON schema；未修改 M4 prompts、Gap schema 或 M4 scientific logic。
- 修复 `VERIFY_SOLUTION` schema injection：prompt 与 `LLMRequest.output_schema` 现在都使用 `SolutionVerificationDraft.model_json_schema()`，覆盖 top-level `comparisons` 与 nested `result`。
- 修复 `VERIFY_SOLUTION` 的调用前 State precondition：显式 solution 必须与 `ResearchState.current_solution.id` 一致；不一致时在 LLM、provider、Memory、State 和 trajectory mutation 前抛出 `PreconditionError`。
- 新增 M5 provider/cache/Gap/Solution/integration tests；完整 offline suite 当前为 `84 passed` / `84 tests OK`。
- 新增显式 `verify-gap` CLI；本次真实输出失败 artifact 为 `outputs/real_verify_gap_m5/verification_error.json`，没有伪造 GapVerificationResult。
- 补齐 Semantic Scholar provider 的请求 pacing、429/5xx/timeout/connection 有限重试、`Retry-After`/bounded backoff、永久错误与 malformed response 分类；`min_request_interval_seconds` 通过 JSON 配置读取，clock/sleep 可注入以保持离线测试无真实等待。
- Semantic Scholar 运行参数继续从本地 JSON 文件 `config/semantic_scholar.json` 读取；提供 `config/semantic_scholar.json.example`、默认 CLI 路径和 `--semantic-scholar-config` 覆盖选项。若 JSON 的 `api_key` 为空、空白或缺失，则回退读取 `SEMANTIC_SCHOLAR_API_KEY`；非空 JSON 值优先。真实配置文件加入 `.gitignore` 并设置为仅 owner 可读；API key 不进入源码或 audit metadata。
- 增加 `auth_mode`：`x-api-key` 兼容官方 API，`bearer` 发送 `Authorization: Bearer <token>` 以兼容当前中转站示例；当前配置使用 `https://s2api.ominiai.cn/s2`、`bearer` 和 `cache/semantic_scholar_proxy`，真实 key 仅保存在本地配置文件且只检查 configured。
- 保留两次早期失败 smoke artifact：代理环境下的请求和沙箱内移除代理环境变量的请求均在 transport/DNS 层失败；它们不是当前 host-network 结果。
- 新增独立持久化 `SemanticScholarCache`：默认写入 `cache/semantic_scholar/`，按 operation + request parameters 内容寻址，分别保存成功的 Academic Graph search 和 Recommendations JSON 响应。缓存跨 provider instance、trajectory 和 CLI 运行复用；失败或 malformed response 不保存成功缓存，缓存文件不包含 API key。
- provider 记录 global cache hit、partial hit、命中的 operation、cache directory 和实际 HTTP request count；完整 global cache hit 不增加 scientific search count，partial hit 只对未缓存请求发起 API 调用。
- 修正科学搜索计数：cache miss 在 related-pool acquisition 开始时恰计一次，即使 provider 失败；cache hit 计零；provider HTTP request、retry 和多个 endpoint 请求只计 provider request count，不重复计 scientific search。
- 为 `VERIFY_SOLUTION` 增加内部 `SolutionVerificationDraft`，一次 structured LLM call 同时返回最多五个合法去重的 `ComparisonRecord` 和嵌套 `SolutionVerificationResult`；比较记录只持久化到 `solution_verified` trajectory/audit，不进入 State 或 `VerificationRecord.result`。
- 加强 Solution comparison contract：comparison paper refs 必须存在于 PaperMemory、无重复、最多五个，且必须与最终 `related_papers` 严格一致；novelty evidence 与最终 novelty status 必须一致。
- `core_overlap` 在完整验证校验后确定性写入一个 `rejected_solution` FailureRecord；distinct、partial_overlap 和 shared elements 不写 rejected-solution failure。
- 在 host-network 环境中移除 HTTP proxy 环境变量后重新执行 ByteFlow Academic Graph title resolution：Bearer 请求成功，`/s2/graph/v1/paper/search` 返回可解析 JSON，provider request count=1、retry count=0，解析到 provider paper id `dfc6c2d34f329d10347604a17f2c616a52c3c0f7`。
- 成功响应已写入全局缓存：`cache/semantic_scholar_proxy/paper_search/15ee8ff7ee4d9cb5367cca14bf99f4a1a74b49cbeb573c87286340e510a4f8f5.json`；本次为 cache miss，后续相同请求可由 global cache 命中而不重复请求 API。
- 按中转站提供的 autocomplete 示例执行一次直接请求：HTTP 200、返回 `matches` JSON 字段；完整 ByteFlow 标题返回 0 matches。该 endpoint 已证明 Bearer/网络可用，但项目当前 title resolution 使用 `/graph/v1/paper/search`，不是 autocomplete。

## 尚未开始内容

- M12 real single-paper pilot 已实现并完成成功 ByteFlow 运行；M13 batch runner 尚未实现。
  M5 Related-work Verification/Semantic Scholar、M6 Inspiration Retrieval、M7 Targeted
  Revision、M8 Experiment Design、M9 Scientific Controller、M10 Commit/Finalizer 与
  M11 Offline Runner/replay 已实现。
- M4 不实现 Gap Verification；此前官方 Semantic Scholar provider smoke 因 HTTP 429 停止，未伪造结果，也未执行 Solution Ideation。当前中转站已完成 ByteFlow Academic Graph + Recommendations related-pool smoke，并已使用现有缓存完成一次真实 `VERIFY_GAP`；尚未执行 `VERIFY_SOLUTION` real run。
- Semantic Scholar provider 已实现并完成 ByteFlow 的真实 title resolution 与 Recommendations 检索；M6 的 arXiv adapter 仅完成离线 transport 测试，本轮未调用真实 arXiv API。
- 除本次 ByteFlow 单篇 M12 real pilot 外，尚未运行其他真实 LLM/网络测试；本次
  pilot 的第一次真实调用已经发生，按单次尝试约束不自动重跑。

## 已解决的人工裁决

- Gap Discovery and Management 负责 gap generation/evaluation/selection/refine/switch；Related-work Verification 负责 VerifyGap/VerifySolution；不建立独立 Gap Verification module。
- Component 只允许 `ready`/`modified`/`stale`；新生成组件为 `ready`，novelty 仍为 `unchecked`。
- canonical Experiment 不含 `design`，使用 `objective`、`setup`、`baselines[]`、`metrics[]`、`what_it_validates`、`validation_criterion`、`status`。
- CurrentSolution.status 只使用 `developed`；core overlap 后 candidate=`rejected`、current_solution=`null`，历史写入 Memory。
- RefineGap 保持 gap id，通过 `gap_refined` trajectory event 和 `has_refined_gap(gap_id)` query 限制一次。
- 所有候选 rejected 且无 Current Solution 时，合法 Action 为 `SOLUTION_IDEATION`，重新生成三个候选，不进入 Inspiration。
- Commit Gate 通过 `get_latest_gap_verification(active_gap.id)` 查询 Memory，不增加 State 字段。

当前没有新增 method-level open issue。早期失败 artifact 来自受限沙箱、代理/DNS 路径或官方匿名 API 429；host-network 中转站 Bearer Academic Graph、Recommendations 和一次真实 `VERIFY_GAP` 均已成功。真实 VERIFY_GAP 使用 abstract-only evidence，未修改结果、未继续 Solution Ideation；若后续 provider/LLM context 或服务限制无法满足请求，必须显式报错并等待人工调整，不改变科研定义。

## 已确认技术选择

- 输入从 MinerU Markdown 开始，不实现 PDF → Markdown。
- SeedPaperContext 由 Markdown 结构解析 + LLM 结构化抽取形成；缺失字段使用 `None`/`[]`，不得凭常识补全。
- 所有科研模块依赖 `LLMClient`；DeepSeek 只通过 `DeepSeekClient(LLMClient)` 接入。
- 从 M2 开始的 DeepSeek structured-output prompt 必须明确 JSON 输出要求、expected schema/structure 和合理 `max_tokens`；M1 不新增 Prompt abstraction。
- DeepSeek 的 API key、base URL、model name 和运行参数继续通过环境变量/配置读取；Semantic Scholar 的 API key、base URL、auth mode、timeout/retry 和 pacing 统一通过本地 JSON 配置读取，源码不写死。
- 外部论文检索使用 `RelatedPaperProvider` / `InspirationSearchProvider` 抽象；M5 已实现 Semantic Scholar related-paper provider，M6 已实现 provider-neutral 的 arXiv Inspiration adapter。
- Related Paper Pool 继续按 Seed Paper trajectory 缓存；Semantic Scholar API 响应另有独立的持久化 global cache，Research Memory 仍保持 trajectory-local，第一版不使用向量数据库或 memory agent。provider HTTP request count 与 scientific search count 分开记录。
- Revision 使用 lightweight patch + deterministic merge；Finalizer 只读 CommitSnapshot。
- 默认 unit/integration tests 使用 fake LLM/provider，禁止真实 API/网络。
- Real Single-Paper Integration Pilot 必须在 Batch Runner 前，并记录调用、成本 metadata、trajectory、final state 和 proposal/unresolved issues。
- M0/M1/M2/M3 测试在 `myrag` conda 环境中运行；Pydantic v2 是 canonical schema 的运行时依赖，pytest 作为测试依赖，当前也通过标准库 unittest 验证。
- M2 `SeedContextConfig` 只保留 `max_tokens`（输出 token budget）和 `audit_excerpt_chars`；不再存在 `max_chars` 或 `SEED_CONTEXT_MAX_CHARS`，`input_chars` 只记录调用 metadata/audit，不能触发切分。
- M2 的真实 LLM 入口是显式 `extract-seed-context <markdown>`；默认测试和 parser-only smoke 不触发网络。
- 只保留文档规定的 Scientific Action Space、状态和控制逻辑；工程层只增加必要的 schema、adapter、serialization、logging、caching 和 test seams。
- M4 Gap Discovery is method-improvement oriented；Gap evidence refs 只使用实际非空的 objective/core method/components/findings/limitations，不使用 `experiment_context.*`；GapAgenda 只写 trajectory，不进入 Research State/Research Memory。
- 实验结果只有在揭示方法 weakness/failure/bottleneck 时才能支持 Method Gap；缺失实验、baseline、metric、scale test 或 evaluation coverage 本身不构成 Gap。Gap 也可以来自 author-stated limitation/future direction、trade-off、constraint 或 grounded method opportunity；正常成功输出必须为 1–5 个。
- M4 Gap Discovery 要求四个 reasoning boundary：method-level actionability、evidence sufficiency、whole-method consistency、calibrated confidence。完整 supplied method evidence 必须先检查 compensation、mitigation、bypass、contradiction 和 severity weakening；design fact 不是自动 limitation，priority 与 confidence 独立。
- Real-paper smoke 只能作为 evaluation case；任何 Prompt 改进必须先抽象为 paper-agnostic failure pattern，不得加入论文名称、专有术语、组件、算法、公式、heuristic 或 special case。
- M4 active selection 使用人工裁决的顺序规则：直接选择模型按 worth exploring 排序后的第一个 candidate，包括 low-confidence candidate；不增加 score/rank/weighted formula。
- M4 RefineGap 只接受 `partially_resolved`，每个 gap 由 trajectory `has_refined_gap()` 限制一次；SwitchGap 不调用 LLM、保持 agenda 顺序并清空旧 solution state。
- M4 Solution Ideation 只读取 `rejected_solution`/`rejected_mechanism` Failure Memory，latest Gap Verification 必须为 `valid`；Current Solution 形成后 next action 固定为 `VERIFY_RELATED_WORK`。
- M7 Targeted Revision 的 Inspiration 必须由本次 execution 的 `inspiration_record_id` 显式触发；历史 Inspiration 不自动注入。selected candidate 之后才可 deterministic 增加其 grounded paper ref，且 trajectory step 必须先通过无副作用 preflight。
- M8 Experiment Design 只在 active gap、distinct Current Solution、全部 ready Components、latest `VERIFY_GAP=valid` 和 latest `VERIFY_SOLUTION=distinct` 同时满足时运行；不搜索论文、不写 Experiment Memory。
- M8 使用内部 `KeyClaimDraft`、`ExperimentProposalDraft` 和 `ExperimentDesignDraft`；canonical `Experiment` 不增加 claim/dependency/design/result 字段。初始设计确定性分配 E IDs，stale redesign 只更新 stale IDs，ready Experiments 原样保留。
- M8 的每个 Key Claim 和 Experiment 必须覆盖闭合，至少有 core_solution claim、Component claim 和 `Seed Method` baseline；`what_it_validates` 必须包含 exact unit anchor。
- M8 initial mode 要求 2–4 Claims；stale redesign mode 允许 1–4 Claims，且新 Claims 只映射 stale redesign proposals。ready Experiments 不因复用相同 Component/core anchor 而被重新解释为新 Claim coverage；audit payload 记录 `design_mode`。
- M9 Scientific Controller 只选择固定七个 Scientific Actions，不新增 `REFINE_GAP` 或 `VERIFY_COMPONENT`，不执行科研 modules，不修改 State。RefineGap 是 Gap 模块内部动作，partial resolution 若没有后续 `gap_refined` event 会显式失败。
- M9 采用 deterministic-first routing：唯一 legal action 不调用 LLM，多个 legal actions 才进行一次 Controller structured call；非法 action 不替换、不重试。
- M9 `ControllerContext` 只发送 compact state/trajectory/failure/budget/action-mask summaries；不发送完整论文、Memory、trajectory 或 paper-pool abstracts。
- M9 `BudgetTracker` 只在 action 实际执行后更新 step/search；cache hit 的 scientific search count 为 0，provider HTTP request count 不计入预算。Solution Ideation、Targeted Revision、Switch Gap 预留 mandatory `VERIFY_RELATED_WORK` step。
- M9 `record_decision_result()` 只在 action 执行完成后写入真实 `DecisionRecord.result_summary`，Controller select 本身不写 pending decision。
- M9 主动 `SEEK_INSPIRATION` 必须满足 `remaining_searches >= 1` 且 `remaining_steps >= 3`，为 found → Revision → Verification 保留最坏路径；已 found 的 Inspiration 后续 Revision 仍只需 2-step reserve。
- M10 Commit Gate 完全 deterministic，不调用 LLM/retrieval/network，不修改 State/Memory/Verification；五项条件按 fixed stable order 生成 `unmet_conditions`，`commit_allowed` 严格等价于 unmet 为空。
- M10 必须使用 matching latest Gap/Solution Verification；State novelty 与 latest Solution Verification 冲突时抛出 `PreconditionError`，不降级成普通 Gate failure。
- M10 `create_commit_snapshot()` 内部重新运行 Gate；Snapshot 只包含 committed gap/solution/experiments/latest verification/evidence refs，不包含 candidate pool、full Memory、trajectory 或 budget，并保证 deep structural isolation。
- M10 `CommitModule.commit()` 成功记录 `research_committed`/`COMMIT`，State diff 与 Memory writes 为空；Gate failure 或 invalid step 不记录事件。
- M10 `FinalizerDraft` 只包含 `novelty_positioning` 和 2–3 条 expected contributions；CommitSnapshot 中的 Gap、Solution、Rationale、Components、Experiments 由 Python exact materialize，Finalizer 不允许改写或预测实验结果。
- M10 提供 `commit_snapshot.json` 与 `final_research_proposal.json` 的 schema/version envelope persistence helpers；不实现 Runner 或 retry loop。

## M11 Offline End-to-End Trajectory

### 已完成

- 新增 `TrajectoryModules` 与 `TrajectoryRunner`。Runner 通过 dependency injection 接收已有科研模块和 Controller，不构造真实 DeepSeek、Semantic Scholar 或 arXiv client，也不增加 Scientific Action、planner 或 reflection loop。
- Runner 启动时只做一次 Gap Discovery bootstrap；bootstrap 不写 DecisionRecord、不增加 scientific budget。之后每轮重新计算 deterministic Commit Gate，调用 Controller，并且只 dispatch 一个固定 Action。
- 固定 dispatch 已覆盖 Solution Ideation、Inspiration、Gap/Solution Verification target、Targeted Revision、Experiment Design、Switch Gap、Commit + Finalizer。VERIFY target、changed-units focus 和当前 InspirationRecord trigger 都由 runner deterministic resolution 完成。
- trajectory event step 与 `ControllerBudget.steps_used` 解耦。`controller_decided`/`budget_updated` 和 Gap 内部 Refine 的 action 为 `None`；RefineGap 不增加 step、不写 DecisionRecord，并保证 refine event 晚于原 Verification。
- 搜索预算使用 M5/M6 module cumulative `scientific_search_count` 的 action delta；cache hit 为 0，provider HTTP request 不进入 scientific search budget。已开始但失败的 Action 仍消耗 step 及实际 search delta，dispatch 前 Runner invariant failure 不伪装 Action。
- Finalizer 只在 CommitModule 成功创建 Snapshot 后调用一次；budget stop、unresolved gap、revision/verification failure 和 action failure 都不会生成 fake proposal。Commit 后 Finalizer failure 保留 Snapshot，不 rollback。
- 新增 `TrajectoryReplayRecord`、`RunManifest`、`TrajectoryRunResult` 和 `trajectory_replay.jsonl`。Replay journal 是 engineering-only reconstruction record：保存完整 `state_after` 和六类 Memory 顶层 category 的 canonical after-value，不改造既有 `trajectory.jsonl` 为 event sourcing。
- `reconstruct_context()` 从 Seed、persisted scientific trajectory 和 replay journal 创建 fresh State/Memory stores，依次 validation/replace，重建 exact State/Memory/trajectory，不读取 final state/memory artifact，不执行任何 LLM/provider/controller。
- Runner artifacts：`seed_context.json`、`research_state.json`、`research_memory.json`、`trajectory.jsonl`、`trajectory_replay.jsonl`、`run_manifest.json`；Commit 成功时额外写 Snapshot/Proposal，其他状态写 unresolved result。

### M11 offline scenarios

- valid commit：Gap → Solution → Gap/Solution Verification → Experiments → Commit → Finalizer。
- partial-overlap verification-only revision：Targeted Revision 后使用 `changed_units` focus 重新验证。
- Inspiration-guided revision：Controller 多 action mask 使用一次 Fake Controller LLM，Runner 传递本次 found InspirationRecord ID。
- resolved gap switch：`SWITCH_GAP` 是真实 Action，清理旧 solution/experiments 后强制新 Gap Verification。
- partial Gap：RefineGap 是 internal transition，最多一次、action=None、budget +0。
- budget exhausted 与 provider/action failure：保留 unresolved/error artifact，不继续执行或伪造 Proposal。

### M11 本轮验证

- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `207 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `207 tests`, `OK`。
- `PYTHONPATH=src conda run -n myrag python -m compileall -q src tests` → 通过。
- 全部 M11 tests 使用 generic FakeLLM/Fake provider；本轮不调用真实 DeepSeek、Semantic Scholar、arXiv 或其他网络。
- 本轮未修改 `docs/method_overview.md`、`docs/implementation_spec.md` 或 `data/`。

## 下一 milestone

M12 已通过，下一步等待人工审核并明确批准后再进入 M13；本轮不进入 M13。

## 验证记录

- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `16 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `16 tests`, `OK`。
- `PYTHONPATH=src conda run -n myrag python -m research_agent --help` → 成功。
- `PYTHONPATH=src conda run -n myrag python -m research_agent offline-smoke` → 成功。
- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `24 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `24 tests`, `OK`。
- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `36 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `36 tests`, `OK`。
- CLI help 确认 `extract-seed-context` 只提供 `--output-dir` 与 `--max-tokens`，不再提供 `--max-chars`。
- 三篇真实 Markdown 的 deterministic full-body smoke：ByteFlow 41 sections/20 main-body sections/43,234 body chars/cut line 349；Bayesian 48/16/38,968/cut line 280；Cautious 35/14/44,162/cut line 346。三篇均保留完整正文 section，References/Appendix 尾部未进入 reading context。
- ByteFlow real smoke：通过加载 `.bashrc` 末尾的三条配置执行；DeepSeek config 为 `provider=deepseek`、`model=deepseek-v4-flash`、`timeout_seconds=60`、`max_retries=0`。实际发起 1 次 HTTP request，未发生 adapter retry。
- ByteFlow real smoke 成功输出：`outputs/real_seed_smoke/MinerU_markdown_10107_ByteFlow_Language_Modeli_2090055877820440591/seed_context.json` 与 `seed_extraction_audit.json`。
- ByteFlow metadata：`provider=deepseek`、`model=deepseek-v4-flash`、`request_count=1`、`prompt_tokens=14,432`、`completion_tokens=5,101`、`total_tokens=19,533`、`input_chars=44,442`、`validation=success`。
- ByteFlow audit：37 anchors；evidence section ids 全部属于 main-paper body（body sections S1–S20），没有 References/Appendix id；warnings 为 `references_removed_from_line:349`、`appendix_removed_from_line:627`；`SeedPaperContext` 不包含 audit 字段。
- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `50 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `50 tests`, `OK`。
- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `67 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `67 tests`, `OK`。
- `discover-gaps --help` → 成功；CLI 仅接收已有 `SeedPaperContext` JSON，不提供 Markdown extraction 或 Solution Ideation 路径。
- ByteFlow M4 real smoke 输出：`outputs/real_gap_smoke/gap_discovery_result.json`、`gap_discovery_audit.json`、`research_state.json`、`research_memory.json`、`trajectory.jsonl`。
- 此前 ByteFlow M4 revised-boundary evaluation metadata：`provider=deepseek`、`model=deepseek-v4-flash`、`request_count=1`、`prompt_tokens=1,679`、`completion_tokens=3,757`、`total_tokens=5,436`、`input_chars=6,986`、`request_id=26118648-3806-46f7-88cf-2540d1a0d327`；该历史结果保留用于 failure-pattern 对照。
- 此前 ByteFlow M4 revised-boundary result：3 method-oriented Candidate Gaps，G1 为 Active；G1=`medium/high`，G2=`low/medium`，G3=`low/medium`；未执行 Gap Verification 或 Solution Ideation。
- 上一轮 Prompt gate 修改后的 offline validation：`PYTHONPATH=src conda run -n myrag python -m pytest -q` → `74 passed`；`PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `74 tests`, `OK`。
- 上一轮 ByteFlow real Gap smoke：只从已有 `seed_context.json` 运行 Gap Discovery，未重新执行 Seed Context extraction、Gap Verification、Solution Ideation 或论文检索；metadata 为 `provider=deepseek`、`model=deepseek-v4-flash`、`request_count=1`、`prompt_tokens=2,386`、`completion_tokens=928`、`total_tokens=3,314`、`input_chars=10,473`、`request_id=4957747c-d569-4b96-b350-8a1a74cb89c7`，warnings=[]。
- 上一轮 real result：`candidate_gap_count=0`、`gap_agenda=[]`、`active_gap=null`、`next_required_action=null`；未自动改写任何 real output，未进入 M5。
- 本轮最新 offline validation：`PYTHONPATH=src conda run -n myrag python -m pytest -q` → `77 passed`；`PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `77 tests`, `OK`。
- M4 历史 real extraction 状态：此前两次使用当时人工裁决要求的真实 `extract-seed-context` 命令均被执行环境拒绝；该历史记录不影响本轮 M5，未在本轮重新抽取 Markdown。
- M5 中间版本 offline validation：`pytest`/`unittest` 均为 `84`；完成工程补齐后为 `90`；加入全局 cache 后为 `95`；加入 Bearer proxy adapter 后重新运行，最终为 `96 passed` / `96 tests`, `OK`。
- M5 CLI help：`verify-gap` 命令成功注册；要求显式传入已有 M4 trajectory directory，不会自动执行 Seed extraction 或 Gap Discovery。
- M5 real smoke 历史尝试：复用 `outputs/real_seed_smoke_m4_recall/.../seed_context.json` 与 `outputs/real_gap_smoke_m4_recall/.../research_state.json` 的 Active Gap G1；官方 Semantic Scholar seed resolution 返回 HTTP 429。provider request count=2（当时一次 retry），DeepSeek request count=0，resolved provider paper id unavailable。该失败 artifact 保留为历史记录，不能当作当前中转站结果。
- 本轮 host-network proxy smoke：`config/semantic_scholar.json` 的 key 已配置，Bearer header 生效；移除大小写形式 HTTP proxy 环境变量后，项目实际 `/s2/graph/v1/paper/search` 请求成功 1 次、无 retry，解析 provider paper id 并写入全局 cache。直接 autocomplete 示例请求 HTTP 200。用户 shell 中的 proxy 配置未被修改，API key 未输出。
- 当前 M5 定向离线回归：`tests/test_m5.py` → `23 passed`。
- 当前完整离线回归：`PYTHONPATH=src conda run -n myrag python -m pytest -q` → `100 passed`；`PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `100 tests`, `OK`。
- 当前成功 smoke audit：`outputs/real_semantic_scholar_proxy_academic_graph_smoke_host/academic_graph_smoke.json`；示例 endpoint audit：`outputs/real_semantic_scholar_proxy_autocomplete_smoke/autocomplete_smoke.json`。独立读取缓存确认 `cache_hit=true`，未为该确认再次调用 API。
- 新 Bayesian Attention real smoke：配置文件 `api_key` 为空时成功回退到 `SEMANTIC_SCHOLAR_API_KEY`；移除代理环境变量后 `/s2/graph/v1/paper/search` 请求 1 次、0 次重试，解析 provider paper id `aab8daac42fff3ff02d3efefc0e30b3443545874` 并写入全局缓存。随后以禁止网络的 transport 验证同一请求，缓存命中、provider request count=0、transport calls=0。审计 artifact 为 `outputs/real_semantic_scholar_proxy_bayesian_attention_smoke/academic_graph_smoke.json`。
- ByteFlow Recommendations real smoke：复用已有 `paper_search` 全局缓存，只发起 1 次 `/s2/recommendations/v1/papers/forpaper/{paper_id}` 请求，返回 20 篇相关论文，provider request count=1、retry count=0，并写入 `cache/semantic_scholar_proxy/recommendations/0c0c884659e37e530adc329a5e03155dbfc00f5a3afe2fa7cd6cb1b8e6b08a5d.json`。随后以禁止网络的 transport 验证，`paper_search` 与 `recommendations` 均命中，provider request count=0、transport calls=0。审计 artifact 为 `outputs/real_semantic_scholar_proxy_byteflow_recommendations_smoke/recommendations_smoke.json`。
- 真实 `VERIFY_GAP`：复用 `outputs/real_seed_smoke_m4_recall/.../seed_context.json` 和 `outputs/real_gap_smoke_m4_recall` 中 Active Gap G1；Semantic Scholar `cache_scope=global`、`cached_operations=["paper_search","recommendations"]`、`provider_http_request_count=0`、`scientific_search_count=0`、`pool_size=20`、`pool_papers_with_abstract=20`。DeepSeek `request_count=1`，`model=deepseek-v4-flash`，`prompt_tokens=8,452`、`completion_tokens=3,308`、`total_tokens=11,760`、`input_chars=40,631`，validation success；结果 `status=valid`。新 artifact 为 `outputs/real_verify_gap_m5_proxy/`。

## M5 standalone contract repair and M6 completion

- M5 `verify_gap` 现在在 `state_store` 且无 `TrajectoryContext` 时，也会在 provider acquisition/LLM 之前验证 `state.active_gap` 存在且 ID 与 target 一致。
- M5 `verify_solution` 的 state-store-only 路径继续要求 State 中存在 Current Solution；显式传入 solution 时 ID 必须一致。mismatch 的两条路径均在 provider、LLM、Memory、State 和 trajectory mutation 前抛出 `PreconditionError`。
- 新增 `BridgeGenerationDraft` 作为 M6 Call 1 内部 artifact：`InspirationNeed`、恰好五个去重 proposed entities、最多三个 proposed-subset selected entities；不写入 Research State。
- `InspirationRetrievalModule` 依赖 `LLMClient` 与 `InspirationSearchProvider`，Call 1 构造 Need/实体，Call 2 做 arXiv-grounded 或 LLM-only transfer，最多两次 structured LLM call。
- M6 只读取 active gap/current solution、latest relevant verification、`rejected_mechanism` failures、previous bridge names 和少量 related context；不把完整 State/Memory/trajectory 或 M5 related pool 作为 grounding papers 发送给 Call 2。
- 选中实体只生成一次按顺序拼接的 `OR` 查询；搜索最多 20 篇。arXiv adapter 仅使用官方 Atom API，规范化 `arxiv:<canonical-id>`、去版本、去重、保留顺序，并显式处理 timeout/retry/pacing/malformed XML。
- Inspiration search 只使用 trajectory-local normalized exact query cache；cache hit 不增加 `scientific_search_count`。provider failure 不写成功 `InspirationSearchRecord`；successful empty search 可写空记录并进入 fallback。
- grounded candidates 必须使用 selected entity、已检索/Memory 中的 paper ref、`grounding="arxiv"`，同一 paper 最多一个；fallback candidates 必须 `grounding="none"` 且 `paper_ref=null`。没有 selected 时只有 Call 1，不调用 provider/Call 2。
- M6 不修改 Research State；存在 context 时记录 `inspiration_retrieved` / `SEEK_INSPIRATION`，State diff 为空；`found` 只暴露 `TARGETED_REVISION` 后续语义。
- 新增 generic offline tests 覆盖 M5 standalone preconditions、M6 entity/query/cache/provider/retry/grounded/fallback/state-boundary contracts。未使用任何真实论文名称、专有机制或 paper-specific branch。

### 本轮验证

- `conda run -n myrag python -m pytest -q`：首次在当前 shell 缺少 `python` 的直接调用不适用；改用项目要求的 `conda run` 后通过。
- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `114 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → `114 tests`, `OK`。
- 本轮没有真实 DeepSeek、Semantic Scholar 或 arXiv 网络调用；全部 M6 tests 使用 FakeLLM/Fake provider/注入式 XML transport。

## M6 contract repair and M7 Targeted Revision completion

- M6 Inspiration prompt 明确区分 seed-specific facts 与一般科学知识：一般科学知识
  只可用于 Bridge Entity hypothesis generation 或 conceptual transfer，不作为外部
  evidence、retrieved fact 或 citation claim；arXiv `source_problem`/
  `source_mechanism` 仍只能来自 supplied title/abstract。
- M6 `InspirationNeed.rejected_mechanisms` 由 Memory 中的
  `FailureRecord(type="rejected_mechanism")` deterministic materialize，LLM 输出
  不能覆盖、遗漏、改写、合并该字段。
- M6 Bridge Entity 使用 canonical identity：selected entity 和 candidate bridge
  entity 若同名但定义被改写，统一 canonicalize 为 proposed/selected definition。
- 新增 M6 generic tests，覆盖 knowledge boundary、authoritative rejected mechanisms
  和 Bridge Entity canonicalization；没有加入任何真实论文专有术语或分支。
- 新增 M7 `TargetedRevisionModule`、`TargetedRevisionDraft` 和
  `targeted_revision.md`。M7 使用单次 structured LLM call，输入经过 deterministic
  evidence/precondition gate；不调用 M5 verification module。
- M7 Revision Need 从 partial-overlap verification、useful Inspiration 和允许的
  failure memory 构建；LLM 只能返回 lightweight patch。runtime Inspiration ref
  采用 `<InspirationRecord.id>:<candidate_index>`，一次 revision 最多选一个。
- M7 merge 只允许 `core_idea`、`rationale` 和已有 Component ID；patch 外字段及
  evidence refs 保留，未知/重复/no-op/仅 rationale/声明不一致的 patch 拒绝。
  Component、core 和 Experiment 的 `modified`/`stale` 传播、verification focus、
  `novelty_status="unchecked"` 与 `developed` 状态均由 deterministic code materialize。
- M7 成功写入 RevisionRecord、revision memory 和 `solution_revised` trajectory
  event，下一动作是 `VERIFY_RELATED_WORK`；unable-to-revise 保持 State 不变并写入
  failed-revision memory/event。M7 不执行 re-verification、Experiment Design、
  Controller、Commit 或 Finalizer。

### M7 本轮验证

- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `131 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` →
  `Ran 131 tests ... OK`。
- M6 定向与 M7 全部测试均使用 FakeLLM、fake provider 和本地 Memory/trajectory
  fixture；没有真实 DeepSeek、Semantic Scholar、arXiv 或其他网络调用。
- 本轮未修改 `docs/`、`data/`，未进入 M8。

## M7 contract cleanup and M8 Experiment Design completion

- M7 新增 execution-only 参数 `inspiration_record_id`。未提供时，当前 Revision 的
  `available_inspirations=[]`，历史 Inspiration Memory 不会自动触发或强制选择
  Inspiration；提供时只读取精确匹配且 `inspirations` 非空的一个 record。
- M7 的 selected Inspiration evidence 在 LLM 输出并完成 ref validation 后才确定性加入；
  未选择候选的 grounded paper refs 不会进入 FailureRecord，fallback candidate 的
  `paper_ref=null` 也不会制造 evidence。Verification refs 仍只来自当前 relevant
  Solution Verification。
- `TrajectoryRecorder.validate_next_step(step)` 新增为无副作用 runtime helper，检查
  integer/non-negative/non-decreasing step。Targeted Revision 在 LLM、Memory、State 和
  trajectory mutation 前执行；非法 step 的 LLM call、Memory、State、trajectory 均为零。
- 新增 M8 internal drafts：`KeyClaimDraft`、`ExperimentProposalDraft`、
  `ExperimentDesignDraft`；canonical `Experiment` schema 未增加任何 claim/status/id
  控制字段。
- `ExperimentDesignModule` 在 LLM 前检查 active gap、distinct solution、ready
  Components、latest `VERIFY_GAP=valid`、latest `VERIFY_SOLUTION=distinct`、State ID、
  experiment count 和 trajectory step。输入只含 Seed experiment context、verification
  summaries、changed_units 和 experiment targets，不含完整 trajectory/Memory/论文池。
- M8 只调用一次 structured LLM。初始设计确定性 materialize `E1...En`；stale redesign
  只接受每个 stale ID 一次，ready Experiment defensive-preserve，最终数量保持 2–4。
- M8 确定性校验 initial 2–4 / redesign 1–4 ordered claims、core/mechanism coverage、双向 claim coverage、
  `what_it_validates` exact unit anchors、`Seed Method` baseline、质量字段、重复实验、
  unknown refs 和 extra fields。
- M8 成功只更新 `ResearchState.experiments`，记录 `experiments_designed` audit，
  `memory_writes=[]`；全部 ready 时 zero-call/no-event/no-op。

### M8 本轮验证

- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → `152 passed`。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` →
  `Ran 152 tests ... OK`。
- `PYTHONPATH=src conda run -n myrag python -m compileall -q src tests` → 通过。
- 所有 M7/M8 测试均为 generic offline fixtures；本轮没有真实 DeepSeek、Semantic Scholar、
  arXiv 或其他网络调用。
- 本轮未修改 `docs/`、`data/`。

## M8 stale-redesign repair and M9 Scientific Controller completion

- M8 将 initial 与 stale redesign 分成两套 deterministic validation：initial 要求
  2–4 Claims、每个 Claim/Experiment 显式 coverage、core_solution Claim、Component
  Claim 和 exact `Seed Method` baseline；redesign 允许 1–4 Claims，Claims 只覆盖 stale
  proposals，ready Experiments 原样保留。最终 Experiment State 仍做 2–4、anchor、字段非空
  和 canonical duplicate checks。`Experiment` canonical schema 未增加 Claim 字段。
- M8 redesign audit 的 `claim_coverage` 只记录新 Claims 到 stale Experiment IDs 的映射，
  并记录 `design_mode`；ready Experiment 不会仅因旧的 unit anchor 被解释成新 Claim coverage。
- 新增 `controller/budget.py` 的 immutable `ControllerBudget` 与 mutable `BudgetTracker`。
  Budget 只在 action execution 后按 `+1 step + scientific_search_count` 更新，HTTP request
  count 与 cache hit 不计入 scientific search。
- 新增 `controller/scientific_controller.py` 和 `prompts/controller.md`。Controller 固定
  七个既有 Actions，先做 State/Memory consistency、mandatory route、action mask 和 budget
  检查；单一合法动作零 LLM，多动作恰好一次 structured Controller call。Controller 不
  import/执行科研 modules，也不修改 State。
- `ControllerContext` 是 deterministic compact summary，不含完整 Seed Paper、Memory、
  trajectory 或 paper-pool abstracts；`ControllerRoutingResult` 是 execution-only 的
  decision/stopped 结果。`record_decision_result()` 只在实际 action 完成后写真实
  `DecisionRecord`，不写 pending/fake result。
- 明确处理 new gap/no verification、valid gap/no solution、resolved gap、unchecked/revised
  solution、Inspiration found、partial overlap、distinct experiment states、commit-ready
  states、zero-search verification 和 stopped/unresolved paths。裸 `partially_resolved`
  没有后续 `gap_refined` event 时抛出 `ControllerRoutingError`；没有新增 Refine/Component
  Verification action。

### M9 本轮验证（含本轮 reserve 修复）

- `PYTHONPATH=src conda run -n myrag python -m pytest -q tests/test_m9.py` → 28 passed。
- M9 与既有 suite 回归 → 181 passed。
- 本轮 M9 测试全部使用 generic FakeLLM、trajectory-local Memory 和本地 budget fixture；
  没有真实 DeepSeek、Semantic Scholar、arXiv 或其他网络调用。
- 本轮未修改 `docs/method_overview.md`、`docs/implementation_spec.md` 或 `data/`。

## M9 reserve repair and M10 Commit + Finalizer completion

- 修复 M9 `SEEK_INSPIRATION` action mask：主动 Inspiration 必须同时满足至少 1 个
  scientific search 和 3 个剩余 step，预留 Inspiration found → Targeted Revision →
  `VERIFY_RELATED_WORK` 的最坏路径；预算仍按每个实际执行 Action 单独更新。
- 新增 generic tests 覆盖 partial overlap 的 2/3-step mask、distinct missing/stale
  experiments、commit-ready 的 2/3-step mask、commit gate 未通过时停止，以及 Inspiration
  found 后 2-step Revision route。
- 新增 deterministic `check_commit_gate()`：按 Active Gap、Gap Verification、Current
  Solution、Solution Verification/novelty、Component readiness、Experiment readiness
  的稳定顺序生成 unmet conditions。最新 matching verification 由 Memory query 获取；
  State/Verification novelty 冲突显式抛出 `PreconditionError`。
- 新增 `create_commit_snapshot()` 与 `CommitModule`。Snapshot 内部重新执行 Gate，只保存
  committed gap/solution/experiments、实际 latest verification results 和 first-seen 去重
  evidence refs；Pydantic deep copies 保证与 State、Memory 和 Snapshot 自身隔离。
- `CommitModule.commit()` 先做 trajectory step preflight，成功记录
  `research_committed`/`COMMIT` 事件，State diff 与 Memory writes 为空；Gate failure/invalid
  step 不产生 Commit event。
- 新增 `FinalizerDraft` 和 `FinalizerModule`。Finalizer 只调用一次 structured LLM，输入为
  CommitSnapshot、SeedPaperContext 的 objective/core method/findings/limitations summary、
  novelty context 和 evidence refs。Draft 只含文本字段与 2–3 条 expected contributions，
  不含 Components/Experiments。
- FinalResearchProposal 由 Python deterministic materialize：method components 与
  experimental plan 是 Snapshot 的 exact deep copy；Finalizer 不得新增、删除、修改组件或
  实验，也不得声称实验已经执行。Prompt 明确 shared elements、`distinct != zero overlap`、
  abstract-only evidence boundary 和 expected/potential contribution boundary。
- 提供 `save/load_commit_snapshot()` 与 `save/load_final_research_proposal()`，继续使用
  schema/version envelope。

### M10 本轮验证

- `PYTHONPATH=src conda run -n myrag python -m pytest -q tests/test_m9.py tests/test_m10.py` → 44 passed。
- `PYTHONPATH=src conda run -n myrag python -m pytest -q` → 197 passed。
- `PYTHONPATH=src conda run -n myrag python -m unittest discover -s tests -v` → 197 tests, OK。
- `PYTHONPATH=src conda run -n myrag python -m compileall -q src tests` → 通过。
- 全部测试使用 FakeLLM 和本地 fixture；没有真实 DeepSeek、Semantic Scholar、arXiv 或其他网络调用。
- 本轮未修改 `docs/method_overview.md`、`docs/implementation_spec.md` 或 `data/`，未进入 M11。

## M12 implementation and real-pilot attempts

- 修复 M11 Runner 的 Inspiration direct handoff：只检查 trajectory 中最近一个
  `event.action is not None` 的实际 Scientific Action；只有该 action 是
  `SEEK_INSPIRATION`、事件类型是 `inspiration_retrieved`、结果为 `found` 且唯一的
  `InspirationRecord` 含非空 inspirations 时，才把该记录交给 Targeted Revision。
  历史 Inspiration 不会在后续 Revision/Verification 后再次触发。
- 新增 execution-only `models/pilot.py`、`llm/recording.py` 和 `runner/pilot.py`：提供
  单次 real opt-in guard、DeepSeek shared `RecordingLLMClient`、调用/token 审计、
  Semantic Scholar/arXiv provider accounting、safe `PilotManifest` 和 `provider_audit`。
  Pilot 不选择 Scientific Action，不改变 State/Memory 科研定义；CLI 为
  `real-pilot`，固定默认 budget 为 12 steps / 2 searches。
- 新增 `tests/test_m12.py`，覆盖 opt-in、缺失配置零调用、调用失败脱敏、token metadata
  缺失、manifest closed schema、Inspiration single-use、Semantic Scholar cache metadata
  和未选择 arXiv 时 zero request。
- 离线验收：`pytest` 为 `221 passed`；`unittest` 为 `221 tests, OK`；`compileall` 通过。
- 第一次失败 artifact 保留在 `outputs/m12_real_byteflow/`，失败点是 Gap Discovery
  structured output 非 JSON。
- 第二次完整 pilot 使用 `outputs/m12_real_byteflow_unbounded/`，运行时移除了
  HTTP/HTTPS proxy 环境变量，并在共享 `RecordingLLMClient` 中移除所有转发请求的
  `max_tokens` 字段；没有修改科研 schema 或 workflow。
- 第二次运行成功 `committed`：10 次 DeepSeek call 全部成功，累计
  prompt/completion/total tokens 为 `49950/34733/84683`；5 个 scientific steps、
  0 个 scientific searches；trajectory events=22、replay records=6。
- Semantic Scholar related pool 从 global cache 命中 `paper_search` 和
  `recommendations`，HTTP requests=0，pool=20，20 篇有 abstract；arXiv 未被选择，
  HTTP requests=0。
- Decision sequence 为 `VERIFY_RELATED_WORK → SOLUTION_IDEATION →
  VERIFY_RELATED_WORK → DESIGN_EXPERIMENT → COMMIT`。Gap Verification=`valid`，
  Solution Verification=`distinct`，生成 3 个 ready experiments，Commit Gate 通过，
  Finalizer 生成 proposal。
- replay 只读重建后的 State、Memory 和 trajectory 均与最终 artifact 精确一致。
- 本轮没有修改 `docs/method_overview.md`、`docs/implementation_spec.md` 或 `data/`。
- Open issue：第一次失败表明中转服务的 structured-output 行为可能受输出长度限制影响；
  第二次移除本地 `max_tokens` 后已成功，但仍应将该结果视为当前 provider/model 配置下的
  evaluation evidence。原始完整响应继续不写入 artifact。M13 不启动。

## M12.1 Design-Rationale Preservation Hardening（本轮）

### 已完成内容

- 在现有 `SeedPaperContext` 中只增加 `method_constraints: list[str]`；LLM draft 使用
  `list[EvidenceItem]`，materialization 继续使用已有递归 source-id validation 和
  `SourceAnchor`，没有增加 rejected-alternatives、design-choices、trade-offs 或
  assumptions 平行字段。
- `method_constraints` 的抽取语义限定为作者明确写出的 design rationale、约束、
  acknowledged trade-off、rejected alternative 及其 rejection reason，以及后续方法
  需要保留或显式克服的条件。不能从 architecture fact 自行推断；不能当作 Gap。
- Gap evidence vocabulary 新增 `seed:method_constraints[i]`；仍不允许任何
  `seed:experiment_context.*`。Gap Refine 在 trajectory context 下把 Seed constraints
  显式传入同一个 refine prompt。
- 更新 `seed_context.md`、`gap_discovery.md`、`gap_refine.md`、`solution_form.md`、
  `solution_select_develop.md`：加入 paper-agnostic Design-Rationale Consistency Gate。
  deliberate choice 不是自动 Gap；挑战 rejected alternative 时必须处理 rejection
  reason；constraint-preserving opportunity 可以保留，但不能把 rejected alternative
  当作从未考虑。
- 没有增加 `consistency_check`、critic/reviewer/reflection agent、second-pass LLM、
  Scientific Action、State/Memory 字段或新的 workflow 分支；Solution 仍是 exactly
  three candidates + one Select/Develop call。
- 新增 `tests/test_m12_1.py`，只使用 generic mechanism A/B 和 stability constraint C
  fixtures，覆盖 Seed extraction/anchors、invalid source、Gap evidence/refine propagation、
  Solution constraint context 和 two-call/three-candidate contract。

### 本轮验证与真实回归

- 完整 offline suite：pytest `229 passed`；unittest `Ran 229 tests ... OK`；
  `compileall -q src tests` 通过。新增 `tests/test_m12_1.py` 的 8 个 generic tests。
- 真实回归输出：`outputs/m12_real_design_rationale_regression/`；没有覆盖既有
  `outputs/m12_real_byteflow*` artifact。`run_manifest.json` 为 committed，
  `commit_gate_passed=true`，trajectory events=22，replay records=6。
- 真实 DeepSeek 共 10 次 structured calls，`provider=deepseek`，`model=deepseek-v4-flash`；
  总 tokens 为 prompt/completion/total = `52362/45966/98328`，10 次均成功、metadata 完整，
  每次实际转发 `max_tokens=null`。llm_calls.jsonl 未包含密钥、认证头或完整 prompt。
- Seed extraction 一次 call：body `input_chars=44442`；完整 request audit 的
  `input_chars=50637`；prompt/completion/total = `14760/7909/22669`。
  warnings 仅为 `references_removed_from_line:349` 和 `appendix_removed_from_line:627`。
- SeedPaperContext 抽取到 6 条 author-explicit `method_constraints`，anchors 只落在
  main-paper sections S7–S10/S8；References/Appendix 没有成为 evidence。`limitations=[]`，
  未观察到模型凭 outside knowledge 发明 limitation 的迹象。
- Gap Discovery 返回 3 个 Candidate Gaps，G1 active；Gap verification 为 `valid`。
  G1/G2 使用 method component + method_constraint evidence，G3 使用 method component +
  method_constraint evidence；没有使用 experiment_context refs。G1 medium/high priority、
  G2 medium/high priority、G3 low/medium priority，置信度与 evidence strength 匹配。
- Solution Ideation 生成恰好 3 个 candidates，S1 selected/developed，4 个 components ready；
  Solution Verification 为 `distinct`；设计 rationale 中明确保留 fixed Top-K、static
  computation graph、causality、tractability 和 memory/resource 约束，没有简单重提被拒绝
  的 global-threshold/ragged alternative。
- Experiment Design 生成 E1/E2；Controller/action sequence 为
  `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK → DESIGN_EXPERIMENT → COMMIT`。
  没有 Inspiration/Revision；arXiv HTTP requests=0。
- Semantic Scholar 使用 bearer 中转站，global cache 命中 `paper_search` 与
  `recommendations`，related pool=20，HTTP requests=0；全局检索缓存未重复付费请求。
- replay 只读重建结果：State、Memory、trajectory 均与最终 artifacts 相等。
- 人工式质量检查：3 个 gap 都是 method-level opportunity/trade-off，不是纯实验 gap；未见
  unsupported confirmed failure，也未见明显 architecture-fact→system-failure 跳跃。G1/G2
  说明了约束保留，G3 保持 low confidence。未发现 paper-specific specialization。
- 抽取质量观察：本轮 `limitations=[]`，未保留新的 future-work 条目；旧的
  `outputs/m12_real_byteflow_unbounded/seed_context.json` 曾有 2 条 evaluation-scope
  limitations。该差异不在真实调用后自动修补，作为后续人工评估的 paper-agnostic
  extraction recall issue 记录。

### 当前状态

- M12.1 完成；M13 Batch Runner / Research Experiment Support 尚未开始。
- 未修改 `docs/method_overview.md`、`docs/implementation_spec.md` 或 `data/`；
  未修改 AGENTS.md（所需 anti-overfitting 规则已存在）。
- 下一步等待人工审核；本轮不进入 M13。

## M12.2 Scientific Controller Inspiration Boundary（本轮）

### 当前执行内容

- 不增加 Action、Agent、reflection/critic loop、numeric innovation score，且不修改
  Research State、Research Memory、Gap、Solution 或 Verification schema。
- 在现有 Controller deterministic action mask 中复用 Active Gap、CurrentSolution 和
  latest Solution Verification：local/incremental solution 且 gap 具有高优先级或结构性
  机会时，或 distinctness 主要被描述为 implementation choice 时，开放可选的
  `SEEK_INSPIRATION`；明确 new mechanism/interaction/representation 或 substantive
  redesign 时不开放该分支。
- `inspiration_opportunity` 只作为 execution-only compact Controller context，不写入
  State/Memory；只有多个合法 Action 时才调用 Controller LLM。
- 更新 `controller.md`，明确 distinct 与 high priority 单独不强制 Inspiration，清晰新
  机制可直接进入 Experiment/Commit；加入 generic tests，未加入真实论文术语。

### 本轮验证与真实回归

- `tests/test_m9.py` 定向回归：`33 passed`。
- 完整 offline suite：pytest `234 passed`；unittest `234 tests, OK`；
  `compileall -q src tests` 通过。
- 真实回归输出：`outputs/m12_2_real_controller_inspiration_regression/`，未覆盖旧
  artifact。Seed extraction、Gap Discovery、Gap Verification 和 Solution Ideation 已
  完成；Solution Verification 的第二次 `VERIFY_RELATED_WORK` 在真实 DeepSeek 第 6 次
  调用时因 `The read operation timed out` 失败。
- 真实 metadata：`provider=deepseek`、`model=deepseek-v4-flash`、
  `llm_request_count=6`（5 次成功、1 次失败）；成功调用 token subtotal 为
  prompt/completion/total `32091/28867/60958`，失败调用及整体 pilot token totals 为
  `None/unavailable`；失败调用 `input_chars=47741`、`max_tokens=null`。Semantic Scholar
  global cache 命中 `paper_search`/`recommendations`，HTTP `0`；arXiv HTTP `0`。
- Partial Action sequence 为
  `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK`，失败动作仍为
  `VERIFY_RELATED_WORK`。未到达后续 Controller Inspiration 分支，因此本次 real case
  没有观察到 Inspiration trigger 的改变；没有 Inspiration、Commit 或 Finalizer，
  `commit_gate_passed=false`。
- 按本轮边界未继续调 Prompt、未重跑真实请求、未修改 paper-specific 规则；该失败只作为
  provider/real-evaluation artifact 保留。
- 随后按人工要求将命令级 `DEEPSEEK_TIMEOUT_SECONDS` 调整为 `180`，保持
  `DEEPSEEK_MAX_RETRIES=0`、移除 proxy、且继续不发送 `max_tokens`，在新目录
  `outputs/m12_2_real_controller_inspiration_timeout180/` 重新执行一次完整 pilot。
- 180 秒运行成功 `committed`：10 次 DeepSeek 调用全部成功，token 为
  prompt/completion/total `52764/31474/84238`；Semantic Scholar global cache 命中，HTTP
  `0`；arXiv HTTP `0`；trajectory events `22`、replay records `6`。
- 新 Inspiration gate 在真实运行中实际改变了 Controller 的合法 Action mask：Experiment
  Design 阶段为 `DESIGN_EXPERIMENT + SEEK_INSPIRATION`，成熟阶段为
  `COMMIT + SEEK_INSPIRATION`。Controller 两次均选择了直接推进的 Action，因此没有强制
  执行 Inspiration；最终 sequence 仍为
  `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK → DESIGN_EXPERIMENT → COMMIT`。
- Commit Gate、Finalizer 均成功；用 replay journal 重建后的 State、Memory、trajectory
  分别与最终 artifact 相等。此前 60 秒失败 artifact 保留，不覆盖。

### Open issues / 下一步

- 当前没有新增 method-level ambiguity；确定性 gate 是工程层 Controller routing rule，
  不改变科研定义。此前的 60 秒 Solution Verification read timeout 通过提高命令级
  timeout 得到解决；未修改 Prompt 或 scientific schema。
- M12.2 已完成并停止；其后进入 M13 Batch Evaluation 实施，本状态文件末尾记录当前
  M13 进度。

## M13 — Batch Evaluation and Evaluation Framework（当前轮次）

### 已完成

- 新增 `models/evaluation.py`：`EvaluationGap`、`PaperEvaluation`、`EvaluationReport`。
  这些是 execution-only closed contracts，记录 execution/workflow/gap/solution/
  inspiration/cost/reliability 的 count/categorical metadata，不进入 Research State、
  Research Memory，也不计算科研质量总分。
- 新增 `runner/batch_evaluation.py`：按固定随机 seed 发现 Markdown、选择子集、为每篇
  paper 建立独立目录、调用现有 `RealSinglePaperPilot`、单篇失败隔离并写出根级
  `evaluation_report.json`。当前明确保持串行，未进入并行实现。
- 新增 `human_review_template.json`，只供人工填写 Gap Quality、Solution Quality 和
  Final Proposal Quality，不被 workflow 读取。
- 新增 `scripts/run_m13_batch_5_serial.sh`：固定 `max_papers=5`、`selection_seed=1301`、
  `max_steps=12`、`max_searches=2`，默认 `DEEPSEEK_TIMEOUT_SECONDS=180`，要求显式真实服务
  配置并拒绝覆盖已有 paper output。
- Semantic Scholar global cache 的 miss-check、provider request 和 atomic write 增加
  per-entry file lock；新增 arXiv global response cache，避免后续重复相关工作/灵感检索
  造成不必要 API 成本。缓存不记录密钥，不修改 `data/`。

### 离线验证

- `pytest -q`：`242 passed`。
- `PYTHONPATH=src python -m unittest discover -s tests -q`：`242 tests, OK`。
- `python -m compileall -q src tests`：通过。
- 新增 M13 generic tests，并新增 arXiv global-cache regression；不使用真实论文专属 fixture。
- 新增 Revision full-solution 状态收敛 regression：完整 distinct verification 后，只有被完整
  验证的 `modified/stale` Components 才确定性转为 `ready`；保留 Controller 对非法 distinct
  状态的拒绝。
- 新增 initial Experiment Design canonical baseline regression：core effectiveness 实验缺少
  literal `Seed Method` 时，在 materialization 阶段补入该内部 canonical baseline；空 baseline
  仍然拒绝，redesign 中已有 ready 实验仍保持不变。

### 真实执行状态

- `outputs/m13_real_batch_5_serial_rerun_2/evaluation_report.json` 已产生完整 batch report：
  5 篇论文中 3 篇 committed、2 篇 failed，5 篇均完成 Gap Discovery 和最终 Solution
  Verification，5 篇 State/Memory/Replay consistency 均通过。
- AtlasKV 失败于 Targeted Revision 后的 full-solution distinct verification routing：core
  revision 使未 patch 的 Components 保持 `stale`，但 full verification 后没有把这些已覆盖
  的 Components 收敛为 `ready`。Antislop 失败于 Experiment Design 的 canonical `Seed Method`
  baseline contract。两者均不是 LLM/API transport failure。
- 两篇触发 Inspiration Retrieval 的真实 artifact 均显示 `status=found`，每篇返回 3 条
  arXiv-grounded inspiration candidates；由于后续没有 TARGETED_REVISION，报告中的
  `inspiration_used=0` 表示没有合并进最终 Solution，不表示检索为空。
- 本轮修复保持 paper-agnostic：没有加入论文名称、专有组件、算法、公式或 special case；
  未修改 `docs/`、`data/` 或历史 batch outputs。

### Anti-overfitting / 边界检查

- 未修改 `docs/` 或 `data/`，未修改任何科学 prompt、Gap/Solution/Controller 定义、
  Action、State、Memory 或 workflow 分支。
- M13 prompts 不新增；代码没有 paper-specific branch；测试使用 generic fixtures。
- 下一步：人工审核修复后的 contract 与真实失败论文是否需要重跑；除非另有明确授权，不进入
  并行优化或 M14。

## Correctness hardening before M13 continuation（本轮）

### 当前阶段

M13 batch 已有离线实现和历史真实结果；本轮是进入下一次 batch 前的通用 correctness
hardening，不是新的 milestone 扩展，不恢复五篇 batch。

### 已完成内容

- Markdown Appendix 检测已改为保守的显式 marker 规则：识别 Appendix/Appendices、
  Supplementary/Supplemental 等 heading，不再把任意单字母 heading 当作 Appendix；
  References 与 Appendix 的最早边界仍决定 main-body cutoff。
- Component freshness contract 已明确：`ready`=当前版本无未验证修改，`modified`=直接
  修改后未重新验证，`stale`=旧验证因核心/依赖变化失效；它与整体
  `novelty_status` 正交。Full-solution `distinct`/`partial_overlap` verification 会
  收敛所有被覆盖的 pending Components；changed-units 必须覆盖全部 pending modified，
  且存在 stale 时拒绝局部 focus，要求 full-solution verification。
- Targeted Revision 继续使用 lightweight patch + deterministic merge，并在一次 LLM
  输入中传入已有 `SeedPaperContext.method_constraints`；prompt 要求保持或明确处理作者
  说出的 design rationale、trade-off 与 rejected-alternative reason。
- Scientific Controller 已删除所有基于 solution/gap 文本关键词、正则、同义词或
  embedding 的 Inspiration gate。`SEEK_INSPIRATION` 只由现有 state/verification、
  action mask 和 productive reserve 决定；多个合法 Action 才调用 Controller LLM，
  不增加 Action、Agent、score 或 schema。
- 新增 generic regression 覆盖 Appendix false positive、freshness transitions、
  changed-units invariants、重新验证后第二次 Revision、method constraint wiring、
  Inspiration wording invariance 和 budget reserve。

### 本轮验证结果

- 完整 offline suite：`pytest` 为 `253 passed`；`unittest discover -s tests -v` 为
  `Ran 253 tests ... OK`；`compileall -q src tests` 通过。
- 37 篇 Markdown 的只读 parser audit 已写入 `outputs/parser_audit.json`：
  `cut_by_references=37`、`papers_with_explicit_appendix=5`、
  `cut_by_explicit_appendix=0`、`cut_by_structurally_inferred_appendix=0`、
  `no_cut=0`、`suspiciously_short=0`、`title_accidentally_appendix=0`。5 篇显式
  Appendix 都晚于 References，因此实际 cutoff 全部由 References 决定。
- Fresh AtlasKV canonical regression 保存在
  `outputs/real_atlaskv_correctness_regression_final2/`，状态为 `committed`；10 次
  DeepSeek request 全部成功，`prompt/completion/total=58684/33035/91719`，
  `token_metadata_complete=true`。Semantic Scholar 使用全局 cache 的
  `paper_search` 与 `recommendations`，HTTP `0`，pool size `20`；arXiv HTTP `0`。
- Canonical action sequence 为
  `VERIFY_RELATED_WORK → SOLUTION_IDEATION → VERIFY_RELATED_WORK → DESIGN_EXPERIMENT
  → COMMIT`。两个 Controller 多动作 mask 分别为
  `DESIGN_EXPERIMENT + SEEK_INSPIRATION` 与 `COMMIT + SEEK_INSPIRATION`，Controller
  两次均选择直接推进，没有强制 Inspiration。
- 最终 Current Solution 为 `distinct`，C1–C4 和 E1–E3 均为 `ready`；没有再次出现
  `Targeted Revision cannot run before re-verification`。State、Memory、trajectory
  replay 三者均与 replay reconstruction 相等。
- 为处理本地工具会话超时，本轮曾产生两个额外真实尝试 artifact：
  `outputs/real_atlaskv_correctness_regression/` 的首个 DNS transport failure、
  `outputs/real_atlaskv_correctness_regression_final/` 的 related-paper reference
  contract failure；它们均未被当作 canonical 成功结果，保留用于成本/失败审计。另一个
  `outputs/real_atlaskv_correctness_regression_network/` 也完整 committed；后续不再重跑。

M13 五篇 batch 不在本轮范围内，等待人工审核后再决定是否恢复。

### Open issues

- 本轮只记录通用工程 correctness，不根据 AtlasKV 或其他真实论文增加 prompt/code 特调。
- 若 AtlasKV 仍暴露科研输出质量问题，只记录其 paper-agnostic failure pattern，等待人工
  审核，不在本轮继续调 Prompt。
- 额外真实尝试暴露的 related-paper reference failure 属于真实 LLM/provider 输出稳定性
  artifact；本轮不修改 schema、prompt 或 scientific logic 处理它。

### 已确认技术选择

- Parser 的 FALSE POSITIVE 风险高于 lettered-only Appendix 的 FALSE NEGATIVE；因此不做
  单字母结构推断。
- `partial_overlap + all Components ready` 是合法状态；Controller 可在
  `TARGETED_REVISION` 与有 reserve 的 `SEEK_INSPIRATION` 中选择。`distinct` 不自动要求
  Inspiration。
- Targeted Revision 的任何 modified/stale precondition 不放宽；必须先
  Revision → Verification → Revision。
- Real-paper smoke 只作 evaluation case；所有通用修复必须先抽象为 paper-agnostic principle。

### 下一 milestone

无。本轮已完成离线验证、parser audit 和 AtlasKV regression 后停止，等待人工审核；
不进入 M13 batch continuation。

## RISE 工程化 P0 — 源码契约核对（2026-10-13）

### Open issues

- P0 按当前源码冻结：Commit Gate 通过且无新科研证据时，Controller 直接返回 `COMMIT`。但 `docs/implementation_spec.md` 第 3.11 节仍允许 Controller 在 Gate 通过后选择 `SEEK_INSPIRATION`；源码和 `test_commit_ready_state_does_not_reopen_inspiration_without_new_evidence` 与该文本不一致。进入 P1 前需确认方法文档应同步当前源码，还是另行授权改变 Controller 行为。P0 不修改科研代码或方法定义。
