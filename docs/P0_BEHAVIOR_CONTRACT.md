# P0：RISE 科研执行契约与迁移边界

本文以当前源码和离线测试为准，供 P1 Graph 迁移及 P2 恢复设计使用。不得把 Graph 路由规则另写成一套科研 Controller，也不得改变 Prompt、State/Memory 含义、预算或终止结果。

## 1. 当前执行流程

1. `RealSinglePaperPilot.run()` 先做配置预检，再由 `SeedContextExtractor.extract_file_with_reading()` 读取完整 MinerU Markdown，生成 `SeedPaperContext` 和完整阅读上下文。此步骤发生在 `TrajectoryRunner` 之外；异常时 Pilot 写 `preflight_failed` 或 `failed` manifest 并重新抛出，尚无 `TrajectoryRunResult`。
2. `TrajectoryRunner.run()` 初始化 `TrajectoryContext`，调用一次 `GapDiscoveryModule.discover()` 做 Bootstrap：生成并评估 GapAgenda、按现有规则选择 Active Gap、更新 State 和 Trajectory。Bootstrap 不是 Scientific Action，不消耗 Action Step 或 Search Budget。
3. 每轮先无副作用地计算 Commit Gate，再由 `ScientificController.decide()` 根据 Research State、Research Memory、完整 Trajectory、Budget、Gate 条件和 Action Mask 选路。单一合法动作可确定性路由；存在多个合法动作时才请求 Controller LLM。非法/结构错误决策不会被 Runner 猜测修复。
4. Runner 对一个高层 Action 调用对应模块。Action 可包含多个内部 LLM 操作；但成功或已开始后失败都按一个 Action Step 记账。Runner 写决策/结果、预算事件及 replay 记录，再回到 Controller。
5. `COMMIT` 只在 Gate 通过时进入当前路由；当前成熟、可提交状态直接选 `COMMIT`，不会无新证据地重开普通 Inspiration。Commit 生成只读 `CommitSnapshot` 并记录 `research_committed`，随后 Runner 在 Action 循环外调用只读 Finalizer。Finalizer 成功产生 Proposal；失败返回 `finalization_failed`，保留 Snapshot。

Commit Gate 是纯确定性五项检查：Active Gap 存在且 active；其最新匹配 Gap Verification 为 `valid`；Current Solution 完整且 developed、novelty 为 `distinct`，并有最新匹配的 distinct Solution Verification；全部组件 ready；Experiment 数量为 2–4 且全部 ready。不能在 Graph Router 中复制或放宽这些规则。

## 2. 操作边界、预算和 State/Memory/Trajectory 影响

| 操作 | Scientific Action | Step / Search | 当前主要效果 |
|---|---|---|---|
| Markdown 解析与 Seed Context Extraction | 否，Runner 前置 | 0 / 0 | 产出 Seed Context、完整阅读上下文及 Pilot 审计文件；还没有 TrajectoryContext。 |
| Gap Discovery Bootstrap | 否 | 0 / 0 | 生成 GapAgenda，Trajectory 记录发现/选择事件，Research State 设置 Active Gap；不写科研 Action 决策记录。 |
| Controller 决策 / Commit Gate | 否 | 0 / 0 | Gate 只读；决策不直接修改 Research State/Memory。Runner 记录 decision event；停止时记录 `controller_stopped`。 |
| `SOLUTION_IDEATION` | 是 | 1 / 通常 0 | 更新候选及 Current Solution；模块结果进入 Trajectory。高层动作无论内部 LLM 次数均只计 1 Step。 |
| `VERIFY_RELATED_WORK`（Gap） | 是 | 1 / 按模块报告的 Search Delta | 记录 Verification/Search/论文池等 Memory；Gap Verification 不直接改 Research State。`partially_resolved` 后 Runner 最多内部执行一次 RefineGap。 |
| `VERIFY_RELATED_WORK`（Solution） | 是 | 1 / 按模块报告的 Search Delta | 写 Verification Memory；更新 novelty State。`core_overlap` 会拒绝候选并清除 Current Solution/Experiments，同时写失败记录。 |
| `SEEK_INSPIRATION` | 是 | 1 / 按模块报告的 Search Delta | 写 Inspiration/Search/论文 Memory 及 Trajectory，不直接改 Research State；后续 Revision 只消费 Controller 当前传递的最新 found Inspiration 记录。 |
| `TARGETED_REVISION` | 是 | 1 / 0 | 使用轻量 Patch 和确定性合并更新 State；写 Revision Memory；不能完成时同时写 Failure Memory。后续仍按 Controller 规则重新 Verification。 |
| `DESIGN_EXPERIMENT` | 是 | 1 / 0 | 有效设计写入 State Experiments；`unable_to_design` 是结构化科研结果，不等同 Runner 异常或伪造实验。 |
| `SWITCH_GAP` | 是 | 1 / 0 | 终结当前 Gap 并按 Agenda 顺序切到下一个 Candidate；重置候选方案、Current Solution 和 Experiments。终结状态由最新 Gap Verification 决定：`resolved` 才标 resolved，其余为 rejected。 |
| `COMMIT` | 是 | 1 / 0 | 生成 Snapshot、记录 Commit 事件；CommitModule 不修改 State 或 Memory。 |
| Finalizer | 否，Commit 后置步骤 | 0 / 0 | 只读 Snapshot 和 Seed 摘要，输出 Proposal；失败不回退 Commit、不另记 Action Step。错误在 RunResult/manifest 中，不新增科研状态。 |
| Gap Refine | 否，Gap Verification Action 内部操作 | 0 / 0 | 只允许每个 Active Gap 一次；更新 Gap/Agenda/Trajectory，不产生新的 Action 或决策记录。后续再验证仍为 partial 时不再 Refine，按 Controller 规则 reject/stop 或 switch。 |

Search Budget 使用模块 `scientific_search_count` 的 Action 前后差值，不等于 provider HTTP 请求数；缓存命中通常不计新 Search。`TrajectoryEvent.step` 是事件序号，不可直接当作 `BudgetTracker.steps_used`。每个已开始的模块 Action 失败仍计 1 Step，并计入已报告的 Search Delta；模块调用前的 Runner invariant 失败不计 Action Step。

Runner 为每个 Action 追加 `controller_decided`、`budget_updated` 和决策结果；科研模块按实际结果写 state diff、Memory 写入及领域事件。`decision_log` 保存执行后的结果（失败也记录），不是 Controller 决策调用的副作用。

## 3. 异常语义与未来 task 状态映射

当前异常定义在 `errors.py`。LLM 类包含 `LLMTransportError`、`StructuredOutputError`/`MalformedLLMResponseError`；检索类包含 `RetrievalTransportError`、`RetrievalHTTPError`、配置和 Seed Paper 解析错误；领域/执行类包括 `ExtractionError`、`ModuleContractError`、`PreconditionError`、`ControllerRoutingError`、`RunnerInvariantError` 和序列化错误。

`TrajectoryRunner` 在 Bootstrap、Controller 和 Action 边界捕获异常并分别产出 `failed`。在已派发模块内抛出的任何异常（包括 LLM、结构化输出、Retrieval、科研模块契约错误）均按当前语义终止该次 RISE 运行；Runner 不依据异常的 `retryable` 属性自动重放 Action。现有 `RecordingLLMClient` 的一次结构化格式纠错重试、Provider 内已配置的有限 HTTP 重试属于原有调用层行为，保持不变；它们不能成为新的 Worker/Graph Action 重试。

| RISE / 执行结果 | 建议业务 `task_status` | `rise_outcome` | 备注 |
|---|---|---|---|
| `committed` | `completed` | `committed` | Proposal 与 Snapshot 均已生成。 |
| `stopped` | `completed` | `stopped` | Controller 按科研条件/预算正常终止；可没有 Proposal。 |
| `failed` | `failed` | `failed` | Bootstrap、Controller 或已派发 Action 的异常；保存原始 `error_type`/摘要，不自动重试该 Action。 |
| `finalization_failed` | `failed` | `finalization_failed` | Commit 已成立、Snapshot 保留，但 Proposal 缺失。 |
| Worker/数据库/Checkpoint 故障且未恢复 | `failed` | `null`，若此前已有真实 RISE 终态则保留该值 | 这是任务基础设施失败，不能伪造成科研 `failed` 结果。恢复中保持非终态状态。 |
| 用户取消 | `cancelled` | `null` 或已有的真实 RISE 结果 | 取消是业务生命周期，不新增 RISE outcome；保留最后持久化的科研状态。 |
| Pilot 预检/Seed Extraction 在 Runner 前失败 | `failed` | `null` | 现有 Pilot Manifest 的 `preflight_failed`/`failed` 不是 `TrajectoryRunResult` 枚举；另存失败阶段与错误类型。 |

取消应在高层 Action 的持久化边界阻止后续 Action。Worker 崩溃可从最后持久化边界恢复，但不能将已返回的科研 Action 错误改成重试策略。

## 4. LangGraph 与持久化迁移映射

| 现有职责 | Graph 迁移建议 |
|---|---|
| Pilot 预检、Markdown 解析、Seed Extraction | 图外输入准备；将 Markdown/完整阅读上下文用稳定文件引用和 checksum 传入 Bootstrap。Seed Extraction 失败不伪装成 Action。 |
| `TrajectoryContext.initialize()` + `GapDiscovery.discover()` | 单独 Bootstrap Node；只执行一次并持久化初始 GapAgenda/State/Trajectory。 |
| Commit Gate + `ScientificController.decide()` | Controller Node/条件边。领域路由与 Action Mask 继续由 ScientificController 决定，Graph 只按已返回的 Action 分发。保存 `ControllerDecision`、允许动作、Gate 输入/结果，避免 checkpoint 后重新抽样不同决策。 |
| `TrajectoryRunner._dispatch()` 七种 Action | 每个高层 Action 一个 Node/Adapter。VERIFY Node 可根据当前状态执行 Gap 或 Solution Verification；Gap Refine 保持在该 Action 内，不拆成可单独计费节点。 |
| Commit + Finalizer | Commit Node 产出不可变 Snapshot；独立 Finalizer Node 产出 Proposal，保持 `finalization_failed` 与 Snapshot 保留。 |

可恢复 Graph State 至少包括：schema/version、task/thread/input 引用及 checksum、Seed Context、完整 Research State、完整 Research Memory、足以重放 Controller 路由的全部 Trajectory 历史、Budget 上限/已用值、待执行的 Controller 决策与 Action/节点边界、CommitSnapshot、Final Proposal 引用或内容、真实 RISE 终态和错误摘要。只保留最近事件不足以路由：Controller 使用 Gap Refine、重复 partial overlap/revision 历史以及决策 Memory。

`ResearchMemory` 是科研领域记忆；LangGraph Checkpoint 是可恢复执行状态；`tasks` 是任务队列/租约/业务状态。三者职责分开，不能让任务表另存一份可独立变更的 State/Memory 作为第二事实源。当前 `TrajectoryReplayRecord` 保存结束运行后的 State/Memory/Budget 变化，用于 `reconstruct_context()`；Runner 只在结束时写 artifacts，它不是正在运行的 Checkpoint，也不能单独实现中断恢复。

当前 Action 的外部 LLM/检索调用与未来 Checkpoint 提交无法天然处于同一事务。Worker 若在调用完成而 checkpoint 尚未保存时崩溃，恢复可能再次调用 Provider，存在重复费用/请求风险；P2 必须定义“仅从最后已提交边界继续”的 at-least-once 外部调用语义、Action attempt identity 和事件去重。不能宣称 exactly-once，也不得把 RISE Action 异常纳入 Worker 自动重试。业务事件应在 Checkpoint 已提交后幂等投影，或通过同事务 outbox，避免 checkpoint 与 UI 事件不一致。

## 5. 旧新 Runner 等价比较字段

固定相同 Seed/完整阅读上下文、FakeLLM 序列、FakeRetrieval 结果和预算，对比：

- 有序 `ControllerDecision.action` 与最终 `TrajectoryRunResult.status`、`error_type`。
- 逐步 Research State 全量 Pydantic dump；Research Memory 各分类及 `decision_log`。
- 有序 Trajectory 事件的 `event_type/action/result_ref/result_payload/state_diff/memory_writes/budget/error`（排除非确定 timestamp 或规范化后比较）。
- 每步 Budget 的 max/used/remaining、模块 `scientific_search_count` Delta 与终态 RunBudget。
- Commit Gate 结果、CommitSnapshot、Final Proposal，含 `finalization_failed` 时 Snapshot 是否保留。
- `TrajectoryReplayRecord` 序列及 `reconstruct_context()` 的最终 State/Memory/Trajectory 一致性。

## 6. P0 测试覆盖与代码入口

原有 `test_m11.py` 已覆盖正常 Commit 和完整 Action 顺序、预算耗尽、Gap Refine/Second Partial/Switch、Solution Verification、Inspiration→Revision、Retrieval Action 失败及 replay。`test_m9.py` 覆盖 Controller 路由/预算规则；`test_m10.py` 覆盖 Gate、Snapshot、Commit 与 Finalizer 模块契约。本 P0 仅加强 Provider 失败后的单次调用断言，并补 Runner 级 Finalizer 失败测试，不改科研实现。

主要源码入口：`runner/pilot.py`、`runner/trajectory_runner.py`、`controller/scientific_controller.py`、`controller/budget.py`、`trajectory/context.py`、`serialization/trajectory.py`、`models/runner.py`、`errors.py`、`modules/commit.py`、`modules/finalizer.py` 及对应 `modules/*`。

## 7. P1 前确认项与验证记录

- **Controller / Commit Gate 文档冲突**：`docs/implementation_spec.md` 第 3.11 节写明 Gate 通过后 Controller 仍可选择 `SEEK_INSPIRATION`；当前 `ScientificController` 的 commit-ready 分支仅返回 `COMMIT`，且 `tests/test_m9.py::test_commit_ready_state_does_not_reopen_inspiration_without_new_evidence` 固定了这一行为。P0 依用户要求以当前代码冻结为准；P1 前需确认是将方法文档更新为当前代码，还是另行授权改变 Controller 行为。P0 不改 Controller 或方法定义。
- `IMPLEMENTATION_STATUS.md` 的历史记录称 `data/` 有 37 篇 Markdown；本次实际检查为 257 篇、约 21 MiB。原始输入均保留并进入初始基线；约 278 MiB 的 `outputs/` 和 15 MiB 的 `cache/` 均未纳入 Git。
- 验证：`PYTHONPATH=src:. /home/tmh/miniconda3/envs/myrag/bin/pytest -q`（项目根目录执行），`333 passed`；目标文件 `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_m11.py' -q`（项目根目录执行），`14 tests OK`。测试使用离线 Fake 输入。
