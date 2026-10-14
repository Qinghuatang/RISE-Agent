# P2-A：Checkpoint 与故障恢复机制设计

> 状态：设计完成，待确认后进入 P2-B。
> 依据：当前 P1 Graph 实现、P0 行为契约及 RISE 工程规范。
> 约束：本设计不改变科研模块、Prompt、Controller、预算或科研终态语义。

## 1. 现状与结论

当前 `LangGraphTrajectoryRunner` 使用 LangGraph 1.2.15，但编译 Graph 时未注入 Checkpointer；`run()` 每次都构造初始状态，没有恢复入口。`RISEGraphState` 已保存 Research State、Research Memory、完整 Trajectory、Budget、Replay、Controller 路由决策、CommitSnapshot、Final Proposal 和 RISE 终态。`RISEActionAdapter` 在节点边界重建临时 `TrajectoryContext`，并以完整快照替换列表字段，没有追加式 reducer。

方案可行。P2-B 需补齐 PostgreSQL Checkpointer、稳定 `thread_id`、新建/恢复分离入口和论文输入解析器；不需要改科研核心。

当前项目的 `agent` extra 只包含 `langgraph==1.2.15`。实际开发环境为 LangGraph 1.2.15、`langgraph-checkpoint` 4.3.0，未安装 PostgreSQL Checkpointer 或 psycopg。建议采用官方 `langgraph-checkpoint-postgres` 的同步 `PostgresSaver`，与当前同步 `graph.invoke()` 匹配。候选锁定版本为 3.1.3；该版本发布于 2026-10-12，需在 P2-B 安装并验证依赖解析、PostgreSQL 连接及 Graph 集成后，才作为最终锁定版本。3.1.2 的官方依赖声明要求 `langgraph-checkpoint>=4.3.0,<5.0.0`，与当前环境版本匹配。

## 2. Checkpointer 集成

- 编译使用 `graph.compile(checkpointer=postgres_saver)`；Graph 调用设置 `durability="sync"`。LangGraph 会在下一执行步开始前同步持久化本步状态，但不会在节点内部的每个 LLM/Retrieval 操作后保存。
- 使用任务 UUID 作为稳定 `thread_id`，并让 `RISEGraphState.run_id` 与之对应。恢复时复用原 ID；禁止对已有 Thread 重新发送初始 Graph State。LangGraph 保存和读取线程快照时通过 `thread_id` 定位状态。
- 由运行时资源管理器持有同步 `PostgresSaver` 及数据库连接，覆盖 Graph 编译与执行。首次部署或数据库初始化时调用 `PostgresSaver.setup()`；Checkpointer 内部表及其迁移由该库管理，不纳入业务 Alembic。
- Graph State 使用现有 `to_payload()` 的 JSON 兼容快照。LLM Client、检索 Provider、数据库连接、Session 和 `FullPaperReadingContext` 等运行对象不进入 Checkpoint。
- Graph State `schema_version` 和应用 Graph 版本必须校验。遇到不支持的旧版本时，显式迁移或停止并报错；不得静默丢弃字段或从 Bootstrap 重开任务。

LangGraph Checkpoint 是科研执行状态唯一权威来源。未来业务 `tasks` 表只保存 task/thread ID、队列生命周期、输入引用和错误摘要，不重复存放可独立修改的 Research State/Memory。`ResearchMemory` 是科研领域记忆；Checkpoint 是当前 Thread 的执行快照；业务表保存任务管理元数据。

## 3. 恢复边界

恢复入口以同一 `thread_id` 检查 LangGraph 的最新 State Snapshot、待执行节点和 RISE 终态；使用当前 Graph 从已有 Checkpoint 继续，不手动选取旧快照，不重新传初始 State。终态检查优先于执行恢复。

| 中断位置 | 恢复行为 |
|---|---|
| Bootstrap 节点完成并保存 | 从 Controller 继续；Bootstrap 不计 Action Step/Search Budget。 |
| Controller 决策已保存、Action 未开始 | 执行 Graph State 中保存的决策，不重调 Controller。 |
| Scientific Action 完成并保存 | 从后续 Controller 或 Finalizer 继续；已提交 Action 不重放。 |
| Scientific Action 节点执行中、输出未保存 | 从该高层 Action 节点起点重放整个节点。节点内尚未保存的上下文修改丢弃；外部 LLM/Retrieval 调用可能重复。 |
| Commit 已保存、Finalizer 未运行 | 保留 CommitSnapshot，只运行 Finalizer。 |
| Commit/Finalizer 节点结果未保存 | 从最近保存边界重放相应节点。Commit 是快照生成；Finalizer 可能重复一次 LLM 调用。 |
| RISE 终态已保存 | `committed`、`stopped`、`failed`、`finalization_failed` 直接返回已有结果，不再执行节点。 |
| 非终态但 Graph 没有待执行节点 | 作为 Checkpoint/Graph 版本不一致错误处理；不能重新从 START 运行。 |

同步持久化仅提供节点边界的持久化保证。未保存节点允许按既定 at-least-once 边界重放；不承诺 LLM、检索或节点内其他副作用 exactly-once。不得给科研 Action 配置 LangGraph retry policy。已保存的科研 Action 错误仍以旧 Runner 语义终止；Worker 不得把持久化的 `rise_status="failed"` 当作可重试任务。

存在不可消除的模糊窗口：Action 已在外部执行，但进程在其结果 Checkpoint 确认前崩溃，恢复只能从上一 Checkpoint 重放该节点。若 Action 的失败结果已保存，恢复不会重试；若失败结果未保存，系统无法仅凭 Checkpoint 判断异常是否已经发生。这是 at-least-once 执行边界，不是新增科研错误重试策略。

## 4. 输入、状态与副作用一致性

### 论文输入

当前 `PaperInputReference` 只校验 locator 和 SHA-256 字段格式，`FullPaperReadingContext` 由 Graph Runner 闭包提供。P2-B 应在启动 Graph 前先将 Markdown 原子写入稳定存储，记录不可变 locator 和内容 SHA-256；Bootstrap 恢复时重新读取、核对摘要，并按当前解析器重建完整阅读上下文。文件缺失或摘要不符时应停止恢复并报告输入完整性错误，不可换用新文件、截断或重新初始化。

### State、Memory、Trajectory 与预算

- 每个节点从完整 `RISEGraphState` 建立临时 Context/Budget，完成后返回完整、校验过的快照。`trajectory_events`、`replay_records`、Memory 和 Budget 按快照替换，不在恢复入口额外 append。
- 从 Action 前的 Checkpoint 重放时，未保存的 State/Memory/Trajectory/Replay/预算更改不会进入后续运行；重放成功后只提交一次完整 Action 结果。
- `TrajectoryEvent.step` 是事件序号，不代表 Action Budget；预算仍由 Adapter/旧 Runner 的原计数规则更新。
- CommitSnapshot 留在 Graph State。Finalizer 成功 Proposal 或 `finalization_failed` 结果持久化后，重复恢复均返回同一终态。

### Checkpoint 与业务事件

Checkpoint 写入错误属于基础设施故障，向 Worker 传播；不能转换成 RISE `failed`，也不能从内存结果假设已保存。后续 P3 的任务事件应以已保存的 Graph State/Trajectory 幂等投影，并能从 Checkpoint 对账补齐。Checkpoint 与业务事件不是同一事务；事件表不能反向修改科研状态。

## 5. 业务状态映射

| RISE 终态 | 业务 `task_status` | `rise_outcome` |
|---|---|---|
| `committed` / `stopped` | `completed` | 保留对应 RISE 终态 |
| `failed` / `finalization_failed` | `failed` | 保留对应 RISE 终态 |
| 尚未恢复完成的基础设施错误 | 恢复期间保持非终态；重试耗尽后 `failed` | `null`，除非 Checkpoint 已有真实 RISE 终态 |

这只是未来业务层映射；P2 不建立任务队列表。

## 6. 实施与验收

### P2-B：PostgreSQL Checkpointer 与恢复入口

1. 在 `pyproject.toml` 增加 PostgreSQL Checkpointer 可选依赖并固定经实测兼容的版本。
2. 新增运行时 Checkpointer 工厂/生命周期管理和一次性 `setup()` 入口。
3. 修改 `runtime/graph_runner.py`，支持注入 Checkpointer、`durability="sync"`、稳定 thread ID，以及互斥的 `start` / `resume` 流程。
4. 为恢复路径增加论文引用解析与摘要验证；对 Graph State Schema 或 Graph 版本不兼容明确失败。
5. 不修改 Controller、科研模块、Prompt、预算规则或旧 `TrajectoryRunner`。

P2-B 验收：启用 Checkpointer 的 Graph 每步完成后都能读取完整有效快照；首次执行与恢复严格分流；已有 Thread 不会被初始输入覆盖；Checkpointer/DB 故障作为基础设施错误传播；不启用 Action 自动重试。

### P2-C：持久化与故障注入测试

使用真实 PostgreSQL 和独立进程运行 FakeLLM/FakeRetrieval 场景，至少覆盖：Bootstrap 后、Controller 决策后、Action 完成后、Action 执行中、Commit 后、Finalizer 中断、PostgreSQL 写入失败、输入摘要错误、Schema 不兼容、终态重复恢复。

验收：已保存的 Action 不重复；节点执行中断时只按文档边界重放；恢复后 State/Memory/Trajectory/Replay/Budget/CommitSnapshot/Proposal 正确；持久化的科研失败和 Finalizer 失败不重试；原 P0/P1 离线回归测试继续通过。测试报告必须记录实际使用的命令、数据库版本和结果。

## 7. P2-B 启动前需处理的技术点

1. 为现有总是初始化的 `run()` 增加显式恢复入口，避免 Thread 被重新初始化。
2. 从经过摘要验证的论文引用重建 Bootstrap 所需阅读上下文。
3. 在 P2 单进程执行时串行化同一 Thread；P3 Worker 再使用租约/fencing 解决跨进程单一执行者问题。Postgres Checkpointer 本身不提供业务任务租约。
4. 当前环境没有 PostgreSQL Checkpointer/psycopg，包版本与实际数据库恢复行为需在 P2-B/P2-C 完成实测。
5. 完整 Trajectory 和 Memory 随每个节点更新，PostgreSQL 会保留多个 Checkpoint 版本；先在 P2-C 记录 Checkpoint 数量/体积，终态清理与保留策略依据实测另定，不在 P2-A 引入剪枝。

## 参考

- [RISE 工程设计规范](RISE_Engineering_Specification_V1.md)
- [P0 科研行为契约](P0_BEHAVIOR_CONTRACT.md)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Durability](https://reference.langchain.com/python/langgraph/types/Durability)
- [PostgresSaver README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md)
- [langgraph-checkpoint-postgres 3.1.3 release](https://github.com/langchain-ai/langgraph/releases/tag/checkpointpostgres%3D%3D3.1.3)
