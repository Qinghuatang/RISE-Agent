# RISE Engineering Specification V1

> 状态：四轮架构决策已确认；具体接口与字段需在 P0 对照源码校验。  
> 源码目录：`/home/tmh/project/rise-agent`  
> 目标：在不改变 RISE 科研语义的前提下，将现有离线科研 Agent 演进为可单机部署、可观测、可恢复的 Agent 应用服务。

## 1. 目标、约束和非目标

### 目标

- 保留完整论文 Markdown 输入、当前科研方法及最终 Proposal 生成行为。
- 使用 LangGraph 接管原 `TrajectoryRunner` 的执行编排；Scientific Controller 仍负责科研决策。
- 提供 FastAPI、后台 Worker、PostgreSQL 任务队列、同步持久化 Checkpoint、SSE 事件、异常恢复及简洁 Web UI。
- 提供自动化回归、故障注入测试、结构化日志、部署与可复现的工程交付资料。

### 不可变科研约束

- 保留 `ScientificController` 的 Action mask、合法动作、路由选择及 Commit Gate 的**当前实际代码行为**（Gate 通过直接 Commit）；不要依据存在差异的旧文档擅改。
- 保留 `ResearchState`、`ResearchMemory`、`TrajectoryContext`、`BudgetTracker`、所有科研模块、Prompt、科研检索接口、Patch 确定性合并与实验设计的现有语义。
- 保留 Gap Refinement、Gap/Solution Verification 目标选择、Inspiration→Revision 上下文依赖、Step/Search 预算、CommitSnapshot/Finalizer 的真实执行顺序与失败语义。
- 完整 Markdown 的解析和科研输入边界不变；应用层可拒绝不合法或无法容纳的输入，**不可静默截断或替换摘要**。
- 原 `TrajectoryRunner`/CLI 与离线回归工具在迁移验收前保持可运行；测试不得随意放宽以掩盖科研行为变更。

### 第一版非目标

- 不做科研算法创新、Prompt 改写、模型微调或新的科研评价规则。
- 不做公网匿名开放、完整用户认证、复杂多租户、高并发集群、Kubernetes、微服务化、强制 Redis/Celery、全量 MCP 改造或新向量数据库重建。
- 不承诺外部 LLM/检索调用 exactly-once；不实现用户手动暂停/通用 Resume 或取消后恢复。

## 2. 技术架构与职责

1. **Scientific Domain Layer**：保留现有 `modules/`、`models/`、`controller/`、`memory/`、`llm/`、`retrieval/`、`ingestion/` 等科研实现。
2. **Agent Runtime Layer**：LangGraph `StateGraph`、Controller Node、Scientific Action Nodes、Conditional Routing、适配器、PostgreSQL Checkpointer（`sync` durability）。一个高层 Scientific Action 对应一个主要节点，内部原有科研操作不强制拆分。
3. **Task Execution Layer**：PostgreSQL 任务表队列、独立 Worker、租约/心跳、原子领取、并发槽位 1、取消检查和安全恢复。
4. **Application Layer**：FastAPI REST API、从持久化事件表恢复读取的 SSE、轻量前端。
5. **Infrastructure**：PostgreSQL 业务表 + 官方 LangGraph Checkpoint 表，本地持久化文件卷，外部 LLM 推理服务，结构化日志和后续 Langfuse。

优先采用 Python、Pydantic、LangGraph、FastAPI、SQLAlchemy、Alembic、pytest、Docker Compose。模型服务独立于 Agent 部署，可通过已有 `LLMClient`/provider 接口接入 API 或独立 vLLM 服务；不得改变科研默认模型行为。

## 3. 运行时迁移契约

- 保留旧版 Runner 作为行为参考。新增 `RISEActionAdapter`：从已持久化 Graph State 重建临时 `TrajectoryContext`/预算/模块依赖 → 执行一个原有高层 Action → 校验领域数据 → 返回可序列化的 Graph State 更新。
- Graph State 应包含 schema 版本、task/thread ID、输入引用、Seed 上下文、科研 State/必要 Memory、预算、执行恢复必需轨迹/上下文、Controller 决策、Commit 快照、Final Proposal 或引用、终止结果。具体字段以 P0 源码审计和序列化验证为准。
- **LangGraph Checkpoint 是科研执行状态的唯一权威快照**；业务任务表不维护第二份可修改的 Research State。LLM Client、网络连接、数据库 Session 等运行时对象不得序列化进 Checkpoint。
- 大体积论文、附件、Proposal、归档和必要文献内容放入持久化文件卷，Graph State 使用内容校验过的稳定引用。先原子保存文件，再将引用放入可能提交的状态。
- `sync` durability 在 Graph 步骤之间持久化。Action 内崩溃可能重新执行尚未提交的整个 Action；科研上已确认且持久化的步骤不能因恢复而重复提交。
- 当前科研模块失败继续按旧版错误路径处理；基础设施失败属于任务执行层，不应伪装为科研决策。

## 4. 数据契约与一致性

### 核心业务表

| 表 | 内容 |
|---|---|
| `tasks` | `task_id`、`thread_id`、`task_status`、`rise_outcome`、输入引用、幂等请求键/摘要、租约所有权/有效期、心跳、恢复次数、错误摘要、时间戳 |
| `task_events` | 事件唯一 ID、task_id、递增事件序号、类型、Action、摘要、时间戳；支持断线续传和幂等插入 |
| `task_artifacts` | artifact_id、task_id、类型、相对路径、大小、校验值及创建时间 |

另由官方 PostgreSQL Checkpointer 管理检查点相关表，业务代码不重建其内部 Schema。数据库访问由 SQLAlchemy 2.x + Alembic 管理业务表迁移。`task_id`/`thread_id` 采用稳定一对一映射；一次新的科研实验创建新任务，不复写旧 Thread。

### 双状态语义

- `task_status`：`queued`、`running`、`completed`、`failed`、`cancelled`；如需要内部 `cancel_requested`/`recovering` 标记，作为明确的附属字段或非终态，不混淆最终状态。
- `rise_outcome`：沿用源码实际枚举（例如 `committed`、`stopped`、`failed`、`finalization_failed`），在 P0 确认，不擅自引入新科研枚举。
- `completed` 只表示一次 RISE 流程正常终结，不等于必定 `committed`。科研失败可作为 RISE 正常终结的结果；真正 Worker/基础设施不可恢复异常对应任务 `failed`。

### 轨迹与应用事件

- 原有 `Trajectory` 是科研轨迹语义来源；`task_events` 是其面向前端的持久化投影，不反向影响科研决策。
- 已完成 Action 的 `action_completed`、`state_updated` 等事件，应根据确认持久化的结果幂等写入，使用稳定事件键/唯一约束。若 Checkpoint 已提交但事件写入失败，恢复/对账负责补齐。
- `action_started` 允许作为尝试级实时状态单独记录，不等于科研 Action 已提交；恢复后可补充尝试中断状态。
- Markdown、论文资料、Proposal 和归档使用 Task-ID 隔离目录，持久化到 Docker Volume；先临时写入再原子替换，保存内容摘要。第一版保留完整 Checkpoint、轨迹和结果，支持按任务清理并明确处理关联记录；容量策略以后根据实测优化。

## 5. 调度、所有权和恢复

- API 仅写入任务记录与已校验输入，及时返回任务 ID；Worker 独立进程执行，默认只有 **1 个执行槽位**，支持多个 queued 任务排队。
- Worker 用 PostgreSQL 事务配合 `FOR UPDATE SKIP LOCKED` 原子领取任务；心跳必须独立于阻塞的模型/检索调用运行。
- 每个运行任务有 `worker_id`、`lease_token`、`heartbeat_at`、`lease_expires_at`。业务更新应校验当前所有权。
- **不能仅凭租约过期就在同一个 LangGraph Thread 上并发启动新的执行**：恢复前须确认旧执行已经终止/隔离，确保单 Thread 仅一个活跃 Graph 执行者；单机版通过受控进程管理实现这一保障。若无法确认，进入可诊断的安全失败/待处置状态，而非冒险并行运行。
- Worker/容器/基础设施失败自动从最后可靠检查点恢复，恢复尝试上限初始 3 次（可配置）。科研 Action 自身失败沿用原 RISE 语义，不额外无限重试。恢复后先对账业务任务状态、科研终态、事件及 artifacts。
- 优先保证科研状态和事件/文件副作用幂等；外部 LLM、检索请求允许因未提交 Action 重试，明确记录可能的重复计费。外部调用设置可配置超时，整体任务只监控时长、不随意引入新的科研终止规则。
- 用户取消为终态：API 写入取消请求，Worker 在高层 Action 的安全完成/持久化边界停止后续动作并标记 `cancelled`；不支持主动取消后恢复，已终态任务不得继续执行。
- 创建任务支持可选 `Idempotency-Key`：同键同请求返回已有 task_id，同键不同内容报冲突；独立提交相同 Markdown 仍可创建多个实验。

## 6. HTTP API 与前端事件

- `POST /api/tasks`：上传 UTF-8 `.md` 文件或粘贴 Markdown，内容非空、限制文件大小、校验编码及当前模型上下文可容纳性；错误应明确，不做静默裁剪。支持可选幂等键。
- `GET /api/tasks`：分页任务列表。
- `GET /api/tasks/{id}`：任务生命周期、科研结果状态、进度和必要摘要。
- `GET /api/tasks/{id}/events`：持久化 `task_events` 的可重连 SSE；事件含稳定 ID/序号，支持 `Last-Event-ID` 补发。
- `GET /api/tasks/{id}/result`：返回最终 Proposal 或未 Commit 的正常结束结果。
- `POST /api/tasks/{id}/cancel`：请求边界安全取消。
- `GET /health/live`、`GET /health/ready`：存活与依赖就绪检查。
- 第一版不公开通用 Resume 接口；自动恢复由 Worker 负责。前端只需 Markdown 提交、任务列表、Action 级进度、Research State 摘要及 Proposal 查看；无需 Token 流或复杂 UI。
- 仅用于本地/受控环境。未实现认证时，API 默认绑定本机地址；不应默认暴露公网。

## 7. 可观测性与测试要求

- 日志至少支持 Task ID、Action、Controller 决策、模型/检索调用耗时及错误、预算/Token、Checkpoint、租约与恢复、终态。敏感字段/API Key 不得输出；发送给外部 Trace 平台的原文/Prompt 由显式配置控制。先结构化日志，P4 再集成 Langfuse。
- 测试分层：Unit（Controller/State/Budget/Patch/序列化）、Graph Regression（旧版 Runner vs LangGraph）、Integration（API/Worker/数据库）、Fault Injection、端到端。
- 回归用 FakeLLM/FakeRetrieval，比较 Action 序列、关键状态、Memory、Budget、Commit 与失败/终止路径，而非仅比较 Proposal 文字。覆盖 Gap Refine/Switch、Verification 目标、Partial Overlap、预算耗尽、Commit、科研失败、Finalizer 失败。
- 故障注入至少覆盖：Checkpoint 后 Worker 崩溃、Action 内崩溃、Checkpoint 后事件丢失、租约过期与旧执行并发防护、安全取消、API 重启、数据库暂时不可用、幂等提交。
- 不得删除/放宽失败测试以伪造兼容；未运行的测试不得声称通过。

## 8. 部署与交付

- Docker Compose 三个核心服务：FastAPI、Worker、PostgreSQL；数据库与文件数据使用持久化 Volume，前端可由 FastAPI 提供静态页面。
- API 默认映射 `127.0.0.1`，数据库不公开端口；配置使用 `.env.example` 和环境变量，真实密钥不得提交 Git。明确初始化/数据库迁移步骤，添加健康检查、重启策略、版本锁定和 README。
- Git 在原目录初始化：先检查 `.gitignore`、敏感信息和生成文件，再建立初始科研版本提交，后续分支推进工程化。旧版 CLI 在迁移验证前始终保留。

## 9. 分阶段实施与验收门槛

| 阶段 | 工作 | 必须验收 |
|---|---|---|
| **P0 基线** | 源码审计、Git 基线、测试清点与补齐、Fake 数据固定、状态/预算/轨迹契约冻结 | 旧版真实测试运行结果可复现；当前代码与旧文档差异已记录；不修改科研行为 |
| **P1 LangGraph** | Graph State、Action Adapter、Controller routing、逐 Action 迁移、终止和 Finalizer | 新旧离线执行在固定替身下行为等价，保留旧 Runner |
| **P2 Checkpoint** | PostgreSQL Checkpointer（sync）、持久化/恢复、状态 Schema、Action 级崩溃测试 | 从持久化动作边界恢复；无已提交科研动作重复；结果/事件一致性可恢复 |
| **P3 服务** | FastAPI、PostgreSQL 任务队列、单 Worker、租约/心跳/恢复/取消、SSE | 提交立即得到任务 ID；排队执行；崩溃自动恢复且同任务不并发；API 不阻塞 |
| **P4 观测/评测** | 结构化日志、指标、Langfuse、固定工程测试与故障注入 | 可定位失败原因；回归和故障测试有真实报告 |
| **P5 展示/部署** | 轻量前端、Compose、CI、README、部署说明、演示 | 新环境可以按文档启动，演示端到端运行、轨迹查看和中断恢复 |

每阶段均提交：设计与实际修改摘要、变更文件、重要取舍、准确测试命令/结果、对科研行为的影响、遗留问题。未经用户明确确认不得提前进入下一阶段或改变科研算法/Prompt/Controller 决策。优先实现可运行、可验证的单机系统，不做不必要的扩展。

## 10. 启动前的代码对齐检查（不是改变已定方案）

1. 核实实际科研枚举、Controller 决策、Commit Gate、Budget 消耗、`TrajectoryContext` 快照和可序列化对象，以及旧版测试目录/有效测试数。
2. 确认完整 Markdown 的输入长度、当前 LLM Provider 超时及检索缓存语义；只调整工程级校验与配置。
3. 设计 Graph State 到临时 Context 的双向适配与版本迁移；明确哪些轨迹字段是科学决策必需、哪些仅供观察。
4. 制定 Action Checkpoint 之外的恢复/幂等限制；尤其确保租约 fencing 不能被误认为 LangGraph Checkpointer 的自动并发锁。
5. 若源码与本规范存在事实冲突，**先报告差异和建议，不擅自更改科研方法**。

**规范地位**：本文件为工程化工作约束。实际科研行为以冻结时的源码为准；实现细节若需调整，先说明原因并获得明确确认。
