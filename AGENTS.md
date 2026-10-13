# AGENTS.md

## 项目边界

- `docs/method_overview.md` 与 `docs/implementation_spec.md` 共同构成本项目的 method-level source of truth。实现、测试、重构和实验解释都必须以这两份文档为准。
- 不得为了工程便利自行增加、删除或改写科研模块、Agent 层级、Action、状态、评分公式、dependency graph、memory agent、planner 或 critique/reflection loop。
- 如果方法文档之间存在冲突、遗漏或无法直接落地的地方，必须记录为 open issue 并报告；不得通过修改科研定义来消除问题。
- `data/` 是原始论文 Markdown 数据。不得修改、清洗覆盖、重命名或删除其中的原始文件。任何规范化结果都必须写入独立的缓存或输出目录。
- 当前输入从 MinerU 生成的 Markdown 开始；本项目第一版不实现 PDF → Markdown。

## 工程实现约束

- Scientific modules 只接受结构化输入并返回结构化输出；不得用未声明的自由格式字符串在模块之间传递状态。
- 模块不得直接调用 DeepSeek HTTP/API。所有 LLM 推理必须依赖 `LLMClient`；`DeepSeekClient` 只能作为该接口的基础设施实现。
- 论文检索必须依赖 provider/interface adapter。科研模块不能深度耦合 Semantic Scholar、arXiv 或其他具体 API。
- API key、base URL、model name、超时和预算等配置必须通过环境变量或配置对象提供，禁止写死在源码中。
- Prompt 模板与 Python 业务逻辑尽量分离；结构化输出必须经过 schema 校验。
- Research State 只保存当前有效状态；论文内容、证据、失败方向、检索记录和决策历史写入 Research Memory 或 trajectory output。
- Targeted Revision 必须遵守方法文档中的 lightweight patch + deterministic merge；不能用一次完整重写替代确定性合并。
- Finalizer 是只读整理模块，不得修改 Research State、添加未经验证的机制或虚构实验结果。

## 测试和迭代纪律

- unit tests 默认禁止真实 DeepSeek/API 调用和真实论文网络检索；使用 fake/stub `LLMClient` 与 provider。
- 每个 milestone 完成后必须运行测试，并记录验证结果。
- 需要真实 LLM 或网络的测试只能显式 opt-in，不能成为默认测试路径。
- Real Single-Paper Integration Pilot 必须先于 batch execution；没有显式 pilot flag 和 credentials 时不得触发真实 LLM 或论文网络请求。
- 实现失败时不得改变科研定义、放宽状态条件或伪造 evidence 来让测试通过。
- 任何 method-level ambiguity 都必须进入 `IMPLEMENTATION_STATUS.md` 的 open issues，并在实现前获得人工确认。
- 第一版优先简单、清晰、可测试和可记录 trajectory；不得提前引入向量数据库、memory agent、复杂 planner 或生产级分布式架构。
- Real-paper smoke tests are evaluation cases, not prompt-specialization targets. Failures observed on real papers may motivate prompt improvements only after being abstracted into paper-agnostic reasoning principles. Do not add paper names, paper-specific terminology, components, algorithms, formulas, heuristics, or special-case rules to prompts or scientific modules to improve one smoke-test result.
- Gap Discovery must preserve scientific exploration: absence of a demonstrated failure does not imply absence of a research gap. A valid Seed Paper should yield at least one grounded method-level research opportunity; this must never be achieved through paper-specific prompt specialization or unsupported failure claims.

## 数据与输出

- 输入 Markdown 的原文只读；解析、LLM 抽取和规范化结果写入独立输出/缓存。
- 每次高层 Scientific Action、模块结果、状态变更、预算变化和 Memory 写入都应能在 trajectory 中追踪。
- 任何缓存都必须能够区分输入 paper、配置、prompt/schema 版本，避免实验和 ablation 之间静默复用错误结果。
- Related-work 与 Inspiration provider 的全局缓存只能保存无密钥的成功 provider response；
  必须使用稳定 request key、原子写入和（未来并行时）miss-check/request/write 锁，缓存命中
  不得伪造新的 provider request 或 scientific search 记录。
