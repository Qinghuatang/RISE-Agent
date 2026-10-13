# Scientific Research Ideation Agent：第一版架构与实现契约

本文记录当前阶段的架构与已确认 contract，不改变 `docs/` 中的方法定义。若本文与方法文档冲突，以方法文档和已记录的人工裁决为准。

## 1. 方法理解与整体 workflow

给定一篇 Seed Paper，系统把科研想法生成建模为外部证据驱动的状态演化：

```text
MinerU Markdown
  -> SeedPaperContext
  -> Candidate Gaps (<= 5)
  -> Active Gap
  -> Gap Verification
  -> Solution Ideation (3 candidates; select + develop)
  -> Solution Verification
  -> {Inspiration Retrieval / Targeted Revision / Experiment Design}
  -> re-verification when required
  -> Commit Gate
  -> Controller chooses COMMIT
  -> frozen CommitSnapshot
  -> read-only Finalizer
```

Research State 表示“当前正在研究什么以及它处于什么状态”；Scientific Controller 只决定下一项科研能力，不执行论文阅读、生成、比较或修改。完整证据和历史不塞入 State，而放进 trajectory-local Research Memory。

### 1.1 Research State

Research State 的最小结构是 `active_gap`、`candidate_solutions`、`current_solution` 和 `experiments`。它只保存下一步决策需要的当前有效内容：

- Active Gap：问题导向的研究缺口，以及 confidence、priority、状态和证据引用。
- Candidate Solutions：按文档固定生成三个轻量候选，只有 `core_idea`、`rationale` 和 candidate/selected/rejected 状态。
- Current Solution：选中的核心方案、rationale、2–5 个 Components、整体 `novelty_status` 和 evidence refs。
- Experiments：关键实验及其 ready/stale 状态。

`unchecked`、`distinct`、`partial_overlap`、`core_overlap` 是 Current Solution 的整体新颖性状态。它们不是 Component 的新颖性标签；Component 只承载方法文档规定的修改/失效状态和 evidence refs。

### 1.2 Scientific Controller

固定 Action Space 只有：

```text
SOLUTION_IDEATION
SEEK_INSPIRATION
VERIFY_RELATED_WORK
TARGETED_REVISION
DESIGN_EXPERIMENT
SWITCH_GAP
COMMIT
```

Controller 先应用 deterministic rule、allowed-action mask 和 budget 限制。下一步唯一明确时不调用 LLM，例如：

- 新 Active Gap → `VERIFY_RELATED_WORK`。
- 有效 Gap 且没有可用 Solution → `SOLUTION_IDEATION`。
- 新形成或修改后的 Solution → 按影响范围重新验证。
- Gap resolved → `SWITCH_GAP`。
- `current_solution == null` 且所有 Candidate Solutions 都是 `rejected` → `SOLUTION_IDEATION`，重新生成三个候选。

只有存在多个合法方向时才调用 LLM Controller，例如 `partial_overlap` 下在 `TARGETED_REVISION` 与 `SEEK_INSPIRATION` 中选择，或 `distinct` 下在 `SEEK_INSPIRATION` 与 `DESIGN_EXPERIMENT` 中选择。Step budget 按高层 Action 计数，search budget 只按实际外部搜索计数；命中 cache 不增加 search。

M12.2 的 Inspiration 边界后来按 correctness hardening 收敛：
`SEEK_INSPIRATION` 不是 solution 已存在时必然跳过，也不是 distinct 或 high-priority gap
出现时必然触发。它只在已有 State/Verification 条件和 productive reserve（至少一个
search、至少三个后续 step）允许时进入合法 action mask。Controller 不再检查 solution
文字中的 lexical cue，也不使用关键词、正则、同义词表、embedding classifier、innovation
score 或数值公式。只有多个合法 Action 时才调用 Controller LLM；单一合法 Action 仍
deterministic。这样保持 Inspiration 可选，同时不把某篇论文的措辞变成 routing rule。

### 1.3 Gap Discovery and Management

初始 Gap 只从 Seed Paper 生成，不主动检索外部论文。模块生成、评估并保留 1–5 个有实质差异的 Candidate Gaps，依据为 method relevance、scientific value、actionability、evidence strength、priority 和 researchability；它不使用人为数值评分公式。正常成功的初始生成不返回空 agenda。

Gap Agenda 保存 candidate/active/resolved/rejected 状态。选出 Active Gap 后，必须先交给 Related-work Verification：

- `valid`：保留 Gap，进入 Solution Ideation。
- `partially_resolved`：最多执行一次 RefineGap，只收窄/明确原问题，再验证一次。
- `resolved`：标记失效并从既有候选中 Switch Gap。

第一版 `GapAgenda` 是 Gap 模块的独立结构化结果，最多包含五个 Gap；它不属于
`ResearchState` 或 `ResearchMemory`，后续需要通过 trajectory 记录 `gap_refined` 等事件。

### 1.4 Solution Ideation

只有通过 Gap Verification 的 Active Gap 才能形成方案。第一次 LLM 操作生成三个核心机制实质不同的 Candidate Solutions；第二次 LLM 操作同时完成候选选择和展开，生成 2–5 个 Components 的 Current Solution。该模块不主动使用外部 Inspiration，也不负责新颖性判断。Current Solution 形成后强制进入 Solution-level Related-work Verification。

### 1.5 Related-work Verification

第一版使用 seed-paper-centric retrieval：首次需要验证时获取一个相关论文池（文档指定 Top 20 和 Semantic Scholar 方向），后续 Gap、Solution 和 focused re-verification 复用 cache，不重复搜索。

Verification 只有两个任务：`VERIFY_GAP` 和 `VERIFY_SOLUTION`。每次先从论文池筛选最多五篇相关论文，再基于标题和摘要做结构化比较。`VERIFY_SOLUTION` 同时看 Core Idea 和 Components，可使用 `full_solution` 或 `changed_units` 作为验证范围。

Solution 的三种新颖性判断边界如下：

| 状态 | 证据含义 | 后续含义 |
| --- | --- | --- |
| `distinct` | 可以共享标准技术、通用组件、相似任务或辅助机制，但没有实质覆盖主要 novelty-bearing mechanism | 可继续找灵感或设计实验；是 Commit Gate 的必需状态 |
| `partial_overlap` | 一个或多个局部 novelty-bearing element 被实质覆盖，但整体核心机制仍有实质区别 | 在 `TARGETED_REVISION` 与 `SEEK_INSPIRATION` 间由 Controller 选择 |
| `core_overlap` | 相同/高度相似问题与实质相同核心机制同时被已有工作覆盖 | 拒绝 Current Solution，保留 Active Gap，尝试剩余候选 |

普通 overlap 不等于 novelty overlap；只有真正削弱主要新颖性主张的部分才进入 `novelty_relevant_overlaps`。

### 1.6 Inspiration Retrieval

该模块不是寻找与当前方案最相似的论文，而是执行 Dynamic Entity Bridging：

```text
Research Need
  -> 5 Bridge Entities
  -> select <= 3
  -> one combined external search
  -> <= 3 grounded Inspiration Candidates
```

Bridge Entity 只能是 scientific concept、mechanism、method、phenomenon 或 system principle。LLM 先抽象 `required_function`，再生成/选择可搜索实体；外部论文只用于 grounding，LLM 提取 `source_mechanism`、`transferable_principle` 和 `adaptation_hint`。搜索不可用时可 LLM-only fallback，但必须标记 `grounding = none`。Inspiration 不直接修改 State，后续修改由 Targeted Revision 完成。

### 1.7 Targeted Revision

触发条件是最新 partial-overlap evidence，或调用方通过运行时参数
`inspiration_record_id` 明确提供的一个 useful InspirationRecord；历史
Inspiration Memory 不会自动成为本次 Revision input。Revision Need 明确
`problem_to_fix`、`affected_units` 和 constraints。LLM 只返回 lightweight patch，优先
修改受影响 Component，必要时同步修改 rationale/Core Idea 以保持一致；它不得重写
无关字段。

代码对 patch 执行 deterministic merge：未出现在 patch 中的字段保持原值；成功后将实际变化的 Component 标记为 `modified`、novelty 置为 `unchecked`，按 `changed_units` 让受影响 Experiment 变为 stale，并固定回到 focused/full re-verification。M7 实现 typed lightweight patch 的语义校验与 merge。Revision/Fault evidence refs 只使用当前 relevant verification refs 和最终实际 selected Inspiration 的 grounded paper ref；未选择的候选不能贡献 evidence。Trajectory step 在 LLM、Memory、State 写入前由 recorder 的无副作用 preflight 校验。若无法保持局部性质，返回 `unable_to_revise`，不强行改写科研方向。

### 1.8 Experiment Design

只有 Research State 中的 Solution 已由最新 `VERIFY_SOLUTION` 验证为 `distinct`、
所有 Component 为 `ready`，且最新 Active Gap Verification 为 `valid` 后，Controller
才可选择 Experiment Design。模块还会从 Research Memory 确认这两条最新 verification，
而不是只信任 State 的 `novelty_status`。它根据 Active Gap、Current Solution、Seed
Paper 的 experiment context、verification summary、既有 experiments 和
verification focus/changed units，识别 2–4 个关键可验证 Claim，生成 2–4 个尽可能少
但足够区分的实验。

`KeyClaimDraft` 与 `ExperimentProposalDraft`/`ExperimentDesignDraft` 只是本次 LLM
调用和 trajectory audit 的内部 artifact，不进入 Research State。LLM 不提供
Experiment ID/status；初次设计由代码确定性分配 `E1...En`，stale redesign 复用原
Experiment ID 并只替换 stale 设计，ready Experiment 原样保留。每个 Claim 和每个
Experiment 必须覆盖关系闭合，至少有一个 `core_solution` Claim 和一个 Component
Claim；`what_it_validates` 必须包含对应的 `core_solution` 或完整 Component ID token，
整体 effectiveness 至少使用精确的 `Seed Method` baseline。

实验优先复用 Seed Paper 的 tasks/datasets、metrics、baselines 和 setting；一般科学
知识只能帮助提出合理控制或 metric，不能伪装成 Seed/Related Work 事实。只设计验证
方式，不预测结果，不执行论文检索。若已有 2–4 个 Experiment 且全部 `ready`，模块
deterministic no-op，不调用 LLM、不写 State 或 trajectory；若存在 stale，只重做
stale，最终数量保持 2–4。

### 1.9 Research Memory

Research Memory 是单篇 Seed Paper trajectory 内的结构化存储，不跨 paper 共享，不引入 memory agent、向量数据库或 planner。它保存：

```text
paper_memory
verification_memory
inspiration_memory
revision_memory
failure_memory
decision_log
```

Memory 通过确定性代码在模块完成后写入；模块只读取与当前任务相关的子集。Failure Memory 只记录 rejected gap/solution/mechanism 和 failed revision，不把普通 shared element 当成失败。RefineGap 的“最多一次”不增加 Research State 字段，而由 trajectory 中的 `gap_refined` 事件和 `has_refined_gap(gap_id)` deterministic query 约束。

读取/写入边界按方法规范固定为：

| 使用者 | 读取的 Relevant Memory | 完成后写入 |
| --- | --- | --- |
| Solution Ideation | rejected solutions、rejected mechanisms | 实际被拒绝的 Solution/mechanism 的 FailureRecord（如有） |
| Related-work Verification | related paper pool、previous verification results | VerificationRecord、论文池/筛选结果缓存 |
| Inspiration Retrieval | previous bridge entities、rejected mechanisms、相关 verification context | InspirationRecord、搜索记录和 paper refs |
| Targeted Revision | latest verification evidence、selected inspiration、failed revisions | RevisionRecord；失败时 FailureRecord |
| Scientific Controller | last action/result、relevant failure memory | DecisionRecord |
| Commit + Finalizer | CommitSnapshot、必要的最新 novelty context 和 evidence refs | CommitSnapshot/FinalResearchProposal 不回写 Research State |

Gap Discovery 使用 Seed Paper 和 Gap Agenda；它只在需要依据验证结果做 refine/switch 时读取相关 Gap Verification 记录。普通 shared elements 不写入 Failure Memory。

### 1.10 Commit + Finalizer

Commit Gate 是 deterministic hard gate，不调用 LLM；必须同时满足：

1. 最近 Gap Verification 为 `valid`；
2. Current Solution 存在且为 developed，包含 core idea、rationale 和 components；
3. `novelty_status == distinct`；
4. Experiments 已设计且当前均为 ready（通常 2–4 个）；
5. 不存在 modified/stale Component、stale Experiment 或 unchecked novelty。

Gate 只回答“能不能结束”。即使 `commit_allowed = true`，Scientific Controller 仍可在预算允许且存在明确提升空间时选择继续 `SEEK_INSPIRATION`；它负责“现在是否值得结束”。Controller 选择 `COMMIT` 后，代码冻结 `CommitSnapshot`。Finalizer 只读 Snapshot 和必要的 novelty context，生成 Research Gap、Proposed Solution、Rationale、Method Components、Novelty Positioning、Experimental Plan 和 Expected Contributions，不回写 State、不增添新机制、不虚构结果。

## 2. State、Memory 和确定性更新的所有权

LLM 永远只产生经过 schema 校验的候选结果，不直接改写 Research State。建议的写入边界如下：

| 组件 | 允许修改/写入 |
| --- | --- |
| Seed context ingestion | 生成独立的 SeedPaperContext 输出，不改原 Markdown |
| Gap Discovery and Management | Candidate Gap agenda，以及 `research_state.active_gap` 的选择、refine、switch |
| Solution Ideation | Candidate 状态和 Current Solution 的形成 |
| Verification application | 根据 Gap/Solution Verification 结果更新 Gap 状态、novelty status、summary 和 evidence refs；core overlap 时拒绝 Current Solution |
| Targeted Revision merge | 以 patch 确定性更新 Current Solution、Component 状态和受影响 Experiment 的 stale 状态 |
| Experiment Design | 写入/替换实验设计并做 coverage/status 检查 |
| Scientific Controller | 不改 State；只返回 ControllerDecision 和 decision log |
| Research Memory store | 只写结构化历史、证据、失败记录、论文缓存和决策记录 |
| Commit module | 不继续演化 State；只检查 Gate 并创建冻结 Snapshot |
| Finalizer | 只读 Snapshot，返回 proposal |

确定性规则包括：Controller rule trigger/action mask/budget、Commit Gate、schema 校验、patch merge、Component/Experiment stale 传播、search/step 计数、cache 命中判断和 Memory 写入时机。需要 LLM 的位置包括 Markdown → SeedPaperContext、Gap 生成/归纳、Solution 生成与展开、Verification 筛选与比较、Bridge Entity/机制提取、Revision patch、Experiment Design、存在多动作时的 Controller 选择和 Finalizer 组织。具体调用次数以实现规范明确的次数为准；没有固定次数的 Gap/Seed 抽取部分只作工程实现选择，不改变科研流程。

## 3. SeedPaperContext 设计

### 3.1 核心 schema

第一版只保留后续模块真正使用的信息：

```python
class ExperimentContext:
    tasks_or_datasets: list[str]
    metrics: list[str]
    major_baselines: list[str]
    experimental_setting: str | None


class SeedPaperContext:
    paper_id: str
    title: str | None
    abstract: str | None
    research_objective: str | None
    core_method: str | None
    main_components: list[str]
    main_findings: list[str]
    limitations: list[str]
    method_constraints: list[str]
    experiment_context: ExperimentContext
```

这是第一版 schema contract，已由 M1 的 Pydantic 模型实现。`paper_id` 建议使用完整输入文件 stem 作为稳定 key，而不是仅依赖标题或猜测外部数据库 ID；这样可以直接关联 trajectory 和 cache。

字段来源与处理规则：

| 字段 | 来源 | 抽取策略 |
| --- | --- | --- |
| `paper_id` | 文件名 | deterministic 取得，不让 LLM 生成 |
| `title` | 首个一级 heading/标题区域 | 可做明确的排版级规范化；不改变语义，不确定时保留原文 |
| `abstract` | `ABSTRACT` heading 下的原文 | 直接保留，必要时仅做空白合并 |
| `research_objective` | Introduction、Problem Setup、Conclusion | LLM 归纳，但必须有原文依据 |
| `core_method` | Method/Methodology/Algorithm 及摘要 | LLM 归纳核心机制，不展开不存在的实现细节 |
| `main_components` | Method 下的主要模块/算法步骤 | LLM 结构化归纳；不把普通实验操作当组件 |
| `main_findings` | Results、Conclusion、摘要 | 只记录论文声称或表格明确支持的发现，不补充新结论 |
| `limitations` | Limitations、Discussion、Conclusion 或明确承认的边界 | 没有明确信息时为空列表，不把模型推测写成论文 limitation |
| `method_constraints` | 作者明确的设计 rationale、约束、trade-off、被拒替代方案及拒绝理由 | 只抽取作者明确支持的约束；不得从设计事实自行推断，也不直接成为 Gap |
| experiment context | Experiments/Experimental Setup/Results | 只记录论文实际出现的 tasks/datasets、metrics、baselines 和 setting |

### 3.2 Markdown → SeedPaperContext

输入不是 PDF，而是原始 Markdown。建议的第一版处理顺序：

1. deterministic 读取并保留原文；解析 heading 层级、section span、HTML table、公式块、图片占位和引用/参考文献边界。
2. 只做版面级处理：合并明显的换行空白、保留公式原文和 table 原文；不做“智能纠错”来改变科学含义。
3. 构建完整的 main-paper reading context：保留从论文开头到 References/Bibliography 或 Appendix/Appendices/Supplementary Material 中最早边界之前的全部正文 section，不按科学相关性筛选、不按字符数切分、不做 synthesis；然后只发起一次 structured LLM extraction call。若 provider context limit 无法容纳输入，显式报错并由人工调整，不自动改变读取方式。
4. 要求 LLM 返回严格 schema，并对每个非空归纳字段提供输入 section/短 source anchor 的审计信息；anchor 是 ingestion audit envelope，不进入科学模块的核心 Context schema。
5. deterministic 校验：字段类型、列表长度/空值、标题/摘要是否来自输入、实验字段是否有原文支持；缺失信息写 `None`/`[]`，不让 LLM 以常识填空。

### 3.3 实际数据观察

`data/` 当前有 37 个 Markdown 文件。文件名形态为：

```text
MinerU_markdown_<numeric-id>_<truncated-title>_<long-export-id>.md
```

抽样检查了：

- `data/MinerU_markdown_10107_ByteFlow_Language_Modeli_2090055877820440591.md`
- `data/MinerU_markdown_13056_Bayesian_Attention_Mecha_2090055813601447937.md`
- `data/MinerU_markdown_14690_Cautious_Optimizers_Impr_2090055813601447940.md`

三篇分别约 816/1,046/1,267 行、11,190/16,646/17,715 词，约 84–106 KB。共同特点和风险是：

- 标题在首个 `#`，摘要在文件开头的 `## ABSTRACT` 下，通常容易定位，但标题含 `ADAP-TIVE`、`COM-MONSENSE` 等断词。
- Section heading 以 `## 1 ...`、`## 3.1 ...`、`## REFERENCES`、`## A ...` 等形式出现，appendix 和 references 可能很长。
- 公式同时出现 `$$...$$`、行内 `$...$`、HTML `<eq>`、`<sup>/<sub>`；MinerU 会把 LaTeX 命令和下标拆成带空格的形式，如 `$p _ { i j }$`。
- 表格不是 pipe Markdown，而是单行 `<table><tr><td>...` HTML；数值、rowspan/colspan 和 `<eq>` 可能全部挤在一行，必须按 HTML/标签解析或作为完整 section 文本传递。
- 图片以 `![image](https://cdn-mineru...)` 占位并附 Figure caption；第一版 SeedPaperContext 不应假设能从图片恢复曲线/表格细节。
- 有明显转换 artifact：ByteFlow 中 `high resolustion`、`languagespecific`、`codingrate` 等词粘连/拼写问题；Cautious Optimizers 的算法块被转换成 `# param ...` 的伪 heading，且公式中出现 `OP T`、`C AdamW` 等分裂文本。
- References 通常有明确 heading，但前文大量作者年份引用；不能把 references 当作实验 baseline 或 limitation。

因此抽取器应依赖 section-aware parsing，而不是只用正则匹配标题/摘要；应把公式、table 和 citation 当作需保留的证据文本，并允许摘要/实验上下文缺失。Seed 抽取阶段把完整 main-paper body 一次发送给 LLM；文件长度只作为 input metadata，不能触发截断或分块。若具体 provider 的 context limit 不足，应显式报告并由人工调整，而不是静默改变科研读取方式；后续科研模块只接收结构化 Context。

## 4. Planned repository structure

M0 已创建 package、LLM infrastructure、serialization、CLI，M1/M2 已加入 contracts 和 Seed Context ingestion；下面其余目录是后续 milestone 的目标布局：

```text
research-agent/
├── AGENTS.md
├── ARCHITECTURE.md
├── PLAN.md
├── IMPLEMENTATION_STATUS.md
├── pyproject.toml
├── docs/                         # immutable method source of truth
├── data/                         # immutable MinerU Markdown inputs
├── src/research_agent/
│   ├── models/                   # schemas only; no orchestration
│   │   ├── seed.py               # SeedPaperContext, ExperimentContext
│   │   ├── seed_extraction.py    # extraction draft and audit, not Research State
│   │   ├── base.py               # StrictModel, shared validation primitives
│   │   ├── state.py              # ResearchState, Solution, Component, Experiment
│   │   ├── gap.py                # CandidateGap, standalone GapAgenda
│   │   ├── verification.py       # Gap/Solution results and comparison records
│   │   ├── inspiration.py        # Need, BridgeEntity, Inspiration results
│   │   ├── revision.py           # RevisionNeed, RevisionResult, patches
│   │   ├── memory.py             # ResearchMemory records
│   │   ├── controller.py         # ControllerDecision
│   │   ├── commit.py             # CommitCheck, CommitSnapshot, proposal
│   │   ├── runner.py              # execution-only replay/manifest/run result contracts
│   │   └── trajectory.py          # action/event records for audit
│   ├── modules/                  # exactly the documented scientific capabilities
│   │   ├── seed_context.py       # Markdown -> SeedPaperContext adapter
│   │   ├── gap_discovery.py      # Gap Discovery and Management
│   │   ├── solution_ideation.py  # Form / Select + Develop
│   │   ├── related_work_verification.py
│   │   ├── inspiration_retrieval.py
│   │   ├── targeted_revision.py
│   │   ├── experiment_design.py
│   │   ├── commit.py             # Commit Gate + snapshot
│   │   └── finalizer.py          # read-only final proposal
│   ├── controller/
│   │   ├── scientific_controller.py
│   │   └── rules.py              # deterministic triggers and action masks
│   ├── memory/
│   │   ├── store.py              # trajectory-local ResearchMemory store
│   │   └── __init__.py           # store export; deterministic queries stay in store.py
│   ├── runtime/
│   │   └── state_store.py         # latest validated ResearchState only
│   ├── trajectory/
│   │   ├── diff.py                # deterministic State diff
│   │   ├── recorder.py            # append-only JSONL audit log
│   │   └── context.py             # seed-local runtime container
│   ├── llm/
│   │   ├── client.py             # LLMClient protocol
│   │   ├── deepseek.py           # DeepSeekClient adapter
│   │   └── config.py             # env/config loading
│   ├── retrieval/
│   │   ├── interfaces.py         # RelatedPaperProvider, InspirationSearchProvider
│   │   ├── semantic_scholar_cache.py # persistent global Semantic Scholar JSON cache
│   │   ├── arxiv_cache.py             # persistent global arXiv response cache
│   │   └── metadata.py           # PaperMetadata/PaperPool
│   ├── prompts/                  # versioned prompt templates, separated from logic
│   ├── ingestion/
│   │   └── markdown.py            # deterministic parser and full-body cutoff
│   ├── runner/
│   │   ├── trajectory_runner.py
│   │   └── batch_evaluation.py
│   └── serialization/            # JSONL/JSON output, envelopes and replay journal
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── outputs/                      # generated trajectories/cache; never raw data
```

十个科研部分的落点是：Research State → `models/state.py`；Scientific Controller → `controller/scientific_controller.py`；Gap Discovery and Management → `modules/gap_discovery.py`；Solution Ideation → `modules/solution_ideation.py`；Related-work Verification → `modules/related_work_verification.py`；Inspiration Retrieval → `modules/inspiration_retrieval.py`；Targeted Revision → `modules/targeted_revision.py`；Experiment Design → `modules/experiment_design.py`；Research Memory → `memory/store.py` + `models/memory.py`；Commit + Finalizer → `modules/commit.py` + `modules/finalizer.py`。

M3 的 runtime foundation 落在 `runtime/state_store.py`、`memory/store.py` 和
`trajectory/{diff,recorder,context}.py`。`TrajectoryContext` 只聚合
`SeedPaperContext`、State store、Memory store 和 recorder，不选择 Action、不调用
LLM、不执行科研 transition。M3 的 JSON/JSONL persistence 使用现有 versioned
envelope；trajectory 只做 append/load 和顺序校验，不实现 event-sourcing、reverse
patch 或 replay engine。M11 新增独立 engineering replay journal，不能把 M3 的
scientific audit event 当成完整 event source。

## 5. Main interfaces and data models

以下是第一版结构化契约；M1 已用 Pydantic v2 实现。所有 canonical schemas 继承
`StrictModel`，使用 `ConfigDict(extra="forbid")`；枚举只使用方法文档已定义的值。

### 5.1 State models

```python
class CandidateGap:
    id: str
    statement: str
    confidence: Literal["low", "medium", "high"]
    confidence_reason: str
    priority: Literal["low", "medium", "high"]
    priority_reason: str
    evidence_refs: list[str]
    status: Literal["candidate", "active", "resolved", "rejected"]


class GapAgenda:
    gaps: list[CandidateGap]  # <= 5; not ResearchState or ResearchMemory


class CandidateSolution:
    id: str
    core_idea: str
    rationale: str
    status: Literal["candidate", "selected", "rejected"]


class Component:
    id: str
    description: str
    status: Literal["ready", "modified", "stale"]
    evidence_refs: list[str]


class CurrentSolution:
    id: str
    core_idea: str
    rationale: str
    components: list[Component]  # 2–5
    status: Literal["developed"]
    novelty_status: Literal["unchecked", "distinct", "partial_overlap", "core_overlap"]
    novelty_summary: str | None
    evidence_refs: list[str]


class Experiment:
    id: str
    objective: str
    setup: str
    baselines: list[str]
    metrics: list[str]
    what_it_validates: str
    validation_criterion: str
    status: Literal["ready", "stale"]


class ResearchState:
    active_gap: CandidateGap | None
    candidate_solutions: list[CandidateSolution]
    current_solution: CurrentSolution | None
    experiments: list[Experiment]
```

`Experiment` 的 canonical schema 是 `objective`、`setup`、`baselines`、`metrics`、`what_it_validates`、`validation_criterion` 和 `status`；不保留独立的 `design` 字段。

### 5.2 Verification models

```python
class GapVerificationResult:
    status: Literal["valid", "partially_resolved", "resolved"]
    summary: str
    covered_aspects: list[str]
    remaining_gap: str | None
    related_papers: list[str]
    evidence_level: Literal["abstract_only", "full_text"]


class ComparisonRecord:
    paper_ref: str
    problem_relation: Literal["same", "related", "different"]
    mechanism_relation: Literal["same", "partial", "different", "unclear"]
    shared_elements: list[str]
    novelty_relevant_overlaps: list[str]
    key_difference: str


class SolutionOverlap:
    component_id: str | None
    overlap_summary: str
    matched_paper_ref: str


class SolutionVerificationResult:
    novelty_status: Literal["distinct", "partial_overlap", "core_overlap"]
    summary: str
    shared_elements: list[str]
    novelty_relevant_overlaps: list[SolutionOverlap]
    distinctive_elements: list[str]
    key_differences: list[str]
    related_papers: list[str]
    evidence_level: Literal["abstract_only", "full_text"]


class VerificationFocus:
    mode: Literal["full_solution", "changed_units"]
    changed_units: list[str]
```

`VerificationFocus` 的 deterministic validation 要求 `full_solution` 不带 changed units，
而 `changed_units` 必须至少包含一个 unit；`unchecked` 只属于 `CurrentSolution`，不属于任何 Verification Result。

### 5.3 Inspiration and revision models

```python
class BridgeEntity:
    name: str
    type: Literal["scientific concept", "mechanism", "method", "phenomenon", "system principle"]
    bridge_reason: str


class BridgeGenerationDraft:
    inspiration_need: InspirationNeed
    proposed_entities: list[BridgeEntity]  # exactly 5
    selected_entities: list[BridgeEntity]  # 0–3, subset of proposed


class InspirationNeed:
    current_problem: str
    required_function: str
    current_mechanism: str
    rejected_mechanisms: list[str]
    relevant_context: str


class InspirationCandidate:
    bridge_entity: BridgeEntity
    paper_ref: str | None
    source_problem: str
    source_mechanism: str
    transferable_principle: str
    adaptation_hint: str
    grounding: Literal["arxiv", "none"]


class InspirationResult:
    status: Literal["found", "no_useful_inspiration"]
    inspirations: list[InspirationCandidate]


class RevisionNeed:
    problem_to_fix: str
    affected_units: list[str]
    constraints: list[str]


class RevisionResult:
    status: Literal["revised", "unable_to_revise"]
    revised_core_idea: str | None
    revised_rationale: str | None
    revised_components: list[ComponentRevisionPatch]
    changed_units: list[str]
    revision_summary: str
    failure_reason: str | None


class ComponentRevisionPatch:
    component_id: str
    description: str
```

`apply_revision(current_solution, revision_result)` 是确定性函数：只合并非空 patch，拒绝未知 Component id 或未声明 changed unit，并把实验 stale 传播交给确定性状态更新逻辑。

### 5.4 Memory, Controller and Commit models

```python
class ResearchMemory:
    paper_memory: PaperMemory
    verification_memory: list[VerificationRecord]
    inspiration_memory: list[InspirationRecord]
    revision_memory: list[RevisionRecord]
    failure_memory: list[FailureRecord]
    decision_log: list[DecisionRecord]


class ControllerDecision:
    action: Literal[
        "SOLUTION_IDEATION", "SEEK_INSPIRATION", "VERIFY_RELATED_WORK",
        "TARGETED_REVISION", "DESIGN_EXPERIMENT", "SWITCH_GAP", "COMMIT"
    ]
    reason: str


class CommitSnapshot:
    active_gap: CandidateGap
    current_solution: CurrentSolution
    experiments: list[Experiment]
    latest_gap_verification: GapVerificationResult
    latest_solution_verification: SolutionVerificationResult
    relevant_evidence_refs: list[str]
```

`ResearchMemory` 的具体 records 对应实现规范第 9 节：PaperRecord/RelatedPaperPool、VerificationRecord、InspirationRecord、RevisionRecord、FailureRecord、DecisionRecord。第一版只提供结构化过滤接口，例如 `get_related_papers`、`get_latest_gap_verification(active_gap.id)`、`get_relevant_failures`、`get_relevant_inspirations` 和 `get_revision_history`；不引入 Memory Agent。

`ResearchMemory` 的顶层字段严格只有 `paper_memory`、`verification_memory`、
`inspiration_memory`、`revision_memory`、`failure_memory` 和 `decision_log`；`GapAgenda`
不写入其中。

### 5.5 Seed extraction contracts

M2 的 `SeedContextExtractionDraft` 是 LLM-only 的中间结构，不是新的科研 State。其
科学归纳字段使用 `value + source_section_ids[]`，列表项也携带 source ids；确定性
代码校验 source id 后才物化为 M1 的 `SeedPaperContext`。`SourceAnchor` 和
`SeedExtractionAudit` 单独保存 field path、section、line span、短 excerpt、warnings
和 LLM call metadata，最终 `SeedPaperContext` 不携带 audit 字段。

`title`、`abstract` 和 `paper_id` 由 Markdown parser/file stem 决定，LLM 返回的同名
字段不会覆盖 deterministic 值。输入保留完整 main-paper body；References/Bibliography
和 Appendix/Appendices/Supplementary Material 中最早出现的尾部边界及其后内容不进入
reading context。该过程不进行 relevant-section selection、字符数截断、chunk extraction
或 synthesis；因此每篇论文只有一次 SeedContext LLM call。每个非空归纳字段的 source
section id 必须来自实际发送的 main-paper body，尾部 section id 会被拒绝。

### 5.6 LLM and retrieval infrastructure

```python
class LLMClient(Protocol):
    def complete_structured(
        self,
        request: LLMRequest,
        output_schema: type[T],
    ) -> StructuredLLMResponse[T]: ...


class DeepSeekClient(LLMClient):
    """DeepSeek adapter; scientific modules depend only on LLMClient."""


class RelatedPaperProvider(Protocol):
    def get_related_paper_pool(
        self, seed_context: SeedPaperContext, *, limit: int = 20
    ) -> RelatedPaperPool: ...


class InspirationSearchProvider(Protocol):
    def search(self, query: str, limit: int = 20) -> list[PaperRecord]: ...
```

`DeepSeekClient` 的配置来自 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 以及可选 timeout/retry 配置；源码不放默认 secret。M5 的 `SemanticScholarProvider` 从本地 `config/semantic_scholar.json` 读取 `api_key`、`base_url`、`auth_mode`、timeout/retry 和 pacing 配置，CLI 可通过 `--semantic-scholar-config` 指定其他 JSON 文件。若 JSON 中 `api_key` 为空、空白或缺失，则回退读取 `SEMANTIC_SCHOLAR_API_KEY`；非空 JSON 值优先于环境变量，两者都没有时在真实请求前显式报配置错误。`auth_mode=x-api-key` 使用官方 `x-api-key` header，`auth_mode=bearer` 使用兼容中转站示例的 `Authorization: Bearer <token>` header。真实配置文件被 `.gitignore` 排除，仓库只保留 `api_key` 为空的 `.example` 模板；global cache key 包含 base URL，避免官方源和中转源混用缓存。pacing 的 clock/sleep 可注入以保持离线测试无真实等待。M6 已实现 `ArxivInspirationSearchProvider`：配置来自 `ARXIV_BASE_URL`、`ARXIV_TIMEOUT_SECONDS`、`ARXIV_MAX_RETRIES` 和 `ARXIV_MIN_REQUEST_INTERVAL_SECONDS`，通过官方 Atom search interface 返回规范化 `PaperRecord`；真实网络测试都必须显式 opt-in。

科研模块只依赖 `LLMClient` 和 provider interfaces，通过依赖注入取得实现。真实客户端、真实检索和 fake 客户端都可以替换，不改变方法代码。

## 6. Trajectory、输出和实验支持

每次高层 Action 至少记录：step、action、输入 State 摘要、结构化 module result、State diff、Memory writes、search/step budget、LLM/provider 调用 metadata、错误和时间戳。记录应支持 JSONL，输出到 `outputs/`，原始 Markdown 仍只读。

M3 当前只提供 `research_state.json`、`research_memory.json` 和
`trajectory.jsonl` 的 deterministic persistence。State 保存最新状态，Memory 保存
方法规定的 trajectory-local 历史，Trajectory 是 append-only audit log；State diff
仅用于 debugging/trajectory analysis，不用于反向重建 State 或 Memory。

M2 的 extraction output 只写到独立的 `outputs/<paper_id>/seed_context.json` 和
`seed_extraction_audit.json`。显式 `extract-seed-context` CLI 才会构造
`DeepSeekClient` 并发起真实请求；默认 CLI help、offline smoke、parser-only smoke
和全部测试均不触网。

Trajectory 记录属于工程审计数据，不是新的科研 State、Action 或 Memory agent。Batch runner 只负责加载多篇 Seed Paper、隔离每篇 trajectory、复用配置并汇总结果，不能跨 paper 共享 Research Memory。

第一版测试默认完全离线：fake `LLMClient` 返回 fixture，fake providers 返回 fixture paper pool，测试 deterministic rule/merge/gate/state diff。M0 的 `DeepSeekClient` 只实现可注入的 infrastructure adapter，不在测试中发起网络；真实 DeepSeek 和论文 provider 只能由显式标记的 smoke/integration test 运行。M12 的 Real Single-Paper Integration Pilot 必须先于 M13 batch execution。

### 5.7 M5 Related-work Verification implementation

M5 只实现方法规定的 `VERIFY_GAP` 和 `VERIFY_SOLUTION` 两个 task。`RelatedWorkVerificationModule` 依赖 provider interface 与 `LLMClient`，不导入 HTTP 实现；每个 task 进行一次 structured LLM call，分别动态注入 `GapVerificationResult` 与 `SolutionVerificationDraft` 的 canonical schema。论文池只在 trajectory-local `ResearchMemory.paper_memory.related_paper_pool` 为空时建立，最大 20 篇；后续 Gap、Solution 和 focused re-verification 复用同一池，cache hit 不增加 scientific search count。

`SemanticScholarProvider` 先以 SeedPaperContext.title 通过 Academic Graph 做 deterministic normalized-title resolution，再调用官方 Recommendations API。local `paper_id` 保持 trajectory identity，resolved provider paper id 只进入 SearchRecord/audit，不进入 ResearchState。PaperRecord 只保存标题、摘要和既有 metadata；VerificationResult 只保存稳定 paper refs。除 trajectory-local `ResearchMemory` 外，M5 还使用独立的持久化 `SemanticScholarCache`：默认目录为 `cache/semantic_scholar/`，按 operation + request parameters 生成内容哈希，分别保存成功的 Academic Graph search JSON 和 Recommendations JSON。缓存无自动 TTL，跨 provider instance、trajectory 和 CLI 运行复用；缓存文件不保存 API key，真实 cache 目录被 `.gitignore` 排除。provider HTTP request count 与一次 related-pool acquisition 的 scientific search count 分开记录：trajectory/global 完整 cache hit 计零；cache miss 在 provider 调用前恰好计一次 scientific search，即使 provider 失败；global partial hit 仍会记录实际剩余请求。HTTP retries 和 graph/recommendations 请求只增加 provider request count。timeout/connection、429、5xx 只在有限次数内重试，429 优先遵循 `Retry-After`，其余使用 bounded backoff；400/401/403/404 与 malformed response 不重试。所有 pacing、retry、HTTP status、cache scope 和失败信息显式进入 audit/error metadata，不 fallback 到其他检索源。

Gap Verification 不改写 Gap statement/confidence/priority，只写 VerificationRecord；Solution Verification 使用一次 structured LLM call 返回内部 `SolutionVerificationDraft`：其中包含最多五个不重复、且真实存在于 PaperMemory 的 `ComparisonRecord`，以及嵌套的 canonical `SolutionVerificationResult`。比较记录只持久化在 `solution_verified` trajectory payload（可选 audit），不进入 ResearchState 或 `VerificationRecord.result`；最终 `related_papers` 必须与比较记录引用严格一致，novelty evidence 也必须与最终 novelty status 一致。结果校验后再 deterministic 更新 novelty status，`core_overlap` 清空 CurrentSolution、拒绝对应 CandidateSolution、清空 experiments，并写入一个 `rejected_solution` FailureRecord；distinct/partial/shared elements 不写该失败记录。focused distinct 只将 focus 中实际为 modified 的 Component 改为 ready。结果中的相关论文最多五篇，unknown/duplicate paper refs、partial 无 overlap、core 无 core evidence 均拒绝。v1 证据粒度固定为 `abstract_only`。

M5 的两个 verification 入口都在 provider/LLM 前做 target precondition。若有
`TrajectoryContext`，State 从 context 读取；否则若调用方提供 `ResearchStateStore`，
State 从该 store 读取。`VERIFY_GAP` 要求 State 中有同 ID active gap；
`VERIFY_SOLUTION` 要求 State 中有 Current Solution，显式 solution 的 ID 还必须匹配。
失败只抛出 `PreconditionError`，不改变 provider、LLM、Memory、State 或 trajectory。

### 5.8 M6 Inspiration Retrieval implementation

`InspirationRetrievalModule` 只依赖 `LLMClient` 与
`InspirationSearchProvider`。在 active gap 和 Current Solution 存在后，Call 1
把 active gap、solution summary、latest relevant verification、仅
`rejected_mechanism` FailureRecord、previous bridge entity names 和少量 related-work
context 组装为紧凑输入，返回内部 `BridgeGenerationDraft`。该 draft 要求恰好五个
去重实体，selected 是其中 0–3 个；实体类型仍只有方法文档规定的五种。Call 1
不会写 State。

若 selected 非空，模块按选中顺序构造一个 `E1 OR E2 OR E3` 查询，并且一次
SEEK_INSPIRATION 最多调用 provider 一次、limit=20。`ArxivInspirationSearchProvider`
使用官方 arXiv Atom programmatic search endpoint，稳定化 `arxiv:<canonical-id>`、
去掉版本后缀、去重并保留顺序；只读取 title/abstract/metadata，不下载 PDF 或抓取
全文。provider 负责显式 timeout、pacing、bounded retry 和 malformed XML error；它
不在内部切换其他 provider。

同一 trajectory 的 `InspirationSearchRecord.query` 使用 normalized exact match 做
缓存复用；只有所有 paper refs 都存在于 `PaperMemory.papers` 时才可命中，否则显式
报 cache corruption。cache hit 不增加 scientific search count；真实 search attempt
（包括 provider 内部 retry）只增加一次 scientific search count。provider failure 进入
第二次 LLM 的 LLM-only fallback，但不写成功 SearchRecord；成功空结果可以写空
SearchRecord。

Call 2 只接收 InspirationNeed、selected entities、solution summary、rejected
mechanisms 和最多 20 篇 title/abstract。grounded 模式的 candidate 必须为
`grounding="arxiv"`、引用当前已检索且已写入 PaperMemory 的 paper、使用 selected
entity，且一篇 paper 最多贡献一个 candidate。fallback 模式所有 candidate 都必须是
`grounding="none"`、`paper_ref=null`，不得伪造外部归因。两种路径最多返回三个
candidate；没有 selected 时只做 Call 1 并直接返回 `no_useful_inspiration`，没有第
三次 fallback call。

M6 只确定性写入 PaperMemory、InspirationSearchRecord 和 InspirationRecord（ID 为
`I1`、`I2`…），并在存在 `TrajectoryContext` 时记录 `inspiration_retrieved` /
`SEEK_INSPIRATION` 事件。State before/after 必须相等；found 只暴露后续
`TARGETED_REVISION` 语义，不执行 Revision。M6 的离线测试使用 generic fixtures，
不执行真实 arXiv/DeepSeek/Semantic Scholar 请求。

### 5.9 M7 Targeted Revision implementation

`TargetedRevisionModule` 要求 Research State 中存在 active gap 和 Current
Solution，并在任何 LLM 调用前校验显式输入的 ID、未完成的 modified/stale
Component、partial-overlap 或 useful Inspiration evidence，以及 trajectory-local
Memory 的 seed identity。失败的前置条件不会调用 LLM，也不会写 State、Memory 或
trajectory。M7 不重新调用 M5 的 verification module。

Revision Need 由 deterministic code 从最新 `VERIFY_SOLUTION` 的 partial-overlap
结果、可用 InspirationRecord 和允许的 failure memory 构造；它包含待解决问题、去重
的 affected units 和 minimum-change/consistency 等约束。发送给 LLM 的输入只包括
active gap、完整 Current Solution、Revision Need、最新相关 verification、带运行时
引用的 Inspiration candidates 和相关 failure memory，不包含完整 trajectory、论文池
或无关 Memory。

一次 structured LLM call 返回内部 `TargetedRevisionDraft`：其中的
`selected_inspiration_ref` 使用 `<InspirationRecord.id>:<candidate_index>` 形式，
成功 revision 最多且必须选择一个可用 Inspiration；没有 Inspiration 时必须为
`null`。LLM 不能控制 Component/Experiment status、novelty、evidence 或实验字段，
也不能新增 Component；prompt schema 和 `extra="forbid"` 共同拒绝这些越权输出。

确定性 merge 只允许 `core_idea`、`rationale` 和已有 Component ID，且
`changed_units` 必须与实际 patch 完全一致；未知、重复、no-op 或仅 rationale 的
patch 被拒绝。未修改字段和 evidence refs 保留。Component patch 标记对应 Component
为 `modified`；core change 使未修改 Component 变为 `stale`；novelty 重置为
`unchecked`，summary 清空，solution 保持 `developed`。core change 会使所有
Experiment stale；Component change 只使 `what_it_validates` 明确包含该 Component
ID 的 Experiment stale；已有 stale 不会被恢复。core 或至少两个 Component 的修改
使用 `full_solution` focus，单个 Component 修改使用 `changed_units` focus。

成功 Revision 写入 `RevisionRecord`、`solution_revised`/
`TARGETED_REVISION` trajectory event 和 `revision_memory`，下一动作固定为
`VERIFY_RELATED_WORK`，不在 M7 内执行 re-verification。`unable_to_revise` 保持
State 不变，写入 `RevisionRecord` 与 `failed_revision` FailureRecord、相应事件和
failure memory，下一动作为 `None`。写入在所有 schema/语义校验完成后进行，保证
成功和失败路径不会产生半成品状态。M7 离线测试全部使用 generic FakeLLM 和
trajectory-local memory，不调用 DeepSeek、论文 provider、M5 verification、M8 或
Controller。

### 5.10 M8 Experiment Design implementation

`ExperimentDesignModule` 只依赖 `LLMClient`、SeedPaperContext、ResearchStateStore/
TrajectoryContext 和 trajectory-local Memory。它在 LLM 前检查 active gap、Current
Solution、`novelty_status="distinct"`、全部 Component `ready`、最新
`VERIFY_GAP=valid`、最新 `VERIFY_SOLUTION=distinct`、实验数量和 trajectory step。
显式对象与 State ID 不一致、缺少 verification、modified/stale Component、单个或
超过四个既有 Experiment、非法 step 都在 LLM 前失败且不改变 State/Memory/trajectory。

输入只包括 active gap、完整 Current Solution、SeedPaperContext.experiment_context、
两条最新 verification summary、deterministically derived changed_units、existing
Experiments 和 stale redesign targets；不包含完整 trajectory、Research Memory、Top-20
paper pool 或 arXiv results。M8 不调用任何 retrieval provider，scientific search count
不变化。

一次 structured LLM call 返回 `ExperimentDesignDraft`，schema 由
`model_json_schema()` 动态注入。INITIAL Draft 包含 2–4 个 ordered `KeyClaimDraft`；
REDESIGN Draft 针对 stale targets 可包含 1–4 个 Claim。确定性校验 Claim target、
initial Claim/Experiment coverage、redesign stale-only coverage、core/mechanism coverage、
`what_it_validates` anchor、Seed Method baseline、必填质量字段、canonical duplicate、
stale-only ID contract 和 final count；LLM 不能写 canonical Experiment 的
id/status/results/design/claim_ids。Ready Experiments 原样保留，不根据相同的
`core_solution`/Component anchor 重新解释为本轮新的 Claim coverage。

成功时只更新 `ResearchState.experiments`，不改变 gap、solution、novelty 或 Component；
Context 路径记录 `experiments_designed` / `DESIGN_EXPERIMENT`，payload 保存
`design_mode`、claims、design summary、experiment IDs、redesigned IDs 和 claim coverage。
REDESIGN 的 claim coverage 只映射新 Claims 到 redesigned stale IDs，`memory_writes=[]`。
全部 ready 时返回 no-op result，不产生 event。M8 不实现 Commit、Finalizer 或 runner。

### 5.11 M9 Scientific Controller implementation

`ScientificController` 是纯 routing boundary：它读取 `ResearchState`、trajectory-local
`ResearchMemory`、紧凑 trajectory summary、外部传入的 `commit_allowed`/unmet conditions
和 runtime budget，只选择已有 Scientific Action，不 import/执行任何科研 module，不生成
scientific content，不写 State，也不写 pending `DecisionRecord`。

固定 Action Space 为：

```text
SOLUTION_IDEATION, SEEK_INSPIRATION, VERIFY_RELATED_WORK,
TARGETED_REVISION, DESIGN_EXPERIMENT, SWITCH_GAP, COMMIT
```

`REFINE_GAP` 是 Gap 模块内部动作，`VERIFY_COMPONENT` 不存在。裸的
`partially_resolved` Gap Verification 如果没有其后的 `gap_refined` trajectory event，
会显式抛出 `ControllerRoutingError`；RefineGap 之后唯一的下一验证动作仍是
`VERIFY_RELATED_WORK`。Revision 后也统一使用 `VERIFY_RELATED_WORK`，具体
`VerificationFocus` 由已有 verification/revision contract 决定，Controller 不选择 target。

Routing 顺序是 deterministic-first：先验证 State/Memory consistency，再应用 mandatory
route、action mask 和 budget。新 Gap、unchecked Solution、refined Gap 都优先进入
`VERIFY_RELATED_WORK`；valid Gap 且无 Current Solution 进入 `SOLUTION_IDEATION`；resolved
Gap 进入 `SWITCH_GAP`；Inspiration found 后进入 `TARGETED_REVISION`。Partial overlap
可在 revision/inspiration 中选择，但 partial/ distinct verification 不能与任何
`modified`/`stale` Component 共存；否则先报告 routing contract error。Distinct solution
下 empty/stale experiments 可在 design/inspiration 中选择，2–4 个 ready experiments 时
不再提供 DESIGN_EXPERIMENT。`SEEK_INSPIRATION` 是否进入这些已有候选只由 State、最新
verification 和预算/action mask 决定，不由 solution 文本分类器决定。
Commit eligibility 只消费外部 boolean，不实现 Commit Gate。

当 action mask 只有一个动作时不调用 LLM；只有多个 legal actions 时进行一次 structured
Controller call。Prompt 动态注入 `ControllerDecision.model_json_schema()`；非法 action
不替换、不重试而显式失败。ControllerContext 只包含 compact state summaries、relevant
`rejected_solution`/`rejected_mechanism`/`failed_revision` failures、last action/result、
commit input、budget 和 action mask，不包含完整 paper、Memory、trajectory 或 paper-pool
abstracts。

`ControllerBudget`/`BudgetTracker` 不进入 Research State。Controller decision 不消耗 step；
未来执行模块后由 runner 以模块报告的 `scientific_search_count` 调用
`record_execution()`，每次 action +1 step，cache hit/search count 0。Solution Ideation、
Targeted Revision、Switch Gap 启动前必须保留一个后续 verification step；主动
`SEEK_INSPIRATION` 还必须保留最坏路径的 3 个 step（Inspiration → Revision → Verification）。
search budget 为 0 时屏蔽 Inspiration，但已有 local paper pool 时仍可验证。预算耗尽或无合法动作返回
execution-only `ControllerRoutingResult(status="stopped", unresolved_issues=[...])`。

Component freshness 与整体 novelty verdict 是正交维度。`ready` 表示当前 Component
版本没有未验证修改，`modified` 表示直接修改后尚未验证，`stale` 表示旧验证因核心或
依赖变化失效；`unchecked`、`distinct`、`partial_overlap`、`core_overlap` 仍只表示
Current Solution 的整体 verification verdict。Successful `distinct` 或
`partial_overlap` full-solution verification 会把所有已覆盖的 modified/stale
Components 确定性收敛为 `ready`；changed-units verification 只有在 focus 恰好覆盖
全部 pending modified Components 且不存在 stale Component 时才允许局部收敛。否则必须
使用 full-solution focus，不得留下 silent pending component。Targeted Revision 的
`lightweight patch + deterministic merge` precondition unchanged：任何 modified/stale
Component 都必须先完成 Revision → Verification，才可再次 Revision。

`record_decision_result()` 只在实际 action 完成后写一个 `DecisionRecord` 和真实 result
summary；M9 不实现 runner、batch runner 或真实 LLM/network trajectory。

### 5.12 M10 Commit Gate, Snapshot and Finalizer implementation

`modules/commit.py` 提供无副作用的 `check_commit_gate()`。Gate 不调用 LLM、retrieval 或
网络，只读取 `ResearchState` 与 trajectory-local `ResearchMemory`，按固定顺序检查：
active Gap 与最新 matching `VERIFY_GAP=valid`、完整 Current Solution、最新 matching
`VERIFY_SOLUTION=distinct`、Component readiness，以及 2–4 个全部 ready 的 Experiments。
`commit_allowed` 严格等价于 deterministic `unmet_conditions` 为空。State novelty 与最新
Solution Verification 冲突时抛出 `PreconditionError`，不把 runtime inconsistency 降级为
普通 Gate failure。

`create_commit_snapshot()` 内部会重新执行 Gate，不能信任调用方传入的旧 boolean。Snapshot
只包含 active gap、current solution、experiments、两个实际 latest verification result 和
first-seen 去重的 relevant evidence refs；不包含 candidate solutions、完整 Memory、trajectory
或 budget。所有嵌套 Pydantic objects 都通过 structural copy 隔离。`CommitModule.commit()`
在 trajectory step preflight 后创建 Snapshot，成功记录 `research_committed`/`COMMIT` 事件，
State diff 与 Memory writes 均为空；Gate failure 或非法 step 不记录事件。

`FinalizerModule` 只接受已创建的 CommitSnapshot 和 SeedPaperContext compact summary，
只做一次 structured LLM call。`FinalizerDraft` 是 narrative-only，只包含
`novelty_positioning` 和 2–3 条 expected contributions，不包含 committed Gap、Solution、
Rationale、Components 或 Experiments。Python exact materialize `FinalResearchProposal`：
Gap、Solution、Rationale、Components 与 Experiments 全部来自 CommitSnapshot 的 deep copy，
Finalizer 不能改写它们。Prompt 明确 shared elements、`distinct !=` zero overlap、
abstract-only evidence boundary、experiment plans 未执行和 expected/potential contributions；
Finalizer 不写 State、Memory、Verification 或 Snapshot。

提供 `commit_snapshot.json` 与 `final_research_proposal.json` 的 schema/version envelope
持久化 helpers。M10 不实现 Runner、replay、real pilot 或 batch runner。

### 5.13 M11 Offline End-to-End Trajectory implementation

`runner/trajectory_runner.py` 是 execution-only orchestration boundary。它接收显式
`TrajectoryModules` bundle（Gap、Solution、Verification、Inspiration、Revision、Experiment、
Controller、Commit、Finalizer），不构造 `DeepSeekClient`、Semantic Scholar 或 arXiv provider，
不实现 planner、reflection 或任何额外 Scientific Action。Runner 只在开始时执行一次
Gap Discovery bootstrap；bootstrap 不写 `DecisionRecord`，也不增加 `ControllerBudget.steps_used`。

固定 dispatch 为：`SOLUTION_IDEATION` → `ideate`，`SEEK_INSPIRATION` → `retrieve`，
`VERIFY_RELATED_WORK` → deterministic `VERIFY_GAP`/`VERIFY_SOLUTION` target，
`TARGETED_REVISION` → `revise`，`DESIGN_EXPERIMENT` → `design`，`SWITCH_GAP` → `switch_gap`，
`COMMIT` → `commit` 后仅调用一次 Finalizer。Controller 选择后，Runner 不覆盖其合法决定；
唯一 action 仍由 Controller deterministic route，多个 action 才使用一次 injected Controller LLM。

Trajectory event step 仅表示 audit ordering、VerificationRecord step、DecisionRecord step
和 Refine ordering；scientific budget step 只在实际 high-level Action attempt 后加一。每个
Action 的 search accounting 使用 related/inspiration module cumulative counter 的 delta，
cache hit 为 0，provider HTTP requests 不进入 scientific search budget。失败的已开始 Action
仍记录 step 和已发生的 search delta；Runner invariant 在 dispatch 前失败时不伪装成 Action。
`controller_decided` 和 `budget_updated` 都是 `action=None` 的 audit events；RefineGap 是
internal transition，不写 DecisionRecord、不增加 step，且 `gap_refined.action=None`。

`serialization/trajectory.py` 提供独立 `trajectory_replay.jsonl`。每条
`TrajectoryReplayRecord` 保存 sequence、event step、kind、action、完整 validated `state_after`、
六个允许 Memory 顶层 category 的 canonical after-value patch 和 `ControllerBudget` snapshot。
它不是 Research State/Memory/Action，也不写入 Memory。`reconstruct_context()` 从 Seed、原
scientific `trajectory.jsonl` 和 replay journal 创建 fresh stores，顺序 replace 完整 State、
apply/validate Memory category，再返回 `TrajectoryContext`；不读取 final state/memory artifact，
不调用任何 LLM/provider/controller，也不尝试由 `state_diff` 或 `memory_writes` 反向推导。

Runner 输出是 execution artifact：`seed_context.json`、`research_state.json`、
`research_memory.json`、`trajectory.jsonl`、`trajectory_replay.jsonl`、`run_manifest.json`；
只有 Commit 成功才写 `commit_snapshot.json` 与 `final_research_proposal.json`，stopped/failed
只写 unresolved result。M11 generic offline fixtures 覆盖 valid commit、verification-only
revision、Inspiration→Revision、resolved switch、partial refine、budget exhaustion、action
failure、persistence/replay equality 和 malformed replay validation。

## 7. M4 implementation details

M4 is now implemented in the following small, provider-neutral pieces:

- `models/gap.py` contains the LLM-only `GapCandidateDraft`,
  `GapDiscoveryDraft` and `GapRefinementDraft`, plus the engineering
  `GapDiscoveryResult`.  `models/solution.py` contains the FormSolution and
  Select+Develop drafts and the module result.  These drafts use
  `extra="forbid"`; IDs and lifecycle fields are intentionally absent from LLM
  output schemas.
- `modules/gap_discovery.py` performs one SeedPaper-grounded structured call,
  validates the stable `seed:*` evidence vocabulary, assigns `G1...`, and
  performs deterministic active selection.  Selection takes the first
  candidate in the model-provided worth-exploring order, including when its
  confidence is `low`; it does not construct a score.
- `modules/solution_ideation.py` enforces the latest
  `ResearchMemoryStore.get_latest_gap_verification(active_gap.id)` result to be
  `valid` before making any LLM call.  It performs FormSolution exactly once
  when needed, then one Select+Develop call, assigns `S<number>` and `C<number>`
  deterministically, and initializes the developed solution with `ready`
  components and `novelty_status="unchecked"`.
- `prompts/gap_discovery.md`, `gap_refine.md`, `solution_form.md` and
  `solution_select_develop.md` are separate task prompts.  Each injects the
  exact `model_json_schema()` of its Pydantic draft at runtime.

Gap Discovery is method-improvement oriented.  Its initial evidence vocabulary
contains only non-empty `seed:research_objective`, `seed:core_method`, indexed
`seed:main_components`, indexed `seed:main_findings`, indexed
`seed:limitations`, and indexed `seed:method_constraints`.  `ExperimentContext`
remains in SeedPaperContext for later Experiment Design, but
experiment/dataset/metric/baseline/setting fields are not provided as initial Gap
evidence.  Unknown, empty-field, and legacy `seed:experiment_context.*`
references fail before State materialization.

The Gap prompt requires a substantive method, mechanism, capability,
architecture, representation, optimization, or inference change to resolve the
problem.  Missing experiments, datasets, baselines, metrics, ablations,
training scale, or evaluation coverage are not Candidate Gaps.  An observed
experimental weakness may support a method gap, but missing evaluation alone
cannot.  RefineGap preserves this same boundary and cannot turn a method gap
into an evaluation request.

Gap Discovery also applies four paper-agnostic reasoning gates in the same
structured call: method-level actionability, evidence sufficiency, whole-method
consistency, and calibrated confidence.  A Research Gap may be an explicit
limitation, author-stated future method direction, observed weakness, trade-off,
constraint, or grounded method opportunity; direct failure evidence is not
required.  A design fact is not automatically a failure, and unsupported
causal wording remains prohibited.  Before returning a gap, the model must
inspect all supplied method evidence, not only the evidence it plans to cite,
and check for compensation, mitigation, bypass, addressing mechanisms,
contradiction, and severity weakening.  A clear contradiction suppresses the
gap; partial mitigation requires residual, non-absolute wording.  High
confidence is reserved for explicit limitations/unresolved problems or direct
observed failures; medium can include strong author-stated directions or
trade-offs; low includes grounded opportunities without direct failure
evidence.  Priority remains independent from confidence.  A successful initial
draft must contain 1–5 candidates.

The same evidence-consistency and causal-calibration boundary is present in
`gap_refine.md`.  M12.1 adds a design-rationale consistency gate: a deliberate
choice is not automatically a gap; a proposed change must inspect the stated
rationale/rejected alternative and preserve or explicitly handle the relevant
constraint.  `method_constraints` is a Seed extraction/audit field and a legal
Gap evidence source, but is not copied into CandidateGap, State, or Memory.
No `consistency_check` field, extra agent, critic, reflection loop, or second
verification call was added; the existing one-call structured contract and
deterministic materialization remain unchanged.

`GapAgenda` is recorded as a trajectory payload and is not copied into
Research State or Research Memory.  Discovery, RefineGap and SwitchGap update
only the documented State fields and append `gap_discovered`, `active_gap_selected`,
`gap_refined`, or `gap_switched` audit events.  RefineGap accepts only
`partially_resolved`, preserves the original ID/confidence/priority/evidence,
and is bounded by `has_refined_gap()`.  SwitchGap preserves agenda order,
marks the old gap `resolved`/`rejected`, and clears candidate solutions,
Current Solution and experiments when changing gaps.

Solution Ideation reads only `rejected_solution` and `rejected_mechanism`
Failure Memory.  Remaining candidate solutions are reused without FormSolution;
only when every existing candidate is rejected are old candidates replaced by a
new sequential set.  A developed solution always returns the deterministic next
required action `VERIFY_RELATED_WORK`; M4 does not implement Verification,
retrieval, Inspiration, Revision, Experiments, Controller, Commit or Finalizer.

The explicit `discover-gaps` CLI consumes the existing SeedPaperContext output,
records the result/state/memory/trajectory and a safe call-metadata audit, and
does not rerun Markdown extraction.  The M4 real smoke is Gap Discovery only.

Real-paper smoke outputs are evaluation cases.  Any failure observed there may
motivate a prompt change only after abstraction into a paper-agnostic reasoning
principle; prompts and scientific modules must not contain paper names,
paper-specific mechanisms, or special-case branches.

## 8. Confirmed contract decisions

以下决定已由人工审核确认，后续实现必须直接遵守：

1. `method_overview.md` 的 “Gap Discovery and Verification” 是总体能力描述；代码边界拆为 `Gap Discovery and Management` 与 `Related-work Verification`，后者负责 `VerifyGap` 和 `VerifySolution`，不建立独立 Gap Verification module。
2. Component 只允许 `ready`、`modified`、`stale`；新生成 Current Solution 的 Components 使用 `ready`，但 Current Solution 的 novelty 仍为 `unchecked`。
3. canonical Experiment 不含 `design`，只含 `objective`、`setup`、`baselines[]`、`metrics[]`、`what_it_validates`、`validation_criterion`、`status`；status 只有 `ready`/`stale`。
4. CurrentSolution.status 只使用 `developed`。发生 `core_overlap` 时对应 Candidate 标记 `rejected`，Current Solution 设为 `null`，历史写入 Research Memory。
5. RefineGap 保持同一 gap id，只更新 Active Gap 内容；通过 trajectory 的 `gap_refined` 事件和 `has_refined_gap(gap_id)` query 保证最多一次，不增加 State 字段或 Action。
6. Current Solution 为空且所有候选都 rejected 时，合法 Action 是 `SOLUTION_IDEATION`；重新生成三个候选并替换旧候选，旧 rejected 内容保留在 Memory；不转入 `SEEK_INSPIRATION`。
7. Commit Gate 通过 `get_latest_gap_verification(active_gap.id)` 从 Research Memory 查询 `target_ref` 匹配且 `created_at_step` 最大的记录，不向 Research State 增加 latest verification 字段；CommitSnapshot 冻结该实际结果。

当前没有待人工裁决的 method-level open issue。后续若发现新的方法歧义，仍须按 `AGENTS.md` 记录并暂停相关决策。

## 9. M12 Real Single-Paper Integration Pilot

`runner/pilot.py` 是显式 opt-in 的 execution-only wiring。它先用同一个
`RecordingLLMClient` 包装一个 `DeepSeekClient`，完成完整 MinerU main-paper body 到
`SeedPaperContext` 的一次 structured extraction；随后把同一个 recorder 注入 Gap、Solution、
Verification、Inspiration、Revision、Experiment、Controller 和 Finalizer。Pilot 不选择
Scientific Action、不指定 gap/solution/inspiration、不加载历史 State/Memory/trajectory，
只把现有 `TrajectoryRunner`、真实 `SemanticScholarProvider`、真实
`ArxivInspirationSearchProvider` 和固定 12-step/2-search budget 连接起来。arXiv 只有在
Controller 实际选择 `SEEK_INSPIRATION` 时才会被调用。

`models/pilot.py` 的 `PilotManifest`、`ProviderAudit` 和 `LLMCallAudit` 是 execution-only
审计，不进入 Research State、Research Memory 或任何 method-level schema。`llm_calls.jsonl`
只记录 task、schema、provider/model、request id、input chars、token usage 和安全错误信息；
不会记录 API key、认证头、完整 prompt 或环境变量。token metadata 不完整时记录 `null`，
不伪造 0。Semantic Scholar 审计保留 HTTP request count、global cache scope 和命中 operation；
arXiv 未被选择时 request count 为 0。

M12 的 M11 trigger repair 规定 Inspiration 是 single-use direct handoff：只有 trajectory
最近一个实际 action 正好是成功的 `SEEK_INSPIRATION`，且唯一 Memory InspirationRecord
包含非空结果时，Targeted Revision 才能读取该 record。历史 Inspiration 不会跨越后续
Revision/Verification 被重新使用。

第一次 ByteFlow real pilot 保留在 `outputs/m12_real_byteflow/`，因 Gap Discovery
structured output 非 JSON 而失败。按人工要求移除 pilot 对 DeepSeek 请求的本地
`max_tokens` 转发限制后，第二次运行保存在 `outputs/m12_real_byteflow_unbounded/`，
并完成 SeedContext、Gap、Verification、Solution、Experiment、Commit 和 Finalizer。
第二次审计显示 10 次 DeepSeek call 全部成功，累计 prompt/completion/total tokens 为
`49950/34733/84683`，所有实际转发请求的 `max_tokens` 均为 `null`。Semantic Scholar
的 `paper_search` 与 `recommendations` 从 global cache 命中，HTTP requests=0、related
pool=20；arXiv 未被选择，HTTP requests=0。最终 `committed`，Commit Gate 通过，Finalizer
生成 proposal；trajectory events=22、replay records=6，replay 对 State/Memory/trajectory
均精确一致。M12 通过，M13 保持未开始。

## 10. M12.1 Design-Rationale Preservation Hardening

M12.1 preserves author-explicit design rationale in the existing Seed extraction
call by adding only `SeedPaperContext.method_constraints: list[str]` and the
corresponding internal `EvidenceItem` draft field. The full main-paper body,
one-call/no-chunking ingestion contract, deterministic tail removal, source
anchors, and deterministic title/abstract/paper identity remain unchanged.
`method_constraints` contains only author-supported constraints, deliberate
choices and reasons, acknowledged trade-offs, explicitly rejected alternatives
and rejection reasons, and conditions that later method changes must preserve or
explicitly overcome. Architecture facts alone do not create constraints,
limitations, or gaps.

`modules/gap_discovery.py` exposes non-empty constraints as
`seed:method_constraints[i]` and still excludes all `experiment_context` refs.
The Gap Discovery and RefineGap prompts now require the existing one-call
evidence-sufficiency and whole-method checks to include a design-rationale
consistency gate. A gap may challenge a deliberate choice only as a defensible
constraint-preserving opportunity or an explicitly handled trade-off; it may not
simply reintroduce an author-rejected alternative. Confidence remains only
`low`/`medium`/`high`, with no numerical score or new draft field.

The FormSolution and Select+Develop prompts inspect the same constraints and
require any developed solution to retain or explicitly overcome the relevant
rejection reason. The module still makes exactly the existing two calls
(exactly three candidates, then one select+develop call); no critic, reflection,
verification action, or new scientific state was added. Generic offline tests
cover rationale extraction/anchors, invalid source IDs, legal Gap evidence,
Refine propagation, and constraint-aware two-call Solution ideation. The fresh ByteFlow
regression in `outputs/m12_real_design_rationale_regression/` completed committed with
10 DeepSeek calls, 3 grounded method-level gaps, S1 selected, 2 experiments, cache-only
Semantic Scholar retrieval, and replay equality; no paper-specific prompt rule was added.

## 10.2 M12.2 Scientific Controller Inspiration Boundary

M12.2 only refines the existing Scientific Controller routing boundary. It does not add an
Action, Agent, reflection/critic loop, score, State field, Memory field, or method-level
schema. The rule is paper-agnostic and uses only existing structured fields. A feasible
distinct solution may still leave a worthwhile mechanism-exploration opportunity when it is
primarily incremental/local, the active gap is high-priority or structurally important, or
verification describes its distinctness mainly as an implementation choice. A clear new
mechanism can proceed directly to experiment design or commit when the other deterministic
conditions permit it. Distinctness and priority are each insufficient by themselves.

The current implementation does not expose a lexical opportunity gate. `SEEK_INSPIRATION`
remains optional because it is offered only when the existing state/verification route and
budget/action preconditions pass. The LLM is used only to choose among multiple already legal
Actions; it does not decide whether a new action or innovation score exists. Generic tests
cover wording invariance, partial-overlap reserve behavior, distinct missing-experiment
behavior, commit-ready reserve behavior, and illegal-action rejection. Real-paper smoke runs
remain evaluation cases, not prompt-specialization targets.

M12.2 的 fresh real evaluation 保存在
`outputs/m12_2_real_controller_inspiration_regression/`。它在 Solution Verification 的
真实 DeepSeek read operation timeout 处停止，未到达 Inspiration routing；因此该 artifact
不能证明 real action sequence 已因本 gate 改变，也没有 Commit/Finalizer 成功结论。离线
generic tests 和 deterministic routing contract 通过；该 real failure 不反向产生
paper-specific prompt 或 controller rule。

## 10.3 Correctness hardening before batch continuation

本轮不是新的 milestone，也不改变 `docs/` 中的方法定义。Markdown parser 对 Appendix
采用保守规则：只识别显式 `Appendix`/`Appendices`、`Supplementary`、`Supplemental`
等 heading marker，移除通用的单字母 heading 推断，因此标题或正文的 `A ...`、`C ...`
不会被误截断。References 与 Appendix 仍取最早边界，两个都不存在时保留完整正文；全量
37 篇 data audit 只读生成独立 `outputs/parser_audit.json`。

Targeted Revision user input 现在显式携带已有 `SeedPaperContext.method_constraints`。
Prompt 要求保留作者明确的 rationale/trade-off/rejected-alternative reason，或在改变
设计选择时加入处理、放宽或补偿该理由的机制；没有增加 State、Memory、Action 或新的
scientific schema。

Controller 删除所有 solution/gap wording 的 Inspiration keyword heuristic，保留七个
Action、deterministic mandatory routes、productive reserve 与多合法 Action 才调用一次
LLM 的边界。P0 generic tests 覆盖 Component freshness、第二次 Revision 在重新验证后
可运行、Revision constraint wiring、state-based Inspiration mask 和 parser false-positive
防护。完成 offline suite 与 parser audit 后只运行一次 fresh AtlasKV regression；不自动
恢复 M13 五篇 batch。

本轮验证中 offline suite 为 pytest/unittest 各 253 个测试通过；
`outputs/parser_audit.json` 覆盖 37 篇 Markdown，标题误判和结构推断均为 0。Canonical
AtlasKV regression `outputs/real_atlaskv_correctness_regression_final2/` committed，10 次
DeepSeek call 成功，Semantic Scholar 与 arXiv 均未产生 HTTP request（related pool 命中
global cache），并完成 State/Memory/trajectory replay equality。Controller 两次提供了
可选 Inspiration 但均选择直接推进；没有再次出现 Revision-before-reverification
contract error。

## 11. M13 Batch Evaluation and Human Review Interface

M13 是 execution-only evaluation infrastructure。`BatchEvaluationRunner` 只负责发现
Markdown、按确定的随机 seed 选择子集、为每篇 paper 分配独立 output directory、调用现有
`RealSinglePaperPilot` 并汇总 metadata；它不复制或重写 `TrajectoryRunner` 的科研 workflow，
不选择 gap/solution/action，也不跨 paper 共享 `ResearchState` 或 `ResearchMemory`。

每篇输出位于 `outputs/<batch_name>/<paper_id>/`，由既有单篇 pilot 产生
`seed_context.json`、State/Memory、trajectory/replay、Commit/Finalizer 和 manifest。根目录的
`evaluation_report.json` 使用 `models/evaluation.py` 的 closed structured contracts，只保存
执行计数、action/status、Gap/Solution/novelty、Inspiration/Revision、LLM/provider cost 和
replay consistency 等 count/categorical metrics；没有总分、ranking、ablation action 或新的
科研状态。`human_review_template.json` 只提供后续人工填写的 Gap、Solution 和 Final Proposal
质量字段，不被 workflow 读取。

当前 `BatchEvaluationRunner` 保持串行，避免在没有完成人工审查前扩大真实请求并发。真实执行
必须显式 `--allow-real-services`；`scripts/run_m13_batch_5_serial.sh` 固定抽取 5 篇、
`selection_seed=1301`，并将 timeout 默认设为 180 秒。未来若加入并行调度，仍必须保持每篇
executor/output/State/Memory 独立。

检索缓存属于 provider infrastructure 而非 Research Memory：Semantic Scholar 的 JSON
response cache 在同一个 operation/request key 上对 miss-check、HTTP 请求和原子写入加文件锁；
arXiv Inspiration Search 新增独立的全局文本响应 cache，同样使用 content-addressed key、
原子替换和锁。这样后续并行实现中，多个 paper 同时请求同一 related/inspiration query 时，
后到者可以复用已完成的响应而不重复付费；缓存命中不消耗 provider request count。缓存内容不含
API key，也不写回 `data/`。
