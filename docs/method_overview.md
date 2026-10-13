# 3 Method

## 3.1 Overview

给定一篇种子论文 P，我们的目标不是让大语言模型一次性生成一个完整科研思路，而是将科研思路生成建模为一个**由外部知识驱动的迭代式科研状态演化过程**。

直接从论文生成研究方案存在两个主要问题。首先，开放式生成缺少明确的研究目标约束，模型容易提出宽泛或与原论文弱相关的改进。其次，模型主要依赖参数知识，难以可靠判断当前问题是否已经被解决，也难以主动利用局部研究领域之外的知识形成新的方法方向。

为此，我们提出一个 **gap-driven scientific ideation agent**。系统首先从种子论文中识别值得进一步研究的 Research Gap，并围绕该 Gap 建立结构化的 Research State。随后，一个轻量 Scientific Controller 根据当前科研状态，在方案生成、外部灵感获取、相关工作验证、针对性修改和实验设计等科研模块之间动态选择下一步操作。

整个过程可以概括为：

Gap→Research State→Scientific Action→External Evidence→State Revision.

与固定的“检索—生成—验证”流水线不同，我们的方法允许智能体根据当前方案以及新获得的文献证据动态改变后续路径。例如，当已有工作只覆盖当前方案的某个组件时，系统仅针对该组件寻找替代机制并进行局部修改；当核心方案已被已有工作覆盖时，则放弃当前方案并探索其他候选方向；当研究缺口本身已经被解决时，则进一步切换研究目标。

------

# 3.2 Overall Architecture

整体架构包含一个共享的 **Research State**、一个高层 **Scientific Controller**，以及六类面向科研任务的功能模块：

1. **Gap Discovery and Verification**
2. **Solution Ideation**
3. **Related-work Verification**
4. **Inspiration Retrieval**
5. **Targeted Revision**
6. **Experiment Design**

此外，Research Memory 保存检索结果、失败方向以及历史决策，并由 Finalizer 在探索结束后生成完整科研提案。

整体交互流程如下：

```
Seed Paper
    ↓
Gap Discovery & Verification
    ↓
Research State
    ↓
Scientific Controller
    ↓
Scientific Action
    ↓
Specialized Module
    ↓
New Evidence / New Solution
    ↓
Research State Update
    ↓
        Repeat
    ↓
Experiment Design
    ↓
Commit & Finalization
```

这里，**Research State 表示当前科研思路是什么，而 Scientific Controller 决定下一步应该进行哪类科研活动**。具体任务则由专门模块完成，从而避免让单个通用 Agent 同时承担规划、检索、阅读和方案生成等所有职责。

------

# 3.3 Research State

Research State 用于维护当前有效的科研思路，而不是保存完整推理历史。其核心内容包括：

```
Research State
├── Active Gap
├── Candidate Solutions
├── Current Solution
│   ├── Core Idea
│   ├── Rationale
│   ├── Components
│   └── Novelty Status
└── Experiments
```

其中，**Active Gap** 定义当前需要解决的研究问题；**Candidate Solutions** 保存若干尚未深入展开的候选方向；**Current Solution** 表示当前正在发展的主要方案，并进一步拆分为若干可独立验证和修改的方法组件。

这种中等粒度的结构化表示使系统能够区分：

- Research Gap 是否仍然有效；
- 整体 Solution 是否存在已有工作重合；
- 哪个具体 Component 需要修改；
- 哪些实验因为方案变化而需要重新设计。

完整的检索记录、失败方案和历史证据不直接写入 Research State，而保存在 Research Memory 中，从而避免状态不断膨胀。

------

# 3.4 Gap Discovery and Verification

系统首先根据种子论文生成若干候选 Research Gaps。每个 Gap 都需要有种子论文中的依据，并从两个简单维度进行评估：

- **Confidence**：种子论文在多大程度上支持该 Gap 的存在；
- **Priority**：该问题是否值得作为后续研究目标。

系统最多保留五个具有实质差异的候选 Gap，并选择其中最值得探索的一个作为 **Active Gap**。

与直接生成 Solution 不同，我们要求 Active Gap 在进入方法设计前先接受一次外部相关工作验证：

```
Candidate Gaps
      ↓
Select Active Gap
      ↓
Related-work Verification
      ↓
valid / partially resolved / resolved
```

如果 Gap 已经被已有研究解决，则切换到其他候选 Gap；如果只是部分被解决，则允许对原 Gap 进行一次收窄后重新验证；只有验证为仍然有效的 Gap 才进入 Solution Ideation。

因此，Gap 不只是初始 Prompt 中的描述，而是整个 Agent 后续科研探索的稳定目标。

------

# 3.5 Scientific Controller

Scientific Controller 是系统的高层科研动作路由器。它不负责具体执行论文搜索、方案生成或方案修改，而只决定：

> **当前应该调用哪一种科研能力。**

第一版使用有限的模块级 Action Space：

```
SOLUTION_IDEATION
SEEK_INSPIRATION
VERIFY_RELATED_WORK
TARGETED_REVISION
DESIGN_EXPERIMENT
SWITCH_GAP
COMMIT
```

为了减少不必要的大模型调用，我们采用 **rule-guided control**。

对于下一步唯一明确的情况，系统直接执行规则。例如：

```
New Active Gap
→ VERIFY_RELATED_WORK


Valid Gap + No Solution
→ SOLUTION_IDEATION


Newly Developed Solution
→ VERIFY_RELATED_WORK


Revised Solution
→ Re-verification
```

只有存在多个合理科研方向时才调用 LLM Controller。例如，当方案与已有工作部分重合时，Controller 可以选择直接进行局部修改，或先寻找新的外部灵感；当方案已经通过新颖性验证后，则可以选择继续寻找灵感或进入实验设计。

因此 Controller 的动态性集中在真正存在科研选择的节点，而不是机械地为每一个内部操作进行规划。

------

## Budget-aware Control

为防止 Agent 无限探索，Controller 同时观察剩余的 step budget 和 search budget。

一次高层科研模块调用计为一个 step，而一次真实的外部论文搜索计为一次 search。Budget 不仅作为 Controller 判断是否继续探索的上下文，也用于限制当前合法 Action。

例如，当方案已经成熟且预算接近耗尽时，Controller 应更倾向于 Experiment Design 或 Commit，而不是继续进行边际价值较低的搜索。

------

# 3.6 Solution Ideation

当 Active Gap 验证通过后，Solution Ideation Module 负责生成初始解决方案。

该模块采用一个简单的 **generate–select–develop** 过程。

首先，根据：

- Active Gap；
- 种子论文的核心方法；
- Gap verification summary；
- 已失败的历史方案；

一次生成三个具有实质不同核心机制的 Candidate Solutions。

每个候选只包含：

```
Core Idea
Rationale
```

随后，通过一次额外的 LLM 调用同时完成候选选择和方案展开。

候选方案主要从以下方面进行比较：

- 与 Active Gap 的直接对应程度；
- 核心机制是否清晰；
- 是否具有实际研究可行性；
- 是否不是简单的工程性调整；
- 是否避免重复已经失败的研究方向。

最终选择的方案被展开成：

```
Current Solution
├── Core Idea
├── Rationale
└── 2–5 Components
```

组件只需要具体到能够支持后续 Component-level Verification 和局部修改，而不要求在这一阶段生成完整算法、公式或实验配置。

Current Solution 一旦形成，系统强制进入 Related-work Verification。

------

# 3.7 Related-work Verification

Related-work Verification 用于判断：

1. 当前 Gap 是否已经被解决；
2. Current Solution 是否已有高度类似工作；
3. 某个 Component 是否已有实质相同实现。

为了保持第一版系统简单，我们采用 **seed-paper-centric retrieval**。

首先，根据种子论文一次性获得一组相关论文，并缓存为：

```
Related Paper Pool
```

之后的 Gap、Solution 和 Component Verification 均复用这一论文池，而不重复进行复杂 Query Generation。

对于每次验证，系统根据当前 Target 从相关论文中筛选最多若干篇最相关工作，并基于标题和摘要进行结构化比较。

对于 Solution，验证结果分为：

```
no_overlap
partial_overlap
core_overlap
```

其中 `partial_overlap` 要求进一步定位具体发生重合的 Component，而 `core_overlap` 表示已有工作同时覆盖了基本相同的问题和核心机制。

这种设计使后续系统能够区分：

```
Component overlap
→ Local Revision


Core Solution overlap
→ Reject Current Solution


Gap resolved
→ Switch Gap
```

而不是每次发现相关工作都重新生成整个科研思路。

------

# 3.8 Inspiration Retrieval via Dynamic Entity Bridging

Related-work Verification 的目标是寻找“已经存在什么”，而 Inspiration Retrieval 的目标则是：

> **寻找当前局部研究领域之外，哪些科学机制可能迁移过来解决当前问题。**

这里我们借鉴 entity-centric scientific ideation 的思想，但不构建大规模静态 Entity Knowledge Base。

我们提出一个轻量的 **Dynamic Entity Bridging** 过程：

```
Current Research Need
        ↓
Bridge Entity Generation
        ↓
External Knowledge Grounding
        ↓
Transferable Mechanism
```

首先，系统根据当前 Gap、Solution weakness 或 overlapped Component 抽象出当前真正需要的功能。例如：

> “如何决定什么时候需要进行状态修正，而不使用已有的 uncertainty threshold？”

随后，LLM 不是直接生成新的 Solution，而是提出若干可能建立跨领域联系的 **Bridge Entities**。实体主要限制为：

- scientific concept；
- mechanism；
- method；
- phenomenon；
- system principle。

例如：

```
event-triggered control
optimal stopping
change-point detection
```

系统首先产生五个候选 Bridge Entities，再选择其中三个与当前科研需求结构联系最强、同时不同于已失败机制的实体。

这些实体被组合成一次在线 arXiv 检索，从真实论文摘要中寻找相关机制。随后 LLM 提取：

```
Source Mechanism
Transferable Principle
Adaptation Hint
```

从而形成最多三个 Inspiration Candidates。

因此 Inspiration Retrieval 并不是直接搜索与 Current Solution 最相似的论文，而是通过：

Research Need→Bridge Entity→External Mechanism

显式扩展 Solution Space。

如果在线检索不可用，系统仍可以退化为 LLM-only Entity Bridging，但会明确记录该灵感没有外部文献 grounding。

------

# 3.9 Evidence-Grounded Targeted Revision

Inspiration 和 Verification 都不会直接修改 Research State。

当系统已经获得明确的 overlap evidence 或新的 Inspiration 后，由 Targeted Revision Module 对当前 Solution 进行针对性修改。

我们区分三种影响层级。

### Local Component Revision

如果已有工作只覆盖某个 Component，则保留：

- Active Gap；
- Core Solution；
- 其他 Components；

只修改目标 Component。

### Solution Replanning

如果核心 Solution 已经被已有工作覆盖，则拒绝当前方案，但保留 Active Gap，并优先尝试剩余 Candidate Solutions 或新的 Inspiration。

### Gap Switching

如果外部文献表明当前 Gap 本身已经被解决，则不继续修改 Solution，而直接切换到其他候选 Gap。

因此：

Local Revision=Solution Replanning=Gap Switching.

修改完成后，只重新验证受到影响的科研单元。例如只修改单个 Component 时优先执行 Component-level Verification；只有整体核心机制发生变化时才重新验证整个 Solution。

------

# 3.10 Experiment Design and Commit

当 Current Solution 已通过相关工作验证后，Scientific Controller 可以选择继续寻找灵感，或者进入 Experiment Design。

Experiment Design Module 根据：

```
Active Gap
+
Current Solution
+
Components
```

生成若干关键实验，每个实验主要描述：

- 实验目标；
- 基本设计；
- 它验证 Solution 的哪个部分。

我们不要求 Agent 在这一阶段直接产生完整实验代码，而是确保最终研究方案具有明确的可验证路径。

当：

- Gap 已验证；
- Solution 已形成并通过相关工作检查；
- 不存在未解决的核心重合；
- Experiment 已完成；
- 没有重要 stale 内容；

系统允许 `COMMIT` 进入 Controller 的 Action Space。

是否立即结束仍由 Controller 根据当前方案质量和剩余预算决定。

------

# 3.11 Research Memory and Final Output

Research Memory 保存当前 State 之外的过程信息，主要包括：

```
Retrieved Papers
Rejected Gaps
Rejected Solutions
Failed Mechanisms
Inspiration Records
Controller Decisions
```

其作用主要是避免 Agent 重复探索已经失败的方向。

最终，当 Controller 选择 Commit 时，Finalizer 根据当前 Research State 生成结构化科研提案，包括：

```
Research Gap


Proposed Solution


Solution Rationale


Method Components


Experimental Design


Novelty Analysis


Expected Contributions
```

Contribution 不在探索过程中持续维护，而是在最终阶段根据 Gap、Solution、Related Work differences 和 Experiment Design 统一总结。

------

# 3.12 Method Summary

我们的整体方法可以概括为一个 **gap-driven, evidence-seeking research state evolution process**：

Gap→Solution→Evidence Seeking→Localized Revision→Experiment

其中 Scientific Controller 根据当前 Research State 动态调度不同科研能力，而外部文献既用于验证当前想法，也用于通过 Dynamic Entity Bridging 引入新的解决机制。

方法的关键并不在于设计一个更加自由的通用 ReAct Agent，而在于为科研思路生成引入三个明确的结构性约束：

**Gap-centered exploration**：先确定仍然开放且值得研究的问题，再围绕该问题展开方案空间。

**State-conditioned scientific actions**：根据当前 Research State 动态选择科研活动，而不是执行固定 pipeline。

**Evidence-localized evolution**：外部知识并不触发整个方案重新生成，而是根据其实际影响范围进行局部修改、方案重规划或研究目标切换。

最终，科研思路不再由单次开放式 Prompt 直接产生，而是通过：

> **提出 → 检索 → 验证 → 修改 → 再验证**

逐步演化为一个具有明确 Research Gap、具体 Solution 和可验证 Experiment Design 的科研方案。