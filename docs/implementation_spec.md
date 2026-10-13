

### 1. Research State

定义 Agent 持续维护的科研状态。

### 2. Scientific Controller

负责规则路由、Action Mask、动态 Action 选择以及 Budget-aware Control。

### 3. Gap Discovery and Management

负责 Candidate Gap 的生成、评价、选择、Refine 和 Switch。

### 4. Solution Ideation

负责：

```
FormSolution
→ SelectSolution
→ DevelopSolution
```

### 5. Related-work Verification

统一负责：

```
VerifyGap
VerifySolution
VerifyComponent
```

### 6. Inspiration Retrieval

负责：

```
Research Need
→ Dynamic Entity Bridging
→ arXiv Grounding
→ Transferable Inspiration
```

### 7. Targeted Revision

负责根据 Verification / Inspiration 对具体科研单元进行局部修改和状态更新。

### 8. Experiment Design

根据稳定 Solution 形成关键实验设计。

### 9. Research Memory

保存：

```
papers
verification results
inspirations
failed directions
controller history
```

以及相关缓存。

### 10. Commit + Finalizer

负责：

```
Commit Eligibility
→ Controller decides Commit
→ Final Research Proposal
```

# 1. Research State

## 1.1 作用

Research State 用于保存 Agent 当前有效的科研状态，是所有科研模块共享和持续更新的核心数据结构。

它主要回答三个问题：

> 当前正在研究什么 Gap？
>  当前正在发展什么 Solution？
>  当前方案已经验证和发展到什么程度？

Research State 只保存**影响后续科研决策的当前状态**。完整论文内容、检索历史、失败轨迹等由 Research Memory 保存。

------

## 1.2 整体结构

第一版 Research State 设计为：

```
ResearchState
{
    active_gap


    candidate_solutions[]


    current_solution


    experiments[]
}
```

整体关系为：

```
Active Gap
    ↓
Candidate Solutions
    ↓
Current Solution
    ↓
Components
    ↓
Experiments
```

------

# 1.3 Active Gap

`active_gap` 表示当前 Agent 正在尝试解决的研究缺口。

```
ActiveGap
{
    id


    statement


    confidence
    confidence_reason


    priority
    priority_reason


    status


    evidence_refs
}
```

### `statement`

用一段简洁描述表示当前研究缺口，重点包含：

- 当前研究场景；
- 尚未解决的限制或能力缺失；
- 该问题带来的影响。

Gap 应保持**问题导向**，不包含具体 Solution。

------

### `confidence`

表示种子论文对该 Gap 存在性的支持程度：

```
low
medium
high
```

同时保存：

```
confidence_reason
```

用于简要说明判断依据。

------

### `priority`

表示该 Gap 是否值得继续探索：

```
low
medium
high
```

同时保存：

```
priority_reason
```

主要反映：

- 是否接近种子论文的核心问题；
- 是否具有明确研究价值；
- 是否能够形成可研究的问题。

------

### `status`

用于表示当前 Gap 的状态，例如：

```
active
resolved
rejected
```

Active Gap 正常情况下为：

```
active
```

如果 Related-work Verification 发现该问题已经解决，则变为：

```
resolved
```

------

### `evidence_refs`

只保存支持该 Gap 的证据引用 ID：

```
evidence_refs[]
```

完整证据内容存入 Research Memory。

------

# 1.4 Candidate Solutions

针对一个经过验证的 Active Gap，Solution Ideation 一次生成固定 3 个轻量 Candidate Solutions。

结构：

```
CandidateSolution
{
    id


    core_idea


    rationale


    status
}
```

------

### `core_idea`

表示候选方案最核心的解决机制。

要求：

> 能明确说明准备采用什么机制解决 Active Gap。

不同 Candidate 应具有实质不同的核心机制。

------

### `rationale`

解释：

> 为什么该机制有可能解决当前 Gap。

用于建立：

```
Gap
→ Solution
```

之间的机制联系。

------

### `status`

建议使用：

```
candidate
selected
rejected
```

例如：

```
S1 = selected
S2 = candidate
S3 = candidate
```

如果 S1 后续被判定 `core_overlap`：

```
S1.status = rejected
```

然后可以继续从 S2、S3 中选择，而不需要立即重新生成 Candidate Solutions。

------

# 1.5 Current Solution

被选中的 Candidate 会被进一步展开为 Current Solution：

```
CurrentSolution
{
    id


    core_idea


    rationale


    components[]


    status


    novelty_status
    novelty_summary


    evidence_refs
}
```

Current Solution 是后续 Verification、Inspiration、Revision 和 Experiment Design 的主要操作对象。

------

## 1.6 Core Idea

`core_idea` 描述整个 Solution 的核心解决机制。

应能够说明：

- 对原有方法的哪个关键部分进行改变；
- 引入什么核心机制；
- 该机制如何针对 Active Gap。

粒度要求是：

> 足够支持后续 Solution-level Related-work Verification。

不需要在 Research State 中保存完整 Method Section。

------

# 1.7 Rationale

`rationale` 用于解释：

> 为什么当前 Solution 能够解决 Active Gap。

它保持为一段简洁机制说明，并随核心 Solution 的重要修改同步更新。

------

# 1.8 Components

Current Solution 进一步拆分为 **2–5 个主要 Components**。

```
Component
{
    id


    description


    status


    evidence_refs
}
```

### `description`

需要说明：

> 该 Component 是什么，以及它在整个 Solution 中承担什么功能。

粒度只需要达到：

> 可以支持 Solution Verification 中的局部 overlap 定位、Focused Re-verification 和 Targeted Revision。

例如：

```
Component C2


Description:
Detects significant changes in intermediate reasoning
states and determines whether corrective intervention
should be triggered.
```

不要求保存：

- 详细代码；
- 超参数；
- 完整算法公式；
- 网络结构细节。

------

### `status`

Component 的状态只用于表示其当前是否稳定有效，以及是否因为 Revision 需要重新检查：

```
ready
modified
stale
```

### `ready`

表示当前 Component 处于正常有效状态，没有待处理的修改。

### `modified`

表示该 Component 刚刚经过 Targeted Revision，需要进行 Focused Re-verification。

例如：

```
C2 revised
→ C2.status = modified
```

### `stale`

表示由于 Core Idea 或其他关键部分发生变化，该 Component 当前描述或作用可能已经与新的 Solution 不完全一致，需要重新检查或更新。

Component 本身不再维护：

```
distinct
overlapped
```

这样的新颖性状态。

Solution 的整体新颖性统一由：

```
current_solution.novelty_status
```

维护，而具体 novelty-relevant overlap 由 Related-work Verification Result 记录。

------

### `evidence_refs`

保存与该 Component 相关的 Verification Evidence 引用，例如定位到该 Component 的 novelty-relevant overlap 或 focused re-verification evidence。

完整比较内容仍保存到 Research Memory。

------

# 1.9 Novelty Status

Current Solution 保存：

```
novelty_status
novelty_summary
```

其中：

```
novelty_status =
unchecked
distinct
partial_overlap
core_overlap
```

### `unchecked`

Solution 尚未完成 Related-work Verification，或 Solution 经过修改后尚未重新验证。

------

### `distinct`

已有工作可以与 Current Solution 共享：

- 标准技术；
- 通用组件；
- 相似任务设定；
- 辅助机制；

但没有实质覆盖当前 Solution 中**承担主要新颖性的关键机制或设计**。

因此：

> `distinct` 不表示与已有工作完全没有重合，而表示已有重合不足以削弱当前 Solution 的核心新颖性主张。

例如：

```
Shared with prior work:
standard encoder
change detection module


Distinctive mechanism:
using reasoning-state transitions
to dynamically trigger corrective intervention
```

此时仍可判断为：

```
novelty_status = distinct
```

------

### `partial_overlap`

已有工作已经实质覆盖 Current Solution 中一个或多个**局部的新颖性承载部分**，从而部分削弱当前方案的新颖性，但整体核心机制仍然保持实质区别。

例如：

```
Current Solution
├── Core Idea
├── C1
├── C2 ← novelty-bearing mechanism overlapped
└── C3
```

如果 C2 原本承担重要创新作用，但已有论文存在实质相同的设计，而整体 Core Idea 仍然不同，则：

```
novelty_status = partial_overlap
```

此时可以根据具体 overlap evidence 进入：

```
TARGETED_REVISION
```

或：

```
SEEK_INSPIRATION
```

------

### `core_overlap`

已有工作已经实质覆盖 Current Solution 的**核心解决机制**，使当前方案的主要新颖性主张无法成立。

通常表现为已有工作同时具有：

```
相同或高度相似的研究问题
+
实质相同的核心解决机制
```

此时：

```
novelty_status = core_overlap
```

Current Solution 会被拒绝。

------

### 三种验证结果的边界

核心判断标准是：

> **已有工作的重合是否影响当前 Solution 所声称的主要新颖性。**

```
只共享标准或辅助技术
        ↓
     distinct


局部 novelty-bearing element 被覆盖
        ↓
 partial_overlap


核心 novelty mechanism 被覆盖
        ↓
  core_overlap
```

因此：

| Novelty Status    | 重合情况                                       | 对当前新颖性的影响          |
| ----------------- | ---------------------------------------------- | --------------------------- |
| `distinct`        | 标准技术、辅助组件或非关键元素存在共享         | 不实质削弱核心新颖性        |
| `partial_overlap` | 一个或多个局部 novelty-bearing elements 被覆盖 | 部分削弱，可尝试局部修改    |
| `core_overlap`    | 核心解决机制被覆盖                             | 根本削弱，拒绝当前 Solution |

------

### `novelty_summary`

保存一段简短的新颖性判断摘要，重点说明：

- 哪些内容与已有工作共享；
- 哪些部分承担当前方案的主要区别；
- 如果存在 novelty-relevant overlap，具体发生在哪里。

例如 `distinct`：

```
The solution shares standard state-detection techniques
with prior work, while its transition-conditioned
correction mechanism remains substantively distinct.
```

例如 `partial_overlap`：

```
Component C2, which constitutes a local novelty-bearing
mechanism, substantially overlaps with prior work,
while the overall solution mechanism remains distinct.
```

详细的论文比较记录仍保存到 Research Memory。

------

# 1.10 Experiments

最终 Research State 可以保存多个实验：

```
Experiment
{
    id


    objective


    design


    what_it_validates


    status
}
```

通常保留若干关键实验即可。

------

### `objective`

实验希望回答什么研究问题。

------

### `design`

实验如何进行的简要设计。

不要求保存完整实验代码。

------

### `what_it_validates`

明确指出该实验验证：

- 整体 Solution；
- 某个核心机制；
- 某个 Component。

这样后续当 Solution 被修改时，可以判断哪些 Experiment 需要重新设计。

------

### `status`

例如：

```
ready
stale
```

如果相关 Solution 或 Component 发生重要变化：

```
experiment.status = stale
```

------

# 1.11 状态更新原则

Research State 使用**局部更新**，而不是每次重新生成整个状态。

例如：

### Component 修改

```
C2 revised
→ C2.status = modified
→ novelty_status = unchecked
```

随后执行：

```
Solution Verification
verification_focus = [C2]
```

即针对修改后的 C2 进行 Focused Re-verification，同时保留整个 Current Solution 作为判断上下文。

如果重新验证通过：

```
C2.status = ready
```

并根据当前完整 Verification Result 更新：

```
current_solution.novelty_status
```

------

### Core Solution 修改

保持：

```
core_idea changed
→ novelty_status = unchecked
→ relevant experiments = stale
```

如果 Core Idea 的变化还会使某些 Components 与新的整体机制不再一致，可以将相关 Component 标记为：

```
component.status = stale
```

后续在更新 Solution 时重新检查。

------

### Solution 被判 Core Overlap

保持：

```
current_solution.novelty_status = core_overlap
→ corresponding candidate.status = rejected
→ current_solution = null
```

随后可以从剩余 Candidate Solutions 中重新选择。

------

### Gap 被判 resolved

保持：

```
active_gap.status = resolved
```

随后交给 Gap Discovery and Management 切换新的 Active Gap。

------

# 1.12 Research State 与 Research Memory 的边界

Research State 只保留：

> **当前 Agent 做下一步决策真正需要看到的内容。**

因此其中主要是：

```
Active Gap
Candidate Solutions
Current Solution
Components
Novelty Status
Experiments
Evidence References
```

而类似：

```
完整论文摘要
论文比较记录
历史检索结果
Rejected Solutions 的详细原因
Bridge Entities
Inspiration Records
Controller Decision History
```

统一保存到 Research Memory。

这样可以保证 Research State 始终保持紧凑。

------

# 1.13 第一版建议的数据结构

最终可以整理为：

```
ResearchState
{
    active_gap:
    {
        id
        statement


        confidence
        confidence_reason


        priority
        priority_reason


        status
        evidence_refs
    }


    candidate_solutions:
    [
        {
            id
            core_idea
            rationale
            status
        }
    ]


    current_solution:
    {
        id


        core_idea
        rationale


        components:
        [
            {
                id
                description
                status
                evidence_refs
            }
        ]


        status


        novelty_status
        novelty_summary


        evidence_refs
    }


    experiments:
    [
        {
            id
            objective
            design
            what_it_validates
            status
        }
    ]
}
```

## 核心设计原则

这一版 Research State 可以概括为三点：

**结构化但不过细。**
 只拆到 Gap、Solution、Component、Experiment 四个实际需要操作的粒度。

**只保存当前有效状态。**
 历史轨迹和完整证据交给 Research Memory。

**支持局部更新。**
 外部证据只修改实际受影响的科研单元，为后续 Targeted Revision 和 Local Re-verification 提供基础。



# 2. Scientific Controller

## 2.1 作用

Scientific Controller 是系统的**高层科研动作调度器**，负责根据当前 Research State 决定：

> **下一步应该调用哪一个科研功能模块。**

Controller 只负责模块级决策，不执行具体科研任务。整体关系为：

```
Research State
      ↓
Scientific Controller
      ↓
Scientific Action
      ↓
Specialized Module
      ↓
Research State Update
```

具体的方案生成、论文比较、灵感获取、局部修改等操作由对应模块完成。

------

## 2.2 Scientific Action Space

第一版使用固定的有限 Action Space：

```
ScientificAction
{
    SOLUTION_IDEATION


    SEEK_INSPIRATION


    VERIFY_RELATED_WORK


    TARGETED_REVISION


    DESIGN_EXPERIMENT


    SWITCH_GAP


    COMMIT
}
```

对应关系为：

| Action                | 调用模块                     |
| --------------------- | ---------------------------- |
| `SOLUTION_IDEATION`   | Solution Ideation            |
| `SEEK_INSPIRATION`    | Inspiration Retrieval        |
| `VERIFY_RELATED_WORK` | Related-work Verification    |
| `TARGETED_REVISION`   | Targeted Revision            |
| `DESIGN_EXPERIMENT`   | Experiment Design            |
| `SWITCH_GAP`          | Gap Discovery and Management |
| `COMMIT`              | Commit + Finalizer           |

Controller 只能从当前允许的 Action 中选择，不生成新的 Action。

------

# 2.3 Rule-guided Control

对于下一步唯一明确的情况，直接通过规则触发，不调用 LLM Controller。

主要规则如下。

### Active Gap 刚被选出

```
New Active Gap
→ VERIFY_RELATED_WORK
```

首先验证 Gap 是否仍然成立。

------

### Gap 验证有效且尚无 Solution

```
Gap = valid
AND
no usable Current Solution
→ SOLUTION_IDEATION
```

如果已经存在尚未使用的 Candidate Solutions，则 Solution Ideation 直接从剩余候选中选择和展开。

------

### Current Solution 刚完成生成

```
Current Solution = developed
→ VERIFY_RELATED_WORK
```

一个具体 Solution 形成后，必须首先进行相关工作验证。

------

### Gap 已被解决

```
Gap = resolved
→ SWITCH_GAP
```

切换到其他 Candidate Gap。

------

### Targeted Revision 完成

根据实际修改范围重新验证：

```
single component changed
→ Component-level Verification
```

或者：

```
core solution changed
→ Solution-level Verification
```

因此不需要 Controller 决定“是否重新验证”。

------

# 2.4 Controller 真正负责的动态决策

只有当前存在多个合理科研方向时才调用 LLM Controller。

主要包括三类场景。

## Partial Overlap

当：

```
novelty_status = partial_overlap
```

允许：

```
TARGETED_REVISION


SEEK_INSPIRATION
```

这里的 `partial_overlap` 表示：

> 已有工作实质覆盖了 Current Solution 中一个或多个局部的 novelty-bearing elements，从而部分削弱当前方案的新颖性，但整体核心机制仍然具有实质区别。

Controller 判断：

- 如果已有 Verification Evidence 足以支持针对性修改，则直接 `TARGETED_REVISION`；
- 如果受到影响的关键机制缺少合适替代方案，则 `SEEK_INSPIRATION`。

------

## Solution 已通过 Verification

当：

```
novelty_status = distinct
```

并且方案已经稳定，可以选择：

```
SEEK_INSPIRATION


DESIGN_EXPERIMENT
```

这里的 `distinct` 表示：

> Current Solution 可以与已有工作共享标准技术、辅助组件或其他非关键元素，但已有工作没有实质覆盖当前 Solution 所承担主要新颖性的关键机制。

因此，`distinct` 并不要求 Current Solution 与已有工作完全没有重合，而是表示：

> 当前发现的重合不足以削弱 Solution 的核心新颖性主张。

Controller 判断：

- 当前 Solution 是否仍存在值得进一步扩展的机制空间；
- 是否值得消耗剩余搜索预算继续寻找外部灵感；
- 或者当前方案已经具有足够明确的差异性，可以进入 Experiment Design。

------

## 满足 Commit 条件

当 Commit Gate 已通过时：

```
commit_allowed = true
```

`COMMIT` 才进入合法动作集合。

此时 Controller 可以决定：

```
COMMIT
```

或者在预算允许且存在明确改进空间时继续：

```
SEEK_INSPIRATION
```

因此：

> **规则判断是否具备结束条件，Controller 判断当前是否值得结束。**

------

# 2.5 Allowed Actions

在调用 LLM Controller 前，系统首先根据当前 Research State 构造：

```
allowed_actions
```

例如：

```
Current Solution developed
novelty_status = unchecked
```

则：

```
allowed_actions =
[
    VERIFY_RELATED_WORK
]
```

只有一个合法 Action 时直接执行。

------

如果：

```
novelty_status = partial_overlap
```

则：

```
allowed_actions =
[
    TARGETED_REVISION,
    SEEK_INSPIRATION
]
```

此时才调用 Controller。

其中 `partial_overlap` 表示当前存在**会实质削弱局部新颖性主张的重合**，因此需要优先修正或寻找新的机制。

------

如果：

```
novelty_status = distinct
experiments = empty
```

则：

```
allowed_actions =
[
    SEEK_INSPIRATION,
    DESIGN_EXPERIMENT
]
```

此时说明当前 Solution 的主要 novelty-bearing mechanism 已经与已有工作保持足够实质区别，可以进入实验设计；如果 Controller 判断仍有明显提升空间，也可以继续寻找 Inspiration。

这种 Action Mask 可以避免 LLM 选择当前状态下不合理的行为。

------

# 2.6 Controller Context

Controller 不读取完整论文或完整历史，只读取完成当前决策需要的紧凑上下文：

```
ControllerContext
{
    active_gap_summary


    current_solution_summary


    component_status


    novelty_status


    experiment_status


    last_action
    last_result


    relevant_failure_memory


    commit_allowed


    budget
    {
        steps_used
        max_steps


        searches_used
        max_searches


        remaining_steps
        remaining_searches
    }


    allowed_actions
}
```

其中：

### `last_action / last_result`

帮助 Controller 理解刚刚发生了什么。

### `relevant_failure_memory`

提供与当前决策直接相关的失败方向，例如：

```
- Solution S1 was rejected due to core overlap.
- Uncertainty-threshold triggering has already been rejected.
```

用于减少重复探索。

### `budget`

帮助 Controller 判断：

> 继续探索带来的潜在收益是否值得额外成本。

------

# 2.7 Budget-aware Control

第一版维护两类简单预算。

## Step Budget

每执行一次高层 Scientific Action / Specialized Module：

```
steps_used += 1
```

例如：

```
SOLUTION_IDEATION
VERIFY_RELATED_WORK
SEEK_INSPIRATION
TARGETED_REVISION
DESIGN_EXPERIMENT
```

各算一个 step。

模块内部的多个 LLM operation 不额外算 Agent step。

------

## Search Budget

每真正发生一次外部学术检索：

```
searches_used += 1
```

例如：

```
Semantic Scholar related-paper retrieval
→ +1 search
Inspiration arXiv search
→ +1 search
```

如果直接读取已有 Cache：

```
→ +0 search
```

因此 Search Budget 对应实际发生的外部知识获取行为。

------

## Budget 对 Action 的限制

Budget 同时影响 `allowed_actions`。

例如搜索预算已经耗尽：

```
remaining_searches = 0
```

则不能继续执行需要新外部检索的 Inspiration Retrieval。

如果：

```
remaining_steps = 1
AND
commit_allowed = true
```

则可以直接将：

```
allowed_actions = [COMMIT]
```

------

# 2.8 Budget Exhaustion

如果 Budget 用尽：

### 已满足 Commit Gate

```
budget exhausted
AND
commit_allowed = true
→ COMMIT
```

### 尚未满足 Commit Gate

停止当前 trajectory，返回：

```
best_current_research_state
+
unresolved_issues
```

例如说明：

```
Solution verification incomplete.


Experiment design not finished.
```

不强行把未完成状态包装成完整科研方案。

------

# 2.9 Controller Output

Controller 输出保持简单：

```
ControllerDecision
{
    action


    reason
}
```

例如：

```
{
    action: SEEK_INSPIRATION,


    reason:
    "The current solution is valid but its core mechanism
    still has clear room for improvement, and sufficient
    search budget remains."
}
```

`reason` 只保存简短的决策理由，用于：

- debugging；
- trajectory analysis；
- case study；
- 方法可解释性分析。

------

# 2.10 第一版执行逻辑

代码层可以设计为：

```
ScientificController
{
    check_rule_trigger()


    get_allowed_actions()


    build_context()


    select_action()
}
```

整体执行：

```
Research State
      ↓
Check Rule Trigger
      ↓
有确定动作？
   /       \
 Yes        No
  ↓          ↓
直接执行   Generate Allowed Actions
             ↓
       只有一个 Action？
          /       \
        Yes        No
         ↓          ↓
       直接执行    Build Context
                     ↓
                LLM Controller
                     ↓
                Select Action
                     ↓
              Specialized Module
                     ↓
              Update Research State
                     ↓
                   Repeat
```

------

## 核心实现原则

Scientific Controller 最终可以概括为四点：

**Constrained**：只从固定 Scientific Action Space 中选择。

**Rule-guided**：明确且唯一的下一步直接由规则触发。

**State-conditioned**：真正的开放科研决策根据当前 Research State 动态完成。

**Budget-aware**：Controller 明确知道剩余 step/search budget，并据此决定是否值得继续探索。

因此 Scientific Controller 的核心作用不是执行科研任务，而是：

> **在正确的科研状态下，选择下一项最值得执行的科研能力。**



# 3. Gap Discovery and Management

## 3.1 作用

Gap Discovery and Management 负责从 Seed Paper 中发现值得继续研究的问题，并维护当前 Research Gap 的候选集合与切换过程。

它主要完成：

```
Generate Gaps
→ Evaluate Gaps
→ Select Active Gap
→ Refine Gap if needed
→ Switch Gap if needed
```

其中，**Gap 的外部文献验证由 Related-work Verification Module 完成**。本模块根据验证结果决定是否继续、收窄或切换 Gap。

------

## 3.2 Candidate Gap 生成

系统首先根据 Seed Paper 生成多个 Candidate Gaps。

输入主要包括：

```
Seed Paper Context
├── research objective
├── core method
├── main findings
├── limitations
└── relevant experimental observations
```

初始 Gap Discovery **只使用 Seed Paper 本身的信息**，不主动检索外部论文。

候选数量：

```
Candidate Gaps ≤ 5
```

不要求必须生成满 5 个，只保留有明确依据且具有实质差异的 Gap。

------

## 3.3 Candidate Gap 结构

每个 Candidate Gap 保存：

```
CandidateGap
{
    id


    statement


    confidence
    confidence_reason


    priority
    priority_reason


    evidence_refs


    status
}
```

其中 `statement` 应明确描述：

> **当前研究场景中存在什么尚未解决的问题，以及该问题为什么会限制现有方法。**

Gap 本身保持问题导向，不提前包含具体 Solution。

------

# 3.4 Gap Confidence

`confidence` 表示：

> Seed Paper 中的证据在多大程度上支持该 Gap 确实存在。

采用：

```
low
medium
high
```

判断原则：

### High

Seed Paper 有直接陈述、实验结果或明确 limitation 支持该 Gap。

### Medium

论文没有直接声明，但可以从方法或实验结果中合理推断。

### Low

主要来自模型自己的扩展推测，论文支持较弱。

同时保存：

```
confidence_reason
```

简要说明依据。

------

# 3.5 Gap Priority

`priority` 表示：

> 如果该 Gap 确实存在，它是否值得作为后续科研目标。

采用：

```
low
medium
high
```

主要考虑：

- 是否接近 Seed Paper 的核心研究问题；
- 是否具有明确科学或方法价值；
- 是否能够形成一个可继续研究的问题；
- 是否有希望形成实质方法改进。

同时保存：

```
priority_reason
```

不使用人为构造的数值评分公式。

------

# 3.6 Gap Agenda

Gap Module 内部维护一个轻量的：

```
GapAgenda
{
    candidate_gaps[]
}
```

每个 Gap 的状态使用：

```
candidate
active
resolved
rejected
```

### `candidate`

尚未被选择的可用 Gap。

### `active`

当前 Agent 正在探索的 Gap。

### `resolved`

外部 Related-work Verification 表明该 Gap 已经被已有研究实质解决。

### `rejected`

该 Gap 本身证据不足、优先级较低或不值得继续探索。

Gap Agenda 用于后续切换研究目标，而不需要每次重新从 Seed Paper 生成 Gap。

------

# 3.7 Active Gap Selection

Candidate Gaps 生成并评估后，由 Gap Module 自己选择 Active Gap。

选择主要遵循：

```
Confidence
+
Priority
+
Problem Clarity
+
Researchability
```

基本原则：

1. 优先避免 `low confidence` Gap；
2. 优先选择 `high priority` Gap；
3. 如果多个 Gap 接近，则选择问题定义更明确、后续更容易形成具体研究方案的 Gap。

该过程属于 Gap Module 内部决策，不交给 Scientific Controller。

选中后：

```
candidate_gap.status = active
research_state.active_gap = candidate_gap
```

随后系统规则直接触发：

```
VERIFY_RELATED_WORK
```

对 Active Gap 进行外部验证。

------

# 3.8 与 Related-work Verification 的交互

Gap Verification 返回三种结果：

```
valid


partially_resolved


resolved
```

Gap Module 根据结果进行处理。

------

## Valid

```
Gap Verification = valid
→ 保留当前 Active Gap
→ 进入 Solution Ideation
```

表示在当前 related-work evidence 中，该 Gap 仍然值得继续研究。

------

## Partially Resolved

```
Gap Verification = partially_resolved
→ RefineGap
→ VerifyGap again
```

表示已有工作已经解决 Gap 的一部分，但仍然存在明确剩余问题。

------

## Resolved

```
Gap Verification = resolved
→ Gap.status = resolved
→ SwitchGap
```

不再围绕当前 Gap 生成新的 Solution。

------

# 3.9 RefineGap

每个 Active Gap 最多允许执行 **一次 RefineGap**。

输入：

```
Current Active Gap


covered_aspects


remaining_gap


verification_summary
```

RefineGap 的目标是：

> 根据已有工作的覆盖范围，把原 Gap 收窄为仍然没有被解决的部分。

例如：

```
Original Gap
→ 已有工作覆盖其中一部分
→ Refined Gap
```

要求：

- 保持与原 Gap 相同的核心研究问题；
- 只能收窄或明确边界；
- 不能重新创造一个完全不同的问题。

Refine 后：

```
→ VerifyGap again
```

第二次验证之后不再继续循环 Refine：

```
valid
→ continue


resolved
→ SwitchGap
```

这样避免 Agent 不断缩小问题来人为制造“novel gap”。

------

# 3.10 SwitchGap

当前 Gap 被判定为：

```
resolved
```

或不再适合继续探索时，执行：

```
SwitchGap
```

Gap Module 从 Gap Agenda 中已有的：

```
status = candidate
```

Gap 中重新选择。

流程：

```
Current Gap unavailable
        ↓
Select next candidate
        ↓
Set as Active Gap
        ↓
VERIFY_RELATED_WORK
```

不会重新执行 Gap Generation 和 Gap Evaluation。

只有当当前 Gap Agenda 中不存在可用候选时，才结束该 Seed Paper 的 Gap exploration。

------

# 3.11 Candidate Gap 的复用

Gap Agenda 在整个 Seed Paper trajectory 中持续保留。

例如：

```
G1 = active
G2 = candidate
G3 = candidate
```

如果：

```
G1 → resolved
```

则：

```
G1 = resolved
G2 → active
G3 = candidate
```

这样系统能够快速切换研究目标，而不需要反复调用 LLM 重新发现问题。

------

# 3.12 Research State 更新

Gap Module 只把当前 Active Gap 写入 Research State：

```
research_state.active_gap
```

完整 Candidate Gap 集合由 Gap Agenda 管理。

当切换 Gap 时：

```
old_gap.status = resolved / rejected


new_gap.status = active


research_state.active_gap = new_gap
```

随后围绕新 Gap 重新开始：

```
Gap Verification
→ Solution Ideation
```

------

# 3.13 第一版代码接口

可以实现为：

```
GapDiscoveryModule
{
    generate_gaps(
        seed_paper_context
    )


    evaluate_gaps(
        candidate_gaps
    )


    select_active_gap(
        gap_agenda
    )


    refine_gap(
        active_gap,
        verification_result
    )


    switch_gap(
        gap_agenda
    )
}
```

主要执行流程：

```
Seed Paper
    ↓
generate_gaps()
    ↓
evaluate_gaps()
    ↓
select_active_gap()
    ↓
VerifyGap
    ↓
valid
│
├→ Solution Ideation
│
partially_resolved
│
├→ refine_gap()
│  → VerifyGap again
│
resolved
│
└→ switch_gap()
   → VerifyGap
```

## 核心实现原则

**Seed-paper grounded**：初始 Gap 只从 Seed Paper 中发现，保证研究目标有明确来源。

**Lightweight evaluation**：使用 Confidence 和 Priority 的离散等级及简短理由，不设计复杂评分公式。

**Verify before ideation**：Active Gap 必须经过 Related-work Verification 后才能进入 Solution Ideation。

**Bounded refinement**：Gap 最多收窄一次，避免无限修改研究问题。

**Reusable Gap Agenda**：候选 Gap 持续保留，在当前 Gap 失效时直接切换，提高 Agent 探索效率。



# 4. Solution Ideation

## 4.1 作用

Solution Ideation 负责在 **Active Gap 已通过验证** 后，生成多个候选解决方案，并从中选择一个最值得继续发展的方向，进一步展开为 Current Solution。

整体流程固定为：

```
Verified Active Gap
        ↓
FormSolution
        ↓
3 Candidate Solutions
        ↓
Select + Develop
        ↓
Current Solution
        ↓
Solution-level Verification
```

该模块只负责**形成方案**，不负责外部灵感检索、新颖性判断和实验设计。

------

## 4.2 模块输入

第一版输入保持精简：

```
SolutionIdeationInput
{
    active_gap


    seed_paper_context


    gap_verification_summary


    relevant_failure_memory
}
```

其中：

- `active_gap`：当前已经验证有效的 Research Gap；
- `seed_paper_context`：Seed Paper 的核心方法、主要组件和与 Gap 直接相关的信息；
- `gap_verification_summary`：说明外部相关工作验证后，当前仍然开放的问题边界；
- `relevant_failure_memory`：当前 Gap 下已经失败的 Solution 或 mechanism，用于避免重复生成。

------

# 4.3 FormSolution

第一次 LLM 调用一次性生成固定 **3 个 Candidate Solutions**。

结构为：

```
CandidateSolution
{
    id


    core_idea


    rationale


    status
}
```

初始状态：

```
status = candidate
```

------

## 4.4 Candidate Solution 生成要求

三个 Candidate 必须是：

> **核心机制上存在实质差异的三个解决方向。**

要求：

- 都直接针对 Active Gap；
- 不只是同一方案的表述变化；
- 不只是简单参数调整或边缘工程修改；
- 与已经失败的方案保持明显区别。

例如，更希望得到：

```
S1: uncertainty-triggered correction


S2: retrieval-based external state repair


S3: hierarchical checkpointing
```

而不是三个高度相似的 adaptive mechanism 变体。

------

# 4.5 Core Idea

`core_idea` 需要明确说明：

> **采用什么核心机制，以及它准备如何解决 Active Gap。**

应具体到“科研方法方向”的粒度，而不是泛泛描述。

例如应说明：

- 改变 Seed Paper 的哪个关键环节；
- 引入什么机制；
- 该机制承担什么作用。

但不需要展开成完整 Method Section。

------

# 4.6 Rationale

`rationale` 用于说明：

> **为什么这个 Solution 可能解决当前 Gap。**

它主要建立：

```
Active Gap
→ Core Mechanism
```

之间的逻辑联系。

保持为简短机制解释即可。

------

# 4.7 Candidate 阶段不主动使用外部灵感

FormSolution 只使用当前已有的信息：

```
Seed Paper
+
Active Gap
+
Gap Verification Summary
+
Failure Memory
```

不调用 Inspiration Retrieval。

这样保持清晰分工：

```
Solution Ideation
→ 基于当前知识形成初始方案


Inspiration Retrieval
→ Controller 判断需要扩展 Solution Space 时再调用
```

------

# 4.8 Select + Develop

第二次 LLM 调用同时完成：

```
SelectSolution
+
DevelopSolution
```

即：

```
3 Candidate Solutions
        ↓
Compare
        ↓
Select Best Candidate
        ↓
Develop into Current Solution
```

这样整个 Solution Ideation 第一版只需要两次主要 LLM 调用。

------

# 4.9 Candidate 选择标准

使用五个自然语言标准进行整体比较，不做数值打分。

### Gap Alignment

是否直接解决 Active Gap。

### Mechanism Clarity

是否能够清楚说明方案做什么、为什么可能有效。

### Research Feasibility

是否能够进一步形成实际可实现、可实验验证的方法。

### Non-triviality

是否超越简单参数调整或边缘工程优化。

### Failure Avoidance

是否避免重复 Research Memory 中已经失败的 Solution 或 mechanism。

输出：

```
SolutionSelection
{
    selected_solution_id


    selection_reason
}
```

其中 `selection_reason` 保持简短。

------

# 4.10 DevelopSolution

被选中的 Candidate 进一步展开为：

```
CurrentSolution
{
    id


    core_idea


    rationale


    components[]


    status


    novelty_status


    novelty_summary


    evidence_refs
}
```

其中 `core_idea` 和 `rationale` 可以在 Candidate 基础上进一步明确，但不需要生成过多实现细节。

------

# 4.11 Components

Current Solution 被拆分为 **2–5 个主要 Components**。

每个 Component：

```
Component
{
    id


    description


    status


    evidence_refs
}
```

`description` 只需要说明：

> **该组件是什么，以及它在整个 Solution 中承担什么作用。**

粒度要求是：

> 足够支持后续 Component-level Related-work Verification 和 Targeted Revision。

不要求生成：

- 详细算法；
- loss function；
- 超参数；
- 完整网络结构；
- 实现代码。

------

# 4.12 Solution 状态初始化

DevelopSolution 完成后：

```
current_solution.status = developed


current_solution.novelty_status = unchecked


current_solution.novelty_summary = null


components[*].status = unchecked


experiments = empty
```

同时 Candidate 状态更新为：

```
selected candidate → selected


other candidates → candidate
```

------

# 4.13 Current Solution 形成后立即验证

Current Solution 一旦完成 Develop，系统直接触发：

```
VERIFY_RELATED_WORK
```

即：

```
DevelopSolution
        ↓
Current Solution
        ↓
novelty_status = unchecked
        ↓
Solution-level Verification
```

不经过 Controller 再决定是否验证。

这样可以避免在一个已经被已有研究覆盖的方案上继续投入 Inspiration 或 Experiment Design。

------

# 4.14 Candidate Pool 的后续复用

三个 Candidate Solutions 会持续保留。

例如：

```
S1 = selected
S2 = candidate
S3 = candidate
```

如果后续：

```
S1 → core_overlap
```

则：

```
S1.status = rejected


current_solution = null
```

如果 S2、S3 仍可用：

```
SOLUTION_IDEATION
→ Select + Develop from remaining candidates
```

不重新生成三个 Candidate。

只有当所有 Candidate 都被拒绝时，才需要重新生成新的方案或由 Controller 进入 Inspiration Retrieval。

------

# 4.15 第一版代码接口

可以实现为：

```
SolutionIdeationModule
{
    form_solutions(
        active_gap,
        seed_paper_context,
        gap_verification_summary,
        failure_memory
    )


    select_and_develop(
        active_gap,
        seed_paper_context,
        candidate_solutions,
        failure_memory
    )
}
```

执行逻辑：

```
if candidate_solutions is empty:


    candidate_solutions =
        form_solutions(...)


current_solution =
    select_and_develop(...)


update Research State


→ VERIFY_RELATED_WORK
```

## 核心实现原则

**Multi-candidate exploration**：一次生成三个机制不同的候选方向，避免过早收敛到第一个方案。

**Two-call implementation**：采用“生成候选”和“选择 + 展开”两次主要 LLM 调用，控制实现成本。

**Moderate solution granularity**：Current Solution 只展开到 Core Idea、Rationale 和 2–5 个 Components，具体到能够被后续检索和修改即可。

**Verify after develop**：一个具体 Solution 一旦形成，立即进入 Related-work Verification，再决定是否继续扩展。



# 5. Related-work Verification

## 5.1 作用

Related-work Verification 负责利用已有相关论文验证当前研究状态中的两个核心对象：

```
Research Gap
Current Solution
```

分别回答：

- 当前 Gap 是否仍然真实存在；
- 当前 Solution 是否与已有工作发生实质重合，以及重合具体发生在哪里。

整体流程为：

```
Seed Paper
    ↓
Build Related Paper Pool
    ↓
Determine Verification Task
    ↓
Gap Verification
or
Solution Verification
    ↓
Paper Screening
    ↓
Structured Comparison
    ↓
Verification Result
    ↓
Update Research State
```

------

## 5.2 Related Paper Pool

第一版采用 **Seed-Paper-Centric Retrieval**。

对于每篇 Seed Paper，只在首次需要外部验证时获取一次相关论文：

```
Seed Paper
    ↓
Semantic Scholar Recommendations
    ↓
Top 20 Related Papers
    ↓
Local Cache
```

形成：

```
RelatedPaperPool
{
    seed_paper_id


    papers[]
}
```

每篇论文只保存必要信息：

```
PaperMetadata
{
    paper_id


    title
    abstract


    authors
    year
    venue


    url
}
```

后续 Gap Verification 和 Solution Verification 均复用该论文池。

------

## 5.3 Verification Task

Related-work Verification 只包含两个任务：

```
VERIFY_GAP
VERIFY_SOLUTION
```

模块根据当前 Research State 和最近一次修改自动确定需要执行哪个任务。

------

# 5.4 Gap Verification

Gap Verification 用于判断：

> 当前 Active Gap 是否仍然是一个尚未被已有研究充分解决的问题。

输入：

```
GapVerificationInput
{
    active_gap


    related_paper_pool
}
```

------

## 5.5 Gap Paper Screening

首先使用一次 LLM 调用：

```
Active Gap
+
Top 20 Related Papers
```

筛选最多：

```
≤5 relevant papers
```

重点寻找：

- 是否研究相同或高度相似的问题；
- 是否直接处理当前 limitation；
- 是否已经覆盖当前 Gap 的核心内容。

不要求必须选满 5 篇。

------

## 5.6 Gap Verification Result

第二次 LLM 调用对筛选论文进行比较，输出：

```
GapVerificationResult
{
    status


    summary


    covered_aspects
    remaining_gap


    related_papers


    evidence_level
}
```

其中：

```
status =
valid
partially_resolved
resolved
```

### `valid`

当前论文池中没有发现实质解决该 Gap 的工作。

后续：

```
→ SOLUTION_IDEATION
```

### `partially_resolved`

已有工作覆盖 Gap 的一部分，但仍存在明确未解决部分。

返回：

```
covered_aspects
remaining_gap
```

供 Gap Discovery and Management 执行：

```
RefineGap
→ VerifyGap again
```

### `resolved`

已有工作已经实质解决该 Gap 的核心问题。

后续：

```
Gap.status = resolved
→ SWITCH_GAP
```

------

# 5.7 Solution Verification

Solution Verification 用于判断：

> 当前 Solution 是否与已有研究发生实质重合，以及重合具体位于哪些部分。

它同时检查：

```
Core Idea
+
Components
```

因此不再单独划分其他验证层级。

输入：

```
SolutionVerificationInput
{
    active_gap


    current_solution
    {
        core_idea
        rationale
        components[]
    }


    verification_focus


    related_paper_pool
}
```

------

# 5.8 Verification Focus

Solution Verification 支持两种执行范围：

```
verification_focus =
full_solution
or
changed_units
```

### `full_solution`

用于：

- Current Solution 首次生成；
- Core Idea 被修改；
- 多个关键 Components 被修改；
- 整体机制发生明显变化。

此时同时检查：

```
Core Idea
C1
C2
...
```

------

### `changed_units`

用于 Targeted Revision 只修改局部内容的情况。

例如：

```
changed_units = [C2]
```

此时主要重新检查修改后的 C2，同时保留整个 Current Solution 作为上下文。

这只是验证范围控制，不是新的 Verification 类型。

------

# 5.9 Solution Paper Screening

使用一次 LLM 调用，根据：

```
Current Solution
+
Verification Focus
+
Top 20 Related Papers
```

筛选：

```
≤5 relevant papers
```

筛选重点包括：

- 是否解决相同或高度相关的问题；
- 是否采用相似的核心解决机制；
- 是否存在与 Current Solution 关键 Components 相似的设计；
- 这些相似内容属于标准/辅助技术共享，还是可能实质影响当前 Solution 的新颖性主张。

这里发现某个相似 Component **并不自动意味着存在 novelty overlap**，后续仍需结合该部分在 Current Solution 中承担的作用进行判断。

------

# 5.10 Structured Comparison

第二次 LLM 调用对关键论文进行结构化比较。

每篇论文形成：

```
ComparisonRecord
{
    paper_ref


    problem_relation


    mechanism_relation


    shared_elements


    novelty_relevant_overlaps


    key_difference
}
```

其中：

```
problem_relation =
same
related
different
mechanism_relation =
same
partial
different
unclear
```

### `shared_elements`

记录 Current Solution 与已有工作共享、但**不实质削弱当前新颖性主张**的内容，例如：

- 标准技术；
- 通用组件；
- 已知辅助机制。

### `novelty_relevant_overlaps`

记录已有工作中真正覆盖 Current Solution **novelty-bearing elements** 的部分。

该字段是后续区分：

```
distinct
partial_overlap
core_overlap
```

的主要依据。

------

# 5.11 Solution Verification Result

最终输出：

```
SolutionVerificationResult
{
    novelty_status


    summary


    shared_elements[]


    novelty_relevant_overlaps[]


    distinctive_elements[]


    key_differences


    related_papers


    evidence_level
}
```

其中：

```
novelty_status =
distinct
partial_overlap
core_overlap
```

------

## Distinct

已有工作可以与 Current Solution 共享：

- 标准技术；
- 通用组件；
- 相似任务设置；
- 辅助机制；

但没有实质覆盖 Current Solution 中**承担主要新颖性的关键机制或设计**。

即：

```
shared elements exist
BUT
no novelty-bearing element is substantially covered
→ distinct
```

此时：

```
current_solution.novelty_status = distinct
```

`distinct` 不表示与已有工作完全没有重合，而表示：

> 当前发现的重合不足以削弱 Current Solution 的核心新颖性主张。

------

## Partial Overlap

已有工作已经实质覆盖 Current Solution 中一个或多个**局部 novelty-bearing elements**，从而部分削弱当前方案的新颖性，但整体核心机制仍然具有实质区别。

需要进一步定位：

```
novelty_relevant_overlaps
[
    {
        component_id


        overlap_summary


        matched_paper_ref
    }
]
```

例如：

```
Component C2, which carries a local novelty claim,
is substantially covered by prior work,
while the overall core mechanism remains distinct.
```

随后可进入：

```
TARGETED_REVISION
```

或由 Controller 判断是否先：

```
SEEK_INSPIRATION
```

------

## Core Overlap

如果已有工作已经实质覆盖 Current Solution 的**核心解决机制**，使其主要新颖性主张无法成立，则：

```
novelty_status = core_overlap
```

典型情况是已有工作同时具有：

```
相同或高度相似的研究问题
+
实质相同的核心解决机制
```

此时 Current Solution 被拒绝，并返回 Solution Ideation 从剩余 Candidate Solutions 中选择其他方案。

------

## 三种状态的判断边界

```
仅共享标准/辅助技术
        ↓
     distinct


局部 novelty-bearing element 被覆盖
        ↓
 partial_overlap


核心 novelty mechanism 被覆盖
        ↓
  core_overlap
```

也就是说：

> **Overlap 本身不是问题，是否覆盖 novelty-bearing elements 才决定新颖性状态。**

------

# 5.12 Focused Re-verification

Targeted Revision 后，只重新验证实际发生变化的部分。

例如：

```
C2 revised
        ↓
Solution Verification
verification_focus = [C2]
```

重新验证时，需要判断修改后的 C2：

> 是否仍然存在会削弱当前新颖性主张的实质重合。

如果该局部 novelty-relevant overlap 已经消除，并且不存在其他 novelty-relevant overlap：

```
current_solution.novelty_status = distinct
```

如果仍然存在其他局部 novelty-bearing elements 被已有工作覆盖：

```
current_solution.novelty_status = partial_overlap
```

如果重新验证发现修改后的方案已经与已有工作核心机制实质重合：

```
current_solution.novelty_status = core_overlap
```

如果修改涉及整体核心机制，则重新执行：

```
verification_focus = full_solution
```

因此整体形成：

```
Full Solution Verification
        ↓
Novelty-relevant Overlap Localization
        ↓
Targeted Revision
        ↓
Focused Re-verification
```



# 5.13 Evidence Level

第一版主要使用论文的：

```
Title
+
Abstract
```

进行验证。

因此默认记录：

```
evidence_level = abstract_only
```

后续如果加入全文读取，可扩展为：

```
full_text
```

该字段只表示验证使用的证据粒度。

------

# 5.14 Research State 更新

Verification 完成后，只将当前决策需要的信息写回 Research State。

### Gap

保持不变：

```
valid
→ 保持 Active Gap


partially_resolved
→ RefineGap


resolved
→ Gap.status = resolved
```

### Solution

修改为：

```
distinct
→ novelty_status = distinct
partial_overlap
→ novelty_status = partial_overlap
→ novelty_summary 更新
core_overlap
→ novelty_status = core_overlap
→ reject Current Solution
```

其中：

- `distinct` 允许存在标准技术或辅助组件的共享；
- `partial_overlap` 只用于真正覆盖局部 novelty-bearing elements 的情况；
- `core_overlap` 表示核心新颖性机制被已有工作实质覆盖。

如果发生 `partial_overlap`，可同时更新对应 Component 的局部状态，方便 Targeted Revision 定位。

完整论文比较记录保存到 Research Memory。

------

# 5.15 缓存与成本

Related Paper Pool 在整个 Seed Paper trajectory 中只获取一次：

```
GetRelatedPapers(seed_paper)
→ +1 external search
```

后续所有：

```
Gap Verification
Solution Verification
Focused Re-verification
```

均复用本地 Cache：

```
→ +0 new searches
```

一次正常 Verification 主要包含：

```
LLM Call 1
→ Screen ≤5 Papers


LLM Call 2
→ Compare + Judge
```

------

# 5.16 第一版代码接口

可以实现为：

```
RelatedWorkVerificationModule
{
    get_related_paper_pool(
        seed_paper
    )


    determine_verification_task(
        research_state,
        latest_result
    )


    screen_papers(
        verification_target,
        verification_focus,
        related_paper_pool
    )


    verify_gap(
        active_gap,
        selected_papers
    )


    verify_solution(
        current_solution,
        verification_focus,
        selected_papers
    )


    update_state(
        verification_result
    )
}
```

## 核心实现原则

**Two-task verification**：只区分 Gap Verification 和 Solution Verification。

**Whole-solution reasoning**：Solution Verification 同时检查 Core Idea 和 Components，并在发生局部重合时定位具体 `overlap_items`。

**Focused re-verification**：局部修改后仅重点验证发生变化的部分，不重复完整验证。

**Seed-paper-centric retrieval**：整个 trajectory 复用同一个 Related Paper Pool，保持第一版实现轻量。

**Structured evidence output**：Verification 不只返回“是否新颖”，还明确指出已有工作覆盖了什么、当前方案仍然不同在哪里。



# 6. Inspiration Retrieval

## 6.1 作用

Inspiration Retrieval 负责在当前科研过程遇到瓶颈或需要进一步扩展方案时，主动寻找**可以迁移到当前问题中的外部科学机制或方法思想**。

它的核心目标不是查找“最相似的论文”，而是：

> **从当前 Research Need 出发，通过 Dynamic Entity Bridging 建立跨领域知识连接，并提取可迁移的科学机制。**

整体流程为：

```
SEEK_INSPIRATION
        ↓
Build Inspiration Need
        ↓
Dynamic Entity Bridging
        ↓
Select Bridge Entities
        ↓
Online Grounding
        ↓
Transferable Mechanism Extraction
        ↓
Inspiration Candidates
        ↓
TARGETED_REVISION
```

------

## 6.2 Inspiration Need

模块首先根据当前 Research State 和最近的 Verification / Failure 信息，构造一个简洁的：

```
InspirationNeed
{
    current_problem


    required_function


    current_mechanism


    rejected_mechanisms


    relevant_context
}
```

### `current_problem`

描述当前为什么需要新的灵感，例如：

```
The current triggering mechanism overlaps
with prior work and requires an alternative.
```

### `required_function`

描述新的机制真正需要完成什么功能，是整个 Inspiration Retrieval 的核心条件。

例如：

```
Determine when corrective intervention
should be activated based on system dynamics.
```

### `current_mechanism`

保存当前采用的机制，用于避免生成与现有方案高度相似的方向。

### `rejected_mechanisms`

来自 Research Memory，用于避免重复探索已经失败、被拒绝或发生明显 overlap 的机制。

### `relevant_context`

提供理解 Research Need 所需的少量上下文，例如：

```
Active Gap
Current Solution Summary
Relevant Verification Summary
Relevant Related Work
```

不直接输入完整 Research State 或完整历史。

------

# 6.3 Dynamic Entity Bridging

Dynamic Entity Bridging 是该模块的核心机制。

其基本思想为：

```
Current Research Need
        ↓
Functional Abstraction
        ↓
Bridge Entities
        ↓
External Scientific Knowledge
        ↓
Transferable Mechanism
```

LLM 不直接从当前问题生成新的 Solution，而是先寻找：

> **在其他科学问题中承担相似功能或解决相似结构性问题的科学概念和机制。**

这些中间概念称为 **Bridge Entities**。

------

## 6.4 Bridge Entity 类型

第一版主要允许以下类型：

```
scientific concept


mechanism


method


phenomenon


system principle
```

例如：

```
event-triggered control
optimal stopping
change-point detection
active sensing
selective computation
```

每个 Entity 保存：

```
BridgeEntity
{
    name


    type


    bridge_reason
}
```

其中 `bridge_reason` 简要说明：

> 为什么该 Entity 与当前 `required_function` 存在结构性联系。

------

# 6.5 Bridge Entity Generation

第一次 LLM 调用输入：

```
BridgeEntityInput
{
    inspiration_need


    active_gap


    current_solution_summary


    verification_summary


    rejected_mechanisms


    related_work_context
}
```

其中 `related_work_context` 复用 Related-work Verification 已经得到的少量论文摘要，不产生新的检索成本。

一次生成：

```
5 Candidate Bridge Entities
```

要求这些 Entity：

- 与 `required_function` 有明确联系；
- 尽量提供不同的机制方向；
- 避免重复 Current Solution；
- 避免重复 rejected mechanisms；
- 能够作为明确的学术检索概念。

------

# 6.6 Bridge Entity Selection

在同一次 LLM 调用中，从 5 个候选中选择最多 3 个用于后续 grounding。

主要依据三个标准：

### Structural Relevance

是否真正对应当前需要解决的功能问题。

### Novel Direction

是否能够提供不同于现有机制和失败方向的新知识连接。

### Searchability

是否是明确、稳定、适合进行文献搜索的科学概念。

最终：

```
5 proposed entities
→ select ≤3
```

不进行数值评分。

------

# 6.7 Online Grounding

选中的 Bridge Entities 被组合为一次外部文献检索。

例如：

```
event-triggered control
optimal stopping
change-point detection
```

形成一个联合查询：

```
E1 OR E2 OR E3
```

第一版使用 arXiv 作为默认 grounding backend：

```
Selected Bridge Entities
        ↓
One Combined Query
        ↓
arXiv Search
        ↓
Top 20 Papers
```

因此一次 `SEEK_INSPIRATION` 正常情况下只消耗：

```
1 external search
```

代码层保留统一 Search Backend 接口，使后续可以替换其他论文检索源。

------

# 6.8 Grounded Mechanism Extraction

第二次 LLM 调用使用：

```
GroundingInput
{
    inspiration_need


    selected_bridge_entities


    current_solution_summary


    rejected_mechanisms


    retrieved_papers
}
```

第一版主要读取论文的：

```
Title
+
Abstract
```

LLM 的任务不是判断哪篇论文与当前工作最相似，而是：

> **识别哪些外部论文中的机制能够迁移到当前 Research Need。**

------

# 6.9 Inspiration Candidate

最终最多返回 3 个：

```
InspirationCandidate
{
    bridge_entity


    paper_ref


    source_problem


    source_mechanism


    transferable_principle


    adaptation_hint


    grounding
}
```

### `source_problem`

原论文解决的科学问题。

### `source_mechanism`

原论文中实际采用的核心机制。

### `transferable_principle`

从原机制中抽象出的、可以迁移到当前问题的核心原则。

这是 Inspiration Candidate 中最重要的内容。

### `adaptation_hint`

简要说明：

> 该 transferable principle 可以如何用于当前科研问题。

它只提供修改方向，不直接修改 Current Solution。

------

# 6.10 Grounding Fallback

正常情况下：

```
grounding = arxiv
```

如果出现：

- Search Budget 不足；
- arXiv 请求失败；
- 搜索结果中没有可用机制；

则进入 LLM-only fallback。

流程：

```
Bridge Entities
        ↓
LLM Conceptual Transfer
        ↓
Inspiration Candidates
```

并明确标记：

```
grounding = none
```

表示该灵感来自模型参数知识，而不是外部论文证据。

------

# 6.11 模块输出与后续流程

如果成功获得可用 Inspiration：

```
InspirationResult
{
    status = found


    inspirations[]
}
```

则直接进入：

```
TARGETED_REVISION
```

由 Targeted Revision 决定如何将灵感具体融入 Current Solution。

如果没有形成合理的 Bridge Entity 或 transferable mechanism：

```
status = no_useful_inspiration
```

则返回 Scientific Controller，由其重新选择下一步科研动作。

------

# 6.12 Research Memory 更新

完整 Inspiration 过程保存到 Research Memory：

```
InspirationRecord
{
    inspiration_need


    proposed_entities


    selected_entities


    search_query


    grounding_status


    inspirations


    paper_refs
}
```

主要用于：

- 避免重复生成相同 Bridge Entities；
- 避免重复探索失败机制；
- 为后续 Targeted Revision 提供证据；
- 支持 Agent trajectory 分析。

Research State 中不保存完整 Inspiration 历史。

------

# 6.13 第一版代码接口

可以实现为：

```
InspirationRetrievalModule
{
    build_inspiration_need(
        research_state,
        latest_result,
        research_memory
    )


    generate_bridge_entities(
        inspiration_need,
        related_work_context
    )


    build_combined_query(
        selected_entities
    )


    search_grounding_papers(
        query,
        search_budget
    )


    extract_transferable_mechanisms(
        inspiration_need,
        selected_entities,
        papers
    )


    fallback_llm_inspiration(
        inspiration_need,
        selected_entities
    )
}
```

整体执行：

```
SEEK_INSPIRATION
        ↓
Build Inspiration Need
        ↓
LLM Call 1
Generate 5 Bridge Entities
Select ≤3
        ↓
Search Available?
     /       \
   Yes        No
    ↓          ↓
arXiv       LLM-only
Search
    ↓
LLM Call 2
Extract Transferable Mechanisms
     \        /
       ↓
≤3 Inspiration Candidates
        ↓
found ?
 /      \
Yes      No
 ↓        ↓
TARGETED Controller
REVISION
```

## 核心实现原则

**Research-need-conditioned**：整个检索过程围绕当前 `required_function` 展开，而不是围绕固定对象层级。

**Dynamic Entity Bridging**：通过 Bridge Entity 显式连接当前问题与其他科学领域中的机制。

**Mechanism-oriented retrieval**：目标是寻找可迁移机制，而不是寻找最相似论文。

**Lightweight grounding**：使用一次联合在线检索完成外部知识 grounding。

**Revision-oriented output**：输出 transferable principle 和 adaptation hint，真正的方案修改统一交给 Targeted Revision。



# 7. Targeted Revision

## 7.1 作用

Targeted Revision 负责根据 Related-work Verification 的重合证据或 Inspiration Retrieval 提供的新机制，对 Current Solution 进行**最小必要修改**。

核心目标是：

> **只修改真正受当前问题影响的内容，并保持修改后的 Solution 整体一致。**

整体流程为：

```
Revision Evidence / Inspiration
        ↓
Build Revision Need
        ↓
Identify Affected Units
        ↓
LLM Targeted Revision
        ↓
Lightweight Patch
        ↓
Deterministic Merge
        ↓
New Current Solution
        ↓
Focused Re-verification
```

------

## 7.2 触发条件

第一版主要处理两类情况。

### Partial Overlap

当 Related-work Verification 返回：

```
overlap_level = partial_overlap
```

并给出具体：

```
overlap_items
```

Targeted Revision 根据这些 overlap evidence 修改对应部分。

### Inspiration Available

当 Inspiration Retrieval 返回：

```
status = found
```

Targeted Revision 从 Inspiration Candidates 中选择一个最适合当前 Revision Need 的候选，并将其 transferable principle 适配到 Current Solution。

对于：

```
core_overlap
```

不进入 Targeted Revision，而是拒绝当前 Solution。

------

## 7.3 Revision Input

统一输入：

```
RevisionInput
{
    active_gap


    current_solution


    revision_reason


    verification_result


    inspirations


    relevant_failure_memory
}
```

其中 `verification_result` 和 `inspirations` 根据当前触发原因提供。

`relevant_failure_memory` 用于避免修改后重新回到已经失败或被否定的机制方向。

------

## 7.4 Revision Need

在修改前，先根据当前 evidence 构造：

```
RevisionNeed
{
    problem_to_fix


    affected_units


    constraints
}
```

### `problem_to_fix`

明确说明当前 Solution 存在什么具体问题。

例如：

```
C2 uses a triggering mechanism that
substantially overlaps prior work.
```

### `affected_units`

表示当前证据直接指出需要修改的主要对象，例如：

```
[C2]
```

### `constraints`

约束本次修改应保持的内容，例如：

```
- preserve the Active Gap
- preserve the core solution when possible
- preserve unaffected Components
- avoid rejected mechanisms
```

------

## 7.5 Minimum Necessary Change

Targeted Revision 遵循：

> **Minimum Necessary Change**

也就是只修改解决 `problem_to_fix` 所必需的内容。

例如：

```
Current Solution
├── Core Idea
├── C1
├── C2  ← overlap
├── C3
└── C4
```

如果问题只发生在 C2，则优先：

```
C2 → C2'
```

其他内容尽量保持不变。

------

## 7.6 Necessary Consistency Update

局部修改不能破坏整个 Solution 的内部一致性。

因此，LLM 除了修改 `affected_units`，还可以同步修改为了保持一致性而**确实需要联动变化的字段**。

例如：

```
affected_units = [C2]
```

但 C2 改变后，C3 的描述和 rationale 也需要同步更新，则最终可以得到：

```
changed_units =
[C2, C3, rationale]
```

因此：

- `affected_units`：修改前预计需要处理的主要对象；
- `changed_units`：修改后实际发生变化的全部对象。

------

## 7.7 Inspiration-guided Revision

如果存在多个 Inspiration Candidates，一次 Revision 只选择一个最适合当前问题的候选。

Targeted Revision 使用的不是外部方法本身，而是其中的：

```
transferable_principle
adaptation_hint
```

执行过程为：

```
Transferable Principle
        ↓
Current Research Context
        ↓
Adapted Mechanism
```

因此 Inspiration Retrieval 负责提供：

> 可以借鉴什么机制或原理。

Targeted Revision 负责：

> 如何把这个原理转化为当前 Solution 中合理的设计。

------

## 7.8 Core Idea 修改

Targeted Revision 允许修改：

```
core_idea
```

但只适用于 Current Solution 仍然具有保留价值，而新的 Inspiration 需要对核心机制进行较明显调整的情况。

如果修改会导致几乎整个 Solution 都需要重新设计，则可以直接返回：

```
status = unable_to_revise
```

由 Scientific Controller 重新规划。

------

## 7.9 一次 LLM 调用完成 Revision

第一版只使用一次主要 LLM 调用。

输入：

```
Active Gap
+
Current Solution
+
Revision Need
+
Verification Evidence
+
Selected Inspiration
+
Relevant Failure Memory
```

要求模型：

1. 解决 `problem_to_fix`；
2. 优先修改 `affected_units`；
3. 保持无关内容不变；
4. 必要时同步更新相关字段以维持整体一致性；
5. 返回所有实际发生变化的字段。

------

## 7.10 Revision Result

Targeted Revision 输出轻量 Patch：

```
RevisionResult
{
    status


    revised_core_idea


    revised_rationale


    revised_components[]


    changed_units[]


    revision_summary


    failure_reason
}
```

不需要修改的字段返回为空。

例如只修改 C2：

```
{
    status: revised,


    revised_core_idea: null,


    revised_rationale: "...",


    revised_components:
    [
        {
            id: C2,
            description: "..."
        }
    ],


    changed_units:
    [
        C2,
        rationale
    ],


    revision_summary: "..."
}
```

------

## 7.11 Deterministic Merge

Revision Result 不需要再经过一次 LLM 重写完整 Solution。

由代码直接执行：

```
apply_revision(
    current_solution,
    revision_result
)
```

将 Patch 合并到原来的结构化 Current Solution。

例如：

```
Old Solution
+
revised C2
+
revised rationale
        ↓
New Current Solution
```

未出现在 Patch 中的字段保持原值不变。

因此：

> Research State 中的 `current_solution` 始终保存最新的完整 Solution。

------

## 7.12 Research State 更新

Revision 成功后，由代码进行确定性更新。

例如：

```
C2.description = revised_description


current_solution.rationale = revised_rationale


C2.status = modified


current_solution.novelty_status = unchecked
```

如果修改了：

```
core_idea
```

则同样更新对应字段，并将：

```
novelty_status = unchecked
```

等待重新验证。

------

## 7.13 Experiment 状态更新

如果 Experiments 已经存在，则根据 `changed_units` 使用简单失效规则。

### Core Idea 修改

```
core_idea changed
→ all experiments = stale
```

### Component 修改

如果某个 Experiment 的：

```
what_it_validates
```

涉及该 Component，则：

```
experiment.status = stale
```

无关实验保持不变。

------

## 7.14 Revision Failure

Targeted Revision 允许：

```
status = unable_to_revise
```

例如：

- 当前 overlap 无法在保持方案核心逻辑的情况下局部解决；
- Inspiration 无法合理适配；
- 必要修改范围过大，已经不再属于 targeted revision。

同时返回：

```
failure_reason
```

随后：

```
→ Scientific Controller
```

由 Controller 重新决定下一步科研动作。

------

## 7.15 Focused Re-verification

Revision 成功后固定进入：

```
VERIFY_RELATED_WORK
```

验证范围由 `changed_units` 决定。

### 局部修改

例如：

```
changed_units = [C2, rationale]
```

则：

```
Solution Verification
verification_focus = [C2]
```

其中 rationale 只是整体上下文调整，不需要作为独立验证对象。

### 核心修改

如果：

```
core_idea changed
```

或多个主要 Components 同时发生变化，则：

```
verification_focus = full_solution
```

形成完整闭环：

```
Verification Evidence
        ↓
Targeted Revision
        ↓
Lightweight Patch
        ↓
Deterministic Merge
        ↓
Changed Units
        ↓
Focused Re-verification
```

------

## 7.16 Research Memory 更新

每次 Revision 保存：

```
RevisionRecord
{
    revision_reason


    affected_units


    selected_inspiration_ref


    changed_units


    revision_summary


    status
}
```

用于记录 Solution 的演化过程，并避免重复执行相同修改。

------

## 7.17 第一版代码接口

可以实现为：

```
TargetedRevisionModule
{
    build_revision_need(
        research_state,
        verification_result,
        inspiration_result,
        research_memory
    )


    select_inspiration(
        revision_need,
        inspiration_candidates
    )


    revise_solution(
        active_gap,
        current_solution,
        revision_need,
        selected_inspiration,
        failure_memory
    )


    apply_revision(
        current_solution,
        revision_result
    )


    update_stale_status(
        research_state,
        changed_units
    )
}
```

------

## 核心实现原则

**Evidence-localized**：根据具体 overlap 或当前科研瓶颈定位修改对象。

**Minimum necessary change**：只修改解决当前问题必需变化的内容。

**Consistency-aware**：允许同步调整必要的相关字段，保证修改后的 Solution 自洽。

**Lightweight patching**：LLM 只返回发生变化的字段。

**Deterministic merge**：由代码将 Patch 合并到原 Current Solution，不再额外调用 LLM 重写整个方案。

**Focused re-verification**：根据 `changed_units` 决定重新验证范围。



# 8. Experiment Design

## 8.1 作用

Experiment Design 负责将已经通过相关工作验证的 Current Solution 转化为一组**能够验证核心科研主张的关键实验**。

核心目标是：

> **用尽可能少但足够有区分力的实验，验证当前 Solution 是否真正解决 Active Gap，以及改进是否来自其关键机制。**

整体流程为：

```
Verified Current Solution
        ↓
Identify Key Claims
        ↓
Design 2–4 Key Experiments
        ↓
Check Claim Coverage
        ↓
Update Research State
```

------

## 8.2 触发条件

当 Current Solution 完成 Related-work Verification，并满足：

```
novelty_status = distinct
```

时，可以进入 Experiment Design。

这里 `distinct` 表示：

> Current Solution 可以与已有工作共享标准技术或辅助组件，但支撑其主要科研贡献的关键机制仍然具有实质区别。

随后 Scientific Controller 可以在：

```
SEEK_INSPIRATION
DESIGN_EXPERIMENT
```

之间选择。

------

## 8.3 模块输入

第一版输入：

```
ExperimentDesignInput
{
    active_gap


    current_solution


    seed_experiment_context


    verification_summary


    existing_experiments


    changed_units
}
```

### `active_gap`

明确实验最终需要验证的研究问题。

### `current_solution`

主要使用：

```
core_idea
rationale
components[]
```

确定当前方法需要验证哪些机制。

### `seed_experiment_context`

从 Seed Paper 中保留与实验直接相关的信息，例如：

```
tasks / datasets
evaluation metrics
major baselines
experimental setting
```

### `verification_summary`

提供当前 Solution 相对于已有工作的关键区别，使实验能够针对真正的 novelty-bearing elements 进行验证。

### `existing_experiments`

用于 Solution 修改后复用仍然有效的实验。

### `changed_units`

用于判断哪些已有实验需要重新设计。

------

# 8.4 Key Claim Identification

在生成实验时，LLM 先在同一次调用内部识别 Current Solution 最重要的可验证 Claim。

例如：

```
Claim 1:
The proposed solution mitigates the failure
identified by the Active Gap.


Claim 2:
The key mechanism C2 is responsible for
the observed improvement.


Claim 3:
The mechanism remains effective under
important variations of the target condition.
```

这些 Claim 只是 Experiment Design 的内部规划信息，不需要写入 Research State。

------

# 8.5 实验数量

第一版生成：

```
2–4 key experiments
```

实验数量根据 Solution 的复杂程度动态决定。

原则是：

> **使用最少数量的实验覆盖主要科研 Claim，而不是为了固定数量生成重复实验。**

------

# 8.6 Experiment 结构

每个实验保存：

```
Experiment
{
    id


    objective


    setup


    baselines


    metrics


    what_it_validates


    validation_criterion


    status
}
```

------

## `objective`

说明：

> 这个实验希望回答什么科研问题。

------

## `setup`

简要描述：

- 使用什么任务或数据；
- 主要实验条件；
- 如何进行关键对比。

保持研究计划级粒度，不生成实现代码和详细超参数。

------

## `baselines`

只保留真正必要的比较对象。

优先来自：

```
Seed Paper
+
Related-work Verification
```

典型包括：

- Seed Paper 原方法；
- 与当前 Gap 直接相关的已有方法；
- 必要的标准 baseline。

Experiment Design 本身不额外执行文献检索。

------

## `metrics`

优先复用 Seed Paper 已有评价指标。

如果 Current Solution 引入了 Seed Paper 原指标无法衡量的新能力，再加入必要的新指标。

------

## `what_it_validates`

明确当前实验验证哪个科研主张，例如：

```
core solution
Component C2
specific mechanism claim
```

该字段也是后续判断实验是否因 Solution 修改而失效的重要依据。

------

## `validation_criterion`

说明：

> 什么样的实验观察能够支持当前科研 Claim。

例如：

```
The proposed method should improve performance
under long-horizon state drift compared with
the seed method, while avoiding substantially
more correction operations.
```

不预测具体提升数值，也不虚构实验结果。

------

## `status`

第一版只需要：

```
ready
stale
```

------

# 8.7 Seed-paper-grounded Experiment Design

为了保证实验具有实际可执行性，第一版遵循：

> **如果 Seed Paper 的实验环境已经能够验证 Current Solution，则优先复用该实验设置。**

例如 Seed Paper 已经包含：

```
Dataset A
Dataset B


Metric X
Metric Y


Baseline M
```

则优先设计：

```
Seed Method M
vs.
Current Solution
```

在相同任务和指标上的对比。

只有 Current Solution 引入的新能力无法通过已有 setting 验证时，才补充新的实验条件。

------

# 8.8 实验覆盖要求

实验集合不采用固定的 Main / Ablation / Robustness 分类，但需要覆盖几个核心问题：

### Solution Effectiveness

是否真正缓解了 Active Gap 所对应的问题。

### Mechanism Validation

当前 Solution 中承担主要新颖性的关键机制是否确实发挥作用。

### Necessary Boundary Check

如果某个边界条件对方法有效性非常重要，则设计必要的验证实验。

并不是每种类型都必须单独对应一个实验，而是确保最终实验集合覆盖主要科研 Claim。

------

# 8.9 一次 LLM 调用完成设计

第一版使用一次主要 LLM 调用：

```
Active Gap
+
Current Solution
+
Seed Experimental Context
+
Verification Summary
+
Existing Experiments
```

完成：

```
Identify Key Claims
+
Design / Update Experiments
```

之后代码进行简单检查：

- 是否至少有一个实验验证整体 Solution；
- 关键 novelty-bearing mechanism 是否得到验证；
- 每个实验是否包含必要字段；
- 是否存在明显重复实验。

不额外调用 LLM Reviewer。

------

# 8.10 Solution 修改后的实验更新

Experiment Design 与 Targeted Revision 使用局部更新机制连接。

如果：

```
changed_units = [C2]
```

并且：

```
E2.what_it_validates
```

涉及 C2，则：

```
E2.status = stale
```

其他实验保持：

```
ready
```

下一次执行：

```
DESIGN_EXPERIMENT
```

时只重新设计 stale experiments。

例如：

```
E1 = ready → keep


E2 = stale → redesign


E3 = ready → keep
```

------

如果：

```
core_idea changed
```

则所有实验标记：

```
status = stale
```

因为整个实验逻辑可能需要重新检查。

------

# 8.11 Experiment Design Result

输出：

```
ExperimentDesignResult
{
    status


    experiments[]


    design_summary
}
```

正常情况下：

```
status = ready
```

生成或更新后的 Experiments 写入：

```
research_state.experiments
```

------

# 8.12 后续流程

实验设计完成后，Research State 已包含：

```
Verified Active Gap
+
Distinct Current Solution
+
Ready Experiments
```

此时系统可以检查 Commit 条件。

如果方案仍有明显提升空间且预算允许，Scientific Controller 仍可以继续：

```
SEEK_INSPIRATION
```

否则可以进入：

```
COMMIT
```

------

# 8.13 第一版代码接口

可以实现为：

```
ExperimentDesignModule
{
    build_experiment_context(
        research_state,
        seed_paper_context,
        verification_result
    )


    design_experiments(
        active_gap,
        current_solution,
        seed_experiment_context,
        verification_summary,
        existing_experiments,
        changed_units
    )


    check_coverage(
        current_solution,
        experiments
    )


    update_experiments(
        research_state,
        experiment_result
    )
}
```

## 核心实现原则

**Claim-oriented**：实验围绕 Current Solution 的关键科研主张设计。

**Seed-paper-grounded**：优先复用 Seed Paper 已有实验环境，提高可执行性和可比较性。

**Minimal but sufficient**：生成 2–4 个关键实验，用最少数量覆盖主要验证需求。

**No fabricated results**：只设计验证方式和支持条件，不预测未经实验验证的结果。

**Revision-aware**：Solution 修改后只重新设计受影响的 stale experiments，而不重新生成全部实验。



# 9. Research Memory

## 9.1 作用

Research Memory 用于保存单篇 Seed Paper 科研轨迹中已经获得的外部知识、验证证据、失败经验和关键演化记录。

它与 Research State 的分工为：

```
Research State
→ 保存当前最新、有效的科研状态


Research Memory
→ 保存形成当前状态所依赖的知识与历史经验
```

Research Memory 主要服务三个目标：

- 复用已经获取的论文和验证结果；
- 避免重复探索已经失败的方向；
- 为后续模块提供与当前任务相关的少量历史信息。

第一版只在**单篇 Seed Paper trajectory 内使用**，不同 Seed Paper 之间不共享 Memory。

------

## 9.2 整体结构

第一版 Research Memory 包含：

```
ResearchMemory
{
    paper_memory


    verification_memory


    inspiration_memory


    revision_memory


    failure_memory


    decision_log
}
```

各类 Memory 采用结构化记录，不引入额外的 Memory Agent 或复杂检索系统。

------

# 9.3 Paper Memory

Paper Memory 负责缓存 Agent 已经实际获取和使用过的论文。

```
PaperMemory
{
    papers


    related_paper_pool


    inspiration_search_records[]
}
```

单篇论文记录：

```
PaperRecord
{
    paper_id


    title
    abstract


    authors
    year
    venue


    source
    url
}
```

第一版只保存：

```
metadata + abstract
```

不保存 PDF 或全文。

------

## Related Paper Pool

Related-work Verification 获取的 Seed Paper 相关论文池直接缓存到：

```
related_paper_pool
```

后续：

```
Gap Verification
Solution Verification
Focused Re-verification
```

均复用该缓存，不重复外部检索。

------

## Inspiration Search Record

Inspiration Retrieval 的搜索结果保存为：

```
InspirationSearchRecord
{
    query


    bridge_entities


    paper_refs[]
}
```

如果后续再次使用相同或高度相近的 Bridge Entity，可以优先复用已有搜索结果。

第一版使用简单的 Entity / Query 匹配即可。

------

# 9.4 Verification Memory

每次 Related-work Verification 的完整结果保存为：

```
VerificationRecord
{
    id


    task


    target_ref


    verification_focus


    result


    paper_refs[]


    created_at_step
}
```

其中：

```
task =
VERIFY_GAP
VERIFY_SOLUTION
```

------

## Gap Verification Record

主要保存：

```
status


summary


covered_aspects


remaining_gap


paper_refs
```

用于后续 RefineGap 和 Gap 管理。

------

## Solution Verification Record

主要保存：

```
novelty_status


shared_elements


novelty_relevant_overlaps


distinctive_elements


key_differences


paper_refs
```

其中：

### `shared_elements`

表示与已有工作共享，但**不实质削弱当前 Solution 核心新颖性**的标准技术、辅助组件或普通设计元素。

这些内容不会自动进入 Failure Memory。

### `novelty_relevant_overlaps`

表示真正覆盖当前 Solution novelty-bearing elements 的已有工作。

这部分可用于：

```
Targeted Revision
Inspiration Retrieval
Failure Memory
```

### `distinctive_elements`

记录 Current Solution 相对于已有工作的关键区别，可供后续 Controller 和 Finalizer 使用。

------

# 9.5 Inspiration Memory

每次 Inspiration Retrieval 保存：

```
InspirationRecord
{
    id


    inspiration_need


    proposed_entities[]


    selected_entities[]


    search_query


    inspirations[]


    grounding_status


    paper_refs[]


    created_at_step
}
```

主要用于：

- 避免重复生成相同 Bridge Entity；
- 为 Targeted Revision 提供 Inspiration Candidates；
- 保存尚未被使用的 Inspiration，供后续继续利用。

一次 Inspiration Retrieval 最多产生多个 Candidate，但 Targeted Revision 一次只使用其中一个，其余候选继续保留。

------

# 9.6 Revision Memory

每次 Targeted Revision 保存：

```
RevisionRecord
{
    id


    revision_reason


    affected_units


    selected_inspiration_ref


    changed_units


    revision_summary


    status


    failure_reason
}
```

它主要记录：

```
为什么修改
→ 修改了什么
→ 修改是否成功
```

Research Memory 不需要保存每个历史版本的完整 Current Solution。

最新完整 Solution 始终保存在 Research State 中。

------

# 9.7 Failure Memory

Failure Memory 只保存**真正导致科研方向失效或不值得重复探索的内容**。

结构：

```
FailureRecord
{
    type


    object_ref


    summary


    reason


    evidence_refs
}
```

支持：

```
type =
rejected_gap
rejected_solution
rejected_mechanism
failed_revision
```

------

## Rejected Gap

记录被判定不值得继续探索或已经失效的 Gap。

------

## Rejected Solution

例如：

```
Solution S1
→ core_overlap
```

则记录当前 Solution 被拒绝的核心原因及相关 evidence。

后续 Solution Ideation 可以避免重新生成高度相似的方案。

------

## Rejected Mechanism

只记录真正发生：

```
novelty-relevant overlap
```

并因此不值得继续采用的机制。

例如：

```
uncertainty-threshold triggering
```

如果它直接导致 `partial_overlap` 或 `core_overlap`，可以写入 Failure Memory。

普通的：

```
shared_elements
```

不作为失败方向记录。

------

## Failed Revision

如果：

```
Targeted Revision
→ unable_to_revise
```

则保存失败原因，避免再次执行相同的无效修改。

------

# 9.8 Controller Decision Log

保留一个轻量的决策轨迹：

```
DecisionRecord
{
    step


    action


    reason


    result_summary
}
```

例如：

```
action:
SEEK_INSPIRATION


reason:
The current novelty-relevant overlap cannot
be resolved using existing information.


result_summary:
Three inspiration candidates were found.
```

Decision Log 主要用于：

- debugging；
- trajectory analysis；
- case study；
- Agent 行为分析。

它不会作为每次 Controller 的完整输入。

------

# 9.9 Memory Retrieval

第一版采用结构化检索，不使用：

```
vector database
embedding retrieval
memory agent
```

主要接口例如：

```
get_related_papers(seed_id)


get_latest_verification(target_ref)


get_relevant_failures(active_gap_id)


get_relevant_inspirations()


get_revision_history(solution_id)
```

根据：

```
record type
gap / solution reference
recency
```

进行简单过滤即可。

------

# 9.10 各模块只读取 Relevant Memory

Research Memory 不会整体输入 LLM。

不同模块只读取当前任务真正需要的信息。

### Solution Ideation

读取：

```
rejected solutions
rejected mechanisms
```

### Related-work Verification

读取：

```
related paper pool
previous verification results
```

### Inspiration Retrieval

读取：

```
previous bridge entities
rejected mechanisms
relevant related-work context
```

### Targeted Revision

读取：

```
latest verification evidence
selected inspiration
relevant failed revisions
```

### Scientific Controller

主要读取：

```
last_action
last_result
relevant_failure_memory
```

这样可以控制上下文长度，并减少无关历史信息对推理的干扰。

------

# 9.11 Memory 写入时机

各模块执行完成后，由代码按照确定性规则写入对应 Memory。

例如：

```
Related-work Verification
→ save VerificationRecord
Inspiration Retrieval
→ save InspirationRecord
Targeted Revision
→ save RevisionRecord
core_overlap / rejected mechanism
→ save FailureRecord
Scientific Controller
→ append DecisionRecord
```

不需要额外的 LLM 判断“什么内容值得记住”。

------

# 9.12 第一版代码接口

可以实现为：

```
ResearchMemory
{
    save_paper()


    save_verification()


    save_inspiration()


    save_revision()


    save_failure()


    save_decision()




    get_related_papers()


    get_latest_verification()


    get_relevant_failures()


    get_relevant_inspirations()


    get_revision_history()
}
```

## 核心实现原则

**Trajectory-local**：第一版每篇 Seed Paper 独立维护 Research Memory。

**Structured and lightweight**：采用结构化记录和简单过滤，不引入复杂 Memory Retrieval。

**Evidence-preserving**：保存 Verification、Inspiration 和 Revision 的完整关键依据。

**Failure-aware**：只记录真正导致科研方向失效的 novelty-relevant failures，不把普通技术共享当作失败。

**Context-efficient**：各模块只读取与当前任务直接相关的 Relevant Memory，而不是重新加载完整历史。



# 10. Commit + Finalizer

## 10.1 作用

Commit + Finalizer 负责在科研探索达到可接受完成度后，判断当前 Research State 是否具备结束条件，并将最终结构化状态整理为一份完整、可读的科研方案。

二者职责明确区分：

```
Commit
→ 判断当前科研方案是否可以结束


Finalizer
→ 将已经确定的科研内容整理为最终输出
```

整体流程为：

```
Research State
      ↓
Check Commit Gate
      ↓
commit_allowed ?
    /        \
   No         Yes
   ↓           ↓
Continue     Scientific Controller
Research          ↓
             COMMIT ?
             /    \
           No      Yes
           ↓        ↓
       Continue   Freeze State
       Research      ↓
                Commit Snapshot
                     ↓
                  Finalizer
                     ↓
           Final Research Proposal
```

------

## 10.2 Commit Gate

Commit Gate 使用确定性规则检查当前科研方案是否达到最低完成要求，不调用 LLM。

需要同时满足：

```
CommitGate
{
    active_gap_verified


    solution_ready


    novelty_verified


    experiments_ready


    no_pending_revision
}
```

------

### Active Gap Verified

当前必须存在有效的 Active Gap，并且最近一次 Gap Verification 结果为：

```
valid
```

即当前 Research Gap 经过相关工作检查后仍然成立。

------

### Solution Ready

必须存在完整的：

```
current_solution
```

并且：

```
current_solution.status = developed
```

至少包含：

```
core_idea
rationale
components[]
```

------

### Novelty Verified

Current Solution 必须满足：

```
current_solution.novelty_status = distinct
```

这里 `distinct` 表示：

> Current Solution 可以与已有工作共享标准技术、辅助组件或非关键元素，但承担主要新颖性的核心机制没有被已有工作实质覆盖。

因此 Commit Gate 不要求与已有工作完全没有重合。

以下状态均不能 Commit：

```
unchecked
partial_overlap
core_overlap
```

------

### Experiments Ready

必须已经完成 Experiment Design，并且当前实验均处于：

```
status = ready
```

第一版通常要求存在：

```
2–4 key experiments
```

从而保证最终科研方案具有明确的验证路径。

------

### No Pending Revision

Research State 中不能存在尚未处理的修改或失效状态。

例如不能存在：

```
Component.status = modified
Component.status = stale


Experiment.status = stale


novelty_status = unchecked
```

因此 Commit 的对象必须是当前已经稳定的 Research State。

------

## 10.3 Commit Gate 输出

输出保持简单：

```
CommitCheck
{
    commit_allowed


    unmet_conditions[]
}
```

例如：

```
{
    commit_allowed: false,


    unmet_conditions:
    [
        "Solution has not completed re-verification.",
        "Experiment E2 is stale."
    ]
}
```

如果：

```
unmet_conditions = []
```

则：

```
commit_allowed = true
```

------

## 10.4 与 Scientific Controller 的交互

当：

```
commit_allowed = true
```

时，`COMMIT` 才进入 Controller 的合法动作集合。

例如：

```
allowed_actions =
[
    COMMIT,
    SEEK_INSPIRATION
]
```

Scientific Controller 根据：

- 当前 Solution 是否已经足够成熟；
- 是否仍存在明确且具体的提升空间；
- 剩余 Step / Search Budget；

决定是否立即结束。

因此：

> **Commit Gate 决定“是否可以结束”，Scientific Controller 决定“现在是否值得结束”。**

------

## 10.5 Commit Snapshot

当 Scientific Controller 选择：

```
COMMIT
```

后，由代码冻结当前最终状态，形成：

```
CommitSnapshot
{
    active_gap


    current_solution


    experiments


    latest_gap_verification


    latest_solution_verification


    relevant_evidence_refs
}
```

Commit Snapshot 表示：

> Agent 最终确认采用、并且已经通过当前验证流程的科研状态。

之后 Finalizer 只读取 Commit Snapshot 和必要的 Research Memory，不再修改 Research State。

------

# 10.6 Finalizer

Finalizer 负责将结构化的 Commit Snapshot 转换为连贯的人类可读科研方案。

其作用为：

```
Structured Research State
        ↓
Finalizer
        ↓
Final Research Proposal
```

Finalizer 只负责内容组织和语言表达，不继续进行科研探索。

它不能：

- 新增未经验证的核心机制；
- 修改 Current Solution；
- 增加新的 Method Component；
- 虚构实验结果；
- 扩大未经 Verification 支持的新颖性主张。

------

## 10.7 Finalizer 输入

第一版输入：

```
FinalizerInput
{
    commit_snapshot


    seed_paper_summary


    novelty_context


    evidence_refs
}
```

其中 `novelty_context` 主要来自最新的 Solution Verification：

```
shared_elements


distinctive_elements


key_differences


related_paper_refs
```

不需要输入完整 Research Memory。

------

# 10.8 Final Research Proposal

Finalizer 最终生成：

```
FinalResearchProposal
{
    research_gap


    proposed_solution


    solution_rationale


    method_components


    novelty_positioning


    experimental_plan


    expected_contributions
}
```

------

## Research Gap

根据最终 Active Gap 整理：

> 当前研究场景中仍然存在什么尚未解决的问题，以及该问题为什么值得进一步研究。

不重新创造新的 Gap。

------

## Proposed Solution

根据：

```
current_solution.core_idea
```

整理整体解决方案。

重点明确：

- 方法改变什么；
- 引入什么核心机制；
- 如何针对 Active Gap。

------

## Solution Rationale

根据：

```
current_solution.rationale
```

解释：

> 为什么当前 Solution 有能力解决 Active Gap。

------

## Method Components

将 Current Solution 中最终确定的：

```
2–5 Components
```

组织成连贯的方法结构。

Finalizer 可以改善语言表达，但不能改变 Components 的实际含义。

------

# 10.9 Novelty Positioning

Novelty Positioning 基于：

```
shared_elements


distinctive_elements


key_differences
```

组织当前 Solution 与已有工作的关系。

基本结构为：

```
What is shared
+
What is different
+
Why the difference matters
```

因此 Finalizer 不声称：

```
The proposed method has no overlap with prior work.
```

而应准确体现：

> Current Solution 可以复用已有技术，但其主要 novelty-bearing mechanism 与现有工作仍然具有实质区别。

------

# 10.10 Experimental Plan

Finalizer 将 Research State 中已经 `ready` 的 Experiments 整理成实验计划。

每个实验主要包括：

```
objective
setup
baselines
metrics
what_it_validates
validation_criterion
```

不生成未经执行的实验结果，也不预测具体性能提升数值。

------

# 10.11 Expected Contributions

Finalizer 根据：

```
Active Gap
+
Current Solution
+
Distinctive Elements
+
Experimental Plan
```

总结若干预期贡献。

通常组织为：

```
2–3 expected contributions
```

这些贡献必须能够从最终 Research State 中直接推导，不能临时增加新的科研主张。

由于实验尚未实际执行，应使用：

```
expected contribution
potential contribution
```

而不是将其表述为已经被实验证明的结论。

------

# 10.12 Finalizer 执行方式

第一版只使用一次 LLM 调用：

```
Commit Snapshot
+
Relevant Verification Context
        ↓
LLM Finalizer
        ↓
Final Research Proposal
```

不需要额外设计：

```
Draft
→ Critique
→ Rewrite
```

因为在进入 Finalizer 前，科研内容已经由前面的 Agent trajectory 确定。

------

# 10.13 Finalizer 为只读模块

Finalizer 不回写 Research State。

执行关系为：

```
Research State
        ↓
Commit Snapshot
        ↓
Finalizer
        ↓
Final Proposal
```

而不存在：

```
Finalizer
→ Modify Research State
```

从而保证最终输出与真正经过 Verification 的 Solution 保持一致。

------

# 10.14 Budget Exhaustion

如果预算耗尽且：

```
commit_allowed = true
```

则：

```
→ COMMIT
→ Finalizer
```

如果预算耗尽但：

```
commit_allowed = false
```

则停止当前 trajectory，并返回：

```
best_current_research_state


unresolved_issues
```

例如：

```
- Solution verification remains incomplete.
- Experiment E2 is stale.
```

不将未完成状态包装成完整科研方案。

------

# 10.15 第一版代码接口

可以实现为：

```
CommitModule
{
    check_commit_gate(
        research_state,
        research_memory
    )


    create_commit_snapshot(
        research_state,
        research_memory
    )
}
```

以及：

```
Finalizer
{
    build_finalization_context(
        commit_snapshot,
        seed_paper_summary,
        research_memory
    )


    generate_final_proposal(
        finalization_context
    )
}
```

整体执行：

```
Research State
      ↓
check_commit_gate()
      ↓
commit_allowed
      ↓
Scientific Controller
      ↓
COMMIT
      ↓
create_commit_snapshot()
      ↓
Finalizer
      ↓
Final Research Proposal
```

## 核心实现原则

**Hard-gated completion**：只有 Gap、Solution、Novelty 和 Experiments 均达到稳定状态时才允许 Commit。

**Controller-decided stopping**：满足 Commit Gate 后仍由 Scientific Controller 判断是否值得继续探索。

**State freezing**：使用 Commit Snapshot 固定最终经过验证的科研状态。

**Read-only finalization**：Finalizer 只整理已有科研内容，不再修改或扩展 Solution。

**Evidence-aligned output**：Novelty 和 Contribution 必须与 Related-work Verification 和当前 Research State 保持一致。