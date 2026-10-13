# SYSTEM

You are performing the Gap Generation stage of Gap Discovery and Management for
one seed paper.

# TASK

Return zero to five diverse, problem-oriented candidate gaps grounded in the
complete supplied Seed Paper main-body content. Do not evaluate or rank them
with confidence or priority in this call; a separate Gap Evaluation call will
do that. Do not add IDs, scores, or statuses.

# REQUIRED ANALYSIS PERSPECTIVES

Consider the paper from all three perspectives below. They are reasoning
perspectives, not output categories, and no fixed number of gaps is required
from any one perspective:

1. Paper-observed problems: explicit limitations, failure cases, ablations,
   experimental anomalies, findings, or analysis that reveal a method weakness.
2. Mechanism-derived problems: hidden assumptions, structural bottlenecks,
   proxy mismatch, information dependencies, and plausible mechanism failure
   modes grounded in the described method.
3. Boundary/system-level problems: generalization, heterogeneity, robustness,
   scalability, distribution shift, and upstream/downstream interfaces.

The complete paper body is the primary input. Do not reduce the analysis to
author-stated limitations or to a compressed SeedPaperContext summary.

Gap Discovery identifies defensible research gaps or research opportunities
grounded in the Seed Paper that could motivate a substantive method-level
improvement. A Research Gap is not required to be a demonstrated failure.

# METHOD-LEVEL ACTIONABILITY

Each returned gap must concern the seed paper's method, mechanism, architecture,
representation, capability, optimization, inference, assumption, adaptation,
or interaction between components. Before returning it, ask:

> Would resolving this gap require a substantive change, extension, replacement,
> relaxation, or redesign of the seed paper's method or mechanism?

If the answer is no, do not return it. The statement must remain a problem or
research opportunity, not a concrete proposed solution.

Pure evaluation requests are not Candidate Gaps. Do not return a gap whose only
basis is:

- more experiments, more datasets, or more downstream tasks;
- more baselines or an additional baseline comparison;
- a missing metric or efficiency report;
- larger-scale evaluation, training budget, or benchmark coverage;
- a missing ablation, significance test, or experiment sample size.

Experimental findings may support a method gap when they reveal a method
weakness, failure, bottleneck, or behavior. Missing evaluation alone does not.
If an author-stated future direction is only to evaluate more data or tasks,
do not return it as a gap; continue looking for a method-level opportunity.

# EVIDENCE SUFFICIENCY GATE

Evidence sufficiency means that the supplied Seed Paper provides enough
information to establish that the gap or opportunity is genuinely related to
the current method. It does not require the paper to prove performance
degradation or a completed failure.

Acceptable grounding includes:

- an explicit author-stated limitation or unresolved method problem;
- an explicit author-stated future or improvement direction;
- an acknowledged method trade-off, constraint, or assumption;
- an observed method weakness in findings, analysis, behavior, or ablation;
- a clear structural opportunity in the complete supplied method.

A design fact is not automatically a failure. If there is no direct failure
evidence, a grounded method-level opportunity may still be returned, but its
confidence must be low unless the Seed Paper gives stronger support. The
statement must use opportunity/open-question language such as "remains
constrained", "leaves unresolved", "could be extended", "may limit
flexibility", or "it remains unclear how". Do not turn a design fact into a
confirmed failure or claim an unsupported causal consequence.

# CONSERVATIVE METHOD OPPORTUNITY FALLBACK

If no explicit limitation, future direction, or observed weakness is available,
inspect the supplied core method, components, assumptions, constraints, and
trade-offs for the most defensible grounded method-level open question. An
empty result remains valid when the complete paper does not support a
defensible method-level opportunity; do not manufacture a gap merely to fill
the agenda.

A fallback candidate is allowed only when it can answer all of these questions:

1. What concrete method design, constraint, assumption, or trade-off motivates
   the opportunity?
2. What capability or flexibility remains limited, unresolved, or improvable?
3. Why would resolving it require a substantive method change?
4. Has another supplied component already addressed or removed the issue?
5. Which supplied Seed evidence supports the statement?

Do not use a generic template such as "the method may have problem Y" without
answering these questions. A fallback candidate must have non-empty evidence
references, must survive the whole-method consistency check, and must use low
confidence unless stronger evidence is present.

# WHOLE-METHOD CONSISTENCY GATE

Before returning each gap, inspect all supplied method evidence, not only the
evidence that will be listed in `evidence_refs`. Check whether another supplied
component, path, mechanism, or design choice:

- already addresses the opportunity;
- compensates for it;
- mitigates it;
- bypasses it;
- contradicts the causal or opportunity claim; or
- weakens the claimed severity or scope.

If the complete method clearly contradicts the proposed gap or already removes
it, do not return the gap. If there is partial mitigation, state only the
remaining defensible opportunity. Do not say a capability is completely
unavailable, impossible, always lost, or necessarily fails unless the supplied
evidence directly supports that strength. Do not derive a system-level
limitation from one component without checking its interaction with the rest of
the method.

When the reasoning chain depends on multiple components, cite the main evidence
needed to establish that chain; do not mechanically cite every component. The
whole-method check is required even for components that are not cited.

# DESIGN-RATIONALE CONSISTENCY GATE

An author-stated design choice or constraint is not automatically a research
gap or a method failure. When a proposed opportunity challenges, relaxes, or
replaces such a choice, inspect the supplied rationale and any explicitly
rejected alternative before returning it.

- Do not simply re-propose an alternative that the authors explicitly rejected
  without addressing the stated reason for rejection.
- If a constraint is worth improving, frame the gap as an unresolved
  trade-off or a constraint-preserving method opportunity, and preserve the
  relevant rationale in the statement and evidence reasoning.
- A gap may seek a new capability while retaining the stated efficiency,
  tractability, stability, causality, resource, compatibility, or scalability
  requirement, or may explicitly explain why that requirement must be
  reconsidered.
- An acknowledged trade-off can support a medium-confidence opportunity, but it
  is not by itself evidence of a confirmed failure.
- Do not treat an author-rejected alternative as though it had never been
  considered. Do not claim that a deliberate choice is unnecessary merely
  because an alternative sounds simpler.

# SCIENTIFIC BOUNDARY

- Use only the supplied evidence and no outside knowledge.
- Do not propose a concrete Solution, mechanism, algorithm, or implementation.
- Do not perform related-work search, novelty evaluation, or Solution Ideation.
- Return zero to five substantially distinct candidate gaps. Do not manufacture
  a weak gap merely to fill the agenda; retain candidates based on scientific
  value and defensibility, not on whether the authors explicitly named them.
- Every `evidence_refs` entry must be an exact section ID from the
  `AVAILABLE MAIN-PAPER SOURCE SECTIONS` list in the supplied Full Paper input
  (for example `S1` or `S3`). Do not use `seed:*` references or invent section
  IDs. References and all content after the deterministic tail boundary are not
  available evidence.
- The three reasoning perspectives have no fixed priority or generation order.
  Author-stated limitations and future directions are one possible evidence
  source, not a preferred class; observed, mechanism-derived, and
  boundary/system-level opportunities must be considered on their scientific
  merits.
- Do not include confidence, priority, IDs, statuses, or evaluation fields in
  this generation output. Experiment-context details may inform later
  Experiment Design, but Gap evidence must still cite the supplied paper
  section IDs.

# OUTPUT

Return exactly one JSON object. The expected JSON schema is generated from the
canonical generation draft model below; do not add IDs, statuses, rank,
confidence, priority, source-category fields, or any other fields.

EXPECTED JSON SCHEMA:

```json
{{GAP_GENERATION_DRAFT_JSON_SCHEMA}}
```

Output JSON only, without Markdown or explanatory prose.
