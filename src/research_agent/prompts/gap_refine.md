# SYSTEM

You are performing the bounded RefineGap operation.

# TASK

Narrow or clarify the supplied active gap using the supplied partially resolved
verification result.  Preserve the core research problem and focus only on the
remaining gap.

# RESIDUAL-GAP OUTPUT

The refined statement must describe only the remaining unresolved method-level
problem.  Use the verification's `remaining_gap` as the primary source for
what still needs to be stated.  Do not repeat covered aspects, retrieved-paper
contributions, or benchmark/evaluation coverage in the refined statement.
Covered aspects belong in verification evidence and memory, not in the core of
the new active-gap wording.  Do not write a statement of the form "prior work
covers X but Y remains"; return the residual Y directly, without restating X.
The refined statement must stand on its own as a defensible research problem.

# METHOD-LEVEL GAP REQUIREMENT

The refined gap must remain a limitation of the seed paper's method, mechanism,
architecture, representation, capability, adaptation, optimization, or inference
behavior.  Resolving it must require a substantive method/mechanism change,
extension, replacement, or redesign.  Do not refine a method gap into a missing
dataset test, baseline comparison, metric report, ablation, or larger-scale
evaluation.

# EVIDENCE SUFFICIENCY AND WHOLE-METHOD CONSISTENCY

Use the supplied active gap, verification result, and the Seed Paper evidence
available through them.  A design property is not automatically a limitation;
the refined statement may remain an author-grounded opportunity or open
methodological question even without direct failure evidence.  Keep it supported
by evidence rather than merely relabeling an unrelated risk as a gap.

Before returning the refinement, check the complete supplied method evidence,
not only the evidence that was originally cited.  Look for components, paths,
or mechanisms that compensate for, mitigate, bypass, or already address the
proposed limitation, as well as evidence that contradicts the causal claim or
weakens its severity.  If mitigation exists, state only the remaining
defensible issue.  Do not introduce absolute claims such as completely
unavailable, impossible, always lost, or necessarily fails.  Do not infer a
system-level limitation from one local component without checking its
interaction with the rest of the method.

The refinement must not introduce a new unsupported causal claim, and it must
not conflict with the Seed Paper's own method description.  If the evidence
supports an opportunity or only structural reasoning, use calibrated wording
such as "may limit", "may constrain", "could restrict", "could be extended",
or "leaves unresolved whether" while still requiring a concrete evidence basis.

# DESIGN-RATIONALE CONSISTENCY

Preserve any author-stated method constraint, design rationale, acknowledged
trade-off, or explicitly rejected alternative relevant to the active gap. Do
not narrow the gap into simply reusing a rejected alternative without handling
the reason it was rejected. If the remaining opportunity challenges a deliberate
choice, describe the unresolved trade-off or the constraint-preserving change
that remains needed. Do not introduce a new unsupported causal claim or imply
that an author-rejected alternative was never considered.

# SCIENTIFIC BOUNDARY

- Use only the supplied active gap and verification result.
- Keep the original problem identity; do not create a new research question.
- Do not introduce a concrete solution, method, mechanism, or novelty claim.
- Return only a refined statement and a concise reason for the refinement.
- Do not return an ID, confidence, priority, status, or evidence references.

# OUTPUT

Return exactly one JSON object matching the generated schema below.  Output JSON
only, without Markdown or explanatory prose.

EXPECTED JSON SCHEMA:

```json
{{GAP_REFINEMENT_DRAFT_JSON_SCHEMA}}
```
