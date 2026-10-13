# SYSTEM

You are performing SelectSolution and DevelopSolution in one structured call.

# TASK

Choose one ID from the supplied available candidate solutions and develop it
into a concise Current Solution.  First critique each available candidate for
scientific weaknesses and failure modes, then compare candidates using Gap Alignment,
Mechanism Clarity, Research Feasibility, Non-triviality, and Failure Avoidance;
use qualitative reasoning rather than numerical scores.

The developed solution must contain two to five component descriptions.  Each
component must be concrete enough for later related-work verification and local
revision, but do not write a full method section, equations, hyperparameters,
or experiments.

# SCIENTIFIC BOUNDARY

- Select only a candidate whose supplied status is candidate.  Never select a
  rejected candidate and never invent an ID.
- Use only the supplied active gap, seed context, available candidates, and
  relevant rejected_solution/rejected_mechanism memory.
- Inspect method_constraints, author-stated design rationales, and explicitly
  rejected alternatives before selecting and developing a candidate. Do not
  select or develop a proposal that simply reintroduces a rejected alternative,
  removes a deliberate choice, or violates a stated constraint without a new
  mechanism that handles the original rejection reason.
- The developed components must retain the relevant constraint-handling
  mechanism and explain how the design rationale is preserved or deliberately
  overcome. Do not silently drop that handling during development.
- The critique must explicitly check information availability, dependency
  consistency, proxy validity, algorithmic invariants, and whether added
  complexity is justified by the gap. Prefer a technically coherent candidate
  over a clever but fragile one.
- Technical soundness is essential.

  Before proposing or developing a solution, ensure that its core mechanism is
  computational and scientifically well-defined.

  In particular:

  - every operation must use information that is actually available at the
    point where it is needed;
  - dependencies between computational steps must be internally consistent;
  - representations, states, caches, parameters, or intermediate results may
    only be shared, compared, or reused when they are genuinely compatible;
  - the proposed mechanism must provide a concrete causal path from the
    modification to the claimed benefit;
  - key algorithmic steps must be implementable rather than relying on an
    unspecified operation;
  - the added computation or memory should not fundamentally negate the
    efficiency or design objective that the method is intended to preserve.

  Prefer a simpler technically coherent mechanism over a more novel-looking
  mechanism that relies on questionable assumptions.
- Do not perform novelty evaluation, Inspiration Retrieval, or Experiment Design.
- Do not return Research State lifecycle fields.  Python deterministically sets
  candidate status, component IDs/statuses, CurrentSolution status, and
  novelty_status.

# OUTPUT

Return exactly one JSON object matching the canonical Pydantic draft schema below.
Output JSON only, with no Markdown or explanatory prose.

EXPECTED JSON SCHEMA:

```json
{{SOLUTION_DEVELOPMENT_DRAFT_JSON_SCHEMA}}
```
