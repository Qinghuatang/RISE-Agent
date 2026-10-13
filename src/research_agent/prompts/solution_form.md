# SYSTEM

You are performing FormSolution after a Research Gap has passed related-work
verification.

# TASK

Generate exactly three Candidate Solutions with substantially different core
mechanisms.  Every candidate must directly target the active gap, identify what
key part of the seed method it changes, state the mechanism introduced, and
explain why that mechanism could address the remaining gap.

# SCIENTIFIC BOUNDARY

- Use only the supplied seed context, active gap, valid Gap Verification summary,
  and relevant failure memory.
- Do not search papers or call Inspiration Retrieval.
- Do not evaluate novelty and do not design experiments.
- Do not merely adjust a parameter, make a routine engineering optimization, or
  restate the same mechanism three times.
- Avoid mechanisms in the supplied rejected_solution or rejected_mechanism memory.
- Inspect the supplied method_constraints and author-stated design rationales.
  Do not simply reintroduce an explicitly rejected alternative, undo a
  deliberate choice, or violate a stated efficiency, tractability, stability,
  causality, resource, compatibility, or scalability constraint unless the
  candidate also addresses the reason for rejection or explains the necessary
  constraint trade-off.
- A candidate should preserve the relevant design rationale when possible and
  make its constraint-handling mechanism explicit. Do not treat a design fact
  as a limitation without supplied evidence.
- Technical soundness is essential.

  Before proposing a solution, ensure that its core mechanism is computational
  and scientifically well-defined.

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
- Do not invent IDs or statuses.  Python assigns those deterministically.

# OUTPUT

Return exactly one JSON object with exactly three candidate objects.  The
canonical expected JSON schema is injected below from the Pydantic model.  Do
not add fields or explanatory prose.

EXPECTED JSON SCHEMA:

```json
{{FORM_SOLUTIONS_DRAFT_JSON_SCHEMA}}
```

Output JSON only.
