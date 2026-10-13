You are performing VERIFY_SOLUTION for a scientific research trajectory.

Return exactly one JSON object matching the supplied internal
SolutionVerificationDraft JSON schema. It must contain `comparisons` and a
nested final `result`. Return JSON only.

Use only the supplied Seed Paper context, the complete Current Solution, the
verification focus, and the supplied related-paper titles and abstracts. Do
not use outside knowledge and do not claim an exhaustive literature search.

First create at most five ComparisonRecord objects, one for each paper that
actually participates in the novelty decision. These comparison records are
the evidence basis for the nested final result. Do not repeat a paper
reference. Then produce the final SolutionVerificationResult.

Compare the problem, core idea, mechanism, component roles, and key
differences. Even when the focus is changed_units, inspect the full Current
Solution; the focus only identifies what should receive priority.

Apply these novelty boundaries:

- distinct: standard/supporting techniques, datasets, settings, or ordinary
  architectural elements may be shared, but no substantial coverage of the
  main novelty-bearing mechanism is shown;
- partial_overlap: one or more local novelty-bearing elements are
  substantially covered, while the core mechanism remains distinct and
  salvageable;
- core_overlap: a supplied paper substantially covers the core
  novelty-bearing mechanism, so the current solution should be rejected.

Do not infer overlap from the same domain, task, backbone, dataset, or generic
technique alone. A partial result must contain at least one
novelty_relevant_overlaps item. A core result must identify evidence covering
the core mechanism. Use at most five related paper references. Missing
abstracts are unavailable evidence; do not pretend to have read them.

The final result.related_papers must equal the comparison paper_ref list in the
same order. A matched_paper_ref in the final novelty overlaps must be one of
those comparison refs. Ordinary shared elements may support distinct; only
novelty-bearing overlaps support partial_overlap or core_overlap.

PAPER REFERENCE FORMAT
For every paper reference in `comparisons`, `result.related_papers`, and
`matched_paper_ref`, copy the exact `paper_ref` identifier supplied in the
Related Paper Pool. Never use the paper title, author, URL, venue, or a
self-generated identifier as a paper reference.

EXPECTED INTERNAL JSON SCHEMA:
{{SOLUTION_VERIFICATION_DRAFT_JSON_SCHEMA}}
