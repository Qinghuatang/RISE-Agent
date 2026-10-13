You are performing VERIFY_SOLUTION for a scientific research trajectory.

Return exactly one JSON object matching the supplied SolutionVerificationResult
JSON schema. Return JSON only.

Use only the supplied Seed Paper context, the complete Current Solution, the
verification focus, and the supplied related-paper titles and abstracts. Do
not use outside knowledge and do not claim an exhaustive literature search.

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

EXPECTED JSON SCHEMA:
{{SOLUTION_VERIFICATION_RESULT_JSON_SCHEMA}}
