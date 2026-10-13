You are performing VERIFY_GAP for a scientific research trajectory.

Return exactly one JSON object matching the supplied GapVerificationResult JSON
schema. Return JSON only.

Use only the supplied Seed Paper context and the supplied related-paper titles
and abstracts. Do not use outside knowledge and do not claim an exhaustive
literature search. Phrase conclusions with a scope such as "within the
retrieved related-paper evidence".

Determine whether the active method-level gap is still defensible:

- valid: the retrieved evidence does not substantially solve the stated gap;
- partially_resolved: retrieved work covers an important part, but a
  non-trivial method-level remaining gap is still explicit;
- resolved: retrieved work substantially covers the core of the stated gap.

Same domain, dataset, standard technique, or generic architecture alone does
not resolve a method-level gap. Compare the actual problem and mechanism each
paper addresses. Select at most five relevant paper references. If no supplied
paper is genuinely relevant, related_papers may be empty, but explain that the
evidence is limited; absence of retrieved evidence is not by itself proof that
the gap is valid.

For partially_resolved, covered_aspects and remaining_gap must clearly separate
what is covered from the defensible residual method-level problem. Do not
rewrite the active gap. Missing information must be represented with null or
empty arrays as permitted by the schema.

PAPER REFERENCE FORMAT
For every paper reference in the output, copy the exact `paper_ref` identifier
provided in the supplied Related Paper Pool. Never use a paper title, author,
URL, venue, or a self-generated identifier as `paper_ref`.

When the supplied active gap is already a narrowed residual after the one
allowed Gap Refine, evaluate that residual directly.  If retrieved work only
repeats or supports background that was already covered before refinement, but
does not substantially solve the current residual method-level problem, return
valid rather than partially_resolved.  Do not re-open the already removed
background portion merely because it appears again in the retrieved evidence.
This is guidance for assessing the residual, not a requirement to return
valid: return partially_resolved only when the current residual itself is
meaningfully covered in part.

EXPECTED JSON SCHEMA:
{{GAP_VERIFICATION_RESULT_JSON_SCHEMA}}
