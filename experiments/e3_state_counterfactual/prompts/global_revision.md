You are performing the Global Revision branch of an experiment comparing a
historical localized revision with a whole-solution rewrite.

Using only the supplied Active Gap, Current Solution, revision need,
verification evidence, selected inspiration (if any), author-stated method
constraints, and relevant failure memory, construct one complete revised
solution. The Active Gap must remain unchanged. Do not generate new Candidate
Gaps, perform Solution Ideation, retrieve papers, or use the historical local
revision or any final proposal.

WHOLE-SOLUTION REWRITE
This branch must regenerate the complete Current Solution, not a local patch.
Reconsider the core idea, rationale, and every Component together so that the
result is internally coherent. Components may be reorganized, added, removed,
replaced, or split, and the rewrite may substantially change any part of the
Current Solution when needed. Treat the supplied solution as the starting
scientific context, not as a preservation or minimal-edit constraint. Do not
merely restate it; construct the complete solution that best addresses the
supplied revision need and evidence.

SCIENTIFIC AND TECHNICAL SOUNDNESS
The revised mechanism must remain implementable and scientifically coherent.
Use only information available at the required stage, maintain consistent
dependencies and information flow, and respect the supplied method
constraints. If an Inspiration is supplied, adapt its transferable principle
only when it creates a substantive mechanism-level change; do not use it as
rhetorical support or copy an external method wholesale.

Do not design experiments, claim a novelty verdict, or claim that the revised
solution has been verified. If a complete coherent rewrite cannot be formed,
return unable_to_revise with no solution fields.

Return only one JSON object matching this schema:

{{E3_GLOBAL_REVISION_DRAFT_JSON_SCHEMA}}

For a revised result, every Component must be included in the one complete
solution and there must be 2-5 Components. Do not include Component status,
evidence_refs, novelty fields, experiments, or any field outside the schema.
