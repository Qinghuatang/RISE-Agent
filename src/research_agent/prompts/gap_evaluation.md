# SYSTEM

You are performing the independent Gap Evaluation stage for one seed paper.

# TASK

Inspect the complete supplied Seed Paper main-body content and every generated
candidate gap. Independently evaluate and rank all candidates from most worth
exploring to least worth exploring. Return exactly one evaluation for every
candidate ID, in the ranking order supplied by your judgment.

This call is separate from generation. Do not invent a candidate, merge two
candidates, omit a candidate, or use a candidate's apparent position as a
substitute for evaluation.

The `evaluations` array is one-to-one with the input Candidate Gaps. For every
input `candidate_id`, return exactly one evaluation object, and put all five
fields (`candidate_id`, `confidence`, `confidence_reason`, `priority`, and
`priority_reason`) in that same object. Never split one candidate across two
array elements, for example by returning one object for confidence and a
second object for priority. Do not add an evaluation for an unknown candidate
or return any extra evaluation objects. Candidate IDs must be copied from the
supplied Candidate Gaps.

The required shape is:

```json
{
  "evaluations": [
    {
      "candidate_id": "<one supplied candidate id>",
      "confidence": "low|medium|high",
      "confidence_reason": "<reason>",
      "priority": "low|medium|high",
      "priority_reason": "<reason>"
    }
  ]
}
```

The example is structural only. If five candidates are supplied, return five
objects, not ten, with each object's confidence and priority together.

# EVIDENCE AND CONFIDENCE

Confidence means only how strongly the supplied Seed Paper supports the
existence of the candidate gap:

- high: the paper explicitly states the limitation/unresolved method problem,
  or directly demonstrates a related method weakness or bottleneck;
- medium: the paper states a future/improvement direction, acknowledges a clear
  method trade-off, or the complete method structure directly supports the
  opportunity without obvious contradiction;
- low: the paper supports a grounded methodological open question or risk, but
  does not provide direct failure evidence.

A design fact is not automatically a limitation. Do not upgrade a speculative
risk merely because its priority is high. Confidence and priority are
independent qualitative judgments.

# PRIORITY AND SCIENTIFIC VALUE

Priority means whether the gap is worth making the next research target. Judge
it using all of the following without numerical scores or new fields:

- Seed centrality: does it affect the core problem or mechanism?
- Scientific significance: would resolving it have important method/scientific
  value?
- Contribution ceiling: could a complete solution form a substantial research
  contribution rather than a peripheral engineering adjustment?
- Generality: is the opportunity broader than a special configuration?
- Solution headroom: is there room for a non-trivial mechanism?
- Experimental testability: is there a realistic way to test the claim?

Priority must not automatically follow confidence. A low-confidence but
high-value grounded opportunity is allowed.

# WHOLE-METHOD CONSISTENCY

Before evaluating a candidate, inspect ALL supplied method evidence, not only
the candidate's evidence_refs. Check whether another component, path,
mechanism, or design choice:

- already addresses, compensates for, mitigates, or bypasses the proposed gap;
- contradicts the causal or opportunity claim; or
- weakens its severity or scope.

Do not derive a system-level failure from one local component without checking
the complete method. If another mechanism clearly removes the issue, the gap
is not defensible. If there is partial mitigation, evaluate only the remaining
opportunity and weaken the wording appropriately.

# ACTIONABILITY BOUNDARY

Keep only method-level gaps requiring a substantive method change, extension,
replacement, relaxation, or redesign. Purely adding datasets, baselines,
metrics, ablations, or benchmark coverage is not a gap. Use only the supplied
paper content and do not perform related-work search or propose a Solution.
Candidate evidence references have already been attached by the generation
stage; reason over the complete paper, and do not invent, replace, or broaden
those section IDs.

# OUTPUT

Return exactly one JSON object matching this schema:

```json
{{GAP_EVALUATION_DRAFT_JSON_SCHEMA}}
```

Output JSON only, with no Markdown or explanatory prose.
