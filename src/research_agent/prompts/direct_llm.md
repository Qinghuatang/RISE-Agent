You are an expert NLP/ML researcher.

Read the supplied research paper carefully and propose ONE concrete follow-up research idea that could plausibly become a new research paper.

Your goal is not to summarize the paper. Your goal is to identify an important method-level research opportunity and develop a technically specific proposal for addressing it.

Requirements:

1. Research Problem

Identify one specific and meaningful research problem grounded in the seed paper.

The problem may arise from an explicit limitation, an observed weakness, a methodological trade-off or constraint, or a well-grounded open opportunity.

The problem must be method-level.

Purely using more data, a larger model, a new benchmark, additional metrics, or additional ablations is not sufficient.

2. Proposed Method

Propose one concrete method that addresses the identified problem.

The proposal should specify:

- the core mechanism;
- 2–5 major technical components;
- how the components interact;
- how the method differs substantively from the seed paper.

Avoid vague suggestions such as “use a better module”, “introduce adaptive learning”, or “combine multiple methods” without explaining the actual mechanism.

3. Technical Rationale

Explain why the proposed mechanism could solve the identified problem.

State the important assumptions, expected advantages, and possible trade-offs.

Respect important design rationales and constraints stated in the seed paper.

Do not casually reverse a deliberate design choice unless you explain how the new method overcomes the original reason for that choice.

4. Experimental Plan

Design 2–4 experiments that could test the proposal.

For each experiment specify:

- objective;
- setup;
- important baselines, including the Seed Method;
- metrics;
- what claim or component it validates;
- a concrete criterion for interpreting whether the proposal is supported.

Do not invent experimental results.

5. Scientific Quality

Prefer a technically meaningful mechanism over a superficial extension.

The proposal should be specific enough that another researcher could understand how to implement and test it.

You may reason about potential novelty relative to the seed paper, but do not claim that the idea has been globally verified as novel through an exhaustive literature search.

Return ONLY valid JSON in the following format:

{
  "research_problem": "...",
  "proposed_method": {
    "core_idea": "...",
    "components": [
      {
        "name": "...",
        "description": "..."
      }
    ]
  },
  "technical_rationale": "...",
  "experimental_plan": [
    {
      "objective": "...",
      "setup": "...",
      "baselines": ["Seed Method", "..."],
      "metrics": ["..."],
      "what_it_validates": "...",
      "validation_criterion": "..."
    }
  ],
  "expected_contributions": [
    "...",
    "..."
  ]
}

For structural validation only, the canonical JSON Schema is:

{{DIRECT_PROPOSAL_JSON_SCHEMA}}

Formatting constraint: every item in `experimental_plan` MUST include the exact
literal string `"Seed Method"` in its `baselines` JSON array. Do not omit it,
rename it, or replace it with a paper-specific description.

Seed paper:

<SEED_PAPER>
{paper_markdown}
</SEED_PAPER>
