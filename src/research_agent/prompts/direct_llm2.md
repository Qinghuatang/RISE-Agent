You are an expert researcher in NLP and machine learning.

Read the provided seed paper carefully and propose ONE concrete follow-up research direction that could plausibly be developed into a new research paper.

This is a single-pass scientific ideation task. You have access only to the provided seed paper. Do not perform or assume any external literature search, tool use, iterative revision, or additional evidence retrieval.

Your goal is to produce a self-contained and technically detailed research proposal rather than a high-level idea sketch.

The proposal should extend, challenge, or address an important limitation or open opportunity arising from the seed paper. It should contain a substantive scientific contribution rather than a superficial extension such as only using more data, a larger model, additional benchmarks, or straightforward application to another dataset.

Construct the proposal with the following requirements.

1. Research Problem

Clearly explain:
- the relevant contribution or assumption of the seed paper;
- the specific unresolved limitation, weakness, or research opportunity that motivates the follow-up work;
- why this issue is scientifically meaningful;
- the concrete research question that the proposed work will address.

The research problem should be sufficiently specific that the proposed method can be directly motivated from it.

2. Proposed Method

Describe a technically concrete solution.

The method should include:
- the central technical idea;
- the major components or mechanisms needed to implement it;
- how these components interact;
- the training, inference, or optimization procedure when applicable;
- how the proposed method differs substantively from the approach in the seed paper.

Do not merely provide a name or high-level description of the method. Explain enough technical detail for a researcher to understand how the proposed approach could actually be implemented and studied.

3. Method Rationale

Explain why the proposed mechanism is expected to address the identified research problem.

Make the causal or technical reasoning explicit:
- which limitation each major component addresses;
- what assumptions the approach relies on;
- why the proposed mechanism should improve upon or complement the seed paper.

Avoid repeating the method description without additional scientific reasoning.

4. Experimental Plan

Design a concrete experimental plan capable of testing the central claims of the proposal.

Where applicable, specify:
- tasks or experimental settings;
- suitable datasets or benchmarks;
- comparison baselines;
- evaluation metrics;
- the main experimental comparisons;
- up to three informative ablation studies;
- the evaluation protocol needed to test whether the proposed mechanism works as intended.

The experiments should directly test the key scientific claims of the proposed method rather than merely report additional benchmark results.

Important constraints:

- Propose exactly ONE coherent research direction.
- Base the proposal primarily on the scientific content of the seed paper.
- Do not invent experimental results.
- Do not claim that the idea has been exhaustively verified as novel against the entire literature.
- Do not fabricate citations or prior work that are not provided in the seed paper.
- Avoid vague claims such as "improves performance" without explaining the mechanism being tested.
- Avoid unnecessary repetition between sections.
- Make the proposal detailed enough to be evaluated as a potential research project, not merely as a short brainstorming note.
- Return only the requested JSON object and no additional commentary.

Return ONLY valid JSON in the following format:

{
  "research_problem": "...",
  "proposed_method": "...",
  "method_rationale": "...",
  "experimental_plan": "..."
}

Seed paper:

<SEED_PAPER>

{paper_markdown}

</SEED_PAPER>