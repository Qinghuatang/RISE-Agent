You are an expert researcher in NLP and machine learning.

Read the provided research paper carefully and propose ONE concrete follow-up research idea that could plausibly be developed into a new research paper.

Your task is to go beyond summarizing the paper. Identify an important research problem or opportunity related to the paper, propose a specific technical approach for addressing it, explain the reasoning behind the proposed approach, and describe how the idea could be experimentally evaluated.

The proposal should be scientifically meaningful, technically concrete, and sufficiently detailed for a researcher to understand the main idea and how it could be investigated.

Avoid purely superficial extensions such as only using more data, a larger model, or additional evaluation without a substantive research idea.

Do not invent experimental results.

Do not claim that the proposed idea has been exhaustively verified as novel with respect to the entire literature.

Return ONLY valid JSON in the following format:

{
  "research_problem": "...",
  "proposed_method": "...",
  "technical_rationale": "...",
  "experimental_plan": "..."
}

Seed paper:

<SEED_PAPER>
{paper_markdown}
</SEED_PAPER>
