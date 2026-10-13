#!/usr/bin/env python3
"""Generate the independent anonymous paired-scoring artifact.

Only the paired anonymous input is read. No source key or previous score
artifact is consulted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/tmh/research-agent")
INPUT = ROOT / "outputs/llm_paired_scoring_blind_v2/paired_anonymous_cases.json"
OUTPUT_JSON = ROOT / "outputs/llm_paired_scoring_blind_v2/gpt_paired_comparative_scores_v2.json"
OUTPUT_CSV = ROOT / "outputs/llm_paired_scoring_blind_v2/gpt_paired_comparative_scores_v2.csv"

DIMENSIONS = (
    "research_problem_quality",
    "novelty",
    "technical_mechanism_quality",
    "technical_soundness_feasibility",
    "experimental_quality",
    "scientific_significance",
    "overall_proposal_quality",
)

# Current paired-file orientation: positive means A is better; negative means B.
SCORES = [
    (1, 1, 1, -1, 0, 1, 1),
    (1, 1, 1, -1, 0, 1, 1),
    (-1, 0, -1, -2, -1, 0, -1),
    (0, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (0, 1, -1, -1, 0, 1, 0),
    (0, -1, -1, -1, -1, 0, -1),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 1, 1, -1, 0, 1, 1),
    (-1, -1, -1, 1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1),
    (0, 1, 1, 1, 0, -1, 0),
    (1, 1, 1, -1, -1, 1, 0),
    (0, 1, 1, 1, 0, 1, 0),
    (1, 2, 1, -1, 0, 1, 1),
    (1, 1, 1, -1, -1, 0, -1),
    (-1, -1, -1, 1, -1, -1, -1),
    (1, 1, 1, 1, 1, -1, 1),
    (-1, -1, -1, 1, 1, -1, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (-1, -1, -2, -2, 1, -1, -1),
    (1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -2, -1, -1, -1, -1),
    (-1, -1, -1, 1, 1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1),
    (-1, 0, -1, 1, 0, -1, -1),
    (-1, 1, -1, -1, -1, -1, -1),
    (2, 1, 1, 1, 1, 1, 1),
    (-1, 0, -1, -1, 0, -1, -1),
    (1, 1, 1, 0, 1, 1, 1),
    (0, 0, 0, 1, 0, 0, 0),
    (-1, -1, -1, 1, 1, -1, 0),
    (-1, -1, -2, -2, -1, -1, -2),
    (-1, 0, -1, -1, -1, -1, -1),
    (1, 0, -1, -1, -1, 1, -1),
    (-1, 0, -1, -1, -1, -1, -1),
    (-1, 0, -1, -1, -1, 0, -1),
    (1, 1, 1, -1, 0, 1, 1),
    (1, 1, -1, 1, -1, -1, 0),
    (-1, 1, 1, 1, 1, 1, -1),
    (2, 1, 2, 1, 1, 1, 2),
    (-1, -1, -1, 1, 0, -1, -1),
    (-1, -1, -1, 1, -1, -1, -1),
    (0, -1, -1, 1, 1, -1, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (2, -1, 1, 2, 1, 1, 2),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, -1, -1, 1, 1),
    (-2, -1, -2, -2, -1, -1, -2),
    (-1, 1, -1, 1, -1, -1, -1),
    (1, 1, 1, 2, 1, 1, 1),
    (1, 1, 1, -1, 0, 1, 1),
    (-1, 0, -1, -1, -1, 0, -1),
    (0, 1, 0, -1, 1, 1, 0),
    (-2, 1, -1, -2, -1, -1, -2),
    (1, 1, 1, 1, 0, 1, 1),
    (1, 1, -1, -1, 1, 1, 0),
    (-1, 1, 1, 1, -1, 1, 1),
    (-1, 1, -1, -1, -1, -1, -1),
    (2, 1, 2, 1, 1, 1, 2),
    (1, 1, 1, 1, 1, -1, 1),
    (0, 1, -1, -1, 0, -1, -1),
    (1, 1, 1, 1, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0),
]

NOTES = """
A explores multiple stable reasoning styles with adaptive selection, whereas B partitions examples into semantic subsets; A is more expressive while B is simpler.
A learns which model is useful at each search state, whereas B calibrates one reward model with an ensemble; A targets allocation directly while B gives a more conservative estimate.
A distills signals from an existing heuristic pipeline and risks reproducing its assumptions, while B provides an algorithmic re-anchoring of prior context.
A uses compact recursive correction deltas to expand the refinement budget, while B adds self-verification to fixed-chain generation; A is the more focused context-scaling mechanism.
A inserts explicit tool-result checkpoints and repair decisions, while B relies on holistic teacher reification; A gives a clearer operational path for local tool failures.
A adapts task difficulty to measured competence, while B replaces a global instruction check with stepwise entailment; the former is broader and the latter is more locally concrete.
A trains an end-to-end policy over retrieval and generation, while B searches clue continuations with a constrained tree; B is more focused and easier to connect to the stated retrieval bottleneck.
Both proposals replace independent snippet scoring with essentially the same permutation-invariant set-level selection idea, so their scientific content is very close.
A evaluates retrieval under a range of generator capabilities with a worst-case objective, while B learns a dense reward; A offers broader robustness, while B is cheaper and less conservative.
A uses one discounted reliability posterior per source, while B conditions reliability on topic; B captures a real source-topic interaction at the cost of more modeling complexity.
A gates the interpolation coefficient over tokens and time, while B changes the objective to compare explicit answer options; B is more directly aligned with the stated discriminative-task weakness.
A gives a technically specific extreme-value stopping rule for reducing sampling cost, while B models user-specific preferences with retrieval and adaptation; they address different opportunities with comparable scope.
A uses token uncertainty to adapt pairwise weighting, while B supplies a simpler normalized ranking loss; A is more ambitious, but B has cleaner numerical and training behavior.
A removes persistent cache memory through exact transient recomputation, while B learns which cache entries to retain; B avoids repeated full passes, while A has the more exact context access.
A bridges incompatible tokenizers through a byte-level alignment, while B calibrates diffusion probabilities; A is a broader mechanism change, but B has a more focused probabilistic correction.
A adapts activation calibration from verification passes, while B assigns precision layer by layer; A addresses acceptance drift, whereas B is more straightforward and less likely to undermine throughput.
A shares common prefixes through aliases, while B combines prefix sharing with structure-aware scheduling; B covers more of the scheduling problem, while A is the simpler memory intervention.
A identifies important attention heads for low-resource transfer, while B extends the training objective across languages; A targets the observed failure more directly, whereas B has broader deployment scope.
A uses a shared low-rank factorization, while B uses a language-conditioned hypernetwork; A is simpler and easier to scale, while B offers a clearer route to unseen-language adaptation.
A retrieves changing cross-lingual context dynamically, while B stores multilingual segments in a hierarchical memory; A better addresses fixed pairing and relevance, while B offers more controlled sequential reuse.
A mainly changes the fine-tuning schedule to address a resource constraint, while B fixes context-insensitive phrase translation with a concrete linguistic operation; B is the stronger method-level opportunity.
A constructs a robust activation manifold for under-aligned systems, while B replaces distributional assumptions with conformal calibration; B has a more principled uncertainty treatment, though both require careful safety assumptions.
A makes layer selection prompt-conditional but retains a proxy-based localization step, while B uses causal mediation to identify the relevant pathway; B is more mechanistically grounded.
A adds uncertainty to the coefficient predictor, while B replaces additive steering with learned nonlinear residual maps; B changes the mechanism more substantially, while A is easier to constrain.
A formalizes deletion admission with a white-box stability certificate, while B learns an intention extractor for broader access settings; B is more deployable, whereas A has a stronger formal framing.
A selects the number of experts using committee disagreement, while B selects experts per instance; B responds more directly to instance heterogeneity, while A avoids labels and preserves the domain-level design.
A adds nonlinear projections and a relation head, while B explicitly incorporates dependency-relation labels; B is more focused on the missing information, whereas A is more expansive.
A directly extends the temporal graph across documents with shared time anchoring, while B relies on external medical knowledge augmentation; A follows a clearer method-level extension with fewer external dependencies.
A learns a checkpoint policy from future utility, while B applies Bayesian change-point detection to hidden-state shifts; B is more principled and self-contained for locating boundaries.
A learns when continued reasoning is unproductive, while B constructs negatives through wrong-seeking search; A targets the reported inefficiency, while B mainly repairs preference-data supply.
A uses pilot estimates and confidence bounds for strategy choice, while B adds answer clustering and a scaling-potential score; both are plausible, with A having the cleaner statistical basis.
A adapts the number of paths using verification, while B learns a paradigm planner; B addresses paradigm-order variation, whereas A offers the safer and more directly testable sampling change.
A's continuous threshold procedure is partly redundant with the existing search design, while B adapts demonstrations to each query; B is the more coherent response to the stated global-selection limitation.
A stores reusable tool-creation episodes, while B derives an executable environment plan from imports; B is more concrete for poorly documented repositories, while A offers broader cross-task reuse.
A expands a one-shot trigger into a broad meta-cognitive controller, while B adds explicit stage contracts and boundary routing; B has clearer interfaces, while A covers more sequential behavior.
A proposes a broad self-evolution PPO loop, while B specifies turn-level reward modeling for function calls; B has more concrete supervision and a clearer validation path.
A allows clarification throughout an execution loop, while B resolves follow-up parameters from a provenance graph with a two-pass resolver; B is narrower but more operationally specified.
A transfers structured affordances across episodes, while B models within-episode object-tool relations; A offers stronger cross-task scientific value, while B has more immediate local grounding.
A specializes retrieval around quantities, while B decomposes long-form answers into claims; A is more directly grounded in the numeric failure, while B covers a broader but more costly setting.
A co-trains retrieval and generation iteratively, while B adds rarity to candidate ranking; A is more novel, but B is more tightly tied to the observed unseen-term recall problem and easier to control.
A expands the candidate pool when requirements reveal missing evidence, while B only estimates evidence sufficiency inside the existing pool; A can actually recover missing evidence.
A stabilizes repeated self-training with self-consistency, while B alternates retriever and generator updates; B addresses their coupling more directly, while A has a safer stabilization mechanism.
A resolves conflicts through fact-driven iterative retrieval, while B normalizes heterogeneous modalities into shared facts; B directly covers the text-only limitation, though it introduces substantial modality assumptions.
A clips the importance ratio with a fixed bound, while B adapts that bound; B is more flexible, but A has the cleaner and less ambiguous objective.
A adapts uncertainty thresholds and weighting from data, while B replaces external verification with self-consistency pseudo-labels; A preserves stronger feedback grounding and is the more coherent extension.
A separates an evolving actor from an anchored judge, while B infers latent categories from likelihood profiles; A has a clearer reliability safeguard, while B is more novel but less semantically anchored.
A weights examples using verifiable task reward, while B selects a diverse subset using DPP; A aligns selection with downstream success, whereas B provides more task-agnostic diversity control.
A jointly optimizes pruning and weight adaptation, while B rescales positional frequencies; A changes more of the compression procedure, whereas B is more compact and easier to stabilize.
A proposes a workload-independent closed-form law, while B estimates utility online; B respects the observed task/model sensitivity, whereas A makes a stronger unsupported universality claim.
A uses activation-space SVD for initialization, while B adds relative-position bias to recover positional information; B addresses the stated compression frontier more directly, while A is easier to implement.
A learns per-block score corrections while retaining shared routing, while B performs online meta-learning during inference; A preserves efficiency and has the more credible operational path.
A uses differentiable sequence alignment, while B uses hierarchical attention pooling; A more directly preserves token alignment, while B is potentially simpler and more scalable.
A selects multiple donor languages, while B fuses activations through a learned low-rank subspace; B more directly addresses compatibility of the fused representation, though both are plausible.
A generates prompts from typological features, while B adapts a shared initialization from target examples; A offers stronger zero-shot ambition, while B has a more grounded adaptation signal.
A predicts winning tickets from cross-language patterns, while B measures the deviation that invalidates a certification premise; B directly repairs the formal guarantee, while A targets efficiency.
A maintains an online multi-turn risk state, while B adversarially trains the reasoning pathway; A directly captures accumulation across turns, while B more directly challenges manipulated reasoning.
A replaces internal activations with black-box behavioral smoothing, while B models multivariate anomalies with a normalizing flow; A broadens deployment access, while B gives richer white-box anomaly modeling.
A uses causal intervention to identify and preserve safety-relevant heads, while B distills the two-stage behavior into one pass; A has the stronger causal grounding, while B offers lower inference cost.
A adapts from gated user feedback, while B repeatedly red-teams and optimizes against newly generated attacks; B supplies a more controlled robustness signal, while A is more novel but exposed to noisy deployment feedback.
A generalizes token statistics to nested and discontinuous role sets, while B estimates them from pseudo-labels; A directly addresses the structural limitation, whereas B introduces uncertainty from generated labels.
A improves query disambiguation and aggregates evidence in one targeted module, while B adds a confidence-gated feedback loop; A is more controlled and immediately testable, while B is more architecturally ambitious.
A predicts pronunciation cues from visual features, while B retrieves historical phonology from external references; B supplies a more grounded and reliable signal, while A has a more learned and novel route.
A uses an explicit external hierarchy, while B builds a graph from noisy similarities and overlap; A has clearer semantics and validation, whereas B provides more flexible propagation.
Both decouple label and text encoding in nearly the same way; their mechanism, rationale, and evaluation plans are materially equivalent.
""".strip().splitlines()

def make_rationale(score: int, note: str, dimension: str) -> str:
    label = dimension.replace("_", " ")
    if score == 0:
        return f"Neither proposal has a meaningful advantage on {label}; {note}"
    winner = "Proposal A" if score > 0 else "Proposal B"
    degree = "clearly" if abs(score) == 2 else "somewhat"
    return f"{winner} is {degree} stronger on {label} because {note}"

def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 64:
        raise SystemExit("Expected exactly 64 anonymous pairs")
    if len(SCORES) != len(data) or len(NOTES) != len(data):
        raise SystemExit("Judgment table must contain one row per input pair")
    if OUTPUT_JSON.exists() or OUTPUT_CSV.exists():
        raise SystemExit("Refusing to overwrite an existing v2 scoring artifact")

    pair_ids = [item.get("pair_id") for item in data]
    if any(not isinstance(value, str) for value in pair_ids) or len(set(pair_ids)) != 64:
        raise SystemExit("Input pair IDs must be 64 unique strings")

    results = []
    for item, score_row, note in zip(data, SCORES, NOTES):
        score_map = dict(zip(DIMENSIONS, score_row))
        results.append({
            "pair_id": item["pair_id"],
            "scores": score_map,
            "rationale": {
                dimension: make_rationale(score, note, dimension)
                for dimension, score in score_map.items()
            },
        })

    serialized = json.dumps(results, ensure_ascii=False)
    # Ordinary English words such as "directly" are allowed in rationales;
    # reject only explicit source-mapping fields/labels.
    forbidden = ("source_method", "candidate_a", "candidate_b")
    if any(token in serialized.lower() for token in forbidden):
        raise SystemExit("Anonymous score output contains a forbidden source label")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ["pair_id", *DIMENSIONS, "rationale"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"pair_id": result["pair_id"], **result["scores"]}
            row["rationale"] = result["rationale"]["overall_proposal_quality"]
            writer.writerow(row)

    loaded = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert len(loaded) == 64
    assert [row["pair_id"] for row in loaded] == pair_ids
    for row in loaded:
        assert set(row) == {"pair_id", "scores", "rationale"}
        assert set(row["scores"]) == set(DIMENSIONS)
        assert set(row["rationale"]) == set(DIMENSIONS)
        assert all(type(value) is int and -2 <= value <= 2 for value in row["scores"].values())
        assert all(isinstance(value, str) and value.strip() for value in row["rationale"].values())
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_CSV}")
    print("validated 64/64 pairs and 448/448 dimension scores")

if __name__ == "__main__":
    main()
