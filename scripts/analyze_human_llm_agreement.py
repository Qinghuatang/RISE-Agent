#!/usr/bin/env python3
"""Offline analysis of the human evaluation and existing LLM blind scores.

No proposal, rating, source key, or LLM result is modified.  The source keys
are read only to convert anonymous positions to method labels for analysis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scipy.stats import binomtest, spearmanr
except ImportError as exc:  # pragma: no cover - environment setup failure
    raise SystemExit("scipy is required for exact binomial and Spearman tests") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(f"P{i:03d}" for i in range(1, 81))
EXPERTS = tuple(f"expert_{i:02d}" for i in range(1, 6))
REASONS = ("problem", "mechanism", "feasibility", "experiment", "significance", "differentiation", "overall")
RANKINGS = {"A>B>C", "A>C>B", "B>A>C", "B>C>A", "C>A>B", "C>B>A"}
COMPARISONS = ("rise_vs_direct", "rise_vs_researchagent")


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AnalysisError(f"invalid JSON: {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception as exc:
            raise AnalysisError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise AnalysisError(f"JSONL record is not an object: {path}:{line_number}")
        rows.append(item)
    return rows


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def exact_ci(successes: int, total: int) -> tuple[float, float]:
    result = binomtest(successes, total, 0.5)
    interval = result.proportion_ci(confidence_level=0.95, method="exact")
    return float(interval.low), float(interval.high)


def exact_sign_p(successes: int, total: int) -> float:
    return float(binomtest(successes, total, 0.5, alternative="two-sided").pvalue)


def fleiss_kappa(votes_by_seed: list[list[bool]]) -> float:
    if not votes_by_seed:
        return float("nan")
    n = len(votes_by_seed[0])
    if n < 2:
        return float("nan")
    category_counts = [sum(votes) for votes in votes_by_seed]
    total = len(votes_by_seed) * n
    p_positive = sum(category_counts) / total
    p_negative = 1.0 - p_positive
    p_bar = statistics.mean(
        (count * (count - 1) + (n - count) * (n - count - 1)) / (n * (n - 1))
        for count in category_counts
    )
    p_expected = p_positive * p_positive + p_negative * p_negative
    if math.isclose(p_expected, 1.0):
        return float("nan")
    return float((p_bar - p_expected) / (1.0 - p_expected))


def kappa_binary(human: list[bool], llm: list[bool]) -> float:
    if len(human) != len(llm) or not human:
        return float("nan")
    n = len(human)
    agreement = sum(a == b for a, b in zip(human, llm)) / n
    h_pos = sum(human) / n
    l_pos = sum(llm) / n
    expected = h_pos * l_pos + (1 - h_pos) * (1 - l_pos)
    if math.isclose(expected, 1.0):
        return float("nan")
    return float((agreement - expected) / (1 - expected))


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else float("nan")


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def read_human_ratings(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    h1_key = read_json(root / "private/h1_source_key.json")
    h2_key = read_json(root / "private/h2_source_key.json")
    issues: list[str] = []
    h1: list[dict[str, Any]] = []
    h2: list[dict[str, Any]] = []
    for expert in EXPERTS:
        h1_files = sorted((root / expert / "H1").glob("session_*/ratings.csv"))
        h2_files = sorted((root / expert / "H2").glob("session_*/ratings.csv"))
        if len(h1_files) != 3:
            issues.append(f"{expert}: expected 3 H1 session files, found {len(h1_files)}")
        if len(h2_files) != 3:
            issues.append(f"{expert}: expected 3 H2 session files, found {len(h2_files)}")
        for path in h1_files:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    pid = row.get("case_id", "").strip()
                    ranking = row.get("ranking", "").strip()
                    if pid not in SEEDS:
                        issues.append(f"H1 invalid case id {pid} in {path}")
                        continue
                    if ranking not in RANKINGS:
                        issues.append(f"H1 invalid or missing ranking for {expert}/{pid}: {ranking!r}")
                        continue
                    try:
                        confidence = int(row.get("confidence", ""))
                    except ValueError:
                        issues.append(f"H1 invalid confidence for {expert}/{pid}")
                        continue
                    if confidence not in {1, 2, 3}:
                        issues.append(f"H1 confidence outside 1..3 for {expert}/{pid}")
                    reason = row.get("primary_reason", "").strip()
                    if reason not in REASONS:
                        issues.append(f"H1 invalid primary_reason for {expert}/{pid}: {reason!r}")
                    mapping = h1_key.get(expert, {}).get(pid)
                    if not isinstance(mapping, dict) or set(mapping.values()) != {"rise", "direct", "researchagent"}:
                        issues.append(f"H1 source key is invalid for {expert}/{pid}")
                        continue
                    ordered = [mapping[position] for position in ranking.split(">")]
                    h1.append({"expert": expert, "case_id": pid, "ranking": ranking, "ordered": ordered, "confidence": confidence, "reason": reason})
        for path in h2_files:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    pid = row.get("case_id", "").strip()
                    winner_position = row.get("winner", "").strip()
                    if pid not in SEEDS:
                        issues.append(f"H2 invalid case id {pid} in {path}")
                        continue
                    if winner_position not in {"A", "B"}:
                        issues.append(f"H2 invalid or missing winner for {expert}/{pid}: {winner_position!r}")
                        continue
                    try:
                        confidence = int(row.get("confidence", ""))
                    except ValueError:
                        issues.append(f"H2 invalid confidence for {expert}/{pid}")
                        continue
                    if confidence not in {1, 2, 3}:
                        issues.append(f"H2 confidence outside 1..3 for {expert}/{pid}")
                    reason = row.get("primary_reason", "").strip()
                    if reason not in REASONS:
                        issues.append(f"H2 invalid primary_reason for {expert}/{pid}: {reason!r}")
                    mapping = h2_key.get(expert, {}).get(pid)
                    if not isinstance(mapping, dict) or set(mapping.values()) != {"full", "flat"}:
                        issues.append(f"H2 source key is invalid for {expert}/{pid}")
                        continue
                    h2.append({"expert": expert, "case_id": pid, "winner": mapping[winner_position], "position": winner_position, "confidence": confidence, "reason": reason})
    # Duplicate/missing checks are performed at the raw rating level.
    for label, rows, expected_per_expert in (("H1", h1, 48), ("H2", h2, 48)):
        by_expert = Counter(row["expert"] for row in rows)
        by_seed = Counter(row["case_id"] for row in rows)
        if any(by_expert[e] != expected_per_expert for e in EXPERTS):
            issues.append(f"{label} per-expert counts: {dict(by_expert)}")
        if set(by_seed) != set(SEEDS):
            issues.append(f"{label} seed coverage is not P001-P080")
        if any(count != 3 for count in by_seed.values()):
            issues.append(f"{label} seed rating counts are not all 3")
    if len(h1) != 240:
        issues.append(f"H1 total rankings is {len(h1)}, expected 240")
    if len(h2) != 240:
        issues.append(f"H2 total judgments is {len(h2)}, expected 240")
    return h1, h2, issues


def read_h1_llm(root: Path) -> tuple[dict[str, dict[str, dict[str, float]]], list[str], str]:
    result_dir = root / "result"
    key_dir = root / "keys"
    files = sorted(result_dir.glob("llm*_batch*_results.jsonl"))
    issues: list[str] = []
    scores: dict[str, dict[str, dict[str, float]]] = {comparison: {judge: {} for judge in ("llm1", "llm2", "llm3")} for comparison in COMPARISONS}
    key_by_judge: dict[str, dict[str, dict[str, Any]]] = {}
    for judge in ("llm1", "llm2", "llm3"):
        path = key_dir / f"{judge}_source_key.json"
        if not path.exists():
            issues.append(f"missing H1 LLM source key: {path}")
            continue
        key_by_judge[judge] = {item["case_id"]: item for item in read_json(path).get("cases", [])}
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        match = re.match(r"(llm[123])_batch", path.name)
        if not match:
            issues.append(f"cannot identify H1 LLM judge from {path.name}")
            continue
        judge = match.group(1)
        for row in read_jsonl(path):
            cid = row.get("case_id")
            key = key_by_judge.get(judge, {}).get(cid)
            if not key:
                issues.append(f"H1 LLM result has no source-key mapping: {judge}/{cid}")
                continue
            comparison = key.get("comparison")
            seed = key.get("seed_id")
            if comparison not in COMPARISONS or seed not in SEEDS:
                issues.append(f"invalid H1 LLM key mapping: {judge}/{cid}")
                continue
            score = row.get("scores", {}).get("overall_proposal_quality")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not -2 <= score <= 2:
                issues.append(f"invalid H1 LLM overall score: {judge}/{cid}: {score!r}")
                continue
            identity = (judge, comparison, seed)
            if identity in seen:
                issues.append(f"duplicate H1 LLM score: {identity}")
            seen.add(identity)
            # Source keys guarantee the score direction is positive for RISE.
            signed = float(score) if key.get("proposal_a_source") == "rise" else -float(score)
            scores[comparison][judge][seed] = signed
    for comparison in COMPARISONS:
        for judge in ("llm1", "llm2", "llm3"):
            if set(scores[comparison][judge]) != set(SEEDS):
                issues.append(f"H1 LLM coverage incomplete for {judge}/{comparison}: {len(scores[comparison][judge])}/80")
    return scores, issues, str(result_dir)


def read_h2_llm(root: Path) -> tuple[dict[str, dict[str, float]], list[str], str, str]:
    key_path = root / "source_key.json"
    result_dir = root / "result"
    issues: list[str] = []
    key = read_json(key_path) if key_path.exists() else {}
    scores: dict[str, dict[str, float]] = {judge: {} for judge in ("result1", "result2", "result3")}
    files = sorted(result_dir.glob("result*.jsonl"))
    for path in files:
        judge = path.stem
        if judge not in scores:
            continue
        for row in read_jsonl(path):
            cid = row.get("case_id")
            item = key.get(cid)
            if not item:
                issues.append(f"H2 LLM result has no source-key mapping: {judge}/{cid}")
                continue
            seed = item.get("seed_id")
            raw = row.get("overall_proposal_quality")
            if seed not in SEEDS or not isinstance(raw, (int, float)) or isinstance(raw, bool) or not -2 <= raw <= 2:
                issues.append(f"invalid H2 LLM result: {judge}/{cid}")
                continue
            if seed in scores[judge]:
                issues.append(f"duplicate H2 LLM score: {judge}/{seed}")
            # H2 score is A minus B; orient it as Full minus Flat.
            scores[judge][seed] = float(raw) if item.get("proposal_a") == "full" else -float(raw)
    for judge in scores:
        if set(scores[judge]) != set(SEEDS):
            issues.append(f"H2 LLM coverage incomplete for {judge}: {len(scores[judge])}/80")
    return scores, issues, str(result_dir), str(key_path)


def human_binary(h1: list[dict[str, Any]], h2: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_h1: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_h2: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in h1:
        by_h1[row["case_id"]].append(row)
    for row in h2:
        by_h2[row["case_id"]].append(row)
    h1_out: dict[str, dict[str, Any]] = {}
    for pid in SEEDS:
        rows = by_h1[pid]
        values: dict[str, Any] = {}
        for comparison, baseline in (("rise_vs_direct", "direct"), ("rise_vs_researchagent", "researchagent")):
            votes = [row["ordered"].index("rise") < row["ordered"].index(baseline) for row in rows]
            count = sum(votes)
            values[comparison] = {"rise_votes": count, "baseline_votes": 3 - count, "rise_wins": count >= 2, "margin": (2 * count - 3) / 3, "votes": votes}
        h1_out[pid] = values
    h2_out: dict[str, dict[str, Any]] = {}
    for pid in SEEDS:
        rows = by_h2[pid]
        votes = [row["winner"] == "full" for row in rows]
        count = sum(votes)
        h2_out[pid] = {"full_votes": count, "flat_votes": 3 - count, "full_wins": count >= 2, "margin": (2 * count - 3) / 3, "votes": votes}
    return h1_out, h2_out


def human_human_stats(h1_out: dict[str, dict[str, Any]], h2_out: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for comparison in COMPARISONS:
        votes = [h1_out[pid][comparison]["votes"] for pid in SEEDS]
        unanimous = sum(len(set(vote)) == 1 for vote in votes)
        result[comparison] = {"unanimous_cases": unanimous, "majority_cases": 80 - unanimous, "unanimous_rate": unanimous / 80, "fleiss_kappa": fleiss_kappa(votes)}
    votes = [h2_out[pid]["votes"] for pid in SEEDS]
    unanimous = sum(len(set(vote)) == 1 for vote in votes)
    result["full_vs_flat"] = {"unanimous_cases": unanimous, "majority_cases": 80 - unanimous, "unanimous_rate": unanimous / 80, "fleiss_kappa": fleiss_kappa(votes)}
    return result


def agreement_stats(human: dict[str, dict[str, Any]], llm_scores: dict[str, dict[str, float]], direction_name: str) -> dict[str, Any]:
    ensemble = {pid: statistics.mean(llm_scores[judge][pid] for judge in llm_scores) for pid in SEEDS}
    llm_winner = {pid: ("positive" if score > 0 else "negative" if score < 0 else "tie") for pid, score in ensemble.items()}
    human_winner: dict[str, str] = {}
    for pid in SEEDS:
        positive = human[pid]["rise_wins"] if direction_name != "full_vs_flat" else human[pid]["full_wins"]
        human_winner[pid] = "positive" if positive else "negative"
    included = [pid for pid in SEEDS if llm_winner[pid] != "tie"]
    h = [human_winner[pid] == "positive" for pid in included]
    l = [llm_winner[pid] == "positive" for pid in included]
    matches = sum(a == b for a, b in zip(h, l))
    rho, rho_p = spearmanr([human[pid]["margin"] for pid in SEEDS], [ensemble[pid] for pid in SEEDS])
    return {
        "ensemble_score": ensemble,
        "llm_winner": llm_winner,
        "llm_ties": len(SEEDS) - len(included),
        "simple_agreement": {"agree": matches, "eligible_cases": len(included), "rate": matches / len(included) if included else None},
        "cohen_kappa": kappa_binary(h, l),
        "spearman": {"rho": float(rho), "p_value": float(rho_p)},
    }


def descriptive_human(h1: list[dict[str, Any]], h2: list[dict[str, Any]], h1_out: dict[str, dict[str, Any]], h2_out: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rank_counts: dict[str, dict[str, int]] = {}
    for method in ("rise", "direct", "researchagent"):
        ranks = [row["ordered"].index(method) + 1 for row in h1]
        rank_counts[method] = {"first": ranks.count(1), "second": ranks.count(2), "third": ranks.count(3), "mean_rank": statistics.mean(ranks)}
    reasons_h1 = Counter(row["reason"] for row in h1)
    reasons_h2 = Counter(row["reason"] for row in h2)
    rise_reason = Counter()
    for row in h1:
        for baseline in ("direct", "researchagent"):
            if row["ordered"].index("rise") < row["ordered"].index(baseline):
                rise_reason[row["reason"]] += 1
    return {"h1_rank_counts": rank_counts, "h1_reason_counts": dict(reasons_h1), "h1_reason_when_rise_preferred": dict(rise_reason), "h2_reason_counts": dict(reasons_h2), "h1_confidence_mean": statistics.mean(row["confidence"] for row in h1), "h2_confidence_mean": statistics.mean(row["confidence"] for row in h2)}


def confidence_stats(h1: list[dict[str, Any]], h2: list[dict[str, Any]], h1_out: dict[str, dict[str, Any]], h2_out: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, rows, outcome_map, comparison in (
        ("h1_rise_vs_direct", h1, h1_out, "rise_vs_direct"),
        ("h1_rise_vs_researchagent", h1, h1_out, "rise_vs_researchagent"),
        ("h2_full_vs_flat", h2, h2_out, "full_vs_flat"),
    ):
        by_seed = outcome_map if comparison == "full_vs_flat" else {pid: outcome_map[pid][comparison] for pid in SEEDS}
        unanimous_seeds = {pid for pid in SEEDS if len(set(by_seed[pid]["votes"])) == 1}
        unanimous_values = [row["confidence"] for row in rows if row["case_id"] in unanimous_seeds]
        mixed_values = [row["confidence"] for row in rows if row["case_id"] not in unanimous_seeds]
        counts = Counter(row["confidence"] for row in rows)
        result[label] = {"mean": statistics.mean(row["confidence"] for row in rows), "proportions": {str(i): counts[i] / len(rows) for i in (1, 2, 3)}, "unanimous_case_mean": mean_or_nan(unanimous_values), "non_unanimous_case_mean": mean_or_nan(mixed_values)}
    return result


def expert_stats(h1: list[dict[str, Any]], h2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for expert in EXPERTS:
        h1e = [row for row in h1 if row["expert"] == expert]
        h2e = [row for row in h2 if row["expert"] == expert]
        result.append({"expert_id": expert, "h1_cases": len(h1e), "h1_rise_vs_direct_rate": mean_or_nan(row["ordered"].index("rise") < row["ordered"].index("direct") for row in h1e), "h1_rise_vs_researchagent_rate": mean_or_nan(row["ordered"].index("rise") < row["ordered"].index("researchagent") for row in h1e), "h1_confidence_mean": mean_or_nan(row["confidence"] for row in h1e), "h2_cases": len(h2e), "h2_full_rate": mean_or_nan(row["winner"] == "full" for row in h2e), "h2_confidence_mean": mean_or_nan(row["confidence"] for row in h2e)})
    return result


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_human(stats: dict[str, Any], hh: dict[str, Any], confidence: dict[str, Any], descriptive: dict[str, Any]) -> str:
    lines = ["# Human expert results", "", "The inferential unit is the seed paper (80 seeds). Each result below first aggregates the three expert judgments for a seed by majority vote.", "", "## Main pairwise results", "", "| Comparison | RISE/Full wins | Baseline wins | Preference rate | 95% exact binomial CI | Exact sign-test p |", "|---|---:|---:|---:|---:|---:|"]
    for key, name in (("rise_vs_direct", "RISE-Full vs Direct-Full"), ("rise_vs_researchagent", "RISE-Full vs ResearchAgent"), ("full_vs_flat", "RISE-Full vs RISE-Flat")):
        item = stats[key]
        lines.append(f"| {name} | {item['positive_wins']}/80 | {item['negative_wins']}/80 | {item['preference_rate']:.3f} | [{item['ci_low']:.3f}, {item['ci_high']:.3f}] | {item['sign_test_p']:.4g} |")
    lines += ["", "## H1 three-way descriptive ranking", "", "| Method | First | Second | Third | Mean rank (lower is better) |", "|---|---:|---:|---:|---:|"]
    for method, label in (("rise", "RISE-Full"), ("direct", "Direct-Full"), ("researchagent", "ResearchAgent")):
        item = descriptive["h1_rank_counts"][method]
        lines.append(f"| {label} | {item['first']} | {item['second']} | {item['third']} | {item['mean_rank']:.3f} |")
    lines += ["", "## Human-human agreement", "", "| Comparison | Unanimous | 2/1 majority | Unanimous rate | Fleiss kappa |", "|---|---:|---:|---:|---:|"]
    for key, label in (("rise_vs_direct", "RISE-Full vs Direct-Full"), ("rise_vs_researchagent", "RISE-Full vs ResearchAgent"), ("full_vs_flat", "RISE-Full vs RISE-Flat")):
        item = hh[key]
        lines.append(f"| {label} | {item['unanimous_cases']} | {item['majority_cases']} | {item['unanimous_rate']:.3f} | {item['fleiss_kappa']:.3f} |")
    lines += ["", "## Auxiliary confidence and primary reason", "", f"- H1 mean confidence: {descriptive['h1_confidence_mean']:.3f}; H2 mean confidence: {descriptive['h2_confidence_mean']:.3f}.", f"- H1 primary reasons: `{descriptive['h1_reason_counts']}`.", f"- H1 reasons on individual pairwise judgments where RISE was preferred: `{descriptive['h1_reason_when_rise_preferred']}`.", f"- H2 primary reasons: `{descriptive['h2_reason_counts']}`.", "", "Confidence and primary_reason are descriptive quality-control fields, not causal explanations."]
    return "\n".join(lines) + "\n"


def markdown_agreement(agreement: dict[str, Any], sensitivity: dict[str, Any]) -> str:
    lines = ["# Human–LLM evaluation agreement", "", "LLM ensemble scores are the mean of the three existing judges' Overall Proposal Quality scores. Binary agreement and Cohen's kappa exclude LLM ensemble ties; Spearman uses all 80 seeds, including ties.", "", "| Comparison | Human wins | LLM ties | Simple agreement | Cohen kappa | Spearman rho | Spearman p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("rise_vs_direct", "RISE-Full vs Direct-Full"), ("rise_vs_researchagent", "RISE-Full vs ResearchAgent"), ("full_vs_flat", "RISE-Full vs RISE-Flat")):
        item = agreement[key]
        human_wins = sensitivity.get(key, {}).get("human_positive_wins", "see human results")
        lines.append(f"| {label} | {human_wins} | {item['llm_ties']} | {item['simple_agreement']['agree']}/{item['simple_agreement']['eligible_cases']} = {item['simple_agreement']['rate']:.3f} | {item['cohen_kappa']:.3f} | {item['spearman']['rho']:.3f} | {item['spearman']['p_value']:.4g} |")
    item = sensitivity["rise_vs_researchagent"]
    lines += ["", "## P064 sensitivity: RISE-Full vs ResearchAgent", "", f"- Full 80 seeds: human RISE wins {item['human_positive_wins_full']}/80 ({item['human_rate_full']:.3f}); Human–LLM agreement = {item['agreement_full']:.3f} excluding {item['llm_ties_full']} LLM ties.", f"- Excluding P064: human RISE wins {item['human_positive_wins_excluding_p064']}/79 ({item['human_rate_excluding_p064']:.3f}); Human–LLM agreement = {item['agreement_excluding_p064']:.3f} excluding {item['llm_ties_excluding_p064']} LLM ties.", "- This is a sensitivity analysis only; the primary analysis remains all 80 seeds."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", type=Path, default=Path("/home/tmh/baseline/human_evaluation_v2"))
    parser.add_argument("--h1-llm-dir", type=Path, default=Path("/home/tmh/baseline/e1_blind_eval_v2"))
    parser.add_argument("--h2-llm-dir", type=Path, default=REPO_ROOT / "outputs/state_evolution_ablation_blind_eval_final")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/tmh/baseline/human_evaluation_v2/analysis"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise AnalysisError(f"refusing to overwrite existing analysis directory: {args.output_dir}")
    h1, h2, human_issues = read_human_ratings(args.human_dir)
    h1_llm, h1_llm_issues, h1_llm_result_dir = read_h1_llm(args.h1_llm_dir)
    h2_llm, h2_llm_issues, h2_llm_result_dir, h2_llm_key_path = read_h2_llm(args.h2_llm_dir)
    all_issues = human_issues + h1_llm_issues + h2_llm_issues
    tmp = Path(tempfile.mkdtemp(prefix="human_llm_analysis.", dir=str(args.output_dir.parent)))
    try:
        validation_lines = ["# Data validation", "", "## Human ratings", "", f"- H1 ratings: {len(h1)}/240; H2 judgments: {len(h2)}/240.", "- H1/H2 coverage and per-expert/per-seed counts were checked from the session CSV files.", "- Ranking values were checked as strict complete A/B/C permutations; H2 winners were checked as A/B; confidence and primary_reason codebooks were checked.", "- Private source keys were read read-only and validated for every rating.", "", "## Existing LLM results", "", f"- H1 result structure: six JSONL files under `{h1_llm_result_dir}` (three judges × two batches), mapped through the corresponding private keys; each judge has one RISE–Direct and one RISE–ResearchAgent score per seed.", f"- H2 score structure used: three JSONL files under `{h2_llm_result_dir}`, mapped through `{h2_llm_key_path}`. The supplied `{args.h2_llm_dir}`/state-evolution run tree itself contains proposal artifacts; its score files were not treated as scores.", "- No LLM/API was called and no input file was modified.", "", "## Issues", ""]
        validation_lines.extend(f"- {issue}" for issue in all_issues or ["None."])
        (tmp / "data_validation.md").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
        if all_issues:
            raise AnalysisError("data validation failed; see data_validation.md")

        h1_out, h2_out = human_binary(h1, h2)
        hh = human_human_stats(h1_out, h2_out)
        agreements = {
            "rise_vs_direct": agreement_stats({pid: h1_out[pid]["rise_vs_direct"] for pid in SEEDS}, {judge: h1_llm["rise_vs_direct"][judge] for judge in h1_llm["rise_vs_direct"]}, "rise_vs_direct"),
            "rise_vs_researchagent": agreement_stats({pid: h1_out[pid]["rise_vs_researchagent"] for pid in SEEDS}, {judge: h1_llm["rise_vs_researchagent"][judge] for judge in h1_llm["rise_vs_researchagent"]}, "rise_vs_researchagent"),
            "full_vs_flat": agreement_stats({pid: h2_out[pid] for pid in SEEDS}, h2_llm, "full_vs_flat"),
        }
        # Main human comparison statistics, using seed-level majority.
        main_stats: dict[str, Any] = {}
        for key, positive_label in (("rise_vs_direct", "rise"), ("rise_vs_researchagent", "rise"), ("full_vs_flat", "full")):
            positive = sum((h1_out[pid][key]["rise_wins"] if key != "full_vs_flat" else h2_out[pid]["full_wins"]) for pid in SEEDS)
            low, high = exact_ci(positive, 80)
            main_stats[key] = {"positive_wins": positive, "negative_wins": 80 - positive, "preference_rate": positive / 80, "ci_low": low, "ci_high": high, "sign_test_p": exact_sign_p(positive, 80), "positive_label": positive_label}
        descriptive = descriptive_human(h1, h2, h1_out, h2_out)
        confidence = confidence_stats(h1, h2, h1_out, h2_out)
        experts = expert_stats(h1, h2)
        # P064 sensitivity for the RISE vs ResearchAgent comparison.
        full_ag = agreements["rise_vs_researchagent"]
        excluded = [pid for pid in SEEDS if pid != "P064"]
        llm_non_tie_full = [pid for pid in SEEDS if full_ag["llm_winner"][pid] != "tie"]
        llm_non_tie_ex = [pid for pid in excluded if full_ag["llm_winner"][pid] != "tie"]
        sensitivity = {
            "rise_vs_direct": {"human_positive_wins": main_stats["rise_vs_direct"]["positive_wins"]},
            "rise_vs_researchagent": {
                "human_positive_wins": main_stats["rise_vs_researchagent"]["positive_wins"],
                "human_positive_wins_full": main_stats["rise_vs_researchagent"]["positive_wins"],
                "human_rate_full": main_stats["rise_vs_researchagent"]["positive_wins"] / 80,
                "agreement_full": full_ag["simple_agreement"]["rate"],
                "llm_ties_full": full_ag["llm_ties"],
                "human_positive_wins_excluding_p064": sum(h1_out[pid]["rise_vs_researchagent"]["rise_wins"] for pid in excluded),
                "human_rate_excluding_p064": sum(h1_out[pid]["rise_vs_researchagent"]["rise_wins"] for pid in excluded) / 79,
                "agreement_excluding_p064": sum((h1_out[pid]["rise_vs_researchagent"]["rise_wins"] == (full_ag["llm_winner"][pid] == "positive")) for pid in llm_non_tie_ex) / len(llm_non_tie_ex),
                "llm_ties_excluding_p064": 79 - len(llm_non_tie_ex),
            },
            "full_vs_flat": {"human_positive_wins": main_stats["full_vs_flat"]["positive_wins"]},
        }
        # One row per seed, with all paired human and LLM information.
        seed_rows: list[dict[str, Any]] = []
        for pid in SEEDS:
            row: dict[str, Any] = {"paper_id": pid}
            for key, prefix in (("rise_vs_direct", "human_rise_vs_direct"), ("rise_vs_researchagent", "human_rise_vs_researchagent")):
                item = h1_out[pid][key]
                row[f"{prefix}_rise_votes"] = item["rise_votes"]
                row[f"{prefix}_majority_winner"] = "rise" if item["rise_wins"] else "baseline"
                ag = agreements[key]
                row[f"llm_{key}_ensemble_score"] = ag["ensemble_score"][pid]
                row[f"llm_{key}_winner"] = ag["llm_winner"][pid]
                row[f"human_llm_{key}_agreement"] = ("NA_LLM_TIE" if ag["llm_winner"][pid] == "tie" else item["rise_wins"] == (ag["llm_winner"][pid] == "positive"))
            item = h2_out[pid]
            row["human_full_vs_flat_full_votes"] = item["full_votes"]
            row["human_full_vs_flat_majority_winner"] = "full" if item["full_wins"] else "flat"
            ag = agreements["full_vs_flat"]
            row["llm_full_vs_flat_ensemble_score"] = ag["ensemble_score"][pid]
            row["llm_full_vs_flat_winner"] = ag["llm_winner"][pid]
            row["human_llm_full_vs_flat_agreement"] = ("NA_LLM_TIE" if ag["llm_winner"][pid] == "tie" else item["full_wins"] == (ag["llm_winner"][pid] == "positive"))
            seed_rows.append(row)
        fields = list(seed_rows[0])
        csv_write(tmp / "seed_level_results.csv", seed_rows, fields)
        csv_write(tmp / "expert_level_results.csv", experts, list(experts[0]))
        (tmp / "human_results.md").write_text(markdown_human(main_stats, hh, confidence, descriptive), encoding="utf-8")
        (tmp / "human_llm_agreement.md").write_text(markdown_agreement(agreements, sensitivity), encoding="utf-8")
        detailed = {
            "data_validation": {"passed": True, "h1_ratings": len(h1), "h2_ratings": len(h2), "seed_count": 80, "h1_llm_result_dir": h1_llm_result_dir, "h2_llm_result_dir": h2_llm_result_dir, "h2_llm_source_key": h2_llm_key_path},
            "human_main": main_stats,
            "human_human_agreement": hh,
            "human_descriptive": descriptive,
            "confidence": confidence,
            "expert_level": experts,
            "human_llm_agreement": agreements,
            "p064_sensitivity": sensitivity,
            "inference_unit": "seed paper; 80 seeds",
        }
        dump_json(tmp / "detailed_statistics.json", safe_json(detailed))
        tmp.rename(args.output_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"created: {args.output_dir}")
    print("validated: H1 240/240, H2 240/240, LLM mappings complete")
    print(f"RISE vs Direct human majority: {main_stats['rise_vs_direct']['positive_wins']}/80")
    print(f"RISE vs ResearchAgent human majority: {main_stats['rise_vs_researchagent']['positive_wins']}/80")
    print(f"Full vs Flat human majority: {main_stats['full_vs_flat']['positive_wins']}/80")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
