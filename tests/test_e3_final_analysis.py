import unittest

from experiments.e3_state_counterfactual.final_analysis import (
    build_summary,
    exact_mcnemar_p_value,
    paired_contingency,
)


def _row(paper_id, targeted, global_result, trigger="overlap_repair"):
    return {
        "paper_id": paper_id,
        "trigger_type": trigger,
        "before_outcome": "partial_overlap" if trigger == "overlap_repair" else "distinct",
        "targeted_outcome": targeted,
        "global_outcome": global_result,
    }


class E3FinalAnalysisTests(unittest.TestCase):
    def test_exact_mcnemar_uses_only_discordant_pairs(self):
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p_value(2, 0), 0.5)

    def test_paired_contingency_keeps_seed_as_sampling_unit(self):
        rows = [
            _row("paper-1", "distinct", "distinct"),
            _row("paper-2", "distinct", "partial_overlap"),
            _row("paper-3", "core_overlap", "distinct"),
            _row("paper-4", "partial_overlap", "core_overlap"),
        ]
        result = paired_contingency(rows)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["discordant_pairs"], 2)
        self.assertEqual(result["targeted_better_discordant"], 1)
        self.assertEqual(result["global_better_discordant"], 1)
        self.assertEqual(
            result["cells"]["targeted_distinct_global_distinct"], 1
        )

    def test_subset_summary_preserves_distinct_rate_definitions(self):
        rows = [
            _row("paper-1", "distinct", "partial_overlap"),
            _row("paper-2", "partial_overlap", "partial_overlap"),
            _row("paper-3", "distinct", "distinct", "inspiration_strengthening"),
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["overlap_repair"]["case_count"], 2)
        self.assertEqual(
            summary["overlap_repair"]["repair_to_distinct_rate"]["targeted"],
            0.5,
        )
        self.assertEqual(summary["inspiration_strengthening"]["case_count"], 1)
        self.assertEqual(
            summary["inspiration_strengthening"]["remain_distinct_rate"]["targeted"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
