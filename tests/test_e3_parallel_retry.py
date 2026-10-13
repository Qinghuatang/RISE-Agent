import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from experiments.e3_state_counterfactual.parallel_retry import (
    run_incomplete_cases_parallel,
    select_incomplete_cases,
)


class E3ParallelRetryTests(unittest.TestCase):
    def test_selects_only_cases_missing_a_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = SimpleNamespace(paper_id="paper-a")
            incomplete = SimpleNamespace(paper_id="paper-b")
            complete_dir = root / "unified_verification" / complete.paper_id
            complete_dir.mkdir(parents=True)
            for branch in ("targeted", "global"):
                (complete_dir / f"unified_verification_{branch}.json").write_text("{}")
            selected = select_incomplete_cases([incomplete, complete], root)
            self.assertEqual([case.paper_id for case in selected], ["paper-b"])

    def test_parallel_retry_isolates_failures_and_keeps_paper_granularity(self):
        cases = [SimpleNamespace(paper_id=f"paper-{index}") for index in range(4)]
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_runner(case, output_dir, **kwargs):
            nonlocal active, max_active
            self.assertFalse(kwargs["overwrite"])
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            if case.paper_id == "paper-1":
                raise RuntimeError("synthetic worker failure")
            return {"paper_id": case.paper_id, "status": "completed"}

        with tempfile.TemporaryDirectory() as tmp:
            results = run_incomplete_cases_parallel(
                cases,
                tmp,
                deepseek=Mock(),
                verification_config=Mock(),
                workers=2,
                case_runner=fake_runner,
            )
        self.assertGreaterEqual(max_active, 2)
        self.assertEqual([item["paper_id"] for item in results], [
            "paper-0", "paper-1", "paper-2", "paper-3"
        ])
        by_id = {item["paper_id"]: item for item in results}
        self.assertEqual(by_id["paper-1"]["status"], "failed")
        self.assertEqual(by_id["paper-0"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
