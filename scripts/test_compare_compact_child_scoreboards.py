#!/usr/bin/env python3
"""Dependency-free tests for compact child scoreboard comparison."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_compact_child_scoreboards as comparator


def write_scoreboard(path: Path, rows: list[dict], *, wrapped: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows} if wrapped else rows
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def child_row(
    dataset: str,
    method: str,
    ndcg: float,
    recall: float,
    *,
    baseline_prefix: str = "eos",
    metrics_path: str | None = "/tmp/volatile.metrics.json",
) -> dict:
    return {
        "dataset": dataset,
        "category": "long_retrieval",
        "baseline": f"{baseline_prefix}-128d-child-turboquant-child",
        "forward_backend": f"{baseline_prefix}-128d-child",
        "method": method,
        "metrics_path": metrics_path,
        "ndcg_at_10": ndcg,
        "recall_at_100": recall,
        "vector_bytes": {"turboquant_ip_b2_child_max": 4248, "turboquant_ip_b4_child_max": 8024}.get(
            method, 15576
        ),
        "dense_vector_bytes": 45056,
        "total_vector_bytes": {"turboquant_ip_b2_child_max": 4248, "turboquant_ip_b4_child_max": 8024}.get(
            method, 15576
        ),
    }


class CompareCompactChildScoreboardsTest(unittest.TestCase):
    def test_parses_top_level_list_and_dict_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_scoreboard(
                root / "baseline.json",
                [child_row("repo-docs", "turboquant_ip_b4_child_max", 0.60, 1.0)],
                wrapped=False,
            )
            candidate = write_scoreboard(
                root / "candidate.json",
                [child_row("repo-docs", "turboquant_ip_b4_child_max", 0.61, 1.0, baseline_prefix="candidate")],
                wrapped=True,
            )

            summary = comparator.compare_scoreboards(
                baseline, candidate, bits={"4"}, min_ndcg_delta=0.0
            )

        self.assertFalse(summary["quality_claim"])
        self.assertIn("evaluation/triage only", summary["evidence_label"])
        self.assertEqual(summary["matched_rows"], 1)
        self.assertEqual(summary["decision"], "pass")
        self.assertAlmostEqual(summary["row_deltas"][0]["ndcg_delta"], 0.01)

    def test_q4_child_pass_and_fail_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_scoreboard(
                root / "baseline.json",
                [
                    child_row("repo-docs", "turboquant_ip_b2_child_max", 0.55, 0.90),
                    child_row("repo-docs", "turboquant_ip_b4_child_max", 0.60, 0.95),
                ],
            )
            pass_candidate = write_scoreboard(
                root / "candidate-pass.json",
                [
                    child_row("repo-docs", "turboquant_ip_b2_child_max", 0.50, 0.90, baseline_prefix="candidate"),
                    child_row("repo-docs", "turboquant_ip_b4_child_max", 0.62, 0.96, baseline_prefix="candidate"),
                ],
            )
            fail_candidate = write_scoreboard(
                root / "candidate-fail.json",
                [
                    child_row("repo-docs", "turboquant_ip_b2_child_max", 0.70, 0.90, baseline_prefix="candidate"),
                    child_row("repo-docs", "turboquant_ip_b4_child_max", 0.59, 0.96, baseline_prefix="candidate"),
                ],
            )

            passing = comparator.compare_scoreboards(baseline, pass_candidate, bits={"4"})
            failing = comparator.compare_scoreboards(baseline, fail_candidate, bits={"4"})

        self.assertEqual(passing["matched_rows"], 1)
        self.assertEqual(passing["decision"], "pass")
        self.assertEqual(failing["decision"], "fail")
        self.assertTrue(
            any(reason.startswith("ndcg_delta_below_threshold") for reason in failing["decision_reasons"])
        )

    def test_missing_rows_are_reported_and_fail_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_scoreboard(
                root / "baseline.json",
                [
                    child_row("repo-docs", "turboquant_ip_b4_child_max", 0.60, 1.0),
                    child_row("repo-docs", "turboquant_ip_b8_child_max", 0.61, 1.0),
                ],
            )
            candidate = write_scoreboard(
                root / "candidate.json",
                [
                    child_row("repo-docs", "turboquant_ip_b4_child_max", 0.62, 1.0, baseline_prefix="candidate"),
                    child_row("other", "turboquant_ip_b4_child_max", 0.62, 1.0, baseline_prefix="candidate"),
                ],
            )

            summary = comparator.compare_scoreboards(baseline, candidate)

        self.assertEqual(summary["decision"], "fail")
        self.assertEqual(len(summary["missing_candidate_rows"]), 1)
        self.assertEqual(len(summary["missing_baseline_rows"]), 1)
        self.assertIn("missing_candidate_rows", summary["decision_reasons"])
        self.assertIn("missing_baseline_rows", summary["decision_reasons"])

    def test_compact_rows_ignore_volatile_fields_and_keep_nonvolatile_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_row = child_row(
                "repo-docs",
                "turboquant_ip_b4_child_max",
                0.60,
                1.0,
                baseline_prefix="eos",
                metrics_path=None,
            )
            baseline_row["notes"] = None
            candidate_row = child_row(
                "repo-docs",
                "turboquant_ip_b4_child_max",
                0.60,
                1.0,
                baseline_prefix="candidate-experiment",
                metrics_path=None,
            )
            candidate_row["notes"] = None
            baseline = write_scoreboard(
                root / "baseline.json",
                [baseline_row],
            )
            candidate = write_scoreboard(
                root / "candidate.json",
                [candidate_row],
            )

            summary = comparator.compare_scoreboards(baseline, candidate)

        self.assertEqual(summary["matched_rows"], 1)
        self.assertEqual(summary["missing_baseline_rows"], [])
        self.assertEqual(summary["missing_candidate_rows"], [])
        delta = summary["row_deltas"][0]
        self.assertNotIn("metrics_path", delta["baseline_row"])
        self.assertNotIn("metrics_path", delta["candidate_row"])
        self.assertIn("notes", delta["baseline_row"])
        self.assertIn("notes", delta["candidate_row"])
        self.assertIsNone(delta["baseline_row"]["notes"])
        self.assertIsNone(delta["candidate_row"]["notes"])

    def test_deterministic_matching_and_tsv_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_scoreboard(
                root / "baseline.json",
                [
                    child_row("b", "turboquant_ip_b4_child_max", 0.60, 1.0),
                    child_row("a", "turboquant_ip_b4_child_max", 0.60, 1.0),
                ],
            )
            candidate = write_scoreboard(
                root / "candidate.json",
                [
                    child_row("a", "turboquant_ip_b4_child_max", 0.61, 1.0, baseline_prefix="candidate"),
                    child_row("b", "turboquant_ip_b4_child_max", 0.61, 1.0, baseline_prefix="candidate"),
                ],
            )
            output = root / "summary.tsv"

            summary = comparator.compare_scoreboards(baseline, candidate)
            comparator.write_tsv(output, summary)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual([row["dataset"] for row in rows], ["a", "b"])
        self.assertEqual([row["status"] for row in rows], ["matched", "matched"])

    def test_threshold_behavior_checks_ndcg_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_scoreboard(
                root / "baseline.json",
                [child_row("repo-docs", "turboquant_ip_b4_child_max", 0.60, 0.95)],
            )
            candidate = write_scoreboard(
                root / "candidate.json",
                [child_row("repo-docs", "turboquant_ip_b4_child_max", 0.62, 0.955, baseline_prefix="candidate")],
            )

            passing = comparator.compare_scoreboards(
                baseline, candidate, min_ndcg_delta=0.01, min_recall_delta=0.0
            )
            failing = comparator.compare_scoreboards(
                baseline, candidate, min_ndcg_delta=0.03, min_recall_delta=0.01
            )

        self.assertEqual(passing["decision"], "pass")
        self.assertEqual(failing["decision"], "fail")
        self.assertTrue(
            any(reason.startswith("ndcg_delta_below_threshold") for reason in failing["decision_reasons"])
        )
        self.assertTrue(
            any(reason.startswith("recall_delta_below_threshold") for reason in failing["decision_reasons"])
        )


if __name__ == "__main__":
    unittest.main()
