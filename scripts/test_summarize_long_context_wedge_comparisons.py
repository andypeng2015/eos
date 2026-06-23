#!/usr/bin/env python3
"""Dependency-free tests for long-context wedge comparison summarization."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_long_context_wedge_comparisons as summarizer


def write_comparison(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def row(
    *,
    dataset: str | None = None,
    baseline: str = "eos",
    model: str = "eos",
    method: str = "direct_token_span_fusion_rrf_k10_lambda025",
    bits: int = 4,
    ndcg: float = 0.8,
    recall: float = 1.0,
    child_count: int = 20,
    storage_multiple: float = 1.0,
    quality_claim: bool = False,
) -> dict:
    result = {
        "baseline": baseline,
        "model": model,
        "method": method,
        "bits": bits,
        "ndcg_at_10": ndcg,
        "recall_at_100": recall,
        "child_count": child_count,
        "storage_multiple": storage_multiple,
        "quality_claim": quality_claim,
        "metrics_path": f"/tmp/{model}-{method}.metrics.json",
    }
    if dataset is not None:
        result["dataset"] = dataset
    return result


def external_row(model: str, ndcg: float, *, bits: int = 4, recall: float = 1.0) -> dict:
    return row(
        baseline="external_chunked",
        model=model,
        method="chunk96_overlap16_turboquant_ip_b4_child_top2-mean",
        bits=bits,
        ndcg=ndcg,
        recall=recall,
        child_count=100,
        storage_multiple=2.0,
    )


class SummarizeLongContextWedgeComparisonsTest(unittest.TestCase):
    def test_pass_case_where_eos_beats_both_external_q4_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_comparison(
                Path(tmp) / "pass" / "comparison.json",
                [
                    row(dataset="alpha", ndcg=0.91, storage_multiple=1.0),
                    external_row("qwen3-0.6b", 0.90),
                    external_row("mxbai-large", 0.88),
                ],
            )

            summary = summarizer.summarize_comparisons([str(path)])

        self.assertFalse(summary["quality_claim"])
        self.assertEqual(summary["decision"], "pass")
        self.assertEqual(summary["datasets_where_eos_beats_all_external"], ["alpha"])
        self.assertAlmostEqual(summary["mean_gap_eos_minus_best_external_ndcg_at_10"], 0.01)

    def test_fail_case_where_eos_trails_external_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_comparison(
                Path(tmp) / "eos-official-longembed-cache-compare-v2-qmsum-20260622T122816Z" / "comparison.json",
                [
                    row(ndcg=0.5545454642810175, recall=1.0, storage_multiple=0.9572265625),
                    external_row("qwen3-0.6b", 0.8762934699274822),
                    external_row("mxbai-large", 0.8069465347786748),
                ],
            )

            summary = summarizer.summarize_comparisons([str(path)])

        self.assertEqual(summary["decision"], "fail")
        self.assertFalse(summary["quality_claim"])
        self.assertEqual(summary["datasets"][0]["dataset"], "qmsum")
        comparisons = summary["datasets"][0]["external_comparisons"]
        by_model = {item["external_model"]: item for item in comparisons}
        self.assertAlmostEqual(by_model["qwen3-0.6b"]["ndcg_delta"], -0.3217480056464647)
        self.assertAlmostEqual(
            by_model["qwen3-0.6b"]["storage_multiple_ratio_eos_over_external"],
            0.9572265625 / 2.0,
        )
        self.assertIn("eos_trails_required_external_ndcg", summary["decision_reasons"])

    def test_missing_required_external_row_fails_and_tsv_marks_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_comparison(
                root / "missing" / "comparison.json",
                [
                    row(dataset="beta", ndcg=0.9),
                    external_row("qwen3-0.6b", 0.8),
                ],
            )
            output_tsv = root / "summary.tsv"

            summary = summarizer.summarize_comparisons([str(path)])
            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(summary["decision"], "fail")
        self.assertEqual(
            summary["missing_required_external_rows"],
            [{"dataset": "beta", "model": "mxbai-large", "bits": 4}],
        )
        missing = [item for item in rows if item["missing_external"] == "true"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["external_model"], "mxbai-large")
        self.assertEqual(missing[0]["quality_claim"], "false")

    def test_quality_claim_true_errors_unless_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_comparison(
                Path(tmp) / "claim" / "comparison.json",
                [
                    row(dataset="gamma", ndcg=0.9, quality_claim=True),
                    external_row("qwen3-0.6b", 0.8),
                    external_row("mxbai-large", 0.7),
                ],
            )

            with self.assertRaises(ValueError):
                summarizer.summarize_comparisons([str(path)])
            summary = summarizer.summarize_comparisons([str(path)], allow_quality_claims=True)
            cli = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "summarize_long_context_wedge_comparisons.py"),
                    "--allow-quality-claims",
                    str(path),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertFalse(summary["quality_claim"])
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertIn('"quality_claim": false', cli.stdout)

    def test_deterministic_label_inference_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zeta = write_comparison(
                root / "zeta-run" / "comparison.json",
                [
                    row(dataset="zeta", ndcg=0.8),
                    external_row("qwen3-0.6b", 0.7),
                    external_row("mxbai-large", 0.6),
                ],
            )
            alpha = write_comparison(
                root / "eos-official-longembed-cache-compare-v2-qmsum-20260622T122816Z" / "comparison.json",
                [
                    row(ndcg=0.8),
                    external_row("qwen3-0.6b", 0.7),
                    external_row("mxbai-large", 0.6),
                ],
            )

            summary = summarizer.summarize_comparisons([str(zeta), str(alpha)])

        self.assertEqual([item["dataset"] for item in summary["datasets"]], ["qmsum", "zeta"])
        self.assertEqual([item["dataset"] for item in summary["inputs"]], ["qmsum", "zeta"])


if __name__ == "__main__":
    unittest.main()
