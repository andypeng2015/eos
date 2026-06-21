#!/usr/bin/env python3
"""Dependency-free tests for agreement teacher-score combiner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_row(query: str, positive: str, negatives: list[str], source: str = "fixture") -> dict:
    return {
        "source": source,
        "query": query,
        "positive": positive,
        "negatives": negatives,
    }


def scored_row(
    query: str,
    positive: str,
    negatives: list[str],
    scores: list[float] | None,
    source: str = "fixture",
) -> dict:
    row = base_row(query, positive, negatives, source=source)
    if scores is not None:
        row["teacher_scores"] = scores
    return row


class CombineAgreementTeacherScoresTest(unittest.TestCase):
    def test_cli_averages_only_complete_agreeing_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            qwen = root / "qwen.jsonl"
            mxbai = root / "mxbai.jsonl"
            output = root / "combined.jsonl"
            manifest = root / "manifest.json"
            scores = root / "scores.jsonl"
            rows = [
                base_row("q1", "p1", ["n1", "n2"], source="scifact"),
                base_row("q2", "p2", ["n3"], source="nfcorpus"),
                base_row("q3", "p3", ["n4"], source="fiqa"),
            ]
            write_jsonl(base, rows)
            write_jsonl(
                qwen,
                [
                    scored_row("q1", "p1", ["n1", "n2"], [0.8, 0.2, 0.1], source="scifact"),
                    scored_row("q2", "p2", ["n3"], [0.4, 0.5], source="nfcorpus"),
                    scored_row("q3", "p3", ["n4"], None, source="fiqa"),
                ],
            )
            write_jsonl(
                mxbai,
                [
                    scored_row("q1", "p1", ["n1", "n2"], [0.6, 0.1, 0.2], source="scifact"),
                    scored_row("q2", "p2", ["n3"], [0.7, 0.2], source="nfcorpus"),
                    scored_row("q3", "p3", ["n4"], [0.9, 0.1], source="fiqa"),
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "combine_agreement_teacher_scores.py"),
                    "--base",
                    str(base),
                    "--teacher",
                    f"qwen={qwen}",
                    "--teacher",
                    f"mxbai={mxbai}",
                    "--output-jsonl",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--scores-jsonl",
                    str(scores),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            combined = read_jsonl(output)
            summary = json.loads(manifest.read_text(encoding="utf-8"))
            score_rows = read_jsonl(scores)

        self.assertIn("with_teacher_scores=1", completed.stdout)
        self.assertEqual(len(combined), 3)
        self.assertEqual(combined[0]["teacher_scores"], [0.7, 0.15000000000000002, 0.15000000000000002])
        self.assertNotIn("teacher_scores", combined[1])
        self.assertNotIn("teacher_scores", combined[2])
        self.assertEqual(summary["coverage"]["with_teacher_scores"], 1)
        self.assertEqual(summary["coverage"]["cleared_count"], 2)
        self.assertAlmostEqual(summary["coverage"]["agreement_keep_rate"], 1.0 / 3.0)
        self.assertEqual(summary["teachers"]["qwen"]["examples_complete"], 2)
        self.assertEqual(summary["teachers"]["qwen"]["examples_missing"], 1)
        self.assertEqual(summary["teachers"]["qwen"]["margin_pass"], 1)
        self.assertEqual(summary["teachers"]["mxbai"]["margin_pass"], 3)
        self.assertEqual(len(score_rows), 3)

    def test_cli_respects_min_margin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            qwen = root / "qwen.jsonl"
            mxbai = root / "mxbai.jsonl"
            output = root / "combined.jsonl"
            manifest = root / "manifest.json"
            write_jsonl(base, [base_row("q", "p", ["n"])])
            write_jsonl(qwen, [scored_row("q", "p", ["n"], [0.6, 0.55])])
            write_jsonl(mxbai, [scored_row("q", "p", ["n"], [0.9, 0.1])])

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "combine_agreement_teacher_scores.py"),
                    "--base",
                    str(base),
                    "--teacher",
                    f"qwen={qwen}",
                    "--teacher",
                    f"mxbai={mxbai}",
                    "--output-jsonl",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--min-margin",
                    "0.1",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            combined = read_jsonl(output)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertNotIn("teacher_scores", combined[0])
        self.assertEqual(summary["coverage"]["with_teacher_scores"], 0)
        self.assertEqual(summary["teachers"]["qwen"]["margin_pass"], 0)
        self.assertEqual(summary["teachers"]["mxbai"]["margin_pass"], 1)

    def test_cli_applies_default_source_to_source_less_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            qwen = root / "qwen.jsonl"
            mxbai = root / "mxbai.jsonl"
            output = root / "combined.jsonl"
            manifest = root / "manifest.json"
            scores = root / "scores.jsonl"
            base_without_source = {"query": "q", "positive": "p", "negatives": ["n"]}
            scored_without_source = dict(base_without_source, teacher_scores=[0.9, 0.1])
            write_jsonl(base, [base_without_source])
            write_jsonl(qwen, [scored_without_source])
            write_jsonl(mxbai, [dict(base_without_source, teacher_scores=[0.7, 0.2])])

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "combine_agreement_teacher_scores.py"),
                    "--base",
                    str(base),
                    "--teacher",
                    f"qwen={qwen}",
                    "--teacher",
                    f"mxbai={mxbai}",
                    "--output-jsonl",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--scores-jsonl",
                    str(scores),
                    "--default-source",
                    "scifact",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            combined = read_jsonl(output)
            summary = json.loads(manifest.read_text(encoding="utf-8"))
            score_rows = read_jsonl(scores)

        self.assertEqual(combined[0]["source"], "scifact")
        self.assertEqual(summary["default_source"], "scifact")
        self.assertEqual(summary["coverage"]["source_counts"], {"scifact": {"examples": 1, "kept": 1, "cleared": 0}})
        self.assertTrue(all(row["source"] == "scifact" for row in score_rows))

    def test_cli_default_source_preserves_matching_existing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            qwen = root / "qwen.jsonl"
            mxbai = root / "mxbai.jsonl"
            output = root / "combined.jsonl"
            write_jsonl(base, [base_row("q", "p", ["n"], source="scifact")])
            write_jsonl(qwen, [scored_row("q", "p", ["n"], [0.9, 0.1], source="scifact")])
            write_jsonl(mxbai, [scored_row("q", "p", ["n"], [0.7, 0.2], source="scifact")])

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "combine_agreement_teacher_scores.py"),
                    "--base",
                    str(base),
                    "--teacher",
                    f"qwen={qwen}",
                    "--teacher",
                    f"mxbai={mxbai}",
                    "--output-jsonl",
                    str(output),
                    "--default-source",
                    "scifact",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            combined = read_jsonl(output)

        self.assertEqual(combined[0]["source"], "scifact")

    def test_cli_default_source_rejects_conflicting_existing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            qwen = root / "qwen.jsonl"
            mxbai = root / "mxbai.jsonl"
            output = root / "combined.jsonl"
            write_jsonl(base, [base_row("q", "p", ["n"], source="fiqa")])
            write_jsonl(qwen, [scored_row("q", "p", ["n"], [0.9, 0.1], source="fiqa")])
            write_jsonl(mxbai, [scored_row("q", "p", ["n"], [0.7, 0.2], source="fiqa")])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "combine_agreement_teacher_scores.py"),
                    "--base",
                    str(base),
                    "--teacher",
                    f"qwen={qwen}",
                    "--teacher",
                    f"mxbai={mxbai}",
                    "--output-jsonl",
                    str(output),
                    "--default-source",
                    "scifact",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("existing source 'fiqa' does not match", completed.stderr)

    def test_cli_rejects_signature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            qwen = root / "qwen.jsonl"
            mxbai = root / "mxbai.jsonl"
            output = root / "combined.jsonl"
            write_jsonl(base, [base_row("q", "p", ["n"])])
            write_jsonl(qwen, [scored_row("q", "different", ["n"], [1.0, 0.0])])
            write_jsonl(mxbai, [scored_row("q", "p", ["n"], [1.0, 0.0])])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "combine_agreement_teacher_scores.py"),
                    "--base",
                    str(base),
                    "--teacher",
                    f"qwen={qwen}",
                    "--teacher",
                    f"mxbai={mxbai}",
                    "--output-jsonl",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("example signature does not match", completed.stderr)


if __name__ == "__main__":
    unittest.main()
