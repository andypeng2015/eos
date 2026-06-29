#!/usr/bin/env python3
"""Tests for teacher-guided score-spectrum adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_retrieval_teacher_guided_score_spectrum.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def guided_row(
    row_id: str,
    query: str,
    policy: str,
    scores: list[float],
    candidate_dev_positive_flags: dict | None = None,
) -> dict:
    flags = candidate_dev_positive_flags or {}
    return {
        "schema": "eos.embedding_text_hard_negative.teacher_guided.v1",
        "source": "fixture",
        "row_id": row_id,
        "query_id": row_id.replace("r", "q"),
        "query": query,
        "positive_doc_id": f"p-{row_id}",
        "positive": f"positive {row_id}",
        "negative_doc_ids": [f"n-{row_id}-1", f"n-{row_id}-2"],
        "negatives": [f"negative {row_id} one", f"negative {row_id} two"],
        "teacher_scores": scores,
        "candidate_dev_positive_flags": flags,
        "teacher_guide": {
            "policy": policy,
            "hard_label_allowed": policy == "clean_agreement",
            "soft_only": policy != "clean_agreement",
            "required_teachers": ["qwen", "mxbai"],
            "per_teacher_margins": {"qwen": 0.1, "mxbai": 0.2},
        },
        "legal_gates": {
            "train_allowed_for_research": True,
            "release_train_allowed": False,
            "commercial_use_allowed": False,
        },
        "train_allowed_for_research": True,
        "release_train_allowed": False,
        "commercial_use_allowed": False,
    }


class TeacherGuidedScoreSpectrumTest(unittest.TestCase):
    def test_cli_converts_clean_and_conflict_soft_and_excludes_dev_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide = root / "guide.jsonl"
            manifest_in = root / "guide.manifest.json"
            full = root / "full.jsonl"
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            excluded = root / "excluded.jsonl"
            manifest = root / "manifest.json"
            write_jsonl(
                guide,
                [
                    guided_row("r1", "query one", "clean_agreement", [0.9, 0.2, 0.1]),
                    guided_row("r2", "query two", "conflict", [0.1, 0.8, 0.2]),
                    guided_row("r3", "query three", "ambiguous_soft_only", [0.5, 0.5, 0.4]),
                    guided_row(
                        "r4",
                        "query four",
                        "clean_agreement",
                        [0.8, 0.1, 0.0],
                        candidate_dev_positive_flags={"n1": {"overlaps_dev_positive": True}},
                    ),
                    guided_row(
                        "r5",
                        "query five",
                        "clean_agreement",
                        [0.7, 0.2, 0.1],
                        candidate_dev_positive_flags={
                            "n1": {"overlaps_dev_positive": False},
                            "n2": {"overlaps_dev_positive": False, "nested": {"confirmed": False}},
                        },
                    ),
                ],
            )
            manifest_in.write_text(json.dumps({"schema": "fixture", "policy": {"conflict_policy": "emit_soft_only"}}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--guide-jsonl",
                    str(guide),
                    "--guide-manifest",
                    str(manifest_in),
                    "--output-full-jsonl",
                    str(full),
                    "--output-train-jsonl",
                    str(train),
                    "--output-eval-jsonl",
                    str(eval_path),
                    "--excluded-jsonl",
                    str(excluded),
                    "--manifest",
                    str(manifest),
                    "--train-count",
                    "2",
                    "--eval-count",
                    "1",
                    "--split-seed",
                    "173",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            rows = read_jsonl(full)
            train_rows = read_jsonl(train)
            eval_rows = read_jsonl(eval_path)
            excluded_rows = read_jsonl(excluded)
            got_manifest = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("eligible=4", completed.stdout)
        self.assertEqual(len(rows), 4)
        by_row_id = {row["row_id"]: row for row in rows}
        self.assertIn("r5", by_row_id)
        self.assertNotIn("r4", by_row_id)
        self.assertEqual(by_row_id["r1"]["hard_negative_eligible"], [False, True, True])
        self.assertEqual(by_row_id["r1"]["hard_loss_weight"], 1.0)
        self.assertEqual(by_row_id["r2"]["hard_negative_eligible"], [False, False, False])
        self.assertEqual(by_row_id["r2"]["hard_loss_weight"], 0.0)
        self.assertEqual(by_row_id["r2"]["soft_loss_weight"], 1.0)
        self.assertEqual(by_row_id["r2"]["recovery_loss_weight"], 0.0)
        for row in rows:
            self.assertEqual(row["positive_indexes"], [0])
            self.assertEqual(row["selected_positive_index"], 0)
            self.assertAlmostEqual(sum(row["target_probabilities"]), 1.0, places=6)
            self.assertEqual(row["legal_gates"]["train_allowed_for_research"], True)
            self.assertEqual(row["release_train_allowed"], False)
            self.assertEqual(row["commercial_use_allowed"], False)
        self.assertEqual(len(excluded_rows), 1)
        self.assertEqual(excluded_rows[0]["reason"], "dev_positive_negative_flag")
        self.assertEqual(excluded_rows[0]["row_id"], "r4")
        self.assertEqual(len({row["query"] for row in train_rows} & {row["query"] for row in eval_rows}), 0)
        self.assertEqual(got_manifest["counts"]["policy_counts"], {"clean_agreement": 2, "conflict": 1, "ambiguous_soft_only": 1})
        self.assertEqual(got_manifest["counts"]["dev_positive_flag_rows"], 1)
        self.assertEqual(got_manifest["counts"]["query_overlap"], 0)
        self.assertEqual(got_manifest["validation"]["full"]["rows"], 4)


if __name__ == "__main__":
    unittest.main()
