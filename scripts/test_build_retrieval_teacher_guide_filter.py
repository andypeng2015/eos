#!/usr/bin/env python3
"""Tests for retrieval teacher guide cache/filter harness."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_retrieval_teacher_guide_filter.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_row(row_id: str, query_id: str, positive_id: str, negative_ids: list[str]) -> dict:
    return {
        "schema": "eos.embedding_text_hard_negative.research_stagea.v1",
        "source": "fixture/source",
        "row_id": row_id,
        "dataset": "fixture",
        "split": "train",
        "query_id": query_id,
        "positive_doc_id": positive_id,
        "negative_doc_ids": negative_ids,
        "query": f"query {query_id}",
        "positive": f"positive {positive_id}",
        "negatives": [f"negative {doc_id}" for doc_id in negative_ids],
        "roles": {"query": "query", "positive": "document", "negatives": "document"},
        "legal_gates": {
            "train_allowed_for_research": True,
            "release_train_allowed": False,
            "commercial_use_allowed": False,
        },
    }


def candidate_scores(row: dict, scores: list[float]) -> list[dict]:
    candidates = [row["positive"]] + row["negatives"]
    doc_ids = [row["positive_doc_id"]] + row["negative_doc_ids"]
    out = []
    for index, (candidate, doc_id, score) in enumerate(zip(candidates, doc_ids, scores)):
        out.append(
            {
                "source": row["source"],
                "row_id": row["row_id"],
                "query_id": row["query_id"],
                "query": row["query"],
                "candidate": candidate,
                "candidate_doc_id": doc_id,
                "candidate_index": index,
                "role": "positive" if index == 0 else "negative",
                "score": score,
            }
        )
    return out


class RetrievalTeacherGuideFilterTest(unittest.TestCase):
    def test_cli_joins_scores_and_accounts_policy_drops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "rows.jsonl"
            qwen_path = root / "qwen.jsonl"
            mxbai_path = root / "mxbai.jsonl"
            output_path = root / "filtered.jsonl"
            manifest_path = root / "manifest.json"

            rows = [
                base_row("r-clean", "q1", "p1", ["n1", "n2"]),
                base_row("r-ambiguous", "q2", "p2", ["n3"]),
                base_row("r-conflict", "q3", "p3", ["n4"]),
                base_row("r-missing", "q4", "p4", ["n5"]),
            ]
            write_jsonl(rows_path, rows)
            write_jsonl(
                qwen_path,
                candidate_scores(rows[0], [0.9, 0.2, 0.1])
                + candidate_scores(rows[1], [0.5, 0.5])
                + candidate_scores(rows[2], [0.1, 0.8])
                + candidate_scores(rows[3], [0.7, 0.2]),
            )
            write_jsonl(
                mxbai_path,
                candidate_scores(rows[0], [0.8, 0.3, 0.2])
                + candidate_scores(rows[1], [0.55, 0.55])
                + candidate_scores(rows[2], [0.9, 0.1])
                + candidate_scores(rows[3], [0.6]),  # missing negative score.
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--rows-jsonl",
                    str(rows_path),
                    "--teacher-cache",
                    f"qwen={qwen_path}",
                    "--teacher-cache",
                    f"mxbai={mxbai_path}",
                    "--teacher-model",
                    "qwen=fixture-qwen",
                    "--teacher-model",
                    "mxbai=fixture-mxbai",
                    "--teacher-config",
                    'qwen={"score_scale":"fixture"}',
                    "--output-jsonl",
                    str(output_path),
                    "--manifest",
                    str(manifest_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            output_rows = read_jsonl(output_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("emitted=2", completed.stdout)
        self.assertEqual(len(output_rows), 2)
        self.assertEqual(output_rows[0]["teacher_guide"]["policy"], "clean_agreement")
        self.assertTrue(output_rows[0]["teacher_guide"]["hard_label_allowed"])
        self.assertEqual(output_rows[1]["teacher_guide"]["policy"], "ambiguous_soft_only")
        self.assertFalse(output_rows[1]["teacher_guide"]["hard_label_allowed"])
        self.assertTrue(output_rows[1]["teacher_guide"]["soft_only"])
        for row in output_rows:
            self.assertEqual(len(row["teacher_scores"]), 1 + len(row["negatives"]))
            self.assertEqual(set(row["teacher_guide"]["per_teacher_scores"]), {"qwen", "mxbai"})
            self.assertEqual(
                row["legal_gates"],
                {
                    "train_allowed_for_research": True,
                    "release_train_allowed": False,
                    "commercial_use_allowed": False,
                },
            )
        self.assertEqual(manifest["counts"]["rows_seen"], 4)
        self.assertEqual(manifest["counts"]["rows_emitted"], 2)
        self.assertEqual(manifest["counts"]["clean_agreement"], 1)
        self.assertEqual(manifest["counts"]["ambiguous_soft_only"], 1)
        self.assertEqual(manifest["counts"]["conflict"], 1)
        self.assertEqual(manifest["counts"]["conflict_drop"], 1)
        self.assertEqual(manifest["counts"]["missing_score_drop"], 1)
        self.assertEqual(manifest["policy"]["missing_score_policy"], "strict_drop")
        self.assertTrue(manifest["validation"]["no_row_emitted_without_required_scores"])
        self.assertTrue(manifest["legal_gate_accounting"]["research_only_preserved"])

    def test_cli_accepts_row_level_teacher_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "rows.jsonl"
            qwen_path = root / "qwen-row-level.jsonl"
            output_path = root / "filtered.jsonl"
            manifest_path = root / "manifest.json"
            rows = [base_row("r1", "q1", "p1", ["n1"])]
            write_jsonl(rows_path, rows)
            row_level = dict(rows[0])
            row_level["teacher_scores"] = [0.7, 0.1]
            write_jsonl(qwen_path, [row_level])

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--rows-jsonl",
                    str(rows_path),
                    "--teacher-cache",
                    f"qwen={qwen_path}",
                    "--output-jsonl",
                    str(output_path),
                    "--manifest",
                    str(manifest_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            output_rows = read_jsonl(output_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(output_rows), 1)
        self.assertEqual(output_rows[0]["teacher_scores"], [0.7, 0.1])
        self.assertEqual(manifest["inputs"]["teacher_caches"]["qwen"]["cache_counts"]["row_level_examples"], 1)


if __name__ == "__main__":
    unittest.main()
