#!/usr/bin/env python3
"""Tests for provenance-safe agreement-teacher prep builder."""

from __future__ import annotations

import csv
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


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(rows)


class BuildProvenanceSafeAgreementTeacherPrepTest(unittest.TestCase):
    def test_cli_keeps_only_exact_safe_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "manta-embed-v1"
            raw = dataset / "raw" / "toy" / "toy"
            processed = dataset / "processed"
            prep = root / "prep"
            out = root / "out"

            write_jsonl(
                raw / "queries.jsonl",
                [
                    {"_id": "q-safe", "text": "safe query"},
                    {"_id": "q-test", "text": "test query"},
                    {"_id": "q-amb-1", "text": "ambiguous query"},
                    {"_id": "q-amb-2", "text": "ambiguous query"},
                    {"_id": "q-neg", "text": "test negative query"},
                ],
            )
            write_jsonl(
                raw / "corpus.jsonl",
                [
                    {"_id": "d-safe", "title": "Safe", "text": "doc"},
                    {"_id": "d-test", "title": "Test", "text": "doc"},
                    {"_id": "d-neg", "title": "", "text": "negative"},
                    {"_id": "d-test-neg", "title": "", "text": "test negative"},
                    {"_id": "d-amb", "title": "", "text": "ambiguous positive"},
                ],
            )
            write_qrels(
                raw / "qrels" / "train.tsv",
                [
                    ("q-safe", "d-safe", 1),
                    ("q-amb-1", "d-amb", 1),
                    ("q-amb-2", "d-amb", 1),
                    ("q-neg", "d-safe", 1),
                ],
            )
            write_qrels(raw / "qrels" / "test.tsv", [("q-test", "d-test", 1), ("q-test", "d-test-neg", 1)])

            rows = [
                {"query": "safe query", "positive": "Safe\ndoc", "negatives": ["negative"]},
                {"query": "test query", "positive": "Test\ndoc", "negatives": ["negative"]},
                {"query": "ambiguous query", "positive": "ambiguous positive", "negatives": ["negative"]},
                {"query": "test negative query", "positive": "Safe\ndoc", "negatives": ["test negative"]},
            ]
            write_jsonl(processed / "toy.train-hard-negatives.jsonl", rows)
            for teacher in ("t1", "t2"):
                scored = [dict(row, teacher_scores=[0.9, 0.1]) for row in rows]
                write_jsonl(
                    prep / "toy" / teacher / f"toy.{teacher}.teacher-scored.train-hard-negatives.jsonl",
                    scored,
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "build_provenance_safe_agreement_teacher_prep.py"),
                    "--repo-root",
                    str(root),
                    "--prep-root",
                    str(prep),
                    "--output-root",
                    str(out),
                    "--datasets",
                    "toy",
                    "--teachers",
                    "t1,t2",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            manifest = json.loads((out / "provenance-safe-prep.manifest.json").read_text(encoding="utf-8"))
            kept_rows = [
                json.loads(line)
                for line in (out / "toy" / "toy.provenance-safe.base.train-hard-negatives.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            teacher_rows = [
                json.loads(line)
                for line in (
                    out / "toy" / "t1" / "toy.t1.provenance-safe.teacher-scored.train-hard-negatives.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertIn("rows_kept=1", completed.stdout)
        self.assertEqual(manifest["totals"], {"rows_dropped": 3, "rows_input": 4, "rows_kept": 1})
        self.assertEqual(manifest["datasets"][0]["counts"]["test_negative_doc_id"], 1)
        self.assertEqual(manifest["safety_gate"]["status"], "passed")
        self.assertEqual(manifest["safety_gate"]["test_negative_doc_ids"], 0)
        self.assertEqual(kept_rows[0]["source"], "toy")
        self.assertEqual(kept_rows[0]["split"], "train")
        self.assertEqual(kept_rows[0]["query_id"], "q-safe")
        self.assertEqual(kept_rows[0]["positive_doc_id"], "d-safe")
        self.assertEqual(kept_rows[0]["negative_doc_ids"], ["d-neg"])
        self.assertEqual(teacher_rows[0]["row_id"], kept_rows[0]["row_id"])
        self.assertEqual(teacher_rows[0]["teacher_scores"], [0.9, 0.1])


if __name__ == "__main__":
    unittest.main()
