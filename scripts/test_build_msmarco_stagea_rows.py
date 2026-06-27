#!/usr/bin/env python3
"""Tests for bounded MS MARCO Passage Stage A row builder."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_msmarco_stagea_rows.py"


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


class BuildMSMarcoStageARowsTest(unittest.TestCase):
    def test_builder_emits_reader_compatible_rows_and_leak_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            out = root / "out"
            write_jsonl(
                corpus_root / "queries.jsonl",
                [
                    {"_id": "q1", "text": "query one"},
                    {"_id": "q2", "text": "query two"},
                ],
            )
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "p1", "title": "", "text": "positive one"},
                    {"_id": "p2", "title": "", "text": "positive two"},
                    {"_id": "n1", "title": "", "text": "negative one"},
                    {"_id": "n2", "title": "", "text": "negative two"},
                    {"_id": "n3", "title": "", "text": "negative three"},
                    {"_id": "ddev", "title": "", "text": "dev positive"},
                ],
            )
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1), ("q2", "p2", 1)])
            write_qrels(corpus_root / "qrels" / "dev.tsv", [("qd", "ddev", 1)])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(out),
                    "--max-rows",
                    "2",
                    "--negatives-per-row",
                    "2",
                    "--candidate-pool-size",
                    "4",
                    "--seed",
                    "7",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            leak = json.loads((out / "reports" / "leak-report.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (
                    out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertIn("manifest", completed.stdout)
        self.assertEqual(manifest["schema"], "eos.msmarco_passage_stagea_rows.v1")
        self.assertEqual(manifest["legal_gates"]["release_train_allowed"], False)
        self.assertEqual(manifest["legal_gates"]["commercial_use_allowed"], False)
        self.assertEqual(manifest["legal_gates"]["train_allowed_for_research"], True)
        self.assertEqual(manifest["counts"]["rows_emitted"], 2)
        self.assertEqual(leak["status"], "passed")
        self.assertEqual(leak["validation"]["counts"]["rows_checked"], 2)
        for row in rows:
            self.assertIn("query", row)
            self.assertIn("positive", row)
            self.assertIn("negatives", row)
            self.assertEqual(row["roles"]["query"], "query")
            self.assertEqual(row["roles"]["positive"], "document")
            self.assertEqual(row["legal_gates"], manifest["legal_gates"])
            self.assertNotIn(row["positive_doc_id"], row["negative_doc_ids"])
            self.assertNotIn("ddev", row["negative_doc_ids"])


if __name__ == "__main__":
    unittest.main()
