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


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
                    "--negatives-per-query",
                    "2",
                    "--candidate-pool-size",
                    "4",
                    "--max-corpus-docs",
                    "6",
                    "--seed",
                    "7",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            leak = json.loads((out / "reports" / "leak-report.json").read_text(encoding="utf-8"))
            rows_path = out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl"
            rows = read_rows(rows_path)
            first_output = rows_path.read_text(encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(root / "out-repeat"),
                    "--max-rows",
                    "2",
                    "--negatives-per-query",
                    "2",
                    "--candidate-pool-size",
                    "4",
                    "--max-corpus-docs",
                    "6",
                    "--seed",
                    "7",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            repeated_output = (
                root
                / "out-repeat"
                / "artifacts"
                / "msmarco-passage.stagea.train-hard-negatives.jsonl"
            ).read_text(encoding="utf-8")

        self.assertIn("manifest", completed.stdout)
        self.assertEqual(manifest["schema"], "eos.msmarco_passage_stagea_rows.v1")
        self.assertEqual(manifest["legal_gates"]["release_train_allowed"], False)
        self.assertEqual(manifest["legal_gates"]["commercial_use_allowed"], False)
        self.assertEqual(manifest["legal_gates"]["train_allowed_for_research"], True)
        self.assertEqual(manifest["legal_gates"]["test_rows_train_allowed"], False)
        self.assertEqual(manifest["builder_args"]["split"], "train")
        self.assertEqual(manifest["builder_args"]["max_corpus_docs"], 6)
        self.assertEqual(manifest["counts"]["qrels_seen"]["positive_rows"], 2)
        self.assertEqual(manifest["counts"]["rows_emitted"], 2)
        self.assertEqual(manifest["counts"]["negative_candidates_emitted"], 4)
        self.assertEqual(manifest["counts"]["negative_candidates_considered"], 3)
        self.assertEqual(leak["status"], "passed")
        self.assertEqual(leak["validation"]["counts"]["rows_checked"], 2)
        self.assertEqual(first_output, repeated_output)
        for row in rows:
            self.assertIn("query", row)
            self.assertIn("positive", row)
            self.assertIn("negatives", row)
            self.assertEqual(row["roles"]["query"], "query")
            self.assertEqual(row["roles"]["positive"], "document")
            self.assertEqual(row["legal_gates"], manifest["legal_gates"])
            self.assertFalse(row["split_policy"]["test_rows_train_allowed"])
            self.assertNotIn(row["positive_doc_id"], row["negative_doc_ids"])
            self.assertNotIn("ddev", row["negative_doc_ids"])

    def test_builder_accounts_missing_query_and_positive_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            out = root / "out"
            write_jsonl(corpus_root / "queries.jsonl", [{"_id": "q1", "text": "query one"}])
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "n1", "title": "", "text": "negative one"},
                    {"_id": "n2", "title": "", "text": "negative two"},
                    {"_id": "p2", "title": "", "text": "positive two"},
                ],
            )
            write_qrels(
                corpus_root / "qrels" / "train.tsv",
                [
                    ("q-missing", "p0", 1),
                    ("q1", "p-missing", 1),
                    ("q1", "p2", 1),
                    ("q1", "n1", 0),
                ],
            )

            subprocess.run(
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
                    "3",
                    "--negatives-per-query",
                    "1",
                    "--candidate-pool-size",
                    "2",
                    "--seed",
                    "11",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            rows = read_rows(out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl")

        self.assertEqual(len(rows), 1)
        self.assertEqual(manifest["counts"]["qrels_seen"]["positive_rows"], 3)
        self.assertEqual(manifest["counts"]["qrels_seen"]["ignored_nonpositive_or_malformed"], 1)
        self.assertEqual(manifest["counts"]["missing_query_skips"], 1)
        self.assertEqual(manifest["counts"]["missing_doc_skips"], 1)
        self.assertEqual(manifest["counts"]["rows_emitted"], 1)


if __name__ == "__main__":
    unittest.main()
