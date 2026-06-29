#!/usr/bin/env python3
"""Dependency-free tests for hard-negative BEIR materialization."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "test_rows_train_allowed": False,
}
LEGACY_LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MaterializeHardNegativesBEIRTest(unittest.TestCase):
    def run_materializer(self, hard: Path, out: Path, manifest: Path, *, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "materialize_hard_negatives_beir.py"),
                "--input-jsonl",
                str(hard),
                "--output-dir",
                str(out),
                "--dataset",
                "fixture",
                "--split",
                "train",
                "--manifest",
                str(manifest),
            ],
            check=check,
            text=True,
            capture_output=True,
        )

    def test_materializes_canonical_ids_aliases_qrels_and_legal_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hard = root / "hard.jsonl"
            out = root / "beir"
            manifest = root / "manifest.json"
            write_jsonl(
                hard,
                [
                    {
                        "row_id": "r1",
                        "source": "fixture",
                        "query_id": "q2",
                        "positive_doc_id": "d2",
                        "negative_doc_ids": ["n2"],
                        "query": "same query",
                        "positive": "same positive",
                        "negatives": ["unique negative"],
                        "legal_gates": LEGAL_GATES,
                    },
                    {
                        "row_id": "r2",
                        "source": "fixture",
                        "query_id": "q1",
                        "positive_doc_id": "d1",
                        "negative_doc_ids": ["n3"],
                        "query": "same query",
                        "positive": "same positive",
                        "negatives": ["other negative"],
                        "legal_gates": LEGAL_GATES,
                    },
                ],
            )

            self.run_materializer(hard, out, manifest)
            summary = json.loads(manifest.read_text(encoding="utf-8"))
            corpus = read_jsonl(out / "corpus.jsonl")
            queries = read_jsonl(out / "queries.jsonl")
            qrels = (out / "qrels" / "train.tsv").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["legal_gates"], LEGAL_GATES)
        self.assertEqual(summary["legal_gate_counts"]["canonical_four_gate_rows"], 2)
        self.assertEqual(summary["counts"]["rows"], 2)
        self.assertEqual(summary["counts"]["unique_queries"], 1)
        self.assertEqual(summary["counts"]["unique_docs"], 3)
        self.assertEqual(summary["counts"]["qrel_pairs"], 1)
        self.assertEqual(summary["counts"]["duplicate_query_texts"], 1)
        self.assertEqual(summary["counts"]["duplicate_doc_texts"], 1)
        self.assertEqual(queries, [{"_id": "q1", "text": "same query"}])
        self.assertIn({"_id": "d1", "title": "", "text": "same positive"}, corpus)
        self.assertEqual(qrels, ["query-id\tcorpus-id\tscore", "q1\td1\t1"])
        self.assertEqual(summary["aliases"]["queries"], {"q1": ["q2"]})
        self.assertEqual(summary["aliases"]["docs"], {"d1": ["d2"]})
        self.assertEqual(summary["row_to_beir_id_maps"][0]["beir_query_id"], "q1")
        self.assertEqual(summary["row_to_beir_id_maps"][0]["beir_positive_doc_id"], "d1")

    def test_accepts_legacy_three_gate_rows_with_omission_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hard = root / "hard.jsonl"
            out = root / "beir"
            manifest = root / "manifest.json"
            write_jsonl(
                hard,
                [
                    {
                        "row_id": "r1",
                        "source": "fixture",
                        "query_id": "q1",
                        "positive_doc_id": "d1",
                        "negative_doc_ids": ["n1"],
                        "query": "legacy query",
                        "positive": "legacy positive",
                        "negatives": ["legacy negative"],
                        "legal_gates": LEGACY_LEGAL_GATES,
                    },
                ],
            )

            self.run_materializer(hard, out, manifest)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(summary["legal_gates"], LEGAL_GATES)
        self.assertEqual(summary["legal_gate_counts"]["legacy_three_gate_rows"], 1)
        self.assertEqual(summary["legal_gate_counts"]["legacy_test_rows_train_allowed_omitted"], 1)
        self.assertNotIn("commercial_use_allowed_mismatch", summary["legal_gate_counts"])
        self.assertEqual(summary["counts"]["rows"], 1)
        self.assertEqual(len(summary["row_to_beir_id_maps"]), 1)

    def test_rejects_unsafe_or_unusable_legal_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hard = root / "hard.jsonl"
            out = root / "beir"
            manifest = root / "manifest.json"
            unsafe = dict(LEGAL_GATES)
            unsafe["commercial_use_allowed"] = True
            write_jsonl(
                hard,
                [
                    {
                        "row_id": "r1",
                        "source": "fixture",
                        "query_id": "q1",
                        "positive_doc_id": "d1",
                        "negative_doc_ids": ["n1"],
                        "query": "unsafe query",
                        "positive": "unsafe positive",
                        "negatives": ["unsafe negative"],
                        "legal_gates": unsafe,
                    },
                    {
                        "row_id": "r2",
                        "source": "fixture",
                        "query_id": "q2",
                        "positive_doc_id": "d2",
                        "negative_doc_ids": ["n2"],
                        "query": "missing core query",
                        "positive": "missing core positive",
                        "negatives": ["missing core negative"],
                        "legal_gates": {"release_train_allowed": False, "commercial_use_allowed": False},
                    },
                ],
            )

            result = self.run_materializer(hard, out, manifest, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("legal gate mismatch", result.stderr)
        self.assertIn("commercial_use_allowed_mismatch", result.stderr)
        self.assertIn("train_allowed_for_research_missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
