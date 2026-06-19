#!/usr/bin/env python3
"""Dependency-free tests for the LongEmbed-to-BEIR adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import convert_longembed_to_beir as adapter


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_splits() -> dict[str, list[dict]]:
    return {
        "corpus": [
            {"doc_id": "distractor", "title": "Distractor", "text": "not relevant"},
            {"doc_id": "relevant", "title": "Relevant", "text": "answer text"},
            {"doc_id": "relevant", "title": "Duplicate", "text": "ignored duplicate"},
        ],
        "queries": [
            {"qid": "q1", "query": "find the answer"},
            {"qid": "q2", "query": "ignored by max query cap"},
        ],
        "qrels": [
            {"qid": "q1", "doc_id": "relevant", "relevance": 2},
            {"qid": "q1", "doc_id": "distractor", "relevance": 0},
            {"qid": "q2", "doc_id": "distractor", "relevance": 1},
        ],
    }


class ConvertLongEmbedToBEIRTest(unittest.TestCase):
    def test_field_extraction_accepts_longembed_aliases(self) -> None:
        corpus = adapter.normalize_corpus_row({"id": 17, "context": "body", "title": "Title"})
        query = adapter.normalize_query_row({"query_id": "q-17", "question": "what"})
        qrel = adapter.normalize_qrel_row({"query_id": "q-17", "corpus_id": 17, "score": "3"})

        self.assertEqual(corpus, adapter.TextRow("17", "body", "Title"))
        self.assertEqual(query, adapter.TextRow("q-17", "what"))
        self.assertEqual(qrel, adapter.QrelRow("q-17", "17", 3.0))

    def test_caps_preserve_relevant_docs_and_positive_qrels_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = adapter.convert_dataset(
                "needle",
                base_splits(),
                Path(tmp),
                max_docs=1,
                max_queries=1,
            )
            out_dir = result.output_dir
            corpus = read_jsonl(out_dir / "corpus.jsonl")
            queries = read_jsonl(out_dir / "queries.jsonl")
            qrels = (out_dir / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()
            manifest = json.loads((out_dir / "dataset-manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([row["_id"] for row in corpus], ["relevant"])
        self.assertEqual([row["_id"] for row in queries], ["q1"])
        self.assertEqual(qrels, ["query-id\tcorpus-id\tscore", "q1\trelevant\t2"])
        self.assertEqual(manifest["counts"]["qrel_pairs"], 1)
        self.assertIn("No model quality claim", " ".join(manifest["caveats"]))

    def test_context_length_filter_uses_fields_and_drops_other_length_qrels(self) -> None:
        splits = {
            "corpus": [
                {"doc_id": "needle-4096-doc", "text": "short", "context_length": 4096},
                {"doc_id": "needle-8192-doc", "text": "long", "context_length": 8192},
            ],
            "queries": [
                {"qid": "needle-4096-q", "query": "short?", "context_length": 4096},
                {"qid": "needle-8192-q", "query": "long?", "context_length": 8192},
            ],
            "qrels": [
                {"qid": "needle-4096-q", "doc_id": "needle-4096-doc", "score": 1},
                {"qid": "needle-8192-q", "doc_id": "needle-8192-doc", "score": 1},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = adapter.convert_dataset("needle", splits, Path(tmp), context_length=4096)
            out_dir = result.output_dir
            corpus = read_jsonl(out_dir / "corpus.jsonl")
            queries = read_jsonl(out_dir / "queries.jsonl")
            qrels = (out_dir / "qrels" / "test.tsv").read_text(encoding="utf-8")
            manifest = json.loads((out_dir / "dataset-manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([row["_id"] for row in corpus], ["needle-4096-doc"])
        self.assertEqual([row["_id"] for row in queries], ["needle-4096-q"])
        self.assertIn("needle-4096-q\tneedle-4096-doc\t1", qrels)
        self.assertEqual(manifest["context_length_filter"]["applied_to_splits"], ["corpus", "queries", "qrels"])

    def test_missing_qrel_reference_fails_without_source_filter(self) -> None:
        splits = base_splits()
        splits["qrels"] = [{"qid": "q1", "doc_id": "missing-doc", "score": 1}]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "qrels reference missing ids"):
                adapter.convert_dataset("needle", splits, Path(tmp))

    def test_cli_converts_local_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            for split, rows in base_splits().items():
                write_jsonl(root / "needle" / f"{split}.jsonl", rows)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "convert_longembed_to_beir.py"),
                    "--input-root",
                    str(root),
                    "--dataset",
                    "needle",
                    "--output-root",
                    str(output),
                    "--max-docs",
                    "1",
                    "--max-queries",
                    "1",
                    "--split-name",
                    "dev",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("wrote needle", completed.stdout)
            self.assertTrue((output / "needle" / "corpus.jsonl").is_file())
            self.assertTrue((output / "needle" / "qrels" / "dev.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
