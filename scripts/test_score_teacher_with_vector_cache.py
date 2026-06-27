#!/usr/bin/env python3
"""Dependency-free tests for vector-cache teacher score bridge."""

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


class ScoreTeacherWithVectorCacheTest(unittest.TestCase):
    def test_scores_jsonl_dedupes_with_stable_row_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            hard = root / "hard.jsonl"
            doc_vectors = root / "doc-vectors.jsonl"
            query_vectors = root / "query-vectors.jsonl"
            output = root / "scored.jsonl"
            scores = root / "scores.jsonl"
            manifest = root / "manifest.json"

            write_jsonl(
                dataset / "corpus.jsonl",
                [
                    {"_id": "pos-a", "title": "", "text": "same positive"},
                    {"_id": "neg-a", "title": "", "text": "same negative"},
                    {"_id": "pos-b", "title": "", "text": "same positive"},
                    {"_id": "neg-b", "title": "", "text": "same negative"},
                ],
            )
            write_jsonl(dataset / "queries.jsonl", [{"_id": "q", "text": "same query"}])
            write_jsonl(
                hard,
                [
                    {
                        "source": "fixture",
                        "row_id": "row-a",
                        "query_id": "q",
                        "positive_doc_id": "pos-a",
                        "negative_doc_ids": ["neg-a"],
                        "query": "same query",
                        "positive": "same positive",
                        "negatives": ["same negative"],
                    },
                    {
                        "source": "fixture",
                        "row_id": "row-b",
                        "query_id": "q",
                        "positive_doc_id": "pos-b",
                        "negative_doc_ids": ["neg-b"],
                        "query": "same query",
                        "positive": "same positive",
                        "negatives": ["same negative"],
                    },
                ],
            )
            write_jsonl(
                doc_vectors,
                [
                    {"id": "pos-a", "embedding": [1.0, 0.0]},
                    {"id": "neg-a", "embedding": [0.0, 1.0]},
                ],
            )
            write_jsonl(query_vectors, [{"id": "q", "embedding": [1.0, 0.0]}])

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "score_teacher_with_vector_cache.py"),
                    "--hard-negatives",
                    str(hard),
                    "--dataset-dir",
                    str(dataset),
                    "--doc-vectors",
                    str(doc_vectors),
                    "--query-vectors",
                    str(query_vectors),
                    "--output-jsonl",
                    str(output),
                    "--scores-jsonl",
                    str(scores),
                    "--manifest",
                    str(manifest),
                    "--skip-empty-beir-text",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            score_rows = read_jsonl(scores)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(len(score_rows), 4)
        self.assertEqual(
            [(row["row_id"], row["candidate_doc_id"], row["candidate_index"]) for row in score_rows],
            [
                ("row-a", "pos-a", 0),
                ("row-a", "neg-a", 1),
                ("row-b", "pos-b", 0),
                ("row-b", "neg-b", 1),
            ],
        )
        self.assertEqual(summary["coverage"]["examples_scored"], 2)
        self.assertEqual(summary["coverage"]["import_score_rows"], 4)

    def test_skip_empty_beir_text_is_explicit_and_scores_exportable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            hard = root / "hard.jsonl"
            doc_vectors = root / "doc-vectors.jsonl"
            query_vectors = root / "query-vectors.jsonl"
            output = root / "scored.jsonl"
            scores = root / "scores.jsonl"
            manifest = root / "manifest.json"

            write_jsonl(
                dataset / "corpus.jsonl",
                [
                    {"_id": "empty", "title": "", "text": ""},
                    {"_id": "pos", "title": "", "text": "positive text"},
                    {"_id": "neg", "title": "", "text": "negative text"},
                ],
            )
            write_jsonl(dataset / "queries.jsonl", [{"_id": "q", "text": "find positive"}])
            write_jsonl(
                hard,
                [
                    {
                        "source": "fixture",
                        "row_id": "row-1",
                        "query_id": "q",
                        "positive_doc_id": "pos",
                        "negative_doc_ids": ["neg"],
                        "query": "find positive",
                        "positive": "positive text",
                        "negatives": ["negative text"],
                    }
                ],
            )
            write_jsonl(
                doc_vectors,
                [
                    {"id": "pos", "embedding": [1.0, 0.0]},
                    {"id": "neg", "embedding": [0.0, 1.0]},
                ],
            )
            write_jsonl(query_vectors, [{"id": "q", "embedding": [1.0, 0.0]}])

            strict = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "score_teacher_with_vector_cache.py"),
                    "--hard-negatives",
                    str(hard),
                    "--dataset-dir",
                    str(dataset),
                    "--doc-vectors",
                    str(doc_vectors),
                    "--query-vectors",
                    str(query_vectors),
                    "--output-jsonl",
                    str(output),
                    "--scores-jsonl",
                    str(scores),
                    "--manifest",
                    str(manifest),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn("empty text", strict.stderr)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "score_teacher_with_vector_cache.py"),
                    "--hard-negatives",
                    str(hard),
                    "--dataset-dir",
                    str(dataset),
                    "--doc-vectors",
                    str(doc_vectors),
                    "--query-vectors",
                    str(query_vectors),
                    "--output-jsonl",
                    str(output),
                    "--scores-jsonl",
                    str(scores),
                    "--manifest",
                    str(manifest),
                    "--skip-empty-beir-text",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            rows = read_jsonl(output)
            score_rows = read_jsonl(scores)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("scored=1", completed.stdout)
        self.assertEqual(rows[0]["teacher_scores"], [1.0, 0.0])
        self.assertEqual(
            [
                {
                    "row_id": row["row_id"],
                    "query_id": row["query_id"],
                    "candidate_doc_id": row["candidate_doc_id"],
                    "candidate_index": row["candidate_index"],
                    "role": row["role"],
                }
                for row in score_rows
            ],
            [
                {
                    "row_id": "row-1",
                    "query_id": "q",
                    "candidate_doc_id": "pos",
                    "candidate_index": 0,
                    "role": "positive",
                },
                {
                    "row_id": "row-1",
                    "query_id": "q",
                    "candidate_doc_id": "neg",
                    "candidate_index": 1,
                    "role": "negative",
                },
            ],
        )
        self.assertEqual(summary["coverage"]["examples_scored"], 1)
        self.assertTrue(summary["beir"]["empty_text_skip_enabled"])
        self.assertEqual(summary["beir"]["empty_doc_texts_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
