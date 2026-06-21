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
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("scored=1", completed.stdout)
        self.assertEqual(rows[0]["teacher_scores"], [1.0, 0.0])
        self.assertEqual(summary["coverage"]["examples_scored"], 1)
        self.assertTrue(summary["beir"]["empty_text_skip_enabled"])
        self.assertEqual(summary["beir"]["empty_doc_texts_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
