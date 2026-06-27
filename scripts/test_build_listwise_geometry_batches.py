#!/usr/bin/env python3
"""Dependency-free tests for listwise geometry batch builder."""

from __future__ import annotations

import json
import math
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


def base_fixture(root: Path, *, include_d3_vector: bool = True) -> dict[str, Path]:
    dataset = root / "dataset"
    hard = root / "hard.jsonl"
    doc_vectors = root / "doc-vectors.jsonl"
    query_vectors = root / "query-vectors.jsonl"
    output = root / "geometry.jsonl"
    manifest = root / "manifest.json"

    write_jsonl(
        dataset / "corpus.jsonl",
        [
            {"_id": "d1", "title": "", "text": "positive one"},
            {"_id": "d2", "title": "", "text": "shared document"},
            {"_id": "d3", "title": "", "text": "diagonal negative"},
        ],
    )
    write_jsonl(
        dataset / "queries.jsonl",
        [
            {"_id": "q1", "text": "query one"},
            {"_id": "q2", "text": "query two"},
        ],
    )
    write_jsonl(
        hard,
        [
            {
                "row_id": "row-1",
                "source": "fixture-a",
                "query": "query one",
                "positive": "positive one",
                "negatives": ["shared document"],
                "query_id": "q1",
                "positive_doc_id": "d1",
                "negative_doc_ids": ["d2"],
                "teacher_scores": [0.8, 0.2],
            },
            {
                "source": "fixture-b",
                "query": "query two",
                "positive": "shared document",
                "negatives": ["diagonal negative"],
            },
        ],
    )
    doc_rows = [
        {"id": "d1", "embedding": [1.0, 0.0]},
        {"id": "d2", "embedding": [0.0, 1.0]},
    ]
    if include_d3_vector:
        doc_rows.append({"id": "d3", "embedding": [1.0, 1.0]})
    write_jsonl(doc_vectors, doc_rows)
    write_jsonl(
        query_vectors,
        [
            {"id": "q1", "embedding": [1.0, 0.0]},
            {"id": "q2", "embedding": [1.0, 1.0]},
        ],
    )
    return {
        "dataset": dataset,
        "hard": hard,
        "doc_vectors": doc_vectors,
        "query_vectors": query_vectors,
        "output": output,
        "manifest": manifest,
    }


def run_builder(paths: dict[str, Path], *extra: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_listwise_geometry_batches.py"),
            "--hard-negatives",
            str(paths["hard"]),
            "--dataset-dir",
            str(paths["dataset"]),
            "--doc-vectors",
            str(paths["doc_vectors"]),
            "--query-vectors",
            str(paths["query_vectors"]),
            "--output-jsonl",
            str(paths["output"]),
            "--manifest",
            str(paths["manifest"]),
            "--model-id",
            "fixture-teacher",
            "--batch-size",
            "2",
            *extra,
        ],
        check=check,
        text=True,
        capture_output=True,
    )


class BuildListwiseGeometryBatchesTest(unittest.TestCase):
    def test_outputs_cosine_matrix_deduped_documents_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = base_fixture(Path(tmp))
            completed = run_builder(paths)
            rows = read_jsonl(paths["output"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        self.assertIn("examples_written=2", completed.stdout)
        self.assertEqual(len(rows), 1)
        batch = rows[0]
        self.assertEqual(batch["schema"], "eos.listwise_geometry_batch.v1")
        self.assertEqual(batch["teacher_model_id"], "fixture-teacher")
        self.assertEqual(batch["score"], "cosine")
        self.assertTrue(batch["normalized"])
        self.assertFalse(batch["release_train_allowed"])
        self.assertFalse(batch["commercial_use_allowed"])
        self.assertTrue(batch["train_allowed_for_research"])
        self.assertEqual(batch["source_counts"], {"fixture-a": 1, "fixture-b": 1})

        self.assertEqual([query["id"] for query in batch["queries"]], ["q1", "q2"])
        self.assertEqual([doc["id"] for doc in batch["documents"]], ["d1", "d2", "d3"])
        self.assertEqual([doc["role"] for doc in batch["documents"]], ["positive", "mixed", "negative"])
        self.assertEqual(len(batch["teacher_similarity"]), 2)
        self.assertEqual([len(row) for row in batch["teacher_similarity"]], [3, 3])

        root_half = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(batch["teacher_similarity"][0][0], 1.0)
        self.assertAlmostEqual(batch["teacher_similarity"][0][1], 0.0)
        self.assertAlmostEqual(batch["teacher_similarity"][0][2], root_half)
        self.assertAlmostEqual(batch["teacher_similarity"][1][0], root_half)
        self.assertAlmostEqual(batch["teacher_similarity"][1][1], root_half)
        self.assertAlmostEqual(batch["teacher_similarity"][1][2], 1.0)

        self.assertEqual(batch["examples"][0]["row_id"], "row-1")
        self.assertEqual(batch["examples"][0]["query_id"], "q1")
        self.assertEqual(batch["examples"][0]["positive_doc_id"], "d1")
        self.assertEqual(batch["examples"][0]["negative_doc_ids"], ["d2"])
        self.assertEqual(batch["examples"][1]["query_id"], "q2")
        self.assertEqual(batch["examples"][1]["row_id"], "source-line-000001")
        self.assertEqual(batch["examples"][1]["source"], "fixture-b")
        self.assertEqual(batch["examples"][1]["positive_doc_id"], "d2")
        self.assertEqual(batch["examples"][1]["negative_doc_ids"], ["d3"])

        self.assertEqual(manifest["schema"], "eos.listwise_geometry_batches_manifest.v1")
        self.assertFalse(manifest["quality_claim"])
        self.assertEqual(manifest["teacher_model_id"], "fixture-teacher")
        self.assertFalse(manifest["release_train_allowed"])
        self.assertFalse(manifest["commercial_use_allowed"])
        self.assertTrue(manifest["train_allowed_for_research"])
        self.assertEqual(manifest["coverage"]["examples_seen"], 2)
        self.assertEqual(manifest["coverage"]["examples_written"], 2)
        self.assertEqual(manifest["coverage"]["examples_dropped"], 0)
        self.assertEqual(manifest["coverage"]["batches_written"], 1)
        self.assertEqual(manifest["coverage"]["queries_written"], 2)
        self.assertEqual(manifest["coverage"]["documents_written"], 3)
        self.assertEqual(manifest["source_counts"], {"fixture-a": 1, "fixture-b": 1})
        self.assertIn("hard_negatives", manifest["sha256"])

    def test_missing_vector_fails_by_default_or_drops_with_allow_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = base_fixture(Path(tmp), include_d3_vector=False)
            strict = run_builder(paths, check=False)
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn("incomplete coverage", strict.stderr)

            completed = run_builder(paths, "--allow-missing")
            rows = read_jsonl(paths["output"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        self.assertIn("dropped=1", completed.stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["examples"]), 1)
        self.assertEqual(rows[0]["documents"], [
            {"id": "d1", "text": "positive one", "role": "positive"},
            {"id": "d2", "text": "shared document", "role": "negative"},
        ])
        self.assertEqual(manifest["coverage"]["examples_seen"], 2)
        self.assertEqual(manifest["coverage"]["examples_written"], 1)
        self.assertEqual(manifest["coverage"]["examples_dropped"], 1)
        self.assertEqual(manifest["coverage"]["missing_doc_vector"], 1)
        self.assertEqual(manifest["missing_samples"][0]["kind"], "doc_vector")
        self.assertFalse(manifest["quality_claim"])

    def test_duplicate_query_starts_new_batch_before_batch_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = base_fixture(Path(tmp))
            rows = read_jsonl(paths["hard"])
            rows.append(
                {
                    "row_id": "row-3",
                    "source": "fixture-c",
                    "query_id": "q1",
                    "positive_doc_id": "d1",
                    "negative_doc_ids": ["d3"],
                }
            )
            write_jsonl(paths["hard"], rows)

            completed = run_builder(paths, "--batch-size", "3")
            batches = read_jsonl(paths["output"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        self.assertIn("examples_written=3", completed.stdout)
        self.assertEqual(len(batches), 2)
        self.assertEqual([query["id"] for query in batches[0]["queries"]], ["q1", "q2"])
        self.assertEqual([query["id"] for query in batches[1]["queries"]], ["q1"])
        self.assertEqual(batches[1]["examples"][0]["row_id"], "row-3")
        self.assertEqual(manifest["coverage"]["batches_written"], 2)
        self.assertEqual(manifest["coverage"]["queries_written"], 3)
        self.assertEqual(manifest["source_counts"], {"fixture-a": 1, "fixture-b": 1, "fixture-c": 1})

    def test_missing_source_uses_deterministic_unknown_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = base_fixture(Path(tmp))
            write_jsonl(
                paths["hard"],
                [
                    {
                        "query_id": "q1",
                        "positive_doc_id": "d1",
                        "negative_doc_ids": ["d2"],
                    }
                ],
            )

            run_builder(paths)
            rows = read_jsonl(paths["output"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["examples"][0]["row_id"], "source-line-000000")
        self.assertEqual(rows[0]["examples"][0]["source"], "unknown")
        self.assertEqual(rows[0]["source_counts"], {"unknown": 1})
        self.assertEqual(manifest["source_counts"], {"unknown": 1})


if __name__ == "__main__":
    unittest.main()
