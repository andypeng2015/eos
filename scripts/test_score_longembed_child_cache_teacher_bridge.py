#!/usr/bin/env python3
"""Dependency-free tests for LongEmbed child-cache teacher-score bridge."""

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

import score_longembed_child_cache_teacher_bridge as scorer


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_dataset(root: Path, name: str, docs: dict[str, str], queries: dict[str, str]) -> Path:
    dataset_dir = root / name
    write_jsonl(
        dataset_dir / "corpus.jsonl",
        [{"_id": doc_id, "text": text, "title": ""} for doc_id, text in docs.items()],
    )
    write_jsonl(
        dataset_dir / "queries.jsonl",
        [{"_id": query_id, "text": text} for query_id, text in queries.items()],
    )
    return dataset_dir


def write_cache(
    root: Path,
    teacher: str,
    dataset: scorer.DatasetIndex,
    child_vectors: dict[str, list[float]],
    query_vectors: dict[str, list[float]],
) -> Path:
    cache_dir = root / teacher / dataset.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": dataset.name,
        "dataset_dir": str(dataset.dataset_dir),
        "model_name": teacher,
        "normalize_embeddings": True,
        "native_dim": 2,
        "output_dim": 2,
        "document_chunk_words": 2,
        "document_chunk_overlap": 0,
        "document_chunk_min_words": 1,
        "document_vector_rows": len(child_vectors),
        "query_vector_rows": len(query_vectors),
        "child_chunk_count": len(child_vectors),
        "quality_claim": False,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for child_id, vector in child_vectors.items():
        parent_id = child_id.split("#chunk-", 1)[0]
        rows.append({"parent_id": parent_id, "child_id": child_id, "embedding": vector})
    write_jsonl(cache_dir / "child-doc-vectors.jsonl", rows)
    write_jsonl(
        cache_dir / "query-vectors.jsonl",
        [{"id": query_id, "embedding": vector} for query_id, vector in query_vectors.items()],
    )
    return root / teacher


def hard_row(dataset: str, query_id: str, query: str, positive: str, negatives: list[str]) -> dict:
    return {
        "schema": "manta.embedding_quality_frontier_hard_negative.v1",
        "source": f"longembed-fusion-topk-remine-v1:{dataset}:quality-frontier:qwen+mxbai",
        "query": query,
        "positive": positive,
        "negatives": negatives,
        "metadata": {
            "query_id": query_id,
            "sources": [f"{dataset}:quality-frontier:qwen", f"{dataset}:quality-frontier:mxbai"],
            "source_files": [f"runs/root/{dataset}/comparison/hard-negatives.jsonl"],
        },
    }


class LongEmbedChildCacheTeacherBridgeTest(unittest.TestCase):
    def test_chunk_reconstruction_matches_exporter_child_ids(self) -> None:
        chunks = scorer.chunk_document_text(
            "doc",
            "one two three four five",
            chunk_words=3,
            overlap=1,
            min_words=2,
        )

        self.assertEqual(
            chunks,
            [
                scorer.DocumentChunk("doc", "doc#chunk-0000", "one two three"),
                scorer.DocumentChunk("doc", "doc#chunk-0001", "three four five"),
            ],
        )

    def test_parent_score_uses_max_child_score(self) -> None:
        query = [1.0, 0.0]
        child_vectors = [[0.0, 1.0], [0.8, 0.1], [0.25, 0.0]]

        self.assertAlmostEqual(scorer.score_parent(query, child_vectors), 0.8)

    def test_cli_routes_mixed_datasets_and_attaches_agreement_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_a = write_dataset(
                root,
                "qmsum",
                {"doc_pos": "alpha beta gamma", "doc_neg": "delta epsilon"},
                {"query_0": "find alpha"},
            )
            ds_b = write_dataset(
                root,
                "2wikimqa",
                {"doc_pos": "red blue green", "doc_neg": "black white"},
                {"query_0": "find red"},
            )
            index_a = scorer.load_dataset_index("qmsum", ds_a)
            index_b = scorer.load_dataset_index("2wikimqa", ds_b)
            teacher1 = write_cache(
                root,
                "qwen",
                index_a,
                {
                    "doc_pos#chunk-0000": [1.0, 0.0],
                    "doc_pos#chunk-0001": [0.8, 0.0],
                    "doc_neg#chunk-0000": [0.0, 1.0],
                },
                {"query_0": [1.0, 0.0]},
            )
            write_cache(
                root,
                "qwen",
                index_b,
                {
                    "doc_pos#chunk-0000": [0.0, 1.0],
                    "doc_pos#chunk-0001": [0.0, 0.8],
                    "doc_neg#chunk-0000": [1.0, 0.0],
                },
                {"query_0": [0.0, 1.0]},
            )
            teacher2 = write_cache(
                root,
                "mxbai",
                index_a,
                {
                    "doc_pos#chunk-0000": [1.0, 0.0],
                    "doc_pos#chunk-0001": [0.7, 0.0],
                    "doc_neg#chunk-0000": [0.0, 1.0],
                },
                {"query_0": [1.0, 0.0]},
            )
            write_cache(
                root,
                "mxbai",
                index_b,
                {
                    "doc_pos#chunk-0000": [0.0, 1.0],
                    "doc_pos#chunk-0001": [0.0, 0.7],
                    "doc_neg#chunk-0000": [1.0, 0.0],
                },
                {"query_0": [0.0, 1.0]},
            )
            hard_negatives = root / "hard.jsonl"
            write_jsonl(
                hard_negatives,
                [
                    hard_row("qmsum", "query_0", "find alpha", "alpha beta gamma", ["delta epsilon"]),
                    hard_row("2wikimqa", "query_0", "find red", "red blue green", ["black white"]),
                ],
            )
            output = root / "out.jsonl"
            manifest = root / "manifest.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "score_longembed_child_cache_teacher_bridge.py"),
                    "--hard-negatives",
                    str(hard_negatives),
                    "--dataset-dir",
                    f"qmsum={ds_a}",
                    "--dataset-dir",
                    f"2wikimqa={ds_b}",
                    "--teacher-cache",
                    f"qwen={teacher1}",
                    "--teacher-cache",
                    f"mxbai={teacher2}",
                    "--output-jsonl",
                    str(output),
                    "--manifest",
                    str(manifest),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            rows = read_jsonl(output)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("with_teacher_scores=2", completed.stdout)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("teacher_scores" in row for row in rows))
        self.assertEqual(summary["coverage"]["dataset_counts"]["qmsum"]["scored"], 1)
        self.assertEqual(summary["coverage"]["dataset_counts"]["2wikimqa"]["scored"], 1)
        self.assertEqual(summary["averaged_scores"]["agreement_keep_rate"], 1.0)
        self.assertEqual(summary["averaged_scores"]["positive_top1_rate"], 1.0)
        self.assertFalse(summary["quality_claim"])

    def test_cli_clears_scores_when_teacher_disagrees_on_positive_top1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = write_dataset(
                root,
                "qmsum",
                {
                    "doc_pos": "alpha beta",
                    "doc_neg": "delta epsilon",
                    "doc_pos2": "gamma theta",
                    "doc_neg2": "iota kappa",
                },
                {"query_0": "find alpha", "query_1": "find gamma"},
            )
            index = scorer.load_dataset_index("qmsum", ds)
            teacher1 = write_cache(
                root,
                "qwen",
                index,
                {
                    "doc_pos#chunk-0000": [1.0, 0.0],
                    "doc_neg#chunk-0000": [0.0, 1.0],
                    "doc_pos2#chunk-0000": [1.0, 0.0],
                    "doc_neg2#chunk-0000": [0.0, 1.0],
                },
                {"query_0": [1.0, 0.0], "query_1": [1.0, 0.0]},
            )
            teacher2 = write_cache(
                root,
                "mxbai",
                index,
                {
                    "doc_pos#chunk-0000": [0.0, 1.0],
                    "doc_neg#chunk-0000": [1.0, 0.0],
                    "doc_pos2#chunk-0000": [1.0, 0.0],
                    "doc_neg2#chunk-0000": [0.0, 1.0],
                },
                {"query_0": [1.0, 0.0], "query_1": [1.0, 0.0]},
            )
            hard_negatives = root / "hard.jsonl"
            write_jsonl(
                hard_negatives,
                [
                    hard_row("qmsum", "query_0", "find alpha", "alpha beta", ["delta epsilon"]),
                    hard_row("qmsum", "query_1", "find gamma", "gamma theta", ["iota kappa"]),
                ],
            )
            output = root / "out.jsonl"
            manifest = root / "manifest.json"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "score_longembed_child_cache_teacher_bridge.py"),
                    "--hard-negatives",
                    str(hard_negatives),
                    "--dataset-dir",
                    f"qmsum={ds}",
                    "--teacher-cache",
                    f"qwen={teacher1}",
                    "--teacher-cache",
                    f"mxbai={teacher2}",
                    "--output-jsonl",
                    str(output),
                    "--manifest",
                    str(manifest),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            rows = read_jsonl(output)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertNotIn("teacher_scores", rows[0])
        self.assertIn("teacher_scores", rows[1])
        self.assertEqual(summary["coverage"]["examples_with_teacher_scores"], 1)
        self.assertEqual(summary["coverage"]["cleared_teacher_disagreement"], 1)
        self.assertEqual(summary["averaged_scores"]["agreement_keep_rate"], 0.5)
        self.assertEqual(summary["averaged_scores"]["positive_top1_rate"], 1.0)
        self.assertEqual(summary["teacher_agreement"]["qwen"]["positive_top1"], 2)
        self.assertEqual(summary["teacher_agreement"]["mxbai"]["positive_top1"], 1)


if __name__ == "__main__":
    unittest.main()
