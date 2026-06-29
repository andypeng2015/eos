#!/usr/bin/env python3
"""Dependency-free tests for selected BGE package gate summarization."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as summarizer


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_lines(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"id": str(index), "vector": [0.1, 0.2]}) + "\n")
    return path


def manifest(dataset: str, package_sha: str | None = None, identity_sha: str | None = None) -> dict:
    return {
        "schema": "manta.pretrained_bert_retrieval_vector_export.v1",
        "dataset": dataset,
        "package_path": summarizer.DEFAULT_PACKAGE_PATH,
        "package_sha256": package_sha or summarizer.DEFAULT_PACKAGE_SHA256,
        "package_identity_sha256": identity_sha or summarizer.DEFAULT_IDENTITY_SHA256,
        "documents": 100,
        "queries": 10,
        "written_documents": 100,
        "written_queries": 10,
        "native_dim": 384,
        "output_dim": 384,
        "query_prefix": summarizer.DEFAULT_QUERY_PREFIX,
        "document_prefix": "",
        "pooling": "cls",
        "normalization": "l2",
        "max_length": 512,
        "quality_claim": False,
    }


def dense_metrics(ndcg: float, recall: float) -> dict:
    return {
        "schema": "manta.embedding_retrieval_metrics.v1",
        "quality": {
            "ndcg_at_10": ndcg,
            "recall_at_100": recall,
        },
    }


def turboquant_metrics(
    dense_ndcg: float,
    dense_recall: float,
    q8_ndcg: float,
    q8_recall: float,
    q4_ndcg: float,
    q4_recall: float,
) -> dict:
    return {
        "schema": "manta.embedding_turboquant_retrieval_metrics.v1",
        "dense": {"quality": {"ndcg_at_10": dense_ndcg, "recall_at_100": dense_recall}},
        "rows": [
            {
                "bits": 8,
                "method": "turboquant_ip_b8",
                "quality": {"ndcg_at_10": q8_ndcg, "recall_at_100": q8_recall},
                "compression_ratio": 4.0,
            },
            {
                "bits": 4,
                "method": "turboquant_ip_b4",
                "quality": {"ndcg_at_10": q4_ndcg, "recall_at_100": q4_recall},
                "compression_ratio": 8.0,
            },
        ],
    }


def write_complete_dataset(
    root: Path,
    dataset: str,
    dense_ndcg: float,
    dense_recall: float,
    q8_ndcg: float,
    q8_recall: float,
    q4_ndcg: float,
    q4_recall: float,
    package_sha: str | None = None,
    identity_sha: str | None = None,
) -> None:
    write_lines(root / dataset / "vectors" / "doc-vectors.jsonl", 2)
    write_lines(root / dataset / "vectors" / "query-vectors.jsonl", 2)
    write_json(root / dataset / "vectors" / "manifest.json", manifest(dataset, package_sha, identity_sha))
    write_json(root / dataset / "eval" / "dense.metrics.json", dense_metrics(dense_ndcg, dense_recall))
    write_json(
        root / dataset / "eval" / "turboquant-q8-q4.metrics.json",
        turboquant_metrics(dense_ndcg, dense_recall, q8_ndcg, q8_recall, q4_ndcg, q4_recall),
    )


class SummarizeBgeSelectedPackageGateTest(unittest.TestCase):
    def test_complete_fixture_computes_macros_and_q_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_dataset(root, "a", 0.2, 0.4, 0.25, 0.5, 0.15, 0.35)
            write_complete_dataset(root, "b", 0.4, 0.8, 0.45, 0.7, 0.35, 0.75)

            summary = summarizer.build_summary(
                run_root=root,
                datasets=["a", "b"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["complete_dataset_count"], 2)
        self.assertTrue(aggregate["all_complete"])
        self.assertAlmostEqual(aggregate["macro"]["dense"]["ndcg_at_10"], 0.3)
        self.assertAlmostEqual(aggregate["macro"]["dense"]["recall_at_100"], 0.6)
        self.assertAlmostEqual(aggregate["macro"]["q8"]["ndcg_at_10"], 0.35)
        self.assertAlmostEqual(aggregate["macro"]["q8"]["recall_at_100"], 0.6)
        self.assertAlmostEqual(aggregate["macro"]["q4"]["ndcg_at_10"], 0.25)
        self.assertAlmostEqual(aggregate["macro"]["q4"]["recall_at_100"], 0.55)
        self.assertAlmostEqual(summary["datasets"][0]["q8"]["ndcg_at_10_delta_vs_dense"], 0.05)
        self.assertAlmostEqual(summary["datasets"][0]["q4"]["recall_at_100_delta_vs_dense"], -0.05)

    def test_incomplete_fiqa_fixture_marks_missing_artifacts_and_defers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_dataset(root, "scifact", 0.7, 0.9, 0.71, 0.89, 0.65, 0.85)
            write_complete_dataset(root, "nfcorpus", 0.3, 0.4, 0.31, 0.41, 0.25, 0.35)
            write_lines(root / "fiqa" / "vectors" / "doc-vectors.jsonl", 3)

            summary = summarizer.build_summary(
                run_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        fiqa = next(dataset for dataset in summary["datasets"] if dataset["dataset"] == "fiqa")
        self.assertEqual(fiqa["status"], "incomplete")
        self.assertEqual(fiqa["partial_doc_vector_lines"], 3)
        self.assertIn("query_vectors", fiqa["missing_artifacts"])
        self.assertIn("vector_manifest", fiqa["missing_artifacts"])
        self.assertIn("dense_metrics", fiqa["missing_artifacts"])
        self.assertIn("turboquant_metrics", fiqa["missing_artifacts"])
        self.assertFalse(summary["aggregate"]["all_complete"])
        self.assertFalse(summary["aggregate"]["quality_claim"])
        self.assertFalse(summary["aggregate"]["default_alias_changed"])
        self.assertEqual(summary["aggregate"]["promotion_recommendation"], "defer")
        self.assertIn("incomplete datasets: fiqa", summary["aggregate"]["blockers"])

    def test_identity_mismatch_sets_consistency_false_and_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_dataset(
                root,
                "scifact",
                0.7,
                0.9,
                0.71,
                0.89,
                0.65,
                0.85,
                identity_sha="wrong",
            )

            summary = summarizer.build_summary(
                run_root=root,
                datasets=["scifact"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertFalse(summary["aggregate"]["identity_consistent"])
        self.assertEqual(summary["aggregate"]["identity_checked_manifest_count"], 1)
        self.assertEqual(summary["aggregate"]["identity_mismatched_datasets"], ["scifact"])
        self.assertIn("package identity mismatch: scifact", summary["aggregate"]["blockers"])
        self.assertEqual(summary["aggregate"]["promotion_recommendation"], "defer")

    def test_tsv_writer_emits_dense_q8_q4_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_dataset(root, "scifact", 0.7, 0.9, 0.71, 0.89, 0.65, 0.85)
            output_tsv = root / "summary.tsv"
            summary = summarizer.build_summary(
                run_root=root,
                datasets=["scifact"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual([(row["dataset"], row["storage"]) for row in rows], [
            ("scifact", "dense"),
            ("scifact", "q8"),
            ("scifact", "q4"),
        ])
        q8 = rows[1]
        self.assertEqual(q8["status"], "complete")
        self.assertEqual(q8["ndcg_at_10"], "0.71")
        self.assertEqual(q8["compression_ratio"], "4.0")
        self.assertEqual(q8["identity_match"], "true")


if __name__ == "__main__":
    unittest.main()
