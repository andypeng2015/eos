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
        first = summary["datasets"][0]
        self.assertEqual(first["expected_documents"], 100)
        self.assertEqual(first["expected_queries"], 10)
        self.assertEqual(first["doc_vector_lines"], 100)
        self.assertEqual(first["query_vector_lines"], 10)
        self.assertEqual(first["vector_progress_completed"], 110)
        self.assertEqual(first["vector_progress_total"], 110)
        self.assertEqual(first["vector_progress_ratio"], 1.0)
        self.assertEqual(first["vector_progress_percent"], 100.0)
        self.assertIsNotNone(first["doc_vector_file"])
        self.assertIsNotNone(first["query_vector_file"])

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
        self.assertEqual(fiqa["expected_documents"], 57638)
        self.assertEqual(fiqa["expected_queries"], 6648)
        self.assertEqual(fiqa["doc_vector_lines"], 3)
        self.assertIsNone(fiqa["query_vector_lines"])
        self.assertEqual(fiqa["vector_progress_completed"], 3)
        self.assertEqual(fiqa["vector_progress_total"], 64286)
        self.assertAlmostEqual(fiqa["vector_progress_ratio"], 3 / 64286)
        self.assertAlmostEqual(fiqa["vector_progress_percent"], (3 / 64286) * 100.0)
        self.assertGreater(fiqa["doc_vector_file"]["size_bytes"], 0)
        self.assertIn("mtime_utc", fiqa["doc_vector_file"])
        self.assertIsNone(fiqa["query_vector_file"])
        self.assertIn("query_vectors", fiqa["missing_artifacts"])
        self.assertIn("vector_manifest", fiqa["missing_artifacts"])
        self.assertIn("dense_metrics", fiqa["missing_artifacts"])
        self.assertIn("turboquant_metrics", fiqa["missing_artifacts"])
        self.assertFalse(summary["aggregate"]["all_complete"])
        self.assertFalse(summary["aggregate"]["quality_claim"])
        self.assertFalse(summary["aggregate"]["default_alias_changed"])
        self.assertEqual(summary["aggregate"]["promotion_recommendation"], "defer")
        self.assertIn("incomplete datasets: fiqa", summary["aggregate"]["blockers"])
        policy = summary["aggregate"]["quality_policy"]
        self.assertFalse(policy["non_default_promotion_policy_pass"])
        self.assertFalse(policy["dense_policy_pass"])
        self.assertFalse(policy["q8_policy_pass"])
        self.assertEqual(policy["per_dataset"]["fiqa"]["ready"], False)
        self.assertIn("fiqa: dataset gate incomplete", policy["blockers"])

    def test_quality_policy_allows_non_default_review_without_q4_release_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_dataset(root, "scifact", 0.7, 0.9, 0.697, 0.88, 0.5, 0.8)

            summary = summarizer.build_summary(
                run_root=root,
                datasets=["scifact"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        policy = summary["aggregate"]["quality_policy"]
        self.assertTrue(policy["dense_policy_pass"])
        self.assertTrue(policy["q8_policy_pass"])
        self.assertFalse(policy["q4_release_profile_pass"])
        self.assertTrue(policy["non_default_promotion_policy_pass"])
        self.assertEqual(policy["q4_release_profile_decision"], "diagnostic_storage_only")
        self.assertEqual(summary["aggregate"]["promotion_recommendation"], "review")
        scifact_policy = policy["per_dataset"]["scifact"]
        self.assertTrue(scifact_policy["dense"]["pass"])
        self.assertTrue(scifact_policy["q8"]["pass"])
        self.assertFalse(scifact_policy["q4"]["pass"])
        self.assertEqual(scifact_policy["q4"]["release_profile_decision"], "diagnostic_storage_only")
        self.assertAlmostEqual(
            scifact_policy["dense"]["ndcg_at_10_delta_vs_current_default_dense"],
            0.7 - summarizer.DEFAULT_BASELINE_METRICS["scifact"]["ndcg_at_10"],
        )

    def test_quality_policy_q8_failure_defers_non_default_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_dataset(root, "scifact", 0.7, 0.9, 0.68, 0.88, 0.65, 0.85)

            summary = summarizer.build_summary(
                run_root=root,
                datasets=["scifact"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        policy = summary["aggregate"]["quality_policy"]
        self.assertTrue(policy["dense_policy_pass"])
        self.assertFalse(policy["q8_policy_pass"])
        self.assertFalse(policy["non_default_promotion_policy_pass"])
        self.assertEqual(summary["aggregate"]["promotion_recommendation"], "defer")
        self.assertGreater(policy["per_dataset"]["scifact"]["q8"]["ndcg_at_10_drop_vs_dense"], 0.005)

    def test_parse_baseline_metrics_override(self) -> None:
        baselines = summarizer.parse_baseline_metrics("custom:0.1:0.2,scifact:0.3:0.4")

        self.assertEqual(baselines["custom"], {"ndcg_at_10": 0.1, "recall_at_100": 0.2})
        self.assertEqual(baselines["scifact"], {"ndcg_at_10": 0.3, "recall_at_100": 0.4})

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
        self.assertEqual(q8["expected_documents"], "100")
        self.assertEqual(q8["expected_queries"], "10")
        self.assertEqual(q8["doc_vector_lines"], "100")
        self.assertEqual(q8["query_vector_lines"], "10")
        self.assertEqual(q8["vector_progress_completed"], "110")
        self.assertEqual(q8["vector_progress_total"], "110")
        self.assertEqual(q8["vector_progress_ratio"], "1.0")
        self.assertEqual(q8["vector_progress_percent"], "100.0")
        self.assertNotEqual(q8["doc_vector_size_bytes"], "")
        self.assertNotEqual(q8["doc_vector_mtime_utc"], "")
        self.assertEqual(q8["current_default_dense_ndcg_at_10"], str(summarizer.DEFAULT_BASELINE_METRICS["scifact"]["ndcg_at_10"]))
        self.assertEqual(q8["policy_ready"], "true")
        self.assertEqual(q8["policy_pass"], "true")
        self.assertNotEqual(q8["ndcg_at_10_ratio_vs_dense"], "")
        self.assertNotEqual(q8["q8_ndcg_at_10_drop_vs_dense"], "")

    def test_tsv_writer_emits_incomplete_fiqa_progress_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_lines(root / "fiqa" / "vectors" / "doc-vectors.jsonl", 7)
            output_tsv = root / "summary.tsv"
            summary = summarizer.build_summary(
                run_root=root,
                datasets=["fiqa"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        dense = rows[0]
        self.assertEqual(dense["dataset"], "fiqa")
        self.assertEqual(dense["status"], "incomplete")
        self.assertEqual(dense["expected_documents"], "57638")
        self.assertEqual(dense["expected_queries"], "6648")
        self.assertEqual(dense["doc_vector_lines"], "7")
        self.assertEqual(dense["query_vector_lines"], "")
        self.assertEqual(dense["vector_progress_completed"], "7")
        self.assertEqual(dense["vector_progress_total"], "64286")
        self.assertEqual(dense["partial_doc_vector_lines"], "7")
        self.assertNotEqual(dense["vector_progress_ratio"], "")
        self.assertNotEqual(dense["vector_progress_percent"], "")
        self.assertNotEqual(dense["doc_vector_size_bytes"], "")
        self.assertNotEqual(dense["doc_vector_mtime_utc"], "")
        self.assertEqual(dense["query_vector_size_bytes"], "")
        self.assertEqual(dense["query_vector_mtime_utc"], "")


if __name__ == "__main__":
    unittest.main()
