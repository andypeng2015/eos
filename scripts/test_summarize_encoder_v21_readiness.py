#!/usr/bin/env python3
"""Dependency-free tests for encoder-v2.1 readiness summarization."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import summarize_encoder_v21_readiness as summarizer
import summarize_bge_selected_package_gate as bge_summarizer


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


def write_binary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def encoder_metrics(ndcg: float) -> dict:
    return {
        "schema": "manta.embedding_train_metrics.v1",
        "config": {"eval_only": True, "select_metric": "retrieval_ndcg"},
        "final_eval": {
            "retrieval_ndcg_at_10": ndcg,
            "retrieval_map_at_100": 0.01,
            "retrieval_recall_at_100": 0.2,
            "pair_count": 1670,
            "positive_count": 835,
            "negative_count": 835,
        },
        "workload": {
            "actual_eval_passes": 1,
            "actual_eval_pairs": 1670,
            "actual_eval_examples": 1670,
        },
    }


def bge_manifest(dataset: str) -> dict:
    return {
        "schema": "manta.pretrained_bert_retrieval_vector_export.v1",
        "dataset": dataset,
        "package_path": bge_summarizer.DEFAULT_PACKAGE_PATH,
        "package_sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "package_identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        "documents": 100,
        "queries": 10,
        "written_documents": 100,
        "written_queries": 10,
        "native_dim": 384,
        "output_dim": 384,
        "query_prefix": bge_summarizer.DEFAULT_QUERY_PREFIX,
        "document_prefix": "",
        "pooling": "cls",
        "normalization": "l2",
        "max_length": 512,
        "quality_claim": False,
    }


def dense_metrics() -> dict:
    return {
        "schema": "manta.embedding_retrieval_metrics.v1",
        "quality": {
            "ndcg_at_10": 0.5,
            "recall_at_100": 0.7,
        },
    }


def turboquant_metrics() -> dict:
    return {
        "schema": "manta.embedding_turboquant_retrieval_metrics.v1",
        "rows": [
            {
                "bits": 8,
                "method": "turboquant_ip_b8",
                "quality": {"ndcg_at_10": 0.49, "recall_at_100": 0.69},
            },
            {
                "bits": 4,
                "method": "turboquant_ip_b4",
                "quality": {"ndcg_at_10": 0.45, "recall_at_100": 0.65},
            },
        ],
    }


def write_complete_bge_dataset(root: Path, dataset: str) -> None:
    write_lines(root / dataset / "vectors" / "doc-vectors.jsonl", 2)
    write_lines(root / dataset / "vectors" / "query-vectors.jsonl", 2)
    write_json(root / dataset / "vectors" / "manifest.json", bge_manifest(dataset))
    write_json(root / dataset / "eval" / "dense.metrics.json", dense_metrics())
    write_json(root / dataset / "eval" / "turboquant-q8-q4.metrics.json", turboquant_metrics())


def write_complete_encoder_fixture(root: Path) -> tuple[Path, Path, Path]:
    run_root = root / "encoder"
    bge_root = root / "bge"
    descriptor = root / "descriptor.md"
    descriptor.write_text("# descriptor\n", encoding="utf-8")
    write_binary(run_root / "bin" / "eos")
    for sidecar in summarizer.REQUIRED_SIDECARS:
        path = run_root / sidecar
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    write_json(
        run_root / summarizer.DEFAULT_TINY_METRICS,
        encoder_metrics(summarizer.EXPECTED_TINY_NDCG_AT_10),
    )
    write_json(
        run_root / summarizer.DEFAULT_CAPPED_METRICS,
        encoder_metrics(summarizer.EXPECTED_CAPPED_NDCG_AT_10),
    )
    for dataset in ("scifact", "nfcorpus", "fiqa"):
        write_complete_bge_dataset(bge_root, dataset)
    return run_root, bge_root, descriptor


class SummarizeEncoderV21ReadinessTest(unittest.TestCase):
    def test_ready_fixture_allows_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root, bge_root, descriptor = write_complete_encoder_fixture(Path(tmp))

            summary = summarizer.build_summary(
                run_root=run_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertTrue(summary["launch_allowed"])
        self.assertEqual(summary["blockers"], [])
        self.assertFalse(summary["quality_claim"])
        self.assertFalse(summary["training_run"])
        self.assertTrue(summary["sidecars"]["complete"])
        self.assertTrue(summary["baselines"]["tiny"]["matches_expected"])
        self.assertTrue(summary["baselines"]["capped"]["matches_expected"])
        self.assertTrue(summary["bge_gate"]["all_complete"])

    def test_incomplete_fiqa_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root, bge_root, descriptor = write_complete_encoder_fixture(Path(tmp))
            fiqa = bge_root / "fiqa"
            os.remove(fiqa / "vectors" / "query-vectors.jsonl")
            os.remove(fiqa / "vectors" / "manifest.json")
            os.remove(fiqa / "eval" / "dense.metrics.json")
            os.remove(fiqa / "eval" / "turboquant-q8-q4.metrics.json")

            summary = summarizer.build_summary(
                run_root=run_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertIn("selected BGE gate incomplete", summary["blockers"])
        self.assertIn("active or partial export marker present", summary["blockers"])
        self.assertTrue(any("incomplete datasets: fiqa" in blocker for blocker in summary["blockers"]))

    def test_missing_sidecar_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root, bge_root, descriptor = write_complete_encoder_fixture(Path(tmp))
            os.remove(run_root / "eos-embed-v1.memory.mll")

            summary = summarizer.build_summary(
                run_root=run_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertIn("eos-embed-v1.memory.mll", summary["sidecars"]["missing"])
        self.assertTrue(any(blocker.startswith("missing sidecars:") for blocker in summary["blockers"]))

    def test_baseline_mismatch_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root, bge_root, descriptor = write_complete_encoder_fixture(Path(tmp))
            write_json(run_root / summarizer.DEFAULT_CAPPED_METRICS, encoder_metrics(0.01))

            summary = summarizer.build_summary(
                run_root=run_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertFalse(summary["baselines"]["capped"]["matches_expected"])
        self.assertTrue(
            any("capped baseline retrieval_ndcg_at_10 mismatch" in blocker for blocker in summary["blockers"])
        )

    def test_tsv_writer_emits_key_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root, bge_root, descriptor = write_complete_encoder_fixture(root)
            output_tsv = root / "readiness.tsv"
            summary = summarizer.build_summary(
                run_root=run_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("summary", "launch_allowed")]["value"], "true")
        self.assertEqual(keyed[("baseline", "tiny.retrieval_ndcg_at_10")]["status"], "pass")
        self.assertEqual(
            keyed[("baseline", "capped.retrieval_ndcg_at_10")]["expected"],
            str(summarizer.EXPECTED_CAPPED_NDCG_AT_10),
        )
        self.assertEqual(keyed[("bge_gate", "all_complete")]["value"], "true")

    def test_require_ready_nonzero_when_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root, bge_root, descriptor = write_complete_encoder_fixture(root)
            os.remove(run_root / "eos-embed-v1.weights.mll")
            output_json = root / "summary.json"
            output_tsv = root / "summary.tsv"

            code = summarizer.main(
                [
                    "--run-root",
                    str(run_root),
                    "--bge-gate-root",
                    str(bge_root),
                    "--descriptor",
                    str(descriptor),
                    "--datasets",
                    "scifact,nfcorpus,fiqa",
                    "--output-json",
                    str(output_json),
                    "--output-tsv",
                    str(output_tsv),
                    "--require-ready",
                ]
            )
            json_exists = output_json.exists()
            tsv_exists = output_tsv.exists()

        self.assertEqual(code, 2)
        self.assertTrue(json_exists)
        self.assertTrue(tsv_exists)


if __name__ == "__main__":
    unittest.main()
