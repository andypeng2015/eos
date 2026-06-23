#!/usr/bin/env python3
"""Dependency-free tests for default Eos embedder status summarization."""

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

import summarize_eos_default_embedder_status as summarizer


GIB = 1024 * 1024 * 1024


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def manifest() -> dict:
    return {
        "schema": "eos.default_embedder_asset.v1",
        "asset_id": "corkscrewdb-default-embedder",
        "model_name": "eos-embed-v1",
        "artifact": {"filename": "model.mll", "sha256": "aaa", "bytes": 10},
        "tokenizer": {"filename": "tok.mll", "sha256": "bbb", "bytes": 5},
        "source_release": {
            "directory": "runs/source",
            "package_sha256": "ccc",
            "sealed_sha256": "aaa",
        },
        "dense_short_metrics": [
            {"dataset": "a", "ndcg_at_10": 0.3, "recall_at_100": 0.6},
            {"dataset": "b", "ndcg_at_10": 0.6, "recall_at_100": 0.9},
        ],
        "dense_gate_evidence": {"status": "PASS", "checks": 2},
        "compact_policy": {
            "profile": "q4/fp16/rerank-overfetch=200",
            "method": "turboquant_ip_b4_overfetch200_fp16_rerank",
            "bits": 4,
            "rerank_overfetch": 200,
            "strict_current_compact_non_regression": True,
            "gate_evidence": {"status": "PASS", "checks": 3},
        },
        "long_context_evidence": {
            "qmsum_promoted_package_retarget": {
                "dataset": "datasets/longembed-official/qmsum",
                "run_dir": "runs/qmsum",
                "quality_claim": False,
                "eos_direct_ndcg_at_10": 0.52,
                "eos_token_span_q4_ndcg_at_10": 0.55,
                "qwen3_q4_ndcg_at_10": 0.88,
                "mxbai_q4_ndcg_at_10": 0.81,
            },
            "repo_docs_capped_cache": {
                "dataset": "datasets/longembed/repo-docs",
                "quality_claim": False,
                "eos_token_span_q4_ndcg_at_10": 0.64,
                "qwen3_q4_ndcg_at_10": 0.73,
            },
        },
    }


class SummarizeEosDefaultEmbedderStatusTest(unittest.TestCase):
    def test_macro_metric_computation(self) -> None:
        macro = summarizer.compute_macro_dense_metrics(
            [
                {"dataset": "x", "ndcg_at_10": 0.2, "recall_at_100": 0.5},
                {"dataset": "y", "ndcg_at_10": 0.4, "recall_at_100": 0.7},
            ]
        )

        self.assertAlmostEqual(macro["macro"]["ndcg_at_10"], 0.3)
        self.assertAlmostEqual(macro["macro"]["recall_at_100"], 0.6)
        self.assertEqual(macro["macro"]["dataset_count"], 2)

    def test_low_disk_with_reclaim_estimate_records_blockers_and_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = write_json(root / "manifest.json", manifest())
            reclaim_summary = write_json(
                root / "reclaim-summary.json",
                {
                    "schema": "eos.run_reclaim_summary.v1",
                    "dry_run": True,
                    "total_estimated_reclaim_bytes": 1234,
                },
            )
            reclaim_manifest = write_json(
                root / "reclaim-manifest.json",
                {"schema": "eos.run_reclaim_manifest.v1", "paths": [{"path": "runs/a"}]},
            )

            status = summarizer.build_status(
                repo_root=root,
                manifest_json=manifest_path,
                reclaim_summary_json=reclaim_summary,
                reclaim_manifest_json=reclaim_manifest,
                free_bytes=lambda path: 14 * GIB,
                clock=lambda: "2026-06-23T00:00:00Z",
            )

        self.assertFalse(status["quality_claim"])
        self.assertIn("long_context_eval_disk_blocked", status["blockers"])
        self.assertIn("training_disk_blocked", status["blockers"])
        self.assertIn(
            "apply audited reclaim manifest after explicit approval",
            status["next_actions"],
        )
        self.assertEqual(status["disk"]["reclaim"]["estimated_reclaim_bytes"], 1234)
        self.assertEqual(status["long_context"]["product_wedge_status"], "unproven_or_negative")

    def test_missing_optional_reclaim_files_warn_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = write_json(root / "manifest.json", manifest())

            status = summarizer.build_status(
                repo_root=root,
                manifest_json=manifest_path,
                reclaim_summary_json=root / "missing-summary.json",
                reclaim_manifest_json=root / "missing-manifest.json",
                free_bytes=lambda path: 25 * GIB,
            )

        self.assertEqual(status["blockers"], [])
        self.assertEqual(len(status["warnings"]), 2)
        self.assertTrue(all("optional" in warning for warning in status["warnings"]))

    def test_quality_claim_is_false_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = write_json(root / "manifest.json", manifest())

            status = summarizer.build_status(
                repo_root=root,
                manifest_json=manifest_path,
                free_bytes=lambda path: 25 * GIB,
            )

        self.assertFalse(status["quality_claim"])
        self.assertFalse(status["long_context"]["quality_claim"])

    def test_tsv_writer_emits_key_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = write_json(root / "manifest.json", manifest())
            output_tsv = root / "status.tsv"
            status = summarizer.build_status(
                repo_root=root,
                manifest_json=manifest_path,
                free_bytes=lambda path: 25 * GIB,
            )

            summarizer.write_tsv(output_tsv, status)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        pairs = {(row["section"], row["key"]): row["value"] for row in rows}
        self.assertEqual(pairs[("status", "schema")], summarizer.STATUS_SCHEMA)
        self.assertEqual(pairs[("status", "quality_claim")], "false")
        self.assertIn(("dense_short", "macro_ndcg_at_10"), pairs)
        self.assertEqual(pairs[("long_context", "product_wedge_status")], "unproven_or_negative")


if __name__ == "__main__":
    unittest.main()
