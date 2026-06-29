#!/usr/bin/env python3
"""Dependency-free tests for Eos Embedder 1 release readiness summarization."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as bge_summarizer
import summarize_eos_embedder1_release_readiness as summarizer


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


def bge_manifest(dataset: str, package_sha: str | None = None, identity_sha: str | None = None) -> dict:
    return {
        "schema": "manta.pretrained_bert_retrieval_vector_export.v1",
        "dataset": dataset,
        "package_path": bge_summarizer.DEFAULT_PACKAGE_PATH,
        "package_sha256": package_sha or bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "package_identity_sha256": identity_sha or bge_summarizer.DEFAULT_IDENTITY_SHA256,
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


def write_complete_bge_dataset(
    root: Path,
    dataset: str,
    package_sha: str | None = None,
    identity_sha: str | None = None,
) -> None:
    write_lines(root / dataset / "vectors" / "doc-vectors.jsonl", 2)
    write_lines(root / dataset / "vectors" / "query-vectors.jsonl", 2)
    write_json(root / dataset / "vectors" / "manifest.json", bge_manifest(dataset, package_sha, identity_sha))
    write_json(root / dataset / "eval" / "dense.metrics.json", dense_metrics())
    write_json(root / dataset / "eval" / "turboquant-q8-q4.metrics.json", turboquant_metrics())


def write_complete_gate(root: Path) -> None:
    for dataset in ("scifact", "nfcorpus", "fiqa"):
        write_complete_bge_dataset(root, dataset)


class SummarizeEosEmbedder1ReleaseReadinessTest(unittest.TestCase):
    def test_complete_bge_gate_makes_non_default_ready_but_default_defer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertEqual(summary["schema"], summarizer.SUMMARY_SCHEMA)
        self.assertEqual(summary["non_default_candidate_status"], "ready_for_review")
        self.assertEqual(summary["default_swap_status"], "defer")
        self.assertEqual(summary["blockers"]["non_default"], [])
        self.assertIn("default provider bridge missing", summary["blockers"]["default_swap"])
        self.assertFalse(summary["quality_claim"])
        self.assertFalse(summary["default_alias_changed"])
        self.assertEqual(summary["identity"]["public_name"], "Eos Embedder 1")
        self.assertEqual(summary["identity"]["public_id"], "eos-embedder-1")
        self.assertEqual(
            summary["identity"]["candidate_provider_id"],
            "corkscrewdb-imported-bge-eos-embed-v1-candidate",
        )

    def test_incomplete_fiqa_defers_non_default_with_bge_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bge_dataset(root, "scifact")
            write_complete_bge_dataset(root, "nfcorpus")
            write_lines(root / "fiqa" / "vectors" / "doc-vectors.jsonl", 3)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        self.assertTrue(any("selected BGE gate incomplete" in blocker for blocker in summary["blockers"]["non_default"]))
        self.assertTrue(any("fiqa" in blocker.lower() for blocker in summary["blockers"]["non_default"]))
        self.assertFalse(summary["bge_gate"]["all_complete"])

    def test_identity_mismatch_defers_non_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bge_dataset(root, "scifact", identity_sha="wrong")
            write_complete_bge_dataset(root, "nfcorpus")
            write_complete_bge_dataset(root, "fiqa")

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        self.assertFalse(summary["bge_gate"]["identity_consistent"])
        self.assertIn("scifact", summary["bge_gate"]["identity_mismatched_datasets"])
        self.assertTrue(
            any("identity inconsistent" in blocker for blocker in summary["blockers"]["non_default"])
        )

    def test_require_non_default_ready_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "complete"
            incomplete = root / "incomplete"
            write_complete_gate(complete)
            write_complete_bge_dataset(incomplete, "scifact")
            write_complete_bge_dataset(incomplete, "nfcorpus")

            complete_code = summarizer.main(
                [
                    "--bge-gate-root",
                    str(complete),
                    "--datasets",
                    "scifact,nfcorpus,fiqa",
                    "--output-json",
                    str(root / "complete.json"),
                    "--output-tsv",
                    str(root / "complete.tsv"),
                    "--require-non-default-ready",
                ]
            )
            incomplete_code = summarizer.main(
                [
                    "--bge-gate-root",
                    str(incomplete),
                    "--datasets",
                    "scifact,nfcorpus,fiqa",
                    "--output-json",
                    str(root / "incomplete.json"),
                    "--output-tsv",
                    str(root / "incomplete.tsv"),
                    "--require-non-default-ready",
                ]
            )

        self.assertEqual(complete_code, 0)
        self.assertEqual(incomplete_code, 2)

    def test_require_default_ready_fails_even_when_bge_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)

            code = summarizer.main(
                [
                    "--bge-gate-root",
                    str(root),
                    "--datasets",
                    "scifact,nfcorpus,fiqa",
                    "--output-json",
                    str(root / "summary.json"),
                    "--output-tsv",
                    str(root / "summary.tsv"),
                    "--require-default-ready",
                ]
            )

        self.assertEqual(code, 2)

    def test_tsv_writer_emits_key_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            output_tsv = root / "readiness.tsv"
            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(
            keyed[("summary", "non_default_candidate_status")]["value"],
            "ready_for_review",
        )
        self.assertEqual(keyed[("summary", "default_swap_status")]["value"], "defer")
        self.assertEqual(keyed[("summary", "quality_claim")]["value"], "false")
        self.assertEqual(keyed[("identity", "public_id")]["value"], "eos-embedder-1")
        self.assertEqual(keyed[("bge_gate", "all_complete")]["value"], "true")
        self.assertEqual(keyed[("default_swap_gate", "default_provider_bridge")]["status"], "missing")

    def test_scan_paths_flags_public_v6_and_allows_internal_run_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root / "gate")
            docs = root / "docs"
            docs.mkdir()
            (docs / "bad.md").write_text("Public release v6 is ready.\n", encoding="utf-8")
            (docs / "allowed.md").write_text(
                "v6 is an internal run label.\nThe v6 experiment remains private.\n",
                encoding="utf-8",
            )
            ignored = root / "runs" / "generated.md"
            ignored.parent.mkdir()
            ignored.write_text("Public release v6 is ignored here.\n", encoding="utf-8")

            summary = summarizer.build_summary(
                bge_gate_root=root / "gate",
                datasets=["scifact", "nfcorpus", "fiqa"],
                scan_paths=[docs, ignored.parent],
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        matches = summary["public_name_hygiene"]["matches"]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["path"].endswith("bad.md"))
        self.assertIn("public-name hygiene", summary["blockers"]["non_default"][0])


if __name__ == "__main__":
    unittest.main()
