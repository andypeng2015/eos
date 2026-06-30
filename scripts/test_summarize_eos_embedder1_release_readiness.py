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
            "ndcg_at_10": 0.7,
            "recall_at_100": 0.9,
        },
    }


def turboquant_metrics() -> dict:
    return {
        "schema": "manta.embedding_turboquant_retrieval_metrics.v1",
        "rows": [
            {
                "bits": 8,
                "method": "turboquant_ip_b8",
                "quality": {"ndcg_at_10": 0.697, "recall_at_100": 0.89},
            },
            {
                "bits": 4,
                "method": "turboquant_ip_b4",
                "quality": {"ndcg_at_10": 0.65, "recall_at_100": 0.85},
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


def candidate_smoke_manifest() -> dict:
    return {
        "schema": "manta.imported_bge_eos_embed_v1_candidate_smoke.v1",
        "candidate_public_id": "eos-embedder-1",
        "candidate_model_name": "eos-embedder-1",
        "candidate_display_name": "Eos Embedder 1",
        "legacy_model_name": "eos-embed-v1",
        "candidate_status": "non_default_reference_candidate",
        "source_model": "BAAI/bge-small-en-v1.5",
        "quality_claim": False,
        "default_alias_changed": False,
        "package": {
            "sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
            "identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        },
        "role_contract": {
            "query_prefix": bge_summarizer.DEFAULT_QUERY_PREFIX,
            "document_prefix": bge_summarizer.DEFAULT_DOCUMENT_PREFIX,
            "pooling": "cls",
            "max_length": 512,
        },
        "direct_embed_smoke": {
            "query_norm": 1.00000001,
            "document_norm": 0.99999999,
        },
        "caveats": ["fixture caveat"],
    }


def role_aware_provider_smoke_manifest() -> dict:
    fingerprint = summarizer.backend_fingerprint(
        bge_summarizer.DEFAULT_PACKAGE_SHA256,
        bge_summarizer.DEFAULT_IDENTITY_SHA256,
    )
    return {
        "schema": "eos.imported_bge_role_aware_provider_smoke.v1",
        "public_id": "eos-embedder-1",
        "public_model_name": "eos-embedder-1",
        "display_name": "Eos Embedder 1",
        "legacy_model_name": "eos-embed-v1",
        "source_model": "BAAI/bge-small-en-v1.5",
        "provider_id": "corkscrewdb-imported-bge-eos-embed-v1-candidate",
        "dim": 384,
        "backend_fingerprint": fingerprint,
        "candidate_package_sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "candidate_identity": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        "quality_claim": False,
        "default_alias_changed": False,
        "all_top1_ok": True,
        "query_role_calls": 3,
        "document_role_calls": 4,
        "encode_calls": 3,
        "encode_batch_calls": 4,
        "db_manifest_embedding": {
            "id": "corkscrewdb-imported-bge-eos-embed-v1-candidate",
            "dim": 384,
            "backend_fingerprint": fingerprint,
        },
    }


def corkscrewdb_serving_smoke_manifest() -> dict:
    return {
        "schema": "eos.imported_bge_serving_candidate_manifest.v1",
        "candidate": {
            "public_identity": "eos-embedder-1",
            "display_name": "Eos Embedder 1",
            "model_name": "eos-embedder-1",
            "legacy_model_name": "eos-embed-v1",
            "quality_claim": False,
            "default_alias_changed": False,
        },
        "package": {
            "sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
            "identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        },
        "corkscrewdb_smoke": {
            "quantized_only": True,
            "index_type": "flat",
            "layout": "single_parent_vectors",
        },
        "offline_comparison": {
            "q4": {
                "delta": {"ndcg_at_10": 0.0, "recall_at_100": 0.0},
                "corkscrew": {"p95_ms": 12.0},
            },
            "q8": {
                "delta": {"ndcg_at_10": 0.0, "recall_at_100": 0.0},
                "corkscrew": {"p95_ms": 24.0},
            },
        },
        "caveats": ["fixture serving caveat"],
    }


def write_valid_non_default_evidence(root: Path) -> dict[str, Path]:
    evidence_root = root / "evidence"
    paths = {
        "candidate_smoke_evidence": write_json(evidence_root / "candidate-smoke.json", candidate_smoke_manifest()),
        "role_aware_provider_smoke_evidence": write_json(
            evidence_root / "role-aware-provider-smoke.json",
            role_aware_provider_smoke_manifest(),
        ),
        "corkscrewdb_serving_smoke_evidence": write_json(
            evidence_root / "corkscrewdb-serving-smoke.json",
            corkscrewdb_serving_smoke_manifest(),
        ),
    }
    return paths


def default_provider_bridge_manifest() -> dict:
    fingerprint = summarizer.backend_fingerprint(
        bge_summarizer.DEFAULT_PACKAGE_SHA256,
        bge_summarizer.DEFAULT_IDENTITY_SHA256,
    )
    return {
        "schema": "eos.embedder1_default_provider_bridge_evidence.v1",
        "package_sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        "default_provider_id": "corkscrewdb-imported-bge-eos-embed-v1-candidate",
        "dim": 384,
        "backend_fingerprint": fingerprint,
        "role_contract": {
            "query_prefix": bge_summarizer.DEFAULT_QUERY_PREFIX,
            "document_prefix": bge_summarizer.DEFAULT_DOCUMENT_PREFIX,
            "pooling": "cls",
            "normalization": "l2",
            "max_length": 512,
        },
        "default_alias_changed": False,
        "dry_run": True,
        "legacy_default_preserved": True,
    }


def default_release_smoke_manifest() -> dict:
    return {
        "schema": "eos.embedder1_default_release_smoke.v1",
        "package_sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        "default_provider_id": "corkscrewdb-imported-bge-eos-embed-v1-candidate",
        "dim": 384,
        "query_role_smoke_passed": True,
        "document_role_smoke_passed": True,
        "new_384d_db_smoke_passed": True,
        "mismatch_smoke_passed": True,
        "quality_claim": False,
    }


def legacy_256d_migration_policy_smoke_manifest() -> dict:
    return {
        "schema": "eos.embedder1_legacy_256d_migration_policy_smoke.v1",
        "legacy_256d_open_passed": True,
        "legacy_provider_available": True,
        "mismatch_rejects_clearly": True,
        "in_place_upgrade_supported": False,
        "reembed_rebuild_required": True,
    }


def throughput_gate_manifest() -> dict:
    return {
        "schema": "eos.embedder1_startup_load_encode_throughput_gate.v1",
        "package_sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        "cold_load_ms": 4500.0,
        "first_query_encode_ms": 25.0,
        "warm_batch64_docs_per_second": 16.0,
        "peak_rss_mb": 512.0,
        "explicit_owner_exception": False,
    }


def default_asset_size_policy_manifest() -> dict:
    return {
        "schema": "eos.embedder1_default_asset_size_policy.v1",
        "package_sha256": bge_summarizer.DEFAULT_PACKAGE_SHA256,
        "identity_sha256": bge_summarizer.DEFAULT_IDENTITY_SHA256,
        "package_bytes": 179_307_385,
        "default_in_repo_asset_bytes": 5_271_971,
        "selected_policy": "explicit_external_package",
        "large_default_asset_approved": False,
    }


def write_valid_default_evidence(root: Path) -> dict[str, Path]:
    evidence_root = root / "default-evidence"
    return {
        "default_provider_bridge": write_json(evidence_root / "default-provider-bridge.json", default_provider_bridge_manifest()),
        "default_release_smoke": write_json(evidence_root / "default-release-smoke.json", default_release_smoke_manifest()),
        "legacy_256d_migration_policy_smoke": write_json(
            evidence_root / "legacy-256d-migration-policy-smoke.json",
            legacy_256d_migration_policy_smoke_manifest(),
        ),
        "startup_load_encode_throughput_gate": write_json(evidence_root / "throughput-gate.json", throughput_gate_manifest()),
        "default_asset_size_policy": write_json(
            evidence_root / "default-asset-size-policy.json",
            default_asset_size_policy_manifest(),
        ),
    }


def evidence_cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--candidate-smoke-evidence",
        str(paths["candidate_smoke_evidence"]),
        "--role-aware-provider-smoke-evidence",
        str(paths["role_aware_provider_smoke_evidence"]),
        "--corkscrewdb-serving-smoke-evidence",
        str(paths["corkscrewdb_serving_smoke_evidence"]),
    ]


def default_evidence_cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--default-provider-bridge-evidence",
        str(paths["default_provider_bridge"]),
        "--default-release-smoke-evidence",
        str(paths["default_release_smoke"]),
        "--legacy-256d-migration-evidence",
        str(paths["legacy_256d_migration_policy_smoke"]),
        "--throughput-gate-evidence",
        str(paths["startup_load_encode_throughput_gate"]),
        "--default-asset-size-policy-evidence",
        str(paths["default_asset_size_policy"]),
    ]


class SummarizeEosEmbedder1ReleaseReadinessTest(unittest.TestCase):
    def test_complete_bge_gate_makes_non_default_ready_but_default_defer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            evidence = write_valid_non_default_evidence(root)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertEqual(summary["schema"], summarizer.SUMMARY_SCHEMA)
        self.assertEqual(summary["non_default_candidate_status"], "ready_for_review")
        self.assertEqual(summary["default_swap_status"], "defer")
        self.assertEqual(summary["blockers"]["non_default"], [])
        self.assertTrue(summary["bge_gate"]["non_default_promotion_policy_pass"])
        self.assertTrue(summary["bge_gate"]["dense_policy_pass"])
        self.assertTrue(summary["bge_gate"]["q8_policy_pass"])
        self.assertTrue(summary["non_default_evidence"]["all_valid"])
        self.assertEqual(summary["non_default_evidence"]["gates"]["candidate_smoke"]["status"], "pass")
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
            evidence = write_valid_non_default_evidence(root)
            output_tsv = root / "readiness.tsv"

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
            )
            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        self.assertEqual(summary["default_swap_status"], "defer")
        self.assertTrue(any("selected BGE gate incomplete" in blocker for blocker in summary["blockers"]["non_default"]))
        self.assertTrue(any("fiqa" in blocker.lower() for blocker in summary["blockers"]["non_default"]))
        self.assertFalse(summary["bge_gate"]["all_complete"])
        fiqa = next(dataset for dataset in summary["bge_gate"]["datasets"] if dataset["dataset"] == "fiqa")
        self.assertEqual(fiqa["expected_documents"], 57638)
        self.assertEqual(fiqa["expected_queries"], 6648)
        self.assertEqual(fiqa["doc_vector_lines"], 3)
        self.assertIsNone(fiqa["query_vector_lines"])
        self.assertEqual(fiqa["vector_progress_completed"], 3)
        self.assertEqual(fiqa["vector_progress_total"], 64286)
        self.assertAlmostEqual(fiqa["vector_progress_percent"], (3 / 64286) * 100.0)
        self.assertGreater(fiqa["doc_vector_size_bytes"], 0)
        self.assertIsNotNone(fiqa["doc_vector_mtime_utc"])
        progress = summary["bge_gate"]["incomplete_dataset_progress"][0]
        self.assertEqual(progress["dataset"], "fiqa")
        self.assertEqual(progress["vector_progress_completed"], 3)
        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("bge_progress", "fiqa.vector_progress_completed")]["value"], "3")
        self.assertEqual(keyed[("bge_progress", "fiqa.vector_progress_total")]["value"], "64286")
        self.assertEqual(keyed[("bge_progress", "fiqa.doc_vector_lines")]["value"], "3")
        self.assertEqual(keyed[("bge_progress", "fiqa.expected_queries")]["value"], "6648")

    def test_identity_mismatch_defers_non_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bge_dataset(root, "scifact", identity_sha="wrong")
            write_complete_bge_dataset(root, "nfcorpus")
            write_complete_bge_dataset(root, "fiqa")
            evidence = write_valid_non_default_evidence(root)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        self.assertFalse(summary["bge_gate"]["identity_consistent"])
        self.assertIn("scifact", summary["bge_gate"]["identity_mismatched_datasets"])
        self.assertTrue(
            any("identity inconsistent" in blocker for blocker in summary["blockers"]["non_default"])
        )

    def test_complete_bge_gate_quality_failure_defers_non_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            write_json(root / "fiqa" / "eval" / "dense.metrics.json", dense_metrics() | {"quality": {"ndcg_at_10": 0.01, "recall_at_100": 0.02}})
            bad_turboquant = turboquant_metrics()
            bad_turboquant["rows"][0]["quality"]["ndcg_at_10"] = 0.005
            write_json(root / "fiqa" / "eval" / "turboquant-q8-q4.metrics.json", bad_turboquant)
            evidence = write_valid_non_default_evidence(root)
            output_tsv = root / "readiness.tsv"

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
            )
            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        self.assertTrue(summary["bge_gate"]["all_complete"])
        self.assertTrue(summary["bge_gate"]["identity_consistent"])
        self.assertFalse(summary["bge_gate"]["non_default_promotion_policy_pass"])
        self.assertFalse(summary["bge_gate"]["dense_policy_pass"])
        self.assertFalse(summary["bge_gate"]["q8_policy_pass"])
        self.assertTrue(summary["non_default_evidence"]["all_valid"])
        blockers = "\n".join(summary["blockers"]["non_default"])
        self.assertIn("selected BGE non-default quality policy failed", blockers)
        self.assertIn("fiqa: dense below current-default dense baseline", blockers)
        self.assertIn("fiqa: q8 is not near dense", blockers)
        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("bge_quality_policy", "non_default_promotion_policy_pass")]["status"], "block")
        self.assertEqual(keyed[("bge_quality_policy_detail", "fiqa.dense")]["status"], "block")
        self.assertEqual(keyed[("bge_quality_policy_detail", "fiqa.q8")]["status"], "block")

    def test_require_non_default_ready_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "complete"
            incomplete = root / "incomplete"
            write_complete_gate(complete)
            write_complete_bge_dataset(incomplete, "scifact")
            write_complete_bge_dataset(incomplete, "nfcorpus")
            evidence = write_valid_non_default_evidence(root)

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
                + evidence_cli_args(evidence)
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
                + evidence_cli_args(evidence)
            )

        self.assertEqual(complete_code, 0)
        self.assertEqual(incomplete_code, 2)

    def test_require_default_ready_fails_even_when_bge_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            evidence = write_valid_non_default_evidence(root)

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
                + evidence_cli_args(evidence)
            )

        self.assertEqual(code, 2)

    def test_require_default_ready_passes_with_valid_default_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)

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
                + evidence_cli_args(evidence)
                + default_evidence_cli_args(default_evidence)
            )

        self.assertEqual(code, 0)

    def test_tsv_writer_emits_key_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            evidence = write_valid_non_default_evidence(root)
            output_tsv = root / "readiness.tsv"
            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
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
        self.assertEqual(keyed[("bge_quality_policy", "non_default_promotion_policy_pass")]["status"], "pass")
        self.assertEqual(keyed[("bge_quality_policy_detail", "fiqa.q8")]["status"], "pass")
        self.assertEqual(keyed[("default_swap_gate", "default_provider_bridge")]["status"], "missing")
        self.assertEqual(keyed[("non_default_evidence", "candidate_smoke")]["status"], "pass")
        self.assertIn("query_norm", keyed[("non_default_evidence_detail", "candidate_smoke")]["value"])

    def test_valid_default_evidence_makes_default_ready_when_other_gates_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                default_gate_evidence_paths={key: str(value) for key, value in default_evidence.items()},
                **non_default_evidence,
            )

        self.assertEqual(summary["non_default_candidate_status"], "ready_for_review")
        self.assertEqual(summary["default_swap_status"], "ready_for_review")
        self.assertTrue(summary["default_swap_gates"]["all_valid"])
        self.assertEqual(summary["blockers"]["default_swap"], [])
        self.assertTrue(all(gate["status"] == "pass" for gate in summary["default_swap_gates"]["gates"].values()))

    def test_malformed_default_evidence_fails_with_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)
            default_evidence["default_release_smoke"].write_text("{", encoding="utf-8")

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                default_gate_evidence_paths={key: str(value) for key, value in default_evidence.items()},
                **non_default_evidence,
            )

        self.assertEqual(summary["default_swap_status"], "defer")
        self.assertEqual(summary["default_swap_gates"]["gates"]["default_release_smoke"]["status"], "fail")
        blockers = "\n".join(summary["blockers"]["default_swap"])
        self.assertIn("default release smoke evidence failed validation", blockers)
        self.assertIn("invalid JSON", blockers)

    def test_default_evidence_identity_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)
            bad_bridge = default_provider_bridge_manifest()
            bad_bridge["identity_sha256"] = "wrong"
            write_json(default_evidence["default_provider_bridge"], bad_bridge)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                default_gate_evidence_paths={key: str(value) for key, value in default_evidence.items()},
                **non_default_evidence,
            )

        self.assertEqual(summary["default_swap_status"], "defer")
        blockers = "\n".join(summary["blockers"]["default_swap"])
        self.assertIn("default provider bridge evidence failed validation", blockers)
        self.assertIn("identity mismatch", blockers)

    def test_default_throughput_threshold_failure_blocks_default_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)
            bad_throughput = throughput_gate_manifest()
            bad_throughput["cold_load_ms"] = 5001.0
            bad_throughput["warm_batch64_docs_per_second"] = 9.9
            write_json(default_evidence["startup_load_encode_throughput_gate"], bad_throughput)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                default_gate_evidence_paths={key: str(value) for key, value in default_evidence.items()},
                **non_default_evidence,
            )

        self.assertEqual(summary["default_swap_status"], "defer")
        blockers = "\n".join(summary["blockers"]["default_swap"])
        self.assertIn("cold_load_ms exceeds 5000ms ceiling", blockers)
        self.assertIn("warm batch64 throughput below 10 docs/s floor", blockers)

    def test_tsv_writer_emits_default_gate_detail_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)
            output_tsv = root / "readiness.tsv"
            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                default_gate_evidence_paths={key: str(value) for key, value in default_evidence.items()},
                **non_default_evidence,
            )

            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("default_swap_gate", "default_provider_bridge")]["status"], "pass")
        self.assertIn(
            "backend_fingerprint",
            keyed[("default_swap_gate_detail", "default_provider_bridge")]["value"],
        )

    def test_complete_bge_gate_missing_evidence_defers_non_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        self.assertFalse(summary["non_default_evidence"]["all_valid"])
        self.assertTrue(
            any("candidate smoke evidence missing" in blocker for blocker in summary["blockers"]["non_default"])
        )
        self.assertTrue(
            any("role-aware provider smoke evidence missing" in blocker for blocker in summary["blockers"]["non_default"])
        )
        self.assertTrue(
            any("CorkScrewDB serving smoke evidence missing" in blocker for blocker in summary["blockers"]["non_default"])
        )

    def test_bad_role_provider_fingerprint_or_missing_role_calls_defers_non_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            evidence = write_valid_non_default_evidence(root)
            bad_role = role_aware_provider_smoke_manifest()
            bad_role["backend_fingerprint"] = "wrong"
            bad_role["db_manifest_embedding"]["backend_fingerprint"] = "wrong"
            bad_role["query_role_calls"] = 0
            write_json(evidence["role_aware_provider_smoke_evidence"], bad_role)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        blockers = "\n".join(summary["blockers"]["non_default"])
        self.assertIn("role-aware provider smoke evidence failed validation", blockers)
        self.assertIn("backend fingerprint mismatch", blockers)
        self.assertIn("query_role_calls must be > 0", blockers)

    def test_bad_serving_q8_delta_or_p95_defers_non_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root)
            evidence = write_valid_non_default_evidence(root)
            bad_serving = corkscrewdb_serving_smoke_manifest()
            bad_serving["offline_comparison"]["q8"]["delta"]["ndcg_at_10"] = 0.01
            bad_serving["offline_comparison"]["q8"]["corkscrew"]["p95_ms"] = 60.0
            write_json(evidence["corkscrewdb_serving_smoke_evidence"], bad_serving)

            summary = summarizer.build_summary(
                bge_gate_root=root,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **evidence,
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        blockers = "\n".join(summary["blockers"]["non_default"])
        self.assertIn("CorkScrewDB serving q8 nDCG delta exceeds tolerance", blockers)
        self.assertIn("CorkScrewDB serving q8 p95 exceeds 50ms ceiling", blockers)

    def test_scan_paths_flags_public_v6_and_allows_internal_run_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_gate(root / "gate")
            evidence = write_valid_non_default_evidence(root)
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
                **evidence,
            )

        self.assertEqual(summary["non_default_candidate_status"], "defer")
        matches = summary["public_name_hygiene"]["matches"]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["path"].endswith("bad.md"))
        self.assertTrue(any("public-name hygiene" in blocker for blocker in summary["blockers"]["non_default"]))


if __name__ == "__main__":
    unittest.main()
