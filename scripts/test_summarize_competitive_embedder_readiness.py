#!/usr/bin/env python3
"""Dependency-free tests for competitive embedder readiness rollup."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import summarize_competitive_embedder_readiness as summarizer
from test_summarize_dynamic_remine_readiness import write_stagea_ready
from test_summarize_encoder_v21_readiness import write_complete_encoder_fixture
from test_summarize_eos_embedder1_release_readiness import (
    write_valid_default_evidence,
    write_valid_non_default_evidence,
)


def write_dynamic_fixture(root: Path) -> tuple[Path, Path]:
    stagea_root = root / "stagea"
    descriptor = root / "guided-negative-dynamic-remine-plan-v1.md"
    descriptor.write_text("# dynamic-remine plan\n", encoding="utf-8")
    write_stagea_ready(stagea_root)
    return stagea_root, descriptor


class SummarizeCompetitiveEmbedderReadinessTest(unittest.TestCase):
    def test_complete_fixture_rolls_up_ready_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            encoder_root, bge_root, encoder_descriptor = write_complete_encoder_fixture(root)
            dynamic_root, dynamic_descriptor = write_dynamic_fixture(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)

            summary = summarizer.build_summary(
                bge_gate_root=bge_root,
                encoder_run_root=encoder_root,
                encoder_descriptor=encoder_descriptor,
                dynamic_stagea_root=dynamic_root,
                dynamic_descriptor=dynamic_descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                embedder1_default_gate_evidence_paths={key: str(value) for key, value in default_evidence.items()},
                embedder1_candidate_smoke_evidence=non_default_evidence["candidate_smoke_evidence"],
                embedder1_role_aware_provider_smoke_evidence=non_default_evidence[
                    "role_aware_provider_smoke_evidence"
                ],
                embedder1_corkscrewdb_serving_smoke_evidence=non_default_evidence[
                    "corkscrewdb_serving_smoke_evidence"
                ],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertEqual(summary["schema"], summarizer.SUMMARY_SCHEMA)
        self.assertFalse(summary["summary"]["waiting_on_fiqa"])
        self.assertEqual(summary["packets"]["selected_bge_full_gate"]["status"], "ready_for_review")
        self.assertEqual(summary["packets"]["encoder_v21_controlled_training"]["status"], "ready_to_launch")
        self.assertEqual(summary["packets"]["dynamic_remine"]["status"], "ready_to_launch")
        self.assertEqual(summary["packets"]["eos_embedder1_non_default"]["status"], "ready_for_review")
        self.assertEqual(summary["packets"]["eos_embedder1_default_swap"]["status"], "ready_for_review")
        self.assertEqual(summary["public_identity_policy"]["public_name"], "Eos Embedder 1")
        self.assertEqual(summary["public_identity_policy"]["public_id"], "eos-embedder-1")
        self.assertFalse(summary["public_identity_policy"]["internal_v_labels_are_release_versions"])
        self.assertFalse(summary["quality_claim"])
        self.assertFalse(summary["training_run"])

    def test_incomplete_fiqa_waits_and_preserves_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            encoder_root, bge_root, encoder_descriptor = write_complete_encoder_fixture(root)
            fiqa = bge_root / "fiqa"
            os.remove(fiqa / "vectors" / "query-vectors.jsonl")
            os.remove(fiqa / "vectors" / "manifest.json")
            os.remove(fiqa / "eval" / "dense.metrics.json")
            os.remove(fiqa / "eval" / "turboquant-q8-q4.metrics.json")
            dynamic_root, dynamic_descriptor = write_dynamic_fixture(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            output_tsv = root / "rollup.tsv"

            summary = summarizer.build_summary(
                bge_gate_root=bge_root,
                encoder_run_root=encoder_root,
                encoder_descriptor=encoder_descriptor,
                dynamic_stagea_root=dynamic_root,
                dynamic_descriptor=dynamic_descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                embedder1_candidate_smoke_evidence=non_default_evidence["candidate_smoke_evidence"],
                embedder1_role_aware_provider_smoke_evidence=non_default_evidence[
                    "role_aware_provider_smoke_evidence"
                ],
                embedder1_corkscrewdb_serving_smoke_evidence=non_default_evidence[
                    "corkscrewdb_serving_smoke_evidence"
                ],
                clock=lambda: "2026-06-29T00:00:00Z",
            )
            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertTrue(summary["summary"]["waiting_on_fiqa"])
        self.assertEqual(summary["packets"]["selected_bge_full_gate"]["status"], "waiting_on_fiqa")
        self.assertEqual(summary["packets"]["encoder_v21_controlled_training"]["status"], "waiting_on_fiqa")
        self.assertEqual(summary["packets"]["dynamic_remine"]["status"], "waiting_on_fiqa")
        self.assertEqual(summary["packets"]["eos_embedder1_non_default"]["status"], "defer")
        self.assertEqual(summary["packets"]["eos_embedder1_default_swap"]["status"], "defer")
        self.assertIn("Wait for the active/incomplete FiQA selected-BGE export", summary["next_action"])
        self.assertIn("`--require-non-default-promotion-policy`", summary["arbiter_next_action"])
        self.assertEqual(summary["active_export"]["dataset"], "fiqa")
        self.assertTrue(summary["active_export"]["present"])
        self.assertEqual(summary["active_export"]["status"], "partial_artifacts_present")
        self.assertIsNone(summary["active_export"]["pid"])
        self.assertIsNone(summary["active_export"]["command"])
        progress = next(
            item for item in summary["packets"]["selected_bge_full_gate"]["progress"] if item["dataset"] == "fiqa"
        )
        self.assertEqual(progress["expected_documents"], 57638)
        self.assertEqual(progress["expected_queries"], 6648)
        self.assertEqual(progress["doc_vector_lines"], 2)
        self.assertIsNone(progress["query_vector_lines"])
        self.assertEqual(progress["vector_progress_completed"], 2)
        self.assertEqual(progress["vector_progress_total"], 64286)
        self.assertAlmostEqual(progress["vector_progress_percent"], (2 / 64286) * 100.0)
        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("summary", "waiting_on_fiqa")]["status"], "waiting_on_fiqa")
        self.assertEqual(keyed[("packet", "selected_bge_full_gate")]["status"], "waiting_on_fiqa")
        self.assertEqual(keyed[("packet", "encoder_v21_controlled_training")]["status"], "waiting_on_fiqa")
        self.assertEqual(keyed[("progress", "fiqa")]["status"], "incomplete")
        self.assertIn("2/64286", keyed[("progress", "fiqa")]["progress"])

    def test_require_unblocked_next_action_exits_2_when_waiting_on_fiqa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            encoder_root, bge_root, encoder_descriptor = write_complete_encoder_fixture(root)
            fiqa = bge_root / "fiqa"
            os.remove(fiqa / "vectors" / "query-vectors.jsonl")
            os.remove(fiqa / "vectors" / "manifest.json")
            os.remove(fiqa / "eval" / "dense.metrics.json")
            os.remove(fiqa / "eval" / "turboquant-q8-q4.metrics.json")
            dynamic_root, dynamic_descriptor = write_dynamic_fixture(root)

            code = summarizer.main(
                [
                    "--bge-gate-root",
                    str(bge_root),
                    "--encoder-run-root",
                    str(encoder_root),
                    "--encoder-descriptor",
                    str(encoder_descriptor),
                    "--dynamic-stagea-root",
                    str(dynamic_root),
                    "--dynamic-descriptor",
                    str(dynamic_descriptor),
                    "--datasets",
                    "scifact,nfcorpus,fiqa",
                    "--output-json",
                    str(root / "rollup.json"),
                    "--output-tsv",
                    str(root / "rollup.tsv"),
                    "--require-unblocked-next-action",
                ]
            )

        self.assertEqual(code, 2)

    def test_cli_active_export_metadata_is_preserved_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            encoder_root, bge_root, encoder_descriptor = write_complete_encoder_fixture(root)
            fiqa = bge_root / "fiqa"
            os.remove(fiqa / "vectors" / "query-vectors.jsonl")
            os.remove(fiqa / "vectors" / "manifest.json")
            os.remove(fiqa / "eval" / "dense.metrics.json")
            os.remove(fiqa / "eval" / "turboquant-q8-q4.metrics.json")
            dynamic_root, dynamic_descriptor = write_dynamic_fixture(root)
            output_json = root / "rollup.json"
            command = "eos export-pretrained-bert-retrieval-vectors --dataset fiqa"

            code = summarizer.main(
                [
                    "--bge-gate-root",
                    str(bge_root),
                    "--encoder-run-root",
                    str(encoder_root),
                    "--encoder-descriptor",
                    str(encoder_descriptor),
                    "--dynamic-stagea-root",
                    str(dynamic_root),
                    "--dynamic-descriptor",
                    str(dynamic_descriptor),
                    "--datasets",
                    "scifact,nfcorpus,fiqa",
                    "--output-json",
                    str(output_json),
                    "--output-tsv",
                    str(root / "rollup.tsv"),
                    "--active-export-pid",
                    "281659",
                    "--active-export-command",
                    command,
                ]
            )

            data = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertTrue(data["active_export"]["present"])
        self.assertEqual(data["active_export"]["pid"], 281659)
        self.assertEqual(data["active_export"]["command"], command)


if __name__ == "__main__":
    unittest.main()
