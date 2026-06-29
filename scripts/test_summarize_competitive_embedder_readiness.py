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
import summarize_dynamic_remine_readiness as dynamic_readiness
from test_summarize_compact_native_readiness import write_compact_native_fixture
from test_summarize_dynamic_remine_readiness import write_stagea_ready
from test_summarize_encoder_v21_readiness import write_complete_encoder_fixture
from test_summarize_eos_embedder1_release_readiness import (
    write_valid_default_evidence,
    write_valid_non_default_evidence,
)
from test_summarize_stageabc_pretraining_readiness import write_stageabc_fixture


def write_dynamic_fixture(root: Path) -> tuple[Path, Path]:
    stagea_root = root / "stagea"
    descriptor = root / "guided-negative-dynamic-remine-plan-v1.md"
    descriptor.write_text("# dynamic-remine plan\n", encoding="utf-8")
    write_stagea_ready(stagea_root)
    (stagea_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (stagea_root / "vectors").mkdir(parents=True, exist_ok=True)
    (stagea_root / "beir").mkdir(parents=True, exist_ok=True)
    expected_score_rows = 128 * dynamic_readiness.DEFAULT_EXPECTED_CANDIDATES_PER_ROW
    (stagea_root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "fixture-stagea",
                "coverage": {
                    "examples_seen": 128,
                    "examples_scored": 128,
                    "examples_written": 128,
                    "missing_examples": 0,
                    "import_score_rows": expected_score_rows,
                    "candidate_rows_scored": expected_score_rows,
                },
                "vectors": {"doc_vector_rows": expected_score_rows, "query_vector_rows": 100},
                "beir": {"corpus_rows": expected_score_rows, "query_rows": 100},
                "scores": {"positive_top1_rate": 1.0, "margin": {"min": 0.1}},
                "teacher_model_id": dynamic_readiness.DEFAULT_TEACHER_MODEL_ID,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stagea_root / "artifacts" / "validation-summary.json").write_text(
        json.dumps(
            {
                "validation": {
                    "rows_ge_128": True,
                    "scoring_complete": True,
                    "guide_missing_drop_0": True,
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stagea_root / "vectors" / "manifest.json").write_text(
        json.dumps(
            {
                "package_sha256": "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763",
                "package_identity_sha256": "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stagea_root / "beir" / "manifest.json").write_text(
        json.dumps({"counts": {"unique_docs": expected_score_rows, "unique_queries": 100}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (stagea_root / "guide-filter-manifest.json").write_text(
        json.dumps(
            {
                "schema": dynamic_readiness.GUIDE_FILTER_SCHEMA,
                "counts": {
                    "rows_seen": 128,
                    "rows_emitted": 128,
                    "clean_agreement": 128,
                    "missing_score_drop": 0,
                    "conflict": 0,
                    "ambiguous_soft_only": 0,
                },
                "legal_gates": {
                    "train_allowed_for_research": True,
                    "release_train_allowed": False,
                    "commercial_use_allowed": False,
                    "test_rows_train_allowed": False,
                },
                "legal_gate_accounting": {"research_only_preserved": True},
                "inputs": {
                    "teacher_caches": {
                        dynamic_readiness.DEFAULT_TEACHER_LABEL: {
                            "model_id": dynamic_readiness.DEFAULT_TEACHER_MODEL_ID,
                            "config": {
                                "package_sha256": "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763",
                                "package_identity": "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637",
                            },
                        }
                    }
                },
                "quality_claim": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return stagea_root, descriptor


def write_major_lever_reports(root: Path) -> tuple[Path, Path]:
    compact_report = root / "compact-bge-listwise-larger-validation-split211-seed191-v1-report.md"
    compact_report.write_text(
        "\n".join(
            [
                "# compact report",
                "- 2000-doc gate: nDCG@10 `+0.062803213`, MAP@100 `+0.050985152`, recall@100 `+0.195312500`.",
                "- 4000-doc gate: nDCG@10 `+0.057674953`, MAP@100 `+0.048607354`, recall@100 `+0.097656250`.",
                "This remains evidence-only and research-only. It supports continued validation, not promotion.",
            ]
        ),
        encoding="utf-8",
    )
    stageabc_report = root / "retrieval-pretraining-distillation-pipeline-map-v1-report.md"
    stageabc_report.write_text("Stage A row builder. Stage B guide cache. Stage C compact adaptation.\n", encoding="utf-8")
    return compact_report, stageabc_report


def compact_rollup_paths(root: Path) -> dict[str, Path]:
    return write_compact_native_fixture(root)


def compact_cli_args(paths: dict[str, Path]) -> list[str]:
    option_names = {
        "compact_native_student_report": "--compact-native-student-report",
        "compact_native_architecture_map_report": "--compact-native-architecture-map-report",
        "compact_native_manifest_checkpoint_foundation_report": "--compact-native-manifest-checkpoint-foundation-report",
        "compact_native_generic_bootstrap_report": "--compact-native-generic-bootstrap-report",
        "compact_native_train_guard_report": "--compact-native-train-guard-report",
        "compact_native_serving_parity_report": "--compact-native-serving-parity-report",
        "compact_native_default_embedding_source": "--compact-native-default-embedding-source",
        "compact_native_runtime_embedding_model_source": "--compact-native-runtime-embedding-model-source",
        "compact_native_backend_tensor_ops_source": "--compact-native-backend-tensor-ops-source",
        "compact_native_default_embedding_test": "--compact-native-default-embedding-test",
        "compact_native_runtime_embedding_model_test": "--compact-native-runtime-embedding-model-test",
        "compact_native_runtime_embedding_trainer_test": "--compact-native-runtime-embedding-trainer-test",
        "compact_native_backend_compact_attention_test": "--compact-native-backend-compact-attention-test",
        "compact_native_cmd_eos_main_test": "--compact-native-cmd-eos-main-test",
        "compact_native_heads2_lr_bracket_report": "--compact-native-heads2-lr-bracket-report",
        "compact_native_heads2_lr_bracket_gate_log": "--compact-native-heads2-lr-bracket-gate-log",
        "compact_native_laststep_movement_report": "--compact-native-laststep-movement-report",
        "compact_native_bge_pre_retrieval_2000": "--compact-native-bge-pre-retrieval-2000",
        "compact_native_bge_post_retrieval_2000": "--compact-native-bge-post-retrieval-2000",
        "compact_native_bge_pre_retrieval_4000": "--compact-native-bge-pre-retrieval-4000",
        "compact_native_bge_post_retrieval_4000": "--compact-native-bge-post-retrieval-4000",
        "compact_native_bge_pre_listwise": "--compact-native-bge-pre-listwise",
        "compact_native_bge_post_listwise": "--compact-native-bge-post-listwise",
        "compact_native_bge_train_metrics": "--compact-native-bge-train-metrics",
        "compact_native_laststep_pre_retrieval": "--compact-native-laststep-pre-retrieval",
        "compact_native_laststep_post_retrieval": "--compact-native-laststep-post-retrieval",
        "compact_native_laststep_train_metrics": "--compact-native-laststep-train-metrics",
    }
    args: list[str] = []
    for key, option in option_names.items():
        args.extend([option, str(paths[key])])
    return args


class SummarizeCompetitiveEmbedderReadinessTest(unittest.TestCase):
    def test_complete_fixture_rolls_up_ready_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            encoder_root, bge_root, encoder_descriptor = write_complete_encoder_fixture(root)
            dynamic_root, dynamic_descriptor = write_dynamic_fixture(root)
            compact_paths = compact_rollup_paths(root)
            stageabc_paths = write_stageabc_fixture(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            default_evidence = write_valid_default_evidence(root)

            summary = summarizer.build_summary(
                bge_gate_root=bge_root,
                encoder_run_root=encoder_root,
                encoder_descriptor=encoder_descriptor,
                dynamic_stagea_root=dynamic_root,
                dynamic_descriptor=dynamic_descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **compact_paths,
                **stageabc_paths,
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
        self.assertEqual(
            summary["packets"]["compact_native_student"]["status"],
            "evidence_positive_blocked_by_training_movement",
        )
        self.assertEqual(
            summary["packets"]["stageabc_pretraining_distillation"]["status"],
            "evidence_ready_not_training_ready",
        )
        self.assertEqual(summary["packets"]["role_asymmetry"]["status"], "release_identity_gate_ready")
        self.assertEqual(summary["packets"]["quantization_profile"]["status"], "q8_ready_for_review")
        self.assertAlmostEqual(
            summary["packets"]["compact_native_student"]["details"]["compact_native_readiness"]["components"][
                "bge_listwise_validation"
            ]["details"]["retrieval_2000"]["ndcg_at_10"]["delta"],
            0.06,
        )
        self.assertFalse(summary["packets"]["compact_native_student"]["details"]["promotion_ready"])
        self.assertFalse(summary["packets"]["compact_native_student"]["details"]["training_ready"])
        self.assertTrue(
            summary["packets"]["compact_native_student"]["details"]["compact_native_readiness"]["components"][
                "serving_parity"
            ]["details"]["trainer_serving_numeric_parity"]
        )
        self.assertFalse(summary["packets"]["stageabc_pretraining_distillation"]["details"]["training_ready"])
        self.assertEqual(summary["summary"]["quantization_profile_status"], "q8_ready_for_review")
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
            compact_paths = compact_rollup_paths(root)
            stageabc_paths = write_stageabc_fixture(root)
            non_default_evidence = write_valid_non_default_evidence(root)
            output_tsv = root / "rollup.tsv"

            summary = summarizer.build_summary(
                bge_gate_root=bge_root,
                encoder_run_root=encoder_root,
                encoder_descriptor=encoder_descriptor,
                dynamic_stagea_root=dynamic_root,
                dynamic_descriptor=dynamic_descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                **compact_paths,
                **stageabc_paths,
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
        self.assertEqual(summary["packets"]["compact_native_student"]["status"], "waiting_on_fiqa")
        self.assertEqual(
            summary["packets"]["stageabc_pretraining_distillation"]["status"],
            "evidence_ready_not_training_ready",
        )
        self.assertEqual(summary["packets"]["role_asymmetry"]["status"], "release_identity_gate_ready")
        self.assertEqual(summary["packets"]["quantization_profile"]["status"], "waiting_on_fiqa")
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
        self.assertEqual(keyed[("packet", "compact_native_student")]["status"], "waiting_on_fiqa")
        self.assertEqual(
            keyed[("packet", "stageabc_pretraining_distillation")]["status"],
            "evidence_ready_not_training_ready",
        )
        self.assertEqual(keyed[("packet", "role_asymmetry")]["status"], "release_identity_gate_ready")
        self.assertEqual(keyed[("packet", "quantization_profile")]["status"], "waiting_on_fiqa")
        self.assertEqual(keyed[("progress", "fiqa")]["status"], "incomplete")
        self.assertIn("2/64286", keyed[("progress", "fiqa")]["progress"])

    def test_missing_major_lever_reports_defer_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            encoder_root, bge_root, encoder_descriptor = write_complete_encoder_fixture(root)
            dynamic_root, dynamic_descriptor = write_dynamic_fixture(root)

            summary = summarizer.build_summary(
                bge_gate_root=bge_root,
                encoder_run_root=encoder_root,
                encoder_descriptor=encoder_descriptor,
                dynamic_stagea_root=dynamic_root,
                dynamic_descriptor=dynamic_descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                compact_native_student_report=root / "missing-compact.md",
                compact_native_architecture_map_report=root / "missing-architecture.md",
                compact_native_manifest_checkpoint_foundation_report=root / "missing-foundation.md",
                compact_native_generic_bootstrap_report=root / "missing-bootstrap.md",
                compact_native_train_guard_report=root / "missing-guard.md",
                compact_native_serving_parity_report=root / "missing-serving.md",
                compact_native_default_embedding_source=root / "missing-default-embedding.go",
                compact_native_runtime_embedding_model_source=root / "missing-runtime-embedding.go",
                compact_native_backend_tensor_ops_source=root / "missing-tensor-ops.go",
                compact_native_default_embedding_test=root / "missing-default-embedding-test.go",
                compact_native_runtime_embedding_model_test=root / "missing-runtime-embedding-test.go",
                compact_native_runtime_embedding_trainer_test=root / "missing-runtime-embedding-trainer-test.go",
                compact_native_backend_compact_attention_test=root / "missing-compact-attention-test.go",
                compact_native_cmd_eos_main_test=root / "missing-main-test.go",
                compact_native_heads2_lr_bracket_report=root / "missing-bracket.md",
                compact_native_heads2_lr_bracket_gate_log=root / "missing-bracket.log",
                compact_native_laststep_movement_report=root / "missing-laststep.md",
                compact_native_bge_pre_retrieval_2000=root / "missing-pre-2000.json",
                compact_native_bge_post_retrieval_2000=root / "missing-post-2000.json",
                compact_native_bge_pre_retrieval_4000=root / "missing-pre-4000.json",
                compact_native_bge_post_retrieval_4000=root / "missing-post-4000.json",
                compact_native_bge_pre_listwise=root / "missing-pre-listwise.json",
                compact_native_bge_post_listwise=root / "missing-post-listwise.json",
                compact_native_bge_train_metrics=root / "missing-bge-train.json",
                compact_native_laststep_pre_retrieval=root / "missing-last-pre.json",
                compact_native_laststep_post_retrieval=root / "missing-last-post.json",
                compact_native_laststep_train_metrics=root / "missing-last-train.json",
                stageabc_pretraining_distillation_report=root / "missing-stageabc.md",
                stageabc_stagea_row_manifest=root / "missing-stagea-manifest.json",
                stageabc_stagea_leak_report=root / "missing-stagea-leak.json",
                stageabc_imported_bge_manifest=root / "missing-imported-manifest.json",
                stageabc_imported_bge_validation=root / "missing-imported-validation.json",
                stageabc_imported_bge_guide_filter_manifest=root / "missing-imported-guide.json",
                stageabc_qwen_mxbai_manifest=root / "missing-qwen-mxbai-manifest.json",
                stageabc_qwen_mxbai_independent_summary=root / "missing-qwen-mxbai-summary.json",
                stageabc_listwise_qwen3_metrics=root / "missing-qwen3-metrics.json",
                stageabc_listwise_mxbai_metrics=root / "missing-mxbai-metrics.json",
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertEqual(summary["packets"]["compact_native_student"]["status"], "partial_evidence_waiting_validation")
        self.assertEqual(
            summary["packets"]["stageabc_pretraining_distillation"]["status"],
            "partial_evidence_waiting_implementation",
        )
        self.assertEqual(summary["packets"]["role_asymmetry"]["status"], "defer")

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
            compact_paths = compact_rollup_paths(root)
            stageabc_paths = write_stageabc_fixture(root)

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
                    *compact_cli_args(compact_paths),
                    "--stageabc-pretraining-distillation-report",
                    str(stageabc_paths["stageabc_pretraining_distillation_report"]),
                    "--stageabc-stagea-row-manifest",
                    str(stageabc_paths["stageabc_stagea_row_manifest"]),
                    "--stageabc-stagea-leak-report",
                    str(stageabc_paths["stageabc_stagea_leak_report"]),
                    "--stageabc-imported-bge-manifest",
                    str(stageabc_paths["stageabc_imported_bge_manifest"]),
                    "--stageabc-imported-bge-validation",
                    str(stageabc_paths["stageabc_imported_bge_validation"]),
                    "--stageabc-imported-bge-guide-filter-manifest",
                    str(stageabc_paths["stageabc_imported_bge_guide_filter_manifest"]),
                    "--stageabc-qwen-mxbai-manifest",
                    str(stageabc_paths["stageabc_qwen_mxbai_manifest"]),
                    "--stageabc-qwen-mxbai-independent-summary",
                    str(stageabc_paths["stageabc_qwen_mxbai_independent_summary"]),
                    "--stageabc-listwise-qwen3-metrics",
                    str(stageabc_paths["stageabc_listwise_qwen3_metrics"]),
                    "--stageabc-listwise-mxbai-metrics",
                    str(stageabc_paths["stageabc_listwise_mxbai_metrics"]),
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
            compact_paths = compact_rollup_paths(root)
            stageabc_paths = write_stageabc_fixture(root)
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
                    *compact_cli_args(compact_paths),
                    "--stageabc-pretraining-distillation-report",
                    str(stageabc_paths["stageabc_pretraining_distillation_report"]),
                    "--stageabc-stagea-row-manifest",
                    str(stageabc_paths["stageabc_stagea_row_manifest"]),
                    "--stageabc-stagea-leak-report",
                    str(stageabc_paths["stageabc_stagea_leak_report"]),
                    "--stageabc-imported-bge-manifest",
                    str(stageabc_paths["stageabc_imported_bge_manifest"]),
                    "--stageabc-imported-bge-validation",
                    str(stageabc_paths["stageabc_imported_bge_validation"]),
                    "--stageabc-imported-bge-guide-filter-manifest",
                    str(stageabc_paths["stageabc_imported_bge_guide_filter_manifest"]),
                    "--stageabc-qwen-mxbai-manifest",
                    str(stageabc_paths["stageabc_qwen_mxbai_manifest"]),
                    "--stageabc-qwen-mxbai-independent-summary",
                    str(stageabc_paths["stageabc_qwen_mxbai_independent_summary"]),
                    "--stageabc-listwise-qwen3-metrics",
                    str(stageabc_paths["stageabc_listwise_qwen3_metrics"]),
                    "--stageabc-listwise-mxbai-metrics",
                    str(stageabc_paths["stageabc_listwise_mxbai_metrics"]),
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
