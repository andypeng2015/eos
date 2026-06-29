#!/usr/bin/env python3
"""Dependency-free tests for Stage A/B/C pretraining readiness summary."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import summarize_stageabc_pretraining_readiness as summarizer


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def research_only_gates() -> dict[str, Any]:
    return {
        "train_allowed_for_research": True,
        "release_train_allowed": False,
        "commercial_use_allowed": False,
        "test_rows_train_allowed": False,
    }


def write_stageabc_fixture(root: Path) -> dict[str, Path]:
    base = root / "stageabc"
    pipeline_map = base / "retrieval-pretraining-distillation-pipeline-map-v1-report.md"
    pipeline_map.parent.mkdir(parents=True, exist_ok=True)
    pipeline_map.write_text("Stage A\nStage B\nStage C\n", encoding="utf-8")
    stagea_manifest = write_json(
        base / "stagea" / "manifest.json",
        {
            "counts": {"rows_emitted": 16},
            "legal_gates": research_only_gates(),
            "split_policy": {"test_or_eval_rows_used": False, "test_rows_train_allowed": False},
            "reader_compatibility": {"matches_runtime_embedding_text_hard_negative_dataset_go": True},
        },
    )
    stagea_leak = write_json(
        base / "stagea" / "reports" / "leak-report.json",
        {"status": "passed", "validation": {"status": "passed", "counts": {"rows_checked": 16}}},
    )
    imported_manifest = write_json(
        base / "imported-bge" / "manifest.json",
        {
            "coverage": {
                "examples_seen": 256,
                "examples_scored": 256,
                "examples_written": 256,
                "missing_examples": 0,
            },
            "scores": {"positive_top1_rate": 1.0},
            "teacher_model_id": "BAAI/bge-small-en-v1.5#imported-mll-fixture",
        },
    )
    imported_validation = write_json(
        base / "imported-bge" / "artifacts" / "validation-summary.json",
        {
            "rows_emitted": 256,
            "validation": {"scoring_complete": True},
            "score_coverage": {"missing_examples": 0},
        },
    )
    imported_guide = write_json(
        base / "imported-bge" / "guide-filter-manifest.json",
        {
            "counts": {"rows_seen": 256, "rows_emitted": 256, "clean_agreement": 256},
            "coverage": {"drop_samples": []},
            "inputs": {
                "teacher_caches": {
                    "imported_bge_small_en_v1_5": {
                        "config": {"package_identity": "identity-fixture", "package_sha256": "sha-fixture"}
                    }
                }
            },
            "validation": {"no_row_emitted_without_required_scores": True},
        },
    )
    qwen_mxbai_manifest = write_json(
        base / "qwen-mxbai" / "manifest.json",
        {
            "counts": {
                "rows_seen": 5000,
                "rows_emitted": 2805,
                "clean_agreement": 2800,
                "ambiguous_soft_only": 5,
                "conflict": 2195,
                "conflict_drop": 2195,
            },
            "coverage": {
                "teachers": {
                    "qwen3_0_6b_real": {"complete_rows": 5000},
                    "mxbai_large_real": {"complete_rows": 5000},
                }
            },
        },
    )
    qwen_mxbai_summary = write_json(
        base / "qwen-mxbai" / "reports" / "guide-filter-independent-summary.json",
        {
            "emitted_dev_positive_flags": {"rows": 423, "refs": 456},
            "policy_counts_output": {"clean_agreement": 2800, "ambiguous_soft_only": 5},
            "validation": {"no_row_emitted_without_required_scores": True},
        },
    )
    qwen3_metrics = write_json(base / "listwise" / "qwen3.metrics.json", listwise_metrics())
    mxbai_metrics = write_json(base / "listwise" / "mxbai.metrics.json", listwise_metrics())
    return {
        "stageabc_pretraining_distillation_report": pipeline_map,
        "stageabc_stagea_row_manifest": stagea_manifest,
        "stageabc_stagea_leak_report": stagea_leak,
        "stageabc_imported_bge_manifest": imported_manifest,
        "stageabc_imported_bge_validation": imported_validation,
        "stageabc_imported_bge_guide_filter_manifest": imported_guide,
        "stageabc_qwen_mxbai_manifest": qwen_mxbai_manifest,
        "stageabc_qwen_mxbai_independent_summary": qwen_mxbai_summary,
        "stageabc_listwise_qwen3_metrics": qwen3_metrics,
        "stageabc_listwise_mxbai_metrics": mxbai_metrics,
    }


def listwise_metrics() -> dict[str, Any]:
    return {
        "config": {"eval_only": True},
        "workload": {"actual_train_examples": 0},
        "profile_delta": {"optimizer_updates": 0},
        "summary": {"steps_run": 0},
        "final_listwise_geometry_eval": {
            "query_count": 2379,
            "batch_count": 75,
            "teacher_cross_entropy": 3.6,
            "teacher_kl": 2.5,
            "teacher_top1_agreement": 0.38,
            "any_positive_top1": 0.39,
        },
    }


class SummarizeStageABCPretrainingReadinessTest(unittest.TestCase):
    def test_fixture_builds_conservative_evidence_ready_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = write_stageabc_fixture(root)
            summary = summarizer.build_summary(
                pipeline_map_report=paths["stageabc_pretraining_distillation_report"],
                stagea_row_manifest=paths["stageabc_stagea_row_manifest"],
                stagea_leak_report=paths["stageabc_stagea_leak_report"],
                imported_bge_manifest=paths["stageabc_imported_bge_manifest"],
                imported_bge_validation=paths["stageabc_imported_bge_validation"],
                imported_bge_guide_filter_manifest=paths["stageabc_imported_bge_guide_filter_manifest"],
                qwen_mxbai_manifest=paths["stageabc_qwen_mxbai_manifest"],
                qwen_mxbai_independent_summary=paths["stageabc_qwen_mxbai_independent_summary"],
                listwise_qwen3_metrics=paths["stageabc_listwise_qwen3_metrics"],
                listwise_mxbai_metrics=paths["stageabc_listwise_mxbai_metrics"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertEqual(summary["schema"], summarizer.SUMMARY_SCHEMA)
        self.assertEqual(summary["status"], "evidence_ready_not_training_ready")
        self.assertFalse(summary["training_ready"])
        self.assertFalse(summary["release_train_allowed"])
        self.assertFalse(summary["quality_claim"])
        self.assertEqual(summary["components"]["pipeline_map"]["status"], "pass")
        self.assertEqual(summary["components"]["stage_a_row_builder"]["status"], "pass")
        self.assertEqual(summary["components"]["imported_bge_teacher_bridge"]["status"], "pass")
        self.assertEqual(summary["components"]["qwen_mxbai_guide_filter"]["status"], "evidence_ready_with_flags")
        self.assertEqual(summary["components"]["listwise_eval_only"]["status"], "eval_only_evidence")
        self.assertEqual(
            summary["components"]["stage_c_compact_adaptation"]["status"],
            "deferred_until_dense_acceptance",
        )
        warnings = " | ".join(summary["components"]["qwen_mxbai_guide_filter"]["warnings"])
        self.assertIn("dev-positive", warnings)
        self.assertIn("ambiguous soft-only", warnings)

    def test_missing_inputs_do_not_raise_and_report_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = summarizer.build_summary(
                pipeline_map_report=root / "missing.md",
                stagea_row_manifest=root / "missing-stagea.json",
                stagea_leak_report=root / "missing-leak.json",
                imported_bge_manifest=root / "missing-imported.json",
                imported_bge_validation=root / "missing-validation.json",
                imported_bge_guide_filter_manifest=root / "missing-guide.json",
                qwen_mxbai_manifest=root / "missing-qwen.json",
                qwen_mxbai_independent_summary=root / "missing-qwen-summary.json",
                listwise_qwen3_metrics=root / "missing-qwen3-metrics.json",
                listwise_mxbai_metrics=root / "missing-mxbai-metrics.json",
            )

        self.assertEqual(summary["status"], "partial_evidence_waiting_implementation")
        self.assertEqual(summary["components"]["pipeline_map"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["stage_a_row_builder"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["imported_bge_teacher_bridge"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["qwen_mxbai_guide_filter"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["listwise_eval_only"]["status"], "missing_evidence")

    def test_missing_pipeline_map_blocks_even_when_json_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = write_stageabc_fixture(root)
            missing_pipeline = root / "missing-pipeline-map.md"
            summary = summarizer.build_summary(
                pipeline_map_report=missing_pipeline,
                stagea_row_manifest=paths["stageabc_stagea_row_manifest"],
                stagea_leak_report=paths["stageabc_stagea_leak_report"],
                imported_bge_manifest=paths["stageabc_imported_bge_manifest"],
                imported_bge_validation=paths["stageabc_imported_bge_validation"],
                imported_bge_guide_filter_manifest=paths["stageabc_imported_bge_guide_filter_manifest"],
                qwen_mxbai_manifest=paths["stageabc_qwen_mxbai_manifest"],
                qwen_mxbai_independent_summary=paths["stageabc_qwen_mxbai_independent_summary"],
                listwise_qwen3_metrics=paths["stageabc_listwise_qwen3_metrics"],
                listwise_mxbai_metrics=paths["stageabc_listwise_mxbai_metrics"],
            )

        self.assertEqual(summary["status"], "partial_evidence_waiting_implementation")
        self.assertEqual(summary["components"]["pipeline_map"]["status"], "missing_evidence")
        self.assertIn("missing_evidence", " | ".join(summary["components"]["pipeline_map"]["blockers"]))
        self.assertEqual(summary["components"]["stage_a_row_builder"]["status"], "pass")
        self.assertEqual(summary["components"]["imported_bge_teacher_bridge"]["status"], "pass")

    def test_incomplete_pipeline_map_blocks_even_when_json_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = write_stageabc_fixture(root)
            incomplete_pipeline = root / "incomplete-pipeline-map.md"
            incomplete_pipeline.write_text("Stage A\nStage B\n", encoding="utf-8")
            summary = summarizer.build_summary(
                pipeline_map_report=incomplete_pipeline,
                stagea_row_manifest=paths["stageabc_stagea_row_manifest"],
                stagea_leak_report=paths["stageabc_stagea_leak_report"],
                imported_bge_manifest=paths["stageabc_imported_bge_manifest"],
                imported_bge_validation=paths["stageabc_imported_bge_validation"],
                imported_bge_guide_filter_manifest=paths["stageabc_imported_bge_guide_filter_manifest"],
                qwen_mxbai_manifest=paths["stageabc_qwen_mxbai_manifest"],
                qwen_mxbai_independent_summary=paths["stageabc_qwen_mxbai_independent_summary"],
                listwise_qwen3_metrics=paths["stageabc_listwise_qwen3_metrics"],
                listwise_mxbai_metrics=paths["stageabc_listwise_mxbai_metrics"],
            )

        self.assertEqual(summary["status"], "partial_evidence_waiting_implementation")
        self.assertEqual(summary["components"]["pipeline_map"]["status"], "planned_not_ready")
        self.assertIn("mentions_stage_c", " | ".join(summary["components"]["pipeline_map"]["blockers"]))

    def test_cli_writes_json_and_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = write_stageabc_fixture(root)
            output_json = root / "summary.json"
            output_tsv = root / "summary.tsv"
            code = summarizer.main(
                [
                    "--pipeline-map-report",
                    str(paths["stageabc_pretraining_distillation_report"]),
                    "--stagea-row-manifest",
                    str(paths["stageabc_stagea_row_manifest"]),
                    "--stagea-leak-report",
                    str(paths["stageabc_stagea_leak_report"]),
                    "--imported-bge-manifest",
                    str(paths["stageabc_imported_bge_manifest"]),
                    "--imported-bge-validation",
                    str(paths["stageabc_imported_bge_validation"]),
                    "--imported-bge-guide-filter-manifest",
                    str(paths["stageabc_imported_bge_guide_filter_manifest"]),
                    "--qwen-mxbai-manifest",
                    str(paths["stageabc_qwen_mxbai_manifest"]),
                    "--qwen-mxbai-independent-summary",
                    str(paths["stageabc_qwen_mxbai_independent_summary"]),
                    "--listwise-qwen3-metrics",
                    str(paths["stageabc_listwise_qwen3_metrics"]),
                    "--listwise-mxbai-metrics",
                    str(paths["stageabc_listwise_mxbai_metrics"]),
                    "--output-json",
                    str(output_json),
                    "--output-tsv",
                    str(output_tsv),
                ]
            )
            data = json.loads(output_json.read_text(encoding="utf-8"))
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(code, 0)
        self.assertEqual(data["status"], "evidence_ready_not_training_ready")
        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("component", "pipeline_map")]["status"], "pass")
        self.assertEqual(
            keyed[("component", "qwen_mxbai_guide_filter")]["status"],
            "evidence_ready_with_flags",
        )


if __name__ == "__main__":
    unittest.main()
