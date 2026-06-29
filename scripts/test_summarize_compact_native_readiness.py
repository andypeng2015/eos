#!/usr/bin/env python3
"""Dependency-free tests for compact native readiness summary."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import summarize_compact_native_readiness as summarizer


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def retrieval_metrics(ndcg: float, map100: float, recall100: float) -> dict[str, Any]:
    return {
        "schema": "manta.embedding_retrieval_metrics.v1",
        "quality": {
            "ndcg_at_10": ndcg,
            "map_at_100": map100,
            "recall_at_100": recall100,
        },
    }


def listwise_metrics(ce: float, kl: float, top1: float, any_positive: float) -> dict[str, Any]:
    return {
        "schema": "manta.embedding_train_metrics.v1",
        "final_listwise_geometry_eval": {
            "teacher_cross_entropy": ce,
            "teacher_kl": kl,
            "teacher_top1_agreement": top1,
            "any_positive_top1": any_positive,
        },
    }


def train_metrics(*, steps_run: int, optimizer_updates: int, restored_best: bool, best_step: int) -> dict[str, Any]:
    return {
        "schema": "manta.embedding_train_metrics.v1",
        "summary": {
            "steps_run": steps_run,
            "best_step": best_step,
            "restored_best": restored_best,
        },
        "profile_delta": {"optimizer_updates": optimizer_updates},
    }


def write_compact_native_fixture(root: Path) -> dict[str, Path]:
    base = root / "compact-native"
    architecture_map = base / "compact-native-student-architecture-map-v1-report.md"
    architecture_map.parent.mkdir(parents=True, exist_ok=True)
    architecture_map.write_text("## Implementation Slices\n## Verification Plan\n", encoding="utf-8")
    foundation = base / "compact-native-manifest-checkpoint-foundation-v1-report.md"
    foundation.write_text(
        "architecture metadata\n"
        "generic checkpoint tensor retention\n",
        encoding="utf-8",
    )
    bootstrap = base / "compact-native-generic-bootstrap-v1-report.md"
    bootstrap.write_text("exact-name tensors\ncopyOverlappingTensor\n", encoding="utf-8")
    train_guard = base / "compact-native-batched-guard-v1-report.md"
    train_guard.write_text(
        "scalar-fallback guards\n"
        "Compact backprop/optimizer updates remain explicitly unsupported\n",
        encoding="utf-8",
    )
    serving_parity = base / "compact-native-serving-parity-v1-report.md"
    serving_parity.write_text(
        "This is a gate, not true serving parity.\n"
        "A future implementation still needs true multi-head compact serving parity.\n",
        encoding="utf-8",
    )
    default_embedding_source = base / "models" / "default_embedding.go"
    default_embedding_source.parent.mkdir(parents=True, exist_ok=True)
    default_embedding_source.write_text(
        "if cfg.AttentionHeads > 1 {\n"
        '    fmt.Fprintf(&b, "compact_multihead_attention_h%d", cfg.AttentionHeads)\n'
        "}\n",
        encoding="utf-8",
    )
    runtime_embedding_model_source = base / "runtime" / "embedding_model.go"
    runtime_embedding_model_source.parent.mkdir(parents=True, exist_ok=True)
    runtime_embedding_model_source.write_text(
        '!moduleHasKernelOp(mod, "masked_softmax") && !moduleHasKernelOp(mod, "compact_multihead_attention")\n'
        '!moduleHasScaledAttentionMatMul(mod) && !moduleHasKernelOp(mod, "compact_multihead_attention")\n',
        encoding="utf-8",
    )
    backend_tensor_ops_source = base / "runtime" / "backend" / "tensor_ops.go"
    backend_tensor_ops_source.parent.mkdir(parents=True, exist_ok=True)
    backend_tensor_ops_source.write_text(
        'case "compact_multihead_attention":\n'
        "func compactMultiheadAttentionTensor() {}\n"
        "num_attention_heads\n"
        "hidden%heads != 0\n",
        encoding="utf-8",
    )
    default_embedding_test = base / "models" / "default_embedding_test.go"
    default_embedding_test.write_text(
        "TestInitDefaultEmbeddingPackageCreatesCompactMultiHeadServingGraph\n"
        "compact_multihead_attention num_attention_heads = %q, want 2\n",
        encoding="utf-8",
    )
    runtime_embedding_model_test = base / "runtime" / "embedding_model_test.go"
    runtime_embedding_model_test.write_text(
        "TestLoadEmbeddingAcceptsCompactMultiHeadServingGraph\n"
        "compact_multihead_attention_h2\n",
        encoding="utf-8",
    )
    backend_compact_attention_test = base / "runtime" / "backend" / "compact_attention_ops_test.go"
    backend_compact_attention_test.write_text(
        "TestCompactMultiheadAttentionTensorMatchesReference\n"
        "TestCompactMultiheadAttentionTensorBatchedMasked\n",
        encoding="utf-8",
    )
    cmd_eos_main_test = base / "cmd" / "eos" / "main_test.go"
    cmd_eos_main_test.parent.mkdir(parents=True, exist_ok=True)
    cmd_eos_main_test.write_text(
        "TestRunInitModelCreatesCompactMultiHeadServingGraph\n"
        '"--attention-heads", "2"\n',
        encoding="utf-8",
    )
    bge_report = base / "compact-bge-listwise-larger-validation-split211-seed191-v1-report.md"
    bge_report.write_text("Decision gate result: pass. evidence-only and research-only.\n", encoding="utf-8")
    heads_report = base / "compact-native-heads2-lr-bracket-v1-report.md"
    heads_report.write_text("best_step=0\nrestored_best=true\n", encoding="utf-8")
    heads_log = base / "verification-gate.stdout.log"
    heads_log.write_text("PASS compact-native-heads2-lr-bracket-v1\n", encoding="utf-8")
    laststep_report = base / "compact-native-laststep-movement-diagnostic-v1-report.md"
    laststep_report.write_text("negative last-step movement diagnostic\n", encoding="utf-8")

    pre2000 = write_json(base / "metrics" / "pre.retrieval.2000.metrics.json", retrieval_metrics(0.20, 0.18, 0.50))
    post2000 = write_json(base / "metrics" / "post.retrieval.2000.metrics.json", retrieval_metrics(0.26, 0.23, 0.69))
    pre4000 = write_json(base / "metrics" / "pre.retrieval.4000.metrics.json", retrieval_metrics(0.19, 0.17, 0.40))
    post4000 = write_json(base / "metrics" / "post.retrieval.4000.metrics.json", retrieval_metrics(0.24, 0.21, 0.49))
    pre_listwise = write_json(base / "metrics" / "pre.listwise.metrics.json", listwise_metrics(2.73, 2.20, 0.32, 0.32))
    post_listwise = write_json(base / "metrics" / "post.listwise.metrics.json", listwise_metrics(2.35, 1.82, 0.39, 0.39))
    bge_train = write_json(base / "metrics" / "train.metrics.json", train_metrics(
        steps_run=224,
        optimizer_updates=224,
        restored_best=False,
        best_step=224,
    ))
    laststep_pre = write_json(base / "metrics" / "laststep.pre.retrieval.metrics.json", retrieval_metrics(0.24, 0.22, 0.79))
    laststep_post = write_json(base / "metrics" / "laststep.post.retrieval.metrics.json", retrieval_metrics(0.18, 0.15, 0.56))
    laststep_train = write_json(base / "metrics" / "laststep.train.metrics.json", train_metrics(
        steps_run=32,
        optimizer_updates=32,
        restored_best=False,
        best_step=32,
    ))

    return {
        "compact_native_architecture_map_report": architecture_map,
        "compact_native_manifest_checkpoint_foundation_report": foundation,
        "compact_native_generic_bootstrap_report": bootstrap,
        "compact_native_train_guard_report": train_guard,
        "compact_native_serving_parity_report": serving_parity,
        "compact_native_default_embedding_source": default_embedding_source,
        "compact_native_runtime_embedding_model_source": runtime_embedding_model_source,
        "compact_native_backend_tensor_ops_source": backend_tensor_ops_source,
        "compact_native_default_embedding_test": default_embedding_test,
        "compact_native_runtime_embedding_model_test": runtime_embedding_model_test,
        "compact_native_backend_compact_attention_test": backend_compact_attention_test,
        "compact_native_cmd_eos_main_test": cmd_eos_main_test,
        "compact_native_student_report": bge_report,
        "compact_native_heads2_lr_bracket_report": heads_report,
        "compact_native_heads2_lr_bracket_gate_log": heads_log,
        "compact_native_laststep_movement_report": laststep_report,
        "compact_native_bge_pre_retrieval_2000": pre2000,
        "compact_native_bge_post_retrieval_2000": post2000,
        "compact_native_bge_pre_retrieval_4000": pre4000,
        "compact_native_bge_post_retrieval_4000": post4000,
        "compact_native_bge_pre_listwise": pre_listwise,
        "compact_native_bge_post_listwise": post_listwise,
        "compact_native_bge_train_metrics": bge_train,
        "compact_native_laststep_pre_retrieval": laststep_pre,
        "compact_native_laststep_post_retrieval": laststep_post,
        "compact_native_laststep_train_metrics": laststep_train,
    }


class SummarizeCompactNativeReadinessTest(unittest.TestCase):
    def test_fixture_builds_evidence_positive_but_blocked_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = write_compact_native_fixture(root)
            output_tsv = root / "compact.tsv"
            summary = summarizer.build_summary(
                architecture_map_report=paths["compact_native_architecture_map_report"],
                manifest_checkpoint_foundation_report=paths["compact_native_manifest_checkpoint_foundation_report"],
                generic_bootstrap_report=paths["compact_native_generic_bootstrap_report"],
                compact_train_guard_report=paths["compact_native_train_guard_report"],
                serving_parity_report=paths["compact_native_serving_parity_report"],
                default_embedding_source=paths["compact_native_default_embedding_source"],
                runtime_embedding_model_source=paths["compact_native_runtime_embedding_model_source"],
                backend_tensor_ops_source=paths["compact_native_backend_tensor_ops_source"],
                default_embedding_test=paths["compact_native_default_embedding_test"],
                runtime_embedding_model_test=paths["compact_native_runtime_embedding_model_test"],
                backend_compact_attention_test=paths["compact_native_backend_compact_attention_test"],
                cmd_eos_main_test=paths["compact_native_cmd_eos_main_test"],
                bge_listwise_validation_report=paths["compact_native_student_report"],
                heads2_lr_bracket_report=paths["compact_native_heads2_lr_bracket_report"],
                heads2_lr_bracket_gate_log=paths["compact_native_heads2_lr_bracket_gate_log"],
                laststep_movement_report=paths["compact_native_laststep_movement_report"],
                bge_pre_retrieval_2000=paths["compact_native_bge_pre_retrieval_2000"],
                bge_post_retrieval_2000=paths["compact_native_bge_post_retrieval_2000"],
                bge_pre_retrieval_4000=paths["compact_native_bge_pre_retrieval_4000"],
                bge_post_retrieval_4000=paths["compact_native_bge_post_retrieval_4000"],
                bge_pre_listwise=paths["compact_native_bge_pre_listwise"],
                bge_post_listwise=paths["compact_native_bge_post_listwise"],
                bge_train_metrics=paths["compact_native_bge_train_metrics"],
                laststep_pre_retrieval=paths["compact_native_laststep_pre_retrieval"],
                laststep_post_retrieval=paths["compact_native_laststep_post_retrieval"],
                laststep_train_metrics=paths["compact_native_laststep_train_metrics"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )
            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(summary["schema"], summarizer.SUMMARY_SCHEMA)
        self.assertEqual(summary["status"], "evidence_positive_blocked_by_training_movement")
        self.assertFalse(summary["promotion_ready"])
        self.assertFalse(summary["training_ready"])
        self.assertFalse(summary["release_train_allowed"])
        self.assertFalse(summary["quality_claim"])
        self.assertEqual(summary["components"]["architecture_plan"]["status"], "pass")
        self.assertEqual(summary["components"]["manifest_checkpoint_foundation"]["status"], "evidence_ready")
        self.assertEqual(summary["components"]["generic_bootstrap"]["status"], "evidence_ready")
        self.assertEqual(summary["components"]["compact_train_guard"]["status"], "evidence_ready")
        self.assertEqual(summary["components"]["serving_parity"]["status"], "source_and_tests_ready")
        self.assertIn(
            "historical gate report exists but is superseded by current source/test evidence",
            summary["components"]["serving_parity"]["warnings"],
        )
        self.assertEqual(summary["components"]["bge_listwise_validation"]["status"], "evidence_positive")
        self.assertEqual(
            summary["components"]["heads2_lr_bracket"]["status"],
            "diagnostic_only_restored_best_static",
        )
        self.assertEqual(summary["components"]["laststep_movement"]["status"], "negative_diagnostic")
        self.assertGreater(
            summary["components"]["bge_listwise_validation"]["details"]["retrieval_2000"]["ndcg_at_10"]["delta"],
            0,
        )
        self.assertLess(
            summary["components"]["laststep_movement"]["details"]["retrieval"]["recall_at_100"]["delta"],
            0,
        )
        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("component", "serving_parity")]["status"], "source_and_tests_ready")

    def test_missing_inputs_do_not_raise_and_report_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = summarizer.build_summary(
                architecture_map_report=root / "missing-architecture.md",
                manifest_checkpoint_foundation_report=root / "missing-foundation.md",
                generic_bootstrap_report=root / "missing-bootstrap.md",
                compact_train_guard_report=root / "missing-guard.md",
                serving_parity_report=root / "missing-serving.md",
                default_embedding_source=root / "missing-default-embedding.go",
                runtime_embedding_model_source=root / "missing-runtime-embedding.go",
                backend_tensor_ops_source=root / "missing-tensor-ops.go",
                default_embedding_test=root / "missing-default-embedding-test.go",
                runtime_embedding_model_test=root / "missing-runtime-embedding-test.go",
                backend_compact_attention_test=root / "missing-compact-attention-test.go",
                cmd_eos_main_test=root / "missing-main-test.go",
                bge_listwise_validation_report=root / "missing-bge.md",
                heads2_lr_bracket_report=root / "missing-heads.md",
                heads2_lr_bracket_gate_log=root / "missing-heads.log",
                laststep_movement_report=root / "missing-laststep.md",
                bge_pre_retrieval_2000=root / "missing-pre2000.json",
                bge_post_retrieval_2000=root / "missing-post2000.json",
                bge_pre_retrieval_4000=root / "missing-pre4000.json",
                bge_post_retrieval_4000=root / "missing-post4000.json",
                bge_pre_listwise=root / "missing-pre-listwise.json",
                bge_post_listwise=root / "missing-post-listwise.json",
                bge_train_metrics=root / "missing-train.json",
                laststep_pre_retrieval=root / "missing-laststep-pre.json",
                laststep_post_retrieval=root / "missing-laststep-post.json",
                laststep_train_metrics=root / "missing-laststep-train.json",
            )

        self.assertEqual(summary["status"], "partial_evidence_waiting_validation")
        self.assertEqual(summary["components"]["architecture_plan"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["serving_parity"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["bge_listwise_validation"]["status"], "missing_evidence")
        self.assertEqual(summary["components"]["laststep_movement"]["status"], "missing_evidence")
        self.assertTrue(summary["blockers"])
        self.assertFalse(summary["promotion_ready"])


if __name__ == "__main__":
    unittest.main()
