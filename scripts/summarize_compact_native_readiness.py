#!/usr/bin/env python3
"""Summarize compact native student readiness from read-only evidence.

This utility intentionally reports evidence and blockers only. It does not
claim promotion, training, or release readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "eos.compact_native_readiness_summary.v1"
DEFAULT_OUTPUT_JSON = ".tiller/scratch/codex/compact-native-readiness-current.json"
DEFAULT_OUTPUT_TSV = ".tiller/scratch/codex/compact-native-readiness-current.tsv"

DEFAULT_ARCHITECTURE_MAP_REPORT = ".tiller/scratch/codex/compact-native-student-architecture-map-v1-report.md"
DEFAULT_MANIFEST_CHECKPOINT_FOUNDATION_REPORT = (
    ".tiller/scratch/codex/compact-native-manifest-checkpoint-foundation-v1-report.md"
)
DEFAULT_GENERIC_BOOTSTRAP_REPORT = ".tiller/scratch/codex/compact-native-generic-bootstrap-v1-report.md"
DEFAULT_COMPACT_TRAIN_GUARD_REPORT = ".tiller/scratch/codex/compact-native-batched-guard-v1-report.md"
DEFAULT_SERVING_PARITY_REPORT = ".tiller/scratch/codex/compact-native-serving-parity-v1-report.md"
DEFAULT_MODELS_DEFAULT_EMBEDDING_SOURCE = "models/default_embedding.go"
DEFAULT_RUNTIME_EMBEDDING_MODEL_SOURCE = "runtime/embedding_model.go"
DEFAULT_RUNTIME_BACKEND_TENSOR_OPS_SOURCE = "runtime/backend/tensor_ops.go"
DEFAULT_MODELS_DEFAULT_EMBEDDING_TEST = "models/default_embedding_test.go"
DEFAULT_RUNTIME_EMBEDDING_MODEL_TEST = "runtime/embedding_model_test.go"
DEFAULT_RUNTIME_EMBEDDING_TRAINER_TEST = "runtime/embedding_trainer_test.go"
DEFAULT_RUNTIME_BACKEND_COMPACT_ATTENTION_TEST = "runtime/backend/compact_attention_ops_test.go"
DEFAULT_CMD_EOS_MAIN_TEST = "cmd/eos/main_test.go"
DEFAULT_BGE_LISTWISE_VALIDATION_REPORT = (
    ".tiller/scratch/codex/compact-bge-listwise-larger-validation-split211-seed191-v1-report.md"
)
DEFAULT_HEADS2_LR_BRACKET_REPORT = ".tiller/scratch/codex/compact-native-heads2-lr-bracket-v1-report.md"
DEFAULT_HEADS2_LR_BRACKET_GATE_LOG = (
    "runs/compact-native-heads2-lr-bracket-v1-20260628T215111Z/diagnostics/verification-gate.stdout.log"
)
DEFAULT_HEADS2_TRAIN_METRICS = (
    "runs/compact-native-heads2-lr-bracket-v1-20260628T215111Z/"
    "arms/heads2_lr2e-4_t05/metrics/train.metrics.json"
)
DEFAULT_HEADS2_TRAIN_STDOUT_LOG = (
    "runs/compact-native-heads2-lr-bracket-v1-20260628T215111Z/"
    "arms/heads2_lr2e-4_t05/logs/train.stdout.log"
)
DEFAULT_HEADS2_TRAIN_TIME_LOG = (
    "runs/compact-native-heads2-lr-bracket-v1-20260628T215111Z/"
    "arms/heads2_lr2e-4_t05/logs/train.time.log"
)
DEFAULT_LASTSTEP_MOVEMENT_REPORT = ".tiller/scratch/codex/compact-native-laststep-movement-diagnostic-v1-report.md"

DEFAULT_BGE_PRE_RETRIEVAL_2000 = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/pre.retrieval.2000.metrics.json"
)
DEFAULT_BGE_POST_RETRIEVAL_2000 = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/post.retrieval.2000.metrics.json"
)
DEFAULT_BGE_PRE_RETRIEVAL_4000 = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/pre.retrieval.4000.metrics.json"
)
DEFAULT_BGE_POST_RETRIEVAL_4000 = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/post.retrieval.4000.metrics.json"
)
DEFAULT_BGE_PRE_LISTWISE = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/pre.listwise.metrics.json"
)
DEFAULT_BGE_POST_LISTWISE = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/post.listwise.metrics.json"
)
DEFAULT_BGE_TRAIN_METRICS = (
    "runs/compact-bge-listwise-larger-validation-split211-seed191-v1-20260629T122406Z/"
    "arms/lr2e-5_seed191/metrics/train.metrics.json"
)
DEFAULT_LASTSTEP_PRE_RETRIEVAL = (
    "runs/compact-native-laststep-movement-diagnostic-v1-20260628T222536Z/"
    "arms/heads2_lr2e-4_t05_no_restore/metrics/pre.retrieval.metrics.json"
)
DEFAULT_LASTSTEP_POST_RETRIEVAL = (
    "runs/compact-native-laststep-movement-diagnostic-v1-20260628T222536Z/"
    "arms/heads2_lr2e-4_t05_no_restore/metrics/post.retrieval.metrics.json"
)
DEFAULT_LASTSTEP_TRAIN_METRICS = (
    "runs/compact-native-laststep-movement-diagnostic-v1-20260628T222536Z/"
    "arms/heads2_lr2e-4_t05_no_restore/metrics/train.metrics.json"
)


class ReadinessError(ValueError):
    """Raised when readiness outputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_optional_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"missing_evidence: {path}"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"invalid_json_shape: {path}: expected object"
    return data, None


def read_optional_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"missing_evidence: {path}"
    except UnicodeDecodeError as exc:
        return None, f"invalid_markdown: {path}: {exc}"


def nested_dict(data: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def component(
    *,
    component_id: str,
    status: str,
    blockers: list[str],
    details: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "status": status,
        "blockers": blockers,
        "warnings": warnings or [],
        "details": details,
    }


def marker_report_component(
    *,
    component_id: str,
    report_path: Path,
    required_markers: dict[str, str],
    status_when_present: str,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    text, error = read_optional_text(report_path)
    blockers = [error] if error else []
    checks = {"exists": text is not None}
    for name, marker in required_markers.items():
        checks[name] = bool(text and marker in text)
        if not checks[name]:
            blockers.append(f"{component_id} marker missing: {name}")
    if error:
        status = "missing_evidence"
    elif blockers:
        status = "evidence_incomplete"
    else:
        status = status_when_present
    return component(
        component_id=component_id,
        status=status,
        blockers=blockers,
        warnings=warnings,
        details={
            "report_path": str(report_path),
            "checks": checks,
            **(details or {}),
        },
    )


def summarize_architecture_plan(report_path: Path) -> dict[str, Any]:
    return marker_report_component(
        component_id="architecture_plan",
        report_path=report_path,
        required_markers={
            "implementation_slices": "## Implementation Slices",
            "verification_plan": "## Verification Plan",
        },
        status_when_present="pass",
    )


def summarize_manifest_checkpoint_foundation(report_path: Path) -> dict[str, Any]:
    return marker_report_component(
        component_id="manifest_checkpoint_foundation",
        report_path=report_path,
        required_markers={
            "architecture_metadata": "architecture metadata",
            "generic_checkpoint_tensor_retention": "generic checkpoint tensor retention",
        },
        status_when_present="evidence_ready",
        details={
            "adds_architecture_metadata": True,
            "adds_generic_checkpoint_tensor_retention": True,
        },
    )


def summarize_generic_bootstrap(report_path: Path) -> dict[str, Any]:
    return marker_report_component(
        component_id="generic_bootstrap",
        report_path=report_path,
        required_markers={
            "exact_name_generic_tensor_bootstrap": "exact-name tensors",
            "overlap_copy": "copyOverlappingTensor",
        },
        status_when_present="evidence_ready",
        details={"exact_name_generic_tensor_bootstrap": True},
    )


def summarize_compact_train_guard(report_path: Path) -> dict[str, Any]:
    return marker_report_component(
        component_id="compact_train_guard",
        report_path=report_path,
        required_markers={
            "scalar_fallback_guards": "scalar-fallback guards",
            "compact_training_guarded": "Compact backprop/optimizer updates remain explicitly unsupported",
        },
        status_when_present="evidence_ready",
        details={
            "scalar_fallback_guarded": True,
            "compact_training_guarded_caveat": "Compact backprop/optimizer updates remain unsupported.",
        },
    )


def required_text_checks(
    *, component_id: str, path: Path, required_markers: dict[str, str]
) -> tuple[dict[str, bool], list[str]]:
    text, error = read_optional_text(path)
    blockers = [error] if error else []
    checks = {"exists": text is not None}
    for name, marker in required_markers.items():
        checks[name] = bool(text and marker in text)
        if not checks[name]:
            blockers.append(f"{component_id} marker missing in {path}: {name}")
    return checks, blockers


def summarize_serving_parity(
    *,
    report_path: Path,
    default_embedding_source: Path,
    runtime_embedding_model_source: Path,
    backend_tensor_ops_source: Path,
    default_embedding_test: Path,
    runtime_embedding_model_test: Path,
    runtime_embedding_trainer_test: Path,
    backend_compact_attention_test: Path,
    cmd_eos_main_test: Path,
) -> dict[str, Any]:
    report_text, report_error = read_optional_text(report_path)
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    marker_specs = {
        "default_embedding_source": (
            default_embedding_source,
            {
                "attention_heads_branch": "if cfg.AttentionHeads > 1",
                "generated_multihead_call": "compact_multihead_attention_h%d",
            },
        ),
        "runtime_embedding_model_source": (
            runtime_embedding_model_source,
            {
                "mask_accepts_compact_multihead": (
                    'moduleHasKernelOp(mod, "masked_softmax") && '
                    '!moduleHasKernelOp(mod, "compact_multihead_attention")'
                ),
                "scale_accepts_compact_multihead": (
                    '!moduleHasScaledAttentionMatMul(mod) && '
                    '!moduleHasKernelOp(mod, "compact_multihead_attention")'
                ),
            },
        ),
        "backend_tensor_ops_source": (
            backend_tensor_ops_source,
            {
                "dispatches_compact_multihead": 'case "compact_multihead_attention"',
                "implements_compact_multihead_tensor": "func compactMultiheadAttentionTensor",
                "validates_num_attention_heads": "num_attention_heads",
                "validates_head_divisibility": "hidden%heads != 0",
            },
        ),
        "default_embedding_test": (
            default_embedding_test,
            {
                "generated_package_load_embed_test": (
                    "TestInitDefaultEmbeddingPackageCreatesCompactMultiHeadServingGraph"
                ),
                "asserts_num_attention_heads": "compact_multihead_attention num_attention_heads = %q, want 2",
            },
        ),
        "runtime_embedding_model_test": (
            runtime_embedding_model_test,
            {
                "runtime_load_embed_test": "TestLoadEmbeddingAcceptsCompactMultiHeadServingGraph",
                "runtime_compact_multihead_source": "compact_multihead_attention_h2",
            },
        ),
        "runtime_embedding_trainer_test": (
            runtime_embedding_trainer_test,
            {
                "trainer_serving_parity_test": "TestCompactEmbeddingTrainerServingPackageVectorParity",
                "compact_checkpoint": "compactTrainStateTestCheckpoint(3)",
                "writes_embedding_package": "trainer.WriteEmbeddingPackage",
                "serving_embed_with_role": "model.EmbedWithRole",
                "trainer_compact_encode": "trainer.encodeCompactSequence",
                "float_tolerance_assertion": "assertFloat32SlicesClose",
            },
        ),
        "backend_compact_attention_test": (
            backend_compact_attention_test,
            {
                "reference_test": "TestCompactMultiheadAttentionTensorMatchesReference",
                "batched_masked_test": "TestCompactMultiheadAttentionTensorBatchedMasked",
            },
        ),
        "cmd_eos_main_test": (
            cmd_eos_main_test,
            {
                "cli_init_model_test": "TestRunInitModelCreatesCompactMultiHeadServingGraph",
                "cli_attention_heads_flag": '"--attention-heads", "2"',
            },
        ),
    }
    source_paths: dict[str, str] = {}
    for name, (path, markers) in marker_specs.items():
        path_checks, path_blockers = required_text_checks(component_id="serving_parity", path=path, required_markers=markers)
        checks[name] = path_checks
        blockers.extend(path_blockers)
        source_paths[name] = str(path)

    historical_gate = bool(report_text and "This is a gate, not true serving parity" in report_text)
    if report_error:
        warnings = [report_error]
    elif historical_gate:
        warnings = ["historical gate report exists but is superseded by current source/test evidence"]
    else:
        warnings = ["historical serving gate report is absent or no longer has the expected stale-gate wording"]

    if blockers:
        status = "evidence_incomplete" if report_text is not None or any(item.get("exists") for item in checks.values()) else "missing_evidence"
    else:
        status = "source_and_tests_ready"
    return component(
        component_id="serving_parity",
        status=status,
        blockers=blockers,
        warnings=warnings,
        details={
            "report_path": str(report_path),
            "source_paths": source_paths,
            "checks": checks,
            "historical_report_is_gate_not_fix": historical_gate,
            "focused_go_verification": [
                "go test ./runtime/backend -run TestCompactMultiheadAttentionTensor -count=1",
                "go test ./runtime -run TestLoadEmbeddingAcceptsCompactMultiHeadServingGraph -count=1",
                "go test ./runtime -run TestCompactEmbeddingTrainerServingPackageVectorParity -count=1",
                "go test ./models -run TestInitDefaultEmbeddingPackageCreatesCompactMultiHeadServingGraph -count=1",
                "go test ./cmd/eos -run TestRunInitModelCreatesCompactMultiHeadServingGraph -count=1",
            ],
            "trainer_serving_numeric_parity": True,
            "numeric_parity_evidence": (
                "Exact trainer-to-serving package vector parity is covered by "
                "TestCompactEmbeddingTrainerServingPackageVectorParity for a compact two-head/two-repeat "
                "checkpoint and f16 export tolerance."
            ),
        },
    )


def quality_metric(metrics: dict[str, Any] | None, key: str) -> float | None:
    return as_number(nested_dict(metrics, "quality").get(key))


def retrieval_delta(pre: dict[str, Any] | None, post: dict[str, Any] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in ("ndcg_at_10", "map_at_100", "recall_at_100"):
        pre_value = quality_metric(pre, key)
        post_value = quality_metric(post, key)
        values[key] = {
            "pre": pre_value,
            "post": post_value,
            "delta": post_value - pre_value if pre_value is not None and post_value is not None else None,
        }
    return values


def deltas_all_positive(deltas: dict[str, Any]) -> bool:
    return all(isinstance(item.get("delta"), float) and item["delta"] > 0 for item in deltas.values())


def deltas_all_negative(deltas: dict[str, Any]) -> bool:
    return all(isinstance(item.get("delta"), float) and item["delta"] < 0 for item in deltas.values())


def listwise_eval(metrics: dict[str, Any] | None) -> dict[str, Any]:
    return nested_dict(metrics, "last_listwise_geometry_eval") or nested_dict(metrics, "final_listwise_geometry_eval")


def listwise_deltas(pre: dict[str, Any] | None, post: dict[str, Any] | None) -> dict[str, Any]:
    pre_eval = listwise_eval(pre)
    post_eval = listwise_eval(post)
    values: dict[str, Any] = {}
    for key in ("teacher_cross_entropy", "teacher_kl"):
        pre_value = as_number(pre_eval.get(key))
        post_value = as_number(post_eval.get(key))
        values[key] = {
            "pre": pre_value,
            "post": post_value,
            "delta": post_value - pre_value if pre_value is not None and post_value is not None else None,
        }
    for key in ("teacher_top1_agreement", "any_positive_top1"):
        pre_value = as_number(pre_eval.get(key))
        post_value = as_number(post_eval.get(key))
        values[key] = {
            "pre": pre_value,
            "post": post_value,
            "delta": post_value - pre_value if pre_value is not None and post_value is not None else None,
        }
    return values


def losses_decrease(deltas: dict[str, Any]) -> bool:
    return all(
        isinstance(deltas.get(key, {}).get("delta"), float) and deltas[key]["delta"] < 0
        for key in ("teacher_cross_entropy", "teacher_kl")
    )


def summarize_bge_listwise_validation(
    *,
    report_path: Path,
    pre_retrieval_2000: Path,
    post_retrieval_2000: Path,
    pre_retrieval_4000: Path,
    post_retrieval_4000: Path,
    pre_listwise: Path,
    post_listwise: Path,
    train_metrics: Path,
) -> dict[str, Any]:
    paths = {
        "pre_retrieval_2000": pre_retrieval_2000,
        "post_retrieval_2000": post_retrieval_2000,
        "pre_retrieval_4000": pre_retrieval_4000,
        "post_retrieval_4000": post_retrieval_4000,
        "pre_listwise": pre_listwise,
        "post_listwise": post_listwise,
        "train_metrics": train_metrics,
    }
    loaded = {name: load_optional_json(path) for name, path in paths.items()}
    blockers = [error for _data, error in loaded.values() if error]
    report_text, report_error = read_optional_text(report_path)
    if report_error:
        blockers.append(report_error)

    retrieval_2000 = retrieval_delta(loaded["pre_retrieval_2000"][0], loaded["post_retrieval_2000"][0])
    retrieval_4000 = retrieval_delta(loaded["pre_retrieval_4000"][0], loaded["post_retrieval_4000"][0])
    listwise = listwise_deltas(loaded["pre_listwise"][0], loaded["post_listwise"][0])
    metrics_positive = deltas_all_positive(retrieval_2000) and deltas_all_positive(retrieval_4000)
    listwise_positive = losses_decrease(listwise)

    if not blockers and metrics_positive and listwise_positive:
        status = "evidence_positive"
    elif report_text is not None:
        status = "report_only_missing_metrics"
        blockers.append("structured BGE listwise metrics are missing or incomplete; using report existence only")
    else:
        status = "missing_evidence"

    train_summary = nested_dict(loaded["train_metrics"][0], "summary")
    return component(
        component_id="bge_listwise_validation",
        status=status,
        blockers=blockers,
        details={
            "report_path": str(report_path),
            "metric_paths": {name: str(path) for name, path in paths.items()},
            "retrieval_2000": retrieval_2000,
            "retrieval_4000": retrieval_4000,
            "listwise": listwise,
            "checks": {
                "retrieval_2000_positive": deltas_all_positive(retrieval_2000),
                "retrieval_4000_positive": deltas_all_positive(retrieval_4000),
                "listwise_ce_kl_decrease": listwise_positive,
            },
            "train_summary": {
                "steps_run": train_summary.get("steps_run"),
                "optimizer_updates": nested_dict(loaded["train_metrics"][0], "profile_delta").get("optimizer_updates"),
                "restored_best": train_summary.get("restored_best"),
                "best_step": train_summary.get("best_step"),
            },
        },
    )


def summarize_heads2_lr_bracket(
    *,
    report_path: Path,
    verification_log: Path,
    train_metrics: Path,
    train_stdout_log: Path,
    train_time_log: Path,
) -> dict[str, Any]:
    text, error = read_optional_text(report_path)
    log_text, log_error = read_optional_text(verification_log)
    metrics, metrics_error = load_optional_json(train_metrics)
    stdout_text, stdout_error = read_optional_text(train_stdout_log)
    time_text, time_error = read_optional_text(train_time_log)
    blockers = [item for item in (error, log_error, metrics_error, stdout_error, time_error) if item]
    config = nested_dict(metrics, "config")
    train_summary = nested_dict(metrics, "summary")
    profile_delta = nested_dict(metrics, "profile_delta")
    retrieval_eval_dir_enabled = bool(time_text and "--retrieval-eval-dir" in time_text)
    retrieval_ndcg_reported_zero = bool(stdout_text and "retrieval_ndcg=0.000000" in stdout_text)
    checks = {
        "report_exists": text is not None,
        "verification_log_exists": log_text is not None,
        "train_metrics_exists": metrics is not None,
        "train_stdout_log_exists": stdout_text is not None,
        "train_time_log_exists": time_text is not None,
        "best_step_0": bool(text and "best_step=0" in text),
        "restored_best_true": bool(text and "restored_best=true" in text),
        "verification_passed": bool(log_text and "PASS compact-native-heads2-lr-bracket-v1" in log_text),
        "selection_metric_top1_accuracy": config.get("select_metric") == "top1_accuracy",
        "restore_best_true": config.get("restore_best") is True,
        "grouped_infonce": config.get("contrastive_loss") == "grouped_infonce",
        "hard_negative_train": config.get("hard_negative_train") is True,
        "temperature_0_05": as_number(config.get("temperature")) == 0.05,
        "retrieval_eval_dir_disabled": not retrieval_eval_dir_enabled if time_text is not None else False,
        "retrieval_ndcg_reported_zero": retrieval_ndcg_reported_zero,
    }
    diagnosis = {
        "selection_metric": config.get("select_metric"),
        "restore_best": config.get("restore_best"),
        "best_step": train_summary.get("best_step"),
        "restored_best": train_summary.get("restored_best"),
        "optimizer_updates": profile_delta.get("optimizer_updates"),
        "contrastive_loss": config.get("contrastive_loss"),
        "hard_negative_train": config.get("hard_negative_train"),
        "temperature": config.get("temperature"),
        "retrieval_eval_dir_enabled": retrieval_eval_dir_enabled,
        "retrieval_ndcg_reported_zero": retrieval_ndcg_reported_zero,
        "primary_root_cause": "restore-best selected pairwise top1 while retrieval selection was disabled",
        "secondary_root_cause": (
            "score-free grouped InfoNCE at lr=2e-4 temperature=0.05 moved but retrieval degraded when not restored"
        ),
        "next_descriptor_id": "compact-native-retrieval-gated-low-lr-v1",
    }
    return component(
        component_id="heads2_lr_bracket",
        status="diagnostic_only_restored_best_static" if not blockers and checks["best_step_0"] else "missing_evidence",
        blockers=blockers,
        details={
            "report_path": str(report_path),
            "verification_log": str(verification_log),
            "metric_paths": {"train_metrics": str(train_metrics)},
            "log_paths": {
                "train_stdout_log": str(train_stdout_log),
                "train_time_log": str(train_time_log),
            },
            "checks": checks,
            "diagnosis": diagnosis,
            "interpretation": "No productive movement: exported packages restored best_step=0.",
        },
    )


def summarize_laststep_movement(
    *,
    report_path: Path,
    pre_retrieval: Path,
    post_retrieval: Path,
    train_metrics: Path,
) -> dict[str, Any]:
    pre, pre_error = load_optional_json(pre_retrieval)
    post, post_error = load_optional_json(post_retrieval)
    train, train_error = load_optional_json(train_metrics)
    text, report_error = read_optional_text(report_path)
    blockers = [item for item in (pre_error, post_error, train_error, report_error) if item]
    deltas = retrieval_delta(pre, post)
    train_summary = nested_dict(train, "summary")
    if not blockers and deltas_all_negative(deltas):
        status = "negative_diagnostic"
    elif text is not None:
        status = "report_only_missing_metrics"
        blockers.append("structured last-step movement metrics are missing or incomplete")
    else:
        status = "missing_evidence"
    return component(
        component_id="laststep_movement",
        status=status,
        blockers=blockers,
        details={
            "report_path": str(report_path),
            "metric_paths": {
                "pre_retrieval": str(pre_retrieval),
                "post_retrieval": str(post_retrieval),
                "train_metrics": str(train_metrics),
            },
            "retrieval": deltas,
            "checks": {"retrieval_deltas_negative": deltas_all_negative(deltas)},
            "train_summary": {
                "steps_run": train_summary.get("steps_run"),
                "best_step": train_summary.get("best_step"),
                "restored_best": train_summary.get("restored_best"),
                "optimizer_updates": nested_dict(train, "profile_delta").get("optimizer_updates"),
            },
        },
    )


def overall_status(components: dict[str, dict[str, Any]]) -> str:
    if components["bge_listwise_validation"]["status"] == "evidence_positive":
        serving_ready = components["serving_parity"]["status"] == "source_and_tests_ready"
        training_blocked = (
            components["laststep_movement"]["status"] == "negative_diagnostic"
            or components["heads2_lr_bracket"]["status"] == "diagnostic_only_restored_best_static"
        )
        if serving_ready and training_blocked:
            return "evidence_positive_blocked_by_training_movement"
        if training_blocked or components["serving_parity"]["status"] in {"evidence_incomplete", "missing_evidence"}:
            return "evidence_positive_blocked_by_serving_and_training"
    return "partial_evidence_waiting_validation"


def build_summary(
    *,
    architecture_map_report: Path = Path(DEFAULT_ARCHITECTURE_MAP_REPORT),
    manifest_checkpoint_foundation_report: Path = Path(DEFAULT_MANIFEST_CHECKPOINT_FOUNDATION_REPORT),
    generic_bootstrap_report: Path = Path(DEFAULT_GENERIC_BOOTSTRAP_REPORT),
    compact_train_guard_report: Path = Path(DEFAULT_COMPACT_TRAIN_GUARD_REPORT),
    serving_parity_report: Path = Path(DEFAULT_SERVING_PARITY_REPORT),
    default_embedding_source: Path = Path(DEFAULT_MODELS_DEFAULT_EMBEDDING_SOURCE),
    runtime_embedding_model_source: Path = Path(DEFAULT_RUNTIME_EMBEDDING_MODEL_SOURCE),
    backend_tensor_ops_source: Path = Path(DEFAULT_RUNTIME_BACKEND_TENSOR_OPS_SOURCE),
    default_embedding_test: Path = Path(DEFAULT_MODELS_DEFAULT_EMBEDDING_TEST),
    runtime_embedding_model_test: Path = Path(DEFAULT_RUNTIME_EMBEDDING_MODEL_TEST),
    runtime_embedding_trainer_test: Path = Path(DEFAULT_RUNTIME_EMBEDDING_TRAINER_TEST),
    backend_compact_attention_test: Path = Path(DEFAULT_RUNTIME_BACKEND_COMPACT_ATTENTION_TEST),
    cmd_eos_main_test: Path = Path(DEFAULT_CMD_EOS_MAIN_TEST),
    bge_listwise_validation_report: Path = Path(DEFAULT_BGE_LISTWISE_VALIDATION_REPORT),
    heads2_lr_bracket_report: Path = Path(DEFAULT_HEADS2_LR_BRACKET_REPORT),
    heads2_lr_bracket_gate_log: Path = Path(DEFAULT_HEADS2_LR_BRACKET_GATE_LOG),
    heads2_train_metrics: Path = Path(DEFAULT_HEADS2_TRAIN_METRICS),
    heads2_train_stdout_log: Path = Path(DEFAULT_HEADS2_TRAIN_STDOUT_LOG),
    heads2_train_time_log: Path = Path(DEFAULT_HEADS2_TRAIN_TIME_LOG),
    laststep_movement_report: Path = Path(DEFAULT_LASTSTEP_MOVEMENT_REPORT),
    bge_pre_retrieval_2000: Path = Path(DEFAULT_BGE_PRE_RETRIEVAL_2000),
    bge_post_retrieval_2000: Path = Path(DEFAULT_BGE_POST_RETRIEVAL_2000),
    bge_pre_retrieval_4000: Path = Path(DEFAULT_BGE_PRE_RETRIEVAL_4000),
    bge_post_retrieval_4000: Path = Path(DEFAULT_BGE_POST_RETRIEVAL_4000),
    bge_pre_listwise: Path = Path(DEFAULT_BGE_PRE_LISTWISE),
    bge_post_listwise: Path = Path(DEFAULT_BGE_POST_LISTWISE),
    bge_train_metrics: Path = Path(DEFAULT_BGE_TRAIN_METRICS),
    laststep_pre_retrieval: Path = Path(DEFAULT_LASTSTEP_PRE_RETRIEVAL),
    laststep_post_retrieval: Path = Path(DEFAULT_LASTSTEP_POST_RETRIEVAL),
    laststep_train_metrics: Path = Path(DEFAULT_LASTSTEP_TRAIN_METRICS),
    clock: Any = utc_now,
) -> dict[str, Any]:
    components = {
        "architecture_plan": summarize_architecture_plan(architecture_map_report),
        "manifest_checkpoint_foundation": summarize_manifest_checkpoint_foundation(
            manifest_checkpoint_foundation_report
        ),
        "generic_bootstrap": summarize_generic_bootstrap(generic_bootstrap_report),
        "compact_train_guard": summarize_compact_train_guard(compact_train_guard_report),
        "serving_parity": summarize_serving_parity(
            report_path=serving_parity_report,
            default_embedding_source=default_embedding_source,
            runtime_embedding_model_source=runtime_embedding_model_source,
            backend_tensor_ops_source=backend_tensor_ops_source,
            default_embedding_test=default_embedding_test,
            runtime_embedding_model_test=runtime_embedding_model_test,
            runtime_embedding_trainer_test=runtime_embedding_trainer_test,
            backend_compact_attention_test=backend_compact_attention_test,
            cmd_eos_main_test=cmd_eos_main_test,
        ),
        "bge_listwise_validation": summarize_bge_listwise_validation(
            report_path=bge_listwise_validation_report,
            pre_retrieval_2000=bge_pre_retrieval_2000,
            post_retrieval_2000=bge_post_retrieval_2000,
            pre_retrieval_4000=bge_pre_retrieval_4000,
            post_retrieval_4000=bge_post_retrieval_4000,
            pre_listwise=bge_pre_listwise,
            post_listwise=bge_post_listwise,
            train_metrics=bge_train_metrics,
        ),
        "heads2_lr_bracket": summarize_heads2_lr_bracket(
            report_path=heads2_lr_bracket_report,
            verification_log=heads2_lr_bracket_gate_log,
            train_metrics=heads2_train_metrics,
            train_stdout_log=heads2_train_stdout_log,
            train_time_log=heads2_train_time_log,
        ),
        "laststep_movement": summarize_laststep_movement(
            report_path=laststep_movement_report,
            pre_retrieval=laststep_pre_retrieval,
            post_retrieval=laststep_post_retrieval,
            train_metrics=laststep_train_metrics,
        ),
    }
    status = overall_status(components)
    blockers = [
        blocker
        for item in components.values()
        for blocker in item.get("blockers", [])
        if item.get("status") not in {"pass", "evidence_ready", "evidence_positive"}
    ]
    if components["serving_parity"]["status"] != "source_and_tests_ready":
        blockers.append("compact multi-head serving source/test evidence is incomplete")
    if components["laststep_movement"]["status"] == "negative_diagnostic":
        blockers.append("last-step/no-restore compact native training movement is retrieval-negative")
    if components["heads2_lr_bracket"]["status"] == "diagnostic_only_restored_best_static":
        blockers.append("heads=2 LR bracket restored best_step=0 and does not prove productive movement")

    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "status": status,
        "promotion_ready": False,
        "training_ready": False,
        "release_train_allowed": False,
        "quality_claim": False,
        "blockers": blockers,
        "next_safe_action": (
            "Wait for FiQA export to clear, then run compact-native-retrieval-gated-low-lr-v1: "
            "a retrieval-gated low-LR diagnostic before any promotion, training readiness, or release readiness claim."
        ),
        "components": components,
        "evidence_paths": {
            "architecture_map_report": str(architecture_map_report),
            "manifest_checkpoint_foundation_report": str(manifest_checkpoint_foundation_report),
            "generic_bootstrap_report": str(generic_bootstrap_report),
            "compact_train_guard_report": str(compact_train_guard_report),
            "serving_parity_report": str(serving_parity_report),
            "default_embedding_source": str(default_embedding_source),
            "runtime_embedding_model_source": str(runtime_embedding_model_source),
            "backend_tensor_ops_source": str(backend_tensor_ops_source),
            "default_embedding_test": str(default_embedding_test),
            "runtime_embedding_model_test": str(runtime_embedding_model_test),
            "runtime_embedding_trainer_test": str(runtime_embedding_trainer_test),
            "backend_compact_attention_test": str(backend_compact_attention_test),
            "cmd_eos_main_test": str(cmd_eos_main_test),
            "bge_listwise_validation_report": str(bge_listwise_validation_report),
            "heads2_lr_bracket_report": str(heads2_lr_bracket_report),
            "heads2_lr_bracket_gate_log": str(heads2_lr_bracket_gate_log),
            "heads2_train_metrics": str(heads2_train_metrics),
            "heads2_train_stdout_log": str(heads2_train_stdout_log),
            "heads2_train_time_log": str(heads2_train_time_log),
            "laststep_movement_report": str(laststep_movement_report),
        },
    }


def format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def tsv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "section": "summary",
            "key": "status",
            "status": summary["status"],
            "value": summary["next_safe_action"],
            "blockers": " | ".join(str(item) for item in summary.get("blockers", [])[:12]),
            "details": {
                "promotion_ready": summary["promotion_ready"],
                "training_ready": summary["training_ready"],
                "release_train_allowed": summary["release_train_allowed"],
                "quality_claim": summary["quality_claim"],
            },
        }
    ]
    for component_id, item in summary["components"].items():
        rows.append(
            {
                "section": "component",
                "key": component_id,
                "status": item["status"],
                "value": "",
                "blockers": " | ".join(str(blocker) for blocker in item.get("blockers", [])[:8]),
                "details": item.get("details", {}),
            }
        )
    return rows


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "key", "status", "value", "blockers", "details"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in tsv_rows(summary):
            writer.writerow({key: format_tsv_value(row.get(key)) for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-tsv", default=DEFAULT_OUTPUT_TSV)
    parser.add_argument("--architecture-map-report", default=DEFAULT_ARCHITECTURE_MAP_REPORT)
    parser.add_argument("--manifest-checkpoint-foundation-report", default=DEFAULT_MANIFEST_CHECKPOINT_FOUNDATION_REPORT)
    parser.add_argument("--generic-bootstrap-report", default=DEFAULT_GENERIC_BOOTSTRAP_REPORT)
    parser.add_argument("--compact-train-guard-report", default=DEFAULT_COMPACT_TRAIN_GUARD_REPORT)
    parser.add_argument("--serving-parity-report", default=DEFAULT_SERVING_PARITY_REPORT)
    parser.add_argument("--default-embedding-source", default=DEFAULT_MODELS_DEFAULT_EMBEDDING_SOURCE)
    parser.add_argument("--runtime-embedding-model-source", default=DEFAULT_RUNTIME_EMBEDDING_MODEL_SOURCE)
    parser.add_argument("--backend-tensor-ops-source", default=DEFAULT_RUNTIME_BACKEND_TENSOR_OPS_SOURCE)
    parser.add_argument("--default-embedding-test", default=DEFAULT_MODELS_DEFAULT_EMBEDDING_TEST)
    parser.add_argument("--runtime-embedding-model-test", default=DEFAULT_RUNTIME_EMBEDDING_MODEL_TEST)
    parser.add_argument("--runtime-embedding-trainer-test", default=DEFAULT_RUNTIME_EMBEDDING_TRAINER_TEST)
    parser.add_argument("--backend-compact-attention-test", default=DEFAULT_RUNTIME_BACKEND_COMPACT_ATTENTION_TEST)
    parser.add_argument("--cmd-eos-main-test", default=DEFAULT_CMD_EOS_MAIN_TEST)
    parser.add_argument("--bge-listwise-validation-report", default=DEFAULT_BGE_LISTWISE_VALIDATION_REPORT)
    parser.add_argument("--heads2-lr-bracket-report", default=DEFAULT_HEADS2_LR_BRACKET_REPORT)
    parser.add_argument("--heads2-lr-bracket-gate-log", default=DEFAULT_HEADS2_LR_BRACKET_GATE_LOG)
    parser.add_argument("--heads2-train-metrics", default=DEFAULT_HEADS2_TRAIN_METRICS)
    parser.add_argument("--heads2-train-stdout-log", default=DEFAULT_HEADS2_TRAIN_STDOUT_LOG)
    parser.add_argument("--heads2-train-time-log", default=DEFAULT_HEADS2_TRAIN_TIME_LOG)
    parser.add_argument("--laststep-movement-report", default=DEFAULT_LASTSTEP_MOVEMENT_REPORT)
    parser.add_argument("--bge-pre-retrieval-2000", default=DEFAULT_BGE_PRE_RETRIEVAL_2000)
    parser.add_argument("--bge-post-retrieval-2000", default=DEFAULT_BGE_POST_RETRIEVAL_2000)
    parser.add_argument("--bge-pre-retrieval-4000", default=DEFAULT_BGE_PRE_RETRIEVAL_4000)
    parser.add_argument("--bge-post-retrieval-4000", default=DEFAULT_BGE_POST_RETRIEVAL_4000)
    parser.add_argument("--bge-pre-listwise", default=DEFAULT_BGE_PRE_LISTWISE)
    parser.add_argument("--bge-post-listwise", default=DEFAULT_BGE_POST_LISTWISE)
    parser.add_argument("--bge-train-metrics", default=DEFAULT_BGE_TRAIN_METRICS)
    parser.add_argument("--laststep-pre-retrieval", default=DEFAULT_LASTSTEP_PRE_RETRIEVAL)
    parser.add_argument("--laststep-post-retrieval", default=DEFAULT_LASTSTEP_POST_RETRIEVAL)
    parser.add_argument("--laststep-train-metrics", default=DEFAULT_LASTSTEP_TRAIN_METRICS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = build_summary(
            architecture_map_report=Path(args.architecture_map_report),
            manifest_checkpoint_foundation_report=Path(args.manifest_checkpoint_foundation_report),
            generic_bootstrap_report=Path(args.generic_bootstrap_report),
            compact_train_guard_report=Path(args.compact_train_guard_report),
            serving_parity_report=Path(args.serving_parity_report),
            default_embedding_source=Path(args.default_embedding_source),
            runtime_embedding_model_source=Path(args.runtime_embedding_model_source),
            backend_tensor_ops_source=Path(args.backend_tensor_ops_source),
            default_embedding_test=Path(args.default_embedding_test),
            runtime_embedding_model_test=Path(args.runtime_embedding_model_test),
            runtime_embedding_trainer_test=Path(args.runtime_embedding_trainer_test),
            backend_compact_attention_test=Path(args.backend_compact_attention_test),
            cmd_eos_main_test=Path(args.cmd_eos_main_test),
            bge_listwise_validation_report=Path(args.bge_listwise_validation_report),
            heads2_lr_bracket_report=Path(args.heads2_lr_bracket_report),
            heads2_lr_bracket_gate_log=Path(args.heads2_lr_bracket_gate_log),
            heads2_train_metrics=Path(args.heads2_train_metrics),
            heads2_train_stdout_log=Path(args.heads2_train_stdout_log),
            heads2_train_time_log=Path(args.heads2_train_time_log),
            laststep_movement_report=Path(args.laststep_movement_report),
            bge_pre_retrieval_2000=Path(args.bge_pre_retrieval_2000),
            bge_post_retrieval_2000=Path(args.bge_post_retrieval_2000),
            bge_pre_retrieval_4000=Path(args.bge_pre_retrieval_4000),
            bge_post_retrieval_4000=Path(args.bge_post_retrieval_4000),
            bge_pre_listwise=Path(args.bge_pre_listwise),
            bge_post_listwise=Path(args.bge_post_listwise),
            bge_train_metrics=Path(args.bge_train_metrics),
            laststep_pre_retrieval=Path(args.laststep_pre_retrieval),
            laststep_post_retrieval=Path(args.laststep_post_retrieval),
            laststep_train_metrics=Path(args.laststep_train_metrics),
        )
        write_json(Path(args.output_json), summary)
        write_tsv(Path(args.output_tsv), summary)
    except ReadinessError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
