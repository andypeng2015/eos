#!/usr/bin/env python3
"""Summarize Stage A/B/C retrieval pretraining-distillation readiness.

This utility is read-only with respect to run inputs. It checks existing
manifests and metrics from bounded Stage A/B/C evidence slices, then writes
JSON/TSV summary artifacts only when invoked with output paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "eos.stageabc_pretraining_readiness_summary.v1"
DEFAULT_OUTPUT_JSON = ".tiller/scratch/codex/stageabc-pretraining-readiness-current.json"
DEFAULT_OUTPUT_TSV = ".tiller/scratch/codex/stageabc-pretraining-readiness-current.tsv"
DEFAULT_PIPELINE_MAP_REPORT = ".tiller/scratch/codex/retrieval-pretraining-distillation-pipeline-map-v1-report.md"
DEFAULT_STAGEA_ROW_MANIFEST = (
    "runs/retrieval-stagea-msmarco-row-builder-v1-dry-run-20260629T185405Z/manifest.json"
)
DEFAULT_STAGEA_LEAK_REPORT = (
    "runs/retrieval-stagea-msmarco-row-builder-v1-dry-run-20260629T185405Z/reports/leak-report.json"
)
DEFAULT_IMPORTED_BGE_MANIFEST = (
    "runs/retrieval-stagea-msmarco-bge-teacher-score-bridge-v1-scale256-20260629T192026Z/manifest.json"
)
DEFAULT_IMPORTED_BGE_VALIDATION = (
    "runs/retrieval-stagea-msmarco-bge-teacher-score-bridge-v1-scale256-20260629T192026Z/"
    "artifacts/validation-summary.json"
)
DEFAULT_IMPORTED_BGE_GUIDE_FILTER_MANIFEST = (
    "runs/retrieval-stagea-msmarco-bge-teacher-score-bridge-v1-scale256-20260629T192026Z/"
    "guide-filter-manifest.json"
)
DEFAULT_QWEN_MXBAI_MANIFEST = "runs/retrieval-qwen-mxbai-5k-guide-filter-v1-20260627T200210Z/manifest.json"
DEFAULT_QWEN_MXBAI_INDEPENDENT_SUMMARY = (
    "runs/retrieval-qwen-mxbai-5k-guide-filter-v1-20260627T200210Z/"
    "reports/guide-filter-independent-summary.json"
)
DEFAULT_LISTWISE_QWEN3_METRICS = (
    "runs/retrieval-clean5k-listwise-eval-only-v1-20260627T203639Z/diagnostics/qwen3.metrics.json"
)
DEFAULT_LISTWISE_MXBAI_METRICS = (
    "runs/retrieval-clean5k-listwise-eval-only-v1-20260627T203639Z/diagnostics/mxbai.metrics.json"
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


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def nested_dict(data: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def legal_gates_are_research_only(gates: dict[str, Any]) -> bool:
    return (
        gates.get("train_allowed_for_research") is True
        and gates.get("release_train_allowed") is False
        and gates.get("commercial_use_allowed") is False
        and gates.get("test_rows_train_allowed") is False
    )


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


def summarize_pipeline_map(report_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    text: str | None = None
    read_error: str | None = None
    try:
        text = report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        read_error = f"missing_evidence: {report_path}"
        blockers.append(read_error)
    except UnicodeDecodeError as exc:
        read_error = f"invalid_markdown: {report_path}: {exc}"
        blockers.append(read_error)

    checks = {
        "exists": text is not None,
        "mentions_stage_a": bool(text and "Stage A" in text),
        "mentions_stage_b": bool(text and "Stage B" in text),
        "mentions_stage_c": bool(text and "Stage C" in text),
    }
    for key, passed in checks.items():
        if not passed and key != "exists":
            blockers.append(f"pipeline_map check failed: {key}")
    return component(
        component_id="pipeline_map",
        status="pass" if not blockers else "missing_evidence" if read_error else "planned_not_ready",
        blockers=blockers,
        details={
            "report_path": str(report_path),
            "checks": checks,
        },
    )


def summarize_stage_a_row_builder(manifest_path: Path, leak_report_path: Path) -> dict[str, Any]:
    manifest, manifest_error = load_optional_json(manifest_path)
    leak_report, leak_error = load_optional_json(leak_report_path)
    blockers = [item for item in [manifest_error, leak_error] if item]
    counts = nested_dict(manifest, "counts")
    legal_gates = nested_dict(manifest, "legal_gates")
    split_policy = nested_dict(manifest, "split_policy")
    leak_validation = nested_dict(leak_report, "validation")

    rows_emitted = as_int(counts.get("rows_emitted"))
    checks = {
        "rows_emitted_gt_0": rows_emitted is not None and rows_emitted > 0,
        "research_only_legal_gates": legal_gates_are_research_only(legal_gates),
        "no_test_rows_for_training": split_policy.get("test_or_eval_rows_used") is False
        and split_policy.get("test_rows_train_allowed") is False,
        "leak_report_passed": leak_report is not None and leak_report.get("status") == "passed",
        "validation_passed": leak_validation.get("status") == "passed",
    }
    for key, passed in checks.items():
        if not passed:
            blockers.append(f"stage_a_row_builder check failed: {key}")
    return component(
        component_id="stage_a_row_builder",
        status="pass" if not blockers else "missing_evidence" if manifest_error or leak_error else "planned_not_ready",
        blockers=blockers,
        details={
            "manifest_path": str(manifest_path),
            "leak_report_path": str(leak_report_path),
            "rows_emitted": rows_emitted,
            "legal_gates": legal_gates,
            "split_policy": split_policy,
            "checks": checks,
            "reader_compatibility": nested_dict(manifest, "reader_compatibility"),
        },
    )


def guide_filter_missing_drop_is_zero(guide_manifest: dict[str, Any] | None) -> bool:
    if not isinstance(guide_manifest, dict):
        return False
    counts = nested_dict(guide_manifest, "counts")
    coverage = nested_dict(guide_manifest, "coverage")
    validation = nested_dict(guide_manifest, "validation")
    explicit = counts.get("missing_score_drop")
    if explicit is not None:
        return explicit == 0
    return (
        counts.get("rows_seen") == counts.get("rows_emitted")
        and validation.get("no_row_emitted_without_required_scores") is True
        and not coverage.get("drop_samples")
    )


def summarize_imported_bge_teacher_bridge(
    manifest_path: Path,
    validation_path: Path,
    guide_filter_manifest_path: Path,
) -> dict[str, Any]:
    manifest, manifest_error = load_optional_json(manifest_path)
    validation, validation_error = load_optional_json(validation_path)
    guide_manifest, guide_error = load_optional_json(guide_filter_manifest_path)
    blockers = [item for item in [manifest_error, validation_error, guide_error] if item]

    coverage = nested_dict(manifest, "coverage")
    guide_inputs = nested_dict(guide_manifest, "inputs")
    teacher_caches = nested_dict(guide_inputs, "teacher_caches")
    imported_teacher = nested_dict(teacher_caches, "imported_bge_small_en_v1_5")
    imported_config = nested_dict(imported_teacher, "config")
    validation_checks = nested_dict(validation, "validation")
    scores = nested_dict(manifest, "scores")

    examples_seen = as_int(coverage.get("examples_seen"))
    examples_scored = as_int(coverage.get("examples_scored"))
    examples_written = as_int(coverage.get("examples_written"))
    missing_examples = as_int(coverage.get("missing_examples"))
    rows_emitted = as_int(validation.get("rows_emitted") if isinstance(validation, dict) else None)
    positive_top1_rate = as_number(scores.get("positive_top1_rate"))
    package_identity = imported_config.get("package_identity")
    package_sha256 = imported_config.get("package_sha256")
    checks = {
        "rows_ge_128": rows_emitted is not None and rows_emitted >= 128,
        "examples_scored_seen_written_match": examples_seen == examples_scored == examples_written
        and examples_seen is not None,
        "missing_examples_0": missing_examples == 0,
        "guide_filter_missing_drop_0": guide_filter_missing_drop_is_zero(guide_manifest),
        "positive_top1_rate_ge_1": positive_top1_rate is not None and positive_top1_rate >= 1.0,
        "package_identity_present": bool(package_identity and package_sha256),
    }
    if validation_checks:
        checks["validation_scoring_complete"] = validation_checks.get("scoring_complete") is True
    for key, passed in checks.items():
        if not passed:
            blockers.append(f"imported_bge_teacher_bridge check failed: {key}")
    return component(
        component_id="imported_bge_teacher_bridge",
        status="pass" if not blockers else "missing_evidence" if manifest_error or validation_error or guide_error else "planned_not_ready",
        blockers=blockers,
        details={
            "manifest_path": str(manifest_path),
            "validation_path": str(validation_path),
            "guide_filter_manifest_path": str(guide_filter_manifest_path),
            "examples_seen": examples_seen,
            "examples_scored": examples_scored,
            "examples_written": examples_written,
            "missing_examples": missing_examples,
            "rows_emitted": rows_emitted,
            "positive_top1_rate": positive_top1_rate,
            "package_identity": package_identity,
            "package_sha256": package_sha256,
            "teacher_model_id": manifest.get("teacher_model_id") if isinstance(manifest, dict) else None,
            "checks": checks,
        },
    )


def qwen_mxbai_missing_score_drops_zero(manifest: dict[str, Any] | None, independent: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    counts = nested_dict(manifest, "counts")
    explicit = counts.get("missing_score_drop")
    if explicit is not None:
        return explicit == 0
    validation = nested_dict(independent, "validation")
    coverage = nested_dict(manifest, "coverage")
    teachers = nested_dict(coverage, "teachers")
    return (
        validation.get("no_row_emitted_without_required_scores") is True
        and all(
            isinstance(value, dict) and value.get("complete_rows") == counts.get("rows_seen")
            for value in teachers.values()
        )
    )


def summarize_qwen_mxbai_guide_filter(manifest_path: Path, independent_summary_path: Path) -> dict[str, Any]:
    manifest, manifest_error = load_optional_json(manifest_path)
    independent, independent_error = load_optional_json(independent_summary_path)
    blockers = [item for item in [manifest_error, independent_error] if item]
    counts = nested_dict(manifest, "counts")
    emitted_flags = nested_dict(independent, "emitted_dev_positive_flags")
    policy_counts_output = nested_dict(independent, "policy_counts_output")
    checks = {
        "rows_seen_5000": counts.get("rows_seen") == 5000,
        "rows_emitted_2805": counts.get("rows_emitted") == 2805,
        "missing_score_drops_0": qwen_mxbai_missing_score_drops_zero(manifest, independent),
        "clean_ambiguous_conflict_counts_recorded": all(
            key in counts for key in ("clean_agreement", "ambiguous_soft_only", "conflict", "conflict_drop")
        ),
    }
    for key, passed in checks.items():
        if not passed:
            blockers.append(f"qwen_mxbai_guide_filter check failed: {key}")
    warnings = [
        (
            "dev-positive negative flags remain in emitted rows "
            f"(rows={emitted_flags.get('rows')}, refs={emitted_flags.get('refs')}); downstream handling is required"
        ),
        (
            "ambiguous soft-only rows require explicit downstream handling "
            f"(ambiguous_soft_only={counts.get('ambiguous_soft_only')})"
        ),
    ]
    status = "evidence_ready_with_flags" if not blockers else "missing_evidence" if manifest_error or independent_error else "planned_not_ready"
    return component(
        component_id="qwen_mxbai_guide_filter",
        status=status,
        blockers=blockers,
        warnings=warnings,
        details={
            "manifest_path": str(manifest_path),
            "independent_summary_path": str(independent_summary_path),
            "counts": counts,
            "policy_counts_output": policy_counts_output,
            "emitted_dev_positive_flags": emitted_flags,
            "teacher_coverage": nested_dict(nested_dict(manifest, "coverage"), "teachers"),
            "checks": checks,
        },
    )


def metrics_eval_only_checks(metrics: dict[str, Any] | None) -> dict[str, bool]:
    workload = nested_dict(metrics, "workload")
    profile = nested_dict(metrics, "profile_delta")
    summary = nested_dict(metrics, "summary")
    final_eval = nested_dict(metrics, "final_listwise_geometry_eval")
    config = nested_dict(metrics, "config")
    return {
        "eval_only_true": config.get("eval_only") is True,
        "actual_train_examples_0": workload.get("actual_train_examples") == 0,
        "optimizer_updates_0": profile.get("optimizer_updates") == 0,
        "steps_run_0": summary.get("steps_run") == 0,
        "query_count_2379": final_eval.get("query_count") == 2379,
        "batch_count_75": final_eval.get("batch_count") == 75,
    }


def compact_listwise_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    workload = nested_dict(metrics, "workload")
    profile = nested_dict(metrics, "profile_delta")
    summary = nested_dict(metrics, "summary")
    final_eval = nested_dict(metrics, "final_listwise_geometry_eval")
    return {
        "actual_train_examples": workload.get("actual_train_examples"),
        "optimizer_updates": profile.get("optimizer_updates"),
        "steps_run": summary.get("steps_run"),
        "query_count": final_eval.get("query_count"),
        "batch_count": final_eval.get("batch_count"),
        "teacher_cross_entropy": final_eval.get("teacher_cross_entropy"),
        "teacher_kl": final_eval.get("teacher_kl"),
        "teacher_top1_agreement": final_eval.get("teacher_top1_agreement"),
        "any_positive_top1": final_eval.get("any_positive_top1"),
    }


def summarize_listwise_eval_only(qwen3_metrics_path: Path, mxbai_metrics_path: Path) -> dict[str, Any]:
    qwen3, qwen_error = load_optional_json(qwen3_metrics_path)
    mxbai, mxbai_error = load_optional_json(mxbai_metrics_path)
    blockers = [item for item in [qwen_error, mxbai_error] if item]
    checks = {
        "qwen3": metrics_eval_only_checks(qwen3),
        "mxbai": metrics_eval_only_checks(mxbai),
    }
    for teacher, teacher_checks in checks.items():
        for key, passed in teacher_checks.items():
            if not passed:
                blockers.append(f"listwise_eval_only {teacher} check failed: {key}")
    return component(
        component_id="listwise_eval_only",
        status="eval_only_evidence" if not blockers else "missing_evidence" if qwen_error or mxbai_error else "planned_not_ready",
        blockers=blockers,
        warnings=["diagnostic only; these metrics are not optimizer updates, training proof, or retrieval quality proof"],
        details={
            "qwen3_metrics_path": str(qwen3_metrics_path),
            "mxbai_metrics_path": str(mxbai_metrics_path),
            "qwen3": compact_listwise_metrics(qwen3),
            "mxbai": compact_listwise_metrics(mxbai),
            "checks": checks,
        },
    )


def summarize_stage_c_compact_adaptation() -> dict[str, Any]:
    return component(
        component_id="stage_c_compact_adaptation",
        status="deferred_until_dense_acceptance",
        blockers=["Stage C compact adaptation is deferred until dense retrieval acceptance exists"],
        details={
            "training_ready": False,
            "policy": "dense accepted artifact first; compact adaptation cannot rescue dense retrieval misses",
        },
    )


def build_summary(
    *,
    pipeline_map_report: Path = Path(DEFAULT_PIPELINE_MAP_REPORT),
    stagea_row_manifest: Path = Path(DEFAULT_STAGEA_ROW_MANIFEST),
    stagea_leak_report: Path = Path(DEFAULT_STAGEA_LEAK_REPORT),
    imported_bge_manifest: Path = Path(DEFAULT_IMPORTED_BGE_MANIFEST),
    imported_bge_validation: Path = Path(DEFAULT_IMPORTED_BGE_VALIDATION),
    imported_bge_guide_filter_manifest: Path = Path(DEFAULT_IMPORTED_BGE_GUIDE_FILTER_MANIFEST),
    qwen_mxbai_manifest: Path = Path(DEFAULT_QWEN_MXBAI_MANIFEST),
    qwen_mxbai_independent_summary: Path = Path(DEFAULT_QWEN_MXBAI_INDEPENDENT_SUMMARY),
    listwise_qwen3_metrics: Path = Path(DEFAULT_LISTWISE_QWEN3_METRICS),
    listwise_mxbai_metrics: Path = Path(DEFAULT_LISTWISE_MXBAI_METRICS),
    clock: Any = utc_now,
) -> dict[str, Any]:
    components = {
        "pipeline_map": summarize_pipeline_map(pipeline_map_report),
        "stage_a_row_builder": summarize_stage_a_row_builder(stagea_row_manifest, stagea_leak_report),
        "imported_bge_teacher_bridge": summarize_imported_bge_teacher_bridge(
            imported_bge_manifest,
            imported_bge_validation,
            imported_bge_guide_filter_manifest,
        ),
        "qwen_mxbai_guide_filter": summarize_qwen_mxbai_guide_filter(
            qwen_mxbai_manifest,
            qwen_mxbai_independent_summary,
        ),
        "listwise_eval_only": summarize_listwise_eval_only(listwise_qwen3_metrics, listwise_mxbai_metrics),
        "stage_c_compact_adaptation": summarize_stage_c_compact_adaptation(),
    }
    hard_blockers = [
        blocker
        for item in components.values()
        for blocker in item.get("blockers", [])
        if item["id"] != "stage_c_compact_adaptation"
        and str(blocker).startswith(
            ("missing_evidence", "invalid_json", "invalid_markdown", "pipeline_map", "stage_", "imported_", "qwen_", "listwise_")
        )
    ]
    flags = [
        warning
        for item in components.values()
        for warning in item.get("warnings", [])
    ]
    if hard_blockers:
        overall_status = "partial_evidence_waiting_implementation"
    else:
        overall_status = "evidence_ready_not_training_ready"
    blockers = [
        "Stage A/B/C evidence is read-only readiness evidence, not a training run",
        "No large-scale Stage A/B training, retrieval-gated pilot, or dense acceptance exists",
        "Stage C compact adaptation remains deferred until dense acceptance",
    ]
    if hard_blockers:
        blockers.extend(hard_blockers)
    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "status": overall_status,
        "training_ready": False,
        "release_train_allowed": False,
        "quality_claim": False,
        "blockers": blockers,
        "warnings": flags,
        "next_safe_action": (
            "Use this as conservative readiness evidence; require explicit downstream handling for dev-positive "
            "flags and soft-only rows before any bounded research-only training descriptor."
        ),
        "evidence_paths": {
            "pipeline_map_report": str(pipeline_map_report),
            "pipeline_map_report_exists": pipeline_map_report.exists(),
            "stagea_row_manifest": str(stagea_row_manifest),
            "stagea_leak_report": str(stagea_leak_report),
            "imported_bge_manifest": str(imported_bge_manifest),
            "imported_bge_validation": str(imported_bge_validation),
            "imported_bge_guide_filter_manifest": str(imported_bge_guide_filter_manifest),
            "qwen_mxbai_manifest": str(qwen_mxbai_manifest),
            "qwen_mxbai_independent_summary": str(qwen_mxbai_independent_summary),
            "listwise_qwen3_metrics": str(listwise_qwen3_metrics),
            "listwise_mxbai_metrics": str(listwise_mxbai_metrics),
        },
        "components": components,
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
            "key": "stageabc_pretraining_distillation",
            "status": summary["status"],
            "value": summary["next_safe_action"],
            "blockers": " | ".join(str(item) for item in summary.get("blockers", [])),
            "details": {
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
                "blockers": " | ".join(str(blocker) for blocker in item.get("blockers", [])),
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
    parser.add_argument("--pipeline-map-report", default=DEFAULT_PIPELINE_MAP_REPORT)
    parser.add_argument("--stagea-row-manifest", default=DEFAULT_STAGEA_ROW_MANIFEST)
    parser.add_argument("--stagea-leak-report", default=DEFAULT_STAGEA_LEAK_REPORT)
    parser.add_argument("--imported-bge-manifest", default=DEFAULT_IMPORTED_BGE_MANIFEST)
    parser.add_argument("--imported-bge-validation", default=DEFAULT_IMPORTED_BGE_VALIDATION)
    parser.add_argument("--imported-bge-guide-filter-manifest", default=DEFAULT_IMPORTED_BGE_GUIDE_FILTER_MANIFEST)
    parser.add_argument("--qwen-mxbai-manifest", default=DEFAULT_QWEN_MXBAI_MANIFEST)
    parser.add_argument("--qwen-mxbai-independent-summary", default=DEFAULT_QWEN_MXBAI_INDEPENDENT_SUMMARY)
    parser.add_argument("--listwise-qwen3-metrics", default=DEFAULT_LISTWISE_QWEN3_METRICS)
    parser.add_argument("--listwise-mxbai-metrics", default=DEFAULT_LISTWISE_MXBAI_METRICS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = build_summary(
            pipeline_map_report=Path(args.pipeline_map_report),
            stagea_row_manifest=Path(args.stagea_row_manifest),
            stagea_leak_report=Path(args.stagea_leak_report),
            imported_bge_manifest=Path(args.imported_bge_manifest),
            imported_bge_validation=Path(args.imported_bge_validation),
            imported_bge_guide_filter_manifest=Path(args.imported_bge_guide_filter_manifest),
            qwen_mxbai_manifest=Path(args.qwen_mxbai_manifest),
            qwen_mxbai_independent_summary=Path(args.qwen_mxbai_independent_summary),
            listwise_qwen3_metrics=Path(args.listwise_qwen3_metrics),
            listwise_mxbai_metrics=Path(args.listwise_mxbai_metrics),
        )
        write_json(Path(args.output_json), summary)
        write_tsv(Path(args.output_tsv), summary)
    except ReadinessError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
