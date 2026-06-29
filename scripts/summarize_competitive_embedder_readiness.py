#!/usr/bin/env python3
"""Summarize competitive embedder goal readiness across current packets.

This utility is read-only with respect to run inputs. It composes the selected
BGE package gate, encoder-v2.1, dynamic-remine, and Eos Embedder 1 readiness
summaries into one JSON/TSV rollup for orchestration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as bge_gate
import summarize_dynamic_remine_readiness as dynamic_remine
import summarize_encoder_v21_readiness as encoder_v21
import summarize_eos_embedder1_release_readiness as embedder1
import summarize_stageabc_pretraining_readiness as stageabc


SUMMARY_SCHEMA = "eos.competitive_embedder_readiness_rollup.v1"
DEFAULT_OUTPUT_JSON = ".tiller/scratch/codex/competitive-embedder-goal-readiness-current.json"
DEFAULT_OUTPUT_TSV = ".tiller/scratch/codex/competitive-embedder-goal-readiness-current.tsv"
DEFAULT_COMPACT_NATIVE_STUDENT_REPORT = (
    ".tiller/scratch/codex/compact-bge-listwise-larger-validation-split211-seed191-v1-report.md"
)
DEFAULT_STAGEABC_PRETRAINING_DISTILLATION_REPORT = stageabc.DEFAULT_PIPELINE_MAP_REPORT


class RollupError(ValueError):
    """Raised when rollup inputs or outputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_datasets(value: str) -> list[str]:
    try:
        return bge_gate.parse_datasets(value)
    except bge_gate.SummaryError as exc:
        raise RollupError(str(exc)) from exc


def active_export_markers(bge_summary: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    for dataset in bge_summary.get("datasets", []):
        if not isinstance(dataset, dict) or dataset.get("status") == "complete":
            continue
        present = dataset.get("present_artifacts")
        if dataset.get("partial_doc_vector_lines") is not None or present:
            pieces = [
                f"{dataset.get('dataset')}: partial_doc_vector_lines={dataset.get('partial_doc_vector_lines')}",
                f"vector_progress={dataset.get('vector_progress_completed')}/{dataset.get('vector_progress_total')}",
            ]
            if dataset.get("vector_progress_percent") is not None:
                pieces.append(f"vector_progress_percent={dataset.get('vector_progress_percent')}")
            pieces.extend(
                [
                    f"expected_documents={dataset.get('expected_documents')}",
                    f"expected_queries={dataset.get('expected_queries')}",
                    f"doc_vector_lines={dataset.get('doc_vector_lines')}",
                    f"query_vector_lines={dataset.get('query_vector_lines')}",
                ]
            )
            doc_file = dataset.get("doc_vector_file") if isinstance(dataset.get("doc_vector_file"), dict) else {}
            if doc_file.get("mtime_utc"):
                pieces.append(f"doc_vector_mtime_utc={doc_file.get('mtime_utc')}")
            markers.append("; ".join(pieces))
    return markers


def compact_bge_progress(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in summary.get("datasets", []):
        if not isinstance(dataset, dict):
            continue
        doc_file = dataset.get("doc_vector_file") if isinstance(dataset.get("doc_vector_file"), dict) else {}
        query_file = dataset.get("query_vector_file") if isinstance(dataset.get("query_vector_file"), dict) else {}
        rows.append(
            {
                "dataset": dataset.get("dataset"),
                "status": dataset.get("status"),
                "expected_documents": dataset.get("expected_documents"),
                "expected_queries": dataset.get("expected_queries"),
                "doc_vector_lines": dataset.get("doc_vector_lines"),
                "query_vector_lines": dataset.get("query_vector_lines"),
                "partial_doc_vector_lines": dataset.get("partial_doc_vector_lines"),
                "vector_progress_completed": dataset.get("vector_progress_completed"),
                "vector_progress_total": dataset.get("vector_progress_total"),
                "vector_progress_percent": dataset.get("vector_progress_percent"),
                "doc_vector_size_bytes": doc_file.get("size_bytes"),
                "doc_vector_mtime_utc": doc_file.get("mtime_utc"),
                "query_vector_size_bytes": query_file.get("size_bytes"),
                "query_vector_mtime_utc": query_file.get("mtime_utc"),
                "present_artifacts": dataset.get("present_artifacts", []),
                "missing_artifacts": dataset.get("missing_artifacts", []),
            }
        )
    return rows


def compact_bge_gate(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    policy = aggregate.get("quality_policy") if isinstance(aggregate.get("quality_policy"), dict) else {}
    return {
        "run_root": summary.get("run_root"),
        "all_complete": aggregate.get("all_complete"),
        "complete_dataset_count": aggregate.get("complete_dataset_count"),
        "expected_dataset_count": aggregate.get("expected_dataset_count"),
        "identity_consistent": aggregate.get("identity_consistent"),
        "promotion_recommendation": aggregate.get("promotion_recommendation"),
        "quality_policy": {
            "non_default_promotion_policy_pass": policy.get("non_default_promotion_policy_pass"),
            "dense_policy_pass": policy.get("dense_policy_pass"),
            "q8_policy_pass": policy.get("q8_policy_pass"),
            "q4_release_profile_pass": policy.get("q4_release_profile_pass"),
            "q4_release_profile_decision": policy.get("q4_release_profile_decision"),
            "blockers": policy.get("blockers", []),
            "macro": policy.get("macro", {}),
            "per_dataset": policy.get("per_dataset", {}),
        },
        "blockers": aggregate.get("blockers", []),
        "progress": compact_bge_progress(summary),
        "active_export_markers": active_export_markers(summary),
    }


def status_for_bge_gate(bge: dict[str, Any]) -> str:
    if bge["quality_policy"].get("non_default_promotion_policy_pass") is True:
        return "ready_for_review"
    if not bge.get("all_complete") and bge.get("active_export_markers"):
        return "waiting_on_fiqa"
    return "blocked"


def launch_status(summary: dict[str, Any]) -> str:
    if summary.get("launch_allowed") is True:
        return "ready_to_launch"
    bge = summary.get("bge_gate") if isinstance(summary.get("bge_gate"), dict) else {}
    if not bge.get("all_complete") and bge.get("active_export_markers"):
        return "waiting_on_fiqa"
    return "blocked"


def packet(
    *,
    packet_id: str,
    title: str,
    status: str,
    blockers: list[str],
    next_safe_action: str,
    progress: Any = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": packet_id,
        "title": title,
        "status": status,
        "blockers": blockers,
        "next_safe_action": next_safe_action,
        "progress": progress,
        "details": details or {},
    }


def report_path_detail(path: Path) -> dict[str, Any]:
    return {
        "evidence_path": str(path),
        "exists": path.exists(),
    }


def read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def parse_compact_native_student_deltas(text: str) -> dict[str, str]:
    deltas: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 2000-doc gate:"):
            parts = stripped.split("`")
            if len(parts) >= 6:
                deltas["2000_doc_ndcg_delta"] = parts[1]
                deltas["2000_doc_map_delta"] = parts[3]
                deltas["2000_doc_recall_delta"] = parts[5]
        if stripped.startswith("- 4000-doc gate:"):
            parts = stripped.split("`")
            if len(parts) >= 6:
                deltas["4000_doc_ndcg_delta"] = parts[1]
                deltas["4000_doc_map_delta"] = parts[3]
                deltas["4000_doc_recall_delta"] = parts[5]
    return {key: value for key, value in deltas.items() if value}


def compact_native_student_packet(report_path: Path, waiting_on_fiqa: bool) -> dict[str, Any]:
    text = read_optional_text(report_path)
    if text is None:
        return packet(
            packet_id="compact_native_student",
            title="Compact native student/listwise distillation",
            status="missing_evidence",
            blockers=[f"compact native student report missing: {report_path}"],
            next_safe_action="Locate or create compact native/listwise validation evidence before planning promotion.",
            details=report_path_detail(report_path),
        )
    status = "waiting_on_fiqa" if waiting_on_fiqa else "evidence_only_waiting_validation"
    return packet(
        packet_id="compact_native_student",
        title="Compact native student/listwise distillation",
        status=status,
        blockers=["existing compact native/listwise evidence is validation-only, not promotion proof"],
        next_safe_action="Third deterministic larger split or broader validation after FiQA clears."
        if waiting_on_fiqa
        else "Run a third deterministic larger split or broader validation before any promotion discussion.",
        details={
            **report_path_detail(report_path),
            "parsed_deltas": parse_compact_native_student_deltas(text),
            "promotion_ready": False,
            "quality_claim": False,
        },
    )


def stageabc_pretraining_distillation_packet(stageabc_summary: dict[str, Any]) -> dict[str, Any]:
    return packet(
        packet_id="stageabc_pretraining_distillation",
        title="Stage A/B/C pretraining-distillation",
        status=stageabc_summary.get("status", "planned_not_ready"),
        blockers=[str(item) for item in stageabc_summary.get("blockers", [])],
        next_safe_action=str(stageabc_summary.get("next_safe_action") or "Resolve Stage A/B/C evidence blockers."),
        details={
            "training_ready": stageabc_summary.get("training_ready"),
            "release_train_allowed": stageabc_summary.get("release_train_allowed"),
            "quality_claim": stageabc_summary.get("quality_claim"),
            "evidence_paths": stageabc_summary.get("evidence_paths", {}),
            "components": stageabc_summary.get("components", {}),
            "warnings": stageabc_summary.get("warnings", []),
        },
    )


def role_asymmetry_packet(release: dict[str, Any]) -> dict[str, Any]:
    evidence = release.get("non_default_evidence") if isinstance(release.get("non_default_evidence"), dict) else {}
    gates = evidence.get("gates") if isinstance(evidence.get("gates"), dict) else {}
    role_gate = gates.get("role_aware_provider_smoke") if isinstance(gates.get("role_aware_provider_smoke"), dict) else {}
    role_status = role_gate.get("status")
    if role_status == "pass":
        status = "release_identity_gate_ready"
        blockers: list[str] = []
        action = "Keep query/document role handling in the Eos Embedder 1 release identity checklist."
    else:
        status = "defer"
        blockers = [str(item) for item in role_gate.get("blockers", [])] if isinstance(role_gate.get("blockers"), list) else []
        if not blockers:
            blockers = ["role-aware provider smoke evidence is missing or not valid"]
        action = "Supply valid role-aware provider smoke evidence before treating role asymmetry as release-ready."
    return packet(
        packet_id="role_asymmetry",
        title="Role asymmetry",
        status=status,
        blockers=blockers,
        next_safe_action=action,
        details={
            "release_non_default_candidate_status": release.get("non_default_candidate_status"),
            "role_aware_provider_smoke_status": role_status,
            "evidence_path": role_gate.get("evidence_path"),
            "public_name": release.get("public_name"),
            "public_id": release.get("public_id"),
            "role_contract": release.get("identity", {}).get("role_contract")
            if isinstance(release.get("identity"), dict)
            else None,
        },
    )


def quantization_profile_packet(bge: dict[str, Any], waiting_on_fiqa: bool) -> dict[str, Any]:
    policy = bge.get("quality_policy") if isinstance(bge.get("quality_policy"), dict) else {}
    if waiting_on_fiqa:
        status = "waiting_on_fiqa"
        blockers = ["FiQA dense/q8/q4 metrics are incomplete"]
        action = bge_wait_action()
    elif policy.get("q8_policy_pass") is True:
        status = "q8_ready_for_review"
        blockers = []
        action = "Review q8 preservation; keep q4 as a separate release-profile decision."
    else:
        status = "blocked"
        blockers = [str(item) for item in policy.get("blockers", [])]
        action = "Resolve quantization quality-policy blockers before release review."
    return packet(
        packet_id="quantization_profile",
        title="Quantization profile readiness",
        status=status,
        blockers=blockers,
        next_safe_action=action,
        details={
            "q8_policy_pass": policy.get("q8_policy_pass"),
            "q4_release_profile_pass": policy.get("q4_release_profile_pass"),
            "q4_release_profile_decision": policy.get("q4_release_profile_decision"),
            "dense_policy_pass": policy.get("dense_policy_pass"),
            "non_default_promotion_policy_pass": policy.get("non_default_promotion_policy_pass"),
            "macro": policy.get("macro", {}),
            "per_dataset": policy.get("per_dataset", {}),
        },
    )


def bge_wait_action() -> str:
    return (
        "Wait for the active/incomplete FiQA selected-BGE export to finish. "
        "After FiQA manifest, dense, q8, and q4 metrics exist, rerun selected BGE with "
        "`--require-non-default-promotion-policy`."
    )


def first_non_heavy_action(packets: dict[str, dict[str, Any]]) -> str | None:
    for item in packets.values():
        action = item.get("next_safe_action")
        if item.get("status") in {"ready_for_review", "blocked", "defer"} and isinstance(action, str):
            if "launch" not in action.lower() and "training" not in action.lower() and "export" not in action.lower():
                return action
    return None


def build_summary(
    *,
    bge_gate_root: Path,
    encoder_run_root: Path,
    encoder_descriptor: Path,
    dynamic_stagea_root: Path,
    dynamic_descriptor: Path,
    datasets: list[str],
    compact_native_student_report: Path = Path(DEFAULT_COMPACT_NATIVE_STUDENT_REPORT),
    stageabc_pretraining_distillation_report: Path = Path(DEFAULT_STAGEABC_PRETRAINING_DISTILLATION_REPORT),
    stageabc_stagea_row_manifest: Path = Path(stageabc.DEFAULT_STAGEA_ROW_MANIFEST),
    stageabc_stagea_leak_report: Path = Path(stageabc.DEFAULT_STAGEA_LEAK_REPORT),
    stageabc_imported_bge_manifest: Path = Path(stageabc.DEFAULT_IMPORTED_BGE_MANIFEST),
    stageabc_imported_bge_validation: Path = Path(stageabc.DEFAULT_IMPORTED_BGE_VALIDATION),
    stageabc_imported_bge_guide_filter_manifest: Path = Path(stageabc.DEFAULT_IMPORTED_BGE_GUIDE_FILTER_MANIFEST),
    stageabc_qwen_mxbai_manifest: Path = Path(stageabc.DEFAULT_QWEN_MXBAI_MANIFEST),
    stageabc_qwen_mxbai_independent_summary: Path = Path(stageabc.DEFAULT_QWEN_MXBAI_INDEPENDENT_SUMMARY),
    stageabc_listwise_qwen3_metrics: Path = Path(stageabc.DEFAULT_LISTWISE_QWEN3_METRICS),
    stageabc_listwise_mxbai_metrics: Path = Path(stageabc.DEFAULT_LISTWISE_MXBAI_METRICS),
    encoder_binary: Path | None = None,
    encoder_tiny_metrics: Path | None = None,
    encoder_capped_metrics: Path | None = None,
    embedder1_default_gate_evidence_paths: dict[str, str | None] | None = None,
    embedder1_candidate_smoke_evidence: str | Path | None = None,
    embedder1_role_aware_provider_smoke_evidence: str | Path | None = None,
    embedder1_corkscrewdb_serving_smoke_evidence: str | Path | None = None,
    embedder1_scan_paths: list[Path] | None = None,
    active_export_pid: int | None = None,
    active_export_command: str | None = None,
    clock: Any = utc_now,
) -> dict[str, Any]:
    bge_summary = bge_gate.build_summary(run_root=bge_gate_root, datasets=datasets, clock=clock)
    bge = compact_bge_gate(bge_summary)
    encoder = encoder_v21.build_summary(
        run_root=encoder_run_root,
        bge_gate_root=bge_gate_root,
        descriptor=encoder_descriptor,
        datasets=datasets,
        binary_path=encoder_binary,
        tiny_metrics_path=encoder_tiny_metrics,
        capped_metrics_path=encoder_capped_metrics,
        clock=clock,
    )
    dynamic = dynamic_remine.build_summary(
        stagea_root=dynamic_stagea_root,
        bge_gate_root=bge_gate_root,
        descriptor=dynamic_descriptor,
        datasets=datasets,
        clock=clock,
    )
    release = embedder1.build_summary(
        bge_gate_root=bge_gate_root,
        datasets=datasets,
        default_gate_evidence_paths=embedder1_default_gate_evidence_paths or {},
        candidate_smoke_evidence=embedder1_candidate_smoke_evidence,
        role_aware_provider_smoke_evidence=embedder1_role_aware_provider_smoke_evidence,
        corkscrewdb_serving_smoke_evidence=embedder1_corkscrewdb_serving_smoke_evidence,
        scan_paths=embedder1_scan_paths or [],
        clock=clock,
    )
    stageabc_summary = stageabc.build_summary(
        pipeline_map_report=stageabc_pretraining_distillation_report,
        stagea_row_manifest=stageabc_stagea_row_manifest,
        stagea_leak_report=stageabc_stagea_leak_report,
        imported_bge_manifest=stageabc_imported_bge_manifest,
        imported_bge_validation=stageabc_imported_bge_validation,
        imported_bge_guide_filter_manifest=stageabc_imported_bge_guide_filter_manifest,
        qwen_mxbai_manifest=stageabc_qwen_mxbai_manifest,
        qwen_mxbai_independent_summary=stageabc_qwen_mxbai_independent_summary,
        listwise_qwen3_metrics=stageabc_listwise_qwen3_metrics,
        listwise_mxbai_metrics=stageabc_listwise_mxbai_metrics,
        clock=clock,
    )
    waiting_on_fiqa = not bge["all_complete"] and bool(bge["active_export_markers"])
    next_action = bge_wait_action() if waiting_on_fiqa else None

    packets = {
        "selected_bge_full_gate": packet(
            packet_id="selected_bge_full_gate",
            title="Selected BGE full gate",
            status=status_for_bge_gate(bge),
            blockers=bge["blockers"],
            next_safe_action=bge_wait_action()
            if status_for_bge_gate(bge) == "waiting_on_fiqa"
            else "Review selected-BGE non-default promotion policy results.",
            progress=bge["progress"],
            details={
                "quality_policy": bge["quality_policy"],
                "active_export_markers": bge["active_export_markers"],
            },
        ),
        "encoder_v21_controlled_training": packet(
            packet_id="encoder_v21_controlled_training",
            title="Encoder-v2.1 controlled training",
            status=launch_status(encoder),
            blockers=encoder.get("blockers", []),
            next_safe_action=bge_wait_action()
            if launch_status(encoder) == "waiting_on_fiqa"
            else "Launch the controlled encoder-v2.1 descriptor only if resource policy allows.",
            progress=encoder.get("bge_gate", {}).get("incomplete_dataset_progress", []),
            details={"launch_allowed": encoder.get("launch_allowed")},
        ),
        "dynamic_remine": packet(
            packet_id="dynamic_remine",
            title="Dynamic remine",
            status=launch_status(dynamic),
            blockers=dynamic.get("blockers", []),
            next_safe_action=bge_wait_action()
            if launch_status(dynamic) == "waiting_on_fiqa"
            else "Launch the dynamic-remine smoke only after selected-BGE gate clears.",
            progress=dynamic.get("bge_gate", {}).get("incomplete_dataset_progress", []),
            details={"launch_allowed": dynamic.get("launch_allowed")},
        ),
        "eos_embedder1_non_default": packet(
            packet_id="eos_embedder1_non_default",
            title="Eos Embedder 1 non-default candidate",
            status=release["non_default_candidate_status"],
            blockers=release["blockers"]["non_default"],
            next_safe_action=bge_wait_action()
            if waiting_on_fiqa
            else "Review Eos Embedder 1 non-default evidence and release identity.",
            progress=release.get("bge_gate", {}).get("incomplete_dataset_progress", []),
            details={"public_name": release["public_name"], "public_id": release["public_id"]},
        ),
        "eos_embedder1_default_swap": packet(
            packet_id="eos_embedder1_default_swap",
            title="Eos Embedder 1 default swap",
            status=release["default_swap_status"],
            blockers=release["blockers"]["default_swap"],
            next_safe_action="Keep default swap deferred until non-default review and default-swap evidence pass.",
            progress=release.get("bge_gate", {}).get("incomplete_dataset_progress", []),
            details={"public_name": release["public_name"], "public_id": release["public_id"]},
        ),
        "compact_native_student": compact_native_student_packet(compact_native_student_report, waiting_on_fiqa),
        "stageabc_pretraining_distillation": stageabc_pretraining_distillation_packet(stageabc_summary),
        "role_asymmetry": role_asymmetry_packet(release),
        "quantization_profile": quantization_profile_packet(bge, waiting_on_fiqa),
    }

    if next_action is None:
        next_action = first_non_heavy_action(packets) or "No non-heavy next action is currently unblocked."

    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "quality_claim": False,
        "training_run": False,
        "public_identity_policy": {
            "public_name": embedder1.DEFAULT_PUBLIC_NAME,
            "public_id": embedder1.DEFAULT_PUBLIC_ID,
            "internal_v_labels_are_release_versions": False,
            "note": "Internal v-labels are experiment labels, not public release versions.",
        },
        "active_export": {
            "dataset": "fiqa",
            "present": waiting_on_fiqa,
            "status": "partial_artifacts_present" if waiting_on_fiqa else "not_detected",
            "pid": active_export_pid if waiting_on_fiqa else None,
            "command": active_export_command if waiting_on_fiqa and active_export_command else None,
            "markers": bge["active_export_markers"],
        },
        "summary": {
            "selected_bge_status": packets["selected_bge_full_gate"]["status"],
            "encoder_v21_status": packets["encoder_v21_controlled_training"]["status"],
            "dynamic_remine_status": packets["dynamic_remine"]["status"],
            "eos_embedder1_non_default_status": packets["eos_embedder1_non_default"]["status"],
            "eos_embedder1_default_swap_status": packets["eos_embedder1_default_swap"]["status"],
            "compact_native_student_status": packets["compact_native_student"]["status"],
            "stageabc_pretraining_distillation_status": packets["stageabc_pretraining_distillation"]["status"],
            "role_asymmetry_status": packets["role_asymmetry"]["status"],
            "quantization_profile_status": packets["quantization_profile"]["status"],
            "waiting_on_fiqa": waiting_on_fiqa,
        },
        "next_action": next_action,
        "arbiter_next_action": next_action,
        "packets": packets,
        "underlying": {
            "selected_bge": bge_summary,
            "encoder_v21": encoder,
            "dynamic_remine": dynamic,
            "eos_embedder1": release,
            "stageabc_pretraining_distillation": stageabc_summary,
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
    rows: list[dict[str, Any]] = [
        {
            "section": "summary",
            "key": "waiting_on_fiqa",
            "status": "waiting_on_fiqa" if summary["summary"]["waiting_on_fiqa"] else "open",
            "value": summary["summary"]["waiting_on_fiqa"],
            "blockers": "",
            "progress": "",
        },
        {
            "section": "summary",
            "key": "next_action",
            "status": "",
            "value": summary["next_action"],
            "blockers": "",
            "progress": "",
        },
    ]
    for packet_id, item in summary["packets"].items():
        rows.append(
            {
                "section": "packet",
                "key": packet_id,
                "status": item["status"],
                "value": item["next_safe_action"],
                "blockers": " | ".join(str(blocker) for blocker in item.get("blockers", [])[:8]),
                "progress": format_tsv_value(item.get("progress")),
            }
        )
    for progress in summary["packets"]["selected_bge_full_gate"].get("progress") or []:
        rows.append(
            {
                "section": "progress",
                "key": progress.get("dataset"),
                "status": progress.get("status"),
                "value": "",
                "blockers": " | ".join(str(item) for item in progress.get("missing_artifacts", [])),
                "progress": (
                    f"{progress.get('vector_progress_completed')}/{progress.get('vector_progress_total')} "
                    f"({progress.get('vector_progress_percent')}%)"
                ),
            }
        )
    return rows


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "key", "status", "value", "blockers", "progress"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in tsv_rows(summary):
            writer.writerow({key: format_tsv_value(row.get(key)) for key in fieldnames})


def default_gate_evidence_map(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "default_provider_bridge": args.default_provider_bridge_evidence,
        "default_release_smoke": args.default_release_smoke_evidence,
        "legacy_256d_migration_policy_smoke": args.legacy_256d_migration_evidence,
        "startup_load_encode_throughput_gate": args.throughput_gate_evidence,
        "default_asset_size_policy": args.default_asset_size_policy_evidence,
    }


def parse_scan_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-tsv", default=DEFAULT_OUTPUT_TSV)
    parser.add_argument("--datasets", default=bge_gate.DEFAULT_DATASETS)
    parser.add_argument("--bge-gate-root", default=bge_gate.DEFAULT_RUN_ROOT)
    parser.add_argument("--encoder-run-root", default=encoder_v21.DEFAULT_RUN_ROOT)
    parser.add_argument("--encoder-descriptor", default=encoder_v21.DEFAULT_DESCRIPTOR)
    parser.add_argument("--encoder-binary")
    parser.add_argument("--encoder-tiny-metrics")
    parser.add_argument("--encoder-capped-metrics")
    parser.add_argument("--dynamic-stagea-root", default=dynamic_remine.DEFAULT_STAGEA_ROOT)
    parser.add_argument("--dynamic-descriptor", default=dynamic_remine.DEFAULT_DESCRIPTOR)
    parser.add_argument("--compact-native-student-report", default=DEFAULT_COMPACT_NATIVE_STUDENT_REPORT)
    parser.add_argument(
        "--stageabc-pretraining-distillation-report",
        default=DEFAULT_STAGEABC_PRETRAINING_DISTILLATION_REPORT,
    )
    parser.add_argument("--stageabc-stagea-row-manifest", default=stageabc.DEFAULT_STAGEA_ROW_MANIFEST)
    parser.add_argument("--stageabc-stagea-leak-report", default=stageabc.DEFAULT_STAGEA_LEAK_REPORT)
    parser.add_argument("--stageabc-imported-bge-manifest", default=stageabc.DEFAULT_IMPORTED_BGE_MANIFEST)
    parser.add_argument("--stageabc-imported-bge-validation", default=stageabc.DEFAULT_IMPORTED_BGE_VALIDATION)
    parser.add_argument(
        "--stageabc-imported-bge-guide-filter-manifest",
        default=stageabc.DEFAULT_IMPORTED_BGE_GUIDE_FILTER_MANIFEST,
    )
    parser.add_argument("--stageabc-qwen-mxbai-manifest", default=stageabc.DEFAULT_QWEN_MXBAI_MANIFEST)
    parser.add_argument(
        "--stageabc-qwen-mxbai-independent-summary",
        default=stageabc.DEFAULT_QWEN_MXBAI_INDEPENDENT_SUMMARY,
    )
    parser.add_argument("--stageabc-listwise-qwen3-metrics", default=stageabc.DEFAULT_LISTWISE_QWEN3_METRICS)
    parser.add_argument("--stageabc-listwise-mxbai-metrics", default=stageabc.DEFAULT_LISTWISE_MXBAI_METRICS)
    parser.add_argument("--scan-paths", default="")
    parser.add_argument("--candidate-smoke-evidence")
    parser.add_argument("--role-aware-provider-smoke-evidence")
    parser.add_argument("--corkscrewdb-serving-smoke-evidence")
    parser.add_argument("--default-provider-bridge-evidence")
    parser.add_argument("--default-release-smoke-evidence")
    parser.add_argument("--legacy-256d-migration-evidence")
    parser.add_argument("--throughput-gate-evidence")
    parser.add_argument("--default-asset-size-policy-evidence")
    parser.add_argument("--active-export-pid", type=int)
    parser.add_argument("--active-export-command")
    parser.add_argument("--require-unblocked-next-action", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_summary(
            bge_gate_root=Path(args.bge_gate_root),
            encoder_run_root=Path(args.encoder_run_root),
            encoder_descriptor=Path(args.encoder_descriptor),
            dynamic_stagea_root=Path(args.dynamic_stagea_root),
            dynamic_descriptor=Path(args.dynamic_descriptor),
            datasets=parse_datasets(args.datasets),
            compact_native_student_report=Path(args.compact_native_student_report),
            stageabc_pretraining_distillation_report=Path(args.stageabc_pretraining_distillation_report),
            stageabc_stagea_row_manifest=Path(args.stageabc_stagea_row_manifest),
            stageabc_stagea_leak_report=Path(args.stageabc_stagea_leak_report),
            stageabc_imported_bge_manifest=Path(args.stageabc_imported_bge_manifest),
            stageabc_imported_bge_validation=Path(args.stageabc_imported_bge_validation),
            stageabc_imported_bge_guide_filter_manifest=Path(args.stageabc_imported_bge_guide_filter_manifest),
            stageabc_qwen_mxbai_manifest=Path(args.stageabc_qwen_mxbai_manifest),
            stageabc_qwen_mxbai_independent_summary=Path(args.stageabc_qwen_mxbai_independent_summary),
            stageabc_listwise_qwen3_metrics=Path(args.stageabc_listwise_qwen3_metrics),
            stageabc_listwise_mxbai_metrics=Path(args.stageabc_listwise_mxbai_metrics),
            encoder_binary=Path(args.encoder_binary) if args.encoder_binary else None,
            encoder_tiny_metrics=Path(args.encoder_tiny_metrics) if args.encoder_tiny_metrics else None,
            encoder_capped_metrics=Path(args.encoder_capped_metrics) if args.encoder_capped_metrics else None,
            embedder1_default_gate_evidence_paths=default_gate_evidence_map(args),
            embedder1_candidate_smoke_evidence=args.candidate_smoke_evidence,
            embedder1_role_aware_provider_smoke_evidence=args.role_aware_provider_smoke_evidence,
            embedder1_corkscrewdb_serving_smoke_evidence=args.corkscrewdb_serving_smoke_evidence,
            embedder1_scan_paths=parse_scan_paths(args.scan_paths),
            active_export_pid=args.active_export_pid,
            active_export_command=args.active_export_command,
        )
        write_json(Path(args.output_json), summary)
        write_tsv(Path(args.output_tsv), summary)
    except (
        RollupError,
        bge_gate.SummaryError,
        encoder_v21.ReadinessError,
        dynamic_remine.ReadinessError,
        embedder1.ReadinessError,
        stageabc.ReadinessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.require_unblocked_next_action and summary["summary"]["waiting_on_fiqa"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
