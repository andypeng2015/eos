#!/usr/bin/env python3
"""Summarize Eos Embedder 1 release readiness.

This utility is read-only with respect to run inputs. It reuses the selected
BGE package gate summary, adds release identity metadata, and separates
first-class non-default readiness from default-swap readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as bge_gate


SUMMARY_SCHEMA = "eos.embedder1_release_readiness.v1"
DEFAULT_PUBLIC_NAME = "Eos Embedder 1"
DEFAULT_PUBLIC_ID = "eos-embedder-1"
DEFAULT_LEGACY_MODEL_NAME = "eos-embed-v1"
DEFAULT_BGE_GATE_ROOT = bge_gate.DEFAULT_RUN_ROOT
DEFAULT_DATASETS = bge_gate.DEFAULT_DATASETS
DEFAULT_CANDIDATE_PROVIDER_ID = "corkscrewdb-imported-bge-eos-embed-v1-candidate"
DEFAULT_ROLE_CONTRACT_SCHEMA = "manta.pretrained_bert_retrieval_role_contract.v1"
DEFAULT_CANDIDATE_SMOKE_SCHEMA = "manta.imported_bge_eos_embed_v1_candidate_smoke.v1"
DEFAULT_ROLE_AWARE_PROVIDER_SMOKE_SCHEMA = "eos.imported_bge_role_aware_provider_smoke.v1"
DEFAULT_CORKSCREWDB_SERVING_SMOKE_SCHEMA = "eos.imported_bge_serving_candidate_manifest.v1"
DEFAULT_PROVIDER_BRIDGE_SCHEMA = "eos.embedder1_default_provider_bridge_evidence.v1"
DEFAULT_RELEASE_SMOKE_SCHEMA = "eos.embedder1_default_release_smoke.v1"
LEGACY_256D_MIGRATION_POLICY_SMOKE_SCHEMA = "eos.embedder1_legacy_256d_migration_policy_smoke.v1"
STARTUP_LOAD_ENCODE_THROUGHPUT_GATE_SCHEMA = "eos.embedder1_startup_load_encode_throughput_gate.v1"
DEFAULT_ASSET_SIZE_POLICY_SCHEMA = "eos.embedder1_default_asset_size_policy.v1"
NORM_TOLERANCE = 1e-3
OFFLINE_DELTA_TOLERANCE = 1e-9
Q4_P95_MS_CEILING = 25.0
Q8_P95_MS_CEILING = 50.0
DEFAULT_COLD_LOAD_MS_CEILING = 5000.0
DEFAULT_WARM_BATCH64_DOCS_PER_SECOND_FLOOR = 10.0
DEFAULT_EXTERNAL_PACKAGE_BYTES_CEILING = 200_000_000
DEFAULT_IN_REPO_ASSET_BYTES_CEILING = 25_000_000
DEFAULT_ASSET_SIZE_POLICIES = {
    "explicit_external_package",
    "lazy_download_package",
    "approved_large_in_repo_asset",
}
DEFAULT_SWAP_GATES = [
    ("default_provider_bridge", "default provider bridge missing"),
    ("default_release_smoke", "default release smoke missing"),
    ("legacy_256d_migration_policy_smoke", "legacy 256d migration policy/smoke missing"),
    ("startup_load_encode_throughput_gate", "startup/load/encode throughput gate missing"),
    ("default_asset_size_policy", "default asset size policy missing"),
]
IGNORED_SCAN_DIRS = {
    ".git",
    "__pycache__",
    "generated",
    "scratch",
    "runs",
    "datasets",
    "dataset",
}
IGNORED_SCAN_EXTENSIONS = {
    ".bin",
    ".mll",
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".gz",
    ".zst",
    ".jsonl",
}
PUBLIC_V6_PATTERN = re.compile(r"\bv6\b", re.IGNORECASE)


class ReadinessError(ValueError):
    """Raised when readiness inputs or outputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_datasets(value: str) -> list[str]:
    try:
        return bge_gate.parse_datasets(value)
    except bge_gate.SummaryError as exc:
        raise ReadinessError(str(exc)) from exc


def parse_scan_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def backend_fingerprint(package_sha256: str, identity_sha256: str) -> str:
    return f"eos-imported-bge:{package_sha256}:{identity_sha256}"


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReadinessError(f"required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReadinessError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReadinessError(f"{path}: expected top-level JSON object")
    return data


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def release_identity(
    *,
    public_name: str,
    public_id: str,
    legacy_model_name: str,
    package_path: str,
    package_sha256: str,
    identity_sha256: str,
    model: str,
    snapshot: str,
) -> dict[str, Any]:
    return {
        "public_name": public_name,
        "public_id": public_id,
        "legacy_model_name": legacy_model_name,
        "package_path": package_path,
        "package_sha256": package_sha256,
        "identity_sha256": identity_sha256,
        "source_model": model,
        "snapshot": snapshot,
        "dim": bge_gate.DEFAULT_DIM,
        "pooling": bge_gate.DEFAULT_POOLING,
        "normalization": bge_gate.DEFAULT_NORMALIZATION,
        "max_length": bge_gate.DEFAULT_MAX_LENGTH,
        "role_contract": {
            "schema": DEFAULT_ROLE_CONTRACT_SCHEMA,
            "query_role": "query",
            "document_role": "document",
            "query_prefix": bge_gate.DEFAULT_QUERY_PREFIX,
            "document_prefix": bge_gate.DEFAULT_DOCUMENT_PREFIX,
        },
        "candidate_provider_id": DEFAULT_CANDIDATE_PROVIDER_ID,
        "backend_fingerprint_shape": backend_fingerprint(package_sha256, identity_sha256),
    }


def compact_bge_progress(dataset: dict[str, Any]) -> dict[str, Any]:
    doc_file = dataset.get("doc_vector_file") if isinstance(dataset.get("doc_vector_file"), dict) else {}
    query_file = dataset.get("query_vector_file") if isinstance(dataset.get("query_vector_file"), dict) else {}
    return {
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


def compact_bge_summary(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    datasets: list[dict[str, Any]] = []
    incomplete_progress: list[dict[str, Any]] = []
    for dataset in summary.get("datasets", []):
        progress = compact_bge_progress(dataset)
        compact = {
            "dataset": dataset.get("dataset"),
            "status": dataset.get("status"),
            "identity_match": dataset.get("identity_match"),
            "missing_artifacts": dataset.get("missing_artifacts", []),
            "present_artifacts": dataset.get("present_artifacts", []),
            "partial_doc_vector_lines": dataset.get("partial_doc_vector_lines"),
            "expected_documents": progress.get("expected_documents"),
            "expected_queries": progress.get("expected_queries"),
            "doc_vector_lines": progress.get("doc_vector_lines"),
            "query_vector_lines": progress.get("query_vector_lines"),
            "vector_progress_completed": progress.get("vector_progress_completed"),
            "vector_progress_total": progress.get("vector_progress_total"),
            "vector_progress_percent": progress.get("vector_progress_percent"),
            "doc_vector_size_bytes": progress.get("doc_vector_size_bytes"),
            "doc_vector_mtime_utc": progress.get("doc_vector_mtime_utc"),
            "query_vector_size_bytes": progress.get("query_vector_size_bytes"),
            "query_vector_mtime_utc": progress.get("query_vector_mtime_utc"),
            "blockers": dataset.get("blockers", []),
            "dense": dataset.get("dense"),
            "q8": dataset.get("q8"),
            "q4": dataset.get("q4"),
        }
        datasets.append(compact)
        if dataset.get("status") != "complete":
            incomplete_progress.append(progress)
    return {
        "run_root": summary["run_root"],
        "summary_schema": summary.get("schema"),
        "all_complete": aggregate.get("all_complete"),
        "complete_dataset_count": aggregate.get("complete_dataset_count"),
        "expected_dataset_count": aggregate.get("expected_dataset_count"),
        "identity_consistent": aggregate.get("identity_consistent"),
        "identity_checked_manifest_count": aggregate.get("identity_checked_manifest_count"),
        "identity_mismatched_datasets": aggregate.get("identity_mismatched_datasets", []),
        "quality_claim": aggregate.get("quality_claim"),
        "default_alias_changed": aggregate.get("default_alias_changed"),
        "promotion_recommendation": aggregate.get("promotion_recommendation"),
        "macro": aggregate.get("macro", {}),
        "blockers": aggregate.get("blockers", []),
        "incomplete_dataset_progress": incomplete_progress,
        "datasets": datasets,
    }


def is_binaryish(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return True
    return b"\x00" in sample


def should_skip_scan_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORED_SCAN_DIRS:
        return True
    return path.suffix.lower() in IGNORED_SCAN_EXTENSIONS


def iter_scan_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if should_skip_scan_path(path):
            continue
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            continue
        for child in path.rglob("*"):
            if child.is_file() and not should_skip_scan_path(child):
                files.append(child)
    return sorted(files)


def scan_public_name_hygiene(paths: list[Path]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    missing: list[str] = []
    scan_files = iter_scan_files(paths)
    for path in paths:
        if not path.exists():
            missing.append(str(path))

    for path in scan_files:
        if is_binaryish(path):
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_number, line in enumerate(handle, start=1):
                    lowered = line.lower()
                    if not PUBLIC_V6_PATTERN.search(line):
                        continue
                    if "internal" in lowered or "run label" in lowered or "experiment" in lowered:
                        continue
                    matches.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
        except OSError:
            continue
    blockers = [
        f"public-name hygiene: {match['path']}:{match['line']} contains public v6"
        for match in matches
    ]
    warnings = [f"scan path missing: {path}" for path in missing]
    return {
        "paths": [str(path) for path in paths],
        "files_scanned": len(scan_files),
        "matches": matches,
        "blockers": blockers,
        "warnings": warnings,
    }


def default_gate_evidence_map(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "default_provider_bridge": args.default_provider_bridge_evidence,
        "default_release_smoke": args.default_release_smoke_evidence,
        "legacy_256d_migration_policy_smoke": args.legacy_256d_migration_evidence,
        "startup_load_encode_throughput_gate": args.throughput_gate_evidence,
        "default_asset_size_policy": args.default_asset_size_policy_evidence,
    }


def summarize_default_gates(
    evidence_paths: dict[str, str | None],
    *,
    package_sha256: str,
    identity_sha256: str,
    legacy_model_name: str,
    public_name: str,
    model: str,
) -> dict[str, Any]:
    validators = {
        "default_provider_bridge": validate_default_provider_bridge,
        "default_release_smoke": validate_default_release_smoke,
        "legacy_256d_migration_policy_smoke": validate_legacy_256d_migration_policy_smoke,
        "startup_load_encode_throughput_gate": validate_startup_load_encode_throughput_gate,
        "default_asset_size_policy": validate_default_asset_size_policy,
    }
    gates: dict[str, Any] = {}
    for gate, missing_message in DEFAULT_SWAP_GATES:
        gates[gate] = summarize_evidence_gate(
            gate.replace("_", " "),
            evidence_paths.get(gate),
            validators[gate],
            missing_message=missing_message,
            package_sha256=package_sha256,
            identity_sha256=identity_sha256,
            legacy_model_name=legacy_model_name,
            public_name=public_name,
            model=model,
        )
    blockers = [blocker for gate in gates.values() for blocker in gate.get("blockers", [])]
    return {
        "gates": gates,
        "all_present": all(gate.get("status") != "missing" for gate in gates.values()),
        "all_valid": not blockers,
        "blockers": blockers,
    }


def add_check(blockers: list[str], condition: bool, message: str) -> None:
    if not condition:
        blockers.append(message)


def close_to(value: Any, expected: float, tolerance: float) -> bool:
    number = as_number(value)
    return number is not None and abs(number - expected) <= tolerance


def validate_candidate_smoke(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    legacy_model_name: str,
    public_name: str,
    model: str,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    add_check(blockers, data.get("schema") == DEFAULT_CANDIDATE_SMOKE_SCHEMA, "candidate smoke schema mismatch")
    add_check(blockers, data.get("candidate_model_name") == legacy_model_name, "candidate smoke model name mismatch")
    add_check(blockers, data.get("candidate_display_name") == public_name, "candidate smoke display name mismatch")
    add_check(
        blockers,
        data.get("candidate_status") == "non_default_reference_candidate",
        "candidate smoke status is not non_default_reference_candidate",
    )
    add_check(blockers, data.get("source_model") == model, "candidate smoke source model mismatch")
    add_check(blockers, data.get("quality_claim") is False, "candidate smoke quality_claim must be false")
    add_check(
        blockers,
        data.get("default_alias_changed") is False,
        "candidate smoke default_alias_changed must be false",
    )
    add_check(blockers, nested(data, "package", "sha256") == package_sha256, "candidate smoke package sha mismatch")
    add_check(
        blockers,
        nested(data, "package", "identity_sha256") == identity_sha256,
        "candidate smoke package identity mismatch",
    )
    add_check(
        blockers,
        nested(data, "role_contract", "query_prefix") == bge_gate.DEFAULT_QUERY_PREFIX,
        "candidate smoke query prefix mismatch",
    )
    add_check(
        blockers,
        nested(data, "role_contract", "document_prefix") == bge_gate.DEFAULT_DOCUMENT_PREFIX,
        "candidate smoke document prefix mismatch",
    )
    add_check(
        blockers,
        nested(data, "role_contract", "pooling") == bge_gate.DEFAULT_POOLING,
        "candidate smoke pooling mismatch",
    )
    add_check(
        blockers,
        nested(data, "role_contract", "max_length") == bge_gate.DEFAULT_MAX_LENGTH,
        "candidate smoke max length mismatch",
    )
    add_check(
        blockers,
        close_to(nested(data, "direct_embed_smoke", "query_norm"), 1.0, NORM_TOLERANCE),
        "candidate smoke query norm is not close to 1.0",
    )
    add_check(
        blockers,
        close_to(nested(data, "direct_embed_smoke", "document_norm"), 1.0, NORM_TOLERANCE),
        "candidate smoke document norm is not close to 1.0",
    )
    details = {
        "schema": data.get("schema"),
        "candidate_model_name": data.get("candidate_model_name"),
        "candidate_display_name": data.get("candidate_display_name"),
        "package_sha256": nested(data, "package", "sha256"),
        "identity_sha256": nested(data, "package", "identity_sha256"),
        "query_norm": nested(data, "direct_embed_smoke", "query_norm"),
        "document_norm": nested(data, "direct_embed_smoke", "document_norm"),
        "caveats": data.get("caveats", []),
    }
    return blockers, details


def validate_role_aware_provider_smoke(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected_fingerprint = backend_fingerprint(package_sha256, identity_sha256)
    add_check(blockers, data.get("schema") == DEFAULT_ROLE_AWARE_PROVIDER_SMOKE_SCHEMA, "role-aware provider smoke schema mismatch")
    add_check(blockers, data.get("provider_id") == DEFAULT_CANDIDATE_PROVIDER_ID, "role-aware provider id mismatch")
    add_check(blockers, data.get("dim") == bge_gate.DEFAULT_DIM, "role-aware provider dim mismatch")
    add_check(
        blockers,
        data.get("backend_fingerprint") == expected_fingerprint,
        "role-aware provider backend fingerprint mismatch",
    )
    add_check(
        blockers,
        data.get("candidate_package_sha256") == package_sha256,
        "role-aware provider package sha mismatch",
    )
    add_check(blockers, data.get("candidate_identity") == identity_sha256, "role-aware provider identity mismatch")
    add_check(blockers, data.get("quality_claim") is False, "role-aware provider quality_claim must be false")
    add_check(
        blockers,
        data.get("default_alias_changed") is False,
        "role-aware provider default_alias_changed must be false",
    )
    add_check(blockers, data.get("all_top1_ok") is True, "role-aware provider top1 smoke failed")
    for field in ("query_role_calls", "document_role_calls", "encode_calls", "encode_batch_calls"):
        add_check(blockers, isinstance(data.get(field), int) and data.get(field) > 0, f"role-aware provider {field} must be > 0")
    add_check(
        blockers,
        nested(data, "db_manifest_embedding", "id") == DEFAULT_CANDIDATE_PROVIDER_ID,
        "role-aware provider db manifest embedding id mismatch",
    )
    add_check(
        blockers,
        nested(data, "db_manifest_embedding", "dim") == bge_gate.DEFAULT_DIM,
        "role-aware provider db manifest embedding dim mismatch",
    )
    add_check(
        blockers,
        nested(data, "db_manifest_embedding", "backend_fingerprint") == expected_fingerprint,
        "role-aware provider db manifest backend fingerprint mismatch",
    )
    details = {
        "schema": data.get("schema"),
        "provider_id": data.get("provider_id"),
        "dim": data.get("dim"),
        "backend_fingerprint": data.get("backend_fingerprint"),
        "all_top1_ok": data.get("all_top1_ok"),
        "query_role_calls": data.get("query_role_calls"),
        "document_role_calls": data.get("document_role_calls"),
        "encode_calls": data.get("encode_calls"),
        "encode_batch_calls": data.get("encode_batch_calls"),
    }
    return blockers, details


def validate_serving_smoke(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    add_check(blockers, data.get("schema") == DEFAULT_CORKSCREWDB_SERVING_SMOKE_SCHEMA, "CorkScrewDB serving smoke schema mismatch")
    add_check(blockers, nested(data, "candidate", "quality_claim") is False, "CorkScrewDB serving quality_claim must be false")
    add_check(
        blockers,
        nested(data, "candidate", "default_alias_changed") is False,
        "CorkScrewDB serving default_alias_changed must be false",
    )
    add_check(blockers, nested(data, "package", "sha256") == package_sha256, "CorkScrewDB serving package sha mismatch")
    add_check(
        blockers,
        nested(data, "package", "identity_sha256") == identity_sha256,
        "CorkScrewDB serving package identity mismatch",
    )
    add_check(
        blockers,
        nested(data, "corkscrewdb_smoke", "quantized_only") is True,
        "CorkScrewDB serving smoke quantized_only must be true",
    )
    add_check(
        blockers,
        nested(data, "corkscrewdb_smoke", "index_type") == "flat",
        "CorkScrewDB serving smoke index_type must be flat",
    )
    add_check(
        blockers,
        nested(data, "corkscrewdb_smoke", "layout") == "single_parent_vectors",
        "CorkScrewDB serving smoke layout must be single_parent_vectors",
    )
    comparisons = data.get("offline_comparison") if isinstance(data.get("offline_comparison"), dict) else {}
    details: dict[str, Any] = {
        "schema": data.get("schema"),
        "candidate": data.get("candidate", {}),
        "package_sha256": nested(data, "package", "sha256"),
        "identity_sha256": nested(data, "package", "identity_sha256"),
        "quantized_only": nested(data, "corkscrewdb_smoke", "quantized_only"),
        "index_type": nested(data, "corkscrewdb_smoke", "index_type"),
        "layout": nested(data, "corkscrewdb_smoke", "layout"),
        "offline_comparison": {},
        "caveats": data.get("caveats", []),
    }
    for key, ceiling in (("q4", Q4_P95_MS_CEILING), ("q8", Q8_P95_MS_CEILING)):
        comparison = comparisons.get(key) if isinstance(comparisons, dict) else None
        if not isinstance(comparison, dict):
            blockers.append(f"CorkScrewDB serving missing {key} offline comparison")
            details["offline_comparison"][key] = None
            continue
        ndcg_delta = as_number(nested(comparison, "delta", "ndcg_at_10"))
        recall_delta = as_number(nested(comparison, "delta", "recall_at_100"))
        p95_ms = as_number(nested(comparison, "corkscrew", "p95_ms"))
        add_check(
            blockers,
            ndcg_delta is not None and abs(ndcg_delta) <= OFFLINE_DELTA_TOLERANCE,
            f"CorkScrewDB serving {key} nDCG delta exceeds tolerance",
        )
        add_check(
            blockers,
            recall_delta is not None and abs(recall_delta) <= OFFLINE_DELTA_TOLERANCE,
            f"CorkScrewDB serving {key} recall delta exceeds tolerance",
        )
        add_check(
            blockers,
            p95_ms is not None and 0.0 < p95_ms <= ceiling,
            f"CorkScrewDB serving {key} p95 exceeds {ceiling:g}ms ceiling",
        )
        details["offline_comparison"][key] = {
            "ndcg_at_10_delta": ndcg_delta,
            "recall_at_100_delta": recall_delta,
            "p95_ms": p95_ms,
            "p95_ms_ceiling": ceiling,
        }
    return blockers, details


def identity_field(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data.get(key)
    if key == "identity_sha256":
        return nested(data, "package", "identity_sha256")
    if key == "package_sha256":
        return nested(data, "package", "sha256")
    return None


def validate_identity_fields(
    blockers: list[str],
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    label: str,
) -> None:
    add_check(blockers, identity_field(data, "package_sha256") == package_sha256, f"{label} package sha mismatch")
    add_check(blockers, identity_field(data, "identity_sha256") == identity_sha256, f"{label} identity mismatch")


def role_contract_value(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data.get(key)
    return nested(data, "role_contract", key)


def validate_default_provider_bridge(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected_fingerprint = backend_fingerprint(package_sha256, identity_sha256)
    add_check(blockers, data.get("schema") == DEFAULT_PROVIDER_BRIDGE_SCHEMA, "default provider bridge schema mismatch")
    validate_identity_fields(
        blockers,
        data,
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        label="default provider bridge",
    )
    provider_id = data.get("provider_id") or data.get("default_provider_id")
    add_check(blockers, isinstance(provider_id, str) and bool(provider_id), "default provider bridge provider id missing")
    add_check(blockers, data.get("dim") == bge_gate.DEFAULT_DIM, "default provider bridge dim mismatch")
    add_check(
        blockers,
        data.get("backend_fingerprint") == expected_fingerprint,
        "default provider bridge backend fingerprint mismatch",
    )
    add_check(
        blockers,
        role_contract_value(data, "query_prefix") == bge_gate.DEFAULT_QUERY_PREFIX,
        "default provider bridge query prefix mismatch",
    )
    add_check(
        blockers,
        role_contract_value(data, "document_prefix") == bge_gate.DEFAULT_DOCUMENT_PREFIX,
        "default provider bridge document prefix mismatch",
    )
    add_check(blockers, role_contract_value(data, "pooling") == bge_gate.DEFAULT_POOLING, "default provider bridge pooling mismatch")
    add_check(
        blockers,
        role_contract_value(data, "normalization") == bge_gate.DEFAULT_NORMALIZATION,
        "default provider bridge normalization mismatch",
    )
    add_check(
        blockers,
        role_contract_value(data, "max_length") == bge_gate.DEFAULT_MAX_LENGTH,
        "default provider bridge max length mismatch",
    )
    add_check(
        blockers,
        data.get("default_alias_changed") is False or data.get("dry_run") is True,
        "default provider bridge changed default alias outside dry run",
    )
    add_check(
        blockers,
        data.get("legacy_default_preserved") is True,
        "default provider bridge did not preserve legacy default",
    )
    details = {
        "schema": data.get("schema"),
        "provider_id": provider_id,
        "dim": data.get("dim"),
        "backend_fingerprint": data.get("backend_fingerprint"),
        "default_alias_changed": data.get("default_alias_changed"),
        "dry_run": data.get("dry_run"),
        "legacy_default_preserved": data.get("legacy_default_preserved"),
    }
    return blockers, details


def validate_default_release_smoke(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    add_check(blockers, data.get("schema") == DEFAULT_RELEASE_SMOKE_SCHEMA, "default release smoke schema mismatch")
    validate_identity_fields(
        blockers,
        data,
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        label="default release smoke",
    )
    add_check(
        blockers,
        isinstance(data.get("default_provider_id"), str) and bool(data.get("default_provider_id")),
        "default release smoke default provider id missing",
    )
    add_check(blockers, data.get("dim") == bge_gate.DEFAULT_DIM, "default release smoke dim mismatch")
    for field in (
        "query_role_smoke_passed",
        "document_role_smoke_passed",
        "new_384d_db_smoke_passed",
        "mismatch_smoke_passed",
    ):
        add_check(blockers, data.get(field) is True, f"default release smoke {field} must be true")
    add_check(blockers, data.get("quality_claim") is False, "default release smoke quality_claim must be false")
    details = {
        "schema": data.get("schema"),
        "default_provider_id": data.get("default_provider_id"),
        "dim": data.get("dim"),
        "query_role_smoke_passed": data.get("query_role_smoke_passed"),
        "document_role_smoke_passed": data.get("document_role_smoke_passed"),
        "new_384d_db_smoke_passed": data.get("new_384d_db_smoke_passed"),
        "mismatch_smoke_passed": data.get("mismatch_smoke_passed"),
        "quality_claim": data.get("quality_claim"),
    }
    return blockers, details


def validate_legacy_256d_migration_policy_smoke(
    data: dict[str, Any],
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    add_check(
        blockers,
        data.get("schema") == LEGACY_256D_MIGRATION_POLICY_SMOKE_SCHEMA,
        "legacy 256d migration policy smoke schema mismatch",
    )
    expected = {
        "legacy_256d_open_passed": True,
        "legacy_provider_available": True,
        "mismatch_rejects_clearly": True,
        "in_place_upgrade_supported": False,
        "reembed_rebuild_required": True,
    }
    for field, value in expected.items():
        add_check(blockers, data.get(field) is value, f"legacy 256d migration policy smoke {field} must be {str(value).lower()}")
    details = {"schema": data.get("schema"), **{field: data.get(field) for field in expected}}
    return blockers, details


def validate_startup_load_encode_throughput_gate(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    add_check(
        blockers,
        data.get("schema") == STARTUP_LOAD_ENCODE_THROUGHPUT_GATE_SCHEMA,
        "startup/load/encode throughput gate schema mismatch",
    )
    validate_identity_fields(
        blockers,
        data,
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        label="startup/load/encode throughput gate",
    )
    owner_exception = data.get("explicit_owner_exception") is True
    cold_load_ms = as_number(data.get("cold_load_ms"))
    first_query_encode_ms = as_number(data.get("first_query_encode_ms"))
    warm_batch64_docs_per_second = as_number(data.get("warm_batch64_docs_per_second"))
    peak_rss_mb = as_number(data.get("peak_rss_mb"))
    add_check(
        blockers,
        cold_load_ms is not None and cold_load_ms > 0.0,
        "startup/load/encode throughput gate cold_load_ms must be > 0",
    )
    add_check(
        blockers,
        owner_exception or (cold_load_ms is not None and cold_load_ms <= DEFAULT_COLD_LOAD_MS_CEILING),
        f"startup/load/encode throughput gate cold_load_ms exceeds {DEFAULT_COLD_LOAD_MS_CEILING:g}ms ceiling",
    )
    add_check(
        blockers,
        first_query_encode_ms is not None and first_query_encode_ms > 0.0,
        "startup/load/encode throughput gate first_query_encode_ms must be > 0",
    )
    add_check(
        blockers,
        warm_batch64_docs_per_second is not None and warm_batch64_docs_per_second > 0.0,
        "startup/load/encode throughput gate warm_batch64_docs_per_second must be > 0",
    )
    add_check(
        blockers,
        owner_exception
        or (
            warm_batch64_docs_per_second is not None
            and warm_batch64_docs_per_second >= DEFAULT_WARM_BATCH64_DOCS_PER_SECOND_FLOOR
        ),
        "startup/load/encode throughput gate warm batch64 throughput below 10 docs/s floor",
    )
    add_check(
        blockers,
        peak_rss_mb is not None and peak_rss_mb > 0.0,
        "startup/load/encode throughput gate peak_rss_mb must be > 0",
    )
    details = {
        "schema": data.get("schema"),
        "cold_load_ms": cold_load_ms,
        "cold_load_ms_ceiling": DEFAULT_COLD_LOAD_MS_CEILING,
        "first_query_encode_ms": first_query_encode_ms,
        "warm_batch64_docs_per_second": warm_batch64_docs_per_second,
        "warm_batch64_docs_per_second_floor": DEFAULT_WARM_BATCH64_DOCS_PER_SECOND_FLOOR,
        "peak_rss_mb": peak_rss_mb,
        "explicit_owner_exception": owner_exception,
    }
    return blockers, details


def validate_default_asset_size_policy(
    data: dict[str, Any],
    *,
    package_sha256: str,
    identity_sha256: str,
    **_: Any,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    add_check(
        blockers,
        data.get("schema") == DEFAULT_ASSET_SIZE_POLICY_SCHEMA,
        "default asset size policy schema mismatch",
    )
    validate_identity_fields(
        blockers,
        data,
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        label="default asset size policy",
    )
    package_bytes = as_number(data.get("package_bytes"))
    default_in_repo_asset_bytes = as_number(data.get("default_in_repo_asset_bytes"))
    large_default_asset_approved = data.get("large_default_asset_approved") is True
    selected_policy = data.get("selected_policy")
    add_check(blockers, package_bytes is not None and package_bytes > 0.0, "default asset size policy package_bytes must be > 0")
    add_check(
        blockers,
        package_bytes is not None and package_bytes <= DEFAULT_EXTERNAL_PACKAGE_BYTES_CEILING,
        "default asset size policy package_bytes exceeds 200000000 byte ceiling",
    )
    add_check(
        blockers,
        default_in_repo_asset_bytes is not None and default_in_repo_asset_bytes >= 0.0,
        "default asset size policy default_in_repo_asset_bytes must be >= 0",
    )
    add_check(
        blockers,
        large_default_asset_approved
        or (
            default_in_repo_asset_bytes is not None
            and default_in_repo_asset_bytes <= DEFAULT_IN_REPO_ASSET_BYTES_CEILING
        ),
        "default asset size policy default_in_repo_asset_bytes exceeds 25000000 byte ceiling",
    )
    add_check(
        blockers,
        selected_policy in DEFAULT_ASSET_SIZE_POLICIES,
        "default asset size policy selected_policy is not approved",
    )
    details = {
        "schema": data.get("schema"),
        "package_bytes": package_bytes,
        "package_bytes_ceiling": DEFAULT_EXTERNAL_PACKAGE_BYTES_CEILING,
        "default_in_repo_asset_bytes": default_in_repo_asset_bytes,
        "default_in_repo_asset_bytes_ceiling": DEFAULT_IN_REPO_ASSET_BYTES_CEILING,
        "large_default_asset_approved": large_default_asset_approved,
        "selected_policy": selected_policy,
    }
    return blockers, details


def summarize_evidence_gate(
    name: str,
    raw_path: str | Path | None,
    validator: Any,
    *,
    missing_message: str | None = None,
    package_sha256: str,
    identity_sha256: str,
    legacy_model_name: str,
    public_name: str,
    model: str,
) -> dict[str, Any]:
    if raw_path is None:
        return {
            "status": "missing",
            "evidence_path": None,
            "blockers": [missing_message or f"{name} evidence missing: evidence path not supplied"],
            "details": {},
        }
    path = Path(raw_path)
    if not path.exists():
        return {
            "status": "missing",
            "evidence_path": str(path),
            "blockers": [f"{missing_message}: {path}" if missing_message else f"{name} evidence missing: {path}"],
            "details": {},
        }
    try:
        data = load_json_object(path)
        blockers, details = validator(
            data,
            package_sha256=package_sha256,
            identity_sha256=identity_sha256,
            legacy_model_name=legacy_model_name,
            public_name=public_name,
            model=model,
        )
    except ReadinessError as exc:
        return {
            "status": "fail",
            "evidence_path": str(path),
            "blockers": [f"{name} evidence failed validation: {exc}"],
            "details": {},
        }
    return {
        "status": "pass" if not blockers else "fail",
        "evidence_path": str(path),
        "blockers": [f"{name} evidence failed validation: {blocker}" for blocker in blockers],
        "details": details,
    }


def summarize_non_default_evidence(
    *,
    candidate_smoke_evidence: str | Path | None,
    role_aware_provider_smoke_evidence: str | Path | None,
    corkscrewdb_serving_smoke_evidence: str | Path | None,
    package_sha256: str,
    identity_sha256: str,
    legacy_model_name: str,
    public_name: str,
    model: str,
) -> dict[str, Any]:
    gates = {
        "candidate_smoke": summarize_evidence_gate(
            "candidate smoke",
            candidate_smoke_evidence,
            validate_candidate_smoke,
            package_sha256=package_sha256,
            identity_sha256=identity_sha256,
            legacy_model_name=legacy_model_name,
            public_name=public_name,
            model=model,
        ),
        "role_aware_provider_smoke": summarize_evidence_gate(
            "role-aware provider smoke",
            role_aware_provider_smoke_evidence,
            validate_role_aware_provider_smoke,
            package_sha256=package_sha256,
            identity_sha256=identity_sha256,
            legacy_model_name=legacy_model_name,
            public_name=public_name,
            model=model,
        ),
        "corkscrewdb_serving_smoke": summarize_evidence_gate(
            "CorkScrewDB serving smoke",
            corkscrewdb_serving_smoke_evidence,
            validate_serving_smoke,
            package_sha256=package_sha256,
            identity_sha256=identity_sha256,
            legacy_model_name=legacy_model_name,
            public_name=public_name,
            model=model,
        ),
    }
    blockers = [blocker for gate in gates.values() for blocker in gate.get("blockers", [])]
    return {
        "gates": gates,
        "all_valid": not blockers,
        "blockers": blockers,
    }


def build_summary(
    *,
    bge_gate_root: Path,
    datasets: list[str],
    public_name: str = DEFAULT_PUBLIC_NAME,
    public_id: str = DEFAULT_PUBLIC_ID,
    legacy_model_name: str = DEFAULT_LEGACY_MODEL_NAME,
    package_path: str = bge_gate.DEFAULT_PACKAGE_PATH,
    package_sha256: str = bge_gate.DEFAULT_PACKAGE_SHA256,
    identity_sha256: str = bge_gate.DEFAULT_IDENTITY_SHA256,
    model: str = bge_gate.DEFAULT_MODEL,
    snapshot: str = bge_gate.DEFAULT_SNAPSHOT,
    default_gate_evidence_paths: dict[str, str | None] | None = None,
    candidate_smoke_evidence: str | Path | None = None,
    role_aware_provider_smoke_evidence: str | Path | None = None,
    corkscrewdb_serving_smoke_evidence: str | Path | None = None,
    scan_paths: list[Path] | None = None,
    clock: Any = utc_now,
) -> dict[str, Any]:
    bge_summary = bge_gate.build_summary(
        run_root=bge_gate_root,
        datasets=datasets,
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        model=model,
        snapshot=snapshot,
    )
    bge = compact_bge_summary(bge_summary)
    scan = scan_public_name_hygiene(scan_paths or [])
    default_gates = summarize_default_gates(
        default_gate_evidence_paths or {},
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        legacy_model_name=legacy_model_name,
        public_name=public_name,
        model=model,
    )
    non_default_evidence = summarize_non_default_evidence(
        candidate_smoke_evidence=candidate_smoke_evidence,
        role_aware_provider_smoke_evidence=role_aware_provider_smoke_evidence,
        corkscrewdb_serving_smoke_evidence=corkscrewdb_serving_smoke_evidence,
        package_sha256=package_sha256,
        identity_sha256=identity_sha256,
        legacy_model_name=legacy_model_name,
        public_name=public_name,
        model=model,
    )

    bge_ready = bool(bge["all_complete"] and bge["identity_consistent"])
    non_default_blockers: list[str] = []
    if not bge["all_complete"]:
        non_default_blockers.append("selected BGE gate incomplete")
    if not bge["identity_consistent"]:
        non_default_blockers.append("selected BGE gate identity inconsistent")
    non_default_blockers.extend(f"bge gate: {blocker}" for blocker in bge.get("blockers", []))
    non_default_blockers.extend(non_default_evidence["blockers"])
    non_default_blockers.extend(scan["blockers"])

    non_default_status = (
        "ready_for_review"
        if bge_ready and non_default_evidence["all_valid"] and not scan["blockers"]
        else "defer"
    )
    default_swap_blockers = list(default_gates["blockers"])
    if non_default_status != "ready_for_review":
        default_swap_blockers.append("non-default candidate not ready for review")
    default_swap_blockers.extend(scan["blockers"])
    default_swap_status = "ready_for_review" if not default_swap_blockers else "defer"

    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "quality_claim": False,
        "default_alias_changed": False,
        "public_name": public_name,
        "public_id": public_id,
        "legacy_model_name": legacy_model_name,
        "non_default_candidate_status": non_default_status,
        "default_swap_status": default_swap_status,
        "blockers": {
            "non_default": non_default_blockers,
            "default_swap": default_swap_blockers,
        },
        "warnings": scan["warnings"],
        "identity": release_identity(
            public_name=public_name,
            public_id=public_id,
            legacy_model_name=legacy_model_name,
            package_path=package_path,
            package_sha256=package_sha256,
            identity_sha256=identity_sha256,
            model=model,
            snapshot=snapshot,
        ),
        "bge_gate": bge,
        "non_default_evidence": non_default_evidence,
        "default_swap_gates": default_gates,
        "public_name_hygiene": scan,
    }


def format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def tsv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    identity = summary["identity"]
    rows: list[dict[str, Any]] = [
        {
            "section": "summary",
            "key": "non_default_candidate_status",
            "value": summary["non_default_candidate_status"],
            "status": summary["non_default_candidate_status"],
        },
        {
            "section": "summary",
            "key": "default_swap_status",
            "value": summary["default_swap_status"],
            "status": summary["default_swap_status"],
        },
        {"section": "summary", "key": "quality_claim", "value": summary["quality_claim"], "status": ""},
        {
            "section": "summary",
            "key": "default_alias_changed",
            "value": summary["default_alias_changed"],
            "status": "",
        },
        {"section": "identity", "key": "public_name", "value": identity["public_name"], "status": ""},
        {"section": "identity", "key": "public_id", "value": identity["public_id"], "status": ""},
        {"section": "identity", "key": "package_sha256", "value": identity["package_sha256"], "status": ""},
        {"section": "identity", "key": "identity_sha256", "value": identity["identity_sha256"], "status": ""},
        {
            "section": "identity",
            "key": "candidate_provider_id",
            "value": identity["candidate_provider_id"],
            "status": "",
        },
        {
            "section": "identity",
            "key": "backend_fingerprint_shape",
            "value": identity["backend_fingerprint_shape"],
            "status": "",
        },
        {
            "section": "bge_gate",
            "key": "all_complete",
            "value": summary["bge_gate"]["all_complete"],
            "status": "",
        },
        {
            "section": "bge_gate",
            "key": "identity_consistent",
            "value": summary["bge_gate"]["identity_consistent"],
            "status": "",
        },
    ]
    for gate, data in summary["default_swap_gates"]["gates"].items():
        rows.append(
            {
                "section": "default_swap_gate",
                "key": gate,
                "value": data.get("evidence_path"),
                "status": data.get("status"),
            }
        )
        rows.append(
            {
                "section": "default_swap_gate_detail",
                "key": gate,
                "value": json.dumps(data.get("details", {}), sort_keys=True, separators=(",", ":")),
                "status": data.get("status"),
            }
        )
    for gate, data in summary["non_default_evidence"]["gates"].items():
        rows.append(
            {
                "section": "non_default_evidence",
                "key": gate,
                "value": data.get("evidence_path"),
                "status": data.get("status"),
            }
        )
        rows.append(
            {
                "section": "non_default_evidence_detail",
                "key": gate,
                "value": json.dumps(data.get("details", {}), sort_keys=True, separators=(",", ":")),
                "status": data.get("status"),
            }
        )
    rows.extend(
        {"section": "blocker.non_default", "key": str(index), "value": blocker, "status": "block"}
        for index, blocker in enumerate(summary["blockers"]["non_default"], start=1)
    )
    rows.extend(
        {"section": "blocker.default_swap", "key": str(index), "value": blocker, "status": "block"}
        for index, blocker in enumerate(summary["blockers"]["default_swap"], start=1)
    )
    rows.extend(
        {"section": "warning", "key": str(index), "value": warning, "status": "warn"}
        for index, warning in enumerate(summary["warnings"], start=1)
    )
    for progress in summary["bge_gate"].get("incomplete_dataset_progress", []):
        dataset = progress.get("dataset")
        for key, value in progress.items():
            if key == "dataset":
                continue
            rows.append(
                {
                    "section": "bge_progress",
                    "key": f"{dataset}.{key}",
                    "value": value,
                    "status": "block" if progress.get("status") != "complete" else "pass",
                }
            )
    return rows


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "key", "value", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in tsv_rows(summary):
            writer.writerow({key: format_tsv_value(row.get(key)) for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge-gate-root", default=DEFAULT_BGE_GATE_ROOT)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--public-name", default=DEFAULT_PUBLIC_NAME)
    parser.add_argument("--public-id", default=DEFAULT_PUBLIC_ID)
    parser.add_argument("--legacy-model-name", default=DEFAULT_LEGACY_MODEL_NAME)
    parser.add_argument("--package-path", default=bge_gate.DEFAULT_PACKAGE_PATH)
    parser.add_argument("--package-sha256", default=bge_gate.DEFAULT_PACKAGE_SHA256)
    parser.add_argument("--identity-sha256", default=bge_gate.DEFAULT_IDENTITY_SHA256)
    parser.add_argument("--model", default=bge_gate.DEFAULT_MODEL)
    parser.add_argument("--snapshot", default=bge_gate.DEFAULT_SNAPSHOT)
    parser.add_argument("--scan-paths", default="")
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--require-non-default-ready", action="store_true")
    parser.add_argument("--require-default-ready", action="store_true")
    parser.add_argument("--default-provider-bridge-evidence")
    parser.add_argument("--default-release-smoke-evidence")
    parser.add_argument("--legacy-256d-migration-evidence")
    parser.add_argument("--throughput-gate-evidence")
    parser.add_argument("--default-asset-size-policy-evidence")
    parser.add_argument("--candidate-smoke-evidence")
    parser.add_argument("--role-aware-provider-smoke-evidence")
    parser.add_argument("--corkscrewdb-serving-smoke-evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        output_json = (
            Path(args.output_json)
            if args.output_json
            else Path(args.bge_gate_root) / "eos-embedder1-release-readiness.json"
        )
        output_tsv = (
            Path(args.output_tsv)
            if args.output_tsv
            else Path(args.bge_gate_root) / "eos-embedder1-release-readiness.tsv"
        )
        summary = build_summary(
            bge_gate_root=Path(args.bge_gate_root),
            datasets=parse_datasets(args.datasets),
            public_name=args.public_name,
            public_id=args.public_id,
            legacy_model_name=args.legacy_model_name,
            package_path=args.package_path,
            package_sha256=args.package_sha256,
            identity_sha256=args.identity_sha256,
            model=args.model,
            snapshot=args.snapshot,
            default_gate_evidence_paths=default_gate_evidence_map(args),
            candidate_smoke_evidence=args.candidate_smoke_evidence,
            role_aware_provider_smoke_evidence=args.role_aware_provider_smoke_evidence,
            corkscrewdb_serving_smoke_evidence=args.corkscrewdb_serving_smoke_evidence,
            scan_paths=parse_scan_paths(args.scan_paths),
        )
        write_json(output_json, summary)
        write_tsv(output_tsv, summary)
    except (ReadinessError, bge_gate.SummaryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.require_non_default_ready and summary["non_default_candidate_status"] != "ready_for_review":
        return 2
    if args.require_default_ready and summary["default_swap_status"] != "ready_for_review":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
