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


def compact_bge_summary(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
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
        "datasets": [
            {
                "dataset": dataset.get("dataset"),
                "status": dataset.get("status"),
                "identity_match": dataset.get("identity_match"),
                "missing_artifacts": dataset.get("missing_artifacts", []),
                "present_artifacts": dataset.get("present_artifacts", []),
                "partial_doc_vector_lines": dataset.get("partial_doc_vector_lines"),
                "blockers": dataset.get("blockers", []),
                "dense": dataset.get("dense"),
                "q8": dataset.get("q8"),
                "q4": dataset.get("q4"),
            }
            for dataset in summary.get("datasets", [])
        ],
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


def summarize_default_gates(evidence_paths: dict[str, str | None]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    blockers: list[str] = []
    for gate, missing_message in DEFAULT_SWAP_GATES:
        raw_path = evidence_paths.get(gate)
        path = Path(raw_path) if raw_path else None
        exists = path.exists() if path else False
        gates[gate] = {
            "status": "present" if exists else "missing",
            "evidence_path": str(path) if path else None,
        }
        if not exists:
            blockers.append(missing_message)
    return {
        "gates": gates,
        "all_present": not blockers,
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
    default_gates = summarize_default_gates(default_gate_evidence_paths or {})

    bge_ready = bool(bge["all_complete"] and bge["identity_consistent"])
    non_default_blockers: list[str] = []
    if not bge["all_complete"]:
        non_default_blockers.append("selected BGE gate incomplete")
    if not bge["identity_consistent"]:
        non_default_blockers.append("selected BGE gate identity inconsistent")
    non_default_blockers.extend(f"bge gate: {blocker}" for blocker in bge.get("blockers", []))
    non_default_blockers.extend(scan["blockers"])

    non_default_status = "ready_for_review" if bge_ready and not scan["blockers"] else "defer"
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
