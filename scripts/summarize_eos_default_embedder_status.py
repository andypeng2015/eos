#!/usr/bin/env python3
"""Summarize the current default Eos embedder status.

This is a read-only/status-only utility. It consolidates manifest capability
evidence, disk gates, and optional reclaim estimates into reproducible JSON and
TSV artifacts. It never trains, evaluates, deletes, stages, commits, or pushes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


STATUS_SCHEMA = "eos.default_embedder_status.v1"
QUALITY_CLAIM = False
CLAIM_BOUNDARY = (
    "Current default-embedder status summary only. Long-context evidence is "
    "diagnostic/non-claim, and this artifact is not model-quality or product "
    "claim evidence."
)
DEFAULT_MANIFEST = "assets/corkscrewdb-default-embedder/manifest.json"
DEFAULT_RECLAIM_MANIFEST = ".tiller/scratch/codex/eos-reclaim-approved-candidates-v1.manifest.json"
DEFAULT_RECLAIM_SUMMARY = ".tiller/scratch/codex/eos-reclaim-approved-candidates-v1.summary.json"
GIB = 1024 * 1024 * 1024


FreeBytes = Callable[[Path], int]
Clock = Callable[[], str]


class StatusError(ValueError):
    """Raised when required status inputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root_from(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StatusError(f"required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StatusError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StatusError(f"{path}: expected top-level JSON object")
    return data


def load_optional_json_object(path: Path, warnings: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        warnings.append(f"optional {label} not found: {path}")
        return None
    try:
        return load_json_object(path)
    except StatusError as exc:
        warnings.append(f"optional {label} unreadable: {exc}")
        return None


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def min_free_bytes(min_free_gb: float) -> int:
    if min_free_gb < 0:
        raise StatusError("free-space thresholds must be non-negative")
    return int(min_free_gb * GIB)


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compute_macro_dense_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ndcg = as_number(row.get("ndcg_at_10"))
        recall = as_number(row.get("recall_at_100"))
        dataset = str(row.get("dataset", "")).strip()
        if ndcg is None or recall is None or not dataset:
            continue
        ndcg_values.append(ndcg)
        recall_values.append(recall)
        compact_rows.append(
            {
                "dataset": dataset,
                "ndcg_at_10": ndcg,
                "recall_at_100": recall,
            }
        )
    return {
        "rows": compact_rows,
        "macro": {
            "ndcg_at_10": mean(ndcg_values),
            "recall_at_100": mean(recall_values),
            "dataset_count": len(compact_rows),
        },
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compact_model_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    tokenizer = manifest.get("tokenizer") if isinstance(manifest.get("tokenizer"), dict) else {}
    source = manifest.get("source_release") if isinstance(manifest.get("source_release"), dict) else {}
    return {
        "asset_id": manifest.get("asset_id"),
        "model_name": manifest.get("model_name"),
        "artifact_filename": artifact.get("filename"),
        "artifact_sha256": artifact.get("sha256"),
        "artifact_bytes": artifact.get("bytes"),
        "tokenizer_filename": tokenizer.get("filename"),
        "tokenizer_sha256": tokenizer.get("sha256"),
        "source_release_directory": source.get("directory"),
        "source_release_package_sha256": source.get("package_sha256"),
        "source_release_sealed_sha256": source.get("sealed_sha256"),
    }


def compact_gate(gate: Any) -> dict[str, Any] | None:
    if not isinstance(gate, dict):
        return None
    result = {
        "status": gate.get("status"),
        "checks": gate.get("checks"),
        "baseline": gate.get("baseline"),
        "candidate": gate.get("candidate"),
        "gate_log": gate.get("gate_log"),
    }
    for key in ("deltas_vs_s40", "deltas_vs_s40_compact_anchor"):
        if key in gate:
            result[key] = gate[key]
    return result


def compact_policy_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = manifest.get("compact_policy") if isinstance(manifest.get("compact_policy"), dict) else {}
    result = {
        "profile": policy.get("profile"),
        "baseline": policy.get("baseline"),
        "method": policy.get("method"),
        "bits": policy.get("bits"),
        "rerank_overfetch": policy.get("rerank_overfetch"),
        "rerank_storage": policy.get("rerank_storage"),
        "total_compression_ratio": policy.get("total_compression_ratio"),
        "strict_current_compact_non_regression": policy.get(
            "strict_current_compact_non_regression"
        ),
        "gate": compact_gate(policy.get("gate_evidence")),
    }
    return result


def dataset_label(key: str, row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        value = dataset.strip().rstrip("/")
        return value.split("/")[-1]
    lowered = key.lower()
    if "qmsum" in lowered:
        return "qmsum"
    if "2wikimqa" in lowered:
        return "2wikimqa"
    if "repo_docs" in lowered or "repo-docs" in lowered:
        return "repo-docs"
    return key


def best_eos_ndcg(row: dict[str, Any]) -> float | None:
    values = [
        as_number(value)
        for key, value in row.items()
        if key.startswith("eos_") and key.endswith("_ndcg_at_10")
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def external_q4_values(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in row.items():
        if not key.endswith("_q4_ndcg_at_10"):
            continue
        number = as_number(value)
        if number is None:
            continue
        model = key[: -len("_q4_ndcg_at_10")]
        values[model] = number
    return values


def summarize_long_context(manifest: dict[str, Any]) -> dict[str, Any]:
    evidence = manifest.get("long_context_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    rows: list[dict[str, Any]] = []
    any_external_negative = False
    any_external = False
    for key, value in evidence.items():
        if not isinstance(value, dict):
            continue
        external = external_q4_values(value)
        eos_best = best_eos_ndcg(value)
        if eos_best is None and not external:
            continue
        gaps = {
            model: eos_best - metric
            for model, metric in external.items()
            if eos_best is not None
        }
        if external:
            any_external = True
        if any(gap < 0.0 for gap in gaps.values()):
            any_external_negative = True
        rows.append(
            {
                "evidence_key": key,
                "dataset": dataset_label(key, value),
                "status": value.get("status"),
                "run_dir": value.get("run_dir"),
                "quality_claim": value.get("quality_claim", QUALITY_CLAIM),
                "eos_best_ndcg_at_10": eos_best,
                "external_q4_ndcg_at_10": external,
                "eos_minus_external_q4_ndcg_at_10": gaps,
                "caveat": value.get("caveat"),
            }
        )
    product_wedge_status = "no_external_q4_rows"
    if any_external_negative:
        product_wedge_status = "unproven_or_negative"
    elif any_external:
        product_wedge_status = "diagnostic_only"
    return {
        "quality_claim": QUALITY_CLAIM,
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest_wedge": manifest.get("long_context_wedge"),
        "evidence_rows": rows,
        "product_wedge_status": product_wedge_status,
    }


def extract_reclaim_estimate(
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    estimate = None
    source = None
    dry_run = None
    if summary:
        estimate = summary.get("total_estimated_reclaim_bytes")
        dry_run = summary.get("dry_run")
        source = "summary.total_estimated_reclaim_bytes"
        if estimate is None and isinstance(summary.get("paths"), list):
            total = 0
            found = False
            for row in summary["paths"]:
                if isinstance(row, dict) and isinstance(row.get("estimated_bytes"), int):
                    total += int(row["estimated_bytes"])
                    found = True
            if found:
                estimate = total
                source = "summary.paths.estimated_bytes"
    if estimate is None and manifest and isinstance(manifest.get("paths"), list):
        total = 0
        found = False
        for row in manifest["paths"]:
            if isinstance(row, dict) and isinstance(row.get("estimated_bytes"), int):
                total += int(row["estimated_bytes"])
                found = True
        if found:
            estimate = total
            source = "manifest.paths.estimated_bytes"
    return {
        "estimated_reclaim_bytes": estimate if isinstance(estimate, int) else None,
        "estimated_reclaim_gb": bytes_to_gib(estimate) if isinstance(estimate, int) else None,
        "estimate_source": source,
        "dry_run": dry_run if isinstance(dry_run, bool) else None,
        "manifest_path_count": len(manifest.get("paths", [])) if manifest else None,
    }


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / GIB


def build_status(
    *,
    repo_root: Path,
    manifest_json: Path,
    reclaim_manifest_json: Path | None = None,
    reclaim_summary_json: Path | None = None,
    disk_path: Path | None = None,
    long_context_min_free_gb: float = 20.0,
    training_min_free_gb: float = 15.0,
    free_bytes: FreeBytes = disk_free_bytes,
    clock: Clock = utc_now,
) -> dict[str, Any]:
    warnings: list[str] = []
    manifest = load_json_object(manifest_json)
    reclaim_summary = (
        load_optional_json_object(reclaim_summary_json, warnings, "reclaim summary")
        if reclaim_summary_json
        else None
    )
    reclaim_manifest = (
        load_optional_json_object(reclaim_manifest_json, warnings, "reclaim manifest")
        if reclaim_manifest_json
        else None
    )
    reclaim = extract_reclaim_estimate(reclaim_summary, reclaim_manifest)
    disk_target = (disk_path or repo_root).resolve()
    free = int(free_bytes(disk_target))
    long_threshold = min_free_bytes(long_context_min_free_gb)
    train_threshold = min_free_bytes(training_min_free_gb)
    blockers: list[str] = []
    next_actions: list[str] = []
    if free < long_threshold:
        blockers.append("long_context_eval_disk_blocked")
        if reclaim.get("estimated_reclaim_bytes") is not None:
            next_actions.append("apply audited reclaim manifest after explicit approval")
        else:
            next_actions.append("free disk above long-context threshold before rerun")
    if free < train_threshold:
        blockers.append("training_disk_blocked")
    if not next_actions:
        next_actions.append("rerun bounded long-context product-wedge evaluation when explicitly requested")

    dense_short = compute_macro_dense_metrics(
        manifest.get("dense_short_metrics") if isinstance(manifest.get("dense_short_metrics"), list) else []
    )
    long_context = summarize_long_context(manifest)
    return {
        "schema": STATUS_SCHEMA,
        "quality_claim": QUALITY_CLAIM,
        "claim_boundary": CLAIM_BOUNDARY,
        "created_at": clock(),
        "model": compact_model_fields(manifest),
        "dense_short": {
            **dense_short,
            "gate": compact_gate(manifest.get("dense_gate_evidence")),
        },
        "compact_policy": compact_policy_fields(manifest),
        "long_context": long_context,
        "disk": {
            "path": str(disk_target),
            "free_bytes": free,
            "free_gb": bytes_to_gib(free),
            "long_context_min_free_gb": long_context_min_free_gb,
            "long_context_min_free_bytes": long_threshold,
            "training_min_free_gb": training_min_free_gb,
            "training_min_free_bytes": train_threshold,
            "reclaim": reclaim,
        },
        "blockers": blockers,
        "next_actions": next_actions,
        "warnings": warnings,
        "sources": {
            "manifest_json": str(manifest_json),
            "reclaim_manifest_json": str(reclaim_manifest_json) if reclaim_manifest_json else None,
            "reclaim_summary_json": str(reclaim_summary_json) if reclaim_summary_json else None,
        },
    }


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def tsv_rows(status: dict[str, Any]) -> list[dict[str, str]]:
    dense_macro = status["dense_short"]["macro"]
    disk = status["disk"]
    rows = [
        {"section": "status", "key": "schema", "value": status["schema"]},
        {"section": "status", "key": "quality_claim", "value": "false"},
        {"section": "model", "key": "model_name", "value": scalar_text(status["model"].get("model_name"))},
        {"section": "model", "key": "artifact_sha256", "value": scalar_text(status["model"].get("artifact_sha256"))},
        {"section": "dense_short", "key": "macro_ndcg_at_10", "value": scalar_text(dense_macro.get("ndcg_at_10"))},
        {"section": "dense_short", "key": "macro_recall_at_100", "value": scalar_text(dense_macro.get("recall_at_100"))},
        {"section": "dense_short", "key": "dense_gate_status", "value": scalar_text(status["dense_short"]["gate"].get("status") if status["dense_short"].get("gate") else None)},
        {"section": "compact_policy", "key": "profile", "value": scalar_text(status["compact_policy"].get("profile"))},
        {"section": "compact_policy", "key": "compact_gate_status", "value": scalar_text(status["compact_policy"].get("gate", {}).get("status") if status["compact_policy"].get("gate") else None)},
        {"section": "long_context", "key": "product_wedge_status", "value": scalar_text(status["long_context"].get("product_wedge_status"))},
        {"section": "disk", "key": "free_bytes", "value": scalar_text(disk.get("free_bytes"))},
        {"section": "disk", "key": "free_gb", "value": scalar_text(disk.get("free_gb"))},
        {"section": "disk", "key": "estimated_reclaim_bytes", "value": scalar_text(disk["reclaim"].get("estimated_reclaim_bytes"))},
        {"section": "decision", "key": "blockers", "value": scalar_text(status.get("blockers"))},
        {"section": "decision", "key": "next_actions", "value": scalar_text(status.get("next_actions"))},
        {"section": "decision", "key": "warnings", "value": scalar_text(status.get("warnings"))},
    ]
    for row in status["dense_short"].get("rows", []):
        rows.append(
            {
                "section": "dense_short_dataset",
                "key": row["dataset"],
                "value": f"ndcg_at_10={row['ndcg_at_10']};recall_at_100={row['recall_at_100']}",
            }
        )
    for row in status["long_context"].get("evidence_rows", []):
        rows.append(
            {
                "section": "long_context_dataset",
                "key": row["dataset"],
                "value": scalar_text(
                    {
                        "eos_best_ndcg_at_10": row.get("eos_best_ndcg_at_10"),
                        "external_q4_ndcg_at_10": row.get("external_q4_ndcg_at_10"),
                        "gaps": row.get("eos_minus_external_q4_ndcg_at_10"),
                    }
                ),
            }
        )
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "key", "value"), delimiter="\t")
        writer.writeheader()
        writer.writerows(tsv_rows(status))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to git root/cwd.")
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST)
    parser.add_argument("--reclaim-manifest-json", default=DEFAULT_RECLAIM_MANIFEST)
    parser.add_argument("--reclaim-summary-json", default=DEFAULT_RECLAIM_SUMMARY)
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--disk-path", default=None)
    parser.add_argument("--long-context-min-free-gb", type=float, default=20.0)
    parser.add_argument("--training-min-free-gb", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repo_root = repo_root_from(args.repo_root)
        status = build_status(
            repo_root=repo_root,
            manifest_json=resolve_path(repo_root, args.manifest_json),
            reclaim_manifest_json=resolve_path(repo_root, args.reclaim_manifest_json)
            if args.reclaim_manifest_json
            else None,
            reclaim_summary_json=resolve_path(repo_root, args.reclaim_summary_json)
            if args.reclaim_summary_json
            else None,
            disk_path=resolve_path(repo_root, args.disk_path) if args.disk_path else repo_root,
            long_context_min_free_gb=args.long_context_min_free_gb,
            training_min_free_gb=args.training_min_free_gb,
        )
        if args.output_json:
            write_json(resolve_path(repo_root, args.output_json), status)
        else:
            json.dump(status, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        if args.output_tsv:
            write_tsv(resolve_path(repo_root, args.output_tsv), status)
    except StatusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
