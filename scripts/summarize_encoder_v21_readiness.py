#!/usr/bin/env python3
"""Summarize encoder-v2.1 controlled-training launch readiness.

This utility is read-only with respect to run inputs. It checks descriptor,
binary, sidecar, baseline metric, and selected BGE gate artifacts, then writes
JSON/TSV summaries only to the requested or default output paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as bge_gate


SUMMARY_SCHEMA = "eos.encoder_v21_readiness_summary.v1"
DEFAULT_RUN_ROOT = "runs/encoder-v21-current-binary-tiny-preflight-v1-20260629T180756Z"
DEFAULT_BGE_GATE_ROOT = "runs/bge-selected-package-full-gate-v1-20260629T000000Z"
DEFAULT_DESCRIPTOR = ".tiller/scratch/codex/encoder-v21-controlled-training-ready-descriptor-v1.md"
DEFAULT_DATASETS = "scifact,nfcorpus,fiqa"
DEFAULT_TINY_METRICS = "pretrain-retrieval-diagnostic.metrics.json"
DEFAULT_CAPPED_METRICS = "pretrain-retrieval-capped.metrics.json"
DEFAULT_BINARY = "bin/eos"
EXPECTED_TINY_NDCG_AT_10 = 0.06666667
EXPECTED_CAPPED_NDCG_AT_10 = 0.08068873
DEFAULT_METRIC_TOLERANCE = 1e-6
REQUIRED_SIDECARS = [
    "eos-embed-v1.train.mll",
    "eos-embed-v1.embed-train.mll",
    "eos-embed-v1.weights.mll",
    "eos-embed-v1.embedding.mll",
    "eos-embed-v1.memory.mll",
    "eos-embed-v1.package.mll",
    "eos-embed-v1.train-profile.mll",
    "eos-embed-v1.tokenizer.mll",
    "eos-embed-v1.eval.tokens.jsonl",
    "eos-embed-v1.mll",
]


class ReadinessError(ValueError):
    """Raised when readiness inputs or outputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_datasets(value: str) -> list[str]:
    try:
        return bge_gate.parse_datasets(value)
    except bge_gate.SummaryError as exc:
        raise ReadinessError(str(exc)) from exc


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
    return float(value)


def approximately_equal(left: float | None, right: float, tolerance: float) -> bool:
    return left is not None and abs(left - right) <= tolerance


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    for section in ("final_eval", "best_eval", "last_eval"):
        values = metrics.get(section)
        if isinstance(values, dict):
            number = as_number(values.get(key))
            if number is not None:
                return number
    return None


def compact_eval_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    workload = metrics.get("workload") if isinstance(metrics.get("workload"), dict) else {}
    config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    return {
        "retrieval_ndcg_at_10": metric_value(metrics, "retrieval_ndcg_at_10"),
        "retrieval_map_at_100": metric_value(metrics, "retrieval_map_at_100"),
        "retrieval_recall_at_100": metric_value(metrics, "retrieval_recall_at_100"),
        "pair_count": metric_value(metrics, "pair_count"),
        "positive_count": metric_value(metrics, "positive_count"),
        "negative_count": metric_value(metrics, "negative_count"),
        "actual_eval_passes": workload.get("actual_eval_passes"),
        "actual_eval_pairs": workload.get("actual_eval_pairs"),
        "actual_eval_examples": workload.get("actual_eval_examples"),
        "eval_only": config.get("eval_only"),
        "select_metric": config.get("select_metric"),
    }


def summarize_baseline(path: Path, expected_ndcg: float, tolerance: float) -> dict[str, Any]:
    exists = path.exists()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "expected_retrieval_ndcg_at_10": expected_ndcg,
        "retrieval_ndcg_at_10_tolerance": tolerance,
        "observed_retrieval_ndcg_at_10": None,
        "matches_expected": False,
        "metrics": None,
    }
    if not exists:
        return result
    metrics = load_json_object(path)
    observed = metric_value(metrics, "retrieval_ndcg_at_10")
    result["observed_retrieval_ndcg_at_10"] = observed
    result["matches_expected"] = approximately_equal(observed, expected_ndcg, tolerance)
    result["metrics"] = compact_eval_metrics(metrics)
    return result


def summarize_sidecars(run_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in REQUIRED_SIDECARS:
        path = run_root / name
        exists = path.exists()
        files[name] = {"path": str(path), "exists": exists}
        if not exists:
            missing.append(name)
    return {
        "run_root": str(run_root),
        "required": REQUIRED_SIDECARS,
        "files": files,
        "missing": missing,
        "complete": not missing,
    }


def summarize_binary(path: Path) -> dict[str, Any]:
    exists = path.exists()
    executable = exists and os.access(path, os.X_OK)
    return {
        "path": str(path),
        "exists": exists,
        "executable": executable,
        "ready": exists and executable,
    }


def active_export_markers_from_bge_summary(summary: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    for dataset in summary.get("datasets", []):
        if not isinstance(dataset, dict):
            continue
        name = str(dataset.get("dataset"))
        if dataset.get("status") == "complete":
            continue
        partial_lines = dataset.get("partial_doc_vector_lines")
        present = dataset.get("present_artifacts")
        missing = dataset.get("missing_artifacts")
        if partial_lines is not None:
            markers.append(f"{name}: partial doc vector export lines={partial_lines}")
        elif present:
            markers.append(
                f"{name}: incomplete artifacts present={','.join(str(item) for item in present)} "
                f"missing={','.join(str(item) for item in missing or [])}"
            )
    return markers


def summarize_bge_gate(run_root: Path, datasets: list[str]) -> dict[str, Any]:
    summary = bge_gate.build_summary(run_root=run_root, datasets=datasets)
    aggregate = summary["aggregate"]
    markers = active_export_markers_from_bge_summary(summary)
    return {
        "run_root": str(run_root),
        "datasets": datasets,
        "summary_schema": summary.get("schema"),
        "all_complete": aggregate.get("all_complete"),
        "complete_dataset_count": aggregate.get("complete_dataset_count"),
        "expected_dataset_count": aggregate.get("expected_dataset_count"),
        "identity_consistent": aggregate.get("identity_consistent"),
        "promotion_recommendation": aggregate.get("promotion_recommendation"),
        "blockers": aggregate.get("blockers", []),
        "active_export_markers": markers,
        "summary": summary,
    }


def build_summary(
    *,
    run_root: Path,
    bge_gate_root: Path,
    descriptor: Path,
    datasets: list[str],
    binary_path: Path | None = None,
    tiny_metrics_path: Path | None = None,
    capped_metrics_path: Path | None = None,
    metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    clock: Any = utc_now,
) -> dict[str, Any]:
    binary = summarize_binary(binary_path or run_root / DEFAULT_BINARY)
    sidecars = summarize_sidecars(run_root)
    tiny = summarize_baseline(
        tiny_metrics_path or run_root / DEFAULT_TINY_METRICS,
        EXPECTED_TINY_NDCG_AT_10,
        metric_tolerance,
    )
    capped = summarize_baseline(
        capped_metrics_path or run_root / DEFAULT_CAPPED_METRICS,
        EXPECTED_CAPPED_NDCG_AT_10,
        metric_tolerance,
    )
    bge = summarize_bge_gate(bge_gate_root, datasets)
    descriptor_exists = descriptor.exists()

    blockers: list[str] = []
    warnings: list[str] = []
    if not descriptor_exists:
        blockers.append(f"descriptor missing: {descriptor}")
    if not binary["exists"]:
        blockers.append(f"current binary missing: {binary['path']}")
    elif not binary["executable"]:
        blockers.append(f"current binary is not executable: {binary['path']}")
    if not sidecars["complete"]:
        blockers.append("missing sidecars: " + ",".join(sidecars["missing"]))
    if not tiny["exists"]:
        blockers.append(f"tiny baseline metrics missing: {tiny['path']}")
    elif not tiny["matches_expected"]:
        blockers.append(
            "tiny baseline retrieval_ndcg_at_10 mismatch: "
            f"observed={tiny['observed_retrieval_ndcg_at_10']} expected={EXPECTED_TINY_NDCG_AT_10}"
        )
    if not capped["exists"]:
        blockers.append(f"capped baseline metrics missing: {capped['path']}")
    elif not capped["matches_expected"]:
        blockers.append(
            "capped baseline retrieval_ndcg_at_10 mismatch: "
            f"observed={capped['observed_retrieval_ndcg_at_10']} expected={EXPECTED_CAPPED_NDCG_AT_10}"
        )
    if not bge["all_complete"]:
        blockers.append("selected BGE gate incomplete")
    if not bge["identity_consistent"]:
        blockers.append("selected BGE gate identity inconsistent")
    for blocker in bge["blockers"]:
        blockers.append(f"bge gate: {blocker}")
    if bge["active_export_markers"]:
        blockers.append("active or partial export marker present")
    else:
        warnings.append("no active export marker was discoverable from BGE artifacts")

    launch_allowed = not blockers
    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "run_root": str(run_root),
        "descriptor": {
            "path": str(descriptor),
            "exists": descriptor_exists,
        },
        "current_binary": binary,
        "sidecars": sidecars,
        "baselines": {
            "tiny": tiny,
            "capped": capped,
        },
        "bge_gate": bge,
        "blockers": blockers,
        "warnings": warnings,
        "launch_allowed": launch_allowed,
        "quality_claim": False,
        "training_run": False,
    }


def format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def tsv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"section": "summary", "key": "launch_allowed", "value": summary["launch_allowed"], "status": ""},
        {"section": "summary", "key": "quality_claim", "value": summary["quality_claim"], "status": ""},
        {"section": "summary", "key": "training_run", "value": summary["training_run"], "status": ""},
        {"section": "descriptor", "key": "exists", "value": summary["descriptor"]["exists"], "status": ""},
        {"section": "current_binary", "key": "exists", "value": summary["current_binary"]["exists"], "status": ""},
        {
            "section": "current_binary",
            "key": "executable",
            "value": summary["current_binary"]["executable"],
            "status": "",
        },
        {"section": "sidecars", "key": "complete", "value": summary["sidecars"]["complete"], "status": ""},
        {
            "section": "baseline",
            "key": "tiny.retrieval_ndcg_at_10",
            "value": summary["baselines"]["tiny"]["observed_retrieval_ndcg_at_10"],
            "expected": summary["baselines"]["tiny"]["expected_retrieval_ndcg_at_10"],
            "status": "pass" if summary["baselines"]["tiny"]["matches_expected"] else "fail",
        },
        {
            "section": "baseline",
            "key": "capped.retrieval_ndcg_at_10",
            "value": summary["baselines"]["capped"]["observed_retrieval_ndcg_at_10"],
            "expected": summary["baselines"]["capped"]["expected_retrieval_ndcg_at_10"],
            "status": "pass" if summary["baselines"]["capped"]["matches_expected"] else "fail",
        },
        {"section": "bge_gate", "key": "all_complete", "value": summary["bge_gate"]["all_complete"], "status": ""},
        {
            "section": "bge_gate",
            "key": "identity_consistent",
            "value": summary["bge_gate"]["identity_consistent"],
            "status": "",
        },
    ]
    rows.extend(
        {"section": "blocker", "key": str(index), "value": blocker, "status": "block"}
        for index, blocker in enumerate(summary["blockers"], start=1)
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
    fieldnames = ["section", "key", "value", "expected", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in tsv_rows(summary):
            writer.writerow({key: format_tsv_value(row.get(key)) for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--bge-gate-root", default=DEFAULT_BGE_GATE_ROOT)
    parser.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--binary")
    parser.add_argument("--tiny-metrics")
    parser.add_argument("--capped-metrics")
    parser.add_argument("--metric-tolerance", type=float, default=DEFAULT_METRIC_TOLERANCE)
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_root = Path(args.run_root)
        output_json = Path(args.output_json) if args.output_json else run_root / "encoder-v21-readiness-summary.json"
        output_tsv = Path(args.output_tsv) if args.output_tsv else run_root / "encoder-v21-readiness-summary.tsv"
        summary = build_summary(
            run_root=run_root,
            bge_gate_root=Path(args.bge_gate_root),
            descriptor=Path(args.descriptor),
            datasets=parse_datasets(args.datasets),
            binary_path=Path(args.binary) if args.binary else None,
            tiny_metrics_path=Path(args.tiny_metrics) if args.tiny_metrics else None,
            capped_metrics_path=Path(args.capped_metrics) if args.capped_metrics else None,
            metric_tolerance=args.metric_tolerance,
        )
        write_json(output_json, summary)
        write_tsv(output_tsv, summary)
    except (ReadinessError, bge_gate.SummaryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.require_ready and not summary["launch_allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
