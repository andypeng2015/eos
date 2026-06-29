#!/usr/bin/env python3
"""Summarize the selected BGE package full-gate run.

This utility is read-only with respect to run inputs. It consolidates vector
manifests, dense metrics, and TurboQuant q8/q4 metrics into JSON and TSV status
artifacts. It does not train, export, evaluate, delete, stage, commit, or push.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "eos.bge_selected_package_gate_summary.v1"
DEFAULT_RUN_ROOT = "runs/bge-selected-package-full-gate-v1-20260629T000000Z"
DEFAULT_DATASETS = "scifact,nfcorpus,fiqa"
DEFAULT_PACKAGE_PATH = (
    "runs/pretrained-bert-current-hf-parity-v1-20260629T090818Z/bge/"
    "bge-small-en-v1.5.imported.mll"
)
DEFAULT_PACKAGE_SHA256 = "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763"
DEFAULT_IDENTITY_SHA256 = "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SNAPSHOT = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_DOCUMENT_PREFIX = ""
DEFAULT_POOLING = "cls"
DEFAULT_NORMALIZATION = "l2"
DEFAULT_DIM = 384
DEFAULT_MAX_LENGTH = 512


class SummaryError(ValueError):
    """Raised when summary inputs or outputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SummaryError(f"required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SummaryError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError(f"{path}: expected top-level JSON object")
    return data


def load_optional_json_object(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json_object(path)
    except SummaryError as exc:
        blockers.append(f"{path}: unreadable JSON: {exc}")
        return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def parse_datasets(value: str) -> list[str]:
    datasets = [part.strip() for part in value.split(",") if part.strip()]
    if not datasets:
        raise SummaryError("--datasets must include at least one dataset")
    return datasets


def artifact_paths(run_root: Path, dataset: str) -> dict[str, Path]:
    dataset_root = run_root / dataset
    return {
        "doc_vectors": dataset_root / "vectors" / "doc-vectors.jsonl",
        "query_vectors": dataset_root / "vectors" / "query-vectors.jsonl",
        "vector_manifest": dataset_root / "vectors" / "manifest.json",
        "dense_metrics": dataset_root / "eval" / "dense.metrics.json",
        "turboquant_metrics": dataset_root / "eval" / "turboquant-q8-q4.metrics.json",
    }


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _line in handle:
            count += 1
    return count


def quality_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    quality = metrics.get("quality") if isinstance(metrics, dict) else None
    if not isinstance(quality, dict):
        quality = {}
    return {
        "ndcg_at_10": as_number(quality.get("ndcg_at_10")),
        "recall_at_100": as_number(quality.get("recall_at_100")),
    }


def turboquant_rows(metrics: dict[str, Any] | None, dense: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    rows = metrics.get("rows") if isinstance(metrics, dict) else None
    if not isinstance(rows, list):
        return result
    dense_ndcg = dense.get("ndcg_at_10")
    dense_recall = dense.get("recall_at_100")
    for row in rows:
        if not isinstance(row, dict):
            continue
        bits = row.get("bits")
        if bits not in (8, 4):
            continue
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        ndcg = as_number(quality.get("ndcg_at_10"))
        recall = as_number(quality.get("recall_at_100"))
        result[f"q{bits}"] = {
            "bits": bits,
            "method": row.get("method"),
            "ndcg_at_10": ndcg,
            "recall_at_100": recall,
            "ndcg_at_10_delta_vs_dense": as_number(row.get("ndcg_at_10_delta"))
            if as_number(row.get("ndcg_at_10_delta")) is not None
            else delta(ndcg, dense_ndcg),
            "recall_at_100_delta_vs_dense": as_number(row.get("recall_at_100_delta"))
            if as_number(row.get("recall_at_100_delta")) is not None
            else delta(recall, dense_recall),
            "compression_ratio": as_number(row.get("compression_ratio")),
            "total_compression_ratio": as_number(row.get("total_compression_ratio")),
        }
    return result


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def compact_manifest(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "dataset": manifest.get("dataset"),
        "package_path": manifest.get("package_path"),
        "package_sha256": manifest.get("package_sha256"),
        "package_identity_sha256": manifest.get("package_identity_sha256"),
        "embedding_space_id": manifest.get("embedding_space_id"),
        "documents": manifest.get("documents"),
        "queries": manifest.get("queries"),
        "written_documents": manifest.get("written_documents"),
        "written_queries": manifest.get("written_queries"),
        "native_dim": manifest.get("native_dim"),
        "output_dim": manifest.get("output_dim"),
        "query_prefix": manifest.get("query_prefix"),
        "document_prefix": manifest.get("document_prefix", manifest.get("doc_prefix")),
        "pooling": manifest.get("pooling"),
        "normalization": manifest.get("normalization"),
        "max_length": manifest.get("max_length"),
        "quality_claim": manifest.get("quality_claim"),
        "created_at": manifest.get("created_at"),
    }


def manifest_identity_matches(
    manifest: dict[str, Any],
    *,
    expected_package_sha256: str,
    expected_identity_sha256: str,
) -> bool:
    return (
        manifest.get("package_sha256") == expected_package_sha256
        and manifest.get("package_identity_sha256") == expected_identity_sha256
    )


def summarize_dataset(
    run_root: Path,
    dataset: str,
    *,
    expected_package_sha256: str,
    expected_identity_sha256: str,
    count_partial_vector_lines: bool,
) -> dict[str, Any]:
    paths = artifact_paths(run_root, dataset)
    present = [name for name, path in paths.items() if path.exists()]
    missing = [name for name, path in paths.items() if not path.exists()]
    blockers: list[str] = []

    manifest = load_optional_json_object(paths["vector_manifest"], blockers)
    dense_metrics = load_optional_json_object(paths["dense_metrics"], blockers)
    turboquant_metrics = load_optional_json_object(paths["turboquant_metrics"], blockers)
    dense = quality_metrics(dense_metrics)
    q_rows = turboquant_rows(turboquant_metrics, dense)
    has_q8_q4 = "q8" in q_rows and "q4" in q_rows
    complete = manifest is not None and dense_metrics is not None and has_q8_q4

    for artifact in missing:
        blockers.append(f"{dataset}: missing {artifact}")
    if turboquant_metrics is not None and not has_q8_q4:
        missing_bits = [name for name in ("q8", "q4") if name not in q_rows]
        blockers.append(f"{dataset}: missing turboquant rows {','.join(missing_bits)}")

    identity_match = None
    if manifest is not None:
        identity_match = manifest_identity_matches(
            manifest,
            expected_package_sha256=expected_package_sha256,
            expected_identity_sha256=expected_identity_sha256,
        )
        if not identity_match:
            blockers.append(f"{dataset}: vector manifest package identity mismatch")

    partial_doc_vector_lines = None
    if (
        count_partial_vector_lines
        and manifest is None
        and paths["doc_vectors"].exists()
        and not paths["query_vectors"].exists()
    ):
        partial_doc_vector_lines = count_lines(paths["doc_vectors"])

    return {
        "dataset": dataset,
        "status": "complete" if complete else "incomplete",
        "present_artifacts": present,
        "missing_artifacts": missing,
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "dense": dense if dense_metrics is not None else None,
        "q8": q_rows.get("q8"),
        "q4": q_rows.get("q4"),
        "vector_manifest": compact_manifest(manifest),
        "identity_match": identity_match,
        "partial_doc_vector_lines": partial_doc_vector_lines,
        "blockers": blockers,
    }


def metric_macro(datasets: list[dict[str, Any]], storage: str) -> dict[str, Any]:
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    used: list[str] = []
    for dataset in datasets:
        if dataset.get("status") != "complete":
            continue
        metrics = dataset.get(storage)
        if not isinstance(metrics, dict):
            continue
        ndcg = as_number(metrics.get("ndcg_at_10"))
        recall = as_number(metrics.get("recall_at_100"))
        if ndcg is None or recall is None:
            continue
        ndcg_values.append(ndcg)
        recall_values.append(recall)
        used.append(str(dataset.get("dataset")))
    return {
        "dataset_count": len(used),
        "datasets": used,
        "ndcg_at_10": mean(ndcg_values),
        "recall_at_100": mean(recall_values),
    }


def build_summary(
    *,
    run_root: Path,
    datasets: list[str],
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    identity_sha256: str = DEFAULT_IDENTITY_SHA256,
    model: str = DEFAULT_MODEL,
    snapshot: str = DEFAULT_SNAPSHOT,
    count_partial_vector_lines: bool = True,
    clock: Any = utc_now,
) -> dict[str, Any]:
    dataset_summaries = [
        summarize_dataset(
            run_root,
            dataset,
            expected_package_sha256=package_sha256,
            expected_identity_sha256=identity_sha256,
            count_partial_vector_lines=count_partial_vector_lines,
        )
        for dataset in datasets
    ]

    complete_count = sum(1 for dataset in dataset_summaries if dataset["status"] == "complete")
    all_complete = complete_count == len(dataset_summaries)
    present_manifest_count = sum(1 for dataset in dataset_summaries if dataset["vector_manifest"] is not None)
    identity_mismatches = [
        dataset["dataset"]
        for dataset in dataset_summaries
        if dataset["identity_match"] is False
    ]
    identity_consistent = not identity_mismatches
    blockers: list[str] = []
    for dataset in dataset_summaries:
        blockers.extend(dataset["blockers"])
    if identity_mismatches:
        blockers.append("package identity mismatch: " + ",".join(identity_mismatches))
    if not all_complete:
        incomplete = [dataset["dataset"] for dataset in dataset_summaries if dataset["status"] != "complete"]
        blockers.append("incomplete datasets: " + ",".join(incomplete))

    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "run_root": str(run_root),
        "expected_package": {
            "path": DEFAULT_PACKAGE_PATH,
            "package_sha256": package_sha256,
            "identity_sha256": identity_sha256,
            "model": model,
            "snapshot": snapshot,
            "query_prefix": DEFAULT_QUERY_PREFIX,
            "document_prefix": DEFAULT_DOCUMENT_PREFIX,
            "pooling": DEFAULT_POOLING,
            "normalization": DEFAULT_NORMALIZATION,
            "dim": DEFAULT_DIM,
            "max_length": DEFAULT_MAX_LENGTH,
        },
        "datasets": dataset_summaries,
        "aggregate": {
            "complete_dataset_count": complete_count,
            "expected_dataset_count": len(dataset_summaries),
            "all_complete": all_complete,
            "macro": {
                "dense": metric_macro(dataset_summaries, "dense"),
                "q8": metric_macro(dataset_summaries, "q8"),
                "q4": metric_macro(dataset_summaries, "q4"),
            },
            "identity_consistent": identity_consistent,
            "identity_checked_manifest_count": present_manifest_count,
            "identity_mismatched_datasets": identity_mismatches,
            "blockers": blockers,
            "quality_claim": False,
            "default_alias_changed": False,
            "promotion_recommendation": "review" if all_complete and identity_consistent else "defer",
        },
    }


def format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def tsv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = summary["aggregate"]
    for dataset in summary["datasets"]:
        manifest = dataset.get("vector_manifest") if isinstance(dataset.get("vector_manifest"), dict) else {}
        for storage in ("dense", "q8", "q4"):
            metrics = dataset.get(storage) if isinstance(dataset.get(storage), dict) else {}
            rows.append(
                {
                    "dataset": dataset["dataset"],
                    "storage": storage,
                    "status": dataset["status"],
                    "ndcg_at_10": metrics.get("ndcg_at_10"),
                    "recall_at_100": metrics.get("recall_at_100"),
                    "ndcg_at_10_delta_vs_dense": metrics.get("ndcg_at_10_delta_vs_dense")
                    if storage != "dense"
                    else "",
                    "recall_at_100_delta_vs_dense": metrics.get("recall_at_100_delta_vs_dense")
                    if storage != "dense"
                    else "",
                    "compression_ratio": metrics.get("compression_ratio") if storage != "dense" else "",
                    "documents": manifest.get("documents"),
                    "queries": manifest.get("queries"),
                    "partial_doc_vector_lines": dataset.get("partial_doc_vector_lines"),
                    "present_artifacts": ",".join(dataset.get("present_artifacts", [])),
                    "missing_artifacts": ",".join(dataset.get("missing_artifacts", [])),
                    "identity_match": dataset.get("identity_match"),
                    "all_complete": aggregate["all_complete"],
                    "promotion_recommendation": aggregate["promotion_recommendation"],
                }
            )
    return rows


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = tsv_rows(summary)
    fieldnames = [
        "dataset",
        "storage",
        "status",
        "ndcg_at_10",
        "recall_at_100",
        "ndcg_at_10_delta_vs_dense",
        "recall_at_100_delta_vs_dense",
        "compression_ratio",
        "documents",
        "queries",
        "partial_doc_vector_lines",
        "present_artifacts",
        "missing_artifacts",
        "identity_match",
        "all_complete",
        "promotion_recommendation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_tsv_value(row.get(key)) for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--package-sha256", default=DEFAULT_PACKAGE_SHA256)
    parser.add_argument("--identity-sha256", default=DEFAULT_IDENTITY_SHA256)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--count-partial-vector-lines",
        dest="count_partial_vector_lines",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-count-partial-vector-lines",
        dest="count_partial_vector_lines",
        action="store_false",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_root = Path(args.run_root)
        datasets = parse_datasets(args.datasets)
        output_json = Path(args.output_json) if args.output_json else run_root / "selected-package-gate-summary.json"
        output_tsv = Path(args.output_tsv) if args.output_tsv else run_root / "selected-package-gate-summary.tsv"
        summary = build_summary(
            run_root=run_root,
            datasets=datasets,
            package_sha256=args.package_sha256,
            identity_sha256=args.identity_sha256,
            model=args.model,
            snapshot=args.snapshot,
            count_partial_vector_lines=args.count_partial_vector_lines,
        )
        write_json(output_json, summary)
        write_tsv(output_tsv, summary)
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.require_complete and not summary["aggregate"]["all_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
