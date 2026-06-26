#!/usr/bin/env python3
"""Run pretrained external-vector retrieval frontier exports and evals.

The harness keeps provider work at the existing SentenceTransformers vector-cache
boundary, then calls the Eos dense and TurboQuant vector-cache evaluators.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_PRESETS = ("e5-small-v2", "bge-small-en-v1.5", "all-minilm-l6-v2")
DEFAULT_DATASETS = ("scifact", "nfcorpus", "fiqa")
DEFAULT_BITS = "8,4"
DATASET_DIR_TEMPLATE = "datasets/manta-embed-v1/raw/{dataset}/{dataset}"
OUTPUT_PATH_FLAGS = {"--metrics-json", "--metrics-tsv", "--per-query-jsonl"}


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: list[str]
    log_path: Path


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export pretrained SentenceTransformers retrieval caches and evaluate "
            "dense/q8/q4 storage with Eos vector-cache evaluators."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--presets", default=",".join(DEFAULT_PRESETS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--dataset-dir-template", default=DATASET_DIR_TEMPLATE)
    parser.add_argument("--bits", default=DEFAULT_BITS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--go", default="go")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=None,
        help="Override summary TSV path. Default: <run-root>/summary.tsv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Override summary JSON path. Default: <run-root>/summary.json.",
    )
    return parser


def dataset_dir_for(args: argparse.Namespace, dataset: str) -> Path:
    return args.repo_root / args.dataset_dir_template.format(dataset=dataset)


def qrels_path_for(dataset_dir: Path) -> Path:
    return dataset_dir / "qrels" / "test.tsv"


def cache_dir_for(run_root: Path, preset: str, dataset: str) -> Path:
    return run_root / "vector-caches" / preset / dataset


def metrics_dir_for(run_root: Path, preset: str, dataset: str) -> Path:
    return run_root / "metrics" / preset / dataset


def command_specs(args: argparse.Namespace, run_root: Path) -> list[CommandSpec]:
    specs: list[CommandSpec] = []
    log_dir = run_root / "logs"
    for preset in csv_values(args.presets):
        for dataset in csv_values(args.datasets):
            dataset_dir = dataset_dir_for(args, dataset)
            cache_root = run_root / "vector-caches" / preset
            cache_dir = cache_dir_for(run_root, preset, dataset)
            metrics_dir = metrics_dir_for(run_root, preset, dataset)
            if not args.skip_export:
                argv = [
                    args.python,
                    "scripts/export_retrieval_vectors.py",
                    "--preset",
                    preset,
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-root",
                    str(cache_root),
                    "--dataset-name",
                    dataset,
                    "--qrels",
                    str(qrels_path_for(dataset_dir)),
                    "--batch-size",
                    str(args.batch_size),
                ]
                if args.device:
                    argv.extend(["--device", args.device])
                if args.max_docs > 0:
                    argv.extend(["--max-docs", str(args.max_docs)])
                if args.max_queries > 0:
                    argv.extend(["--max-queries", str(args.max_queries)])
                specs.append(
                    CommandSpec(
                        f"export:{preset}:{dataset}",
                        argv,
                        log_dir / f"{preset}.{dataset}.export.log",
                    )
                )
            if not args.skip_eval:
                dense_json = metrics_dir / "dense.metrics.json"
                tq_json = metrics_dir / "turboquant.metrics.json"
                tq_tsv = metrics_dir / "turboquant.metrics.tsv"
                common = [
                    "--dataset",
                    dataset,
                    "--backend",
                    preset,
                    "--artifact",
                    preset,
                    "--doc-vectors",
                    str(cache_dir / "doc-vectors.jsonl"),
                    "--query-vectors",
                    str(cache_dir / "query-vectors.jsonl"),
                    "--top-k",
                    str(args.top_k),
                ]
                if args.max_docs > 0:
                    common.extend(["--max-docs", str(args.max_docs)])
                if args.max_queries > 0:
                    common.extend(["--max-queries", str(args.max_queries)])
                specs.append(
                    CommandSpec(
                        f"dense:{preset}:{dataset}",
                        [
                            args.go,
                            "run",
                            "./cmd/eos",
                            "eval-retrieval-vectors",
                            *common,
                            "--metrics-json",
                            str(dense_json),
                            str(dataset_dir),
                        ],
                        log_dir / f"{preset}.{dataset}.dense.log",
                    )
                )
                specs.append(
                    CommandSpec(
                        f"turboquant:{preset}:{dataset}",
                        [
                            args.go,
                            "run",
                            "./cmd/eos",
                            "eval-retrieval-vectors-turboquant",
                            *common,
                            "--bits",
                            args.bits,
                            "--metrics-json",
                            str(tq_json),
                            "--metrics-tsv",
                            str(tq_tsv),
                            str(dataset_dir),
                        ],
                        log_dir / f"{preset}.{dataset}.turboquant.log",
                    )
                )
    return specs


def run_command(spec: CommandSpec, cwd: Path) -> CommandResult:
    prepare_command_output_dirs(spec)
    proc = subprocess.run(spec.argv, cwd=cwd, text=True, capture_output=True)
    spec.log_path.write_text(
        "$ " + " ".join(spec.argv) + "\n\n"
        + proc.stdout
        + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def prepare_command_output_dirs(spec: CommandSpec) -> None:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(spec.argv[:-1]):
        if value in OUTPUT_PATH_FLAGS:
            Path(spec.argv[index + 1]).parent.mkdir(parents=True, exist_ok=True)


def quality_values(obj: dict[str, Any]) -> tuple[float | None, float | None]:
    quality = obj.get("quality") or {}
    return quality.get("ndcg_at_10"), quality.get("recall_at_100")


def manifest_values(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_summary(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preset in csv_values(args.presets):
        for dataset in csv_values(args.datasets):
            cache_dir = cache_dir_for(run_root, preset, dataset)
            metrics_dir = metrics_dir_for(run_root, preset, dataset)
            manifest = manifest_values(cache_dir / "manifest.json")
            dense_path = metrics_dir / "dense.metrics.json"
            tq_path = metrics_dir / "turboquant.metrics.json"
            dense_metric = json.loads(dense_path.read_text(encoding="utf-8")) if dense_path.is_file() else {}
            tq_metric = json.loads(tq_path.read_text(encoding="utf-8")) if tq_path.is_file() else {}
            output_dim = manifest.get("output_dim")
            doc_rows = manifest.get("document_vector_rows")
            dense_vector_bytes = None
            if isinstance(output_dim, int) and isinstance(doc_rows, int):
                dense_vector_bytes = output_dim * doc_rows * 4
            dense_ndcg, dense_recall = quality_values(dense_metric)
            tq_dense = tq_metric.get("dense") or {}
            if dense_ndcg is None and tq_dense:
                dense_ndcg, dense_recall = quality_values(tq_dense)
            rows.append(
                {
                    "preset": preset,
                    "dataset": dataset,
                    "storage": "dense",
                    "bits": "",
                    "method": "float32",
                    "ndcg_at_10": dense_ndcg,
                    "recall_at_100": dense_recall,
                    "native_dim": manifest.get("native_dim"),
                    "output_dim": output_dim,
                    "vector_bytes": tq_dense.get("vector_bytes", dense_vector_bytes),
                    "dense_vector_bytes": tq_dense.get("vector_bytes", dense_vector_bytes),
                    "compression_ratio": 1.0 if dense_ndcg is not None else None,
                    "metrics_json": str(dense_path) if dense_path.is_file() else "",
                }
            )
            for row in tq_metric.get("rows") or []:
                bits = row.get("bits")
                if bits not in {4, 8}:
                    continue
                ndcg, recall = quality_values(row)
                rows.append(
                    {
                        "preset": preset,
                        "dataset": dataset,
                        "storage": f"q{bits}",
                        "bits": bits,
                        "method": row.get("method"),
                        "ndcg_at_10": ndcg,
                        "recall_at_100": recall,
                        "native_dim": manifest.get("native_dim"),
                        "output_dim": output_dim,
                        "vector_bytes": row.get("vector_bytes"),
                        "dense_vector_bytes": row.get("dense_vector_bytes"),
                        "compression_ratio": row.get("compression_ratio"),
                        "metrics_json": str(tq_path),
                    }
                )
    return rows


def write_outputs(
    args: argparse.Namespace,
    run_root: Path,
    commands: list[dict[str, Any]],
    failed_commands: list[str],
) -> dict[str, Any]:
    summary_tsv = args.summary_tsv or run_root / "summary.tsv"
    summary_json = args.summary_json or run_root / "summary.json"
    rows = collect_summary(args, run_root)
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preset",
        "dataset",
        "storage",
        "bits",
        "method",
        "ndcg_at_10",
        "recall_at_100",
        "native_dim",
        "output_dim",
        "vector_bytes",
        "dense_vector_bytes",
        "compression_ratio",
        "metrics_json",
    ]
    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "quality_claim": False,
        "dry_run": args.dry_run,
        "presets": csv_values(args.presets),
        "datasets": csv_values(args.datasets),
        "bits": csv_values(args.bits),
        "commands": commands,
        "failed_commands": failed_commands,
        "summary_rows": rows,
        "summary_json": str(summary_json),
        "summary_tsv": str(summary_tsv),
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def execute(
    args: argparse.Namespace,
    runner: Callable[[CommandSpec, Path], CommandResult] = run_command,
) -> dict[str, Any]:
    run_root = args.run_root or args.repo_root / "runs" / f"pretrained-retrieval-frontier-{timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    failed: list[str] = []
    for spec in command_specs(args, run_root):
        item = {
            "label": spec.label,
            "argv": spec.argv,
            "log_path": str(spec.log_path),
            "executed": False,
            "returncode": None,
        }
        if not args.dry_run:
            prepare_command_output_dirs(spec)
            result = runner(spec, args.repo_root)
            item["executed"] = True
            item["returncode"] = result.returncode
            if result.returncode != 0:
                failed.append(spec.label)
                commands.append(item)
                break
        commands.append(item)
    payload = write_outputs(args, run_root, commands, failed)
    if failed:
        raise SystemExit(f"command failed: {failed[0]}; see logs under {run_root / 'logs'}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_docs < 0 or args.max_queries < 0:
        raise SystemExit("--max-docs and --max-queries must be non-negative")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    payload = execute(args)
    print(f"summary_json: {payload['summary_json']}")
    print(f"summary_tsv: {payload['summary_tsv']}")
    if args.dry_run:
        print("dry_run: commands were planned but not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
