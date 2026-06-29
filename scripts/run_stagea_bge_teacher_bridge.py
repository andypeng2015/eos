#!/usr/bin/env python3
"""Run the Stage A MS MARCO imported-BGE teacher scoring bridge.

The harness turns the scratch-proven Stage A -> BEIR -> imported BGE vectors ->
teacher score cache -> guide filter sequence into a reproducible runner. Dry-run
mode writes command metadata only and does not require inputs to exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_ACQUISITION_RUN_ROOT = Path("runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z")
DEFAULT_CORPUS_ROOT = DEFAULT_ACQUISITION_RUN_ROOT / "beir-msmarco-passage-research-only"
DEFAULT_PACKAGE = Path("runs/pretrained-bert-current-hf-parity-v1-20260629T090818Z/bge/bge-small-en-v1.5.imported.mll")
DEFAULT_EOS_BIN = "env GOWORK=off go run ./cmd/eos"
TEACHER_LABEL = "imported_bge_small_en_v1_5"
BGE_PACKAGE_SHA256 = "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763"
BGE_IDENTITY = "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637"
BGE_SNAPSHOT = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_DOC_PREFIX = ""
BGE_POOLING = "cls"
BGE_NORMALIZATION = "l2"


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


@dataclass(frozen=True)
class BridgePaths:
    run_root: Path
    stagea_root: Path
    beir_root: Path
    vectors_root: Path
    artifacts_root: Path
    logs_root: Path
    stagea_rows: Path
    stagea_manifest: Path
    beir_manifest: Path
    vector_manifest: Path
    scored_rows: Path
    score_rows: Path
    scoring_manifest: Path
    guide_filtered_rows: Path
    guide_manifest: Path
    summary_json: Path
    summary_tsv: Path


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def split_command(value: str) -> list[str]:
    parts = value.split()
    if not parts:
        raise SystemExit("--eos-bin must not be empty")
    return parts


def teacher_model_id() -> str:
    return f"{BGE_MODEL}@{BGE_SNAPSHOT}#imported-mll-{BGE_IDENTITY}"


def teacher_config() -> dict[str, Any]:
    return {
        "package_sha256": BGE_PACKAGE_SHA256,
        "identity": BGE_IDENTITY,
        "snapshot": BGE_SNAPSHOT,
        "model": BGE_MODEL,
        "query_prefix": BGE_QUERY_PREFIX,
        "document_prefix": BGE_DOC_PREFIX,
        "pooling": BGE_POOLING,
        "normalization": BGE_NORMALIZATION,
        "score": "cosine",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--acquisition-run-root", type=Path, default=DEFAULT_ACQUISITION_RUN_ROOT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--eos-bin",
        default=DEFAULT_EOS_BIN,
        help="Eos command. Defaults to 'env GOWORK=off go run ./cmd/eos'; explicit binary paths are honored.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split", default="train")
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--candidate-pool-size", type=int, default=8192)
    parser.add_argument("--max-corpus-docs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-emitted-rows", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-guide-filter", action="store_true")
    return parser


def default_label(args: argparse.Namespace) -> str:
    return args.label or f"scale{args.max_rows}"


def resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return args.run_root
    return args.repo_root / "runs" / f"retrieval-stagea-msmarco-bge-teacher-score-bridge-v1-{timestamp()}"


def paths_for(run_root: Path, label: str) -> BridgePaths:
    stagea_root = run_root / "stagea-rows"
    beir_root = run_root / "beir"
    vectors_root = run_root / "vectors"
    artifacts_root = run_root / "artifacts"
    logs_root = run_root / "logs"
    return BridgePaths(
        run_root=run_root,
        stagea_root=stagea_root,
        beir_root=beir_root,
        vectors_root=vectors_root,
        artifacts_root=artifacts_root,
        logs_root=logs_root,
        stagea_rows=stagea_root / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl",
        stagea_manifest=stagea_root / "manifest.json",
        beir_manifest=beir_root / "manifest.json",
        vector_manifest=vectors_root / "manifest.json",
        scored_rows=artifacts_root / f"msmarco-passage.stagea.{label}.bge-teacher-scored.jsonl",
        score_rows=artifacts_root / f"msmarco-passage.stagea.{label}.bge-teacher-score-rows.jsonl",
        scoring_manifest=run_root / "manifest.json",
        guide_filtered_rows=artifacts_root / f"msmarco-passage.stagea.{label}.bge-guide-filtered.jsonl",
        guide_manifest=run_root / "guide-filter-manifest.json",
        summary_json=run_root / "summary.json",
        summary_tsv=run_root / "summary.tsv",
    )


def command_specs(args: argparse.Namespace, paths: BridgePaths, label: str) -> list[CommandSpec]:
    dataset = f"msmarco-stagea-bge-real-v1-{label}"
    specs = [
        CommandSpec(
            "build-stagea-rows",
            [
                args.python,
                "scripts/build_msmarco_stagea_rows.py",
                "--corpus-root",
                str(args.corpus_root),
                "--acquisition-run-root",
                str(args.acquisition_run_root),
                "--output-root",
                str(paths.stagea_root),
                "--manifest",
                str(paths.stagea_manifest),
                "--split",
                args.split,
                "--max-rows",
                str(args.max_rows),
                "--negatives-per-query",
                str(args.negatives_per_query),
                "--candidate-pool-size",
                str(args.candidate_pool_size),
                "--max-corpus-docs",
                str(args.max_corpus_docs),
                "--seed",
                str(args.seed),
            ],
            paths.logs_root / "01-build-stagea-rows.log",
        ),
        CommandSpec(
            "materialize-beir",
            [
                args.python,
                "scripts/materialize_hard_negatives_beir.py",
                "--input-jsonl",
                str(paths.stagea_rows),
                "--output-dir",
                str(paths.beir_root),
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--manifest",
                str(paths.beir_manifest),
            ],
            paths.logs_root / "02-materialize-beir.log",
        ),
        CommandSpec(
            "export-imported-bge-vectors",
            [
                *split_command(args.eos_bin),
                "export-pretrained-bert-retrieval-vectors",
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--qrels",
                str(paths.beir_root / "qrels" / f"{args.split}.tsv"),
                "--query-prefix",
                BGE_QUERY_PREFIX,
                "--document-prefix",
                BGE_DOC_PREFIX,
                "--batch-size",
                str(args.batch_size),
                "--progress-every",
                str(args.progress_every),
                "--manifest-json",
                str(paths.vector_manifest),
                str(args.package),
                str(paths.beir_root),
                str(paths.vectors_root),
            ],
            paths.logs_root / "03-export-imported-bge-vectors.log",
        ),
        CommandSpec(
            "score-teacher-vector-cache",
            [
                args.python,
                "scripts/score_teacher_with_vector_cache.py",
                "--hard-negatives",
                str(paths.stagea_rows),
                "--dataset-dir",
                str(paths.beir_root),
                "--doc-vectors",
                str(paths.vectors_root / "doc-vectors.jsonl"),
                "--query-vectors",
                str(paths.vectors_root / "query-vectors.jsonl"),
                "--output-jsonl",
                str(paths.scored_rows),
                "--scores-jsonl",
                str(paths.score_rows),
                "--manifest",
                str(paths.scoring_manifest),
                "--model-id",
                teacher_model_id(),
                "--dataset",
                dataset,
                "--score",
                "cosine",
            ],
            paths.logs_root / "04-score-teacher-vector-cache.log",
        ),
    ]
    if not args.skip_guide_filter:
        specs.append(
            CommandSpec(
                "build-guide-filter",
                [
                    args.python,
                    "scripts/build_retrieval_teacher_guide_filter.py",
                    "--rows-jsonl",
                    str(paths.scored_rows),
                    "--teacher-cache",
                    f"{TEACHER_LABEL}={paths.score_rows}",
                    "--teacher-model",
                    f"{TEACHER_LABEL}={teacher_model_id()}",
                    "--teacher-config",
                    f"{TEACHER_LABEL}={json.dumps(teacher_config(), sort_keys=True, separators=(',', ':'))}",
                    "--output-jsonl",
                    str(paths.guide_filtered_rows),
                    "--manifest",
                    str(paths.guide_manifest),
                ],
                paths.logs_root / "05-guide-filter.log",
            )
        )
    return specs


def run_command(spec: CommandSpec, cwd: Path) -> CommandResult:
    prepare_command_dirs(spec)
    proc = subprocess.run(spec.argv, cwd=cwd, text=True, capture_output=True)
    spec.log_path.write_text(
        "$ " + " ".join(spec.argv) + "\n\n"
        + proc.stdout
        + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def prepare_command_dirs(spec: CommandSpec) -> None:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def nested_bool(obj: dict[str, Any], *keys: str) -> bool | None:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, bool) else None


def validate_legal_gates(*manifests: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    ok = True
    for index, manifest in enumerate(manifests):
        if not manifest:
            continue
        name = str(manifest.get("schema") or f"manifest_{index}")
        gates = manifest.get("legal_gates")
        split_policy = manifest.get("split_policy")
        legal_accounting = manifest.get("legal_gate_accounting")
        value = None
        if isinstance(gates, dict) and "test_rows_train_allowed" in gates:
            value = gates.get("test_rows_train_allowed")
        elif isinstance(split_policy, dict) and "test_rows_train_allowed" in split_policy:
            value = split_policy.get("test_rows_train_allowed")
        elif isinstance(legal_accounting, dict):
            merged = legal_accounting.get("input_merged_gates")
            if isinstance(merged, dict):
                value = merged.get("test_rows_train_allowed")
        if value is not None:
            checks[f"{name}.test_rows_train_allowed"] = value
            ok = ok and value is False
        preserved = nested_bool(manifest, "legal_gate_accounting", "research_only_preserved")
        if preserved is not None:
            checks[f"{name}.research_only_preserved"] = preserved
            ok = ok and preserved
    checks["passed"] = ok
    return checks


def validate_after(label: str, args: argparse.Namespace, paths: BridgePaths) -> dict[str, Any]:
    if label == "build-stagea-rows":
        manifest = read_json(paths.stagea_manifest)
        rows = int((manifest.get("counts") or {}).get("rows_emitted") or count_jsonl_rows(paths.stagea_rows))
        return {
            "stagea_rows_emitted": rows,
            "min_emitted_rows": args.min_emitted_rows,
            "passed": rows >= args.min_emitted_rows,
            "legal_gates": validate_legal_gates(manifest),
        }
    if label == "export-imported-bge-vectors":
        beir = read_json(paths.beir_manifest)
        vector = read_json(paths.vector_manifest)
        beir_counts = beir.get("counts") or {}
        query_count = int(beir_counts.get("unique_queries") or 0)
        doc_count = int(beir_counts.get("unique_docs") or 0)
        query_vector_rows = int(
            vector.get("query_vector_rows")
            or vector.get("queries")
            or count_jsonl_rows(paths.vectors_root / "query-vectors.jsonl")
        )
        doc_vector_rows = int(
            vector.get("document_vector_rows")
            or vector.get("documents")
            or count_jsonl_rows(paths.vectors_root / "doc-vectors.jsonl")
        )
        return {
            "beir_queries": query_count,
            "beir_docs": doc_count,
            "query_vector_rows": query_vector_rows,
            "doc_vector_rows": doc_vector_rows,
            "passed": query_vector_rows == query_count and doc_vector_rows == doc_count,
        }
    if label == "score-teacher-vector-cache":
        stagea = read_json(paths.stagea_manifest)
        scoring = read_json(paths.scoring_manifest)
        rows = int((stagea.get("counts") or {}).get("rows_emitted") or count_jsonl_rows(paths.stagea_rows))
        coverage = scoring.get("coverage") or {}
        examples_seen = int(coverage.get("examples_seen") or 0)
        examples_scored = int(coverage.get("examples_scored") or 0)
        missing = int(coverage.get("missing_examples") or 0)
        score_rows = int(coverage.get("import_score_rows") or count_jsonl_rows(paths.score_rows))
        expected_score_rows = rows * (1 + args.negatives_per_query)
        return {
            "rows_emitted": rows,
            "examples_seen": examples_seen,
            "examples_scored": examples_scored,
            "missing_examples": missing,
            "import_score_rows": score_rows,
            "expected_import_score_rows": expected_score_rows,
            "passed": examples_seen == rows
            and examples_scored == rows
            and missing == 0
            and score_rows == expected_score_rows,
            "legal_gates": validate_legal_gates(stagea),
        }
    if label == "build-guide-filter":
        guide = read_json(paths.guide_manifest)
        counts = guide.get("counts") or {}
        missing_drop = int(counts.get("missing_score_drop") or 0)
        return {
            "missing_score_drop": missing_drop,
            "passed": missing_drop == 0,
            "legal_gates": validate_legal_gates(guide),
        }
    return {"passed": True}


def write_summary_tsv(path: Path, commands: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "executed", "returncode", "passed"], delimiter="\t")
        writer.writeheader()
        for command in commands:
            validation = command.get("validation") or {}
            writer.writerow(
                {
                    "label": command["label"],
                    "executed": command["executed"],
                    "returncode": command["returncode"],
                    "passed": validation.get("passed"),
                }
            )


def write_summary(
    args: argparse.Namespace,
    paths: BridgePaths,
    label: str,
    commands: list[dict[str, Any]],
    failed_commands: list[str],
) -> dict[str, Any]:
    payload = {
        "schema": "eos.stagea_bge_teacher_bridge_runner.v1",
        "quality_claim": False,
        "training_run": False,
        "dry_run": args.dry_run,
        "label": label,
        "run_root": str(paths.run_root),
        "paths": {
            "stagea_rows": str(paths.stagea_rows),
            "beir": str(paths.beir_root),
            "vectors": str(paths.vectors_root),
            "scored_rows": str(paths.scored_rows),
            "score_rows": str(paths.score_rows),
            "guide_filtered_rows": str(paths.guide_filtered_rows),
            "scoring_manifest": str(paths.scoring_manifest),
            "guide_manifest": str(paths.guide_manifest),
            "summary_json": str(paths.summary_json),
            "summary_tsv": str(paths.summary_tsv),
        },
        "bge_teacher": {
            "label": TEACHER_LABEL,
            "model_id": teacher_model_id(),
            "package": str(args.package),
            **teacher_config(),
        },
        "options": {
            "split": args.split,
            "max_rows": args.max_rows,
            "negatives_per_query": args.negatives_per_query,
            "candidate_pool_size": args.candidate_pool_size,
            "max_corpus_docs": args.max_corpus_docs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "min_emitted_rows": args.min_emitted_rows,
            "progress_every": args.progress_every,
            "skip_guide_filter": args.skip_guide_filter,
        },
        "commands": commands,
        "failed_commands": failed_commands,
    }
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    write_summary_tsv(paths.summary_tsv, commands)
    paths.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def execute(
    args: argparse.Namespace,
    runner: Callable[[CommandSpec, Path], CommandResult] = run_command,
) -> dict[str, Any]:
    label = default_label(args)
    run_root = resolve_run_root(args)
    paths = paths_for(run_root, label)
    run_root.mkdir(parents=True, exist_ok=True)
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.logs_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    failed: list[str] = []
    for spec in command_specs(args, paths, label):
        item: dict[str, Any] = {
            "label": spec.label,
            "argv": spec.argv,
            "log_path": str(spec.log_path),
            "executed": False,
            "returncode": None,
        }
        if not args.dry_run:
            prepare_command_dirs(spec)
            result = runner(spec, args.repo_root)
            item["executed"] = True
            item["returncode"] = result.returncode
            if result.returncode != 0:
                failed.append(spec.label)
                commands.append(item)
                break
            validation = validate_after(spec.label, args, paths)
            item["validation"] = validation
            if not validation.get("passed", False):
                failed.append(spec.label)
                commands.append(item)
                break
        commands.append(item)
    payload = write_summary(args, paths, label, commands, failed)
    if failed:
        raise SystemExit(f"command failed or validation failed: {failed[0]}; see {paths.summary_json}")
    return payload


def validate_args(args: argparse.Namespace) -> None:
    if args.max_rows <= 0:
        raise SystemExit("--max-rows must be positive")
    if args.negatives_per_query <= 0:
        raise SystemExit("--negatives-per-query must be positive")
    if args.candidate_pool_size < args.negatives_per_query:
        raise SystemExit("--candidate-pool-size must be >= --negatives-per-query")
    if args.max_corpus_docs < 0:
        raise SystemExit("--max-corpus-docs must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.min_emitted_rows < 0:
        raise SystemExit("--min-emitted-rows must be non-negative")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    if not args.split.strip() or "/" in args.split or "\\" in args.split:
        raise SystemExit("--split must be a simple split name")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    payload = execute(args)
    print(f"summary_json: {payload['paths']['summary_json']}")
    print(f"summary_tsv: {payload['paths']['summary_tsv']}")
    if args.dry_run:
        print("dry_run: commands were planned but not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
