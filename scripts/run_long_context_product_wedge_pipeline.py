#!/usr/bin/env python3
"""Guarded runner for the bounded long-context product-wedge path.

The default mode is a dry-run preflight. Cleanup and eval execution both require
paired confirmation flags. The generated summary preserves the diagnostic
claim boundary with quality_claim=false.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


QUALITY_CLAIM = False
PIPELINE_SCHEMA = "eos.long_context_product_wedge_pipeline.v1"
DEFAULT_DATASETS = "qmsum,2wikimqa"
DEFAULT_RECLAIM_MANIFEST = ".tiller/scratch/codex/eos-reclaim-approved-candidates-v1.manifest.json"
DEFAULT_RUN_ROOT_PREFIX = "runs/eos-current-default-long-context-product-wedge-v1"
DEFAULT_SUMMARY_ROOT = "runs/eos-current-default-long-context-product-wedge-summary-v1"
DEFAULT_MIN_FREE_GB = 20.0
QWEN_CACHE_ROOT = "runs/external-vector-caches/qwen3-0.6b-longembed-real-doc20-128d"
MXBAI_CACHE_ROOT = "runs/external-vector-caches/mxbai-large-longembed-real-doc20-128d"

BASE_EVAL_ENV = {
    "EOS_LC_WEDGE_MAX_DOCS": "20",
    "EOS_LC_WEDGE_MAX_QUERIES": "20",
    "EOS_LC_WEDGE_RETARGET_MAX_SEQ": "4096",
    "EOS_LC_WEDGE_MAX_TOKENS": "4096",
    "EOS_LC_WEDGE_OUTPUT_DIM": "128",
    "EOS_LC_WEDGE_TOKEN_SPAN": "128",
    "EOS_LC_WEDGE_TOKEN_OVERLAP": "32",
    "EOS_LC_WEDGE_BITS": "4",
    "EOS_LC_WEDGE_PARENT_AGGREGATION": "top2-mean",
    "EOS_LC_WEDGE_EXTERNAL_MODE": "cache",
    "EOS_LC_WEDGE_QWEN3_CACHE_ROOT": QWEN_CACHE_ROOT,
    "EOS_LC_WEDGE_MXBAI_CACHE_ROOT": MXBAI_CACHE_ROOT,
}


class PipelineError(ValueError):
    """Raised for invalid guarded pipeline requests."""


@dataclass(frozen=True)
class CommandSpec:
    label: str
    args: list[str]
    env: dict[str, str] | None = None

    def as_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"label": self.label, "args": self.args}
        if self.env:
            result["env"] = dict(sorted(self.env.items()))
        return result


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[CommandSpec, Path], CommandResult]
FreeBytes = Callable[[Path], int]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root_from(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def parse_datasets(value: str) -> list[str]:
    datasets = [part.strip() for part in value.split(",") if part.strip()]
    if not datasets:
        raise PipelineError("--datasets must include at least one dataset")
    return datasets


def min_free_bytes(min_free_gb: float) -> int:
    if min_free_gb < 0:
        raise PipelineError("--min-free-gb must be non-negative")
    return int(min_free_gb * 1024 * 1024 * 1024)


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def default_runner(spec: CommandSpec, cwd: Path) -> CommandResult:
    env = os.environ.copy()
    if spec.env:
        env.update(spec.env)
    completed = subprocess.run(
        spec.args,
        cwd=str(cwd),
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def run_dir_for(prefix: str, dataset: str) -> str:
    return f"{prefix}-{dataset}"


def expected_comparison_paths(prefix: str, datasets: list[str]) -> dict[str, str]:
    return {
        dataset: f"{run_dir_for(prefix, dataset)}/comparison.json"
        for dataset in datasets
    }


def build_reclaim_command(manifest: str, output_json: Path | None) -> CommandSpec:
    summary_path = str(output_json) if output_json else ".tiller/scratch/codex/eos-product-wedge-pipeline-reclaim-summary.json"
    tsv_path = str(Path(summary_path).with_suffix(".tsv"))
    return CommandSpec(
        label="reclaim",
        args=[
            sys.executable,
            "scripts/plan_run_reclaim.py",
            "execute",
            manifest,
            "--apply",
            "--yes-delete-approved-run-artifacts",
            "--output-json",
            summary_path,
            "--output-tsv",
            tsv_path,
        ],
    )


def build_eval_command(dataset: str, run_root_prefix: str) -> CommandSpec:
    env = dict(BASE_EVAL_ENV)
    env.update(
        {
            "EOS_LC_WEDGE_DATASET_NAME": dataset,
            "EOS_LC_WEDGE_DATASET_DIR": f"datasets/longembed-official/{dataset}",
            "EOS_LC_WEDGE_RUN_DIR": run_dir_for(run_root_prefix, dataset),
        }
    )
    return CommandSpec(
        label=f"eval:{dataset}",
        args=["ferrous-wheel", "run", "scripts/eval_eos_long_context_wedge.fw"],
        env=env,
    )


def build_summary_command(
    *,
    datasets: list[str],
    comparison_paths: dict[str, str],
    summary_root: str,
) -> CommandSpec:
    args = [
        sys.executable,
        "scripts/summarize_long_context_wedge_comparisons.py",
        *[f"{dataset}={comparison_paths[dataset]}" for dataset in datasets],
        "--output-json",
        f"{summary_root}/summary.json",
        "--output-tsv",
        f"{summary_root}/summary.tsv",
    ]
    return CommandSpec(label="summary", args=args)


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._-/:=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def command_to_shell(spec: CommandSpec) -> str:
    env = ""
    if spec.env:
        env = " ".join(f"{key}={shell_quote(value)}" for key, value in sorted(spec.env.items())) + " "
    return env + " ".join(shell_quote(arg) for arg in spec.args)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_plan(path: Path, commands: list[CommandSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Command plan only. Review guard flags before copying commands to a shell.",
        "",
    ]
    for spec in commands:
        lines.append(f"# {spec.label}")
        lines.append("# " + command_to_shell(spec))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def command_record(spec: CommandSpec, executed: bool, result: CommandResult | None = None) -> dict[str, Any]:
    record = spec.as_json()
    record["shell"] = command_to_shell(spec)
    record["executed"] = executed
    if result is not None:
        record["returncode"] = result.returncode
        if result.stdout:
            record["stdout_head"] = result.stdout[:2000]
        if result.stderr:
            record["stderr_head"] = result.stderr[:2000]
    return record


def ensure_dual_flags(args: argparse.Namespace) -> None:
    if args.apply_reclaim != args.yes_delete_approved_run_artifacts:
        raise PipelineError(
            "cleanup execution requires both --apply-reclaim and --yes-delete-approved-run-artifacts"
        )
    if args.run_eval != args.yes_run_long_context_eval:
        raise PipelineError(
            "eval execution requires both --run-eval and --yes-run-long-context-eval"
        )


def comparison_files_exist(repo_root: Path, comparison_paths: dict[str, str]) -> bool:
    return all((repo_root / rel_path).is_file() for rel_path in comparison_paths.values())


def execute_pipeline(
    args: argparse.Namespace,
    *,
    runner: Runner = default_runner,
    free_bytes: FreeBytes = disk_free_bytes,
) -> dict[str, Any]:
    ensure_dual_flags(args)
    repo_root = repo_root_from(args.repo_root)
    datasets = parse_datasets(args.datasets)
    threshold = min_free_bytes(args.min_free_gb)
    warnings: list[str] = []
    blockers: list[str] = []

    comparison_paths = expected_comparison_paths(args.run_root_prefix, datasets)
    reclaim_output = None
    if args.output_json:
        reclaim_output = Path(str(args.output_json) + ".reclaim-summary.json")
    reclaim_spec = build_reclaim_command(args.reclaim_manifest, reclaim_output)
    eval_specs = [build_eval_command(dataset, args.run_root_prefix) for dataset in datasets]
    summary_spec = build_summary_command(
        datasets=datasets,
        comparison_paths=comparison_paths,
        summary_root=args.summary_root,
    )
    planned_commands = [reclaim_spec, *eval_specs, summary_spec]

    free_before = free_bytes(repo_root)
    free_after_reclaim = None
    reclaim_result = None
    reclaim_executed = False
    if args.apply_reclaim:
        reclaim_result = runner(reclaim_spec, repo_root)
        reclaim_executed = True
        if reclaim_result.returncode != 0:
            blockers.append("reclaim command failed; eval skipped")
        free_after_reclaim = free_bytes(repo_root)
    else:
        warnings.append("reclaim not applied; dry-run/preflight only")

    free_for_eval = free_after_reclaim if free_after_reclaim is not None else free_before
    eval_blocked = False
    eval_results: list[dict[str, Any]] = []
    if free_for_eval < threshold:
        eval_blocked = True
        if reclaim_executed:
            blockers.append("free disk below threshold after reclaim")
        else:
            blockers.append("free disk below threshold and reclaim was not applied")
    if reclaim_result is not None and reclaim_result.returncode != 0:
        eval_blocked = True

    if args.run_eval and not eval_blocked:
        for spec in eval_specs:
            result = runner(spec, repo_root)
            eval_results.append(command_record(spec, True, result))
            if result.returncode != 0:
                blockers.append(f"{spec.label} command failed")
                break
    else:
        for spec in eval_specs:
            eval_results.append(command_record(spec, False))

    summary_result = None
    summary_executed = False
    summary_decision = "planned"
    comparisons_ready = comparison_files_exist(repo_root, comparison_paths)
    eval_attempted = any(item["executed"] for item in eval_results)
    eval_succeeded = eval_attempted and all(item.get("returncode") == 0 for item in eval_results)
    if args.run_summary:
        if comparisons_ready or eval_succeeded:
            summary_result = runner(summary_spec, repo_root)
            summary_executed = True
            summary_decision = "executed" if summary_result.returncode == 0 else "failed"
            if summary_result.returncode != 0:
                blockers.append("summary command failed")
        else:
            summary_decision = "blocked_missing_expected_comparisons"
            blockers.append("summary requested but expected comparison files are missing")

    return {
        "schema": PIPELINE_SCHEMA,
        "quality_claim": QUALITY_CLAIM,
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "dry_run": not (args.apply_reclaim or args.run_eval or args.run_summary),
        "flags": {
            "apply_reclaim": args.apply_reclaim,
            "yes_delete_approved_run_artifacts": args.yes_delete_approved_run_artifacts,
            "run_eval": args.run_eval,
            "yes_run_long_context_eval": args.yes_run_long_context_eval,
            "run_summary": args.run_summary,
        },
        "free_bytes_before": free_before,
        "free_bytes_after_reclaim": free_after_reclaim,
        "free_bytes_for_eval": free_for_eval,
        "min_free_gb": args.min_free_gb,
        "min_free_bytes": threshold,
        "reclaim_manifest": args.reclaim_manifest,
        "reclaim_command": command_record(reclaim_spec, reclaim_executed, reclaim_result),
        "eval_blocked": eval_blocked,
        "eval_commands": eval_results,
        "expected_comparison_paths": comparison_paths,
        "summary_command": command_record(summary_spec, summary_executed, summary_result),
        "summary_decision": summary_decision,
        "warnings": warnings,
        "blockers": blockers,
        "planned_commands": [spec.as_json() | {"shell": command_to_shell(spec)} for spec in planned_commands],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or run the guarded bounded long-context product-wedge pipeline."
    )
    parser.add_argument("--repo-root", type=Path, help="Repository root. Defaults to detected current repo.")
    parser.add_argument("--reclaim-manifest", default=DEFAULT_RECLAIM_MANIFEST)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--run-root-prefix", default=DEFAULT_RUN_ROOT_PREFIX)
    parser.add_argument("--summary-root", default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--apply-reclaim", action="store_true")
    parser.add_argument("--yes-delete-approved-run-artifacts", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--yes-run-long-context-eval", action="store_true")
    parser.add_argument("--run-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = execute_pipeline(args)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_plan:
        commands = []
        for record in summary["planned_commands"]:
            commands.append(CommandSpec(record["label"], record["args"], record.get("env")))
        write_plan(args.output_plan, commands)
    if not args.output_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "planned long-context product-wedge pipeline: "
            f"datasets={len(summary['expected_comparison_paths'])} "
            f"quality_claim={str(summary['quality_claim']).lower()} "
            f"blockers={len(summary['blockers'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
