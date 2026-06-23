#!/usr/bin/env python3
"""Plan a guarded LongEmbed repair candidate run.

This utility is a dependency-free preflight and command-plan generator only. It
validates existing teacher-signal and per-query diagnostic artifacts, then emits
an auditable JSON summary plus an optional commented shell packet. It never
executes training, scoring, cleanup, staging, committing, pushing, or model
artifact mutation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PLAN_SCHEMA = "manta.guarded_longembed_repair_candidate_plan.v1"
TEACHER_MANIFEST_SCHEMA = "manta.longembed_child_cache_teacher_score_bridge.v1"
GAP_SUMMARY_SCHEMA = "eos.long_context_perquery_gap_summary.v1"
GAP_MISS_SCHEMA = "eos.long_context_perquery_gap_summary.v1.candidate"
QUALITY_CLAIM = False
CLAIM_BOUNDARY = (
    "Plan/preflight only for one guarded candidate. This is not benchmark, "
    "promotion, release, or product-quality evidence."
)

DEFAULT_EVAL_JSONL = "datasets/manta-embed-v1/processed/eval.jsonl"
DEFAULT_HARD_EVAL_JSONL = "datasets/manta-embed-v1/processed/hard-eval.jsonl"


class PlanError(ValueError):
    """Raised for unsafe or invalid plan inputs."""


FreeBytes = Callable[[Path], int]


@dataclass(frozen=True)
class CommandPlan:
    label: str
    env: dict[str, str]
    args: list[str]
    unset_env: tuple[str, ...] = ("EOS_BOOTSTRAP_ARTIFACT",)

    def as_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "unset_env": list(self.unset_env),
            "env": dict(sorted(self.env.items())),
            "args": list(self.args),
            "shell": command_to_shell(self),
            "commented_only": True,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return path.resolve()
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


def require_file(repo_root: Path, value: str | Path, label: str) -> Path:
    path = resolve_path(repo_root, value)
    if not path.is_file():
        raise PlanError(f"{label} does not exist or is not a file: {path}")
    return path


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def min_free_bytes(min_free_gb: float) -> int:
    if min_free_gb < 0:
        raise PlanError("--min-free-gb must be non-negative")
    return int(min_free_gb * 1024 * 1024 * 1024)


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PlanError(f"expected JSON object: {path}")
    return data


def known_teacher_audit_path(path: Path) -> bool:
    return "eos-longembed-current-cache-teacher-signal-audit-v1" in path.as_posix()


def validate_teacher_manifest(path: Path, *, expected_rows: int) -> tuple[dict[str, Any], list[str]]:
    manifest = load_json(path)
    warnings: list[str] = []
    if manifest.get("schema") != TEACHER_MANIFEST_SCHEMA:
        raise PlanError(f"{path}: teacher manifest schema must be {TEACHER_MANIFEST_SCHEMA}")
    if manifest.get("quality_claim") is True:
        raise PlanError(f"{path}: teacher manifest quality_claim must not be true")
    if "quality_claim" not in manifest and known_teacher_audit_path(path):
        warnings.append(f"{path}: missing quality_claim accepted for known teacher audit artifact")
    elif manifest.get("quality_claim") is not False:
        raise PlanError(f"{path}: teacher manifest quality_claim must be false")

    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise PlanError(f"{path}: teacher manifest missing coverage object")
    examples_written = coverage.get("examples_written")
    if isinstance(examples_written, int) and examples_written != expected_rows:
        raise PlanError(
            f"{path}: coverage.examples_written={examples_written} does not match rows={expected_rows}"
        )
    dataset_counts = coverage.get("dataset_counts")
    if not isinstance(dataset_counts, dict):
        raise PlanError(f"{path}: teacher manifest missing coverage.dataset_counts")
    return {
        "path": str(path),
        "schema": manifest.get("schema"),
        "quality_claim": manifest.get("quality_claim"),
        "coverage": coverage,
        "datasets": list(manifest.get("datasets") or []),
    }, warnings


def classify_row(row: dict[str, Any]) -> str:
    for key in ("dataset", "source"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("dataset", "source"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return "unknown"


def validate_filtered_jsonl(path: Path, *, min_rows: int, min_scored_rows: int) -> dict[str, Any]:
    row_count = 0
    scored_rows = 0
    max_line_bytes = 0
    by_source: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            line_bytes = len(raw.encode("utf-8"))
            max_line_bytes = max(max_line_bytes, line_bytes)
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PlanError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise PlanError(f"{path}:{line_no}: expected JSON object row")
            if not isinstance(row.get("query"), str) or not row.get("query"):
                raise PlanError(f"{path}:{line_no}: missing non-empty query")
            if not isinstance(row.get("positive"), str) or not row.get("positive"):
                raise PlanError(f"{path}:{line_no}: missing non-empty positive")
            negatives = row.get("negatives")
            if not isinstance(negatives, list):
                raise PlanError(f"{path}:{line_no}: negatives must be a list")
            teacher_scores = row.get("teacher_scores")
            if teacher_scores is not None:
                if not isinstance(teacher_scores, list):
                    raise PlanError(f"{path}:{line_no}: teacher_scores must be a list when present")
                expected = 1 + len(negatives)
                if len(teacher_scores) != expected:
                    raise PlanError(
                        f"{path}:{line_no}: teacher_scores length {len(teacher_scores)} "
                        f"does not match positive+negatives length {expected}"
                    )
                scored_rows += 1
            row_count += 1
            by_source[classify_row(row)] += 1

    if row_count < min_rows:
        raise PlanError(f"{path}: row count {row_count} below required minimum {min_rows}")
    if scored_rows < min_scored_rows:
        raise PlanError(f"{path}: scored row count {scored_rows} below required minimum {min_scored_rows}")
    return {
        "path": str(path),
        "rows": row_count,
        "rows_with_teacher_scores": scored_rows,
        "max_line_bytes": max_line_bytes,
        "by_source_or_dataset": dict(sorted(by_source.items())),
    }


def validate_gap_summary(path: Path, *, min_consensus_misses: int) -> dict[str, Any]:
    summary = load_json(path)
    if summary.get("schema") != GAP_SUMMARY_SCHEMA:
        raise PlanError(f"{path}: gap summary schema must be {GAP_SUMMARY_SCHEMA}")
    if summary.get("quality_claim") is not False:
        raise PlanError(f"{path}: gap summary quality_claim must be false")
    required_ints = (
        "count_external_consensus_misses",
        "count_eos_matches_external",
        "query_count",
        "dataset_count",
    )
    for key in required_ints:
        if not isinstance(summary.get(key), int):
            raise PlanError(f"{path}: gap summary missing integer {key}")
    if summary["count_external_consensus_misses"] < min_consensus_misses:
        raise PlanError(
            f"{path}: count_external_consensus_misses={summary['count_external_consensus_misses']} "
            f"below required minimum {min_consensus_misses}"
        )
    datasets = summary.get("datasets")
    if not isinstance(datasets, list):
        raise PlanError(f"{path}: gap summary datasets must be a list")
    dataset_names: set[str] = set()
    dataset_summaries: list[dict[str, Any]] = []
    for item in datasets:
        if not isinstance(item, dict):
            raise PlanError(f"{path}: dataset summary entries must be objects")
        if item.get("quality_claim") is not False:
            raise PlanError(f"{path}: dataset summary quality_claim must be false")
        name = item.get("dataset")
        if not isinstance(name, str) or not name:
            raise PlanError(f"{path}: dataset summary missing dataset")
        dataset_names.add(name)
        dataset_summaries.append(
            {
                "dataset": name,
                "query_count": item.get("query_count"),
                "count_external_consensus_misses": item.get("count_external_consensus_misses"),
                "count_eos_matches_external": item.get("count_eos_matches_external"),
                "mean_best_external_minus_best_eos_ndcg_at_10": item.get(
                    "mean_best_external_minus_best_eos_ndcg_at_10"
                ),
            }
        )
    for required in ("qmsum", "2wikimqa"):
        if required not in dataset_names:
            raise PlanError(f"{path}: gap summary missing dataset {required}")
    return {
        "path": str(path),
        "schema": summary.get("schema"),
        "quality_claim": summary.get("quality_claim"),
        "query_count": summary["query_count"],
        "dataset_count": summary["dataset_count"],
        "count_external_consensus_misses": summary["count_external_consensus_misses"],
        "count_eos_matches_external": summary["count_eos_matches_external"],
        "mean_best_external_minus_best_eos_ndcg_at_10": summary.get(
            "mean_best_external_minus_best_eos_ndcg_at_10"
        ),
        "parameters": summary.get("parameters") if isinstance(summary.get("parameters"), dict) else {},
        "datasets": dataset_summaries,
    }


def validate_consensus_misses_jsonl(
    path: Path,
    *,
    summary_consensus_misses: int,
) -> dict[str, Any]:
    rows = 0
    by_dataset: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PlanError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise PlanError(f"{path}:{line_no}: expected JSON object row")
            if row.get("quality_claim") is True:
                raise PlanError(f"{path}:{line_no}: consensus miss quality_claim must not be true")
            if row.get("quality_claim") is not False:
                raise PlanError(f"{path}:{line_no}: consensus miss quality_claim must be false")
            if row.get("schema") not in (None, GAP_MISS_SCHEMA):
                raise PlanError(f"{path}:{line_no}: unexpected consensus miss schema {row.get('schema')!r}")
            dataset = row.get("dataset")
            if isinstance(dataset, str) and dataset:
                by_dataset[dataset] += 1
            rows += 1
    if rows > summary_consensus_misses:
        raise PlanError(
            f"{path}: consensus miss rows={rows} exceed summary "
            f"count_external_consensus_misses={summary_consensus_misses}"
        )
    return {
        "path": str(path),
        "rows": rows,
        "summary_count_external_consensus_misses": summary_consensus_misses,
        "matches_summary_count": rows == summary_consensus_misses,
        "row_count_within_summary": rows <= summary_consensus_misses,
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def require_manifest_path(repo_root: Path, manifest: dict[str, Any], dotted: str) -> str:
    current: Any = manifest
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PlanError(f"default manifest missing required path field: {dotted}")
        current = current[part]
    if not isinstance(current, str) or not current:
        raise PlanError(f"default manifest required path field is not a string: {dotted}")
    path = resolve_path(repo_root, current)
    if not path.is_file():
        raise PlanError(f"default manifest path {dotted} does not exist or is not a file: {path}")
    return str(path)


def validate_default_manifest(repo_root: Path, path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    source_release = manifest.get("source_release")
    dense_gate = manifest.get("dense_gate_evidence")
    compact_policy = manifest.get("compact_policy")
    compact_gate = compact_policy.get("gate_evidence") if isinstance(compact_policy, dict) else None
    if not isinstance(source_release, dict):
        raise PlanError("default manifest missing source_release object")
    if not isinstance(dense_gate, dict):
        raise PlanError("default manifest missing dense_gate_evidence object")
    if not isinstance(compact_policy, dict) or not isinstance(compact_gate, dict):
        raise PlanError("default manifest missing compact_policy.gate_evidence object")

    package = require_manifest_path(repo_root, manifest, "source_release.package")
    tokenizer = require_manifest_path(repo_root, manifest, "source_release.tokenizer")
    dense_anchor = require_manifest_path(repo_root, manifest, "dense_gate_evidence.candidate")
    compact_anchor = require_manifest_path(repo_root, manifest, "compact_policy.gate_evidence.candidate")
    compact_baseline_anchor = require_manifest_path(
        repo_root, manifest, "compact_policy.gate_evidence.baseline"
    )
    if compact_policy.get("bits") != 4:
        raise PlanError("default manifest compact_policy.bits must be 4")
    if compact_policy.get("rerank_storage") != "fp16":
        raise PlanError("default manifest compact_policy.rerank_storage must be fp16")
    if compact_policy.get("rerank_overfetch") != 200:
        raise PlanError("default manifest compact_policy.rerank_overfetch must be 200")
    return {
        "path": str(path),
        "asset_id": manifest.get("asset_id"),
        "source_release": {
            "package": package,
            "tokenizer": tokenizer,
            "directory": source_release.get("directory"),
        },
        "dense_gate": {
            "anchor_scoreboard": dense_anchor,
            "baseline_scoreboard": dense_gate.get("baseline"),
            "status": dense_gate.get("status"),
            "checks": dense_gate.get("checks"),
        },
        "compact_gate": {
            "current_compact_comparator": compact_anchor,
            "baseline_scoreboard": compact_baseline_anchor,
            "status": compact_gate.get("status"),
            "checks": compact_gate.get("checks"),
            "profile": compact_policy.get("profile"),
            "bits": compact_policy.get("bits"),
            "rerank_storage": compact_policy.get("rerank_storage"),
            "rerank_overfetch": compact_policy.get("rerank_overfetch"),
        },
    }


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._-/:=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def command_to_shell(plan: CommandPlan) -> str:
    parts = ["env"]
    for name in plan.unset_env:
        parts.extend(["-u", name])
    for key, value in sorted(plan.env.items()):
        parts.append(f"{key}={shell_quote(value)}")
    parts.extend(shell_quote(arg) for arg in plan.args)
    return " ".join(parts)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_plan(path: Path, commands: list[CommandPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# PLAN ONLY: commented commands below are not executed by this script.",
        "# Review every path, guard, compact post-gate, and claim boundary before copying a command.",
        "# This plan is not benchmark, promotion, release, or product-quality evidence.",
        "",
    ]
    for command in commands:
        lines.append(f"# {command.label}")
        for shell_line in command_to_shell(command).splitlines():
            lines.append("# " + shell_line)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_planned_env(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    paths: dict[str, Path],
    default_manifest: dict[str, Any],
) -> dict[str, str]:
    run_dir = resolve_path(repo_root, Path(args.run_root) / args.run_id)
    return {
        "EOS_REPO_ROOT": str(repo_root),
        "EOS_MODEL_NAME": "eos-embed-v1",
        "EOS_GUARD_RUN_ID": args.run_id,
        "EOS_GUARD_RUN_DIR": str(run_dir),
        "EOS_GUARD_CANDIDATE_RUN_ID": args.candidate_run_id,
        "EOS_GUARD_ANCHOR_SCOREBOARD": default_manifest["dense_gate"]["anchor_scoreboard"],
        "EOS_GUARD_DATASETS": "scifact,nfcorpus,fiqa",
        "EOS_GUARD_METRICS": "ndcg_at_10,recall_at_100",
        "EOS_GUARD_CATEGORY": "short_retrieval",
        "EOS_GUARD_BASELINE": "eos",
        "EOS_GUARD_TOLERANCE": "0",
        "EOS_GUARD_FAIL_ON_GATE": "0",
        "EOS_GUARD_ALLOW_DIRTY": "1",
        "EOS_INITIAL_ARTIFACT": default_manifest["source_release"]["package"],
        "EOS_TOKENIZER": default_manifest["source_release"]["tokenizer"],
        "EOS_PACKAGE_TOKENIZER": default_manifest["source_release"]["tokenizer"],
        "EOS_TRAIN_JSONL": str(paths["train_jsonl"]),
        "EOS_PROTECTED_LONGEMBED_EVAL_JSONL": str(paths["eval_jsonl"]),
        "EOS_EVAL_JSONL": str(resolve_path(repo_root, DEFAULT_EVAL_JSONL)),
        "EOS_HARD_EVAL_JSONL": str(resolve_path(repo_root, DEFAULT_HARD_EVAL_JSONL)),
        "EOS_PRETOKENIZE_JSONL": "1",
        "EOS_QUALITY_TARGET": "pairwise",
        "EOS_HARD_NEGATIVE_TRAIN": "1",
        "EOS_HARD_NEGATIVES_PER_QUERY": "5",
        "EOS_EPOCHS": "1",
        "EOS_BATCH_SIZE": "8",
        "EOS_LR": "0.00000005",
        "EOS_CONTRASTIVE_LOSS": "infonce",
        "EOS_TEMPERATURE": "0.05",
        "EOS_TEACHER_LOSS_WEIGHT": "0.05",
        "EOS_TEACHER_TEMPERATURE": "1.5",
        "EOS_TEACHER_SCORE_NORMALIZATION": "example_zscore",
        "EOS_RESTORE_BEST": "false",
        "EOS_SELECT_METRIC": "score_margin",
        "EOS_EVAL_EVERY": "1",
        "EOS_EVAL_EVERY_STEPS": "0",
        "EOS_PATIENCE": "3",
        "EOS_PROGRESS_EVERY": "1",
        "EOS_SKIP_TESTS": "1",
        "EOS_MATRYOSHKA_DIMS": "256",
        "EOS_TURBOQUANT_PREFIX_OBJECTIVES": "256:4=0.05",
        "EOS_TURBOQUANT_PREFIX_SCORE_MODE": "prepared-ip",
        "EOS_TURBOQUANT_PREFIX_SEED": "5581486560434873699",
        "EOS_TRAIN_ENABLE_ACTIVATION_ACCEL": "1",
        "EOS_TRAIN_ENABLE_FAST_GELU": "1",
    }


def decision_for(blockers: list[str]) -> str:
    if not blockers:
        return "ready"
    if blockers == ["free disk below threshold"]:
        return "ready_when_disk_clear"
    return "blocked_by_input_validation"


def generate_plan(
    args: argparse.Namespace,
    *,
    free_bytes: FreeBytes = disk_free_bytes,
) -> dict[str, Any]:
    repo_root = repo_root_from(args.repo_root)
    if not (repo_root / ".git").exists():
        raise PlanError(f"--repo-root must point at a git worktree: {repo_root}")

    paths = {
        "default_manifest": require_file(repo_root, args.default_embedder_manifest, "--default-embedder-manifest"),
        "train_jsonl": require_file(repo_root, args.train_jsonl, "--train-jsonl"),
        "eval_jsonl": require_file(repo_root, args.eval_jsonl, "--eval-jsonl"),
        "train_manifest": require_file(repo_root, args.train_teacher_manifest, "--train-teacher-manifest"),
        "eval_manifest": require_file(repo_root, args.eval_teacher_manifest, "--eval-teacher-manifest"),
        "gap_summary": require_file(repo_root, args.gap_summary_json, "--gap-summary-json"),
        "eval_corpus_jsonl": require_file(repo_root, DEFAULT_EVAL_JSONL, "--default eval jsonl"),
        "hard_eval_jsonl": require_file(repo_root, DEFAULT_HARD_EVAL_JSONL, "--default hard eval jsonl"),
    }
    if args.consensus_misses_jsonl:
        paths["consensus_misses"] = require_file(
            repo_root, args.consensus_misses_jsonl, "--consensus-misses-jsonl"
        )

    default_manifest = validate_default_manifest(repo_root, paths["default_manifest"])
    train_summary = validate_filtered_jsonl(
        paths["train_jsonl"],
        min_rows=args.min_train_rows,
        min_scored_rows=args.min_scored_train_rows,
    )
    eval_summary = validate_filtered_jsonl(
        paths["eval_jsonl"],
        min_rows=args.min_eval_rows,
        min_scored_rows=0,
    )
    train_manifest, train_manifest_warnings = validate_teacher_manifest(
        paths["train_manifest"], expected_rows=train_summary["rows"]
    )
    eval_manifest, eval_manifest_warnings = validate_teacher_manifest(
        paths["eval_manifest"], expected_rows=eval_summary["rows"]
    )
    gap_summary = validate_gap_summary(paths["gap_summary"], min_consensus_misses=args.min_consensus_misses)
    consensus_summary = None
    if "consensus_misses" in paths:
        consensus_summary = validate_consensus_misses_jsonl(
            paths["consensus_misses"],
            summary_consensus_misses=gap_summary["count_external_consensus_misses"],
        )

    threshold = min_free_bytes(args.min_free_gb)
    free = free_bytes(repo_root)
    blockers: list[str] = []
    if free < threshold:
        blockers.append("free disk below threshold")
    warnings = [
        "plan-only dry-run; no training, scoring, cleanup, VCS, export, or eval executed",
        "previous agreement-teacher run failed NFCorpus compact retention; compact post-gate is mandatory",
    ]
    warnings.extend(train_manifest_warnings)
    warnings.extend(eval_manifest_warnings)

    env = build_planned_env(args, repo_root=repo_root, paths=paths, default_manifest=default_manifest)
    command = CommandPlan(
        label="guarded LongEmbed repair candidate",
        env=env,
        args=["ferrous-wheel", "run", "scripts/run_manta_embed_v1_guarded_candidate.fw"],
    )
    files = {key: repo_relative(repo_root, value) for key, value in sorted(paths.items())}

    compact_requirement = {
        "required": True,
        "profile": "q4/fp16/rerank-overfetch=200",
        "bits": 4,
        "rerank_storage": "fp16",
        "rerank_overfetch": 200,
        "comparator_scoreboard": default_manifest["compact_gate"]["current_compact_comparator"],
        "fail_context": "previous agreement-teacher run failed NFCorpus compact retention",
        "must_pass_before_promotion": True,
    }

    return {
        "schema": PLAN_SCHEMA,
        "created_at": utc_now(),
        "quality_claim": QUALITY_CLAIM,
        "dry_run": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "decision": decision_for(blockers),
        "repo_root": str(repo_root),
        "files": files,
        "default_embedder_manifest": default_manifest,
        "teacher_signal": {
            "train": train_summary,
            "eval": eval_summary,
            "train_manifest": train_manifest,
            "eval_manifest": eval_manifest,
        },
        "per_query_gap_diagnosis": {
            "summary": gap_summary,
            "consensus_misses": consensus_summary,
        },
        "run": {
            "run_id": args.run_id,
            "run_root": str(resolve_path(repo_root, args.run_root)),
            "run_dir": str(resolve_path(repo_root, Path(args.run_root) / args.run_id)),
            "candidate_run_id": args.candidate_run_id,
        },
        "disk_free_bytes": free,
        "min_free_gb": args.min_free_gb,
        "min_free_bytes": threshold,
        "compact_post_gate_requirement": compact_requirement,
        "planned_env": dict(sorted(env.items())),
        "planned_shell_command": command_to_shell(command),
        "planned_commands": [command.as_json()],
        "blockers": blockers,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate LongEmbed repair inputs and emit a guarded candidate plan."
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--default-embedder-manifest", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--train-teacher-manifest", required=True)
    parser.add_argument("--eval-teacher-manifest", required=True)
    parser.add_argument("--gap-summary-json", required=True)
    parser.add_argument("--consensus-misses-jsonl")
    parser.add_argument("--run-id", default="eos-current-default-longembed-repair-plan-v1")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--candidate-run-id", default="candidate")
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--min-scored-train-rows", type=int, default=1)
    parser.add_argument("--min-eval-rows", type=int, default=1)
    parser.add_argument("--min-consensus-misses", type=int, default=1)
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-plan", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = generate_plan(args)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_plan:
        commands = [
            CommandPlan(
                record["label"],
                record["env"],
                record["args"],
                tuple(record.get("unset_env", [])),
            )
            for record in summary["planned_commands"]
        ]
        write_plan(args.output_plan, commands)

    if args.output_json:
        print(
            "planned guarded LongEmbed repair candidate: "
            f"train_rows={summary['teacher_signal']['train']['rows']} "
            f"scored_train_rows={summary['teacher_signal']['train']['rows_with_teacher_scores']} "
            f"decision={summary['decision']} blockers={len(summary['blockers'])}"
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
