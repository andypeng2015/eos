#!/usr/bin/env python3
"""Plan a guarded frontier-curriculum embed-m candidate run.

This is a preflight and command-plan generator only. It validates that the
frontier curriculum inputs are internally consistent, then emits an auditable
JSON summary and a commented shell plan. It never executes training, scoring,
cleanup, staging, committing, or pushing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PLAN_SCHEMA = "manta.guarded_frontier_curriculum_candidate_plan.v1"
CURRICULUM_MANIFEST_SCHEMA = "manta.embedding_frontier_curriculum.v1.manifest"
QUALITY_CLAIM = False
CLAIM_BOUNDARY = (
    "Training plan/preflight only. This is not benchmark, promotion, or "
    "product-quality evidence."
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
    unset_env: tuple[str, ...] = ("EOS_BOOTSTRAP_ARTIFACT", "EOS_TRAIN_ENABLE_FAST_GELU")

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


def count_jsonl_rows(path: Path) -> tuple[int, dict[str, float | None]]:
    rows = 0
    recall_gaps: list[float] = []
    ndcg_gaps: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise PlanError(f"{path}:{line_no}: expected JSON object row")
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                recall_gap = metadata.get("frontier_curriculum_teacher_minus_eos_recall_at_100")
                ndcg_gap = metadata.get("frontier_curriculum_teacher_minus_eos_ndcg_at_10")
                if isinstance(recall_gap, (int, float)):
                    recall_gaps.append(float(recall_gap))
                if isinstance(ndcg_gap, (int, float)):
                    ndcg_gaps.append(float(ndcg_gap))
    return rows, {
        "selected_teacher_minus_eos_recall_at_100_min": min(recall_gaps) if recall_gaps else None,
        "selected_teacher_minus_eos_recall_at_100_mean": mean(recall_gaps),
        "selected_teacher_minus_eos_recall_at_100_max": max(recall_gaps) if recall_gaps else None,
        "selected_teacher_minus_eos_ndcg_at_10_min": min(ndcg_gaps) if ndcg_gaps else None,
        "selected_teacher_minus_eos_ndcg_at_10_mean": mean(ndcg_gaps),
        "selected_teacher_minus_eos_ndcg_at_10_max": max(ndcg_gaps) if ndcg_gaps else None,
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise PlanError(f"{label} must be numeric")
    return float(value)


def validate_manifest(
    manifest: dict[str, Any],
    *,
    actual_rows: int,
    min_rows: int,
    max_rows: int,
    min_selected_recall_gap: float,
) -> dict[str, Any]:
    if manifest.get("schema") != CURRICULUM_MANIFEST_SCHEMA:
        raise PlanError(
            f"curriculum manifest schema must be {CURRICULUM_MANIFEST_SCHEMA}"
        )
    if manifest.get("quality_claim") is not False:
        raise PlanError("curriculum manifest quality_claim must be false")
    output_row_count = manifest.get("output_row_count")
    if not isinstance(output_row_count, int):
        raise PlanError("curriculum manifest output_row_count must be an integer")
    if output_row_count != actual_rows:
        raise PlanError(
            f"curriculum manifest output_row_count={output_row_count} "
            f"does not match actual JSONL rows={actual_rows}"
        )
    if output_row_count < min_rows or output_row_count > max_rows:
        raise PlanError(
            f"curriculum output_row_count={output_row_count} outside "
            f"--min-rows/--max-rows range {min_rows}..{max_rows}"
        )
    recall_stats = manifest.get("selected_teacher_minus_eos_recall_at_100")
    if not isinstance(recall_stats, dict):
        raise PlanError("curriculum manifest missing selected recall-gap stats")
    recall_min = number(recall_stats.get("min"), "selected recall-gap min")
    if recall_min < min_selected_recall_gap:
        raise PlanError(
            f"selected recall-gap min {recall_min} is below "
            f"--min-selected-recall-gap {min_selected_recall_gap}"
        )
    ndcg_stats = manifest.get("selected_teacher_minus_eos_ndcg_at_10")
    if not isinstance(ndcg_stats, dict):
        ndcg_stats = {}
    return {
        "row_count": output_row_count,
        "selected_teacher_minus_eos_recall_at_100": dict(recall_stats),
        "selected_teacher_minus_eos_ndcg_at_10": dict(ndcg_stats),
    }


def validate_jsonl_metadata_gap(
    jsonl_gap_stats: dict[str, float | None],
    *,
    min_selected_recall_gap: float,
) -> None:
    recall_min = jsonl_gap_stats.get("selected_teacher_minus_eos_recall_at_100_min")
    if recall_min is None:
        return
    if recall_min < min_selected_recall_gap:
        raise PlanError(
            f"selected JSONL metadata recall-gap min {recall_min} is below "
            f"--min-selected-recall-gap {min_selected_recall_gap}"
        )


def build_planned_env(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, str]:
    run_dir = resolve_path(paths["repo_root"], Path(args.run_root) / args.run_id)
    env = {
        "EOS_REPO_ROOT": str(paths["repo_root"]),
        "EOS_MODEL_NAME": args.model_name,
        "EOS_INITIAL_ARTIFACT": str(paths["initial_artifact"]),
        "EOS_TRAIN_JSONL": str(paths["curriculum_jsonl"]),
        "EOS_EVAL_JSONL": str(paths["eval_jsonl"]),
        "EOS_HARD_EVAL_JSONL": str(paths["hard_eval_jsonl"]),
        "EOS_TOKENIZER": str(paths["tokenizer"]),
        "EOS_PACKAGE_TOKENIZER": str(paths["tokenizer"]),
        "EOS_GUARD_ANCHOR_SCOREBOARD": str(paths["anchor_scoreboard"]),
        "EOS_GUARD_RUN_ID": args.run_id,
        "EOS_GUARD_RUN_DIR": str(run_dir),
        "EOS_GUARD_CANDIDATE_RUN_ID": args.candidate_run_id,
        "EOS_VOCAB_SIZE": str(args.vocab_size),
        "EOS_MAX_SEQ": str(args.max_seq),
        "EOS_EMBEDDING_DIM": str(args.embedding_dim),
        "EOS_HIDDEN_DIM": str(args.hidden_dim),
        "EOS_ENCODER_REPEATS": str(args.encoder_repeats),
        "EOS_BATCH_SIZE": str(args.batch),
        "EOS_EPOCHS": str(args.epochs),
        "EOS_LR": str(args.lr),
        "EOS_CONTRASTIVE_LOSS": args.contrastive_loss,
        "EOS_GROUPED_LOSS_WEIGHT": str(args.grouped_loss_weight),
        "EOS_HARD_NEGATIVE_TRAIN": "1",
        "EOS_HARD_NEGATIVES_PER_QUERY": str(args.hard_negatives_per_query),
        "EOS_TEACHER_LOSS_WEIGHT": str(args.teacher_loss_weight),
        "EOS_QUALITY_TARGET": args.quality_target,
        "EOS_SELECT_METRIC": args.select_metric,
        "EOS_RESTORE_BEST": args.restore_best,
        "EOS_PRETOKENIZE_JSONL": str(args.pretokenize),
        "EOS_EVAL_EVERY": str(args.eval_every),
        "EOS_EVAL_EVERY_STEPS": str(args.eval_every_steps),
        "EOS_GUARD_DATASETS": args.guard_datasets,
        "EOS_GUARD_METRICS": args.guard_metrics,
        "EOS_GUARD_CATEGORY": args.guard_category,
        "EOS_GUARD_BASELINE": args.guard_baseline,
        "EOS_GUARD_TOLERANCE": str(args.guard_tolerance),
        "EOS_GUARD_FAIL_ON_GATE": str(args.guard_fail_on_gate),
    }
    return env


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
        "# Review every path, guard, and claim boundary before copying a command.",
        "# This plan is not benchmark, promotion, or product-quality evidence.",
        "",
    ]
    for command in commands:
        lines.append(f"# {command.label}")
        for shell_line in command_to_shell(command).splitlines():
            lines.append("# " + shell_line)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_plan(
    args: argparse.Namespace,
    *,
    free_bytes: FreeBytes = disk_free_bytes,
) -> dict[str, Any]:
    repo_root = repo_root_from(args.repo_root)
    if not (repo_root / ".git").exists():
        raise PlanError(f"--repo-root must point at a git worktree: {repo_root}")

    paths = {
        "repo_root": repo_root,
        "curriculum_jsonl": require_file(repo_root, args.curriculum_jsonl, "--curriculum-jsonl"),
        "curriculum_manifest": require_file(repo_root, args.curriculum_manifest, "--curriculum-manifest"),
        "initial_artifact": require_file(repo_root, args.initial_artifact, "--initial-artifact"),
        "tokenizer": require_file(repo_root, args.tokenizer, "--tokenizer"),
        "anchor_scoreboard": require_file(repo_root, args.anchor_scoreboard, "--anchor-scoreboard"),
        "eval_jsonl": require_file(repo_root, args.eval_jsonl, "--eval-jsonl"),
        "hard_eval_jsonl": require_file(repo_root, args.hard_eval_jsonl, "--hard-eval-jsonl"),
    }

    actual_rows, jsonl_gap_stats = count_jsonl_rows(paths["curriculum_jsonl"])
    manifest = load_json(paths["curriculum_manifest"])
    manifest_summary = validate_manifest(
        manifest,
        actual_rows=actual_rows,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_selected_recall_gap=args.min_selected_recall_gap,
    )
    validate_jsonl_metadata_gap(
        jsonl_gap_stats,
        min_selected_recall_gap=args.min_selected_recall_gap,
    )

    threshold = min_free_bytes(args.min_free_gb)
    free = free_bytes(repo_root)
    blockers: list[str] = []
    warnings: list[str] = []
    if free < threshold:
        blockers.append("free disk below threshold")
    warnings.append("plan-only dry-run; no training, scoring, cleanup, VCS, or eval executed")

    env = build_planned_env(args, paths)
    command = CommandPlan(
        label="guarded frontier-curriculum candidate",
        env=env,
        args=["ferrous-wheel", "run", "scripts/run_manta_embed_v1_guarded_candidate.fw"],
    )

    files = {key: str(value) for key, value in paths.items() if key != "repo_root"}
    files["repo_root"] = str(repo_root)
    run_dir = resolve_path(repo_root, Path(args.run_root) / args.run_id)

    return {
        "schema": PLAN_SCHEMA,
        "created_at": utc_now(),
        "quality_claim": QUALITY_CLAIM,
        "dry_run": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "repo_root": str(repo_root),
        "files": files,
        "run": {
            "run_id": args.run_id,
            "run_root": str(resolve_path(repo_root, args.run_root)),
            "run_dir": str(run_dir),
            "candidate_run_id": args.candidate_run_id,
            "model_name": args.model_name,
        },
        "row_count": manifest_summary["row_count"],
        "selected_gap_stats": {
            "manifest": {
                "teacher_minus_eos_recall_at_100": manifest_summary[
                    "selected_teacher_minus_eos_recall_at_100"
                ],
                "teacher_minus_eos_ndcg_at_10": manifest_summary[
                    "selected_teacher_minus_eos_ndcg_at_10"
                ],
            },
            "jsonl_metadata": jsonl_gap_stats,
        },
        "disk_free_bytes": free,
        "min_free_gb": args.min_free_gb,
        "min_free_bytes": threshold,
        "planned_env": dict(sorted(env.items())),
        "planned_shell_command": command_to_shell(command),
        "planned_commands": [command.as_json()],
        "blockers": blockers,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate frontier curriculum inputs and emit a guarded embed-m candidate plan."
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--curriculum-jsonl", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--initial-artifact", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--anchor-scoreboard", required=True)
    parser.add_argument("--eval-jsonl", default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--hard-eval-jsonl", default=DEFAULT_HARD_EVAL_JSONL)
    parser.add_argument("--run-id", default="eos-embed-m-frontier-curriculum-guarded-candidate-v1")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--candidate-run-id", default="candidate")
    parser.add_argument("--model-name", default="manta-embed-m")
    parser.add_argument("--vocab-size", default="16384")
    parser.add_argument("--max-seq", default="512")
    parser.add_argument("--embedding-dim", default="192")
    parser.add_argument("--hidden-dim", default="384")
    parser.add_argument("--encoder-repeats", default="3")
    parser.add_argument("--batch", default="16")
    parser.add_argument("--epochs", default="1")
    parser.add_argument("--lr", default="0.0000005")
    parser.add_argument("--contrastive-loss", default="hybrid_infonce")
    parser.add_argument("--grouped-loss-weight", default="0.05")
    parser.add_argument("--hard-negatives-per-query", default="3")
    parser.add_argument("--teacher-loss-weight", default="0")
    parser.add_argument("--quality-target", default="pairwise")
    parser.add_argument("--select-metric", default="mrr")
    parser.add_argument("--restore-best", default="false")
    parser.add_argument("--pretokenize", default="1")
    parser.add_argument("--eval-every", default="999")
    parser.add_argument("--eval-every-steps", default="0")
    parser.add_argument("--guard-datasets", default="scifact,nfcorpus,fiqa")
    parser.add_argument("--guard-metrics", default="ndcg_at_10,recall_at_100")
    parser.add_argument("--guard-category", default="short_retrieval")
    parser.add_argument("--guard-baseline", default="eos")
    parser.add_argument("--guard-tolerance", default="0")
    parser.add_argument("--guard-fail-on-gate", default="1")
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--min-selected-recall-gap", type=float, default=0.0)
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
            "planned guarded frontier-curriculum candidate: "
            f"rows={summary['row_count']} "
            f"quality_claim={str(summary['quality_claim']).lower()} "
            f"dry_run={str(summary['dry_run']).lower()} "
            f"blockers={len(summary['blockers'])}"
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
