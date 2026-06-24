#!/usr/bin/env python3
"""Summarize guarded LongEmbed repair lane post-run evidence.

This is a dependency-free read-only summarizer. It inspects a guarded repair
plan plus the dense and compact guarded runner manifests, then emits an
explicit non-promotion decision. It never executes planned commands or mutates
candidate artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "manta.guarded_longembed_repair_candidate_plan.v1"
SUMMARY_SCHEMA = "manta.guarded_longembed_repair_decision_summary.v1"
QUALITY_CLAIM = False
DENSE_LABEL = "guarded LongEmbed repair candidate"
COMPACT_LABEL = "mandatory compact q4/fp16/rerank-overfetch=200 post-gate"


class DecisionError(ValueError):
    """Raised for invalid plan or manifest evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise DecisionError(f"{label} cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecisionError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DecisionError(f"{label} must be a JSON object: {path}")
    return data


def require_false_quality_claim(data: dict[str, Any], label: str) -> None:
    if data.get("quality_claim") is not False:
        raise DecisionError(f"{label} quality_claim must be false")


def validate_plan(path: Path) -> dict[str, Any]:
    plan = load_json_object(path, "--plan-json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise DecisionError(f"--plan-json schema must be {PLAN_SCHEMA}")
    require_false_quality_claim(plan, "--plan-json")

    commands = plan.get("planned_commands")
    if not isinstance(commands, list):
        raise DecisionError("--plan-json planned_commands must be a list")
    labels = [item.get("label") for item in commands if isinstance(item, dict)]
    missing = [label for label in (DENSE_LABEL, COMPACT_LABEL) if label not in labels]
    if missing:
        raise DecisionError("--plan-json missing planned command label(s): " + ", ".join(missing))

    compact_requirement = plan.get("compact_post_gate_requirement")
    if not isinstance(compact_requirement, dict):
        raise DecisionError("--plan-json missing compact_post_gate_requirement object")
    if compact_requirement.get("required") is not True:
        raise DecisionError("--plan-json compact_post_gate_requirement.required must be true")

    return plan


def compact_command_env(plan: dict[str, Any]) -> dict[str, Any]:
    commands = plan.get("planned_commands")
    if not isinstance(commands, list):
        return {}
    for command in commands:
        if isinstance(command, dict) and command.get("label") == COMPACT_LABEL:
            env = command.get("env")
            return env if isinstance(env, dict) else {}
    return {}


def scalar_or_none(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def require_manifest_text(manifest: dict[str, Any], path: Path, key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise DecisionError(f"{path}: manifest missing non-empty {key}")
    return value


def require_manifest_int(manifest: dict[str, Any], path: Path, key: str) -> int:
    value = manifest.get(key)
    if not isinstance(value, int):
        raise DecisionError(f"{path}: manifest missing integer {key}")
    return value


def summarize_manifest(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {
            "label": label,
            "provided": False,
            "path": None,
            "present": False,
            "accepted": False,
        }
    if not path.is_file():
        raise DecisionError(f"{label} manifest does not exist or is not a file: {path}")
    manifest = load_json_object(path, f"{label} manifest")
    gate_status = require_manifest_text(manifest, path, "gate_status")
    gate_exit_code = require_manifest_int(manifest, path, "gate_exit_code")
    scoreboard_json = require_manifest_text(manifest, path, "scoreboard_json")
    anchor_scoreboard = require_manifest_text(manifest, path, "anchor_scoreboard")
    candidate_sealed = manifest.get("candidate_sealed_artifact")
    if not isinstance(candidate_sealed, str) or not candidate_sealed:
        candidate_sealed = require_manifest_text(manifest, path, "candidate_artifact")
    accepted = gate_status == "accepted" and gate_exit_code == 0
    return {
        "label": label,
        "provided": True,
        "path": str(path),
        "present": True,
        "gate_status": gate_status,
        "gate_exit_code": gate_exit_code,
        "accepted": accepted,
        "scoreboard_json": scoreboard_json,
        "anchor_scoreboard": anchor_scoreboard,
        "candidate_sealed_artifact": candidate_sealed,
        "summary_tsv": scalar_or_none(manifest.get("summary_tsv")),
        "run_dir": scalar_or_none(manifest.get("run_dir")),
        "candidate_dir": scalar_or_none(manifest.get("candidate_dir")),
        "baseline": scalar_or_none(manifest.get("baseline")),
        "method": scalar_or_none(manifest.get("method")),
        "bits": scalar_or_none(manifest.get("bits")),
        "metrics": scalar_or_none(manifest.get("metrics")),
    }


def compact_requirement_summary(plan: dict[str, Any]) -> dict[str, Any]:
    requirement = dict(plan["compact_post_gate_requirement"])
    env = compact_command_env(plan)
    metrics = requirement.get("required_metrics")
    if metrics is None:
        env_metrics = env.get("EOS_GUARD_METRICS")
        if isinstance(env_metrics, str) and env_metrics:
            metrics = [item for item in env_metrics.split(",") if item]
    return {
        "required": requirement.get("required"),
        "must_pass_before_promotion": requirement.get("must_pass_before_promotion"),
        "profile": requirement.get("profile"),
        "comparator_scoreboard": requirement.get("comparator_scoreboard"),
        "required_metrics": metrics,
        "bits": requirement.get("bits"),
        "rerank_storage": requirement.get("rerank_storage"),
        "rerank_overfetch": requirement.get("rerank_overfetch"),
        "fail_context": requirement.get("fail_context"),
        "planned_command_label": COMPACT_LABEL,
        "planned_guard_metrics": env.get("EOS_GUARD_METRICS"),
        "planned_guard_baseline": env.get("EOS_GUARD_BASELINE"),
        "planned_guard_method": env.get("EOS_GUARD_METHOD"),
    }


def decide(dense: dict[str, Any], compact: dict[str, Any]) -> str:
    if not dense.get("provided"):
        return "pending"
    if not dense.get("accepted"):
        return "no_promote_dense_gate_failed"
    if not compact.get("provided"):
        return "pending"
    if not compact.get("accepted"):
        return "no_promote_compact_gate_failed"
    return "short_dense_and_compact_passed_needs_long_context_review"


def summarize_decision(
    *,
    plan_json: Path,
    dense_manifest: Path | None,
    compact_manifest: Path | None,
) -> dict[str, Any]:
    plan = validate_plan(plan_json)
    dense = summarize_manifest(dense_manifest, "dense")
    compact = summarize_manifest(compact_manifest, "compact")
    decision = decide(dense, compact)
    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "quality_claim": QUALITY_CLAIM,
        "decision": decision,
        "promotion_status": "not_promoted",
        "claim_boundary": (
            "Dense and compact short-retrieval gates are not sufficient promotion "
            "or product-quality evidence for the LongEmbed repair lane."
        ),
        "plan": {
            "path": str(plan_json),
            "schema": plan.get("schema"),
            "quality_claim": plan.get("quality_claim"),
            "decision": plan.get("decision"),
            "run": plan.get("run") if isinstance(plan.get("run"), dict) else {},
        },
        "compact_post_gate_requirement": compact_requirement_summary(plan),
        "dense_manifest": dense,
        "compact_manifest": compact,
        "evidence_complete": bool(dense.get("provided") and compact.get("provided")),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("decision", summary["decision"]),
        ("quality_claim", summary["quality_claim"]),
        ("promotion_status", summary["promotion_status"]),
        ("evidence_complete", summary["evidence_complete"]),
        ("plan_json", summary["plan"]["path"]),
        ("compact_profile", summary["compact_post_gate_requirement"].get("profile")),
        (
            "compact_comparator_scoreboard",
            summary["compact_post_gate_requirement"].get("comparator_scoreboard"),
        ),
        (
            "compact_required_metrics",
            summary["compact_post_gate_requirement"].get("required_metrics"),
        ),
    ]
    for prefix in ("dense", "compact"):
        manifest = summary[f"{prefix}_manifest"]
        rows.extend(
            [
                (f"{prefix}_manifest", manifest.get("path")),
                (f"{prefix}_gate_status", manifest.get("gate_status")),
                (f"{prefix}_gate_exit_code", manifest.get("gate_exit_code")),
                (f"{prefix}_accepted", manifest.get("accepted")),
                (f"{prefix}_scoreboard_json", manifest.get("scoreboard_json")),
                (f"{prefix}_anchor_scoreboard", manifest.get("anchor_scoreboard")),
                (f"{prefix}_candidate_sealed_artifact", manifest.get("candidate_sealed_artifact")),
                (f"{prefix}_summary_tsv", manifest.get("summary_tsv")),
            ]
        )
    text = "key\tvalue\n" + "".join(f"{key}\t{tsv_value(value)}\n" for key, value in rows)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--dense-manifest", type=Path)
    parser.add_argument("--compact-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_decision(
            plan_json=args.plan_json,
            dense_manifest=args.dense_manifest,
            compact_manifest=args.compact_manifest,
        )
    except DecisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_tsv:
        write_tsv(args.output_tsv, summary)
    if not args.output_json and not args.output_tsv:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "summarized guarded LongEmbed repair decision: "
            f"decision={summary['decision']} quality_claim=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
