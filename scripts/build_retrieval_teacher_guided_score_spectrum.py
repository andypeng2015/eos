#!/usr/bin/env python3
"""Convert teacher-guided retrieval rows to text score-spectrum JSONL.

This adapter is intentionally research-only. It preserves restrictive legal
gates, carries guide-filter provenance, and can keep teacher-disagreement rows
as soft-only supervision without treating them as hard labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "eos.retrieval_teacher_guided_score_spectrum.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-jsonl", required=True, type=Path)
    parser.add_argument("--guide-manifest", type=Path)
    parser.add_argument("--output-full-jsonl", required=True, type=Path)
    parser.add_argument("--output-train-jsonl", required=True, type=Path)
    parser.add_argument("--output-eval-jsonl", required=True, type=Path)
    parser.add_argument("--excluded-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=173)
    parser.add_argument("--softmax-temperature", type=float, default=0.05)
    parser.add_argument("--clean-soft-loss-weight", type=float, default=0.25)
    parser.add_argument("--clean-recovery-loss-weight", type=float, default=0.25)
    parser.add_argument("--exclude-dev-positive-flags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        raise ValueError("teacher_scores are empty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("--softmax-temperature must be finite and positive")
    scaled = [float(score) / temperature for score in scores]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("teacher_scores must be finite")
    max_score = max(scaled)
    exps = [math.exp(value - max_score) for value in scaled]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("softmax total is invalid")
    probs = [value / total for value in exps]
    # Keep serialized probabilities summing exactly enough for strict checks.
    probs[-1] += 1.0 - sum(probs)
    return probs


def has_dev_positive_negative_flag(row: dict[str, Any]) -> bool:
    flags = row.get("candidate_dev_positive_flags") or {}
    if not isinstance(flags, dict):
        return bool(flags)

    def flag_value_truthy(value: Any) -> bool:
        if isinstance(value, dict):
            return any(flag_value_truthy(nested) for nested in value.values())
        return bool(value)

    return any(flag_value_truthy(value) for value in flags.values())


def legal_gates_ok(row: dict[str, Any]) -> bool:
    nested = row.get("legal_gates") or {}
    if nested != LEGAL_GATES:
        return False
    return all(row.get(key) == expected for key, expected in LEGAL_GATES.items())


def row_policy(row: dict[str, Any]) -> str:
    return stable_text((row.get("teacher_guide") or {}).get("policy"))


def candidate_doc_ids(row: dict[str, Any]) -> list[str]:
    return [stable_text(row.get("positive_doc_id"))] + [stable_text(value) for value in row.get("negative_doc_ids") or []]


def candidate_texts(row: dict[str, Any]) -> list[str]:
    return [stable_text(row.get("positive"))] + [stable_text(value) for value in row.get("negatives") or []]


def convert_row(
    row: dict[str, Any],
    line_number: int,
    source_hash: str,
    temperature: float,
    clean_soft_loss_weight: float,
    clean_recovery_loss_weight: float,
) -> dict[str, Any]:
    policy = row_policy(row)
    if policy not in {"clean_agreement", "ambiguous_soft_only", "conflict"}:
        raise ValueError(f"line {line_number}: unsupported teacher_guide policy {policy!r}")

    doc_ids = candidate_doc_ids(row)
    texts = candidate_texts(row)
    scores = [float(value) for value in row.get("teacher_scores") or []]
    if len(doc_ids) != len(texts) or len(scores) != len(texts):
        raise ValueError(
            f"line {line_number}: candidate/text/teacher score length mismatch "
            f"doc_ids={len(doc_ids)} texts={len(texts)} scores={len(scores)}"
        )
    if not texts or not texts[0] or not stable_text(row.get("query")):
        raise ValueError(f"line {line_number}: missing query or positive text")

    probabilities = softmax(scores, temperature)
    is_clean = policy == "clean_agreement"
    selected_positive_index = 0
    out = {
        "row_id": stable_text(row.get("row_id")),
        "source": f"{stable_text(row.get('source'))}:qwen3-mxbai-guide-conflict-soft",
        "query": stable_text(row.get("query")),
        "candidate_doc_ids": doc_ids,
        "candidate_texts": texts,
        "positive_indexes": [0],
        "selected_positive_index": selected_positive_index,
        "hard_negative_eligible": ([False] + [True] * (len(texts) - 1)) if is_clean else [False] * len(texts),
        "target_probabilities": probabilities,
        "hard_loss_weight": 1.0 if is_clean else 0.0,
        "soft_loss_weight": clean_soft_loss_weight if is_clean else 1.0,
        "recovery_loss_weight": clean_recovery_loss_weight if is_clean else 0.0,
        "train_policy": "research_only_qwen3_mxbai_clean_plus_conflict_soft_no_dev_positive_negatives",
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": source_hash,
        "teacher_guide_policy": policy,
        "teacher_guide_soft_only": not is_clean,
        "teacher_guide_hard_label_allowed": is_clean,
        "source_query_id": stable_text(row.get("query_id")),
        "source_positive_doc_id": stable_text(row.get("positive_doc_id")),
        "teacher_score_count": len(scores),
        "teacher_score_min": min(scores),
        "teacher_score_max": max(scores),
        "teacher_score_positive": scores[0],
    }
    teacher_guide = row.get("teacher_guide")
    if isinstance(teacher_guide, dict):
        out["teacher_guide_required_teachers"] = teacher_guide.get("required_teachers") or []
        out["teacher_guide_per_teacher_margins"] = teacher_guide.get("per_teacher_margins") or {}
    return out


def split_rows(rows: list[dict[str, Any]], train_count: int, eval_count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_count < 0 or eval_count < 0:
        raise ValueError("train/eval counts must be non-negative")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    train_queries: set[str] = set()
    eval_queries: set[str] = set()

    for row in shuffled:
        query = row["query"]
        if len(train) < train_count and query not in eval_queries:
            train.append(row)
            train_queries.add(query)
            continue
        if len(eval_rows) < eval_count and query not in train_queries:
            eval_rows.append(row)
            eval_queries.add(query)
        if len(train) >= train_count and len(eval_rows) >= eval_count:
            break
    return train, eval_rows


def validate_score_spectrum(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["rows"] += 1
        n = len(row.get("candidate_texts") or [])
        probs = row.get("target_probabilities") or []
        if len(row.get("candidate_doc_ids") or []) != n:
            counts["candidate_id_length_mismatch"] += 1
        if len(row.get("hard_negative_eligible") or []) != n:
            counts["hard_negative_length_mismatch"] += 1
        if len(probs) != n:
            counts["probability_length_mismatch"] += 1
        if abs(sum(float(value) for value in probs) - 1.0) > 1e-6:
            counts["probability_sum_bad"] += 1
        if row.get("legal_gates") != LEGAL_GATES or any(row.get(key) != expected for key, expected in LEGAL_GATES.items()):
            counts["legal_gate_bad"] += 1
        if row.get("teacher_guide_policy") == "clean_agreement":
            if row.get("selected_positive_index") != 0 or row.get("positive_indexes") != [0]:
                counts["clean_positive_index_bad"] += 1
            if row.get("hard_negative_eligible") != [False] + [True] * (n - 1):
                counts["clean_hard_negative_eligible_bad"] += 1
            if row.get("hard_loss_weight") != 1.0 or row.get("soft_loss_weight") <= 0 or row.get("recovery_loss_weight") <= 0:
                counts["clean_weight_bad"] += 1
        else:
            if any(row.get("hard_negative_eligible") or []):
                counts["soft_only_hard_negative_eligible_bad"] += 1
            if row.get("hard_loss_weight") != 0.0 or row.get("recovery_loss_weight") != 0.0 or row.get("soft_loss_weight") != 1.0:
                counts["soft_only_weight_bad"] += 1
    return counts


def main() -> None:
    args = parse_args()
    if not args.guide_jsonl.is_file():
        raise SystemExit(f"missing guide JSONL: {args.guide_jsonl}")
    if args.guide_manifest and not args.guide_manifest.is_file():
        raise SystemExit(f"missing guide manifest: {args.guide_manifest}")

    source_hash = sha256_file(args.guide_jsonl)
    guide_manifest = json.loads(args.guide_manifest.read_text(encoding="utf-8")) if args.guide_manifest else {}
    counts: Counter[str] = Counter()
    excluded: list[dict[str, Any]] = []
    excluded_samples: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for line_number, row in iter_jsonl(args.guide_jsonl):
        counts["input_rows"] += 1
        policy = row_policy(row)
        counts[f"input_policy_{policy or 'missing'}"] += 1
        if not legal_gates_ok(row):
            counts["legal_gate_bad_input_rows"] += 1
        if has_dev_positive_negative_flag(row):
            counts["dev_positive_flag_rows"] += 1
            if args.exclude_dev_positive_flags:
                excluded_row = {
                    "line": line_number,
                    "row_id": stable_text(row.get("row_id")),
                    "query_id": stable_text(row.get("query_id")),
                    "reason": "dev_positive_negative_flag",
                    "teacher_guide_policy": policy,
                    "candidate_dev_positive_flags": row.get("candidate_dev_positive_flags") or {},
                }
                excluded.append(excluded_row)
                if len(excluded_samples) < args.sample_limit:
                    excluded_samples.append(excluded_row)
                continue
        try:
            eligible.append(
                convert_row(
                    row,
                    line_number,
                    source_hash,
                    args.softmax_temperature,
                    args.clean_soft_loss_weight,
                    args.clean_recovery_loss_weight,
                )
            )
        except ValueError as exc:
            counts["malformed_rows"] += 1
            excluded_row = {
                "line": line_number,
                "row_id": stable_text(row.get("row_id")),
                "query_id": stable_text(row.get("query_id")),
                "reason": "malformed_row",
                "error": str(exc),
                "teacher_guide_policy": policy,
            }
            excluded.append(excluded_row)
            if len(excluded_samples) < args.sample_limit:
                excluded_samples.append(excluded_row)

    train_rows, eval_rows = split_rows(eligible, args.train_count, args.eval_count, args.split_seed)
    train_queries = {row["query"] for row in train_rows}
    eval_queries = {row["query"] for row in eval_rows}
    validation = {
        "full": dict(validate_score_spectrum(eligible)),
        "train": dict(validate_score_spectrum(train_rows)),
        "eval": dict(validate_score_spectrum(eval_rows)),
    }
    policy_counts = Counter(row["teacher_guide_policy"] for row in eligible)
    train_policy_counts = Counter(row["teacher_guide_policy"] for row in train_rows)
    eval_policy_counts = Counter(row["teacher_guide_policy"] for row in eval_rows)
    teacher_score_lengths = [int(row["teacher_score_count"]) for row in eligible]

    write_jsonl(args.output_full_jsonl, eligible)
    write_jsonl(args.output_train_jsonl, train_rows)
    write_jsonl(args.output_eval_jsonl, eval_rows)
    write_jsonl(args.excluded_jsonl, excluded)

    artifact_hashes = {
        "guide_jsonl": source_hash,
        "full_score_spectrum": sha256_file(args.output_full_jsonl),
        "train_score_spectrum": sha256_file(args.output_train_jsonl),
        "eval_score_spectrum": sha256_file(args.output_eval_jsonl),
        "excluded_jsonl": sha256_file(args.excluded_jsonl),
    }
    if args.guide_manifest:
        artifact_hashes["guide_manifest"] = sha256_file(args.guide_manifest)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_retrieval_teacher_guided_score_spectrum.py",
        "objective": "research-only Qwen3/mxbai clean plus conflict-soft score-spectrum planning; no promotion/default/alias movement",
        "inputs": {
            "guide_jsonl": str(args.guide_jsonl),
            "guide_manifest": str(args.guide_manifest) if args.guide_manifest else "",
            "guide_manifest_schema": guide_manifest.get("schema", ""),
            "guide_policy": guide_manifest.get("policy", {}),
            "guide_inputs": guide_manifest.get("inputs", {}),
        },
        "outputs": {
            "full_score_spectrum": str(args.output_full_jsonl),
            "train_score_spectrum": str(args.output_train_jsonl),
            "eval_score_spectrum": str(args.output_eval_jsonl),
            "excluded_jsonl": str(args.excluded_jsonl),
            "manifest": str(args.manifest),
        },
        "source_artifact_hashes": artifact_hashes,
        "legal_gates": dict(LEGAL_GATES),
        "filters": {
            "exclude_dev_positive_negative_flags": args.exclude_dev_positive_flags,
            "soft_only_rows_used": policy_counts.get("ambiguous_soft_only", 0) + policy_counts.get("conflict", 0),
        },
        "score_spectrum": {
            "target_probability_transform": f"softmax(teacher_scores / {args.softmax_temperature} after max-shift)",
            "clean": {
                "hard_loss_weight": 1.0,
                "soft_loss_weight": args.clean_soft_loss_weight,
                "recovery_loss_weight": args.clean_recovery_loss_weight,
                "hard_negative_eligible": "positive false; negatives true",
            },
            "soft_only": {
                "hard_loss_weight": 0.0,
                "soft_loss_weight": 1.0,
                "recovery_loss_weight": 0.0,
                "hard_negative_eligible": "all false",
            },
        },
        "counts": {
            **dict(counts),
            "eligible_rows": len(eligible),
            "excluded_rows": len(excluded),
            "selected_train_rows": len(train_rows),
            "selected_eval_rows": len(eval_rows),
            "query_overlap": len(train_queries & eval_queries),
            "policy_counts": dict(policy_counts),
            "train_policy_counts": dict(train_policy_counts),
            "eval_policy_counts": dict(eval_policy_counts),
        },
        "excluded_samples": excluded_samples,
        "split": {
            "seed": args.split_seed,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "query_overlap": len(train_queries & eval_queries),
        },
        "teacher_score_lengths": {
            "min": min(teacher_score_lengths) if teacher_score_lengths else None,
            "max": max(teacher_score_lengths) if teacher_score_lengths else None,
            "mean": statistics.fmean(teacher_score_lengths) if teacher_score_lengths else None,
        },
        "validation": validation,
        "quality_claim": False,
    }
    write_json(args.manifest, manifest)
    print(
        "teacher-guided score-spectrum rows: "
        f"input={counts.get('input_rows', 0)} eligible={len(eligible)} excluded={len(excluded)} "
        f"train={len(train_rows)} eval={len(eval_rows)} "
        f"policies={dict(policy_counts)} query_overlap={len(train_queries & eval_queries)}"
    )
    print(f"full: {args.output_full_jsonl}")
    print(f"train: {args.output_train_jsonl}")
    print(f"eval: {args.output_eval_jsonl}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
