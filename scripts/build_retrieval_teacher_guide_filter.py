#!/usr/bin/env python3
"""Join teacher score caches to retrieval hard-negative rows and guide-filter them.

This is a non-GPU substrate builder. It does not score teacher models. It consumes
runtime-compatible hard-negative rows plus one or more teacher score/cache JSONL
files, then emits rows with complete required teacher coverage and explicit guide
policy metadata for later hard-negative or soft/listwise training.

Minimal teacher cache JSONL schema, candidate-level:

    {"query":"...","candidate":"...","score":0.87}

Recommended fields for stable joins:

    {
      "source":"...",
      "row_id":"...",
      "query_id":"...",
      "candidate_doc_id":"...",
      "candidate_index":0,
      "role":"positive",
      "query":"...",
      "candidate":"...",
      "score":0.87
    }

The loader also accepts runtime exported request-compatible scored rows keyed by
``source/query/candidate`` and optional ``example_index/candidate_index`` fields,
and row-level hard-negative examples with ``teacher_scores``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA = "eos.retrieval_teacher_guide_cache_filter.v1"
ROW_SCHEMA = "eos.embedding_text_hard_negative.teacher_guided.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


@dataclass
class ScoreHit:
    score: float
    key_kind: str


@dataclass
class TeacherCache:
    label: str
    path: Path
    model_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    keys: dict[tuple[Any, ...], float] = field(default_factory=dict)
    counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", required=True, type=Path)
    parser.add_argument(
        "--teacher-cache",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Teacher score/cache JSONL. Repeat for qwen/mxbai/bge/etc.",
    )
    parser.add_argument(
        "--teacher-model",
        action="append",
        default=[],
        metavar="LABEL=MODEL_ID",
        help="Model identity/revision/path provenance for a teacher label.",
    )
    parser.add_argument(
        "--teacher-config",
        action="append",
        default=[],
        metavar="LABEL=JSON",
        help="Additional teacher config JSON object, e.g. prefixes/normalization/device.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument(
        "--ambiguous-epsilon",
        type=float,
        default=0.0,
        help="Rows with no negative above positive but margin <= min-margin become ambiguous_soft_only.",
    )
    parser.add_argument(
        "--ambiguous-policy",
        choices=["emit_soft_only", "drop"],
        default="emit_soft_only",
    )
    parser.add_argument(
        "--conflict-policy",
        choices=["drop", "emit_soft_only"],
        default="drop",
    )
    parser.add_argument("--drop-sample-limit", type=int, default=50)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def norm_text(value: Any) -> str:
    return stable_text(value).casefold()


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


def parse_label_paths(values: list[str], flag: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} must be LABEL=PATH, got {value!r}")
        label, raw = value.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"{flag} has empty label: {value!r}")
        if label in seen:
            raise SystemExit(f"duplicate teacher label: {label}")
        seen.add(label)
        out.append((label, Path(raw)))
    if not out:
        raise SystemExit("at least one --teacher-cache LABEL=PATH input is required")
    return out


def parse_label_values(values: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} must be LABEL=VALUE, got {value!r}")
        label, raw = value.split("=", 1)
        out[label.strip()] = raw.strip()
    return out


def parse_label_json(values: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--teacher-config must be LABEL=JSON, got {value!r}")
        label, raw = value.split("=", 1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--teacher-config {label}: invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit(f"--teacher-config {label}: JSON must be an object")
        out[label.strip()] = parsed
    return out


def finite_score(row: dict[str, Any], path: Path, line_number: int) -> float | None:
    raw = row.get("score", row.get("teacher_score"))
    if raw is None:
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: non-numeric score {raw!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"{path}:{line_number}: non-finite score {raw!r}")
    return score


def add_key(cache: TeacherCache, key: tuple[Any, ...], score: float) -> None:
    if len(key) <= 1:
        return
    existing = cache.keys.get(key)
    if existing is not None:
        cache.counts["duplicate_keys"] += 1
        if score <= existing:
            return
    cache.keys[key] = score


def row_candidate_doc_ids(row: dict[str, Any]) -> list[str]:
    return [stable_text(row.get("positive_doc_id"))] + [
        stable_text(value) for value in row.get("negative_doc_ids") or []
    ]


def add_candidate_score_keys(cache: TeacherCache, row: dict[str, Any], score: float) -> None:
    source = stable_text(row.get("source"))
    query = norm_text(row.get("query"))
    candidate = norm_text(row.get("candidate"))
    row_id = stable_text(row.get("row_id"))
    query_id = stable_text(row.get("query_id"))
    candidate_doc_id = stable_text(row.get("candidate_doc_id") or row.get("doc_id"))
    candidate_index = row.get("candidate_index")
    example_index = row.get("example_index")
    request_id = stable_text(row.get("request_id"))

    if request_id:
        add_key(cache, ("request_id", request_id), score)
    if row_id and candidate_doc_id:
        add_key(cache, ("row_doc", row_id, candidate_doc_id), score)
    if row_id and candidate_index is not None:
        add_key(cache, ("row_index", row_id, int(candidate_index)), score)
    if query_id and candidate_doc_id:
        add_key(cache, ("query_doc", query_id, candidate_doc_id), score)
    if source and query and candidate:
        add_key(cache, ("source_text", source, query, candidate), score)
    if query and candidate:
        add_key(cache, ("text", query, candidate), score)
    if example_index is not None and candidate_index is not None:
        add_key(cache, ("example_index", int(example_index), int(candidate_index)), score)


def add_row_level_scores(cache: TeacherCache, row: dict[str, Any], path: Path, line_number: int) -> bool:
    raw_scores = row.get("teacher_scores") or row.get("scores")
    if not isinstance(raw_scores, list):
        return False
    candidates = [row.get("positive")] + list(row.get("negatives") or [])
    if len(raw_scores) != len(candidates):
        raise ValueError(
            f"{path}:{line_number}: row-level scores length {len(raw_scores)} "
            f"does not match candidates {len(candidates)}"
        )
    doc_ids = row_candidate_doc_ids(row)
    for index, raw in enumerate(raw_scores):
        try:
            score = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric score {raw!r}") from exc
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number}: non-finite score {raw!r}")
        synthetic = {
            "source": row.get("source"),
            "query": row.get("query"),
            "candidate": candidates[index],
            "row_id": row.get("row_id"),
            "query_id": row.get("query_id"),
            "candidate_doc_id": doc_ids[index] if index < len(doc_ids) else "",
            "candidate_index": index,
        }
        add_candidate_score_keys(cache, synthetic, score)
        cache.counts["candidate_scores"] += 1
    cache.counts["row_level_examples"] += 1
    return True


def load_teacher_cache(
    label: str,
    path: Path,
    model_id: str,
    config: dict[str, Any],
) -> TeacherCache:
    if not path.is_file():
        raise SystemExit(f"missing teacher cache for {label}: {path}")
    cache = TeacherCache(label=label, path=path, model_id=model_id, config=config)
    for line_number, row in iter_jsonl(path):
        cache.counts["jsonl_rows"] += 1
        if add_row_level_scores(cache, row, path, line_number):
            continue
        score = finite_score(row, path, line_number)
        if score is None:
            cache.counts["rows_without_score"] += 1
            continue
        add_candidate_score_keys(cache, row, score)
        cache.counts["candidate_scores"] += 1
    cache.counts["join_keys"] = len(cache.keys)
    return cache


def candidate_records(row: dict[str, Any], example_index: int) -> list[dict[str, Any]]:
    candidates = [row.get("positive")] + list(row.get("negatives") or [])
    doc_ids = row_candidate_doc_ids(row)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        records.append(
            {
                "source": stable_text(row.get("source")),
                "row_id": stable_text(row.get("row_id")),
                "query_id": stable_text(row.get("query_id")),
                "query": stable_text(row.get("query")),
                "candidate": stable_text(candidate),
                "candidate_doc_id": doc_ids[index] if index < len(doc_ids) else "",
                "candidate_index": index,
                "role": "positive" if index == 0 else "negative",
                "example_index": example_index,
            }
        )
    return records


def lookup_score(cache: TeacherCache, cand: dict[str, Any]) -> ScoreHit | None:
    key_candidates = [
        ("row_doc", cand["row_id"], cand["candidate_doc_id"]),
        ("row_index", cand["row_id"], cand["candidate_index"]),
        ("query_doc", cand["query_id"], cand["candidate_doc_id"]),
        ("source_text", cand["source"], norm_text(cand["query"]), norm_text(cand["candidate"])),
        ("text", norm_text(cand["query"]), norm_text(cand["candidate"])),
        ("example_index", cand["example_index"], cand["candidate_index"]),
    ]
    for key in key_candidates:
        if key in cache.keys:
            return ScoreHit(cache.keys[key], str(key[0]))
    return None


def margin(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    return scores[0] - max(scores[1:])


def classify_row(per_teacher_scores: dict[str, list[float]], min_margin: float, ambiguous_epsilon: float) -> tuple[str, dict[str, float]]:
    margins = {label: margin(scores) for label, scores in per_teacher_scores.items()}
    if all(value > min_margin for value in margins.values()):
        return "clean_agreement", margins
    if all(value >= -abs(ambiguous_epsilon) for value in margins.values()):
        return "ambiguous_soft_only", margins
    return "conflict", margins


def average_scores(per_teacher_scores: dict[str, list[float]], teacher_order: list[str]) -> list[float]:
    count = len(next(iter(per_teacher_scores.values())))
    return [
        statistics.fmean(per_teacher_scores[label][index] for label in teacher_order)
        for index in range(count)
    ]


def legal_gate_status(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], Counter[str]]:
    counts: Counter[str] = Counter()
    merged = dict(LEGAL_GATES)
    for row in rows:
        gates = row.get("legal_gates") or {}
        if gates != LEGAL_GATES:
            counts["legal_gate_mismatch"] += 1
        for key, expected in LEGAL_GATES.items():
            if gates.get(key) != expected:
                counts[f"{key}_mismatch"] += 1
        if gates:
            merged.update({key: gates.get(key) for key in LEGAL_GATES})
    return merged, counts


def add_drop_sample(samples: list[dict[str, Any]], limit: int, sample: dict[str, Any]) -> None:
    if len(samples) < limit:
        samples.append(sample)


def validate_output_rows(rows: list[dict[str, Any]], required_teachers: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        candidates = [row.get("positive")] + list(row.get("negatives") or [])
        if len(row.get("teacher_scores") or []) != len(candidates):
            counts["teacher_scores_length_mismatch"] += 1
        per_teacher = row.get("teacher_guide", {}).get("per_teacher_scores", {})
        for label in required_teachers:
            if len(per_teacher.get(label) or []) != len(candidates):
                counts["per_teacher_scores_length_mismatch"] += 1
        if row.get("legal_gates") != LEGAL_GATES:
            counts["legal_gate_mismatch"] += 1
        counts["rows_checked"] += 1
    return counts


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.min_margin):
        raise SystemExit("--min-margin must be finite")
    if not math.isfinite(args.ambiguous_epsilon):
        raise SystemExit("--ambiguous-epsilon must be finite")
    if args.drop_sample_limit < 0:
        raise SystemExit("--drop-sample-limit must be non-negative")
    if not args.rows_jsonl.is_file():
        raise SystemExit(f"missing rows JSONL: {args.rows_jsonl}")

    teacher_specs = parse_label_paths(args.teacher_cache, "--teacher-cache")
    model_ids = parse_label_values(args.teacher_model, "--teacher-model")
    configs = parse_label_json(args.teacher_config)
    teacher_order = [label for label, _ in teacher_specs]
    caches = [
        load_teacher_cache(label, path, model_ids.get(label, ""), configs.get(label, {}))
        for label, path in teacher_specs
    ]

    input_rows = [row for _, row in iter_jsonl(args.rows_jsonl)]
    gates, legal_counts = legal_gate_status(input_rows)
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    teacher_coverage: dict[str, Counter[str]] = {cache.label: Counter() for cache in caches}
    key_kind_counts: dict[str, Counter[str]] = {cache.label: Counter() for cache in caches}
    emitted: list[dict[str, Any]] = []
    drop_samples: list[dict[str, Any]] = []
    all_avg_scores: list[float] = []

    for example_index, row in enumerate(input_rows):
        counts["rows_seen"] += 1
        source = stable_text(row.get("source"))
        source_counts[source]["rows_seen"] += 1
        candidates = candidate_records(row, example_index)
        if not candidates or not candidates[0]["candidate"] or not candidates[0]["query"]:
            counts["malformed_row_drop"] += 1
            source_counts[source]["malformed_row_drop"] += 1
            add_drop_sample(drop_samples, args.drop_sample_limit, {"example_index": example_index, "reason": "malformed_row"})
            continue

        per_teacher_scores: dict[str, list[float]] = {}
        missing: dict[str, list[int]] = {}
        for cache in caches:
            scores: list[float] = []
            missing_indices: list[int] = []
            for cand in candidates:
                hit = lookup_score(cache, cand)
                if hit is None:
                    missing_indices.append(int(cand["candidate_index"]))
                    teacher_coverage[cache.label]["missing_candidate_scores"] += 1
                    continue
                scores.append(hit.score)
                teacher_coverage[cache.label]["scored_candidate_hits"] += 1
                key_kind_counts[cache.label][hit.key_kind] += 1
            if missing_indices:
                missing[cache.label] = missing_indices
            else:
                per_teacher_scores[cache.label] = scores
                teacher_coverage[cache.label]["complete_rows"] += 1
        if missing:
            counts["missing_score_drop"] += 1
            source_counts[source]["missing_score_drop"] += 1
            for label, indices in missing.items():
                teacher_coverage[label]["missing_rows"] += 1
                teacher_coverage[label]["missing_indices_total"] += len(indices)
            add_drop_sample(
                drop_samples,
                args.drop_sample_limit,
                {
                    "example_index": example_index,
                    "row_id": row.get("row_id"),
                    "query_id": row.get("query_id"),
                    "reason": "missing_score_drop",
                    "missing": missing,
                },
            )
            continue

        policy, margins = classify_row(per_teacher_scores, args.min_margin, args.ambiguous_epsilon)
        counts[policy] += 1
        source_counts[source][policy] += 1
        for label, value in margins.items():
            if value > args.min_margin:
                teacher_coverage[label]["positive_margin_pass_rows"] += 1
            elif value >= -abs(args.ambiguous_epsilon):
                teacher_coverage[label]["ambiguous_margin_rows"] += 1
            else:
                teacher_coverage[label]["conflict_margin_rows"] += 1

        emit = policy == "clean_agreement"
        if policy == "ambiguous_soft_only" and args.ambiguous_policy == "emit_soft_only":
            emit = True
        if policy == "conflict" and args.conflict_policy == "emit_soft_only":
            emit = True
        if not emit:
            drop_key = f"{policy}_drop"
            counts[drop_key] += 1
            source_counts[source][drop_key] += 1
            add_drop_sample(
                drop_samples,
                args.drop_sample_limit,
                {
                    "example_index": example_index,
                    "row_id": row.get("row_id"),
                    "query_id": row.get("query_id"),
                    "reason": drop_key,
                    "margins": margins,
                },
            )
            continue

        averaged = average_scores(per_teacher_scores, teacher_order)
        all_avg_scores.extend(averaged)
        candidate_policies = []
        for cand_index, cand in enumerate(candidates):
            candidate_policies.append(
                {
                    "candidate_index": cand_index,
                    "role": cand["role"],
                    "candidate_doc_id": cand["candidate_doc_id"],
                    "policy": policy,
                    "teacher_scores": {label: per_teacher_scores[label][cand_index] for label in teacher_order},
                }
            )
        out = dict(row)
        out["schema"] = ROW_SCHEMA
        out["teacher_scores"] = averaged
        out["teacher_guide"] = {
            "schema": f"{SCHEMA}.row_policy",
            "policy": policy,
            "hard_label_allowed": policy == "clean_agreement",
            "soft_only": policy != "clean_agreement",
            "required_teachers": teacher_order,
            "per_teacher_margins": margins,
            "per_teacher_scores": per_teacher_scores,
            "candidate_policies": candidate_policies,
            "missing_score_policy": "strict_drop",
            "conflict_policy": args.conflict_policy,
            "ambiguous_policy": args.ambiguous_policy,
        }
        out["legal_gates"] = dict(LEGAL_GATES)
        emitted.append(out)
        counts["rows_emitted"] += 1
        source_counts[source]["rows_emitted"] += 1

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")
    write_jsonl(args.output_jsonl, emitted)
    validation_counts = validate_output_rows(emitted, teacher_order)
    output_hash = sha256_file(args.output_jsonl)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_retrieval_teacher_guide_filter.py",
        "inputs": {
            "rows_jsonl": str(args.rows_jsonl),
            "rows_jsonl_sha256": sha256_file(args.rows_jsonl),
            "teacher_caches": {
                cache.label: {
                    "path": str(cache.path),
                    "sha256": sha256_file(cache.path),
                    "model_id": cache.model_id,
                    "config": cache.config,
                    "cache_counts": dict(cache.counts),
                }
                for cache in caches
            },
        },
        "outputs": {
            "filtered_jsonl": str(args.output_jsonl),
            "filtered_jsonl_sha256": output_hash,
            "manifest": str(manifest_path),
        },
        "policy": {
            "required_teacher_coverage": "all_candidates_all_teachers",
            "missing_score_policy": "strict_drop",
            "min_margin": args.min_margin,
            "ambiguous_epsilon": args.ambiguous_epsilon,
            "ambiguous_policy": args.ambiguous_policy,
            "conflict_policy": args.conflict_policy,
        },
        "legal_gates": dict(LEGAL_GATES),
        "legal_gate_accounting": {
            "input_merged_gates": gates,
            "counts": dict(legal_counts),
            "research_only_preserved": legal_counts.get("legal_gate_mismatch", 0) == 0,
        },
        "counts": dict(counts),
        "coverage": {
            "teachers": {
                label: dict(counter)
                for label, counter in teacher_coverage.items()
            },
            "join_key_hits": {
                label: dict(counter)
                for label, counter in key_kind_counts.items()
            },
            "source_counts": {source: dict(counter) for source, counter in source_counts.items()},
            "drop_samples": drop_samples,
        },
        "score_summary": {
            "averaged_teacher_scores": {
                "count": len(all_avg_scores),
                "min": min(all_avg_scores) if all_avg_scores else None,
                "max": max(all_avg_scores) if all_avg_scores else None,
                "mean": statistics.fmean(all_avg_scores) if all_avg_scores else None,
            }
        },
        "validation": {
            "counts": dict(validation_counts),
            "no_row_emitted_without_required_scores": validation_counts.get("teacher_scores_length_mismatch", 0) == 0
            and validation_counts.get("per_teacher_scores_length_mismatch", 0) == 0,
            "output_rows_parse": True,
        },
        "schema_notes": {
            "input_row_required_fields": ["query", "positive", "negatives"],
            "input_row_recommended_fields": ["row_id", "query_id", "positive_doc_id", "negative_doc_ids", "legal_gates"],
            "teacher_cache_minimal_candidate_schema": ["query", "candidate", "score"],
            "teacher_cache_recommended_candidate_schema": [
                "source",
                "row_id",
                "query_id",
                "candidate_doc_id",
                "candidate_index",
                "role",
                "query",
                "candidate",
                "score",
            ],
            "request_compatibility": "runtime export-teacher-score-requests rows are supported after a score field is added",
        },
        "quality_claim": False,
    }
    write_json(manifest_path, manifest)
    print(
        "teacher guide filtered rows: "
        f"input={counts.get('rows_seen', 0)} emitted={counts.get('rows_emitted', 0)} "
        f"clean={counts.get('clean_agreement', 0)} ambiguous={counts.get('ambiguous_soft_only', 0)} "
        f"conflict={counts.get('conflict', 0)} missing_drop={counts.get('missing_score_drop', 0)}"
    )
    print(f"output: {args.output_jsonl}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
