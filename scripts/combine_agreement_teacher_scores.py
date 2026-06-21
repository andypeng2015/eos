#!/usr/bin/env python3
"""Combine scored hard-negative JSONL files by teacher agreement.

The script preserves every base hard-negative example, but only attaches
averaged teacher_scores when every teacher has complete scores and ranks the
labeled positive above all negatives by at least --min-margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TeacherStats:
    label: str
    path: Path
    examples_seen: int = 0
    examples_complete: int = 0
    examples_missing: int = 0
    positive_top1: int = 0
    margin_pass: int = 0
    agreement_pass: int = 0
    candidate_scores: int = 0
    margins: list[float] = field(default_factory=list)
    positive_scores: list[float] = field(default_factory=list)
    negative_scores: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach averaged teacher_scores only where all teachers agree with labels."
    )
    parser.add_argument("--base", required=True, type=Path, help="Base hard-negative JSONL.")
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        metavar="LABEL=SCORED.jsonl",
        help="Teacher-scored hard-negative JSONL. Repeat for each teacher.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Summary manifest path. Default: <output-jsonl>.agreement-manifest.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=None,
        help="Averaged import-compatible score rows for kept examples.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Minimum teacher positive-vs-best-negative margin required per teacher.",
    )
    parser.add_argument(
        "--missing-sample-limit",
        type=int,
        default=30,
        help="Maximum cleared-example diagnostics to include in the manifest.",
    )
    parser.add_argument(
        "--default-source",
        default="",
        help=(
            "Source label to write for output rows that lack source. Rows with an existing "
            "different source are rejected instead of relabeled."
        ),
    )
    return parser.parse_args()


def parse_teacher_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--teacher must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"--teacher has empty label: {value!r}")
        if label in seen:
            raise SystemExit(f"duplicate teacher label: {label!r}")
        seen.add(label)
        specs.append((label, Path(raw_path)))
    if len(specs) < 2:
        raise SystemExit("at least two --teacher LABEL=PATH inputs are required")
    return specs


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in iter_jsonl(path)]


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def example_signature(row: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        stable_text(row.get("source")),
        stable_text(row.get("query")),
        stable_text(row.get("positive")),
        tuple(stable_text(value) for value in row.get("negatives") or []),
    )


def labeled_source(row: dict[str, Any], default_source: str, index: int) -> str:
    source = stable_text(row.get("source"))
    if source == "":
        return default_source
    if default_source and source != default_source:
        raise SystemExit(
            f"base line {index + 1}: existing source {source!r} does not match "
            f"--default-source {default_source!r}"
        )
    return source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def valid_scores(row: dict[str, Any], candidate_count: int) -> list[float] | None:
    raw = row.get("teacher_scores")
    if not isinstance(raw, list) or len(raw) != candidate_count:
        return None
    scores: list[float] = []
    for value in raw:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        scores.append(score)
    return scores


def margin(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    return scores[0] - max(scores[1:])


def add_cleared_sample(
    samples: list[dict[str, Any]],
    limit: int,
    example_index: int,
    source: str,
    reason: str,
    teacher_reasons: dict[str, str],
) -> None:
    if len(samples) >= limit:
        return
    samples.append(
        {
            "example_index": example_index,
            "source": source,
            "reason": reason,
            "teacher_reasons": teacher_reasons,
        }
    )


def main() -> int:
    args = parse_args()
    if args.missing_sample_limit < 0:
        raise SystemExit("--missing-sample-limit must be non-negative")
    if not math.isfinite(args.min_margin):
        raise SystemExit("--min-margin must be finite")
    args.default_source = args.default_source.strip()

    teacher_specs = parse_teacher_specs(args.teacher)
    input_paths = [args.base] + [path for _, path in teacher_specs]
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")

    manifest_path = args.manifest or args.output_jsonl.with_suffix(
        args.output_jsonl.suffix + ".agreement-manifest.json"
    )
    scores_path = args.scores_jsonl
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if scores_path is not None:
        scores_path.parent.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(args.base)
    teacher_rows: dict[str, list[dict[str, Any]]] = {}
    teacher_stats: dict[str, TeacherStats] = {}
    for label, path in teacher_specs:
        rows = load_jsonl(path)
        if len(rows) != len(base_rows):
            raise SystemExit(
                f"{path}: row count {len(rows)} does not match base {len(base_rows)}"
            )
        teacher_rows[label] = rows
        teacher_stats[label] = TeacherStats(label=label, path=path)

    for index, base in enumerate(base_rows):
        base_sig = example_signature(base)
        for label, rows in teacher_rows.items():
            scored_sig = example_signature(rows[index])
            if scored_sig != base_sig:
                raise SystemExit(
                    f"{teacher_stats[label].path}:{index + 1}: example signature does not "
                    f"match base line {index + 1}"
                )

    examples = len(base_rows)
    with_teacher_scores = 0
    cleared_count = 0
    source_counts: dict[str, dict[str, int]] = {}
    all_averaged_scores: list[float] = []
    positive_averaged_scores: list[float] = []
    negative_averaged_scores: list[float] = []
    averaged_margins: list[float] = []
    cleared_samples: list[dict[str, Any]] = []
    import_score_rows = 0

    score_handle = scores_path.open("w", encoding="utf-8") if scores_path is not None else None
    try:
        with args.output_jsonl.open("w", encoding="utf-8") as out_handle:
            for index, base in enumerate(base_rows):
                source = labeled_source(base, args.default_source, index)
                candidates = [str(base.get("positive") or "")] + [
                    str(value or "") for value in base.get("negatives") or []
                ]
                source_bucket = source_counts.setdefault(source, {"examples": 0, "kept": 0, "cleared": 0})
                source_bucket["examples"] += 1

                row_out = dict(base)
                if source:
                    row_out["source"] = source
                row_out.pop("teacher_scores", None)
                per_teacher_scores: dict[str, list[float]] = {}
                teacher_reasons: dict[str, str] = {}
                keep = True
                for label, rows in teacher_rows.items():
                    stats = teacher_stats[label]
                    stats.examples_seen += 1
                    scores = valid_scores(rows[index], len(candidates))
                    if scores is None:
                        stats.examples_missing += 1
                        teacher_reasons[label] = "missing_or_invalid_scores"
                        keep = False
                        continue
                    stats.examples_complete += 1
                    stats.candidate_scores += len(scores)
                    stats.positive_scores.append(scores[0])
                    stats.negative_scores.extend(scores[1:])
                    teacher_margin = margin(scores)
                    stats.margins.append(teacher_margin)
                    if teacher_margin >= 0:
                        stats.positive_top1 += 1
                    if teacher_margin >= args.min_margin:
                        stats.margin_pass += 1
                    else:
                        teacher_reasons[label] = f"margin_below_min:{teacher_margin:.9g}"
                        keep = False
                    per_teacher_scores[label] = scores

                if keep:
                    averaged = [
                        statistics.fmean(per_teacher_scores[label][candidate_index] for label, _ in teacher_specs)
                        for candidate_index in range(len(candidates))
                    ]
                    for label in per_teacher_scores:
                        teacher_stats[label].agreement_pass += 1
                    row_out["teacher_scores"] = averaged
                    with_teacher_scores += 1
                    source_bucket["kept"] += 1
                    all_averaged_scores.extend(averaged)
                    positive_averaged_scores.append(averaged[0])
                    negative_averaged_scores.extend(averaged[1:])
                    averaged_margins.append(margin(averaged))
                    if score_handle is not None:
                        for candidate, score in zip(candidates, averaged):
                            score_handle.write(
                                json.dumps(
                                    {
                                        "source": source,
                                        "query": row_out.get("query", ""),
                                        "candidate": candidate,
                                        "score": score,
                                        "score_scale": "average_cosine",
                                        "teacher_model_id": "+".join(label for label, _ in teacher_specs),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            import_score_rows += 1
                else:
                    cleared_count += 1
                    source_bucket["cleared"] += 1
                    reason = "teacher_agreement_failed"
                    if len(teacher_reasons) == len(teacher_specs):
                        reason = "all_teachers_failed"
                    add_cleared_sample(
                        cleared_samples,
                        args.missing_sample_limit,
                        index,
                        source,
                        reason,
                        teacher_reasons,
                    )

                out_handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
    finally:
        if score_handle is not None:
            score_handle.close()

    teacher_manifest = {}
    for label, stats in teacher_stats.items():
        teacher_manifest[label] = {
            "path": str(stats.path),
            "examples_seen": stats.examples_seen,
            "examples_complete": stats.examples_complete,
            "examples_missing": stats.examples_missing,
            "scored_coverage": stats.examples_complete / examples if examples else 0.0,
            "positive_top1": stats.positive_top1,
            "positive_top1_rate": stats.positive_top1 / stats.examples_complete
            if stats.examples_complete
            else 0.0,
            "margin_pass": stats.margin_pass,
            "margin_pass_rate": stats.margin_pass / stats.examples_complete
            if stats.examples_complete
            else 0.0,
            "agreement_pass": stats.agreement_pass,
            "agreement_pass_rate": stats.agreement_pass / examples if examples else 0.0,
            "candidate_scores": stats.candidate_scores,
            "positive_scores": score_summary(stats.positive_scores),
            "negative_scores": score_summary(stats.negative_scores),
            "margins": score_summary(stats.margins),
        }

    manifest = {
        "schema": "manta.agreement_teacher_scores.v1",
        "base": str(args.base),
        "output_jsonl": str(args.output_jsonl),
        "scores_jsonl": str(scores_path) if scores_path is not None else "",
        "min_margin": args.min_margin,
        "default_source": args.default_source,
        "teachers": teacher_manifest,
        "inputs_sha256": {str(path): sha256_file(path) for path in input_paths},
        "coverage": {
            "examples": examples,
            "with_teacher_scores": with_teacher_scores,
            "cleared_count": cleared_count,
            "agreement_keep_rate": with_teacher_scores / examples if examples else 0.0,
            "import_score_rows": import_score_rows,
            "source_counts": source_counts,
        },
        "averaged_scores": {
            "all": score_summary(all_averaged_scores),
            "positive": score_summary(positive_averaged_scores),
            "negative": score_summary(negative_averaged_scores),
            "margins": score_summary(averaged_margins),
            "positive_top1_rate": sum(1 for value in averaged_margins if value >= 0) / len(averaged_margins)
            if averaged_margins
            else 0.0,
        },
        "caveats": [
            "Rows are matched by line order after validating source/query/positive/negatives signatures.",
            "Examples that fail any teacher coverage or margin gate are preserved without teacher_scores.",
            "Averaged teacher_scores are evidence for guarded candidate training only, not a model-quality claim.",
        ],
        "cleared_samples": cleared_samples,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "combined agreement teacher scores: "
        f"examples={examples} with_teacher_scores={with_teacher_scores} "
        f"cleared={cleared_count} keep_rate={manifest['coverage']['agreement_keep_rate']:.6f}"
    )
    print(f"output_jsonl: {args.output_jsonl}")
    if scores_path is not None:
        print(f"scores_jsonl: {scores_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
