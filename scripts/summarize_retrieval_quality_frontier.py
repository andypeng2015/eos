#!/usr/bin/env python3
"""Summarize mined retrieval frontier gaps into train-effort priorities.

Inputs are one or more JSONL files produced by
`scripts/mine_retrieval_quality_frontier.py`. The output is a triage artifact:
it ranks buckets where a teacher beats Eos on mined frontier rows, but it is
not benchmark, promotion, or model-quality evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FRONTIER_SCHEMA = "manta.embedding_quality_frontier_mine.v1"
SUMMARY_SCHEMA = "manta.embedding_quality_frontier_priority_summary.v1"
CLAIM_BOUNDARY = (
    "Prioritization/triage only. This summary ranks mined frontier gaps for "
    "train-safe investigation and is not benchmark, promotion, or product-quality evidence."
)
WEAK_TEACHER_HIT_COVERAGE = 0.5


@dataclass
class BucketStats:
    bucket_type: str
    bucket_key: str
    count: int = 0
    ndcg_gap_sum: float = 0.0
    ndcg_gap_max: float = 0.0
    recall_gap_sum: float = 0.0
    recall_gap_max: float = 0.0
    rank_improvement_sum: float = 0.0
    rank_improvement_max: float = 0.0
    teacher_hit_count: int = 0
    negative_recall_gap_count: int = 0
    datasets: set[str] = field(default_factory=set)
    teacher_labels: set[str] = field(default_factory=set)
    eos_labels: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)

    def add(self, row: dict[str, Any], source_path: str) -> None:
        dataset = text(row.get("dataset"))
        teacher_label = text(nested(row, "teacher", "label"))
        eos_label = text(nested(row, "eos", "label"))
        ndcg_gap = number(nested(row, "delta", "teacher_minus_eos_ndcg_at_10"))
        recall_gap = number(nested(row, "delta", "teacher_minus_eos_recall_at_100"))
        rank_improvement = number(nested(row, "delta", "first_relevant_rank_improvement"))

        self.count += 1
        self.ndcg_gap_sum += ndcg_gap
        self.ndcg_gap_max = max(self.ndcg_gap_max, ndcg_gap)
        self.recall_gap_sum += recall_gap
        self.recall_gap_max = max(self.recall_gap_max, recall_gap)
        self.rank_improvement_sum += rank_improvement
        self.rank_improvement_max = max(self.rank_improvement_max, rank_improvement)
        if teacher_has_hit(row):
            self.teacher_hit_count += 1
        if recall_gap < 0:
            self.negative_recall_gap_count += 1
        if dataset:
            self.datasets.add(dataset)
        if teacher_label:
            self.teacher_labels.add(teacher_label)
        if eos_label:
            self.eos_labels.add(eos_label)
        self.source_paths.add(source_path)

    def as_dict(self) -> dict[str, Any]:
        mean_ndcg_gap = self.ndcg_gap_sum / self.count if self.count else 0.0
        mean_recall_gap = self.recall_gap_sum / self.count if self.count else 0.0
        mean_rank_improvement = self.rank_improvement_sum / self.count if self.count else 0.0
        teacher_hit_coverage = self.teacher_hit_count / self.count if self.count else 0.0
        risky_reasons = []
        if self.negative_recall_gap_count:
            risky_reasons.append("negative_recall_gap")
        if teacher_hit_coverage < WEAK_TEACHER_HIT_COVERAGE:
            risky_reasons.append("weak_teacher_hit_coverage")
        return {
            "bucket_type": self.bucket_type,
            "bucket_key": self.bucket_key,
            "count": self.count,
            "mean_teacher_minus_eos_ndcg_at_10": mean_ndcg_gap,
            "max_teacher_minus_eos_ndcg_at_10": self.ndcg_gap_max,
            "mean_teacher_minus_eos_recall_at_100": mean_recall_gap,
            "max_teacher_minus_eos_recall_at_100": self.recall_gap_max,
            "mean_first_relevant_rank_improvement": mean_rank_improvement,
            "max_first_relevant_rank_improvement": self.rank_improvement_max,
            "teacher_hit_count": self.teacher_hit_count,
            "teacher_hit_coverage": teacher_hit_coverage,
            "negative_recall_gap_count": self.negative_recall_gap_count,
            "risky": bool(risky_reasons),
            "risky_reasons": risky_reasons,
            "priority_score": priority_score(
                self.count,
                mean_ndcg_gap,
                mean_recall_gap,
                mean_rank_improvement,
                bool(risky_reasons),
            ),
            "datasets": sorted(self.datasets),
            "teacher_labels": sorted(self.teacher_labels),
            "eos_labels": sorted(self.eos_labels),
            "source_paths": sorted(self.source_paths),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank train-safe quality-gap buckets from mined retrieval frontier JSONL."
    )
    parser.add_argument("frontier_jsonl", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Minimum teacher-minus-Eos nDCG@10 gap for rows included in buckets.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Limit ranked buckets in output. 0 means no limit.",
    )
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "")


def number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def teacher_has_hit(row: dict[str, Any]) -> bool:
    top_relevant = nested(row, "teacher", "top_relevant")
    return isinstance(top_relevant, list) and len(top_relevant) > 0


def priority_score(
    count: int,
    mean_ndcg_gap: float,
    mean_recall_gap: float,
    mean_rank_improvement: float,
    risky: bool,
) -> float:
    count_weight = math.sqrt(max(count, 1))
    recall_component = max(mean_recall_gap, 0.0) * 0.25
    rank_component = max(min(mean_rank_improvement, 100.0), 0.0) / 1000.0
    score = count_weight * mean_ndcg_gap + recall_component + rank_component
    if risky:
        score *= 0.5
    return score


def load_frontier_rows(paths: list[Path], min_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        seen = 0
        used = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                seen += 1
                row = json.loads(line)
                schema = text(row.get("schema"))
                if schema and schema != FRONTIER_SCHEMA:
                    raise ValueError(f"{path}:{line_no}: expected schema {FRONTIER_SCHEMA}, got {schema}")
                gap = number(nested(row, "delta", "teacher_minus_eos_ndcg_at_10"))
                if gap < min_gap:
                    continue
                row = dict(row)
                row["_source_path"] = str(path)
                row["_source_line"] = line_no
                rows.append(row)
                used += 1
        inputs.append({"path": str(path), "rows_seen": seen, "rows_used": used})
    return rows, inputs


def bucket_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    dataset = text(row.get("dataset")) or "unknown"
    teacher_label = text(nested(row, "teacher", "label")) or "unknown"
    eos_label = text(nested(row, "eos", "label")) or "unknown"
    source_path = text(row.get("_source_path")) or "unknown"
    return [
        ("dataset", dataset),
        ("teacher_label", teacher_label),
        ("eos_label", eos_label),
        ("source_path", source_path),
        ("dataset_teacher_eos", f"{dataset}\t{teacher_label}\t{eos_label}"),
        ("dataset_teacher_eos_source", f"{dataset}\t{teacher_label}\t{eos_label}\t{source_path}"),
    ]


def summarize_frontiers(paths: list[Path], min_gap: float = 0.0, top_k: int = 0) -> dict[str, Any]:
    rows, inputs = load_frontier_rows(paths, min_gap)
    buckets: dict[tuple[str, str], BucketStats] = {}
    for row in rows:
        source_path = text(row.get("_source_path"))
        for bucket_type, bucket_key in bucket_keys(row):
            key = (bucket_type, bucket_key)
            if key not in buckets:
                buckets[key] = BucketStats(bucket_type=bucket_type, bucket_key=bucket_key)
            buckets[key].add(row, source_path)

    bucket_rows = [bucket.as_dict() for bucket in buckets.values()]
    bucket_rows.sort(
        key=lambda bucket: (
            -number(bucket["priority_score"]),
            -int(bucket["count"]),
            bucket["bucket_type"],
            bucket["bucket_key"],
        )
    )
    if top_k > 0:
        bucket_rows = bucket_rows[:top_k]

    return {
        "schema": SUMMARY_SCHEMA,
        "manifest": {
            "quality_claim": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_schema": FRONTIER_SCHEMA,
            "min_gap": min_gap,
            "top_k": top_k,
            "weak_teacher_hit_coverage_threshold": WEAK_TEACHER_HIT_COVERAGE,
        },
        "inputs": inputs,
        "rows_seen": sum(item["rows_seen"] for item in inputs),
        "rows_used": sum(item["rows_used"] for item in inputs),
        "bucket_count": len(bucket_rows),
        "buckets": bucket_rows,
    }


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, buckets: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "priority_score",
                "bucket_type",
                "bucket_key",
                "count",
                "mean_teacher_minus_eos_ndcg_at_10",
                "max_teacher_minus_eos_ndcg_at_10",
                "mean_teacher_minus_eos_recall_at_100",
                "max_teacher_minus_eos_recall_at_100",
                "mean_first_relevant_rank_improvement",
                "max_first_relevant_rank_improvement",
                "teacher_hit_count",
                "teacher_hit_coverage",
                "negative_recall_gap_count",
                "risky",
                "risky_reasons",
                "datasets",
                "teacher_labels",
                "eos_labels",
                "source_paths",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for index, bucket in enumerate(buckets, 1):
            row = dict(bucket)
            row["rank"] = index
            for key in ("risky_reasons", "datasets", "teacher_labels", "eos_labels", "source_paths"):
                row[key] = ",".join(str(value) for value in row.get(key, []))
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    summary = summarize_frontiers(args.frontier_jsonl, min_gap=args.min_gap, top_k=args.top_k)
    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_tsv:
        write_tsv(args.output_tsv, summary["buckets"])
    if not args.output_json and not args.output_tsv:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "summarized retrieval quality frontier: "
            f"rows_used={summary['rows_used']} buckets={summary['bucket_count']}"
        )


if __name__ == "__main__":
    main()
