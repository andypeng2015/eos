#!/usr/bin/env python3
"""Build an auditable retrieval frontier hard-negative curriculum.

The config is JSON from ``--config-json PATH`` or ``--config-json -``.

Example:

  {
    "dedupe": true,
    "max_negatives": 3,
    "quality_claim": false,
    "sources": [
      {
        "name": "fiqa-qwen-primary",
        "role": "primary",
        "priority": 10,
        "cap": 96,
        "min_gap": 0.30,
        "min_recall_gap": 0.0,
        "frontier_jsonl": "runs/.../qwen.frontier.jsonl",
        "hard_negatives_jsonl": "runs/.../qwen.hard-negatives.jsonl"
      }
    ]
  }

The JSONL output preserves Eos hard-negative rows and only enriches metadata.
It is training input selection evidence, not benchmark or promotion evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FRONTIER_SCHEMA = "manta.embedding_quality_frontier_mine.v1"
CURRICULUM_MANIFEST_SCHEMA = "manta.embedding_frontier_curriculum.v1.manifest"
DEFAULT_CLAIM_BOUNDARY = (
    "Training input selection only. This curriculum is not benchmark, promotion, "
    "or product-quality evidence."
)


@dataclass
class SourceConfig:
    name: str
    frontier_jsonl: Path
    hard_negatives_jsonl: Path
    role: str = "primary"
    cap: int | None = None
    min_gap: float = 0.0
    min_recall_gap: float = 0.0
    max_negative_recall_gap_count: int | None = None
    priority: int = 0


@dataclass
class CandidateRow:
    source: SourceConfig
    frontier: dict[str, Any]
    hard_negative: dict[str, Any]
    source_rank: int

    @property
    def dedupe_key(self) -> tuple[str, str]:
        return (
            text(self.hard_negative.get("query")).strip(),
            text(self.hard_negative.get("positive")).strip(),
        )


@dataclass
class SourceStats:
    name: str
    role: str
    priority: int
    frontier_jsonl: str
    hard_negatives_jsonl: str
    rows_seen: int = 0
    hard_negative_rows_seen: int = 0
    joined: int = 0
    eligible: int = 0
    capped: int = 0
    selected: int = 0
    deduped: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "priority": self.priority,
            "frontier_jsonl": self.frontier_jsonl,
            "hard_negatives_jsonl": self.hard_negatives_jsonl,
            "rows_seen": self.rows_seen,
            "hard_negative_rows_seen": self.hard_negative_rows_seen,
            "joined": self.joined,
            "eligible": self.eligible,
            "capped": self.capped,
            "selected": self.selected,
            "deduped": self.deduped,
            "dropped": dict(sorted(self.dropped.items())),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Eos hard-negative curriculum from frontier/HN JSONL inputs."
    )
    parser.add_argument("--config-json", required=True, help="Config JSON path, or '-' for stdin.")
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--summary-tsv", type=Path, default=None)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "")


def number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def teacher_has_hit(frontier: dict[str, Any]) -> bool:
    top_relevant = nested(frontier, "teacher", "top_relevant")
    return isinstance(top_relevant, list) and bool(top_relevant)


def gap(frontier: dict[str, Any]) -> float:
    return number(nested(frontier, "delta", "teacher_minus_eos_ndcg_at_10"))


def recall_gap(frontier: dict[str, Any]) -> float:
    return number(nested(frontier, "delta", "teacher_minus_eos_recall_at_100"))


def rank_improvement(frontier: dict[str, Any]) -> float:
    return number(nested(frontier, "delta", "first_relevant_rank_improvement"))


def load_config(path: str) -> dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"missing config JSON: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_source_config(raw: dict[str, Any], index: int) -> SourceConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"sources[{index}]: expected object")
    missing = [key for key in ("name", "frontier_jsonl", "hard_negatives_jsonl") if not raw.get(key)]
    if missing:
        raise ValueError(f"sources[{index}]: missing required field(s): {', '.join(missing)}")
    cap = raw.get("cap")
    if cap is not None:
        cap = integer(cap, -1)
        if cap < 0:
            raise ValueError(f"sources[{index}]: cap must be non-negative")
    max_negative_recall_gap_count = raw.get("max_negative_recall_gap_count")
    if max_negative_recall_gap_count is not None:
        max_negative_recall_gap_count = integer(max_negative_recall_gap_count, -1)
        if max_negative_recall_gap_count < 0:
            raise ValueError(
                f"sources[{index}]: max_negative_recall_gap_count must be non-negative"
            )
    source = SourceConfig(
        name=text(raw["name"]),
        frontier_jsonl=Path(text(raw["frontier_jsonl"])),
        hard_negatives_jsonl=Path(text(raw["hard_negatives_jsonl"])),
        role=text(raw.get("role")) or "primary",
        cap=cap,
        min_gap=number(raw.get("min_gap")),
        min_recall_gap=number(raw.get("min_recall_gap")),
        max_negative_recall_gap_count=max_negative_recall_gap_count,
        priority=integer(raw.get("priority"), 0),
    )
    if not source.frontier_jsonl.is_file():
        raise ValueError(f"source {source.name}: missing frontier_jsonl: {source.frontier_jsonl}")
    if not source.hard_negatives_jsonl.is_file():
        raise ValueError(
            f"source {source.name}: missing hard_negatives_jsonl: {source.hard_negatives_jsonl}"
        )
    return source


def parse_sources(config: dict[str, Any]) -> list[SourceConfig]:
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("config must contain a non-empty sources list")
    return [parse_source_config(raw, index) for index, raw in enumerate(raw_sources)]


def load_frontier(path: Path, source_name: str) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    seen = 0
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
            qid = text(row.get("query_id"))
            if not qid:
                raise ValueError(f"{path}:{line_no}: missing query_id")
            if qid in rows:
                raise ValueError(f"{path}:{line_no}: duplicate query_id {qid!r} in source {source_name}")
            rows[qid] = row
    return rows, seen


def load_hard_negatives(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            seen += 1
            row = json.loads(line)
            qid = text(nested(row, "metadata", "query_id"))
            if not qid:
                raise ValueError(f"{path}:{line_no}: hard-negative row missing metadata.query_id")
            if qid in rows:
                raise ValueError(f"{path}:{line_no}: duplicate metadata.query_id {qid!r}")
            rows[qid] = row
    return rows, seen


def valid_hard_negative(row: dict[str, Any], max_negatives: int | None) -> bool:
    if not text(row.get("query")).strip():
        return False
    if not text(row.get("positive")).strip():
        return False
    negatives = row.get("negatives")
    if not isinstance(negatives, list):
        return False
    if max_negatives is not None:
        negatives = negatives[:max_negatives]
    return any(text(negative).strip() for negative in negatives)


def sort_candidates(candidates: list[CandidateRow]) -> None:
    candidates.sort(
        key=lambda candidate: (
            -gap(candidate.frontier),
            -recall_gap(candidate.frontier),
            -rank_improvement(candidate.frontier),
            text(candidate.frontier.get("query_id")),
        )
    )


def build_source_candidates(
    source: SourceConfig,
    *,
    require_teacher_hit: bool,
    max_negatives: int | None,
) -> tuple[list[CandidateRow], SourceStats]:
    stats = SourceStats(
        name=source.name,
        role=source.role,
        priority=source.priority,
        frontier_jsonl=str(source.frontier_jsonl),
        hard_negatives_jsonl=str(source.hard_negatives_jsonl),
    )
    frontier_by_qid, stats.rows_seen = load_frontier(source.frontier_jsonl, source.name)
    hard_negative_by_qid, stats.hard_negative_rows_seen = load_hard_negatives(
        source.hard_negatives_jsonl
    )
    candidates: list[CandidateRow] = []
    negative_recall_gap_used = 0
    for qid in sorted(frontier_by_qid):
        frontier = frontier_by_qid[qid]
        hard_negative = hard_negative_by_qid.get(qid)
        if hard_negative is None:
            stats.drop("missing_hard_negative")
            continue
        stats.joined += 1
        if gap(frontier) < source.min_gap:
            stats.drop("below_min_gap")
            continue
        row_recall_gap = recall_gap(frontier)
        if row_recall_gap < source.min_recall_gap:
            stats.drop("below_min_recall_gap")
            continue
        if row_recall_gap < 0:
            if source.max_negative_recall_gap_count is None:
                stats.drop("negative_recall_gap")
                continue
            if negative_recall_gap_used >= source.max_negative_recall_gap_count:
                stats.drop("negative_recall_gap_cap")
                continue
            negative_recall_gap_used += 1
        if require_teacher_hit and not teacher_has_hit(frontier):
            stats.drop("missing_teacher_hit")
            continue
        if not valid_hard_negative(hard_negative, max_negatives):
            stats.drop("invalid_hard_negative_text")
            continue
        candidates.append(CandidateRow(source=source, frontier=frontier, hard_negative=hard_negative, source_rank=0))

    sort_candidates(candidates)
    stats.eligible = len(candidates)
    if source.cap is not None:
        stats.capped = max(0, len(candidates) - source.cap)
        candidates = candidates[: source.cap]
    for index, candidate in enumerate(candidates, 1):
        candidate.source_rank = index
    return candidates, stats


def enriched_hard_negative(candidate: CandidateRow, max_negatives: int | None) -> dict[str, Any]:
    source = candidate.source
    frontier = candidate.frontier
    hard_negative = candidate.hard_negative
    original_metadata = hard_negative.get("metadata")
    if not isinstance(original_metadata, dict):
        original_metadata = {}
    metadata = dict(original_metadata)
    metadata.update(
        {
            "frontier_curriculum_selected_source_name": source.name,
            "frontier_curriculum_selected_source_role": source.role,
            "frontier_curriculum_selected_source_priority": source.priority,
            "frontier_curriculum_selected_source_rank": candidate.source_rank,
            "frontier_curriculum_frontier_jsonl": str(source.frontier_jsonl),
            "frontier_curriculum_hard_negatives_jsonl": str(source.hard_negatives_jsonl),
            "frontier_curriculum_teacher_minus_eos_ndcg_at_10": gap(frontier),
            "frontier_curriculum_teacher_minus_eos_recall_at_100": recall_gap(frontier),
            "frontier_curriculum_first_relevant_rank_improvement": rank_improvement(frontier),
            "frontier_curriculum_dataset": text(frontier.get("dataset")),
            "frontier_curriculum_query_id": text(frontier.get("query_id")),
            "frontier_curriculum_teacher_label": text(nested(frontier, "teacher", "label")),
            "frontier_curriculum_eos_label": text(nested(frontier, "eos", "label")),
            "frontier_curriculum_original_hard_negative_source": text(hard_negative.get("source")),
            "frontier_curriculum_original_hard_negative_metadata": dict(original_metadata),
        }
    )
    negatives = hard_negative.get("negatives")
    if not isinstance(negatives, list):
        negatives = []
    if max_negatives is not None:
        negatives = negatives[:max_negatives]
    return {
        "schema": hard_negative.get("schema"),
        "source": hard_negative.get("source"),
        "query": hard_negative.get("query"),
        "positive": hard_negative.get("positive"),
        "negatives": negatives,
        "metadata": metadata,
    }


def selected_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def build_curriculum(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = parse_sources(config)
    dedupe = bool(config.get("dedupe", True))
    max_negatives_raw = config.get("max_negatives")
    max_negatives = None if max_negatives_raw is None else integer(max_negatives_raw, -1)
    if max_negatives is not None and max_negatives < 0:
        raise ValueError("max_negatives must be non-negative when provided")
    quality_claim = bool(config.get("quality_claim", False))
    if quality_claim:
        raise ValueError("quality_claim must be false; curriculum output is not quality evidence")
    require_teacher_hit = bool(config.get("require_teacher_hit", True))
    claim_boundary = text(config.get("claim_boundary")) or DEFAULT_CLAIM_BOUNDARY

    all_candidates: list[CandidateRow] = []
    source_stats: dict[str, SourceStats] = {}
    for source in sources:
        candidates, stats = build_source_candidates(
            source, require_teacher_hit=require_teacher_hit, max_negatives=max_negatives
        )
        all_candidates.extend(candidates)
        source_stats[source.name] = stats

    all_candidates.sort(
        key=lambda candidate: (
            candidate.source.priority,
            sources.index(candidate.source),
            candidate.source_rank,
            text(candidate.frontier.get("query_id")),
        )
    )

    output_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_removals = 0
    selected_gaps: list[float] = []
    selected_recall_gaps: list[float] = []
    for candidate in all_candidates:
        key = candidate.dedupe_key
        if dedupe and key in seen_keys:
            source_stats[candidate.source.name].deduped += 1
            duplicate_removals += 1
            continue
        if dedupe:
            seen_keys.add(key)
        output_rows.append(enriched_hard_negative(candidate, max_negatives))
        source_stats[candidate.source.name].selected += 1
        selected_gaps.append(gap(candidate.frontier))
        selected_recall_gaps.append(recall_gap(candidate.frontier))

    manifest = {
        "schema": CURRICULUM_MANIFEST_SCHEMA,
        "quality_claim": False,
        "claim_boundary": claim_boundary,
        "source_schema": FRONTIER_SCHEMA,
        "dedupe": dedupe,
        "require_teacher_hit": require_teacher_hit,
        "max_negatives": max_negatives,
        "source_count": len(sources),
        "sources": [source_stats[source.name].as_dict() for source in sources],
        "duplicate_removal_count": duplicate_removals,
        "output_row_count": len(output_rows),
        "selected_teacher_minus_eos_ndcg_at_10": selected_stats(selected_gaps),
        "selected_teacher_minus_eos_recall_at_100": selected_stats(selected_recall_gaps),
        "caveat": DEFAULT_CLAIM_BOUNDARY,
    }
    return output_rows, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_tsv(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "role",
                "priority",
                "rows_seen",
                "hard_negative_rows_seen",
                "joined",
                "eligible",
                "capped",
                "selected",
                "deduped",
                "dropped",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for source in manifest["sources"]:
            row = dict(source)
            row["dropped"] = json.dumps(row.get("dropped", {}), sort_keys=True)
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames or []})


def main() -> None:
    args = parse_args()
    try:
        rows, manifest = build_curriculum(load_config(args.config_json))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    write_jsonl(args.output_jsonl, rows)
    write_json(args.manifest_json, manifest)
    if args.summary_tsv:
        write_summary_tsv(args.summary_tsv, manifest)
    print(
        "built retrieval frontier curriculum: "
        f"rows={manifest['output_row_count']} duplicates_removed={manifest['duplicate_removal_count']} "
        f"jsonl={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
