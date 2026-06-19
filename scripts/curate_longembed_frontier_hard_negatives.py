#!/usr/bin/env python3
"""Curate mined LongEmbed frontier hard negatives into guarded bundles.

The input rows are protected candidate training data mined from capped eval
slices. The generated manifest deliberately records that boundary so downstream
training can use the rows without treating them as benchmark-quality evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA = "manta.embedding_longembed_frontier_hard_negative_curation.v1"
TRAIN_CLAIM_BOUNDARY = (
    "Inputs were mined from capped official eval slices and are protected "
    "candidate training data only; they must not be used to claim benchmark "
    "quality on the same slice."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and split LongEmbed quality-frontier hard-negative JSONL rows."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input hard-negative JSONL. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=int, default=80)
    parser.add_argument("--eval-ratio", type=int, default=20)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument(
        "--source-prefix",
        default="",
        help="Optional output source prefix; original sources are kept in metadata.sources.",
    )
    return parser.parse_args()


@dataclass
class CuratedRow:
    query_id: str
    query: str
    positive: str
    schema: str = ""
    negatives: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)
    teacher_labels: set[str] = field(default_factory=set)
    eos_labels: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    merge_count: int = 0

    def key(self) -> tuple[str, str, str]:
        return (self.query_id, self.query, self.positive)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def require_text(row: dict[str, Any], name: str, path: Path, line_no: int) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_no}: missing non-empty {name!r}")
    return value


def row_query_id(row: dict[str, Any], path: Path, line_no: int) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}:{line_no}: missing metadata object")
    value = metadata.get("query_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_no}: missing metadata.query_id")
    return value


def sorted_list(values: set[str]) -> list[str]:
    return sorted(v for v in values if v)


def merge_metadata(
    base: dict[str, Any],
    incoming: dict[str, Any],
    teacher_labels: set[str],
    eos_labels: set[str],
    sources: set[str],
    source_files: set[str],
    merge_count: int,
) -> dict[str, Any]:
    merged = dict(base)
    merged["query_id"] = incoming.get("query_id", merged.get("query_id"))
    merged["sources"] = sorted_list(sources)
    merged["teacher_labels"] = sorted_list(teacher_labels)
    merged["eos_labels"] = sorted_list(eos_labels)
    merged["source_files"] = sorted_list(source_files)
    merged["merged_input_rows"] = merge_count

    for key in (
        "teacher_top_relevant_doc_ids",
        "eos_top_nonrelevant_doc_ids",
    ):
        values: list[str] = []
        seen: set[str] = set()
        for candidate in (base.get(key), incoming.get(key)):
            if not isinstance(candidate, list):
                continue
            for item in candidate:
                text = str(item)
                if text and text not in seen:
                    seen.add(text)
                    values.append(text)
        if values:
            merged[key] = values

    gaps = []
    for candidate in (base.get("teacher_minus_eos_ndcg_at_10"), incoming.get("teacher_minus_eos_ndcg_at_10")):
        if isinstance(candidate, int | float):
            gaps.append(float(candidate))
    if gaps:
        merged["teacher_minus_eos_ndcg_at_10_max"] = max(gaps)
    return merged


def normalize_inputs(paths: list[Path], max_negatives: int) -> tuple[list[CuratedRow], dict[str, Any]]:
    if max_negatives <= 0:
        raise ValueError("--max-negatives must be positive")

    by_key: dict[tuple[str, str, str], CuratedRow] = {}
    source_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    file_counts: dict[str, int] = {}
    input_rows = 0

    for path in paths:
        rows = read_jsonl(path)
        file_counts[str(path)] = len(rows)
        for index, row in enumerate(rows, 1):
            input_rows += 1
            query_id = row_query_id(row, path, index)
            query = require_text(row, "query", path, index)
            positive = require_text(row, "positive", path, index)
            schema = str(row.get("schema", "") or "")
            source = str(row.get("source", "") or "")
            metadata = row.get("metadata")
            assert isinstance(metadata, dict)
            teacher_label = str(metadata.get("teacher_label", "") or "")
            eos_label = str(metadata.get("eos_label", "") or "")
            negatives_value = row.get("negatives")
            if not isinstance(negatives_value, list):
                raise ValueError(f"{path}:{index}: missing negatives list")

            key = (query_id, query, positive)
            curated = by_key.get(key)
            if curated is None:
                curated = CuratedRow(
                    query_id=query_id,
                    query=query,
                    positive=positive,
                    schema=schema,
                    metadata=dict(metadata),
                )
                by_key[key] = curated
            elif not curated.schema and schema:
                curated.schema = schema

            curated.merge_count += 1
            curated.sources.add(source)
            curated.source_files.add(str(path))
            curated.teacher_labels.add(teacher_label)
            curated.eos_labels.add(eos_label)
            source_counts[source] += 1
            teacher_counts[teacher_label] += 1

            seen_negatives = set(curated.negatives)
            for value in negatives_value:
                negative = str(value)
                if not negative or negative == positive or negative in seen_negatives:
                    continue
                if len(curated.negatives) >= max_negatives:
                    break
                curated.negatives.append(negative)
                seen_negatives.add(negative)
            curated.metadata = merge_metadata(
                curated.metadata,
                metadata,
                curated.teacher_labels,
                curated.eos_labels,
                curated.sources,
                curated.source_files,
                curated.merge_count,
            )

    rows = sorted(by_key.values(), key=lambda row: row.key())
    stats = {
        "input_rows": input_rows,
        "unique_rows": len(rows),
        "duplicate_rows": input_rows - len(rows),
        "merged_rows": sum(1 for row in rows if row.merge_count > 1),
        "source_files": file_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_counts": dict(sorted(teacher_counts.items())),
    }
    return rows, stats


def split_score(seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).hexdigest()


def split_rows(
    rows: list[CuratedRow], train_ratio: int, eval_ratio: int, seed: int
) -> tuple[list[CuratedRow], list[CuratedRow]]:
    if train_ratio < 0 or eval_ratio < 0 or train_ratio + eval_ratio <= 0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")

    by_qid: dict[str, list[CuratedRow]] = {}
    for row in rows:
        by_qid.setdefault(row.query_id, []).append(row)

    query_ids = sorted(by_qid, key=lambda qid: (split_score(seed, qid), qid))
    train_count = round(len(query_ids) * train_ratio / (train_ratio + eval_ratio))
    if len(query_ids) > 1:
        train_count = min(max(train_count, 1), len(query_ids) - 1)
    train_ids = set(query_ids[:train_count])

    train: list[CuratedRow] = []
    eval_rows: list[CuratedRow] = []
    for row in rows:
        if row.query_id in train_ids:
            train.append(row)
        else:
            eval_rows.append(row)
    return train, eval_rows


def output_source(row: CuratedRow, source_prefix: str) -> str:
    sources = sorted_list(row.sources)
    if source_prefix:
        suffix = "+".join(sources)
        return f"{source_prefix}:{suffix}" if suffix else source_prefix
    return "+".join(sources)


def to_json_row(row: CuratedRow, source_prefix: str) -> dict[str, Any]:
    out = {
        "source": output_source(row, source_prefix),
        "query": row.query,
        "positive": row.positive,
        "negatives": row.negatives,
        "metadata": row.metadata,
    }
    if row.schema:
        out["schema"] = row.schema
    return out


def write_jsonl(path: Path, rows: list[CuratedRow], source_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_row(row, source_prefix), sort_keys=True, separators=(",", ":")) + "\n")


def write_summary_tsv(path: Path, train: list[CuratedRow], eval_rows: list[CuratedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "query_id",
                "sources",
                "teacher_labels",
                "negative_count",
                "merged_input_rows",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for split_name, rows in (("train", train), ("eval", eval_rows)):
            for row in rows:
                writer.writerow(
                    {
                        "split": split_name,
                        "query_id": row.query_id,
                        "sources": ",".join(sorted_list(row.sources)),
                        "teacher_labels": ",".join(sorted_list(row.teacher_labels)),
                        "negative_count": len(row.negatives),
                        "merged_input_rows": row.merge_count,
                    }
                )


def manifest(
    stats: dict[str, Any],
    train: list[CuratedRow],
    eval_rows: list[CuratedRow],
    seed: int,
    train_ratio: int,
    eval_ratio: int,
    max_negatives: int,
    source_prefix: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source_files": stats["source_files"],
        "row_counts": {
            "input": stats["input_rows"],
            "unique": stats["unique_rows"],
            "duplicate": stats["duplicate_rows"],
            "merged": stats["merged_rows"],
            "train": len(train),
            "eval": len(eval_rows),
        },
        "train_count": len(train),
        "eval_count": len(eval_rows),
        "source_counts": stats["source_counts"],
        "teacher_counts": stats["teacher_counts"],
        "seed": seed,
        "split_ratios": {"train": train_ratio, "eval": eval_ratio},
        "max_negatives": max_negatives,
        "source_prefix": source_prefix,
        "quality_claim": False,
        "train_claim_boundary": TRAIN_CLAIM_BOUNDARY,
        "caveats": [
            TRAIN_CLAIM_BOUNDARY,
            "Train/eval split prevents query-id overlap inside this curated bundle only.",
            "Rows retain mined teacher labels and sources for guarded candidate training audits.",
        ],
    }


def curate(args: argparse.Namespace) -> dict[str, Any]:
    rows, stats = normalize_inputs(args.input, args.max_negatives)
    train, eval_rows = split_rows(rows, args.train_ratio, args.eval_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train-hard-negatives.jsonl"
    eval_path = args.output_dir / "eval-hard-negatives.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    summary_path = args.output_dir / "summary.tsv"

    write_jsonl(train_path, train, args.source_prefix)
    write_jsonl(eval_path, eval_rows, args.source_prefix)
    write_summary_tsv(summary_path, train, eval_rows)
    data = manifest(
        stats,
        train,
        eval_rows,
        args.seed,
        args.train_ratio,
        args.eval_ratio,
        args.max_negatives,
        args.source_prefix,
    )
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> None:
    args = parse_args()
    data = curate(args)
    print(
        "curated longembed frontier hard negatives: "
        f"input={data['row_counts']['input']} unique={data['row_counts']['unique']} "
        f"train={data['train_count']} eval={data['eval_count']} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
