#!/usr/bin/env python3
"""Mine guarded LongEmbed frontier hard-negative rows from per-query failures.

The source examples are protected candidate data from capped LongEmbed slices.
Outputs are suitable for guarded data-generation audits only and intentionally
carry ``quality_claim=false`` plus an explicit claim boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "manta.embedding_longembed_frontier_teacher_pair.v1"
CLAIM_BOUNDARY = (
    "Inputs were mined from capped official LongEmbed slices and are protected "
    "candidate training data only; they must not be used to claim benchmark "
    "quality on the same slice."
)
CATEGORY_FAMILIES = {
    "worst_compact_loss_vs_direct": "direct_rescue",
    "direct_wins_token_span_loses": "direct_rescue",
    "token_span_wins_direct_loses": "token_span_rescue",
    "sparse_parent_compact_loses_dense_rank": "sparse_compact_preservation",
    "external_wins_when_eos_loses": "external_teacher_win",
}


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    dataset_dir: Path
    run_dir: Path | None
    sparse_per_query: Path | None


@dataclass
class DatasetIndex:
    name: str
    dataset_dir: Path
    corpus: dict[str, str]
    queries: dict[str, str]


@dataclass
class ProfileRows:
    profile: str
    path: Path
    rows: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build guarded hard-negative rows from LongEmbed frontier candidate TSV rows."
    )
    parser.add_argument("--candidate-tsv", required=True, type=Path)
    parser.add_argument("--frontier-report", type=Path, default=None)
    parser.add_argument("--per-query-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--run-dir", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=int, default=80)
    parser.add_argument("--eval-ratio", type=int, default=20)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument(
        "--teacher-eligible-dataset",
        action="append",
        default=["qmsum", "2wikimqa"],
        help="Dataset to include in teacher-eligible split files. May be repeated.",
    )
    return parser.parse_args()


def parse_mapping(values: list[str], flag: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name:
            raise SystemExit(f"{flag} has empty name: {value!r}")
        out[name] = Path(raw_path)
    return out


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def row_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError(f"{path}:{line_number}: row has no _id/id")
    return str(value)


def corpus_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return title + "\n" + text
    return title or text


def load_dataset_index(name: str, dataset_dir: Path) -> DatasetIndex:
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    for path in (corpus_path, queries_path):
        if not path.is_file():
            raise SystemExit(f"missing dataset file: {path}")
    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    for line_number, row in iter_jsonl(corpus_path):
        corpus[row_id(row, corpus_path, line_number)] = corpus_text(row)
    for line_number, row in iter_jsonl(queries_path):
        queries[row_id(row, queries_path, line_number)] = str(row.get("text") or row.get("query") or "")
    return DatasetIndex(name=name, dataset_dir=dataset_dir, corpus=corpus, queries=queries)


def load_profile_rows(profile: str, path: Path) -> ProfileRows:
    if not path.is_file():
        raise SystemExit(f"missing per-query profile file for {profile}: {path}")
    rows = {}
    for line_number, row in iter_jsonl(path):
        query_id = row.get("query_id")
        if query_id is None:
            raise ValueError(f"{path}:{line_number}: missing query_id")
        rows[str(query_id)] = row
    return ProfileRows(profile=profile, path=path, rows=rows)


def sparse_profile_path(paths: DatasetPaths) -> Path:
    if paths.sparse_per_query is None:
        raise SystemExit(f"missing sparse per-query path for {paths.name}")
    return paths.sparse_per_query


def profile_path(paths: DatasetPaths, profile: str) -> Path:
    if profile in {"eos/sparse_parent_dense", "eos/sparse_parent_q4", "eos/sparse_parent_q6", "eos/sparse_parent_q7", "eos/sparse_parent_q8"}:
        return sparse_profile_path(paths)
    if paths.run_dir is None:
        return sparse_profile_path(paths)
    if profile == "eos/direct_single_vector":
        return paths.run_dir / "direct-eos.per-query.jsonl"
    if profile in {"eos/token_span_dense", "eos/token_span_q4"}:
        return paths.run_dir / "eos-token-span-multivector-turboquant.per-query.jsonl"
    if profile.startswith("eos/direct_token_span_fusion"):
        return paths.run_dir / "direct-token-span-fusion.per-query.jsonl"
    if profile.startswith("external/qwen3_0.6b"):
        return paths.run_dir / "external-qwen3-0.6b-multivector-turboquant.per-query.jsonl"
    if profile.startswith("external/mxbai_large"):
        return paths.run_dir / "external-mxbai-large-multivector-turboquant.per-query.jsonl"
    return sparse_profile_path(paths)


def default_dataset_paths(
    dataset_dirs: dict[str, Path],
    run_dirs: dict[str, Path],
    per_query_dir: Path,
) -> dict[str, DatasetPaths]:
    names = sorted(dataset_dirs)
    paths: dict[str, DatasetPaths] = {}
    for name in names:
        sparse = per_query_dir / f"{name}-sparse-parent-q4678.per-query.jsonl"
        paths[name] = DatasetPaths(
            name=name,
            dataset_dir=dataset_dirs[name],
            run_dir=run_dirs.get(name),
            sparse_per_query=sparse if sparse.is_file() else None,
        )
    return paths


def split_score(seed: int, dataset: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{query_id}".encode("utf-8")).hexdigest()


def assign_splits(rows: list[dict[str, Any]], train_ratio: int, eval_ratio: int, seed: int) -> dict[tuple[str, str], str]:
    if train_ratio < 0 or eval_ratio < 0 or train_ratio + eval_ratio <= 0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")
    query_keys = sorted(
        {(row["metadata"]["dataset"], row["metadata"]["query_id"]) for row in rows},
        key=lambda item: (split_score(seed, item[0], item[1]), item),
    )
    train_count = round(len(query_keys) * train_ratio / (train_ratio + eval_ratio))
    if len(query_keys) > 1:
        train_count = min(max(train_count, 1), len(query_keys) - 1)
    train_keys = set(query_keys[:train_count])
    return {key: ("train" if key in train_keys else "eval") for key in query_keys}


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int_or_none(value: str) -> int | None:
    if value == "":
        return None
    return int(float(value))


def hard_negative_docs(profile_row: dict[str, Any], positive_doc_id: str, max_negatives: int) -> list[dict[str, Any]]:
    ranked = profile_row.get("top_k")
    if not isinstance(ranked, list):
        return []
    first_relevant = profile_row.get("first_relevant_rank")
    try:
        first_relevant_rank = int(first_relevant) if first_relevant is not None else None
    except (TypeError, ValueError):
        first_relevant_rank = None

    def is_candidate(item: dict[str, Any]) -> bool:
        return str(item.get("doc_id")) != positive_doc_id and int(item.get("relevance") or 0) <= 0

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    if first_relevant_rank and first_relevant_rank > 1:
        for item in ranked:
            if int(item.get("rank") or 0) >= first_relevant_rank:
                continue
            doc_id = str(item.get("doc_id") or "")
            if doc_id and is_candidate(item) and doc_id not in seen:
                selected.append(item)
                seen.add(doc_id)
            if len(selected) >= max_negatives:
                return selected
    for item in ranked:
        doc_id = str(item.get("doc_id") or "")
        if doc_id and is_candidate(item) and doc_id not in seen:
            selected.append(item)
            seen.add(doc_id)
        if len(selected) >= max_negatives:
            break
    return selected


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dirs = parse_mapping(args.dataset_dir, "--dataset-dir")
    run_dirs = parse_mapping(args.run_dir, "--run-dir")
    if not dataset_dirs:
        raise SystemExit("at least one --dataset-dir is required")
    dataset_paths = default_dataset_paths(dataset_dirs, run_dirs, args.per_query_dir)
    datasets = {name: load_dataset_index(name, path) for name, path in dataset_dirs.items()}
    profile_cache: dict[tuple[str, str], ProfileRows] = {}

    candidate_rows = 0
    skipped_rows = Counter()
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    profile_sources: dict[str, str] = {}

    with args.candidate_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 2):
            candidate_rows += 1
            dataset_name = row["dataset"]
            dataset = datasets.get(dataset_name)
            paths = dataset_paths.get(dataset_name)
            if dataset is None or paths is None:
                skipped_rows["unknown_dataset"] += 1
                continue
            category = row["category"]
            family = CATEGORY_FAMILIES.get(category, "frontier_rescue")
            query_id = row["query_id"]
            positive_doc_id = row["winner_relevant_doc_id"] or row["loser_relevant_doc_id"]
            query = dataset.queries.get(query_id)
            positive = dataset.corpus.get(positive_doc_id)
            if query is None or positive is None:
                skipped_rows["missing_query_or_positive"] += 1
                continue
            loser_profile = row["loser_profile"]
            profile_key = (dataset_name, loser_profile)
            if profile_key not in profile_cache:
                path = profile_path(paths, loser_profile)
                profile_cache[profile_key] = load_profile_rows(loser_profile, path)
                profile_sources[f"{dataset_name}:{loser_profile}"] = str(path)
            profile_rows = profile_cache[profile_key]
            profile_row = profile_rows.rows.get(query_id)
            if profile_row is None:
                skipped_rows["missing_profile_query"] += 1
                continue
            negative_items = hard_negative_docs(profile_row, positive_doc_id, args.max_negatives)
            negative_doc_ids = [str(item.get("doc_id")) for item in negative_items]
            negatives = [dataset.corpus[doc_id] for doc_id in negative_doc_ids if doc_id in dataset.corpus]
            if not negatives:
                skipped_rows["no_negatives"] += 1
                continue

            key = (dataset_name, category, query_id, positive_doc_id, loser_profile)
            if key in seen_keys:
                skipped_rows["duplicate_key"] += 1
                continue
            seen_keys.add(key)
            metadata = {
                "dataset": dataset_name,
                "query_id": query_id,
                "category": category,
                "rescue_family": family,
                "positive_doc_id": positive_doc_id,
                "negative_doc_ids": negative_doc_ids[: len(negatives)],
                "winner_profile": row["winner_profile"],
                "loser_profile": loser_profile,
                "winner_ndcg_at_10": as_float(row, "winner_ndcg_at_10"),
                "loser_ndcg_at_10": as_float(row, "loser_ndcg_at_10"),
                "delta_ndcg_at_10": as_float(row, "delta_ndcg_at_10"),
                "winner_first_relevant_rank": as_int_or_none(row["winner_first_relevant_rank"]),
                "loser_first_relevant_rank": as_int_or_none(row["loser_first_relevant_rank"]),
                "winner_relevant_doc_id": row["winner_relevant_doc_id"],
                "loser_relevant_doc_id": row["loser_relevant_doc_id"],
                "loser_per_query_path": str(profile_rows.path),
                "candidate_tsv_line": line_number,
                "negative_rank_evidence": [
                    {
                        "doc_id": str(item.get("doc_id")),
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "dense_rank": item.get("dense_rank"),
                        "compact_rank": item.get("compact_rank"),
                    }
                    for item in negative_items[: len(negatives)]
                ],
                "quality_claim": False,
                "claim_boundary": CLAIM_BOUNDARY,
            }
            rows.append(
                {
                    "schema": SCHEMA,
                    "source": f"longembed-frontier-teacher-pair-miner:{dataset_name}:{category}:{loser_profile}",
                    "query": query,
                    "positive": positive,
                    "negatives": negatives,
                    "metadata": metadata,
                }
            )

    stats = {
        "candidate_rows": candidate_rows,
        "emitted_rows": len(rows),
        "skipped_rows": dict(sorted(skipped_rows.items())),
        "profile_sources": dict(sorted(profile_sources.items())),
    }
    return rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row["metadata"].get(key, "")) for row in rows)
    return dict(sorted(counts.items()))


def counter_by_pair(rows: list[dict[str, Any]], first: str, second: str) -> dict[str, int]:
    counts = Counter(
        f"{row['metadata'].get(first)}\t{row['metadata'].get(second)}"
        for row in rows
    )
    return dict(sorted(counts.items()))


def mine(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_negatives <= 0:
        raise SystemExit("--max-negatives must be positive")
    rows, stats = build_rows(args)
    split_map = assign_splits(rows, args.train_ratio, args.eval_ratio, args.seed)
    for row in rows:
        key = (row["metadata"]["dataset"], row["metadata"]["query_id"])
        row["metadata"]["split"] = split_map[key]

    train = [row for row in rows if row["metadata"]["split"] == "train"]
    eval_rows = [row for row in rows if row["metadata"]["split"] == "eval"]
    teacher_datasets = set(args.teacher_eligible_dataset)
    teacher_train = [row for row in train if row["metadata"]["dataset"] in teacher_datasets]
    teacher_eval = [row for row in eval_rows if row["metadata"]["dataset"] in teacher_datasets]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all-hard-negatives.jsonl", rows)
    write_jsonl(args.output_dir / "train-hard-negatives.jsonl", train)
    write_jsonl(args.output_dir / "eval-hard-negatives.jsonl", eval_rows)
    write_jsonl(args.output_dir / "teacher-eligible-train-hard-negatives.jsonl", teacher_train)
    write_jsonl(args.output_dir / "teacher-eligible-eval-hard-negatives.jsonl", teacher_eval)
    for category in sorted({row["metadata"]["category"] for row in rows}):
        write_jsonl(args.output_dir / "by-category" / f"{category}.jsonl", [row for row in rows if row["metadata"]["category"] == category])

    manifest = {
        "schema": "manta.embedding_longembed_frontier_teacher_pair_manifest.v1",
        "quality_claim": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "candidate_tsv": str(args.candidate_tsv),
        "frontier_report": str(args.frontier_report) if args.frontier_report else None,
        "per_query_dir": str(args.per_query_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "split_ratios": {"train": args.train_ratio, "eval": args.eval_ratio},
        "max_negatives": args.max_negatives,
        "row_counts": {
            **stats,
            "train": len(train),
            "eval": len(eval_rows),
            "teacher_eligible_train": len(teacher_train),
            "teacher_eligible_eval": len(teacher_eval),
        },
        "counts_by_dataset": counter_by(rows, "dataset"),
        "counts_by_category": counter_by(rows, "category"),
        "counts_by_rescue_family": counter_by(rows, "rescue_family"),
        "counts_by_dataset_category": counter_by_pair(rows, "dataset", "category"),
        "teacher_eligible_datasets": sorted(teacher_datasets),
        "caveats": [
            CLAIM_BOUNDARY,
            "Train/eval split only prevents query-id overlap inside this generated candidate bundle.",
            "Rows are generated for guarded hard-negative training experiments; no training or benchmark claim is made here.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = mine(args)
    print(
        "mined longembed frontier teacher pairs: "
        f"input={manifest['row_counts']['candidate_rows']} emitted={manifest['row_counts']['emitted_rows']} "
        f"train={manifest['row_counts']['train']} eval={manifest['row_counts']['eval']} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
