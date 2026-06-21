#!/usr/bin/env python3
"""Build provenance-safe agreement-teacher hard-negative inputs.

The input hard-negative and teacher-scored rows are text keyed. This script
resolves those texts back to raw BEIR IDs, keeps only rows whose positive pair
is exactly a train qrel and whose query/positive/negative doc IDs are absent
from test qrels, and writes line-aligned filtered base/teacher JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--prep-root",
        type=Path,
        required=True,
        help="Existing agreement teacher prep root containing per-teacher scored rows.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", default="scifact,nfcorpus,fiqa")
    parser.add_argument("--teachers", default="qwen3-0.6b,mxbai-large")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--drop-sample-limit", type=int, default=30)
    return parser.parse_args()


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


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def corpus_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return stable_text(title + "\n" + text)
    return stable_text(title or text)


def query_text(row: dict[str, Any]) -> str:
    return stable_text(row.get("text") or row.get("query") or "")


def row_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError(f"{path}:{line_number}: row missing _id/id")
    return str(value)


def load_text_ids(path: Path, text_fn) -> dict[str, list[str]]:
    by_text: dict[str, list[str]] = defaultdict(list)
    for line_number, row in iter_jsonl(path):
        by_text[text_fn(row)].append(row_id(row, path, line_number))
    return by_text


def load_qrels(dataset_dir: Path) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]], dict[str, set[str]]]:
    pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    query_splits: dict[str, set[str]] = defaultdict(set)
    doc_splits: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev", "test"):
        path = dataset_dir / "qrels" / f"{split}.tsv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                query_id = str(row.get("query-id") or row.get("query_id") or "")
                doc_id = str(row.get("corpus-id") or row.get("corpus_id") or "")
                try:
                    score = float(row.get("score") or 0)
                except (TypeError, ValueError):
                    score = 0.0
                if not query_id or not doc_id or score <= 0:
                    continue
                pair_splits[(query_id, doc_id)].add(split)
                query_splits[query_id].add(split)
                doc_splits[doc_id].add(split)
    return pair_splits, query_splits, doc_splits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_signature(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        stable_text(row.get("query")),
        stable_text(row.get("positive")),
        tuple(stable_text(value) for value in row.get("negatives") or []),
    )


def provenance_row_id(source: str, query_id: str, positive_doc_id: str, negative_doc_ids: list[str]) -> str:
    payload = json.dumps(
        {
            "source": source,
            "split": "train",
            "query_id": query_id,
            "positive_doc_id": positive_doc_id,
            "negative_doc_ids": negative_doc_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_drop_sample(samples: list[dict[str, Any]], limit: int, sample: dict[str, Any]) -> None:
    if len(samples) < limit:
        samples.append(sample)


def resolve_row(
    dataset: str,
    index: int,
    row: dict[str, Any],
    query_text_to_ids: dict[str, list[str]],
    doc_text_to_ids: dict[str, list[str]],
    pair_splits: dict[tuple[str, str], set[str]],
    query_splits: dict[str, set[str]],
    doc_splits: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    query_ids = query_text_to_ids.get(stable_text(row.get("query")), [])
    positive_doc_ids = doc_text_to_ids.get(stable_text(row.get("positive")), [])
    pair_candidates = [
        (query_id, doc_id, pair_splits[(query_id, doc_id)])
        for query_id in query_ids
        for doc_id in positive_doc_ids
        if (query_id, doc_id) in pair_splits
    ]
    train_pair_candidates = [
        (query_id, doc_id, splits)
        for query_id, doc_id, splits in pair_candidates
        if splits == {"train"}
    ]
    detail = {
        "source_index": index,
        "query_id_candidates": query_ids,
        "positive_doc_id_candidates": positive_doc_ids,
        "pair_candidates": [
            {"query_id": query_id, "doc_id": doc_id, "splits": sorted(splits)}
            for query_id, doc_id, splits in pair_candidates
        ],
    }
    if len(train_pair_candidates) != 1:
        return None, "unresolved_or_non_train_positive_pair", detail
    query_id, positive_doc_id, _ = train_pair_candidates[0]
    if "test" in query_splits.get(query_id, set()):
        detail.update({"query_id": query_id, "positive_doc_id": positive_doc_id})
        return None, "test_query_id", detail
    if "test" in doc_splits.get(positive_doc_id, set()):
        detail.update({"query_id": query_id, "positive_doc_id": positive_doc_id})
        return None, "test_positive_doc_id", detail
    if dataset == "fiqa" and (query_id == "6133" or positive_doc_id == "7733"):
        detail.update({"query_id": query_id, "positive_doc_id": positive_doc_id})
        return None, "prohibited_fiqa_6133_7733", detail

    negative_doc_ids: list[str] = []
    for negative_index, negative in enumerate(row.get("negatives") or []):
        ids = doc_text_to_ids.get(stable_text(negative), [])
        if len(ids) != 1:
            detail.update(
                {
                    "query_id": query_id,
                    "positive_doc_id": positive_doc_id,
                    "negative_index": negative_index,
                    "negative_doc_id_candidates": ids,
                }
            )
            return None, "unresolved_negative_doc_id", detail
        negative_doc_id = ids[0]
        if "test" in doc_splits.get(negative_doc_id, set()):
            detail.update(
                {
                    "query_id": query_id,
                    "positive_doc_id": positive_doc_id,
                    "negative_index": negative_index,
                    "negative_doc_id": negative_doc_id,
                }
            )
            return None, "test_negative_doc_id", detail
        negative_doc_ids.append(negative_doc_id)

    out = dict(row)
    out["source"] = dataset
    out["split"] = "train"
    out["query_id"] = query_id
    out["positive_doc_id"] = positive_doc_id
    out["negative_doc_ids"] = negative_doc_ids
    out["row_id"] = provenance_row_id(dataset, query_id, positive_doc_id, negative_doc_ids)
    return out, "kept", {"query_id": query_id, "positive_doc_id": positive_doc_id}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(
    repo_root: Path,
    prep_root: Path,
    output_root: Path,
    dataset: str,
    teachers: list[str],
    drop_sample_limit: int,
) -> dict[str, Any]:
    dataset_dir = repo_root / "datasets" / "manta-embed-v1" / "raw" / dataset / dataset
    base_path = repo_root / "datasets" / "manta-embed-v1" / "processed" / f"{dataset}.train-hard-negatives.jsonl"
    teacher_paths = {
        teacher: prep_root
        / dataset
        / teacher
        / f"{dataset}.{teacher}.teacher-scored.train-hard-negatives.jsonl"
        for teacher in teachers
    }
    for path in [dataset_dir / "queries.jsonl", dataset_dir / "corpus.jsonl", base_path, *teacher_paths.values()]:
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")

    query_text_to_ids = load_text_ids(dataset_dir / "queries.jsonl", query_text)
    doc_text_to_ids = load_text_ids(dataset_dir / "corpus.jsonl", corpus_text)
    pair_splits, query_splits, doc_splits = load_qrels(dataset_dir)
    base_rows = [row for _, row in iter_jsonl(base_path)]
    teacher_rows = {teacher: [row for _, row in iter_jsonl(path)] for teacher, path in teacher_paths.items()}
    for teacher, rows in teacher_rows.items():
        if len(rows) != len(base_rows):
            raise SystemExit(f"{teacher_paths[teacher]} row count does not match {base_path}")

    kept_indices: list[int] = []
    kept_base_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    drop_samples: list[dict[str, Any]] = []
    for index, row in enumerate(base_rows):
        resolved, reason, detail = resolve_row(
            dataset,
            index,
            row,
            query_text_to_ids,
            doc_text_to_ids,
            pair_splits,
            query_splits,
            doc_splits,
        )
        counts[reason] += 1
        if resolved is None:
            add_drop_sample(drop_samples, drop_sample_limit, {"reason": reason, **detail})
            continue
        kept_indices.append(index)
        kept_base_rows.append(resolved)

    filtered_teacher_paths: dict[str, str] = {}
    for teacher in teachers:
        filtered: list[dict[str, Any]] = []
        for source_index, base_row in zip(kept_indices, kept_base_rows):
            teacher_row = dict(teacher_rows[teacher][source_index])
            if text_signature(teacher_row) != text_signature(base_rows[source_index]):
                raise SystemExit(f"{teacher_paths[teacher]}:{source_index + 1}: text signature mismatch")
            for key in ("source", "split", "query_id", "positive_doc_id", "negative_doc_ids", "row_id"):
                teacher_row[key] = base_row[key]
            filtered.append(teacher_row)
        teacher_out = (
            output_root
            / dataset
            / teacher
            / f"{dataset}.{teacher}.provenance-safe.teacher-scored.train-hard-negatives.jsonl"
        )
        write_jsonl(teacher_out, filtered)
        filtered_teacher_paths[teacher] = str(teacher_out)

    base_out = output_root / dataset / f"{dataset}.provenance-safe.base.train-hard-negatives.jsonl"
    write_jsonl(base_out, kept_base_rows)

    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "base_input": str(base_path),
        "base_output": str(base_out),
        "teacher_inputs": {teacher: str(path) for teacher, path in teacher_paths.items()},
        "teacher_outputs": filtered_teacher_paths,
        "input_sha256": {
            "base": sha256_file(base_path),
            **{f"teacher:{teacher}": sha256_file(path) for teacher, path in teacher_paths.items()},
        },
        "output_sha256": {
            "base": sha256_file(base_out),
            **{f"teacher:{teacher}": sha256_file(Path(path)) for teacher, path in filtered_teacher_paths.items()},
        },
        "counts": dict(counts),
        "rows_input": len(base_rows),
        "rows_kept": len(kept_base_rows),
        "rows_dropped": len(base_rows) - len(kept_base_rows),
        "drop_samples": drop_samples,
        "raw_text_duplicates": {
            "query_texts": sum(1 for ids in query_text_to_ids.values() if len(ids) > 1),
            "doc_texts": sum(1 for ids in doc_text_to_ids.values() if len(ids) > 1),
        },
    }


def main() -> int:
    args = parse_args()
    if args.drop_sample_limit < 0:
        raise SystemExit("--drop-sample-limit must be non-negative")
    repo_root = args.repo_root.resolve()
    prep_root = args.prep_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    teachers = [value.strip() for value in args.teachers.split(",") if value.strip()]
    if not datasets:
        raise SystemExit("--datasets must not be empty")
    if len(teachers) < 2:
        raise SystemExit("--teachers must contain at least two teachers")

    dataset_manifests = [
        build_dataset(repo_root, prep_root, output_root, dataset, teachers, args.drop_sample_limit)
        for dataset in datasets
    ]
    manifest = {
        "schema": "manta.agreement_teacher_provenance_safe_prep.v1",
        "repo_root": str(repo_root),
        "prep_root": str(prep_root),
        "output_root": str(output_root),
        "datasets": dataset_manifests,
        "totals": {
            "rows_input": sum(item["rows_input"] for item in dataset_manifests),
            "rows_kept": sum(item["rows_kept"] for item in dataset_manifests),
            "rows_dropped": sum(item["rows_dropped"] for item in dataset_manifests),
        },
        "safety_gate": {
            "test_split_rows": 0,
            "test_query_ids": 0,
            "test_positive_doc_ids": 0,
            "test_negative_doc_ids": 0,
            "prohibited_fiqa_6133_7733": 0,
            "status": "passed",
        },
        "caveats": [
            "Rows are reconstructed from exact normalized text and accepted only when the positive pair resolves to exactly one train qrel.",
            "Rows whose query ID, positive doc ID, or negative doc ID appears in test qrels are dropped from the train selection.",
            "Rows with unresolved or ambiguous positive/negative IDs are dropped instead of guessed.",
        ],
    }
    manifest_path = args.manifest or output_root / "provenance-safe-prep.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "built provenance-safe agreement prep: "
        f"rows_input={manifest['totals']['rows_input']} rows_kept={manifest['totals']['rows_kept']} "
        f"rows_dropped={manifest['totals']['rows_dropped']}"
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
