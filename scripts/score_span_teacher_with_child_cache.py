#!/usr/bin/env python3
"""Score span hard-negative rows with exported child-vector caches.

This maps repo-docs span hard-negative query/candidate texts back to the
run-local queries/chunks JSONL IDs, scores candidates against external
query/child-vector caches, and writes both import-compatible score rows and
hard-negative rows with `teacher_scores` attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score repo-docs span hard negatives using child-vector caches."
    )
    parser.add_argument("--hard-negatives", required=True, type=Path)
    parser.add_argument("--queries-jsonl", required=True, type=Path)
    parser.add_argument("--chunks-jsonl", required=True, type=Path)
    parser.add_argument("--child-vectors", required=True, type=Path)
    parser.add_argument("--query-vectors", required=True, type=Path)
    parser.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
        help="Hard-negative JSONL with teacher_scores attached.",
    )
    parser.add_argument(
        "--scores-jsonl",
        required=True,
        type=Path,
        help="Import-compatible per-example score rows.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Summary manifest path. Default: <output-jsonl>.manifest.json",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", default="repo-docs-span")
    parser.add_argument(
        "--score",
        choices=("cosine", "dot"),
        default="cosine",
        help="Score to compute. Cosine normalizes vectors before dot product.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum hard-negative examples to score; 0 means all.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write rows without teacher_scores when text/vector coverage is incomplete.",
    )
    parser.add_argument(
        "--missing-sample-limit",
        type=int,
        default=20,
        help="Maximum missing-match examples to include in the manifest.",
    )
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


def stable_text(value: str) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def query_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("query") or "").strip()


def row_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError(f"{path}:{line_number}: row is missing _id/id")
    return str(value)


def load_query_text_ids(path: Path) -> tuple[dict[str, str], int, int]:
    by_text: dict[str, str] = {}
    duplicate_texts = 0
    rows = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        item_id = row_id(row, path, line_number)
        text = stable_text(query_text(row))
        if not text:
            raise ValueError(f"{path}:{line_number}: empty query text for id={item_id!r}")
        if text in by_text:
            duplicate_texts += 1
            continue
        by_text[text] = item_id
    return by_text, rows, duplicate_texts


def load_chunk_ids(path: Path) -> tuple[dict[str, str], dict[str, str], int, int]:
    by_text: dict[str, str] = {}
    text_by_id: dict[str, str] = {}
    duplicate_texts = 0
    rows = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        item_id = row_id(row, path, line_number)
        text = stable_text(str(row.get("text") or ""))
        if not text:
            raise ValueError(f"{path}:{line_number}: empty chunk text for id={item_id!r}")
        text_by_id[item_id] = text
        if text in by_text:
            duplicate_texts += 1
            continue
        by_text[text] = item_id
    return by_text, text_by_id, rows, duplicate_texts


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def vector_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    value = row.get("child_id", row.get("id", row.get("_id")))
    if value is None:
        raise ValueError(f"{path}:{line_number}: vector row is missing child_id/id/_id")
    return str(value)


def load_vectors(path: Path, normalize: bool) -> tuple[dict[str, list[float]], int, int]:
    vectors: dict[str, list[float]] = {}
    rows = 0
    zero_norm = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        item_id = vector_id(row, path, line_number)
        raw = row.get("embedding", row.get("vector"))
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{path}:{line_number}: vector row has no embedding")
        vector = [float(value) for value in raw]
        if normalize:
            norm = vector_norm(vector)
            if norm == 0 or not math.isfinite(norm):
                zero_norm += 1
                continue
            vector = [value / norm for value in vector]
        vectors[item_id] = vector
    return vectors, rows, zero_norm


def score_vectors(query_vector: list[float], chunk_vector: list[float]) -> float:
    if len(query_vector) != len(chunk_vector):
        raise ValueError(
            f"vector dimension mismatch: query={len(query_vector)} chunk={len(chunk_vector)}"
        )
    return float(sum(a * b for a, b in zip(query_vector, chunk_vector)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_summary(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "pstdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }


def sample_missing(
    samples: list[dict[str, Any]],
    limit: int,
    kind: str,
    example_index: int,
    text: str,
    item_id: str | None = None,
) -> None:
    if len(samples) >= limit:
        return
    sample: dict[str, Any] = {
        "kind": kind,
        "example_index": example_index,
        "text_preview": stable_text(text)[:240],
    }
    if item_id:
        sample["id"] = item_id
    samples.append(sample)


def source_chunk_id(source: str) -> str:
    marker = ":"
    if marker not in source:
        return ""
    return source.rsplit(marker, 1)[-1]


def resolve_chunk_id(
    candidate: str,
    candidate_index: int,
    source: str,
    chunk_text_to_id: dict[str, str],
    chunk_text_by_id: dict[str, str],
) -> str | None:
    text = stable_text(candidate)
    if candidate_index == 0:
        hinted = source_chunk_id(source)
        if hinted and chunk_text_by_id.get(hinted) == text:
            return hinted
    return chunk_text_to_id.get(text)


def main() -> int:
    args = parse_args()
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")
    if args.missing_sample_limit < 0:
        raise SystemExit("--missing-sample-limit must be non-negative")

    for path in (
        args.hard_negatives,
        args.queries_jsonl,
        args.chunks_jsonl,
        args.child_vectors,
        args.query_vectors,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")

    manifest_path = args.manifest or args.output_jsonl.with_suffix(
        args.output_jsonl.suffix + ".manifest.json"
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    query_text_to_id, query_rows, duplicate_queries = load_query_text_ids(args.queries_jsonl)
    chunk_text_to_id, chunk_text_by_id, chunk_rows, duplicate_chunks = load_chunk_ids(
        args.chunks_jsonl
    )
    normalize_vectors = args.score == "cosine"
    child_vectors, child_vector_rows, zero_child_vectors = load_vectors(
        args.child_vectors, normalize_vectors
    )
    query_vectors, query_vector_rows, zero_query_vectors = load_vectors(
        args.query_vectors, normalize_vectors
    )

    examples_seen = 0
    examples_written = 0
    examples_scored = 0
    missing_examples = 0
    missing_query_text = 0
    missing_chunk_text = 0
    missing_query_vector = 0
    missing_child_vector = 0
    candidate_rows = 0
    import_score_rows = 0
    import_score_keys: set[tuple[str, str, str]] = set()
    all_scores: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    positive_top1 = 0
    missing_samples: list[dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out_handle, args.scores_jsonl.open(
        "w", encoding="utf-8"
    ) as score_handle:
        for line_number, example in iter_jsonl(args.hard_negatives):
            if args.max_examples > 0 and examples_seen >= args.max_examples:
                break
            examples_seen += 1
            query = str(example.get("query") or "")
            positive = str(example.get("positive") or "")
            negatives = [str(value or "") for value in example.get("negatives") or []]
            candidates = [positive] + negatives
            source = str(example.get("source") or "")

            query_id = query_text_to_id.get(stable_text(query))
            query_vector = query_vectors.get(query_id or "")
            if query_id is None:
                missing_query_text += 1
                sample_missing(
                    missing_samples,
                    args.missing_sample_limit,
                    "query_text",
                    examples_seen - 1,
                    query,
                )
            elif query_vector is None:
                missing_query_vector += 1
                sample_missing(
                    missing_samples,
                    args.missing_sample_limit,
                    "query_vector",
                    examples_seen - 1,
                    query,
                    query_id,
                )

            scores: list[float] = []
            complete = query_vector is not None
            for candidate_index, candidate in enumerate(candidates):
                chunk_id = resolve_chunk_id(
                    candidate,
                    candidate_index,
                    source,
                    chunk_text_to_id,
                    chunk_text_by_id,
                )
                child_vector = child_vectors.get(chunk_id or "")
                if chunk_id is None:
                    missing_chunk_text += 1
                    complete = False
                    sample_missing(
                        missing_samples,
                        args.missing_sample_limit,
                        "chunk_text",
                        examples_seen - 1,
                        candidate,
                    )
                    continue
                if child_vector is None:
                    missing_child_vector += 1
                    complete = False
                    sample_missing(
                        missing_samples,
                        args.missing_sample_limit,
                        "child_vector",
                        examples_seen - 1,
                        candidate,
                        chunk_id,
                    )
                    continue
                if query_vector is not None:
                    score = score_vectors(query_vector, child_vector)
                    scores.append(score)
                    all_scores.append(score)
                    if candidate_index == 0:
                        positive_scores.append(score)
                    else:
                        negative_scores.append(score)
                    candidate_rows += 1

            if complete and len(scores) == len(candidates):
                example["teacher_scores"] = scores
                for candidate, score in zip(candidates, scores):
                    import_key = (source, query, candidate)
                    if import_key in import_score_keys:
                        continue
                    import_score_keys.add(import_key)
                    score_handle.write(
                        json.dumps(
                            {
                                "source": source,
                                "query": query,
                                "candidate": candidate,
                                "score": score,
                                "score_scale": args.score,
                                "teacher_model_id": args.model_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    import_score_rows += 1
                examples_scored += 1
                if negatives:
                    best_negative = max(scores[1:])
                    margins.append(scores[0] - best_negative)
                    if scores[0] >= best_negative:
                        positive_top1 += 1
            else:
                missing_examples += 1
                example.pop("teacher_scores", None)
                if not args.allow_missing:
                    raise SystemExit(
                        f"{args.hard_negatives}:{line_number}: incomplete child-cache coverage "
                        f"for example index {examples_seen - 1}; use --allow-missing to keep going"
                    )

            out_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            examples_written += 1

    manifest = {
        "schema": "manta.span_child_vector_cache_teacher_scores.v1",
        "dataset": args.dataset,
        "teacher_model_id": args.model_id,
        "score_scale": args.score,
        "hard_negatives": str(args.hard_negatives),
        "queries_jsonl": str(args.queries_jsonl),
        "chunks_jsonl": str(args.chunks_jsonl),
        "child_vectors": str(args.child_vectors),
        "query_vectors": str(args.query_vectors),
        "output_jsonl": str(args.output_jsonl),
        "scores_jsonl": str(args.scores_jsonl),
        "sha256": {
            "hard_negatives": sha256_file(args.hard_negatives),
            "queries_jsonl": sha256_file(args.queries_jsonl),
            "chunks_jsonl": sha256_file(args.chunks_jsonl),
            "child_vectors": sha256_file(args.child_vectors),
            "query_vectors": sha256_file(args.query_vectors),
        },
        "coverage": {
            "examples_seen": examples_seen,
            "examples_written": examples_written,
            "examples_scored": examples_scored,
            "missing_examples": missing_examples,
            "candidate_rows_scored": candidate_rows,
            "import_score_rows": import_score_rows,
            "missing_query_text": missing_query_text,
            "missing_chunk_text": missing_chunk_text,
            "missing_query_vector": missing_query_vector,
            "missing_child_vector": missing_child_vector,
        },
        "corpus": {
            "query_rows": query_rows,
            "chunk_rows": chunk_rows,
            "duplicate_query_texts": duplicate_queries,
            "duplicate_chunk_texts": duplicate_chunks,
        },
        "vectors": {
            "child_vector_rows": child_vector_rows,
            "query_vector_rows": query_vector_rows,
            "zero_child_vectors": zero_child_vectors,
            "zero_query_vectors": zero_query_vectors,
            "normalized_for_scoring": normalize_vectors,
        },
        "scores": {
            "all": score_summary(all_scores),
            "positive": score_summary(positive_scores),
            "negative": score_summary(negative_scores),
            "margin": score_summary(margins),
            "positive_top1_rate": positive_top1 / len(margins) if margins else 0.0,
        },
        "missing_samples": missing_samples,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "scored span child-vector teacher: "
        f"examples={examples_seen} scored={examples_scored} "
        f"missing={missing_examples} candidate_rows={candidate_rows}"
    )
    print(f"output_jsonl: {args.output_jsonl}")
    print(f"scores_jsonl: {args.scores_jsonl}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
