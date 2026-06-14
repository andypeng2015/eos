#!/usr/bin/env python3
"""Score hard-negative examples with a BEIR vector cache teacher.

This bridges external vector-cache teachers into the existing Eos
`import-teacher-scores`/`audit-teacher-scores` flow. It maps example query and
candidate texts back to BEIR IDs, loads cached query/document embeddings, scores
each positive and negative, then writes both import-compatible score rows and a
hard-negative JSONL with `teacher_scores` attached.
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
        description="Score text hard negatives using BEIR document/query vector caches."
    )
    parser.add_argument("--hard-negatives", required=True, type=Path)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="BEIR-style directory containing corpus.jsonl and queries.jsonl.",
    )
    parser.add_argument("--doc-vectors", required=True, type=Path)
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
    parser.add_argument("--dataset", default="scifact")
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


def corpus_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return title + "\n" + text
    return title or text


def query_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("query") or "").strip()


def row_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError(f"{path}:{line_number}: row is missing _id/id")
    return str(value)


def load_text_ids(path: Path, text_fn) -> tuple[dict[str, str], int, int]:
    by_text: dict[str, str] = {}
    duplicate_texts = 0
    rows = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        item_id = row_id(row, path, line_number)
        text = stable_text(text_fn(row))
        if not text:
            raise ValueError(f"{path}:{line_number}: empty text for id={item_id!r}")
        if text in by_text:
            duplicate_texts += 1
            continue
        by_text[text] = item_id
    return by_text, rows, duplicate_texts


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def load_vectors(path: Path, normalize: bool) -> tuple[dict[str, list[float]], int, int]:
    vectors: dict[str, list[float]] = {}
    rows = 0
    zero_norm = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        item_id = row.get("id", row.get("_id"))
        if item_id is None:
            raise ValueError(f"{path}:{line_number}: vector row is missing id")
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
        vectors[str(item_id)] = vector
    return vectors, rows, zero_norm


def score_vectors(query_vector: list[float], doc_vector: list[float]) -> float:
    if len(query_vector) != len(doc_vector):
        raise ValueError(
            f"vector dimension mismatch: query={len(query_vector)} doc={len(doc_vector)}"
        )
    return float(sum(a * b for a, b in zip(query_vector, doc_vector)))


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
    samples: list[dict[str, Any]], limit: int, kind: str, example_index: int, text: str
) -> None:
    if len(samples) >= limit:
        return
    samples.append(
        {
            "kind": kind,
            "example_index": example_index,
            "text_preview": stable_text(text)[:240],
        }
    )


def main() -> int:
    args = parse_args()
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")
    if args.missing_sample_limit < 0:
        raise SystemExit("--missing-sample-limit must be non-negative")

    corpus_path = args.dataset_dir / "corpus.jsonl"
    queries_path = args.dataset_dir / "queries.jsonl"
    for path in (
        args.hard_negatives,
        corpus_path,
        queries_path,
        args.doc_vectors,
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

    query_text_to_id, query_rows, duplicate_queries = load_text_ids(queries_path, query_text)
    doc_text_to_id, doc_rows, duplicate_docs = load_text_ids(corpus_path, corpus_text)
    normalize_vectors = args.score == "cosine"
    doc_vectors, doc_vector_rows, zero_doc_vectors = load_vectors(
        args.doc_vectors, normalize_vectors
    )
    query_vectors, query_vector_rows, zero_query_vectors = load_vectors(
        args.query_vectors, normalize_vectors
    )

    examples_seen = 0
    examples_written = 0
    examples_scored = 0
    missing_examples = 0
    missing_query_text = 0
    missing_doc_text = 0
    missing_query_vector = 0
    missing_doc_vector = 0
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
                sample_missing(missing_samples, args.missing_sample_limit, "query_text", examples_seen - 1, query)
            elif query_vector is None:
                missing_query_vector += 1
                sample_missing(missing_samples, args.missing_sample_limit, "query_vector", examples_seen - 1, query)

            scores: list[float] = []
            complete = query_vector is not None
            for candidate_index, candidate in enumerate(candidates):
                doc_id = doc_text_to_id.get(stable_text(candidate))
                doc_vector = doc_vectors.get(doc_id or "")
                if doc_id is None:
                    missing_doc_text += 1
                    complete = False
                    sample_missing(
                        missing_samples,
                        args.missing_sample_limit,
                        "doc_text",
                        examples_seen - 1,
                        candidate,
                    )
                    continue
                if doc_vector is None:
                    missing_doc_vector += 1
                    complete = False
                    sample_missing(
                        missing_samples,
                        args.missing_sample_limit,
                        "doc_vector",
                        examples_seen - 1,
                        candidate,
                    )
                    continue
                if query_vector is not None:
                    score = score_vectors(query_vector, doc_vector)
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
                        f"{args.hard_negatives}:{line_number}: incomplete cache coverage "
                        f"for example index {examples_seen - 1}; use --allow-missing to keep going"
                    )

            out_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            examples_written += 1

    manifest = {
        "schema": "manta.vector_cache_teacher_scores.v1",
        "dataset": args.dataset,
        "teacher_model_id": args.model_id,
        "score_scale": args.score,
        "hard_negatives": str(args.hard_negatives),
        "output_jsonl": str(args.output_jsonl),
        "scores_jsonl": str(args.scores_jsonl),
        "dataset_dir": str(args.dataset_dir),
        "doc_vectors": str(args.doc_vectors),
        "query_vectors": str(args.query_vectors),
        "sha256": {
            "hard_negatives": sha256_file(args.hard_negatives),
            "doc_vectors": sha256_file(args.doc_vectors),
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
            "missing_doc_text": missing_doc_text,
            "missing_query_vector": missing_query_vector,
            "missing_doc_vector": missing_doc_vector,
        },
        "beir": {
            "query_rows": query_rows,
            "corpus_rows": doc_rows,
            "duplicate_query_texts": duplicate_queries,
            "duplicate_doc_texts": duplicate_docs,
        },
        "vectors": {
            "doc_vector_rows": doc_vector_rows,
            "query_vector_rows": query_vector_rows,
            "zero_doc_vectors": zero_doc_vectors,
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
        "scored vector-cache teacher: "
        f"examples={examples_seen} scored={examples_scored} "
        f"missing={missing_examples} candidate_rows={candidate_rows}"
    )
    print(f"output_jsonl: {args.output_jsonl}")
    print(f"scores_jsonl: {args.scores_jsonl}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
