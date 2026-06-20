#!/usr/bin/env python3
"""Score LongEmbed parent hard negatives with external child-vector caches.

External LongEmbed caches store document vectors at child-chunk granularity.
This script reconstructs those child IDs from the cache manifest and dataset
corpus, scores each parent candidate by max child score, and attaches averaged
teacher_scores only when all configured teachers have full coverage and each
teacher ranks the labeled positive top-1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple


class DocumentChunk(NamedTuple):
    parent_id: str
    child_id: str
    text: str


@dataclass
class DatasetIndex:
    name: str
    dataset_dir: Path
    corpus_path: Path
    queries_path: Path
    doc_text_to_ids: dict[str, list[str]]
    query_text_to_ids: dict[str, list[str]]
    query_ids: set[str]
    doc_ids: set[str]
    corpus_rows: int
    query_rows: int
    duplicate_doc_texts: int
    duplicate_query_texts: int


@dataclass
class TeacherDatasetCache:
    teacher_label: str
    dataset_name: str
    cache_dir: Path
    manifest_path: Path
    child_vectors_path: Path
    query_vectors_path: Path
    manifest: dict[str, Any]
    child_vectors_by_parent: dict[str, list[list[float]]]
    query_vectors: dict[str, list[float]]
    child_vector_rows: int
    query_vector_rows: int
    zero_child_vectors: int
    zero_query_vectors: int
    reconstructed_child_count: int
    missing_reconstructed_child_vectors: int


@dataclass
class TeacherStats:
    examples_complete: int = 0
    examples_missing: int = 0
    positive_top1: int = 0
    candidate_rows_scored: int = 0
    margins: list[float] = field(default_factory=list)
    all_scores: list[float] = field(default_factory=list)
    positive_scores: list[float] = field(default_factory=list)
    negative_scores: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach agreement-filtered teacher_scores from LongEmbed child-vector caches."
    )
    parser.add_argument("--hard-negatives", required=True, type=Path)
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Dataset mapping. Repeat for qmsum and 2wikimqa.",
    )
    parser.add_argument(
        "--teacher-cache",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Teacher cache root containing <dataset>/manifest.json. Repeat for each teacher.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=None,
        help="Import-compatible averaged score rows. Default: <output-jsonl>.scores.jsonl",
    )
    parser.add_argument(
        "--teacher-scores-jsonl",
        type=Path,
        default=None,
        help="Per-teacher diagnostic score rows. Default: <output-jsonl>.per-teacher-scores.jsonl",
    )
    parser.add_argument("--score", choices=("cosine", "dot"), default="cosine")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--missing-sample-limit", type=int, default=30)
    return parser.parse_args()


def parse_mapping(values: list[str], flag: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"{flag} has empty name: {value!r}")
        if name in out:
            raise SystemExit(f"duplicate {flag} name {name!r}")
        out[name] = Path(raw_path)
    if not out:
        raise SystemExit(f"at least one {flag} is required")
    return out


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


def add_text_id(mapping: dict[str, list[str]], text: str, item_id: str) -> None:
    mapping.setdefault(stable_text(text), []).append(item_id)


def load_dataset_index(name: str, dataset_dir: Path) -> DatasetIndex:
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    for path in (corpus_path, queries_path):
        if not path.is_file():
            raise SystemExit(f"missing dataset file: {path}")

    doc_text_to_ids: dict[str, list[str]] = {}
    query_text_to_ids: dict[str, list[str]] = {}
    doc_ids: set[str] = set()
    query_ids: set[str] = set()
    corpus_rows = 0
    query_rows = 0

    for line_number, row in iter_jsonl(corpus_path):
        corpus_rows += 1
        item_id = row_id(row, corpus_path, line_number)
        doc_ids.add(item_id)
        text = corpus_text(row)
        if not stable_text(text):
            raise ValueError(f"{corpus_path}:{line_number}: empty corpus text for {item_id!r}")
        add_text_id(doc_text_to_ids, text, item_id)

    for line_number, row in iter_jsonl(queries_path):
        query_rows += 1
        item_id = row_id(row, queries_path, line_number)
        query_ids.add(item_id)
        text = query_text(row)
        if not stable_text(text):
            raise ValueError(f"{queries_path}:{line_number}: empty query text for {item_id!r}")
        add_text_id(query_text_to_ids, text, item_id)

    return DatasetIndex(
        name=name,
        dataset_dir=dataset_dir,
        corpus_path=corpus_path,
        queries_path=queries_path,
        doc_text_to_ids=doc_text_to_ids,
        query_text_to_ids=query_text_to_ids,
        query_ids=query_ids,
        doc_ids=doc_ids,
        corpus_rows=corpus_rows,
        query_rows=query_rows,
        duplicate_doc_texts=sum(1 for ids in doc_text_to_ids.values() if len(ids) > 1),
        duplicate_query_texts=sum(1 for ids in query_text_to_ids.values() if len(ids) > 1),
    )


def chunk_document_text(
    parent_id: str,
    text: str,
    chunk_words: int,
    overlap: int,
    min_words: int,
) -> list[DocumentChunk]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0 or len(words) <= chunk_words:
        return [DocumentChunk(parent_id, f"{parent_id}#chunk-0000", " ".join(words))]
    step = chunk_words - overlap
    if step <= 0:
        raise ValueError(f"invalid chunk config for {parent_id}: words={chunk_words} overlap={overlap}")

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_words_list = words[start:end]
        if chunks and len(chunk_words_list) < min_words:
            break
        chunks.append(
            DocumentChunk(
                parent_id,
                f"{parent_id}#chunk-{len(chunks):04d}",
                " ".join(chunk_words_list),
            )
        )
        if end >= len(words):
            break
        start += step
    if not chunks:
        chunks.append(DocumentChunk(parent_id, f"{parent_id}#chunk-0000", " ".join(words)))
    return chunks


def reconstruct_chunks(dataset: DatasetIndex, manifest: dict[str, Any]) -> list[DocumentChunk]:
    chunk_words = int(manifest.get("document_chunk_words") or 0)
    overlap = int(manifest.get("document_chunk_overlap") or 0)
    min_words = int(manifest.get("document_chunk_min_words") or 1)
    chunks: list[DocumentChunk] = []
    for line_number, row in iter_jsonl(dataset.corpus_path):
        parent_id = row_id(row, dataset.corpus_path, line_number)
        chunks.extend(chunk_document_text(parent_id, corpus_text(row), chunk_words, overlap, min_words))
    return chunks


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize_vector(vector: list[float]) -> list[float] | None:
    norm = vector_norm(vector)
    if norm == 0 or not math.isfinite(norm):
        return None
    return [value / norm for value in vector]


def load_query_vectors(path: Path, normalize: bool) -> tuple[dict[str, list[float]], int, int]:
    vectors: dict[str, list[float]] = {}
    rows = 0
    zero_norm = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        value = row.get("id", row.get("_id"))
        if value is None:
            raise ValueError(f"{path}:{line_number}: query vector row is missing id")
        raw = row.get("embedding", row.get("vector"))
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{path}:{line_number}: vector row has no embedding")
        vector = [float(item) for item in raw]
        if normalize:
            normalized = normalize_vector(vector)
            if normalized is None:
                zero_norm += 1
                continue
            vector = normalized
        vectors[str(value)] = vector
    return vectors, rows, zero_norm


def load_child_vectors(
    path: Path,
    reconstructed_child_ids: set[str],
    normalize: bool,
) -> tuple[dict[str, list[float]], int, int]:
    vectors: dict[str, list[float]] = {}
    rows = 0
    zero_norm = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        child_id = row.get("child_id", row.get("id", row.get("_id")))
        if child_id is None:
            raise ValueError(f"{path}:{line_number}: child vector row is missing child_id/id/_id")
        child_id = str(child_id)
        if child_id not in reconstructed_child_ids:
            raise ValueError(f"{path}:{line_number}: unknown child_id {child_id!r} for reconstructed manifest chunks")
        raw = row.get("embedding", row.get("vector"))
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{path}:{line_number}: vector row has no embedding")
        vector = [float(item) for item in raw]
        if normalize:
            normalized = normalize_vector(vector)
            if normalized is None:
                zero_norm += 1
                continue
            vector = normalized
        vectors[child_id] = vector
    return vectors, rows, zero_norm


def load_teacher_dataset_cache(
    teacher_label: str,
    cache_root: Path,
    dataset: DatasetIndex,
    normalize: bool,
) -> TeacherDatasetCache:
    cache_dir = cache_root / dataset.name
    manifest_path = cache_dir / "manifest.json"
    child_vectors_path = cache_dir / "child-doc-vectors.jsonl"
    query_vectors_path = cache_dir / "query-vectors.jsonl"
    for path in (manifest_path, child_vectors_path, query_vectors_path):
        if not path.is_file():
            raise SystemExit(f"missing teacher cache file: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = reconstruct_chunks(dataset, manifest)
    child_ids = {chunk.child_id for chunk in chunks}
    child_vectors, child_vector_rows, zero_child_vectors = load_child_vectors(
        child_vectors_path, child_ids, normalize
    )
    query_vectors, query_vector_rows, zero_query_vectors = load_query_vectors(query_vectors_path, normalize)
    child_vectors_by_parent: dict[str, list[list[float]]] = defaultdict(list)
    missing_reconstructed = 0
    for chunk in chunks:
        vector = child_vectors.get(chunk.child_id)
        if vector is None:
            missing_reconstructed += 1
            continue
        child_vectors_by_parent[chunk.parent_id].append(vector)
    return TeacherDatasetCache(
        teacher_label=teacher_label,
        dataset_name=dataset.name,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        child_vectors_path=child_vectors_path,
        query_vectors_path=query_vectors_path,
        manifest=manifest,
        child_vectors_by_parent=dict(child_vectors_by_parent),
        query_vectors=query_vectors,
        child_vector_rows=child_vector_rows,
        query_vector_rows=query_vector_rows,
        zero_child_vectors=zero_child_vectors,
        zero_query_vectors=zero_query_vectors,
        reconstructed_child_count=len(chunks),
        missing_reconstructed_child_vectors=missing_reconstructed,
    )


def score_vectors(query_vector: list[float], doc_vector: list[float]) -> float:
    if len(query_vector) != len(doc_vector):
        raise ValueError(f"vector dimension mismatch: query={len(query_vector)} doc={len(doc_vector)}")
    return float(sum(a * b for a, b in zip(query_vector, doc_vector)))


def score_parent(query_vector: list[float], child_vectors: list[list[float]]) -> float:
    if not child_vectors:
        raise ValueError("cannot score parent without child vectors")
    return max(score_vectors(query_vector, child_vector) for child_vector in child_vectors)


def extract_dataset_hints(example: dict[str, Any], dataset_names: set[str]) -> set[str]:
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    fields: list[Any] = [
        example.get("source"),
        metadata.get("source"),
        metadata.get("sources"),
        metadata.get("source_files"),
    ]
    hints: set[str] = set()
    for field in fields:
        values = field if isinstance(field, list) else [field]
        for value in values:
            if value is None:
                continue
            tokens = [token for token in re.split(r"[^A-Za-z0-9_]+", str(value)) if token]
            for token in tokens:
                if token in dataset_names:
                    hints.add(token)
    return hints


def unique_lookup(mapping: dict[str, list[str]], text: str) -> str | None:
    ids = mapping.get(stable_text(text)) or []
    if len(ids) != 1:
        return None
    return ids[0]


def infer_dataset(
    example: dict[str, Any],
    datasets: dict[str, DatasetIndex],
    line_number: int,
    input_path: Path,
) -> str:
    hints = extract_dataset_hints(example, set(datasets))
    if len(hints) == 1:
        return next(iter(hints))
    if len(hints) > 1:
        raise ValueError(f"{input_path}:{line_number}: ambiguous dataset hints: {sorted(hints)}")

    query = str(example.get("query") or "")
    positive = str(example.get("positive") or "")
    matching: set[str] = set()
    for name, dataset in datasets.items():
        if unique_lookup(dataset.query_text_to_ids, query) and unique_lookup(dataset.doc_text_to_ids, positive):
            matching.add(name)
    if len(matching) == 1:
        return next(iter(matching))
    raise ValueError(f"{input_path}:{line_number}: cannot infer dataset from metadata or text")


def resolve_query_id(example: dict[str, Any], dataset: DatasetIndex) -> str | None:
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    query_id = metadata.get("query_id")
    if query_id is not None and str(query_id) in dataset.query_ids:
        return str(query_id)
    return unique_lookup(dataset.query_text_to_ids, str(example.get("query") or ""))


def resolve_parent_id(candidate: str, dataset: DatasetIndex) -> str | None:
    return unique_lookup(dataset.doc_text_to_ids, candidate)


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


def sample_missing(samples: list[dict[str, Any]], limit: int, sample: dict[str, Any]) -> None:
    if len(samples) < limit:
        samples.append(sample)


def teacher_score_example(
    teacher_cache: TeacherDatasetCache,
    dataset: DatasetIndex,
    query_id: str | None,
    candidates: list[str],
) -> tuple[list[float] | None, str | None]:
    if query_id is None:
        return None, "query"
    query_vector = teacher_cache.query_vectors.get(query_id)
    if query_vector is None:
        return None, "query_vector"
    scores: list[float] = []
    for candidate in candidates:
        parent_id = resolve_parent_id(candidate, dataset)
        if parent_id is None:
            return None, "candidate_parent"
        child_vectors = teacher_cache.child_vectors_by_parent.get(parent_id) or []
        if not child_vectors:
            return None, "candidate_child_vectors"
        scores.append(score_parent(query_vector, child_vectors))
    return scores, None


def positive_top1(scores: list[float]) -> bool:
    return len(scores) == 1 or scores[0] >= max(scores[1:])


def main() -> int:
    args = parse_args()
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")
    if args.missing_sample_limit < 0:
        raise SystemExit("--missing-sample-limit must be non-negative")
    if not args.hard_negatives.is_file():
        raise SystemExit(f"missing input file: {args.hard_negatives}")

    dataset_dirs = parse_mapping(args.dataset_dir, "--dataset-dir")
    teacher_roots = parse_mapping(args.teacher_cache, "--teacher-cache")
    manifest_path = args.manifest or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")
    scores_path = args.scores_jsonl or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".scores.jsonl")
    teacher_scores_path = args.teacher_scores_jsonl or args.output_jsonl.with_suffix(
        args.output_jsonl.suffix + ".per-teacher-scores.jsonl"
    )
    for path in (args.output_jsonl, manifest_path, scores_path, teacher_scores_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    normalize = args.score == "cosine"
    datasets = {name: load_dataset_index(name, path) for name, path in dataset_dirs.items()}
    teacher_caches: dict[str, dict[str, TeacherDatasetCache]] = {}
    for teacher_label, cache_root in teacher_roots.items():
        teacher_caches[teacher_label] = {
            dataset_name: load_teacher_dataset_cache(teacher_label, cache_root, dataset, normalize)
            for dataset_name, dataset in datasets.items()
        }

    teacher_stats = {teacher_label: TeacherStats() for teacher_label in teacher_roots}
    dataset_counts = {name: {"examples": 0, "scored": 0, "cleared": 0} for name in datasets}
    examples_seen = 0
    examples_written = 0
    examples_with_average_scores = 0
    cleared_missing_coverage = 0
    cleared_teacher_disagreement = 0
    averaged_score_rows = 0
    per_teacher_score_rows = 0
    missing_samples: list[dict[str, Any]] = []
    average_margins: list[float] = []
    average_scores_all: list[float] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out_handle, scores_path.open(
        "w", encoding="utf-8"
    ) as score_handle, teacher_scores_path.open("w", encoding="utf-8") as teacher_score_handle:
        for line_number, example in iter_jsonl(args.hard_negatives):
            if args.max_examples > 0 and examples_seen >= args.max_examples:
                break
            examples_seen += 1
            dataset_name = infer_dataset(example, datasets, line_number, args.hard_negatives)
            dataset = datasets[dataset_name]
            dataset_counts[dataset_name]["examples"] += 1
            query = str(example.get("query") or "")
            positive = str(example.get("positive") or "")
            negatives = [str(value or "") for value in example.get("negatives") or []]
            candidates = [positive] + negatives
            source = str(example.get("source") or "")
            query_id = resolve_query_id(example, dataset)

            per_teacher_scores: dict[str, list[float]] = {}
            missing_reasons: dict[str, str] = {}
            teacher_agreement = True
            for teacher_label, caches_by_dataset in teacher_caches.items():
                scores, missing_reason = teacher_score_example(
                    caches_by_dataset[dataset_name], dataset, query_id, candidates
                )
                stats = teacher_stats[teacher_label]
                if scores is None:
                    stats.examples_missing += 1
                    missing_reasons[teacher_label] = missing_reason or "unknown"
                    teacher_agreement = False
                    continue
                per_teacher_scores[teacher_label] = scores
                stats.examples_complete += 1
                stats.candidate_rows_scored += len(scores)
                stats.all_scores.extend(scores)
                if scores:
                    stats.positive_scores.append(scores[0])
                    stats.negative_scores.extend(scores[1:])
                if len(scores) > 1:
                    margin = scores[0] - max(scores[1:])
                    stats.margins.append(margin)
                    if margin >= 0:
                        stats.positive_top1 += 1
                    else:
                        teacher_agreement = False
                if not positive_top1(scores):
                    teacher_agreement = False
                for candidate, score in zip(candidates, scores):
                    teacher_score_handle.write(
                        json.dumps(
                            {
                                "source": source,
                                "dataset": dataset_name,
                                "query_id": query_id,
                                "query": query,
                                "candidate": candidate,
                                "score": score,
                                "score_scale": args.score,
                                "teacher_model_id": caches_by_dataset[dataset_name].manifest.get(
                                    "model_name", teacher_label
                                ),
                                "teacher_label": teacher_label,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    per_teacher_score_rows += 1

            if len(per_teacher_scores) == len(teacher_caches) and teacher_agreement:
                averaged = [
                    statistics.fmean(scores[index] for scores in per_teacher_scores.values())
                    for index in range(len(candidates))
                ]
                example["teacher_scores"] = averaged
                examples_with_average_scores += 1
                dataset_counts[dataset_name]["scored"] += 1
                average_scores_all.extend(averaged)
                if len(averaged) > 1:
                    average_margins.append(averaged[0] - max(averaged[1:]))
                for candidate, score in zip(candidates, averaged):
                    score_handle.write(
                        json.dumps(
                            {
                                "source": source,
                                "dataset": dataset_name,
                                "query_id": query_id,
                                "query": query,
                                "candidate": candidate,
                                "score": score,
                                "score_scale": args.score,
                                "teacher_model_id": "+".join(sorted(teacher_roots)),
                                "teacher_labels": sorted(teacher_roots),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    averaged_score_rows += 1
            else:
                example.pop("teacher_scores", None)
                dataset_counts[dataset_name]["cleared"] += 1
                if len(per_teacher_scores) != len(teacher_caches):
                    cleared_missing_coverage += 1
                    sample_missing(
                        missing_samples,
                        args.missing_sample_limit,
                        {
                            "kind": "missing_coverage",
                            "example_index": examples_seen - 1,
                            "line_number": line_number,
                            "dataset": dataset_name,
                            "query_id": query_id,
                            "reasons": missing_reasons,
                            "source": source,
                        },
                    )
                else:
                    cleared_teacher_disagreement += 1

            out_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            examples_written += 1

    manifest = {
        "schema": "manta.longembed_child_cache_teacher_score_bridge.v1",
        "quality_claim": False,
        "score_scale": args.score,
        "hard_negatives": str(args.hard_negatives),
        "output_jsonl": str(args.output_jsonl),
        "scores_jsonl": str(scores_path),
        "teacher_scores_jsonl": str(teacher_scores_path),
        "teachers": sorted(teacher_roots),
        "datasets": sorted(datasets),
        "coverage": {
            "examples_seen": examples_seen,
            "examples_written": examples_written,
            "examples_with_teacher_scores": examples_with_average_scores,
            "examples_cleared": examples_seen - examples_with_average_scores,
            "cleared_missing_coverage": cleared_missing_coverage,
            "cleared_teacher_disagreement": cleared_teacher_disagreement,
            "averaged_score_rows": averaged_score_rows,
            "per_teacher_score_rows": per_teacher_score_rows,
            "dataset_counts": dataset_counts,
        },
        "teacher_agreement": {
            label: {
                "examples_complete": stats.examples_complete,
                "examples_missing": stats.examples_missing,
                "positive_top1": stats.positive_top1,
                "positive_top1_rate": stats.positive_top1 / stats.examples_complete
                if stats.examples_complete
                else 0.0,
                "candidate_rows_scored": stats.candidate_rows_scored,
                "margin": score_summary(stats.margins),
                "all_scores": score_summary(stats.all_scores),
                "positive_scores": score_summary(stats.positive_scores),
                "negative_scores": score_summary(stats.negative_scores),
            }
            for label, stats in teacher_stats.items()
        },
        "averaged_scores": {
            "all_scores": score_summary(average_scores_all),
            "margin": score_summary(average_margins),
            "agreement_keep_rate": examples_with_average_scores / examples_seen if examples_seen else 0.0,
            "positive_top1_rate": 1.0 if examples_with_average_scores else 0.0,
        },
        "dataset_indexes": {
            name: {
                "dataset_dir": str(dataset.dataset_dir),
                "corpus_rows": dataset.corpus_rows,
                "query_rows": dataset.query_rows,
                "duplicate_doc_texts": dataset.duplicate_doc_texts,
                "duplicate_query_texts": dataset.duplicate_query_texts,
            }
            for name, dataset in datasets.items()
        },
        "cache_indexes": {
            teacher_label: {
                dataset_name: {
                    "cache_dir": str(cache.cache_dir),
                    "model_name": cache.manifest.get("model_name"),
                    "output_dim": cache.manifest.get("output_dim"),
                    "document_chunk_words": cache.manifest.get("document_chunk_words"),
                    "document_chunk_overlap": cache.manifest.get("document_chunk_overlap"),
                    "document_chunk_min_words": cache.manifest.get("document_chunk_min_words"),
                    "reconstructed_child_count": cache.reconstructed_child_count,
                    "child_vector_rows": cache.child_vector_rows,
                    "query_vector_rows": cache.query_vector_rows,
                    "missing_reconstructed_child_vectors": cache.missing_reconstructed_child_vectors,
                    "zero_child_vectors": cache.zero_child_vectors,
                    "zero_query_vectors": cache.zero_query_vectors,
                }
                for dataset_name, cache in caches_by_dataset.items()
            }
            for teacher_label, caches_by_dataset in teacher_caches.items()
        },
        "sha256": {
            "hard_negatives": sha256_file(args.hard_negatives),
            **{
                f"dataset:{name}:corpus": sha256_file(dataset.corpus_path)
                for name, dataset in datasets.items()
            },
            **{
                f"dataset:{name}:queries": sha256_file(dataset.queries_path)
                for name, dataset in datasets.items()
            },
            **{
                f"cache:{teacher}:{dataset_name}:manifest": sha256_file(cache.manifest_path)
                for teacher, caches in teacher_caches.items()
                for dataset_name, cache in caches.items()
            },
            **{
                f"cache:{teacher}:{dataset_name}:child_vectors": sha256_file(cache.child_vectors_path)
                for teacher, caches in teacher_caches.items()
                for dataset_name, cache in caches.items()
            },
            **{
                f"cache:{teacher}:{dataset_name}:query_vectors": sha256_file(cache.query_vectors_path)
                for teacher, caches in teacher_caches.items()
                for dataset_name, cache in caches.items()
            },
        },
        "missing_samples": missing_samples,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "scored longembed child-cache teachers: "
        f"examples={examples_seen} with_teacher_scores={examples_with_average_scores} "
        f"cleared={examples_seen - examples_with_average_scores}"
    )
    print(f"output_jsonl: {args.output_jsonl}")
    print(f"scores_jsonl: {scores_path}")
    print(f"teacher_scores_jsonl: {teacher_scores_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
