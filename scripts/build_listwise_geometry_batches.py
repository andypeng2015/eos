#!/usr/bin/env python3
"""Build listwise teacher-geometry batches from hard negatives and vector caches.

The output preserves a teacher's query-document similarity matrix for each
batch without changing trainer code. Rows are strict by default: unresolved
BEIR text/IDs or missing vectors fail the run. With --allow-missing, incomplete
examples are dropped and coverage diagnostics are recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "eos.listwise_geometry_batch.v1"
MANIFEST_SCHEMA = "eos.listwise_geometry_batches_manifest.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-negatives", required=True, type=Path)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="BEIR-style directory containing corpus.jsonl and queries.jsonl.",
    )
    parser.add_argument("--doc-vectors", required=True, type=Path)
    parser.add_argument("--query-vectors", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        "--teacher-model-id",
        dest="model_id",
        default="",
        help="Teacher model identifier to write as teacher_model_id.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score", choices=("cosine", "dot"), default="cosine")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum hard-negative examples to inspect; 0 means all.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Drop examples with unresolved text/IDs or missing vectors instead of failing.",
    )
    parser.add_argument(
        "--skip-empty-beir-text",
        action="store_true",
        help="Skip empty BEIR corpus/query rows instead of failing while loading text maps.",
    )
    parser.add_argument(
        "--missing-sample-limit",
        type=int,
        default=20,
        help="Maximum missing coverage samples to include in the manifest.",
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


def stable_text(value: Any) -> str:
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


def load_text_maps(
    path: Path, text_fn, skip_empty: bool
) -> tuple[dict[str, str], dict[str, str], int, int, int]:
    text_to_id: dict[str, str] = {}
    id_to_text: dict[str, str] = {}
    duplicate_texts = 0
    empty_texts = 0
    rows = 0
    for line_number, row in iter_jsonl(path):
        rows += 1
        item_id = row_id(row, path, line_number)
        text = stable_text(text_fn(row))
        if not text:
            empty_texts += 1
            if skip_empty:
                continue
            raise ValueError(f"{path}:{line_number}: empty text for id={item_id!r}")
        id_to_text[item_id] = text
        if text in text_to_id:
            duplicate_texts += 1
            continue
        text_to_id[text] = item_id
    return text_to_id, id_to_text, rows, duplicate_texts, empty_texts


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
    score = float(sum(a * b for a, b in zip(query_vector, doc_vector)))
    if not math.isfinite(score):
        raise ValueError("non-finite teacher similarity score")
    return score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_missing_sample(
    samples: list[dict[str, Any]],
    limit: int,
    *,
    kind: str,
    example_index: int,
    line_number: int,
    item_id: str = "",
    text: str = "",
) -> None:
    if len(samples) >= limit:
        return
    sample: dict[str, Any] = {
        "kind": kind,
        "example_index": example_index,
        "line_number": line_number,
    }
    if item_id:
        sample["id"] = item_id
    if text:
        sample["text_preview"] = stable_text(text)[:240]
    samples.append(sample)


def resolve_item(
    *,
    explicit_id: Any,
    text: Any,
    text_to_id: dict[str, str],
    id_to_text: dict[str, str],
) -> tuple[str | None, str]:
    item_id = str(explicit_id) if explicit_id not in (None, "") else ""
    normalized_text = stable_text(text)
    if item_id:
        if not normalized_text:
            normalized_text = id_to_text.get(item_id, "")
        return item_id, normalized_text
    if not normalized_text:
        return None, ""
    resolved_id = text_to_id.get(normalized_text)
    return resolved_id, normalized_text


def source_label(row: dict[str, Any]) -> str:
    source = stable_text(row.get("source"))
    return source if source else "unknown"


def provenance(row: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("row_id", "source"):
        if row.get(key) not in (None, ""):
            out[key] = row[key]
    out["query_id"] = resolved["query_id"]
    out["positive_doc_id"] = resolved["positive_doc_id"]
    out["negative_doc_ids"] = resolved["negative_doc_ids"]
    return out


def finalize_batch(
    *,
    batch_examples: list[dict[str, Any]],
    batch_index: int,
    model_id: str,
    score: str,
) -> dict[str, Any]:
    doc_order: list[str] = []
    doc_text: dict[str, str] = {}
    doc_roles: dict[str, set[str]] = {}
    doc_vector_by_id: dict[str, list[float]] = {}
    for item in batch_examples:
        for doc_id, text, role in item["documents"]:
            if doc_id not in doc_text:
                doc_order.append(doc_id)
                doc_text[doc_id] = text
                doc_roles[doc_id] = set()
                doc_vector_by_id[doc_id] = item["doc_vectors"][doc_id]
            doc_roles[doc_id].add(role)

    queries = [{"id": item["query_id"], "text": item["query_text"]} for item in batch_examples]
    documents = []
    for doc_id in doc_order:
        roles = doc_roles[doc_id]
        role = "mixed" if len(roles) > 1 else next(iter(roles))
        documents.append({"id": doc_id, "text": doc_text[doc_id], "role": role})

    teacher_similarity: list[list[float]] = []
    for item in batch_examples:
        query_vector = item["query_vector"]
        row_scores = [score_vectors(query_vector, doc_vector_by_id[doc_id]) for doc_id in doc_order]
        teacher_similarity.append(row_scores)

    row: dict[str, Any] = {
        "schema": SCHEMA,
        "batch_id": f"batch-{batch_index:06d}",
        "source_counts": dict(Counter(source_label(item["raw"]) for item in batch_examples)),
        "examples": [provenance(item["raw"], item) for item in batch_examples],
        "queries": queries,
        "documents": documents,
        "teacher_similarity": teacher_similarity,
        "score": score,
        "normalized": score == "cosine",
    }
    if model_id:
        row["teacher_model_id"] = model_id
    return row


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
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

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    query_text_to_id, query_id_to_text, query_rows, duplicate_queries, empty_queries = load_text_maps(
        queries_path, query_text, args.skip_empty_beir_text
    )
    doc_text_to_id, doc_id_to_text, doc_rows, duplicate_docs, empty_docs = load_text_maps(
        corpus_path, corpus_text, args.skip_empty_beir_text
    )
    normalize_vectors = args.score == "cosine"
    doc_vectors, doc_vector_rows, zero_doc_vectors = load_vectors(args.doc_vectors, normalize_vectors)
    query_vectors, query_vector_rows, zero_query_vectors = load_vectors(
        args.query_vectors, normalize_vectors
    )

    examples_seen = 0
    examples_written = 0
    examples_dropped = 0
    batches_written = 0
    missing_query_text = 0
    missing_doc_text = 0
    missing_query_vector = 0
    missing_doc_vector = 0
    source_counts: Counter[str] = Counter()
    missing_samples: list[dict[str, Any]] = []
    batch_examples: list[dict[str, Any]] = []
    written_query_ids: set[str] = set()
    written_doc_ids: set[str] = set()
    queries_written = 0
    documents_written = 0

    def handle_incomplete(line_number: int, reason: str) -> None:
        if not args.allow_missing:
            raise SystemExit(
                f"{args.hard_negatives}:{line_number}: incomplete coverage ({reason}) "
                "for listwise geometry example; use --allow-missing to drop incomplete examples"
            )

    with args.output_jsonl.open("w", encoding="utf-8") as out_handle:
        for line_number, row in iter_jsonl(args.hard_negatives):
            if args.max_examples > 0 and examples_seen >= args.max_examples:
                break
            example_index = examples_seen
            examples_seen += 1

            query_id, resolved_query_text = resolve_item(
                explicit_id=row.get("query_id"),
                text=row.get("query"),
                text_to_id=query_text_to_id,
                id_to_text=query_id_to_text,
            )
            complete = True
            if query_id is None or not resolved_query_text:
                missing_query_text += 1
                complete = False
                add_missing_sample(
                    missing_samples,
                    args.missing_sample_limit,
                    kind="query_text",
                    example_index=example_index,
                    line_number=line_number,
                    item_id=str(row.get("query_id") or ""),
                    text=str(row.get("query") or ""),
                )
            query_vector = query_vectors.get(query_id or "")
            if query_id is not None and query_vector is None:
                missing_query_vector += 1
                complete = False
                add_missing_sample(
                    missing_samples,
                    args.missing_sample_limit,
                    kind="query_vector",
                    example_index=example_index,
                    line_number=line_number,
                    item_id=query_id,
                    text=resolved_query_text,
                )

            negatives = [str(value or "") for value in row.get("negatives") or []]
            negative_ids_raw = row.get("negative_doc_ids") or []
            if not isinstance(negative_ids_raw, list):
                negative_ids_raw = []
            candidate_specs = [
                ("positive", row.get("positive_doc_id"), row.get("positive")),
            ]
            for idx, negative_text in enumerate(negatives):
                explicit_negative_id = negative_ids_raw[idx] if idx < len(negative_ids_raw) else None
                candidate_specs.append(("negative", explicit_negative_id, negative_text))

            resolved_docs: list[tuple[str, str, str]] = []
            doc_vector_map: dict[str, list[float]] = {}
            for role, explicit_doc_id, text in candidate_specs:
                doc_id, resolved_doc_text = resolve_item(
                    explicit_id=explicit_doc_id,
                    text=text,
                    text_to_id=doc_text_to_id,
                    id_to_text=doc_id_to_text,
                )
                if doc_id is None or not resolved_doc_text:
                    missing_doc_text += 1
                    complete = False
                    add_missing_sample(
                        missing_samples,
                        args.missing_sample_limit,
                        kind="doc_text",
                        example_index=example_index,
                        line_number=line_number,
                        item_id=str(explicit_doc_id or ""),
                        text=str(text or ""),
                    )
                    continue
                doc_vector = doc_vectors.get(doc_id)
                if doc_vector is None:
                    missing_doc_vector += 1
                    complete = False
                    add_missing_sample(
                        missing_samples,
                        args.missing_sample_limit,
                        kind="doc_vector",
                        example_index=example_index,
                        line_number=line_number,
                        item_id=doc_id,
                        text=resolved_doc_text,
                    )
                    continue
                resolved_docs.append((doc_id, resolved_doc_text, role))
                doc_vector_map[doc_id] = doc_vector

            if complete and query_vector is not None:
                try:
                    for doc_id, _, _ in resolved_docs:
                        score_vectors(query_vector, doc_vector_map[doc_id])
                except ValueError as exc:
                    complete = False
                    add_missing_sample(
                        missing_samples,
                        args.missing_sample_limit,
                        kind="vector_score",
                        example_index=example_index,
                        line_number=line_number,
                        text=str(exc),
                    )

            if not complete:
                examples_dropped += 1
                handle_incomplete(line_number, "missing text/vector")
                continue

            positive_doc_id = resolved_docs[0][0]
            negative_doc_ids = [doc_id for doc_id, _, role in resolved_docs if role == "negative"]
            source_counts[source_label(row)] += 1
            batch_examples.append(
                {
                    "raw": row,
                    "query_id": query_id,
                    "query_text": resolved_query_text,
                    "query_vector": query_vector,
                    "positive_doc_id": positive_doc_id,
                    "negative_doc_ids": negative_doc_ids,
                    "documents": resolved_docs,
                    "doc_vectors": doc_vector_map,
                }
            )
            examples_written += 1
            written_query_ids.add(query_id)
            written_doc_ids.update(doc_id for doc_id, _, _ in resolved_docs)

            if len(batch_examples) >= args.batch_size:
                batch_row = finalize_batch(
                    batch_examples=batch_examples,
                    batch_index=batches_written,
                    model_id=args.model_id,
                    score=args.score,
                )
                out_handle.write(json.dumps(batch_row, ensure_ascii=False) + "\n")
                batches_written += 1
                queries_written += len(batch_row["queries"])
                documents_written += len(batch_row["documents"])
                batch_examples = []

        if batch_examples:
            batch_row = finalize_batch(
                batch_examples=batch_examples,
                batch_index=batches_written,
                model_id=args.model_id,
                score=args.score,
            )
            out_handle.write(json.dumps(batch_row, ensure_ascii=False) + "\n")
            batches_written += 1
            queries_written += len(batch_row["queries"])
            documents_written += len(batch_row["documents"])

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "quality_claim": False,
        "teacher_model_id": args.model_id,
        "score": args.score,
        "normalized": normalize_vectors,
        "batch_size": args.batch_size,
        "max_examples": args.max_examples,
        "allow_missing": args.allow_missing,
        "skip_empty_beir_text": args.skip_empty_beir_text,
        "inputs": {
            "hard_negatives": str(args.hard_negatives),
            "dataset_dir": str(args.dataset_dir),
            "corpus_jsonl": str(corpus_path),
            "queries_jsonl": str(queries_path),
            "doc_vectors": str(args.doc_vectors),
            "query_vectors": str(args.query_vectors),
        },
        "outputs": {
            "output_jsonl": str(args.output_jsonl),
            "manifest": str(args.manifest),
        },
        "sha256": {
            "hard_negatives": sha256_file(args.hard_negatives),
            "corpus_jsonl": sha256_file(corpus_path),
            "queries_jsonl": sha256_file(queries_path),
            "doc_vectors": sha256_file(args.doc_vectors),
            "query_vectors": sha256_file(args.query_vectors),
        },
        "coverage": {
            "examples_seen": examples_seen,
            "examples_written": examples_written,
            "examples_dropped": examples_dropped,
            "batches_written": batches_written,
            "queries_written": queries_written,
            "documents_written": documents_written,
            "unique_query_ids_written": len(written_query_ids),
            "unique_doc_ids_written": len(written_doc_ids),
            "missing_query_text": missing_query_text,
            "missing_doc_text": missing_doc_text,
            "missing_query_vector": missing_query_vector,
            "missing_doc_vector": missing_doc_vector,
        },
        "source_counts": dict(source_counts),
        "beir": {
            "query_rows": query_rows,
            "corpus_rows": doc_rows,
            "duplicate_query_texts": duplicate_queries,
            "duplicate_doc_texts": duplicate_docs,
            "empty_query_texts_skipped": empty_queries if args.skip_empty_beir_text else 0,
            "empty_doc_texts_skipped": empty_docs if args.skip_empty_beir_text else 0,
        },
        "vectors": {
            "doc_vector_rows": doc_vector_rows,
            "query_vector_rows": query_vector_rows,
            "zero_doc_vectors": zero_doc_vectors,
            "zero_query_vectors": zero_query_vectors,
        },
        "missing_samples": missing_samples,
    }
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "built listwise geometry batches: "
        f"examples_seen={examples_seen} examples_written={examples_written} "
        f"dropped={examples_dropped} batches={batches_written}"
    )
    print(f"output_jsonl: {args.output_jsonl}")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
