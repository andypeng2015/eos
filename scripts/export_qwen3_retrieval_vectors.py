#!/usr/bin/env python3
"""Export BEIR document/query vector caches with SentenceTransformers.

The output layout matches `eos eval-retrieval-vectors`:

  <output_root>/<dataset_name>/doc-vectors.jsonl
  <output_root>/<dataset_name>/query-vectors.jsonl

When document chunking is enabled, documents are exported as child vectors for
`eos eval-retrieval-multivector-turboquant`:

  <output_root>/<dataset_name>/child-doc-vectors.jsonl
  <output_root>/<dataset_name>/query-vectors.jsonl

Rows are JSON objects with `id` and `embedding` fields. Embeddings are
L2-normalized by default because SentenceTransformers embedding models are
commonly used with cosine similarity; the Eos evaluator normalizes vectors
again before scoring, so this choice should not change ranking for nonzero
vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a scientific claim, retrieve documents that support or "
    "refute the claim\nQuery: "
)


class DocumentChunk(NamedTuple):
    parent_id: str
    child_id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BEIR-compatible document/query vector caches with SentenceTransformers."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="BEIR-style directory containing corpus.jsonl and queries.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root directory for <dataset-name>/doc-vectors.jsonl and query-vectors.jsonl.",
    )
    parser.add_argument(
        "--dataset-name",
        default="scifact",
        help="Dataset subdirectory name under --output-root.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformers model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum corpus rows to export; 0 means all rows.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum query rows to export; 0 means all rows.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch/SentenceTransformers device such as cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--query-prefix",
        default=DEFAULT_QUERY_PREFIX,
        help="Instruction prefix prepended to query text. Use '' to disable.",
    )
    parser.add_argument(
        "--document-prefix",
        default="",
        help="Instruction prefix prepended to document title/text. Use '' to disable.",
    )
    parser.add_argument(
        "--document-chunk-words",
        type=int,
        default=0,
        help=(
            "When positive, export documents as overlapping word chunks to "
            "child-doc-vectors.jsonl instead of doc-vectors.jsonl."
        ),
    )
    parser.add_argument(
        "--document-chunk-overlap",
        type=int,
        default=0,
        help="Word overlap between adjacent document chunks; requires --document-chunk-words.",
    )
    parser.add_argument(
        "--document-chunk-min-words",
        type=int,
        default=1,
        help="Minimum words for a trailing document chunk when chunking is enabled.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not request normalized embeddings from SentenceTransformers.",
    )
    return parser.parse_args()


def require_sentence_transformers():
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing optional Python dependencies for vector export. "
            "Install them in a local venv, for example: "
            "python3 -m venv .venv-qwen3 && "
            ". .venv-qwen3/bin/activate && "
            "pip install 'sentence-transformers>=3' torch\n"
            f"Original import error: {exc}"
        ) from exc
    return SentenceTransformer


def iter_jsonl(path: Path, limit: int) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        count = 0
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            yield row
            count += 1
            if limit > 0 and count >= limit:
                break


def row_id(row: dict, path: Path) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError(f"{path}: row is missing '_id' or 'id': {row!r}")
    return str(value)


def corpus_text(row: dict) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return title + "\n" + text
    return title or text


def query_text(row: dict) -> str:
    return str(row.get("text") or row.get("query") or "").strip()


def batches(items: list[tuple[str, str]], batch_size: int) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_items(path: Path, limit: int, text_fn: Callable[[dict], str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in iter_jsonl(path, limit):
        item_id = row_id(row, path)
        text = text_fn(row)
        if not text:
            raise ValueError(f"{path}: row {item_id!r} has empty text")
        out.append((item_id, text))
    return out


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

    chunks: list[DocumentChunk] = []
    step = chunk_words - overlap
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = words[start:end]
        if chunks and len(chunk) < min_words:
            break
        chunks.append(
            DocumentChunk(
                parent_id,
                f"{parent_id}#chunk-{len(chunks):04d}",
                " ".join(chunk),
            )
        )
        if end >= len(words):
            break
        start += step
    if not chunks:
        chunks.append(DocumentChunk(parent_id, f"{parent_id}#chunk-0000", " ".join(words)))
    return chunks


def chunk_documents(
    docs: list[tuple[str, str]],
    chunk_words: int,
    overlap: int,
    min_words: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for parent_id, text in docs:
        chunks.extend(chunk_document_text(parent_id, text, chunk_words, overlap, min_words))
    return chunks


def embedding_to_list(embedding) -> list[float]:
    values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    return [float(value) for value in values]


def write_vectors(
    model,
    items: list[tuple[str, str]],
    output_path: Path,
    prefix: str,
    batch_size: int,
    normalize: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        written = 0
        for batch in batches(items, batch_size):
            ids = [item_id for item_id, _ in batch]
            texts = [prefix + text for _, text in batch]
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )
            for item_id, embedding in zip(ids, embeddings):
                vector = embedding_to_list(embedding)
                handle.write(json.dumps({"id": item_id, "embedding": vector}) + "\n")
                written += 1
        print(f"wrote {written} rows to {output_path}", flush=True)


def write_child_vectors(
    model,
    chunks: list[DocumentChunk],
    output_path: Path,
    prefix: str,
    batch_size: int,
    normalize: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        written = 0
        for batch in batches(chunks, batch_size):
            texts = [prefix + chunk.text for chunk in batch]
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )
            for chunk, embedding in zip(batch, embeddings):
                vector = embedding_to_list(embedding)
                handle.write(
                    json.dumps(
                        {
                            "parent_id": chunk.parent_id,
                            "child_id": chunk.child_id,
                            "embedding": vector,
                        }
                    )
                    + "\n"
                )
                written += 1
        print(f"wrote {written} rows to {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_docs < 0 or args.max_queries < 0:
        raise SystemExit("--max-docs and --max-queries must be non-negative")
    if args.document_chunk_words < 0:
        raise SystemExit("--document-chunk-words must be non-negative")
    if args.document_chunk_overlap < 0:
        raise SystemExit("--document-chunk-overlap must be non-negative")
    if args.document_chunk_min_words <= 0:
        raise SystemExit("--document-chunk-min-words must be positive")
    if args.document_chunk_words == 0 and args.document_chunk_overlap != 0:
        raise SystemExit("--document-chunk-overlap requires --document-chunk-words")
    if args.document_chunk_words > 0 and args.document_chunk_overlap >= args.document_chunk_words:
        raise SystemExit("--document-chunk-overlap must be smaller than --document-chunk-words")

    corpus_path = args.dataset_dir / "corpus.jsonl"
    queries_path = args.dataset_dir / "queries.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"missing corpus file: {corpus_path}")
    if not queries_path.is_file():
        raise SystemExit(f"missing queries file: {queries_path}")

    SentenceTransformer = require_sentence_transformers()
    print(f"loading {args.model_name}", flush=True)
    model = SentenceTransformer(args.model_name, device=args.device)

    docs = load_items(corpus_path, args.max_docs, corpus_text)
    queries = load_items(queries_path, args.max_queries, query_text)
    if not docs:
        raise SystemExit("no corpus rows selected")
    if not queries:
        raise SystemExit("no query rows selected")

    out_dir = args.output_root / args.dataset_name
    normalize = not args.no_normalize
    chunks: list[DocumentChunk] = []
    if args.document_chunk_words > 0:
        chunks = chunk_documents(
            docs,
            args.document_chunk_words,
            args.document_chunk_overlap,
            args.document_chunk_min_words,
        )
        if not chunks:
            raise SystemExit("document chunking selected no chunks")
    print(
        f"exporting dataset={args.dataset_name} docs={len(docs)} "
        f"queries={len(queries)} child_chunks={len(chunks)} normalize={normalize}",
        flush=True,
    )
    if args.document_chunk_words > 0:
        write_child_vectors(
            model,
            chunks,
            out_dir / "child-doc-vectors.jsonl",
            args.document_prefix,
            args.batch_size,
            normalize,
        )
    else:
        write_vectors(
            model,
            docs,
            out_dir / "doc-vectors.jsonl",
            args.document_prefix,
            args.batch_size,
            normalize,
        )
    write_vectors(
        model,
        queries,
        out_dir / "query-vectors.jsonl",
        args.query_prefix,
        args.batch_size,
        normalize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
