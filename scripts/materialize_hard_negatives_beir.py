#!/usr/bin/env python3
"""Materialize runtime hard-negative JSONL as a bounded BEIR-style dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "eos.hard_negatives_beir_materialization.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="msmarco-stagea-bge-real-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_text(row: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    text = stable_text(row.get(key))
    if not text:
        raise ValueError(f"{path}:{line_number}: missing or empty {key}")
    return text


def synthetic_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def choose_canonical(ids: list[str], fallback_prefix: str, text: str) -> str:
    clean = sorted({stable_text(value) for value in ids if stable_text(value)})
    if clean:
        return clean[0]
    return synthetic_id(fallback_prefix, text)


def collect_text_id(text_to_ids: dict[str, list[str]], text: str, item_id: Any) -> None:
    ids = text_to_ids.setdefault(text, [])
    clean = stable_text(item_id)
    if clean and clean not in ids:
        ids.append(clean)


def main() -> int:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise SystemExit(f"missing input JSONL: {args.input_jsonl}")
    split = stable_text(args.split)
    if not split:
        raise SystemExit("--split must be non-empty")

    rows: list[dict[str, Any]] = []
    query_text_to_ids: dict[str, list[str]] = {}
    doc_text_to_ids: dict[str, list[str]] = {}
    gate_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for line_number, row in iter_jsonl(args.input_jsonl):
        query = require_text(row, "query", args.input_jsonl, line_number)
        positive = require_text(row, "positive", args.input_jsonl, line_number)
        negatives = [stable_text(value) for value in row.get("negatives") or []]
        if any(not value for value in negatives):
            raise ValueError(f"{args.input_jsonl}:{line_number}: empty negative text")
        negative_doc_ids = list(row.get("negative_doc_ids") or [])
        if negative_doc_ids and len(negative_doc_ids) != len(negatives):
            raise ValueError(
                f"{args.input_jsonl}:{line_number}: negative_doc_ids length "
                f"{len(negative_doc_ids)} does not match negatives {len(negatives)}"
            )
        gates = row.get("legal_gates") or {}
        if gates != LEGAL_GATES:
            gate_counts["legal_gate_mismatch"] += 1
        for key, expected in LEGAL_GATES.items():
            if gates.get(key) != expected:
                gate_counts[f"{key}_mismatch"] += 1
        source_counts[stable_text(row.get("source"))] += 1
        rows.append(row)
        collect_text_id(query_text_to_ids, query, row.get("query_id"))
        collect_text_id(doc_text_to_ids, positive, row.get("positive_doc_id"))
        for index, negative in enumerate(negatives):
            collect_text_id(
                doc_text_to_ids,
                negative,
                negative_doc_ids[index] if index < len(negative_doc_ids) else "",
            )

    if not rows:
        raise SystemExit("input JSONL had no rows")
    if gate_counts:
        raise SystemExit(f"legal gate mismatch in input rows: {dict(gate_counts)}")

    query_canonical = {
        text: choose_canonical(ids, "q", text) for text, ids in query_text_to_ids.items()
    }
    doc_canonical = {
        text: choose_canonical(ids, "d", text) for text, ids in doc_text_to_ids.items()
    }

    query_aliases = {
        canonical: sorted(set(ids) - {canonical})
        for text, ids in query_text_to_ids.items()
        for canonical in [query_canonical[text]]
        if sorted(set(ids) - {canonical})
    }
    doc_aliases = {
        canonical: sorted(set(ids) - {canonical})
        for text, ids in doc_text_to_ids.items()
        for canonical in [doc_canonical[text]]
        if sorted(set(ids) - {canonical})
    }

    corpus_rows = [
        {"_id": doc_canonical[text], "title": "", "text": text}
        for text in sorted(doc_canonical, key=lambda value: doc_canonical[value])
    ]
    query_rows = [
        {"_id": query_canonical[text], "text": text}
        for text in sorted(query_canonical, key=lambda value: query_canonical[value])
    ]

    qrels: dict[tuple[str, str], int] = {}
    row_maps: list[dict[str, Any]] = []
    for example_index, row in enumerate(rows):
        query = stable_text(row.get("query"))
        positive = stable_text(row.get("positive"))
        negatives = [stable_text(value) for value in row.get("negatives") or []]
        positive_beir_doc_id = doc_canonical[positive]
        query_beir_id = query_canonical[query]
        qrels[(query_beir_id, positive_beir_doc_id)] = 1
        row_maps.append(
            {
                "example_index": example_index,
                "row_id": stable_text(row.get("row_id")),
                "query_id": stable_text(row.get("query_id")),
                "beir_query_id": query_beir_id,
                "positive_doc_id": stable_text(row.get("positive_doc_id")),
                "beir_positive_doc_id": positive_beir_doc_id,
                "negative_doc_ids": [stable_text(value) for value in row.get("negative_doc_ids") or []],
                "beir_negative_doc_ids": [doc_canonical[text] for text in negatives],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir = args.output_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    queries_path = args.output_dir / "queries.jsonl"
    qrels_path = qrels_dir / f"{split}.tsv"
    manifest_path = args.manifest or args.output_dir / "manifest.json"

    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(queries_path, query_rows)
    with qrels_path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, doc_id in sorted(qrels):
            handle.write(f"{query_id}\t{doc_id}\t1\n")

    duplicate_query_texts = sum(1 for ids in query_text_to_ids.values() if len(ids) > 1)
    duplicate_doc_texts = sum(1 for ids in doc_text_to_ids.values() if len(ids) > 1)
    candidate_count = sum(1 + len(row.get("negatives") or []) for row in rows)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": args.dataset,
        "split": split,
        "source_row_path": str(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "output_dir": str(args.output_dir),
        "paths": {
            "corpus": str(corpus_path),
            "queries": str(queries_path),
            "qrels": str(qrels_path),
        },
        "counts": {
            "rows": len(rows),
            "candidate_texts": candidate_count,
            "unique_queries": len(query_rows),
            "unique_docs": len(corpus_rows),
            "qrel_pairs": len(qrels),
            "duplicate_query_texts": duplicate_query_texts,
            "duplicate_doc_texts": duplicate_doc_texts,
            "query_aliases": sum(len(values) for values in query_aliases.values()),
            "doc_aliases": sum(len(values) for values in doc_aliases.values()),
        },
        "legal_gates": dict(LEGAL_GATES),
        "legal_gate_counts": dict(gate_counts),
        "source_counts": dict(source_counts),
        "aliases": {
            "queries": query_aliases,
            "docs": doc_aliases,
        },
        "row_to_beir_id_maps": row_maps,
        "sha256": {
            "corpus": sha256_file(corpus_path),
            "queries": sha256_file(queries_path),
            "qrels": sha256_file(qrels_path),
        },
    }
    write_json(manifest_path, manifest)
    print(
        "materialized hard negatives BEIR: "
        f"rows={len(rows)} queries={len(query_rows)} docs={len(corpus_rows)} qrels={len(qrels)}"
    )
    print(f"output_dir: {args.output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
