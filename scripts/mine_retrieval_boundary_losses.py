#!/usr/bin/env python3
"""Build guarded hard-negative rows from compact top-100 boundary losses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEMA = "manta.embedding_compact_boundary_loss_hard_negative.v1"
MANIFEST_SCHEMA = SCHEMA + ".manifest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine Eos-compatible hard-negative rows from lost top-100 positives."
    )
    parser.add_argument("--lost-tsv", required=True, type=Path)
    parser.add_argument("--candidate-per-query", required=True, type=Path)
    parser.add_argument("--anchor-per-query", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument(
        "--method",
        default="turboquant_ip_b4_overfetch200_fp16_rerank",
        help="Method name to record in row metadata.",
    )
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--boundary-start", type=int, default=90)
    parser.add_argument("--boundary-end", type=int, default=105)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def read_beir_texts(path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = str(row.get("_id", "") or "")
            if not item_id:
                raise ValueError(f"{path}:{line_no}: missing _id")
            title = str(row.get("title", "") or "").strip()
            text = str(row.get("text", "") or "").strip()
            texts[item_id] = f"{title} {text}".strip() if title and text else title or text
    return texts


def read_lost_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"query_id", "doc_id", "anchor_rank", "candidate_rank_top130"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    return rows


def rows_by_query(path: Path, method: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(iter_jsonl(path), 1):
        if method and str(row.get("method", "") or "") != method:
            continue
        query_id = str(row.get("query_id", "") or "")
        if not query_id:
            raise ValueError(f"{path}:{line_no}: missing query_id")
        if query_id in out:
            raise ValueError(f"{path}:{line_no}: duplicate query_id {query_id}")
        out[query_id] = row
    if not out:
        raise ValueError(f"{path}: no rows matched method {method!r}")
    return out


def top_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    docs = row.get("top_k", [])
    if not isinstance(docs, list):
        raise ValueError(f"query {row.get('query_id')}: top_k is not a list")
    return docs


def select_boundary_negatives(
    candidate_row: dict[str, Any],
    *,
    positive_doc_id: str,
    candidate_rank: int,
    max_negatives: int,
    boundary_start: int,
    boundary_end: int,
) -> list[dict[str, Any]]:
    """Return deterministic nonrelevant candidate docs above the lost positive.

    Boundary-window docs are preferred, then earlier high-scoring nonrelevant
    docs fill any remaining slots. Both lists preserve candidate rank order.
    """

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_from(docs: list[dict[str, Any]]) -> None:
        for doc in docs:
            if len(selected) >= max_negatives:
                return
            doc_id = str(doc.get("doc_id", "") or "")
            rank = int(doc.get("rank", 0) or 0)
            relevance = float(doc.get("relevance", 0) or 0)
            if not doc_id or doc_id == positive_doc_id or doc_id in seen:
                continue
            if rank <= 0 or rank >= candidate_rank:
                continue
            if relevance > 0:
                continue
            seen.add(doc_id)
            selected.append(doc)

    docs = sorted(top_docs(candidate_row), key=lambda doc: int(doc.get("rank", 0) or 0))
    boundary_docs = [
        doc
        for doc in docs
        if boundary_start <= int(doc.get("rank", 0) or 0) <= boundary_end
    ]
    add_from(boundary_docs)
    add_from(docs)
    return selected


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lost_rows = read_lost_rows(args.lost_tsv)
    candidate_rows = rows_by_query(args.candidate_per_query, args.method)
    anchor_rows = rows_by_query(args.anchor_per_query, args.method)
    queries = read_beir_texts(args.dataset_dir / "queries.jsonl")
    docs = read_beir_texts(args.dataset_dir / "corpus.jsonl")

    output_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for lost in lost_rows:
        query_id = str(lost["query_id"])
        positive_doc_id = str(lost["doc_id"])
        candidate_rank = int(lost["candidate_rank_top130"])
        anchor_rank = int(lost["anchor_rank"])
        query_text = queries.get(query_id, "")
        positive_text = docs.get(positive_doc_id, "")
        candidate = candidate_rows.get(query_id)
        anchor = anchor_rows.get(query_id)
        if not query_text or not positive_text or candidate is None or anchor is None:
            skipped.append(
                {
                    "query_id": query_id,
                    "doc_id": positive_doc_id,
                    "missing_query_text": not bool(query_text),
                    "missing_positive_text": not bool(positive_text),
                    "missing_candidate_row": candidate is None,
                    "missing_anchor_row": anchor is None,
                }
            )
            continue

        negatives = select_boundary_negatives(
            candidate,
            positive_doc_id=positive_doc_id,
            candidate_rank=candidate_rank,
            max_negatives=args.max_negatives,
            boundary_start=args.boundary_start,
            boundary_end=args.boundary_end,
        )
        negative_texts: list[str] = []
        negative_doc_ids: list[str] = []
        negative_ranks: list[int] = []
        for doc in negatives:
            doc_id = str(doc.get("doc_id", "") or "")
            text = docs.get(doc_id, "")
            if not text:
                continue
            negative_doc_ids.append(doc_id)
            negative_ranks.append(int(doc.get("rank", 0) or 0))
            negative_texts.append(text)
        if not negative_texts:
            skipped.append(
                {
                    "query_id": query_id,
                    "doc_id": positive_doc_id,
                    "reason": "no usable nonrelevant candidate negatives",
                }
            )
            continue

        output_rows.append(
            {
                "schema": SCHEMA,
                "source": "nfcorpus:compact-boundary-loss:q4-fp16-overfetch200",
                "query": query_text,
                "positive": positive_text,
                "negatives": negative_texts,
                "metadata": {
                    "query_id": query_id,
                    "positive_doc_id": positive_doc_id,
                    "anchor_rank": anchor_rank,
                    "candidate_rank_top130": candidate_rank,
                    "candidate_negative_doc_ids": negative_doc_ids,
                    "candidate_negative_ranks": negative_ranks,
                    "source_run": args.source_run,
                    "method": args.method,
                    "quality_claim": False,
                    "relevant_count": int(lost.get("relevant_count", 0) or 0),
                    "query_recall_delta": float(lost.get("query_recall_delta", 0) or 0),
                },
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "quality_claim": False,
        "claim_boundary": (
            "Protected candidate repair data only; these compact boundary-loss "
            "rows are not benchmark-quality or promotion evidence."
        ),
        "source_run": args.source_run,
        "method": args.method,
        "lost_rows": len(lost_rows),
        "hard_negative_rows": len(output_rows),
        "skipped_rows": len(skipped),
        "skipped": skipped,
        "max_negatives": args.max_negatives,
        "boundary_rank_window": [args.boundary_start, args.boundary_end],
        "represented_positive_doc_ids": [
            row["metadata"]["positive_doc_id"] for row in output_rows
        ],
        "represented_query_ids": [row["metadata"]["query_id"] for row in output_rows],
        "negative_counts": [len(row["negatives"]) for row in output_rows],
    }
    return output_rows, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_negatives <= 0:
        raise SystemExit("--max-negatives must be positive")
    rows, manifest = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "mined compact boundary losses: "
        f"lost_rows={manifest['lost_rows']} hard_negative_rows={manifest['hard_negative_rows']} "
        f"skipped_rows={manifest['skipped_rows']} output={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
