#!/usr/bin/env python3
"""Mine retrieval queries where Eos trails a stronger per-query baseline.

Inputs are `manta.embedding_retrieval_per_query.v1` JSONL files emitted by
`eos eval-retrieval --per-query-jsonl` and
`eos eval-retrieval-vectors --per-query-jsonl`.
Multimethod diagnostics such as `eval-retrieval-multivector-turboquant` can be
filtered with `--method` or `--bits`. Use `--eos-method`/`--teacher-method`
and `--eos-bits`/`--teacher-bits` when comparing asymmetric surfaces such as
direct Eos rows against a q4 external teacher row.

The primary artifact is a JSONL frontier report. When `--dataset-dir` and
`--hard-negatives-jsonl` are provided, the script also writes Eos-compatible
text hard-negative rows:

  {"source":"fiqa:quality-frontier:mxbai","query":"...","positive":"...",
   "negatives":["..."]}

Those rows are intended as guarded training input candidates, not promotion
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FRONTIER_SCHEMA = "manta.embedding_quality_frontier_mine.v1"
HARD_NEGATIVE_SCHEMA = "manta.embedding_quality_frontier_hard_negative.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine Eos-vs-teacher retrieval frontier failures from per-query JSONL."
    )
    parser.add_argument("--eos-per-query", required=True, type=Path)
    parser.add_argument("--teacher-per-query", required=True, type=Path)
    parser.add_argument("--eos-label", default="eos")
    parser.add_argument("--teacher-label", default="teacher")
    parser.add_argument(
        "--method",
        default="",
        help=(
            "Optional exact per-query method filter for both inputs. "
            "Used as the default for --eos-method and --teacher-method."
        ),
    )
    parser.add_argument(
        "--eos-method",
        default=None,
        help="Optional exact per-query method filter for the Eos input only.",
    )
    parser.add_argument(
        "--teacher-method",
        default=None,
        help="Optional exact per-query method filter for the teacher input only.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help=(
            "Optional per-query bit-width filter for both inputs. "
            "Used as the default for --eos-bits and --teacher-bits."
        ),
    )
    parser.add_argument(
        "--eos-bits",
        type=int,
        default=None,
        help="Optional per-query bit-width filter for the Eos input only.",
    )
    parser.add_argument(
        "--teacher-bits",
        type=int,
        default=None,
        help="Optional per-query bit-width filter for the teacher input only.",
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional BEIR dataset dir for text hard-negative output.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--hard-negatives-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--min-ndcg-gap",
        type=float,
        default=0.0,
        help="Minimum teacher minus Eos nDCG@10 gap to emit.",
    )
    parser.add_argument(
        "--min-teacher-ndcg",
        type=float,
        default=0.0,
        help="Require teacher nDCG@10 at least this value.",
    )
    parser.add_argument(
        "--require-teacher-hit",
        action="store_true",
        help="Only emit rows where the teacher has a relevant doc in top-k.",
    )
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=5,
        help="Maximum Eos top-ranked non-relevant docs for hard-negative rows.",
    )
    return parser.parse_args()


def matches_filter(row: dict[str, Any], method: str, bits: int | None) -> bool:
    if method and str(row.get("method", "")) != method:
        return False
    if bits is not None and int(row.get("bits", 0) or 0) != bits:
        return False
    return True


def load_per_query(path: Path, method: str = "", bits: int | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not matches_filter(row, method, bits):
                continue
            qid = str(row.get("query_id", ""))
            if not qid:
                raise ValueError(f"{path}:{line_no}: missing query_id")
            if qid in rows:
                suffix = ""
                if method or bits is not None:
                    suffix = f" after filters method={method!r} bits={bits!r}"
                raise ValueError(f"{path}:{line_no}: duplicate query_id {qid!r}{suffix}")
            rows[qid] = row
    if not rows:
        suffix = ""
        if method or bits is not None:
            suffix = f" matching method={method!r} bits={bits!r}"
        raise ValueError(f"{path}: no per-query rows{suffix}")
    return rows


def metric(row: dict[str, Any], name: str) -> float:
    for field in ("quality", "fusion_quality", "direct_quality", "token_span_quality"):
        value = row.get(field)
        if isinstance(value, dict) and name in value:
            return float(value.get(name, 0.0) or 0.0)
    return 0.0


def first_rank(row: dict[str, Any]) -> int:
    if "first_relevant_rank" in row:
        return int(row.get("first_relevant_rank", 0) or 0)
    if "fusion_first_relevant_rank" in row:
        return int(row.get("fusion_first_relevant_rank", 0) or 0)
    return int(row.get("first_relevant_rank", 0) or 0)


def relevant_top_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [doc for doc in row.get("top_k", []) if float(doc.get("relevance", 0) or 0) > 0]


def nonrelevant_top_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [doc for doc in row.get("top_k", []) if float(doc.get("relevance", 0) or 0) <= 0]


def build_frontier_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    eos_method = args.method if args.eos_method is None else args.eos_method
    teacher_method = args.method if args.teacher_method is None else args.teacher_method
    eos_bits = args.bits if args.eos_bits is None else args.eos_bits
    teacher_bits = args.bits if args.teacher_bits is None else args.teacher_bits
    eos_rows = load_per_query(args.eos_per_query, eos_method, eos_bits)
    teacher_rows = load_per_query(args.teacher_per_query, teacher_method, teacher_bits)
    shared_ids = sorted(set(eos_rows) & set(teacher_rows))
    if not shared_ids:
        raise ValueError("no shared query_id values between Eos and teacher rows")

    mined: list[dict[str, Any]] = []
    for qid in shared_ids:
        eos = eos_rows[qid]
        teacher = teacher_rows[qid]
        eos_ndcg = metric(eos, "ndcg_at_10")
        teacher_ndcg = metric(teacher, "ndcg_at_10")
        gap = teacher_ndcg - eos_ndcg
        teacher_relevant = relevant_top_docs(teacher)
        if gap < args.min_ndcg_gap:
            continue
        if teacher_ndcg < args.min_teacher_ndcg:
            continue
        if args.require_teacher_hit and not teacher_relevant:
            continue
        dataset = args.dataset or str(eos.get("dataset") or teacher.get("dataset") or "")
        row = {
            "schema": FRONTIER_SCHEMA,
            "dataset": dataset,
            "query_id": qid,
            "relevant_count": int(eos.get("relevant_count", teacher.get("relevant_count", 0)) or 0),
            "eos": {
                "label": args.eos_label,
                "ndcg_at_10": eos_ndcg,
                "recall_at_100": metric(eos, "recall_at_100"),
                "first_relevant_rank": first_rank(eos),
                "top_nonrelevant": nonrelevant_top_docs(eos)[: args.negatives_per_query],
                "top_relevant": relevant_top_docs(eos)[:5],
            },
            "teacher": {
                "label": args.teacher_label,
                "ndcg_at_10": teacher_ndcg,
                "recall_at_100": metric(teacher, "recall_at_100"),
                "first_relevant_rank": first_rank(teacher),
                "top_relevant": teacher_relevant[:5],
                "top_nonrelevant": nonrelevant_top_docs(teacher)[: args.negatives_per_query],
            },
            "delta": {
                "teacher_minus_eos_ndcg_at_10": gap,
                "teacher_minus_eos_recall_at_100": metric(teacher, "recall_at_100")
                - metric(eos, "recall_at_100"),
                "first_relevant_rank_improvement": rank_improvement(first_rank(eos), first_rank(teacher)),
            },
        }
        mined.append(row)

    mined.sort(
        key=lambda row: (
            row["delta"]["teacher_minus_eos_ndcg_at_10"],
            row["delta"]["first_relevant_rank_improvement"],
        ),
        reverse=True,
    )
    if args.limit > 0:
        mined = mined[: args.limit]
    return mined


def rank_improvement(eos_rank: int, teacher_rank: int) -> int:
    if eos_rank == 0 and teacher_rank == 0:
        return 0
    if eos_rank == 0:
        return 1_000_000 - teacher_rank
    if teacher_rank == 0:
        return -1_000_000 + eos_rank
    return eos_rank - teacher_rank


def read_beir_jsonl(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = str(row.get("_id", ""))
            if not item_id:
                raise ValueError(f"{path}:{line_no}: missing _id")
            rows[item_id] = {
                "title": str(row.get("title", "") or ""),
                "text": str(row.get("text", "") or ""),
            }
    return rows


def beir_text(row: dict[str, str]) -> str:
    title = row.get("title", "").strip()
    text = row.get("text", "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "query_id",
                "eos_ndcg_at_10",
                "teacher_ndcg_at_10",
                "ndcg_at_10_gap",
                "eos_first_relevant_rank",
                "teacher_first_relevant_rank",
                "rank_improvement",
                "teacher_top_relevant_doc_ids",
                "eos_top_nonrelevant_doc_ids",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "query_id": row["query_id"],
                    "eos_ndcg_at_10": row["eos"]["ndcg_at_10"],
                    "teacher_ndcg_at_10": row["teacher"]["ndcg_at_10"],
                    "ndcg_at_10_gap": row["delta"]["teacher_minus_eos_ndcg_at_10"],
                    "eos_first_relevant_rank": row["eos"]["first_relevant_rank"],
                    "teacher_first_relevant_rank": row["teacher"]["first_relevant_rank"],
                    "rank_improvement": row["delta"]["first_relevant_rank_improvement"],
                    "teacher_top_relevant_doc_ids": ",".join(
                        str(doc.get("doc_id", "")) for doc in row["teacher"]["top_relevant"]
                    ),
                    "eos_top_nonrelevant_doc_ids": ",".join(
                        str(doc.get("doc_id", "")) for doc in row["eos"]["top_nonrelevant"]
                    ),
                }
            )


def write_hard_negatives(path: Path, rows: list[dict[str, Any]], dataset_dir: Path) -> int:
    queries = read_beir_jsonl(dataset_dir / "queries.jsonl")
    docs = read_beir_jsonl(dataset_dir / "corpus.jsonl")
    hard_rows: list[dict[str, Any]] = []
    for row in rows:
        query = queries.get(row["query_id"])
        if not query:
            continue
        positive_doc = first_doc_text(row["teacher"]["top_relevant"], docs)
        if not positive_doc:
            positive_doc = first_doc_text(row["eos"]["top_relevant"], docs)
        if not positive_doc:
            continue
        negatives = []
        seen = set()
        for doc in row["eos"]["top_nonrelevant"]:
            doc_id = str(doc.get("doc_id", ""))
            if not doc_id or doc_id in seen:
                continue
            text = beir_text(docs.get(doc_id, {}))
            if not text:
                continue
            seen.add(doc_id)
            negatives.append(text)
        if not negatives:
            continue
        hard_rows.append(
            {
                "schema": HARD_NEGATIVE_SCHEMA,
                "source": f"{row['dataset']}:quality-frontier:{row['teacher']['label']}",
                "query": beir_text(query),
                "positive": positive_doc,
                "negatives": negatives,
                "metadata": {
                    "query_id": row["query_id"],
                    "teacher_label": row["teacher"]["label"],
                    "eos_label": row["eos"]["label"],
                    "teacher_minus_eos_ndcg_at_10": row["delta"][
                        "teacher_minus_eos_ndcg_at_10"
                    ],
                    "eos_top_nonrelevant_doc_ids": [
                        str(doc.get("doc_id", "")) for doc in row["eos"]["top_nonrelevant"]
                    ],
                    "teacher_top_relevant_doc_ids": [
                        str(doc.get("doc_id", "")) for doc in row["teacher"]["top_relevant"]
                    ],
                },
            }
        )
    write_jsonl(path, hard_rows)
    return len(hard_rows)


def first_doc_text(top_docs: list[dict[str, Any]], docs: dict[str, dict[str, str]]) -> str:
    for doc in top_docs:
        text = beir_text(docs.get(str(doc.get("doc_id", "")), {}))
        if text:
            return text
    return ""


def write_summary(path: Path, rows: list[dict[str, Any]], hard_negative_rows: int) -> None:
    total_gap = sum(row["delta"]["teacher_minus_eos_ndcg_at_10"] for row in rows)
    summary = {
        "schema": FRONTIER_SCHEMA + ".summary",
        "queries": len(rows),
        "hard_negative_rows": hard_negative_rows,
        "quality_claim": False,
        "claim_boundary": (
            "Protected candidate training data only; capped mined rows are not "
            "benchmark or product-quality evidence."
        ),
        "mean_teacher_minus_eos_ndcg_at_10": total_gap / len(rows) if rows else 0.0,
        "max_teacher_minus_eos_ndcg_at_10": rows[0]["delta"]["teacher_minus_eos_ndcg_at_10"]
        if rows
        else 0.0,
        "datasets": sorted({row["dataset"] for row in rows}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_frontier_rows(args)
    write_jsonl(args.output_jsonl, rows)
    if args.output_tsv:
        write_tsv(args.output_tsv, rows)
    hard_negative_rows = 0
    if args.hard_negatives_jsonl:
        if args.dataset_dir is None:
            raise SystemExit("--hard-negatives-jsonl requires --dataset-dir")
        hard_negative_rows = write_hard_negatives(
            args.hard_negatives_jsonl, rows, args.dataset_dir
        )
    if args.summary_json:
        write_summary(args.summary_json, rows, hard_negative_rows)
    print(
        "mined quality frontier: "
        f"queries={len(rows)} hard_negative_rows={hard_negative_rows} "
        f"jsonl={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
