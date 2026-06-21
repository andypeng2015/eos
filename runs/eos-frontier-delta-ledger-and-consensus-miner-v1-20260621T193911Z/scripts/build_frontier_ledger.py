#!/usr/bin/env python3
"""Build a short-set per-query frontier ledger and consensus mining artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ("scifact", "nfcorpus", "fiqa")
QREL_SPLIT = "test"
RUNS = ("eos_dense", "eos_compact_q4_fp16_o200", "bm25", "hybrid_minmax_blend_a050", "qwen3", "mxbai")
TEACHERS = ("qwen3", "mxbai")
EOS_RUNS = ("eos_dense", "eos_compact_q4_fp16_o200")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_qrels(path: Path) -> dict[str, dict[str, float]]:
    rels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("query-id") or raw.startswith("qid"):
                continue
            parts = raw.split()
            if len(parts) >= 4:
                qid, _, docid, score = parts[:4]
            elif len(parts) >= 3:
                qid, docid, score = parts[:3]
            else:
                continue
            try:
                relevance = float(score)
            except ValueError:
                continue
            if relevance > 0:
                rels[qid][docid] = relevance
    return dict(rels)


def first_margin(top_docs: list[dict]) -> float | None:
    if len(top_docs) < 2:
        return None
    return float(top_docs[0].get("score", 0.0)) - float(top_docs[1].get("score", 0.0))


def first_relevant_doc(top_docs: list[dict]) -> str | None:
    for doc in top_docs:
        if float(doc.get("relevance", 0.0)) > 0:
            return str(doc["doc_id"])
    return None


def top_doc(row: dict) -> dict | None:
    docs = row.get("top_k") or []
    return docs[0] if docs else None


def rank_of(row: dict, doc_id: str) -> int:
    for doc in row.get("top_k") or []:
        if str(doc.get("doc_id")) == doc_id:
            return int(doc.get("rank", 0))
    return 0


def quality_value(row: dict, name: str) -> float:
    return float((row.get("quality") or {}).get(name, 0.0))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def dataset_dir(root: Path, dataset: str) -> Path:
    return root / dataset / dataset


def qrels_path(root: Path, dataset: str) -> Path:
    base = dataset_dir(root, dataset) / "qrels"
    split_path = base / QREL_SPLIT
    split_tsv_path = base / f"{QREL_SPLIT}.tsv"
    if split_path.exists():
        return split_path
    if split_tsv_path.exists():
        return split_tsv_path
    return base


def per_query_path(run_root: Path, dataset: str, run: str) -> Path:
    return run_root / "per-query" / f"{dataset}.{run}.per-query.jsonl"


def summarize_run(rows: list[dict]) -> dict:
    return {
        "queries": len(rows),
        "hit_at_1": sum(1 for r in rows if quality_value(r, "hit_at_1") > 0),
        "hit_at_10": sum(1 for r in rows if quality_value(r, "hit_at_10") > 0),
        "hit_at_100": sum(1 for r in rows if quality_value(r, "recall_at_100") > 0),
        "mean_ndcg_at_10": mean([quality_value(r, "ndcg_at_10") for r in rows]),
        "mean_recall_at_100": mean([quality_value(r, "recall_at_100") for r in rows]),
        "mean_first_relevant_rank": mean([float(r.get("first_relevant_rank") or 0) for r in rows]),
        "mean_top1_margin": mean([m for r in rows if (m := first_margin(r.get("top_k") or [])) is not None]),
    }


def build(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    data_root = Path(args.dataset_root)
    ledger_rows: list[dict] = []
    summary: dict[str, object] = {
        "schema": "eos.frontier_delta_ledger_summary.v1",
        "run_root": str(run_root),
        "qrel_split": QREL_SPLIT,
        "datasets": {},
        "sources": list(RUNS),
        "teacher_sources": list(TEACHERS),
        "caveats": [
            "External models are consumed only as local vector-cache scorer boundaries.",
            "FiQA Qwen3 cache is full exportable-text/sanitized-cache evidence, not a raw-row-complete claim.",
            "Consensus artifacts are selection/relabel/drop candidates only; they are not soft teacher-loss targets.",
        ],
    }
    manifest = {
        "schema": "eos.consensus_mined_no_soft_teacher_manifest.v1",
        "eligible_for_later_candidate": False,
        "recommended_training_descriptor": "eos-consensus-mined-no-soft-teacher-candidate",
        "teacher_loss_weight_required": 0,
        "datasets": {},
        "inputs": {},
        "artifacts": {},
        "caveats": list(summary["caveats"]),
    }

    total_consensus = 0
    total_drops = 0
    total_queries = 0

    for dataset in DATASETS:
        qrels = read_qrels(qrels_path(data_root, dataset))
        by_run: dict[str, dict[str, dict]] = {}
        missing_sources = []
        for run in RUNS:
            path = per_query_path(run_root, dataset, run)
            if not path.exists():
                missing_sources.append(run)
                continue
            rows = load_jsonl(path)
            by_run[run] = {str(r["query_id"]): r for r in rows}
            manifest["inputs"][f"{dataset}.{run}.per_query_jsonl"] = str(path)

        query_ids = sorted(qrels.keys(), key=lambda x: (len(x), x))
        total_queries += len(query_ids)
        dataset_summary = {
            "qrel_queries": len(query_ids),
            "missing_sources": missing_sources,
            "sources": {run: summarize_run(list(by_run[run].values())) for run in by_run},
            "full_per_query_sources": sorted([run for run, rows in by_run.items() if len(rows) == len(query_ids)]),
            "row_count_mismatches": {
                run: {"rows": len(rows), "qrel_queries": len(query_ids)}
                for run, rows in by_run.items()
                if len(rows) != len(query_ids)
            },
        }

        consensus_rows: list[dict] = []
        drop_rows: list[dict] = []
        eos_miss_count = 0
        teacher_agree_hit_count = 0
        teacher_same_top1_count = 0
        teacher_conflict_count = 0

        for qid in query_ids:
            per_run = {run: by_run[run].get(qid) for run in by_run if qid in by_run[run]}
            tops = {run: top_doc(row) for run, row in per_run.items() if row}
            top_ids = {run: str(doc["doc_id"]) for run, doc in tops.items() if doc}
            first_rels = {run: first_relevant_doc(row.get("top_k") or []) for run, row in per_run.items() if row}
            teacher_rows = {run: per_run.get(run) for run in TEACHERS if per_run.get(run)}
            teacher_top_ids = {run: top_ids.get(run) for run in TEACHERS if top_ids.get(run)}
            qrel_doc_ids = set(qrels.get(qid, {}))

            eos_miss = any(per_run.get(run) and quality_value(per_run[run], "hit_at_10") == 0 for run in EOS_RUNS)
            eos_hit = any(per_run.get(run) and quality_value(per_run[run], "hit_at_10") > 0 for run in EOS_RUNS)
            qwen_hit = per_run.get("qwen3") is not None and quality_value(per_run["qwen3"], "hit_at_10") > 0
            mxbai_hit = per_run.get("mxbai") is not None and quality_value(per_run["mxbai"], "hit_at_10") > 0
            qwen_top_rel = per_run.get("qwen3") is not None and quality_value(per_run["qwen3"], "hit_at_1") > 0
            mxbai_top_rel = per_run.get("mxbai") is not None and quality_value(per_run["mxbai"], "hit_at_1") > 0

            ledger_row = {
                "dataset": dataset,
                "query_id": qid,
                "relevant_count": len(qrel_doc_ids),
                "source_top1_doc_ids": top_ids,
                "source_first_relevant_doc_ids": {k: v for k, v in first_rels.items() if v},
                "source_first_relevant_ranks": {run: int(row.get("first_relevant_rank") or 0) for run, row in per_run.items()},
                "source_ndcg_at_10": {run: quality_value(row, "ndcg_at_10") for run, row in per_run.items()},
                "source_recall_at_100": {run: quality_value(row, "recall_at_100") for run, row in per_run.items()},
                "source_top1_margins": {run: first_margin(row.get("top_k") or []) for run, row in per_run.items()},
                "eos_any_miss_at_10": eos_miss,
                "external_both_hit_at_10": qwen_hit and mxbai_hit,
                "external_both_hit_at_1": qwen_top_rel and mxbai_top_rel,
                "external_same_top1_doc": len(set(teacher_top_ids.values())) == 1 if len(teacher_top_ids) == 2 else False,
            }
            ledger_rows.append(ledger_row)

            if eos_miss:
                eos_miss_count += 1
            if qwen_hit and mxbai_hit:
                teacher_agree_hit_count += 1
            if ledger_row["external_same_top1_doc"]:
                teacher_same_top1_count += 1

            if eos_miss and qwen_hit and mxbai_hit:
                positive_doc_ids = sorted((set(first_rels.values()) | qrel_doc_ids) - {None})
                positive_doc_ids = [d for d in positive_doc_ids if d]
                consensus_rows.append({
                    "schema": "eos.consensus_relabel_candidate.v1",
                    "dataset": dataset,
                    "query_id": qid,
                    "kind": "eos_miss_external_consensus_positive",
                    "qrel_doc_ids": sorted(qrel_doc_ids),
                    "candidate_positive_doc_ids": positive_doc_ids,
                    "teacher_first_relevant_ranks": {run: int(teacher_rows[run].get("first_relevant_rank") or 0) for run in teacher_rows},
                    "teacher_top1_doc_ids": teacher_top_ids,
                    "teacher_top1_margins": {run: first_margin(teacher_rows[run].get("top_k") or []) for run in teacher_rows},
                    "eos_first_relevant_ranks": {run: int(per_run[run].get("first_relevant_rank") or 0) for run in EOS_RUNS if per_run.get(run)},
                    "use": "silver-positive-or-protection-row",
                })

            teacher_top_set = set(teacher_top_ids.values())
            teacher_disagree = len(teacher_top_set) > 1
            label_conflict = bool(teacher_top_set) and not bool(teacher_top_set & qrel_doc_ids) and not qwen_hit and not mxbai_hit
            eos_teacher_conflict = eos_hit and not (qwen_hit or mxbai_hit)
            if teacher_disagree or label_conflict or eos_teacher_conflict:
                teacher_conflict_count += 1
                drop_rows.append({
                    "schema": "eos.consensus_drop_or_disagreement_case.v1",
                    "dataset": dataset,
                    "query_id": qid,
                    "kind": "teacher_disagreement_or_label_conflict",
                    "qrel_doc_ids": sorted(qrel_doc_ids),
                    "source_top1_doc_ids": top_ids,
                    "source_first_relevant_ranks": {run: int(row.get("first_relevant_rank") or 0) for run, row in per_run.items()},
                    "teacher_disagree_top1": teacher_disagree,
                    "teacher_label_conflict": label_conflict,
                    "eos_hit_teacher_miss_conflict": eos_teacher_conflict,
                    "use": "drop-or-review-before-training",
                })

        dataset_summary.update({
            "eos_any_miss_at_10_queries": eos_miss_count,
            "external_both_hit_at_10_queries": teacher_agree_hit_count,
            "external_same_top1_doc_queries": teacher_same_top1_count,
            "disagreement_or_drop_queries": teacher_conflict_count,
            "consensus_relabel_candidates": len(consensus_rows),
            "consensus_positive_coverage": (len(consensus_rows) / len(query_ids)) if query_ids else 0.0,
        })
        summary["datasets"][dataset] = dataset_summary

        consensus_path = run_root / f"{dataset}.consensus-relabel-candidates.jsonl"
        drop_path = run_root / f"{dataset}.disagreement-drop-cases.jsonl"
        write_jsonl(consensus_path, consensus_rows)
        write_jsonl(drop_path, drop_rows)
        manifest["datasets"][dataset] = {
            "qrel_queries": len(query_ids),
            "consensus_relabel_candidates": len(consensus_rows),
            "drop_or_review_cases": len(drop_rows),
            "missing_sources": missing_sources,
            "row_count_mismatches": dataset_summary["row_count_mismatches"],
            "candidate_files": {"consensus_relabel": str(consensus_path), "drop_or_review": str(drop_path)},
        }
        manifest["artifacts"][f"{dataset}.consensus_relabel"] = str(consensus_path)
        manifest["artifacts"][f"{dataset}.drop_or_review"] = str(drop_path)
        total_consensus += len(consensus_rows)
        total_drops += len(drop_rows)

    summary["totals"] = {
        "qrel_queries": total_queries,
        "ledger_rows": len(ledger_rows),
        "consensus_relabel_candidates": total_consensus,
        "drop_or_review_cases": total_drops,
    }
    manifest["eligible_for_later_candidate"] = total_consensus >= 50
    manifest["totals"] = summary["totals"]
    manifest["recommended_use"] = (
        "Enough silver/protection rows for a bounded no-soft-teacher candidate."
        if manifest["eligible_for_later_candidate"]
        else "Use as audit/protection data first; candidate row count is small."
    )

    ledger_json = run_root / "frontier-ledger-summary.json"
    ledger_tsv = run_root / "frontier-ledger-summary.tsv"
    write_json(ledger_json, {"summary": summary, "ledger": ledger_rows})
    with ledger_tsv.open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "query_id",
            "relevant_count",
            "eos_any_miss_at_10",
            "external_both_hit_at_10",
            "external_both_hit_at_1",
            "external_same_top1_doc",
            "source_first_relevant_ranks",
            "source_ndcg_at_10",
            "source_recall_at_100",
            "source_top1_doc_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in ledger_rows:
            out = dict(row)
            for key in ("source_first_relevant_ranks", "source_ndcg_at_10", "source_recall_at_100", "source_top1_doc_ids"):
                out[key] = json.dumps(out[key], sort_keys=True)
            writer.writerow(out)

    manifest["artifacts"]["frontier_ledger_summary_json"] = str(ledger_json)
    manifest["artifacts"]["frontier_ledger_summary_tsv"] = str(ledger_tsv)
    write_json(run_root / "candidate-training-manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="datasets/manta-embed-v1/raw")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
