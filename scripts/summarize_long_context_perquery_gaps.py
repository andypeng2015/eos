#!/usr/bin/env python3
"""Summarize LongEmbed per-query Eos-vs-external diagnostic gaps.

The output is diagnostic evidence only. It compares already-generated per-query
JSONL artifacts and intentionally carries ``quality_claim=false`` throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUALITY_CLAIM = False
SCHEMA = "eos.long_context_perquery_gap_summary.v1"
CLAIM_BOUNDARY = (
    "Diagnostic comparison of existing per-query artifacts only; not a "
    "benchmark, model-training signal, product-quality claim, or release claim."
)
QUALITY_FIELDS = ("quality", "fusion_quality", "direct_quality", "token_span_quality")
TSV_FIELDS = (
    "dataset",
    "query_id",
    "best_eos_label",
    "best_eos_ndcg_at_10",
    "best_external_label",
    "best_external_ndcg_at_10",
    "best_external_minus_best_eos_ndcg_at_10",
    "trails_all_externals",
    "external_consensus_miss",
    "eos_matches_external",
    "direct_fusion_beats_token_span",
    "token_span_beats_direct_fusion",
    "best_external_top_relevant_doc_ids",
    "best_eos_top_nonrelevant_doc_ids",
    "quality_claim",
)


@dataclass(frozen=True)
class ProfileSpec:
    dataset: str
    label: str
    path: Path


@dataclass(frozen=True)
class RowFilter:
    dataset: str
    label: str
    field: str
    value: str


def parse_profile_spec(value: str) -> ProfileSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DATASET:LABEL=PATH")
    left, path = value.split("=", 1)
    if ":" not in left:
        raise argparse.ArgumentTypeError("expected DATASET:LABEL=PATH")
    dataset, label = left.split(":", 1)
    dataset = dataset.strip()
    label = label.strip()
    if not dataset or not label or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty DATASET, LABEL, and PATH")
    return ProfileSpec(dataset=dataset, label=label, path=Path(path))


def parse_row_filter(value: str) -> RowFilter:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DATASET:LABEL:FIELD=VALUE")
    left, filter_value = value.split("=", 1)
    parts = left.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected DATASET:LABEL:FIELD=VALUE")
    dataset, label, field = (part.strip() for part in parts)
    if not dataset or not label or not field:
        raise argparse.ArgumentTypeError("expected non-empty DATASET, LABEL, and FIELD")
    return RowFilter(dataset=dataset, label=label, field=field, value=filter_value)


def finite_float(value: Any, *, field: str, path: Path, line_no: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}:{line_no}: expected numeric {field}, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}:{line_no}: expected finite {field}, got {value!r}")
    return result


def row_matches_filters(row: dict[str, Any], filters: list[RowFilter]) -> bool:
    for row_filter in filters:
        if str(row.get(row_filter.field) or "") != row_filter.value:
            return False
    return True


def load_per_query(path: Path, filters: list[RowFilter] | None = None) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required per-query path: {path}")
    filters = filters or []
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            if not row_matches_filters(row, filters):
                continue
            qid = str(row.get("query_id") or "")
            if not qid:
                raise ValueError(f"{path}:{line_no}: missing query_id")
            if qid in rows:
                raise ValueError(f"{path}:{line_no}: duplicate query_id {qid!r}")
            metric(row, "ndcg_at_10", path=path, line_no=line_no)
            metric(row, "recall_at_100", path=path, line_no=line_no)
            rows[qid] = row
    if not rows:
        suffix = ""
        if filters:
            suffix = " matching " + ",".join(f"{item.field}={item.value!r}" for item in filters)
        raise ValueError(f"{path}: no per-query rows{suffix}")
    return rows


def metric(row: dict[str, Any], name: str, *, path: Path | None = None, line_no: int = 0) -> float:
    for field in QUALITY_FIELDS:
        value = row.get(field)
        if isinstance(value, dict) and name in value:
            raw = value.get(name)
            if path is not None:
                return finite_float(raw, field=f"{field}.{name}", path=path, line_no=line_no)
            return float(raw or 0.0)
    raw = row.get(name)
    if raw is not None:
        if path is not None:
            return finite_float(raw, field=name, path=path, line_no=line_no)
        return float(raw or 0.0)
    if path is not None:
        raise ValueError(f"{path}:{line_no}: missing metric {name!r}")
    return 0.0


def first_relevant_rank(row: dict[str, Any]) -> int:
    for field in (
        "first_relevant_rank",
        "fusion_first_relevant_rank",
        "direct_first_relevant_rank",
        "token_span_first_relevant_rank",
    ):
        if field in row:
            return int(row.get(field) or 0)
    return 0


def ranked_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("top_k")
    if not isinstance(value, list):
        return []
    docs: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            docs.append(item)
    return docs


def compact_doc(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "doc_id": str(item.get("doc_id") or ""),
        "rank": int(item.get("rank") or 0),
        "relevance": float(item.get("relevance") or 0.0),
    }
    if "score" in item and isinstance(item.get("score"), (int, float)) and not isinstance(item.get("score"), bool):
        result["score"] = float(item["score"])
    if item.get("child_id"):
        result["child_id"] = str(item["child_id"])
    return result


def top_doc_ids(row: dict[str, Any], limit: int) -> list[str]:
    return [str(item.get("doc_id") or "") for item in ranked_docs(row)[:limit] if item.get("doc_id")]


def top_relevant_docs(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return [
        compact_doc(item)
        for item in ranked_docs(row)
        if float(item.get("relevance") or 0.0) > 0.0
    ][:limit]


def top_nonrelevant_docs(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return [
        compact_doc(item)
        for item in ranked_docs(row)
        if float(item.get("relevance") or 0.0) <= 0.0
    ][:limit]


def profile_summary(label: str, row: dict[str, Any], top_k: int) -> dict[str, Any]:
    return {
        "label": label,
        "ndcg_at_10": metric(row, "ndcg_at_10"),
        "recall_at_100": metric(row, "recall_at_100"),
        "first_relevant_rank": first_relevant_rank(row),
        "top_doc_ids": top_doc_ids(row, top_k),
        "top_relevant_docs": top_relevant_docs(row, top_k),
        "top_nonrelevant_docs": top_nonrelevant_docs(row, top_k),
    }


def profile_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
    label, row = item
    rank = first_relevant_rank(row)
    rank_key = rank if rank > 0 else 1_000_000
    return (-metric(row, "ndcg_at_10"), -metric(row, "recall_at_100"), rank_key, label)


def best_profile(rows: dict[str, dict[str, Any]], qid: str) -> tuple[str, dict[str, Any]]:
    return sorted(((label, by_qid[qid]) for label, by_qid in rows.items()), key=profile_sort_key)[0]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def profile_specs_by_dataset(specs: list[ProfileSpec], role: str) -> dict[str, dict[str, Path]]:
    by_dataset: dict[str, dict[str, Path]] = {}
    for spec in specs:
        labels = by_dataset.setdefault(spec.dataset, {})
        if spec.label in labels:
            raise ValueError(f"duplicate {role} profile label for {spec.dataset}: {spec.label}")
        labels[spec.label] = spec.path
    return by_dataset


def filters_by_profile(filters: list[RowFilter]) -> dict[tuple[str, str], list[RowFilter]]:
    by_profile: dict[tuple[str, str], list[RowFilter]] = {}
    for row_filter in filters:
        by_profile.setdefault((row_filter.dataset, row_filter.label), []).append(row_filter)
    return by_profile


def validate_query_sets(dataset: str, role: str, rows: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    query_ids: list[str] | None = None
    owner = ""
    for label in sorted(rows):
        current = sorted(rows[label])
        if query_ids is None:
            query_ids = current
            owner = label
            continue
        if current != query_ids:
            missing = sorted(set(query_ids) - set(current))[:5]
            extra = sorted(set(current) - set(query_ids))[:5]
            raise ValueError(
                f"{dataset}: {role} profile {label!r} query_ids differ from {owner!r}; "
                f"missing={missing} extra={extra}"
            )
    return query_ids or []


def load_dataset_profiles(
    eos_specs: dict[str, dict[str, Path]],
    external_specs: dict[str, dict[str, Path]],
    filters: dict[tuple[str, str], list[RowFilter]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    datasets = sorted(set(eos_specs) | set(external_specs))
    if not datasets:
        raise ValueError("at least one --eos and one --external profile are required")
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for dataset in datasets:
        if dataset not in eos_specs:
            raise ValueError(f"{dataset}: missing --eos profile")
        if dataset not in external_specs:
            raise ValueError(f"{dataset}: missing --external profile")
        eos_rows = {
            label: load_per_query(path, filters.get((dataset, label), []))
            for label, path in sorted(eos_specs[dataset].items())
        }
        external_rows = {
            label: load_per_query(path, filters.get((dataset, label), []))
            for label, path in sorted(external_specs[dataset].items())
        }
        eos_ids = validate_query_sets(dataset, "eos", eos_rows)
        external_ids = validate_query_sets(dataset, "external", external_rows)
        if eos_ids != external_ids:
            missing = sorted(set(eos_ids) - set(external_ids))[:5]
            extra = sorted(set(external_ids) - set(eos_ids))[:5]
            raise ValueError(
                f"{dataset}: eos and external query_ids differ; missing_external={missing} extra_external={extra}"
            )
        loaded[dataset] = {"eos": eos_rows, "external": external_rows}
    return loaded


def compare_dataset(
    dataset: str,
    profiles: dict[str, dict[str, dict[str, Any]]],
    *,
    min_gap: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    eos_rows = profiles["eos"]
    external_rows = profiles["external"]
    query_ids = sorted(next(iter(eos_rows.values())))
    ledger: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for qid in query_ids:
        best_eos_label, best_eos_row = best_profile(eos_rows, qid)
        best_external_label, best_external_row = best_profile(external_rows, qid)
        best_eos_ndcg = metric(best_eos_row, "ndcg_at_10")
        best_external_ndcg = metric(best_external_row, "ndcg_at_10")
        gap = best_external_ndcg - best_eos_ndcg
        all_external_beat_eos = all(
            metric(rows[qid], "ndcg_at_10") - best_eos_ndcg > min_gap
            for rows in external_rows.values()
        )
        eos_matches_external = any(
            metric(rows[qid], "ndcg_at_10") >= best_external_ndcg - min_gap
            for rows in eos_rows.values()
        )
        direct_ndcg = metric(eos_rows["direct-fusion"][qid], "ndcg_at_10") if "direct-fusion" in eos_rows else None
        token_span_ndcg = (
            metric(eos_rows["token-span"][qid], "ndcg_at_10") if "token-span" in eos_rows else None
        )
        direct_beats_token_span = (
            direct_ndcg is not None and token_span_ndcg is not None and direct_ndcg > token_span_ndcg
        )
        token_span_beats_direct = (
            direct_ndcg is not None and token_span_ndcg is not None and token_span_ndcg > direct_ndcg
        )
        external_by_label = {
            label: profile_summary(label, rows[qid], top_k) for label, rows in sorted(external_rows.items())
        }
        eos_by_label = {label: profile_summary(label, rows[qid], top_k) for label, rows in sorted(eos_rows.items())}
        row = {
            "schema": SCHEMA + ".per_query",
            "dataset": dataset,
            "query_id": qid,
            "best_eos": profile_summary(best_eos_label, best_eos_row, top_k),
            "best_external": profile_summary(best_external_label, best_external_row, top_k),
            "gap": {
                "best_external_minus_best_eos_ndcg_at_10": gap,
                "best_external_minus_best_eos_recall_at_100": metric(best_external_row, "recall_at_100")
                - metric(best_eos_row, "recall_at_100"),
            },
            "trails_all_externals": all_external_beat_eos,
            "external_consensus_miss": all_external_beat_eos,
            "eos_matches_external": eos_matches_external,
            "direct_fusion_beats_token_span": direct_beats_token_span,
            "token_span_beats_direct_fusion": token_span_beats_direct,
            "eos_profiles": eos_by_label,
            "external_profiles": external_by_label,
            "quality_claim": QUALITY_CLAIM,
        }
        ledger.append(row)
        if all_external_beat_eos:
            candidates.append(
                {
                    "schema": SCHEMA + ".candidate",
                    "dataset": dataset,
                    "query_id": qid,
                    "best_eos": row["best_eos"],
                    "best_external": row["best_external"],
                    "gap": row["gap"],
                    "external_top_relevant_docs": row["best_external"]["top_relevant_docs"],
                    "external_profiles_top_relevant_docs": {
                        label: summary["top_relevant_docs"] for label, summary in external_by_label.items()
                    },
                    "eos_top_nonrelevant_docs": row["best_eos"]["top_nonrelevant_docs"],
                    "quality_claim": QUALITY_CLAIM,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )

    gaps = [row["gap"]["best_external_minus_best_eos_ndcg_at_10"] for row in ledger]
    summary = {
        "dataset": dataset,
        "query_count": len(ledger),
        "mean_best_eos_ndcg_at_10": mean([row["best_eos"]["ndcg_at_10"] for row in ledger]),
        "mean_best_external_ndcg_at_10": mean([row["best_external"]["ndcg_at_10"] for row in ledger]),
        "mean_best_external_minus_best_eos_ndcg_at_10": mean(gaps),
        "count_trails_all_externals": sum(1 for row in ledger if row["trails_all_externals"]),
        "count_external_consensus_misses": sum(1 for row in ledger if row["external_consensus_miss"]),
        "count_eos_matches_external": sum(1 for row in ledger if row["eos_matches_external"]),
        "count_direct_fusion_beats_token_span": sum(
            1 for row in ledger if row["direct_fusion_beats_token_span"]
        ),
        "count_token_span_beats_direct_fusion": sum(
            1 for row in ledger if row["token_span_beats_direct_fusion"]
        ),
        "worst_gaps": [
            {
                "query_id": row["query_id"],
                "best_eos_label": row["best_eos"]["label"],
                "best_external_label": row["best_external"]["label"],
                "gap": row["gap"]["best_external_minus_best_eos_ndcg_at_10"],
                "quality_claim": QUALITY_CLAIM,
            }
            for row in sorted(
                ledger,
                key=lambda item: (
                    -item["gap"]["best_external_minus_best_eos_ndcg_at_10"],
                    item["dataset"],
                    item["query_id"],
                ),
            )[:10]
        ],
        "quality_claim": QUALITY_CLAIM,
    }
    return summary, ledger, candidates


def summarize(
    eos: list[ProfileSpec],
    external: list[ProfileSpec],
    *,
    min_gap: float = 0.0,
    top_k: int = 5,
    max_candidates: int = 0,
    row_filters: list[RowFilter] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    eos_specs = profile_specs_by_dataset(eos, "eos")
    external_specs = profile_specs_by_dataset(external, "external")
    loaded = load_dataset_profiles(eos_specs, external_specs, filters_by_profile(row_filters or []))

    summaries: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for dataset in sorted(loaded):
        dataset_summary, dataset_ledger, dataset_candidates = compare_dataset(
            dataset, loaded[dataset], min_gap=min_gap, top_k=top_k
        )
        summaries.append(dataset_summary)
        ledger.extend(dataset_ledger)
        candidates.extend(dataset_candidates)

    candidates.sort(
        key=lambda row: (
            -row["gap"]["best_external_minus_best_eos_ndcg_at_10"],
            row["dataset"],
            row["query_id"],
        )
    )
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    all_gaps = [row["gap"]["best_external_minus_best_eos_ndcg_at_10"] for row in ledger]
    summary = {
        "schema": SCHEMA,
        "quality_claim": QUALITY_CLAIM,
        "claim_boundary": CLAIM_BOUNDARY,
        "parameters": {"min_gap": min_gap, "top_k_doc_ids": top_k, "max_candidates": max_candidates},
        "profile_inputs": {
            "eos": [
                {"dataset": spec.dataset, "label": spec.label, "path": str(spec.path)}
                for spec in sorted(eos, key=lambda item: (item.dataset, item.label))
            ],
            "external": [
                {"dataset": spec.dataset, "label": spec.label, "path": str(spec.path)}
                for spec in sorted(external, key=lambda item: (item.dataset, item.label))
            ],
        },
        "row_filters": [
            {
                "dataset": row_filter.dataset,
                "label": row_filter.label,
                "field": row_filter.field,
                "value": row_filter.value,
            }
            for row_filter in sorted(row_filters or [], key=lambda item: (item.dataset, item.label, item.field))
        ],
        "dataset_count": len(summaries),
        "query_count": len(ledger),
        "mean_best_eos_ndcg_at_10": mean([row["best_eos"]["ndcg_at_10"] for row in ledger]),
        "mean_best_external_ndcg_at_10": mean([row["best_external"]["ndcg_at_10"] for row in ledger]),
        "mean_best_external_minus_best_eos_ndcg_at_10": mean(all_gaps),
        "count_trails_all_externals": sum(1 for row in ledger if row["trails_all_externals"]),
        "count_external_consensus_misses": sum(1 for row in ledger if row["external_consensus_miss"]),
        "count_eos_matches_external": sum(1 for row in ledger if row["eos_matches_external"]),
        "count_direct_fusion_beats_token_span": sum(
            1 for row in ledger if row["direct_fusion_beats_token_span"]
        ),
        "count_token_span_beats_direct_fusion": sum(
            1 for row in ledger if row["token_span_beats_direct_fusion"]
        ),
        "datasets": summaries,
        "worst_gaps": [
            {
                "dataset": row["dataset"],
                "query_id": row["query_id"],
                "best_eos_label": row["best_eos"]["label"],
                "best_external_label": row["best_external"]["label"],
                "gap": row["gap"]["best_external_minus_best_eos_ndcg_at_10"],
                "quality_claim": QUALITY_CLAIM,
            }
            for row in sorted(
                ledger,
                key=lambda item: (
                    -item["gap"]["best_external_minus_best_eos_ndcg_at_10"],
                    item["dataset"],
                    item["query_id"],
                ),
            )[:20]
        ],
    }
    return summary, ledger, candidates


def format_tsv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_tsv(path: Path, ledger: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in sorted(ledger, key=lambda item: (item["dataset"], item["query_id"])):
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "query_id": row["query_id"],
                    "best_eos_label": row["best_eos"]["label"],
                    "best_eos_ndcg_at_10": row["best_eos"]["ndcg_at_10"],
                    "best_external_label": row["best_external"]["label"],
                    "best_external_ndcg_at_10": row["best_external"]["ndcg_at_10"],
                    "best_external_minus_best_eos_ndcg_at_10": row["gap"][
                        "best_external_minus_best_eos_ndcg_at_10"
                    ],
                    "trails_all_externals": format_tsv_value(row["trails_all_externals"]),
                    "external_consensus_miss": format_tsv_value(row["external_consensus_miss"]),
                    "eos_matches_external": format_tsv_value(row["eos_matches_external"]),
                    "direct_fusion_beats_token_span": format_tsv_value(
                        row["direct_fusion_beats_token_span"]
                    ),
                    "token_span_beats_direct_fusion": format_tsv_value(
                        row["token_span_beats_direct_fusion"]
                    ),
                    "best_external_top_relevant_doc_ids": ",".join(
                        doc["doc_id"] for doc in row["best_external"]["top_relevant_docs"]
                    ),
                    "best_eos_top_nonrelevant_doc_ids": ",".join(
                        doc["doc_id"] for doc in row["best_eos"]["top_nonrelevant_docs"]
                    ),
                    "quality_claim": "false",
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize diagnostic LongEmbed per-query gaps across Eos and external profiles."
    )
    parser.add_argument("--eos", action="append", default=[], type=parse_profile_spec, metavar="DATASET:LABEL=PATH")
    parser.add_argument(
        "--external", action="append", default=[], type=parse_profile_spec, metavar="DATASET:LABEL=PATH"
    )
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument(
        "--row-filter",
        action="append",
        default=[],
        type=parse_row_filter,
        metavar="DATASET:LABEL:FIELD=VALUE",
        help="Exact row filter for one profile; filters are applied before duplicate query_id checks.",
    )
    parser.add_argument("--top-k-docs", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.eos:
        parser.error("at least one --eos DATASET:LABEL=PATH is required")
    if not args.external:
        parser.error("at least one --external DATASET:LABEL=PATH is required")
    try:
        summary, ledger, candidates = summarize(
            args.eos,
            args.external,
            min_gap=args.min_gap,
            top_k=args.top_k_docs,
            max_candidates=args.max_candidates,
            row_filters=args.row_filter,
        )
        if args.output_json:
            write_json(args.output_json, summary)
        else:
            print(json.dumps(summary, indent=2, sort_keys=True))
        if args.output_tsv:
            write_tsv(args.output_tsv, ledger)
        if args.candidate_jsonl:
            write_jsonl(args.candidate_jsonl, candidates)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
