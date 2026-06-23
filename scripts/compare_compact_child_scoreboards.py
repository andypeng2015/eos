#!/usr/bin/env python3
"""Compare compact child-vector scoreboard rows.

This utility is for evaluation/triage only. Its output is not product-quality
evidence and should not be used as a promotion claim without the surrounding
benchmark provenance and release gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


QUALITY_CLAIM = False
EVIDENCE_LABEL = "evaluation/triage only; not product-quality evidence"
VOLATILE_FIELDS = {
    "metrics_path",
    "elapsed_seconds",
    "documents_per_second",
    "queries_per_second",
    "scores_per_second",
}
IDENTITY_FIELDS = (
    "dataset",
    "category",
    "method",
    "bits",
    "output_dim",
    "baseline_kind",
    "forward_backend_kind",
    "dense_vector_bytes",
    "vector_bytes",
    "total_vector_bytes",
)
TSV_FIELDS = (
    "status",
    "identity",
    "dataset",
    "category",
    "method",
    "bits",
    "output_dim",
    "baseline_kind",
    "baseline_ndcg",
    "candidate_ndcg",
    "ndcg_delta",
    "baseline_recall",
    "candidate_recall",
    "recall_delta",
)


@dataclass(frozen=True)
class ScoreRow:
    identity: tuple[tuple[str, str], ...]
    display: dict[str, Any]
    row: dict[str, Any]

    @property
    def identity_text(self) -> str:
        return "\t".join(f"{key}={value}" for key, value in self.identity)


def load_scoreboard(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows = data["rows"]
    else:
        raise ValueError(f"{path}: expected top-level list or object with list field 'rows'")

    bad = [index for index, row in enumerate(rows) if not isinstance(row, dict)]
    if bad:
        raise ValueError(f"{path}: expected object rows; non-object row indexes: {bad[:5]}")
    return rows


def split_filter(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    result: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.add(part)
    return result


def metric_value(row: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if text in {"bm25", "eos", "cuda", "dense"}:
        return text
    if "turboquant-child" in text:
        return "turboquant-child"
    if "dense-child" in text:
        return "dense-child"
    if text.endswith("-child") or "128d-child" in text:
        return "child"
    return text


def infer_bits(row: dict[str, Any]) -> str:
    value = row.get("bits") or row.get("bit_width") or row.get("quant_bits")
    if value is not None:
        return str(value)
    for field in ("method", "baseline", "forward_backend"):
        match = re.search(r"(?:^|[_-])b([248])(?:[_-]|$)", str(row.get(field, "")))
        if match:
            return match.group(1)
    return ""


def infer_output_dim(row: dict[str, Any]) -> str:
    for field in ("output_dim", "dim", "child_dim", "prefix_dim"):
        value = row.get(field)
        if value is not None:
            return str(value)
    for field in ("baseline", "forward_backend", "method"):
        match = re.search(r"(?:^|[-_])(\d+)d(?:[-_]|$)", str(row.get(field, "")))
        if match:
            return match.group(1)
    return ""


def row_display(row: dict[str, Any]) -> dict[str, Any]:
    method = row.get("method")
    baseline_kind = normalize_label(row.get("baseline"))
    if method is None and baseline_kind:
        method = baseline_kind
    return {
        "dataset": str(row.get("dataset", "")),
        "category": str(row.get("category", "")),
        "method": str(method or ""),
        "bits": infer_bits(row),
        "output_dim": infer_output_dim(row),
        "baseline_kind": baseline_kind,
        "forward_backend_kind": normalize_label(row.get("forward_backend")),
        "dense_vector_bytes": row.get("dense_vector_bytes", ""),
        "vector_bytes": row.get("vector_bytes", ""),
        "total_vector_bytes": row.get("total_vector_bytes", ""),
    }


def identity_for(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    display = row_display(row)
    identity = []
    for field in IDENTITY_FIELDS:
        value = display.get(field)
        if value not in ("", None):
            identity.append((field, str(value)))
    return tuple(identity)


def prepare_rows(
    rows: list[dict[str, Any]],
    *,
    datasets: set[str],
    methods: set[str],
    bits: set[str],
) -> list[ScoreRow]:
    prepared: list[ScoreRow] = []
    for row in rows:
        display = row_display(row)
        if datasets and display["dataset"] not in datasets:
            continue
        method_values = {display["method"], display["baseline_kind"]}
        if methods and not (method_values & methods):
            continue
        if bits and display["bits"] not in bits:
            continue
        prepared.append(ScoreRow(identity_for(row), display, row))
    prepared.sort(key=lambda item: item.identity)
    return prepared


def index_rows(rows: list[ScoreRow], label: str) -> dict[tuple[tuple[str, str], ...], ScoreRow]:
    indexed: dict[tuple[tuple[str, str], ...], ScoreRow] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.identity in indexed:
            duplicates.append(row.identity_text)
        indexed[row.identity] = row
    if duplicates:
        joined = "; ".join(duplicates[:5])
        raise ValueError(f"{label}: duplicate stable row identities: {joined}")
    return indexed


def compact_source_row(row: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in sorted(row.items()):
        if key in VOLATILE_FIELDS:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            compact[key] = value
    return compact


def missing_entry(row: ScoreRow, side: str) -> dict[str, Any]:
    return {
        "identity": dict(row.identity),
        "identity_text": row.identity_text,
        "display": row.display,
        "missing_from": side,
    }


def compare_scoreboards(
    baseline_path: Path,
    candidate_path: Path,
    *,
    datasets: set[str] | None = None,
    methods: set[str] | None = None,
    bits: set[str] | None = None,
    min_ndcg_delta: float = 0.0,
    min_recall_delta: float = 0.0,
) -> dict[str, Any]:
    datasets = datasets or set()
    methods = methods or set()
    bits = bits or set()

    baseline_rows = prepare_rows(
        load_scoreboard(baseline_path), datasets=datasets, methods=methods, bits=bits
    )
    candidate_rows = prepare_rows(
        load_scoreboard(candidate_path), datasets=datasets, methods=methods, bits=bits
    )
    baseline_index = index_rows(baseline_rows, "baseline")
    candidate_index = index_rows(candidate_rows, "candidate")

    baseline_keys = set(baseline_index)
    candidate_keys = set(candidate_index)
    matched_keys = sorted(baseline_keys & candidate_keys)
    missing_candidate_keys = sorted(baseline_keys - candidate_keys)
    missing_baseline_keys = sorted(candidate_keys - baseline_keys)

    deltas = []
    for key in matched_keys:
        baseline = baseline_index[key]
        candidate = candidate_index[key]
        baseline_ndcg = metric_value(baseline.row, ("ndcg_at_10", "ndcg_at_100"))
        candidate_ndcg = metric_value(candidate.row, ("ndcg_at_10", "ndcg_at_100"))
        baseline_recall = metric_value(baseline.row, ("recall_at_100", "recall_at_10"))
        candidate_recall = metric_value(candidate.row, ("recall_at_100", "recall_at_10"))
        ndcg_delta = (
            candidate_ndcg - baseline_ndcg
            if candidate_ndcg is not None and baseline_ndcg is not None
            else None
        )
        recall_delta = (
            candidate_recall - baseline_recall
            if candidate_recall is not None and baseline_recall is not None
            else None
        )
        deltas.append(
            {
                "identity": dict(key),
                "identity_text": baseline.identity_text,
                "display": baseline.display,
                "baseline_ndcg": baseline_ndcg,
                "candidate_ndcg": candidate_ndcg,
                "ndcg_delta": ndcg_delta,
                "baseline_recall": baseline_recall,
                "candidate_recall": candidate_recall,
                "recall_delta": recall_delta,
                "baseline_row": compact_source_row(baseline.row),
                "candidate_row": compact_source_row(candidate.row),
            }
        )

    ndcg_deltas = [row["ndcg_delta"] for row in deltas if row["ndcg_delta"] is not None]
    recall_deltas = [row["recall_delta"] for row in deltas if row["recall_delta"] is not None]
    macro = {
        "ndcg_delta": sum(ndcg_deltas) / len(ndcg_deltas) if ndcg_deltas else None,
        "recall_delta": sum(recall_deltas) / len(recall_deltas) if recall_deltas else None,
    }

    decision_reasons: list[str] = []
    if not matched_keys:
        decision_reasons.append("no_matched_rows")
    if missing_candidate_keys:
        decision_reasons.append("missing_candidate_rows")
    if missing_baseline_keys:
        decision_reasons.append("missing_baseline_rows")
    for row in deltas:
        if row["ndcg_delta"] is None or row["ndcg_delta"] < min_ndcg_delta:
            decision_reasons.append(f"ndcg_delta_below_threshold:{row['identity_text']}")
        if row["recall_delta"] is None or row["recall_delta"] < min_recall_delta:
            decision_reasons.append(f"recall_delta_below_threshold:{row['identity_text']}")

    return {
        "quality_claim": QUALITY_CLAIM,
        "evidence_label": EVIDENCE_LABEL,
        "baseline_scoreboard": str(baseline_path),
        "candidate_scoreboard": str(candidate_path),
        "filters": {
            "dataset": sorted(datasets),
            "methods": sorted(methods),
            "bits": sorted(bits),
        },
        "thresholds": {
            "min_ndcg_delta": min_ndcg_delta,
            "min_recall_delta": min_recall_delta,
        },
        "matched_rows": len(deltas),
        "missing_baseline_rows": [
            missing_entry(candidate_index[key], "baseline") for key in missing_baseline_keys
        ],
        "missing_candidate_rows": [
            missing_entry(baseline_index[key], "candidate") for key in missing_candidate_keys
        ],
        "row_deltas": deltas,
        "macro_mean_deltas": macro,
        "decision": "fail" if decision_reasons else "pass",
        "decision_reasons": decision_reasons,
    }


def tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=TSV_FIELDS)
        writer.writeheader()
        for row in summary["row_deltas"]:
            display = row["display"]
            writer.writerow(
                {
                    "status": "matched",
                    "identity": row["identity_text"],
                    "dataset": display.get("dataset", ""),
                    "category": display.get("category", ""),
                    "method": display.get("method", ""),
                    "bits": display.get("bits", ""),
                    "output_dim": display.get("output_dim", ""),
                    "baseline_kind": display.get("baseline_kind", ""),
                    "baseline_ndcg": tsv_value(row["baseline_ndcg"]),
                    "candidate_ndcg": tsv_value(row["candidate_ndcg"]),
                    "ndcg_delta": tsv_value(row["ndcg_delta"]),
                    "baseline_recall": tsv_value(row["baseline_recall"]),
                    "candidate_recall": tsv_value(row["candidate_recall"]),
                    "recall_delta": tsv_value(row["recall_delta"]),
                }
            )
        for status, key in (
            ("missing_baseline", "missing_baseline_rows"),
            ("missing_candidate", "missing_candidate_rows"),
        ):
            for row in summary[key]:
                display = row["display"]
                writer.writerow(
                    {
                        "status": status,
                        "identity": row["identity_text"],
                        "dataset": display.get("dataset", ""),
                        "category": display.get("category", ""),
                        "method": display.get("method", ""),
                        "bits": display.get("bits", ""),
                        "output_dim": display.get("output_dim", ""),
                        "baseline_kind": display.get("baseline_kind", ""),
                        "baseline_ndcg": "",
                        "candidate_ndcg": "",
                        "ndcg_delta": "",
                        "baseline_recall": "",
                        "candidate_recall": "",
                        "recall_delta": "",
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare compact child-vector scoreboards for triage only."
    )
    parser.add_argument("--baseline-scoreboard", required=True, type=Path)
    parser.add_argument("--candidate-scoreboard", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    parser.add_argument("--min-ndcg-delta", type=float, default=0.0)
    parser.add_argument("--min-recall-delta", type=float, default=0.0)
    parser.add_argument("--dataset", action="append", help="Dataset filter; repeat or comma-separate.")
    parser.add_argument("--methods", action="append", help="Method filter; repeat or comma-separate.")
    parser.add_argument("--bits", action="append", help="Bit-width filter; repeat or comma-separate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = compare_scoreboards(
            args.baseline_scoreboard,
            args.candidate_scoreboard,
            datasets=split_filter(args.dataset),
            methods=split_filter(args.methods),
            bits=split_filter(args.bits),
            min_ndcg_delta=args.min_ndcg_delta,
            min_recall_delta=args.min_recall_delta,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_tsv:
        write_tsv(args.output_tsv, summary)
    if not args.output_json and not args.output_tsv:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
