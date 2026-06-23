#!/usr/bin/env python3
"""Summarize long-context wedge comparison rows into a product-wedge report.

This utility consumes existing `eval_eos_long_context_wedge.fw` comparison JSON
files. The output is diagnostic evidence only: it preserves the claim boundary
and does not make a product-quality claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


QUALITY_CLAIM = False
EVIDENCE_LABEL = "diagnostic long-context comparison summary; not product-quality evidence"
SUMMARY_SCHEMA = "eos.long_context_wedge_comparison_summary.v1"
DEFAULT_EXTERNAL_MODELS = ("qwen3-0.6b", "mxbai-large")
DEFAULT_EXTERNAL_BITS = (4,)
TSV_FIELDS = (
    "dataset",
    "eos_method",
    "eos_ndcg",
    "external_model",
    "external_method",
    "external_ndcg",
    "ndcg_delta",
    "storage_ratio",
    "eos_beats_external",
    "missing_external",
    "quality_claim",
)


def text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def number(value: Any, *, field: str, path: Path, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: row {row_index}: expected numeric {field}, got {value!r}")
    return float(value)


def optional_number(value: Any, *, field: str, path: Path, row_index: int) -> float | None:
    if value is None or value == "":
        return None
    return number(value, field=field, path=path, row_index=row_index)


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "baseline": text(row.get("baseline")),
        "model": text(row.get("model")),
        "method": text(row.get("method")),
        "bits": row.get("bits"),
        "ndcg_at_10": row.get("ndcg_at_10"),
        "recall_at_100": row.get("recall_at_100"),
        "child_count": row.get("child_count"),
        "storage_multiple": row.get("storage_multiple"),
    }
    if row.get("metrics_path"):
        result["metrics_path"] = row["metrics_path"]
    return result


def parse_bits(values: list[str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return default
    bits: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                bits.add(int(part))
            except ValueError as exc:
                raise ValueError(f"invalid bit width {part!r}") from exc
    return tuple(sorted(bits))


def parse_input_spec(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if label.strip():
            return label.strip(), Path(path)
    return None, Path(value)


def infer_dataset_label(path: Path, rows: list[dict[str, Any]], explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    row_datasets = sorted({text(row.get("dataset")) for row in rows if text(row.get("dataset"))})
    if len(row_datasets) == 1:
        return row_datasets[0]

    parent = path.parent.name
    known = (
        "2wikimqa",
        "qmsum",
        "needle",
        "passkey",
        "repo-docs",
        "scifact",
        "nfcorpus",
        "fiqa",
    )
    lowered = parent.lower()
    for marker in known:
        if re.search(rf"(^|[-_]){re.escape(marker)}($|[-_])", lowered):
            return marker
    return parent or path.stem


def load_comparison(path: Path, *, allow_quality_claims: bool) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected top-level list")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index}: expected object row")
        if row.get("quality_claim") is True and not allow_quality_claims:
            raise ValueError(
                f"{path}: row {index}: quality_claim=true; pass --allow-quality-claims for inspection"
            )
        number(row.get("ndcg_at_10"), field="ndcg_at_10", path=path, row_index=index)
        number(row.get("recall_at_100"), field="recall_at_100", path=path, row_index=index)
        for field in ("bits", "child_count", "storage_multiple", "compression_ratio", "dim"):
            if field in row and row[field] is not None:
                optional_number(row[field], field=field, path=path, row_index=index)
        rows.append(row)
    return rows


def metric(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def bits(row: dict[str, Any]) -> int | None:
    value = row.get("bits")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -metric(row, "ndcg_at_10"),
        -metric(row, "recall_at_100"),
        text(row.get("baseline")),
        text(row.get("model")),
        text(row.get("method")),
        bits(row) if bits(row) is not None else -1,
    )


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=row_sort_key)[0]


def is_eos_row(row: dict[str, Any]) -> bool:
    return text(row.get("model")).lower() == "eos" or text(row.get("baseline")).lower() == "eos"


def is_required_external_row(
    row: dict[str, Any],
    *,
    model: str,
    bit_width: int,
) -> bool:
    return (
        text(row.get("baseline")).lower() == "external_chunked"
        and text(row.get("model")).lower() == model.lower()
        and bits(row) == bit_width
    )


def row_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return compact_row(row)


def storage_ratio(eos_row: dict[str, Any] | None, external_row: dict[str, Any] | None) -> float | None:
    if eos_row is None or external_row is None:
        return None
    eos_storage = finite_or_none(metric(eos_row, "storage_multiple"))
    external_storage = finite_or_none(metric(external_row, "storage_multiple"))
    if eos_storage is None or external_storage is None or external_storage == 0.0:
        return None
    return eos_storage / external_storage


def comparison_for(
    dataset: str,
    eos_row: dict[str, Any] | None,
    external_row: dict[str, Any] | None,
    *,
    model: str,
    bit_width: int,
) -> dict[str, Any]:
    missing = external_row is None
    eos_ndcg = metric(eos_row, "ndcg_at_10") if eos_row is not None else None
    external_ndcg = metric(external_row, "ndcg_at_10") if external_row is not None else None
    eos_recall = metric(eos_row, "recall_at_100") if eos_row is not None else None
    external_recall = metric(external_row, "recall_at_100") if external_row is not None else None
    ndcg_delta = None if eos_ndcg is None or external_ndcg is None else eos_ndcg - external_ndcg
    recall_delta = None if eos_recall is None or external_recall is None else eos_recall - external_recall
    return {
        "dataset": dataset,
        "external_model": model,
        "external_bits": bit_width,
        "missing_external": missing,
        "external": row_summary(external_row),
        "ndcg_delta": ndcg_delta,
        "recall_delta": recall_delta,
        "storage_multiple_ratio_eos_over_external": storage_ratio(eos_row, external_row),
        "eos_beats_external": bool(ndcg_delta is not None and ndcg_delta >= 0.0),
    }


def summarize_one_input(
    *,
    label: str,
    path: Path,
    rows: list[dict[str, Any]],
    external_models: tuple[str, ...],
    external_bits: tuple[int, ...],
    preferred_bits: int,
) -> dict[str, Any]:
    eos_rows = [row for row in rows if is_eos_row(row)]
    best_eos = best_row(eos_rows)
    preferred_eos = best_row([row for row in eos_rows if bits(row) == preferred_bits])
    if preferred_eos is None:
        preferred_eos = best_row([row for row in eos_rows if (bits(row) or 0) > 0])

    comparisons = []
    missing_required: list[dict[str, Any]] = []
    required_external_rows: list[dict[str, Any]] = []
    for model in external_models:
        for bit_width in external_bits:
            matches = [
                row
                for row in rows
                if is_required_external_row(row, model=model, bit_width=bit_width)
            ]
            external = best_row(matches)
            if external is None:
                missing_required.append({"dataset": label, "model": model, "bits": bit_width})
            else:
                required_external_rows.append(external)
            comparisons.append(
                comparison_for(label, best_eos, external, model=model, bit_width=bit_width)
            )

    present = [comparison for comparison in comparisons if not comparison["missing_external"]]
    beats_any = any(comparison["eos_beats_external"] for comparison in present)
    beats_all = bool(present) and len(present) == len(comparisons) and all(
        comparison["eos_beats_external"] for comparison in present
    )
    return {
        "dataset": label,
        "source_path": str(path),
        "quality_claim": QUALITY_CLAIM,
        "row_count": len(rows),
        "required_external_models": list(external_models),
        "required_external_bits": list(external_bits),
        "best_eos": row_summary(best_eos),
        "best_eos_preferred_bits": row_summary(preferred_eos),
        "preferred_bits": preferred_bits,
        "best_external_required": row_summary(best_row(required_external_rows)),
        "external_comparisons": comparisons,
        "missing_required_external_rows": missing_required,
        "eos_beats_any_required_external": beats_any,
        "eos_beats_all_required_externals": beats_all,
    }


def mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def summarize_comparisons(
    input_specs: list[str],
    *,
    external_models: tuple[str, ...] = DEFAULT_EXTERNAL_MODELS,
    external_bits: tuple[int, ...] = DEFAULT_EXTERNAL_BITS,
    preferred_bits: int = 4,
    allow_quality_claims: bool = False,
) -> dict[str, Any]:
    datasets = []
    inputs = []
    for spec in input_specs:
        explicit_label, path = parse_input_spec(spec)
        rows = load_comparison(path, allow_quality_claims=allow_quality_claims)
        label = infer_dataset_label(path, rows, explicit_label)
        inputs.append({"dataset": label, "path": str(path), "row_count": len(rows)})
        datasets.append(
            summarize_one_input(
                label=label,
                path=path,
                rows=rows,
                external_models=external_models,
                external_bits=external_bits,
                preferred_bits=preferred_bits,
            )
        )

    datasets.sort(key=lambda item: (item["dataset"], item["source_path"]))
    missing_required = [
        missing
        for dataset in datasets
        for missing in dataset["missing_required_external_rows"]
    ]
    eos_ndcgs = [
        float(dataset["best_eos"]["ndcg_at_10"])
        for dataset in datasets
        if dataset["best_eos"] is not None
    ]
    external_ndcgs = [
        float(dataset["best_external_required"]["ndcg_at_10"])
        for dataset in datasets
        if dataset["best_external_required"] is not None
    ]
    best_external_gaps = [
        float(dataset["best_eos"]["ndcg_at_10"]) - float(dataset["best_external_required"]["ndcg_at_10"])
        for dataset in datasets
        if dataset["best_eos"] is not None and dataset["best_external_required"] is not None
    ]

    decision_reasons: list[str] = []
    if missing_required:
        decision_reasons.append("missing_required_external_rows")
    no_eos = [dataset["dataset"] for dataset in datasets if dataset["best_eos"] is None]
    if no_eos:
        decision_reasons.append("missing_eos_rows")
    trailing = [
        {
            "dataset": dataset["dataset"],
            "external_model": comparison["external_model"],
            "external_bits": comparison["external_bits"],
            "ndcg_delta": comparison["ndcg_delta"],
        }
        for dataset in datasets
        for comparison in dataset["external_comparisons"]
        if not comparison["missing_external"] and not comparison["eos_beats_external"]
    ]
    if trailing:
        decision_reasons.append("eos_trails_required_external_ndcg")
    decision = "pass" if not missing_required and not no_eos and not trailing else "fail"
    if decision == "pass":
        decision_reasons.append("best_eos_ndcg_meets_or_beats_all_required_external_rows")

    return {
        "schema": SUMMARY_SCHEMA,
        "quality_claim": QUALITY_CLAIM,
        "evidence_label": EVIDENCE_LABEL,
        "required_external_baseline": "external_chunked",
        "required_external_models": list(external_models),
        "required_external_bits": list(external_bits),
        "preferred_bits": preferred_bits,
        "allow_quality_claims_input_inspection": allow_quality_claims,
        "inputs": sorted(inputs, key=lambda item: (item["dataset"], item["path"])),
        "dataset_count": len(datasets),
        "missing_required_external_rows": missing_required,
        "datasets_where_eos_beats_any_external": [
            dataset["dataset"] for dataset in datasets if dataset["eos_beats_any_required_external"]
        ],
        "datasets_where_eos_beats_all_external": [
            dataset["dataset"] for dataset in datasets if dataset["eos_beats_all_required_externals"]
        ],
        "mean_best_eos_ndcg_at_10": mean(eos_ndcgs),
        "mean_best_external_ndcg_at_10": mean(external_ndcgs),
        "mean_gap_eos_minus_best_external_ndcg_at_10": mean(best_external_gaps),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "trailing_required_external_rows": trailing,
        "datasets": datasets,
    }


def tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
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
        for dataset in summary["datasets"]:
            eos = dataset["best_eos"] or {}
            for comparison in dataset["external_comparisons"]:
                external = comparison["external"] or {}
                writer.writerow(
                    {
                        "dataset": dataset["dataset"],
                        "eos_method": eos.get("method", ""),
                        "eos_ndcg": tsv_value(eos.get("ndcg_at_10")),
                        "external_model": comparison["external_model"],
                        "external_method": external.get("method", ""),
                        "external_ndcg": tsv_value(external.get("ndcg_at_10")),
                        "ndcg_delta": tsv_value(comparison["ndcg_delta"]),
                        "storage_ratio": tsv_value(
                            comparison["storage_multiple_ratio_eos_over_external"]
                        ),
                        "eos_beats_external": tsv_value(comparison["eos_beats_external"]),
                        "missing_external": tsv_value(comparison["missing_external"]),
                        "quality_claim": tsv_value(summary["quality_claim"]),
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize long-context wedge comparison JSON files for diagnostic triage."
    )
    parser.add_argument("comparison_json", nargs="+", help="Path or dataset=path comparison JSON input.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    parser.add_argument("--preferred-bits", type=int, default=4)
    parser.add_argument(
        "--external-model",
        action="append",
        dest="external_models",
        help="Required external model; repeatable. Defaults to qwen3-0.6b and mxbai-large.",
    )
    parser.add_argument(
        "--external-bits",
        action="append",
        help="Required external bit widths; repeat or comma-separate. Defaults to 4.",
    )
    parser.add_argument(
        "--allow-quality-claims",
        action="store_true",
        help="Allow inspection of inputs containing quality_claim=true; output remains quality_claim=false.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        external_models = tuple(args.external_models or DEFAULT_EXTERNAL_MODELS)
        external_bits = parse_bits(args.external_bits, DEFAULT_EXTERNAL_BITS)
        summary = summarize_comparisons(
            args.comparison_json,
            external_models=external_models,
            external_bits=external_bits,
            preferred_bits=args.preferred_bits,
            allow_quality_claims=args.allow_quality_claims,
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
    else:
        print(
            "summarized long-context wedge comparisons: "
            f"datasets={summary['dataset_count']} decision={summary['decision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
