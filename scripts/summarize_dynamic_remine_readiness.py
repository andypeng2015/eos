#!/usr/bin/env python3
"""Summarize dynamic-remine launch readiness from existing evidence.

This utility is read-only with respect to run inputs. It gates the next
guided-negative dynamic-remine smoke on the selected BGE package full gate and
the Stage A imported-BGE teacher-bridge evidence, then writes JSON/TSV summary
artifacts only to the requested output paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as bge_gate


SUMMARY_SCHEMA = "eos.dynamic_remine_readiness_summary.v1"
DEFAULT_STAGEA_ROOT = "runs/retrieval-stagea-msmarco-bge-teacher-score-bridge-v1-scale256-20260629T192026Z"
DEFAULT_BGE_GATE_ROOT = "runs/bge-selected-package-full-gate-v1-20260629T000000Z"
DEFAULT_DESCRIPTOR = ".tiller/scratch/codex/guided-negative-dynamic-remine-plan-v1.md"
DEFAULT_DATASETS = "scifact,nfcorpus,fiqa"
DEFAULT_EXPECTED_CANDIDATES_PER_ROW = 5
DEFAULT_MIN_STAGEA_ROWS = 128
DEFAULT_MIN_POSITIVE_TOP1_RATE = 1.0
DEFAULT_MIN_MARGIN = 0.0
DEFAULT_TEACHER_LABEL = "imported_bge_small_en_v1_5"
DEFAULT_TEACHER_MODEL_ID = (
    "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    "#imported-mll-a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637"
)
GUIDE_FILTER_SCHEMA = "eos.retrieval_teacher_guide_cache_filter.v1"


class ReadinessError(ValueError):
    """Raised when readiness inputs or outputs are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_datasets(value: str) -> list[str]:
    try:
        return bge_gate.parse_datasets(value)
    except bge_gate.SummaryError as exc:
        raise ReadinessError(str(exc)) from exc


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReadinessError(f"required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReadinessError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReadinessError(f"{path}: expected top-level JSON object")
    return data


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def nested_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def legal_gates_are_research_only(gates: dict[str, Any]) -> bool:
    return (
        gates.get("train_allowed_for_research") is True
        and gates.get("release_train_allowed") is False
        and gates.get("commercial_use_allowed") is False
        and gates.get("test_rows_train_allowed") is False
    )


def summarize_bge_gate(run_root: Path, datasets: list[str]) -> dict[str, Any]:
    summary = bge_gate.build_summary(run_root=run_root, datasets=datasets)
    aggregate = summary["aggregate"]
    incomplete = [
        dataset.get("dataset")
        for dataset in summary.get("datasets", [])
        if isinstance(dataset, dict) and dataset.get("status") != "complete"
    ]
    active_export_markers: list[str] = []
    for dataset in summary.get("datasets", []):
        if not isinstance(dataset, dict) or dataset.get("status") == "complete":
            continue
        partial_lines = dataset.get("partial_doc_vector_lines")
        if partial_lines is not None:
            active_export_markers.append(f"{dataset.get('dataset')}: partial doc vector export lines={partial_lines}")
    return {
        "run_root": str(run_root),
        "datasets": datasets,
        "summary_schema": summary.get("schema"),
        "all_complete": aggregate.get("all_complete"),
        "complete_dataset_count": aggregate.get("complete_dataset_count"),
        "expected_dataset_count": aggregate.get("expected_dataset_count"),
        "identity_consistent": aggregate.get("identity_consistent"),
        "incomplete_datasets": incomplete,
        "active_export_markers": active_export_markers,
        "blockers": aggregate.get("blockers", []),
        "summary": summary,
    }


def summarize_stagea_bridge(
    run_root: Path,
    *,
    expected_candidates_per_row: int,
    min_rows: int,
    min_positive_top1_rate: float,
    min_margin: float,
    expected_package_sha256: str,
    expected_identity_sha256: str,
    expected_teacher_model_id: str,
) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    validation_path = run_root / "artifacts" / "validation-summary.json"
    vector_manifest_path = run_root / "vectors" / "manifest.json"
    beir_manifest_path = run_root / "beir" / "manifest.json"
    manifest = load_json_object(manifest_path)
    validation = load_json_object(validation_path)
    vector_manifest = load_json_object(vector_manifest_path)
    beir_manifest = load_json_object(beir_manifest_path)

    coverage = nested_dict(manifest, "coverage")
    vectors = nested_dict(manifest, "vectors")
    beir = nested_dict(manifest, "beir")
    scores = nested_dict(manifest, "scores")
    margin = nested_dict(scores, "margin")
    beir_counts = nested_dict(beir_manifest, "counts")

    examples_seen = as_int(coverage.get("examples_seen"))
    examples_scored = as_int(coverage.get("examples_scored"))
    examples_written = as_int(coverage.get("examples_written"))
    missing_examples = as_int(coverage.get("missing_examples"))
    import_score_rows = as_int(coverage.get("import_score_rows"))
    candidate_rows_scored = as_int(coverage.get("candidate_rows_scored"))
    expected_score_rows = examples_seen * expected_candidates_per_row if examples_seen is not None else None
    doc_vector_rows = as_int(vectors.get("doc_vector_rows"))
    query_vector_rows = as_int(vectors.get("query_vector_rows"))
    beir_docs = as_int(beir.get("corpus_rows"))
    beir_queries = as_int(beir.get("query_rows"))
    beir_manifest_docs = as_int(beir_counts.get("unique_docs"))
    beir_manifest_queries = as_int(beir_counts.get("unique_queries"))
    positive_top1_rate = as_number(scores.get("positive_top1_rate"))
    margin_min = as_number(margin.get("min"))

    identity_match = (
        vector_manifest.get("package_sha256") == expected_package_sha256
        and vector_manifest.get("package_identity_sha256") == expected_identity_sha256
    )
    teacher_match = manifest.get("teacher_model_id") == expected_teacher_model_id

    blockers: list[str] = []
    if examples_seen is None or examples_seen < min_rows:
        blockers.append(f"Stage A rows below minimum: observed={examples_seen} required={min_rows}")
    if not (examples_seen == examples_scored == examples_written):
        blockers.append(
            "Stage A score coverage mismatch: "
            f"examples_seen={examples_seen} examples_scored={examples_scored} examples_written={examples_written}"
        )
    if missing_examples != 0:
        blockers.append(f"Stage A missing_examples not zero: {missing_examples}")
    if import_score_rows != expected_score_rows:
        blockers.append(
            "Stage A import_score_rows mismatch: "
            f"observed={import_score_rows} expected={expected_score_rows}"
        )
    if candidate_rows_scored != expected_score_rows:
        blockers.append(
            "Stage A candidate_rows_scored mismatch: "
            f"observed={candidate_rows_scored} expected={expected_score_rows}"
        )
    if doc_vector_rows != beir_docs or doc_vector_rows != beir_manifest_docs:
        blockers.append(
            "Stage A doc vector rows mismatch: "
            f"vectors={doc_vector_rows} beir={beir_docs} beir_manifest={beir_manifest_docs}"
        )
    if query_vector_rows != beir_queries or query_vector_rows != beir_manifest_queries:
        blockers.append(
            "Stage A query vector rows mismatch: "
            f"vectors={query_vector_rows} beir={beir_queries} beir_manifest={beir_manifest_queries}"
        )
    if positive_top1_rate is None or positive_top1_rate < min_positive_top1_rate:
        blockers.append(
            "Stage A positive_top1_rate below threshold: "
            f"observed={positive_top1_rate} required={min_positive_top1_rate}"
        )
    if margin_min is None or margin_min < min_margin:
        blockers.append(f"Stage A margin min below threshold: observed={margin_min} required={min_margin}")
    if not identity_match:
        blockers.append("Stage A vector package identity mismatch")
    if not teacher_match:
        blockers.append("Stage A teacher_model_id mismatch")

    validation_flags = nested_dict(validation, "validation")
    failed_validation = [key for key, value in validation_flags.items() if value is not True]
    for key in failed_validation:
        blockers.append(f"Stage A validation flag failed: {key}")

    return {
        "run_root": str(run_root),
        "manifest_path": str(manifest_path),
        "validation_summary_path": str(validation_path),
        "vector_manifest_path": str(vector_manifest_path),
        "beir_manifest_path": str(beir_manifest_path),
        "dataset": manifest.get("dataset"),
        "examples_seen": examples_seen,
        "examples_scored": examples_scored,
        "examples_written": examples_written,
        "missing_examples": missing_examples,
        "import_score_rows": import_score_rows,
        "candidate_rows_scored": candidate_rows_scored,
        "expected_candidates_per_row": expected_candidates_per_row,
        "expected_score_rows": expected_score_rows,
        "doc_vector_rows": doc_vector_rows,
        "query_vector_rows": query_vector_rows,
        "beir_docs": beir_docs,
        "beir_queries": beir_queries,
        "positive_top1_rate": positive_top1_rate,
        "min_positive_top1_rate": min_positive_top1_rate,
        "margin_min": margin_min,
        "min_margin": min_margin,
        "teacher_model_id": manifest.get("teacher_model_id"),
        "expected_teacher_model_id": expected_teacher_model_id,
        "identity_match": identity_match,
        "validation_flags": validation_flags,
        "ready": not blockers,
        "blockers": blockers,
    }


def summarize_guide_filter(
    run_root: Path,
    *,
    min_rows: int,
    expected_teacher_label: str,
    expected_package_sha256: str,
    expected_identity_sha256: str,
) -> dict[str, Any]:
    path = run_root / "guide-filter-manifest.json"
    manifest = load_json_object(path)
    counts = nested_dict(manifest, "counts")
    legal_gates = nested_dict(manifest, "legal_gates")
    legal_accounting = nested_dict(manifest, "legal_gate_accounting")
    inputs = nested_dict(manifest, "inputs")
    teacher_caches = nested_dict(inputs, "teacher_caches")
    teacher_cache = nested_dict(teacher_caches, expected_teacher_label)
    teacher_config = nested_dict(teacher_cache, "config")

    rows_seen = as_int(counts.get("rows_seen"))
    rows_emitted = as_int(counts.get("rows_emitted"))
    clean_agreement = as_int(counts.get("clean_agreement"))
    ambiguous = as_int(counts.get("ambiguous_soft_only")) or 0
    conflict = as_int(counts.get("conflict")) or 0
    missing = as_int(counts.get("missing_score_drop")) or 0
    identity_match = (
        teacher_config.get("package_sha256") == expected_package_sha256
        and teacher_config.get("package_identity") == expected_identity_sha256
    )
    legal_ready = legal_gates_are_research_only(legal_gates)
    research_only_preserved = legal_accounting.get("research_only_preserved")

    blockers: list[str] = []
    if manifest.get("schema") != GUIDE_FILTER_SCHEMA:
        blockers.append(f"guide filter schema mismatch: {manifest.get('schema')}")
    if rows_seen is None or rows_seen < min_rows:
        blockers.append(f"guide filter rows_seen below minimum: observed={rows_seen} required={min_rows}")
    if rows_emitted is None or rows_emitted < min_rows:
        blockers.append(f"guide filter rows_emitted below minimum: observed={rows_emitted} required={min_rows}")
    if clean_agreement is None or clean_agreement < min_rows:
        blockers.append(f"guide filter clean_agreement below minimum: observed={clean_agreement} required={min_rows}")
    if missing != 0:
        blockers.append(f"guide filter missing_score_drop not zero: {missing}")
    if conflict != 0:
        blockers.append(f"guide filter conflict not zero for clean smoke: {conflict}")
    if ambiguous != 0:
        blockers.append(f"guide filter ambiguous_soft_only not zero for clean smoke: {ambiguous}")
    if expected_teacher_label not in teacher_caches:
        blockers.append(f"guide filter teacher cache missing: {expected_teacher_label}")
    elif not identity_match:
        blockers.append("guide filter teacher cache package identity mismatch")
    if not legal_ready or research_only_preserved is not True:
        blockers.append("guide filter legal gates are not research-only")
    if manifest.get("quality_claim") is not False:
        blockers.append("guide filter quality_claim must be false")

    return {
        "path": str(path),
        "schema": manifest.get("schema"),
        "rows_seen": rows_seen,
        "rows_emitted": rows_emitted,
        "clean_agreement": clean_agreement,
        "ambiguous_soft_only": ambiguous,
        "conflict": conflict,
        "missing_score_drop": missing,
        "teacher_label": expected_teacher_label,
        "teacher_cache_present": expected_teacher_label in teacher_caches,
        "teacher_model_id": teacher_cache.get("model_id"),
        "identity_match": identity_match,
        "legal_gates": legal_gates,
        "research_only_preserved": research_only_preserved,
        "quality_claim": manifest.get("quality_claim"),
        "ready": not blockers,
        "blockers": blockers,
    }


def build_summary(
    *,
    stagea_root: Path,
    bge_gate_root: Path,
    descriptor: Path,
    datasets: list[str],
    expected_candidates_per_row: int = DEFAULT_EXPECTED_CANDIDATES_PER_ROW,
    min_stagea_rows: int = DEFAULT_MIN_STAGEA_ROWS,
    min_positive_top1_rate: float = DEFAULT_MIN_POSITIVE_TOP1_RATE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    expected_teacher_label: str = DEFAULT_TEACHER_LABEL,
    expected_teacher_model_id: str = DEFAULT_TEACHER_MODEL_ID,
    expected_package_sha256: str = bge_gate.DEFAULT_PACKAGE_SHA256,
    expected_identity_sha256: str = bge_gate.DEFAULT_IDENTITY_SHA256,
    clock: Any = utc_now,
) -> dict[str, Any]:
    bge = summarize_bge_gate(bge_gate_root, datasets)
    stagea = summarize_stagea_bridge(
        stagea_root,
        expected_candidates_per_row=expected_candidates_per_row,
        min_rows=min_stagea_rows,
        min_positive_top1_rate=min_positive_top1_rate,
        min_margin=min_margin,
        expected_package_sha256=expected_package_sha256,
        expected_identity_sha256=expected_identity_sha256,
        expected_teacher_model_id=expected_teacher_model_id,
    )
    guide = summarize_guide_filter(
        stagea_root,
        min_rows=min_stagea_rows,
        expected_teacher_label=expected_teacher_label,
        expected_package_sha256=expected_package_sha256,
        expected_identity_sha256=expected_identity_sha256,
    )
    descriptor_exists = descriptor.exists()

    blockers: list[str] = []
    warnings: list[str] = []
    if not descriptor_exists:
        blockers.append(f"dynamic remine descriptor missing: {descriptor}")
    if not bge["all_complete"]:
        blockers.append("selected BGE gate incomplete")
    if not bge["identity_consistent"]:
        blockers.append("selected BGE gate identity inconsistent")
    for blocker in bge["blockers"]:
        blockers.append(f"bge gate: {blocker}")
    if bge["active_export_markers"]:
        blockers.append("active or partial selected-BGE export marker present")
    if not stagea["ready"]:
        blockers.extend(f"stagea bridge: {blocker}" for blocker in stagea["blockers"])
    if not guide["ready"]:
        blockers.extend(f"guide filter: {blocker}" for blocker in guide["blockers"])
    if not bge["active_export_markers"] and not bge["all_complete"]:
        warnings.append("selected BGE gate is incomplete but no partial export marker was discoverable")

    return {
        "schema": SUMMARY_SCHEMA,
        "created_at": clock(),
        "quality_claim": False,
        "training_run": False,
        "launch_allowed": not blockers,
        "descriptor": {
            "path": str(descriptor),
            "exists": descriptor_exists,
        },
        "thresholds": {
            "expected_candidates_per_row": expected_candidates_per_row,
            "min_stagea_rows": min_stagea_rows,
            "min_positive_top1_rate": min_positive_top1_rate,
            "min_margin": min_margin,
        },
        "expected_teacher": {
            "label": expected_teacher_label,
            "model_id": expected_teacher_model_id,
            "package_sha256": expected_package_sha256,
            "identity_sha256": expected_identity_sha256,
        },
        "bge_gate": bge,
        "stagea_bridge": stagea,
        "guide_filter": guide,
        "blockers": blockers,
        "warnings": warnings,
    }


def format_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def threshold_status(value: Any, threshold: Any) -> str:
    value_number = as_number(value)
    threshold_number = as_number(threshold)
    if value_number is None or threshold_number is None:
        return "block"
    return "pass" if value_number >= threshold_number else "block"


def equality_status(value: Any, expected: Any) -> str:
    return "pass" if value is not None and value == expected else "block"


def tsv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"section": "summary", "key": "launch_allowed", "value": summary["launch_allowed"], "status": ""},
        {"section": "summary", "key": "quality_claim", "value": summary["quality_claim"], "status": ""},
        {"section": "summary", "key": "training_run", "value": summary["training_run"], "status": ""},
        {"section": "descriptor", "key": "exists", "value": summary["descriptor"]["exists"], "status": ""},
        {"section": "bge_gate", "key": "all_complete", "value": summary["bge_gate"]["all_complete"], "status": ""},
        {
            "section": "bge_gate",
            "key": "identity_consistent",
            "value": summary["bge_gate"]["identity_consistent"],
            "status": "",
        },
        {
            "section": "bge_gate",
            "key": "complete_dataset_count",
            "value": summary["bge_gate"]["complete_dataset_count"],
            "expected": summary["bge_gate"]["expected_dataset_count"],
            "status": "pass" if summary["bge_gate"]["all_complete"] else "block",
        },
        {
            "section": "stagea_bridge",
            "key": "ready",
            "value": summary["stagea_bridge"]["ready"],
            "status": "pass" if summary["stagea_bridge"]["ready"] else "block",
        },
        {
            "section": "stagea_bridge",
            "key": "examples_seen",
            "value": summary["stagea_bridge"]["examples_seen"],
            "expected": summary["thresholds"]["min_stagea_rows"],
            "status": threshold_status(
                summary["stagea_bridge"]["examples_seen"],
                summary["thresholds"]["min_stagea_rows"],
            ),
        },
        {
            "section": "stagea_bridge",
            "key": "import_score_rows",
            "value": summary["stagea_bridge"]["import_score_rows"],
            "expected": summary["stagea_bridge"]["expected_score_rows"],
            "status": equality_status(
                summary["stagea_bridge"]["import_score_rows"],
                summary["stagea_bridge"]["expected_score_rows"],
            ),
        },
        {
            "section": "stagea_bridge",
            "key": "positive_top1_rate",
            "value": summary["stagea_bridge"]["positive_top1_rate"],
            "expected": summary["thresholds"]["min_positive_top1_rate"],
            "status": threshold_status(
                summary["stagea_bridge"]["positive_top1_rate"],
                summary["thresholds"]["min_positive_top1_rate"],
            ),
        },
        {
            "section": "stagea_bridge",
            "key": "margin_min",
            "value": summary["stagea_bridge"]["margin_min"],
            "expected": summary["thresholds"]["min_margin"],
            "status": threshold_status(
                summary["stagea_bridge"]["margin_min"],
                summary["thresholds"]["min_margin"],
            ),
        },
        {
            "section": "guide_filter",
            "key": "ready",
            "value": summary["guide_filter"]["ready"],
            "status": "pass" if summary["guide_filter"]["ready"] else "block",
        },
        {
            "section": "guide_filter",
            "key": "clean_agreement",
            "value": summary["guide_filter"]["clean_agreement"],
            "expected": summary["thresholds"]["min_stagea_rows"],
            "status": threshold_status(
                summary["guide_filter"]["clean_agreement"],
                summary["thresholds"]["min_stagea_rows"],
            ),
        },
        {
            "section": "guide_filter",
            "key": "research_only_preserved",
            "value": summary["guide_filter"]["research_only_preserved"],
            "status": "pass" if summary["guide_filter"]["research_only_preserved"] else "block",
        },
    ]
    rows.extend(
        {"section": "blocker", "key": str(index), "value": blocker, "status": "block"}
        for index, blocker in enumerate(summary["blockers"], start=1)
    )
    rows.extend(
        {"section": "warning", "key": str(index), "value": warning, "status": "warn"}
        for index, warning in enumerate(summary["warnings"], start=1)
    )
    return rows


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "key", "value", "expected", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in tsv_rows(summary):
            writer.writerow({key: format_tsv_value(row.get(key)) for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stagea-root", default=DEFAULT_STAGEA_ROOT)
    parser.add_argument("--bge-gate-root", default=DEFAULT_BGE_GATE_ROOT)
    parser.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--expected-candidates-per-row", type=int, default=DEFAULT_EXPECTED_CANDIDATES_PER_ROW)
    parser.add_argument("--min-stagea-rows", type=int, default=DEFAULT_MIN_STAGEA_ROWS)
    parser.add_argument("--min-positive-top1-rate", type=float, default=DEFAULT_MIN_POSITIVE_TOP1_RATE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--teacher-label", default=DEFAULT_TEACHER_LABEL)
    parser.add_argument("--teacher-model-id", default=DEFAULT_TEACHER_MODEL_ID)
    parser.add_argument("--package-sha256", default=bge_gate.DEFAULT_PACKAGE_SHA256)
    parser.add_argument("--identity-sha256", default=bge_gate.DEFAULT_IDENTITY_SHA256)
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_summary(
            stagea_root=Path(args.stagea_root),
            bge_gate_root=Path(args.bge_gate_root),
            descriptor=Path(args.descriptor),
            datasets=parse_datasets(args.datasets),
            expected_candidates_per_row=args.expected_candidates_per_row,
            min_stagea_rows=args.min_stagea_rows,
            min_positive_top1_rate=args.min_positive_top1_rate,
            min_margin=args.min_margin,
            expected_teacher_label=args.teacher_label,
            expected_teacher_model_id=args.teacher_model_id,
            expected_package_sha256=args.package_sha256,
            expected_identity_sha256=args.identity_sha256,
        )
        output_json = Path(args.output_json) if args.output_json else Path(args.stagea_root) / "dynamic-remine-readiness-summary.json"
        output_tsv = Path(args.output_tsv) if args.output_tsv else Path(args.stagea_root) / "dynamic-remine-readiness-summary.tsv"
        write_json(output_json, summary)
        write_tsv(output_tsv, summary)
    except (ReadinessError, bge_gate.SummaryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.require_ready and not summary["launch_allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
