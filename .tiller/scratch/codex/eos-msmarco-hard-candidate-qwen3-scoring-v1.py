#!/usr/bin/env python3
"""Score mined MS MARCO hard-candidate rows with local Qwen3 embeddings.

This helper emits teacher-score evidence only. It preserves restrictive legal
gates and supports variable negative-candidate counts per row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--schema", default="eos.msmarco_hard_candidate_qwen3_scoring.v1")
    parser.add_argument("--report-title", default="Qwen3 Hard-Candidate Teacher Scoring")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def load_candidates(path: Path, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0:
        raise SystemExit("--max-rows must be positive")
    rows: list[dict[str, Any]] = []
    for _, row in iter_jsonl(path):
        query = stable_text(row.get("query"))
        positive = stable_text(row.get("positive"))
        negatives = [stable_text(value) for value in row.get("negatives") or []]
        negative_doc_ids = [str(value) for value in row.get("negative_doc_ids") or []]
        if not query or not positive or not negatives:
            raise ValueError(f"{path}: row {len(rows) + 1} is missing query/positive/negatives")
        if len(negatives) != len(negative_doc_ids):
            raise ValueError(
                f"{path}: row {len(rows) + 1} has {len(negatives)} negatives but "
                f"{len(negative_doc_ids)} negative_doc_ids"
            )
        if len(negatives) < 16 or len(negatives) > 32:
            raise ValueError(f"{path}: row {len(rows) + 1} has unsupported negative count {len(negatives)}")
        rows.append(row)
        if len(rows) >= max_rows:
            break
    if not rows:
        raise SystemExit("no candidate rows selected")
    return rows


def dataset_root_from_manifest(source_manifest: dict[str, Any]) -> Path:
    if source_manifest.get("dataset_root"):
        return Path(source_manifest["dataset_root"])
    source_data = source_manifest.get("source_data") or {}
    if source_data.get("dataset_root"):
        return Path(source_data["dataset_root"])
    raise KeyError("source manifest does not expose dataset_root or source_data.dataset_root")


def load_dev_positive_doc_ids(source_manifest: dict[str, Any]) -> set[str]:
    dataset_root = dataset_root_from_manifest(source_manifest)
    dev_qrels = dataset_root / "qrels" / "dev.tsv"
    if not dev_qrels.is_file():
        raise FileNotFoundError(f"missing dev qrels for dev-positive negative flagging: {dev_qrels}")
    doc_ids: set[str] = set()
    with dev_qrels.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("query-id"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                doc_ids.add(parts[1])
    return doc_ids


def require_sentence_transformer():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing scoring dependencies. Use .venv-qwen3 with local Qwen3 packages. "
            f"Original import error: {exc}"
        ) from exc
    return SentenceTransformer, torch


def encode_texts(model: Any, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    encoded = model.encode(
        [prefix + text for text in texts],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(encoded, dtype=np.float32)


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "pstdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def int_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    start = time.time()
    args.run_root.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_root / "artifacts"
    reports = args.run_root / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    legal_gate = source_manifest.get("legal_gate") or {}
    if legal_gate.get("release_train_allowed") is not False:
        raise SystemExit("source manifest does not preserve release_train_allowed=false")
    if legal_gate.get("commercial_use_allowed") is not False:
        raise SystemExit("source manifest does not preserve commercial_use_allowed=false")
    if legal_gate.get("train_allowed_for_research") is not True:
        raise SystemExit("source manifest does not preserve train_allowed_for_research=true")

    candidates = load_candidates(args.candidates, args.max_rows)
    dev_positive_doc_ids = load_dev_positive_doc_ids(source_manifest)

    query_texts: list[str] = []
    query_index: dict[str, int] = {}
    doc_texts: list[str] = []
    doc_index: dict[str, int] = {}
    candidate_count_distribution: Counter[int] = Counter()
    manifest_flag_refs = 0
    manifest_flag_rows = 0

    for row in candidates:
        query = stable_text(row["query"])
        if query not in query_index:
            query_index[query] = len(query_texts)
            query_texts.append(query)
        row_doc_texts = [row["positive"], *(row.get("negatives") or [])]
        for candidate in row_doc_texts:
            text = stable_text(candidate)
            if text not in doc_index:
                doc_index[text] = len(doc_texts)
                doc_texts.append(text)
        negative_count = len(row.get("negatives") or [])
        candidate_count_distribution[negative_count] += 1
        row_manifest_flags = sum(1 for value in (row.get("candidate_dev_positive_flags") or {}).values() if value)
        manifest_flag_refs += row_manifest_flags
        manifest_flag_rows += bool(row_manifest_flags)

    expected_score_rows = len(candidates) + sum(k * v for k, v in candidate_count_distribution.items())

    SentenceTransformer, torch = require_sentence_transformer()
    cuda_available = bool(torch.cuda.is_available())
    if args.device.startswith("cuda") and not cuda_available:
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    model = SentenceTransformer(args.model_id, device=args.device, local_files_only=args.local_files_only)

    query_vectors = encode_texts(model, query_texts, args.query_prefix, args.batch_size)
    doc_vectors = encode_texts(model, doc_texts, args.document_prefix, args.batch_size)

    scored_path = artifacts / "msmarco-passage.hard-candidates.qwen3-0.6b.scored.jsonl"
    scores_path = artifacts / "msmarco-passage.hard-candidates.qwen3-0.6b.teacher-scores.jsonl"

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    positive_ranks: list[int] = []
    positive_top1 = 0
    score_rows = 0
    flagged_negative_refs = 0
    flagged_rows = 0
    missing_score_rows = 0
    best_negative_primary_sources: Counter[str] = Counter()
    best_negative_all_sources: Counter[str] = Counter()
    non_top1_best_negative_primary_sources: Counter[str] = Counter()
    source_negative_counts: Counter[str] = Counter()
    source_candidate_rank_values: dict[str, list[int]] = defaultdict(list)
    source_negative_rank_values: dict[str, list[int]] = defaultdict(list)
    source_best_negative_margins: dict[str, list[float]] = defaultdict(list)

    with scored_path.open("w", encoding="utf-8") as scored_handle, scores_path.open(
        "w", encoding="utf-8"
    ) as scores_handle:
        for row_index, row in enumerate(candidates):
            query = stable_text(row["query"])
            positive = stable_text(row["positive"])
            negatives = [stable_text(value) for value in row.get("negatives") or []]
            negative_doc_ids = [str(value) for value in row.get("negative_doc_ids") or []]
            candidate_doc_ids = [str(row.get("positive_doc_id")), *negative_doc_ids]
            candidate_texts = [positive, *negatives]
            qv = query_vectors[query_index[query]]
            doc_positions = [doc_index[text] for text in candidate_texts]
            row_scores_np = doc_vectors[doc_positions] @ qv
            row_scores = [float(value) for value in row_scores_np.tolist()]
            positive_score = row_scores[0]
            negative_row_scores = row_scores[1:]
            best_negative = max(negative_row_scores)
            best_negative_offset = negative_row_scores.index(best_negative)
            best_negative_doc_id = negative_doc_ids[best_negative_offset]
            margin = positive_score - best_negative
            sorted_candidate_indexes = sorted(range(len(row_scores)), key=lambda idx: row_scores[idx], reverse=True)
            rank_by_index = {idx: rank for rank, idx in enumerate(sorted_candidate_indexes, start=1)}
            positive_rank = rank_by_index[0]
            positive_is_top1 = positive_rank == 1

            positive_scores.append(positive_score)
            negative_scores.extend(negative_row_scores)
            margins.append(margin)
            positive_ranks.append(positive_rank)
            positive_top1 += positive_is_top1

            row_flagged = 0
            candidate_sources = row.get("candidate_sources") or {}
            best_sources = [str(value) for value in candidate_sources.get(best_negative_doc_id, [])]
            primary_best_source = best_sources[0] if best_sources else "unknown"
            best_negative_primary_sources[primary_best_source] += 1
            if not positive_is_top1:
                non_top1_best_negative_primary_sources[primary_best_source] += 1
            for source in best_sources or ["unknown"]:
                best_negative_all_sources[source] += 1
                source_best_negative_margins[source].append(margin)

            for candidate_index, (candidate, score, candidate_doc_id) in enumerate(
                zip(candidate_texts, row_scores, candidate_doc_ids)
            ):
                label = "positive" if candidate_index == 0 else "negative"
                candidate_rank = rank_by_index[candidate_index]
                is_dev_positive_negative = label == "negative" and candidate_doc_id in dev_positive_doc_ids
                if is_dev_positive_negative:
                    flagged_negative_refs += 1
                    row_flagged += 1
                sources = [] if label == "positive" else [str(value) for value in candidate_sources.get(candidate_doc_id, [])]
                if label == "negative":
                    for source in sources or ["unknown"]:
                        source_negative_counts[source] += 1
                        source_candidate_rank_values[source].append(candidate_rank)
                        source_negative_rank_values[source].append(1 + sorted(negative_row_scores, reverse=True).index(score))
                scores_handle.write(
                    json.dumps(
                        {
                            "source": row.get("source"),
                            "row_id": row.get("row_id"),
                            "query_id": row.get("query_id"),
                            "query": query,
                            "candidate": candidate,
                            "candidate_doc_id": candidate_doc_id,
                            "candidate_index": candidate_index,
                            "candidate_rank": candidate_rank,
                            "label": label,
                            "score": score,
                            "score_scale": "cosine_normalized_dot",
                            "teacher_model_id": args.model_id,
                            "candidate_sources": sources,
                            "is_global_dev_positive_negative": is_dev_positive_negative,
                            "release_train_allowed": False,
                            "commercial_use_allowed": False,
                            "train_allowed_for_research": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                score_rows += 1
            if row_flagged:
                flagged_rows += 1

            out_row = dict(row)
            out_row["teacher_scores_ready"] = True
            out_row["teacher_model_id"] = args.model_id
            out_row["teacher_score_scale"] = "cosine_normalized_dot"
            out_row["teacher_scores"] = row_scores
            out_row["teacher_positive_margin_over_best_negative"] = margin
            out_row["teacher_positive_rank"] = positive_rank
            out_row["teacher_positive_top1"] = positive_is_top1
            out_row["teacher_best_negative_doc_id"] = best_negative_doc_id
            out_row["teacher_best_negative_score"] = best_negative
            out_row["teacher_best_negative_sources"] = best_sources
            out_row["global_dev_positive_negative_count"] = row_flagged
            out_row["release_train_allowed"] = False
            out_row["commercial_use_allowed"] = False
            out_row["train_allowed_for_research"] = True
            scored_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    if score_rows != expected_score_rows:
        missing_score_rows = expected_score_rows - score_rows

    elapsed_seconds = time.time() - start
    source_analytics = {
        "negative_source_counts": dict(sorted(source_negative_counts.items())),
        "best_negative_primary_source_counts": dict(sorted(best_negative_primary_sources.items())),
        "best_negative_all_source_counts": dict(sorted(best_negative_all_sources.items())),
        "non_top1_best_negative_primary_source_counts": dict(sorted(non_top1_best_negative_primary_sources.items())),
        "source_candidate_rank_summary": {
            source: int_summary(values) for source, values in sorted(source_candidate_rank_values.items())
        },
        "source_negative_only_rank_summary": {
            source: int_summary(values) for source, values in sorted(source_negative_rank_values.items())
        },
        "source_best_negative_margin_summary": {
            source: numeric_summary(values) for source, values in sorted(source_best_negative_margins.items())
        },
    }

    manifest = {
        "schema": args.schema,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_root": str(args.run_root.resolve()),
        "source_candidate_manifest": str(args.source_manifest.resolve()),
        "source_candidate_jsonl": str(args.candidates.resolve()),
        "teacher_model": {
            "model_id": args.model_id,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "query_prefix": args.query_prefix,
            "document_prefix": args.document_prefix,
            "score_scale": "cosine_normalized_dot",
        },
        "sample": {
            "strategy": "deterministic prefix of hard-candidate rows",
            "requested_max_rows": args.max_rows,
            "candidate_rows_scored": len(candidates),
            "teacher_score_rows": score_rows,
            "expected_teacher_score_rows": expected_score_rows,
            "unique_query_texts_encoded": len(query_texts),
            "unique_candidate_texts_encoded": len(doc_texts),
            "full_source_candidate_rows": source_manifest.get("counts", {}).get("output_rows"),
            "full_source_negative_candidates": source_manifest.get("counts", {}).get("negative_candidates_total"),
        },
        "legal_gate": {
            "release_train_allowed": False,
            "commercial_use_allowed": False,
            "train_allowed_for_research": True,
            "policy_basis": "inherited from source hard-candidate manifest; scoring evidence only",
        },
        "preflight": {
            "torch_version": getattr(torch, "__version__", ""),
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "",
            "batch_size": args.batch_size,
        },
        "counts": {
            "candidate_rows_scored": len(candidates),
            "negative_candidates_scored": expected_score_rows - len(candidates),
            "teacher_score_rows": score_rows,
            "expected_teacher_score_rows": expected_score_rows,
            "missing_score_rows": missing_score_rows,
            "global_dev_positive_negative_refs": flagged_negative_refs,
            "rows_with_global_dev_positive_negative": flagged_rows,
            "source_manifest_global_dev_positive_negative_refs": source_manifest.get("leak_audit", {}).get(
                "global_dev_positive_negative_refs"
            ),
            "row_embedded_dev_positive_flag_refs": manifest_flag_refs,
            "row_embedded_dev_positive_flag_rows": manifest_flag_rows,
        },
        "candidate_counts": {
            "negative_count_distribution": dict(sorted((str(k), v) for k, v in candidate_count_distribution.items())),
            "negative_count_summary": int_summary(
                [count for count, freq in candidate_count_distribution.items() for _ in range(freq)]
            ),
        },
        "metrics": {
            "positive_top1_rate": positive_top1 / len(candidates),
            "positive_top1_count": positive_top1,
            "positive_mean_rank": statistics.fmean(positive_ranks),
            "positive_mean_margin_over_best_negative": statistics.fmean(margins),
            "positive_scores": numeric_summary(positive_scores),
            "negative_scores": numeric_summary(negative_scores),
            "margins": numeric_summary(margins),
            "positive_ranks": int_summary(positive_ranks),
        },
        "source_analytics": source_analytics,
        "artifacts": {
            "teacher_score_jsonl": {
                "path": str(scores_path.resolve()),
                "rows": score_rows,
                "sha256": sha256_file(scores_path),
            },
            "scored_candidates_jsonl": {
                "path": str(scored_path.resolve()),
                "rows": len(candidates),
                "sha256": sha256_file(scored_path),
            },
        },
        "source_sha256": {
            "candidate_jsonl": sha256_file(args.candidates),
            "source_manifest": sha256_file(args.source_manifest),
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "helper": str(Path(__file__).resolve()),
        },
    }
    manifest_path = args.run_root / "manifest.json"
    write_json(manifest_path, manifest)

    source_lines = [
        f"  - {source}: `{count}` best negatives"
        for source, count in sorted(best_negative_primary_sources.items(), key=lambda item: (-item[1], item[0]))
    ]
    report_path = reports / "teacher-agreement-margin-report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {args.report_title}",
                "",
                f"- teacher: `{args.model_id}`",
                f"- rows scored: `{len(candidates)}`",
                f"- score rows: `{score_rows}`",
                f"- unique query texts encoded: `{len(query_texts)}`",
                f"- unique candidate texts encoded: `{len(doc_texts)}`",
                "- score scale: `cosine_normalized_dot`",
                f"- positive top-1 rate: `{manifest['metrics']['positive_top1_rate']:.6f}` "
                f"(`{positive_top1}/{len(candidates)}`)",
                f"- positive mean margin over best negative: "
                f"`{manifest['metrics']['positive_mean_margin_over_best_negative']:.6f}`",
                f"- missing score rows: `{missing_score_rows}`",
                f"- candidate count distribution: "
                f"`{json.dumps(manifest['candidate_counts']['negative_count_distribution'], sort_keys=True)}`",
                f"- global-dev-positive negative refs: `{flagged_negative_refs}`",
                f"- rows with global-dev-positive negatives: `{flagged_rows}`",
                f"- row-embedded global-dev-positive refs: `{manifest_flag_refs}`",
                f"- source manifest global-dev-positive refs: "
                f"`{manifest['counts']['source_manifest_global_dev_positive_negative_refs']}`",
                "- release_train_allowed: `false`",
                "- commercial_use_allowed: `false`",
                "- train_allowed_for_research: `true`",
                "",
                "## Best Negative Primary Source Counts",
                "",
                *(source_lines or ["- none"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"wrote manifest: {manifest_path}")
    print(f"wrote scores: {scores_path}")
    print(f"wrote scored candidates: {scored_path}")
    print(f"wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
