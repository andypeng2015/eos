#!/usr/bin/env python3
"""Bounded MS MARCO teacher-scoring smoke over candidate rows.

This helper intentionally scores only a deterministic prefix of the audited
candidate JSONL. It does not create training rows or promotion artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-requests", type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--schema", default="eos.msmarco_bounded_teacher_scoring_smoke.v1")
    parser.add_argument("--report-title", default="Bounded Teacher Scoring Smoke")
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
        raise SystemExit("--max-rows must be positive for this smoke")
    rows: list[dict[str, Any]] = []
    for _, row in iter_jsonl(path):
        query = stable_text(row.get("query"))
        positive = stable_text(row.get("positive"))
        negatives = [stable_text(value) for value in row.get("negatives") or []]
        if not query or not positive or not negatives:
            raise ValueError(f"{path}: candidate row {len(rows)} is missing query/positive/negatives")
        rows.append(row)
        if len(rows) >= max_rows:
            break
    if not rows:
        raise SystemExit("no candidate rows selected")
    return rows


def load_dev_positive_doc_ids(source_manifest: dict[str, Any]) -> set[str]:
    dataset_root = Path(source_manifest["dataset_root"])
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
            "Missing scoring dependencies. Use the existing venv if available: "
            ".venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py ... "
            f"Original import error: {exc}"
        ) from exc
    return SentenceTransformer, torch


def encode_texts(model: Any, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
    encoded = model.encode(
        [prefix + text for text in texts],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [[float(value) for value in row.tolist()] for row in encoded]


def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimension mismatch: {len(left)} != {len(right)}")
    return float(sum(a * b for a, b in zip(left, right)))


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
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
    if args.source_requests is not None and not args.source_requests.is_file():
        raise FileNotFoundError(f"missing source requests JSONL: {args.source_requests}")
    legal_gate = source_manifest.get("legal_gate") or {}
    if legal_gate.get("release_train_allowed") is not False:
        raise SystemExit("source manifest does not preserve release_train_allowed=false")
    if legal_gate.get("commercial_use_allowed") is not False:
        raise SystemExit("source manifest does not preserve commercial_use_allowed=false")

    candidates = load_candidates(args.candidates, args.max_rows)
    dev_positive_doc_ids = load_dev_positive_doc_ids(source_manifest)

    query_texts: list[str] = []
    query_index: dict[str, int] = {}
    doc_texts: list[str] = []
    doc_index: dict[str, int] = {}
    for row in candidates:
        query = stable_text(row["query"])
        if query not in query_index:
            query_index[query] = len(query_texts)
            query_texts.append(query)
        for candidate in [row["positive"], *(row.get("negatives") or [])]:
            text = stable_text(candidate)
            if text not in doc_index:
                doc_index[text] = len(doc_texts)
                doc_texts.append(text)

    SentenceTransformer, torch = require_sentence_transformer()
    cuda_available = bool(torch.cuda.is_available())
    if args.device.startswith("cuda") and not cuda_available:
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    model = SentenceTransformer(args.model_id, device=args.device, local_files_only=args.local_files_only)

    query_vectors = encode_texts(model, query_texts, args.query_prefix, args.batch_size)
    doc_vectors = encode_texts(model, doc_texts, args.document_prefix, args.batch_size)

    scored_path = artifacts / "msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl"
    scores_path = artifacts / "msmarco-passage.qwen3-0.6b.teacher-scores.jsonl"
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    positive_top1 = 0
    score_rows = 0
    flagged_negative_refs = 0
    flagged_rows = 0

    with scored_path.open("w", encoding="utf-8") as scored_handle, scores_path.open(
        "w", encoding="utf-8"
    ) as scores_handle:
        for row_index, row in enumerate(candidates):
            query = stable_text(row["query"])
            positive = stable_text(row["positive"])
            negatives = [stable_text(value) for value in row.get("negatives") or []]
            negative_doc_ids = [str(value) for value in row.get("negative_doc_ids") or []]
            qv = query_vectors[query_index[query]]
            candidates_text = [positive, *negatives]
            row_scores = [dot(qv, doc_vectors[doc_index[text]]) for text in candidates_text]
            positive_scores.append(row_scores[0])
            negative_scores.extend(row_scores[1:])
            best_negative = max(row_scores[1:])
            margin = row_scores[0] - best_negative
            margins.append(margin)
            if row_scores[0] >= best_negative:
                positive_top1 += 1
            row_flagged = 0
            for candidate_index, (candidate, score) in enumerate(zip(candidates_text, row_scores)):
                label = "positive" if candidate_index == 0 else "negative"
                candidate_doc_id = str(row.get("positive_doc_id")) if candidate_index == 0 else ""
                if candidate_index > 0 and candidate_index - 1 < len(negative_doc_ids):
                    candidate_doc_id = negative_doc_ids[candidate_index - 1]
                is_dev_positive_negative = label == "negative" and candidate_doc_id in dev_positive_doc_ids
                if is_dev_positive_negative:
                    flagged_negative_refs += 1
                    row_flagged += 1
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
                            "label": label,
                            "score": score,
                            "score_scale": "cosine_normalized_dot",
                            "teacher_model_id": args.model_id,
                            "is_global_dev_positive_negative": is_dev_positive_negative,
                            "release_train_allowed": False,
                            "commercial_use_allowed": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                score_rows += 1
            if row_flagged:
                flagged_rows += 1
            out_row = dict(row)
            out_row["teacher_model_id"] = args.model_id
            out_row["teacher_score_scale"] = "cosine_normalized_dot"
            out_row["teacher_scores"] = row_scores
            out_row["teacher_positive_margin_over_best_negative"] = margin
            out_row["teacher_positive_top1"] = row_scores[0] >= best_negative
            out_row["global_dev_positive_negative_count"] = row_flagged
            out_row["release_train_allowed"] = False
            out_row["commercial_use_allowed"] = False
            scored_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    elapsed_seconds = time.time() - start
    manifest = {
        "schema": args.schema,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_root": str(args.run_root.resolve()),
        "source_candidate_manifest": str(args.source_manifest.resolve()),
        "source_candidate_jsonl": str(args.candidates.resolve()),
        "source_teacher_score_requests_jsonl": str(args.source_requests.resolve()) if args.source_requests else None,
        "teacher_model": {
            "model_id": args.model_id,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "query_prefix": args.query_prefix,
            "document_prefix": args.document_prefix,
            "score_scale": "cosine_normalized_dot",
        },
        "sample": {
            "strategy": "deterministic prefix of audited teacher candidate rows",
            "requested_max_rows": args.max_rows,
            "candidate_rows_scored": len(candidates),
            "teacher_score_rows": score_rows,
            "unique_query_texts_encoded": len(query_texts),
            "unique_candidate_texts_encoded": len(doc_texts),
            "full_source_candidate_rows": source_manifest.get("counts", {}).get("candidate_rows"),
            "full_source_teacher_request_rows": source_manifest.get("counts", {}).get("teacher_request_rows"),
        },
        "legal_gate": {
            "release_train_allowed": False,
            "commercial_use_allowed": False,
            "train_allowed_for_research": bool(legal_gate.get("train_allowed_for_research")),
            "policy_basis": "inherited from source candidate manifest; scoring evidence only",
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
            "teacher_score_rows": score_rows,
            "missing_score_rows": 0,
            "global_dev_positive_negative_refs": flagged_negative_refs,
            "rows_with_global_dev_positive_negative": flagged_rows,
        },
        "metrics": {
            "positive_top1_rate": positive_top1 / len(candidates),
            "positive_mean_margin_over_best_negative": statistics.fmean(margins),
            "positive_scores": summary(positive_scores),
            "negative_scores": summary(negative_scores),
            "margins": summary(margins),
        },
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
            "teacher_score_requests_jsonl": sha256_file(args.source_requests) if args.source_requests else None,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "helper": str(Path(__file__).resolve()),
        },
    }
    manifest_path = args.run_root / "manifest.json"
    write_json(manifest_path, manifest)
    report_path = reports / "teacher-agreement-margin-report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {args.report_title}",
                "",
                f"- teacher: `{args.model_id}`",
                f"- bound: first `{len(candidates)}` candidate rows, `{score_rows}` score rows",
                "- score scale: `cosine_normalized_dot`",
                f"- positive top-1 rate: `{manifest['metrics']['positive_top1_rate']:.6f}`",
                f"- positive mean margin over best negative: `{manifest['metrics']['positive_mean_margin_over_best_negative']:.6f}`",
                f"- missing score rows: `{manifest['counts']['missing_score_rows']}`",
                f"- global-dev-positive negative refs in subset: `{flagged_negative_refs}`",
                f"- rows with global-dev-positive negatives: `{flagged_rows}`",
                "- release_train_allowed: `false`",
                "- commercial_use_allowed: `false`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")
    print(f"wrote scores: {scores_path}")
    print(f"wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
