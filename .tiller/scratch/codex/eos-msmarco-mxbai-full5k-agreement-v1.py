#!/usr/bin/env python3
"""Score the audited MS MARCO 5k substrate with mxbai and compare to Qwen3."""

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


DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--qwen-scored", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-requests", required=True, type=Path)
    parser.add_argument("--qwen-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--low-margin-threshold", type=float, default=0.0)
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


def load_candidates(path: Path, max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0:
        raise SystemExit("--max-rows must be positive")
    rows: list[dict[str, Any]] = []
    for _, row in iter_jsonl(path):
        query = stable_text(row.get("query"))
        positive = stable_text(row.get("positive"))
        negatives = [stable_text(value) for value in row.get("negatives") or []]
        if not query or not positive or len(negatives) != 3:
            raise ValueError(f"{path}: candidate row {len(rows)} is missing query/positive/3 negatives")
        rows.append(row)
        if len(rows) >= max_rows:
            break
    if len(rows) != max_rows:
        raise SystemExit(f"selected {len(rows)} rows, expected {max_rows}")
    return rows


def load_qwen_rows(path: Path, max_rows: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for _, row in iter_jsonl(path):
        row_id = str(row.get("row_id") or "")
        if not row_id:
            raise ValueError(f"{path}: missing row_id")
        rows[row_id] = row
        if len(rows) >= max_rows:
            break
    if len(rows) != max_rows:
        raise SystemExit(f"loaded {len(rows)} Qwen rows, expected {max_rows}")
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
            "Missing mxbai scoring dependencies. Recommended command: "
            ".venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1.py ... "
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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return numerator / (dx * dy)


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
    qwen_manifest = json.loads(args.qwen_manifest.read_text(encoding="utf-8"))
    legal_gate = source_manifest.get("legal_gate") or {}
    if legal_gate.get("release_train_allowed") is not False:
        raise SystemExit("source manifest does not preserve release_train_allowed=false")
    if legal_gate.get("commercial_use_allowed") is not False:
        raise SystemExit("source manifest does not preserve commercial_use_allowed=false")
    if qwen_manifest.get("counts", {}).get("candidate_rows_scored") != args.max_rows:
        raise SystemExit("Qwen manifest does not match requested row bound")

    candidates = load_candidates(args.candidates, args.max_rows)
    qwen_rows = load_qwen_rows(args.qwen_scored, args.max_rows)
    dev_positive_doc_ids = load_dev_positive_doc_ids(source_manifest)

    query_texts: list[str] = []
    query_index: dict[str, int] = {}
    doc_texts: list[str] = []
    doc_index: dict[str, int] = {}
    for row in candidates:
        row_id = str(row.get("row_id") or "")
        if row_id not in qwen_rows:
            raise ValueError(f"candidate row_id missing from Qwen scored rows: {row_id}")
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

    scored_path = artifacts / "msmarco-passage.mxbai-large.scored.teacher-candidates.jsonl"
    scores_path = artifacts / "msmarco-passage.mxbai-large.teacher-scores.jsonl"
    agreement_path = reports / "qwen3-mxbai-agreement-report.json"

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    qwen_margins: list[float] = []
    mxbai_margins: list[float] = []
    positive_top1 = 0
    score_rows = 0
    flagged_negative_refs = 0
    flagged_rows = 0
    both_positive_top1 = 0
    qwen_non_top1: list[dict[str, Any]] = []
    mxbai_non_top1: list[dict[str, Any]] = []
    either_failed: list[dict[str, Any]] = []
    agreement_filter_count = 0
    agreement_filter_excluded = {
        "global_dev_positive_negative": 0,
        "qwen_non_top1_or_low_margin": 0,
        "mxbai_non_top1_or_low_margin": 0,
    }

    with scored_path.open("w", encoding="utf-8") as scored_handle, scores_path.open(
        "w", encoding="utf-8"
    ) as scores_handle:
        for row_index, row in enumerate(candidates):
            row_id = str(row.get("row_id") or "")
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
            mxbai_top1 = row_scores[0] >= best_negative
            if mxbai_top1:
                positive_top1 += 1

            qwen_row = qwen_rows[row_id]
            qwen_margin = float(qwen_row.get("teacher_positive_margin_over_best_negative"))
            qwen_top1 = bool(qwen_row.get("teacher_positive_top1"))
            qwen_margins.append(qwen_margin)
            mxbai_margins.append(margin)
            if qwen_top1 and mxbai_top1:
                both_positive_top1 += 1
            if not qwen_top1:
                qwen_non_top1.append({"row_index": row_index, "row_id": row_id, "qwen_margin": qwen_margin})
            if not mxbai_top1:
                mxbai_non_top1.append({"row_index": row_index, "row_id": row_id, "mxbai_margin": margin})
            if not qwen_top1 or not mxbai_top1:
                either_failed.append(
                    {
                        "row_index": row_index,
                        "row_id": row_id,
                        "query_id": row.get("query_id"),
                        "qwen_top1": qwen_top1,
                        "qwen_margin": qwen_margin,
                        "mxbai_top1": mxbai_top1,
                        "mxbai_margin": margin,
                    }
                )

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
                            "row_id": row_id,
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

            qwen_ok = qwen_top1 and qwen_margin > args.low_margin_threshold
            mxbai_ok = mxbai_top1 and margin > args.low_margin_threshold
            if row_flagged:
                agreement_filter_excluded["global_dev_positive_negative"] += 1
            elif not qwen_ok:
                agreement_filter_excluded["qwen_non_top1_or_low_margin"] += 1
            elif not mxbai_ok:
                agreement_filter_excluded["mxbai_non_top1_or_low_margin"] += 1
            else:
                agreement_filter_count += 1

            out_row = dict(row)
            out_row["teacher_model_id"] = args.model_id
            out_row["teacher_score_scale"] = "cosine_normalized_dot"
            out_row["teacher_scores"] = row_scores
            out_row["teacher_positive_margin_over_best_negative"] = margin
            out_row["teacher_positive_top1"] = mxbai_top1
            out_row["global_dev_positive_negative_count"] = row_flagged
            out_row["qwen3_teacher_model_id"] = qwen_row.get("teacher_model_id")
            out_row["qwen3_teacher_positive_margin_over_best_negative"] = qwen_margin
            out_row["qwen3_teacher_positive_top1"] = qwen_top1
            out_row["qwen3_mxbai_both_positive_top1"] = qwen_top1 and mxbai_top1
            out_row["agreement_filter_candidate"] = row_flagged == 0 and qwen_ok and mxbai_ok
            out_row["release_train_allowed"] = False
            out_row["commercial_use_allowed"] = False
            scored_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    elapsed_seconds = time.time() - start
    agreement_report = {
        "schema": "eos.msmarco_qwen3_mxbai_full5k_agreement.v1",
        "candidate_rows": len(candidates),
        "both_positive_top1_count": both_positive_top1,
        "both_positive_top1_rate": both_positive_top1 / len(candidates),
        "qwen_positive_top1_count": len(candidates) - len(qwen_non_top1),
        "mxbai_positive_top1_count": positive_top1,
        "qwen_non_top1_count": len(qwen_non_top1),
        "mxbai_non_top1_count": len(mxbai_non_top1),
        "either_failed_count": len(either_failed),
        "either_failed_examples": either_failed[:20],
        "qwen_non_top1_examples": qwen_non_top1[:20],
        "mxbai_non_top1_examples": mxbai_non_top1[:20],
        "margin_correlation_pearson": pearson(qwen_margins, mxbai_margins),
        "qwen_margins": summary(qwen_margins),
        "mxbai_margins": summary(mxbai_margins),
        "margin_delta_mxbai_minus_qwen": summary([b - a for a, b in zip(qwen_margins, mxbai_margins)]),
        "global_dev_positive_negative_refs": flagged_negative_refs,
        "rows_with_global_dev_positive_negative": flagged_rows,
        "low_margin_threshold": args.low_margin_threshold,
        "agreement_filter_candidate_count": agreement_filter_count,
        "agreement_filter_candidate_rate": agreement_filter_count / len(candidates),
        "agreement_filter_excluded": agreement_filter_excluded,
    }
    write_json(agreement_path, agreement_report)

    manifest = {
        "schema": "eos.msmarco_mxbai_full5k_agreement.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_root": str(args.run_root.resolve()),
        "source_candidate_manifest": str(args.source_manifest.resolve()),
        "source_candidate_jsonl": str(args.candidates.resolve()),
        "source_teacher_score_requests_jsonl": str(args.source_requests.resolve()),
        "qwen3_source_manifest": str(args.qwen_manifest.resolve()),
        "qwen3_scored_candidates_jsonl": str(args.qwen_scored.resolve()),
        "teacher_model": {
            "model_id": args.model_id,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "query_prefix": args.query_prefix,
            "document_prefix": args.document_prefix,
            "score_scale": "cosine_normalized_dot",
        },
        "sample": {
            "strategy": "exact same deterministic prefix of audited teacher candidate rows as Qwen3 full5k",
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
            "policy_basis": "inherited from source candidate manifest; scoring/agreement evidence only",
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
            "qwen3_mxbai_agreement": agreement_report,
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
            "agreement_report_json": {
                "path": str(agreement_path.resolve()),
                "sha256": sha256_file(agreement_path),
            },
        },
        "source_sha256": {
            "candidate_jsonl": sha256_file(args.candidates),
            "source_manifest": sha256_file(args.source_manifest),
            "teacher_score_requests_jsonl": sha256_file(args.source_requests),
            "qwen3_manifest": sha256_file(args.qwen_manifest),
            "qwen3_scored_candidates_jsonl": sha256_file(args.qwen_scored),
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
                "# mxbai Full 5k Teacher Scoring And Qwen3 Agreement",
                "",
                f"- teacher: `{args.model_id}`",
                f"- bound: first `{len(candidates)}` candidate rows, `{score_rows}` score rows",
                "- score scale: `cosine_normalized_dot`",
                f"- mxbai positive top-1 rate: `{manifest['metrics']['positive_top1_rate']:.6f}`",
                f"- mxbai positive mean margin over best negative: `{manifest['metrics']['positive_mean_margin_over_best_negative']:.6f}`",
                f"- Qwen3/mxbai both-positive-top1 rate: `{agreement_report['both_positive_top1_rate']:.6f}`",
                f"- margin correlation pearson: `{agreement_report['margin_correlation_pearson']:.6f}`",
                f"- agreement-filter candidate rows: `{agreement_filter_count}`",
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
    print(f"wrote scored candidates: {scored_path}")
    print(f"wrote agreement report: {agreement_path}")
    print(f"wrote markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
