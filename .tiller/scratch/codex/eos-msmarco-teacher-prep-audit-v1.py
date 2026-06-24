#!/usr/bin/env python3
"""Build a bounded provenance-safe MS MARCO teacher candidate audit set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sample-qrels", type=int, default=5000)
    parser.add_argument("--negatives-per-row", type=int, default=3)
    parser.add_argument("--max-leak-examples", type=int, default=20)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_id(split: str, query_id: str, positive_doc_id: str, negative_doc_ids: list[str]) -> str:
    payload = json.dumps(
        {
            "source": "msmarco-passage",
            "split": split,
            "query_id": query_id,
            "positive_doc_id": positive_doc_id,
            "negative_doc_ids": negative_doc_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def iter_qrels(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=2):
            query_id = str(row.get("query-id") or row.get("query_id") or "").strip()
            doc_id = str(row.get("corpus-id") or row.get("corpus_id") or "").strip()
            try:
                score = float(row.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            if query_id and doc_id and score > 0:
                yield line_number, query_id, doc_id, score


def load_qrel_positive_map(path: Path) -> tuple[dict[str, set[str]], set[str], int]:
    positives: dict[str, set[str]] = defaultdict(set)
    doc_ids: set[str] = set()
    rows = 0
    for _, query_id, doc_id, _ in iter_qrels(path):
        positives[query_id].add(doc_id)
        doc_ids.add(doc_id)
        rows += 1
    return positives, doc_ids, rows


def load_sampled_train_qrels(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for line_number, query_id, doc_id, score in iter_qrels(path):
        pair = (query_id, doc_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rows.append(
            {
                "source_qrels_line": line_number,
                "query_id": query_id,
                "positive_doc_id": doc_id,
                "qrel_score": score,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def deterministic_negative_ids(
    query_id: str,
    positive_doc_id: str,
    known_positive_ids: set[str],
    corpus_rows: int,
    count: int,
) -> list[str]:
    negatives: list[str] = []
    seen = {positive_doc_id}
    seed = sha256_text(f"msmarco-passage\0{query_id}\0{positive_doc_id}")
    cursor = int(seed[:16], 16) % corpus_rows
    stride = (int(seed[16:32], 16) % 104729) + 1
    attempts = 0
    while len(negatives) < count:
        candidate = str((cursor + attempts * stride) % corpus_rows)
        attempts += 1
        if candidate in seen or candidate in known_positive_ids:
            continue
        seen.add(candidate)
        negatives.append(candidate)
    return negatives


def load_queries(path: Path, needed_query_ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for _, row in iter_jsonl(path):
        query_id = str(row.get("_id") or row.get("id") or "")
        if query_id in needed_query_ids:
            found[query_id] = str(row.get("text") or row.get("query") or "")
            if len(found) == len(needed_query_ids):
                break
    return found


def load_corpus_texts(path: Path, needed_doc_ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for _, row in iter_jsonl(path):
        doc_id = str(row.get("_id") or row.get("id") or "")
        if doc_id not in needed_doc_ids:
            continue
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        found[doc_id] = f"{title}\n{text}" if title and text else title or text
        if len(found) == len(needed_doc_ids):
            break
    return found


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_report(manifest: dict[str, Any], audit: dict[str, Any]) -> str:
    examples = audit["negative_leak_examples"]
    example_text = "none" if not examples else json.dumps(examples[:5], indent=2, sort_keys=True)
    return "\n".join(
        [
            "# MS MARCO Teacher Prep Leak/Split Audit",
            "",
            f"- Run root: `{manifest['run_root']}`",
            f"- Candidate rows: `{manifest['counts']['candidate_rows']}`",
            f"- Teacher request rows: `{manifest['counts']['teacher_request_rows']}`",
            f"- Split: `{manifest['sample']['split']}`",
            f"- Sampled train qrel rows: `{manifest['sample']['sample_qrels']}`",
            f"- Negatives per row: `{manifest['sample']['negatives_per_row']}`",
            f"- Same-query negative positive leaks: `{audit['candidate_negative_positive_leaks']}`",
            f"- Same-query dev-positive negative leaks: `{audit['candidate_negative_dev_positive_same_query']}`",
            f"- Dev qrels used for candidate construction: `{audit['dev_qrels_used_for_candidate_construction']}`",
            f"- Rows missing split provenance: `{audit['rows_missing_split_provenance']}`",
            f"- Rows missing row_id provenance: `{audit['rows_missing_row_id']}`",
            f"- Global train/dev positive doc overlap: `{audit['global_train_dev_positive_doc_overlap']}`",
            f"- Candidate negatives that are any dev-positive doc: `{audit['candidate_negative_any_dev_positive_doc_ids']}`",
            f"- Gate status: `{audit['gate_status']}`",
            f"- Legal gate: `release_train_allowed={str(manifest['legal_gate']['release_train_allowed']).lower()}`; "
            f"`commercial_use_allowed={str(manifest['legal_gate']['commercial_use_allowed']).lower()}`",
            "",
            "## Negative Leak Examples",
            "",
            "```json",
            example_text,
            "```",
            "",
            "## Artifacts",
            "",
            f"- Candidate JSONL: `{manifest['artifacts']['candidate_jsonl']['path']}`",
            f"- Teacher score request JSONL: `{manifest['artifacts']['teacher_score_requests_jsonl']['path']}`",
            f"- Manifest JSON: `{manifest['artifacts']['manifest_json']['path']}`",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    if args.sample_qrels <= 0:
        raise SystemExit("--sample-qrels must be positive")
    if args.negatives_per_row <= 0:
        raise SystemExit("--negatives-per-row must be positive")

    dataset_root = args.dataset_root.resolve()
    source_manifest_path = args.source_manifest.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(source_manifest_path)
    corpus_path = dataset_root / "corpus.jsonl"
    queries_path = dataset_root / "queries.jsonl"
    train_qrels_path = dataset_root / "qrels" / "train.tsv"
    dev_qrels_path = dataset_root / "qrels" / "dev.tsv"
    for path in (corpus_path, queries_path, train_qrels_path, dev_qrels_path):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")

    corpus_rows = int(source_manifest["counts"]["beir"]["corpus"]["rows"])
    train_positives, train_positive_docs, train_qrel_rows = load_qrel_positive_map(train_qrels_path)
    dev_positives, dev_positive_docs, dev_qrel_rows = load_qrel_positive_map(dev_qrels_path)
    sampled_qrels = load_sampled_train_qrels(train_qrels_path, args.sample_qrels)

    candidate_specs: list[dict[str, Any]] = []
    needed_query_ids: set[str] = set()
    needed_doc_ids: set[str] = set()
    for qrel in sampled_qrels:
        query_id = qrel["query_id"]
        positive_doc_id = qrel["positive_doc_id"]
        known_positive_ids = set(train_positives.get(query_id, set()))
        known_positive_ids.update(dev_positives.get(query_id, set()))
        negative_doc_ids = deterministic_negative_ids(
            query_id=query_id,
            positive_doc_id=positive_doc_id,
            known_positive_ids=known_positive_ids,
            corpus_rows=corpus_rows,
            count=args.negatives_per_row,
        )
        candidate_specs.append({**qrel, "negative_doc_ids": negative_doc_ids})
        needed_query_ids.add(query_id)
        needed_doc_ids.add(positive_doc_id)
        needed_doc_ids.update(negative_doc_ids)

    query_texts = load_queries(queries_path, needed_query_ids)
    doc_texts = load_corpus_texts(corpus_path, needed_doc_ids)
    missing_query_ids = sorted(needed_query_ids - set(query_texts))
    missing_doc_ids = sorted(needed_doc_ids - set(doc_texts))
    if missing_query_ids or missing_doc_ids:
        raise SystemExit(
            "sample could not resolve texts: "
            f"missing_query_ids={missing_query_ids[:5]} missing_doc_ids={missing_doc_ids[:5]}"
        )

    candidate_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    leak_examples: list[dict[str, Any]] = []
    negative_positive_leaks = 0
    negative_dev_same_query = 0
    negative_any_dev_positive_docs: set[str] = set()
    rows_missing_split = 0
    rows_missing_row_id = 0
    for example_index, spec in enumerate(candidate_specs):
        query_id = spec["query_id"]
        positive_doc_id = spec["positive_doc_id"]
        negative_doc_ids = list(spec["negative_doc_ids"])
        row_id = stable_row_id("train", query_id, positive_doc_id, negative_doc_ids)
        row = {
            "source": "msmarco-passage",
            "split": "train",
            "query_id": query_id,
            "positive_doc_id": positive_doc_id,
            "negative_doc_ids": negative_doc_ids,
            "row_id": row_id,
            "source_qrels_line": spec["source_qrels_line"],
            "qrel_score": spec["qrel_score"],
            "query": query_texts[query_id],
            "positive": doc_texts[positive_doc_id],
            "negatives": [doc_texts[doc_id] for doc_id in negative_doc_ids],
            "release_train_allowed": False,
            "commercial_use_allowed": False,
            "teacher_scores_ready": False,
        }
        if row.get("split") != "train":
            rows_missing_split += 1
        if not row.get("row_id"):
            rows_missing_row_id += 1
        for negative_doc_id in negative_doc_ids:
            is_train_positive = negative_doc_id in train_positives.get(query_id, set())
            is_dev_positive = negative_doc_id in dev_positives.get(query_id, set())
            if is_train_positive or is_dev_positive:
                negative_positive_leaks += 1
                if len(leak_examples) < args.max_leak_examples:
                    leak_examples.append(
                        {
                            "row_id": row_id,
                            "query_id": query_id,
                            "negative_doc_id": negative_doc_id,
                            "train_positive": is_train_positive,
                            "dev_positive": is_dev_positive,
                        }
                    )
            if is_dev_positive:
                negative_dev_same_query += 1
            if negative_doc_id in dev_positive_docs:
                negative_any_dev_positive_docs.add(negative_doc_id)
        candidate_rows.append(row)
        candidates = [(positive_doc_id, doc_texts[positive_doc_id], "positive")]
        candidates.extend((doc_id, doc_texts[doc_id], "negative") for doc_id in negative_doc_ids)
        for candidate_index, (doc_id, text, role) in enumerate(candidates):
            request_rows.append(
                {
                    "source": "msmarco-passage",
                    "split": "train",
                    "row_id": row_id,
                    "query_id": query_id,
                    "candidate_doc_id": doc_id,
                    "query": query_texts[query_id],
                    "candidate": text,
                    "role": role,
                    "example_index": example_index,
                    "candidate_index": candidate_index,
                    "release_train_allowed": False,
                    "commercial_use_allowed": False,
                }
            )

    artifacts_dir = run_root / "artifacts"
    reports_dir = run_root / "reports"
    candidates_path = artifacts_dir / "msmarco-passage.teacher-candidates.train-hard-negatives.jsonl"
    requests_path = artifacts_dir / "msmarco-passage.teacher-score-requests.jsonl"
    audit_path = reports_dir / "leak-split-audit.md"
    manifest_path = run_root / "teacher-prep-manifest.json"
    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(requests_path, request_rows)

    train_dev_overlap = train_positive_docs & dev_positive_docs
    audit = {
        "candidate_negative_positive_leaks": negative_positive_leaks,
        "candidate_negative_dev_positive_same_query": negative_dev_same_query,
        "candidate_negative_any_dev_positive_doc_ids": len(negative_any_dev_positive_docs),
        "candidate_negative_any_dev_positive_doc_id_examples": sorted(negative_any_dev_positive_docs)[:50],
        "dev_qrels_used_for_candidate_construction": 0,
        "rows_missing_split_provenance": rows_missing_split,
        "rows_missing_row_id": rows_missing_row_id,
        "candidate_rows_not_train_split": sum(1 for row in candidate_rows if row["split"] != "train"),
        "global_train_dev_positive_doc_overlap": len(train_dev_overlap),
        "global_train_dev_positive_doc_overlap_examples": sorted(train_dev_overlap)[:50],
        "negative_leak_examples": leak_examples,
        "gate_status": "passed"
        if negative_positive_leaks == 0
        and negative_dev_same_query == 0
        and rows_missing_split == 0
        and rows_missing_row_id == 0
        else "failed",
    }
    provenance_counts = Counter(row["source"] for row in candidate_rows)
    manifest = {
        "schema": "eos.msmarco_teacher_prep_audit.v1",
        "created_utc": utc_now(),
        "run_root": str(run_root),
        "source_manifest": str(source_manifest_path),
        "dataset_root": str(dataset_root),
        "sample": {
            "split": "train",
            "sample_qrels": len(sampled_qrels),
            "sample_qrels_bound": args.sample_qrels,
            "sample_strategy": "first unique train qrel pairs from qrels/train.tsv; deterministic corpus-id hash negatives",
            "negatives_per_row": args.negatives_per_row,
            "dev_qrels_used_for_candidate_construction": False,
        },
        "counts": {
            "source_train_qrels_rows": train_qrel_rows,
            "source_dev_qrels_rows": dev_qrel_rows,
            "candidate_rows": len(candidate_rows),
            "teacher_request_rows": len(request_rows),
            "positive_request_rows": len(candidate_rows),
            "negative_request_rows": len(candidate_rows) * args.negatives_per_row,
            "unique_candidate_queries": len({row["query_id"] for row in candidate_rows}),
            "unique_positive_doc_ids": len({row["positive_doc_id"] for row in candidate_rows}),
            "unique_negative_doc_ids": len({doc_id for row in candidate_rows for doc_id in row["negative_doc_ids"]}),
            "candidate_sources": dict(provenance_counts),
        },
        "legal_gate": {
            "train_allowed_for_research": True,
            "release_train_allowed": False,
            "commercial_use_allowed": False,
            "requires_independent_legal_review_for_products_or_services": True,
            "policy_basis": "inherited from MS MARCO acquisition manifest; teacher prep only, no release rows",
        },
        "split_and_leak_audit": audit,
        "schema_compatibility": {
            "candidate_jsonl": "Eos EmbeddingTextHardNegativeExample-compatible fields plus extra provenance fields",
            "teacher_score_requests_jsonl": "Eos export-teacher-score-requests-style fields plus doc-id/row-id provenance",
            "teacher_scores_present": False,
            "training_rows": False,
        },
        "artifacts": {
            "candidate_jsonl": {
                "path": str(candidates_path),
                "rows": len(candidate_rows),
                "sha256": sha256_file(candidates_path),
            },
            "teacher_score_requests_jsonl": {
                "path": str(requests_path),
                "rows": len(request_rows),
                "sha256": sha256_file(requests_path),
            },
            "manifest_json": {
                "path": str(manifest_path),
            },
            "leak_split_audit_md": {
                "path": str(audit_path),
            },
        },
        "source_sha256s": {
            "source_manifest": sha256_file(source_manifest_path),
            "train_qrels": sha256_file(train_qrels_path),
            "dev_qrels": sha256_file(dev_qrels_path),
            "queries": sha256_file(queries_path),
            "corpus": source_manifest["counts"]["beir"]["corpus"]["sha256"],
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(build_report(manifest, audit), encoding="utf-8")
    manifest["artifacts"]["leak_split_audit_md"]["sha256"] = sha256_file(audit_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "built msmarco teacher prep audit: "
        f"candidates={len(candidate_rows)} requests={len(request_rows)} gate={audit['gate_status']}"
    )
    print(f"manifest: {manifest_path}")
    print(f"audit: {audit_path}")
    return 0 if audit["gate_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
