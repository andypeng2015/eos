#!/usr/bin/env python3
"""Build bounded research-only MS MARCO Passage Stage A hard-negative rows.

The builder consumes the BEIR-style MS MARCO Passage corpus produced by
``scripts/acquire_msmarco_passage.py`` or the earlier acquisition packet. It
emits text hard-negative JSONL compatible with
``runtime/embedding_hard_negative_dataset.go`` while preserving source,
provenance, split, legal, and leak-accounting metadata in extra fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "eos.msmarco_passage_stagea_rows.v1"
ROW_SCHEMA = "eos.embedding_text_hard_negative.research_stagea.v1"
DEFAULT_ACQUISITION_RUN = Path("runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z")
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "test_rows_train_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--acquisition-run-root", type=Path, default=DEFAULT_ACQUISITION_RUN)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--negatives-per-query", "--negatives-per-row", dest="negatives_per_query", type=int, default=3)
    parser.add_argument("--candidate-pool-size", type=int, default=20000)
    parser.add_argument("--max-corpus-docs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--drop-sample-limit", type=int, default=50)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def normalize_for_leak(value: Any) -> str:
    return stable_text(value).casefold()


def corpus_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return stable_text(f"{title}\n{text}")
    return stable_text(title or text)


def query_text(row: dict[str, Any]) -> str:
    return stable_text(row.get("text") or row.get("query") or "")


def row_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError(f"{path}:{line_number}: missing _id/id")
    return str(value)


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


def read_sha256s(acquisition_run_root: Path) -> dict[str, str]:
    path = acquisition_run_root / "SHA256SUMS"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "  " not in raw:
            continue
        digest, rel = raw.split("  ", 1)
        out[rel] = digest
    return out


def load_acquisition_manifest(acquisition_run_root: Path) -> dict[str, Any]:
    path = acquisition_run_root / "acquisition-manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, row in iter_jsonl(path):
        qid = row_id(row, path, line_number)
        text = query_text(row)
        if text:
            queries[qid] = text
    return queries


def load_qrels(path: Path) -> tuple[list[tuple[str, str]], dict[str, set[str]], Counter[str]]:
    pairs: list[tuple[str, str]] = []
    positives_by_query: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        has_header = sample.lower().startswith("query-id\t") or sample.lower().startswith("query_id\t")
        if has_header:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                qid = str(row.get("query-id") or row.get("query_id") or "").strip()
                doc_id = str(row.get("corpus-id") or row.get("corpus_id") or "").strip()
                try:
                    score = float(row.get("score") or 0)
                except (TypeError, ValueError):
                    score = 0.0
                if not qid or not doc_id or score <= 0:
                    counts["ignored_nonpositive_or_malformed"] += 1
                    continue
                pairs.append((qid, doc_id))
                positives_by_query[qid].add(doc_id)
        else:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    counts["ignored_malformed"] += 1
                    continue
                qid, doc_id = row[0].strip(), row[2].strip()
                try:
                    score = float(row[3]) if len(row) > 3 else 1.0
                except ValueError:
                    score = 0.0
                if not qid or not doc_id or score <= 0:
                    counts["ignored_nonpositive_or_malformed"] += 1
                    continue
                pairs.append((qid, doc_id))
                positives_by_query[qid].add(doc_id)
    counts["positive_rows"] = len(pairs)
    counts["unique_queries"] = len(positives_by_query)
    counts["unique_positive_docs"] = len({doc_id for _, doc_id in pairs})
    return pairs, positives_by_query, counts


def select_train_pairs(
    train_pairs: list[tuple[str, str]],
    queries: dict[str, str],
    max_rows: int,
) -> tuple[list[tuple[str, str]], Counter[str]]:
    selected: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for qid, doc_id in train_pairs:
        if len(selected) >= max_rows:
            break
        if (qid, doc_id) in seen:
            counts["duplicate_qrel_pair_skipped"] += 1
            continue
        if qid not in queries:
            counts["missing_query_text"] += 1
            continue
        selected.append((qid, doc_id))
        seen.add((qid, doc_id))
    return selected, counts


def load_needed_corpus_and_negative_pool(
    corpus_path: Path,
    needed_doc_ids: set[str],
    positives_by_query: dict[str, set[str]],
    dev_positive_doc_ids: set[str],
    candidate_pool_size: int,
    max_corpus_docs: int,
    rng: random.Random,
) -> tuple[dict[str, str], list[tuple[str, str]], Counter[str]]:
    needed_texts: dict[str, str] = {}
    pool: list[tuple[str, str]] = []
    train_positive_doc_ids = {doc_id for doc_ids in positives_by_query.values() for doc_id in doc_ids}
    counts: Counter[str] = Counter()
    for line_number, row in iter_jsonl(corpus_path):
        counts["corpus_jsonl_rows_seen"] += 1
        if max_corpus_docs > 0 and counts["corpus_docs_scanned"] >= max_corpus_docs:
            counts["corpus_scan_cap_reached"] = 1
            break
        doc_id = row_id(row, corpus_path, line_number)
        text = corpus_text(row)
        if not text:
            counts["empty_corpus_text"] += 1
            continue
        counts["corpus_docs_scanned"] += 1
        if doc_id in needed_doc_ids:
            needed_texts[doc_id] = text
        if doc_id in dev_positive_doc_ids:
            counts["dev_positive_excluded_from_negative_pool"] += 1
            continue
        if doc_id in train_positive_doc_ids:
            counts["train_positive_excluded_from_negative_pool"] += 1
            continue
        counts["negative_pool_candidates_seen"] += 1
        item = (doc_id, text)
        if len(pool) < candidate_pool_size:
            pool.append(item)
        else:
            index = rng.randrange(counts["negative_pool_candidates_seen"])
            if index < candidate_pool_size:
                pool[index] = item
    counts["needed_doc_ids"] = len(needed_doc_ids)
    counts["needed_doc_ids_resolved"] = len(needed_texts)
    counts["negative_pool_size"] = len(pool)
    return needed_texts, pool, counts


def make_row_id(split: str, query_id: str, positive_doc_id: str, negative_doc_ids: list[str]) -> str:
    payload = json.dumps(
        {
            "dataset": "msmarco-passage",
            "split": split,
            "query_id": query_id,
            "positive_doc_id": positive_doc_id,
            "negative_doc_ids": negative_doc_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_drop_sample(samples: list[dict[str, Any]], limit: int, sample: dict[str, Any]) -> None:
    if len(samples) < limit:
        samples.append(sample)


def build_rows(
    selected_pairs: list[tuple[str, str]],
    queries: dict[str, str],
    positive_texts: dict[str, str],
    negative_pool: list[tuple[str, str]],
    positives_by_query: dict[str, set[str]],
    dev_positive_doc_ids: set[str],
    split: str,
    negatives_per_row: int,
    rng: random.Random,
    drop_sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    drop_samples: list[dict[str, Any]] = []
    shuffled_pool = list(negative_pool)
    rng.shuffle(shuffled_pool)
    cursor = 0
    for qid, positive_doc_id in selected_pairs:
        query = queries.get(qid, "")
        positive = positive_texts.get(positive_doc_id, "")
        if not query:
            counts["missing_query_text"] += 1
            add_drop_sample(drop_samples, drop_sample_limit, {"query_id": qid, "reason": "missing_query_text"})
            continue
        if not positive:
            counts["missing_positive_text"] += 1
            add_drop_sample(
                drop_samples,
                drop_sample_limit,
                {"query_id": qid, "positive_doc_id": positive_doc_id, "reason": "missing_positive_text"},
            )
            continue
        positive_norm = normalize_for_leak(positive)
        negative_doc_ids: list[str] = []
        negatives: list[str] = []
        attempts = 0
        max_attempts = max(len(shuffled_pool) * 2, negatives_per_row * 10)
        while len(negatives) < negatives_per_row and attempts < max_attempts and shuffled_pool:
            attempts += 1
            candidate_doc_id, candidate_text = shuffled_pool[cursor % len(shuffled_pool)]
            cursor += 1
            if candidate_doc_id in positives_by_query.get(qid, set()):
                counts["same_query_positive_negative_drops"] += 1
                continue
            if candidate_doc_id in dev_positive_doc_ids:
                counts["dev_positive_negative_drops"] += 1
                continue
            if candidate_doc_id in negative_doc_ids:
                counts["duplicate_negative_doc_id_drops"] += 1
                continue
            if normalize_for_leak(candidate_text) == positive_norm:
                counts["duplicate_positive_text_negative_drops"] += 1
                continue
            negatives.append(candidate_text)
            negative_doc_ids.append(candidate_doc_id)
        if len(negatives) < negatives_per_row:
            counts["insufficient_negatives"] += 1
            add_drop_sample(
                drop_samples,
                drop_sample_limit,
                {
                    "query_id": qid,
                    "positive_doc_id": positive_doc_id,
                    "reason": "insufficient_negatives",
                    "negatives_found": len(negatives),
                },
            )
            continue
        row = {
            "schema": ROW_SCHEMA,
            "row_id": make_row_id(split, qid, positive_doc_id, negative_doc_ids),
            "source": f"msmarco-passage/{split}/qrels/random-corpus-hard-negatives",
            "dataset": "msmarco-passage",
            "split": split,
            "query_id": qid,
            "positive_doc_id": positive_doc_id,
            "negative_doc_ids": negative_doc_ids,
            "query": query,
            "positive": positive,
            "negatives": negatives,
            "roles": {
                "query": "query",
                "positive": "document",
                "negatives": "document",
            },
            "legal_gates": dict(LEGAL_GATES),
            "split_policy": {
                "training_selection_split": split,
                "dev_qrels_used_for": "negative_leak_filtering_and_overlap_accounting_only",
                "test_or_eval_rows_used": False,
                "test_rows_train_allowed": False,
            },
            "provenance": {
                "corpus_format": "BEIR-jsonl",
                "qrels_source": f"qrels/{split}.tsv",
                "negative_source": "corpus.jsonl excluding train/dev qrel positives",
            },
        }
        rows.append(row)
    counts["rows_emitted"] = len(rows)
    return rows, {"counts": dict(counts), "drop_samples": drop_samples}


def validate_rows(rows: list[dict[str, Any]], positives_by_query: dict[str, set[str]], dev_positive_doc_ids: set[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not stable_text(row.get("query")):
            counts["missing_query_text"] += 1
        if not stable_text(row.get("positive")):
            counts["missing_positive_text"] += 1
        positive_norm = normalize_for_leak(row.get("positive"))
        qid = str(row.get("query_id") or "")
        for negative_doc_id, negative in zip(row.get("negative_doc_ids") or [], row.get("negatives") or []):
            if not stable_text(negative):
                counts["missing_negative_text"] += 1
            if negative_doc_id in positives_by_query.get(qid, set()):
                counts["same_query_positive_as_negative"] += 1
            if normalize_for_leak(negative) == positive_norm:
                counts["duplicate_positive_text_as_negative"] += 1
            if negative_doc_id in dev_positive_doc_ids:
                counts["dev_positive_as_negative"] += 1
        if len(row.get("negative_doc_ids") or []) != len(row.get("negatives") or []):
            counts["negative_id_text_length_mismatch"] += 1
        if not row.get("roles"):
            counts["missing_roles"] += 1
        legal = row.get("legal_gates") or {}
        if legal != LEGAL_GATES:
            counts["legal_gate_mismatch"] += 1
        if not row.get("row_id"):
            counts["missing_row_id"] += 1
        counts["rows_checked"] = index + 1
    return {"counts": dict(counts), "status": "passed" if not any(v for k, v in counts.items() if k != "rows_checked") else "failed"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0:
        raise SystemExit("--max-rows must be positive")
    if args.negatives_per_query <= 0:
        raise SystemExit("--negatives-per-query must be positive")
    if args.candidate_pool_size < args.negatives_per_query:
        raise SystemExit("--candidate-pool-size must be >= --negatives-per-query")
    if args.max_corpus_docs < 0:
        raise SystemExit("--max-corpus-docs must be >= 0")
    split = args.split.strip()
    if not split or "/" in split or "\\" in split:
        raise SystemExit("--split must be a simple qrels split name")

    corpus_root = args.corpus_root
    if corpus_root is None:
        corpus_root = args.acquisition_run_root / "beir-msmarco-passage-research-only"
    output_root = args.output_root or (args.manifest.parent if args.manifest else Path("runs") / f"retrieval-stagea-msmarco-row-builder-v1-{utc_stamp()}")
    output_root.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_root / "corpus.jsonl"
    queries_path = corpus_root / "queries.jsonl"
    split_qrels_path = corpus_root / "qrels" / f"{split}.tsv"
    dev_qrels_path = corpus_root / "qrels" / "dev.tsv"
    for path in (corpus_path, queries_path, split_qrels_path):
        if not path.exists():
            raise SystemExit(f"required input missing: {path}")

    rng = random.Random(args.seed)
    queries = load_queries(queries_path)
    split_pairs, positives_by_query, split_qrel_counts = load_qrels(split_qrels_path)
    if dev_qrels_path.exists():
        dev_pairs, _, dev_qrel_counts = load_qrels(dev_qrels_path)
    else:
        dev_pairs, dev_qrel_counts = [], Counter()
    dev_positive_doc_ids = {doc_id for _, doc_id in dev_pairs}
    selected_pairs, selection_counts = select_train_pairs(split_pairs, queries, args.max_rows)
    selected_doc_ids = {doc_id for _, doc_id in selected_pairs}
    selected_query_ids = {qid for qid, _ in selected_pairs}

    positive_texts, negative_pool, corpus_counts = load_needed_corpus_and_negative_pool(
        corpus_path,
        selected_doc_ids,
        positives_by_query,
        dev_positive_doc_ids,
        args.candidate_pool_size,
        args.max_corpus_docs,
        rng,
    )
    rows, build_report = build_rows(
        selected_pairs,
        queries,
        positive_texts,
        negative_pool,
        positives_by_query,
        dev_positive_doc_ids,
        split,
        args.negatives_per_query,
        rng,
        args.drop_sample_limit,
    )
    validation = validate_rows(rows, positives_by_query, dev_positive_doc_ids)

    rows_path = output_root / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl"
    manifest_path = args.manifest or output_root / "manifest.json"
    leak_path = output_root / "reports" / "leak-report.json"
    sample_path = output_root / "reports" / "sample-rows.jsonl"
    write_jsonl(rows_path, rows)
    write_jsonl(sample_path, rows[: min(10, len(rows))])

    sha_map = read_sha256s(args.acquisition_run_root)
    acquisition_manifest = load_acquisition_manifest(args.acquisition_run_root)
    split_dev_positive_overlap = len({doc_id for _, doc_id in split_pairs} & dev_positive_doc_ids)
    source_hashes = {
        "corpus.jsonl": sha_map.get("beir-msmarco-passage-research-only/corpus.jsonl"),
        "queries.jsonl": sha_map.get("beir-msmarco-passage-research-only/queries.jsonl"),
        f"qrels/{split}.tsv": sha_map.get(f"beir-msmarco-passage-research-only/qrels/{split}.tsv"),
        "qrels/dev.tsv": sha_map.get("beir-msmarco-passage-research-only/qrels/dev.tsv"),
    }
    source_hashes = {key: value for key, value in source_hashes.items() if value}
    output_hashes = {
        "rows_jsonl_sha256": sha256_file(rows_path),
        "sample_rows_jsonl_sha256": sha256_file(sample_path),
    }
    leak_report = {
        "schema": f"{SCHEMA}.leak_report",
        "status": validation["status"],
        "validation": validation,
        "build_drops": build_report,
        "split_policy": {
            "training_selection_split": split,
            "dev_qrels_used_for": "negative_leak_filtering_and_overlap_accounting_only",
            "test_or_eval_rows_used": False,
            "test_rows_train_allowed": False,
        },
        "dev_overlap_accounting": {
            "split_dev_positive_doc_overlap": split_dev_positive_overlap,
            "dev_positive_doc_ids_excluded_from_negative_pool": corpus_counts.get(
                "dev_positive_excluded_from_negative_pool", 0
            ),
        },
    }
    write_json(leak_path, leak_report)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_msmarco_stagea_rows.py",
        "builder_args": {
            "corpus_root": str(corpus_root),
            "acquisition_run_root": str(args.acquisition_run_root),
            "split": split,
            "max_rows": args.max_rows,
            "negatives_per_query": args.negatives_per_query,
            "candidate_pool_size": args.candidate_pool_size,
            "max_corpus_docs": args.max_corpus_docs,
            "seed": args.seed,
        },
        "legal_gates": dict(LEGAL_GATES),
        "split_policy": {
            "training_selection_split": split,
            "dev_qrels_used_for": "negative_leak_filtering_and_overlap_accounting_only",
            "test_or_eval_rows_used": False,
            "test_rows_train_allowed": False,
            "release_train_allowed": False,
            "commercial_use_allowed": False,
        },
        "source": {
            "dataset": "MS MARCO Passage",
            "corpus_root": str(corpus_root),
            "acquisition_run_root": str(args.acquisition_run_root),
            "acquisition_manifest_schema": acquisition_manifest.get("schema"),
            "source_hashes": source_hashes,
        },
        "counts": {
            "queries_loaded": len(queries),
            "qrels_seen": dict(split_qrel_counts),
            "split_qrels": dict(split_qrel_counts),
            "dev_qrels": dict(dev_qrel_counts),
            "selected_train_pairs": len(selected_pairs),
            "selected_qrel_pairs": len(selected_pairs),
            "selected_unique_queries": len(selected_query_ids),
            "selected_positive_docs": len(selected_doc_ids),
            "rows_emitted": len(rows),
            "negatives_per_query": args.negatives_per_query,
            "negative_texts_emitted": sum(len(row.get("negatives") or []) for row in rows),
            "negative_candidates_considered": corpus_counts.get("negative_pool_candidates_seen", 0),
            "negative_candidates_emitted": sum(len(row.get("negatives") or []) for row in rows),
            "missing_query_skips": selection_counts.get("missing_query_text", 0) + build_report["counts"].get("missing_query_text", 0),
            "missing_doc_skips": build_report["counts"].get("missing_positive_text", 0),
            "source_mix": {f"msmarco-passage/{split}/qrels/random-corpus-hard-negatives": len(rows)},
            "selection": dict(selection_counts),
            "corpus_scan": dict(corpus_counts),
            "split_dev_positive_doc_overlap": split_dev_positive_overlap,
        },
        "artifacts": {
            "rows_jsonl": str(rows_path),
            "sample_rows_jsonl": str(sample_path),
            "leak_report": str(leak_path),
        },
        "output_hashes": output_hashes,
        "row_schema_fields": [
            "schema",
            "row_id",
            "source",
            "dataset",
            "split",
            "query_id",
            "positive_doc_id",
            "negative_doc_ids",
            "query",
            "positive",
            "negatives",
            "roles",
            "legal_gates",
            "split_policy",
            "provenance",
        ],
        "reader_compatibility": {
            "matches_runtime_embedding_text_hard_negative_dataset_go": True,
            "required_fields": ["query", "positive", "negatives", "source"],
            "extra_fields_preserved_by_reader": True,
        },
        "leak_report_summary": leak_report,
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "rows": str(rows_path), "leak_report": str(leak_path)}))


if __name__ == "__main__":
    main()
