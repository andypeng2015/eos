#!/usr/bin/env python3
"""Acquire MS MARCO passage corpus and audit qrel/query/corpus resolvability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_PAGE = "https://microsoft.github.io/msmarco/Datasets.html"
TERMS_PAGE = "https://microsoft.github.io/msmarco/"
COLLECTION_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
QUERIES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
QRELS_TRAIN_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv"
QRELS_DEV_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"
DEFAULT_MIN_FREE_BYTES = 15 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def head_url(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "ok": True,
                "status": resp.status,
                "content_length": int(resp.headers["Content-Length"])
                if resp.headers.get("Content-Length")
                else None,
                "content_type": resp.headers.get("Content-Type"),
                "last_modified": resp.headers.get("Last-Modified"),
                "etag": resp.headers.get("ETag"),
            }
    except Exception as exc:  # noqa: BLE001 - audit detail
        return {"ok": False, "error": repr(exc)}


def download(url: str, path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(path)


def safe_extract_tar_gz(path: Path, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    root = dst.resolve()
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            target = (dst / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"tar member escapes destination: {member.name}")
            tf.extract(member, dst, filter="data")
            extracted.append(target)
    return extracted


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_probe_inputs(previous_run: Path, run_root: Path) -> dict[str, Path]:
    raw = run_root / "raw"
    raw_queries = raw / "queries"
    prev_raw = previous_run / "raw"
    prev_queries = prev_raw / "queries"
    paths = {
        "queries_archive": raw / "queries.tar.gz",
        "queries_train": raw_queries / "queries.train.tsv",
        "queries_dev": raw_queries / "queries.dev.tsv",
        "queries_eval": raw_queries / "queries.eval.tsv",
        "qrels_train": raw / "qrels.train.tsv",
        "qrels_dev": raw / "qrels.dev.tsv",
    }
    source_paths = {
        "queries_archive": prev_raw / "queries.tar.gz",
        "queries_train": prev_queries / "queries.train.tsv",
        "queries_dev": prev_queries / "queries.dev.tsv",
        "queries_eval": prev_queries / "queries.eval.tsv",
        "qrels_train": prev_raw / "qrels.train.tsv",
        "qrels_dev": prev_raw / "qrels.dev.tsv",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("previous probe inputs missing: " + ", ".join(missing))
    for key, src in source_paths.items():
        link_or_copy(src, paths[key])
    return paths


def read_queries(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    queries: dict[str, str] = {}
    rows = 0
    malformed = 0
    duplicate_ids = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2 or not row[0]:
                malformed += 1
                continue
            rows += 1
            if row[0] in queries:
                duplicate_ids += 1
            queries[row[0]] = row[1]
    return queries, {
        "path": str(path),
        "rows": rows,
        "unique_queries": len(queries),
        "malformed_rows": malformed,
        "duplicate_ids": duplicate_ids,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_qrels(path: Path) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    rows: list[tuple[str, str, str]] = []
    qids: set[str] = set()
    pids: set[str] = set()
    malformed = 0
    rel_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 4 or not row[0] or not row[2]:
                malformed += 1
                continue
            qid, pid, rel = row[0], row[2], row[3]
            rows.append((qid, pid, rel))
            qids.add(qid)
            pids.add(pid)
            rel_counter[rel] += 1
    return rows, {
        "path": str(path),
        "rows": len(rows),
        "unique_queries": len(qids),
        "unique_doc_ids": len(pids),
        "malformed_rows": malformed,
        "relevance_counts": dict(sorted(rel_counter.items())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "_query_ids": qids,
        "_doc_ids": pids,
    }


def write_beir_queries(path: Path, split_queries: dict[str, dict[str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows = 0
    duplicate_across_splits = 0
    with path.open("w", encoding="utf-8") as out:
        for split in ("train", "dev"):
            for qid, text in split_queries[split].items():
                if qid in seen:
                    duplicate_across_splits += 1
                    continue
                seen.add(qid)
                out.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
                rows += 1
    return {
        "path": str(path),
        "rows": rows,
        "unique_queries": len(seen),
        "duplicate_ids_across_splits": duplicate_across_splits,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_beir_qrels(path: Path, rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid, pid, rel in rows:
            writer.writerow([qid, pid, rel])
    return {
        "path": str(path),
        "rows": len(rows),
        "includes_header": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def stream_corpus(
    collection_path: Path,
    needed_train: set[str],
    needed_dev: set[str],
    beir_corpus_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    needed_all = needed_train | needed_dev
    found_all: set[str] = set()
    found_train: set[str] = set()
    found_dev: set[str] = set()
    seen: set[str] = set()
    rows = 0
    malformed = 0
    duplicate_ids = 0
    examples_bad: list[list[str]] = []
    beir_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with collection_path.open("r", encoding="utf-8", errors="replace", newline="") as src, beir_corpus_path.open(
        "w", encoding="utf-8"
    ) as out:
        reader = csv.reader(src, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2 or not row[0]:
                malformed += 1
                if len(examples_bad) < 20:
                    examples_bad.append(row)
                continue
            pid, passage = row[0], row[1]
            rows += 1
            if pid in seen:
                duplicate_ids += 1
            seen.add(pid)
            if pid in needed_all:
                found_all.add(pid)
            if pid in needed_train:
                found_train.add(pid)
            if pid in needed_dev:
                found_dev.add(pid)
            out.write(json.dumps({"_id": pid, "title": "", "text": passage}, ensure_ascii=False) + "\n")
    corpus_counts = {
        "path": str(collection_path),
        "rows": rows,
        "unique_doc_ids": len(seen),
        "malformed_rows": malformed,
        "duplicate_ids": duplicate_ids,
        "bad_row_examples": examples_bad,
        "bytes": collection_path.stat().st_size,
        "sha256": sha256_file(collection_path),
    }
    beir_counts = {
        "path": str(beir_corpus_path),
        "rows": rows,
        "bytes": beir_corpus_path.stat().st_size,
        "sha256": sha256_file(beir_corpus_path),
    }
    resolvability = {
        "needed_unique_doc_ids": len(needed_all),
        "resolved_unique_doc_ids": len(found_all),
        "missing_unique_doc_ids": len(needed_all - found_all),
        "missing_doc_id_examples": sorted(needed_all - found_all)[:50],
        "train": {
            "needed_unique_doc_ids": len(needed_train),
            "resolved_unique_doc_ids": len(found_train),
            "missing_unique_doc_ids": len(needed_train - found_train),
            "missing_doc_id_examples": sorted(needed_train - found_train)[:50],
        },
        "dev": {
            "needed_unique_doc_ids": len(needed_dev),
            "resolved_unique_doc_ids": len(found_dev),
            "missing_unique_doc_ids": len(needed_dev - found_dev),
            "missing_doc_id_examples": sorted(needed_dev - found_dev)[:50],
        },
    }
    return {**corpus_counts, "qrel_doc_resolvability": resolvability}, beir_counts


def clean_counts(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if not k.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--previous-run",
        default=Path("runs/eos-msmarco-data-acquisition-v1-20260624T165140Z"),
        type=Path,
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    previous_run = args.previous_run.resolve()
    raw = run_root / "raw"
    extracted = run_root / "extracted"
    reports = run_root / "reports"
    beir = run_root / "beir-msmarco-passage-research-only"
    for path in (raw, extracted, reports, beir):
        path.mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    disk_before = shutil.disk_usage(run_root.parent)
    collection_head = head_url(COLLECTION_URL, args.timeout)
    if not collection_head.get("ok"):
        raise RuntimeError(f"collection HEAD failed: {collection_head}")
    content_length = collection_head.get("content_length") or 0
    required_floor = args.min_free_bytes + content_length
    if disk_before.free < required_floor:
        raise RuntimeError(
            f"unsafe free space: free={disk_before.free} required_at_least={required_floor} "
            f"(min_free_after={args.min_free_bytes}, collection_bytes={content_length})"
        )

    probe_paths = copy_probe_inputs(previous_run, run_root)
    collection_archive = raw / "collection.tar.gz"
    if not collection_archive.exists():
        download(COLLECTION_URL, collection_archive, args.timeout)
    collection_sha = sha256_file(collection_archive)
    extracted_files = safe_extract_tar_gz(collection_archive, extracted / "collection")
    collection_tsv_matches = [p for p in extracted_files if p.name == "collection.tsv"]
    if not collection_tsv_matches:
        raise FileNotFoundError("collection.tsv not found in extracted collection archive")
    collection_tsv = collection_tsv_matches[0]

    train_queries, train_query_counts = read_queries(probe_paths["queries_train"])
    dev_queries, dev_query_counts = read_queries(probe_paths["queries_dev"])
    eval_queries, eval_query_counts = read_queries(probe_paths["queries_eval"])
    train_qrels, train_qrel_counts = read_qrels(probe_paths["qrels_train"])
    dev_qrels, dev_qrel_counts = read_qrels(probe_paths["qrels_dev"])

    train_missing_queries = sorted(train_qrel_counts["_query_ids"] - set(train_queries))
    dev_missing_queries = sorted(dev_qrel_counts["_query_ids"] - set(dev_queries))
    train_dev_query_overlap = train_qrel_counts["_query_ids"] & dev_qrel_counts["_query_ids"]
    train_dev_doc_overlap = train_qrel_counts["_doc_ids"] & dev_qrel_counts["_doc_ids"]

    beir_query_counts = write_beir_queries(beir / "queries.jsonl", {"train": train_queries, "dev": dev_queries})
    beir_train_qrels = write_beir_qrels(beir / "qrels" / "train.tsv", train_qrels)
    beir_dev_qrels = write_beir_qrels(beir / "qrels" / "dev.tsv", dev_qrels)
    corpus_counts, beir_corpus_counts = stream_corpus(
        collection_tsv,
        train_qrel_counts["_doc_ids"],
        dev_qrel_counts["_doc_ids"],
        beir / "corpus.jsonl",
    )

    local_files = [
        {
            "path": str(collection_archive),
            "bytes": collection_archive.stat().st_size,
            "sha256": collection_sha,
            "source_url": COLLECTION_URL,
            "role": "downloaded_corpus_archive",
        },
        {
            "path": str(collection_tsv),
            "bytes": collection_tsv.stat().st_size,
            "sha256": corpus_counts["sha256"],
            "source_url": COLLECTION_URL,
            "extracted_from": str(collection_archive),
            "role": "extracted_corpus_tsv",
        },
    ]
    for key, source_url in (
        ("queries_archive", QUERIES_URL),
        ("queries_train", QUERIES_URL),
        ("queries_dev", QUERIES_URL),
        ("queries_eval", QUERIES_URL),
        ("qrels_train", QRELS_TRAIN_URL),
        ("qrels_dev", QRELS_DEV_URL),
    ):
        path = probe_paths[key]
        local_files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": source_url,
                "role": "verified_probe_input_reused",
                "copied_or_hardlinked_from": str(previous_run),
            }
        )
    for counts, role in (
        (beir_corpus_counts, "beir_corpus_jsonl"),
        (beir_query_counts, "beir_queries_jsonl"),
        (beir_train_qrels, "beir_train_qrels_tsv"),
        (beir_dev_qrels, "beir_dev_qrels_tsv"),
    ):
        local_files.append(
            {
                "path": counts["path"],
                "bytes": counts["bytes"],
                "sha256": counts["sha256"],
                "source_url": "derived_from_verified_ms_marco_inputs",
                "role": role,
            }
        )

    split_safety = {
        "train_dev_query_overlap": len(train_dev_query_overlap),
        "train_dev_query_overlap_examples": sorted(train_dev_query_overlap)[:50],
        "train_dev_positive_doc_overlap": len(train_dev_doc_overlap),
        "train_dev_positive_doc_overlap_examples": sorted(train_dev_doc_overlap)[:50],
        "train_qrels_queries_missing_from_train_queries": len(train_missing_queries),
        "dev_qrels_queries_missing_from_dev_queries": len(dev_missing_queries),
        "missing_query_examples": {
            "train": train_missing_queries[:50],
            "dev": dev_missing_queries[:50],
        },
        "corpus_resolvability": corpus_counts["qrel_doc_resolvability"],
        "test_boundary": {
            "status": "safe_for_acquisition_audit",
            "details": "No test labels, test top1000, triples, or unrelated candidate files were downloaded. Eval queries from queries.tar.gz were retained only as upstream archive content and are not emitted into BEIR queries.jsonl.",
        },
    }

    disk_after = shutil.disk_usage(run_root.parent)
    manifest = {
        "schema": "eos.msmarco_passage_corpus_acquisition_manifest.v1",
        "dataset": {
            "name": "MS MARCO Passage Ranking",
            "variant": "passage",
            "official_dataset_page": DATASET_PAGE,
            "official_terms_page": TERMS_PAGE,
            "retrieved_at_utc": started,
        },
        "source_urls": {
            "terms": TERMS_PAGE,
            "dataset_page": DATASET_PAGE,
            "collection": COLLECTION_URL,
            "queries": QUERIES_URL,
            "qrels_train": QRELS_TRAIN_URL,
            "qrels_dev": QRELS_DEV_URL,
        },
        "license_terms_summary": {
            "source_urls": [TERMS_PAGE, DATASET_PAGE],
            "summary": (
                "Official MS MARCO pages state that MS MARCO and ORCAS datasets are intended "
                "for non-commercial research purposes only, provided free of charge without "
                "extending license or other intellectual property rights, and provided as-is "
                "with risk because Microsoft may not own underlying document rights."
            ),
            "engineering_policy": {
                "train_allowed_for_research": True,
                "release_train_allowed": False,
                "commercial_use_allowed": False,
                "requires_independent_legal_review_for_products_or_services": True,
                "policy_basis": "official non-commercial research terms; no stronger release-compatible license found in this run",
            },
        },
        "disk_preflight": {
            "path": str(run_root.parent),
            "free_bytes_before": disk_before.free,
            "min_free_bytes_after_collection_download_floor": args.min_free_bytes,
            "collection_head_content_length": content_length,
            "required_free_bytes_before": required_floor,
            "free_bytes_after": disk_after.free,
            "total_bytes": disk_before.total,
        },
        "artifacts": {
            "collection": {
                "url": COLLECTION_URL,
                "head": collection_head,
                "archive_path": str(collection_archive),
                "archive_bytes": collection_archive.stat().st_size,
                "archive_sha256": collection_sha,
                "extracted_path": str(collection_tsv),
                "extracted_bytes": collection_tsv.stat().st_size,
                "extracted_sha256": corpus_counts["sha256"],
            },
            "queries_and_qrels": {
                "source": "reused from previous verified probe run",
                "previous_run": str(previous_run),
                "queries_archive": str(probe_paths["queries_archive"]),
                "qrels_train": str(probe_paths["qrels_train"]),
                "qrels_dev": str(probe_paths["qrels_dev"]),
            },
        },
        "local_files": local_files,
        "counts": {
            "corpus": clean_counts(corpus_counts),
            "queries": {
                "train": train_query_counts,
                "dev": dev_query_counts,
                "eval_retained_from_archive_not_emitted_to_beir": eval_query_counts,
            },
            "qrels": {
                "train": clean_counts(train_qrel_counts),
                "dev": clean_counts(dev_qrel_counts),
            },
            "beir": {
                "root": str(beir),
                "research_only_provenance_gated": True,
                "corpus": beir_corpus_counts,
                "queries": beir_query_counts,
                "qrels_train": beir_train_qrels,
                "qrels_dev": beir_dev_qrels,
            },
        },
        "split_safety": split_safety,
        "train_allowed_policy": {
            "row_family": "msmarco_passage_qrel_positive_pairs",
            "train_rows_allowed_for_research": "train split only, after this manifest's full qrel/query/corpus resolvability check",
            "dev_rows_allowed": "dev sanity/evaluation only, not training or teacher filtering",
            "test_rows_allowed": "never for training or selection",
            "release_default_training_allowed": False,
            "commercial_use_allowed": False,
            "reason": "official non-commercial research terms; this run is acquisition/audit only",
        },
        "skipped_or_missing": {
            "hard_negative_triples": "not downloaded by design",
            "top1000_reranking_rows": "not downloaded by design",
            "release_training_rows": "not emitted; release_train_allowed=false",
            "teacher_vectors_or_scores": "not generated",
            "model_training": "not run",
        },
    }

    write_json(run_root / "acquisition-manifest.json", manifest)
    write_json(reports / "corpus-resolvability.json", split_safety)
    sha_lines = [
        f"{entry['sha256']}  {Path(entry['path']).relative_to(run_root)}"
        for entry in local_files
        if entry.get("sha256")
    ]
    (run_root / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    md = [
        "# MS MARCO Passage Corpus Resolvability Audit",
        "",
        f"- Run root: `{run_root}`",
        f"- Corpus rows: `{corpus_counts['rows']}` rows, `{corpus_counts['unique_doc_ids']}` unique passage IDs",
        f"- Train qrels: `{train_qrel_counts['rows']}` rows, `{train_qrel_counts['unique_queries']}` unique queries, `{train_qrel_counts['unique_doc_ids']}` unique positive passage IDs",
        f"- Dev qrels: `{dev_qrel_counts['rows']}` rows, `{dev_qrel_counts['unique_queries']}` unique queries, `{dev_qrel_counts['unique_doc_ids']}` unique positive passage IDs",
        f"- Train qrels missing query text: `{split_safety['train_qrels_queries_missing_from_train_queries']}`",
        f"- Dev qrels missing query text: `{split_safety['dev_qrels_queries_missing_from_dev_queries']}`",
        f"- Train qrel doc IDs missing corpus text: `{split_safety['corpus_resolvability']['train']['missing_unique_doc_ids']}`",
        f"- Dev qrel doc IDs missing corpus text: `{split_safety['corpus_resolvability']['dev']['missing_unique_doc_ids']}`",
        f"- Train/dev query overlap: `{split_safety['train_dev_query_overlap']}`",
        f"- Train/dev positive-passage overlap: `{split_safety['train_dev_positive_doc_overlap']}`",
        f"- Bad corpus rows: `{corpus_counts['malformed_rows']}`; duplicate corpus IDs: `{corpus_counts['duplicate_ids']}`",
        f"- BEIR-style root: `{beir}`",
        "- Gate: `release_train_allowed=false`; `commercial_use_allowed=false`; research-only/provenance-gated.",
        "- Skipped by design: triples, top1000 files, teacher scoring, rerankers, model training, release training rows.",
    ]
    (reports / "corpus-resolvability.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"manifest": str(run_root / "acquisition-manifest.json"), "report": str(reports / "corpus-resolvability.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
