#!/usr/bin/env python3
"""Bounded MS MARCO passage acquisition probe for Eos distillation planning."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_PAGE = "https://microsoft.github.io/msmarco/Datasets.html"
TERMS_PAGE = "https://microsoft.github.io/msmarco/"

ARTIFACTS = [
    {
        "key": "passage_collection",
        "split": "corpus",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        "official_size": "2.9 GB",
        "official_records": 8841823,
        "format": "tsv: pid, passage",
        "download_policy": "planned_not_downloaded_in_probe",
        "reason": "large corpus; needed for corpus/qrel text resolvability in a full acquisition",
    },
    {
        "key": "passage_queries",
        "split": "all_queries",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
        "official_size": "42.0 MB",
        "official_records": 1010916,
        "format": "tsv: qid, query",
        "download_policy": "download_probe",
    },
    {
        "key": "passage_qrels_dev",
        "split": "dev",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        "official_size": "1.1 MB",
        "official_records": 59273,
        "format": "TREC qrels format",
        "download_policy": "download_probe",
    },
    {
        "key": "passage_qrels_train",
        "split": "train",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
        "official_size": "10.1 MB",
        "official_records": 532761,
        "format": "TREC qrels format",
        "download_policy": "download_probe",
    },
    {
        "key": "passage_collection_and_queries",
        "split": "corpus_plus_queries_qrels",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
        "official_size": "2.9 GB",
        "official_records": 10406754,
        "format": "combined archive",
        "download_policy": "planned_not_downloaded_in_probe",
        "reason": "duplicates corpus-sized payload; full acquisition should choose this or separate corpus/query/qrel files, not both",
    },
    {
        "key": "passage_train_triples_small",
        "split": "train",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz",
        "official_size": "27.1 GB",
        "official_records": 39780811,
        "format": "tsv: query, positive passage, negative passage",
        "download_policy": "blocked_large_probe",
        "reason": "exceeds bounded probe budget; should be a separate explicit large-download descriptor",
    },
    {
        "key": "passage_train_triples_large",
        "split": "train",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.full.tsv.gz",
        "official_size": "272.2 GB",
        "official_records": 397756691,
        "format": "tsv: query, positive passage, negative passage",
        "download_policy": "blocked_large_probe",
        "reason": "far exceeds local safe budget",
    },
    {
        "key": "passage_train_qidpid_triples_full",
        "split": "train",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz",
        "official_size": "5.7 GB",
        "official_records": 397768673,
        "format": "tsv: qid, positive pid, negative pid",
        "download_policy": "planned_not_downloaded_in_probe",
        "reason": "large but feasible later; needs explicit choice after corpus/query/qrel acquisition",
    },
    {
        "key": "passage_top1000_train",
        "split": "train",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.train.tar.gz",
        "official_size": "175.0 GB",
        "official_records": 478002393,
        "format": "tsv: qid, pid, query, passage",
        "download_policy": "blocked_large_probe",
        "reason": "far exceeds local safe budget",
    },
    {
        "key": "passage_top1000_dev",
        "split": "dev",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
        "official_size": "2.5 GB",
        "official_records": 6668967,
        "format": "tsv: qid, pid, query, passage",
        "download_policy": "planned_not_downloaded_in_probe",
        "reason": "not needed for this qrel/query probe; useful for later dev reranking sanity",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    except Exception as exc:  # noqa: BLE001 - captured in audit manifest
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


def count_tsv_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        return sum(1 for line in f if line.rstrip("\n"))


def read_query_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2 and row[0]:
                ids.add(row[0])
    return ids


def read_qrels(path: Path) -> dict[str, Any]:
    qids: set[str] = set()
    pids: set[str] = set()
    rows = 0
    malformed = 0
    rel_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                malformed += 1
                continue
            rows += 1
            qids.add(row[0])
            pids.add(row[2])
            rel_counter[row[3]] += 1
    return {
        "rows": rows,
        "malformed_rows": malformed,
        "unique_queries": len(qids),
        "unique_doc_ids": len(pids),
        "query_ids": qids,
        "doc_ids": pids,
        "relevance_counts": dict(sorted(rel_counter.items())),
    }


def find_query_file(query_root: Path, name: str) -> Path | None:
    matches = sorted(query_root.rglob(name))
    return matches[0] if matches else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    raw = run_root / "raw"
    query_extract = raw / "queries"
    reports = run_root / "reports"
    raw.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    disk = shutil.disk_usage(run_root.parent if run_root.parent.exists() else Path("."))

    artifacts: list[dict[str, Any]] = []
    local_files: list[dict[str, Any]] = []
    for artifact in ARTIFACTS:
        item = dict(artifact)
        item["head"] = head_url(item["url"], args.timeout)
        if item["download_policy"] == "download_probe":
            target = raw / Path(item["url"]).name
            if not target.exists():
                download(item["url"], target, args.timeout)
            item["local_path"] = str(target)
            item["bytes"] = target.stat().st_size
            item["sha256"] = sha256_file(target)
            local_files.append(
                {
                    "path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": item["sha256"],
                    "source_url": item["url"],
                }
            )
            if target.name == "queries.tar.gz":
                extracted = safe_extract_tar_gz(target, query_extract)
                item["extracted_files"] = [str(p) for p in extracted]
                for extracted_path in extracted:
                    if extracted_path.is_file():
                        local_files.append(
                            {
                                "path": str(extracted_path),
                                "bytes": extracted_path.stat().st_size,
                                "sha256": sha256_file(extracted_path),
                                "source_url": item["url"],
                                "extracted_from": str(target),
                            }
                        )
        artifacts.append(item)

    qrels_train_path = raw / "qrels.train.tsv"
    qrels_dev_path = raw / "qrels.dev.tsv"
    train_qrels = read_qrels(qrels_train_path)
    dev_qrels = read_qrels(qrels_dev_path)

    query_files = {
        "train": find_query_file(query_extract, "queries.train.tsv"),
        "dev": find_query_file(query_extract, "queries.dev.tsv"),
        "eval": find_query_file(query_extract, "queries.eval.tsv"),
    }
    query_counts: dict[str, Any] = {}
    query_ids: dict[str, set[str]] = {}
    for split, path in query_files.items():
        if path is None:
            query_counts[split] = {"present": False}
            query_ids[split] = set()
            continue
        ids = read_query_ids(path)
        query_ids[split] = ids
        query_counts[split] = {
            "present": True,
            "path": str(path),
            "rows": count_tsv_rows(path),
            "unique_queries": len(ids),
            "sha256": sha256_file(path),
        }

    train_missing_queries = sorted(train_qrels["query_ids"] - query_ids["train"])
    dev_missing_queries = sorted(dev_qrels["query_ids"] - query_ids["dev"])
    split_safety = {
        "train_dev_query_overlap": len(train_qrels["query_ids"] & dev_qrels["query_ids"]),
        "train_dev_positive_doc_overlap": len(train_qrels["doc_ids"] & dev_qrels["doc_ids"]),
        "train_qrels_queries_missing_from_train_queries": len(train_missing_queries),
        "dev_qrels_queries_missing_from_dev_queries": len(dev_missing_queries),
        "missing_query_examples": {
            "train": train_missing_queries[:20],
            "dev": dev_missing_queries[:20],
        },
        "corpus_resolvability": {
            "status": "not_run",
            "reason": "passage collection was not downloaded in this bounded probe",
            "required_next_artifact": "passage_collection or passage_collection_and_queries",
        },
        "test_boundary": {
            "status": "safe_for_probe",
            "details": "No test labels were downloaded; leaderboard test qrels are not public. Test queries/top1000 are planned only and are not train inputs.",
        },
    }

    manifest = {
        "schema": "eos.msmarco_acquisition_manifest.v1",
        "dataset": {
            "name": "MS MARCO Passage Ranking",
            "variant": "passage",
            "official_dataset_page": DATASET_PAGE,
            "official_terms_page": TERMS_PAGE,
            "retrieved_at_utc": started,
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
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "artifacts": artifacts,
        "local_files": local_files,
        "counts": {
            "official_records_from_dataset_page": {
                a["key"]: a["official_records"] for a in ARTIFACTS
            },
            "probe": {
                "qrels_train": {k: v for k, v in train_qrels.items() if not k.endswith("_ids")},
                "qrels_dev": {k: v for k, v in dev_qrels.items() if not k.endswith("_ids")},
                "queries": query_counts,
            },
        },
        "splits": {
            "train": {
                "downloaded": ["qrels.train.tsv", "queries.train.tsv via queries.tar.gz"],
                "train_allowed_for_research": True,
                "release_train_allowed": False,
            },
            "dev": {
                "downloaded": ["qrels.dev.tsv", "queries.dev.tsv via queries.tar.gz"],
                "train_allowed_for_research": False,
                "selection_or_sanity_only": True,
                "release_train_allowed": False,
            },
            "test": {
                "downloaded": [],
                "train_allowed_for_research": False,
                "release_train_allowed": False,
            },
        },
        "train_allowed_policy": {
            "row_family": "msmarco_passage_qrel_positive_pairs",
            "train_rows_allowed_for_research": "train split only after corpus text is acquired and qrel/corpus/query resolvability is complete",
            "dev_rows_allowed": "dev sanity/evaluation only, not training or teacher filtering",
            "test_rows_allowed": "never for training or selection",
            "release_default_training_allowed": False,
            "reason": "official non-commercial research terms and incomplete corpus resolvability in this probe",
        },
        "split_safety": split_safety,
        "skipped_or_missing": {
            "corpus_text_rows": "not downloaded",
            "hard_negative_triples": "not downloaded",
            "top1000_reranking_rows": "not downloaded",
            "processed_train_jsonl": "not emitted because corpus text is absent and release_train_allowed=false",
        },
    }

    manifest_path = run_root / "acquisition-manifest.json"
    split_path = reports / "split-safety.json"
    write_json(manifest_path, manifest)
    write_json(split_path, split_safety)

    sha_lines = [
        f"{entry['sha256']}  {Path(entry['path']).relative_to(run_root)}"
        for entry in local_files
    ]
    (run_root / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    md = [
        "# MS MARCO Passage Split-Safety Probe",
        "",
        f"- Run root: `{run_root}`",
        f"- Train qrels: `{train_qrels['rows']}` rows, `{train_qrels['unique_queries']}` unique queries, `{train_qrels['unique_doc_ids']}` unique positive pids",
        f"- Dev qrels: `{dev_qrels['rows']}` rows, `{dev_qrels['unique_queries']}` unique queries, `{dev_qrels['unique_doc_ids']}` unique positive pids",
        f"- Train/dev query overlap: `{split_safety['train_dev_query_overlap']}`",
        f"- Train/dev positive-pid overlap: `{split_safety['train_dev_positive_doc_overlap']}`",
        f"- Train qrels missing query text: `{split_safety['train_qrels_queries_missing_from_train_queries']}`",
        f"- Dev qrels missing query text: `{split_safety['dev_qrels_queries_missing_from_dev_queries']}`",
        "- Corpus resolvability: `not_run` because the passage collection was not downloaded in this bounded probe.",
        "- Test boundary: no test labels downloaded; test queries/top1000 remain planned-only and not trainable.",
    ]
    (reports / "split-safety.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"manifest": str(manifest_path), "split_safety": str(split_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
