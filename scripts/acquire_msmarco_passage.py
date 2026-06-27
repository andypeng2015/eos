#!/usr/bin/env python3
"""Acquire or audit MS MARCO Passage inputs with research-only gates.

The default mode downloads or consumes only bounded query/qrels artifacts. The
passage collection is resolved only when --download-corpus or --corpus-path is
explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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


SCHEMA = "eos.msmarco_passage_acquisition_manifest.v1"
DATASET_PAGE = "https://microsoft.github.io/msmarco/Datasets.html"
TERMS_PAGE = "https://microsoft.github.io/msmarco/"

OFFICIAL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "passage_collection": {
        "key": "passage_collection",
        "split": "corpus",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        "official_size": "2.9 GB",
        "official_records": 8841823,
        "format": "tsv: pid, passage",
        "default_policy": "opt_in_only",
    },
    "passage_queries": {
        "key": "passage_queries",
        "split": "all_queries",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
        "official_size": "42.0 MB",
        "official_records": 1010916,
        "format": "tar.gz containing queries.train.tsv, queries.dev.tsv, queries.eval.tsv",
        "default_policy": "bounded_probe",
    },
    "passage_qrels_train": {
        "key": "passage_qrels_train",
        "split": "train",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
        "official_size": "10.1 MB",
        "official_records": 532761,
        "format": "TREC qrels TSV: qid, unused, pid, relevance",
        "default_policy": "bounded_probe",
    },
    "passage_qrels_dev": {
        "key": "passage_qrels_dev",
        "split": "dev",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        "official_size": "1.1 MB",
        "official_records": 59273,
        "format": "TREC qrels TSV: qid, unused, pid, relevance",
        "default_policy": "bounded_probe",
    },
    "passage_collection_and_queries": {
        "key": "passage_collection_and_queries",
        "split": "corpus_plus_queries_qrels",
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
        "official_size": "2.9 GB",
        "official_records": 10406754,
        "format": "combined archive",
        "default_policy": "not_downloaded_by_this_script",
    },
}


POLICY = {
    "train_allowed_for_research": True,
    "train_allowed_for_research_scope": "train rows only after query and corpus resolvability succeeds",
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "test_rows_train_allowed": False,
    "dev_rows_train_allowed": False,
    "requires_independent_legal_review_for_products_or_services": True,
    "basis": "MS MARCO is treated as non-commercial research-only input; no release-training claim is made.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=env_path("EOS_MSMARCO_RUN_ROOT"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=env_path("EOS_MSMARCO_SOURCE_ROOT"),
        help="Optional local source directory containing queries/qrels/corpus artifacts for offline audit.",
    )
    parser.add_argument("--download-corpus", action="store_true", help="Explicitly download collection.tar.gz.")
    parser.add_argument("--corpus-path", type=Path, default=env_path("EOS_MSMARCO_CORPUS_PATH"))
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=int(os.environ.get("EOS_MSMARCO_MIN_FREE_BYTES", str(15 * 1024**3))),
        help="Free-space floor for corpus mode. Default comes from EOS_MSMARCO_MIN_FREE_BYTES or 15 GiB.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-head", action="store_true", help="Skip URL HEAD metadata checks.")
    args = parser.parse_args()
    if args.run_root is None:
        raise SystemExit("--run-root or EOS_MSMARCO_RUN_ROOT is required; this script never writes to /tmp by default")
    if args.download_corpus and args.corpus_path:
        raise SystemExit("use either --download-corpus or --corpus-path, not both")
    return args


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def head_url(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "ok": True,
                "status": response.status,
                "content_length": int(response.headers["Content-Length"])
                if response.headers.get("Content-Length")
                else None,
                "content_type": response.headers.get("Content-Type"),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
            }
    except Exception as exc:  # noqa: BLE001 - surfaced in manifest, not fatal.
        return {"ok": False, "error": repr(exc)}


def download(url: str, path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".download")
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(path)


def copy_source(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)


def find_source_file(source_root: Path | None, names: list[str]) -> Path | None:
    if source_root is None:
        return None
    for name in names:
        direct = source_root / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(source_root.rglob(name))
        if matches:
            return matches[0]
    return None


def obtain_artifact(
    artifact_key: str,
    raw: Path,
    source_root: Path | None,
    timeout: int,
    allow_download: bool,
) -> tuple[Path, dict[str, Any]]:
    artifact = OFFICIAL_ARTIFACTS[artifact_key]
    filename = Path(artifact["url"]).name
    source = find_source_file(source_root, [filename])
    target = raw / filename
    mode = "downloaded"
    if source is not None:
        copy_source(source, target)
        mode = "copied_from_source_root"
    elif allow_download:
        if not target.exists():
            download(artifact["url"], target, timeout)
    elif not target.exists():
        raise SystemExit(f"missing {filename}; supply --source-root or allow network download")
    return target, {
        "key": artifact_key,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "source_url": artifact["url"],
        "acquisition_mode": mode,
    }


def safe_extract_tar_gz(path: Path, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    root = dst.resolve()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            target = (dst / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"tar member escapes destination: {member.name}")
            archive.extract(member, dst, filter="data")
            extracted.append(target)
    return extracted


def maybe_extract_queries(queries_archive: Path, query_extract: Path) -> list[Path]:
    if query_extract.exists() and list(query_extract.rglob("queries.*.tsv")):
        return sorted(p for p in query_extract.rglob("queries.*.tsv") if p.is_file())
    return safe_extract_tar_gz(queries_archive, query_extract)


def count_tsv_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        return sum(1 for line in handle if line.rstrip("\n"))


def read_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2 and row[0]:
                queries[row[0]] = row[1]
    return queries


def read_qrels(path: Path) -> dict[str, Any]:
    qids: set[str] = set()
    pids: set[str] = set()
    rows = 0
    malformed = 0
    rel_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
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


def find_query_file(query_extract: Path, split: str, source_root: Path | None) -> Path | None:
    name = f"queries.{split}.tsv"
    extracted = sorted(query_extract.rglob(name))
    if extracted:
        return extracted[0]
    return find_source_file(source_root, [name])


def ensure_qrels(raw: Path, source_root: Path | None, split: str, timeout: int) -> tuple[Path, dict[str, Any]]:
    key = f"passage_qrels_{split}"
    local, record = obtain_artifact(key, raw, source_root, timeout, allow_download=source_root is None)
    return local, record


def read_corpus_ids(path: Path, wanted: set[str]) -> tuple[set[str], dict[str, int]]:
    found: set[str] = set()
    stats = {"rows_scanned": 0, "malformed_rows": 0}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                stats["malformed_rows"] += 1
                continue
            stats["rows_scanned"] += 1
            if row[0] in wanted:
                found.add(row[0])
    return found, stats


def resolve_corpus_input(
    raw: Path,
    source_root: Path | None,
    corpus_path: Path | None,
    download_corpus: bool,
    timeout: int,
    min_free_bytes: int,
) -> tuple[Path | None, dict[str, Any] | None, list[Path]]:
    if not corpus_path and not download_corpus:
        return None, None, []
    free = shutil.disk_usage(raw).free
    if free < min_free_bytes:
        raise SystemExit(f"free space {free} is below corpus floor {min_free_bytes}")
    extracted: list[Path] = []
    if corpus_path:
        local = raw / corpus_path.name
        copy_source(corpus_path, local)
        record = {
            "key": "passage_collection",
            "path": str(local),
            "bytes": local.stat().st_size,
            "sha256": sha256_file(local),
            "source_url": OFFICIAL_ARTIFACTS["passage_collection"]["url"],
            "acquisition_mode": "copied_from_corpus_path",
        }
    else:
        local, record = obtain_artifact("passage_collection", raw, source_root, timeout, allow_download=True)
    corpus_tsv = local
    if local.suffixes[-2:] == [".tar", ".gz"]:
        extracted = safe_extract_tar_gz(local, raw / "collection")
        candidates = [p for p in extracted if p.name in {"collection.tsv", "collection.tsv.gz"}]
        if not candidates:
            candidates = [p for p in extracted if p.is_file()]
        if not candidates:
            raise SystemExit(f"no corpus file found in {local}")
        corpus_tsv = sorted(candidates)[0]
    return corpus_tsv, record, extracted


def public_counts(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if not k.endswith("_ids")}


def build_reports(run_root: Path, split_safety: dict[str, Any], train_qrels: dict[str, Any], dev_qrels: dict[str, Any]) -> None:
    reports = run_root / "reports"
    write_json(reports / "split-safety.json", split_safety)
    corpus = split_safety["corpus_resolvability"]
    lines = [
        "# MS MARCO Passage Split-Safety Audit",
        "",
        f"- Train qrels: `{train_qrels['rows']}` rows, `{train_qrels['unique_queries']}` unique queries, `{train_qrels['unique_doc_ids']}` unique positive pids",
        f"- Dev qrels: `{dev_qrels['rows']}` rows, `{dev_qrels['unique_queries']}` unique queries, `{dev_qrels['unique_doc_ids']}` unique positive pids",
        f"- Train/dev query overlap: `{split_safety['train_dev_query_overlap']}`",
        f"- Train/dev positive-pid overlap: `{split_safety['train_dev_positive_doc_overlap']}`",
        f"- Train qrels missing query text: `{split_safety['train_qrels_queries_missing_from_train_queries']}`",
        f"- Dev qrels missing query text: `{split_safety['dev_qrels_queries_missing_from_dev_queries']}`",
        f"- Corpus resolvability: `{corpus['status']}`",
    ]
    if corpus["status"] == "run":
        lines.extend(
            [
                f"- Train unresolved positive pids: `{corpus['train_unresolved_positive_pids']}`",
                f"- Dev unresolved positive pids: `{corpus['dev_unresolved_positive_pids']}`",
            ]
        )
    else:
        lines.append("- Corpus resolvability was not run because no corpus mode was explicitly requested.")
    lines.append("- Policy: release training and commercial use are not allowed by this engineering gate.")
    (reports / "split-safety.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha256s(run_root: Path, files: list[dict[str, Any]]) -> None:
    lines = []
    seen: set[Path] = set()
    for entry in files:
        path = Path(entry["path"])
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        lines.append(f"{entry['sha256']}  {path.relative_to(run_root)}")
    (run_root / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    raw = run_root / "raw"
    query_extract = raw / "queries"
    run_root.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    disk = shutil.disk_usage(run_root)
    local_files: list[dict[str, Any]] = []

    queries_archive, query_record = obtain_artifact(
        "passage_queries",
        raw,
        args.source_root,
        args.timeout,
        allow_download=args.source_root is None,
    )
    local_files.append(query_record)
    for extracted in maybe_extract_queries(queries_archive, query_extract):
        if extracted.is_file():
            local_files.append(
                {
                    "key": "passage_queries_extracted",
                    "path": str(extracted),
                    "bytes": extracted.stat().st_size,
                    "sha256": sha256_file(extracted),
                    "source_url": OFFICIAL_ARTIFACTS["passage_queries"]["url"],
                    "extracted_from": str(queries_archive),
                    "acquisition_mode": "extracted",
                }
            )

    qrels_train_path, train_record = ensure_qrels(raw, args.source_root, "train", args.timeout)
    qrels_dev_path, dev_record = ensure_qrels(raw, args.source_root, "dev", args.timeout)
    local_files.extend([train_record, dev_record])

    train_qrels = read_qrels(qrels_train_path)
    dev_qrels = read_qrels(qrels_dev_path)
    query_counts: dict[str, Any] = {}
    query_ids: dict[str, set[str]] = {}
    for split in ("train", "dev", "eval"):
        path = find_query_file(query_extract, split, args.source_root)
        if path is None:
            query_counts[split] = {"present": False}
            query_ids[split] = set()
            continue
        queries = read_queries(path)
        query_ids[split] = set(queries)
        query_counts[split] = {
            "present": True,
            "path": str(path),
            "rows": count_tsv_rows(path),
            "unique_queries": len(queries),
            "sha256": sha256_file(path),
        }

    corpus_path, corpus_record, extracted_corpus = resolve_corpus_input(
        raw,
        args.source_root,
        args.corpus_path,
        args.download_corpus,
        args.timeout,
        args.min_free_bytes,
    )
    if corpus_record is not None:
        local_files.append(corpus_record)
    for extracted in extracted_corpus:
        if extracted.is_file():
            local_files.append(
                {
                    "key": "passage_collection_extracted",
                    "path": str(extracted),
                    "bytes": extracted.stat().st_size,
                    "sha256": sha256_file(extracted),
                    "source_url": OFFICIAL_ARTIFACTS["passage_collection"]["url"],
                    "extracted_from": corpus_record["path"] if corpus_record else None,
                    "acquisition_mode": "extracted",
                }
            )

    train_missing_queries = sorted(train_qrels["query_ids"] - query_ids["train"])
    dev_missing_queries = sorted(dev_qrels["query_ids"] - query_ids["dev"])
    corpus_resolvability: dict[str, Any]
    if corpus_path is None:
        corpus_resolvability = {
            "status": "not_run",
            "reason": "passage collection was not supplied; default probe avoids corpus download",
            "required_opt_in": "--download-corpus or --corpus-path",
        }
    else:
        wanted = train_qrels["doc_ids"] | dev_qrels["doc_ids"]
        found, corpus_stats = read_corpus_ids(corpus_path, wanted)
        train_unresolved = sorted(train_qrels["doc_ids"] - found)
        dev_unresolved = sorted(dev_qrels["doc_ids"] - found)
        corpus_resolvability = {
            "status": "run",
            "corpus_path": str(corpus_path),
            "rows_scanned": corpus_stats["rows_scanned"],
            "malformed_rows": corpus_stats["malformed_rows"],
            "wanted_positive_pids": len(wanted),
            "resolved_positive_pids": len(found),
            "train_unresolved_positive_pids": len(train_unresolved),
            "dev_unresolved_positive_pids": len(dev_unresolved),
            "train_unresolved_examples": train_unresolved[:20],
            "dev_unresolved_examples": dev_unresolved[:20],
        }

    split_safety = {
        "train_dev_query_overlap": len(train_qrels["query_ids"] & dev_qrels["query_ids"]),
        "train_dev_positive_doc_overlap": len(train_qrels["doc_ids"] & dev_qrels["doc_ids"]),
        "train_qrels_queries_missing_from_train_queries": len(train_missing_queries),
        "dev_qrels_queries_missing_from_dev_queries": len(dev_missing_queries),
        "missing_query_examples": {"train": train_missing_queries[:20], "dev": dev_missing_queries[:20]},
        "corpus_resolvability": corpus_resolvability,
        "test_boundary": {
            "status": "safe_for_probe",
            "details": "No test labels are acquired; test rows are never training or selection inputs.",
            "test_rows_train_allowed": False,
        },
    }

    artifact_records = []
    for artifact in OFFICIAL_ARTIFACTS.values():
        item = dict(artifact)
        item["source_url"] = artifact["url"]
        item["head"] = None if args.no_head else head_url(artifact["url"], args.timeout)
        artifact_records.append(item)

    manifest = {
        "schema": SCHEMA,
        "dataset": {
            "name": "MS MARCO Passage Ranking",
            "variant": "passage",
            "official_dataset_page": DATASET_PAGE,
            "official_terms_page": TERMS_PAGE,
            "retrieved_at_utc": started,
        },
        "official_source_urls": {
            "documentation": [TERMS_PAGE, DATASET_PAGE],
            "queries": OFFICIAL_ARTIFACTS["passage_queries"]["url"],
            "qrels_train": OFFICIAL_ARTIFACTS["passage_qrels_train"]["url"],
            "qrels_dev": OFFICIAL_ARTIFACTS["passage_qrels_dev"]["url"],
            "optional_corpus": OFFICIAL_ARTIFACTS["passage_collection"]["url"],
        },
        "license_terms_summary": {
            "source_urls": [TERMS_PAGE, DATASET_PAGE],
            "summary": "Engineering gate treats MS MARCO Passage as non-commercial research-only data.",
            "engineering_policy": POLICY,
        },
        "disk_preflight": {
            "path": str(run_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "corpus_min_free_bytes": args.min_free_bytes,
        },
        "artifacts": artifact_records,
        "local_files": local_files,
        "counts": {
            "official_records_from_dataset_page": {
                key: item["official_records"] for key, item in OFFICIAL_ARTIFACTS.items()
            },
            "probe": {
                "qrels_train": public_counts(train_qrels),
                "qrels_dev": public_counts(dev_qrels),
                "queries": query_counts,
            },
        },
        "splits": {
            "train": {
                "downloaded_or_consumed": ["qrels.train.tsv", "queries.train.tsv"],
                "train_allowed_for_research": True,
                "train_allowed_requires_corpus_resolvability": True,
                "release_train_allowed": False,
            },
            "dev": {
                "downloaded_or_consumed": ["qrels.dev.tsv", "queries.dev.tsv"],
                "train_allowed_for_research": False,
                "selection_or_sanity_only": True,
                "release_train_allowed": False,
            },
            "test": {
                "downloaded_or_consumed": [],
                "train_allowed_for_research": False,
                "test_rows_train_allowed": False,
                "release_train_allowed": False,
            },
        },
        "split_safety": split_safety,
        "skipped_or_missing": {
            "corpus_text_rows": "not downloaded" if corpus_path is None else "resolved/audited",
            "hard_negative_triples": "not downloaded",
            "top1000_reranking_rows": "not downloaded",
            "release_training_jsonl": "not emitted; release_train_allowed=false",
        },
    }

    write_json(run_root / "acquisition-manifest.json", manifest)
    build_reports(run_root, split_safety, train_qrels, dev_qrels)
    write_sha256s(run_root, local_files)
    return {"manifest": str(run_root / "acquisition-manifest.json"), "split_safety": str(run_root / "reports" / "split-safety.json")}


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
