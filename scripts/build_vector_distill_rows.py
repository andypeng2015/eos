#!/usr/bin/env python3
"""Build a role-tagged `eos.vector_distill_example.v1` JSONL dataset.

Reads raw query/document text, computes BAAI/bge-small-en-v1.5 imported-teacher
embeddings with the role-correct prefix (query prefix for queries, empty
prefix for documents/raw text), and writes rows compatible with
`runtime/embedding_vector_distill_dataset.go`:

    {"schema": "eos.vector_distill_example.v1", "id": ..., "text": ...,
     "teacher_vector": [...], "role": "query"|"document"|"raw",
     "train_allowed_for_research": bool, "release_train_allowed": bool,
     "commercial_use_allowed": bool}

`text` is always the RAW input text with no teacher prefix applied -- the
prefix is teacher-side only (see `--query-prefix`/`--document-prefix`); the
student model applies its own role conditioning natively at train time.

Teacher embedding runs through the `eos export-pretrained-bert-retrieval-vectors`
CLI (batched, subprocess), sharded across a small pool of parallel worker
processes for CPU throughput. Each worker gets its own scratch BEIR-shaped
directory (a `corpus.jsonl` holding that shard's texts, a single throwaway
query, and a trivial qrels row) so the same "documents" code path is reused
for both query-role and document-role batches -- only the `--document-prefix`
passed to the worker changes between the two passes. This sidesteps the
exporter's qrels-driven query filtering entirely.

Inputs
------
--queries PATH        JSONL or TSV of query rows (id + text).
--documents PATH       JSONL of document rows (id + text).
--mixed PATH           Alternative to --queries/--documents: a single JSONL
                        file where each row also carries a "kind" field
                        ("query", "document", or "raw").
--relations PATH       Optional JSONL with {"query_id", "positive_doc_id",
                        "negative_doc_ids": [...]} rows (the same shape as
                        runtime hard-negative teacher-scored rows). Used only
                        by --group-order.

Determinism: row order is the input order unless --group-order is set (see
below), and --limit-queries/--limit-docs take a deterministic prefix. No
randomness is introduced anywhere in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = "eos.vector_distill_example.v1"
ROLE_QUERY = "query"
ROLE_DOCUMENT = "document"
ROLE_RAW = "raw"
VALID_ROLES = (ROLE_QUERY, ROLE_DOCUMENT, ROLE_RAW)

DEFAULT_EOS_BIN = "env GOWORK=off go run ./cmd/eos"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_DOCUMENT_PREFIX = ""
DEFAULT_RAW_PREFIX = ""

ID_KEYS = ("id", "_id", "query_id", "doc_id", "document_id", "positive_doc_id")
TEXT_KEYS = ("text", "query", "document", "positive", "passage")


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    role: str


@dataclass
class EmbedShardResult:
    vectors: dict[str, list[float]]
    elapsed_seconds: float
    shard_index: int
    row_count: int


@dataclass
class EmbedBucketReport:
    role: str
    prefix: str
    requested: int
    embedded: int
    skipped_for_time_budget: int
    wall_clock_seconds: float
    shards: int


class TimeBudgetExceeded(Exception):
    """Raised internally to stop submitting new shards once the wall-clock budget is spent."""


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def split_command(value: str) -> list[str]:
    parts = shlex.split(value)
    if not parts:
        raise SystemExit("--eos-bin must not be empty")
    return parts


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return None


def read_jsonl_id_text_rows(path: Path, role: str) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
            row_id = first_present(record, ID_KEYS)
            text = first_present(record, TEXT_KEYS)
            if row_id is None:
                raise SystemExit(f"{path}:{line_no}: no id field found (tried {ID_KEYS})")
            if text is None:
                raise SystemExit(f"{path}:{line_no}: no text field found (tried {TEXT_KEYS})")
            rows.append(Row(id=row_id, text=text, role=role))
    return rows


def read_tsv_id_text_rows(path: Path, role: str) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        first = True
        for line_no, cols in enumerate(reader, start=1):
            if not cols or (len(cols) == 1 and not cols[0].strip()):
                continue
            if first:
                first = False
                lowered = [c.strip().lower() for c in cols]
                if lowered[:2] in (["id", "text"], ["query_id", "text"], ["id", "query"]):
                    continue
            if len(cols) < 2:
                raise SystemExit(f"{path}:{line_no}: expected at least 2 TSV columns (id, text)")
            rows.append(Row(id=cols[0].strip(), text=cols[1].strip(), role=role))
    return rows


def read_id_text_rows(path: Path, role: str) -> list[Row]:
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".txt"):
        return read_tsv_id_text_rows(path, role)
    return read_jsonl_id_text_rows(path, role)


def read_mixed_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
            kind = str(record.get("kind") or record.get("role") or "").strip().lower()
            if kind not in VALID_ROLES:
                raise SystemExit(f"{path}:{line_no}: kind/role must be one of {VALID_ROLES}, got {kind!r}")
            row_id = first_present(record, ID_KEYS)
            text = first_present(record, TEXT_KEYS)
            if row_id is None or text is None:
                raise SystemExit(f"{path}:{line_no}: mixed row requires id and text fields")
            rows.append(Row(id=row_id, text=text, role=kind))
    return rows


def dedupe_rows(rows: list[Row]) -> tuple[list[Row], int]:
    seen: set[str] = set()
    out: list[Row] = []
    dup_count = 0
    for row in rows:
        key = f"{row.role}:{row.id}"
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dup_count


# ---------------------------------------------------------------------------
# Group-order
# ---------------------------------------------------------------------------


def read_relations(path: Path, negatives_cap: int) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
            query_id = record.get("query_id")
            if query_id is None:
                raise SystemExit(f"{path}:{line_no}: relations row missing query_id")
            positive_doc_id = record.get("positive_doc_id")
            negative_doc_ids = list(record.get("negative_doc_ids") or [])
            if negatives_cap > 0:
                negative_doc_ids = negative_doc_ids[:negatives_cap]
            relations.append(
                {
                    "query_id": str(query_id),
                    "positive_doc_id": str(positive_doc_id) if positive_doc_id is not None else None,
                    "negative_doc_ids": [str(x) for x in negative_doc_ids],
                }
            )
    return relations


def apply_group_order(rows: list[Row], relations: list[dict[str, Any]]) -> list[Row]:
    """Reorder rows so each query is immediately followed by its positive doc
    and (capped) negative docs, in relation-file order. Docs already emitted
    for an earlier query are not re-emitted. Rows with no relation entry (or
    referenced docs that never appear in `rows`) are appended at the end in
    original order. The output is a permutation of `rows`: no row is added,
    dropped, or duplicated.
    """
    by_key: dict[str, Row] = {f"{r.role}:{r.id}": r for r in rows}
    emitted: set[str] = set()
    ordered: list[Row] = []

    def emit(role: str, row_id: str) -> None:
        key = f"{role}:{row_id}"
        if key in emitted:
            return
        row = by_key.get(key)
        if row is None:
            return
        ordered.append(row)
        emitted.add(key)

    for relation in relations:
        emit(ROLE_QUERY, relation["query_id"])
        if relation["positive_doc_id"] is not None:
            emit(ROLE_DOCUMENT, relation["positive_doc_id"])
        for neg_id in relation["negative_doc_ids"]:
            emit(ROLE_DOCUMENT, neg_id)

    for row in rows:
        key = f"{row.role}:{row.id}"
        if key not in emitted:
            ordered.append(row)
            emitted.add(key)

    assert len(ordered) == len(rows), "group-order must be a permutation of the input rows"
    return ordered


# ---------------------------------------------------------------------------
# Teacher embedding via subprocess shards
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunk_rows_by_size(rows: list[Row], shard_size: int) -> list[list[Row]]:
    """Split into fixed-size shards (last shard may be smaller). Using a fixed
    shard size (rather than a fixed shard count) keeps per-shard subprocess
    duration roughly constant regardless of total row count, which is what
    gives --time-budget-seconds useful between-wave checkpoint granularity."""
    if not rows:
        return []
    shard_size = max(1, shard_size)
    return [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]


def write_shard_dataset(shard_dir: Path, rows: list[Row]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "qrels").mkdir(parents=True, exist_ok=True)
    with (shard_dir / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"_id": row.id, "text": row.text, "title": ""}) + "\n")
    with (shard_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_id": "__dummy_query__", "text": "placeholder"}) + "\n")
    with (shard_dir / "qrels" / "dummy.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        handle.write(f"__dummy_query__\t{rows[0].id}\t1\n")


def run_shard(
    eos_bin: list[str],
    package_path: Path,
    shard_dir: Path,
    out_dir: Path,
    prefix: str,
    batch_size: int,
    gomaxprocs: int,
    dataset_label: str,
) -> float:
    manifest_path = out_dir / "manifest.json"
    cmd = [
        *eos_bin,
        "export-pretrained-bert-retrieval-vectors",
        "--dataset",
        dataset_label,
        "--split",
        "dummy",
        "--package",
        str(package_path),
        "--query-prefix",
        "",
        "--document-prefix",
        prefix,
        "--batch-size",
        str(batch_size),
        "--max-docs",
        "0",
        "--max-queries",
        "0",
        "--manifest-json",
        str(manifest_path),
        str(shard_dir),
        str(out_dir),
    ]
    env = dict(os.environ)
    env["GOWORK"] = "off"
    if gomaxprocs > 0:
        env["GOMAXPROCS"] = str(gomaxprocs)
    start = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"shard embedding failed (dataset={dataset_label}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return elapsed


def read_doc_vectors(path: Path) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_id = record.get("id") or record.get("_id")
            vector = record.get("vector") or record.get("embedding") or record.get("values")
            if row_id is None or vector is None:
                continue
            vectors[str(row_id)] = [float(v) for v in vector]
    return vectors


def embed_texts(
    rows: list[Row],
    role: str,
    prefix: str,
    args: argparse.Namespace,
    work_root: Path,
) -> tuple[EmbedBucketReport, dict[str, list[float]]]:
    """Embed `rows` (all the same role) via subprocess shards. This is the
    function tests monkeypatch to stub out the teacher entirely. Returns
    (bucket_report, {row_id: vector})."""
    bucket_start = time.monotonic()
    requested = len(rows)
    if requested == 0:
        return EmbedBucketReport(role=role, prefix=prefix, requested=0, embedded=0, skipped_for_time_budget=0, wall_clock_seconds=0.0, shards=0), {}

    eos_bin = split_command(args.eos_bin)
    package_path = Path(args.teacher_package)
    shards = chunk_rows_by_size(rows, args.shard_size)
    vectors: dict[str, list[float]] = {}
    embedded_row_ids: set[str] = set()
    shards_run = 0

    def deadline_exceeded() -> bool:
        if args.time_budget_seconds <= 0:
            return False
        return (time.monotonic() - args.run_start) >= args.time_budget_seconds

    def submit(shard_index: int, shard_rows: list[Row]) -> EmbedShardResult:
        shard_dir = work_root / f"{role}-shard-{shard_index:04d}"
        out_dir = work_root / f"{role}-shard-{shard_index:04d}-out"
        write_shard_dataset(shard_dir, shard_rows)
        elapsed = run_shard(
            eos_bin,
            package_path,
            shard_dir,
            out_dir,
            prefix,
            args.batch_size,
            args.gomaxprocs_per_shard,
            f"{args.dataset_label}-{role}-{shard_index:04d}",
        )
        shard_vectors = read_doc_vectors(out_dir / "doc-vectors.jsonl")
        if not args.keep_work_dir:
            _cleanup_dir(shard_dir)
            _cleanup_dir(out_dir)
        return EmbedShardResult(vectors=shard_vectors, elapsed_seconds=elapsed, shard_index=shard_index, row_count=len(shard_rows))

    wave_size = max(1, args.parallel_shards)
    idx = 0
    while idx < len(shards):
        if deadline_exceeded():
            break
        wave = shards[idx : idx + wave_size]
        wave_start = idx
        idx += wave_size
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [pool.submit(submit, wave_start + offset, shard_rows) for offset, shard_rows in enumerate(wave)]
            for future in futures:
                result = future.result()
                vectors.update(result.vectors)
                shards_run += 1

    for row in rows:
        if row.id in vectors:
            embedded_row_ids.add(row.id)

    return (
        EmbedBucketReport(
            role=role,
            prefix=prefix,
            requested=requested,
            embedded=len(embedded_row_ids),
            skipped_for_time_budget=requested - len(embedded_row_ids),
            wall_clock_seconds=time.monotonic() - bucket_start,
            shards=shards_run,
        ),
        vectors,
    )


def _cleanup_dir(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=None, help="Query rows: JSONL or TSV (id, text).")
    parser.add_argument("--documents", type=Path, default=None, help="Document rows: JSONL (id, text).")
    parser.add_argument("--mixed", type=Path, default=None, help="Single JSONL with a 'kind' field per row instead of --queries/--documents.")
    parser.add_argument("--relations", type=Path, default=None, help="JSONL with query_id/positive_doc_id/negative_doc_ids rows, for --group-order.")
    parser.add_argument("--relations-negatives-cap", type=int, default=0, help="Cap negative_doc_ids per relation row (0 = no cap).")
    parser.add_argument("--group-order", action="store_true", help="Order output so each query is contiguous with its positive+negative docs (requires --relations).")

    parser.add_argument("--teacher-package", required=True, help="Imported BGE .mll package path.")
    parser.add_argument("--eos-bin", default=DEFAULT_EOS_BIN, help="eos command; default 'env GOWORK=off go run ./cmd/eos'.")
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--document-prefix", default=DEFAULT_DOCUMENT_PREFIX)
    parser.add_argument("--raw-prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=400, help="Rows per embedding subprocess shard. Kept roughly constant so --time-budget-seconds gets steady between-wave checkpoints regardless of total row count.")
    parser.add_argument("--parallel-shards", type=int, default=8, help="Concurrent eos subprocess workers per role bucket (one wave = this many shards run at once).")
    parser.add_argument("--gomaxprocs-per-shard", type=int, default=3, help="GOMAXPROCS set on each worker subprocess (avoids goroutine oversubscription across parallel workers).")
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch dir for per-shard BEIR datasets; default a fresh tempdir.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Do not delete per-shard scratch dirs (debugging).")
    parser.add_argument("--time-budget-seconds", type=float, default=0.0, help="Stop launching new embedding shards once this wall-clock budget (from script start) is spent; already-embedded rows are still written. 0 = unlimited.")
    parser.add_argument("--dataset-label", default="vector-distill", help="Label embedded into per-shard dataset names (cosmetic, shows up in shard manifests).")

    parser.add_argument("--seed", type=int, default=0, help="Reserved for deterministic sampling (unused unless --limit-* is combined with future --shuffle support).")
    parser.add_argument("--limit-queries", type=int, default=0, help="Deterministic prefix cap on query rows (0 = no limit).")
    parser.add_argument("--limit-docs", type=int, default=0, help="Deterministic prefix cap on document rows (0 = no limit).")

    parser.add_argument("--train-allowed-for-research", action="store_true", default=False)
    parser.add_argument("--release-train-allowed", action="store_true", default=False)
    parser.add_argument("--commercial-use-allowed", action="store_true", default=False)

    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON path; default <output>.manifest.json.")

    args = parser.parse_args(argv)

    if args.mixed is None and args.queries is None and args.documents is None:
        parser.error("must supply --mixed or at least one of --queries/--documents")
    if args.mixed is not None and (args.queries is not None or args.documents is not None):
        parser.error("--mixed cannot be combined with --queries/--documents")
    if args.group_order and args.relations is None:
        parser.error("--group-order requires --relations")
    if args.manifest is None:
        args.manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    return args


def load_input_rows(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    if args.mixed is not None:
        rows = read_mixed_rows(args.mixed)
    else:
        query_rows: list[Row] = []
        doc_rows: list[Row] = []
        if args.queries is not None:
            query_rows = read_id_text_rows(args.queries, ROLE_QUERY)
            if args.limit_queries > 0:
                query_rows = query_rows[: args.limit_queries]
        if args.documents is not None:
            doc_rows = read_id_text_rows(args.documents, ROLE_DOCUMENT)
            if args.limit_docs > 0:
                doc_rows = doc_rows[: args.limit_docs]
        rows = query_rows + doc_rows
    rows, dup_count = dedupe_rows(rows)
    if dup_count:
        print(f"warning: dropped {dup_count} duplicate (role, id) rows", file=sys.stderr)
    return rows


def role_prefix(role: str, args: argparse.Namespace) -> str:
    return {ROLE_QUERY: args.query_prefix, ROLE_DOCUMENT: args.document_prefix, ROLE_RAW: args.raw_prefix}[role]


def build_rows(args: argparse.Namespace, embedder: Callable[..., Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full pipeline: load inputs, order them, embed by role bucket, and
    assemble output rows + a manifest dict. `embedder` is the seam tests
    monkeypatch (same signature as `embed_texts`). Resolving the module-level
    `embed_texts` name at call time (rather than via a function-def-time
    default argument) means `monkeypatch.setattr(module, "embed_texts", fake)`
    is honored even when callers don't pass `embedder` explicitly."""
    if embedder is None:
        embedder = embed_texts
    args.run_start = time.monotonic()
    input_rows = load_input_rows(args)

    if args.group_order:
        relations = read_relations(args.relations, args.relations_negatives_cap)
        input_rows = apply_group_order(input_rows, relations)

    buckets: dict[str, list[Row]] = {ROLE_QUERY: [], ROLE_DOCUMENT: [], ROLE_RAW: []}
    for row in input_rows:
        buckets[row.role].append(row)

    work_dir_ctx = _static_dir(args.work_dir) if args.work_dir is not None else tempfile.TemporaryDirectory()
    with work_dir_ctx as work_root_str:
        work_root = Path(work_root_str)
        vectors: dict[str, dict[str, list[float]]] = {}
        bucket_reports: list[EmbedBucketReport] = []
        for role in (ROLE_QUERY, ROLE_DOCUMENT, ROLE_RAW):
            rows = buckets[role]
            if not rows:
                continue
            prefix = role_prefix(role, args)
            report, role_vectors = embedder(rows, role, prefix, args, work_root)
            bucket_reports.append(report)
            vectors[role] = role_vectors

    output_rows: list[dict[str, Any]] = []
    counts_by_role = {ROLE_QUERY: 0, ROLE_DOCUMENT: 0, ROLE_RAW: 0}
    skipped_by_role = {ROLE_QUERY: 0, ROLE_DOCUMENT: 0, ROLE_RAW: 0}
    for row in input_rows:
        role_vectors = vectors.get(row.role, {})
        vector = role_vectors.get(row.id)
        if vector is None:
            skipped_by_role[row.role] += 1
            continue
        output_rows.append(
            {
                "schema": SCHEMA,
                "id": row.id,
                "text": row.text,
                "teacher_vector": vector,
                "role": row.role,
                "train_allowed_for_research": bool(args.train_allowed_for_research),
                "release_train_allowed": bool(args.release_train_allowed),
                "commercial_use_allowed": bool(args.commercial_use_allowed),
            }
        )
        counts_by_role[row.role] += 1

    capped = any(r.skipped_for_time_budget > 0 for r in bucket_reports)
    manifest = {
        "schema": "eos.vector_distill_rows_build.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "row_counts_by_role": counts_by_role,
        "skipped_by_role": skipped_by_role,
        "total_rows": len(output_rows),
        "teacher_package_path": str(args.teacher_package),
        "teacher_package_sha256": sha256_file(Path(args.teacher_package)) if Path(args.teacher_package).is_file() else None,
        "query_prefix": args.query_prefix,
        "document_prefix": args.document_prefix,
        "raw_prefix": args.raw_prefix,
        "group_order": bool(args.group_order),
        "relations_path": str(args.relations) if args.relations else None,
        "relations_negatives_cap": args.relations_negatives_cap,
        "capped_for_time_budget": capped,
        "time_budget_seconds": args.time_budget_seconds,
        "wall_clock_seconds": time.monotonic() - args.run_start,
        "buckets": [
            {
                "role": r.role,
                "prefix": r.prefix,
                "requested": r.requested,
                "embedded": r.embedded,
                "skipped_for_time_budget": r.skipped_for_time_budget,
                "wall_clock_seconds": r.wall_clock_seconds,
                "shards": r.shards,
            }
            for r in bucket_reports
        ],
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k not in ("run_start",)
        },
    }
    return output_rows, manifest


class _static_dir:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, *exc: Any) -> None:
        return None


def write_output(rows: list[dict[str, Any]], manifest: dict[str, Any], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, manifest = build_rows(args)
    write_output(rows, manifest, args)
    print(f"wrote {len(rows)} rows -> {args.output}")
    print(f"manifest -> {args.manifest}")
    if manifest["capped_for_time_budget"]:
        print("WARNING: time budget exceeded; some rows were skipped (see manifest.skipped_by_role)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
