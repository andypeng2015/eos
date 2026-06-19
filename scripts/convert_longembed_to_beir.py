#!/usr/bin/env python3
"""Convert official LongEmbed configs into this repo's BEIR retrieval layout.

Source dataset: https://huggingface.co/datasets/dwzhu/LongEmbed
Reference code/paper artifacts: https://github.com/dwzhu-pku/LongEmbed

This script is an acquisition/conversion adapter only. It does not run model
evaluation and does not make a benchmark-quality claim by itself.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SOURCE_DATASET = "dwzhu/LongEmbed"
SOURCE_URLS = [
    "https://huggingface.co/datasets/dwzhu/LongEmbed",
    "https://github.com/dwzhu-pku/LongEmbed",
]
OFFICIAL_CONFIGS = ("narrativeqa", "qmsum", "2wikimqa", "summ_screen_fd", "passkey", "needle")
OFFICIAL_SPLITS = ("corpus", "queries", "qrels")
SYNTHETIC_CONTEXT_LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

CORPUS_ID_KEYS = ("doc_id", "corpus_id", "_id", "id", "pid", "passage_id")
QUERY_ID_KEYS = ("qid", "query_id", "_id", "id")
QREL_QUERY_ID_KEYS = ("qid", "query_id", "_id", "id")
QREL_DOC_ID_KEYS = ("doc_id", "corpus_id", "pid", "passage_id", "document_id")
TITLE_KEYS = ("title", "document_title")
CORPUS_TEXT_KEYS = ("text", "document", "passage", "context", "content", "body")
QUERY_TEXT_KEYS = ("text", "query", "question", "input")
SCORE_KEYS = ("score", "relevance", "label")
CONTEXT_LENGTH_KEYS = ("context_length", "ctx_len", "length", "input_length", "doc_length")


@dataclass(frozen=True)
class TextRow:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class QrelRow:
    query_id: str
    doc_id: str
    score: float


@dataclass
class ConvertResult:
    dataset: str
    output_dir: Path
    corpus_count: int
    query_count: int
    qrel_count: int
    warnings: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LongEmbed corpus/queries/qrels splits into BEIR-compatible "
            "corpus.jsonl, queries.jsonl, and qrels/<split>.tsv files."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "LongEmbed config to convert. May be repeated or comma-separated. "
            f"Defaults to all official configs: {','.join(OFFICIAL_CONFIGS)}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/longembed-official"),
        help="Output root for <dataset>/corpus.jsonl etc. Default: datasets/longembed-official.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Offline fixture/raw root containing <dataset>/{corpus,queries,qrels}.jsonl. "
            "When omitted, the script lazily imports datasets.load_dataset."
        ),
    )
    parser.add_argument("--split-name", default="test", help="Output qrels split filename. Default: test.")
    parser.add_argument("--max-docs", type=int, default=0, help="Maximum documents to emit; 0 means all.")
    parser.add_argument("--max-queries", type=int, default=0, help="Maximum queries to emit; 0 means all.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Optional synthetic length filter for needle/passkey when rows expose "
            "context_length, ctx_len, length, related fields, or IDs containing the numeric length."
        ),
    )
    return parser.parse_args(argv)


def parse_dataset_selection(values: Sequence[str]) -> list[str]:
    if not values:
        return list(OFFICIAL_CONFIGS)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if not name:
                continue
            if name not in OFFICIAL_CONFIGS:
                raise ValueError(
                    f"unsupported LongEmbed config {name!r}; expected one of {', '.join(OFFICIAL_CONFIGS)}"
                )
            if name not in seen:
                seen.add(name)
                out.append(name)
    if not out:
        raise ValueError("no datasets selected")
    return out


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def load_local_split(input_root: Path, dataset: str, split: str) -> list[dict]:
    path = input_root / dataset / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing local LongEmbed split: {path}")
    return list(iter_jsonl(path))


def load_hf_split(dataset: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Online LongEmbed acquisition requires the optional 'datasets' package. "
            "Install it in a local environment, for example: python3 -m pip install datasets. "
            "For no-network verification, pass --input-root with local JSONL fixtures."
        ) from exc
    loaded = load_dataset(SOURCE_DATASET, dataset, split=split)
    return [dict(row) for row in loaded]


def load_splits(dataset: str, input_root: Path | None) -> dict[str, list[dict]]:
    loader = (lambda split: load_local_split(input_root, dataset, split)) if input_root else (
        lambda split: load_hf_split(dataset, split)
    )
    return {split: loader(split) for split in OFFICIAL_SPLITS}


def first_value(row: dict, keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def clean_id(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty {label}")
    return text


def first_text(row: dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            text = "\n".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()
        if text:
            return text
    return ""


def normalize_corpus_row(row: dict) -> TextRow:
    row_id = first_value(row, CORPUS_ID_KEYS)
    if row_id is None:
        raise ValueError(f"corpus row missing id field from {CORPUS_ID_KEYS}: {row!r}")
    text = first_text(row, CORPUS_TEXT_KEYS)
    if not text:
        raise ValueError(f"corpus row {row_id!r} has empty text")
    title = first_text(row, TITLE_KEYS)
    return TextRow(clean_id(row_id, "corpus id"), text, title)


def normalize_query_row(row: dict) -> TextRow:
    row_id = first_value(row, QUERY_ID_KEYS)
    if row_id is None:
        raise ValueError(f"query row missing id field from {QUERY_ID_KEYS}: {row!r}")
    text = first_text(row, QUERY_TEXT_KEYS)
    if not text:
        raise ValueError(f"query row {row_id!r} has empty text")
    return TextRow(clean_id(row_id, "query id"), text)


def normalize_qrel_row(row: dict) -> QrelRow:
    query_id = first_value(row, QREL_QUERY_ID_KEYS)
    doc_id = first_value(row, QREL_DOC_ID_KEYS)
    if query_id is None or doc_id is None:
        raise ValueError(f"qrels row missing query/doc id fields: {row!r}")
    score_value = first_value(row, SCORE_KEYS)
    score = 1.0 if score_value is None else float(score_value)
    return QrelRow(clean_id(query_id, "qrel query id"), clean_id(doc_id, "qrel doc id"), score)


def id_implies_context_length(row: dict, context_length: int) -> bool | None:
    values: list[str] = []
    for key in (*CORPUS_ID_KEYS, *QUERY_ID_KEYS, *QREL_DOC_ID_KEYS, *QREL_QUERY_ID_KEYS):
        value = row.get(key)
        if value is not None:
            values.append(str(value))
    if not values:
        return None
    found_any = False
    for value in values:
        for match in re.findall(r"(?<!\d)(\d{3,5})(?!\d)", value):
            number = int(match)
            if number in SYNTHETIC_CONTEXT_LENGTHS:
                found_any = True
                if number == context_length:
                    return True
    return False if found_any else None


def row_matches_context_length(row: dict, context_length: int) -> bool | None:
    for key in CONTEXT_LENGTH_KEYS:
        if key not in row or row[key] is None:
            continue
        try:
            return int(row[key]) == context_length
        except (TypeError, ValueError):
            continue
    return id_implies_context_length(row, context_length)


def filter_context_rows(
    rows: list[dict],
    context_length: int | None,
    split: str,
    warnings: list[str],
) -> tuple[list[dict], bool]:
    if context_length is None:
        return rows, False
    matched: list[dict] = []
    inspected = 0
    for row in rows:
        result = row_matches_context_length(row, context_length)
        if result is None:
            continue
        inspected += 1
        if result:
            matched.append(row)
    if inspected == 0:
        warnings.append(
            f"context length {context_length} was requested, but {split} rows expose no recognized length field or id marker"
        )
        return rows, False
    warnings.append(
        f"context length {context_length} filter applied to {split}: kept {len(matched)} of {inspected} length-marked rows"
    )
    return matched, True


def dedupe_text_rows(rows: Iterable[TextRow], label: str, warnings: list[str]) -> list[TextRow]:
    out: list[TextRow] = []
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        if row.id in seen:
            duplicates += 1
            continue
        seen.add(row.id)
        out.append(row)
    if duplicates:
        warnings.append(f"ignored {duplicates} duplicate {label} ids using first-row-wins order")
    return out


def dedupe_qrels(rows: Iterable[QrelRow], warnings: list[str]) -> list[QrelRow]:
    by_pair: dict[tuple[str, str], float] = {}
    order: list[tuple[str, str]] = []
    duplicates = 0
    for row in rows:
        if row.score <= 0:
            continue
        key = (row.query_id, row.doc_id)
        if key not in by_pair:
            order.append(key)
            by_pair[key] = row.score
        else:
            duplicates += 1
            by_pair[key] = max(by_pair[key], row.score)
    if duplicates:
        warnings.append(f"merged {duplicates} duplicate positive qrel pairs using max score")
    return [QrelRow(query_id, doc_id, by_pair[(query_id, doc_id)]) for query_id, doc_id in order]


def validate_qrels(
    qrels: list[QrelRow],
    query_ids: set[str],
    doc_ids: set[str],
    allow_drop_missing: bool,
    warnings: list[str],
) -> list[QrelRow]:
    valid: list[QrelRow] = []
    missing_queries: set[str] = set()
    missing_docs: set[str] = set()
    for row in qrels:
        missing_query = row.query_id not in query_ids
        missing_doc = row.doc_id not in doc_ids
        if missing_query:
            missing_queries.add(row.query_id)
        if missing_doc:
            missing_docs.add(row.doc_id)
        if not missing_query and not missing_doc:
            valid.append(row)
    if missing_queries or missing_docs:
        message = (
            f"qrels reference missing ids: {len(missing_queries)} queries, "
            f"{len(missing_docs)} docs"
        )
        if not allow_drop_missing:
            raise ValueError(message)
        warnings.append(message + "; dropped those qrel rows after source filtering")
    return valid


def select_queries(queries: list[TextRow], qrels: list[QrelRow], max_queries: int) -> list[TextRow]:
    if max_queries <= 0:
        return queries
    qrel_query_ids = {row.query_id for row in qrels}
    selected: list[TextRow] = []
    for row in queries:
        if row.id in qrel_query_ids:
            selected.append(row)
            if len(selected) >= max_queries:
                break
    return selected


def select_corpus_preserving_relevant(
    corpus: list[TextRow],
    qrels: list[QrelRow],
    max_docs: int,
    warnings: list[str],
) -> list[TextRow]:
    if max_docs <= 0:
        return corpus
    relevant = {row.doc_id for row in qrels}
    selected: list[TextRow] = []
    selected_ids: set[str] = set()
    for row in corpus:
        if row.id in relevant:
            selected.append(row)
            selected_ids.add(row.id)
    if len(selected) > max_docs:
        warnings.append(
            f"max-docs={max_docs} was exceeded to preserve {len(selected)} qrels-relevant documents"
        )
        return selected
    for row in corpus:
        if len(selected) >= max_docs:
            break
        if row.id not in selected_ids:
            selected.append(row)
            selected_ids.add(row.id)
    return selected


def text_stats(rows: list[TextRow]) -> dict[str, float | int]:
    word_counts = [len(row.text.split()) for row in rows]
    char_counts = [len(row.text) for row in rows]
    if not rows:
        return {
            "rows": 0,
            "words_min": 0,
            "words_max": 0,
            "words_mean": 0.0,
            "chars_min": 0,
            "chars_max": 0,
            "chars_mean": 0.0,
        }
    return {
        "rows": len(rows),
        "words_min": min(word_counts),
        "words_max": max(word_counts),
        "words_mean": round(statistics.fmean(word_counts), 3),
        "chars_min": min(char_counts),
        "chars_max": max(char_counts),
        "chars_mean": round(statistics.fmean(char_counts), 3),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def write_qrels(path: Path, qrels: list[QrelRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            score = int(row.score) if row.score.is_integer() else row.score
            handle.write(f"{row.query_id}\t{row.doc_id}\t{score}\n")
    return len(qrels)


def convert_dataset(
    dataset: str,
    splits: dict[str, list[dict]],
    output_root: Path,
    split_name: str = "test",
    max_docs: int = 0,
    max_queries: int = 0,
    context_length: int | None = None,
    input_root: Path | None = None,
) -> ConvertResult:
    warnings: list[str] = []
    if context_length is not None and context_length not in SYNTHETIC_CONTEXT_LENGTHS:
        warnings.append(
            f"context length {context_length} is outside official synthetic lengths "
            f"{','.join(str(value) for value in SYNTHETIC_CONTEXT_LENGTHS)}"
        )
    filtered_splits: dict[str, list[dict]] = {}
    context_applied: list[str] = []
    for split, rows in splits.items():
        filtered, applied = filter_context_rows(rows, context_length, split, warnings)
        filtered_splits[split] = filtered
        if applied:
            context_applied.append(split)

    corpus = dedupe_text_rows(
        (normalize_corpus_row(row) for row in filtered_splits["corpus"]),
        "corpus",
        warnings,
    )
    queries = dedupe_text_rows(
        (normalize_query_row(row) for row in filtered_splits["queries"]),
        "query",
        warnings,
    )
    qrels = dedupe_qrels((normalize_qrel_row(row) for row in filtered_splits["qrels"]), warnings)

    if not corpus:
        raise ValueError(f"{dataset}: no corpus rows selected")
    if not queries:
        raise ValueError(f"{dataset}: no query rows selected")
    if not qrels:
        raise ValueError(f"{dataset}: no positive qrels selected")

    corpus_ids = {row.id for row in corpus}
    query_ids = {row.id for row in queries}
    qrels = validate_qrels(qrels, query_ids, corpus_ids, bool(context_applied), warnings)
    if not qrels:
        raise ValueError(f"{dataset}: no qrels remain after source filtering")

    selected_queries = select_queries(queries, qrels, max_queries)
    selected_query_ids = {row.id for row in selected_queries}
    selected_qrels = [row for row in qrels if row.query_id in selected_query_ids]
    selected_corpus = select_corpus_preserving_relevant(corpus, selected_qrels, max_docs, warnings)
    selected_doc_ids = {row.id for row in selected_corpus}
    missing_after_caps = [
        row for row in selected_qrels if row.query_id not in selected_query_ids or row.doc_id not in selected_doc_ids
    ]
    if missing_after_caps:
        raise ValueError(f"{dataset}: qrels reference missing ids after caps")
    if not selected_queries or not selected_corpus or not selected_qrels:
        raise ValueError(f"{dataset}: caps/filtering selected an empty BEIR eval slice")

    out_dir = output_root / dataset
    corpus_count = write_jsonl(
        out_dir / "corpus.jsonl",
        ({"_id": row.id, "title": row.title, "text": row.text} for row in selected_corpus),
    )
    query_count = write_jsonl(
        out_dir / "queries.jsonl",
        ({"_id": row.id, "text": row.text} for row in selected_queries),
    )
    qrel_count = write_qrels(out_dir / "qrels" / f"{split_name}.tsv", selected_qrels)

    caveats = [
        "Adapter performs dataset conversion/acquisition only; it does not run retrieval evaluation.",
        "No model quality claim is implied by these files.",
        "Official benchmark proof requires actual eval rows produced by the long-context wedge or scoreboard harness.",
    ]
    if input_root is not None:
        caveats.append("Input came from local JSONL fixture/raw mode, not a live Hugging Face acquisition.")
    if context_length is not None and not context_applied:
        caveats.append("Requested context-length filter was not applied because rows exposed no recognized length signal.")

    manifest = {
        "schema": "eos.longembed.beir.v1",
        "source_dataset": SOURCE_DATASET,
        "source_config": dataset,
        "source_urls": SOURCE_URLS,
        "official_configs": list(OFFICIAL_CONFIGS),
        "official_splits": list(OFFICIAL_SPLITS),
        "synthetic_context_lengths": list(SYNTHETIC_CONTEXT_LENGTHS),
        "split_names": {
            "corpus": "corpus",
            "queries": "queries",
            "qrels_source": "qrels",
            "qrels_output": split_name,
        },
        "counts": {
            "corpus": corpus_count,
            "queries": query_count,
            "qrel_pairs": qrel_count,
            "source_corpus_rows": len(splits["corpus"]),
            "source_query_rows": len(splits["queries"]),
            "source_qrel_rows": len(splits["qrels"]),
        },
        "text_length_stats": {
            "corpus": text_stats(selected_corpus),
            "queries": text_stats(selected_queries),
        },
        "caps": {
            "max_docs": max_docs,
            "max_queries": max_queries,
            "preserve_qrels_relevant_docs": True,
        },
        "context_length_filter": {
            "requested": context_length,
            "applied_to_splits": context_applied,
        },
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "caveats": caveats,
        "warnings": warnings,
    }
    (out_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ConvertResult(dataset, out_dir, corpus_count, query_count, qrel_count, warnings)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_docs < 0 or args.max_queries < 0:
        raise SystemExit("--max-docs and --max-queries must be non-negative")
    if not args.split_name.strip():
        raise SystemExit("--split-name must be non-empty")
    try:
        datasets = parse_dataset_selection(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results: list[ConvertResult] = []
    for dataset in datasets:
        splits = load_splits(dataset, args.input_root)
        result = convert_dataset(
            dataset,
            splits,
            args.output_root,
            split_name=args.split_name,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
            context_length=args.context_length,
            input_root=args.input_root,
        )
        results.append(result)
        print(
            f"wrote {dataset}: corpus={result.corpus_count} "
            f"queries={result.query_count} qrels={result.qrel_count} to {result.output_dir}",
            flush=True,
        )
        for warning in result.warnings:
            print(f"warning[{dataset}]: {warning}", file=sys.stderr, flush=True)
    if not results:
        raise SystemExit("no datasets converted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
