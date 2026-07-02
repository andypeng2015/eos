#!/usr/bin/env python3
"""Build bounded research-only MS MARCO Passage Stage A hard-negative rows.

The builder consumes the BEIR-style MS MARCO Passage corpus produced by
``scripts/acquire_msmarco_passage.py`` or the earlier acquisition packet. It
emits text hard-negative JSONL compatible with
``runtime/embedding_hard_negative_dataset.go`` while preserving source,
provenance, split, legal, and leak-accounting metadata in extra fields.

Hard-negative mining (``--negative-mining``):

* ``random`` (default) preserves the original behavior byte-for-byte: hard
  negatives are a random draw from a reservoir-sampled corpus pool, excluding
  train/dev qrel positives.
* ``bm25`` mines negatives with an in-process BM25 index built over a
  reservoir-sampled corpus subsample (``--mining-pool-size``, default
  200000). The full 8.8M-passage corpus does not fit a naive in-memory
  Python BM25 index, and ``rank_bm25`` is not available in this
  environment, so this is a documented pool-subsample fallback: BM25 ranks
  are only as good as the docs captured in the subsample. Row provenance
  records the mode and pool size.
* ``teacher`` ranks negatives using an externally produced teacher score
  cache (the same JSONL schema written by
  ``scripts/score_teacher_with_vector_cache.py`` /
  ``scripts/run_stagea_bge_teacher_bridge.py``: rows with
  ``query_id``/``candidate_doc_id``/``candidate``/``score``/``role``).
  Requires ``--teacher-score-cache``.

``--false-negative-margin`` (only active when ``--teacher-score-cache`` is
supplied, for either ``bm25`` or ``teacher`` mining) drops any mined
candidate whose teacher score is >= the positive's teacher score minus the
margin, guarding against promoting an actually-relevant passage to a hard
negative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
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
NEGATIVE_MINING_MODES = ("random", "bm25", "teacher")
BM25_K1 = 0.9
BM25_B = 0.4
SOURCE_MIX_LABELS = {
    "random": "random-corpus-hard-negatives",
    "bm25": "bm25-mined-hard-negatives",
    "teacher": "teacher-mined-hard-negatives",
}
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


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
    parser.add_argument(
        "--negative-mining",
        choices=NEGATIVE_MINING_MODES,
        default="random",
        help=(
            "Hard-negative mining strategy. 'random' is the original random-corpus-sample "
            "behavior (default, byte-identical output). 'bm25' mines lexically similar hard "
            "negatives over a corpus subsample. 'teacher' ranks negatives from a teacher "
            "score cache."
        ),
    )
    parser.add_argument(
        "--mining-pool-size",
        type=int,
        default=200000,
        help=(
            "Corpus subsample size used to build the bm25 mining index (and as a random "
            "backfill pool for bm25/teacher mining). The full corpus does not fit an "
            "in-memory Python BM25 index; this bounds the mining pool."
        ),
    )
    parser.add_argument(
        "--teacher-score-cache",
        type=Path,
        default=None,
        help=(
            "Teacher score cache JSONL (score_teacher_with_vector_cache.py / "
            "run_stagea_bge_teacher_bridge.py schema) used to rank 'teacher' mode "
            "candidates and/or apply --false-negative-margin filtering to mined candidates."
        ),
    )
    parser.add_argument(
        "--false-negative-margin",
        type=float,
        default=0.0,
        help=(
            "Drop a mined negative candidate whose teacher score >= the positive's teacher "
            "score minus this margin. Only active when --teacher-score-cache is supplied."
        ),
    )
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


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.split(text.lower()) if token]


class BM25Index:
    """Minimal in-memory BM25 index (Lucene-style idf, k1/b as in runtime/retrieval_bm25.go).

    Built over a bounded document list (see ``--mining-pool-size``); not intended
    for the full 8.8M-passage corpus.
    """

    def __init__(self, docs: list[tuple[str, str]], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total_len = 0
        for doc_index, (doc_id, text) in enumerate(docs):
            tokens = tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            for term, count in Counter(tokens).items():
                self.postings[term].append((doc_index, count))
        self.n_docs = len(docs)
        self.avg_len = (total_len / self.n_docs) if self.n_docs else 0.0
        self.doc_freq = {term: len(postings) for term, postings in self.postings.items()}

    def rank(self, query_text: str) -> list[tuple[str, float]]:
        if not self.n_docs:
            return []
        terms = tokenize(query_text)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        # Weight each term's contribution by its query term frequency (qtf),
        # matching runtime/retrieval_bm25.go's scoreBM25Document (score +=
        # qtf * idf * tfWeight), rather than treating every unique query term
        # as contributing once regardless of how often it appears.
        for term, qtf in Counter(terms).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.doc_freq[term]
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for doc_index, tf in postings:
                doc_len = self.doc_len[doc_index] or 1
                length_norm = (1 - self.b + self.b * doc_len / self.avg_len) if self.avg_len else 1.0
                denom = tf + self.k1 * length_norm
                if denom <= 0:
                    continue
                scores[doc_index] += qtf * idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[doc_index], score) for doc_index, score in ranked]


def resolve_query_id_for_text(query_text_value: str, text_to_qid: dict[str, str]) -> str:
    return text_to_qid.get(normalize_for_leak(query_text_value), "")


def load_teacher_score_cache(
    path: Path, queries: dict[str, str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], float], dict[str, Any]]:
    """Load a teacher score cache JSONL into per-query candidate lists and a
    ``(query_id, doc_id) -> score`` lookup.

    Compatible with ``scripts/score_teacher_with_vector_cache.py`` scores_jsonl
    rows (``query_id``/``candidate_doc_id``/``candidate``/``score``/``role``)
    and the row-level ``teacher_scores`` schema consumed by
    ``scripts/build_retrieval_teacher_guide_filter.py``.
    """
    text_to_qid = {normalize_for_leak(text): qid for qid, text in queries.items()}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query_doc: dict[tuple[str, str], float] = {}
    # Tracks the index of each (qid, doc_id)'s entry within by_query[qid] so a
    # higher-scoring duplicate can replace it in place instead of leaving a
    # stale lower-scoring duplicate entry alongside the new one.
    entry_index: dict[tuple[str, str], int] = {}
    counts: Counter[str] = Counter()

    def record(qid: str, doc_id: str, text: str, score: float, role: str) -> None:
        if not qid or not doc_id:
            return
        key = (qid, doc_id)
        existing = by_query_doc.get(key)
        entry = {"doc_id": doc_id, "text": text, "score": score, "role": role}
        if existing is not None:
            counts["duplicate_query_doc_scores"] += 1
            if score <= existing:
                return
            by_query_doc[key] = score
            by_query[qid][entry_index[key]] = entry
            return
        by_query_doc[key] = score
        entry_index[key] = len(by_query[qid])
        by_query[qid].append(entry)
        counts["candidate_scores"] += 1

    for line_number, row in iter_jsonl(path):
        counts["rows_seen"] += 1
        raw_scores = row.get("teacher_scores") or row.get("scores")
        if isinstance(raw_scores, list):
            qid = str(row.get("query_id") or "").strip() or resolve_query_id_for_text(
                str(row.get("query") or ""), text_to_qid
            )
            if not qid:
                counts["unresolved_query"] += 1
                continue
            doc_ids = [str(row.get("positive_doc_id") or "")] + [
                str(value or "") for value in row.get("negative_doc_ids") or []
            ]
            texts = [str(row.get("positive") or "")] + [str(value or "") for value in row.get("negatives") or []]
            for index, raw_score in enumerate(raw_scores):
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score):
                    continue
                doc_id = doc_ids[index] if index < len(doc_ids) else ""
                text = stable_text(texts[index] if index < len(texts) else "")
                role = "positive" if index == 0 else "negative"
                record(qid, doc_id, text, score, role)
            counts["row_level_examples"] += 1
            continue

        raw_score = row.get("score", row.get("teacher_score"))
        if raw_score is None:
            counts["rows_without_score"] += 1
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            counts["rows_without_score"] += 1
            continue
        if not math.isfinite(score):
            counts["rows_without_score"] += 1
            continue
        qid = str(row.get("query_id") or "").strip() or resolve_query_id_for_text(
            str(row.get("query") or ""), text_to_qid
        )
        if not qid:
            counts["unresolved_query"] += 1
            continue
        doc_id = str(row.get("candidate_doc_id") or row.get("doc_id") or "").strip()
        if not doc_id:
            counts["missing_candidate_doc_id"] += 1
            continue
        text = stable_text(row.get("candidate"))
        role = str(row.get("role") or "").strip().lower()
        record(qid, doc_id, text, score, role)

    counts["queries_with_cache_scores"] = len(by_query)
    counts["query_doc_pairs"] = len(by_query_doc)
    return dict(by_query), by_query_doc, dict(counts)


def select_mined_negatives(
    qid: str,
    positive_norm: str,
    ranked_candidates: list[tuple[str, str]],
    backfill_candidates: list[tuple[str, str]],
    same_query_positive_doc_ids: set[str],
    dev_positive_doc_ids: set[str],
    train_positive_doc_ids: set[str],
    teacher_cache_by_query_doc: dict[tuple[str, str], float],
    positive_teacher_score: float | None,
    false_negative_margin: float,
    negatives_per_row: int,
) -> tuple[list[str], list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    negative_doc_ids: list[str] = []
    negatives: list[str] = []
    seen: set[str] = set()

    def try_candidate(doc_id: str, text: str, source_label: str) -> None:
        if doc_id in seen:
            counts["duplicate_negative_doc_id_drops"] += 1
            return
        seen.add(doc_id)
        if doc_id in same_query_positive_doc_ids:
            counts["same_query_positive_negative_drops"] += 1
            return
        if doc_id in dev_positive_doc_ids:
            counts["dev_positive_negative_drops"] += 1
            return
        if doc_id in train_positive_doc_ids:
            counts["cross_query_train_positive_negative_drops"] += 1
            return
        if not text:
            counts["missing_candidate_text_drops"] += 1
            return
        if normalize_for_leak(text) == positive_norm:
            counts["duplicate_positive_text_negative_drops"] += 1
            return
        if positive_teacher_score is not None:
            candidate_score = teacher_cache_by_query_doc.get((qid, doc_id))
            if candidate_score is not None and candidate_score >= positive_teacher_score - false_negative_margin:
                counts["false_negative_margin_drops"] += 1
                return
        negatives.append(text)
        negative_doc_ids.append(doc_id)
        counts[f"{source_label}_negatives_used"] += 1

    for doc_id, text in ranked_candidates:
        if len(negatives) >= negatives_per_row:
            break
        try_candidate(doc_id, text, "mined_primary")
    if len(negatives) < negatives_per_row:
        for doc_id, text in backfill_candidates:
            if len(negatives) >= negatives_per_row:
                break
            try_candidate(doc_id, text, "mined_backfill")
    return negative_doc_ids, negatives, counts


def build_rows_mined(
    selected_pairs: list[tuple[str, str]],
    queries: dict[str, str],
    positive_texts: dict[str, str],
    negative_pool: list[tuple[str, str]],
    positives_by_query: dict[str, set[str]],
    dev_positive_doc_ids: set[str],
    train_positive_doc_ids: set[str],
    split: str,
    negatives_per_row: int,
    rng: random.Random,
    drop_sample_limit: int,
    mode: str,
    bm25_index: "BM25Index | None",
    teacher_cache_by_query: dict[str, list[dict[str, Any]]],
    teacher_cache_by_query_doc: dict[tuple[str, str], float],
    false_negative_margin: float,
    mining_pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    drop_samples: list[dict[str, Any]] = []
    pool_text_by_id = {doc_id: text for doc_id, text in negative_pool}
    shuffled_backfill = list(negative_pool)
    rng.shuffle(shuffled_backfill)
    source_label = f"msmarco-passage/{split}/qrels/{SOURCE_MIX_LABELS[mode]}"
    queries_with_ranked_candidates = 0
    queries_without_ranked_candidates = 0

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

        if mode == "bm25":
            ranked_ids = bm25_index.rank(query) if bm25_index is not None else []
            ranked_candidates = [(doc_id, pool_text_by_id.get(doc_id, "")) for doc_id, _score in ranked_ids]
        else:
            cache_candidates = teacher_cache_by_query.get(qid, [])
            ranked_sorted = sorted(cache_candidates, key=lambda item: (-item["score"], item["doc_id"]))
            ranked_candidates = [(item["doc_id"], item["text"]) for item in ranked_sorted]

        if ranked_candidates:
            queries_with_ranked_candidates += 1
        else:
            queries_without_ranked_candidates += 1

        positive_teacher_score = teacher_cache_by_query_doc.get((qid, positive_doc_id))

        negative_doc_ids, negatives, select_counts = select_mined_negatives(
            qid=qid,
            positive_norm=positive_norm,
            ranked_candidates=ranked_candidates,
            backfill_candidates=shuffled_backfill,
            same_query_positive_doc_ids=positives_by_query.get(qid, set()),
            dev_positive_doc_ids=dev_positive_doc_ids,
            train_positive_doc_ids=train_positive_doc_ids,
            teacher_cache_by_query_doc=teacher_cache_by_query_doc,
            positive_teacher_score=positive_teacher_score,
            false_negative_margin=false_negative_margin,
            negatives_per_row=negatives_per_row,
        )
        counts.update(select_counts)

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

        if mode == "bm25":
            negative_source = (
                f"BM25 top-k over a {mining_pool_size}-document corpus subsample "
                "excluding train/dev qrel positives"
            )
        else:
            negative_source = (
                "teacher-score-cache-ranked candidates excluding train/dev qrel positives"
            )
        row = {
            "schema": ROW_SCHEMA,
            "row_id": make_row_id(split, qid, positive_doc_id, negative_doc_ids),
            "source": source_label,
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
                "negative_source": negative_source,
                "negative_mining": {
                    "mode": mode,
                    "mining_pool_size": mining_pool_size if mode == "bm25" else None,
                    "bm25_k1": BM25_K1 if mode == "bm25" else None,
                    "bm25_b": BM25_B if mode == "bm25" else None,
                    "teacher_score_cache_applied": bool(teacher_cache_by_query_doc),
                    "false_negative_margin": false_negative_margin if teacher_cache_by_query_doc else None,
                },
            },
        }
        rows.append(row)
    counts["rows_emitted"] = len(rows)
    mining_report = {
        "mode": mode,
        "mining_pool_size": mining_pool_size,
        "queries_with_ranked_candidates": queries_with_ranked_candidates,
        "queries_without_ranked_candidates": queries_without_ranked_candidates,
        "bm25_k1": BM25_K1 if mode == "bm25" else None,
        "bm25_b": BM25_B if mode == "bm25" else None,
        "false_negative_margin": false_negative_margin,
        "false_negative_margin_active": bool(teacher_cache_by_query_doc),
    }
    return rows, {"counts": dict(counts), "drop_samples": drop_samples}, mining_report


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
    if args.mining_pool_size <= 0:
        raise SystemExit("--mining-pool-size must be positive")
    if args.negative_mining != "random" and args.mining_pool_size < args.negatives_per_query:
        raise SystemExit("--mining-pool-size must be >= --negatives-per-query")
    if args.negative_mining == "teacher" and args.teacher_score_cache is None:
        raise SystemExit("--negative-mining teacher requires --teacher-score-cache")
    if args.negative_mining == "random" and args.teacher_score_cache is not None:
        raise SystemExit("--teacher-score-cache requires --negative-mining bm25 or teacher")
    if not math.isfinite(args.false_negative_margin):
        raise SystemExit("--false-negative-margin must be finite")
    if args.negative_mining == "random" and args.false_negative_margin > 0:
        raise SystemExit("--false-negative-margin requires --negative-mining bm25 or teacher")
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
    if args.teacher_score_cache is not None and not args.teacher_score_cache.exists():
        raise SystemExit(f"--teacher-score-cache missing: {args.teacher_score_cache}")

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

    negative_mining = args.negative_mining
    pool_size = args.candidate_pool_size if negative_mining == "random" else args.mining_pool_size
    positive_texts, negative_pool, corpus_counts = load_needed_corpus_and_negative_pool(
        corpus_path,
        selected_doc_ids,
        positives_by_query,
        dev_positive_doc_ids,
        pool_size,
        args.max_corpus_docs,
        rng,
    )

    if negative_mining == "random":
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
        mining_report: dict[str, Any] = {"mode": "random", "teacher_score_cache": None}
    else:
        train_positive_doc_ids = {doc_id for doc_ids in positives_by_query.values() for doc_id in doc_ids}
        teacher_cache_by_query: dict[str, list[dict[str, Any]]] = {}
        teacher_cache_by_query_doc: dict[tuple[str, str], float] = {}
        teacher_cache_counts: dict[str, Any] = {}
        if args.teacher_score_cache is not None:
            teacher_cache_by_query, teacher_cache_by_query_doc, teacher_cache_counts = load_teacher_score_cache(
                args.teacher_score_cache, queries
            )
        bm25_index = BM25Index(negative_pool) if negative_mining == "bm25" else None
        rows, build_report, mining_report = build_rows_mined(
            selected_pairs,
            queries,
            positive_texts,
            negative_pool,
            positives_by_query,
            dev_positive_doc_ids,
            train_positive_doc_ids,
            split,
            args.negatives_per_query,
            rng,
            args.drop_sample_limit,
            negative_mining,
            bm25_index,
            teacher_cache_by_query,
            teacher_cache_by_query_doc,
            args.false_negative_margin,
            args.mining_pool_size,
        )
        mining_report["teacher_score_cache"] = (
            {"path": str(args.teacher_score_cache), "counts": teacher_cache_counts}
            if args.teacher_score_cache is not None
            else None
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
            "negative_mining": negative_mining,
            "mining_pool_size": args.mining_pool_size,
            "mining_pool_size_applied": negative_mining != "random",
            "teacher_score_cache": str(args.teacher_score_cache) if args.teacher_score_cache else None,
            "false_negative_margin": args.false_negative_margin,
            "false_negative_margin_applied": args.teacher_score_cache is not None,
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
            "source_mix": {f"msmarco-passage/{split}/qrels/{SOURCE_MIX_LABELS[negative_mining]}": len(rows)},
            "selection": dict(selection_counts),
            "corpus_scan": dict(corpus_counts),
            "split_dev_positive_doc_overlap": split_dev_positive_overlap,
        },
        "mining": mining_report,
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
