#!/usr/bin/env python3
"""Tests for bounded MS MARCO Passage Stage A row builder."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_msmarco_stagea_rows.py"

_spec = importlib.util.spec_from_file_location("build_msmarco_stagea_rows", SCRIPT)
bms = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(bms)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BuildMSMarcoStageARowsTest(unittest.TestCase):
    def test_builder_emits_reader_compatible_rows_and_leak_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            out = root / "out"
            write_jsonl(
                corpus_root / "queries.jsonl",
                [
                    {"_id": "q1", "text": "query one"},
                    {"_id": "q2", "text": "query two"},
                ],
            )
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "p1", "title": "", "text": "positive one"},
                    {"_id": "p2", "title": "", "text": "positive two"},
                    {"_id": "n1", "title": "", "text": "negative one"},
                    {"_id": "n2", "title": "", "text": "negative two"},
                    {"_id": "n3", "title": "", "text": "negative three"},
                    {"_id": "ddev", "title": "", "text": "dev positive"},
                ],
            )
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1), ("q2", "p2", 1)])
            write_qrels(corpus_root / "qrels" / "dev.tsv", [("qd", "ddev", 1)])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(out),
                    "--max-rows",
                    "2",
                    "--negatives-per-query",
                    "2",
                    "--candidate-pool-size",
                    "4",
                    "--max-corpus-docs",
                    "6",
                    "--seed",
                    "7",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            leak = json.loads((out / "reports" / "leak-report.json").read_text(encoding="utf-8"))
            rows_path = out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl"
            rows = read_rows(rows_path)
            first_output = rows_path.read_text(encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(root / "out-repeat"),
                    "--max-rows",
                    "2",
                    "--negatives-per-query",
                    "2",
                    "--candidate-pool-size",
                    "4",
                    "--max-corpus-docs",
                    "6",
                    "--seed",
                    "7",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            repeated_output = (
                root
                / "out-repeat"
                / "artifacts"
                / "msmarco-passage.stagea.train-hard-negatives.jsonl"
            ).read_text(encoding="utf-8")

        self.assertIn("manifest", completed.stdout)
        self.assertEqual(manifest["schema"], "eos.msmarco_passage_stagea_rows.v1")
        self.assertEqual(manifest["legal_gates"]["release_train_allowed"], False)
        self.assertEqual(manifest["legal_gates"]["commercial_use_allowed"], False)
        self.assertEqual(manifest["legal_gates"]["train_allowed_for_research"], True)
        self.assertEqual(manifest["legal_gates"]["test_rows_train_allowed"], False)
        self.assertEqual(manifest["builder_args"]["split"], "train")
        self.assertEqual(manifest["builder_args"]["max_corpus_docs"], 6)
        self.assertEqual(manifest["counts"]["qrels_seen"]["positive_rows"], 2)
        self.assertEqual(manifest["counts"]["rows_emitted"], 2)
        self.assertEqual(manifest["counts"]["negative_candidates_emitted"], 4)
        self.assertEqual(manifest["counts"]["negative_candidates_considered"], 3)
        self.assertEqual(leak["status"], "passed")
        self.assertEqual(leak["validation"]["counts"]["rows_checked"], 2)
        self.assertEqual(first_output, repeated_output)
        for row in rows:
            self.assertIn("query", row)
            self.assertIn("positive", row)
            self.assertIn("negatives", row)
            self.assertEqual(row["roles"]["query"], "query")
            self.assertEqual(row["roles"]["positive"], "document")
            self.assertEqual(row["legal_gates"], manifest["legal_gates"])
            self.assertFalse(row["split_policy"]["test_rows_train_allowed"])
            self.assertNotIn(row["positive_doc_id"], row["negative_doc_ids"])
            self.assertNotIn("ddev", row["negative_doc_ids"])

    def test_builder_accounts_missing_query_and_positive_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            out = root / "out"
            write_jsonl(corpus_root / "queries.jsonl", [{"_id": "q1", "text": "query one"}])
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "n1", "title": "", "text": "negative one"},
                    {"_id": "n2", "title": "", "text": "negative two"},
                    {"_id": "p2", "title": "", "text": "positive two"},
                ],
            )
            write_qrels(
                corpus_root / "qrels" / "train.tsv",
                [
                    ("q-missing", "p0", 1),
                    ("q1", "p-missing", 1),
                    ("q1", "p2", 1),
                    ("q1", "n1", 0),
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(out),
                    "--max-rows",
                    "3",
                    "--negatives-per-query",
                    "1",
                    "--candidate-pool-size",
                    "2",
                    "--seed",
                    "11",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            rows = read_rows(out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl")

        self.assertEqual(len(rows), 1)
        self.assertEqual(manifest["counts"]["qrels_seen"]["positive_rows"], 3)
        self.assertEqual(manifest["counts"]["qrels_seen"]["ignored_nonpositive_or_malformed"], 1)
        self.assertEqual(manifest["counts"]["missing_query_skips"], 1)
        self.assertEqual(manifest["counts"]["missing_doc_skips"], 1)
        self.assertEqual(manifest["counts"]["rows_emitted"], 1)

    def test_negative_mining_random_default_is_explicit_and_unchanged(self) -> None:
        # Regression guard: passing --negative-mining random explicitly must match
        # the implicit default byte-for-byte, and the source/provenance labels
        # used by the original random-corpus-sample behavior must not move.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            write_jsonl(
                corpus_root / "queries.jsonl",
                [{"_id": "q1", "text": "query one"}, {"_id": "q2", "text": "query two"}],
            )
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "p1", "title": "", "text": "positive one"},
                    {"_id": "p2", "title": "", "text": "positive two"},
                    {"_id": "n1", "title": "", "text": "negative one"},
                    {"_id": "n2", "title": "", "text": "negative two"},
                    {"_id": "n3", "title": "", "text": "negative three"},
                ],
            )
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1), ("q2", "p2", 1)])

            def run(out: Path, extra_args: list[str]) -> None:
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--corpus-root",
                        str(corpus_root),
                        "--acquisition-run-root",
                        str(root / "missing-acquisition"),
                        "--output-root",
                        str(out),
                        "--max-rows",
                        "2",
                        "--negatives-per-query",
                        "2",
                        "--candidate-pool-size",
                        "4",
                        "--seed",
                        "7",
                        *extra_args,
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                )

            run(root / "out-implicit", [])
            run(root / "out-explicit", ["--negative-mining", "random"])

            implicit_rows = (
                root / "out-implicit" / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl"
            ).read_text(encoding="utf-8")
            explicit_rows = (
                root / "out-explicit" / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl"
            ).read_text(encoding="utf-8")
            manifest = json.loads((root / "out-implicit" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(implicit_rows, explicit_rows)
        self.assertEqual(manifest["builder_args"]["negative_mining"], "random")
        self.assertIn("msmarco-passage/train/qrels/random-corpus-hard-negatives", manifest["counts"]["source_mix"])
        self.assertEqual(manifest["mining"]["mode"], "random")

    def test_bm25_mining_prefers_topically_related_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            out = root / "out"
            write_jsonl(
                corpus_root / "queries.jsonl",
                [{"_id": "q1", "text": "electric vehicle battery range"}],
            )
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "p1", "title": "", "text": "long trips need more battery capacity for driving"},
                    {
                        "_id": "hard1",
                        "title": "",
                        "text": "electric vehicle battery chemistry and range anxiety solutions",
                    },
                    {
                        "_id": "hard2",
                        "title": "",
                        "text": "battery electric vehicle charging network expands range",
                    },
                    {"_id": "unrelated1", "title": "", "text": "medieval castle architecture history"},
                    {"_id": "unrelated2", "title": "", "text": "recipe for chocolate cake baking"},
                    {"_id": "unrelated3", "title": "", "text": "stock market analysis quarterly report"},
                ],
            )
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1)])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(out),
                    "--max-rows",
                    "1",
                    "--negatives-per-query",
                    "2",
                    "--candidate-pool-size",
                    "5",
                    "--mining-pool-size",
                    "5",
                    "--seed",
                    "7",
                    "--negative-mining",
                    "bm25",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            rows = read_rows(out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], "msmarco-passage/train/qrels/bm25-mined-hard-negatives")
        self.assertEqual(set(row["negative_doc_ids"]), {"hard1", "hard2"})
        self.assertEqual(row["provenance"]["negative_mining"]["mode"], "bm25")
        self.assertEqual(row["provenance"]["negative_mining"]["mining_pool_size"], 5)
        self.assertEqual(manifest["builder_args"]["negative_mining"], "bm25")
        self.assertEqual(manifest["mining"]["mode"], "bm25")
        self.assertGreaterEqual(manifest["mining"]["queries_with_ranked_candidates"], 1)

    def test_teacher_mining_ranks_by_score_and_drops_false_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            out = root / "out"
            write_jsonl(corpus_root / "queries.jsonl", [{"_id": "q1", "text": "query one"}])
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [{"_id": "p1", "title": "", "text": "positive one"}],
            )
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1)])

            teacher_cache = root / "teacher-scores.jsonl"
            cache_rows = [
                {"query_id": "q1", "candidate_doc_id": "p1", "candidate": "positive one", "score": 0.9, "role": "positive"},
                {"query_id": "q1", "candidate_doc_id": "docA", "candidate": "doc a text", "score": 0.5, "role": "negative"},
                {"query_id": "q1", "candidate_doc_id": "docB", "candidate": "doc b text", "score": 0.3, "role": "negative"},
                {
                    "query_id": "q1",
                    "candidate_doc_id": "docC",
                    "candidate": "doc c near-duplicate text",
                    "score": 0.88,
                    "role": "negative",
                },
                {"query_id": "q1", "candidate_doc_id": "docD", "candidate": "doc d text", "score": 0.6, "role": "negative"},
            ]
            write_jsonl(teacher_cache, cache_rows)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(out),
                    "--max-rows",
                    "1",
                    "--negatives-per-query",
                    "2",
                    "--candidate-pool-size",
                    "2",
                    "--mining-pool-size",
                    "2",
                    "--seed",
                    "7",
                    "--negative-mining",
                    "teacher",
                    "--teacher-score-cache",
                    str(teacher_cache),
                    "--false-negative-margin",
                    "0.05",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            leak = json.loads((out / "reports" / "leak-report.json").read_text(encoding="utf-8"))
            rows = read_rows(out / "artifacts" / "msmarco-passage.stagea.train-hard-negatives.jsonl")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], "msmarco-passage/train/qrels/teacher-mined-hard-negatives")
        # docC (score 0.88) is within the false-negative margin of the positive's
        # score (0.9 - 0.05 = 0.85) and must be dropped; docD/docA (the next
        # highest-scored candidates) are kept, in descending-score order.
        self.assertNotIn("docC", row["negative_doc_ids"])
        self.assertEqual(row["negative_doc_ids"], ["docD", "docA"])
        self.assertEqual(leak["build_drops"]["counts"].get("false_negative_margin_drops"), 1)
        self.assertEqual(manifest["mining"]["mode"], "teacher")
        self.assertTrue(manifest["mining"]["false_negative_margin_active"])
        self.assertEqual(manifest["builder_args"]["false_negative_margin"], 0.05)

    def test_teacher_mining_requires_score_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            write_jsonl(corpus_root / "queries.jsonl", [{"_id": "q1", "text": "query one"}])
            write_jsonl(corpus_root / "corpus.jsonl", [{"_id": "p1", "title": "", "text": "positive one"}])
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1)])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(root / "out"),
                    "--negative-mining",
                    "teacher",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--teacher-score-cache", completed.stderr)

    def test_false_negative_margin_requires_non_random_mining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            write_jsonl(corpus_root / "queries.jsonl", [{"_id": "q1", "text": "query one"}])
            write_jsonl(corpus_root / "corpus.jsonl", [{"_id": "p1", "title": "", "text": "positive one"}])
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1)])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(root / "out"),
                    "--negative-mining",
                    "random",
                    "--false-negative-margin",
                    "0.1",
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--false-negative-margin", completed.stderr)
        self.assertIn("--negative-mining", completed.stderr)

    def test_false_negative_margin_zero_allowed_with_random_mining(self) -> None:
        # The default --false-negative-margin is 0.0 and must not be rejected
        # for the (also default) random mining mode.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_root = root / "beir"
            write_jsonl(corpus_root / "queries.jsonl", [{"_id": "q1", "text": "query one"}])
            write_jsonl(
                corpus_root / "corpus.jsonl",
                [
                    {"_id": "p1", "title": "", "text": "positive one"},
                    {"_id": "n1", "title": "", "text": "negative one"},
                ],
            )
            write_qrels(corpus_root / "qrels" / "train.tsv", [("q1", "p1", 1)])

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--corpus-root",
                    str(corpus_root),
                    "--acquisition-run-root",
                    str(root / "missing-acquisition"),
                    "--output-root",
                    str(root / "out"),
                    "--negatives-per-query",
                    "1",
                    "--candidate-pool-size",
                    "1",
                    "--negative-mining",
                    "random",
                    "--false-negative-margin",
                    "0.0",
                ],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)


class BM25IndexRankTest(unittest.TestCase):
    def test_rank_weights_by_query_term_frequency(self) -> None:
        # runtime/retrieval_bm25.go's scoreBM25Document weights each query
        # term's contribution by its query term frequency (qtf); BM25Index.rank
        # must match. Repeat "battery" in the query so the doc containing
        # "battery" (matched twice by qtf) outranks a doc that only matches
        # each of two distinct single-occurrence terms once.
        docs = [
            ("repeats_query_term", "battery battery capacity"),
            ("distinct_terms", "battery range"),
        ]
        index = bms.BM25Index(docs)
        ranked = index.rank("battery battery")
        self.assertEqual(ranked[0][0], "repeats_query_term")
        # Sanity: with qtf-weighting active, matching "battery" with qtf=2
        # scores strictly higher than a single-occurrence match to the same
        # term set on an otherwise-identical document (verifies the weighting
        # is actually applied, not just tie-broken by doc_id ordering).
        single_index = bms.BM25Index([("single", "battery capacity")])
        single_score = single_index.rank("battery")[0][1]
        doubled_index = bms.BM25Index([("single", "battery capacity")])
        doubled_score = doubled_index.rank("battery battery")[0][1]
        self.assertGreater(doubled_score, single_score)
        self.assertAlmostEqual(doubled_score, 2 * single_score)

    def test_rank_empty_query_or_index_returns_empty(self) -> None:
        index = bms.BM25Index([("d1", "some text")])
        self.assertEqual(index.rank(""), [])
        empty_index = bms.BM25Index([])
        self.assertEqual(empty_index.rank("some query"), [])


class LoadTeacherScoreCacheTest(unittest.TestCase):
    def test_duplicate_query_doc_pair_replaces_stale_lower_score_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "teacher-scores.jsonl"
            write_jsonl(
                cache_path,
                [
                    {"query_id": "q1", "candidate_doc_id": "docA", "candidate": "doc a v1", "score": 0.2, "role": "negative"},
                    # Higher-scoring duplicate for the same (q1, docA) pair: must
                    # REPLACE the first entry in by_query["q1"], not sit
                    # alongside it as a second, stale duplicate.
                    {"query_id": "q1", "candidate_doc_id": "docA", "candidate": "doc a v2", "score": 0.7, "role": "negative"},
                ],
            )
            queries = {"q1": "query one"}
            by_query, by_query_doc, counts = bms.load_teacher_score_cache(cache_path, queries)

        self.assertEqual(len(by_query["q1"]), 1, by_query["q1"])
        entry = by_query["q1"][0]
        self.assertEqual(entry["doc_id"], "docA")
        self.assertEqual(entry["score"], 0.7)
        self.assertEqual(entry["text"], "doc a v2")
        self.assertEqual(by_query_doc[("q1", "docA")], 0.7)
        self.assertEqual(counts["duplicate_query_doc_scores"], 1)
        self.assertEqual(counts["candidate_scores"], 1)

    def test_duplicate_query_doc_pair_keeps_higher_score_when_seen_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "teacher-scores.jsonl"
            write_jsonl(
                cache_path,
                [
                    {"query_id": "q1", "candidate_doc_id": "docA", "candidate": "doc a v1", "score": 0.7, "role": "negative"},
                    {"query_id": "q1", "candidate_doc_id": "docA", "candidate": "doc a v2", "score": 0.2, "role": "negative"},
                ],
            )
            queries = {"q1": "query one"}
            by_query, by_query_doc, counts = bms.load_teacher_score_cache(cache_path, queries)

        self.assertEqual(len(by_query["q1"]), 1, by_query["q1"])
        entry = by_query["q1"][0]
        self.assertEqual(entry["score"], 0.7)
        self.assertEqual(entry["text"], "doc a v1")
        self.assertEqual(by_query_doc[("q1", "docA")], 0.7)

    def test_row_level_teacher_scores_list_ingestion(self) -> None:
        # build_retrieval_teacher_guide_filter.py-style schema: a row per
        # hard-negative example, with a "teacher_scores" list parallel to
        # [positive] + negatives, and query/positive/negative doc ids/texts
        # instead of a flat candidate/score-per-row cache schema.
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "row-level-teacher-scores.jsonl"
            write_jsonl(
                cache_path,
                [
                    {
                        "query_id": "q1",
                        "positive_doc_id": "p1",
                        "negative_doc_ids": ["n1", "n2"],
                        "query": "query one",
                        "positive": "positive text",
                        "negatives": ["negative one text", "negative two text"],
                        "teacher_scores": [0.9, 0.4, 0.1],
                    },
                ],
            )
            queries = {"q1": "query one"}
            by_query, by_query_doc, counts = bms.load_teacher_score_cache(cache_path, queries)

        self.assertEqual(by_query_doc[("q1", "p1")], 0.9)
        self.assertEqual(by_query_doc[("q1", "n1")], 0.4)
        self.assertEqual(by_query_doc[("q1", "n2")], 0.1)
        entries_by_doc = {entry["doc_id"]: entry for entry in by_query["q1"]}
        self.assertEqual(entries_by_doc["p1"]["role"], "positive")
        self.assertEqual(entries_by_doc["p1"]["text"], "positive text")
        self.assertEqual(entries_by_doc["n1"]["role"], "negative")
        self.assertEqual(entries_by_doc["n1"]["text"], "negative one text")
        self.assertEqual(entries_by_doc["n2"]["role"], "negative")
        self.assertEqual(counts["row_level_examples"], 1)
        self.assertEqual(counts["candidate_scores"], 3)

    def test_row_level_teacher_scores_resolves_query_id_from_query_text(self) -> None:
        # Row-level rows may omit query_id and rely on matching the "query"
        # text back to a known query_id (resolve_query_id_for_text).
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "row-level-teacher-scores.jsonl"
            write_jsonl(
                cache_path,
                [
                    {
                        "positive_doc_id": "p1",
                        "negative_doc_ids": ["n1"],
                        "query": "Query One",
                        "positive": "positive text",
                        "negatives": ["negative one text"],
                        "teacher_scores": [0.8, 0.3],
                    },
                ],
            )
            queries = {"q1": "query one"}
            by_query, by_query_doc, counts = bms.load_teacher_score_cache(cache_path, queries)

        self.assertEqual(by_query_doc[("q1", "p1")], 0.8)
        self.assertEqual(by_query_doc[("q1", "n1")], 0.3)
        self.assertEqual(counts.get("unresolved_query", 0), 0)


if __name__ == "__main__":
    unittest.main()
