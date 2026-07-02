#!/usr/bin/env python3
"""Dependency-free tests for the vector-distillation row builder."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_vector_distill_rows as bvdr  # noqa: E402


class MonkeypatchIsolatedTestCase(unittest.TestCase):
    """Base class that snapshots/restores every module-level callable this
    test suite monkeypatches (`embed_texts`, `run_shard`), so a stub left in
    place by one test can never leak into the next test's module state."""

    def setUp(self) -> None:
        super().setUp()
        self._orig_embed_texts = bvdr.embed_texts
        self._orig_run_shard = bvdr.run_shard

    def tearDown(self) -> None:
        bvdr.embed_texts = self._orig_embed_texts
        bvdr.run_shard = self._orig_run_shard
        super().tearDown()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def fake_embed_factory(dim: int = 4):
    """A deterministic stand-in for `embed_texts`: every row gets a vector
    derived from its id (stable across calls), and role/prefix are recorded
    on the returned report so tests can assert the pipeline routed the right
    prefix to the right bucket."""

    def fake_embed(rows, role, prefix, args, work_root):
        vectors = {}
        for row in rows:
            seed = sum(row.id.encode("utf-8"))
            vectors[row.id] = [float((seed + i) % 97) / 97.0 for i in range(dim)]
        report = bvdr.EmbedBucketReport(
            role=role,
            prefix=prefix,
            requested=len(rows),
            embedded=len(rows),
            skipped_for_time_budget=0,
            wall_clock_seconds=0.0,
            shards=1,
        )
        return report, vectors

    return fake_embed


class ReadInputRowsTest(unittest.TestCase):
    def test_reads_jsonl_queries_and_documents_with_flexible_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            queries = tmp / "queries.jsonl"
            documents = tmp / "documents.jsonl"
            write_jsonl(queries, [{"query_id": "q1", "query": "hello world"}])
            write_jsonl(documents, [{"doc_id": "d1", "document": "a document"}])

            q_rows = bvdr.read_id_text_rows(queries, bvdr.ROLE_QUERY)
            d_rows = bvdr.read_id_text_rows(documents, bvdr.ROLE_DOCUMENT)

        self.assertEqual(q_rows, [bvdr.Row(id="q1", text="hello world", role="query")])
        self.assertEqual(d_rows, [bvdr.Row(id="d1", text="a document", role="document")])

    def test_reads_tsv_queries_with_and_without_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with_header = tmp / "with_header.tsv"
            with_header.write_text("id\ttext\nq1\thello\nq2\tworld\n", encoding="utf-8")
            no_header = tmp / "no_header.tsv"
            no_header.write_text("q1\thello\nq2\tworld\n", encoding="utf-8")

            rows_h = bvdr.read_id_text_rows(with_header, bvdr.ROLE_QUERY)
            rows_n = bvdr.read_id_text_rows(no_header, bvdr.ROLE_QUERY)

        self.assertEqual([r.id for r in rows_h], ["q1", "q2"])
        self.assertEqual([r.id for r in rows_n], ["q1", "q2"])
        self.assertEqual(rows_h, rows_n)

    def test_reads_mixed_input_with_kind_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mixed = tmp / "mixed.jsonl"
            write_jsonl(
                mixed,
                [
                    {"id": "q1", "text": "a query", "kind": "query"},
                    {"id": "d1", "text": "a doc", "kind": "document"},
                    {"id": "r1", "text": "raw text", "kind": "raw"},
                ],
            )
            rows = bvdr.read_mixed_rows(mixed)
        self.assertEqual([(r.id, r.role) for r in rows], [("q1", "query"), ("d1", "document"), ("r1", "raw")])

    def test_mixed_rejects_unknown_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mixed = tmp / "mixed.jsonl"
            write_jsonl(mixed, [{"id": "x1", "text": "x", "kind": "bogus"}])
            with self.assertRaises(SystemExit):
                bvdr.read_mixed_rows(mixed)

    def test_dedupe_rows_drops_duplicate_role_id_pairs(self) -> None:
        rows = [
            bvdr.Row(id="a", text="1", role="query"),
            bvdr.Row(id="a", text="2", role="query"),
            bvdr.Row(id="a", text="3", role="document"),
        ]
        unique, dup_count = bvdr.dedupe_rows(rows)
        self.assertEqual(dup_count, 1)
        self.assertEqual([(r.id, r.role, r.text) for r in unique], [("a", "query", "1"), ("a", "document", "3")])


class GroupOrderTest(unittest.TestCase):
    def test_group_order_interleaves_query_with_positive_and_negatives(self) -> None:
        rows = [
            bvdr.Row(id="q1", text="query1", role="query"),
            bvdr.Row(id="q2", text="query2", role="query"),
            bvdr.Row(id="d1", text="doc1", role="document"),
            bvdr.Row(id="d2", text="doc2", role="document"),
            bvdr.Row(id="d3", text="doc3", role="document"),
            bvdr.Row(id="d4", text="doc4", role="document"),
        ]
        relations = [
            {"query_id": "q1", "positive_doc_id": "d1", "negative_doc_ids": ["d2", "d3"]},
            {"query_id": "q2", "positive_doc_id": "d4", "negative_doc_ids": ["d2"]},  # d2 shared/reused
        ]
        ordered = bvdr.apply_group_order(rows, relations)
        ids_roles = [(r.role, r.id) for r in ordered]
        self.assertEqual(
            ids_roles,
            [
                ("query", "q1"),
                ("document", "d1"),
                ("document", "d2"),
                ("document", "d3"),
                ("query", "q2"),
                ("document", "d4"),
                # d2 already emitted for q1, so q2's relation to d2 is a no-op.
            ],
        )
        # group-order must be a permutation: same multiset of rows, no dupes/drops.
        self.assertEqual(sorted((r.role, r.id) for r in ordered), sorted((r.role, r.id) for r in rows))

    def test_group_order_appends_unreferenced_rows_at_end_in_original_order(self) -> None:
        rows = [
            bvdr.Row(id="q1", text="query1", role="query"),
            bvdr.Row(id="orphan1", text="o1", role="document"),
            bvdr.Row(id="d1", text="doc1", role="document"),
            bvdr.Row(id="orphan2", text="o2", role="document"),
        ]
        relations = [{"query_id": "q1", "positive_doc_id": "d1", "negative_doc_ids": []}]
        ordered = bvdr.apply_group_order(rows, relations)
        self.assertEqual([r.id for r in ordered], ["q1", "d1", "orphan1", "orphan2"])

    def test_relations_negatives_cap_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relations.jsonl"
            write_jsonl(path, [{"query_id": "q1", "positive_doc_id": "d1", "negative_doc_ids": ["d2", "d3", "d4", "d5"]}])
            relations = bvdr.read_relations(path, negatives_cap=2)
        self.assertEqual(relations[0]["negative_doc_ids"], ["d2", "d3"])

    def test_group_order_flag_requires_relations(self) -> None:
        with self.assertRaises(SystemExit):
            bvdr.parse_args(["--queries", "q.jsonl", "--group-order", "--output", "out.jsonl", "--teacher-package", "pkg.mll"])


class BuildRowsSchemaTest(MonkeypatchIsolatedTestCase):
    def _build(self, tmp: Path, extra_argv: list[str] | None = None):
        queries = tmp / "queries.jsonl"
        documents = tmp / "documents.jsonl"
        write_jsonl(queries, [{"id": "q1", "text": "side effects of tetracyclines"}])
        write_jsonl(documents, [{"id": "d1", "text": "tetracycline side effects"}, {"id": "d2", "text": "unrelated"}])
        argv = [
            "--queries",
            str(queries),
            "--documents",
            str(documents),
            "--teacher-package",
            str(tmp / "missing-package.mll"),
            "--output",
            str(tmp / "out.jsonl"),
        ] + (extra_argv or [])
        args = bvdr.parse_args(argv)
        bvdr.embed_texts = fake_embed_factory()
        return bvdr.build_rows(args)

    def test_schema_role_and_prefix_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows, manifest = self._build(tmp)

        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["q1"]["schema"], bvdr.SCHEMA)
        self.assertEqual(by_id["q1"]["role"], "query")
        self.assertEqual(by_id["d1"]["role"], "document")
        self.assertEqual(by_id["d2"]["role"], "document")
        # text must be the RAW input text -- no teacher prefix baked in.
        self.assertEqual(by_id["q1"]["text"], "side effects of tetracyclines")
        self.assertNotIn(bvdr.DEFAULT_QUERY_PREFIX, by_id["q1"]["text"])
        # default licensing flags are all False unless explicitly requested.
        for row in rows:
            self.assertFalse(row["train_allowed_for_research"])
            self.assertFalse(row["release_train_allowed"])
            self.assertFalse(row["commercial_use_allowed"])
        # manifest records role counts and the prefixes actually used.
        self.assertEqual(manifest["row_counts_by_role"], {"query": 1, "document": 2, "raw": 0})
        self.assertEqual(manifest["query_prefix"], bvdr.DEFAULT_QUERY_PREFIX)
        self.assertEqual(manifest["document_prefix"], bvdr.DEFAULT_DOCUMENT_PREFIX)
        bucket_prefixes = {b["role"]: b["prefix"] for b in manifest["buckets"]}
        self.assertEqual(bucket_prefixes["query"], bvdr.DEFAULT_QUERY_PREFIX)
        self.assertEqual(bucket_prefixes["document"], bvdr.DEFAULT_DOCUMENT_PREFIX)

    def test_licensing_flags_propagate_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows, _ = self._build(
                tmp,
                ["--train-allowed-for-research", "--release-train-allowed", "--commercial-use-allowed"],
            )
        for row in rows:
            self.assertTrue(row["train_allowed_for_research"])
            self.assertTrue(row["release_train_allowed"])
            self.assertTrue(row["commercial_use_allowed"])

    def test_custom_prefixes_are_threaded_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows, manifest = self._build(tmp, ["--query-prefix", "Q: ", "--document-prefix", "D: "])
        self.assertEqual(manifest["query_prefix"], "Q: ")
        self.assertEqual(manifest["document_prefix"], "D: ")
        # text is still raw regardless of the configured prefix.
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["q1"]["text"], "side effects of tetracyclines")

    def test_determinism_same_inputs_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows1, _ = self._build(tmp)
        with tempfile.TemporaryDirectory() as tmp2:
            tmp2 = Path(tmp2)
            rows2, _ = self._build(tmp2)
        self.assertEqual(rows1, rows2)

    def test_limit_queries_and_docs_take_deterministic_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            queries = tmp / "queries.jsonl"
            documents = tmp / "documents.jsonl"
            write_jsonl(queries, [{"id": f"q{i}", "text": f"query {i}"} for i in range(5)])
            write_jsonl(documents, [{"id": f"d{i}", "text": f"doc {i}"} for i in range(5)])
            args = bvdr.parse_args(
                [
                    "--queries",
                    str(queries),
                    "--documents",
                    str(documents),
                    "--teacher-package",
                    str(tmp / "missing.mll"),
                    "--output",
                    str(tmp / "out.jsonl"),
                    "--limit-queries",
                    "2",
                    "--limit-docs",
                    "3",
                ]
            )
            bvdr.embed_texts = fake_embed_factory()
            rows, manifest = bvdr.build_rows(args)
        query_ids = [r["id"] for r in rows if r["role"] == "query"]
        doc_ids = [r["id"] for r in rows if r["role"] == "document"]
        self.assertEqual(query_ids, ["q0", "q1"])
        self.assertEqual(doc_ids, ["d0", "d1", "d2"])

    def test_group_order_end_to_end_reorders_output_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            queries = tmp / "queries.jsonl"
            documents = tmp / "documents.jsonl"
            relations = tmp / "relations.jsonl"
            write_jsonl(queries, [{"id": "q1", "text": "query one"}, {"id": "q2", "text": "query two"}])
            write_jsonl(
                documents,
                [{"id": "d1", "text": "doc1"}, {"id": "d2", "text": "doc2"}, {"id": "d3", "text": "doc3"}],
            )
            write_jsonl(
                relations,
                [
                    {"query_id": "q2", "positive_doc_id": "d3", "negative_doc_ids": ["d1"]},
                    {"query_id": "q1", "positive_doc_id": "d2", "negative_doc_ids": []},
                ],
            )
            args = bvdr.parse_args(
                [
                    "--queries",
                    str(queries),
                    "--documents",
                    str(documents),
                    "--relations",
                    str(relations),
                    "--group-order",
                    "--teacher-package",
                    str(tmp / "missing.mll"),
                    "--output",
                    str(tmp / "out.jsonl"),
                ]
            )
            bvdr.embed_texts = fake_embed_factory()
            rows, manifest = bvdr.build_rows(args)
        self.assertEqual([r["id"] for r in rows], ["q2", "d3", "d1", "q1", "d2"])
        self.assertTrue(manifest["group_order"])

    def test_skipped_rows_excluded_from_output_and_marked_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            documents = tmp / "documents.jsonl"
            write_jsonl(documents, [{"id": "d1", "text": "doc1"}, {"id": "d2", "text": "doc2"}])

            def partial_embed(rows, role, prefix, args, work_root):
                embedded = rows[:1]
                vectors = {r.id: [0.1, 0.2] for r in embedded}
                report = bvdr.EmbedBucketReport(
                    role=role,
                    prefix=prefix,
                    requested=len(rows),
                    embedded=len(embedded),
                    skipped_for_time_budget=len(rows) - len(embedded),
                    wall_clock_seconds=0.0,
                    shards=1,
                )
                return report, vectors

            bvdr.embed_texts = partial_embed
            args = bvdr.parse_args(
                [
                    "--documents",
                    str(documents),
                    "--teacher-package",
                    str(tmp / "missing.mll"),
                    "--output",
                    str(tmp / "out.jsonl"),
                    "--time-budget-seconds",
                    "1",
                ]
            )
            rows, manifest = bvdr.build_rows(args)
        self.assertEqual([r["id"] for r in rows], ["d1"])
        self.assertTrue(manifest["capped_for_time_budget"])
        self.assertEqual(manifest["skipped_by_role"]["document"], 1)


class ChunkAndShardSchedulingTest(MonkeypatchIsolatedTestCase):
    def test_chunk_rows_by_size_splits_evenly_with_smaller_last_chunk(self) -> None:
        rows = [bvdr.Row(id=str(i), text="t", role="document") for i in range(7)]
        chunks = bvdr.chunk_rows_by_size(rows, 3)
        self.assertEqual([len(c) for c in chunks], [3, 3, 1])

    def test_chunk_rows_by_size_empty_input(self) -> None:
        self.assertEqual(bvdr.chunk_rows_by_size([], 5), [])

    def test_embed_texts_wave_scheduling_respects_time_budget(self) -> None:
        """Exercise the real chunk/wave/time-budget scaffolding in embed_texts,
        with `run_shard` faked out (no real eos subprocess / teacher model).
        `write_shard_dataset` and `read_doc_vectors` run for real against a
        temp work dir, so this covers the shard-file round trip too."""
        rows = [bvdr.Row(id=f"d{i}", text=f"doc {i}", role="document") for i in range(20)]

        original_run_shard = bvdr.run_shard

        def fake_run_shard(eos_bin, package_path, shard_dir, out_dir, prefix, batch_size, gomaxprocs, dataset_label):
            # Simulate real embedding work taking a little wall-clock time,
            # then write a doc-vectors.jsonl exactly like the real exporter
            # would, keyed off the corpus.jsonl the (real) write_shard_dataset
            # already wrote to shard_dir.
            time.sleep(0.05)
            out_dir.mkdir(parents=True, exist_ok=True)
            corpus_rows = []
            with (shard_dir / "corpus.jsonl").open() as handle:
                for line in handle:
                    corpus_rows.append(json.loads(line))
            with (out_dir / "doc-vectors.jsonl").open("w") as handle:
                for row in corpus_rows:
                    handle.write(json.dumps({"id": row["_id"], "vector": [1.0, 2.0]}) + "\n")
            return 0.05

        bvdr.run_shard = fake_run_shard
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work_root = Path(tmp)
                args = bvdr.parse_args(
                    [
                        "--documents",
                        str(work_root / "unused.jsonl"),  # not read directly by embed_texts
                        "--teacher-package",
                        str(work_root / "missing.mll"),
                        "--output",
                        str(work_root / "out.jsonl"),
                        "--shard-size",
                        "4",
                        "--parallel-shards",
                        "2",
                        "--time-budget-seconds",
                        "0.09",
                    ]
                )
                args.run_start = time.monotonic()
                report, vectors = bvdr.embed_texts(rows, bvdr.ROLE_DOCUMENT, "", args, work_root)
        finally:
            bvdr.run_shard = original_run_shard

        # 20 rows / shard-size 4 = 5 shards; parallel-shards=2 => waves of 2.
        # A ~0.09s budget should allow roughly one wave (2 shards / 8 rows)
        # before the loop stops launching more waves.
        self.assertGreater(report.embedded, 0)
        self.assertLess(report.embedded, len(rows))
        self.assertEqual(report.embedded, len(vectors))
        self.assertEqual(report.requested, len(rows))
        self.assertEqual(report.embedded + report.skipped_for_time_budget, report.requested)

    def test_embed_texts_no_time_budget_embeds_everything(self) -> None:
        rows = [bvdr.Row(id=f"d{i}", text=f"doc {i}", role="document") for i in range(9)]

        def fake_run_shard(eos_bin, package_path, shard_dir, out_dir, prefix, batch_size, gomaxprocs, dataset_label):
            out_dir.mkdir(parents=True, exist_ok=True)
            corpus_rows = []
            with (shard_dir / "corpus.jsonl").open() as handle:
                for line in handle:
                    corpus_rows.append(json.loads(line))
            with (out_dir / "doc-vectors.jsonl").open("w") as handle:
                for row in corpus_rows:
                    handle.write(json.dumps({"id": row["_id"], "vector": [3.0]}) + "\n")
            return 0.0

        original_run_shard = bvdr.run_shard
        bvdr.run_shard = fake_run_shard
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work_root = Path(tmp)
                args = bvdr.parse_args(
                    [
                        "--documents",
                        str(work_root / "unused.jsonl"),
                        "--teacher-package",
                        str(work_root / "missing.mll"),
                        "--output",
                        str(work_root / "out.jsonl"),
                        "--shard-size",
                        "4",
                        "--parallel-shards",
                        "3",
                    ]
                )
                args.run_start = time.monotonic()
                report, vectors = bvdr.embed_texts(rows, bvdr.ROLE_DOCUMENT, "", args, work_root)
        finally:
            bvdr.run_shard = original_run_shard

        self.assertEqual(report.embedded, 9)
        self.assertEqual(report.skipped_for_time_budget, 0)
        self.assertEqual(len(vectors), 9)


class ManifestAndOutputWritingTest(MonkeypatchIsolatedTestCase):
    def test_write_output_writes_jsonl_and_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            queries = tmp / "queries.jsonl"
            write_jsonl(queries, [{"id": "q1", "text": "hello"}])
            args = bvdr.parse_args(
                [
                    "--queries",
                    str(queries),
                    "--teacher-package",
                    str(tmp / "pkg.mll"),
                    "--output",
                    str(tmp / "nested" / "out.jsonl"),
                ]
            )
            bvdr.embed_texts = fake_embed_factory()
            rows, manifest = bvdr.build_rows(args)
            bvdr.write_output(rows, manifest, args)

            self.assertTrue(args.output.exists())
            self.assertTrue(args.manifest.exists())
            with args.output.open() as handle:
                written_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(written_rows), 1)
            with args.manifest.open() as handle:
                written_manifest = json.load(handle)
            self.assertEqual(written_manifest["total_rows"], 1)

    def test_manifest_default_path_derives_from_output(self) -> None:
        args = bvdr.parse_args(
            [
                "--queries",
                "q.jsonl",
                "--teacher-package",
                "pkg.mll",
                "--output",
                "runs/foo/data/train.jsonl",
            ]
        )
        self.assertEqual(str(args.manifest), "runs/foo/data/train.jsonl.manifest.json")


if __name__ == "__main__":
    unittest.main()
