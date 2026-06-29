#!/usr/bin/env python3
"""Dependency-free tests for the BEIR vector exporter helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_qwen3_retrieval_vectors as exporter


class FakeModel:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, texts, **_kwargs):
        self.encoded_texts.extend(texts)
        return [[float(index), float(len(text.split()))] for index, text in enumerate(texts)]


class ExportRetrievalVectorsTest(unittest.TestCase):
    def test_load_docs_skips_empty_rows_and_records_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "d-empty", "title": "", "text": ""}),
                        json.dumps({"_id": "d-title", "title": "Only title", "text": ""}),
                        json.dumps({"_id": "d-text", "title": "", "text": "Only text"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            selection = exporter.load_docs(path, limit=0, qrels=None)

        self.assertEqual(selection.items, [("d-title", "Only title"), ("d-text", "Only text")])
        self.assertEqual(selection.empty_skipped, 1)
        self.assertEqual(selection.empty_sample_ids, ["d-empty"])
        self.assertEqual(selection.qrels_placeholder_rows, 0)
        self.assertEqual(selection.qrels_placeholder_sample_ids, [])

    def test_load_docs_qrels_placeholder_full_export_only_keeps_relevant_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            qrels_path = root / "qrels.tsv"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "d1", "title": "", "text": ""}),
                        json.dumps({"_id": "d2", "title": "", "text": ""}),
                        json.dumps({"_id": "d3", "title": "", "text": "real text"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

            qrels = exporter.parse_qrels(qrels_path)
            selection = exporter.load_docs(
                corpus,
                limit=0,
                qrels=qrels,
                empty_document_policy="qrels-placeholder",
                empty_document_placeholder="[EMPTY_DOCUMENT]",
            )

        self.assertEqual(selection.items, [("d1", "[EMPTY_DOCUMENT]"), ("d3", "real text")])
        self.assertEqual(selection.empty_skipped, 1)
        self.assertEqual(selection.empty_sample_ids, ["d2"])
        self.assertEqual(selection.qrels_placeholder_rows, 1)
        self.assertEqual(selection.qrels_placeholder_sample_ids, ["d1"])

    def test_qrels_aware_doc_cap_keeps_relevant_docs_then_fills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            qrels_path = root / "qrels.tsv"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "d1", "title": "", "text": "filler one"}),
                        json.dumps({"_id": "d2", "title": "", "text": "filler two"}),
                        json.dumps({"_id": "d3", "title": "", "text": "relevant three"}),
                        json.dumps({"_id": "d4", "title": "", "text": ""}),
                        json.dumps({"_id": "d5", "title": "", "text": "relevant five"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text(
                "query-id\tcorpus-id\tscore\nq1\td3\t1\nq1\td4\t1\nq2\td5\t1\n",
                encoding="utf-8",
            )

            qrels = exporter.parse_qrels(qrels_path)
            selection = exporter.load_docs(corpus, limit=2, qrels=qrels)

        self.assertEqual([item_id for item_id, _ in selection.items], ["d3", "d5"])
        self.assertEqual(selection.empty_skipped, 1)
        self.assertEqual(selection.empty_sample_ids, ["d4"])

    def test_qrels_placeholder_doc_cap_includes_relevant_empty_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            qrels_path = root / "qrels.tsv"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "d1", "text": "filler one"}),
                        json.dumps({"_id": "d2", "text": ""}),
                        json.dumps({"_id": "d3", "text": "relevant"}),
                        json.dumps({"_id": "d4", "text": ""}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text(
                "query-id\tcorpus-id\tscore\nq1\td2\t1\nq1\td3\t1\n",
                encoding="utf-8",
            )

            qrels = exporter.parse_qrels(qrels_path)
            selection = exporter.load_docs(
                corpus,
                limit=2,
                qrels=qrels,
                empty_document_policy="qrels-placeholder",
                empty_document_placeholder="EMPTY",
            )

        self.assertEqual(selection.items, [("d2", "EMPTY"), ("d3", "relevant")])
        self.assertEqual(selection.empty_skipped, 1)
        self.assertEqual(selection.empty_sample_ids, ["d4"])
        self.assertEqual(selection.qrels_placeholder_rows, 1)
        self.assertEqual(selection.qrels_placeholder_sample_ids, ["d2"])

    def test_qrels_aware_doc_cap_fills_with_non_relevant_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            qrels_path = root / "qrels.tsv"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "d1", "text": "filler one"}),
                        json.dumps({"_id": "d2", "text": "filler two"}),
                        json.dumps({"_id": "d3", "text": "relevant"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text("qid\tQ0\tdocid\tscore\nq1\t0\td3\t1\n", encoding="utf-8")

            qrels = exporter.parse_qrels(qrels_path)
            selection = exporter.load_docs(corpus, limit=3, qrels=qrels)

        self.assertEqual([item_id for item_id, _ in selection.items], ["d3", "d1", "d2"])

    def test_qrels_aware_query_cap_selects_non_empty_qrels_queries_in_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = root / "queries.jsonl"
            qrels_path = root / "qrels.tsv"
            queries.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "q0", "text": "not in qrels"}),
                        json.dumps({"_id": "q1", "text": ""}),
                        json.dumps({"_id": "q2", "text": "second"}),
                        json.dumps({"_id": "q3", "text": "third"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qrels_path.write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq3\td3\t1\n",
                encoding="utf-8",
            )

            qrels = exporter.parse_qrels(qrels_path)
            selection = exporter.load_queries(queries, limit=1, qrels=qrels)

        self.assertEqual(selection.items, [("q2", "second")])
        self.assertEqual(selection.empty_skipped, 1)
        self.assertEqual(selection.empty_sample_ids, ["q1"])

    def test_chunk_document_text_uses_deterministic_overlapping_ids(self) -> None:
        chunks = exporter.chunk_document_text(
            "doc-7",
            "one two three four five six seven eight nine ten",
            chunk_words=4,
            overlap=1,
            min_words=2,
        )

        self.assertEqual(
            chunks,
            [
                exporter.DocumentChunk("doc-7", "doc-7#chunk-0000", "one two three four"),
                exporter.DocumentChunk("doc-7", "doc-7#chunk-0001", "four five six seven"),
                exporter.DocumentChunk("doc-7", "doc-7#chunk-0002", "seven eight nine ten"),
            ],
        )

    def test_chunk_document_text_drops_short_trailing_chunk(self) -> None:
        chunks = exporter.chunk_document_text(
            "doc-8",
            "one two three four five six seven",
            chunk_words=4,
            overlap=0,
            min_words=4,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].child_id, "doc-8#chunk-0000")
        self.assertEqual(chunks[0].text, "one two three four")

    def test_write_child_vectors_writes_parent_child_embedding_rows(self) -> None:
        model = FakeModel()
        chunks = [
            exporter.DocumentChunk("p1", "p1#chunk-0000", "alpha beta"),
            exporter.DocumentChunk("p1", "p1#chunk-0001", "beta gamma delta"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "child-doc-vectors.jsonl"
            result = exporter.write_child_vectors(
                model,
                chunks,
                output_path,
                prefix="doc: ",
                batch_size=1,
                normalize=True,
            )
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(model.encoded_texts, ["doc: alpha beta", "doc: beta gamma delta"])
        self.assertEqual(result, exporter.WriteResult(rows=2, native_dim=2, output_dim=2))
        self.assertEqual(rows[0]["parent_id"], "p1")
        self.assertEqual(rows[0]["child_id"], "p1#chunk-0000")
        self.assertEqual(rows[0]["embedding"], [0.0, 3.0])
        self.assertEqual(rows[1]["child_id"], "p1#chunk-0001")

    def test_prepare_embedding_truncates_and_renormalizes_prefix(self) -> None:
        vector, native_dim = exporter.prepare_embedding([3.0, 4.0, 12.0], output_dim=2)

        self.assertEqual(native_dim, 3)
        self.assertEqual(vector, [0.6, 0.8])

    def test_prepare_embedding_rejects_output_dim_larger_than_native_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds native embedding dimension"):
            exporter.prepare_embedding([1.0, 2.0], output_dim=3)

    def test_write_manifest_marks_external_exports_as_no_quality_claim(self) -> None:
        args = type(
            "Args",
            (),
            {
                "dataset_name": "sample",
                "dataset_dir": Path("datasets/sample"),
                "model_name": "example/model",
                "output_dim": 2,
                "document_chunk_words": 128,
                "document_chunk_overlap": 32,
                "document_chunk_min_words": 1,
                "query_prefix": "query: ",
                "document_prefix": "doc: ",
                "qrels": Path("datasets/sample/qrels/test.tsv"),
                "max_docs": 2,
                "max_queries": 1,
                "empty_document_policy": "qrels-placeholder",
                "empty_document_placeholder": "[EMPTY_DOCUMENT]",
            },
        )()

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "manifest.json"
            exporter.write_manifest(
                output_path,
                args,
                docs=[("doc-1", "alpha")],
                queries=[("query-1", "beta")],
                chunks=[exporter.DocumentChunk("doc-1", "doc-1#chunk-0000", "alpha")],
                vector_result=exporter.WriteResult(rows=1, native_dim=3, output_dim=2),
                query_result=exporter.WriteResult(rows=1, native_dim=3, output_dim=2),
                normalize=True,
                doc_selection=exporter.ItemSelection(
                    items=[("doc-1", "alpha")],
                    raw_rows=2,
                    empty_skipped=1,
                    empty_sample_ids=["doc-empty"],
                    qrels_placeholder_rows=1,
                    qrels_placeholder_sample_ids=["doc-placeholder"],
                ),
                query_selection=exporter.ItemSelection(
                    items=[("query-1", "beta")],
                    raw_rows=1,
                    empty_skipped=0,
                    empty_sample_ids=[],
                    qrels_placeholder_rows=0,
                    qrels_placeholder_sample_ids=[],
                ),
                qrels=exporter.Qrels(
                    by_query={"query-1": {"doc-1": 1.0}},
                    query_order=["query-1"],
                    relevant_docs={"doc-1"},
                ),
            )
            manifest = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIs(manifest["quality_claim"], False)
        self.assertEqual(manifest["document_empty_rows_skipped"], 1)
        self.assertEqual(manifest["document_empty_sample_ids"], ["doc-empty"])
        self.assertEqual(manifest["document_empty_policy"], "qrels-placeholder")
        self.assertEqual(manifest["document_empty_placeholder"], "[EMPTY_DOCUMENT]")
        self.assertEqual(manifest["document_qrels_placeholder_rows"], 1)
        self.assertEqual(manifest["document_qrels_placeholder_sample_ids"], ["doc-placeholder"])
        self.assertEqual(manifest["qrels_path"], "datasets/sample/qrels/test.tsv")
        self.assertIs(manifest["qrels_aware_cap"], True)


if __name__ == "__main__":
    unittest.main()
