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
            )
            manifest = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIs(manifest["quality_claim"], False)


if __name__ == "__main__":
    unittest.main()
