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
            exporter.write_child_vectors(
                model,
                chunks,
                output_path,
                prefix="doc: ",
                batch_size=1,
                normalize=True,
            )
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(model.encoded_texts, ["doc: alpha beta", "doc: beta gamma delta"])
        self.assertEqual(rows[0]["parent_id"], "p1")
        self.assertEqual(rows[0]["child_id"], "p1#chunk-0000")
        self.assertEqual(rows[0]["embedding"], [0.0, 3.0])
        self.assertEqual(rows[1]["child_id"], "p1#chunk-0001")


if __name__ == "__main__":
    unittest.main()
