#!/usr/bin/env python3
"""Dependency-free tests for the preset wrapper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_retrieval_vectors as wrapper


class ExportRetrievalVectorsPresetTest(unittest.TestCase):
    def test_e5_preset_applies_query_and_passage_prefixes(self) -> None:
        args = wrapper.expand_preset(["--preset", "e5-small-v2", "--dataset-name", "scifact"])

        self.assertIn("--model-name", args)
        self.assertEqual(args[args.index("--model-name") + 1], "intfloat/e5-small-v2")
        self.assertEqual(args[args.index("--query-prefix") + 1], "query: ")
        self.assertEqual(args[args.index("--document-prefix") + 1], "passage: ")

    def test_bge_preset_applies_query_instruction_and_no_document_prefix(self) -> None:
        args = wrapper.expand_preset(["--preset", "bge-small-en-v1.5"])

        self.assertEqual(args[args.index("--model-name") + 1], "BAAI/bge-small-en-v1.5")
        self.assertEqual(
            args[args.index("--query-prefix") + 1],
            "Represent this sentence for searching relevant passages: ",
        )
        self.assertEqual(args[args.index("--document-prefix") + 1], "")

    def test_minilm_preset_has_no_prefixes(self) -> None:
        args = wrapper.expand_preset(["--preset", "all-minilm-l6-v2"])

        self.assertEqual(
            args[args.index("--model-name") + 1],
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(args[args.index("--query-prefix") + 1], "")
        self.assertEqual(args[args.index("--document-prefix") + 1], "")

    def test_explicit_prefix_overrides_preset(self) -> None:
        args = wrapper.expand_preset(
            ["--preset", "e5-small-v2", "--query-prefix", "custom q: ", "--document-prefix", "custom d: "]
        )

        self.assertEqual(args[args.index("--query-prefix") + 1], "custom q: ")
        self.assertEqual(args[args.index("--document-prefix") + 1], "custom d: ")


if __name__ == "__main__":
    unittest.main()
