#!/usr/bin/env python3
"""Dependency-free tests for LongEmbed frontier hard-negative curation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import curate_longembed_frontier_hard_negatives as curator


HARD_NEGATIVE_SCHEMA = "manta.embedding_quality_frontier_hard_negative.v1"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hard_row(
    query_id: str,
    query: str,
    positive: str,
    negatives: list[str],
    source: str,
    teacher_label: str,
) -> dict:
    return {
        "schema": HARD_NEGATIVE_SCHEMA,
        "source": source,
        "query": query,
        "positive": positive,
        "negatives": negatives,
        "metadata": {
            "query_id": query_id,
            "teacher_label": teacher_label,
            "eos_label": "eos-q4",
            "teacher_minus_eos_ndcg_at_10": 1.0,
            "teacher_top_relevant_doc_ids": [f"{query_id}-pos"],
            "eos_top_nonrelevant_doc_ids": [f"{query_id}-neg"],
        },
    }


class CurateLongEmbedFrontierHardNegativesTest(unittest.TestCase):
    def test_merge_dedupes_by_query_id_query_positive_and_caps_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.jsonl"
            second = Path(tmp) / "second.jsonl"
            write_jsonl(
                first,
                [
                    hard_row("q1", "question", "positive", ["n1", "n2"], "needle:quality-frontier:qwen", "qwen"),
                    hard_row("q2", "other", "positive2", ["a"], "needle:quality-frontier:qwen", "qwen"),
                ],
            )
            write_jsonl(
                second,
                [
                    hard_row("q1", "question", "positive", ["n2", "n3", "n4"], "needle:quality-frontier:mxbai", "mxbai"),
                ],
            )

            rows, stats = curator.normalize_inputs([first, second], max_negatives=3)

        merged = [row for row in rows if row.query_id == "q1"][0]
        self.assertEqual(stats["input_rows"], 3)
        self.assertEqual(stats["unique_rows"], 2)
        self.assertEqual(stats["duplicate_rows"], 1)
        self.assertEqual(stats["merged_rows"], 1)
        self.assertEqual(merged.negatives, ["n1", "n2", "n3"])
        self.assertEqual(merged.metadata["teacher_labels"], ["mxbai", "qwen"])
        self.assertEqual(merged.metadata["merged_input_rows"], 2)
        self.assertEqual(merged.schema, HARD_NEGATIVE_SCHEMA)

    def test_split_is_deterministic_and_query_disjoint(self) -> None:
        rows = [
            curator.CuratedRow(query_id=f"q{i}", query=f"query {i}", positive=f"pos {i}")
            for i in range(10)
        ]
        train_a, eval_a = curator.split_rows(rows, 80, 20, seed=17)
        train_b, eval_b = curator.split_rows(rows, 80, 20, seed=17)

        self.assertEqual([row.query_id for row in train_a], [row.query_id for row in train_b])
        self.assertEqual([row.query_id for row in eval_a], [row.query_id for row in eval_b])
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(eval_a), 2)
        self.assertFalse({row.query_id for row in train_a} & {row.query_id for row in eval_a})

    def test_cli_writes_guarded_manifest_and_parseable_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_dir = Path(tmp) / "out"
            write_jsonl(
                input_path,
                [
                    hard_row("q1", "question 1", "positive 1", ["n1"], "needle:quality-frontier:qwen", "qwen"),
                    hard_row("q2", "question 2", "positive 2", ["n2"], "needle:quality-frontier:qwen", "qwen"),
                    hard_row("q3", "question 3", "positive 3", ["n3"], "needle:quality-frontier:mxbai", "mxbai"),
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "curate_longembed_frontier_hard_negatives.py"),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "7",
                    "--train-ratio",
                    "2",
                    "--eval-ratio",
                    "1",
                    "--source-prefix",
                    "frontier-curated",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            train = read_jsonl(output_dir / "train-hard-negatives.jsonl")
            eval_rows = read_jsonl(output_dir / "eval-hard-negatives.jsonl")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertIn("curated longembed frontier", completed.stdout)
        self.assertTrue(train)
        self.assertTrue(eval_rows)
        self.assertFalse(manifest["quality_claim"])
        self.assertIn("must not be used to claim benchmark quality", manifest["train_claim_boundary"])
        self.assertFalse(
            {row["metadata"]["query_id"] for row in train}
            & {row["metadata"]["query_id"] for row in eval_rows}
        )
        self.assertTrue(all(row["source"].startswith("frontier-curated:") for row in train + eval_rows))
        self.assertTrue(all(row["schema"] == HARD_NEGATIVE_SCHEMA for row in train + eval_rows))


if __name__ == "__main__":
    unittest.main()
