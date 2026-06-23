#!/usr/bin/env python3
"""Dependency-free tests for retrieval quality frontier prioritization."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_retrieval_quality_frontier as summarizer


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def frontier_row(
    dataset: str,
    query_id: str,
    ndcg_gap: float,
    *,
    recall_gap: float = 0.0,
    rank_improvement: float = 0.0,
    teacher_label: str = "teacher",
    eos_label: str = "eos",
    teacher_hits: int = 1,
) -> dict:
    return {
        "schema": summarizer.FRONTIER_SCHEMA,
        "dataset": dataset,
        "query_id": query_id,
        "eos": {"label": eos_label},
        "teacher": {
            "label": teacher_label,
            "top_relevant": [
                {"doc_id": f"{query_id}-pos-{index}", "relevance": 1}
                for index in range(teacher_hits)
            ],
        },
        "delta": {
            "teacher_minus_eos_ndcg_at_10": ndcg_gap,
            "teacher_minus_eos_recall_at_100": recall_gap,
            "first_relevant_rank_improvement": rank_improvement,
        },
    }


def find_bucket(summary: dict, bucket_type: str, bucket_key: str) -> dict:
    for bucket in summary["buckets"]:
        if bucket["bucket_type"] == bucket_type and bucket["bucket_key"] == bucket_key:
            return bucket
    raise AssertionError(f"missing bucket {bucket_type}:{bucket_key}")


class SummarizeRetrievalQualityFrontierTest(unittest.TestCase):
    def test_aggregates_dataset_teacher_eos_and_source_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier.jsonl"
            write_jsonl(
                frontier,
                [
                    frontier_row("fiqa", "q1", 0.4, recall_gap=0.2, rank_improvement=4),
                    frontier_row("fiqa", "q2", 0.2, recall_gap=0.0, rank_improvement=2),
                    frontier_row("nfcorpus", "q3", 0.6, teacher_label="mxbai"),
                ],
            )

            summary = summarizer.summarize_frontiers([frontier])

        self.assertFalse(summary["manifest"]["quality_claim"])
        self.assertEqual(summary["rows_seen"], 3)
        self.assertEqual(summary["rows_used"], 3)
        dataset = find_bucket(summary, "dataset", "fiqa")
        self.assertEqual(dataset["count"], 2)
        self.assertAlmostEqual(dataset["mean_teacher_minus_eos_ndcg_at_10"], 0.3)
        self.assertAlmostEqual(dataset["max_teacher_minus_eos_ndcg_at_10"], 0.4)
        self.assertAlmostEqual(dataset["mean_teacher_minus_eos_recall_at_100"], 0.1)
        self.assertAlmostEqual(dataset["mean_first_relevant_rank_improvement"], 3.0)
        self.assertEqual(dataset["teacher_hit_count"], 2)
        source = find_bucket(summary, "source_path", str(frontier))
        self.assertEqual(source["count"], 3)
        combo = find_bucket(summary, "dataset_teacher_eos", "fiqa\tteacher\teos")
        self.assertEqual(combo["count"], 2)

    def test_min_gap_filters_rows_before_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier.jsonl"
            write_jsonl(
                frontier,
                [
                    frontier_row("fiqa", "q1", 0.49),
                    frontier_row("fiqa", "q2", 0.50),
                    frontier_row("fiqa", "q3", 0.80),
                ],
            )

            summary = summarizer.summarize_frontiers([frontier], min_gap=0.5)

        dataset = find_bucket(summary, "dataset", "fiqa")
        self.assertEqual(summary["rows_used"], 2)
        self.assertEqual(dataset["count"], 2)
        self.assertAlmostEqual(dataset["mean_teacher_minus_eos_ndcg_at_10"], 0.65)

    def test_risky_negative_recall_and_weak_teacher_hit_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier.jsonl"
            write_jsonl(
                frontier,
                [
                    frontier_row("nfcorpus", "q1", 0.8, recall_gap=-0.1, teacher_hits=0),
                    frontier_row("nfcorpus", "q2", 0.6, recall_gap=0.1, teacher_hits=1),
                    frontier_row("nfcorpus", "q3", 0.4, recall_gap=0.0, teacher_hits=0),
                ],
            )

            summary = summarizer.summarize_frontiers([frontier])

        bucket = find_bucket(summary, "dataset", "nfcorpus")
        self.assertTrue(bucket["risky"])
        self.assertEqual(bucket["negative_recall_gap_count"], 1)
        self.assertEqual(bucket["teacher_hit_count"], 1)
        self.assertAlmostEqual(bucket["teacher_hit_coverage"], 1 / 3)
        self.assertEqual(
            bucket["risky_reasons"], ["negative_recall_gap", "weak_teacher_hit_coverage"]
        )

    def test_tsv_output_contains_ranked_bucket_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier.jsonl"
            output = root / "summary.tsv"
            write_jsonl(frontier, [frontier_row("fiqa", "q1", 0.5)])

            summary = summarizer.summarize_frontiers([frontier], top_k=2)
            summarizer.write_tsv(output, summary["buckets"])
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rank"], "1")
        self.assertIn(rows[0]["bucket_type"], {"dataset", "dataset_teacher_eos"})
        self.assertEqual(rows[0]["risky"], "False")
        self.assertIn("priority_score", rows[0])

    def test_stable_sorting_uses_bucket_type_and_key_for_ties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontier = root / "frontier.jsonl"
            write_jsonl(
                frontier,
                [
                    frontier_row("b", "q1", 0.3, rank_improvement=0),
                    frontier_row("a", "q2", 0.3, rank_improvement=0),
                ],
            )

            summary = summarizer.summarize_frontiers([frontier])

        dataset_keys = [
            bucket["bucket_key"]
            for bucket in summary["buckets"]
            if bucket["bucket_type"] == "dataset"
        ]
        self.assertEqual(dataset_keys, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
