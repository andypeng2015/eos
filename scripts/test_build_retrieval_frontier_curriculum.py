#!/usr/bin/env python3
"""Dependency-free tests for retrieval frontier curriculum building."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_retrieval_frontier_curriculum as builder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def frontier_row(
    query_id: str,
    gap: float,
    *,
    recall_gap: float = 0.0,
    rank_improvement: float = 0.0,
    teacher_hits: int = 1,
) -> dict:
    return {
        "schema": builder.FRONTIER_SCHEMA,
        "dataset": "toy:train",
        "query_id": query_id,
        "eos": {"label": "eos"},
        "teacher": {
            "label": "teacher",
            "top_relevant": [
                {"doc_id": f"{query_id}-pos-{index}", "rank": index + 1, "relevance": 1}
                for index in range(teacher_hits)
            ],
        },
        "delta": {
            "teacher_minus_eos_ndcg_at_10": gap,
            "teacher_minus_eos_recall_at_100": recall_gap,
            "first_relevant_rank_improvement": rank_improvement,
        },
    }


def hard_negative_row(
    query_id: str,
    *,
    query: str | None = None,
    positive: str | None = None,
    negatives: list[str] | None = None,
    source: str = "toy:quality-frontier:teacher",
    metadata: dict | None = None,
) -> dict:
    row_metadata = {"query_id": query_id, "kept": f"original-{query_id}"}
    if metadata:
        row_metadata.update(metadata)
    return {
        "schema": "manta.embedding_quality_frontier_hard_negative.v1",
        "source": source,
        "query": query if query is not None else f"query {query_id}",
        "positive": positive if positive is not None else f"positive {query_id}",
        "negatives": negatives if negatives is not None else [f"negative {query_id} a", f"negative {query_id} b"],
        "metadata": row_metadata,
    }


def source_config(root: Path, name: str, **overrides: object) -> dict:
    values: dict[str, object] = {
        "name": name,
        "frontier_jsonl": str(root / f"{name}.frontier.jsonl"),
        "hard_negatives_jsonl": str(root / f"{name}.hard-negatives.jsonl"),
    }
    values.update(overrides)
    return values


class BuildRetrievalFrontierCurriculumTest(unittest.TestCase):
    def test_joins_and_filters_gap_recall_teacher_hit_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_config(root, "s1", min_gap=0.3, min_recall_gap=0.0)
            write_jsonl(
                Path(source["frontier_jsonl"]),
                [
                    frontier_row("good", 0.7, recall_gap=0.2),
                    frontier_row("low-gap", 0.2, recall_gap=0.2),
                    frontier_row("low-recall", 0.7, recall_gap=-0.1),
                    frontier_row("no-hit", 0.8, recall_gap=0.2, teacher_hits=0),
                    frontier_row("empty-query", 0.9, recall_gap=0.2),
                ],
            )
            write_jsonl(
                Path(source["hard_negatives_jsonl"]),
                [
                    hard_negative_row("good"),
                    hard_negative_row("low-gap"),
                    hard_negative_row("low-recall"),
                    hard_negative_row("no-hit"),
                    hard_negative_row("empty-query", query=""),
                ],
            )

            rows, manifest = builder.build_curriculum({"sources": [source]})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metadata"]["query_id"], "good")
        stats = manifest["sources"][0]
        self.assertEqual(stats["rows_seen"], 5)
        self.assertEqual(stats["joined"], 5)
        self.assertEqual(stats["eligible"], 1)
        self.assertEqual(stats["selected"], 1)
        self.assertEqual(stats["dropped"]["below_min_gap"], 1)
        self.assertEqual(stats["dropped"]["below_min_recall_gap"], 1)
        self.assertEqual(stats["dropped"]["missing_teacher_hit"], 1)
        self.assertEqual(stats["dropped"]["invalid_hard_negative_text"], 1)

    def test_per_source_cap_and_deterministic_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_config(root, "s1", cap=3)
            write_jsonl(
                Path(source["frontier_jsonl"]),
                [
                    frontier_row("q4", 0.7, recall_gap=0.1, rank_improvement=4),
                    frontier_row("q2", 0.8, recall_gap=0.0, rank_improvement=99),
                    frontier_row("q1", 0.8, recall_gap=0.2, rank_improvement=1),
                    frontier_row("q3", 0.8, recall_gap=0.2, rank_improvement=5),
                ],
            )
            write_jsonl(
                Path(source["hard_negatives_jsonl"]),
                [hard_negative_row(qid) for qid in ("q1", "q2", "q3", "q4")],
            )

            rows, manifest = builder.build_curriculum({"sources": [source]})

        self.assertEqual([row["metadata"]["query_id"] for row in rows], ["q3", "q1", "q2"])
        self.assertEqual(
            [row["metadata"]["frontier_curriculum_selected_source_rank"] for row in rows],
            [1, 2, 3],
        )
        self.assertEqual(manifest["sources"][0]["eligible"], 4)
        self.assertEqual(manifest["sources"][0]["capped"], 1)

    def test_global_dedupe_prefers_earlier_priority_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            early = source_config(root, "early", priority=1)
            late = source_config(root, "late", priority=5)
            write_jsonl(Path(early["frontier_jsonl"]), [frontier_row("q-early", 0.4)])
            write_jsonl(Path(late["frontier_jsonl"]), [frontier_row("q-late", 0.9)])
            write_jsonl(
                Path(early["hard_negatives_jsonl"]),
                [hard_negative_row("q-early", query="same query", positive="same positive")],
            )
            write_jsonl(
                Path(late["hard_negatives_jsonl"]),
                [hard_negative_row("q-late", query="same query", positive="same positive")],
            )

            rows, manifest = builder.build_curriculum({"sources": [late, early]})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metadata"]["frontier_curriculum_selected_source_name"], "early")
        source_stats = {source["name"]: source for source in manifest["sources"]}
        self.assertEqual(source_stats["late"]["deduped"], 1)
        self.assertEqual(manifest["duplicate_removal_count"], 1)

    def test_metadata_enrichment_preserves_original_and_truncates_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_config(root, "s1", role="guard", priority=7)
            write_jsonl(Path(source["frontier_jsonl"]), [frontier_row("q1", 0.6, recall_gap=0.3)])
            write_jsonl(
                Path(source["hard_negatives_jsonl"]),
                [
                    hard_negative_row(
                        "q1",
                        negatives=["n1", "n2", "n3"],
                        metadata={"custom": "value"},
                    )
                ],
            )

            rows, manifest = builder.build_curriculum({"sources": [source], "max_negatives": 2})

        row = rows[0]
        self.assertEqual(row["negatives"], ["n1", "n2"])
        self.assertEqual(row["schema"], "manta.embedding_quality_frontier_hard_negative.v1")
        self.assertEqual(row["source"], "toy:quality-frontier:teacher")
        self.assertEqual(row["metadata"]["custom"], "value")
        self.assertEqual(row["metadata"]["frontier_curriculum_selected_source_role"], "guard")
        self.assertEqual(row["metadata"]["frontier_curriculum_selected_source_priority"], 7)
        self.assertAlmostEqual(
            row["metadata"]["frontier_curriculum_teacher_minus_eos_recall_at_100"], 0.3
        )
        self.assertEqual(
            row["metadata"]["frontier_curriculum_original_hard_negative_metadata"]["custom"],
            "value",
        )
        self.assertFalse(manifest["quality_claim"])
        self.assertEqual(manifest["output_row_count"], 1)
        self.assertEqual(manifest["max_negatives"], 2)
        self.assertAlmostEqual(manifest["selected_teacher_minus_eos_ndcg_at_10"]["mean"], 0.6)

    def test_fails_on_duplicate_frontier_query_id_and_missing_hn_query_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_config(root, "s1")
            write_jsonl(
                Path(source["frontier_jsonl"]),
                [frontier_row("q1", 0.6), frontier_row("q1", 0.7)],
            )
            write_jsonl(Path(source["hard_negatives_jsonl"]), [hard_negative_row("q1")])

            with self.assertRaisesRegex(ValueError, "duplicate query_id"):
                builder.build_curriculum({"sources": [source]})

            write_jsonl(Path(source["frontier_jsonl"]), [frontier_row("q1", 0.6)])
            write_jsonl(
                Path(source["hard_negatives_jsonl"]),
                [
                    {
                        "schema": "manta.embedding_quality_frontier_hard_negative.v1",
                        "source": "toy",
                        "query": "q",
                        "positive": "p",
                        "negatives": ["n"],
                        "metadata": {},
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "metadata.query_id"):
                builder.build_curriculum({"sources": [source]})


if __name__ == "__main__":
    unittest.main()
