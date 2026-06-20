#!/usr/bin/env python3
"""Dependency-free tests for retrieval quality frontier mining."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mine_retrieval_quality_frontier as miner


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def per_query(
    query_id: str,
    ndcg: float,
    *,
    method: str | None = None,
    bits: int | None = None,
    rank: int = 1,
    top_k: list[dict] | None = None,
) -> dict:
    row = {
        "schema": "manta.embedding_retrieval_per_query.v1",
        "dataset": "toy",
        "query_id": query_id,
        "relevant_count": 1,
        "first_relevant_rank": rank,
        "quality": {"ndcg_at_10": ndcg, "recall_at_100": 1.0 if rank else 0.0},
        "top_k": top_k
        if top_k is not None
        else [
            {"rank": 1, "doc_id": f"{query_id}-pos", "relevance": 1},
            {"rank": 2, "doc_id": f"{query_id}-neg", "relevance": 0},
        ],
    }
    if method is not None:
        row["method"] = method
    if bits is not None:
        row["bits"] = bits
    return row


def fusion_per_query(query_id: str, ndcg: float, *, method: str) -> dict:
    return {
        "schema": "eos.long_context_wedge_parent_span_fusion_per_query.v1",
        "dataset": "toy",
        "query_id": query_id,
        "method": method,
        "fusion_quality": {"ndcg_at_10": ndcg, "recall_at_100": 0.5},
        "fusion_first_relevant_rank": 3,
    }


def args(eos_path: Path, teacher_path: Path, **overrides: object) -> Namespace:
    values = {
        "eos_per_query": eos_path,
        "teacher_per_query": teacher_path,
        "eos_label": "eos",
        "teacher_label": "teacher",
        "method": "",
        "eos_method": None,
        "teacher_method": None,
        "bits": None,
        "eos_bits": None,
        "teacher_bits": None,
        "dataset": "toy",
        "limit": 0,
        "min_ndcg_gap": 0.000001,
        "min_teacher_ndcg": 0.0,
        "require_teacher_hit": True,
        "negatives_per_query": 2,
    }
    values.update(overrides)
    return Namespace(**values)


class MineRetrievalQualityFrontierTest(unittest.TestCase):
    def test_symmetric_method_and_bits_filters_both_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos = root / "eos.jsonl"
            teacher = root / "teacher.jsonl"
            write_jsonl(
                eos,
                [
                    per_query("q1", 0.1, method="q4", bits=4),
                    per_query("q1", 0.9, method="dense"),
                ],
            )
            write_jsonl(
                teacher,
                [
                    per_query("q1", 0.8, method="q4", bits=4),
                    per_query("q1", 0.2, method="dense"),
                ],
            )

            rows = miner.build_frontier_rows(args(eos, teacher, method="q4", bits=4))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_id"], "q1")
        self.assertAlmostEqual(rows[0]["delta"]["teacher_minus_eos_ndcg_at_10"], 0.7)
        self.assertEqual(rows[0]["eos"]["top_nonrelevant"][0]["doc_id"], "q1-neg")

    def test_asymmetric_direct_eos_to_q4_teacher_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos = root / "direct.jsonl"
            teacher = root / "teacher.jsonl"
            write_jsonl(eos, [per_query("q1", 0.25)])
            write_jsonl(
                teacher,
                [
                    per_query("q1", 0.2, method="dense"),
                    per_query("q1", 0.95, method="q4", bits=4),
                ],
            )

            rows = miner.build_frontier_rows(
                args(teacher_path=teacher, eos_path=eos, teacher_method="q4", teacher_bits=4)
            )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["teacher"]["ndcg_at_10"], 0.95)

    def test_asymmetric_fusion_quality_to_q4_teacher_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos = root / "fusion.jsonl"
            teacher = root / "teacher.jsonl"
            write_jsonl(
                eos,
                [
                    fusion_per_query("q1", 0.15, method="fusion-a"),
                    fusion_per_query("q1", 0.99, method="fusion-b"),
                ],
            )
            write_jsonl(teacher, [per_query("q1", 0.75, method="q4", bits=4)])

            rows = miner.build_frontier_rows(
                args(eos, teacher, eos_method="fusion-a", teacher_method="q4", teacher_bits=4)
            )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["eos"]["ndcg_at_10"], 0.15)
        self.assertEqual(rows[0]["eos"]["first_relevant_rank"], 3)
        self.assertEqual(rows[0]["eos"]["top_nonrelevant"], [])


if __name__ == "__main__":
    unittest.main()
