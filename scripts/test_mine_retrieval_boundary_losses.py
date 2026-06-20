#!/usr/bin/env python3
"""Tests for compact retrieval boundary-loss mining."""

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

import mine_retrieval_boundary_losses as miner


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MineRetrievalBoundaryLossesTest(unittest.TestCase):
    def test_select_boundary_negatives_prefers_window_and_caps(self) -> None:
        row = {
            "query_id": "q1",
            "top_k": [
                {"rank": 1, "doc_id": "early-1", "relevance": 0},
                {"rank": 90, "doc_id": "b90", "relevance": 0},
                {"rank": 91, "doc_id": "rel", "relevance": 1},
                {"rank": 92, "doc_id": "b92", "relevance": 0},
                {"rank": 93, "doc_id": "pos", "relevance": 1},
                {"rank": 94, "doc_id": "b94", "relevance": 0},
                {"rank": 101, "doc_id": "below", "relevance": 0},
            ],
        }

        docs = miner.select_boundary_negatives(
            row,
            positive_doc_id="pos",
            candidate_rank=100,
            max_negatives=3,
            boundary_start=90,
            boundary_end=105,
        )

        self.assertEqual([doc["doc_id"] for doc in docs], ["b90", "b92", "b94"])

    def test_select_boundary_negatives_fills_from_high_scoring_docs(self) -> None:
        row = {
            "query_id": "q1",
            "top_k": [
                {"rank": 1, "doc_id": "early-1", "relevance": 0},
                {"rank": 2, "doc_id": "early-2", "relevance": 0},
                {"rank": 95, "doc_id": "b95", "relevance": 0},
                {"rank": 103, "doc_id": "below-lost", "relevance": 0},
            ],
        }

        docs = miner.select_boundary_negatives(
            row,
            positive_doc_id="pos",
            candidate_rank=102,
            max_negatives=3,
            boundary_start=90,
            boundary_end=105,
        )

        self.assertEqual([doc["doc_id"] for doc in docs], ["b95", "early-1", "early-2"])

    def test_cli_writes_hard_negative_rows_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            write_jsonl(dataset / "queries.jsonl", [{"_id": "q1", "text": "query text"}])
            write_jsonl(
                dataset / "corpus.jsonl",
                [
                    {"_id": "pos", "title": "Positive", "text": "text"},
                    {"_id": "n90", "title": "Ninety", "text": "text"},
                    {"_id": "n91", "title": "Ninety one", "text": "text"},
                ],
            )
            lost = root / "lost.tsv"
            lost.write_text(
                "query_id\trelevant_count\tdoc_id\tanchor_rank\tcandidate_rank_top130\tquery_recall_delta\n"
                "q1\t2\tpos\t100\t102\t-0.5\n",
                encoding="utf-8",
            )
            per_query = {
                "query_id": "q1",
                "method": "turboquant_ip_b4_overfetch200_fp16_rerank",
                "top_k": [
                    {"rank": 90, "doc_id": "n90", "relevance": 0},
                    {"rank": 91, "doc_id": "n91", "relevance": 0},
                    {"rank": 102, "doc_id": "pos", "relevance": 1},
                ],
            }
            write_jsonl(root / "candidate.jsonl", [per_query])
            write_jsonl(
                root / "anchor.jsonl",
                [
                    {
                        "query_id": "q1",
                        "method": "turboquant_ip_b4_overfetch200_fp16_rerank",
                        "top_k": [],
                    }
                ],
            )

            output = root / "out" / "train.jsonl"
            manifest = root / "out" / "manifest.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "mine_retrieval_boundary_losses.py"),
                    "--lost-tsv",
                    str(lost),
                    "--candidate-per-query",
                    str(root / "candidate.jsonl"),
                    "--anchor-per-query",
                    str(root / "anchor.jsonl"),
                    "--dataset-dir",
                    str(dataset),
                    "--output-jsonl",
                    str(output),
                    "--manifest-json",
                    str(manifest),
                    "--source-run",
                    "source-run",
                    "--max-negatives",
                    "8",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            rows = read_jsonl(output)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("hard_negative_rows=1", completed.stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "query text")
        self.assertEqual(rows[0]["positive"], "Positive text")
        self.assertEqual(rows[0]["metadata"]["positive_doc_id"], "pos")
        self.assertEqual(rows[0]["metadata"]["candidate_negative_doc_ids"], ["n90", "n91"])
        self.assertIs(rows[0]["metadata"]["quality_claim"], False)
        self.assertEqual(summary["lost_rows"], 1)
        self.assertEqual(summary["hard_negative_rows"], 1)
        self.assertFalse(summary["quality_claim"])
        self.assertIn("not benchmark-quality", summary["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
