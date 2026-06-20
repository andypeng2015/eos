#!/usr/bin/env python3
"""Dependency-free tests for LongEmbed frontier teacher-pair mining."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mine_longembed_frontier_teacher_pairs as miner


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_dataset(root: Path, name: str) -> Path:
    dataset_dir = root / "datasets" / name
    write_jsonl(
        dataset_dir / "queries.jsonl",
        [
            {"_id": "q1", "text": "question one"},
            {"_id": "q2", "text": "question two"},
            {"_id": "q3", "text": "question three"},
        ],
    )
    write_jsonl(
        dataset_dir / "corpus.jsonl",
        [
            {"_id": "p1", "text": "positive one", "title": ""},
            {"_id": "p2", "text": "positive two", "title": ""},
            {"_id": "p3", "text": "positive three", "title": ""},
            {"_id": "n1", "text": "negative one", "title": ""},
            {"_id": "n2", "text": "negative two", "title": ""},
            {"_id": "n3", "text": "negative three", "title": ""},
        ],
    )
    return dataset_dir


def write_per_query(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "dataset": "qmsum",
                "query_id": "q1",
                "first_relevant_rank": 3,
                "top_k": [
                    {"rank": 1, "doc_id": "n1", "score": 0.9, "relevance": 0},
                    {"rank": 2, "doc_id": "n2", "score": 0.8, "relevance": 0},
                    {"rank": 3, "doc_id": "p1", "score": 0.7, "relevance": 1},
                    {"rank": 4, "doc_id": "n3", "score": 0.6, "relevance": 0},
                ],
            },
            {
                "dataset": "qmsum",
                "query_id": "q2",
                "first_relevant_rank": 1,
                "top_k": [
                    {"rank": 1, "doc_id": "p2", "score": 0.9, "relevance": 1},
                    {"rank": 2, "doc_id": "n3", "score": 0.4, "relevance": 0},
                ],
            },
            {
                "dataset": "qmsum",
                "query_id": "q3",
                "first_relevant_rank": 2,
                "top_k": [
                    {"rank": 1, "doc_id": "n2", "score": 0.9, "relevance": 0},
                    {"rank": 2, "doc_id": "p3", "score": 0.8, "relevance": 1},
                ],
            },
        ],
    )


def write_candidate_tsv(path: Path) -> None:
    rows = [
        {
            "dataset": "qmsum",
            "category": "direct_wins_token_span_loses",
            "query_id": "q1",
            "winner_profile": "eos/direct_single_vector",
            "winner_ndcg_at_10": "1.0",
            "winner_first_relevant_rank": "1",
            "winner_relevant_doc_id": "p1",
            "loser_profile": "eos/token_span_q4",
            "loser_ndcg_at_10": "0.5",
            "loser_first_relevant_rank": "3",
            "loser_relevant_doc_id": "p1",
            "delta_ndcg_at_10": "0.5",
        },
        {
            "dataset": "qmsum",
            "category": "token_span_wins_direct_loses",
            "query_id": "q2",
            "winner_profile": "eos/token_span_q4",
            "winner_ndcg_at_10": "1.0",
            "winner_first_relevant_rank": "1",
            "winner_relevant_doc_id": "p2",
            "loser_profile": "eos/direct_single_vector",
            "loser_ndcg_at_10": "0.5",
            "loser_first_relevant_rank": "4",
            "loser_relevant_doc_id": "p2",
            "delta_ndcg_at_10": "0.5",
        },
        {
            "dataset": "qmsum",
            "category": "external_wins_when_eos_loses",
            "query_id": "q3",
            "winner_profile": "external/qwen3_0.6b_dense",
            "winner_ndcg_at_10": "1.0",
            "winner_first_relevant_rank": "1",
            "winner_relevant_doc_id": "p3",
            "loser_profile": "eos/sparse_parent_q4",
            "loser_ndcg_at_10": "0.0",
            "loser_first_relevant_rank": "11",
            "loser_relevant_doc_id": "p3",
            "delta_ndcg_at_10": "1.0",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class MineLongEmbedFrontierTeacherPairsTest(unittest.TestCase):
    def test_negative_selection_prefers_nonrelevant_docs_before_positive(self) -> None:
        row = {
            "first_relevant_rank": 3,
            "top_k": [
                {"rank": 1, "doc_id": "n1", "relevance": 0},
                {"rank": 2, "doc_id": "n2", "relevance": 0},
                {"rank": 3, "doc_id": "p", "relevance": 1},
                {"rank": 4, "doc_id": "n3", "relevance": 0},
            ],
        }

        self.assertEqual(
            [item["doc_id"] for item in miner.hard_negative_docs(row, "p", 3)],
            ["n1", "n2", "n3"],
        )

    def test_cli_writes_guarded_split_and_category_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = write_dataset(root, "qmsum")
            run_dir = root / "run"
            run_dir.mkdir()
            per_query_dir = root / "frontier"
            per_query_dir.mkdir()
            profile_file = run_dir / "eos-token-span-multivector-turboquant.per-query.jsonl"
            direct_file = run_dir / "direct-eos.per-query.jsonl"
            sparse_file = per_query_dir / "qmsum-sparse-parent-q4678.per-query.jsonl"
            write_per_query(profile_file)
            write_per_query(direct_file)
            write_per_query(sparse_file)
            tsv = root / "candidate-examples.tsv"
            write_candidate_tsv(tsv)
            out = root / "out"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "mine_longembed_frontier_teacher_pairs.py"),
                    "--candidate-tsv",
                    str(tsv),
                    "--per-query-dir",
                    str(per_query_dir),
                    "--dataset-dir",
                    f"qmsum={dataset_dir}",
                    "--run-dir",
                    f"qmsum={run_dir}",
                    "--output-dir",
                    str(out),
                    "--seed",
                    "3",
                    "--max-negatives",
                    "2",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            all_rows = read_jsonl(out / "all-hard-negatives.jsonl")
            train = read_jsonl(out / "train-hard-negatives.jsonl")
            eval_rows = read_jsonl(out / "eval-hard-negatives.jsonl")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            category_rows = read_jsonl(out / "by-category" / "external_wins_when_eos_loses.jsonl")

        self.assertIn("emitted=3", completed.stdout)
        self.assertEqual(len(all_rows), 3)
        self.assertTrue(train)
        self.assertTrue(eval_rows)
        self.assertFalse(manifest["quality_claim"])
        self.assertIn("must not be used to claim benchmark quality", manifest["claim_boundary"])
        self.assertEqual(manifest["counts_by_rescue_family"]["direct_rescue"], 1)
        self.assertEqual(manifest["counts_by_rescue_family"]["token_span_rescue"], 1)
        self.assertEqual(manifest["counts_by_rescue_family"]["external_teacher_win"], 1)
        self.assertEqual(len(category_rows), 1)
        self.assertTrue(all(row["schema"] == miner.SCHEMA for row in all_rows))
        self.assertFalse(
            {(row["metadata"]["dataset"], row["metadata"]["query_id"]) for row in train}
            & {(row["metadata"]["dataset"], row["metadata"]["query_id"]) for row in eval_rows}
        )
        first = [row for row in all_rows if row["metadata"]["query_id"] == "q1"][0]
        self.assertEqual(first["negatives"], ["negative one", "negative two"])
        self.assertEqual(first["metadata"]["rescue_family"], "direct_rescue")


if __name__ == "__main__":
    unittest.main()
