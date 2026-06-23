#!/usr/bin/env python3
"""Dependency-free tests for LongEmbed per-query gap summarization."""

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

import summarize_long_context_perquery_gaps as summarizer


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def row(
    query_id: str,
    *,
    ndcg: float,
    recall: float = 1.0,
    metric_field: str = "quality",
    first_rank: int = 0,
    top_docs: list[tuple[str, int]] | None = None,
) -> dict:
    if top_docs is None:
        top_docs = [("doc_bad", 0), ("doc_good", 1)]
    result = {
        "dataset": "toy",
        "query_id": query_id,
        metric_field: {"ndcg_at_10": ndcg, "recall_at_100": recall},
        "top_k": [
            {"doc_id": doc_id, "rank": index + 1, "relevance": relevance, "score": 1.0 / (index + 1)}
            for index, (doc_id, relevance) in enumerate(top_docs)
        ],
        "quality_claim": False,
    }
    if metric_field == "fusion_quality":
        result["fusion_first_relevant_rank"] = first_rank
    elif metric_field == "direct_quality":
        result["direct_first_relevant_rank"] = first_rank
    elif metric_field == "token_span_quality":
        result["token_span_first_relevant_rank"] = first_rank
    else:
        result["first_relevant_rank"] = first_rank
    return result


class SummarizeLongContextPerQueryGapsTest(unittest.TestCase):
    def test_nested_metrics_consensus_candidates_and_quality_claim_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos_token = write_jsonl(
                root / "eos-token.jsonl",
                [
                    row("q1", ndcg=0.20, metric_field="token_span_quality", first_rank=0),
                    row("q2", ndcg=0.90, metric_field="token_span_quality", first_rank=1),
                ],
            )
            eos_direct = write_jsonl(
                root / "eos-direct.jsonl",
                [
                    row("q1", ndcg=0.30, metric_field="fusion_quality", first_rank=0),
                    row("q2", ndcg=0.70, metric_field="fusion_quality", first_rank=1),
                ],
            )
            qwen3 = write_jsonl(
                root / "qwen3.jsonl",
                [
                    row("q1", ndcg=0.80, metric_field="quality", first_rank=1, top_docs=[("doc_pos", 1)]),
                    row("q2", ndcg=0.80, metric_field="quality", first_rank=1),
                ],
            )
            mxbai = write_jsonl(
                root / "mxbai.jsonl",
                [
                    row("q1", ndcg=0.70, metric_field="direct_quality", first_rank=1),
                    row("q2", ndcg=0.60, metric_field="direct_quality", first_rank=1),
                ],
            )

            summary, ledger, candidates = summarizer.summarize(
                [
                    summarizer.parse_profile_spec(f"toy:token-span={eos_token}"),
                    summarizer.parse_profile_spec(f"toy:direct-fusion={eos_direct}"),
                ],
                [
                    summarizer.parse_profile_spec(f"toy:qwen3={qwen3}"),
                    summarizer.parse_profile_spec(f"toy:mxbai={mxbai}"),
                ],
                min_gap=0.05,
            )

        self.assertFalse(summary["quality_claim"])
        self.assertEqual(summary["count_external_consensus_misses"], 1)
        self.assertEqual(summary["count_eos_matches_external"], 1)
        self.assertEqual(summary["count_direct_fusion_beats_token_span"], 1)
        self.assertEqual(summary["count_token_span_beats_direct_fusion"], 1)
        self.assertEqual([row["query_id"] for row in candidates], ["q1"])
        self.assertFalse(candidates[0]["quality_claim"])
        self.assertEqual(candidates[0]["external_top_relevant_docs"][0]["doc_id"], "doc_pos")
        self.assertEqual([row["query_id"] for row in ledger], ["q1", "q2"])

    def test_tied_external_profiles_are_matches_not_misses_at_default_min_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos = write_jsonl(
                root / "eos.jsonl",
                [
                    row("q_miss", ndcg=0.40),
                    row("q_tie", ndcg=0.70),
                ],
            )
            qwen3 = write_jsonl(
                root / "qwen3.jsonl",
                [
                    row("q_miss", ndcg=0.60),
                    row("q_tie", ndcg=0.70),
                ],
            )
            mxbai = write_jsonl(
                root / "mxbai.jsonl",
                [
                    row("q_miss", ndcg=0.50),
                    row("q_tie", ndcg=0.70),
                ],
            )

            summary, ledger, candidates = summarizer.summarize(
                [summarizer.parse_profile_spec(f"toy:eos={eos}")],
                [
                    summarizer.parse_profile_spec(f"toy:qwen3={qwen3}"),
                    summarizer.parse_profile_spec(f"toy:mxbai={mxbai}"),
                ],
            )

        by_qid = {item["query_id"]: item for item in ledger}
        self.assertEqual(summary["count_eos_matches_external"], 1)
        self.assertEqual(summary["count_external_consensus_misses"], 1)
        self.assertEqual(summary["count_trails_all_externals"], 1)
        self.assertTrue(by_qid["q_tie"]["eos_matches_external"])
        self.assertFalse(by_qid["q_tie"]["external_consensus_miss"])
        self.assertFalse(by_qid["q_tie"]["trails_all_externals"])
        self.assertTrue(by_qid["q_miss"]["external_consensus_miss"])
        self.assertTrue(by_qid["q_miss"]["trails_all_externals"])
        self.assertEqual([item["query_id"] for item in candidates], ["q_miss"])

    def test_duplicate_query_id_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(Path(tmp) / "dupe.jsonl", [row("q1", ndcg=0.1), row("q1", ndcg=0.2)])
            with self.assertRaisesRegex(ValueError, "duplicate query_id"):
                summarizer.load_per_query(path)

    def test_deterministic_sorting_and_tsv_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos = write_jsonl(
                root / "eos.jsonl",
                [
                    row("q2", ndcg=0.1, top_docs=[("eos_bad2", 0), ("eos_good2", 1)]),
                    row("q1", ndcg=0.1, top_docs=[("eos_bad1", 0), ("eos_good1", 1)]),
                ],
            )
            ext = write_jsonl(
                root / "ext.jsonl",
                [
                    row("q2", ndcg=0.9, top_docs=[("ext_good2", 1)]),
                    row("q1", ndcg=0.7, top_docs=[("ext_good1", 1)]),
                ],
            )
            out_tsv = root / "ledger.tsv"

            summary, ledger, candidates = summarizer.summarize(
                [summarizer.parse_profile_spec(f"toy:eos={eos}")],
                [summarizer.parse_profile_spec(f"toy:teacher={ext}")],
            )
            summarizer.write_tsv(out_tsv, ledger)
            with out_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual([item["query_id"] for item in summary["worst_gaps"]], ["q2", "q1"])
        self.assertEqual([item["query_id"] for item in candidates], ["q2", "q1"])
        self.assertEqual([item["query_id"] for item in rows], ["q1", "q2"])
        self.assertEqual(rows[0]["quality_claim"], "false")
        self.assertEqual(rows[0]["best_external_top_relevant_doc_ids"], "ext_good1")
        self.assertEqual(rows[0]["best_eos_top_nonrelevant_doc_ids"], "eos_bad1")

    def test_cli_writes_json_tsv_and_candidate_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos = write_jsonl(root / "eos.jsonl", [row("q1", ndcg=0.1)])
            qwen3 = write_jsonl(root / "qwen3.jsonl", [row("q1", ndcg=0.6)])
            mxbai = write_jsonl(root / "mxbai.jsonl", [row("q1", ndcg=0.7)])
            out_json = root / "summary.json"
            out_tsv = root / "ledger.tsv"
            out_jsonl = root / "candidates.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "summarize_long_context_perquery_gaps.py"),
                    "--eos",
                    f"toy:eos={eos}",
                    "--external",
                    f"toy:qwen3={qwen3}",
                    "--external",
                    f"toy:mxbai={mxbai}",
                    "--output-json",
                    str(out_json),
                    "--output-tsv",
                    str(out_tsv),
                    "--candidate-jsonl",
                    str(out_jsonl),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads(out_json.read_text(encoding="utf-8"))
            candidate = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
            tsv_exists = out_tsv.exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(summary["quality_claim"])
        self.assertFalse(candidate["quality_claim"])
        self.assertTrue(tsv_exists)


if __name__ == "__main__":
    unittest.main()
