#!/usr/bin/env python3
"""Dependency-free tests for guarded LongEmbed repair decision summarization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_guarded_longembed_repair_decision as summarizer


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def plan(*, schema: str | None = None, quality_claim: bool = False) -> dict:
    return {
        "schema": schema or summarizer.PLAN_SCHEMA,
        "quality_claim": quality_claim,
        "decision": "ready",
        "run": {
            "run_id": "repair-run",
            "run_dir": "/tmp/repair-run",
            "candidate_run_id": "candidate",
        },
        "compact_post_gate_requirement": {
            "required": True,
            "must_pass_before_promotion": True,
            "profile": "q4/fp16/rerank-overfetch=200",
            "comparator_scoreboard": "/tmp/current/compact-scoreboard.json",
            "bits": 4,
            "rerank_storage": "fp16",
            "rerank_overfetch": 200,
        },
        "planned_commands": [
            {
                "label": summarizer.DENSE_LABEL,
                "env": {"EOS_GUARD_METRICS": "ndcg_at_10,recall_at_100"},
            },
            {
                "label": summarizer.COMPACT_LABEL,
                "env": {
                    "EOS_GUARD_METRICS": "ndcg_at_10,recall_at_100,total_compression_ratio",
                    "EOS_GUARD_BASELINE": "eos-turboquant-rerank",
                    "EOS_GUARD_METHOD": "turboquant_ip_b4_overfetch200_fp16_rerank",
                },
            },
        ],
    }


def manifest(*, gate_status: str = "accepted", gate_exit_code: int = 0, suffix: str = "dense") -> dict:
    return {
        "repo_root": "/tmp/repo",
        "run_id": f"{suffix}-run",
        "run_dir": f"/tmp/{suffix}-run",
        "candidate_dir": f"/tmp/{suffix}-run/candidate",
        "candidate_sealed_artifact": f"/tmp/{suffix}-run/candidate/eos-embed-v1.mll",
        "scoreboard_json": f"/tmp/{suffix}-run/candidate-scoreboard/scoreboard.json",
        "anchor_scoreboard": f"/tmp/current/{suffix}-anchor-scoreboard.json",
        "summary_tsv": f"/tmp/{suffix}-run/summary.tsv",
        "gate_status": gate_status,
        "gate_exit_code": gate_exit_code,
        "metrics": "ndcg_at_10,recall_at_100",
        "baseline": "eos",
    }


class SummarizeGuardedLongEmbedRepairDecisionTest(unittest.TestCase):
    def test_both_gates_accepted_still_needs_long_context_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_json = write_json(root / "plan.json", plan())
            dense_json = write_json(root / "dense.manifest.json", manifest(suffix="dense"))
            compact_json = write_json(root / "compact.manifest.json", manifest(suffix="compact"))
            output_tsv = root / "decision.tsv"

            summary = summarizer.summarize_decision(
                plan_json=plan_json,
                dense_manifest=dense_json,
                compact_manifest=compact_json,
            )
            summarizer.write_tsv(output_tsv, summary)
            tsv_text = output_tsv.read_text(encoding="utf-8")

        self.assertFalse(summary["quality_claim"])
        self.assertEqual(
            summary["decision"],
            "short_dense_and_compact_passed_needs_long_context_review",
        )
        self.assertEqual(summary["promotion_status"], "not_promoted")
        self.assertEqual(summary["dense_manifest"]["gate_status"], "accepted")
        self.assertEqual(summary["compact_manifest"]["gate_status"], "accepted")
        self.assertEqual(
            summary["compact_post_gate_requirement"]["required_metrics"],
            ["ndcg_at_10", "recall_at_100", "total_compression_ratio"],
        )
        self.assertIn("decision\tshort_dense_and_compact_passed_needs_long_context_review", tsv_text)
        self.assertIn("quality_claim\tfalse", tsv_text)

    def test_dense_rejected_blocks_promotion_before_compact_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = summarizer.summarize_decision(
                plan_json=write_json(root / "plan.json", plan()),
                dense_manifest=write_json(
                    root / "dense.manifest.json",
                    manifest(gate_status="rejected", gate_exit_code=1, suffix="dense"),
                ),
                compact_manifest=write_json(root / "compact.manifest.json", manifest(suffix="compact")),
            )

        self.assertEqual(summary["decision"], "no_promote_dense_gate_failed")
        self.assertFalse(summary["dense_manifest"]["accepted"])

    def test_compact_rejected_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = summarizer.summarize_decision(
                plan_json=write_json(root / "plan.json", plan()),
                dense_manifest=write_json(root / "dense.manifest.json", manifest(suffix="dense")),
                compact_manifest=write_json(
                    root / "compact.manifest.json",
                    manifest(gate_status="rejected", gate_exit_code=2, suffix="compact"),
                ),
            )

        self.assertEqual(summary["decision"], "no_promote_compact_gate_failed")
        self.assertFalse(summary["compact_manifest"]["accepted"])

    def test_missing_compact_manifest_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = summarizer.summarize_decision(
                plan_json=write_json(root / "plan.json", plan()),
                dense_manifest=write_json(root / "dense.manifest.json", manifest(suffix="dense")),
                compact_manifest=None,
            )

        self.assertEqual(summary["decision"], "pending")
        self.assertFalse(summary["evidence_complete"])
        self.assertFalse(summary["compact_manifest"]["provided"])

    def test_plan_quality_claim_true_or_wrong_schema_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense_json = write_json(root / "dense.manifest.json", manifest(suffix="dense"))
            compact_json = write_json(root / "compact.manifest.json", manifest(suffix="compact"))

            with self.assertRaisesRegex(summarizer.DecisionError, "quality_claim"):
                summarizer.summarize_decision(
                    plan_json=write_json(root / "claim-plan.json", plan(quality_claim=True)),
                    dense_manifest=dense_json,
                    compact_manifest=compact_json,
                )
            with self.assertRaisesRegex(summarizer.DecisionError, "schema"):
                summarizer.summarize_decision(
                    plan_json=write_json(root / "schema-plan.json", plan(schema="wrong.schema")),
                    dense_manifest=dense_json,
                    compact_manifest=compact_json,
                )


if __name__ == "__main__":
    unittest.main()
