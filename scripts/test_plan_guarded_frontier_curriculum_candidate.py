#!/usr/bin/env python3
"""Dependency-free tests for guarded frontier curriculum candidate planning."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plan_guarded_frontier_curriculum_candidate as planner


GIB = 1024 * 1024 * 1024


def make_repo(root: Path) -> Path:
    (root / ".git").mkdir()
    (root / "scripts").mkdir()
    return root


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def touch(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def curriculum_row(query_id: str, recall_gap: float = 0.0, ndcg_gap: float = 0.4) -> dict:
    return {
        "schema": "manta.embedding_quality_frontier_hard_negative.v1",
        "source": "toy",
        "query": f"query {query_id}",
        "positive": f"positive {query_id}",
        "negatives": [f"negative {query_id}"],
        "metadata": {
            "query_id": query_id,
            "frontier_curriculum_teacher_minus_eos_recall_at_100": recall_gap,
            "frontier_curriculum_teacher_minus_eos_ndcg_at_10": ndcg_gap,
        },
    }


def manifest(row_count: int, *, quality_claim: bool = False, recall_min: float = 0.0) -> dict:
    return {
        "schema": planner.CURRICULUM_MANIFEST_SCHEMA,
        "quality_claim": quality_claim,
        "output_row_count": row_count,
        "selected_teacher_minus_eos_recall_at_100": {
            "min": recall_min,
            "mean": recall_min,
            "max": recall_min,
        },
        "selected_teacher_minus_eos_ndcg_at_10": {
            "min": 0.4,
            "mean": 0.4,
            "max": 0.4,
        },
    }


def fixture_repo(root: Path, *, row_count: int = 2, quality_claim: bool = False) -> dict[str, Path]:
    make_repo(root)
    paths = {
        "curriculum": root / "curriculum.jsonl",
        "manifest": root / "manifest.json",
        "initial": root / "runs/stage-c/manta-embed-m.mll",
        "tokenizer": root / "runs/stage-c/manta-embed-m.tokenizer.mll",
        "anchor": root / "runs/stage-c/scoreboard.json",
        "eval": root / planner.DEFAULT_EVAL_JSONL,
        "hard_eval": root / planner.DEFAULT_HARD_EVAL_JSONL,
    }
    rows = [curriculum_row(f"q{index}", recall_gap=0.1 * index) for index in range(row_count)]
    write_jsonl(paths["curriculum"], rows)
    touch(paths["manifest"], json.dumps(manifest(row_count, quality_claim=quality_claim)) + "\n")
    for key in ("initial", "tokenizer", "anchor", "eval", "hard_eval"):
        touch(paths[key])
    return paths


def args_for(root: Path, paths: dict[str, Path], **overrides):
    parser = planner.build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(root),
            "--curriculum-jsonl",
            str(paths["curriculum"]),
            "--curriculum-manifest",
            str(paths["manifest"]),
            "--initial-artifact",
            str(paths["initial"]),
            "--tokenizer",
            str(paths["tokenizer"]),
            "--anchor-scoreboard",
            str(paths["anchor"]),
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class PlanGuardedFrontierCurriculumCandidateTest(unittest.TestCase):
    def test_valid_plan_emits_summary_and_guarded_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            summary = planner.generate_plan(
                args_for(root, paths),
                free_bytes=lambda path: 30 * GIB,
            )

        self.assertFalse(summary["quality_claim"])
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["blockers"], [])
        shell = summary["planned_shell_command"]
        self.assertIn("ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw", shell)
        self.assertIn("EOS_INITIAL_ARTIFACT=", shell)
        self.assertIn("EOS_TRAIN_JSONL=", shell)
        self.assertIn("EOS_GUARD_ANCHOR_SCOREBOARD=", shell)
        self.assertEqual(summary["planned_env"]["EOS_MODEL_NAME"], "manta-embed-m")

    def test_manifest_quality_claim_true_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root, quality_claim=True)

            with self.assertRaisesRegex(planner.PlanError, "quality_claim"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

    def test_row_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root, row_count=2)
            paths["manifest"].write_text(json.dumps(manifest(3)) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(planner.PlanError, "output_row_count=3"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

    def test_jsonl_row_recall_gap_below_min_fails_even_when_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root, row_count=2)
            write_jsonl(
                paths["curriculum"],
                [
                    curriculum_row("q0", recall_gap=0.05),
                    curriculum_row("q1", recall_gap=0.20),
                ],
            )
            paths["manifest"].write_text(
                json.dumps(manifest(2, recall_min=0.15)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(planner.PlanError, "JSONL metadata recall-gap min"):
                planner.generate_plan(
                    args_for(root, paths, min_selected_recall_gap=0.10),
                    free_bytes=lambda path: 30 * GIB,
                )

    def test_low_disk_records_blocker_without_failing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            summary = planner.generate_plan(
                args_for(root, paths, min_free_gb=15.0),
                free_bytes=lambda path: 14 * GIB,
            )

        self.assertTrue(summary["dry_run"])
        self.assertIn("free disk below threshold", summary["blockers"])
        self.assertEqual(summary["disk_free_bytes"], 14 * GIB)

    def test_output_plan_commands_are_commented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            args = args_for(root, paths)
            summary = planner.generate_plan(args, free_bytes=lambda path: 30 * GIB)
            plan_path = root / "out" / "plan.sh"
            commands = [
                planner.CommandPlan(
                    record["label"],
                    record["env"],
                    record["args"],
                    tuple(record.get("unset_env", [])),
                )
                for record in summary["planned_commands"]
            ]
            planner.write_plan(plan_path, commands)
            lines = plan_path.read_text(encoding="utf-8").splitlines()

        command_lines = [line for line in lines if "ferrous-wheel run" in line]
        self.assertTrue(command_lines)
        self.assertTrue(all(line.startswith("# ") for line in command_lines))


if __name__ == "__main__":
    unittest.main()
