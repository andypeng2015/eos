#!/usr/bin/env python3
"""Dependency-free tests for guarded LongEmbed repair candidate planning."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plan_guarded_longembed_repair_candidate as planner


GIB = 1024 * 1024 * 1024


def make_repo(root: Path) -> Path:
    (root / ".git").mkdir()
    (root / "scripts").mkdir()
    return root


def touch(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    touch(path, json.dumps(data, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def teacher_row(query_id: str, *, scored: bool = True, scores: list[float] | None = None) -> dict:
    row = {
        "source": "qmsum two-teacher frontier",
        "query": f"query {query_id}",
        "positive": f"positive {query_id}",
        "negatives": [f"negative {query_id} a", f"negative {query_id} b"],
    }
    if scored:
        row["teacher_scores"] = scores if scores is not None else [0.9, 0.2, 0.1]
    return row


def teacher_manifest(rows: int, scored: int, *, quality_claim=False, include_quality=True) -> dict:
    data = {
        "schema": planner.TEACHER_MANIFEST_SCHEMA,
        "datasets": ["qmsum", "2wikimqa"],
        "coverage": {
            "examples_written": rows,
            "examples_seen": rows,
            "examples_with_teacher_scores": scored,
            "dataset_counts": {
                "qmsum": {"examples": rows, "scored": scored, "cleared": rows - scored},
            },
        },
    }
    if include_quality:
        data["quality_claim"] = quality_claim
    return data


def gap_summary(*, quality_claim=False, consensus_misses: int = 2, max_candidates: int = 0) -> dict:
    return {
        "schema": planner.GAP_SUMMARY_SCHEMA,
        "quality_claim": quality_claim,
        "query_count": 4,
        "dataset_count": 2,
        "count_external_consensus_misses": consensus_misses,
        "count_eos_matches_external": 2,
        "mean_best_external_minus_best_eos_ndcg_at_10": 0.25,
        "parameters": {"max_candidates": max_candidates},
        "datasets": [
            {
                "dataset": "qmsum",
                "quality_claim": False,
                "query_count": 2,
                "count_external_consensus_misses": 1,
                "count_eos_matches_external": 1,
                "mean_best_external_minus_best_eos_ndcg_at_10": 0.3,
            },
            {
                "dataset": "2wikimqa",
                "quality_claim": False,
                "query_count": 2,
                "count_external_consensus_misses": 1,
                "count_eos_matches_external": 1,
                "mean_best_external_minus_best_eos_ndcg_at_10": 0.2,
            },
        ],
    }


def miss_row(query_id: str, *, quality_claim=False) -> dict:
    return {
        "schema": planner.GAP_MISS_SCHEMA,
        "quality_claim": quality_claim,
        "dataset": "qmsum",
        "query_id": query_id,
        "gap": {"best_external_minus_best_eos_ndcg_at_10": 1.0},
    }


def default_manifest(root: Path) -> dict:
    paths = {
        "package": root / "runs/current/candidate/eos-embed-v1.mll",
        "tokenizer": root / "runs/current/candidate/eos-embed-v1.tokenizer.mll",
        "dense_candidate": root / "runs/current/candidate-scoreboard/scoreboard.json",
        "dense_baseline": root / "runs/s40/dense-scoreboard/scoreboard.json",
        "compact_candidate": root / "runs/current/compact-q4-fp16-overfetch200-scoreboard/scoreboard.json",
        "compact_baseline": root / "runs/s40/compact-q4-fp16-overfetch200-scoreboard/scoreboard.json",
    }
    for path in paths.values():
        touch(path)
    return {
        "asset_id": "corkscrewdb-default-embedder",
        "source_release": {
            "package": paths["package"].relative_to(root).as_posix(),
            "tokenizer": paths["tokenizer"].relative_to(root).as_posix(),
            "directory": "runs/current",
        },
        "dense_gate_evidence": {
            "candidate": paths["dense_candidate"].relative_to(root).as_posix(),
            "baseline": paths["dense_baseline"].relative_to(root).as_posix(),
            "status": "PASS",
            "checks": 6,
        },
        "compact_policy": {
            "bits": 4,
            "rerank_storage": "fp16",
            "rerank_overfetch": 200,
            "profile": "q4/fp16/rerank-overfetch=200",
            "gate_evidence": {
                "candidate": paths["compact_candidate"].relative_to(root).as_posix(),
                "baseline": paths["compact_baseline"].relative_to(root).as_posix(),
                "status": "PASS",
                "checks": 9,
            },
        },
    }


def fixture_repo(root: Path) -> dict[str, Path]:
    make_repo(root)
    paths = {
        "manifest": root / "assets/corkscrewdb-default-embedder/manifest.json",
        "train_jsonl": root / "runs/teacher/primary.train.filtered.jsonl",
        "eval_jsonl": root / "runs/teacher/primary.eval.filtered.jsonl",
        "train_manifest": root / "runs/teacher/primary.train.manifest.json",
        "eval_manifest": root / "runs/teacher/primary.eval.manifest.json",
        "gap_summary": root / "runs/gaps/summary.json",
        "misses": root / "runs/gaps/external-consensus-misses.jsonl",
        "output_json": root / "out/plan.json",
        "output_plan": root / "out/plan.sh",
    }
    write_json(paths["manifest"], default_manifest(root))
    write_jsonl(paths["train_jsonl"], [teacher_row("a"), teacher_row("b", scored=False)])
    write_jsonl(paths["eval_jsonl"], [teacher_row("eval", scored=True)])
    write_json(paths["train_manifest"], teacher_manifest(2, 1))
    write_json(paths["eval_manifest"], teacher_manifest(1, 1))
    write_json(paths["gap_summary"], gap_summary(consensus_misses=2))
    write_jsonl(paths["misses"], [miss_row("q1"), miss_row("q2")])
    touch(root / planner.DEFAULT_EVAL_JSONL)
    touch(root / planner.DEFAULT_HARD_EVAL_JSONL)
    return paths


def args_for(root: Path, paths: dict[str, Path], **overrides):
    parser = planner.build_parser()
    argv = [
        "--repo-root",
        str(root),
        "--default-embedder-manifest",
        str(paths["manifest"]),
        "--train-jsonl",
        str(paths["train_jsonl"]),
        "--eval-jsonl",
        str(paths["eval_jsonl"]),
        "--train-teacher-manifest",
        str(paths["train_manifest"]),
        "--eval-teacher-manifest",
        str(paths["eval_manifest"]),
        "--gap-summary-json",
        str(paths["gap_summary"]),
        "--consensus-misses-jsonl",
        str(paths["misses"]),
    ]
    args = parser.parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class PlanGuardedLongEmbedRepairCandidateTest(unittest.TestCase):
    def test_valid_plan_emits_summary_and_commented_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            args = args_for(root, paths)
            summary = planner.generate_plan(args, free_bytes=lambda path: 30 * GIB)
            commands = [
                planner.CommandPlan(
                    record["label"],
                    record["env"],
                    record["args"],
                    tuple(record.get("unset_env", [])),
                )
                for record in summary["planned_commands"]
            ]
            planner.write_plan(paths["output_plan"], commands)
            plan_text = paths["output_plan"].read_text(encoding="utf-8")

        self.assertFalse(summary["quality_claim"])
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["decision"], "ready")
        self.assertEqual(summary["teacher_signal"]["train"]["rows"], 2)
        self.assertEqual(summary["teacher_signal"]["train"]["rows_with_teacher_scores"], 1)
        self.assertIn("q4/fp16/rerank-overfetch=200", summary["compact_post_gate_requirement"]["profile"])
        shell = summary["planned_shell_command"]
        self.assertIn("ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw", shell)
        self.assertIn("EOS_INITIAL_ARTIFACT=", shell)
        self.assertIn("EOS_PROTECTED_LONGEMBED_EVAL_JSONL=", shell)
        self.assertIn("EOS_HARD_NEGATIVES_PER_QUERY=5", shell)
        self.assertTrue(all(line.startswith("# ") for line in plan_text.splitlines() if "ferrous-wheel run" in line))
        self.assertEqual(len(summary["planned_commands"]), 2)
        dense_command, compact_command = summary["planned_commands"]
        self.assertEqual(dense_command["label"], "guarded LongEmbed repair candidate")
        self.assertEqual(
            compact_command["label"],
            "mandatory compact q4/fp16/rerank-overfetch=200 post-gate",
        )
        compact_env = compact_command["env"]
        self.assertEqual(
            compact_env["EOS_GUARD_ANCHOR_SCOREBOARD"],
            summary["compact_post_gate_requirement"]["comparator_scoreboard"],
        )
        self.assertEqual(
            compact_env["EOS_GUARD_ANCHOR_SCOREBOARD"],
            str(root / "runs/current/compact-q4-fp16-overfetch200-scoreboard/scoreboard.json"),
        )
        self.assertEqual(compact_env["EOS_GUARD_CANDIDATE_DIR"], str(root / "runs" / args.run_id / "candidate"))
        self.assertEqual(compact_env["EOS_GUARD_SKIP_TRAIN"], "1")
        self.assertEqual(compact_env["EOS_SCOREBOARD_TURBOQUANT"], "1")
        self.assertEqual(compact_env["EOS_SCOREBOARD_TURBOQUANT_BITS"], "4")
        self.assertEqual(compact_env["EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH"], "200")
        self.assertEqual(compact_env["EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE"], "fp16")
        self.assertEqual(compact_env["EOS_GUARD_BASELINE"], "eos-turboquant-rerank")
        self.assertEqual(compact_env["EOS_GUARD_METHOD"], "turboquant_ip_b4_overfetch200_fp16_rerank")
        self.assertEqual(compact_env["EOS_GUARD_BITS"], "4")
        self.assertEqual(compact_env["EOS_GUARD_METRICS"], "ndcg_at_10,recall_at_100,total_compression_ratio")
        self.assertIn("total_compression_ratio", compact_command["shell"])
        self.assertIn("EOS_GUARD_SKIP_TRAIN=1", compact_command["shell"])
        self.assertIn("EOS_SCOREBOARD_TURBOQUANT=1", compact_command["shell"])
        command_lines = [line for line in plan_text.splitlines() if "ferrous-wheel run" in line]
        self.assertEqual(len(command_lines), 2)
        self.assertTrue(all(line.startswith("# ") for line in command_lines))

    def test_gap_summary_quality_claim_true_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            write_json(paths["gap_summary"], gap_summary(quality_claim=True))

            with self.assertRaisesRegex(planner.PlanError, "gap summary quality_claim"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

    def test_consensus_miss_quality_claim_true_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            write_jsonl(paths["misses"], [miss_row("q1", quality_claim=True)])

            with self.assertRaisesRegex(planner.PlanError, "consensus miss quality_claim"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

    def test_teacher_scores_length_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            write_jsonl(paths["train_jsonl"], [teacher_row("bad", scores=[0.9, 0.1])])
            write_json(paths["train_manifest"], teacher_manifest(1, 1))

            with self.assertRaisesRegex(planner.PlanError, "teacher_scores length"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

    def test_low_disk_records_blocker_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            summary = planner.generate_plan(
                args_for(root, paths, min_free_gb=15.0),
                free_bytes=lambda path: 14 * GIB,
            )

        self.assertIn("free disk below threshold", summary["blockers"])
        self.assertEqual(summary["decision"], "ready_when_disk_clear")
        self.assertEqual(summary["disk_free_bytes"], 14 * GIB)

    def test_default_manifest_missing_required_source_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            manifest = default_manifest(root)
            del manifest["source_release"]["package"]
            write_json(paths["manifest"], manifest)

            with self.assertRaisesRegex(planner.PlanError, "source_release.package"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

    def test_consensus_miss_rows_may_be_less_than_summary_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            write_json(paths["gap_summary"], gap_summary(consensus_misses=3, max_candidates=2))
            write_jsonl(paths["misses"], [miss_row("q1"), miss_row("q2")])
            summary = planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)

        misses = summary["per_query_gap_diagnosis"]["consensus_misses"]
        self.assertEqual(misses["rows"], 2)
        self.assertFalse(misses["matches_summary_count"])
        self.assertTrue(misses["row_count_within_summary"])

    def test_consensus_miss_rows_exceeding_summary_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture_repo(root)
            write_json(paths["gap_summary"], gap_summary(consensus_misses=1))
            write_jsonl(paths["misses"], [miss_row("q1"), miss_row("q2")])

            with self.assertRaisesRegex(planner.PlanError, "exceed summary"):
                planner.generate_plan(args_for(root, paths), free_bytes=lambda path: 30 * GIB)


if __name__ == "__main__":
    unittest.main()
