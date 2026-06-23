#!/usr/bin/env python3
"""Dependency-free tests for the guarded product-wedge pipeline runner."""

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

import run_long_context_product_wedge_pipeline as pipeline


def make_repo(root: Path) -> Path:
    (root / ".git").mkdir()
    (root / "scripts").mkdir()
    return root


def args_for(root: Path, **overrides):
    parser = pipeline.build_parser()
    args = parser.parse_args(["--repo-root", str(root)])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[pipeline.CommandSpec] = []

    def __call__(self, spec: pipeline.CommandSpec, cwd: Path) -> pipeline.CommandResult:
        self.calls.append(spec)
        return pipeline.CommandResult(0, stdout=f"ran {spec.label}\n")


class FreeBytesSequence:
    def __init__(self, *values: int) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self, path: Path) -> int:
        self.calls += 1
        if self.values:
            return self.values.pop(0)
        raise AssertionError("free_bytes called more times than expected")


class RunLongContextProductWedgePipelineTest(unittest.TestCase):
    def test_default_dry_run_plans_cleanup_eval_summary_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            runner = FakeRunner()

            summary = pipeline.execute_pipeline(
                args_for(root),
                runner=runner,
                free_bytes=lambda path: 30 * 1024 * 1024 * 1024,
            )

        self.assertEqual(runner.calls, [])
        self.assertFalse(summary["quality_claim"])
        self.assertTrue(summary["dry_run"])
        self.assertFalse(summary["reclaim_command"]["executed"])
        self.assertEqual(len(summary["eval_commands"]), 2)
        self.assertFalse(any(item["executed"] for item in summary["eval_commands"]))
        self.assertFalse(summary["summary_command"]["executed"])
        shells = [item["shell"] for item in summary["planned_commands"]]
        self.assertTrue(any("plan_run_reclaim.py execute" in shell for shell in shells))
        self.assertTrue(any("ferrous-wheel run scripts/eval_eos_long_context_wedge.fw" in shell for shell in shells))
        self.assertTrue(any("summarize_long_context_wedge_comparisons.py" in shell for shell in shells))

    def test_cleanup_cannot_run_unless_both_cleanup_flags_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))

            with self.assertRaises(pipeline.PipelineError):
                pipeline.execute_pipeline(
                    args_for(root, apply_reclaim=True),
                    runner=FakeRunner(),
                    free_bytes=lambda path: 30 * 1024 * 1024 * 1024,
                )

            runner = FakeRunner()
            summary = pipeline.execute_pipeline(
                args_for(root, apply_reclaim=True, yes_delete_approved_run_artifacts=True),
                runner=runner,
                free_bytes=lambda path: 30 * 1024 * 1024 * 1024,
            )

        self.assertEqual([call.label for call in runner.calls], ["reclaim"])
        self.assertTrue(summary["reclaim_command"]["executed"])

    def test_eval_requires_both_flags_and_free_space_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))

            with self.assertRaises(pipeline.PipelineError):
                pipeline.execute_pipeline(
                    args_for(root, run_eval=True),
                    runner=FakeRunner(),
                    free_bytes=lambda path: 30 * 1024 * 1024 * 1024,
                )

            runner = FakeRunner()
            summary = pipeline.execute_pipeline(
                args_for(root, run_eval=True, yes_run_long_context_eval=True),
                runner=runner,
                free_bytes=lambda path: 30 * 1024 * 1024 * 1024,
            )

        self.assertEqual([call.label for call in runner.calls], ["eval:qmsum", "eval:2wikimqa"])
        self.assertFalse(summary["eval_blocked"])
        self.assertTrue(all(item["executed"] for item in summary["eval_commands"]))

    def test_low_disk_blocks_eval_when_reclaim_not_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            runner = FakeRunner()

            summary = pipeline.execute_pipeline(
                args_for(root, run_eval=True, yes_run_long_context_eval=True),
                runner=runner,
                free_bytes=lambda path: 14 * 1024 * 1024 * 1024,
            )

        self.assertEqual(runner.calls, [])
        self.assertTrue(summary["eval_blocked"])
        self.assertIn("free disk below threshold", " ".join(summary["blockers"]))

    def test_low_disk_after_reclaim_blocks_eval(self) -> None:
        gib = 1024 * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            runner = FakeRunner()
            free_bytes = FreeBytesSequence(10 * gib, 14 * gib)

            summary = pipeline.execute_pipeline(
                args_for(
                    root,
                    apply_reclaim=True,
                    yes_delete_approved_run_artifacts=True,
                    run_eval=True,
                    yes_run_long_context_eval=True,
                    min_free_gb=20,
                ),
                runner=runner,
                free_bytes=free_bytes,
            )

        self.assertEqual([call.label for call in runner.calls], ["reclaim"])
        self.assertEqual(free_bytes.calls, 2)
        self.assertTrue(summary["eval_blocked"])
        self.assertEqual(summary["free_bytes_before"], 10 * gib)
        self.assertEqual(summary["free_bytes_after_reclaim"], 14 * gib)
        self.assertEqual(summary["free_bytes_for_eval"], 14 * gib)
        self.assertIn("free disk below threshold after reclaim", summary["blockers"])
        self.assertFalse(any(item["executed"] for item in summary["eval_commands"]))

    def test_eval_allowed_when_post_reclaim_free_space_meets_threshold(self) -> None:
        gib = 1024 * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            runner = FakeRunner()
            free_bytes = FreeBytesSequence(10 * gib, 25 * gib)

            summary = pipeline.execute_pipeline(
                args_for(
                    root,
                    apply_reclaim=True,
                    yes_delete_approved_run_artifacts=True,
                    run_eval=True,
                    yes_run_long_context_eval=True,
                    min_free_gb=20,
                ),
                runner=runner,
                free_bytes=free_bytes,
            )

        self.assertEqual([call.label for call in runner.calls], ["reclaim", "eval:qmsum", "eval:2wikimqa"])
        self.assertEqual(free_bytes.calls, 2)
        self.assertFalse(summary["eval_blocked"])
        self.assertEqual(summary["free_bytes_for_eval"], 25 * gib)
        self.assertTrue(all(item["executed"] for item in summary["eval_commands"]))

    def test_summary_uses_expected_paths_and_executes_with_mock_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            for dataset in ("qmsum", "2wikimqa"):
                path = root / pipeline.run_dir_for(pipeline.DEFAULT_RUN_ROOT_PREFIX, dataset) / "comparison.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]\n", encoding="utf-8")
            runner = FakeRunner()

            summary = pipeline.execute_pipeline(
                args_for(root, run_summary=True),
                runner=runner,
                free_bytes=lambda path: 30 * 1024 * 1024 * 1024,
            )

        self.assertEqual([call.label for call in runner.calls], ["summary"])
        self.assertTrue(summary["summary_command"]["executed"])
        shell = summary["summary_command"]["shell"]
        self.assertIn("qmsum=runs/eos-current-default-long-context-product-wedge-v1-qmsum/comparison.json", shell)
        self.assertIn("2wikimqa=runs/eos-current-default-long-context-product-wedge-v1-2wikimqa/comparison.json", shell)

    def test_output_json_has_quality_claim_false_and_expected_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            output = root / "out" / "summary.json"

            cmd = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "run_long_context_product_wedge_pipeline.py"),
                    "--repo-root",
                    str(root),
                    "--output-json",
                    str(output),
                    "--min-free-gb",
                    "0",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(cmd.returncode, 0, cmd.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(summary["quality_claim"])
        self.assertEqual(
            summary["expected_comparison_paths"]["qmsum"],
            "runs/eos-current-default-long-context-product-wedge-v1-qmsum/comparison.json",
        )
        self.assertEqual(
            summary["expected_comparison_paths"]["2wikimqa"],
            "runs/eos-current-default-long-context-product-wedge-v1-2wikimqa/comparison.json",
        )


if __name__ == "__main__":
    unittest.main()
