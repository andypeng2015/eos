#!/usr/bin/env python3
"""Dependency-free tests for the Stage A BGE teacher bridge runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stagea_bge_teacher_bridge as bridge


def args_for(root: Path, **overrides):
    parser = bridge.build_parser()
    args = parser.parse_args(["--repo-root", str(root), "--run-root", str(root / "run")])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class StageABGETeacherBridgeTest(unittest.TestCase):
    def test_dry_run_builds_all_bridge_commands_with_stable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eos_bin = "runs/pretrained-bert-current-hf-parity-v1-20260629T090818Z/cmd/eos"
            args = args_for(root, dry_run=True, eos_bin=eos_bin)

            payload = bridge.execute(args)

        self.assertEqual(
            [command["label"] for command in payload["commands"]],
            [
                "build-stagea-rows",
                "materialize-beir",
                "export-imported-bge-vectors",
                "score-teacher-vector-cache",
                "build-guide-filter",
            ],
        )
        self.assertFalse(any(command["executed"] for command in payload["commands"]))
        self.assertEqual(payload["label"], "scale256")
        self.assertEqual(
            payload["paths"]["scored_rows"],
            str(root / "run" / "artifacts" / "msmarco-passage.stagea.scale256.bge-teacher-scored.jsonl"),
        )
        self.assertEqual(
            payload["paths"]["score_rows"],
            str(root / "run" / "artifacts" / "msmarco-passage.stagea.scale256.bge-teacher-score-rows.jsonl"),
        )
        self.assertEqual(
            payload["paths"]["guide_filtered_rows"],
            str(root / "run" / "artifacts" / "msmarco-passage.stagea.scale256.bge-guide-filtered.jsonl"),
        )

    def test_dry_run_threads_caps_package_prefixes_and_explicit_eos_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "bge.imported.mll"
            eos_bin = "runs/pretrained-bert-current-hf-parity-v1-20260629T090818Z/cmd/eos"
            args = args_for(
                root,
                dry_run=True,
                eos_bin=eos_bin,
                package=package,
                max_rows=32,
                negatives_per_query=4,
                candidate_pool_size=99,
                max_corpus_docs=1000,
                batch_size=8,
                progress_every=77,
            )

            payload = bridge.execute(args)

        build = payload["commands"][0]["argv"]
        export = payload["commands"][2]["argv"]
        self.assertEqual(export[0], eos_bin)
        self.assertEqual(export[1], "export-pretrained-bert-retrieval-vectors")
        self.assertNotEqual(export[:5], ["env", "GOWORK=off", "go", "run", "./cmd/eos"])
        self.assertEqual(export[export.index("--query-prefix") + 1], bridge.BGE_QUERY_PREFIX)
        self.assertEqual(export[export.index("--document-prefix") + 1], "")
        self.assertEqual(export[export.index("--batch-size") + 1], "8")
        self.assertEqual(export[export.index("--progress-every") + 1], "77")
        self.assertIn(str(package), export)
        self.assertEqual(build[build.index("--max-rows") + 1], "32")
        self.assertEqual(build[build.index("--candidate-pool-size") + 1], "99")
        self.assertEqual(build[build.index("--max-corpus-docs") + 1], "1000")
        self.assertEqual(payload["bge_teacher"]["package_sha256"], bridge.BGE_PACKAGE_SHA256)
        self.assertEqual(payload["bge_teacher"]["identity"], bridge.BGE_IDENTITY)

    def test_default_eos_command_disables_parent_go_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, dry_run=True)

            payload = bridge.execute(args)

        export = payload["commands"][2]["argv"]
        self.assertEqual(export[:5], ["env", "GOWORK=off", "go", "run", "./cmd/eos"])

    def test_dry_run_summary_has_no_quality_or_training_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, dry_run=True)

            payload = bridge.execute(args)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            summary_tsv_exists = (root / "run" / "summary.tsv").is_file()

        self.assertFalse(summary["quality_claim"])
        self.assertFalse(summary["training_run"])
        self.assertTrue(summary["dry_run"])
        self.assertEqual(payload, summary)
        self.assertFalse(any(command["executed"] for command in summary["commands"]))
        self.assertTrue(summary_tsv_exists)

    def test_guide_filter_command_includes_teacher_cache_model_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, dry_run=True)

            payload = bridge.execute(args)

        guide = payload["commands"][-1]["argv"]
        self.assertIn("scripts/build_retrieval_teacher_guide_filter.py", guide)
        self.assertEqual(
            guide[guide.index("--teacher-cache") + 1],
            f"{bridge.TEACHER_LABEL}={root / 'run' / 'artifacts' / 'msmarco-passage.stagea.scale256.bge-teacher-score-rows.jsonl'}",
        )
        config = json.loads(guide[guide.index("--teacher-config") + 1].split("=", 1)[1])
        self.assertEqual(config["model"], bridge.BGE_MODEL)
        self.assertEqual(config["query_prefix"], bridge.BGE_QUERY_PREFIX)
        self.assertEqual(config["document_prefix"], "")
        self.assertEqual(config["pooling"], "cls")
        self.assertEqual(config["normalization"], "l2")

    def test_label_default_and_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_args = args_for(root, dry_run=True, max_rows=17)
            custom_args = args_for(root, run_root=root / "custom-run", dry_run=True, max_rows=17, label="pilot")

            default_payload = bridge.execute(default_args)
            custom_payload = bridge.execute(custom_args)

        self.assertEqual(default_payload["label"], "scale17")
        self.assertIn("scale17", default_payload["paths"]["scored_rows"])
        self.assertEqual(custom_payload["label"], "pilot")
        self.assertIn("pilot", custom_payload["paths"]["scored_rows"])

    def test_skip_guide_filter_removes_final_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, dry_run=True, skip_guide_filter=True)

            payload = bridge.execute(args)

        self.assertEqual(
            [command["label"] for command in payload["commands"]],
            [
                "build-stagea-rows",
                "materialize-beir",
                "export-imported-bge-vectors",
                "score-teacher-vector-cache",
            ],
        )
        self.assertTrue(payload["options"]["skip_guide_filter"])

    def test_failed_runner_writes_failed_command_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, skip_guide_filter=True)

            def fail_first(spec: bridge.CommandSpec, _cwd: Path) -> bridge.CommandResult:
                spec.log_path.write_text("failed\n", encoding="utf-8")
                return bridge.CommandResult(9, stderr=f"failed {spec.label}\n")

            with self.assertRaises(SystemExit):
                bridge.execute(args, runner=fail_first)

            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["failed_commands"], ["build-stagea-rows"])
        self.assertEqual(summary["commands"][0]["returncode"], 9)
        self.assertTrue(summary["commands"][0]["executed"])


if __name__ == "__main__":
    unittest.main()
