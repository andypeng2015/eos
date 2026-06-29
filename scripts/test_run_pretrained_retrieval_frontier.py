#!/usr/bin/env python3
"""Dependency-free tests for the pretrained retrieval frontier runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pretrained_retrieval_frontier as frontier


def args_for(root: Path, **overrides):
    parser = frontier.build_parser()
    args = parser.parse_args(["--repo-root", str(root), "--run-root", str(root / "run")])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class PretrainedRetrievalFrontierTest(unittest.TestCase):
    def test_dry_run_builds_export_dense_and_turboquant_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, presets="e5-small-v2", datasets="scifact", dry_run=True)

            payload = frontier.execute(args)

        labels = [command["label"] for command in payload["commands"]]
        self.assertEqual(labels, ["export:e5-small-v2:scifact", "dense:e5-small-v2:scifact", "turboquant:e5-small-v2:scifact"])
        self.assertFalse(any(command["executed"] for command in payload["commands"]))
        turbo = payload["commands"][2]["argv"]
        self.assertIn("eval-retrieval-vectors-turboquant", turbo)
        self.assertEqual(turbo[turbo.index("--bits") + 1], "8,4")
        export = payload["commands"][0]["argv"]
        self.assertEqual(
            export[export.index("--qrels") + 1],
            str(root / "datasets/manta-embed-v1/raw/scifact/scifact/qrels/test.tsv"),
        )
        self.assertFalse(payload["quality_claim"])

    def test_dry_run_keeps_caps_on_export_and_eval_with_qrels_aware_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(
                root,
                presets="e5-small-v2",
                datasets="scifact",
                max_docs=200,
                max_queries=20,
                dry_run=True,
            )

            payload = frontier.execute(args)

        export = payload["commands"][0]["argv"]
        dense = payload["commands"][1]["argv"]
        self.assertEqual(export[export.index("--qrels") + 1], str(frontier.qrels_path_for(frontier.dataset_dir_for(args, "scifact"))))
        self.assertEqual(export[export.index("--max-docs") + 1], "200")
        self.assertEqual(export[export.index("--max-queries") + 1], "20")
        self.assertEqual(dense[dense.index("--max-docs") + 1], "200")
        self.assertEqual(dense[dense.index("--max-queries") + 1], "20")

    def test_dry_run_threads_empty_document_policy_to_export_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(
                root,
                presets="e5-small-v2",
                datasets="fiqa",
                empty_document_policy="qrels-placeholder",
                empty_document_placeholder="EMPTY",
                dry_run=True,
            )

            payload = frontier.execute(args)

        export = payload["commands"][0]["argv"]
        dense = payload["commands"][1]["argv"]
        self.assertEqual(export[export.index("--empty-document-policy") + 1], "qrels-placeholder")
        self.assertEqual(export[export.index("--empty-document-placeholder") + 1], "EMPTY")
        self.assertNotIn("--empty-document-policy", dense)

    def test_summary_collects_dense_q8_q4_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            cache = run_root / "vector-caches" / "e5-small-v2" / "scifact"
            metrics = run_root / "metrics" / "e5-small-v2" / "scifact"
            cache.mkdir(parents=True)
            metrics.mkdir(parents=True)
            (cache / "manifest.json").write_text(
                json.dumps({"native_dim": 384, "output_dim": 384, "document_vector_rows": 10}),
                encoding="utf-8",
            )
            (metrics / "dense.metrics.json").write_text(
                json.dumps({"quality": {"ndcg_at_10": 0.5, "recall_at_100": 0.75}}),
                encoding="utf-8",
            )
            (metrics / "turboquant.metrics.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "bits": 8,
                                "method": "turboquant_ip_b8",
                                "quality": {"ndcg_at_10": 0.49, "recall_at_100": 0.74},
                                "vector_bytes": 400,
                                "dense_vector_bytes": 15360,
                                "compression_ratio": 38.4,
                            },
                            {
                                "bits": 4,
                                "method": "turboquant_ip_b4",
                                "quality": {"ndcg_at_10": 0.48, "recall_at_100": 0.73},
                                "vector_bytes": 220,
                                "dense_vector_bytes": 15360,
                                "compression_ratio": 69.8,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = args_for(root, presets="e5-small-v2", datasets="scifact")

            rows = frontier.collect_summary(args, run_root)

        self.assertEqual([row["storage"] for row in rows], ["dense", "q8", "q4"])
        self.assertEqual(rows[0]["vector_bytes"], 15360)
        self.assertEqual(rows[1]["ndcg_at_10"], 0.49)

    def test_execute_creates_eval_output_dirs_before_runner_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, presets="e5-small-v2", datasets="scifact", skip_export=True)
            checked_labels: list[str] = []

            def assert_dirs(spec: frontier.CommandSpec, _cwd: Path) -> frontier.CommandResult:
                checked_labels.append(spec.label)
                self.assertTrue(spec.log_path.parent.is_dir())
                for index, value in enumerate(spec.argv[:-1]):
                    if value in frontier.OUTPUT_PATH_FLAGS:
                        self.assertTrue(Path(spec.argv[index + 1]).parent.is_dir())
                return frontier.CommandResult(0, stdout=f"ran {spec.label}\n")

            payload = frontier.execute(args, runner=assert_dirs)

        self.assertEqual(checked_labels, ["dense:e5-small-v2:scifact", "turboquant:e5-small-v2:scifact"])
        self.assertTrue(all(command["executed"] for command in payload["commands"]))

    def test_failed_commands_are_written_to_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for(root, presets="e5-small-v2", datasets="scifact", skip_export=True)

            def fail_first(spec: frontier.CommandSpec, _cwd: Path) -> frontier.CommandResult:
                return frontier.CommandResult(7, stderr=f"failed {spec.label}\n")

            with self.assertRaises(SystemExit):
                frontier.execute(args, runner=fail_first)

            payload = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["failed_commands"], ["dense:e5-small-v2:scifact"])
        self.assertEqual(payload["commands"][0]["returncode"], 7)
        self.assertTrue(payload["commands"][0]["executed"])


if __name__ == "__main__":
    unittest.main()
