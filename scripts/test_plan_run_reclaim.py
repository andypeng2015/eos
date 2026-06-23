#!/usr/bin/env python3
"""Dependency-free tests for run reclaim manifest planning."""

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

import plan_run_reclaim as reclaim


def make_repo(root: Path) -> Path:
    (root / ".git").mkdir()
    (root / "runs").mkdir()
    return root


def write_file(path: Path, text: str = "payload\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class PlanRunReclaimTest(unittest.TestCase):
    def test_manifest_normalizes_absolute_repo_paths_and_rejects_outside(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            run_dir = root / "runs" / "alpha"
            run_dir.mkdir()

            manifest = reclaim.create_manifest(
                paths=[str(run_dir.resolve())],
                output=None,
                reason="test reclaim",
                root=root,
            )

            self.assertFalse(manifest["quality_claim"])
            self.assertEqual(manifest["paths"], [{"path": "runs/alpha", "warnings": []}])
            with self.assertRaises(reclaim.ReclaimError):
                reclaim.create_manifest(
                    paths=[str((root / "assets" / "model.mll").resolve())],
                    output=None,
                    reason="test reclaim",
                    root=root,
                )
            with self.assertRaises(reclaim.ReclaimError):
                reclaim.create_manifest(
                    paths=[str(Path(tmp).parent / "outside")],
                    output=None,
                    reason="test reclaim",
                    root=root,
                )

    def test_dry_run_reports_bytes_and_does_not_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            write_file(root / "runs" / "alpha" / "artifact.txt", "12345")
            manifest = reclaim.create_manifest(
                paths=["runs/alpha"],
                output=None,
                reason="test reclaim",
                root=root,
            )

            summary = reclaim.execute_manifest(manifest)

            self.assertTrue((root / "runs" / "alpha" / "artifact.txt").exists())
            self.assertTrue(summary["dry_run"])
            self.assertFalse(summary["apply"])
            self.assertGreaterEqual(summary["paths"][0]["estimated_bytes"], 5)
            self.assertTrue(summary["paths"][0]["deletion_allowed"])
            self.assertGreaterEqual(summary["total_estimated_reclaim_bytes"], 5)

    def test_apply_requires_both_flags_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            write_file(root / "runs" / "alpha" / "artifact.txt")
            manifest = reclaim.create_manifest(
                paths=["runs/alpha"],
                output=None,
                reason="test reclaim",
                root=root,
            )

            with self.assertRaises(reclaim.ReclaimError):
                reclaim.execute_manifest(manifest, dry_run=False, apply=True)
            with self.assertRaises(reclaim.ReclaimError):
                reclaim.execute_manifest(
                    manifest,
                    dry_run=False,
                    yes_delete_approved_run_artifacts=True,
                )

            self.assertTrue((root / "runs" / "alpha" / "artifact.txt").exists())

    def test_apply_deletes_exact_temp_runs_fixture_with_both_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            write_file(root / "runs" / "alpha" / "artifact.txt")
            write_file(root / "runs" / "beta" / "artifact.txt")
            manifest = reclaim.create_manifest(
                paths=["runs/alpha"],
                output=None,
                reason="test reclaim",
                root=root,
            )

            summary = reclaim.execute_manifest(
                manifest,
                dry_run=False,
                apply=True,
                yes_delete_approved_run_artifacts=True,
            )

            self.assertFalse((root / "runs" / "alpha").exists())
            self.assertTrue((root / "runs" / "beta" / "artifact.txt").exists())
            self.assertTrue(summary["paths"][0]["deleted"])
            self.assertFalse(summary["paths"][0]["exists_after"])

    def test_cache_roots_rejected_unless_allowed_with_warning_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            (root / "runs" / "eos-vector-caches").mkdir()

            with self.assertRaises(reclaim.ReclaimError):
                reclaim.create_manifest(
                    paths=["runs/eos-vector-caches"],
                    output=None,
                    reason="test reclaim",
                    root=root,
                )

            manifest = reclaim.create_manifest(
                paths=["runs/eos-vector-caches"],
                output=None,
                reason="test reclaim",
                root=root,
                allow_cache_roots=True,
            )
            summary = reclaim.execute_manifest(
                manifest,
                allow_cache_roots=True,
            )

            self.assertIn("special cache root", manifest["paths"][0]["warnings"][0])
            self.assertIn("special cache root", summary["paths"][0]["warnings"][0])

    def test_missing_paths_reject_by_default_and_allow_missing_on_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            manifest = reclaim.create_manifest(
                paths=["runs/missing"],
                output=None,
                reason="test reclaim",
                root=root,
            )

            with self.assertRaises(reclaim.ReclaimError):
                reclaim.execute_manifest(manifest)
            summary = reclaim.execute_manifest(manifest, allow_missing=True)
            self.assertEqual(summary["paths"][0]["kind"], "missing")
            self.assertFalse(summary["paths"][0]["deletion_allowed"])

    def test_cli_writes_manifest_and_dry_run_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(Path(tmp))
            write_file(root / "runs" / "alpha" / "artifact.txt")
            output = root / "manifest.json"
            summary_json = root / "summary.json"
            summary_tsv = root / "summary.tsv"

            manifest_cmd = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "plan_run_reclaim.py"),
                    "manifest",
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--reason",
                    "test reclaim",
                    "--path",
                    str(root / "runs" / "alpha"),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            execute_cmd = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "plan_run_reclaim.py"),
                    "execute",
                    str(output),
                    "--output-json",
                    str(summary_json),
                    "--output-tsv",
                    str(summary_tsv),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(manifest_cmd.returncode, 0, manifest_cmd.stderr)
            self.assertEqual(execute_cmd.returncode, 0, execute_cmd.stderr)
            self.assertTrue((root / "runs" / "alpha" / "artifact.txt").exists())
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertFalse(summary["quality_claim"])
            self.assertIn("runs/alpha", summary_tsv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
