#!/usr/bin/env python3
"""Offline tests for MS MARCO Passage acquisition/audit script."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "acquire_msmarco_passage.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(rows)


def write_query_tar(source_root: Path) -> None:
    query_dir = source_root / "queries-src"
    write_tsv(query_dir / "queries.train.tsv", [("q1", "train one"), ("q2", "train two")])
    write_tsv(query_dir / "queries.dev.tsv", [("q3", "dev one"), ("q4", "dev two")])
    write_tsv(query_dir / "queries.eval.tsv", [("qe", "eval one")])
    with tarfile.open(source_root / "queries.tar.gz", "w:gz") as archive:
        for path in sorted(query_dir.iterdir()):
            archive.add(path, arcname=path.name)


def make_source_root(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    write_query_tar(source)
    write_tsv(
        source / "qrels.train.tsv",
        [
            ("q1", "0", "p1", "1"),
            ("q2", "0", "p2", "1"),
            ("q2", "0", "p3", "1"),
        ],
    )
    write_tsv(
        source / "qrels.dev.tsv",
        [
            ("q3", "0", "p2", "1"),
            ("q4", "0", "p4", "1"),
        ],
    )
    return source


def run_script(run_root: Path, source_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-root",
            str(run_root),
            "--source-root",
            str(source_root),
            "--no-head",
            *extra,
        ],
        check=True,
        text=True,
        capture_output=True,
    )


class AcquireMSMarcoPassageTest(unittest.TestCase):
    def test_bounded_probe_manifest_policy_counts_split_and_sha256s(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source_root(root)
            run_root = root / "run"

            completed = run_script(run_root, source)

            manifest = json.loads((run_root / "acquisition-manifest.json").read_text(encoding="utf-8"))
            split = json.loads((run_root / "reports" / "split-safety.json").read_text(encoding="utf-8"))
            sha_lines = (run_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            sha_map = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in sha_lines}

        self.assertIn("acquisition-manifest.json", completed.stdout)
        self.assertEqual(manifest["schema"], "eos.msmarco_passage_acquisition_manifest.v1")
        policy = manifest["license_terms_summary"]["engineering_policy"]
        self.assertTrue(policy["train_allowed_for_research"])
        self.assertFalse(policy["release_train_allowed"])
        self.assertFalse(policy["commercial_use_allowed"])
        self.assertFalse(policy["test_rows_train_allowed"])
        self.assertEqual(manifest["counts"]["probe"]["qrels_train"]["rows"], 3)
        self.assertEqual(manifest["counts"]["probe"]["qrels_dev"]["rows"], 2)
        self.assertEqual(manifest["counts"]["probe"]["queries"]["train"]["rows"], 2)
        self.assertEqual(manifest["counts"]["probe"]["queries"]["dev"]["rows"], 2)
        self.assertEqual(split["train_dev_query_overlap"], 0)
        self.assertEqual(split["train_dev_positive_doc_overlap"], 1)
        self.assertEqual(split["corpus_resolvability"]["status"], "not_run")
        self.assertIn("raw/queries.tar.gz", sha_map)
        self.assertIn("raw/qrels.train.tsv", sha_map)
        self.assertIn("raw/queries/queries.train.tsv", sha_map)

    def test_corpus_path_runs_resolvability_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source_root(root)
            corpus = root / "collection.tsv"
            write_tsv(corpus, [("p1", "passage one"), ("p2", "passage two"), ("p4", "passage four")])
            run_root = root / "run"

            run_script(
                run_root,
                source,
                "--corpus-path",
                str(corpus),
                "--min-free-bytes",
                "1",
            )

            split = json.loads((run_root / "reports" / "split-safety.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_root / "acquisition-manifest.json").read_text(encoding="utf-8"))
            sha_text = (run_root / "SHA256SUMS").read_text(encoding="utf-8")
            copied_corpus = run_root / "raw" / "collection.tsv"
            copied_corpus_sha = sha256_file(copied_corpus)

        corpus_audit = split["corpus_resolvability"]
        self.assertEqual(corpus_audit["status"], "run")
        self.assertEqual(corpus_audit["rows_scanned"], 3)
        self.assertEqual(corpus_audit["train_unresolved_positive_pids"], 1)
        self.assertEqual(corpus_audit["dev_unresolved_positive_pids"], 0)
        self.assertEqual(corpus_audit["train_unresolved_examples"], ["p3"])
        self.assertEqual(manifest["skipped_or_missing"]["corpus_text_rows"], "resolved/audited")
        self.assertIn(f"{copied_corpus_sha}  raw/collection.tsv", sha_text)


if __name__ == "__main__":
    unittest.main()
