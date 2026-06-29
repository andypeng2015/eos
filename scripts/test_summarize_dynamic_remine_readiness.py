#!/usr/bin/env python3
"""Dependency-free tests for dynamic-remine readiness summarization."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_bge_selected_package_gate as bge_gate
import summarize_dynamic_remine_readiness as summarizer
from test_summarize_bge_selected_package_gate import write_complete_dataset, write_lines


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def legal_gates() -> dict:
    return {
        "train_allowed_for_research": True,
        "release_train_allowed": False,
        "commercial_use_allowed": False,
        "test_rows_train_allowed": False,
    }


def stagea_manifest(
    *,
    examples_seen: int = 256,
    examples_scored: int = 256,
    examples_written: int = 256,
    missing_examples: int = 0,
    import_score_rows: int = 1280,
    candidate_rows_scored: int = 1280,
    doc_rows: int = 1280,
    query_rows: int = 239,
    positive_top1_rate: float = 1.0,
    margin_min: float = 0.1,
    teacher_model_id: str = summarizer.DEFAULT_TEACHER_MODEL_ID,
) -> dict:
    return {
        "schema": "manta.vector_cache_teacher_scores.v1",
        "dataset": "msmarco-stagea-bge-real-v1-scale256",
        "teacher_model_id": teacher_model_id,
        "coverage": {
            "examples_seen": examples_seen,
            "examples_scored": examples_scored,
            "examples_written": examples_written,
            "missing_examples": missing_examples,
            "import_score_rows": import_score_rows,
            "candidate_rows_scored": candidate_rows_scored,
        },
        "vectors": {
            "doc_vector_rows": doc_rows,
            "query_vector_rows": query_rows,
        },
        "beir": {
            "corpus_rows": doc_rows,
            "query_rows": query_rows,
        },
        "scores": {
            "positive_top1_rate": positive_top1_rate,
            "margin": {"min": margin_min},
        },
    }


def vector_manifest(
    *,
    doc_rows: int = 1280,
    query_rows: int = 239,
    package_sha: str = bge_gate.DEFAULT_PACKAGE_SHA256,
    identity_sha: str = bge_gate.DEFAULT_IDENTITY_SHA256,
) -> dict:
    return {
        "schema": "manta.pretrained_bert_retrieval_vector_export.v1",
        "package_sha256": package_sha,
        "package_identity_sha256": identity_sha,
        "documents": doc_rows,
        "queries": query_rows,
        "written_documents": doc_rows,
        "written_queries": query_rows,
        "pooling": "cls",
        "normalization": "l2",
    }


def beir_manifest(*, doc_rows: int = 1280, query_rows: int = 239, rows: int = 256) -> dict:
    return {
        "dataset": "msmarco-stagea-bge-real-v1-scale256",
        "counts": {
            "rows": rows,
            "unique_docs": doc_rows,
            "unique_queries": query_rows,
            "qrel_pairs": rows,
        },
        "legal_gates": legal_gates(),
    }


def validation_summary() -> dict:
    return {
        "validation": {
            "rows_ge_128": True,
            "rows_full_256": True,
            "scoring_complete": True,
            "vector_rows_match_beir": True,
            "teacher_scores_len_5": True,
            "guide_missing_drop_0": True,
            "legal_test_rows_train_allowed_false": True,
        }
    }


def guide_manifest(
    *,
    rows_seen: int = 256,
    rows_emitted: int = 256,
    clean_agreement: int = 256,
    ambiguous: int = 0,
    conflict: int = 0,
    missing: int = 0,
    schema: str = summarizer.GUIDE_FILTER_SCHEMA,
    package_sha: str = bge_gate.DEFAULT_PACKAGE_SHA256,
    identity_sha: str = bge_gate.DEFAULT_IDENTITY_SHA256,
    gates: dict | None = None,
) -> dict:
    return {
        "schema": schema,
        "counts": {
            "rows_seen": rows_seen,
            "rows_emitted": rows_emitted,
            "clean_agreement": clean_agreement,
            "ambiguous_soft_only": ambiguous,
            "conflict": conflict,
            "missing_score_drop": missing,
        },
        "inputs": {
            "teacher_caches": {
                summarizer.DEFAULT_TEACHER_LABEL: {
                    "model_id": summarizer.DEFAULT_TEACHER_MODEL_ID,
                    "config": {
                        "package_sha256": package_sha,
                        "package_identity": identity_sha,
                    },
                }
            }
        },
        "legal_gates": gates or legal_gates(),
        "legal_gate_accounting": {"research_only_preserved": True},
        "quality_claim": False,
    }


def write_bge_ready(root: Path) -> None:
    write_complete_dataset(root, "scifact", 0.7, 0.9, 0.71, 0.89, 0.65, 0.85)
    write_complete_dataset(root, "nfcorpus", 0.3, 0.4, 0.31, 0.41, 0.25, 0.35)
    write_complete_dataset(root, "fiqa", 0.4, 0.6, 0.39, 0.59, 0.35, 0.55)


def write_stagea_ready(root: Path) -> None:
    write_json(root / "manifest.json", stagea_manifest())
    write_json(root / "artifacts" / "validation-summary.json", validation_summary())
    write_json(root / "vectors" / "manifest.json", vector_manifest())
    write_json(root / "beir" / "manifest.json", beir_manifest())
    write_json(root / "guide-filter-manifest.json", guide_manifest())


def build_fixture(tmp: Path) -> tuple[Path, Path, Path]:
    bge_root = tmp / "bge"
    stagea_root = tmp / "stagea"
    descriptor = tmp / "guided-negative-dynamic-remine-plan-v1.md"
    descriptor.write_text("# plan\n", encoding="utf-8")
    write_bge_ready(bge_root)
    write_stagea_ready(stagea_root)
    return bge_root, stagea_root, descriptor


class SummarizeDynamicRemineReadinessTest(unittest.TestCase):
    def test_ready_fixture_allows_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bge_root, stagea_root, descriptor = build_fixture(Path(tmp_dir))

            summary = summarizer.build_summary(
                stagea_root=stagea_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
                clock=lambda: "2026-06-29T00:00:00Z",
            )

        self.assertTrue(summary["launch_allowed"])
        self.assertFalse(summary["quality_claim"])
        self.assertFalse(summary["training_run"])
        self.assertEqual(summary["blockers"], [])
        self.assertTrue(summary["bge_gate"]["all_complete"])
        self.assertTrue(summary["stagea_bridge"]["ready"])
        self.assertTrue(summary["guide_filter"]["ready"])

    def test_fiqa_bge_incomplete_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bge_root, stagea_root, descriptor = build_fixture(tmp)
            for path in (bge_root / "fiqa" / "vectors" / "manifest.json", bge_root / "fiqa" / "eval" / "dense.metrics.json"):
                path.unlink()
            (bge_root / "fiqa" / "eval" / "turboquant-q8-q4.metrics.json").unlink()
            (bge_root / "fiqa" / "vectors" / "query-vectors.jsonl").unlink()
            write_lines(bge_root / "fiqa" / "vectors" / "doc-vectors.jsonl", 3)

            summary = summarizer.build_summary(
                stagea_root=stagea_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertIn("fiqa", summary["bge_gate"]["incomplete_datasets"])
        self.assertTrue(summary["stagea_bridge"]["ready"])
        self.assertTrue(summary["guide_filter"]["ready"])
        self.assertTrue(any("selected BGE gate incomplete" in blocker for blocker in summary["blockers"]))

    def test_stagea_score_coverage_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bge_root, stagea_root, descriptor = build_fixture(Path(tmp_dir))
            write_json(
                stagea_root / "manifest.json",
                stagea_manifest(examples_scored=255, missing_examples=1, import_score_rows=1275),
            )

            summary = summarizer.build_summary(
                stagea_root=stagea_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertFalse(summary["stagea_bridge"]["ready"])
        self.assertTrue(any("Stage A score coverage mismatch" in blocker for blocker in summary["stagea_bridge"]["blockers"]))
        self.assertTrue(any("Stage A import_score_rows mismatch" in blocker for blocker in summary["stagea_bridge"]["blockers"]))

    def test_guide_filter_legal_gate_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bge_root, stagea_root, descriptor = build_fixture(Path(tmp_dir))
            bad_gates = legal_gates()
            bad_gates["release_train_allowed"] = True
            write_json(
                stagea_root / "guide-filter-manifest.json",
                guide_manifest(conflict=1, missing=1, gates=bad_gates),
            )

            summary = summarizer.build_summary(
                stagea_root=stagea_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertFalse(summary["guide_filter"]["ready"])
        self.assertTrue(any("conflict not zero" in blocker for blocker in summary["guide_filter"]["blockers"]))
        self.assertTrue(any("missing_score_drop not zero" in blocker for blocker in summary["guide_filter"]["blockers"]))
        self.assertTrue(any("legal gates" in blocker for blocker in summary["guide_filter"]["blockers"]))

    def test_identity_or_teacher_mismatch_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bge_root, stagea_root, descriptor = build_fixture(Path(tmp_dir))
            write_json(stagea_root / "vectors" / "manifest.json", vector_manifest(identity_sha="wrong"))
            write_json(stagea_root / "manifest.json", stagea_manifest(teacher_model_id="wrong-teacher"))

            summary = summarizer.build_summary(
                stagea_root=stagea_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )

        self.assertFalse(summary["launch_allowed"])
        self.assertFalse(summary["stagea_bridge"]["identity_match"])
        self.assertTrue(any("vector package identity mismatch" in blocker for blocker in summary["stagea_bridge"]["blockers"]))
        self.assertTrue(any("teacher_model_id mismatch" in blocker for blocker in summary["stagea_bridge"]["blockers"]))

    def test_tsv_writer_emits_key_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bge_root, stagea_root, descriptor = build_fixture(tmp)
            summary = summarizer.build_summary(
                stagea_root=stagea_root,
                bge_gate_root=bge_root,
                descriptor=descriptor,
                datasets=["scifact", "nfcorpus", "fiqa"],
            )
            output_tsv = tmp / "summary.tsv"

            summarizer.write_tsv(output_tsv, summary)
            with output_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

        keyed = {(row["section"], row["key"]): row for row in rows}
        self.assertEqual(keyed[("summary", "launch_allowed")]["value"], "true")
        self.assertEqual(keyed[("stagea_bridge", "import_score_rows")]["value"], "1280")
        self.assertEqual(keyed[("stagea_bridge", "import_score_rows")]["expected"], "1280")
        self.assertEqual(keyed[("guide_filter", "clean_agreement")]["value"], "256")
        self.assertEqual(keyed[("guide_filter", "research_only_preserved")]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
