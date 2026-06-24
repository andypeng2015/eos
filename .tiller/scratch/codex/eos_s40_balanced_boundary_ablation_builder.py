#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
import random
import re
from pathlib import Path


DATASETS = ("scifact", "nfcorpus", "fiqa")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_beir_jsonl(path):
    out = {}
    for _, row in read_jsonl(path):
        row_id = str(row.get("_id") or row.get("id") or "")
        if not row_id:
            continue
        title = row.get("title") or ""
        text = row.get("text") or ""
        combined = (title + "\n" + text).strip() if title else text.strip()
        out[row_id] = combined
    return out


def load_qrels_ids(path):
    qids = set()
    docids = set()
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if i == 0 and parts[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(parts) < 2:
                continue
            qids.add(parts[0])
            docids.add(parts[1])
    return qids, docids


def load_test_audit_sets(raw_root):
    test = {}
    for ds in DATASETS:
        ds_root = raw_root / ds / ds
        queries = load_beir_jsonl(ds_root / "queries.jsonl")
        corpus = load_beir_jsonl(ds_root / "corpus.jsonl")
        qids, docids = load_qrels_ids(ds_root / "qrels" / "test.tsv")
        pairs = set()
        for qid in qids:
            qtext = queries.get(qid, "")
            if not qtext:
                continue
            for did in docids:
                dtext = corpus.get(did, "")
                if dtext:
                    pairs.add((qtext, dtext))
        # More precise pass over qrels, avoiding full qid x docid cross-product.
        pairs = set()
        with open(ds_root / "qrels" / "test.tsv", "r", encoding="utf-8") as f:
            for i, raw in enumerate(f):
                parts = raw.strip().split()
                if not parts:
                    continue
                if i == 0 and parts[0].lower() in {"query-id", "query_id", "qid"}:
                    continue
                if len(parts) >= 2 and queries.get(parts[0]) and corpus.get(parts[1]):
                    pairs.add((queries[parts[0]], corpus[parts[1]]))
        test[ds] = {
            "query_ids": qids,
            "doc_ids": docids,
            "query_positive_pairs": pairs,
        }
    return test


def family(source):
    src = (source or "").split(":", 1)[0]
    return src if src in DATASETS else ""


def source_ids(row):
    src = row.get("source") or ""
    fam = family(src)
    qid = None
    did = None
    if fam == "scifact":
        # scifact:train:<qid>:<docid>:top10_competitor
        m = re.match(r"^scifact:[^:]+:([^:]+):([^:]+):", src)
        if m:
            qid, did = m.group(1), m.group(2)
    elif fam == "nfcorpus":
        # nfcorpus:train:<qid>:<docid>:top10_competitor
        m = re.match(r"^nfcorpus:[^:]+:([^:]+):([^:]+):", src)
        if m:
            qid, did = m.group(1), m.group(2)
    elif fam == "fiqa":
        m = re.match(r"^fiqa:[^:]+:([^:]+):([^:]+):", src)
        if m:
            qid, did = m.group(1), m.group(2)
    return fam, qid, did


def row_id_sets(row):
    fam, qid, did = source_ids(row)
    qids = set()
    docids = set()
    if qid:
        qids.add(str(qid))
    if did:
        docids.add(str(did))
    if row.get("query_id") is not None:
        qids.add(str(row.get("query_id")))
    for key in ("positive_id", "doc_id", "corpus_id"):
        if row.get(key) is not None:
            docids.add(str(row.get(key)))
    for key in ("negative_ids", "doc_ids"):
        value = row.get(key)
        if isinstance(value, list):
            docids.update(str(v) for v in value if v is not None)
    return fam, qids, docids


def row_is_test_overlap(row, test_sets):
    fam, qids, docids = row_id_sets(row)
    if not fam:
        fam = family(row.get("source"))
    if not fam:
        return False, []
    reasons = []
    audit = test_sets[fam]
    for qid in sorted(qids):
        if qid in audit["query_ids"]:
            reasons.append(f"test_query_id:{qid}")
    for did in sorted(docids):
        if did in audit["doc_ids"]:
            reasons.append(f"test_doc_id:{did}")
    if (row.get("query", ""), row.get("positive", "")) in audit["query_positive_pairs"]:
        reasons.append("test_query_positive_text_pair")
    return bool(reasons), reasons


def stable_key(row):
    body = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_candidates(paths, test_sets):
    rows = {ds: [] for ds in DATASETS}
    rejected = collections.Counter()
    seen = set()
    explicit_id_rows = 0
    for path in paths:
        for _, row in read_jsonl(path):
            ds = family(row.get("source"))
            if ds not in DATASETS:
                continue
            if "query" not in row or "positive" not in row or "negatives" not in row:
                rejected[(ds, "missing_required_fields")] += 1
                continue
            fam, qids, docids = row_id_sets(row)
            if qids or docids:
                explicit_id_rows += 1
            overlap, reasons = row_is_test_overlap(row, test_sets)
            if overlap:
                for reason in reasons:
                    rejected[(ds, reason)] += 1
                continue
            key = stable_key(row)
            if key in seen:
                rejected[(ds, "duplicate_exact_row")] += 1
                continue
            seen.add(key)
            row = dict(row)
            row["_ablation_input_path"] = str(path)
            rows[ds].append(row)
    return rows, rejected, explicit_id_rows


def strip_internal(row):
    row = dict(row)
    row.pop("_ablation_input_path", None)
    return row


def summarize_rows(rows, test_sets):
    by_source = collections.Counter()
    by_family = collections.Counter()
    neg_dist = collections.Counter()
    teacher_rows = 0
    overlap_rows = 0
    explicit_id_rows = 0
    input_paths = collections.Counter()
    for row in rows:
        src = row.get("source", "")
        ds = family(src)
        by_family[ds] += 1
        by_source[src] += 1
        neg_dist[str(len(row.get("negatives") or []))] += 1
        if "teacher_scores" in row or "teacher_score" in row:
            teacher_rows += 1
        overlap, _ = row_is_test_overlap(row, test_sets)
        if overlap:
            overlap_rows += 1
        fam, qids, docids = row_id_sets(row)
        if qids or docids:
            explicit_id_rows += 1
        if row.get("_ablation_input_path"):
            input_paths[row["_ablation_input_path"]] += 1
    return {
        "rows": len(rows),
        "source_family_counts": dict(sorted(by_family.items())),
        "source_counts_top": by_source.most_common(40),
        "teacher_score_rows": teacher_rows,
        "teacher_score_fraction": teacher_rows / len(rows) if rows else 0.0,
        "negative_count_distribution": dict(sorted(neg_dist.items(), key=lambda kv: int(kv[0]))),
        "test_overlap_rows_after_filter": overlap_rows,
        "explicit_source_id_rows": explicit_id_rows,
        "input_paths": dict(input_paths.most_common()),
    }


def choose_balanced(base_rows, per_dataset, seed):
    rng = random.Random(seed)
    out = []
    for ds in DATASETS:
        rows = list(base_rows[ds])
        rng.shuffle(rows)
        if len(rows) < per_dataset:
            raise SystemExit(f"not enough audited rows for {ds}: have {len(rows)}, need {per_dataset}")
        out.extend(rows[:per_dataset])
    rng.shuffle(out)
    return out


def choose_boundary(path, limit, test_sets, seed, source_prefix=None):
    rng = random.Random(seed)
    accepted = []
    rejected = collections.Counter()
    for _, row in read_jsonl(path):
        ds = family(row.get("source"))
        if ds not in DATASETS:
            rejected[("unknown", "source_family")] += 1
            continue
        if source_prefix and not (row.get("source") or "").startswith(source_prefix):
            rejected[(ds, "source_prefix_skip")] += 1
            continue
        overlap, reasons = row_is_test_overlap(row, test_sets)
        if overlap:
            for reason in reasons:
                rejected[(ds, reason)] += 1
            continue
        accepted.append(row)
    rng.shuffle(accepted)
    return accepted[:limit], rejected, len(accepted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--per-dataset", type=int, default=10000)
    ap.add_argument("--boundary-limit", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260624)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    data_dir = run_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_root = repo / "datasets/manta-embed-v1/raw"
    test_sets = load_test_audit_sets(raw_root)

    base_inputs = [
        repo / "datasets/manta-embed-v1/processed/train-mixed-pretrain-plus-beir.jsonl",
        repo / "datasets/manta-embed-v1/processed/shipping-weighted-hard-negatives.jsonl",
        repo / "datasets/manta-embed-v1/processed/train-hard-negatives-plus-model.jsonl",
    ]
    base_rows, base_rejected, explicit_id_rows = load_candidates(base_inputs, test_sets)
    arm_a = choose_balanced(base_rows, args.per_dataset, args.seed)

    nf_boundary_path = repo / "runs/eos-s40-nfcorpus-compact-rank-mining-v1-20260623T022251Z/data/nfcorpus-train.compact-hard-negatives.jsonl"
    nf_fallback_boundary_path = repo / "runs/eos-nfcorpus-compact-top10-diff-and-non-test-repair-20260617T000000Z/data/nfcorpus-top10-competitor-repair.train.jsonl"
    scifact_boundary_path = repo / "runs/eos-compact-non-test-mining-v1-20260617T145423Z/data/scifact-train.compact-hard-negatives.jsonl"
    nf_boundary, nf_rejected, nf_available = choose_boundary(nf_boundary_path, args.boundary_limit, test_sets, args.seed + 1)
    nf_boundary_selected_path = nf_boundary_path
    nf_boundary_source_note = "descriptor-specified compact rank-mining file"
    nf_fallback_rejected = collections.Counter()
    nf_fallback_available = 0
    if len(nf_boundary) < 64:
        nf_fallback, nf_fallback_rejected, nf_fallback_available = choose_boundary(
            nf_fallback_boundary_path,
            args.boundary_limit,
            test_sets,
            args.seed + 4,
            source_prefix="nfcorpus:train:",
        )
        if len(nf_fallback) >= 64:
            nf_boundary = nf_fallback
            nf_boundary_selected_path = nf_fallback_boundary_path
            nf_boundary_source_note = "fallback train-only NFCorpus top10/compact repair file; descriptor-specified file had fewer than 64 audited rows"
    scifact_boundary, scifact_rejected, scifact_available = choose_boundary(scifact_boundary_path, args.boundary_limit, test_sets, args.seed + 2)

    # Replace equal-count base rows from the matching family so arm B stays close to 30k.
    arm_b = list(arm_a)
    for ds, boundary_rows in (("nfcorpus", nf_boundary), ("scifact", scifact_boundary)):
        remove = len(boundary_rows)
        kept = []
        removed = 0
        for row in arm_b:
            if removed < remove and family(row.get("source")) == ds:
                removed += 1
                continue
            kept.append(row)
        arm_b = kept + boundary_rows
    random.Random(args.seed + 3).shuffle(arm_b)

    arm_a_path = data_dir / "arm_a_balanced_clean.train.jsonl"
    arm_b_path = data_dir / "arm_b_balanced_boundary.train.jsonl"
    write_jsonl(arm_a_path, [strip_internal(r) for r in arm_a])
    write_jsonl(arm_b_path, [strip_internal(r) for r in arm_b])

    manifest = {
        "schema": "eos.s40_balanced_boundary_ablation.data_manifest.v1",
        "run_root": str(run_root),
        "seed": args.seed,
        "per_dataset_target": args.per_dataset,
        "base_inputs": [str(p) for p in base_inputs],
        "base_candidate_counts_after_audit": {ds: len(base_rows[ds]) for ds in DATASETS},
        "base_rejections": {f"{k[0]}:{k[1]}": v for k, v in sorted(base_rejected.items())},
        "base_explicit_source_id_rows": explicit_id_rows,
        "test_audit": {
            ds: {
                "test_query_ids": len(test_sets[ds]["query_ids"]),
                "test_doc_ids": len(test_sets[ds]["doc_ids"]),
                "test_query_positive_text_pairs": len(test_sets[ds]["query_positive_pairs"]),
            }
            for ds in DATASETS
        },
        "arms": {
            "arm_a_balanced_clean": {
                "path": str(arm_a_path),
                **summarize_rows(arm_a, test_sets),
            },
            "arm_b_balanced_boundary": {
                "path": str(arm_b_path),
                "boundary_policy": "replace equal-count base rows with audited train-only compact boundary rows",
                "boundary_requested_per_dataset": args.boundary_limit,
                "boundary_sources": {
                    "nfcorpus": str(nf_boundary_selected_path),
                    "nfcorpus_descriptor_specified": str(nf_boundary_path),
                    "nfcorpus_fallback_candidate": str(nf_fallback_boundary_path),
                    "scifact": str(scifact_boundary_path),
                },
                "boundary_source_notes": {
                    "nfcorpus": nf_boundary_source_note,
                    "scifact": "train-only compact hard negatives",
                },
                "boundary_available_after_audit": {
                    "nfcorpus": nf_available,
                    "nfcorpus_fallback": nf_fallback_available,
                    "scifact": scifact_available,
                },
                "boundary_selected": {
                    "nfcorpus": len(nf_boundary),
                    "scifact": len(scifact_boundary),
                },
                "boundary_rejections": {
                    **{f"nfcorpus_descriptor:{k[0]}:{k[1]}": v for k, v in sorted(nf_rejected.items())},
                    **{f"nfcorpus_fallback:{k[0]}:{k[1]}": v for k, v in sorted(nf_fallback_rejected.items())},
                    **{f"scifact:{k[0]}:{k[1]}": v for k, v in sorted(scifact_rejected.items())},
                },
                **summarize_rows(arm_b, test_sets),
            },
        },
    }
    manifest_path = data_dir / "balanced-boundary-data-manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
