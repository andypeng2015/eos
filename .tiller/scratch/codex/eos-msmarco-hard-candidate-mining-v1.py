#!/usr/bin/env python3
import argparse
import collections
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it",
    "its", "may", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}


LEGAL_GATE = {
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "train_allowed_for_research": True,
    "requires_independent_legal_review_for_products_or_services": True,
    "policy_basis": "inherited from MS MARCO acquisition manifest; non-commercial research-only candidate mining substrate",
}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text, keep_stopwords=False):
    toks = TOKEN_RE.findall((text or "").lower())
    if keep_stopwords:
        return toks
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def bucket_count(n, edges, labels):
    for edge, label in zip(edges, labels):
        if n <= edge:
            return label
    return labels[-1]


def query_len_bucket(text):
    n = len(tokenize(text, keep_stopwords=True))
    return bucket_count(n, [3, 6, 10, 16], ["q_len_1_3", "q_len_4_6", "q_len_7_10", "q_len_11_16", "q_len_17_plus"])


def passage_len_bucket(text):
    n = len(tokenize(text, keep_stopwords=True))
    return bucket_count(n, [30, 60, 100, 160], ["p_len_1_30", "p_len_31_60", "p_len_61_100", "p_len_101_160", "p_len_161_plus"])


def rank_bucket(rank):
    if rank is None:
        return "not_retrieved"
    return bucket_count(rank, [1, 10, 100], ["rank_1", "rank_2_10", "rank_11_100", "rank_101_plus"])


def read_jsonl_map(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["_id"])] = row.get("text", "")
    return out


def read_qrels(path):
    by_query = collections.defaultdict(set)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        line_no = 1
        if header and header[:2] != ["query-id", "corpus-id"]:
            qid, _, docid, score = normalize_qrel_row(header)
            by_query[qid].add(docid)
            rows.append((qid, docid, float(score), line_no))
        for raw in reader:
            line_no += 1
            if not raw:
                continue
            qid, docid, score = normalize_qrel_row(raw)
            by_query[qid].add(docid)
            rows.append((qid, docid, float(score), line_no))
    return by_query, rows


def normalize_qrel_row(raw):
    if len(raw) == 4:
        qid, second, third, score = raw
        # BEIR TSV here is query-id, corpus-id, score. TREC qrels have 4 cols.
        if second == "0" and third != "1":
            return str(qid), str(third), score
        return str(qid), str(second), third
    if len(raw) == 3:
        qid, docid, score = raw
        return str(qid), str(docid), score
    raise ValueError(f"bad qrel row: {raw!r}")


def iter_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            title = row.get("title", "")
            if title:
                text = title + " " + text
            yield line_no, str(row["_id"]), text


def resolve_doc_texts(corpus_path, wanted_doc_ids, label):
    wanted = set(map(str, wanted_doc_ids))
    found = {}
    for _, doc_id, text in iter_corpus(corpus_path):
        if doc_id in wanted:
            found[doc_id] = text
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found), key=lambda x: (len(x), x))[:20]
    if missing:
        print(f"{label}: missing {len(wanted) - len(found)} docs, examples={missing}", file=sys.stderr)
    return found


def choose_stratified_queries(train_rows, train_by_query, query_text, positive_text, limit):
    eligible = []
    seen = set()
    for qid, docid, score, line_no in train_rows:
        if qid in seen:
            continue
        q = query_text.get(qid)
        p = positive_text.get(docid)
        if not q or not p:
            continue
        seen.add(qid)
        q_bucket = query_len_bucket(q)
        p_bucket = passage_len_bucket(p)
        eligible.append({
            "query_id": qid,
            "positive_doc_id": docid,
            "qrel_score": score,
            "source_qrels_line": line_no,
            "query": q,
            "positive": p,
            "base_stratum": f"{q_bucket}|{p_bucket}",
            "query_length_bucket": q_bucket,
            "positive_length_bucket": p_bucket,
            "known_train_positive_doc_ids": sorted(train_by_query[qid], key=lambda x: (len(x), x)),
            "select_hash": stable_hash(f"{qid}\t{docid}\t{q}"),
        })
    groups = collections.defaultdict(list)
    for row in eligible:
        groups[row["base_stratum"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: (r["select_hash"], r["query_id"], r["positive_doc_id"]))
    selected = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    selected.sort(key=lambda r: (r["select_hash"], r["query_id"]))
    return selected, eligible


def bm25_idf(total_docs, df):
    return math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))


def first_bm25_pass(corpus_path, query_vocab):
    df = collections.Counter()
    total_len = 0
    total_docs = 0
    max_numeric_doc_id = -1
    for _, doc_id, text in iter_corpus(corpus_path):
        total_docs += 1
        try:
            max_numeric_doc_id = max(max_numeric_doc_id, int(doc_id))
        except ValueError:
            pass
        toks = tokenize(text)
        total_len += len(toks)
        seen = set(toks) & query_vocab
        df.update(seen)
        if total_docs % 1_000_000 == 0:
            print(f"bm25 df pass docs={total_docs}", file=sys.stderr)
    return {
        "document_count": total_docs,
        "average_doc_length": total_len / max(total_docs, 1),
        "df": df,
        "max_numeric_doc_id": max_numeric_doc_id,
    }


def choose_bm25_terms(df, selected_query_terms, max_df, max_postings_sum):
    candidates = [t for t in selected_query_terms if 0 < df.get(t, 0) <= max_df]
    candidates.sort(key=lambda t: (df[t], t))
    kept = []
    postings_sum = 0
    for term in candidates:
        if postings_sum + df[term] > max_postings_sum and kept:
            continue
        kept.append(term)
        postings_sum += df[term]
    return set(kept), postings_sum


def build_bounded_postings(corpus_path, kept_terms):
    postings = {t: [] for t in kept_terms}
    doc_lengths = {}
    docs_with_kept_terms = 0
    for _, doc_id, text in iter_corpus(corpus_path):
        counts = collections.Counter(t for t in tokenize(text) if t in kept_terms)
        if not counts:
            continue
        dl = sum(counts.values())
        doc_lengths[doc_id] = max(dl, 1)
        docs_with_kept_terms += 1
        for term, tf in counts.items():
            postings[term].append((doc_id, tf, doc_lengths[doc_id]))
        if docs_with_kept_terms % 1_000_000 == 0:
            print(f"bm25 postings pass docs_with_terms={docs_with_kept_terms}", file=sys.stderr)
    return postings, doc_lengths, docs_with_kept_terms


def score_query(query_terms, postings, df, total_docs, avgdl, top_k):
    k1 = 0.9
    b = 0.4
    scores = collections.defaultdict(float)
    overlaps = collections.Counter()
    unique_terms = list(dict.fromkeys(t for t in query_terms if t in postings))
    for term in unique_terms:
        idf = bm25_idf(total_docs, df[term])
        for doc_id, tf, dl in postings[term]:
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            scores[doc_id] += idf * (tf * (k1 + 1.0)) / denom
            overlaps[doc_id] += 1
    bm25 = heapq.nlargest(top_k, scores.items(), key=lambda kv: (kv[1], kv[0]))
    lexical = heapq.nlargest(top_k, overlaps.items(), key=lambda kv: (kv[1], scores.get(kv[0], 0.0), kv[0]))
    return bm25, lexical


def deterministic_random_doc_ids(seed_text, total_docs, blocked, count):
    rng = random.Random(int(stable_hash(seed_text)[:16], 16))
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        doc_id = str(rng.randrange(total_docs))
        if doc_id in blocked or doc_id in out:
            continue
        out.append(doc_id)
    return out


def add_candidate(table, doc_id, source, rank, score=None):
    ent = table.setdefault(doc_id, {"sources": [], "ranks": {}, "scores": {}})
    if source not in ent["sources"]:
        ent["sources"].append(source)
    ent["ranks"][source] = rank
    if score is not None:
        ent["scores"][source] = score


def percentiles(values):
    if not values:
        return {}
    values = sorted(values)
    def pct(p):
        idx = int(round((len(values) - 1) * p))
        return values[idx]
    return {
        "min": values[0],
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def write_audit_md(path, manifest):
    lines = [
        "# MS MARCO Hard-Candidate Mining Audit",
        "",
        f"Run root: `{manifest['run_root']}`",
        "",
        "## Counts",
        "",
        f"- selected queries: `{manifest['counts']['selected_queries']}`",
        f"- output rows: `{manifest['counts']['output_rows']}`",
        f"- candidate negatives: `{manifest['counts']['negative_candidates_total']}`",
        f"- candidate count distribution: `{json.dumps(manifest['candidate_audit']['candidate_count_distribution'], sort_keys=True)}`",
        "",
        "## Sources",
        "",
    ]
    for key, value in sorted(manifest["candidate_audit"]["source_contribution_counts"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Leak Checks",
        "",
        f"- same-query train-positive negative leaks: `{manifest['leak_audit']['same_query_train_positive_negative_leaks']}`",
        f"- same-query dev-positive negative flags: `{manifest['leak_audit']['same_query_dev_positive_negative_flags']}`",
        f"- global dev-positive negative refs: `{manifest['leak_audit']['global_dev_positive_negative_refs']}`",
        f"- dev qrels used for construction: `{manifest['leak_audit']['dev_qrels_used_for_candidate_construction']}`",
        "",
        "## Blocked Lanes",
        "",
        f"- current Eos lane: `{manifest['blocked_lanes']['current_eos_lane_status']}`",
        f"- blocker: {manifest['blocked_lanes']['current_eos_lane_blocker']}",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only")
    ap.add_argument("--source-manifest", default="runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json")
    ap.add_argument("--run-root", default="")
    ap.add_argument("--selected-queries", type=int, default=5000)
    ap.add_argument("--candidate-min", type=int, default=16)
    ap.add_argument("--candidate-max", type=int, default=32)
    ap.add_argument("--bm25-top-k", type=int, default=48)
    ap.add_argument("--lexical-top-k", type=int, default=32)
    ap.add_argument("--random-tail", type=int, default=4)
    ap.add_argument("--max-term-df", type=int, default=50000)
    ap.add_argument("--max-postings-sum", type=int, default=9000000)
    args = ap.parse_args()

    start = time.time()
    dataset_root = Path(args.dataset_root)
    corpus_path = dataset_root / "corpus.jsonl"
    queries_path = dataset_root / "queries.jsonl"
    train_qrels_path = dataset_root / "qrels" / "train.tsv"
    dev_qrels_path = dataset_root / "qrels" / "dev.tsv"
    run_root = Path(args.run_root or f"runs/eos-msmarco-hard-candidate-mining-v1-{timestamp()}")
    artifacts_dir = run_root / "artifacts"
    reports_dir = run_root / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("loading qrels and queries", file=sys.stderr)
    train_by_query, train_rows = read_qrels(train_qrels_path)
    dev_by_query, dev_rows = read_qrels(dev_qrels_path)
    query_text = read_jsonl_map(queries_path)
    train_positive_doc_ids = {docid for docs in train_by_query.values() for docid in docs}
    dev_positive_doc_ids = {docid for docs in dev_by_query.values() for docid in docs}

    print(f"resolving train positive texts docs={len(train_positive_doc_ids)}", file=sys.stderr)
    positive_text = resolve_doc_texts(corpus_path, train_positive_doc_ids, "train positives")
    selected, eligible = choose_stratified_queries(train_rows, train_by_query, query_text, positive_text, args.selected_queries)
    if len(selected) != args.selected_queries:
        raise SystemExit(f"selected {len(selected)} queries, want {args.selected_queries}")

    selected_query_terms = set()
    query_terms_by_id = {}
    for row in selected:
        terms = tokenize(row["query"])
        query_terms_by_id[row["query_id"]] = terms
        selected_query_terms.update(terms)

    print(f"bm25 first pass query_terms={len(selected_query_terms)}", file=sys.stderr)
    first = first_bm25_pass(corpus_path, selected_query_terms)
    kept_terms, postings_sum = choose_bm25_terms(first["df"], selected_query_terms, args.max_term_df, args.max_postings_sum)
    print(f"bm25 kept_terms={len(kept_terms)} postings_sum_estimate={postings_sum}", file=sys.stderr)
    postings, _, docs_with_terms = build_bounded_postings(corpus_path, kept_terms)

    candidate_tables = {}
    bm25_positive_ranks = {}
    positive_removed = 0
    dedup_removed = 0
    for idx, row in enumerate(selected, 1):
        qid = row["query_id"]
        blocked = set(row["known_train_positive_doc_ids"])
        table = {}
        bm25, lexical = score_query(
            query_terms_by_id[qid],
            postings,
            first["df"],
            first["document_count"],
            first["average_doc_length"],
            max(args.bm25_top_k, args.lexical_top_k, args.candidate_max + 8),
        )
        pos_rank = None
        for rank, (doc_id, score) in enumerate(bm25, 1):
            if doc_id == row["positive_doc_id"]:
                pos_rank = rank
            if doc_id in blocked:
                positive_removed += 1
                continue
            before = len(table)
            add_candidate(table, doc_id, "bm25_bounded", rank, score)
            dedup_removed += int(len(table) == before)
        for rank, (doc_id, overlap) in enumerate(lexical, 1):
            if doc_id in blocked:
                positive_removed += 1
                continue
            before = len(table)
            add_candidate(table, doc_id, "lexical_overlap_bounded", rank, float(overlap))
            dedup_removed += int(len(table) == before)
        random_ids = deterministic_random_doc_ids(qid, first["document_count"], blocked | set(table), args.random_tail)
        for rank, doc_id in enumerate(random_ids, 1):
            add_candidate(table, doc_id, "random_tail_control", rank, None)
        if len(table) < args.candidate_min:
            fill_ids = deterministic_random_doc_ids("fill:" + qid, first["document_count"], blocked | set(table), args.candidate_min - len(table))
            for rank, doc_id in enumerate(fill_ids, 1):
                add_candidate(table, doc_id, "random_fill_control", rank, None)
        ordered = sorted(table.items(), key=lambda kv: (
            min(kv[1]["ranks"].get("bm25_bounded", 10_000), kv[1]["ranks"].get("lexical_overlap_bounded", 10_000), 10_000),
            kv[0],
        ))[:args.candidate_max]
        candidate_tables[qid] = collections.OrderedDict(ordered)
        bm25_positive_ranks[qid] = pos_rank
        if idx % 500 == 0:
            print(f"scored queries={idx}", file=sys.stderr)

    candidate_doc_ids = set()
    for table in candidate_tables.values():
        candidate_doc_ids.update(table.keys())
    print(f"resolving candidate texts docs={len(candidate_doc_ids)}", file=sys.stderr)
    candidate_text = resolve_doc_texts(corpus_path, candidate_doc_ids, "candidate docs")

    output_path = artifacts_dir / "msmarco-passage.hard-candidates.train.jsonl"
    source_counts = collections.Counter()
    source_first_counts = collections.Counter()
    count_dist = collections.Counter()
    strata_counts = collections.Counter()
    bm25_rank_buckets = collections.Counter()
    current_eos_rank_buckets = collections.Counter()
    global_dev_flags = 0
    same_query_dev_flags = 0
    train_leaks = 0
    rows_with_unresolved = 0
    rows = 0
    with output_path.open("w", encoding="utf-8") as out:
        for ordinal, row in enumerate(selected):
            qid = row["query_id"]
            table = candidate_tables[qid]
            neg_doc_ids = [doc_id for doc_id in table if doc_id in candidate_text]
            if len(neg_doc_ids) != len(table):
                rows_with_unresolved += 1
            neg_doc_ids = neg_doc_ids[:args.candidate_max]
            negatives = [candidate_text[doc_id] for doc_id in neg_doc_ids]
            for doc_id in neg_doc_ids:
                ent = table[doc_id]
                for src in ent["sources"]:
                    source_counts[src] += 1
                if ent["sources"]:
                    source_first_counts[ent["sources"][0]] += 1
                if doc_id in row["known_train_positive_doc_ids"]:
                    train_leaks += 1
                if doc_id in dev_positive_doc_ids:
                    global_dev_flags += 1
                if doc_id in dev_by_query.get(qid, set()):
                    same_query_dev_flags += 1
            bm25_bucket = rank_bucket(bm25_positive_ranks[qid])
            eos_bucket = "blocked"
            strata = {
                "query_length_bucket": row["query_length_bucket"],
                "positive_length_bucket": row["positive_length_bucket"],
                "bm25_positive_rank_bucket": bm25_bucket,
                "current_eos_positive_rank_bucket": eos_bucket,
            }
            for key, value in strata.items():
                strata_counts[f"{key}:{value}"] += 1
            bm25_rank_buckets[bm25_bucket] += 1
            current_eos_rank_buckets[eos_bucket] += 1
            count_dist[len(neg_doc_ids)] += 1
            row_id = stable_hash(f"hard-v1\t{qid}\t{row['positive_doc_id']}\t{','.join(neg_doc_ids)}")
            record = {
                "source": "msmarco-passage",
                "query": row["query"],
                "positive": row["positive"],
                "negatives": negatives,
                "split": "train",
                "query_id": qid,
                "positive_doc_id": row["positive_doc_id"],
                "negative_doc_ids": neg_doc_ids,
                "row_id": row_id,
                "source_qrels_line": row["source_qrels_line"],
                "qrel_score": row["qrel_score"],
                "candidate_sources": {doc_id: table[doc_id]["sources"] for doc_id in neg_doc_ids},
                "candidate_ranks": {doc_id: table[doc_id]["ranks"] for doc_id in neg_doc_ids},
                "candidate_scores": {doc_id: table[doc_id]["scores"] for doc_id in neg_doc_ids if table[doc_id]["scores"]},
                "candidate_dev_positive_flags": {
                    doc_id: {
                        "global_dev_positive": doc_id in dev_positive_doc_ids,
                        "same_query_dev_positive": doc_id in dev_by_query.get(qid, set()),
                    }
                    for doc_id in neg_doc_ids
                    if doc_id in dev_positive_doc_ids or doc_id in dev_by_query.get(qid, set())
                },
                "strata": strata,
                "known_train_positive_doc_ids": row["known_train_positive_doc_ids"],
                "release_train_allowed": False,
                "commercial_use_allowed": False,
                "train_allowed_for_research": True,
                "teacher_scores_ready": False,
            }
            out.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            rows += 1

    manifest_path = run_root / "manifest.json"
    audit_md_path = reports_dir / "candidate-source-audit.md"
    source_manifest_path = Path(args.source_manifest)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_sha = sha256_file(output_path)
    manifest = {
        "schema": "eos.msmarco_hard_candidate_mining.v1",
        "created_utc": utc_now(),
        "run_root": str(run_root),
        "legal_gate": LEGAL_GATE,
        "source_data": {
            "dataset_root": str(dataset_root),
            "source_manifest": str(source_manifest_path),
            "corpus_jsonl": str(corpus_path),
            "queries_jsonl": str(queries_path),
            "train_qrels": str(train_qrels_path),
            "dev_qrels": str(dev_qrels_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "corpus_sha256": source_manifest.get("counts", {}).get("beir", {}).get("corpus", {}).get("sha256"),
            "queries_sha256": source_manifest.get("counts", {}).get("beir", {}).get("queries", {}).get("sha256"),
            "train_qrels_sha256": source_manifest.get("counts", {}).get("beir", {}).get("qrels_train", {}).get("sha256"),
            "dev_qrels_sha256": source_manifest.get("counts", {}).get("beir", {}).get("qrels_dev", {}).get("sha256"),
        },
        "selection": {
            "method": "deterministic round-robin over query length x positive passage length strata, sorted by sha256(query_id, positive_doc_id, query)",
            "eligible_unique_train_queries": len(eligible),
            "selected_queries": len(selected),
            "base_strata_counts_selected": dict(collections.Counter(r["base_stratum"] for r in selected)),
        },
        "bm25": {
            "status": "bounded_approximation",
            "tokenizer": "lowercase alnum tokens, stopword removal, min token length 3",
            "k1": 0.9,
            "b": 0.4,
            "document_count": first["document_count"],
            "average_doc_length": first["average_doc_length"],
            "selected_query_vocab_terms": len(selected_query_terms),
            "kept_terms": len(kept_terms),
            "max_term_df": args.max_term_df,
            "max_postings_sum": args.max_postings_sum,
            "postings_sum_estimate": postings_sum,
            "docs_with_kept_terms": docs_with_terms,
            "positive_rank_buckets": dict(bm25_rank_buckets),
        },
        "blocked_lanes": {
            "current_eos_lane_status": "blocked",
            "current_eos_lane_blocker": "No existing local current-Eos MS MARCO passage corpus/query vector cache or index was found under runs/; descriptor forbids embedding all 8.8M corpus docs.",
        },
        "counts": {
            "selected_queries": len(selected),
            "output_rows": rows,
            "negative_candidates_total": sum(k * v for k, v in count_dist.items()),
            "train_qrel_rows_loaded": len(train_rows),
            "dev_qrel_rows_loaded_for_audit_only": len(dev_rows),
            "unique_train_positive_doc_ids": len(train_positive_doc_ids),
            "unique_dev_positive_doc_ids": len(dev_positive_doc_ids),
            "candidate_doc_ids_resolved": len(candidate_text),
            "candidate_doc_ids_requested": len(candidate_doc_ids),
        },
        "artifacts": {
            "hard_candidates_jsonl": {
                "path": str(output_path),
                "rows": rows,
                "sha256": output_sha,
            },
            "candidate_source_audit_md": {
                "path": str(audit_md_path),
            },
        },
        "candidate_audit": {
            "candidate_count_distribution": {str(k): v for k, v in sorted(count_dist.items())},
            "candidate_count_summary": percentiles(list(count_dist.elements())),
            "source_contribution_counts": dict(source_counts),
            "source_first_contribution_counts": dict(source_first_counts),
            "positive_removal_events": positive_removed,
            "dedup_overlap_events": dedup_removed,
            "rows_with_unresolved_candidate_text": rows_with_unresolved,
            "strata_counts": dict(strata_counts),
            "bm25_positive_rank_buckets": dict(bm25_rank_buckets),
            "current_eos_positive_rank_buckets": dict(current_eos_rank_buckets),
        },
        "leak_audit": {
            "same_query_train_positive_negative_leaks": train_leaks,
            "same_query_dev_positive_negative_flags": same_query_dev_flags,
            "global_dev_positive_negative_refs": global_dev_flags,
            "dev_qrels_used_for_candidate_construction": 0,
        },
        "tool_inventory": {
            "in_repo_bm25": "go run ./cmd/eos eval-retrieval-bm25 and mine-retrieval-hard-negatives exist, but built-in miner emits older BM25-only rows without multi-source provenance; scratch helper used for bounded multi-source substrate.",
            "current_eos_vectors": "export/eval vector-cache commands exist, but no local MS MARCO current-Eos full-corpus cache was found.",
            "lexical_confusers": "implemented in scratch helper from bounded query-vocabulary postings by token-overlap rank.",
            "random_tail": "implemented in scratch helper via deterministic sha256-seeded corpus-id sampling.",
        },
        "runtime": {
            "elapsed_seconds": time.time() - start,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_audit_md(audit_md_path, manifest)
    manifest["artifacts"]["candidate_source_audit_md"]["sha256"] = sha256_file(audit_md_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "manifest": str(manifest_path), "output": str(output_path), "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
