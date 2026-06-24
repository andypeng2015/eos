# eos-msmarco-hard-candidate-mining-v1 Report

## Outcome

Completed a bounded MS MARCO hard-candidate mining substrate from 5,000 deterministic stratified train queries. This replaces the prior deterministic random corpus-ID negative substrate with a candidate union from bounded BM25, bounded lexical overlap confusers, and deterministic random controls. No model training and no external teacher scoring were run.

Run root:

- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/`

Primary artifacts:

- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/manifest.json`
- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/artifacts/msmarco-passage.hard-candidates.train.jsonl`
- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/reports/candidate-source-audit.md`

Scratch helper:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1.py`

## Distillation

The substrate has exactly `5,000` train rows and `135,565` negative candidates. Rows preserve Eos text hard-negative compatibility fields (`source`, `query`, `positive`, `negatives`) plus provenance fields: `split`, `query_id`, `positive_doc_id`, `negative_doc_ids`, `row_id`, `candidate_sources`, `candidate_ranks`, `candidate_scores`, `strata`, and legal gates.

Legal gates are preserved in manifest and every row:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

## Tool Inventory

- In-repo BM25 exists: `go run ./cmd/eos eval-retrieval-bm25` and `go run ./cmd/eos mine-retrieval-hard-negatives`.
- Built-in BM25 miner emits older BM25-only training JSONL and does not preserve the requested multi-source candidate provenance, so a bounded scratch helper was used.
- Current-Eos vector tooling exists (`export-retrieval-vectors`, `eval-retrieval-vectors`, vector-cache commands), but no local current-Eos MS MARCO full-corpus/query cache was found under `runs/`.
- Current-Eos lane status: `blocked`; blocker is exact in the manifest: no existing local MS MARCO cache/index and descriptor forbids embedding all `8,841,823` docs.

## Query Stratification

Selection used deterministic round-robin over query-token-length bucket x positive-passage-token-length bucket, sorted by SHA256 over query ID, positive doc ID, and query text.

Selected strata totals:

- query length: `q_len_1_3=985`, `q_len_4_6=1175`, `q_len_7_10=1175`, `q_len_11_16=1028`, `q_len_17_plus=637`
- positive length: `p_len_1_30=1004`, `p_len_31_60=1176`, `p_len_61_100=1175`, `p_len_101_160=1042`, `p_len_161_plus=603`
- BM25 positive rank bucket: `rank_1=234`, `rank_2_10=666`, `rank_11_100=565`, `not_retrieved=3535`
- current-Eos positive rank bucket: `blocked=5000`

## Candidate Sources

BM25 is a documented bounded approximation, not an exact full BM25 index. It used lowercase alnum tokens, stopword removal, min token length `3`, BM25 `k1=0.9`, `b=0.4`, and retained query-vocabulary terms with `df <= 50,000` under `max_postings_sum=9,000,000`.

BM25 indexing diagnostics:

- documents scanned: `8,841,823`
- selected query vocab terms: `8,090`
- kept BM25 terms: `4,516`
- estimated postings retained: `8,999,068`
- docs with kept terms: `5,330,617`

Candidate source contribution counts:

- `bm25_bounded=112351`
- `lexical_overlap_bounded=112347`
- `random_tail_control=6390`
- `random_fill_control=16824`

Candidate count distribution:

- min/median/mean/p90/max: `16 / 32 / 27.113 / 32 / 32`
- distribution: `16=1463`, `17=11`, `18=8`, `19=9`, `20=11`, `21=8`, `22=9`, `23=4`, `24=14`, `25=9`, `26=7`, `27=4`, `28=5`, `29=4`, `30=6`, `31=6`, `32=3422`

## Leak Checks

- same-query train-positive negative leaks: `0`
- same-query dev-positive negative flags: `0`
- global dev-positive negative refs: `952`
- dev qrels used for candidate construction: `0`
- positive removal events during candidate union: `3018`
- dedup/source-overlap events: `164934`
- unresolved candidate texts: `0`

The `952` global dev-positive refs are audit flags only; they are not same-query dev leaks and dev qrels were not used for construction.

## Files Changed Or Created

Tracked/visible scratch files:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1.py`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1-report.md`

Ignored/generated run artifacts:

- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/manifest.json`
- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/artifacts/msmarco-passage.hard-candidates.train.jsonl`
- `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/reports/candidate-source-audit.md`

Inspected context included the descriptor-specified AGENTS/docs/reports/acquisition files, qrels/corpus/queries, and existing retrieval commands in `cmd/eos/main.go` / `runtime/retrieval_*.go`.

## Verification Commands And Results

Helper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1.py
```

Result: exited `0`.

Mining command:

```bash
python3 .tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1.py
```

Result: exited `0`; wrote `5000` rows.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_hard_candidate_mining.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .legal_gate.train_allowed_for_research == true and .counts.selected_queries == 5000 and .counts.output_rows == 5000 and .leak_audit.same_query_train_positive_negative_leaks == 0 and .leak_audit.dev_qrels_used_for_candidate_construction == 0' \
  runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/manifest.json
```

Result: `true`.

Candidate JSONL independent parse/count/leak check:

```bash
python3 - <<'PY'
# parsed all rows, asserted 5000 rows, 16-32 candidates, provenance and legal gates,
# and no negative doc ID in same-query train-positive qrels
PY
```

Result: `rows=5000`, no bad candidate counts, no missing provenance, no same-query train leaks, `legal_gate_rows=5000`, `dev_positive_flagged_refs=952`.

Line count:

```bash
wc -l runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/artifacts/msmarco-passage.hard-candidates.train.jsonl
```

Result: `5000`.

Final git status summary at report time:

```text
## main...origin/main
?? .tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1.py
?? .tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1-report.md
?? runtime/eos_encoder_invariance_diagnostic_v1_test.go
```

The untracked `runtime/eos_encoder_invariance_diagnostic_v1_test.go` existed before this descriptor and was not touched.

## Caveats Or Residual Risk

- BM25 is bounded by query-vocabulary df/postings caps; it is not exact full-corpus BM25 over all terms.
- Current-Eos lane is blocked due missing local MS MARCO vector cache/index; this descriptor intentionally did not embed the full corpus.
- `1,463` rows needed random fill to reach the 16-candidate floor because bounded BM25/lexical retrieval did not provide enough resolved non-positive candidates.
- Global dev-positive negative flags increased versus the old random substrate (`952` refs). They are explicit row-level flags for downstream filtering/relabeling before training.
- Deterministic full rerun was not repeated because it would rescan the 3.3 GB corpus several times; determinism is by stable SHA256 ordering and seeded random controls.

## Checkpoint Candidate

Yes, but only for the tiny scratch helper/report slice:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1.py`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mining-v1-report.md`

Do not checkpoint generated run artifacts or candidate JSONL.

## Recommended Next Action

Run Qwen3 and mxbai teacher scoring on `runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/artifacts/msmarco-passage.hard-candidates.train.jsonl`, then produce an agreement/filter report that handles the `952` global dev-positive flags before any training-row generation.

## Arbiter Next Action

Proceed to teacher scoring/agreement on this hard-candidate substrate, gated on `output_rows=5000`, legal gates false for release/commercial, same-query train/dev leak checks at zero, and explicit handling of globally dev-positive flagged negatives.
