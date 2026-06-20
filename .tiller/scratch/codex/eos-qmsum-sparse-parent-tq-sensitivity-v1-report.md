# eos-qmsum-sparse-parent-tq-sensitivity-v1 Report

## Outcome

Completed bounded QMSum sparse-parent TurboQuant sensitivity diagnostics using the existing sparse-enabled parent vector cache. No vector export was rerun.

Run directory:

- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/`

Key result: q4 reproduces the prior collapse exactly (`0.4966837425472634` nDCG@10), so the earlier q4 number is not a scoreboard assembly artifact. The collapse is bit-width sensitive: q5 partially recovers, q6/q8 are near dense, and q7 exactly preserves dense nDCG@10 while retaining `4.413793x` vector compression.

## Distillation

Dense sparse-parent nDCG@10 is `0.5464916791430152`. Direct q4 falls to `0.4966837425472634` (`-0.0498079366`) at `7.529412x` compression. This looks like a direct quantization-bit-width issue for the parent representation rather than an unavoidable representation/head failure, because q7 ties dense nDCG@10 with material compression and q6/q8 miss dense by only `-0.0026759737`.

The q4 loss is concentrated in a few queries: 6 queries lose, 1 improves, and 13 are unchanged. The two largest losses are `query_2` and `query_13`, where the relevant document drops from dense rank 1 to q4 rank 2; those two drops alone contribute `-0.738140492` nDCG@10, before one q4 improvement and smaller losses offset it.

## Metrics Table

| Row | bits | nDCG@10 | delta vs dense | recall@100 | vector bytes | compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense parent | 0 | 0.546491679 | 0.000000000 | 1.000000000 | 10240 | 1.000000x |
| q2 | 2 | 0.480189828 | -0.066301851 | 1.000000000 | 720 | 14.222222x |
| q3 | 3 | 0.502195910 | -0.044295769 | 1.000000000 | 1040 | 9.846154x |
| q4 | 4 | 0.496683743 | -0.049807937 | 1.000000000 | 1360 | 7.529412x |
| q5 | 5 | 0.529683920 | -0.016807759 | 1.000000000 | 1680 | 6.095238x |
| q6 | 6 | 0.543815705 | -0.002675974 | 1.000000000 | 2000 | 5.120000x |
| q7 | 7 | 0.546491679 | 0.000000000 | 1.000000000 | 2320 | 4.413793x |
| q8 | 8 | 0.543815705 | -0.002675974 | 1.000000000 | 2640 | 3.878788x |

Answer to the compact-preservation question: q7 preserves dense parent nDCG@10 exactly on this capped diagnostic while retaining `4.41x` compression. q6 and q8 are near-preserving but do not beat/tie dense. q4 does not preserve dense and is worse than q3 in this run.

## Per-query q4 delta summary

Worst q4 losses against dense:

| Query | Relevant doc | dense rank | q4 rank | dense nDCG@10 | q4 nDCG@10 | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| query_2 | doc_48 | 1 | 2 | 1.000000000 | 0.630929754 | -0.369070246 |
| query_13 | doc_156 | 1 | 2 | 1.000000000 | 0.630929754 | -0.369070246 |
| query_9 | doc_57 | 2 | 4 | 0.630929754 | 0.430676558 | -0.200253195 |
| query_15 | doc_97 | 6 | 8 | 0.356207187 | 0.315464877 | -0.040742310 |
| query_14 | doc_141 | 5 | 6 | 0.386852807 | 0.356207187 | -0.030645620 |
| query_16 | doc_77 | 5 | 6 | 0.386852807 | 0.356207187 | -0.030645620 |

Only positive q4 movement:

| Query | dense rank | q4 rank | delta |
| --- | ---: | ---: | ---: |
| query_11 | 10 | 7 | +0.044268507 |

Generated detailed files:

- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/per-query-deltas.tsv`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/per-query-delta-summary.json`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/metrics-table.tsv`

## Files inspected/generated

Inspected:

- `.tiller/scratch/codex/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-report.md`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/logs/eval-eos-sparse-encoder-parent-turboquant.log`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/eos-sparse-encoder-parent-dense.per-query.jsonl`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/eos-sparse-encoder-parent-turboquant.metrics.json`
- `cmd/eos/main.go`
- `runtime/retrieval_turboquant.go`
- `runtime/retrieval_eval.go`
- `datasets/longembed-official/qmsum/queries.jsonl`

Generated:

- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/eos-sparse-encoder-parent-turboquant-sensitivity.metrics.json`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/eos-sparse-encoder-parent-turboquant-sensitivity.metrics.tsv`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/eos-sparse-encoder-parent-turboquant-sensitivity.per-query.jsonl`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/eos-sparse-encoder-parent-turboquant-sensitivity.per-query-driver.metrics.json`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/per-query-deltas.tsv`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/per-query-delta-summary.json`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/metrics-table.tsv`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/per_query_tq_driver.go`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/logs/eval-sensitivity.log`
- `runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-20260620T131622Z/logs/per-query-driver.log`
- `.tiller/scratch/codex/eos-qmsum-sparse-parent-tq-sensitivity-v1-run-root.txt`
- `.tiller/scratch/codex/eos-qmsum-sparse-parent-tq-sensitivity-v1-report.md`

## Exact commands and results

Aggregate official CLI sensitivity sweep:

```bash
RUN_DIR="runs/eos-qmsum-sparse-parent-tq-sensitivity-v1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR/logs"
printf '%s\n' "$RUN_DIR" | tee .tiller/scratch/codex/eos-qmsum-sparse-parent-tq-sensitivity-v1-run-root.txt
/usr/bin/time -p go run ./cmd/eos eval-retrieval-vectors-turboquant \
  --dataset qmsum \
  --backend eos-sparse-encoder-parent-128d \
  --artifact eos-long-context-wedge-maxseq4096.mll \
  --doc-vectors runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/vectors/eos-sparse-encoder-parent/doc-vectors.jsonl \
  --query-vectors runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/vectors/eos-sparse-encoder-parent/query-vectors.jsonl \
  --bits 2,3,4,5,6,7,8 \
  --metrics-json "$RUN_DIR/eos-sparse-encoder-parent-turboquant-sensitivity.metrics.json" \
  --metrics-tsv "$RUN_DIR/eos-sparse-encoder-parent-turboquant-sensitivity.metrics.tsv" \
  datasets/longembed-official/qmsum
```

Result: completed in `real 47.18`; reproduced existing q4 nDCG@10 `0.4966837425472634` exactly and emitted q2 through q8 rows.

Per-query runtime driver:

```bash
/usr/bin/time -p go run "$RUN_DIR/per_query_tq_driver.go" "$RUN_DIR"
```

Result: completed in `real 46.81`; wrote 140 per-query rows (`20` queries x `7` bit widths). The driver's aggregate metrics match the official CLI metrics exactly for dense and q2-q8 nDCG@10.

Validation:

```bash
python3 - <<'PY'
# Parsed generated JSON, JSONL, and counted TSV rows.
PY
```

Results:

- JSON parsed: sensitivity metrics, per-query-driver metrics, per-query delta summary.
- JSONL parsed: `eos-sparse-encoder-parent-turboquant-sensitivity.per-query.jsonl`, `140` rows.
- TSV rows: `per-query-deltas.tsv` has `21` lines, `metrics-table.tsv` has `9` lines, official metrics TSV has `9` lines.

## Caveats / residual risk

- This is capped QMSum evidence over 20 docs and 20 queries; `recall@100=1.0` is not a meaningful large-corpus recall claim.
- `quality_claim=false` remains appropriate. This is a host-reference sparse parent vector-cache diagnostic, not production sparse runtime evidence.
- The per-query CLI flag is not exposed for `eval-retrieval-vectors-turboquant`, so per-query JSONL was produced by a run-local Go driver that calls the same exported runtime evaluator. Product source was not changed.
- The q7 exact tie is over this capped diagnostic only. It supports "q4 collapse is quantization setting sensitive" but should not be promoted without broader LongEmbed coverage.
- Worktree already had unrelated modified source: `scripts/eval_eos_long_context_wedge.fw`. I did not touch it.

## Checkpoint candidate

Yes, report/evidence checkpoint candidate. No product source files were changed. The only helper is run-local diagnostic code under the generated `runs/` directory.

## Arbiter next action

Record sparse-parent QMSum q4 as **NO-PROMOTE**. For this parent representation, q7 is the first measured direct TurboQuant width that preserves dense nDCG@10 on the capped QMSum diagnostic with material compression. Next useful action: either evaluate q6/q7/q8 on broader LongEmbed caches, or add/expose a per-query flag to the single-vector TurboQuant CLI if this diagnostic needs to become a repeatable harness surface.
