# eos-sparse-parent-real-longembed-export-q7-v1 Report

## Outcome

Completed the bounded real-task sparse-parent export and q4/q6/q7/q8 cache-only TurboQuant eval for capped LongEmbed 2WikiMQA and NarrativeQA.

Run directory:

- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/`

Decision: q7 is **not a uniform dense-preserving sparse-parent profile** on these real-task capped slices. It preserves or slightly beats dense on 2WikiMQA, but NarrativeQA repeats the q7 compactness failure pattern with a material dense gap.

## Distillation

Both requested missing sparse-parent caches were generated from the existing capped LongEmbed datasets with `quality_claim=false`.

- 2WikiMQA sparse-parent export used the existing 4096-token retargeted artifact and completed with `full_encoder_applied=true`.
- NarrativeQA sparse-parent export used the existing 1024-token retargeted artifact and completed with `full_encoder_applied=true`.
- 2WikiMQA q7: `0.592745958851` nDCG@10, `+0.003080315582` vs sparse-parent dense at `4.413793x` compression.
- NarrativeQA q7: `0.347400718135` nDCG@10, `-0.019063106101` vs sparse-parent dense at `4.413793x` compression.
- This is capped diagnostic evidence only: 20 docs / 20 queries per dataset; `quality_claim=false`; not release proof.

## Metrics Table

| dataset | row | bits | nDCG@10 | delta vs sparse-parent dense | recall@100 | vector bytes | compression |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | sparse-parent dense | 0 | 0.589665643269 | 0.000000000000 | 1.000000000000 | 10240 | 1.000000x |
| 2wikimqa | sparse-parent q4 | 4 | 0.554983379289 | -0.034682263980 | 1.000000000000 | 1360 | 7.529412x |
| 2wikimqa | sparse-parent q6 | 6 | 0.586199471173 | -0.003466172096 | 1.000000000000 | 2000 | 5.120000x |
| 2wikimqa | sparse-parent q7 | 7 | 0.592745958851 | +0.003080315582 | 1.000000000000 | 2320 | 4.413793x |
| 2wikimqa | sparse-parent q8 | 8 | 0.575567431084 | -0.014098212185 | 1.000000000000 | 2640 | 3.878788x |
| narrativeqa | sparse-parent dense | 0 | 0.366463824236 | 0.000000000000 | 1.000000000000 | 10240 | 1.000000x |
| narrativeqa | sparse-parent q4 | 4 | 0.347105553775 | -0.019358270461 | 1.000000000000 | 1360 | 7.529412x |
| narrativeqa | sparse-parent q6 | 6 | 0.330438887108 | -0.036024937128 | 1.000000000000 | 2000 | 5.120000x |
| narrativeqa | sparse-parent q7 | 7 | 0.347400718135 | -0.019063106101 | 1.000000000000 | 2320 | 4.413793x |
| narrativeqa | sparse-parent q8 | 8 | 0.333686827621 | -0.032776996615 | 1.000000000000 | 2640 | 3.878788x |

## Baseline Comparison

| dataset | baseline row | bits | nDCG@10 | recall@100 | compression |
| --- | --- | ---: | ---: | ---: | ---: |
| 2wikimqa | Eos direct single-vector | 0 | 0.739153782169 | 1.000000000000 | 0 |
| 2wikimqa | Eos token-span dense 128/32 top2-mean | 0 | 0.638982103397 | 1.000000000000 | 0 |
| 2wikimqa | Eos token-span q4 128/32 top2-mean | 4 | 0.603183540008 | 1.000000000000 | 1.447964x |
| 2wikimqa | best listed direct/token-span fusion | 4 | 0.744085102964 | 1.000000000000 | 1.063123x |
| 2wikimqa | external Qwen3 0.6B chunked dense/q4 | 0/4 | 1.000000000000 | 1.000000000000 | 0 / 0.680240x |
| 2wikimqa | external mxbai-large chunked dense/q4 | 0/4 | 1.000000000000 | 1.000000000000 | 0 / 0.680240x |
| narrativeqa | Eos direct single-vector | 0 | 0.428430713650 | 1.000000000000 | 0 |
| narrativeqa | Eos token-span dense 256/64 top2-mean | 0 | 0.372816837137 | 1.000000000000 | 0 |
| narrativeqa | Eos token-span q4 256/64 top2-mean | 4 | 0.363971642703 | 1.000000000000 | 12.047059x |
| narrativeqa | best listed direct/token-span fusion | 4 | 0.428430713650 | 1.000000000000 | 3.002933x |

Generated combined artifact:

- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/combined-real-longembed-sparse-parent-q4678-with-baselines.tsv`

## Files Inspected / Generated

Inspected:

- `.tiller/scratch/codex/eos-sparse-parent-q7-broader-longembed-sweep-v1-report.md`
- `.tiller/scratch/codex/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-report.md`
- `runs/eos-real-longembed-external-compare-v2-2wikimqa/comparison.json`
- `runs/eos-real-longembed-external-compare-v2-2wikimqa/manifest.json`
- `runs/eos-resumable-longembed-narrativeqa-doc20-span256-v1/comparison.json`
- `runs/eos-resumable-longembed-narrativeqa-doc20-span256-v1/manifest.json`
- `datasets/longembed-official/2wikimqa/dataset-manifest.json`
- `datasets/longembed-official/narrativeqa/dataset-manifest.json`
- `scripts/eval_eos_long_context_wedge.fw`
- `cmd/eos/main.go`

Generated:

- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/vectors/2wikimqa/eos-sparse-encoder-parent/manifest.json`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/vectors/2wikimqa/eos-sparse-encoder-parent/doc-vectors.jsonl`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/vectors/2wikimqa/eos-sparse-encoder-parent/query-vectors.jsonl`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/vectors/narrativeqa/eos-sparse-encoder-parent/manifest.json`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/vectors/narrativeqa/eos-sparse-encoder-parent/doc-vectors.jsonl`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/vectors/narrativeqa/eos-sparse-encoder-parent/query-vectors.jsonl`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/2wikimqa-sparse-parent-q4678.metrics.json`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/2wikimqa-sparse-parent-q4678.metrics.tsv`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/narrativeqa-sparse-parent-q4678.metrics.json`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/narrativeqa-sparse-parent-q4678.metrics.tsv`
- `runs/eos-sparse-parent-real-longembed-export-q7-v1-20260620T133121Z/combined-real-longembed-sparse-parent-q4678-with-baselines.tsv`
- `.tiller/scratch/codex/eos-sparse-parent-real-longembed-export-q7-v1.run-root`
- `.tiller/scratch/codex/eos-sparse-parent-real-longembed-export-q7-v1-report.md`

## Exact Commands And Results

Build:

```bash
RUN_ID=eos-sparse-parent-real-longembed-export-q7-v1-$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR/bin" "$RUN_DIR/logs"
printf '%s\n' "$RUN_DIR" > .tiller/scratch/codex/eos-sparse-parent-real-longembed-export-q7-v1.run-root
/usr/bin/time -p go build -trimpath -o "$RUN_DIR/bin/eos" ./cmd/eos
```

Result: success; `real 0.96`.

2WikiMQA sparse-parent export:

```bash
"$RUN_DIR/bin/eos" export-sparse-encoder-vectors \
  --dataset 2wikimqa \
  --output-dim 128 \
  --max-tokens 4096 \
  --min-observed-doc-tokens 256 \
  --top-k 256 \
  --bits 4 \
  --seed 5581486560434873699 \
  --max-docs 20 \
  --max-queries 20 \
  --manifest-json "$RUN_DIR/vectors/2wikimqa/eos-sparse-encoder-parent/manifest.json" \
  runs/eos-real-longembed-external-compare-v2-2wikimqa/eos-long-context-wedge-maxseq4096.mll \
  datasets/longembed-official/2wikimqa \
  "$RUN_DIR/vectors/2wikimqa/eos-sparse-encoder-parent"
```

Result: success; `real 527.41`.

Key manifest/export facts:

- `documents=20`, `queries=20`
- `quality_claim=false`
- `require_full_encoder=true`
- `full_encoder_applied=true`
- `dense_kv_materialized=true`
- `kv_decode=host_reference_decode`
- `sparse_top_k=256`
- `max_tokens=4096`

NarrativeQA sparse-parent export:

```bash
"$RUN_DIR/bin/eos" export-sparse-encoder-vectors \
  --dataset narrativeqa \
  --output-dim 128 \
  --max-tokens 1024 \
  --min-observed-doc-tokens 512 \
  --top-k 256 \
  --bits 4 \
  --seed 5581486560434873699 \
  --max-docs 20 \
  --max-queries 20 \
  --manifest-json "$RUN_DIR/vectors/narrativeqa/eos-sparse-encoder-parent/manifest.json" \
  runs/eos-resumable-longembed-narrativeqa-doc20-span256-v1/eos-long-context-wedge-maxseq1024.mll \
  datasets/longembed-official/narrativeqa \
  "$RUN_DIR/vectors/narrativeqa/eos-sparse-encoder-parent"
```

Result: success; `real 318.94`.

Key manifest/export facts:

- `documents=20`, `queries=20`
- `quality_claim=false`
- `require_full_encoder=true`
- `full_encoder_applied=true`
- `dense_kv_materialized=true`
- `kv_decode=host_reference_decode`
- `sparse_top_k=256`
- `max_tokens=1024`

2WikiMQA cache-only TurboQuant eval:

```bash
"$RUN_DIR/bin/eos" eval-retrieval-vectors-turboquant \
  --dataset 2wikimqa \
  --backend eos-sparse-encoder-parent-128d \
  --artifact eos-long-context-wedge-maxseq4096.mll \
  --doc-vectors "$RUN_DIR/vectors/2wikimqa/eos-sparse-encoder-parent/doc-vectors.jsonl" \
  --query-vectors "$RUN_DIR/vectors/2wikimqa/eos-sparse-encoder-parent/query-vectors.jsonl" \
  --bits 4,6,7,8 \
  --max-docs 20 \
  --max-queries 20 \
  --metrics-json "$RUN_DIR/2wikimqa-sparse-parent-q4678.metrics.json" \
  --metrics-tsv "$RUN_DIR/2wikimqa-sparse-parent-q4678.metrics.tsv" \
  datasets/longembed-official/2wikimqa
```

Result: success; `real 43.44`.

NarrativeQA cache-only TurboQuant eval:

```bash
"$RUN_DIR/bin/eos" eval-retrieval-vectors-turboquant \
  --dataset narrativeqa \
  --backend eos-sparse-encoder-parent-128d \
  --artifact eos-long-context-wedge-maxseq1024.mll \
  --doc-vectors "$RUN_DIR/vectors/narrativeqa/eos-sparse-encoder-parent/doc-vectors.jsonl" \
  --query-vectors "$RUN_DIR/vectors/narrativeqa/eos-sparse-encoder-parent/query-vectors.jsonl" \
  --bits 4,6,7,8 \
  --max-docs 20 \
  --max-queries 20 \
  --metrics-json "$RUN_DIR/narrativeqa-sparse-parent-q4678.metrics.json" \
  --metrics-tsv "$RUN_DIR/narrativeqa-sparse-parent-q4678.metrics.tsv" \
  datasets/longembed-official/narrativeqa
```

Result: success; `real 40.97`.

Validation:

```bash
python3 - <<'PY'
# parsed both sparse-parent manifests, both metrics JSON files,
# both per-dataset TSV files, vector JSONL row counts, and combined TSV
PY
```

Result:

- Validation passed.
- Both sparse-parent vector caches have 20 doc rows and 20 query rows.
- Both metrics JSON files have q4/q6/q7/q8 rows.
- Both per-dataset TSV files have 6 rows including header.
- Combined TSV has 27 rows including header.

## Caveats / Residual Risk

- This is capped LongEmbed diagnostic evidence only, not release proof.
- Both datasets are limited to 20 documents and 20 queries; `recall@100=1.0` is weak because the retrieval cutoff exceeds corpus size.
- Sparse-parent export is host-reference retrieval-cache evidence; manifests record dense K/V materialization and `subquadratic=false` at the max observed document length.
- The 2WikiMQA sparse-parent dense/q7 rows are below the existing Eos direct, token-span q4, and fusion rows, and far below external chunked baselines.
- NarrativeQA sparse-parent dense is below Eos direct and direct/token-span fusion; q7 is also below sparse-parent dense.

## Checkpoint Candidate

Yes, report/evidence checkpoint candidate. No source files changed.

## Arbiter Next Action

Record real-task sparse-parent q7 as **mixed / no-promote**: q7 preserves dense on capped 2WikiMQA but fails to preserve dense on capped NarrativeQA. Keep q7 as a profile candidate only for targeted sparse-parent contexts with dataset-specific validation, not a default compactness claim.
