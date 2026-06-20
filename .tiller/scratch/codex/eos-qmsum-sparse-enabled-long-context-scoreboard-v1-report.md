# eos-qmsum-sparse-enabled-long-context-scoreboard-v1 Report

## Outcome

Completed bounded QMSum sparse-enabled long-context scoreboard evidence.

Primary run directory:

- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/`

Gate decision: **NO-PROMOTE**.

The newly measured host-reference sparse-enabled parent encoder dense row beats Eos direct on capped QMSum without recall loss, but it does **not** beat the existing sparse token-span q4 threshold. The sparse-enabled q4 parent row regresses below both direct and token-span q4.

## Distillation

Useful evidence was produced for the current available sparse-enabled path:

- `export-sparse-encoder-vectors` works on real converted LongEmbed QMSum, consumes 4096-token documents, requires and applies the full encoder, and writes parent doc/query vector caches.
- The sparse-enabled parent dense row reaches nDCG@10 `0.546491679`, above direct `0.517955842`, with recall@100 `1.0`.
- The sparse-enabled parent q4 row reaches only nDCG@10 `0.496683743`, with recall@100 `1.0`.
- This is host-reference retrieval-cache evidence with `quality_claim=false`, not trained sparse LongEmbed proof, sealed runtime sparse inference, or production-quality evidence.

## Metrics Table

| Row | Source | bits | nDCG@10 | recall@100 | Gate note |
| --- | --- | ---: | ---: | ---: | --- |
| Eos direct single-vector | new run | 0 | 0.517955842 | 1.000000000 | direct threshold matched |
| Eos token-span dense 128/32 top2-mean | new run | 0 | 0.539227742 | 1.000000000 | below sparse encoder dense |
| Eos token-span q4 128/32 top2-mean | new run | 4 | 0.516269489 | 1.000000000 | below direct in this rerun |
| Eos sparse-enabled parent dense | new run | 0 | 0.546491679 | 1.000000000 | beats direct, misses prior q4 threshold |
| Eos sparse-enabled parent q4 | new run | 4 | 0.496683743 | 1.000000000 | regresses |
| Best direct/token-span fusion | reconstructed from new per-query rows | 4 | 0.536813696 | 1.000000000 | below sparse encoder dense and prior fusion threshold |
| Prior Eos token-span q4 threshold | existing compare | 4 | 0.551111330 | 1.000000000 | target threshold |
| Prior best Eos fusion threshold | existing compare | 4 | 0.554545464 | 1.000000000 | target threshold |
| External Qwen3 0.6B chunked q4 | existing compare | 4 | 0.876293470 | 1.000000000 | still far ahead |
| External mxbai-large chunked q4 | existing compare | 4 | 0.806946535 | 1.000000000 | still far ahead |

Threshold deltas:

- sparse-enabled parent dense vs direct threshold `0.517956`: `+0.028535679`
- sparse-enabled parent dense vs token-span q4 threshold `0.551111`: `-0.004619321`
- sparse-enabled parent q4 vs token-span q4 threshold `0.551111`: `-0.054427257`

## Files Inspected / Generated

Inspected:

- `docs/local-long-context-embedder-wedge.md`
- `docs/manta-embed-sota-avenues.md`
- `scripts/smoke_eos_sparse_long_context_retrieval.fw`
- `scripts/eval_eos_long_context_wedge.fw`
- `runs/eos-real-longembed-external-compare-v2-qmsum/comparison.json`
- `runs/eos-resumable-longembed-qmsum-doc40-span256-v1/comparison.json`
- `.tiller/scratch/codex/eos-resumable-longembed-qmsum-validation-v1-report.md`

Generated primary artifacts:

- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/comparison.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/comparison.tsv`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/manifest.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/direct-eos.metrics.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/eos-token-span-multivector-turboquant.metrics.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/eos-sparse-encoder-parent-dense.metrics.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/eos-sparse-encoder-parent-turboquant.metrics.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/direct-token-span-fusion.metrics.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/vectors/eos-token-span/manifest.json`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/vectors/eos-sparse-encoder-parent/manifest.json`

Also generated failed sealed-retarget evidence:

- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124842Z/logs/rename-embed.log`

## Exact Commands Run And Results

Initial sealed-artifact attempt:

```bash
EOS_REPO_ROOT=$PWD \
EOS_LC_WEDGE_RUN_DIR=runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124842Z \
EOS_LC_WEDGE_DATASET_NAME=qmsum \
EOS_LC_WEDGE_DATASET_DIR=datasets/longembed-official/qmsum \
EOS_LC_WEDGE_ARTIFACT=runs/current-release-qwen3-nf005-continuation-20260616T224102Z/candidate/eos-embed-v1.sealed.mll \
EOS_LC_WEDGE_RETARGET_MAX_SEQ=4096 \
EOS_LC_WEDGE_TOKEN_SPAN=128 \
EOS_LC_WEDGE_TOKEN_OVERLAP=32 \
EOS_LC_WEDGE_MAX_TOKENS=4096 \
EOS_LC_WEDGE_BITS=4 \
EOS_LC_WEDGE_PARENT_AGGREGATION=top2-mean \
EOS_LC_WEDGE_EXTERNAL_MODE=cache \
EOS_LC_WEDGE_SPARSE_ENCODER=1 \
EOS_LC_WEDGE_TOKEN_SPAN_RESUME=1 \
/usr/bin/time -p ferrous-wheel run scripts/eval_eos_long_context_wedge.fw
```

Result: failed during retarget because sealed artifact sibling files are not present:

```text
open .../candidate/eos-embed-v1.sealed.train.mll: no such file or directory
```

Successful expensive run used the trainable package layout, which provides the sibling files needed by the sparse/token-span exporter:

```bash
EOS_REPO_ROOT=$PWD \
EOS_LC_WEDGE_RUN_DIR=runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z \
EOS_LC_WEDGE_DATASET_NAME=qmsum \
EOS_LC_WEDGE_DATASET_DIR=datasets/longembed-official/qmsum \
EOS_LC_WEDGE_ARTIFACT=runs/current-release-qwen3-nf005-continuation-20260616T224102Z/candidate/eos-embed-v1.mll \
EOS_LC_WEDGE_RETARGET_MAX_SEQ=4096 \
EOS_LC_WEDGE_TOKEN_SPAN=128 \
EOS_LC_WEDGE_TOKEN_OVERLAP=32 \
EOS_LC_WEDGE_MAX_TOKENS=4096 \
EOS_LC_WEDGE_BITS=4 \
EOS_LC_WEDGE_OUTPUT_DIM=128 \
EOS_LC_WEDGE_BASELINE_DIM=1024 \
EOS_LC_WEDGE_PARENT_AGGREGATION=top2-mean \
EOS_LC_WEDGE_PER_QUERY=1 \
EOS_LC_WEDGE_EXTERNAL_MODE=cache \
EOS_LC_WEDGE_QWEN3_CACHE_ROOT=runs/external-vector-caches/qwen3-0.6b-qmsum-128d \
EOS_LC_WEDGE_MXBAI_CACHE_ROOT=runs/external-vector-caches/mxbai-large-qmsum-128d \
EOS_LC_WEDGE_SPARSE_ENCODER=1 \
EOS_LC_WEDGE_SPARSE_TOP_K=256 \
EOS_LC_WEDGE_TOKEN_SPAN_RESUME=1 \
EOS_LC_WEDGE_TOKEN_SPAN_PROGRESS_EVERY=1 \
/usr/bin/time -p ferrous-wheel run scripts/eval_eos_long_context_wedge.fw
```

Result: expensive direct, token-span, sparse-encoder export, and sparse-encoder scoring completed, then final assembly failed because an extra fusion recipe duplicated a default method name:

```text
EOS_LC_WEDGE_PARENT_SPAN_FUSION_EXTRA_RECIPES duplicate method name "direct_token_span_fusion_dense_protect_n1_rrf_k10_lambda05"
real 1218.50
```

Follow-up assembly:

- Parsed the completed new metrics/per-query files.
- Reconstructed direct/token-span fusion rows using the harness RRF/protect logic from `scripts/eval_eos_long_context_wedge.fw`.
- Copied external Qwen3/mxbai rows from existing `runs/eos-real-longembed-external-compare-v2-qmsum/comparison.json`.
- Wrote final `comparison.json`, `comparison.tsv`, and `manifest.json` into the new run directory.

## Verification

JSON/JSONL validation passed for:

- `comparison.json`
- `manifest.json`
- direct, token-span, sparse-encoder dense, sparse-encoder q4, and fusion metrics JSON
- token-span and sparse-encoder export manifests
- token-span progress sidecars
- direct, token-span, sparse-encoder, fusion per-query JSONL files
- token-span and sparse-encoder vector JSONL files

Validated row counts:

- direct per-query rows: `20`
- token-span per-query rows: `40` (`20` dense + `20` q4)
- sparse-encoder parent dense per-query rows: `20`
- fusion per-query rows: `60` (`3` recipes x `20` queries)
- token-span child vectors: `852`
- token-span query vectors: `20`
- sparse-encoder parent doc vectors: `20`
- sparse-encoder query vectors: `20`

Sparse encoder manifest checks:

- `method=experimental_sparse_encoder_host_reference`
- `evidence_level=retrieval_cache_host_reference_sparse_encoder`
- `quality_claim=false`
- `require_full_encoder=true`
- `full_encoder_applied=true`
- `dense_kv_materialized=true`
- `kv_decode=host_reference_decode`
- `sparse_top_k=256`
- max observed document tokens reached `4096`

## Caveats / Residual Risk

- This is capped QMSum smoke evidence over 20 documents and 20 queries; recall@100 is weak because the cutoff exceeds corpus size.
- The sparse-enabled path is host-reference and materializes dense K/V; its manifest explicitly reports `subquadratic=false` for the max document sparse plan.
- The sparse-enabled row is not a trained sparse LongEmbed encoder, not sealed sparse runtime inference, and not production quality evidence.
- The final comparison was assembled after a harness duplicate-recipe failure; the expensive measured Eos artifacts parse and are complete, but the external rows were not regenerated in this run directory.
- The rerun token-span q4 row (`0.516269489`) is below the existing threshold row (`0.551111330`), so the prior token-span q4 threshold remains the stronger Eos token-span comparison point.

## Checkpoint Candidate

Yes, report-only evidence checkpoint candidate.

No source files changed. The useful durable artifacts are generated run evidence and this scratch report.

## Arbiter Next Action

Record **NO-PROMOTE** for sparse-enabled QMSum. The next useful engineering action is to fix the long-context wedge harness so duplicate extra fusion recipes are ignored or overridden cleanly, then run a bounded sparse-encoder q4 quality investigation. The product path still needs trained sparse/compact long-context quality, because host-reference sparse parent q4 currently misses the token-span q4 threshold.
