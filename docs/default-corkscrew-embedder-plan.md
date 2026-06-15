# Default CorkScrewDB Embedder Plan

This plan is scoped to `eos-embed-v1`, the small sealed local default embedder candidate for CorkScrewDB, with reproducible local training, retrieval scoring, and TurboQuant-first serving gates. The shipping alias is `corkscrewdb-default-embedder` once the promotion gates pass. This plan does not claim state-of-the-art quality or superiority over hosted/open embedding models until scored rows exist in the baseline matrix.

## Current Anchor

- Artifact: `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll`
- SHA256: `a7461b47784ea7434cf6048f33f6c281ef19887cfa9d0c699b6f2fba079f2b67`
- Macro nDCG@10: `0.265891`
- Macro recall@100: `0.452844`

Treat this as the local sealed anchor, not as a default-promotion decision by itself. The `manta-embed-v1` artifact and run directory names are legacy names for the same `eos-embed-v1` lineage.

## Scoreboard Promotion Gate

Default embedder candidates must pass the scoreboard gate against the sealed anchor before promotion. The gate compares every selected dataset and metric independently; macro gains are reported for context but do not hide a per-dataset miss.

```bash
go run ./cmd/eos gate-scoreboard \
  --category short_retrieval \
  --baseline eos \
  --datasets scifact,nfcorpus,fiqa \
  --metrics ndcg_at_10,recall_at_100 \
  runs/<candidate-scoreboard>/scoreboard.json \
  runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/scoreboard.json
```

Use `--tolerance` only for an explicitly accepted numeric rounding margin. For TurboQuant rows, add the matching `--baseline`, `--method`, and `--bits` filters so the command compares one unambiguous row per dataset in both scoreboards. The current promoted compact retrieval profile is `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch250_fp16_rerank --bits 4` with `--metrics ndcg_at_10,recall_at_100,total_compression_ratio`, backed by `runs/eos-q4-fp16-overfetch250-gate-20260615T000000Z/`.
`--baseline eos` falls back to legacy `manta` rows when exact `eos` rows are absent, so the gate can compare new scoreboards with the current legacy-named anchor without rewriting provenance.

Hybrid retrieval rows are eligible only as calibrated retrieval-surface evidence. Run `scripts/calibrate_eos_embed_hybrid_retrieval.fw` first, select method/alpha/RRF settings on the configured dev split, and apply the selected setting unchanged to test. The calibration summary must include dense and BM25 sanity rows plus the protection gate deltas against dense `ndcg_at_10` and `recall_at_100`; use that selected setting in any later `eos-hybrid` scoreboard row. Per-query protection beyond optional sentinel query IDs is still a follow-up policy layer, so do not treat a passing hybrid calibration as a dense model promotion.

For FiQA and mixed-objective iteration, use the reusable guarded candidate runner as the standard loop:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GUARD_ANCHOR_SCOREBOARD=runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/scoreboard.json \
ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw
```

The runner composes the existing candidate training script, scoreboard script, and `eos gate-scoreboard`. It writes a guard manifest and summary under `runs/eos-embed-v1-guarded-<timestamp>/`, and rejects the candidate unless every selected dataset metric clears the anchor. Use `EOS_GUARD_FAIL_ON_GATE=0` only for smoke runs where the rejected manifest is the expected output.

## Baseline Matrix

Maintain a scoreboard with explicit rows for:

- Current sealed Eos anchor (`baseline=eos`; legacy anchor rows may still be `baseline=manta`).
- BM25 lexical baseline.
- BGE, Qwen, Jina, Voyage, Cohere, and OpenAI external vector caches.
- Any later local Eos candidate, including `eos-hybrid`, `eos-turboquant`, and `eos-turboquant-rerank` rows when those product surfaces are evaluated.

Rows without evaluated vectors stay marked `not_scored`. Do not replace missing rows with claims from model cards or unrelated public benchmarks. For external models, the provider boundary is a BEIR-aligned pair of caches:

- `doc-vectors.jsonl`
- `query-vectors.jsonl`

Each row needs `id` or `_id` plus one of `vector`, `embedding`, or `values`.

The first practical leading-family local baseline is Qwen3 0.6B:

```bash
python3 scripts/export_retrieval_vectors.py \
  --preset qwen3-0.6b \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b \
  --dataset-name scifact \
  --batch-size 16
```

Mixedbread/mxbai can use the same provider-boundary lane:

```bash
python3 scripts/export_retrieval_vectors.py \
  --preset mxbai-large \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/mxbai-large \
  --dataset-name scifact \
  --batch-size 16
```

This script is a provider-boundary exporter. Its optional Python dependencies stay outside Eos, and its default output is `runs/external-vector-caches/<provider>/<dataset>/doc-vectors.jsonl` plus `query-vectors.jsonl`. The older `scripts/export_qwen3_retrieval_vectors.py` entry point remains available for compatibility and accepts `--model-name` for non-Qwen SentenceTransformers models.
The mxbai preset applies Mixedbread's retrieval query prompt and leaves document text unprefixed.

For parent-child multi-vector evidence, add document chunking flags. This keeps the query cache unchanged but writes `child-doc-vectors.jsonl` rows with `parent_id`, deterministic `child_id`, and `embedding`:

```bash
python3 scripts/export_retrieval_vectors.py \
  --preset qwen3-0.6b \
  --dataset-dir datasets/manta-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b-scifact-child-w128-o32 \
  --dataset-name scifact \
  --batch-size 16 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16
```

Current external comparison state: Qwen3 is locally consolidated for SciFact, NFCorpus, and full exportable-text FiQA. Its useful compact external row is direct q8 at about `3.98x` vector compression, with SciFact q8 nDCG@10 `0.704128`, NFCorpus q8 nDCG@10 `0.368763`, and FiQA q8 nDCG@10 `0.449614`. mxbai remains stronger than Qwen3 in the existing local short-set evidence. Qwen3 FiQA is not raw-row-complete or judged-coverage complete because one judged test document had empty unexportable text and was skipped.

## TurboQuant Gate

Every promotion candidate needs a dense reference and TurboQuant IP document-vector rows over the same vectors:

```bash
go run ./cmd/eos eval-retrieval-vectors \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b/scifact/doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b/scifact/query-vectors.jsonl \
  --metrics-json runs/scifact.qwen3.retrieval.metrics.json \
  datasets/eos-embed-v1/raw/scifact/scifact

go run ./cmd/eos eval-retrieval-vectors-turboquant \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b/scifact/doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --metrics-json runs/scifact.qwen3.turboquant.metrics.json \
  --metrics-tsv runs/scifact.qwen3.turboquant.metrics.tsv \
  datasets/eos-embed-v1/raw/scifact/scifact
```

The scoreboard harness appends the dense row and, when enabled, q2/q4/q8 rows with method, bit width, quality deltas, vector bytes, dense vector bytes, compression, docs/s, and scores/s:

```bash
EOS_SCOREBOARD_EXTERNAL_VECTOR_ROOT=runs/external-vector-caches/qwen3-0.6b \
EOS_SCOREBOARD_EXTERNAL_VECTOR_DATASETS=scifact \
EOS_SCOREBOARD_EXTERNAL_VECTOR_BASELINE=qwen3 \
EOS_SCOREBOARD_EXTERNAL_VECTOR_BACKEND=qwen3 \
EOS_SCOREBOARD_EXTERNAL_VECTOR_ARTIFACT=Qwen/Qwen3-Embedding-0.6B \
EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT=1 \
EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_BITS=2,4,8 \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

Local sealed Eos candidates can emit the same product-surface rows without exporting an external vector cache:

```bash
EOS_SCOREBOARD_ARTIFACT=runs/<candidate>/eos-embed-v1.sealed.mll \
EOS_SCOREBOARD_RETRIEVAL_ROOT=datasets/eos-embed-v1 \
EOS_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
EOS_SCOREBOARD_TURBOQUANT=1 \
EOS_SCOREBOARD_TURBOQUANT_BITS=4 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=250 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16 \
EOS_SCOREBOARD_TURBOQUANT_BASELINE=eos-turboquant \
EOS_SCOREBOARD_TURBOQUANT_RERANK_BASELINE=eos-turboquant-rerank \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

This produces direct `eos-turboquant` rows and fp16 sidecar rerank `eos-turboquant-rerank` rows from one `eval-retrieval-turboquant` metrics file. Use `--baseline eos-turboquant --method turboquant_ip_b4 --bits 4` to inspect direct q4, but do not treat direct q4 or direct q8 as default-promotion candidates. Gate the promoted compact profile with `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch250_fp16_rerank --bits 4 --metrics ndcg_at_10,recall_at_100,total_compression_ratio`.

When the default embedder needs to feed the vector-cache evaluators instead of live `eval-retrieval`, use the Go-native exporter. It loads packaged or sealed Eos embedding artifacts through the runtime tokenizer/batch path and writes the same JSONL shape as external caches:

```bash
go run ./cmd/eos export-retrieval-vectors \
  --dataset scifact \
  --batch-size 64 \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/beir/scifact \
  runs/eos-vector-caches/eos-embed-v1/scifact
```

For Eos-owned parent-child evidence, enable deterministic word chunking. This writes `child-doc-vectors.jsonl` rows with `parent_id`, `child_id`, and `embedding`, plus the unchanged query cache for `eval-retrieval-multivector-turboquant`:

```bash
go run ./cmd/eos export-retrieval-vectors \
  --dataset scifact \
  --batch-size 64 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16 \
  --manifest-json runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32/manifest.json \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/beir/scifact \
  runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32/scifact
```

Record, per dataset and candidate:

- Dense nDCG@10/nDCG@100, MRR@10, precision@1/5/10, hit@1/5/10, MAP@10/MAP@100, and recall@10/100.
- q2/q4/q8 quality deltas, especially nDCG@10 and recall@100 deltas against the dense row.
- Vector bytes, rerank storage, rerank sidecar bytes, total vector bytes, compression ratio, and total compression ratio.
- Quantization docs/s.
- Direct IP scores/s, per-query p50/p95/p99/max scoring latency, and rerank overfetch/rerank score counts when rerank rows are enabled.

Use the quality columns for different failure modes: nDCG and MAP judge ranked relevance, precision/hit@k judge first-screen success, and recall@100 judges candidate-pool coverage for reranking or multi-stage retrieval. Use vector bytes and compression ratio for index-footprint decisions, and throughput columns for the path they actually measure. External vector-cache dense rows measure cache load plus scoring, not live encoder throughput for Qwen, BGE, hosted APIs, or other providers.

The default bit width should be selected from measured q4/q8 rows. q2 is useful pressure testing but should not become default unless quality loss is explicitly acceptable for the target workload.

Use the local serving proxy before default-alias promotion:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_default_embedder_serving.fw
```

The smoke compares `turboquant_ip_b4_overfetch250_fp16_rerank` against the lower-risk fallback `turboquant_ip_b8_overfetch125_fp16_rerank`, writes a summary TSV and manifest under `runs/eos-default-embedder-serving-smoke-<timestamp>/`, and can gate total compression plus optional p95 latency. This is CorkScrewDB-relevant TurboQuant serving evidence only; do not claim CorkScrewDB load/index/search has passed until an external or in-repo CorkScrewDB harness runs that API path.

Current local Eos TurboQuant result: q4/fp16 sidecar rerank at overfetch250 is the promoted compact retrieval profile. It passed the selected-vs-anchor scoreboard gate on SciFact, NFCorpus, and FiQA for `ndcg_at_10,recall_at_100,total_compression_ratio` as `eos-turboquant-rerank` / `turboquant_ip_b4_overfetch250_fp16_rerank` / bits `4`, with total compression `1.590062x`, in `runs/eos-q4-fp16-overfetch250-gate-20260615T000000Z/`. This is a two-stage compact retrieval profile, not q4-only retrieval: direct q4 loses quality on SciFact and FiQA and is not a default-promotion candidate. Direct q8 also remains outside the promoted default path because the useful lower-risk compact fallback is the two-stage q8/fp16 sidecar profile.

Keep q8/fp16 sidecar rerank at overfetch125 as the lower-risk, lower-rerank-cost fallback: `turboquant_ip_b8_overfetch125_fp16_rerank`, total compression `1.326425x`, evidence in `runs/eos-fp16-overfetch125-gate-20260614T000000Z/`.

## Multi-Vector Storage Planning

The direct multi-vector lane is a storage/accounting thesis, not a retrieval-quality claim: one parent CorkScrewDB object can keep many quantized child vectors for windows, events, spans, or time-series observations while staying near the byte budget of one dense fp32 parent vector. Measure that budget with:

```bash
go run ./cmd/eos plan-multivector-storage \
  --dim 128 \
  --baseline-dim 3072 \
  --bits 2,4,8 \
  --vectors-per-object 64,128,256,384 \
  --objects 1000
```

The TSV/JSON rows report `baseline_dim`, `dense_parent_bytes`, `dense_baseline_bytes`, `quantized_vector_bytes`, `total_quantized_bytes`, compression ratios, and `vectors_that_fit_in_one_dense_vector`. When `--baseline-dim` is omitted or `0`, the dense budget is the same dimension as the child vector, preserving the same-dim interpretation: 128-dimensional q2 stores a child vector in 36 bytes, so 14 children fit inside one 512-byte fp32 vector budget. When modeling compact children against a larger baseline, pass `--dim 128 --baseline-dim 3072`; one dense baseline vector is 12,288 bytes, so 341 q2 children fit in that one-vector budget and 128 children cost about `0.375x` of it before object/index metadata.

The first quality harness for that lane is cache-only and still outside the CorkScrewDB API:

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset scifact \
  --backend qwen3-child-cache \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b-scifact-child-w128-o32/scifact/child-doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b-scifact-child-w128-o32/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  --metrics-json runs/scifact.multivector-turboquant.metrics.json \
  --metrics-tsv runs/scifact.multivector-turboquant.metrics.tsv \
  datasets/manta-embed-v1/raw/scifact/scifact
```

Document child-vector JSONL accepts `parent_id`, `child_id`, and one vector field among `vector`, `embedding`, or `values`. When `parent_id` is absent, `id` or `_id` is used as both parent and child, so one-vector caches remain a valid degenerate multi-vector input. The evaluator scores every child vector, aggregates by max child score per parent, and evaluates parent IDs against BEIR qrels. Strict coverage is the default: if any qrels-relevant parent is missing from the child-vector cache, the run fails instead of filtering that parent out and inflating metrics. Use `--allow-missing-relevant` only for diagnostic smoke runs where incomplete qrel coverage is intentional. Its dense row uses the same max-child aggregation over fp32 child vectors; q2/q4/q8 rows quantize children with a deterministic TurboQuant IP seed, configurable with `--quantizer-seed`, and use direct TurboQuant IP scoring without fp16 rerank sidecars. JSON/TSV metrics include `allow_missing_relevant`, `quantizer_seed`, `baseline_dim`, parent count, child-vector count, average and max children per parent, dense baseline bytes, dense child bytes, per-vector quantized bytes, total quantized child bytes, dense-child compression, vectors that fit in one dense baseline, storage multiple versus one dense baseline vector per parent, scored child pairs, quality deltas, scores/s, and query latency summaries. Leave `--baseline-dim` at `0` only for same-dimension accounting; use `1024` or `3072` when the claim is compact children versus a large dense embedder.

First measured SciFact parent-child evidence: Qwen3 0.6B child cache, `128` word chunks, `32` word overlap, and `16` word minimum trailing chunk produced `5,183` parents and `12,468` children, averaging `2.41` children per parent. The run evaluated `300` qrels queries with strict qrels coverage and recorded the deterministic quantizer seed in metrics.

| row | child nDCG@10 | child recall@100 | compression | parent-budget multiple |
| --- | ---: | ---: | ---: | ---: |
| dense-child | 0.717467 | 0.953333 | n/a | n/a |
| q2 | 0.701697 | 0.950000 | 15.75x | 0.15x |
| q4 | 0.714895 | 0.956667 | 7.94x | 0.30x |
| q8 | 0.716310 | 0.953333 | 3.98x | 0.60x |

Compared with the existing one-vector Qwen3 SciFact evidence, dense child-max improves over dense `0.702026` nDCG@10 / `0.946667` recall@100, and direct q8 child-max improves over q8 `0.702657` nDCG@10 / `0.946667` recall@100 while storing q8 children below one dense-parent-vector budget.

Mixedbread `mixedbread-ai/mxbai-embed-large-v1` is the current stronger external SciFact child-cache baseline on the same parent-child lane. The requested `datasets/eos-embed-v1/raw/scifact/scifact` path was absent, so the run used `datasets/manta-embed-v1/raw/scifact/scifact`, matching the Qwen3 child evidence. It used `128` word chunks, `32` overlap, and `16` minimum trailing words, producing `5,183` parents, `12,468` child vectors, `2.405557` average children per parent, and `300` evaluated qrels queries with strict coverage (`allow_missing_relevant=false`).

| row | child nDCG@10 | child recall@100 | compression | parent-budget multiple | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-child | 0.747175 | 0.970000 | n/a | n/a | 12.497 ms |
| q2 | 0.712790 | 0.956667 | 15.75x | 0.15x | 4.754 ms |
| q4 | 0.739489 | 0.965000 | 7.94x | 0.30x | 77.250 ms |
| q8 | 0.747799 | 0.966667 | 3.98x | 0.60x | 157.876 ms |

mxbai is higher than Qwen3 child-max on dense, q2, q4, and q8 nDCG@10 and recall@100. The q8 mxbai row beats Qwen3 q8 by `+0.031489` nDCG@10 and `+0.013334` recall@100. Treat mxbai as the stronger external SciFact child-cache baseline, while Qwen3 remains useful as a compact leading-family baseline.

The sealed Eos/default path is now measured end-to-end for this lane: `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll` exported a full Go-native SciFact child cache from `datasets/manta-embed-v1/raw/scifact/scifact` with `128` word chunks, `32` overlap, and `16` minimum trailing words. Export counts were `5,183` docs, `300` queries, `12,468` children, dim `256`, CUDA backend, and `57.771s` elapsed. The strict eval used `allow_missing_relevant=false`, `339` relevant pairs, `3,740,400` scored child pairs, and quantizer seed `5581486560434873699`.

| row | Eos child nDCG@10 | Eos child recall@100 | compression | parent-budget multiple | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-child | 0.462489 | 0.778111 | n/a | n/a | 3.129 ms |
| q2 | 0.383295 | 0.719667 | 15.06x | 0.16x | 1.159 ms |
| q4 | 0.449435 | 0.773111 | 7.76x | 0.31x | 17.819 ms |
| q8 | 0.461862 | 0.774778 | 3.94x | 0.61x | 39.192 ms |

This proves the sealed `.mll` -> Go-native child vector cache -> strict TurboQuant multivector eval path, but it does not promote the current sealed Eos anchor on SciFact child-cache quality. Eos dense-child is materially below full mxbai `0.747175` nDCG@10 / `0.970000` recall@100 and Qwen3 `0.717467` / `0.953333`; q8 preserves Eos dense-child quality closely, and q4 is near but drops more. The main deficit is model quality, not TurboQuant storage or scoring.

TurboQuant's strategic win for CorkScrewDB is not only q4/q8 compression of one vector. Direct compact child vectors let a parent object carry many cheap vectors for windows, spans, time-series slices, events, and other multi-vector schemas. Be precise in product planning: same-dimension child vectors do not fit hundreds of children inside the budget of one same-dimension fp32 parent vector. Tens to hundreds become plausible when compact child dimensions are planned against a 1024 to 3072 dimensional fp32 baseline with `--baseline-dim` or when the product explicitly budgets multiple dense-parent equivalents. That needs a measured planner and eval lane, not hand-wavy storage copy.

Keep this separate from q4/fp16 rerank. `--sidecar-storage fp16` intentionally adds a per-child fp16 sidecar and shows why sidecars destroy the high-child-count storage argument: the sidecar is useful for quality-preserving two-stage rerank, but it is not the direct hundred-child storage mode.

## Data And Teacher Growth

Grow the training/evaluation signal before increasing model size:

- Add scored teacher batches from BGE, Qwen, Jina, Voyage, Cohere, and OpenAI.
- Keep provider outputs as vector caches or scorer batches, not as provider SDK dependencies inside Eos.
- Use hard negatives and retrieval-error mining to target the weakest datasets in the matrix.
- Try Matryoshka-style dimension slicing before moving to larger embedding dimensions, so CorkScrewDB can trade quality for memory and scoring cost.

## Default Promotion Gate

Do not promote a default CorkScrewDB embedder until all of these are true:

- The sealed `.mll` package verifies by SHA256 and package metadata.
- The baseline matrix has explicit dense, direct TurboQuant, and TurboQuant rerank rows, with missing external rows still visible as `not_scored`.
- A CorkScrewDB load/index/search smoke has passed with the candidate vectors.
- q4 and q8 default choices are measured on the same datasets, with quality deltas, vector bytes, compression, docs/s, scores/s, rerank overfetch where applicable, and serving p50/p95/p99/max attached.
- The docs name the measured default and avoid unsupported standing claims.

## Next Actions

1. Run a CorkScrewDB load/index/search smoke for the q4/fp16/overfetch250 compact retrieval profile.
2. Measure p95 serving latency for q4/fp16/overfetch250 and decide whether the q8/fp16/overfetch125 fallback is needed for lower rerank cost.
3. Keep the full short-set external matrix current, with Qwen3 FiQA labeled as full exportable-text rather than raw-row-complete or judged-coverage complete.
4. Run a protected teacher/data experiment targeted at the remaining quality gap.
