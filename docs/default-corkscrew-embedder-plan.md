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
  --output-dim 128 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16 \
  --manifest-json runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/manifest.json \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/beir/scifact \
  runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/scifact
```

`--output-dim 128` writes a prefix-truncated compact child cache and L2-renormalizes the truncated vectors before writing. The manifest keeps `dimension` as the written vector dimension and records `model_dimension`/`output_dimension` so the truncation is auditable. Treat this as a measurement bridge for the compact-child thesis, not as a trained Matryoshka embedding or native 128d Eos head; the trained compact head remains the stronger future path.

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

The smoke compares `turboquant_ip_b4_overfetch250_fp16_rerank` against the lower-risk fallback `turboquant_ip_b8_overfetch125_fp16_rerank`, writes a summary TSV and manifest under `runs/eos-default-embedder-serving-smoke-<timestamp>/`, and can gate total compression plus optional p95 latency. This is CorkScrewDB-relevant TurboQuant serving evidence only. Use the local flat CorkScrewDB API smoke when the actual `PutVector`/`SearchVector` integration path needs to pass:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_corkscrewdb_child_vectors.fw
```

That smoke defaults to a tiny synthetic time-series child-vector cache, can consume externally prepared child/query/qrels paths with `EOS_CORKSCREW_SMOKE_CHILD_VECTORS`, `EOS_CORKSCREW_SMOKE_QUERY_VECTORS`, and `EOS_CORKSCREW_SMOKE_QRELS`, and requires a local CorkScrewDB checkout rather than silently pulling a network dependency. `EOS_CORKSCREW_SMOKE_OVERFETCH` accepts comma-separated values such as `100,12468`; the harness loads one DB per bit width and reuses it across overfetch values so serving recall/latency sweeps avoid repeated insert cost. Use exhaustive/full child overfetch when comparing against offline cache-evaluator parity, with the expectation that latency rises. Treat it as local flat CorkScrewDB load/index/search/storage accounting evidence, not remote/federation/HNSW evidence or a model-quality benchmark.

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
  --vector-overhead-bytes 32 \
  --packed-object-overhead-bytes 32 \
  --objects 1000
```

For time-series/window-vector planning, derive the child-vector count from series lengths:

```bash
go run ./cmd/eos plan-multivector-storage \
  --dim 128 \
  --baseline-dim 3072 \
  --bits 2,4,8 \
  --series-lengths 256,1024 \
  --window-size 64 \
  --window-stride 16 \
  --vector-overhead-bytes 32 \
  --packed-object-overhead-bytes 32 \
  --objects 1000
```

The planner uses one vector per covering window, including a tail window when needed. This is a storage gate for the TurboQuant time-series vector hypothesis, not a numeric time-series quality benchmark; run a retrieval or forecasting quality harness separately before making quality claims. Do not pass `--vectors-per-object` with `--series-lengths`; explicit conflicts fail so manual and derived modes remain separate.

The TSV/JSON rows report `baseline_dim`, `dense_parent_bytes`, `dense_baseline_bytes`, raw `quantized_vector_bytes`, `vector_overhead_bytes`, `dense_vector_storage_bytes`, `quantized_vector_storage_bytes`, `total_quantized_bytes`, packed-parent fields such as `packed_object_overhead_bytes`, `packed_quantized_storage_bytes`, `packed_total_quantized_bytes`, `packed_vectors_that_fit_in_one_dense_vector`, compression ratios, and optional time-series fields `series_length`, `window_size`, `window_stride`, and `derived_window_count`. When `--baseline-dim` is omitted or `0`, the dense budget is the same dimension as the child vector, preserving the same-dim interpretation: 128-dimensional q2 stores a child vector payload in 36 bytes, so 14 payload-only children fit inside one 512-byte fp32 vector budget; with 32 bytes of packed parent-object overhead, 14 q2, 7 q4, or 3 q8 child payloads fit. When modeling compact children against a larger baseline, pass `--dim 128 --baseline-dim 3072`; one dense baseline vector is 12,288 payload bytes, so 341 q2 payload-only children fit in that one-vector budget and 128 children cost about `0.375x` of it before object/index metadata. With the overhead-aware packed-parent accounting used by the smoke (`--packed-object-overhead-bytes 32`), one 3072d dense parent-vector storage budget fits 341 q2, 180 q4, or 93 q8 128d children. Current CorkScrewDB-style per-child-entry accounting uses `--vector-overhead-bytes`; packed-parent target accounting uses `--packed-object-overhead-bytes` to pay object overhead once per parent while each child remains a compact TurboQuant payload. Treat the packed-parent numbers as an architecture target until a CorkScrewDB API smoke directly measures packed parent-object persistence.

For an executable overhead-aware budget-frontier artifact, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_frontier.fw
```

The smoke uses the planner as the source of truth and writes `summary.tsv`, `manifest.json`, planner JSON, and command logs under `runs/eos-multivector-budget-frontier-smoke-<timestamp>/`. Defaults model `128d` compact children against a `3072d` dense baseline, q2/q4/q8, child counts `1,16,64,100,128,181,256,341`, `32` bytes per current stored vector, `32` bytes per packed parent object, no sidecar, and `1000` objects. Set `EOS_MV_BUDGET_SMOKE_BASELINE_DIMS=128,384,768,1024,1536,3072` to measure the same compact-child shape against multiple dense parent dimensions in one run; this comma-list takes precedence over the backward-compatible `EOS_MV_BUDGET_SMOKE_BASELINE_DIM`. The default gates are current q2 >= `181`, current q4 >= `100`, current q8 >= `64`, packed q2 >= `341`, packed q4 >= `180`, and packed q8 >= `93` children fitting in one dense-vector budget. A deterministic parent-budget frontier run should use `EOS_MV_BUDGET_SMOKE_VECTORS_PER_OBJECT=64,100,128,180,256,341` with the baseline list above to document the precise claim: same-dimension children fit only single-digit to low-tens counts, while a 3072d dense-parent budget fits hundreds of packed q2/q4 child vectors and tens of q8 child vectors. Read this beside the time-series smoke and local CorkScrewDB API smoke: it proves byte accounting only, not retrieval quality, remote serving, HNSW, federation, API latency, or current end-to-end packed parent-object storage.

For the executable bridge from byte accounting to cache-only quality, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_quality.fw
```

The default run uses `runs/eos-128d-child-cache-quality-20260615T000000Z/scifact/child-doc-vectors.jsonl`, matching query vectors, and `datasets/manta-embed-v1/raw/scifact/scifact`. It evaluates dense/q2/q4/q8 parent-child retrieval, infers the `128d` child dimension from dense child-vector bytes, then plans q2/q4/q8 for the actual parent count with child counts including `1,64,100,128,181,256,341`, `--baseline-dim 3072`, `--vector-overhead-bytes 32`, and `--sidecar-storage none`. Current default interpretation: q4 is near dense on the 128d SciFact child cache (`ndcg@10` drop about `0.002630`, `recall@100` drop about `0.001667`) while the overhead-aware planner fits `123` q4 children in one 3072d dense-vector budget, so the `100` child-vector target fits. Treat this as Eos cache evidence only: the 128d cache is a prefix-truncated bridge, not a native Matryoshka artifact, and the planner overhead is an explicit input rather than a measured CorkScrewDB directory or API result.

For the actual local flat CorkScrewDB API budget-quality check on the same compact cache, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_budget_quality.fw
```

The wrapper runs `scripts/smoke_corkscrewdb_child_vectors.fw` against the Eos 128d SciFact child/query cache and test qrels with q4 overfetch `100,12468`, then joins the resulting local `PutVector`/`SearchVector` metrics to the overhead-aware planner row for `--dim 128 --baseline-dim 3072 --vector-overhead-bytes 32 --sidecar-storage none`. It writes `summary.tsv`, `manifest.json`, command logs, nested `corkscrewdb/` artifacts, and `planner.json` under `runs/eos-corkscrewdb-budget-quality-smoke-<timestamp>/`. Current default result: q4 overfetch100 records `ndcg@10=0.407586`, `recall@100=0.724111`, DB directory multiple `0.048935x`, and p95 search latency about `11.8ms`; q4 full overfetch records `recall@100=0.741889`; planner fit remains `123` q4 children, so the `100` child target fits. The packed-parent path now forwards and records the compact storage modes from the child-vector smoke. Verified compact packed q4 run `runs/eos-corkscrewdb-budget-quality-packed-q4-compact-20260616T000000Z/` used `packed_metadata_mode=none` and `packed_child_id_mode=ordinal` on the same real SciFact child cache with local flat exact parent search; it recorded DB bytes `1,653,983`, DB directory multiple `0.025970x`, p95 `9.505725ms`, `ndcg@10=0.407586`, and `recall@100=0.741889`, or `0.4935x` the prior packed full/source DB bytes and `0.5307x` the separate-child default DB bytes. Read this beside the cache-only smoke and pure planner smoke as actual local flat CorkScrewDB API evidence for the compact Eos child cache, not remote serving, federation, HNSW, or native Matryoshka proof.

For HNSW index-search validation on the same compact cache, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_hnsw_quality.fw
```

This wrapper runs the child-vector smoke with `index_type=hnsw`, `vector_storage=raw`, and a post-load HNSW rebuild, then joins the same overhead-aware q4 planner row under `runs/eos-corkscrewdb-hnsw-quality-smoke-<timestamp>/`. Current default result: q4 overfetch100 records `ndcg@10=0.392775`, `recall@100=0.685778`, raw HNSW DB directory multiple `0.347691x`, rebuild time about `20.9s`, and p95 search latency about `1.78ms`; q4 full overfetch records `recall@100=0.741889` and p95 about `33.1ms`. CorkScrewDB rejects HNSW with `quantized_only`, so this smoke deliberately trades compact quantized-only persistence for raw-vector HNSW validation. Keep the raw HNSW DB-directory multiple separate from the flat q4 quantized-only storage multiple; the q4 vector payload/planner columns are planner comparisons, not HNSW persistence claims.

For a first time-series/window quality seam, export text-rendered numeric windows and run the existing parent-child evaluator against parent-series qrels:

```bash
go run ./cmd/eos export-timeseries-vectors \
  --dataset sensor-window-retrieval \
  --batch-size 64 \
  --output-dim 128 \
  --window-size 64 \
  --window-stride 16 \
  --manifest-json runs/timeseries-window-cache-128d/manifest.json \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/timeseries/sensor-series.jsonl \
  data/timeseries/queries.jsonl \
  runs/timeseries-window-cache-128d
```

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset sensor-window-retrieval \
  --backend text-rendered-timeseries-windows \
  --artifact eos-embed-v1-prefix128 \
  --doc-vectors runs/timeseries-window-cache-128d/child-doc-vectors.jsonl \
  --query-vectors runs/timeseries-window-cache-128d/query-vectors.jsonl \
  --qrels data/timeseries/qrels/test.tsv \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  runs/timeseries-window-cache-128d
```

`export-timeseries-vectors` expects one JSONL object per parent series with `id` or `_id` and numeric `values`. It writes deterministic child IDs like `series-id#window-0000`, renders each numeric window as stable text with values and simple stats, embeds those windows with the same Eos text embedder path as retrieval export, and also writes BEIR helper files `corpus.jsonl` and `queries.jsonl` into the output directory. Use that output directory as the evaluator dataset directory and pass parent-series qrels with `--qrels`; the qrels corpus IDs must be parent series IDs. This is a quality harness for text-rendered numeric windows, not a final trained numeric time-series encoder.

Reusable smoke:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_timeseries_window_vectors.fw
```

This script creates a tiny synthetic five-series dataset, regenerates the time-series child-vector cache with the sealed Eos artifact, runs dense/q2/q4/q8 multivector TurboQuant evaluation, and then runs the storage planner with overhead-aware baseline accounting. Its `summary.tsv` and `manifest.json` demonstrate the narrow economics claim: many 128d quantized window vectors can fit under one 3072d dense-vector budget while q8 stays at dense quality on this controlled smoke. Override `EOS_TS_WINDOW_SMOKE_ARTIFACT`, `EOS_TS_WINDOW_SMOKE_OUTPUT_DIM`, `EOS_TS_WINDOW_SMOKE_WINDOW_SIZE`, `EOS_TS_WINDOW_SMOKE_WINDOW_STRIDE`, `EOS_TS_WINDOW_SMOKE_BITS`, `EOS_TS_WINDOW_SMOKE_BASELINE_DIM`, and `EOS_TS_WINDOW_SMOKE_VECTOR_OVERHEAD_BYTES` to reuse the harness on another artifact or shape.

To prove the same synthetic window child vectors through local CorkScrewDB `PutVector`/`SearchVector`, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

The wrapper nests the time-series smoke under `timeseries/`, feeds the generated `child-doc-vectors.jsonl`, `query-vectors.jsonl`, and `qrels.tsv` into `scripts/smoke_corkscrewdb_child_vectors.fw`, and defaults CorkScrewDB to flat `quantized_only` storage. Its joined `summary.tsv` reports parent count, child-window count, derived windows per parent, q4/q8 quality, planner fit, measured DB directory multiple, vector payload multiple, and p95 latency. Current default result: `5` parents, `25` child windows, `5` derived windows per parent; q4 flat/quantized_only records `ndcg@10=1.000000`, `recall@100=1.000000`, planner fit `123`, vector payload multiple `0.034180x`, and DB directory multiple `0.117643x`; q8 records `ndcg@10=0.926186`, `recall@100=1.000000`, planner fit `75`, vector payload multiple `0.060221x`, and DB directory multiple `0.143685x`. Observed p95 latency for the default synthetic smoke has been sub-ms, but it varies by run. This is local flat API evidence for synthetic text-rendered numeric windows only; it is not remote mode, federation, HNSW, a trained numeric time-series encoder, or a claim that DB directory bytes and planner/vector payload bytes are the same accounting layer.

Scale the same path to 100 q4 child windows per parent and enough parent variants to separate fixed DB-directory overhead from per-vector TurboQuant payload accounting with:

```bash
EOS_REPO_ROOT=$PWD \
EOS_CORKSCREW_TS_WINDOW_BITS=4 \
EOS_CORKSCREW_TS_WINDOW_SERIES_LENGTH=1648 \
EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20 \
EOS_CORKSCREW_TS_WINDOW_WINDOW_SIZE=64 \
EOS_CORKSCREW_TS_WINDOW_WINDOW_STRIDE=16 \
EOS_CORKSCREW_TS_WINDOW_TARGET_CHILDREN=100 \
EOS_CORKSCREW_TS_WINDOW_TOP_PARENTS=100 \
EOS_CORKSCREW_TS_WINDOW_OVERFETCH=500 \
EOS_CORKSCREW_TS_WINDOW_MAX_VECTOR_PAYLOAD_MULTIPLE=0.70 \
EOS_CORKSCREW_TS_WINDOW_MAX_DB_DIR_MULTIPLE=0 \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

The 1648/64/16 windowing derives exactly 100 child windows per parent and keeps q8 out of the run. With `EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20`, the five query patterns expand to `100` parent series and `10,000` child windows while keeping the original five pattern queries; each query has `20` relevant parent variants, for `100` relevant query-parent pairs. Verified separate-child run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-20260616T000000Z/` recorded q4 flat/`quantized_only` with planner fit `123`, planner storage multiple `0.811688x`, vector payload multiple `0.683594x`, DB directory multiple `2.293748x`, DB bytes `2,818,558`, `ndcg@10=0.352927`, `recall@100=0.560000`, p95 `11.948319ms`, and overfetch `500`.

Packed parent-object storage has now been measured on the same q4 scaled shape with local flat exact parent search. Verified `EOS_CORKSCREW_SMOKE_PACKED_METADATA_MODE=none` and `EOS_CORKSCREW_SMOKE_PACKED_CHILD_ID_MODE=ordinal` run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-packed-minimal-20260616T000000Z/` recorded `layout=packed_parent_multivectors`, `parent_insert_count=100`, `parent_search_exact=true`, `overfetch_children=0`, DB directory multiple `0.844660x`, DB bytes `1,037,918`, `ndcg@10=0.352927`, `recall@100=1.000000`, and p95 `5.916649ms`. DB bytes were `0.421237x` of the prior full-metadata packed run and `0.368244x` of the separate-child run. This is a storage/API evidence win for packed parent objects; the recall delta reflects exact parent rollup versus bounded separate-child overfetch. It remains synthetic text-rendered local flat evidence, not remote/federation/HNSW, production quality, or a trained numeric time-series encoder claim.

The first quality harness for that lane is cache-only and still outside the CorkScrewDB API:

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset scifact \
  --backend eos-child-cache-128d \
  --artifact eos-embed-v1-prefix128 \
  --doc-vectors runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/scifact/child-doc-vectors.jsonl \
  --query-vectors runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  --metrics-json runs/scifact.128d-child.multivector-turboquant.metrics.json \
  --metrics-tsv runs/scifact.128d-child.multivector-turboquant.metrics.tsv \
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
- A CorkScrewDB load/index/search smoke has passed with the candidate vectors; use `scripts/smoke_corkscrewdb_child_vectors.fw` for the local flat API path and override its input-cache env vars for candidate-specific child vectors. Keep `EOS_CORKSCREW_SMOKE_LAYOUT=separate_child_vectors` for the current child-overfetch baseline, or set `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors` to measure local flat `PutMultiVector` plus exact `SearchParentsVector` packed-parent persistence.
- q4 and q8 default choices are measured on the same datasets, with quality deltas, vector bytes, compression, docs/s, scores/s, rerank overfetch where applicable, and serving p50/p95/p99/max attached.
- The docs name the measured default and avoid unsupported standing claims.

## Next Actions

1. Run `scripts/smoke_corkscrewdb_child_vectors.fw` with candidate child/query/qrels inputs for the compact retrieval profile, using the scaled q4 time-series packed-vs-separate evidence as the current local flat storage reference. Keep q4/fp16 rerank quality evidence separate because this API smoke exercises direct quantized child or exact packed-parent vector search, not sidecar rerank.
2. Measure p95 serving latency for q4/fp16/overfetch250 and decide whether the q8/fp16/overfetch125 fallback is needed for lower rerank cost.
3. Keep the full short-set external matrix current, with Qwen3 FiQA labeled as full exportable-text rather than raw-row-complete or judged-coverage complete.
4. Run a protected teacher/data experiment targeted at the remaining quality gap.
