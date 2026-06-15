# Production Embedding Candidate

Use `scripts/train_manta_embed_v1_candidate.fw` to create a release-grade `eos-embed-v1` candidate. The workflow wraps the current Eos CLI primitives with production guardrails:

- refuses temporary input paths unless explicitly overridden
- refuses dirty repositories unless explicitly overridden
- records repo commit, Go version, selected environment, dataset SHA256, artifact SHA256, and run config
- trains from either a raw corpus or prepared JSONL
- runs eval-only verification on a copied package and requires `optimizer_updates=0`
- runs a separate hard holdout eval by default
- exports and verifies a sealed MLL
- compares dense retrieval against TurboQuant IP-preserving quantized document vectors before default CorkScrewDB embedder promotion
- supports metric gates through environment variables

## Preflight

Run the local preflight before spending trainer time on a candidate:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/verify_manta_production.fw
```

This uses generated tiny fixtures to verify the public Eos surface, build `cmd/eos`, initialize and train a `.mll` package, run eval-only checks with `optimizer_updates=0`, export a sealed `.mll`, and inspect both packages. It is a production path check, not a model quality result.

## Acquire Datasets

The default acquisition workflow downloads public BEIR retrieval datasets (`scifact`, `nfcorpus`, and `fiqa`), converts qrels to Eos text-pair JSONL, writes a tokenizer corpus, and emits initial threshold gates:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/eos-embed-v1 \
ferrous-wheel run scripts/acquire_manta_embed_v1_datasets.fw
```

The processed outputs are:

```text
/data/manta/datasets/eos-embed-v1/processed/train.jsonl
/data/manta/datasets/eos-embed-v1/processed/eval.jsonl
/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl
/data/manta/datasets/eos-embed-v1/processed/tokenizer-corpus.txt
/data/manta/datasets/eos-embed-v1/processed/thresholds.env
```

Review dataset licenses before commercial use. The default avoids MS MARCO because it is commonly distributed with non-commercial research terms.

## Process Corpus Pretraining

Use the process-doc lane when a candidate should include local Tiller/Codex operating language before BEIR alignment. The builder reads local process files such as `AGENTS.md`, `.codex/agents/*.toml`, and `.codex/skills/**/SKILL.md`, chunks them, and emits text hard-negative JSONL with:

```json
{"query":"process guidance in AGENTS.md","positive":"...","negatives":["..."],"source":"process:agents","group_id":"process-agents-md-0"}
```

For a small local smoke:

```bash
EOS_REPO_ROOT=$PWD \
EOS_PRETRAIN_BEIR=0 \
EOS_PROCESS_PRETRAIN=1 \
EOS_PROCESS_PRETRAIN_MAX_DOCS=8 \
EOS_PROCESS_PRETRAIN_MAX_ROWS=16 \
EOS_PRETRAIN_OUT=$PWD/datasets/eos-embed-v1/processed/process-pretrain-smoke.jsonl \
ferrous-wheel run scripts/build_pretrain_pairs.fw
```

For the normal Stage A pretraining file, add process rows alongside the existing BEIR synthetic rows:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/eos-embed-v1/raw \
EOS_PROCESS_PRETRAIN=1 \
EOS_PROCESS_PRETRAIN_INCLUDE_DOCS=1 \
EOS_PRETRAIN_OUT=/data/manta/datasets/eos-embed-v1/processed/pretrain-pairs.jsonl \
ferrous-wheel run scripts/build_pretrain_pairs.fw
```

`EOS_PROCESS_PRETRAIN_INCLUDE_DOCS=1` adds `docs/**/*.md`. Use `EOS_PROCESS_PRETRAIN_PATHS=path1,path2` for extra files or directories, `EOS_PROCESS_PRETRAIN_MAX_DOCS` and `EOS_PROCESS_PRETRAIN_MAX_ROWS` for bounded probes, and `EOS_PROCESS_PRETRAIN_CHUNK_WORDS` when the default `220`-word chunks are too large or small.

A bounded end-to-end process-corpus smoke generated `12` process rows, trained through the hard-negative path with `optimizer_updates=42`, and completed a separate eval-only pass with `optimizer_updates=0`. Treat this as plumbing proof only, not a quality result.

The shipping pipeline reads `/data/manta/datasets/eos-embed-v1/processed/pretrain-pairs.jsonl` when it builds `shipping-mixed-pretrain-plus-beir.jsonl`. If that file already exists, rebuild it with the command above before running:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_SHIP_RUN_ROOT=/data/manta/runs/eos-embed-v1-process-pretrain \
ferrous-wheel run scripts/train_manta_embed_v1_shipping_pipeline.fw
```

For a direct candidate experiment, feed the process JSONL through the existing hard-negative path:

```bash
EOS_REPO_ROOT=$PWD \
EOS_RUN_ROOT=/data/manta/runs \
EOS_RUN_ID=eos-embed-v1-process-pretrain-a \
EOS_TRAIN_JSONL=/data/manta/datasets/eos-embed-v1/processed/pretrain-pairs.jsonl \
EOS_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_HARD_NEGATIVE_TRAIN=1 \
EOS_HARD_NEGATIVES_PER_QUERY=1 \
EOS_EPOCHS=1 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

## Prepared JSONL Path

Prepared JSONL is the preferred path for production because train/eval splits are fixed before training starts.

```bash
EOS_RUN_ROOT=/data/manta/runs \
EOS_REPO_ROOT=$PWD \
EOS_RUN_ID=eos-embed-v1-20260412-a \
EOS_TRAIN_JSONL=/data/manta/datasets/eos-embed-v1/processed/train.jsonl \
EOS_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_TOKENIZER_CORPUS=/data/manta/datasets/eos-embed-v1/processed/tokenizer-corpus.txt \
EOS_THRESHOLDS_ENV=/data/manta/datasets/eos-embed-v1/processed/thresholds.env \
EOS_EPOCHS=3 \
EOS_BATCH_SIZE=1024 \
EOS_LR=0.005 \
EOS_TEMPERATURE=0.05 \
EOS_SELECT_METRIC=score_margin \
EOS_EVAL_EVERY_STEPS=0 \
EOS_MIN_AUC=0.70 \
EOS_MIN_THRESHOLD_ACCURACY=0.65 \
EOS_MIN_SCORE_MARGIN=0.05 \
EOS_MAX_LOSS=0.35 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

`EOS_THRESHOLDS_ENV` loads the acquisition workflow's current gate file and records its SHA256 in the run manifest. Explicitly exported `EOS_*` values still override values from that file. Set `EOS_TOKENIZER=/path/to/tokenizer.mll` when you want to reuse an existing tokenizer instead of training one from `EOS_TOKENIZER_CORPUS`.

When prepared JSONL is text and a tokenizer is available, the production workflow tokenizes train, validation, and hard-eval JSONL into run-local token files before training. Training and eval then read token JSONL directly, which front-loads BPE cost, makes the optimizer profile reflect model work, and records the generated token files in `datasets.sha256`.
Use `eos train-embed --no-tokenizer` when directly training token JSONL beside a sibling tokenizer; otherwise the CLI intentionally auto-discovers that tokenizer and treats the JSONL as text.
Use `eos diagnose-train-metrics /path/to/train.metrics.json` to explain backend use, transfer pressure, and suspicious training/eval counters from any run. Use `eos gate-train-metrics --thresholds /path/to/thresholds.env --scope quality /path/to/final-eval.metrics.json` to apply the same quality gate outside the production workflow. Use `--scope efficiency` for training throughput and backend counters, and `--scope eval-only` to enforce zero optimizer updates on validation runs.

## TurboQuant Retrieval Gate

Default CorkScrewDB embedder promotion requires a vector-index cost check in addition to dense retrieval quality. The repo-local serving proxy is `scripts/smoke_eos_default_embedder_serving.fw`; it compares the promoted q4/fp16/overfetch250 profile against the q8/fp16/overfetch125 fallback and records p50/p95/p99/max per-query scoring latency:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_default_embedder_serving.fw
```

This smoke uses the in-repo TurboQuant evaluator as CorkScrewDB-relevant vector-index and serving evidence. It is not an actual CorkScrewDB API load/index/search smoke. For the local flat CorkScrewDB API path, run `scripts/smoke_corkscrewdb_child_vectors.fw`; it creates or consumes child-vector caches, loads child vectors with CorkScrewDB `PutVector`, queries with `SearchVector`, rolls child hits up to parent IDs by max score, and records storage, latency, nDCG@10, and recall@100:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_corkscrewdb_child_vectors.fw
```

Set `EOS_CORKSCREW_SMOKE_OVERFETCH` to a comma-separated list, for example `100,12468`, to sweep serving recall/latency without rebuilding the per-bit DB for each overfetch value. Exhaustive/full child overfetch can compare against offline cache-evaluator parity, but it is a higher-latency diagnostic rather than the default serving setting.

The CorkScrewDB smoke defaults to tiny synthetic time-series vectors generated by `scripts/smoke_eos_timeseries_window_vectors.fw` and uses a local checkout from `EOS_CORKSCREW_SMOKE_CORKSCREWDB_REPO` or `/home/draco/work/corkscrewdb`. It proves the local flat load/index/search/accounting path only; it is not remote mode, federation, HNSW, or a model-quality benchmark. For lower-level release checks, run `eos eval-retrieval-turboquant` on at least one capped BEIR-style dataset, and on the full selected retrieval set before updating the `corkscrewdb-default-embedder` alias:

```bash
go run ./cmd/eos eval-retrieval-turboquant \
  --dataset scifact \
  --max-docs 200 \
  --max-queries 20 \
  --bits 2,4,8 \
  --rerank-overfetch 200 \
  --rerank-storage fp16 \
  --metrics-json /data/manta/runs/<run-id>/turboquant-scifact.json \
  --metrics-tsv /data/manta/runs/<run-id>/turboquant-scifact.tsv \
  /data/manta/runs/<run-id>/eos-embed-v1.sealed.mll \
  /data/manta/datasets/eos-embed-v1/raw/scifact
```

The JSON/TSV rows include the dense float32 reference, TurboQuant bit width, nDCG@10/nDCG@100, MRR@10, precision@1/5/10, hit@1/5/10, MAP@10/MAP@100, recall@10/100, quality deltas, vector bytes, rerank storage, rerank sidecar bytes, total vector bytes, compression ratio, total compression ratio, quantization docs/s, direct IP scoring throughput, per-query scoring latency summaries, and optional rerank overfetch/score counts. CorkScrewDB integration is a separate local smoke for this gate; these are the CorkScrewDB-relevant storage and scoring metrics that must stay attached to a promoted default embedder.

Interpret the metric groups separately for promotion. Quality metrics decide whether the candidate ranks relevant documents well enough: nDCG and MAP capture ordering, precision/hit@k capture first-screen success, and recall@100 captures candidate-pool coverage. Compression metrics decide whether q4/q8 are worth the footprint reduction. Throughput metrics are path-specific: sealed Eos evals can include local encoder time, while cached external-vector rows only measure cache load/scoring and do not represent live provider or external model encoding throughput.

For CorkScrewDB multi-vector and time-series designs, use the storage planner before running quality experiments:

```bash
go run ./cmd/eos plan-multivector-storage \
  --dim 128 \
  --baseline-dim 3072 \
  --bits 2,4,8 \
  --vectors-per-object 64,128,256,384 \
  --vector-overhead-bytes 32 \
  --objects 1000
```

For time-series/window-vector planning, derive the child-vector count from point counts, window size, and stride instead of hand-entering `--vectors-per-object`:

```bash
go run ./cmd/eos plan-multivector-storage \
  --dim 128 \
  --baseline-dim 3072 \
  --bits 2,4,8 \
  --series-lengths 256,1024 \
  --window-size 64 \
  --window-stride 16 \
  --vector-overhead-bytes 32 \
  --objects 1000
```

This uses one vector per covering window: a series no longer than the window gets one vector, and longer series include a tail window when the stride does not land exactly on the end. Do not pass `--vectors-per-object` with `--series-lengths`; the CLI fails explicit conflicts so manual and derived modes stay distinct.

This planner answers a different question from `eval-retrieval-turboquant`: how many direct quantized child vectors per parent object fit in the storage cost of one dense fp32 baseline vector. It is a storage gate, not a numeric time-series quality benchmark. The output is TSV by default and optional JSON with fields including `dim`, `baseline_dim`, `bits`, `objects`, `vectors_per_object`, optional `series_length`/`window_size`/`window_stride`/`derived_window_count`, `dense_parent_bytes`, `dense_baseline_bytes`, raw `quantized_vector_bytes`, `vector_overhead_bytes`, `dense_vector_storage_bytes`, `quantized_vector_storage_bytes`, `total_quantized_bytes`, compression ratios, and `vectors_that_fit_in_one_dense_vector`. Omitting `--baseline-dim` preserves same-dim accounting (`baseline_dim=dim`); omitting `--vector-overhead-bytes` preserves ideal payload-only accounting. Use `--dim 128 --baseline-dim 3072` to test the large-baseline interpretation where one 3072d fp32 vector has a 12,288-byte payload, q2 128d children have a 36-byte raw payload, and 128 payload-only children use about `0.375x` of the baseline budget. That ideal math can fit q2/q4/q8 as measured, but CorkScrewDB product claims require overhead-aware planning with per-vector/index-entry bytes before claiming hundreds of vectors for the cost of one. The intended lane is direct child vectors for windows, spans, event histories, or per-object time-series slices. Do not enable `--sidecar-storage fp16` for that lane unless the product explicitly accepts the extra storage; fp16 sidecars are for quality-preserving rerank profiles and erase most of the hundred-child storage advantage when attached to every child vector.

Use the executable budget-frontier smoke when the claim needs a repeatable artifact instead of an inline planner command:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_frontier.fw
```

The default smoke runs the planner for `128d` compact children against a `3072d` dense baseline with q2/q4/q8, child counts `1,16,64,100,128,181,256,341`, `32` bytes of per-vector overhead, no sidecar, and `1000` objects. It writes `summary.tsv`, `manifest.json`, planner JSON, and command logs under `runs/eos-multivector-budget-frontier-smoke-<timestamp>/`; the default gates assert q2 fits at least `181`, q4 at least `100`, and q8 at least `64` child vectors in one dense-vector budget. This is storage/accounting only, not retrieval quality and not CorkScrewDB API latency.

Use the budget-quality smoke when the claim must include both cache-only retrieval quality and overhead-aware planner capacity:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_quality.fw
```

The default run uses the local Eos 128d SciFact child cache in `runs/eos-128d-child-cache-quality-20260615T000000Z/`, evaluates dense/q2/q4/q8 max-child parent retrieval, then plans the inferred 128d child vectors against one 3072d dense baseline vector with `32` overhead bytes and no sidecar. Current default interpretation: q4 stays near dense on this cache (`ndcg@10` drop about `0.002630`, `recall@100` drop about `0.001667`) and the planner fits `123` overhead-aware q4 children per dense-vector budget, so the `100` child-vector target fits. This remains cache-only evidence; it does not measure CorkScrewDB API latency, HNSW/index behavior, remote mode, or DB-directory size.

Use the CorkScrewDB budget-quality smoke when the same compact-cache claim must pass through the actual local flat CorkScrewDB API:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_budget_quality.fw
```

The default run feeds the Eos 128d SciFact child cache and qrels into `scripts/smoke_corkscrewdb_child_vectors.fw` with q4 overfetch `100,12468`, then joins those `PutVector`/`SearchVector` metrics to `plan-multivector-storage` with `--dim 128 --baseline-dim 3072 --vector-overhead-bytes 32 --sidecar-storage none`. It writes a combined `summary.tsv`, `manifest.json`, command logs, nested `corkscrewdb/` run artifacts, and `planner.json` under `runs/eos-corkscrewdb-budget-quality-smoke-<timestamp>/`. Current default API evidence: q4 overfetch100 has `ndcg@10=0.407586`, `recall@100=0.724111`, DB directory multiple `0.048935x`, and p95 search latency about `11.8ms`; q4 full overfetch has `recall@100=0.741889`; the planner fit is `123` q4 children, so the 100-child target fits. Treat this as the local flat CorkScrewDB API counterpart to the cache-only budget-quality smoke and pure planner smoke, not remote mode, federation, HNSW, or native Matryoshka evidence.

Use the HNSW/raw CorkScrewDB budget-quality smoke when the same compact cache must validate HNSW index search rather than compact persistence:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_hnsw_quality.fw
```

The HNSW wrapper feeds the same Eos 128d SciFact child cache into `scripts/smoke_corkscrewdb_child_vectors.fw` with `index_type=hnsw`, `vector_storage=raw`, and a post-load HNSW rebuild, then joins the same q4 planner row. Current default evidence: q4 overfetch100 has `ndcg@10=0.392775`, `recall@100=0.685778`, raw HNSW DB directory multiple `0.347691x`, rebuild time about `20.9s`, and p95 search latency about `1.78ms`; q4 full overfetch has `recall@100=0.741889` and p95 about `33.1ms`. CorkScrewDB rejects HNSW with `quantized_only`, so this is raw-vector index-search validation. It intentionally does not claim the flat quantized-only DB-size multiple; use the vector payload/planner columns only for compact q4 planning comparisons.

To move from storage math to a cache-only quality harness for time-series windows, export text-rendered numeric windows and reuse the existing multivector TurboQuant evaluator. The series JSONL has one parent series per row with `id` or `_id` and numeric `values`; qrels must use the parent series IDs as corpus IDs:

```bash
go run ./cmd/eos export-timeseries-vectors \
  --dataset sensor-window-retrieval \
  --batch-size 64 \
  --output-dim 128 \
  --window-size 64 \
  --window-stride 16 \
  --manifest-json /data/manta/runs/<run-id>/sensor-window-export-128d.json \
  /data/manta/runs/<run-id>/eos-embed-v1.sealed.mll \
  /data/manta/timeseries/sensor-series.jsonl \
  /data/manta/timeseries/queries.jsonl \
  /data/manta/runs/<run-id>/sensor-window-cache-128d
```

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset sensor-window-retrieval \
  --backend text-rendered-timeseries-windows \
  --artifact eos-embed-v1-prefix128 \
  --doc-vectors /data/manta/runs/<run-id>/sensor-window-cache-128d/child-doc-vectors.jsonl \
  --query-vectors /data/manta/runs/<run-id>/sensor-window-cache-128d/query-vectors.jsonl \
  --qrels /data/manta/timeseries/qrels/test.tsv \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  /data/manta/runs/<run-id>/sensor-window-cache-128d
```

`export-timeseries-vectors` renders each window as stable text containing the series ID, window bounds, values, and simple stats before embedding with the normal Eos text embedder path. It also writes BEIR helper `corpus.jsonl` and `queries.jsonl` files into the output directory, so use the export directory as the evaluator dataset directory and pass the parent-series qrels with `--qrels`. This is a bridge quality harness for numeric windows represented as text, not a final trained numeric time-series encoder or a CorkScrewDB load/index/search benchmark.

For a reproducible local smoke of that bridge, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_timeseries_window_vectors.fw
```

The smoke generates five synthetic 12-point parent series, exports 128d text-rendered windows with `--window-size 4 --window-stride 2`, evaluates dense/q2/q4/q8 parent-child retrieval, and runs the storage planner with `--baseline-dim 3072 --vector-overhead-bytes 32`. It writes `summary.tsv` and `manifest.json` under `runs/eos-timeseries-window-vector-smoke-<timestamp>` so the claim is concrete: on this synthetic cache, q8 should match dense quality within the configured tolerance while using a small fraction of a large dense baseline-vector budget. Set `EOS_TS_WINDOW_SMOKE_ARTIFACT` if the default sealed artifact is not present.

After storage planning, run the cache-only parent-child quality harness before making product claims:

For Eos-owned/default embedders, create the cache with the Go-native exporter so the sealed `.mll` runtime path, tokenizer, batching, and normalization match `eval-retrieval` without Python or SentenceTransformers:

```bash
go run ./cmd/eos export-retrieval-vectors \
  --dataset scifact \
  --batch-size 64 \
  --output-dim 128 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16 \
  --manifest-json /data/manta/runs/<run-id>/scifact-child-vector-export-128d.json \
  /data/manta/runs/<run-id>/eos-embed-v1.sealed.mll \
  /data/manta/datasets/eos-embed-v1/raw/scifact \
  /data/manta/runs/<run-id>/scifact-child-cache-128d
```

Without `--document-chunk-words`, the exporter writes `doc-vectors.jsonl` and `query-vectors.jsonl` for `eval-retrieval-vectors` or `eval-retrieval-vectors-turboquant`. With chunking enabled, it writes `child-doc-vectors.jsonl` and `query-vectors.jsonl`; child IDs are deterministic `parent#chunk-0000` word windows and the manifest records document count, query count, child-vector count, written dimension, model dimension, backend, artifact, and output paths. `--output-dim 128` is a prefix-truncated compact cache followed by L2 renormalization. It is a measurement bridge for storage/quality evidence, not a trained Matryoshka or native 128d projection head; a trained compact head is the stronger future path before promotion claims.

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset scifact \
  --backend child-cache \
  --artifact <embedder-label> \
  --doc-vectors /data/manta/runs/<run-id>/scifact-child-cache-128d/child-doc-vectors.jsonl \
  --query-vectors /data/manta/runs/<run-id>/scifact-child-cache-128d/query-vectors.jsonl \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  --metrics-json /data/manta/runs/<run-id>/multivector-turboquant-128d.json \
  --metrics-tsv /data/manta/runs/<run-id>/multivector-turboquant-128d.tsv \
  /data/manta/datasets/eos-embed-v1/raw/scifact
```

This evaluator treats qrels document IDs as parent IDs. Each document-vector JSONL row may include `parent_id`, `child_id`, and `vector`/`embedding`/`values`; absent `parent_id` falls back to `id` or `_id` for one-vector compatibility. The dense reference scores all child vectors and rolls them up by max score per parent, then direct TurboQuant rows quantize child vectors and perform the same max-child parent aggregation. Runs fail by default when any qrels-relevant parent is absent from the child-vector cache; `--allow-missing-relevant` preserves the old filtered behavior only for incomplete-cache diagnostics. TurboQuant rows use a deterministic IP quantizer seed, configurable with `--quantizer-seed`, and metrics record both `allow_missing_relevant` and `quantizer_seed`. Pass `--baseline-dim` when the storage claim is compact child vectors versus a larger dense embedder budget; leaving it at `0` uses the child dimension and is only same-dimension accounting. The metrics report `baseline_dim`, parent/child counts, average and max children per parent, dense baseline bytes, dense child bytes, per-vector quantized bytes, total quantized child bytes, dense-child compression, vectors that fit in one dense baseline, storage multiple relative to one dense baseline vector per parent, scored child pairs, quality metrics/deltas, scores/s, and per-query latency. Follow it with `scripts/smoke_corkscrewdb_child_vectors.fw` when the CorkScrewDB `PutVector`/`SearchVector` path itself must be proven.

For hosted or open external embedders, export BEIR-aligned `doc-vectors.jsonl` and `query-vectors.jsonl` caches and run both `eos eval-retrieval-vectors` and `eos eval-retrieval-vectors-turboquant`. `scripts/export_qwen3_retrieval_vectors.py` is the first provider-boundary exporter for the leading Qwen3 family baseline:

```bash
python3 scripts/export_qwen3_retrieval_vectors.py \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b \
  --dataset-name scifact \
  --model-name Qwen/Qwen3-Embedding-0.6B
```

For parent-child multi-vector runs, keep the same provider boundary and add chunking flags. The exporter writes `child-doc-vectors.jsonl` with `parent_id`, deterministic `child_id`, and `embedding`, while still writing the normal `query-vectors.jsonl`:

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

Current SciFact child-cache evidence uses `datasets/manta-embed-v1/raw/scifact/scifact`, because `datasets/eos-embed-v1/raw/scifact/scifact` was absent for the local runs. With `128` word chunks, `32` overlap, `16` minimum trailing words, `5,183` parents, `12,468` child vectors, and strict qrels coverage (`allow_missing_relevant=false`), `mixedbread-ai/mxbai-embed-large-v1` is stronger than Qwen3 0.6B at dense, q2, q4, and q8 child-max retrieval. Its q8 row scored `0.747799` nDCG@10 / `0.966667` recall@100, beating Qwen3 q8 by `+0.031489` nDCG@10 and `+0.013334` recall@100. The sealed Eos artifact `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll` now has full Go-native child-cache evidence on the same strict lane: export counts were `5,183` docs, `300` queries, `12,468` children, dim `256`, CUDA backend, and `57.771s`; strict eval used `339` relevant pairs, `3,740,400` scored child pairs, and seed `5581486560434873699`. Eos q8 scored `0.461862` nDCG@10 / `0.774778` recall@100 at `3.94x` compression and `0.61x` parent budget, closely preserving Eos dense-child `0.462489` / `0.778111` but remaining materially below mxbai and Qwen3. Treat this as pipeline proof, not default-quality promotion evidence.

Use `--backend` and `--artifact` labels to keep BGE, Qwen, Jina, Voyage, Cohere, OpenAI, and sealed Eos rows comparable. In the scoreboard harness, set `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT=1` and `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_BITS=2,4,8` to append q2/q4/q8 rows for the same cache; add `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_RERANK_OVERFETCH=200` and `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_RERANK_STORAGE=fp16` when fp16 sidecar rerank rows should be emitted for an external cache. External baseline rows remain `not_scored` until the dense and TurboQuant cache metrics are present.

Current external comparison state: Qwen3 is locally consolidated for SciFact, NFCorpus, and full exportable-text FiQA. Its useful compact external row is direct q8 at about `3.98x` vector compression, with SciFact q8 nDCG@10 `0.704128`, NFCorpus q8 nDCG@10 `0.368763`, and FiQA q8 nDCG@10 `0.449614`. mxbai remains stronger than Qwen3 in the existing local short-set evidence. Qwen3 FiQA is not raw-row-complete or judged-coverage complete because one judged test document had empty unexportable text and was skipped.

For targeted quality-frontier mining, first emit per-query diagnostics with `--per-query-jsonl` for selected Eos and an external vector cache. Then run `scripts/mine_retrieval_quality_frontier.py` to produce machine-readable Eos-underperformance rows and optional text hard-negative JSONL. Treat the output as protected candidate training data; it still needs the normal selected-vs-anchor gates on SciFact, NFCorpus, and FiQA before any model promotion.

Hybrid lexical+dense retrieval is a separate retrieval/rerank product surface, not a model-promotion shortcut. Before using hybrid rows as CorkScrewDB default-embedder evidence, select fusion parameters on a dev split and apply the selected setting unchanged to test:

```bash
EOS_REPO_ROOT=$PWD \
EOS_HYBRID_CAL_MODE=dense \
EOS_HYBRID_CAL_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_HYBRID_CAL_DATASET_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_HYBRID_CAL_DATASETS=fiqa \
EOS_HYBRID_CAL_DEV_SPLIT=dev \
EOS_HYBRID_CAL_TEST_SPLIT=test \
ferrous-wheel run scripts/calibrate_eos_embed_hybrid_retrieval.fw
```

For external vector caches, set `EOS_HYBRID_CAL_MODE=vectors`, `EOS_HYBRID_CAL_VECTOR_ROOT`, `EOS_HYBRID_CAL_VECTOR_BACKEND`, and optionally `EOS_HYBRID_CAL_VECTOR_ARTIFACT`. The calibration harness writes JSON, TSV, Markdown, per-setting metrics, per-query JSONL, and command logs under `runs/<run-id>/`. It selects by dev `ndcg_at_10`, tie-breaks by `mrr_at_10` then `recall_at_100`, reports dense and BM25 sanity checks, and records a protection gate showing whether selected test hybrid regresses against dense by more than `EOS_HYBRID_CAL_MAX_NDCG10_REGRESSION` or `EOS_HYBRID_CAL_MAX_RECALL100_REGRESSION`.

Use `eos eval-retrieval-hybrid` for sealed Eos artifacts and `eos eval-retrieval-vectors-hybrid` for external vector caches when applying a calibrated setting; both fuse dense top-k and BM25 top-k over their union. The prior FiQA-selected setting was `--method minmax --alpha 0.75`, where `alpha` is the BM25 weight, but new default claims should cite the current calibration summary rather than reusing that setting blindly. RRF is available with `--method rrf --rrf-k 60 --rrf-lambda 1.0`. In scoreboards, set the selected values explicitly:

```bash
EOS_SCOREBOARD_HYBRID_RETRIEVAL=1
EOS_SCOREBOARD_HYBRID_METHOD=minmax
EOS_SCOREBOARD_HYBRID_ALPHA=0.75
EOS_SCOREBOARD_HYBRID_RRF_K=60
EOS_SCOREBOARD_HYBRID_RRF_LAMBDA=1.0
```

This appends `eos-hybrid` rows by default and, when external vector datasets are configured, `<external>-hybrid` rows. Scoreboard `method` is explicit, such as `hybrid_minmax_alpha0.75` or `hybrid_rrf_k60_lambda1.0`, so gates can select the hybrid mode independently from dense Eos, BM25, and TurboQuant rows. Set `EOS_SCOREBOARD_HYBRID_BASELINE_LABEL=manta-hybrid` only when intentionally producing a legacy-labeled scoreboard. The FiQA dev-selected `minmax_alpha0.75` result is evidence for guarded hybrid retrieval because it improved held-out FiQA test nDCG@10 over dense v2, strict anchor, and BM25-only; it is not evidence that the dense model itself should be promoted.

Before updating the default-shipping anchor, run the scoreboard-level promotion gate against the current sealed anchor scoreboard:

```bash
go run ./cmd/eos gate-scoreboard \
  --category short_retrieval \
  --baseline eos \
  --datasets scifact,nfcorpus,fiqa \
  --metrics ndcg_at_10,recall_at_100 \
  /data/manta/runs/<candidate-scoreboard>/scoreboard.json \
  /data/manta/runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/scoreboard.json
```

The command fails on any missing selected row or any dataset metric below the anchor minus `--tolerance`. Macro deltas are printed only as summary evidence; they are not a substitute for every selected dataset metric passing.

Compatibility note: older scoreboards and run directories use the legacy `manta` / `manta-hybrid` row labels and `manta-embed-v1` artifact names for the same embedder lineage. `eos gate-scoreboard --baseline eos` falls back to legacy `manta` rows when exact `eos` rows are absent, and `--baseline eos-hybrid` similarly falls back to `manta-hybrid`.

Canonical active row labels are `eos` for dense `eos-embed-v1`, `eos-hybrid` for lexical+dense hybrid retrieval, `eos-turboquant` for local direct TurboQuant rows, and `eos-turboquant-rerank` for local TurboQuant candidate overfetch plus rerank rows. Enable the promoted local compact lane in the scoreboard with `EOS_SCOREBOARD_TURBOQUANT=1`, `EOS_SCOREBOARD_TURBOQUANT_BITS=4`, `EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=250`, `EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16`, `EOS_SCOREBOARD_TURBOQUANT_BASELINE=eos-turboquant`, and `EOS_SCOREBOARD_TURBOQUANT_RERANK_BASELINE=eos-turboquant-rerank`. Gate the compact profile with `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch250_fp16_rerank --bits 4 --metrics ndcg_at_10,recall_at_100,total_compression_ratio`. The CorkScrewDB shipping alias should point at the promoted `corkscrewdb-default-embedder` artifact only after the dense, hybrid where relevant, and TurboQuant gates have passed.

Current measured TurboQuant promotion surface: q4/fp16 sidecar rerank at overfetch250 is the promoted compact retrieval profile. It passed the selected-vs-anchor scoreboard gate on SciFact, NFCorpus, and FiQA for `ndcg_at_10,recall_at_100,total_compression_ratio` as `eos-turboquant-rerank` / `turboquant_ip_b4_overfetch250_fp16_rerank` / bits `4`, with total compression `1.590062x`, in `runs/eos-q4-fp16-overfetch250-gate-20260615T000000Z/`. This is a two-stage compact retrieval profile, not q4-only retrieval: direct q4 and direct q8 are not default-promotion candidates. Keep q8/fp16 sidecar rerank at overfetch125 as the lower-risk, lower-rerank-cost fallback: `turboquant_ip_b8_overfetch125_fp16_rerank`, total compression `1.326425x`, evidence in `runs/eos-fp16-overfetch125-gate-20260614T000000Z/`.

For candidate iteration, prefer the guarded runner so training, scoring, and the scoreboard gate produce one acceptance manifest:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GUARD_ANCHOR_SCOREBOARD=runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/scoreboard.json \
ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw
```

The runner inherits the normal `EOS_TRAIN_JSONL`, `EOS_EVAL_JSONL`, `EOS_HARD_EVAL_JSONL`, tokenizer, training, and scoreboard knobs. It defaults to `scifact,nfcorpus,fiqa`, `ndcg_at_10,recall_at_100`, `short_retrieval`, and `eos`, writes `runs/eos-embed-v1-guarded-<timestamp>/manifest.json`, and exits nonzero when the candidate is rejected. Set `EOS_GUARD_BASELINE=manta` only for legacy-labeled candidate scoreboards. Set `EOS_GUARD_FAIL_ON_GATE=0` only when you need a non-failing smoke that still records `"gate_status": "rejected"`. Dirty working trees remain blocked by the training script unless the caller sets `EOS_ALLOW_DIRTY=1` or explicitly sets `EOS_GUARD_ALLOW_DIRTY=1`.

For a dry smoke with existing artifacts:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GUARD_CANDIDATE_DIR=runs/<candidate-run> \
EOS_GUARD_SCOREBOARD_JSON=runs/<candidate-scoreboard>/scoreboard.json \
EOS_GUARD_ANCHOR_SCOREBOARD=runs/<anchor-scoreboard>/scoreboard.json \
EOS_GUARD_FAIL_ON_GATE=0 \
ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw
```

Contrastive training uses pair-length-aware bucketing by default so batches reach larger exact-length matmul groups. `EOS_TRAIN_LENGTH_BUCKET_WINDOW` controls the shuffled sort window; larger values can improve grouping but must be profiled because they reduce local length randomness and can increase per-batch working-set pressure.

The production workflow defaults `EOS_EVAL_EVERY_STEPS=0`. Keep within-epoch eval disabled for full candidate runs unless you are debugging convergence; epoch eval, final validation eval, and hard holdout eval still run and are enough for release gating. On the acquired full split, step-level eval every 4 batches adds many full eval passes and dominates transfer without improving the optimizer update itself.

Eval-only candidate gates batch pairwise forward encodes by exact token length. `EOS_TRAIN_PAIR_EVAL_BATCH_SIZE` defaults to `256`; set `EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1` only for scalar A/B checks.

`EOS_TRAIN_ENABLE_FAST_GELU=1` is available for throughput experiments. It changes GELU math from precise tanh to a bounded rational tanh approximation, so use it only when the exact-GELU candidate and fast-GELU candidate are both evaluated against validation and hard holdout gates.

## Metric Thresholds

The default acquired eval files are pairwise positive/negative judgments with one deterministic sampled negative per positive. The initial release gates are:

```text
EOS_SELECT_METRIC=score_margin
EOS_MIN_AUC=0.70
EOS_MIN_THRESHOLD_ACCURACY=0.65
EOS_MIN_SCORE_MARGIN=0.05
EOS_MAX_LOSS=0.35
```

These gates are intentionally concrete rather than advisory. AUC must clear 0.70 as a threshold-free separability check, calibrated threshold accuracy must clear 0.65 against a 0.50 random baseline, positive scores must beat negative scores by at least 0.05 on average, and pairwise loss must stay at or below 0.35. Tighten them after the first stable full-size candidate establishes the project baseline.

## Training Efficiency Gates

Production candidate runs can also enforce hardware-specific training efficiency gates from `train.metrics.json`:

```text
EOS_MIN_TRAIN_PAIRS_PER_SEC=85000
EOS_MIN_OPTIMIZER_STEPS_PER_SEC=0.08
EOS_MAX_MATMUL_RUNS=400000
EOS_MAX_MATMUL_RUN_UPLOAD_MB=950000
EOS_MAX_MATMUL_RUN_DOWNLOAD_MB=485000
```

Set these from a known-good run on the target trainer host. Keep quality gates hardware-independent, and keep efficiency gates host-specific so a CPU fallback or data-transfer regression does not silently produce a slow candidate.

## Corpus Path

Use corpus mode only when you intentionally want this run to mine the train/eval pairs:

```bash
EOS_RUN_ROOT=/data/manta/runs \
EOS_REPO_ROOT=$PWD \
EOS_RUN_ID=eos-embed-v1-20260412-corpus-a \
EOS_CORPUS=/data/manta/corpus/prod-corpus.txt \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/hard-eval.jsonl \
EOS_EPOCHS=3 \
EOS_BATCH_SIZE=1024 \
EOS_MAX_PAIRS=0 \
EOS_EVAL_PAIRS=512 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

Corpus mode writes mined pairs and the tokenizer into the run directory, then records their SHA256 values in `datasets.sha256`.

## JSONL Formats

Token contrastive JSONL:

```json
{"query_tokens":[1,2,3],"positive_tokens":[1,2,3],"query_mask":[1,1,1],"positive_mask":[1,1,1]}
```

Text-pair JSONL:

```json
{"query":"how to reset password","document":"reset your password from account settings","label":1}
{"left":"how to reset password","right":"billing invoice export","label":0}
```

Positive-only text pairs can train contrastively. Mixed positive/negative text pairs are valid for eval-only gates.

Token-pair eval JSONL:

```json
{"left_tokens":[1,2,3],"right_tokens":[1,2,3],"left_mask":[1,1,1],"right_mask":[1,1,1],"target":1}
{"left_tokens":[1,2,3],"right_tokens":[9,8],"left_mask":[1,1,1],"right_mask":[1,1],"target":0}
```

## Release Gate

Current sealed-verified local anchor: `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll`, SHA256 `a7461b47784ea7434cf6048f33f6c281ef19887cfa9d0c699b6f2fba079f2b67`. This is a legacy-named `manta-embed-v1` artifact for the `eos-embed-v1` lineage. It scores above the previous sealed anchor on all retrieval rows, with macro nDCG@10 `0.265891` versus `0.148144` and macro recall@100 `0.452844` versus `0.339353`. The sealed scoreboard is under `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/`.

The sealed-vs-train-package comparison recorded zero nonzero quality or count deltas against `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-scoreboard/`. Treat this as the local sealed anchor; broader release claims still require the normal release-gate evidence below.

A candidate is releasable only when:

- `manifest.json` status is `success`
- `logs/final-eval.log` and `logs/hard-eval.log` report `optimizer_updates=0`
- configured metric gates pass on hard eval
- `logs/inspect-package.log` reports `package verify: OK`
- `logs/inspect-sealed.log` reports `package verify: OK`
- `artifacts.sha256` contains the sealed MLL hash
- `eos gate-scoreboard --datasets scifact,nfcorpus,fiqa --metrics ndcg_at_10,recall_at_100 <candidate-scoreboard>/scoreboard.json <sealed-anchor-scoreboard>/scoreboard.json` passes

The release artifact is:

```text
<run-dir>/eos-embed-v1.sealed.mll
```

Keep the full run directory with the released artifact. It is the audit trail for reproducing or rejecting the candidate later.
