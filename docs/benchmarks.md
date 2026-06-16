# Benchmarks

Eos has three benchmark layers:

- Go microbenchmarks for isolated runtime kernels and trainer math.
- End-to-end training smokes for the native default embedding-model path.
- Embedder scoreboards for retrieval quality and serving efficiency claims.

For the compact child-vector storage frontier that sits beside the embedding benchmarks, see [TurboQuant Multi-Vector Frontier](turboquant-multivector-frontier.md). It centralizes the CorkScrewDB/Eos many-child-vectors-per-parent thesis, evidence boundaries, and promotion gates.

Run the default microbenchmarks with Ferrous Wheel:

```bash
EOS_BENCH_ROOT=$PWD ferrous-wheel run scripts/bench.fw
```

Run CUDA microbenchmarks:

```bash
EOS_BENCH_ROOT=$PWD EOS_BENCH_CUDA=1 ferrous-wheel run scripts/bench.fw
```

Run the sparse-attention x TurboQuant measurement layer:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/bench_sparse_attention.fw
```

The sparse-attention harness first writes a routed preflight plan as `sparse-attention-plan.tsv` and `sparse-attention-plan.json`, then records CUDA benchmark output as `sparse-attention-bench.jsonl`, `sparse-attention-bench.txt`, `sparse-attention-bench-summary.tsv`, and `sparse-attention-scaling.tsv` under `runs/<run-id>/`. The Go benchmark lines and summary table include selected-key count, candidate-key budget, estimated scores per query, score fraction, subquadratic-plan flag, TurboQuant K/V MiB, and logical K/V compression ratio. The scaling table fits log-log time alpha for exact f16 and routed TurboQuant rows, and the harness fails routed TurboQuant when alpha exceeds `EOS_SPARSE_BENCH_MAX_ROUTED_TIME_ALPHA` (`0.95` by default; set `0` to disable during exploratory runs). Keep exact dense costs bounded with `EOS_SPARSE_BENCH_EXACT_KEY_LENS` (`1024,4096` by default) while extending routed scaling with `EOS_SPARSE_BENCH_ROUTED_KEY_LENS` (`1024,4096,16384` by default).

Run the default-model training smoke from a local asset package:

```bash
EOS_BENCH_ROOT=$PWD EOS_BENCH_MODEL_ASSETS=/path/to/assets/eos-embed-v1 ferrous-wheel run scripts/bench.fw
```

The model smoke copies the package into a temporary directory before training. It does not mutate the source asset directory.

Evaluate an existing candidate package against a token JSONL or text-pair JSONL eval file without running optimizer steps:

```bash
go run ./cmd/eos train-embed --eval-only /path/to/eos-embed-v1.mll /path/to/eval-mini.jsonl
```

When the package has a sibling `.tokenizer.mll`, text eval JSONL is tokenized automatically. Pass `--tokenizer /path/to/tokenizer.mll` to use an explicit tokenizer.

For a production candidate run with acquired BEIR-format datasets, provenance, metric gates, sealed export, and SHA256 manifests, use `scripts/acquire_manta_embed_v1_datasets.fw` followed by `scripts/train_manta_embed_v1_candidate.fw`. See `docs/production-embedding.md`.

Build the long-context embedder wedge scoreboard from an existing sealed candidate:

```bash
EOS_REPO_ROOT=$PWD \
EOS_SCOREBOARD_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_SCOREBOARD_PAIRWISE_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_SCOREBOARD_HARD_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_SCOREBOARD_RETRIEVAL_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The scoreboard run writes `scoreboard.tsv`, `scoreboard.json`, per-task metrics JSON, command logs, and a run-local `eos` binary under `runs/<run-id>/`. Dense local rows default to `baseline=eos`; hybrid rows default to `baseline=eos-hybrid`; local TurboQuant rows use `eos-turboquant` for direct quantized scoring and `eos-turboquant-rerank` for quantized candidate overfetch plus rerank storage modes such as fp16; external TurboQuant rows use `<external>-turboquant`. Set `EOS_SCOREBOARD_BASELINE_LABEL=manta` and `EOS_SCOREBOARD_HYBRID_BASELINE_LABEL=manta-hybrid` only when intentionally producing a legacy-labeled scoreboard. Pairwise rows use `EOS_SCOREBOARD_PAIRWISE_ARTIFACT` when set, or infer the sibling trainable package from a sealed artifact path. Add `EOS_SCOREBOARD_LONG_ROOT` and `EOS_SCOREBOARD_LONG_DATASETS` when long-document retrieval datasets are prepared.

Calibrate hybrid retrieval before using `eos-hybrid` rows as product evidence:

```bash
EOS_REPO_ROOT=$PWD \
EOS_HYBRID_CAL_MODE=vectors \
EOS_HYBRID_CAL_VECTOR_ROOT=runs/external-vector-caches/qwen3-0.6b \
EOS_HYBRID_CAL_VECTOR_BACKEND=qwen3 \
EOS_HYBRID_CAL_DATASETS=nfcorpus \
EOS_HYBRID_CAL_DEV_SPLIT=dev \
EOS_HYBRID_CAL_TEST_SPLIT=test \
ferrous-wheel run scripts/calibrate_eos_embed_hybrid_retrieval.fw
```

The calibration run measures dense, BM25, and candidate hybrid settings on dev, chooses by nDCG@10 with MRR@10 and recall@100 tie-breakers, then evaluates only the selected setting on test. Its summary JSON/TSV/Markdown includes dense/BM25 comparisons, selected test metrics, protection gate deltas against dense, command logs, and optional sentinel query rows from `EOS_HYBRID_CAL_SENTINEL_QUERY_IDS`. When applying a selected setting with `eos eval-retrieval-hybrid` or `eos eval-retrieval-vectors-hybrid`, `--dense-protect-top-k N` preserves the original dense top-N prefix before appending the fused hybrid tail; leave it at `0` for the unguarded fusion order.

Retrieval metric glossary:

- `ndcg_at_10` and `ndcg_at_100`: rank-discounted relevance quality at shallow and broad retrieval depths.
- `mrr_at_10`: reciprocal rank of the first relevant result within the first 10 results.
- `precision_at_1`, `precision_at_5`, and `precision_at_10`: relevant results divided by the fixed cutoff.
- `hit_at_1`, `hit_at_5`, and `hit_at_10`: whether at least one relevant result appears by the cutoff, averaged across queries.
- `map_at_10` and `map_at_100`: mean average precision over relevant positives up to the cutoff.
- `recall_at_10` and `recall_at_100`: fraction of qrels positives recovered by the cutoff.

For default-shipping decisions, read these together. nDCG and MAP catch ranking quality, precision/hit@k reflect first-screen usefulness, and recall@100 protects downstream reranking or broader candidate generation. Compression rows (`vector_bytes`, `dense_vector_bytes`, `compression_ratio`) describe index footprint; throughput rows (`documents_per_second`, `queries_per_second`, `scores_per_second`, `docs_per_second`) describe encoder, cache-load, quantization, and scoring speed depending on the path. Cached external-vector rows do not measure live provider or model encoder throughput; they only measure loading/scoring the already-exported vectors.

Score external embedders by exporting document and query vector JSONL caches, then run the same BEIR metric code:

```bash
python3 scripts/export_qwen3_retrieval_vectors.py \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b \
  --dataset-name scifact \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --batch-size 16

go run ./cmd/eos eval-retrieval-vectors \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b/scifact/doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b/scifact/query-vectors.jsonl \
  --metrics-json runs/scifact.qwen3.retrieval.metrics.json \
  datasets/eos-embed-v1/raw/scifact/scifact
```

`scripts/export_qwen3_retrieval_vectors.py` keeps Qwen3/SentenceTransformers dependencies outside the Go runtime. It writes `embedding` arrays and requests normalized embeddings by default; the Eos evaluator normalizes vectors again before scoring. Use caps such as `--max-docs 200 --max-queries 20` for smoke runs.

Each vector cache row must contain `id` or `_id` plus one of `vector`, `embedding`, or `values`, for example `{"id":"doc-1","vector":[0.1,0.2]}`. The evaluator normalizes vectors, requires matching dimensions, scores cosine/dot-product ranking, and emits the same `manta.embedding_retrieval_metrics.v1` JSON as `eval-retrieval`. To add external rows to the Ferrous Wheel scoreboard, place caches at `<EOS_SCOREBOARD_EXTERNAL_VECTOR_ROOT>/<dataset>/doc-vectors.jsonl` and `query-vectors.jsonl`, then set `EOS_SCOREBOARD_EXTERNAL_VECTOR_DATASETS`, `EOS_SCOREBOARD_EXTERNAL_VECTOR_BASELINE`, `EOS_SCOREBOARD_EXTERNAL_VECTOR_BACKEND`, and optionally `EOS_SCOREBOARD_EXTERNAL_VECTOR_ARTIFACT`.

Compare the same external caches after TurboQuant IP document-vector compression:

```bash
go run ./cmd/eos eval-retrieval-vectors-turboquant \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors /data/external-vector-caches/scifact/doc-vectors.jsonl \
  --query-vectors /data/external-vector-caches/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --metrics-json runs/scifact.qwen3.turboquant.metrics.json \
  --metrics-tsv runs/scifact.qwen3.turboquant.metrics.tsv \
  /data/manta/datasets/eos-embed-v1/raw/scifact/scifact
```

This external-cache TurboQuant path is the main apples-to-apples comparison surface for CorkScrewDB default promotion: BGE, Qwen, Jina, Voyage, Cohere, OpenAI, and sealed Eos rows can all be scored as dense vectors and as q2/q4/q8 TurboQuant document indexes without adding provider SDKs to Eos.

The scoreboard can append TurboQuant rows for the same external caches:

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

Append local sealed Eos TurboQuant rows in the same scoreboard:

```bash
EOS_SCOREBOARD_TURBOQUANT=1 \
EOS_SCOREBOARD_TURBOQUANT_BITS=4 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=250 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16 \
EOS_SCOREBOARD_TURBOQUANT_BASELINE=eos-turboquant \
EOS_SCOREBOARD_TURBOQUANT_RERANK_BASELINE=eos-turboquant-rerank \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The direct and rerank rows share the same metrics JSON but use separate scoreboard baselines. Gate the promoted compact reranked profile with `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch250_fp16_rerank --bits 4 --metrics ndcg_at_10,recall_at_100,total_compression_ratio`. Direct q4 and direct q8 are useful diagnostics, but they are not default-promotion candidates; the promoted q4 profile is two-stage q4 retrieval plus fp16 sidecar rerank.

Qwen3 is now locally consolidated for SciFact and NFCorpus only. Its useful compact external row is direct q8 at about `3.98x` vector compression, with SciFact q8 nDCG@10 `0.704128` and NFCorpus q8 nDCG@10 `0.368763`. mxbai remains stronger than Qwen3 in the existing local SciFact/NFCorpus evidence. Qwen3 FiQA is still subset-only, so do not use it for full short-set claims until the cache is repaired.

Run the TurboQuant retrieval storage/scoring gate against the same sealed candidate and a capped BEIR-style corpus:

```bash
go run ./cmd/eos eval-retrieval-turboquant \
  --dataset scifact \
  --max-docs 200 \
  --max-queries 20 \
  --bits 2,4,8 \
  --rerank-overfetch 200 \
  --rerank-storage fp16 \
  --metrics-json runs/turboquant-retrieval-scifact.json \
  --metrics-tsv runs/turboquant-retrieval-scifact.tsv \
  /path/to/eos-embed-v1.sealed.mll \
  /data/manta/datasets/eos-embed-v1/raw/scifact
```

The TurboQuant gate embeds the corpus once, records the dense float32 reference nDCG@10/recall@100, then quantizes document vectors with `m31labs.dev/turboquant` IP-preserving quantizers and reports per-bit quality deltas, vector bytes, rerank storage, rerank sidecar bytes, total vector bytes, compression ratio, total compression ratio, docs/s, scores/s, per-query p50/p95/p99/max scoring latency, and optional rerank overfetch/score counts. Use the capped command for smoke/release checks and remove the caps for a full CorkScrewDB-relevant vector-index promotion run.

For the promoted compact default profile, prefer the repo-local serving proxy:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_default_embedder_serving.fw
```

It compares q4/fp16/overfetch250 with q8/fp16/overfetch125 and writes `summary.tsv` plus `manifest.json` under `runs/eos-default-embedder-serving-smoke-<timestamp>/`. This is not an actual CorkScrewDB API smoke; it is the in-repo TurboQuant serving proxy.

For the local flat CorkScrewDB child-vector API path, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_corkscrewdb_child_vectors.fw
```

The default run generates a tiny synthetic time-series child-vector cache, loads children into CorkScrewDB with `PutVector`, searches query vectors with `SearchVector`, rolls child hits up to parent IDs by max score, and writes per-bit metrics, `summary.tsv`, and `manifest.json` under `runs/eos-corkscrewdb-child-vector-smoke-<timestamp>/`. Set `EOS_CORKSCREW_SMOKE_CHILD_VECTORS`, `EOS_CORKSCREW_SMOKE_QUERY_VECTORS`, and `EOS_CORKSCREW_SMOKE_QRELS` to reuse a full child-vector cache. `EOS_CORKSCREW_SMOKE_OVERFETCH` also accepts comma-separated values such as `100,12468`; the separate-child smoke loads one DB per bit width and reuses it across overfetch values so serving recall/latency sweeps do not pay duplicate insert cost. Exhaustive/full child overfetch is useful when comparing against offline cache-evaluator parity, but it is expected to cost more latency. This is local flat API/storage/latency/quality smoke evidence only; it is not remote mode, federation, HNSW, or a model-quality benchmark.

To measure CorkScrewDB packed parent-object persistence instead of separate child entries, run the same smoke with `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors`, `EOS_CORKSCREW_SMOKE_INDEX_TYPE=flat`, and `EOS_CORKSCREW_SMOKE_VECTOR_STORAGE=quantized_only`. Packed mode groups children by parent, calls `PutMultiVector` once per parent, and queries parents with exact local flat `SearchParentsVector`; it records `layout`, `search_mode`, `parent_search_exact=true`, `parent_insert_count`, and `overfetch_children=0`. Compare its DB directory multiple against the default `separate_child_vectors` row at the same bit width; vector payload accounting stays per child vector.

For the pure multi-vector storage/accounting frontier, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_multivector_budget_frontier.fw
```

This smoke calls `go run ./cmd/eos plan-multivector-storage` for the default `128d` compact-child versus `3072d` dense-baseline shape, records planner JSON, command logs, `summary.tsv`, and `manifest.json` under `runs/eos-multivector-budget-frontier-smoke-<timestamp>/`, and gates current per-child-entry fit counts at q2 >= 181, q4 >= 100, q8 >= 64 plus packed-parent target fit counts at q2 >= 341, q4 >= 180, q8 >= 93 children per dense-vector budget. Interpret it beside the time-series and CorkScrewDB API smokes as byte accounting only: it does not measure retrieval quality, CorkScrewDB API latency, or measured DB-directory cost.

For the product/use-case frontier, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_multivector_usecase_frontier.fw
```

This smoke also calls the planner, but records named scenarios in `summary.tsv` and `manifest.json`: same-dim 100-child control, 100-child document spans, 100 time-series windows derived from `series_length=1648`, `window_size=64`, `window_stride=16`, a 180-child event/trace timeline, and a 341-child q2 frontier. Its gates keep the claim narrow: same-dim 100-child packed q2/q4 fails, 3072d-baseline packed q2/q4 100-child scenarios fit, packed q4 fits 180 children at the edge, and packed q2 fits 341 children. q8 rows are reported as contrast rather than required to fit.

For the combined cache-only quality plus overhead-aware budget gate, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_multivector_budget_quality.fw
```

The default smoke reuses the existing Eos 128d SciFact child cache under `runs/eos-128d-child-cache-quality-20260615T000000Z/`, calls `eval-retrieval-multivector-turboquant`, infers the 128d child shape from the dense child-vector bytes, then calls `plan-multivector-storage` with `--baseline-dim 3072 --vector-overhead-bytes 32` and a `100` child-vector fit gate. Current default interpretation: direct q4 is near dense on this cache (`ndcg@10` drop about `0.002630`, `recall@100` drop about `0.001667`) while the overhead-aware planner fits `123` q4 child vectors in one 3072d dense-vector budget, so 100 child vectors fit with overhead. It remains cache-only evidence, not CorkScrewDB API latency or DB-directory measurement.

For the actual local flat CorkScrewDB API budget-quality smoke on the compact Eos child cache, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_budget_quality.fw
```

This wrapper feeds `runs/eos-128d-child-cache-quality-20260615T000000Z/scifact/child-doc-vectors.jsonl`, matching query vectors, and SciFact test qrels into `scripts/smoke_corkscrewdb_child_vectors.fw` with q4 overfetch `100,12468`, then joins the API metrics to the overhead-aware planner row for `--dim 128 --baseline-dim 3072 --vector-overhead-bytes 32 --sidecar-storage none`. The default run writes `summary.tsv`, `manifest.json`, command logs, nested CorkScrewDB artifacts under `corkscrewdb/`, and `planner.json` under `runs/eos-corkscrewdb-budget-quality-smoke-<timestamp>/`. Current default API result: q4 overfetch100 records `ndcg@10=0.407586`, `recall@100=0.724111`, DB directory multiple `0.048935x`, and p95 search latency about `11.8ms`; q4 full overfetch records `recall@100=0.741889`; the planner fits `123` q4 children in one 3072d dense-vector budget and the 100-child target fits. The packed-parent budget wrapper now forwards and records compact packed knobs. Verified q4 `packed_parent_multivectors` run `runs/eos-corkscrewdb-budget-quality-packed-q4-compact-20260616T000000Z/` used `packed_metadata_mode=none` and `packed_child_id_mode=ordinal` on the real Eos SciFact child cache with local flat exact parent search; it recorded DB bytes `1,653,983`, DB directory multiple `0.025970x`, p95 `9.505725ms`, `ndcg@10=0.407586`, and `recall@100=0.741889`. DB bytes were `0.4935x` of the prior packed full/source run and `0.5307x` of the separate-child default run. This is actual local flat CorkScrewDB evidence for the compact Eos child cache, beside the cache-only quality smoke and pure planner smoke; it is not remote mode, federation, HNSW, or a native Matryoshka embedding result.

The same wrapper also supports `EOS_CORKSCREW_BUDGET_QUALITY_LAYOUT=single_parent_vectors` for a real document-span layout baseline. On the existing full Eos SciFact child cache at `runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-full/scifact/`, the wrapper compared q4/q8 packed `none`/`ordinal` parent multivectors with mean-pooled single-parent vectors over `5,183` parents, `12,468` child spans, `300` queries, and `256d` vectors. Packed run `runs/eos-real-scifact-full-packed-none-ordinal-q4q8-diagnostic2-20260616T000000Z/` recorded q4 `ndcg@10=0.449435`, `recall@100=0.773111`, DB directory multiple `0.032900x`, and p95 `15.934540ms`; q8 recorded `0.461862`, `0.774778`, `0.057958x`, and `30.366188ms`. Single-parent run `runs/eos-real-scifact-full-single-parent-q4q8-diagnostic-20260616T000000Z/` recorded q4 `ndcg@10=0.406498`, `recall@100=0.743111`, DB directory multiple `0.022411x`, and p95 `7.075771ms`; q8 recorded `0.422597`, `0.745889`, `0.032827x`, and `13.242884ms`. Treat this as real local flat document-span layout evidence: packed preserves child-span evidence better at higher storage/latency, while single-parent is smaller and faster but mean-pools span structure away. The diagnostic run disabled old q4 quality/storage/planner floors; it does not claim remote mode, HNSW, federation, or native Matryoshka behavior.

For the same compact cache through CorkScrewDB HNSW/raw-vector indexing, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_hnsw_quality.fw
```

This wrapper uses `scripts/smoke_corkscrewdb_child_vectors.fw` with `EOS_CORKSCREW_SMOKE_INDEX_TYPE=hnsw`, `EOS_CORKSCREW_SMOKE_VECTOR_STORAGE=raw`, and rebuilds HNSW after loading before joining the same planner row. It writes `summary.tsv`, `manifest.json`, command logs, nested CorkScrewDB artifacts under `corkscrewdb/`, and `planner.json` under `runs/eos-corkscrewdb-hnsw-quality-smoke-<timestamp>/`. Current default HNSW result: q4 overfetch100 records `ndcg@10=0.392775`, `recall@100=0.685778`, raw HNSW DB directory multiple `0.347691x`, rebuild time about `20.9s`, and p95 search latency about `1.78ms`; q4 full overfetch records `recall@100=0.741889` and p95 about `33.1ms`. CorkScrewDB rejects `quantized_only` with HNSW, so this smoke trades compact quantized-only persistence for raw-vector HNSW index-search validation. Do not use its raw DB directory multiple as the q4 quantized-only storage claim; keep the q4 vector payload/planner fit columns as separate planner accounting.

For synthetic time-series window child vectors through the actual local flat CorkScrewDB API, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

This wrapper first runs `scripts/smoke_eos_timeseries_window_vectors.fw` into a nested `timeseries/` directory, then feeds the generated `child-doc-vectors.jsonl`, `query-vectors.jsonl`, and `qrels.tsv` into `scripts/smoke_corkscrewdb_child_vectors.fw` using flat `quantized_only` storage by default. It writes a joined `summary.tsv` and `manifest.json` under `runs/eos-corkscrewdb-timeseries-window-smoke-<timestamp>/`, plus nested `timeseries/` and `corkscrewdb/` artifacts. Current default result: `5` parents, `25` child windows, `5` derived windows per parent; q4 flat/quantized_only records `ndcg@10=1.000000`, `recall@100=1.000000`, planner fit `123`, vector payload multiple `0.034180x`, and DB directory multiple `0.117643x`; q8 records `ndcg@10=0.926186`, `recall@100=1.000000`, planner fit `75`, vector payload multiple `0.060221x`, and DB directory multiple `0.143685x`. Observed p95 latency for the default synthetic smoke has been sub-ms, but it varies by run. Treat this as proof that cheap child/window vectors under parent series are usable through local CorkScrewDB `PutVector`/`SearchVector`; the inputs are synthetic text-rendered numeric windows, not a trained numeric time-series encoder, and DB directory size is measured separately from vector-payload/planner accounting.

For a contrastive parent-layout smoke, set `EOS_CORKSCREW_TS_WINDOW_SCENARIO=contrastive_needle` and compare `EOS_CORKSCREW_TS_WINDOW_LAYOUT=packed_parent_multivectors` against `single_parent_vectors` on the same generated local-window inputs. The scenario keeps each parent mostly on a shared baseline and inserts one short distinctive regime, so mean-pooling can dilute the query-relevant local window while packed child multivectors keep the window/facet available. Verified q2 runs recorded `24` parents, `264` child windows, and `11` windows per parent: packed q2 `none`/`ordinal` run `runs/eos-corkscrewdb-timeseries-contrastive-needle-q2-packed-20260616T000000Z/` recorded `ndcg@10=0.769694`, `recall@100=1.000000`, planner packed fit `341`, packed planner multiple `0.034740x`, vector payload multiple `0.032227x`, DB directory multiple `0.042772x`, and p95 `0.045773ms`; single-parent q2 run `runs/eos-corkscrewdb-timeseries-contrastive-needle-q2-single-parent-20260616T000000Z/` recorded `ndcg@10=0.677304`, `recall@100=1.000000`, vector payload multiple `0.002930x`, DB directory multiple `0.019687x`, and p95 `0.032332ms`. q4 showed the same direction but smaller split: packed `0.831824` nDCG@10 versus single-parent `0.818766`, both with recall@100 `1.000000`. Treat this as a local-window/facet preservation smoke, not the q2-341 storage-capacity proof.

For a q4-only parent-variant scale/stress profile with 100 child windows per parent and enough parent objects to separate fixed DB-directory overhead from per-vector TurboQuant payload accounting, run:

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

The 1648/64/16 shape derives exactly 100 covering windows per parent. With `EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20`, the five query patterns expand to `100` parent series and `10,000` child windows while keeping the original five pattern queries; each query has `20` relevant parent variants, for `100` relevant query-parent pairs. Verified separate-child run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-20260616T000000Z/` recorded q4 flat/`quantized_only` with planner fit `123`, planner storage multiple `0.811688x`, vector payload multiple `0.683594x`, DB directory multiple `2.293748x`, DB bytes `2,818,558`, `ndcg@10=0.352927`, `recall@100=0.560000`, p95 `11.948319ms`, and overfetch `500`.

For packed parent-object persistence at the same q4 scaled shape, rerun with `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors`, flat `quantized_only`, and exact parent search. The packed smoke also supports `EOS_CORKSCREW_SMOKE_PACKED_METADATA_MODE=full|minimal|none` and `EOS_CORKSCREW_SMOKE_PACKED_CHILD_ID_MODE=source|ordinal`; defaults preserve the full/source behavior. Verified packed `none`/`ordinal` run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-packed-minimal-20260616T000000Z/` recorded `layout=packed_parent_multivectors`, `parent_insert_count=100`, `parent_search_exact=true`, `overfetch_children=0`, the same planner/vector payload multiples and `ndcg@10=0.352927`, plus `recall@100=1.000000`, DB directory multiple `0.844660x`, DB bytes `1,037,918`, and p95 `5.916649ms`. Against the prior full-metadata packed run, DB bytes were `0.421237x` as large, about `57.88%` lower; against the separate-child run, DB bytes were `0.368244x` as large, about `63.18%` lower. Treat the recall gain as exact parent rollup versus bounded separate-child overfetch, not as model-quality improvement.

Corrected q2-341 packed-parent frontier evidence is now a unified API run rather than planner math alone. Wrapper run `runs/eos-corkscrewdb-timeseries-window-q2-341-compact-v5-unified-20260616T000000Z/`, against CorkScrewDB commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4` (`update(snapshot): Add compact format for quantized ordinal children`), generated the child/query/qrels inputs, packed planner evidence, and measured persisted DB directory evidence for q2 `128d`, `packed_parent_multivectors`, `packed_metadata_mode=none`, and `packed_child_id_mode=ordinal` with `100` parents, `34,100` child windows, and `341` windows per parent. It recorded `quantized_vector_bytes=36`, `quantized_child_bytes=1,227,600`, vector payload multiple `0.9990234375x`, packed planner bytes `12,308`, packed planner multiple `0.999025974025974x`, measured DB directory bytes `1,237,818`, DB directory multiple `1.0073388671875x`, `ndcg@10=0.4493940305106442`, `recall@100=1.000000`, and p95 `1.418733ms`, passing planner-fit, vector-payload, DB-directory, and p95 gates. With CorkScrewDB compact snapshot v5 ordinal encoding, the persisted DB directory is approximately one dense parent-vector budget for this strict shape; without that compact snapshot path, or for richer child records, keep DB directory cost separate.

These scaled time-series rows are synthetic text-rendered smoke evidence: they improve DB overhead measurement by scaling parent objects and show a local flat packed-parent storage win, but they are not production quality, remote/federation/HNSW evidence, or a trained numeric time-series encoder claim. Keep the DB directory max disabled unless a fresh run establishes a stable machine-local threshold.

Current measured local result: q4/fp16 sidecar rerank at overfetch250 is the promoted compact retrieval profile. It passed the selected-vs-anchor scoreboard gate on SciFact, NFCorpus, and FiQA for `ndcg_at_10,recall_at_100,total_compression_ratio` as `eos-turboquant-rerank` / `turboquant_ip_b4_overfetch250_fp16_rerank` / bits `4`, with total compression `1.590062x`, in `runs/eos-q4-fp16-overfetch250-gate-20260615T000000Z/`. This is not q4-only retrieval: direct q4 misses dense quality on SciFact and FiQA, and direct q8 is not the promoted default path. Keep q8/fp16 sidecar rerank at overfetch125 as the lower-risk, lower-rerank-cost fallback: `turboquant_ip_b8_overfetch125_fp16_rerank`, total compression `1.326425x`, evidence in `runs/eos-fp16-overfetch125-gate-20260614T000000Z/`.

Current sealed-verified local anchor:

```text
runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll
```

The June 10 deephard-full candidate is a legacy-named `manta-embed-v1` artifact for the `eos-embed-v1` lineage. It is sealed, inspected, and verified through the full scoreboard path. The sealed artifact SHA256 is `a7461b47784ea7434cf6048f33f6c281ef19887cfa9d0c699b6f2fba079f2b67`. The sealed scoreboard is under `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/`, and its sealed-vs-train-package comparison recorded zero nonzero quality or count deltas against `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-scoreboard/`.

| Dataset | Previous sealed anchor nDCG@10 | Current sealed anchor nDCG@10 | Delta | Previous sealed anchor recall@100 | Current sealed anchor recall@100 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.331139 | 0.482406 | +0.151267 | 0.724111 | 0.775778 | +0.051667 |
| NFCorpus | 0.084325 | 0.197733 | +0.113408 | 0.129067 | 0.235557 | +0.106490 |
| FiQA | 0.028967 | 0.117533 | +0.088566 | 0.164881 | 0.347197 | +0.182316 |
| Macro | 0.148144 | 0.265891 | +0.117747 | 0.339353 | 0.452844 | +0.113491 |

Pairwise eval-only rows completed with `optimizer_updates=0`, CUDA forward backend, eval AUC `0.825890`, and hard AUC `0.812725`.

Run a full retrieval-alignment round from an existing candidate when retrieval is behind the BM25 or open-model baselines:

```bash
EOS_REPO_ROOT=$PWD \
EOS_ALIGN_INITIAL_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_ALIGN_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/run_manta_embed_v1_retrieval_alignment_round.fw
```

The alignment harness writes a baseline scoreboard, run-local model-hard negatives, a retrained candidate package, a candidate scoreboard, and `retrieval-alignment-summary.tsv/json` with per-dataset nDCG@10 and recall@100 deltas.
Candidate training defaults to batch `64` and one explicit hard negative per query. This keeps full mined rounds bounded; larger batches or more explicit negatives should be treated as throughput experiments because pair work grows quickly.
Set `EOS_ALIGN_MODEL_HARD_DATASET_WEIGHTS=dataset=weight,...` to allocate more mined examples to weak datasets in the next mixed round.
Set `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=source=weight,...` to source-balance hard-negative batches when the train JSONL has `source` fields. Family keys such as `fiqa` also apply to mined sources such as `fiqa:model` unless an exact key is present.
Set `EOS_ALIGN_GATE_CANDIDATE=1` for promotion-style rounds. The gate records the summary, then fails when macro nDCG@10 is below `EOS_ALIGN_MIN_MACRO_NDCG_DELTA` or any dataset regresses beyond `EOS_ALIGN_MAX_DATASET_NDCG_REGRESSION`; `EOS_ALIGN_MIN_DATASET_NDCG_RATIO` can enforce an additional per-dataset nDCG ratio floor. Use `EOS_ALIGN_MAX_DATASET_RECALL_AT_100_REGRESSION` and `EOS_ALIGN_MIN_DATASET_RECALL_AT_100_RATIO` to also guard recall@100.
Set `EOS_ALIGN_CANDIDATE_CONTRASTIVE_LOSS=grouped_infonce` to test the query-grouped hard-negative objective. This counts only each query's own positive/negative candidate set in the training loss, which is useful as a retrieval-ranking ablation when corpus ranking regresses. The first grouped-only gated run rejected the candidate, so keep this flag as an experiment rather than a promotion path.
Set `EOS_ALIGN_CANDIDATE_CONTRASTIVE_LOSS=hybrid_infonce` to keep the global hard-negative InfoNCE matrix and add a weighted grouped term. The first weight-`0.25` gated run improved FiQA but regressed SciFact/NFCorpus. The first strict pass used `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.10`, `EOS_ALIGN_CANDIDATE_LR=0.000025`, and `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1`.
Model-hard and BM25 mining can now emit `teacher_scores` for each query's positive plus explicit negatives. Use `eos export-teacher-score-requests <hard-negatives.jsonl> <requests.jsonl>` to hand the same candidate groups to an external scorer, then import scored rows with `eos import-teacher-scores`. Set `EOS_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT` above zero to blend a soft teacher-distribution cross-entropy into hard-negative training, and tune `EOS_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` when the teacher ranking is too sharp or too flat.
A lower-LR ratchet from that strict-pass checkpoint with `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.05`, `EOS_ALIGN_CANDIDATE_LR=0.0000125`, and `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=3,fiqa=1` improved pairwise AUC but failed the retrieval gate. Do not treat repeated ratchets on the same mined blend as the default next step; remine model-hard negatives from the promoted artifact or add teacher-score distillation.
A fresh-mining round from the strict-pass checkpoint with `EOS_ALIGN_MODEL_HARD_DATASET_WEIGHTS=scifact=1,nfcorpus=3,fiqa=1`, `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.05`, `EOS_ALIGN_CANDIDATE_LR=0.0000125`, and `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1` passed the retrieval gate: macro nDCG@10 improved from `0.138397` to `0.145568`. Because recall@100 dipped, future promotion-style rounds should set an explicit recall floor when using this recipe.
A teacher-distilled follow-up from that checkpoint used fresh model-hard examples with `teacher_scores`, `EOS_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT=0.20`, `EOS_ALIGN_CANDIDATE_LR=0.000010`, and recall floors. The first full gated run with `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1` improved macro nDCG@10 to `0.146301` but failed the NFCorpus nDCG floor. Reusing the same teacher-scored JSONL with `scifact=1,nfcorpus=3,fiqa=1` fixed that failure and raised macro nDCG@10 to `0.147862` while recall@100 stayed flat or improved on all three retrieval sets. Holding that recipe and softening `EOS_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` to `1.5` raised macro nDCG@10 again to `0.148143`; the retrieval comparison gate passed with SciFact `0.331139`, NFCorpus `0.084325`, and FiQA `0.028967`. Temperature `1.25` and `2.0` also passed the gate but landed lower at macro `0.147645` and `0.148029`, respectively. LR `0.000008` at temperature `1.5` passed at macro `0.147625` and improved NFCorpus to `0.084927`, but lost enough SciFact to keep LR `0.000010` as the current best. Source weights `scifact=1,nfcorpus=4,fiqa=1` raised SciFact to `0.331362` but failed the NFCorpus nDCG floor and regressed FiQA. Source weights `scifact=2,nfcorpus=3,fiqa=1` passed the older fresh-baseline comparison at macro `0.146288`, but pairwise guardrails fell to validation AUC `0.815529` / hard AUC `0.808770` and macro stayed below the current best, so the local source-weight sweep is closed. Full BM25 teacher-score coverage plus `example_zscore` normalization is now a near-anchor branch: SciFact `0.329417`, NFCorpus `0.085742`, FiQA `0.029123`, macro `0.148094`, validation AUC `0.823282`, and hard AUC `0.815203`. It passed the stale-baseline gate by `+0.002526` macro but missed the current best by `0.000049`. The follow-up SciFact-recovery sampler (`scifact=2,nfcorpus=3,fiqa=1`) failed early with validation/hard AUC `0.817674` / `0.810527` and SciFact `0.326679`, so local normalization/source reshuffling is closed; move to external/synthetic teacher signal.
A deeper Lane B mining round from the temperature-`1.5` best requested `9000` model-hard examples with `EOS_ALIGN_MODEL_HARD_NEGATIVES=5`, `EOS_ALIGN_MODEL_HARD_CANDIDATE_TOP_K=400`, and `EOS_ALIGN_CANDIDATE_HARD_NEGATIVES=2`. The run trained `13,038` blended hard-negative examples at `4262` train pairs/s with CUDA-backed forward/optimizer/activation/contrastive, but failed the promotion gate: macro nDCG@10 fell from `0.148144` to `0.143866`, SciFact fell to `0.320576`, and FiQA fell to `0.026453`. NFCorpus rose slightly to `0.084568`, so the next isolation run should reuse the same mined JSONL with one candidate hard negative.
The one-hard-negative reuse trained the same deep-mined JSONL at `5056` train pairs/s and improved NFCorpus to a new high-water mark of `0.087300`, but it still failed against the current best: SciFact `0.324630`, FiQA `0.025679`, macro `0.145870`, validation AUC `0.802976`, and hard AUC `0.800840`. Since the deep-mined JSONL already contains `5400` NFCorpus model-hard examples, the next reuse should drop the NF3 training source schedule and try balanced source weights.
Balanced source weights on that same deep-mined HN1 JSONL regressed further to macro `0.144915`: SciFact `0.322932`, NFCorpus `0.086364`, FiQA `0.025450`, validation AUC `0.794627`, and hard AUC `0.794320`. Do not continue source-sampler rescue on this file; the next local isolation should keep the NF3 HN1 shape but lower LR to test a smaller update.
Lowering the HN1 NF3 reuse to LR `0.000005` improved FiQA to `0.027508` and kept NFCorpus high at `0.086784`, but SciFact remained low at `0.323136`, macro landed at `0.145809`, validation AUC was `0.803438`, and hard AUC was `0.800508`. The remaining local check is reduced grouped pressure; otherwise move this lane to external-teacher work.
Reducing grouped pressure to `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.025` at LR `0.000005` gave the best Lane B deep-mined balance but still failed promotion: SciFact `0.325439`, NFCorpus `0.086645`, FiQA `0.027204`, macro `0.146429`, validation AUC `0.804674`, and hard AUC `0.801615`. Close this deep-mined file for balanced promotion and move the next improvement path to external teacher import or a larger `embed-m` run.
The first `embed-m` capacity probes separated mechanical viability from quality. The full target shape (`32768` vocab, max sequence `512`, dim `192`, hidden `384`, `3` repeats) initialized, but full-corpus tokenizer training stayed CPU-bound for more than fifteen minutes before any optimizer step, so true `32768`-vocab iteration needs a cached tokenizer artifact or faster trainer first. A cached-tokenizer probe (`16384` vocab, max sequence `512`) trained and sealed at batch `64`, processing `1.393M` actual train pairs in `19m20s` with CUDA forward/optimizer/activation/contrastive and `1460.78` train pairs/s, but rejected hard: validation AUC `0.595854`, hard AUC `0.598887`, SciFact `0.160753`, NFCorpus `0.060778`, FiQA `0.012688`, macro `0.078073`. A scratch-style cached-tokenizer run with pure `infonce`, LR `0.002`, and one epoch rejected even earlier with validation AUC `0.495137` and hard AUC `0.498731`. Do not continue blind random-start `embed-m`; the next larger-model proof needs dimension-compatible bootstrapping, staged pretraining, or imported external teacher scores.

Compare a retrieval-only candidate scoreboard against a prior alignment summary without rerunning the full alignment harness:

```bash
EOS_COMPARE_BASELINE_SUMMARY_JSON=/path/to/retrieval-alignment-summary.json \
EOS_COMPARE_CANDIDATE_SCOREBOARD_JSON=/path/to/candidate-scoreboard/scoreboard.json \
ferrous-wheel run scripts/compare_manta_embed_v1_retrieval_candidate.fw
```

The comparison writes `retrieval-comparison-summary.tsv/json` beside the candidate scoreboard and applies the same macro nDCG@10, per-dataset nDCG, and recall@100 floors by default.

If you want a binary runner instead of `run` mode:

```bash
ferrous-wheel build scripts/bench.fw -o bin/manta-bench
bin/manta-bench
```

Capture CPU or heap profiles for any `eos` command with:

```bash
EOS_CPU_PROFILE=/tmp/manta.cpu.pprof go run ./cmd/eos train-embed ...
EOS_MEM_PROFILE=/tmp/manta.mem.pprof go run ./cmd/eos train-embed ...
```

Then inspect with `go tool pprof -top /tmp/manta.cpu.pprof`.

For repeatable GPU A/B profiles, use the Ferrous Wheel harness:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GPU_PROFILE_ASSETS=/path/to/assets/eos-embed-v1 \
EOS_GPU_PROFILE_TRAIN=/path/to/train-mini.jsonl \
EOS_GPU_PROFILE_EVAL=/path/to/eval-mini.jsonl \
EOS_GPU_PROFILE_VARIANTS=default,disable-batched-forward \
EOS_GPU_PROFILE_ENV_DISABLE_BATCHED_FORWARD=EOS_TRAIN_DISABLE_BATCHED_FORWARD=1 \
ferrous-wheel run scripts/profile_manta_gpu_efficiency.fw
```

The profile harness copies `.mll` package assets per variant, runs `train-embed`, writes each variant's `run.log`, `time.txt`, `cpu.pprof`, and `pprof-top.txt`, then writes a root `summary.tsv` with throughput and accelerator counters.
Set `EOS_GPU_PROFILE_NO_TOKENIZER=1` when profiling already-tokenized JSONL so the copied sibling tokenizer does not force text mode.

## Current Default Model Smoke

The current reference smoke uses:

- model package: `eos-embed-v1`
- encoder repeats: `2`
- tokenizer vocab: `2454`
- max sequence length: `256`
- train set: `4096` contrastive examples
- eval set: `512` contrastive examples
- batch size: `1024`
- loss: InfoNCE
- temperature: `0.05`
- learning rate: `0.005`
- eval cadence: every `4` steps

Latest local CUDA result:

```text
throughput: elapsed=5.139s examples/s=896.74 pairs/s=867250.60 train_examples/s=845.15 train_pairs/s=865437.87 eval_examples/s=1752.73 eval_pairs/s=897395.54 optimizer_steps/s=0.83
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=30 matmul_runs=13644 matmul_run_upload_mb=3727.56 matmul_run_download_mb=2000.00 optimizer_updates=28 activation_calls=0 contrastive_calls=4
```

This is the promoted default benchmark path. It includes CUDA matmul scratch-buffer reuse, grouped batched backward, exact-length grouped contrastive forward for variable-length text, pair-length-aware bucketing across shuffled windows, strided-batched cuBLAS for grouped attention matmuls, rank-3 transpose support for grouped attention backward, batch-1024 contrastive training, sequence matmul bindings disabled by default, Q/K/V forward projection coalescing through a multi-bound-right CUDA path, Q/K/V attention-gradient accumulation through one concatenated shared-left GEMM, combined V/K attention backward gradients in one doubled-batch GEMM, Q/K/V input-gradient accumulation into one resident-right output download with one CUDA sync per accumulated group, and guarded grouped activation-backward helpers kept behind the activation accelerator flag. The larger batch keeps the full in-batch negative set intact, improves contrastive signal, cuts optimizer/contrastive calls on this smoke, and reduces per-pair orchestration overhead. Length bucketing is on by default for CLI contrastive training; set `--length-bucket-batches=false` to disable it or `EOS_TRAIN_LENGTH_BUCKET_WINDOW` to tune the shuffled sort window. Disabling per-sequence matmul bindings trades a small upload increase for a large reduction in backend binding churn. Q/K/V coalescing preserves per-weight residency and quantization while uploading each shared left-hand activation once across query/key/value projections. Concatenated shared-left attention-gradient coalescing computes `input^T*[dQ|dK|dV]` as one standard GEMM, then splits the result back into the Q/K/V weight gradients. V/K gradient coalescing computes `scores^T*dMixed` and `dPreSoftmax^T*Q` in a single strided-batched dispatch because both use the same transpose shape. Input-gradient coalescing computes `dQ*Wq^T + dK*Wk^T + dV*Wv^T` against resident Q/K/V weights, synchronizes once after the accumulated cuBLAS calls, and downloads the accumulated result once.

Pairwise eval-only gates also use exact-length batched forward chunks by default. On the acquired hard eval set, the current default `EOS_TRAIN_PAIR_EVAL_BATCH_SIZE=256` measured `9.85s`, `6704` matmul runs, and `4261.18 MB` uploaded versus `16.16s`, `53504` matmul runs, and `5135.84 MB` uploaded with `EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1`. Eval metrics matched within float tolerance.

Read the throughput line with both lenses:

- `train_examples/s` measures actual encoder training throughput.
- `train_pairs/s` measures in-batch contrastive work and grows roughly with `batch^2`.
- `optimizer_steps/s` protects the benchmark from hiding that very large batches reduce update count on small datasets.

## Recent Perf Delta

The training hot path moved as follows on the same mini smoke:

| Path | Train pairs/s | Matmul runs |
| --- | ---: | ---: |
| Instrumented baseline | `21975.83` | `409600` |
| CUDA scratch reuse | `22933.16` | `409600` |
| Grouped batched backward default | `33328.94` | `237568` |
| Exact-length grouped forward default | `35856.58` | `140056` |
| Strided-batched grouped attention | `42594.29` | `107552` |
| Rank-3 transpose batched attention backward | `82262.50` | `50208` |
| Batch-512 benchmark smoke | `177192.16` | `28848` |
| Batch-1024 default benchmark smoke | `407407.98` | `16464` |
| Disable sequence matmul bindings by default | `707265.87` | `16464` |
| Return grouped bound-right outputs as views | `742477.76` | `16464` |
| Concatenate shared-left Q/K/V gradient matmul | `762372.72` | `15336` |
| Combine attention V/K gradient matmuls | `803746.93` | `14772` |
| Accumulate Q/K/V input gradients with resident weights | `825766.14` | `13644` |
| Single-sync accumulated Q/K/V input gradients | `865437.87` | `13644` |

The main wins came from grouping real text batches by sequence length during backward, coalescing parameter-gradient matmuls into taller `X^T*dY` operations, grouping contrastive forward sequences by exact token length inside each original batch, promoting rank-3 x rank-3 CUDA matmul to `cublasSgemmStridedBatched`, allowing strided-batched matmul to handle transpose flags directly, and increasing the effective contrastive batch. The forward grouping keeps the full in-batch negative set intact and avoids padding, so attention math does not change.

The Q/K/V multi-bound-right path is transfer progress: it reduces matmul run uploads from `4173.15 MB` to `3727.56 MB` while preserving each weight's resident quantized form. The concatenated shared-left gradient path and combined V/K gradient path are dispatch-count wins on top of that. Accumulated input gradients keep the Q/K/V weights resident and reduce matmul run downloads from `2208.41 MB` to `2000.00 MB`. Single-sync accumulation removes the redundant per-term CUDA syncs inside that primitive while keeping the same transfer profile. Batch-1024 A/B measured `845.15 train_examples/s`, `865437.87 train_pairs/s`, and `13644` matmul runs by default versus `748.52 train_examples/s`, `766485.96 train_pairs/s`, and `13644` matmul runs with `EOS_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1`.

Batched forward materialization now keeps downloaded backend outputs as per-sequence views and passes layer outputs forward by view instead of copying them through temporary buffers. On a real acquired-text 512 train / 128 eval A/B at batch 256, trainer elapsed moved from `9.557s` to `8.235s` and train examples/s moved from `57.91` to `70.67`; matmul upload/download counters stayed at `5838.20 MB` / `2990.88 MB`, as expected, because this cuts host copies rather than device transfer.

Attention backprop now also keeps batched Q/K/V gradient outputs as views instead of allocating zeroed per-sequence buffers and copying accelerator outputs into them. On the acquired-text 512 train / 128 eval CPU profile, `backpropAttentionSequences` moved from `4.69s` cumulative to `2.21s`, and `runtime.memclrNoHeapPointers` moved from `3.27s` to `1.26s`. Matmul counters stay unchanged; this is a host allocation and copy reduction inside the existing CUDA dispatch pattern.

The production candidate workflow keeps `EOS_EVAL_EVERY_STEPS=0` by default. Step-level eval is still available for convergence debugging, but it inserts full eval passes into `train.log`; on the acquired full split, `--eval-every-steps 4` adds `21` extra contrastive eval passes across 3 epochs at batch 1024. Keep release gates in the final validation and hard holdout evals so training transfer reflects optimizer work first.

Ranked BPE tokenization removed a startup/data-ingest bottleneck before longer training runs. A direct batch-2048 `train-embed` CPU profile moved tokenizer encode time from `2.13s` cumulative (`BPETokenizer.Encode` -> `bpeMerge` -> `applyMerge`) to `0.25s` cumulative (`bpeMergeRanked`). End-to-end throughput remains dominated by training transfer/orchestration after tokenization, but corpus ingestion no longer burns a large fraction of host CPU.

Ranked BPE now compacts each selected merge in place instead of allocating a fresh token slice for every merge pass. On the acquired-text 512 train / 128 eval CPU profile, `applyRankedMerge` moved from `0.86s` cumulative to `0.45s`; total tokenizer encode time moved from `3.39s` to `3.10s`. The full run remains dominated by backend orchestration, memory clearing, and host-device traffic, so this is a data-ingest allocation cut rather than a matmul-counter change.

Prepared-text ingestion now keeps a trainer-local tokenization cache across train and eval loading. On the same acquired-text 512 train / 128 eval profile, with `1.38x` repeated text fields, `BPETokenizer.Encode` moved from `3.57s` cumulative to `1.98s`, train text tokenization moved from `2.82s` to `1.60s`, pair eval text tokenization moved from `0.75s` to `0.38s`, and trainer elapsed moved from `8.414s` to `7.577s`. Matmul counters stayed unchanged at `2788` runs, `5838.20 MB` uploaded, and `2990.88 MB` downloaded; this is an in-memory ingest optimization, and candidate packages, tokenizers, train profiles, checkpoints, and sealed outputs remain `.mll` artifacts.

Prepared JSONL production runs can now pretokenize once with `eos tokenize-embed` and train with `eos train-embed --no-tokenizer`. On the 512 train / 128 eval mini smoke, a same-code text JSONL profile still spent `3.61s` cumulative in `BPETokenizer.Encode` (`22.59%` of CPU samples), while the token JSONL profile removed tokenizer encode from the hot path. The GPU counters were identical for the two runs at `2712` matmul runs, `5838.20 MB` uploaded, and `2988.88 MB` downloaded; text measured `24948.46` train pairs/s and token JSONL measured `22414.32` train pairs/s in one noisy pair of local runs. Treat this as a training-profile cleanup and reproducibility win, not a claimed device-throughput gain.

Unbound activation acceleration now has a default shape ceiling so `EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1` does not route long-document activation groups through standalone upload/download kernels. On the tokenized acquired 4096 train / 512 eval split, fully unbounded CUDA activation regressed to `1m42.293s` and `42420.11` train pairs/s with `744` activation calls. Host activation measured `1m0.212s` and `73554.96` train pairs/s. Shape-limited opt-in measured `1m0.321s` and `73329.90` train pairs/s with `694` activation calls and identical train/eval metrics. On the smaller 512 / 128 tokenized smoke, the same shape-limited opt-in path measured `25239.78` train pairs/s versus `23997.21` for the host-activation default in the A/B run. Keep activation acceleration as an experiment until activation residency removes the extra transfers.

Fast GELU is available as an opt-in training math approximation for candidate-throughput experiments. `EOS_TRAIN_ENABLE_FAST_GELU=1` replaces the precise tanh call in host GELU forward/backward with a bounded rational tanh approximation and keeps GELU backward on the host so forward/backward use matching math. On the tokenized 512 train / 128 eval smoke, precise GELU measured `24191.69` train pairs/s and fast GELU measured `32525.01` train pairs/s; eval AUC moved from `0.580322` to `0.581543`. On the tokenized 4096 train / 512 eval split, fast GELU measured `86779.56` train pairs/s versus the prior clean precise-GELU baseline of `73554.96`; eval AUC moved from `0.548126` to `0.547104`. Treat this as a speed/quality knob for candidate runs and keep the exact GELU default until larger validation clears the tradeoff.

Batched matmul upload staging now detects already-contiguous split views and uploads the backing span directly instead of copying those views into scratch first. On the same acquired-text 512 train / 128 eval profile, trainer elapsed moved from `7.577s` to `6.396s`, `runtime.memmove` moved from `1.07s` to `0.58s`, `runtime.memclrNoHeapPointers` moved from `0.78s` to `0.60s`, and `flattenFixedFloat32MatricesScratch` was down to `0.34s` cumulative. Matmul counters again stayed unchanged at `2788` runs, `5838.20 MB` uploaded, and `2990.88 MB` downloaded; the win is less host staging around the same CUDA work.

Trainer-owned scratch buffers now reuse transient float32 flattening inputs for batched matmul dispatches. On the same direct batch-2048 CPU profile, `flattenFixedFloat32Matrices` moved from roughly `0.72s` cumulative to `0.09s` cumulative. The remaining host-side allocation/copy profile is mostly broader activation/state materialization and unavoidable host-device transfer until full device residency lands.

## Batch Sweep

Batch size is now the largest exposed training knob. On the same 4096-example mini smoke:

| Batch | Run steps | Train examples/s | Train pairs/s | Matmul runs | Max RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `512` | `8` | `558.19` | `285791.12` | `28848` | `1.02 GB` |
| `1024` | `4` | `845.15` | `865437.87` | `13644` | `1.52 GB` |
| `2048` | `2` | `827.20` | `1694107.56` | `9408` | `2.50 GB` |
| `4096` | `1` | `741.49` | `3037152.08` | `5616` | `4.47 GB` |

Batch 2048 is a useful ceiling or large-corpus setting, but it halves optimizer steps on this mini smoke. Batch 4096 is one update and regresses example throughput despite very high pair throughput. Batch 1024 is the benchmark default because it keeps multiple updates in the smoke and captures most of the real throughput win.

## How Much Faster Can It Get?

The current profile still shows the next bottleneck is backend transfer/orchestration, not raw math:

```text
MatMulRuns ~= 13644 per 4096-example mini smoke at batch 1024
RunUploadedBytes ~= 3.73 GiB
RunDownloadedBytes ~= 2.00 GiB
```

Reasonable next targets:

- `850-1000 train_examples/s`: reduce host materialization and launch overhead inside same-length groups.
- `1000+ train_examples/s`: keep forward/backward intermediates device-resident across full layer groups and move optimizer/activation state through the same device-resident path.
- `1200+ train_examples/s`: persistent device-resident training steps with fused attention/FFN backward and fewer per-layer dispatches.

The practical ceiling for this tiny model is dominated by orchestration and host-device transfer, not raw GEMM throughput. Larger models will shift more time into actual math, but the same device-residency work is still required to get high GPU utilization.

## Relevant Flags

```bash
EOS_TRAIN_DISABLE_BATCHED_BACKWARD=1
```

Disables the promoted grouped batched-backward path and returns to per-sequence backward.

```bash
EOS_TRAIN_DISABLE_BATCHED_FORWARD=1
```

Disables the promoted batched forward path and returns to per-sequence forward encoding. Batched forward is enabled by default because the larger default-model run underfeeds the GPU unless forward work is coalesced aggressively.

```bash
EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1
```

Disables exact-length batched forward chunks for pairwise `train-embed --eval-only` runs. This is useful for A/B checks against the scalar pair encoder.

```bash
EOS_TRAIN_PAIR_EVAL_BATCH_SIZE=256
```

Controls how many pair examples each pairwise eval chunk may contain before grouping by exact token length. The default is `256`; larger chunks reduce dispatch count further but can increase materialization pressure and wall time.

```bash
EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS=1
```

Re-enables per-sequence matmul bindings. These are disabled by default because batch-1024 grouped training spends more time binding and unbinding short-lived sequence tensors than it saves in uploads. Keep this for small-batch experiments and regression checks.

```bash
EOS_TRAIN_DISABLE_QKV_MULTI_BOUND=1
```

Disables Q/K/V forward projection coalescing. By default CUDA uses one uploaded left-hand activation with three resident right-hand Q/K/V matrices for same-length forward groups. This cuts transfer bytes while preserving each right-hand weight's own quantization state.

```bash
EOS_TRAIN_DISABLE_SHARED_LEFT_MATMUL=1
```

Disables shared-left matmul coalescing. By default Eos first tries the concatenated shared-left gradient path for attention backward Q/K/V weight-gradient matmuls, then falls back to the backend shared-left interface when concatenation is disabled or unavailable.

```bash
EOS_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL=1
```

Disables only the concatenated shared-left path. This keeps the older backend shared-left fallback enabled for A/B testing.

```bash
EOS_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD=1
```

Disables only the combined V/K attention backward gradient path. This returns to separate strided-batched GEMMs for `scores^T*dMixed` and `dPreSoftmax^T*Q`.

```bash
EOS_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD=1
```

Disables only the accumulated Q/K/V attention input-gradient path. This returns to three resident right-hand matmuls for `dQ*Wq^T`, `dK*Wk^T`, and `dV*Wv^T`, followed by host accumulation.

```bash
EOS_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1
```

Disables CUDA single-sync accumulation inside the backend accumulated resident-right matmul primitive. This keeps the same logical training path but synchronizes after each accumulated cuBLAS term, which is useful for A/B testing backend launch overhead.

```bash
EOS_TRAIN_ENABLE_FAST_GELU=1
```

Uses a bounded rational tanh approximation for host GELU forward/backward. This is opt-in because it changes training math. It can materially improve CPU-bound trainer throughput while the model still uses host activation math, but production candidates should compare validation and hard-holdout metrics against exact GELU before promotion.

```bash
EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1
```

Enables CUDA/Metal activation backward acceleration. Activation backward batches GELU, softmax, and layernorm across same-length groups before dispatching. It is still opt-in because standalone activation kernels are upload/download-bound without broader activation residency. Large unbound groups fall back to host math by default through `EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS`.

```bash
EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS=1048576
```

Caps unbound activation accelerator calls by `rows * cols`. The default is `1048576`. Set it lower to force more host fallback for memory-transfer-heavy profiles; set it to `0` to remove the shape limit when profiling a fully unbounded activation path.

```bash
EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Enables only the attention softmax backward activation path while keeping GELU and layernorm backward on the host. This is a profiling seam, not a promoted default.

```bash
EOS_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Disables the softmax backward activation path even when the broader activation accelerator is enabled. Use it to isolate GELU/layernorm activation experiments from attention softmax experiments.
