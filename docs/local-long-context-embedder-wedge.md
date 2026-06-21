# Local Long-Context Embedder Wedge

This is the product wedge for Eos's sparse attention work:

```text
Build the best local long-context embedder that can be trained, adapted, exported, and served on consumer GPUs.
```

This is more credible than claiming general frontier-model performance from a desktop card. Embedding models are smaller, retrieval quality can be improved through distillation and hard negatives, long-context embedding is still underdeveloped, and Eos's TurboQuant plus routed sparse attention stack directly attacks the memory and attention-cost bottlenecks that make long-document embedding expensive.

## Target Claim

The target claim is:

```text
Eos can train and serve a best-in-class local long-context embedding model on a single consumer GPU, with sealed .mll export, compressed deployment, and subquadratic long-context attention.
```

This means best-in-class in the local/efficient/long-context class first, then public leaderboard competitiveness second.

## Non-Goals

- Do not claim global MTEB #1 until benchmark evidence exists.
- Do not claim frontier LLM training from scratch on one desktop GPU.
- Do not rely on a hosted API in the measured training or inference path.
- Do not hide performance behind reranking unless the scorecard has a separate rerank column.

## Scorecard

The model should be evaluated on four axes.

| Axis | Primary metric | Initial pass | Strong pass |
| --- | --- | ---: | ---: |
| Local RAG retrieval | project/domain nDCG@10 | beats current CorkScrew baseline | beats strong open embedder baseline |
| BEIR/MTEB retrieval | nDCG@10 / retrieval score | competitive with BGE-M3 class | competitive with current 0.6B-1B leaders |
| Long-context retrieval | LongEmbed-style retrieval score | beats chunked short-context baseline | best local long-context model in class |
| Efficiency | examples/s, VRAM, index size | fits C24 training and serving | fits C16 serving with compressed artifacts |

The public story should lead with the axis we actually win. If the first win is LongEmbed or domain RAG, say that. Overall MTEB can come later.

## Baselines

Use a frozen baseline set for each release candidate.

Open/local baselines:

- `BAAI/bge-m3`
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-Embedding-4B` where hardware allows
- `Qwen/Qwen3-Embedding-8B` as an aspirational high-quality reference
- current `eos-embed-v1` package

API baselines for context, not release dependency:

- current OpenAI embedding model
- current Voyage large embedding model
- current Cohere embedding model

Long-context baselines:

- chunked retrieval with a strong short-context embedder
- late-chunking baseline
- long-context open embedding model if available
- high-budget teacher using chunk retrieval plus reranker

Every baseline run must record model id, revision/version, dimensions, max tokens, precision, chunking policy, pooling policy, and index settings.

## Model Path

### Stage A: Eos Embed V1 Reliable Baseline

Goal: make the existing native embedding pipeline a trustworthy baseline.

Work:

- keep BEIR acquisition and production gates current
- add long-document train/eval splits
- preserve sealed `.mll` export and tokenizer packaging
- record reproducible metrics for every candidate

Pass:

- current production embedding gates pass
- candidate beats the prior Eos baseline on hard eval
- no hidden CPU fallback in the measured candidate run

### Stage B: Distilled Compact Embedder

Goal: close the quality gap with strong public embedders before adding long context.

Work:

- train against positive/negative pairs plus teacher similarities
- mine model-hard negatives
- add multi-dataset mixture weights
- add embedding-dimension and Matryoshka-style truncation experiments
- evaluate dense vector and optional hybrid sparse vector heads separately

Pass:

- beats `eos-embed-v1` by a meaningful margin on hard eval
- is competitive with BGE-M3-class retrieval on selected BEIR/MTEB retrieval tasks
- exports as sealed `.mll`

### Stage C: Long-Context Extension

Goal: embed 32k-128k documents without reducing them to naive chunks.

Work:

- long-context position strategy
- chunk alignment training
- global document embedding plus local span embeddings
- contrastive positives where relevant evidence is far from document start
- train/eval with LongEmbed-style tasks

Pass:

- beats chunked short-context baseline on long-document retrieval
- short-context retrieval does not regress beyond the quality gate
- serving fits C24 with compressed weights and activations

### Stage D: Sparse Long-Context Encoder

Goal: use routed sparse attention where it matters, not just in decode demos.

Work:

- integrate routed sparse attention into the embedding encoder
- add block summaries for routing
- train router with high-budget teacher labels
- enable sparse backward or detached-router training path
- profile dense vs exact sparse vs routed sparse encoder passes

Pass:

- routed encoder clears subquadratic scaling gate from the consumer GPU spec
- quality stays within teacher-comparison gates
- training smoke reaches `100+` optimizer steps on C24 without dense attention materialization

### Stage E: Production Local Embedder

Goal: make the model useful as a product artifact.

Work:

- sealed `.mll` export with tokenizer, weights, memory plan, and metric metadata
- command-line encode path for document/query embeddings
- CorkScrewDB load path
- batch indexing profile
- deterministic package inspection and verification

Pass:

- candidate is better than current Eos baseline on local RAG gates
- long-context candidate beats chunked baseline on long-context gates
- package verifies and serves through normal runtime

## Architecture Direction

The embedder should support three output modes:

| Mode | Purpose | Notes |
| --- | --- | --- |
| single vector | standard ANN retrieval | required for compatibility |
| multi-vector spans | long-document and late-interaction retrieval | optional first, likely important |
| sparse lexical head | hybrid retrieval | optional, useful for exact terms and code |

The internal encoder should move toward:

- TurboQuant-ready weights and activations
- routed sparse attention for long sequences
- block summaries for route decisions
- local window plus selected global blocks
- pooled global vector plus optional span vectors
- training-time teacher labels for router and embedding output

## Data Strategy

Use a mixture, not a single benchmark.

Minimum data groups:

- BEIR-style public retrieval sets with clear licenses
- project/domain RAG data from CorkScrew use cases
- long-document synthetic retrieval: passkey, needle, dispersed evidence
- long-document real retrieval: legal, docs, transcripts, code/docs
- teacher-labeled hard negatives
- query/document instruction variants

Data artifacts must be versioned with SHA256 manifests. Dataset leakage into eval must be treated as a release blocker.

## Training Strategy

The practical path is teacher-heavy, not pure self-supervised pretraining.

Training losses:

- in-batch contrastive loss
- hard-negative contrastive loss
- grouped hard-negative InfoNCE over each query's explicit candidate set
- hybrid hard-negative InfoNCE that keeps global batch candidates and adds a weighted grouped term
- teacher-score distillation over mined candidate groups
- optional pairwise margin loss
- optional Matryoshka/truncation loss
- router block-recall auxiliary loss
- optional sparse lexical head loss

Curriculum:

1. short-context retrieval quality
2. hard negatives
3. long-document chunk alignment
4. long-context sparse encoder
5. compressed deployment fine-tune

## Evaluation Plan

Every candidate needs a machine-readable scoreboard plus the underlying score files:

```text
scoreboard.tsv
scoreboard.json
short-retrieval.metrics.json
long-retrieval.metrics.json
efficiency.metrics.json
```

Build the first scoreboard with:

```bash
EOS_REPO_ROOT=$PWD \
EOS_SCOREBOARD_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_SCOREBOARD_PAIRWISE_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_SCOREBOARD_HARD_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_SCOREBOARD_RETRIEVAL_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The scoreboard harness writes `runs/<run-id>/scoreboard.tsv`, `runs/<run-id>/scoreboard.json`, per-task metrics JSON, command logs, and the exact `eos` binary used for the run. Enable `EOS_SCOREBOARD_BM25_BASELINE=1` to include the in-repo BM25 baseline beside Eos retrieval scores.
Before a real LongEmbed-style dataset is wired into the scoreboard, `scripts/smoke_eos_long_context_chunked_baseline.fw` provides a reproducible synthetic late-needle smoke for the long-document comparison shape; the baseline run is summarized in `.tiller/scratch/codex/eos-long-context-chunked-baseline-smoke-v1-report.md`, and BM25 also reaches a perfect score on that generated dataset.
The `semantic_late_needle` scenario keeps the same synthetic caveat but is now the preferred compact-vector comparison smoke because BM25 no longer saturates nDCG@10; see `.tiller/scratch/codex/eos-semantic-long-context-smoke-v1-report.md`.
External Qwen3/mxbai native and compact 128d late-needle cache comparisons are summarized in `.tiller/scratch/codex/eos-late-needle-external-baseline-v1-report.md`.
`scripts/build_repo_docs_longembed_dataset.fw` adds a small non-synthetic local-doc lane at `datasets/longembed/repo-docs`. It is useful for exercising long-document chunking over real repository text, but its qrels are deterministic path/heading heuristics, so results should be treated as repo-specific harness evidence rather than LongEmbed proof; see `.tiller/scratch/codex/eos-repo-docs-longembed-lane-v1-report.md`.
For short-set parent/single-vector compact-cache evidence, `scripts/eval_eos_prefix_dim_curve.fw` now records the current default artifact's prefix-width curve separately from long-document child-vector evidence. Its rows use `export-retrieval-vectors --output-dim` prefix truncation plus L2 renormalization, not trained Matryoshka/native 128d heads. The full-query `128d` q4/q8 run at `runs/eos-prefix-128-q4q8-full-query-current-default-v1-20260620T021328Z/` keeps `128d q8` as the near-dense compact cache reference at about `3.878788x` vector-payload compression, while `128d q4` is the aggressive compact candidate at about `7.529412x`. q2 is not default-quality-ready without a trained compact head. Keep this lane distinct from repo-docs child-vector and token-span evidence, which exercises multi-vector rollups and long-document retrieval shapes.
`eos smoke-sparse-embedding-encoder` now writes a scorecard bridge beside its manifest: `scorecard.json` and `scorecard.tsv`. Run it with a bounded local smoke such as `go run ./cmd/eos smoke-sparse-embedding-encoder --run-dir runs/eos-sparse-embedding-encoder-smoke-local --backend host --seq-len 64 --query-len 2 --dim 8 --top-k 2 --route-top-blocks 2 --preflight-key-lens 4096,32768`. The row records `evidence_level=smoke_synthetic_kernel_evidence`, `quality_claim=false`, runtime/backend metadata, 32k preflight status, the 32768-key score fraction, TurboQuant bits/seed, and parity status. Treat this as sparse-enabled routed TurboQuant encoder smoke visibility only; it does not score retrieval quality or prove the long-context wedge.
`eos export-sparse-token-pool-vectors` is the first retrieval-cache bridge beyond that synthetic smoke. It loads an existing trainable embedding package, tokenizer, and sibling weights; tokenizes BEIR corpus/query text; gathers actual token_embedding rows; optionally applies manifest Q/K/V/O/projection weights when shapes match; routes host-reference TurboQuant sparse attention over token sequences; and writes `doc-vectors.jsonl` or `child-doc-vectors.jsonl` plus `query-vectors.jsonl`. `--document-chunk-words` keeps the existing word-chunk child-vector path. `--token-span-tokens N` is a separate span-aware child-vector diagnostic mode: each parent document is tokenized and encoded once, then `child-doc-vectors.jsonl` rows are emitted by mean pooling final normalized token rows over spans of `N` tokenizer-output tokens, with `--token-span-overlap` and `--token-span-min-tokens` controlling overlap and trailing-span retention. Token-span mode is mutually exclusive with word chunking and requires full manifest encoder weights; it fails instead of using the legacy token-embedding fallback. `--resume` and `--progress-every N` make long token-span exports restartable by writing progress sidecars for child document and query vectors, then truncating/appending from the last trusted completed record on resume. The manifest uses `method=experimental_sparse_token_pool`, `quality_claim=false`, and records skipped weights, sparse routing knobs, bits/seed, token-span settings when enabled, tokenizer-output token length stats for documents and queries, resume/progress metadata when enabled, and the host-reference dense-K/V decode caveat. Use `--min-observed-doc-tokens` to make a long-context smoke fail when the maximum document tokenizer-output length actually consumed by the encoder stays below the requested threshold. For diagnostic control only, `eos rename-embed --max-seq N <input.mll> <output.mll>` can rewrite a copied training package with a different tokenizer max-sequence contract, optionally with `--name`; this is useful for proving the sparse-token-pool path can consume longer token sequences when the package contract allows it, but it is not a trained LongEmbed artifact, not a quality promotion, and must not replace a production default artifact. Score these caches with `eos eval-retrieval-vectors`, `eos eval-retrieval-vectors-turboquant`, or `eos eval-retrieval-multivector-turboquant` before comparison. Treat every row from this command as prototype retrieval-evaluation evidence only, not production dense embedder output, not a trained sparse encoder, and not LongEmbed proof.
`eos export-sparse-encoder-vectors` is the first distinct sparse encoder retrieval-cache entrypoint. It reuses the same host-reference sparse-token internals but defaults to parent `doc-vectors.jsonl` plus `query-vectors.jsonl`, requires full manifest encoder weights, and fails clearly instead of falling back to token-embedding-only pooling. Its manifest records `method=experimental_sparse_encoder_host_reference`, `evidence_level=retrieval_cache_host_reference_sparse_encoder`, `quality_claim=false`, `require_full_encoder=true`, `full_encoder_applied=true`, sparse top-k/routing/TurboQuant bits and seed, document/query tokenizer-output stats, `dense_kv_materialized=true`, `kv_decode=host_reference_decode`, and a max-observed-document sparse plan with candidate-key budget and score-work fraction. This is host-reference retrieval-cache evidence only: it is not a trained sparse or LongEmbed encoder, not sealed runtime inference, and not production quality evidence. Score the parent caches with `eos eval-retrieval-vectors`; use `eos eval-retrieval-vectors-turboquant --quantizer-seed` when producing deterministic q-bit cache rows.
External embedder baselines can be added without provider API calls by exporting BEIR-aligned vector caches and setting `EOS_SCOREBOARD_EXTERNAL_VECTOR_ROOT`, `EOS_SCOREBOARD_EXTERNAL_VECTOR_DATASETS`, `EOS_SCOREBOARD_EXTERNAL_VECTOR_BASELINE`, and `EOS_SCOREBOARD_EXTERNAL_VECTOR_BACKEND`. The harness expects `<vector-root>/<dataset>/doc-vectors.jsonl` and `query-vectors.jsonl`; each row needs `id` or `_id` plus `vector`, `embedding`, or `values`. These rows use `eos eval-retrieval-vectors` and emit the same retrieval metrics JSON/scoreboard columns as sealed Eos and BM25 rows.
External child-vector caches can be added to the long-retrieval scoreboard with `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_ROOT`, `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_DATASETS`, `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BASELINE`, `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BACKEND`, `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_ARTIFACT`, `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BITS`, and `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BASELINE_DIM`. This path expects `<vector-root>/<dataset>/child-doc-vectors.jsonl` plus `query-vectors.jsonl`, calls `eos eval-retrieval-multivector-turboquant`, and emits `<baseline>-dense-child` and `<baseline>-turboquant-child` rows. The consolidated repo-docs child-vector evidence is below; all q-bit rows use quantizer seed `5581486560434873699`.

| Row | Method | nDCG@10 | recall@100 |
| --- | --- | ---: | ---: |
| BM25 | bm25 | 0.717080 | 1.000000 |
| Eos default | cuda dense | 0.644872 | 1.000000 |
| Eos 128d child | dense child-max | 0.597491 | 1.000000 |
| Eos 128d child | q2 child-max | 0.580548 | 1.000000 |
| Eos 128d child | q4 child-max | 0.603952 | 1.000000 |
| Eos 128d child | q8 child-max | 0.600218 | 1.000000 |
| Qwen3 0.6B 128d child | dense child-max | 0.771170 | 1.000000 |
| Qwen3 0.6B 128d child | q2 child-max | 0.694290 | 1.000000 |
| Qwen3 0.6B 128d child | q4 child-max | 0.739269 | 1.000000 |
| Qwen3 0.6B 128d child | q8 child-max | 0.769390 | 1.000000 |
| mxbai-large 128d child | dense child-max | 0.713075 | 1.000000 |
| mxbai-large 128d child | q2 child-max | 0.627584 | 1.000000 |
| mxbai-large 128d child | q4 child-max | 0.691586 | 1.000000 |
| mxbai-large 128d child | q8 child-max | 0.708765 | 1.000000 |

For repo-docs Qwen3/mxbai/Eos 128d child caches, this makes the direct dense/q-bit child-max evidence land in `scoreboard.json`; it is still repo-specific harness evidence, not LongEmbed proof. The Eos 128d child cache uses prefix truncation plus L2 renormalization from the default 256d artifact, not a trained Matryoshka head.

Repo-docs token-span sweeps now add diagnostic Eos sparse-token-pool child-vector evidence. These used maxseq1024 retargeted packages to prove the path can consume longer token sequences, but the retarget is diagnostic only and not a trained long-context artifact. The important rows are:

| Row | Best span/overlap | Dense nDCG@10 | q4 nDCG@10 | q8 nDCG@10 | Child vectors | q4 storage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Eos 256d token-span | 64/16 | 0.647864 | 0.650561 | 0.652997 | 211 | 2.472656x vs 256d parent |
| Eos 128d compact token-span | 48/12 | 0.660476 | 0.659998 | 0.655515 | 290 | 0.438x vs 1024d parent; 1.751x vs 256d parent |

Both best rows beat Eos default repo-docs dense `0.644872` on this local lane, while still trailing Qwen3 0.6B 128d child q4 `0.739269` and mxbai-large 128d child q4 `0.691586`. Reports and ignored run roots: `runs/eos-lc-token-span-sweep-v1-20260619T004023Z`, `runs/eos-lc-compact-token-span-sweep-v1-20260619T005356Z`, `.tiller/scratch/codex/eos-lc-token-span-sweep-v1-report.md`, and `.tiller/scratch/codex/eos-lc-compact-token-span-sweep-v1-report.md`. Preserve `quality_claim=false`: repo-docs is tiny deterministic path/heading qrels evidence, not LongEmbed proof.

The same `eval-retrieval-multivector-turboquant` path exposes diagnostic parent rollup knobs: `--aggregation max|top2-mean|top3-mean|top5-mean` and `--child-count-penalty FLOAT`. The default remains max-child with no penalty. On the compact 128d `48/12` token-span fixture, anchor q4 nDCG@10 was `0.659998`; max-child with penalty `0.01` reached q4 `0.665813`, closing only a small part of the external gap, while top-N mean hurt this fixture. Keep this as eval/profile evidence only, not a model-quality claim or default-promotion signal.

Official LongEmbed capped token-span evidence is now available for four real-task slices using span `256`, overlap `64`, q4 token-span caches, external baselines disabled, and `quality_claim=false` throughout. The resumable sparse token-vector export capability that made the QMSum, NarrativeQA, and 2WikiMQA runs practical was checkpointed as `a5e18786` (`add: Add resumable sparse token vector export with progress sidecars`); it adds `--resume`, `--progress-every`, and progress sidecars for child document and query vector JSONL outputs. The SummScreenFD row is a complete audited legacy/pre-resume artifact, so it has no resume/progress sidecars even though its artifact set is complete.

The promoted s40 current-default package has a fresh capped official comparison against cached Qwen3 0.6B and mxbai-large child-vector baselines. These rows are still diagnostic: `quality_claim=false`, capped `real-doc20` slices, and external rows use many more child vectors and higher storage than Eos token-span rows.

| Dataset | Best Eos row | Eos nDCG@10 | Qwen3 q4 nDCG@10 | mxbai q4 nDCG@10 | Caveat |
| --- | --- | ---: | ---: | ---: | --- |
| `qmsum` | token-span q4 | 0.560693 | 0.848162 | 0.828480 | Eos uses 416 token-span children; external cached rows use 2,577 children. |
| `2wikimqa` | fusion RRF | 0.745700 | 1.000000 | 0.981546 | Eos uses 426 fusion children; external cached rows use 1,771 children. |

| Dataset | Slice | Direct nDCG@10 | q4 token-span nDCG@10 | Best fusion nDCG@10 | Conservative fusion nDCG@10 | q4 child count | q4 storage multiple | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qmsum` | doc40/query20 | 0.396104612 | 0.359794187 | 0.407512308 | 0.396826356 | 200 | 0.083007812 | q4 token-span alone regressed; conservative fusion gives a tiny no-regression lift; less-conservative fusion lifts aggregate but has per-query regressions. |
| `narrativeqa` | doc20/query20 | 0.428430714 | 0.363971643 | 0.428430714 | 0.428430714 | 100 | 0.083007812 | q4 token-span alone regressed; only the conservative fusion recipe ties direct with no per-query regressions. |
| `2wikimqa` | doc20/query20 | 0.648352471 | 0.550567431 | 0.648352471 | 0.648352471 | 100 | 0.083007812 | q4 token-span alone regressed; only the conservative fusion recipe ties direct with no per-query regressions. |
| `summ_screen_fd` | doc20/query20 | 0.843996310 | 0.760026850 | 0.843996310 | 0.843996310 | 417 | 0.346142578125 | q4 token-span alone regressed; conservative fusion ties direct; less-conservative dense-protect k10 regressed. |

Interpret q4 token-span alone as compact storage and evaluation-path evidence, not default quality evidence: it regressed aggregate nDCG@10 versus direct single-vector retrieval on all four official span256/64 capped slices. The current conservative diagnostic recipe is `direct_token_span_fusion_dense_protect_n1_rrf_k60_lambda010`: QMSum gets a slight aggregate lift, NarrativeQA ties direct exactly, 2WikiMQA ties direct exactly, SummScreenFD ties direct exactly, and none of the post-resume capped runs had per-query nDCG@10 regressions under that recipe. Less-conservative fusion can lift some datasets, especially QMSum, but regresses others, so keep fusion diagnostic until cross-dataset stability improves. Do not promote from these capped LongEmbed rows. The local replay lane around the exact s40 LongEmbed teacher batch and anchor mix is closed: further replay/source-weight/FIQA padding did not clear the strict short-set gate, so the next Eos improvement should change signal family or objective.

Router evidence keeps the product default at direct retrieval. Five-way router-signal conservative abstention over `needle`, `passkey`, `2wikimqa`, `qmsum`, and `narrativeqa` produced only a tiny macro lift (`+0.002460468` vs direct) with `0` held-out regressions and `1` switch across `170` rows; the companion learned router had larger average lift (`+0.014338863`) but `17` held-out regressions and `116` non-direct switches, so it is not source-integratable. On the current official span256/64 layer, static conservative fusion gives `+0.000180436` macro delta vs direct with `0` regressions across `80` rows, while action-only LODO gives `0` lift/`0` regressions and feature-threshold LODO gives `-0.003199038` macro delta with `1` held-out regression. Preserve `quality_claim=false`, treat conservative fusion as capped diagnostic/no-regression evidence only, block learned or feature-router source promotion for now, and redirect router improvement to more official slices or a changed signal family/objective. Reports: `.tiller/scratch/codex/eos-router-fiveway-abstention-v1-report.md`, `.tiller/scratch/codex/eos-router-fiveway-learned-router-fit-v1-report.md`, `.tiller/scratch/codex/eos-current-span256-router-abstention-v1-report.md`.

Rejected s40 LongEmbed replay probes:

| Probe | Outcome |
| --- | --- |
| LongEmbed teacher batch | Rejected on NFCorpus nDCG `-0.000197`; QMSum q4 lifted, but 2Wiki drifted. |
| Anchor-protected continuation | Rejected on NFCorpus recall `-0.000054` and FIQA nDCG `-0.000107`; QMSum q4/fusion stayed positive but with less lift. |
| Balanced anchor sweep best `fiqa24-nf48` | Rejected only on FIQA nDCG `-0.000057`; NFCorpus recall was preserved at `+0.000007`, but the useful QMSum q4/fusion signal fell below s40 and prior probes. |
| FIQA targeted replay with fallback anchors | Rejected on NFCorpus recall `-0.000047` and FIQA nDCG `-0.000057`; the aggregate FIQA miss came from only query `6133`, doc `7733`, rank `2 -> 3`. |
| FIQA single replay | Rejected with the same NFCorpus recall and FIQA nDCG failures; embeddings moved slightly, but the diagnostic fixture rank order stayed unchanged. |

Next LongEmbed work should use a new non-test FIQA-compatible teacher or synthetic signal, a trained compact/Matryoshka objective with a movement gate, or a larger-model/bootstrap path. If LongEmbed remains the target, use a new non-test signal or objective rather than more eval-slice replay micro-tuning around this protected batch.

`scripts/diagnose_eos_embedding_movement.fw` is now the required cheap preflight before expensive compact-head or tiny-continuation sweeps. It compares two Eos artifacts through the retrieval export surface and reports vector movement plus synthetic rank-order changes. The Matryoshka-only 128d probe was exactly pinned at 64d, 128d, and full dimensions, while the TurboQuant-prefix branch moved slightly. Require movement-positive and no-restore evidence before spending another compact-head sweep.

Use `eos eval-retrieval-vectors-turboquant` on those same caches to compare q2/q4/q8 TurboQuant IP document-vector indexes without adding provider SDKs. This is the key CorkScrewDB default-promotion comparison surface because every candidate, including external BGE/Qwen/Jina/Voyage/Cohere/OpenAI caches and sealed local Eos embeddings, can be judged by dense quality plus compressed vector-index quality and cost.
Enable `EOS_SCOREBOARD_HYBRID_RETRIEVAL=1` to add lexical+dense hybrid rows without removing dense, BM25, or TurboQuant rows. The harness runs `eos eval-retrieval-hybrid` for local Eos rows and `eos eval-retrieval-vectors-hybrid` for external vector caches, using `EOS_SCOREBOARD_HYBRID_METHOD`, `EOS_SCOREBOARD_HYBRID_ALPHA`, `EOS_SCOREBOARD_HYBRID_RRF_K`, and `EOS_SCOREBOARD_HYBRID_RRF_LAMBDA`. Treat the FiQA dev-selected `minmax`/`alpha=0.75` setting as guarded hybrid retrieval evidence only; it does not promote the dense embedder.
For pairwise `train-embed --eval-only` rows, the harness uses `EOS_SCOREBOARD_PAIRWISE_ARTIFACT` when set; otherwise it infers the sibling trainable package when `EOS_SCOREBOARD_ARTIFACT` points at a sealed `.mll`.

Run a closed retrieval-alignment round when the scoreboard shows a retrieval gap:

```bash
EOS_REPO_ROOT=$PWD \
EOS_ALIGN_INITIAL_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_ALIGN_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/run_manta_embed_v1_retrieval_alignment_round.fw
```

The alignment harness runs the baseline scoreboard, mines model-hard negatives into the run directory, blends them with the existing BM25 hard-negative file, trains a focused candidate from the initial trainable artifact, runs the candidate scoreboard, and writes `retrieval-alignment-summary.tsv` plus `retrieval-alignment-summary.json`. It is the first loop to run after the current Eos candidate underperforms BM25 or open embedders on BEIR-style retrieval.
The default candidate training pass uses batch `64` and one explicit hard negative per query so full mined BEIR-style runs stay iterative. Increase `EOS_ALIGN_CANDIDATE_BATCH_SIZE` or `EOS_ALIGN_CANDIDATE_HARD_NEGATIVES` only after the planned train-pair count is acceptable for the local GPU.
Use `EOS_ALIGN_MODEL_HARD_DATASET_WEIGHTS=fiqa=2` or similar when a dataset needs more mined examples in the next mixed round.
Use `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=1,fiqa=1` or similar when the train JSONL carries `source` fields and the candidate should build source-balanced hard-negative batches without rewriting the dataset. Exact source keys such as `fiqa:model` override the family key; otherwise `fiqa:model` falls back to `fiqa`.
Use `EOS_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT=0.20` or similar when mined hard negatives carry `teacher_scores` and the candidate should preserve the teacher's soft ordering over each query's positive plus explicit negatives. `EOS_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` controls the softness of that target distribution.
Use `EOS_ALIGN_CANDIDATE_TEACHER_SOURCE_TEMPERATURES=scifact=10,nfcorpus=10,fiqa=10,scifact:model=1.5,nfcorpus:model=1.5,fiqa:model=1.5` when mixed teacher-score sources need different softmax temperatures. Exact keys win first, then source-family keys, then `*`, then the global teacher temperature.
Use `EOS_ALIGN_CANDIDATE_TEACHER_SCORE_NORMALIZATION=source_zscore` when mixed `teacher_scores` are on incompatible scales before the teacher softmax. Supported modes are `source_zscore`, `family_zscore`, and `example_zscore`; leave it empty for raw scores.
For promotion runs, set `EOS_ALIGN_GATE_CANDIDATE=1`. The gate writes the summary first, then fails the run if macro nDCG@10 misses `EOS_ALIGN_MIN_MACRO_NDCG_DELTA` or any dataset regresses more than `EOS_ALIGN_MAX_DATASET_NDCG_REGRESSION`; `EOS_ALIGN_MIN_DATASET_NDCG_RATIO` adds an optional per-dataset nDCG ratio floor. Use `EOS_ALIGN_MAX_DATASET_RECALL_AT_100_REGRESSION` and `EOS_ALIGN_MIN_DATASET_RECALL_AT_100_RATIO` when a candidate must also preserve top-k coverage.

## Current Retrieval Alignment Findings

The credible wedge is improving the in-repo embedder on selected BEIR-style retrieval tasks without losing the consumer-GPU training/export loop. The current best nDCG checkpoint is the teacher-distilled lower-weight hybrid InfoNCE candidate:

| Candidate | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 | Macro | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage C baseline | 0.229650 | 0.052655 | 0.015580 | 0.099295 | superseded |
| Full mined `b64` | 0.296863 | 0.078214 | 0.020453 | 0.131843 | superseded |
| FiQA rescue | 0.306736 | 0.078341 | 0.022386 | 0.135821 | superseded |
| Weighted FiQA2 | 0.307661 | 0.072530 | 0.024936 | 0.135042 | FiQA/SciFact gain, NF regression |
| Source exact kitchen | 0.295726 | 0.051626 | 0.018608 | 0.121987 | rejected |
| Source family safe | 0.305378 | 0.068838 | 0.024872 | 0.133029 | rejected as balanced candidate |
| Grouped InfoNCE only | 0.222989 | 0.048378 | 0.020070 | 0.097146 | rejected by promotion gate |
| Hybrid InfoNCE w0.25 | 0.303582 | 0.069264 | 0.024596 | 0.132481 | FiQA gain, rejected by balanced gate |
| Hybrid InfoNCE w0.10 NF2 LR25 | 0.310343 | 0.077658 | 0.027189 | 0.138397 | superseded; balanced gate pass |
| Hybrid ratchet w0.05 NF3 LR12.5 | 0.309634 | 0.074894 | 0.026093 | 0.136874 | rejected; pairwise AUC up, retrieval down |
| Fresh-mine hybrid w0.05 NF3mine/NF2train LR12.5 | 0.324437 | 0.084556 | 0.027711 | 0.145568 | superseded; gate pass |
| Teacher hybrid w0.05 tw0.10 NF3train LR10 | 0.330978 | 0.083022 | 0.028878 | 0.147626 | rejected; NFCorpus nDCG floor |
| Teacher hybrid w0.05 tw0.20 NF2train LR10 | 0.327075 | 0.082980 | 0.028849 | 0.146301 | rejected; NFCorpus nDCG floor |
| Teacher hybrid w0.05 tw0.20 NF3train LR10 | 0.330614 | 0.084433 | 0.028539 | 0.147862 | superseded; gate criteria pass |
| Teacher hybrid w0.05 tw0.20 tt0.75 NF3train LR10 | 0.329814 | 0.084532 | 0.028868 | 0.147738 | gate pass, below current best |
| Teacher hybrid w0.05 tw0.20 tt1.25 NF3train LR10 | 0.330311 | 0.083897 | 0.028727 | 0.147645 | gate pass, below current best |
| Teacher hybrid w0.05 tw0.20 tt1.50 NF3train LR10 | 0.331139 | 0.084325 | 0.028967 | 0.148143 | current nDCG best; gate criteria pass |
| Teacher hybrid w0.05 tw0.20 tt1.50 NF3train LR8 | 0.329199 | 0.084927 | 0.028750 | 0.147625 | gate pass, NFCorpus high-water mark |
| Teacher hybrid w0.05 tw0.20 tt1.50 NF4train LR10 | 0.331362 | 0.083434 | 0.027883 | 0.147560 | rejected; NFCorpus nDCG floor |
| Teacher hybrid w0.05 tw0.20 tt1.50 SF2/NF3train LR10 | 0.327158 | 0.083568 | 0.028138 | 0.146288 | baseline gate pass, current-best reject |
| Teacher hybrid w0.05 tw0.20 tt1.50 SF1/NF3/FIQA2train LR10 | 0.330051 | 0.084318 | 0.028178 | 0.147516 | baseline gate pass, current-best reject |
| Teacher hybrid w0.05 tw0.05 tt10 NF3train BM25-scored LR10 | 0.330122 | 0.083344 | 0.027986 | 0.147151 | rejected; full-score coverage hurt NFCorpus |
| Teacher hybrid w0.05 tw0.20 BM25 temp10/model temp1.5 NF3train BM25-scored LR10 | 0.323329 | 0.083238 | 0.029619 | 0.145395 | rejected; source temps lifted FiQA only |
| Teacher hybrid w0.05 tw0.05 BM25 temp10/model temp1.5 NF3train BM25-scored LR10 | 0.327390 | 0.082918 | 0.029068 | 0.146459 | rejected; stale macro pass, NFCorpus floor fail |
| Teacher hybrid w0.05 tw0.20 tt1.50 source-zscore NF3train BM25-scored LR10 | 0.322086 | 0.082917 | 0.029375 | 0.144793 | rejected; normalization overpressured SciFact/NF |
| Teacher hybrid w0.05 tw0.05 tt1.50 source-zscore NF3train BM25-scored LR10 | 0.331279 | 0.083764 | 0.028101 | 0.147714 | baseline gate pass, current-best reject |
| Teacher hybrid w0.05 tw0.05 tt1.50 source-zscore SF1/NF3/FIQA2train BM25-scored LR10 | 0.329793 | 0.083956 | 0.028354 | 0.147368 | baseline gate pass, current-best reject |
| Teacher hybrid w0.05 tw0.05 tt1.50 example-zscore NF3train BM25-scored LR10 | 0.329417 | 0.085742 | 0.029123 | 0.148094 | near-anchor; current-best reject by 0.000049 macro |
| Teacher hybrid w0.05 tw0.20 tt1.50 deepmine9k k400 hn2 LR10 | 0.320576 | 0.084568 | 0.026453 | 0.143866 | rejected; hard-negative overpressure |
| Teacher hybrid w0.05 tw0.20 tt1.50 deepmine9k k400 hn1 LR10 | 0.324630 | 0.087300 | 0.025679 | 0.145870 | rejected; NFCorpus high-water mark |
| Teacher hybrid w0.05 tw0.20 tt1.50 deepmine9k k400 balanced hn1 LR10 | 0.322932 | 0.086364 | 0.025450 | 0.144915 | rejected; balanced source worsened |
| Teacher hybrid w0.05 tw0.20 tt1.50 deepmine9k k400 hn1 LR5 | 0.323136 | 0.086784 | 0.027508 | 0.145809 | rejected; smaller LR recovers FiQA only |
| Teacher hybrid w0.025 tw0.20 tt1.50 deepmine9k k400 hn1 LR5 | 0.325439 | 0.086645 | 0.027204 | 0.146429 | rejected; best Lane B balance |
| Embed-m cached16k w0.05 tw0.20 tt1.50 HN1 LR10 | 0.160753 | 0.060778 | 0.012688 | 0.078073 | rejected; cold-start capacity probe |
| Teacher hybrid w0.05 tw0.20 tt2.00 NF3train LR10 | 0.331232 | 0.083795 | 0.029060 | 0.148029 | gate pass, below current best |
| Teacher hybrid w0.05 tw0.35 NF3train LR10 | 0.329387 | 0.083761 | 0.028540 | 0.147229 | gate pass, below current best |

Interpretation:

- Source-aware hard-negative scheduling is now implemented and validated. Simple source balancing alone did not beat the FiQA rescue checkpoint, but source bias combined with a lower-weight hybrid objective produced the first gated balanced improvement.
- Pairwise AUC can improve while corpus ranking regresses, so promotion must use full retrieval scoreboards, not pairwise eval alone.
- The retrieval-alignment harness now has an optional promotion gate so failed candidates leave useful artifacts but cannot quietly pass a release-style run.
- `grouped_infonce` is implemented and validated mechanically, but the first grouped-only gated run regressed every retrieval task. It used `26,076` grouped train pairs per epoch and zero contrastive accelerator calls, then failed the gate with macro nDCG@10 delta `-0.038675`. Treat grouped-only ranking as a negative objective result.
- `hybrid_infonce` keeps the rectangular query-candidate accelerator path active while adding explicit per-query hard-negative ordering. The first weight-`0.25` run processed `1,693,284` train pairs per epoch with `204` contrastive accelerator calls and improved FiQA nDCG@10 from `0.022386` to `0.024596`, but regressed SciFact and NFCorpus enough to fail the balanced gate.
- The lower-weight hybrid run (`grouped_loss_weight=0.10`, LR `0.000025`, source weights `scifact=1,nfcorpus=2,fiqa=1`) passed the strict promotion gate: macro nDCG@10 moved from `0.135821` to `0.138397`, SciFact improved, FiQA improved, and NFCorpus stayed inside the allowed regression band while recall@100 improved. This is the current balanced checkpoint to promote and rebase the next round from.
- A lower-LR ratchet from the promoted checkpoint (`grouped_loss_weight=0.05`, LR `0.0000125`, source weights `scifact=1,nfcorpus=3,fiqa=1`) improved pairwise eval AUC to `0.832026` but failed the promotion gate: macro nDCG@10 moved from `0.138397` to `0.136874`, NFCorpus fell by `0.002764`, and FiQA fell by `0.001096`. Stop ratcheting the same mined blend; the next proof layer should remine model-hard negatives from the promoted artifact or add teacher-score distillation so training sees new ranking signal.
- Fresh mining from the promoted checkpoint validated that proof layer. The fresh-mined candidate used `6000` max model-hard examples with mining weights `scifact=1,nfcorpus=3,fiqa=1`, training source weights `scifact=1,nfcorpus=2,fiqa=1`, `grouped_loss_weight=0.05`, and LR `0.0000125`; it passed the gate with macro nDCG@10 delta `+0.007171`. Recall@100 dipped on all three retrieval sets, especially FiQA (`0.171556` to `0.164380`), so the next promotion run should enable the recall floor and/or teacher-score distillation that preserves wider top-k coverage.
- Teacher-score distillation is now implemented for hard-negative runs. BM25 and model-hard mining can preserve per-candidate teacher scores, tokenization carries them through, and `teacher_loss_weight` blends soft teacher-distribution cross-entropy with the selected hard-negative objective.
- Teacher-score distillation produced the next macro lift. With `teacher_loss_weight=0.20`, LR `0.000010`, `grouped_loss_weight=0.05`, and fresh teacher-scored model-hard JSONL, the NF2 source schedule raised macro nDCG@10 to `0.146301` but failed the NFCorpus nDCG floor by `0.000576`. Reusing the same JSONL with `source_weights=scifact=1,nfcorpus=3,fiqa=1` raised macro nDCG@10 to `0.147862`; SciFact gained `+0.006177`, FiQA gained `+0.000828`, NFCorpus stayed within the floor at `-0.000123`, and recall@100 was flat or better on all three datasets. Lower teacher pressure (`teacher_loss_weight=0.10`) raised macro nDCG@10 to `0.147626` but is a reject because NFCorpus fell by `0.001534`; higher pressure (`teacher_loss_weight=0.35`) passed the gate at macro `0.147229` but did not beat `0.20`.
- Teacher temperature is now an active SOTA knob. Holding `teacher_loss_weight=0.20`, `teacher_temperature=0.75` passed the gate at macro `0.147738` but stayed below the temperature-`1.0` run. Softer `teacher_temperature=1.5` is the new retrieval best at macro `0.148143`: SciFact `+0.006702`, NFCorpus `-0.000231`, and FiQA `+0.001256` versus the fresh-mined baseline, with recall floors intact. `teacher_temperature=1.25` also landed below the temperature-`1.0` run at macro `0.147645`; `2.0` improved SciFact and FiQA further but dropped NFCorpus enough to land at macro `0.148029`. Lowering LR to `0.000008` at temperature `1.5` passed at macro `0.147625` and set the NFCorpus high-water mark at `0.084927`, but gave back too much SciFact. Increasing the source schedule to `nfcorpus=4` rejected: SciFact rose to `0.331362`, but NFCorpus fell below the nDCG floor and FiQA regressed. Adding SciFact pressure with `source_weights=scifact=2,nfcorpus=3,fiqa=1` passed the stale-baseline gate at macro `0.146288`, but missed the current best by `0.001855` and pairwise guardrails dropped to validation AUC `0.815529` / hard AUC `0.808770`. Adding FiQA pressure with `source_weights=scifact=1,nfcorpus=3,fiqa=2` also passed the stale-baseline gate at macro `0.147516`, but missed the current best by `0.000628`; FiQA nDCG fell from `0.028967` to `0.028178` despite the extra sampling weight. Treat the local source-weight branch as closed; the next local refinement should acquire new ranking signal by remapping hard negatives, importing an external teacher, adding synthetic query data, or bootstrapping a larger model.
- Full BM25 teacher-score coverage is mechanically available but not yet a quality win. Auditing the current teacher blend showed that only model-mined rows had scores (`5719/10638` examples), so the BM25 half was regenerated with scores and the acquisition script was fixed to preserve `teacher_scores` when source-tagging hard-negative rows. Training that full-coverage blend with softened BM25 scores (`teacher_loss_weight=0.05`, `teacher_temperature=10`) improved pairwise AUC to `0.822831`, but failed the stale-baseline retrieval gate: macro `0.147151`, NFCorpus nDCG delta `-0.001212`, and all three nDCG rows below the current anchor. Source-specific teacher temperatures are implemented and passed through the alignment scripts, but split-temperature runs still rejected: `teacher_loss_weight=0.20` landed at macro `0.145395`, and `teacher_loss_weight=0.05` landed at macro `0.146459` with stale macro pass but NFCorpus nDCG delta `-0.001638`. Teacher-score normalization is implemented as the next layer. `source_zscore` with `teacher_loss_weight=0.20` rejected at macro `0.144793`, but lowering normalized teacher pressure to `0.05` passed the stale-baseline gate at macro `0.147714`, set SciFact to `0.331279`, and improved pairwise AUC to validation `0.823381` / hard `0.814565`. It still missed the current anchor by `0.000429` macro because FiQA fell to `0.028101`. A FiQA-weighted sampler on that normalized full-score branch passed the stale gate at macro `0.147368` and recovered some FiQA/NFCorpus (`0.028354` / `0.083956`), but SciFact fell to `0.329793`, widening the current-anchor macro miss to `0.000775`. `example_zscore` is the strongest normalized result: SciFact `0.329417`, NFCorpus `0.085742`, FiQA `0.029123`, macro `0.148094`, validation AUC `0.823282`, and hard AUC `0.815203`. It missed the current anchor by only `0.000049` macro. The narrow SciFact-recovery tweak (`scifact=2,nfcorpus=3,fiqa=1`) rejected before full scoreboard with validation/hard AUC `0.817674` / `0.810527` and SciFact `0.326679`. Stop local score-normalization reshuffling and move to stronger external/synthetic teacher signal.
- The first Lane B deeper-mining round from the temperature-`1.5` best used `9000` requested model-hard examples, `candidate_top_k=400`, `5` mined negatives, and `2` candidate hard negatives. Mining produced `919` SciFact, `5400` NFCorpus, and `1800` FiQA examples; the blended training file had `13,038` hard-negative examples and trained `2.245M` actual pairs at `4262` train pairs/s with CUDA forward/optimizer/activation/contrastive. The promotion gate failed: macro fell from `0.148144` to `0.143866`, SciFact lost `0.010563`, FiQA lost `0.002514`, and pairwise AUC dropped to validation `0.802662` / hard `0.800179`. NFCorpus rose slightly to `0.084568`, so reuse the mined JSONL with `hard_negatives_per_query=1` before discarding the deeper mining signal.
- Reusing that Lane B mined JSONL with one candidate hard negative reduced the damage and set the NFCorpus high-water mark at `0.087300`, but still failed against the current best: macro `0.145870`, SciFact `-0.006508`, FiQA `-0.003288`, and pairwise AUC `0.802976` / `0.800840`. Because the mined file already carries `5400` NFCorpus model-hard examples, continuing to train it with `source_weights=scifact=1,nfcorpus=3,fiqa=1` over-allocates NF signal. The next isolation run should keep the same JSONL and HN1 shape but switch training source weights to `scifact=1,nfcorpus=1,fiqa=1`.
- Balanced source weights on the same deep-mined HN1 JSONL did not recover the non-NF tasks. It landed at macro `0.144915`: SciFact `0.322932`, NFCorpus `0.086364`, FiQA `0.025450`, validation AUC `0.794627`, and hard AUC `0.794320`. This closes source-sampler rescue for the deep-mined file. If continuing Lane B locally, use the NF3 HN1 shape with a smaller update, such as LR `0.000005` or `grouped_loss_weight=0.025`; otherwise pivot to external teacher import or `embed-m`.
- Lowering the HN1 NF3 reuse to LR `0.000005` improved FiQA relative to LR `0.000010` (`0.027508` versus `0.025679`) and preserved strong NFCorpus (`0.086784`), but SciFact remained too low at `0.323136` and macro stayed at `0.145809`. Pairwise guardrails stayed weak at validation AUC `0.803438` / hard AUC `0.800508`. A final local Lane B retry can reduce `grouped_loss_weight` to `0.025`; after that, close the deep-mined file for balanced promotion and move to external teacher import or `embed-m`.
- Reducing the grouped term to `0.025` at LR `0.000005` produced the best Lane B balance but still failed promotion: macro `0.146429`, SciFact `0.325439`, NFCorpus `0.086645`, FiQA `0.027204`, validation AUC `0.804674`, hard AUC `0.801615`. This closes the current deep-mined file for balanced promotion. Keep it as evidence that deeper mining can specialize NFCorpus, but the next credible balanced path is external teacher import or a larger `embed-m` run, not more local source/LR/grouped reshuffling.
- A direct `embed-m` run with the target `32768` vocab, max sequence `512`, dim `192`, hidden `384`, and `3` encoder repeats reached model initialization but spent more than fifteen minutes CPU-bound in full-corpus tokenizer training before any optimizer step. Treat true `32768`-vocab `embed-m` as blocked on cached tokenizer artifacts or a faster tokenizer trainer, not on model initialization.
- The cached-tokenizer `embed-m` capacity probe (`vocab=16384`, max sequence `512`, dim `192`, hidden `384`, repeats `3`) trained and sealed successfully at batch `64` on the desktop GPU. It processed `1.393M` actual train pairs in `19m20s` with CUDA forward/optimizer/activation/contrastive accelerators and `1460.78` train pairs/s, but quality collapsed from the current best: validation AUC `0.595854`, hard AUC `0.598887`, SciFact `0.160753`, NFCorpus `0.060778`, FiQA `0.012688`, macro `0.078073`. The fine-tune LR `0.000010` is not a valid random-start recipe for `embed-m`.
- A scratch-style cached-tokenizer `embed-m` pretrain pass with pure hard-negative `infonce`, LR `0.002`, one epoch, and batch `64` also rejected before retrieval: validation AUC `0.495137`, hard AUC `0.498731`, hard score margin `-0.000154`, and `1259.54` train pairs/s. This closes blind random-start `embed-m` as a near-term path. The next larger-model attempt needs either dimension-compatible bootstrapping from a smaller trained model, staged pretraining data, or imported teacher scores rather than another single-epoch scratch LR sweep.
- The subquadratic/TurboQuant kernel path now has shared attention-plan metadata across host reference, CUDA sparse attention, and fused CUDA TurboQuant sparse attention. Each path reports top-k, routing mode, route block counts, selected keys, candidate key budget, estimated per-query score work, and score-work fraction versus dense scoring, so future long-context runs can gate on actual routed work rather than just successful execution.
- `eos plan-sparse-attention` now turns that metadata into a cheap preflight gate: sweep key lengths, route budgets, and TurboQuant bits, then reject configurations whose routed score-work fraction, fitted score-work alpha, or logical K/V memory budget do not fit the target card before launching CUDA benchmarks or training.
- `scripts/bench_sparse_attention.fw` now archives that preflight beside CUDA sparse-attention benchmark JSONL/text, a parsed summary TSV, and a measured scaling-alpha TSV, so exact f16 and routed TurboQuant sparse kernels can be compared from a clean run directory with selected-key, candidate-budget, score-fraction, subquadratic, K/V compression, and routed time-alpha gate metrics attached.
- The sparse benchmark matrix now has separate exact and routed key-length knobs, so the local proof can bound dense/exact costs while pushing routed TurboQuant to longer contexts for the alpha gate.
- Sparse embedding smoke now has CUDA backend evidence and a strict backend-vs-host TurboQuant parity gate, but sparse routing remains calibration evidence. `eos calibrate-sparse-routing` now sweeps `anchor`, `multiprobe`, `summary_mean`, `summary_mean_radius`, `summary_maxnorm`, `summary_blend_radius`, calibration-only `summary_multirep`, calibration-only `summary_hier_multirep`, calibration-only `learned_block_linear`, and teacher-only `oracle_block_max` policies with an optional per-query minimum recall gate. The deployable summary policies use precomputed one-vector-per-block representatives plus optional scalar radius correction, including max-norm representatives and alpha/beta blending. `summary_multirep` precomputes deterministic farthest-point representatives per block and scores the max dot across K representatives, explicitly charging `block_count * K` route scores plus candidate-key work. `summary_hier_multirep` probes a two-level budget: one coarse max-norm representative score per group, then K fine representatives only inside selected coarse groups, with row fields for group size, top groups, coarse group count, and fine summary-score work. `learned_block_linear` trains a tiny deterministic logistic linear scorer from `oracle_block_max` block labels on synthetic calibration seeds, then evaluates from learned weights and cheap block features only. Default-scale anchor and simple block/config sweeps failed the `score_count_fraction <= 0.2`, `recall_avg >= 0.95`, `cosine >= 0.98` gates; best under-budget multiprobe (`4096x8x64`, block `64`, top blocks `8`, probes `4`) reached recall avg `0.486328125`, cosine `0.98335072496811`, and score fraction `0.1875`. A same/held-out seed learned-router probe at `4096x8x64`, block `10`, top blocks `35..40` also failed strict gates; its best learned row reached recall avg/min `0.609375 / 0.46875`, cosine `0.9833500752094315`, and score fraction `0.19775390625`. Coarse oracle block-size sweeping with min recall found no robust pass; its best under-budget row, block `16` / top blocks `32`, reached recall avg/min `0.9921875 / 0.9375`, cosine `0.999981385545`, and score fraction `0.1875`, failing only min recall `0.95`. A later fine-grained teacher-only oracle boundary sweep showed the skipped ideal region: block size `10`, top blocks `38` reached recall avg/min `1 / 1`, cosine `1`, MSE `0`, and score fraction `0.19287109375`; passing ranges were block `8` top blocks `36,37,38`, block `10` `35..40`, block `12` `35..39`, block `14` `34..37`, block `16` `33..35`, and block `24` none. Treat this as upper-bound routing-label evidence, not retrieval quality proof or a deployable selector; the next architecture decision is to use richer learned features or a different block-label objective before runtime or CUDA routing changes.

Short retrieval metrics:

- nDCG@10
- recall@10/100
- MRR@10 where applicable
- score margin
- AUC for pairwise eval

Long retrieval metrics:

- nDCG@10 by document length bucket
- passkey/needle accuracy
- dispersed-evidence retrieval accuracy
- chunked-baseline delta
- long-context truncation robustness

Efficiency metrics:

- train examples/s
- encode docs/s
- encode tokens/s
- peak GPU memory
- index bytes per document
- embedding dimension
- model package bytes
- dense fallback count
- host fallback count
- sparse attention metadata

## Release Gates

Initial local release gate:

- beats prior Eos baseline on hard eval
- no package verification failures
- sealed `.mll` export works
- efficiency metrics recorded

Long-context release gate:

- beats chunked short-context baseline on long-retrieval metrics
- handles at least `32k` document input on C24
- no dense attention fallback in routed mode
- no more than `2%` short-context regression versus non-routed candidate

Best-local claim gate:

- beats named open/local baselines on the chosen public or domain scorecard
- serves locally on C24
- ships reproducible benchmark scripts and manifests
- reports model size, vector size, latency, memory, and quality together

## Immediate Experiments

Experiment 1: baseline scoreboard

- run current Eos embedder against the fixed baseline tasks
- use `scripts/score_manta_embed_v1_baselines.fw` to produce `scoreboard.tsv` and `scoreboard.json`
- include pairwise eval, hard eval, BEIR-style retrieval, optional long retrieval, and BM25 baselines
- identify the smallest set of evals that catches real retrieval regressions

Experiment 2: teacher distillation

- choose a strong teacher baseline
- generate teacher scores for train/eval pairs
- train a compact Eos embedder with teacher similarity loss
- compare to contrastive-only training

Experiment 3: hard-negative loop

- mine model-hard negatives from the current candidate
- retrain with weighted hard negatives
- measure RAG/domain and BEIR deltas
- use `scripts/run_manta_embed_v1_retrieval_alignment_round.fw` for the reproducible baseline -> mine -> train -> score loop
- promote only candidates with positive nDCG@10 movement and no material recall@100 regression on the chosen retrieval set

Experiment 4: long-document eval

- add LongEmbed-style prepared datasets
- benchmark chunked baselines
- establish the first long-context quality target

Experiment 5: sparse encoder prototype

- use routed TurboQuant sparse attention in an encoder block
- compare exact vs routed on small contexts
- profile sequence-length scaling
- record whether quality survives routing

Experiment 6: compressed serving

- export compressed `.mll`
- run local encode/index loop
- report package size, VRAM, docs/s, and index bytes/doc

## Direct Path To The Wedge

The shortest path is:

1. stabilize the existing production embedding pipeline
2. build a scoreboard that includes short retrieval, long retrieval, and efficiency
3. use teacher distillation and hard negatives to get quality
4. add long-context data and chunk-alignment training
5. integrate sparse attention for long sequences
6. prove C24 training/serving
7. publish the best-local claim only on the scorecard we actually win

This path does not require solving general long-context LLM training first. It uses the fact that embedding quality is trainable at smaller scale, long-context retrieval has open space, and Eos can own the deployment/runtime path end to end.

The expanded SOTA experiment surface is tracked in [manta-embed-sota-avenues.md](manta-embed-sota-avenues.md). Treat that file as the queue for objective, data, architecture, retrieval-head, compression, and sparse-long-context lanes.
