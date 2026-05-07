# Local Long-Context Embedder Wedge

This is the product wedge for Manta's sparse attention work:

```text
Build the best local long-context embedder that can be trained, adapted, exported, and served on consumer GPUs.
```

This is more credible than claiming general frontier-model performance from a desktop card. Embedding models are smaller, retrieval quality can be improved through distillation and hard negatives, long-context embedding is still underdeveloped, and Manta's TurboQuant plus routed sparse attention stack directly attacks the memory and attention-cost bottlenecks that make long-document embedding expensive.

## Target Claim

The target claim is:

```text
Manta can train and serve a best-in-class local long-context embedding model on a single consumer GPU, with sealed .mll export, compressed deployment, and subquadratic long-context attention.
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
- current `manta-embed-v1` package

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

### Stage A: Manta Embed V1 Reliable Baseline

Goal: make the existing native embedding pipeline a trustworthy baseline.

Work:

- keep BEIR acquisition and production gates current
- add long-document train/eval splits
- preserve sealed `.mll` export and tokenizer packaging
- record reproducible metrics for every candidate

Pass:

- current production embedding gates pass
- candidate beats the prior Manta baseline on hard eval
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

- beats `manta-embed-v1` by a meaningful margin on hard eval
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

- candidate is better than current Manta baseline on local RAG gates
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
MANTA_REPO_ROOT=$PWD \
MANTA_SCOREBOARD_ARTIFACT=/path/to/manta-embed-v1.sealed.mll \
MANTA_SCOREBOARD_PAIRWISE_JSONL=/data/manta/datasets/manta-embed-v1/processed/eval.jsonl \
MANTA_SCOREBOARD_HARD_JSONL=/data/manta/datasets/manta-embed-v1/processed/hard-eval.jsonl \
MANTA_SCOREBOARD_RETRIEVAL_ROOT=/data/manta/datasets/manta-embed-v1 \
MANTA_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The scoreboard harness writes `runs/<run-id>/scoreboard.tsv`, `runs/<run-id>/scoreboard.json`, per-task metrics JSON, command logs, and the exact `manta` binary used for the run. Enable `MANTA_SCOREBOARD_BM25_BASELINE=1` to include the in-repo BM25 baseline beside Manta retrieval scores.
For pairwise `train-embed --eval-only` rows, the harness uses `MANTA_SCOREBOARD_PAIRWISE_ARTIFACT` when set; otherwise it infers the sibling trainable package when `MANTA_SCOREBOARD_ARTIFACT` points at a sealed `.mll`.

Run a closed retrieval-alignment round when the scoreboard shows a retrieval gap:

```bash
MANTA_REPO_ROOT=$PWD \
MANTA_ALIGN_INITIAL_ARTIFACT=/path/to/manta-embed-v1.sealed.mll \
MANTA_ALIGN_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/run_manta_embed_v1_retrieval_alignment_round.fw
```

The alignment harness runs the baseline scoreboard, mines model-hard negatives into the run directory, blends them with the existing BM25 hard-negative file, trains a focused candidate from the initial trainable artifact, runs the candidate scoreboard, and writes `retrieval-alignment-summary.tsv` plus `retrieval-alignment-summary.json`. It is the first loop to run after the current Manta candidate underperforms BM25 or open embedders on BEIR-style retrieval.
The default candidate training pass uses batch `64` and one explicit hard negative per query so full mined BEIR-style runs stay iterative. Increase `MANTA_ALIGN_CANDIDATE_BATCH_SIZE` or `MANTA_ALIGN_CANDIDATE_HARD_NEGATIVES` only after the planned train-pair count is acceptable for the local GPU.
Use `MANTA_ALIGN_MODEL_HARD_DATASET_WEIGHTS=fiqa=2` or similar when a dataset needs more mined examples in the next mixed round.
Use `MANTA_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=1,fiqa=1` or similar when the train JSONL carries `source` fields and the candidate should build source-balanced hard-negative batches without rewriting the dataset. Exact source keys such as `fiqa:model` override the family key; otherwise `fiqa:model` falls back to `fiqa`.
Use `MANTA_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT=0.20` or similar when mined hard negatives carry `teacher_scores` and the candidate should preserve the teacher's soft ordering over each query's positive plus explicit negatives. `MANTA_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` controls the softness of that target distribution.
For promotion runs, set `MANTA_ALIGN_GATE_CANDIDATE=1`. The gate writes the summary first, then fails the run if macro nDCG@10 misses `MANTA_ALIGN_MIN_MACRO_NDCG_DELTA` or any dataset regresses more than `MANTA_ALIGN_MAX_DATASET_NDCG_REGRESSION`; `MANTA_ALIGN_MIN_DATASET_NDCG_RATIO` adds an optional per-dataset nDCG ratio floor. Use `MANTA_ALIGN_MAX_DATASET_RECALL_AT_100_REGRESSION` and `MANTA_ALIGN_MIN_DATASET_RECALL_AT_100_RATIO` when a candidate must also preserve top-k coverage.

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
| Teacher hybrid w0.05 tw0.20 tt1.50 NF3train LR10 | 0.331139 | 0.084325 | 0.028967 | 0.148143 | current nDCG best; gate criteria pass |
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
- Teacher temperature is now an active SOTA knob. Holding `teacher_loss_weight=0.20`, `teacher_temperature=0.75` passed the gate at macro `0.147738` but stayed below the temperature-`1.0` run. Softer `teacher_temperature=1.5` is the new retrieval best at macro `0.148143`: SciFact `+0.006702`, NFCorpus `-0.000231`, and FiQA `+0.001256` versus the fresh-mined baseline, with recall floors intact. `teacher_temperature=2.0` improved SciFact and FiQA further but dropped NFCorpus enough to land at macro `0.148029`, so the next local refinement should tighten around `1.5` rather than keep softening broadly.

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

- beats prior Manta baseline on hard eval
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

- run current Manta embedder against the fixed baseline tasks
- use `scripts/score_manta_embed_v1_baselines.fw` to produce `scoreboard.tsv` and `scoreboard.json`
- include pairwise eval, hard eval, BEIR-style retrieval, optional long retrieval, and BM25 baselines
- identify the smallest set of evals that catches real retrieval regressions

Experiment 2: teacher distillation

- choose a strong teacher baseline
- generate teacher scores for train/eval pairs
- train a compact Manta embedder with teacher similarity loss
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

This path does not require solving general long-context LLM training first. It uses the fact that embedding quality is trainable at smaller scale, long-context retrieval has open space, and Manta can own the deployment/runtime path end to end.

The expanded SOTA experiment surface is tracked in [manta-embed-sota-avenues.md](manta-embed-sota-avenues.md). Treat that file as the queue for objective, data, architecture, retrieval-head, compression, and sparse-long-context lanes.
