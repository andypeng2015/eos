# Manta Embedder SOTA Avenue Map

This is the working map for trying everything that can plausibly move `manta-embed-v1` toward a best-in-class local embedder. The objective is not one clever loss. SOTA embedding systems combine foundation-model backbones, staged data, synthetic data, hard negatives, distillation, task routing, multi-output retrieval modes, and compression-aware serving. Manta should turn each of those into a measured lane.

## External Signals

Current public systems point at these ingredients:

- Qwen3 Embedding: multi-stage unsupervised pretraining, supervised fine-tuning, synthetic data from foundation models, model merging, 0.6B/4B/8B scales, and paired rerankers. The Qwen model card reports strong MTEB/MMTEB scores, including `Qwen3-Embedding-8B` at `70.58` mean task on MMTEB and `Qwen3-Embedding-0.6B` at `64.33`.
- BGE-M3: dense retrieval, sparse retrieval, and multi-vector retrieval in one model, with self-knowledge distillation across retrieval functions and long inputs up to `8192` tokens.
- Jina embeddings v3: task-specific LoRA adapters, long-context retrieval, and Matryoshka Representation Learning so output dimensions can shrink from `1024` down to small prefixes.
- ReasonEmbed: reasoning-intensive synthetic retrieval data plus adaptive sample weighting for difficult examples.
- SPLADE and ColBERT families: sparse lexical expansion and late-interaction multi-vector retrieval remain separate high-quality lanes from single-vector dense retrieval.

Sources consulted:

- https://huggingface.co/Qwen/Qwen3-Embedding-8B
- https://github.com/QwenLM/Qwen3-Embedding
- https://arxiv.org/abs/2506.05176
- https://arxiv.org/abs/2402.03216
- https://arxiv.org/abs/2409.10173
- https://arxiv.org/abs/2510.08252
- https://arxiv.org/abs/2205.13147
- https://arxiv.org/abs/2107.05720
- https://arxiv.org/abs/2004.12832

## Current Anchor

The current in-repo best is:

```text
runs/manta-embed-v1-teacher-hybrid-w005-tw020-tt150-nf3train-lr10-20260507T053803Z/candidate/manta-embed-v1.sealed.mll
```

Compared with the previous fresh-mined hybrid best:

| Dataset | Previous nDCG@10 | Candidate nDCG@10 | Delta | Previous recall@100 | Candidate recall@100 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.324437 | 0.331139 | +0.006702 | 0.725222 | 0.724111 | -0.001111 |
| NFCorpus | 0.084556 | 0.084325 | -0.000231 | 0.128650 | 0.129067 | +0.000417 |
| FiQA | 0.027711 | 0.028967 | +0.001256 | 0.164380 | 0.164881 | +0.000501 |
| Macro | 0.145568 | 0.148143 | +0.002575 | - | - | - |

This passes the current gate criteria: macro nDCG delta `>= 0.0005`, per-dataset nDCG regression no worse than `0.001`, per-dataset nDCG ratio `>= 0.98`, recall@100 regression no worse than `0.004`, and recall@100 ratio `>= 0.96`.

Rejected nearby probe:

| Probe | Macro | Reason |
| --- | ---: | --- |
| `teacher_loss_weight=0.10`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.147626 | NFCorpus nDCG@10 delta `-0.001534`, outside the `-0.001000` floor |
| `teacher_loss_weight=0.35`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.147229 | Gate pass, but lower macro than the current best |
| `teacher_loss_weight=0.20`, `teacher_temperature=0.75`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.147738 | Gate pass, but lower macro than temperature `1.5` |
| `teacher_loss_weight=0.20`, `teacher_temperature=1.25`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.147645 | Gate pass, but lower macro than temperature `1.0` and `1.5` |
| `teacher_loss_weight=0.20`, `teacher_temperature=2.0`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.148029 | Gate pass, but NFCorpus tradeoff keeps macro below temperature `1.5` |
| `teacher_loss_weight=0.20`, `teacher_temperature=1.5`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000008` | 0.147625 | Gate pass and NFCorpus high-water mark, but SciFact regression keeps macro below LR `0.000010` |
| `teacher_loss_weight=0.20`, `teacher_temperature=1.5`, `source_weights=scifact=1,nfcorpus=4,fiqa=1`, LR `0.000010` | 0.147560 | NFCorpus nDCG@10 delta `-0.001122`, outside the `-0.001000` floor |
| `teacher_loss_weight=0.20`, `teacher_temperature=1.5`, `source_weights=scifact=2,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.146288 | Baseline gate pass, but current-best macro and pairwise AUC both regressed |

## Ready-To-Run Lanes

These require no new model code.

### Lane A: Teacher Loss Shape

Question: is `teacher_loss_weight=0.20` the local optimum, or just the first useful point?

Sweep:

| Var | Values |
| --- | --- |
| `MANTA_TEACHER_LOSS_WEIGHT` | `0.05`, `0.10`, `0.20`, `0.35`, `0.50` |
| `MANTA_TEACHER_TEMPERATURE` | `0.5`, `0.75`, `1.0`, `1.5`, `2.0` |
| `MANTA_LR` | `0.000005`, `0.000008`, `0.000010`, `0.0000125` |
| `MANTA_HARD_NEGATIVE_SOURCE_WEIGHTS` | `scifact=1,nfcorpus=3,fiqa=1`, `scifact=1,nfcorpus=4,fiqa=1`, `scifact=2,nfcorpus=3,fiqa=1` |

Gate:

- candidate macro nDCG@10 beats `0.148143`
- no dataset violates nDCG or recall floors
- pairwise AUC does not fall below `0.818`

### Lane B: Mining Depth And Negative Budget

Question: are we under-sampling the teacher candidate set?

Sweep:

| Var | Values |
| --- | --- |
| `MANTA_ALIGN_MODEL_HARD_MAX_EXAMPLES` | `6000`, `9000`, `12000` |
| `MANTA_ALIGN_MODEL_HARD_NEGATIVES` | `3`, `5`, `8` |
| `MANTA_ALIGN_MODEL_HARD_CANDIDATE_TOP_K` | `100`, `200`, `400`, `800` |
| `MANTA_ALIGN_CANDIDATE_HARD_NEGATIVES` | `1`, `2`, `3` |

Gate:

- train-pair count stays within host budget
- recall@100 improves or stays flat on NFCorpus and FiQA
- nDCG improvement is not only SciFact

### Lane C: Source Scheduling

Question: can source scheduling act as a stable control knob for dataset regressions?

Sweep:

```text
scifact=1,nfcorpus=2,fiqa=1
scifact=1,nfcorpus=3,fiqa=1
scifact=1,nfcorpus=4,fiqa=1
scifact=2,nfcorpus=3,fiqa=1
scifact=1,nfcorpus=3,fiqa=2
scifact=2,nfcorpus=4,fiqa=1
```

Gate:

- per-dataset nDCG deltas form a Pareto improvement or acceptable macro gain
- no source schedule is promoted from pairwise metrics alone

### Lane D: Bigger Compact Models

Question: how much of the current quality ceiling is architecture size?

New-start configs:

| Name | Max seq | Dim | Hidden | Repeats | Vocab | Use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `embed-s` | 256 | 128 | 256 | 2 | 32768 | current control |
| `embed-m` | 512 | 192 | 384 | 3 | 32768 | first quality/VRAM probe |
| `embed-l` | 1024 | 256 | 512 | 4 | 32768 | C24 quality probe |
| `embed-xl-smoke` | 1024 | 384 | 768 | 4 | 32768 | throughput/VRAM smoke |

Gate:

- larger model must improve retrieval, not just pairwise AUC
- docs/s and train pairs/s remain inside C24 target budget
- sealed artifact remains practical for local serving

### Lane E: TurboQuant And Weight Precision

Question: where is the quality/throughput knee for local serving?

Sweep:

| Var | Values |
| --- | --- |
| `MANTA_WEIGHT_BITS` | `4`, `6`, `8` |
| train/eval dtype | current f16 output plus future q-vector variants |
| package mode | trainable `.mll` vs sealed `.mll` |

Gate:

- quality regression is measured against dense/f16 candidate
- package size and encode throughput improve enough to justify regression

## Code Lanes To Unlock

These are likely necessary for true best-in-class local performance.

### Lane F: External Teacher Import

Add a tool that imports query/document/candidate teacher scores from Qwen3, BGE-M3, Jina, Voyage/OpenAI/Gemini APIs, or local TEI servers into the existing `teacher_scores` JSONL field.

Required outputs:

- normalized scores over `positive + negatives`
- teacher model id, revision, prompt/instruction, dimensionality, and score scale in a sidecar manifest
- deterministic fallback when the teacher cannot score an item

### Lane G: Synthetic Query And Reasoning Data

Add a data builder for:

- generated queries from documents
- hard paraphrases
- adversarial near-miss negatives
- multi-hop or dispersed-evidence queries
- domain-specific questions for CorkScrewDB and code/document retrieval

Gate synthetic data by retrieval scoreboards, not by generated-data volume.

### Lane H: Matryoshka Loss

Add a truncation-aware loss over output prefixes, for example:

```text
full dim: 128 or 256
prefix dims: 32, 64, 96, 128, 192, 256
loss = full_loss + sum(prefix_loss[d] * weight[d])
```

This makes Manta vectors cheaper to store, gives CorkScrewDB multiple latency/quality modes, and aligns with SOTA embedding compression practice.

### Lane I: Sparse Lexical Head

Add an optional lexical-weight output trained from BM25/SPLADE-style teachers.

Minimum version:

- per-token vocabulary logits or hashed lexical bins
- sparse regularization
- teacher scores from BM25 and/or SPLADE-like external teacher
- hybrid retrieval scoreboard: dense only, sparse only, dense+sparse

### Lane J: Multi-Vector Late Interaction

Add span/token vector outputs and a late-interaction scorer.

Minimum version:

- document span vectors
- query token/span vectors
- MaxSim or pruned MaxSim scorer
- scoreboard for first-stage dense retrieval plus late-interaction reranking

This is the direct Manta analogue to BGE-M3 multi-vector and ColBERT-style retrieval.

### Lane K: Reranker Distillation

Use the existing rerank/select runtime surface to train a compact reranker from Qwen3/BGE reranker outputs.

Gate:

- candidate reranker improves top-10/top-100 ordering from Manta dense retrieval
- reranker latency remains suitable for local desktop serving

### Lane L: Sparse Long-Context Encoder

Move beyond chunking by integrating routed TurboQuant sparse attention into the embedding encoder.

Milestones:

- dense vs exact sparse vs routed sparse encoder score parity on small contexts
- router trained from high-budget attention/block teacher labels
- sparse backward or detached-router training smoke
- 32k training smoke and 128k inference demo on C24

## First Execution Queue

Priority order:

1. Run Lane B with `9000` mined examples, `5` mined negatives, `candidate_top_k=400`, and candidate hard negatives `2`; the first Lane A source-weight pressure tests around the temperature-`1.5` best are now exhausted.
2. Start `embed-m` from scratch with the current best training recipe, then compare full retrieval and throughput.
3. Implement Lane F so public teachers can write into the same `teacher_scores` path.
4. Implement Lane H before increasing vector dimension aggressively.
5. Implement Lane I and Lane J after single-vector dense gains flatten.
6. Integrate Lane L once short retrieval is stable enough to justify long-context work.

## Promotion Discipline

No SOTA claim without:

- full retrieval scoreboards, not pairwise-only metrics
- per-dataset nDCG and recall floors
- latency, package size, and VRAM measurements
- reproducible run commands and manifests
- explicit teacher provenance
- a named baseline set that includes the strongest local public models we can run
