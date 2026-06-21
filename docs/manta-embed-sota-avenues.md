# Eos Embedder SOTA Avenue Map

This is the working map for trying everything that can plausibly move `manta-embed-v1` toward a best-in-class local embedder. The objective is not one clever loss. SOTA embedding systems combine foundation-model backbones, staged data, synthetic data, hard negatives, distillation, task routing, multi-output retrieval modes, and compression-aware serving. Eos should turn each of those into a measured lane.

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

## Current Dense Candidate

The previous sealed in-repo anchor was:

```text
runs/manta-embed-v1-teacher-hybrid-w005-tw020-tt150-nf3train-lr10-20260507T053803Z/candidate/manta-embed-v1.sealed.mll
```

The current dense local candidate is:

```text
runs/eos-frontier-teacher-sentinel-balance-sweep-v1-s40-20260620T154736Z/eos-embed-v1.sealed.mll
```

Status: s40 frontier-teacher sentinel-balanced release package, sealed and dense short-set gate verified against the previous nf005 default. The release sealed artifact SHA256 is `f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a`; release package SHA256 is `188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf`; tokenizer SHA256 is `64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5`. Package and sealed inspection report `package verify: OK`, sealed inspection reports `package: embedded sealed MLL`, and final plus hard eval logs record `optimizer_updates=0`. The training data was `frontier-teacher-nfcorpus-sentinel-balanced-40.train.jsonl` with 66 filtered frontier-teacher rows plus 40 audited non-test NFCorpus sentinel rows, teacher source weights `frontier-teacher-filtered=1,nfcorpus=1`, LR `0.00000005`, and quality target `pairwise`. The predecessor nf005 package at `runs/current-release-qwen3-nf005-continuation-20260616T224102Z/candidate/`, the targeted-v3 package at `runs/eos-embed-v1-targeted-v3-release-package-20260616T000000Z/`, and the legacy source artifact `runs/eos-embed-v1-targeted-neargate-v3-low-lr-restorebest-20260614T000000Z/targeted-v3-lr000002-restorebest-manta/manta-embed-v1.sealed.mll`, SHA256 `ea776e2fca7fdade7ee05396b2ee8980e220899e2515853c83a4bca34cf87242`, remain comparison provenance only.

The previous strict sealed anchor is:

```text
runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll
```

That June 10 deephard-full artifact is sealed, inspected, and full-scoreboard verified. Its SHA256 is `a7461b47784ea7434cf6048f33f6c281ef19887cfa9d0c699b6f2fba079f2b67`; the sealed scoreboard is under `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/`, and the sealed-vs-train-package comparison recorded zero nonzero quality or count deltas.

Dense comparison against the previous nf005 default:

| Dataset | s40 nDCG@10 | s40 recall@100 | Delta vs nf005 nDCG@10 | Delta vs nf005 recall@100 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.5645379155 | 0.7964444444 | +0.0000000000 | +0.0000000000 |
| NFCorpus | 0.205571 | 0.242059 | +0.000213 | +0.000011 |
| FiQA | 0.121261 | 0.351678 | +0.000151 | +0.000000 |

2026-06-20 hybrid ranking-policy evidence: the `fiqa24-nf48` candidate artifact at `runs/eos-s40-longembed-balanced-anchor-sweep-v1-20260620T195017Z/candidates/fiqa24-nf48/candidate/eos-embed-v1.sealed.mll` passed command-level short retrieval gates with `eval-retrieval-hybrid --method minmax_blend --alpha 0.5 --top-k 100`. This is lexical+dense ranking-policy evidence only; it is not dense model promotion, does not replace the s40 dense default, and does not change shipped assets.

| Dataset | Hybrid nDCG@10 | Hybrid recall@100 | Gate |
| --- | ---: | ---: | --- |
| SciFact | 0.717644867485 | 0.932888888889 | pass |
| NFCorpus | 0.311158654714 | 0.290278895553 | pass |
| FiQA | 0.219415915378 | 0.500980325402 | pass |

Evidence lives in `runs/eos-s40-command-hybrid-validation-v1-20260620T224155Z/command-hybrid-validation.json` and `.tiller/scratch/codex/eos-s40-command-hybrid-validation-v1-report.md`. NFCorpus command-level nDCG@10 is lower than the prior offline simulation by `0.002670461567`, but recall matches and the command gate remains comfortably above the s40 floor.

The s40 package is the dense release-candidate line. Its promoted compact policy is q4/fp16/rerank-overfetch=200, method `turboquant_ip_b4_overfetch200_fp16_rerank`, total compression `1.5900621118x`. It passed strict seeded compact non-regression against the nf005 q4/fp16/o200 anchor: NFCorpus nDCG@10 `+0.000052`, recall@100 `+0.000460`; FiQA nDCG@10 `+0.000038`, recall@100 `+0.000386`; macro nDCG@10 `+0.000030`, recall@100 `+0.000282`. The capped serving smoke in `runs/eos-default-embedder-serving-smoke-20260620T161633Z/` selected q4/fp16/o200 with SciFact nDCG@10 `0.7846268033`, recall@100 `0.95`, total compression `1.5900621118x`, and p95 `0.984950ms`. Current CorkScrewDB local flat packed-parent API evidence is the s40 main-checkout run `runs/eos-s40-current-default-corkscrewdb-budget-quality-packed-q4q8-main-20260620T165050Z/`, using vector cache `runs/eos-vector-caches/eos-s40-current-default-scifact-child-w128-o32-128d/` and CorkScrewDB commit `511f5d24408d9aeba21941954d29cca3569875da`: q4 `packed_parent_multivectors` with `metadata=none`, ordinal child IDs, `quantized_only` storage, and flat index measured `5,183` parents, `12,468` children, `128d`, nDCG@10 `0.452971`, recall@100 `0.755222`, DB directory multiple `0.041675x`, vector payload multiple `0.013312x`, p95 `13.434893ms`, planner fit `180`, target fit `true`, and target storage multiple `0.554545x`. q8 is diagnostic only: nDCG@10 `0.472424`, recall@100 `0.776889`, DB directory multiple `0.066733x`, vector payload multiple `0.025841x`, p95 `21.874919ms`, planner fit `93`, target fit `false`, and target storage multiple `1.074026x`. This evidence covers the local flat API only; it is not remote mode, federation, HNSW, hosted parity, or a service SLO. q8 misses target fit and the DB directory gate.

Historical rejected probes from the prior sealed-anchor lane:

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
| `teacher_loss_weight=0.20`, `teacher_temperature=1.5`, `source_weights=scifact=1,nfcorpus=3,fiqa=2`, LR `0.000010` | 0.147516 | Baseline gate pass, but extra FiQA sampling missed the current best by `0.000628` macro and did not improve FiQA |
| Full BM25-scored blend, `teacher_loss_weight=0.05`, `teacher_temperature=10`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000010` | 0.147151 | Full teacher-score coverage improved pairwise AUC, but failed the stale-baseline NFCorpus floor and missed the current best by `0.000993` macro |
| Full BM25-scored blend, source temperatures `scifact/nfcorpus/fiqa=10` and `*:model=1.5`, `teacher_loss_weight=0.20`, LR `0.000010` | 0.145395 | Source-temperature plumbing works, but stronger full-score distillation regressed SciFact and NFCorpus; FiQA rose to `0.029619` |
| Full BM25-scored blend, source temperatures `scifact/nfcorpus/fiqa=10` and `*:model=1.5`, `teacher_loss_weight=0.05`, LR `0.000010` | 0.146459 | Macro beat the stale baseline, but NFCorpus nDCG@10 delta `-0.001638` failed the floor and current-best macro was missed by `0.001684` |
| Full BM25-scored blend, `teacher_score_normalization=source_zscore`, `teacher_loss_weight=0.20`, `teacher_temperature=1.5`, LR `0.000010` | 0.144793 | Source normalization improved FiQA recall@100, but strong full-score teacher pressure regressed SciFact and NFCorpus |
| Full BM25-scored blend, `teacher_score_normalization=source_zscore`, `teacher_loss_weight=0.05`, `teacher_temperature=1.5`, LR `0.000010` | 0.147714 | Baseline gate pass and strong pairwise AUC, but FiQA nDCG kept macro `0.000429` below the current anchor |
| Full BM25-scored blend, `teacher_score_normalization=source_zscore`, `teacher_loss_weight=0.05`, `teacher_temperature=1.5`, `source_weights=scifact=1,nfcorpus=3,fiqa=2`, LR `0.000010` | 0.147368 | Baseline gate pass, but the FiQA-weighted sampler traded too much SciFact for smaller FiQA/NFCorpus recovery and missed the current anchor by `0.000775` macro |
| Full BM25-scored blend, `teacher_score_normalization=example_zscore`, `teacher_loss_weight=0.05`, `teacher_temperature=1.5`, LR `0.000010` | 0.148094 | Strong stale-baseline gate pass and near-anchor macro; NFCorpus/FiQA rose, but SciFact regression missed the current anchor by `0.000049` macro |
| Full BM25-scored blend, `teacher_score_normalization=example_zscore`, `teacher_loss_weight=0.05`, `teacher_temperature=1.5`, `source_weights=scifact=2,nfcorpus=3,fiqa=1`, LR `0.000010` | - | Rejected before full scoreboard: validation/hard AUC fell to `0.817674`/`0.810527`, and SciFact dropped to `0.326679` |
| Lane B deep mine, `9000` requested examples, `5` mined negatives, `candidate_top_k=400`, `hard_negatives_per_query=2` | 0.143866 | Promotion gate failed; NFCorpus rose slightly, but SciFact and FiQA regressed hard |
| Lane B deep mine reuse, `hard_negatives_per_query=1`, `source_weights=scifact=1,nfcorpus=3,fiqa=1` | 0.145870 | NFCorpus high-water mark, but SciFact and FiQA still fail current-best gate |
| Lane B deep mine reuse, `hard_negatives_per_query=1`, `source_weights=scifact=1,nfcorpus=1,fiqa=1` | 0.144915 | Balanced source sampling reduced NFCorpus gains and did not recover SciFact/FiQA |
| Lane B deep mine reuse, `hard_negatives_per_query=1`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000005` | 0.145809 | Smaller LR recovered FiQA versus LR10, but SciFact still failed the gate |
| Lane B deep mine reuse, `hard_negatives_per_query=1`, `source_weights=scifact=1,nfcorpus=3,fiqa=1`, LR `0.000005`, `grouped_loss_weight=0.025` | 0.146429 | Best Lane B balance, but still below current best and fails SciFact/FiQA floors |
| `embed-m` cached16k, max sequence `512`, dim `192`, hidden `384`, repeats `3`, w0.05 tw0.20 tt1.50 HN1 LR `0.000010` | 0.078073 | Mechanically trains and seals, but random-start fine-tune LR collapses retrieval |
| `embed-m` cached16k scratch `infonce`, LR `0.002`, HN1, pairwise-only | - | Rejected before retrieval: validation AUC `0.495137`, hard AUC `0.498731` |

## Ready-To-Run Lanes

These require no new model code.

### Lane A: Teacher Loss Shape

Question: is `teacher_loss_weight=0.20` the local optimum, or just the first useful point?

Sweep:

| Var | Values |
| --- | --- |
| `EOS_TEACHER_LOSS_WEIGHT` | `0.05`, `0.10`, `0.20`, `0.35`, `0.50` |
| `EOS_TEACHER_TEMPERATURE` | `0.5`, `0.75`, `1.0`, `1.5`, `2.0` |
| `EOS_LR` | `0.000005`, `0.000008`, `0.000010`, `0.0000125` |
| `EOS_HARD_NEGATIVE_SOURCE_WEIGHTS` | `scifact=1,nfcorpus=3,fiqa=1`, `scifact=1,nfcorpus=4,fiqa=1`, `scifact=2,nfcorpus=3,fiqa=1` |

Gate:

- candidate macro nDCG@10 beats the previous sealed anchor `0.148144`, the June 10 strict anchor `0.265891`, and the accepted targeted-v3 dense candidate once that scoreboard is the active comparison
- no dataset violates nDCG or recall floors
- pairwise AUC does not fall below `0.818`

### Lane B: Mining Depth And Negative Budget

Question: are we under-sampling the teacher candidate set?

Sweep:

| Var | Values |
| --- | --- |
| `EOS_ALIGN_MODEL_HARD_MAX_EXAMPLES` | `6000`, `9000`, `12000` |
| `EOS_ALIGN_MODEL_HARD_NEGATIVES` | `3`, `5`, `8` |
| `EOS_ALIGN_MODEL_HARD_CANDIDATE_TOP_K` | `100`, `200`, `400`, `800` |
| `EOS_ALIGN_CANDIDATE_HARD_NEGATIVES` | `1`, `2`, `3` |

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

Status: local source-weight reshuffling around the temperature-`1.5` teacher recipe has not beaten the current anchor. Extra SciFact pressure, extra NFCorpus pressure, and extra FiQA pressure each passed or nearly passed stale-baseline gates in some cases, but all missed the current-best macro or dataset floors. Move source scheduling back behind new signal acquisition: deeper-but-balanced mining, imported external teacher scores, synthetic query data, or larger-model bootstrapping.

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

Status:

- The true `32768`-vocab `embed-m` target initialized but spent more than fifteen minutes CPU-bound in tokenizer training before any optimizer step. Treat full-vocab `embed-m` as blocked on cached tokenizer artifacts or tokenizer trainer improvements.
- The cached-tokenizer `embed-m` shape (`16384` vocab, max sequence `512`, dim `192`, hidden `384`, repeats `3`) trains and seals on the desktop GPU at batch `64`, but the current-best fine-tune recipe is invalid from random initialization: validation/hard AUC `0.595854` / `0.598887`, macro nDCG@10 `0.078073`, and `1460.78` train pairs/s.
- A scratch `infonce` LR `0.002` pass also failed as a bootstrap: validation/hard AUC `0.495137` / `0.498731` with `1259.54` train pairs/s. The next `embed-m` attempt should use staged pretraining or dimension-compatible weight expansion, then apply the teacher-distilled recipe as a fine-tune.

2026-06-15 `embed-m` frontier checkpoint: the half-frontier triple-SciFact guard run is the current local dense `embed-m` frontier, but it is evidence only. It is not the promoted default model, does not replace the CorkScrewDB/default `eos-embed-v1` q4 + fp16 sidecar rerank overfetch-250 path, and should be treated as an `embed-m` validation candidate rather than default alias promotion. Direct retrieval remains the gate; pairwise AUC is not sufficient.

| Candidate | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 | Macro nDCG@10 | Macro recall@100 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| balanced Stage B `embed-m` baseline | 0.365649 | 0.152246 | 0.040103 | 0.185999 | 0.331780 | stronger staged baseline, still below anchor |
| protective replay `embed-m` continuation | 0.365697 | 0.152673 | 0.040820 | 0.186397 | 0.331876 | prior local `embed-m` benchmark |
| prior best local `embed-m` LR `0.0000025` | 0.365759 | 0.152676 | 0.040834 | 0.186423 | 0.331864 | comparison point for the dense local gate |
| half-frontier triple-SciFact guard `embed-m` | 0.366213 | 0.152950 | 0.040485 | 0.186549 | 0.333135 | current local dense `embed-m` frontier; passes previous-best macro and SciFact floor |
| half-frontier compact q8/fp16 overfetch-200 | 0.366273 | 0.152950 | 0.040485 | 0.186569 | 0.333135 | selected compact profile for this artifact only; `1.324138x` total compression |
| June 10 strict anchor | 0.482406 | 0.197733 | 0.117533 | 0.265891 | 0.452844 | previous strict dense anchor |
| targeted-v3 dense candidate | 0.562322 | 0.204117 | 0.120294 | 0.295578 | 0.462973 | previous default |
| nf005 dense candidate | 0.564538 | 0.205358 | 0.121109 | 0.297002 | 0.463390 | predecessor default |
| s40 dense candidate | 0.564538 | 0.205571 | 0.121261 | 0.297123 | 0.463394 | current promoted default |

The half-frontier triple-SciFact guard run is `runs/eos-embed-m-half-frontier-triple-scifact-guard-20260615T000000Z/stage-c-half-frontier-triple-scifact-guard-lr25e-7-hn3-b16/`, sealed SHA256 `58b5b80a71520342062c6e6b7062b35ff95a425cccf9a683d23608192e2ac876`. It starts from the balanced Stage B baseline and trains one LR `0.0000025`, HN3, no-teacher continuation on `240` rows: `48` FiQA dev-frontier rows, `48` NFCorpus dev-frontier rows, and `144` SciFact protective replay rows. It passes the dense local `embed-m` gate against prior best macro nDCG `0.186423` and SciFact floor `0.365459`, but remains far below the s40 dense candidate macro nDCG `0.297123`.

The selected compact profile for this exact artifact is q8/fp16 rerank overfetch-200, method `turboquant_ip_b8_overfetch200_fp16_rerank`, with total compression `1.324138x`. q4/fp16 overfetch `300`, `400`, and `500` preserved nDCG but missed FiQA recall by `-0.000257`, so q4 is not selected for this `embed-m` artifact despite better compression `1.586777x`.

The prior protective replay continuation is `runs/eos-embed-m-fiqa-dev-toprank-protective-replay-probe-20260615T000000Z/`. It starts from the balanced Stage B baseline and trains one LR `0.000002`, HN3, no-teacher continuation on a 96-row blend: `48` FiQA dev top-rank rows, `24` SciFact replay rows, and `24` NFCorpus replay rows.

Negative findings for this branch: FiQA source oversampling regressed; the test-selected microrepair was diagnostic only; dev-heldout top-rank selection generalized directionally but was weaker without protective replay; and the larger scale96 blend was worse than the 96-row protective blend on macro and FiQA nDCG. Future `embed-m` work should treat protective replay as the local comparison point, not as a promotion candidate.

### Lane E: TurboQuant And Weight Precision

Question: where is the quality/throughput knee for local serving?

Sweep:

| Var | Values |
| --- | --- |
| `EOS_WEIGHT_BITS` | `4`, `6`, `8` |
| train/eval dtype | current f16 output plus future q-vector variants |
| package mode | trainable `.mll` vs sealed `.mll` |

Gate:

- quality regression is measured against dense/f16 candidate
- package size and encode throughput improve enough to justify regression

Storage-accounting harness:

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

Use this for the CorkScrewDB direct multi-vector lane: many quantized child vectors under one parent object for windows, spans, or time-series slices. It measures byte budgets only. Omitting `--baseline-dim` keeps same-dim accounting (`baseline_dim=dim`), where a 128d q2 child vector payload is 36 bytes and only 14 payload-only children fit inside one 128d fp32 vector budget; with 32 bytes of packed parent-object overhead, only 14 q2, 7 q4, or 3 q8 children fit. Passing `--baseline-dim 3072` tests compact 128d children against a larger dense baseline: one 3072d fp32 vector is 12,288 payload bytes, so 341 q2 payload-only children fit and 128 children cost about `0.375x` of that one-vector budget before metadata. With packed-parent accounting (`--packed-object-overhead-bytes 32`), one 3072d dense parent-vector storage budget fits 341 q2, 180 q4, or 93 q8 128d children. The executable frontier smoke can now measure multiple parent baselines in one run with `EOS_MV_BUDGET_SMOKE_BASELINE_DIMS=128,384,768,1024,1536,3072`; the comma-list takes precedence over the backward-compatible singular `EOS_MV_BUDGET_SMOKE_BASELINE_DIM`. Current per-child-entry accounting uses `--vector-overhead-bytes`; packed-parent target accounting uses `--packed-object-overhead-bytes` to pay object overhead once per parent while keeping children as compact TurboQuant payloads. TSV/JSON include current storage fields and packed fields such as `packed_quantized_storage_bytes`, `packed_total_quantized_bytes`, and `packed_vectors_that_fit_in_one_dense_vector`. The scaled q4 time-series smoke has now measured local flat packed parent-object persistence with omitted packed metadata and ordinal child IDs: `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-packed-minimal-20260616T000000Z/` stored 100 parents and 10,000 child windows with DB bytes `0.368244x` of the comparable separate-child run while preserving the same vector payload accounting. The corrected scaled q2-341 compact v5 packed time-series evidence is unified in wrapper run `runs/eos-corkscrewdb-timeseries-window-q2-341-compact-v5-unified-20260616T000000Z/`, which generated the child/query/qrels inputs, planner evidence, and measured persisted DB bytes against CorkScrewDB commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4`. The shape stored `100` parents and `34,100` child windows with `341` windows per parent, q2 `128d`, `packed_parent_multivectors`, `packed_metadata_mode=none`, `packed_child_id_mode=ordinal`, `quantized_vector_bytes=36`, `quantized_child_bytes=1,227,600`, vector payload multiple `0.9990234375x`, packed planner bytes `12,308`, and packed planner multiple `0.999025974025974x`; measured DB directory bytes were `1,237,818`, DB directory multiple `1.0073388671875x`, with nDCG@10 `0.4493940305106442`, recall@100 `1.000000`, and p95 `1.418733ms`. Treat these time-series rows as synthetic text-rendered local API evidence, not production quality or a trained numeric time-series encoder result. With CorkScrewDB compact snapshot v5 ordinal encoding, the persisted DB directory is approximately one dense parent-vector budget for this strict shape; without that compact snapshot path, or for richer child records, keep DB directory cost separate from vector payload and planner accounting. The first cache-only quality follow-up is `eval-retrieval-multivector-turboquant`, which reads child-vector JSONL with `parent_id`/`child_id`, scores every child, rolls up by max child score per parent, and compares dense child aggregation against direct q2/q4/q8 TurboQuant child aggregation on BEIR qrels. Export BEIR child caches with `eos export-retrieval-vectors --output-dim 128 --document-chunk-words 128 --document-chunk-overlap 32 --document-chunk-min-words 16`; chunked export writes `child-doc-vectors.jsonl` plus the unchanged `query-vectors.jsonl`, and the manifest records both model and written dimensions. `--output-dim 128` is prefix truncation plus L2 renormalization, not trained Matryoshka; use it as a measured bridge while treating a native/trained 128d head as the stronger future path. The quality harness fails by default if the child-vector cache is missing any qrels-relevant parent; `--allow-missing-relevant` is diagnostic-only. TurboQuant rows are deterministic through `--quantizer-seed`, and metrics record the seed. It is the quality bridge between storage math and a future CorkScrewDB search harness, not a replacement for an API load/index/search smoke. Keep direct child-vector storage separate from q4/fp16 rerank sidecars, because a per-child fp16 sidecar is a quality-preserving rerank option rather than the hundred-vector storage lane.

First measured SciFact evidence for this lane used Qwen3 0.6B child chunks at `128` words, `32` overlap, and `16` minimum trailing words. The cache has `5,183` parents, `12,468` children, and `2.41` average children per parent. On `300` strict-coverage qrels queries, dense child-max scored `0.717467` nDCG@10 / `0.953333` recall@100; direct q8 scored `0.716310` / `0.953333` at `3.98x` child compression and `0.60x` of one dense-parent-vector budget. That improves over the one-vector Qwen3 SciFact dense row `0.702026` / `0.946667` and q8 row `0.702657` / `0.946667`.

Mixedbread `mixedbread-ai/mxbai-embed-large-v1` is the stronger current external SciFact child-cache baseline on this lane. The requested `datasets/eos-embed-v1/raw/scifact/scifact` path was absent, so the run used `datasets/manta-embed-v1/raw/scifact/scifact`, matching the Qwen3 child evidence. With `128` word chunks, `32` overlap, and `16` minimum trailing words, it produced `5,183` parents, `12,468` child vectors, `2.405557` average children per parent, and `300` evaluated qrels queries with strict coverage (`allow_missing_relevant=false`).

| row | child nDCG@10 | child recall@100 | compression | parent-budget multiple | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-child | 0.747175 | 0.970000 | n/a | n/a | 12.497 ms |
| q2 | 0.712790 | 0.956667 | 15.75x | 0.15x | 4.754 ms |
| q4 | 0.739489 | 0.965000 | 7.94x | 0.30x | 77.250 ms |
| q8 | 0.747799 | 0.966667 | 3.98x | 0.60x | 157.876 ms |

mxbai is higher than Qwen3 child-max on dense, q2, q4, and q8 nDCG@10 and recall@100. The q8 mxbai row beats Qwen3 q8 by `+0.031489` nDCG@10 and `+0.013334` recall@100. Keep Qwen3 as a compact leading-family baseline, but use mxbai as the stronger external SciFact child-cache quality target.

The sealed Eos/default path is now measured end-to-end for the same strict lane. `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll` exported a full Go-native SciFact child cache from `datasets/manta-embed-v1/raw/scifact/scifact` with `128` word chunks, `32` overlap, and `16` minimum trailing words: `5,183` docs, `300` queries, `12,468` children, dim `256`, CUDA backend, and `57.771s` elapsed. Strict eval used `allow_missing_relevant=false`, `339` relevant pairs, `3,740,400` scored child pairs, and quantizer seed `5581486560434873699`.

| row | Eos child nDCG@10 | Eos child recall@100 | compression | parent-budget multiple | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-child | 0.462489 | 0.778111 | n/a | n/a | 3.129 ms |
| q2 | 0.383295 | 0.719667 | 15.06x | 0.16x | 1.159 ms |
| q4 | 0.449435 | 0.773111 | 7.76x | 0.31x | 17.819 ms |
| q8 | 0.461862 | 0.774778 | 3.94x | 0.61x | 39.192 ms |

This proves the sealed `.mll` -> Go-native child vector cache -> strict TurboQuant multivector eval path, but it also shows the current sealed Eos anchor is materially below full mxbai and Qwen3 child-cache evidence on SciFact. q8 preserves Eos dense-child quality closely, and q4 is near but drops more; the main deficit is model quality, not TurboQuant storage or scoring.

The strategic TurboQuant lane is multi-vector object design, not only compressing one vector. Direct compact child vectors can make windows, spans, time-series slices, and other child schemas practical per parent object. Same-dimension child vectors do not fit hundreds of children inside one same-dimension fp32 parent-vector budget; the precise parent-budget claim is that packed 128d TurboQuant children fit in single-digit to low-tens counts against a 128d dense parent, but fit tens to hundreds of children when compared against 1024 to 3072 dimensional fp32 dense parent baselines with `--baseline-dim` or when the product explicitly budgets multiple dense-parent equivalents.

## Code Lanes To Unlock

These are likely necessary for true best-in-class local performance.

### Lane F: External Teacher Import

Add a tool that imports query/document/candidate teacher scores from Qwen3, BGE-M3, Jina, Voyage/OpenAI/Gemini APIs, or local TEI servers into the existing `teacher_scores` JSONL field.

Status: the generic landing zone is implemented as `eos import-teacher-scores`. It accepts either one score vector per hard-negative example:

```json
{"source":"scifact","query":"...","scores":[0.91,0.22,0.13]}
```

or one row per query/candidate pair:

```json
{"query":"...","candidate":"document text","score":0.91}
```

The command writes validated text hard-negative JSONL plus a `manta.teacher_score_import.v1` provenance manifest. External scorers should now target this sidecar format first, then let the existing tokenizer and `teacher_loss_weight` path carry scores into training.

Use `eos export-teacher-score-requests <hard-negatives.jsonl> <requests.jsonl>` to generate one external-teacher request per query/candidate pair:

```json
{"source":"scifact","query":"...","candidate":"document text","role":"negative","example_index":0,"candidate_index":1}
```

An external scorer can add a `score` field to those rows and feed them directly into `eos import-teacher-scores`. The export command writes a `manta.teacher_score_requests.v1` manifest and supports `--missing-only` for partially scored files.

Local Eos teachers can bypass the sidecar step with `eos score-teacher-hard-negatives <teacher.mll> <hard-negatives.jsonl> <output.jsonl>`. That command embeds each query and its `positive + negatives`, writes cosine-style `teacher_scores`, and emits a `manta.teacher_hard_negative_score.v1` manifest with artifact, backend, batch size, and teacher provenance.

Before spending a training run on a new teacher, run `eos audit-teacher-scores <hard-negatives.jsonl> <summary.json>`. It reports score coverage, positive top-1 rate, mean positive rank, positive-vs-best-negative margin, and teacher-distribution entropy overall and by source, giving a cheap reject path for teachers that misorder positives or produce unusably flat/sharp targets.

For Qwen3/mxbai-style external teachers, follow the audit with `eos filter-teacher-scores <scored-hard-negatives.jsonl> <filtered.jsonl> <summary.json>`. The default filter keeps each hard-negative example but clears `teacher_scores` unless the teacher ranks the labeled positive top-1 with non-negative margin; `--min-margin` can require a larger safety gap, and `--max-normalized-entropy` can reject overly flat distributions. Train guarded candidates from the filtered JSONL so base hard-negative InfoNCE still uses every example while teacher loss applies only where the teacher agrees with the label.

For cached external embedders, `scripts/score_teacher_with_vector_cache.py` bridges BEIR-style `corpus.jsonl`/`queries.jsonl` plus document/query vector JSONL into complete hard-negative `teacher_scores`. The repeatable plumbing smoke is:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_vector_cache_teacher_scores.fw
```

It builds a tiny deterministic BEIR fixture, scores hard negatives through the vector-cache bridge, runs `go run ./cmd/eos audit-teacher-scores`, and gates on full coverage, zero missing examples, positive top-1 rate `1.0`, and positive mean margin `> 0` before writing `summary.tsv` and `manifest.json`. To adapt the smoke to Qwen3 or mxbai caches, keep the same file contracts but point `--dataset-dir`, `--doc-vectors`, and `--query-vectors` at the real cache, preserve exact hard-negative query/candidate text so the bridge can map back to BEIR IDs, set `--model-id` to the external model, and keep the audit gate before launching `train-embed`.

SciFact vector-cache teacher-signal audit: `runs/eos-vector-cache-teacher-scifact-audit-20260616T000000Z/` scored the full `919`-row SciFact hard-negative file from existing mxbai-large and Qwen3-0.6B BEIR caches with zero missing examples. mxbai scored `1838/1838` candidates with positive top-1 rate `0.792165`, positive mean rank `1.207835`, positive mean margin `0.076113`, and mean normalized entropy `0.997426`. Qwen3 scored `1838/1838` candidates with positive top-1 rate `0.761697`, positive mean rank `1.238303`, positive mean margin `0.105088`, and mean normalized entropy `0.994165`. This is evidence that both external caches produce complete, importable SciFact teacher scores; it is a teacher-signal audit only, not proof that either teacher will improve a training run.

Short-set agreement teacher prep is ready at `runs/eos-shortset-agreement-teacher-prep-v1-20260621T000000Z/`. It scores SciFact, NFCorpus, and FiQA hard-negative train files from local Qwen3 0.6B and mxbai-large vector caches, then writes `shortset.qwen3-mxbai.agreement-filtered.train-hard-negatives.jsonl` with all `4919` base examples preserved and averaged `teacher_scores` only where both teachers rank the labeled positive top-1 with non-negative margin. Agreement coverage is `667/919` SciFact (`0.725789`), `373/2000` NFCorpus (`0.186500`), and `1114/2000` FiQA (`0.557000`), for `2154/4919` overall (`0.437886`). The combined audit reports `scored=2154`, `missing=2765`, positive top-1 rate `1.000000`, and mean positive margin `0.130634`; a consistency `eos filter-teacher-scores` pass kept all `2154` scored rows and cleared `0`. FiQA scoring used explicit exportable-text handling for the raw BEIR corpus empty-text rows; do not describe it as raw-row-complete or judged-coverage-complete evidence.

Agreement-teacher follow-up status: closed without promotion. The source-less guarded candidate improved NFCorpus nDCG@10 by `+0.000075309103`, NFCorpus recall@100 by `+0.001536789246`, and FiQA nDCG@10 by `+0.000027672042`, while failing only SciFact nDCG@10 by `-0.000732471281` versus the s40 dense anchor. The source-labeled hard-negative source-weight attempt with `scifact=2,nfcorpus=1,fiqa=1` failed SciFact and NFCorpus checks; the reusable source-labeling tooling landed as commit `f4b99269aee3c2ab8d434c0e4633550848178a95`. Teacher-source damping with neutral hard-negative sampling restored the source-less NFCorpus and FiQA gains but still failed the same SciFact nDCG delta. A tiny train/dev-safe SciFact sentinel replay found only `38` non-overlapping rows; it did not move SciFact and introduced an NFCorpus recall miss.

Decision: do not promote any of these candidates, and do not sweep source weights, teacher-source weights, or replay sizes around this exact agreement file. The next model-quality run must change the signal family: for example, a stronger non-overlapping SciFact-compatible teacher or synthetic signal, the trained Matryoshka/compact-head lane, or a larger-model bootstrap. Run a cheap audit before any guarded training.

Status: BM25 and model-hard mining can both emit `teacher_scores`, and dataset acquisition now preserves those scores when it rewrites source-tagged hard-negative JSONL. A full BM25-scored blend gave complete score coverage but rejected at macro `0.147151`; BM25 scores were on a much larger scale than model cosine scores. Source-specific teacher temperatures are implemented, including exact source, source-family, and wildcard fallback, but split-temperature runs rejected at macro `0.145395` with `teacher_loss_weight=0.20` and macro `0.146459` with `teacher_loss_weight=0.05`. Teacher-score normalization is available in `train-embed` and the alignment scripts with `source_zscore`, `family_zscore`, and `example_zscore`. `source_zscore` at `teacher_loss_weight=0.20` rejected at macro `0.144793`, while reducing teacher pressure to `0.05` passed the stale-baseline gate at macro `0.147714` with SciFact `0.331279`, NFCorpus inside the floor, validation AUC `0.823381`, and hard AUC `0.814565`. It still missed the current anchor by `0.000429` macro because FiQA fell to `0.028101`. Adding FiQA sampler pressure (`scifact=1,nfcorpus=3,fiqa=2`) lifted FiQA to `0.028354` and NFCorpus to `0.083956`, but SciFact fell to `0.329793`, macro slipped to `0.147368`, and the current anchor miss widened to `0.000775`. `example_zscore` is the best normalized branch so far: SciFact `0.329417`, NFCorpus `0.085742`, FiQA `0.029123`, macro `0.148094`, validation AUC `0.823282`, and hard AUC `0.815203`. It passed the stale-baseline gate by `+0.002526` macro but missed the current anchor by only `0.000049` because SciFact fell below the current-best row. A narrow SciFact-recovery tweak (`scifact=2,nfcorpus=3,fiqa=1`) failed early: validation/hard AUC fell to `0.817674`/`0.810527`, and SciFact retrieval dropped to `0.326679`. Stop local score-normalization reshuffling here; the next improvement path should bring in stronger external/synthetic teacher signal.

Required outputs:

- normalized scores over `positive + negatives`
- request rows for every query/candidate pair that an external teacher must score
- teacher model id, revision, prompt/instruction, dimensionality, and score scale in a sidecar manifest
- deterministic fallback when the teacher cannot score an item; by default the importer fails incomplete examples, and `--allow-missing` can preserve unscored examples for smoke checks

### Lane G: Synthetic Query And Reasoning Data

Add a data builder for:

- generated queries from documents
- hard paraphrases
- adversarial near-miss negatives
- multi-hop or dispersed-evidence queries
- domain-specific questions for CorkScrewDB and code/document retrieval

Gate synthetic data by retrieval scoreboards, not by generated-data volume.

Status: local process-doc pretraining now has a first landing zone in `scripts/build_pretrain_pairs.fw`. Set `EOS_PROCESS_PRETRAIN=1` to add chunks from `AGENTS.md`, `.codex/agents/*.toml`, and `.codex/skills/**/SKILL.md` to the hard-negative pretraining JSONL; set `EOS_PROCESS_PRETRAIN_INCLUDE_DOCS=1` to include `docs/**/*.md`. The output uses the existing text hard-negative fields (`query`, `positive`, `negatives`, `source`, `group_id`), so it can be blended into `processed/pretrain-pairs.jsonl` before the shipping pipeline or used directly with `EOS_HARD_NEGATIVE_TRAIN=1` for a candidate smoke. A bounded process-corpus smoke generated `12` process rows, reached hard-negative training with `optimizer_updates=42`, and completed a separate eval-only pass with `optimizer_updates=0`; this proves the plumbing path only, not model quality. This is not generated query data and has no quality claim yet; it is the local-process corpus lane needed before synthetic Tiller/Codex questions or external teacher scoring are layered on.

### Lane H: Matryoshka Loss

Add a truncation-aware loss over output prefixes, for example:

```text
full dim: 128 or 256
prefix dims: 32, 64, 96, 128, 192, 256
loss = full_loss + sum(prefix_loss[d] * weight[d])
```

This makes Eos vectors cheaper to store, gives CorkScrewDB multiple latency/quality modes, and aligns with SOTA embedding compression practice.

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

This is the direct Eos analogue to BGE-M3 multi-vector and ColBERT-style retrieval.

### Lane K: Reranker Distillation

Use the existing rerank/select runtime surface to train a compact reranker from Qwen3/BGE reranker outputs.

Gate:

- candidate reranker improves top-10/top-100 ordering from Eos dense retrieval
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

1. Change the model-quality signal family before the next guarded candidate: audit a stronger non-overlapping SciFact-compatible teacher or synthetic signal, the trained Matryoshka/compact-head lane, or a larger-model bootstrap before spending a guarded training run.
2. Add an `embed-m` bootstrap layer before more capacity runs: cache or accelerate the `32768` tokenizer path, then try dimension-compatible weight expansion or staged pretraining before teacher fine-tuning.
3. Implement Lane H before increasing vector dimension aggressively.
4. Implement Lane I and Lane J after single-vector dense gains flatten.
5. Integrate Lane L once short retrieval is stable enough to justify long-context work.

## Promotion Discipline

No SOTA claim without:

- full retrieval scoreboards, not pairwise-only metrics
- per-dataset nDCG and recall floors
- latency, package size, and VRAM measurements
- reproducible run commands and manifests
- explicit teacher provenance
- a named baseline set that includes the strongest local public models we can run
