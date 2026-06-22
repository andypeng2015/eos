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

2026-06-20 hybrid ranking-policy evidence began with the `fiqa24-nf48` candidate artifact at `runs/eos-s40-longembed-balanced-anchor-sweep-v1-20260620T195017Z/candidates/fiqa24-nf48/candidate/eos-embed-v1.sealed.mll`, which passed command-level short retrieval gates with `eval-retrieval-hybrid --method minmax_blend --alpha 0.5 --top-k 100`. That remains candidate-only lexical+dense ranking-policy evidence: it is not dense model promotion, does not replace the s40 dense default, and does not change shipped assets.

| Dataset | Hybrid nDCG@10 | Hybrid recall@100 | Gate |
| --- | ---: | ---: | --- |
| SciFact | 0.717644867485 | 0.932888888889 | pass |
| NFCorpus | 0.311158654714 | 0.290278895553 | pass |
| FiQA | 0.219415915378 | 0.500980325402 | pass |

Evidence lives in `runs/eos-s40-command-hybrid-validation-v1-20260620T224155Z/command-hybrid-validation.json` and `.tiller/scratch/codex/eos-s40-command-hybrid-validation-v1-report.md`. NFCorpus command-level nDCG@10 is lower than the prior offline simulation by `0.002670461567`, but recall matches and the command gate remains comfortably above the s40 floor.

Current-default hybrid policy evidence now exists for the durable default asset `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`, SHA256 `f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a`, under `runs/eos-s40-current-default-hybrid-policy-v1-20260621T103500Z/`. The command-level path uses the same `eval-retrieval-hybrid --method minmax_blend --alpha 0.5 --top-k 100` policy over the current s40 dense anchor and passes all three datasets:

| Dataset | Hybrid nDCG@10 | Delta vs s40 dense | Hybrid recall@100 | Delta vs s40 dense |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.717644867485 | +0.153106951976 | 0.932888888889 | +0.136444444444 |
| NFCorpus | 0.311170830256 | +0.105599664989 | 0.290149896585 | +0.048091200495 |
| FiQA | 0.219415915378 | +0.098154974764 | 0.500671683426 | +0.148993474804 |

Full uncapped CorkScrewDB `SearchMulti` API smokes for the same default asset, using BM25-dot sparse vectors and public weighted dense+sparse fusion, also passed:

| Dataset | Hybrid nDCG@10 | Hybrid recall@100 | overall_pass |
| --- | ---: | ---: | --- |
| SciFact | 0.724378172602 | 0.932888888889 | true |
| NFCorpus | 0.315090347196 | 0.293990495284 | true |
| FiQA | 0.223742712175 | 0.504002444743 | true |

This is local lexical+dense ranking-policy evidence for the current default asset, not dense model promotion. No default asset changed.

The s40 package is the dense release-candidate line. Its promoted compact policy is q4/fp16/rerank-overfetch=200, method `turboquant_ip_b4_overfetch200_fp16_rerank`, total compression `1.5900621118x`. It passed strict seeded compact non-regression against the nf005 q4/fp16/o200 anchor: NFCorpus nDCG@10 `+0.000052`, recall@100 `+0.000460`; FiQA nDCG@10 `+0.000038`, recall@100 `+0.000386`; macro nDCG@10 `+0.000030`, recall@100 `+0.000282`. The capped serving smoke in `runs/eos-default-embedder-serving-smoke-20260620T161633Z/` selected q4/fp16/o200 with SciFact nDCG@10 `0.7846268033`, recall@100 `0.95`, total compression `1.5900621118x`, and p95 `0.984950ms`. Current CorkScrewDB local flat packed-parent API evidence is the s40 main-checkout run `runs/eos-s40-current-default-corkscrewdb-budget-quality-packed-q4q8-main-20260620T165050Z/`, using vector cache `runs/eos-vector-caches/eos-s40-current-default-scifact-child-w128-o32-128d/` and CorkScrewDB commit `511f5d24408d9aeba21941954d29cca3569875da`: q4 `packed_parent_multivectors` with `metadata=none`, ordinal child IDs, `quantized_only` storage, and flat index measured `5,183` parents, `12,468` children, `128d`, nDCG@10 `0.452971`, recall@100 `0.755222`, DB directory multiple `0.041675x`, vector payload multiple `0.013312x`, p95 `13.434893ms`, planner fit `180`, target fit `true`, and target storage multiple `0.554545x`. q8 is diagnostic only: nDCG@10 `0.472424`, recall@100 `0.776889`, DB directory multiple `0.066733x`, vector payload multiple `0.025841x`, p95 `21.874919ms`, planner fit `93`, target fit `false`, and target storage multiple `1.074026x`. This evidence covers the local flat API only; it is not remote mode, federation, HNSW, hosted parity, or a service SLO. q8 misses target fit and the DB directory gate.

Current capped LongEmbed evidence for the same s40 line is diagnostic and not a product-quality claim. The official capped doc20/query20 external-cache blocker is cleared for `qmsum` and `2wikimqa`, and the v2 cache-mode rows preserve `quality_claim=false`. On `qmsum`, Eos direct nDCG@10 is `0.517955842`, best Eos fusion is `0.536813696`, Qwen3 128d q4 child is `0.876293470`, and mxbai 128d q4 child is `0.806946535`. On `2wikimqa`, the v2 external-only run has Eos direct `0.739153782`, best Eos fusion `0.744085103`, Qwen3 q4 `1.000000000`, and mxbai q4 `1.000000000`; the later sparse-enabled rerun's best Eos fusion is `0.739699216`, so cite the run used. Sparse encoder parent rows are present and subquadratic in the host-reference audit, but are quality-negative versus direct on both datasets: `qmsum` sparse dense/q4 `0.458166919`/`0.420963072`, and `2wikimqa` sparse dense/q4 `0.504805535`/`0.505901435`. Preserve `quality_claim=false`, capped doc20/query20 caveats, external-cache caveats, host-reference sparse-encoder caveats, and external child-count/storage caveats when using these rows. The next long-context claim needs a changed objective/model or stronger comparable real-doc Eos rows before any product-quality claim.

Router status for this lane: keep direct as the default. Five-way conservative abstention is capped diagnostic evidence only (`+0.002460468` macro vs direct, `0` regressions, `1` switch / `170` rows), five-way learned routing is blocked despite higher average lift (`+0.014338863`) because it has `17` held-out regressions and `116` non-direct switches, and current span256/64 conservative fusion remains no-regression diagnostic evidence (`+0.000180436` static macro, `0` regressions / `80` rows; action-only LODO `0` lift, feature-threshold LODO `-0.003199038` with `1` regression). Reports: `.tiller/scratch/codex/eos-router-fiveway-abstention-v1-report.md`, `.tiller/scratch/codex/eos-router-fiveway-learned-router-fit-v1-report.md`, `.tiller/scratch/codex/eos-current-span256-router-abstention-v1-report.md`.

The local s40 LongEmbed replay lane is closed. The LongEmbed teacher batch candidate was rejected on NFCorpus nDCG `-0.000197`; anchor-protected continuation shifted failures to NFCorpus recall `-0.000054` and FIQA nDCG `-0.000107`; the best balanced anchor sweep (`fiqa24-nf48`) failed only FIQA nDCG `-0.000057` while preserving NFCorpus recall `+0.000007`, but lost the useful QMSum q4/fusion signal; both FIQA targeted replay and single-row replay reproduced NFCorpus recall `-0.000047` plus FIQA nDCG `-0.000057`. The targeted diagnosis found the FIQA aggregate miss came from query `6133`, doc `7733`, rank `2 -> 3`, but replay did not repair it. Do not promote these candidates and do not spend more runs on replay/source-weight/FIQA padding around this exact protected LongEmbed batch. The next improvement should change signal family or objective: non-test FIQA-compatible teacher or synthetic signal, trained compact/Matryoshka with a movement gate, or larger-model/bootstrap work.

Movement diagnostics are now part of promotion discipline for compact-head and tiny-continuation lanes. `scripts/diagnose_eos_embedding_movement.fw` compares two Eos packages through the retrieval export surface before expensive sweeps. It showed the Matryoshka-only 128d probe was exactly pinned at 64d, 128d, and full dimensions, while the TurboQuant-prefix branch moved slightly. Require movement-positive/no-restore checks before spending another compact-head or micro-continuation sweep.

2026-06-22 Matryoshka q4 status: the s40 continuation at `runs/eos-matryoshka-128q4-no-clear-child-quality-v1-seed5581486560434873699-20260622T035557Z/` used valid explicit replacement semantics for `EOS_MATRYOSHKA_DIMS=64,128`, `EOS_MATRYOSHKA_WEIGHTS=1,1`, and `EOS_TURBOQUANT_PREFIX_OBJECTIVES=128:4=0.05` with prepared-IP scoring and no clear-prefix flag. The run was movement-positive at 64d, 128d, and full dimensions, but quality was negative: repo-docs 128d child q4 nDCG@10 regressed from `0.603952` to `0.593129`, and the dense short-set gate failed versus s40 on SciFact nDCG@10, NFCorpus nDCG@10, and NFCorpus recall@100. Do not promote this candidate and do not repeat this exact no-clear Matryoshka 64/128 + q4 prepared-IP recipe; the next compact-child attempt needs a changed signal, data mix, or objective.

2026-06-22 teacher-guided q4 child-rank status: the compact continuation at `runs/eos-teacher-guided-128q4-child-rankmargin-v1-20260622T044411Z/` started from s40, used teacher-guided 128d q4 child-rank objectives, and was mechanically valid and movement-positive at 64d, 128d, and full dimensions. Quality did not pass: the dense short gate failed versus s40 on NFCorpus recall@100 and FiQA nDCG@10, and repo-docs 128d q4 child nDCG@10 was `0.593812`, below the bridge `0.603952` and target `0.608952`. Do not promote this candidate and do not repeat this exact teacher-guided q4 child-rank recipe; the next compact-child run must change the signal family or protection strategy rather than tune this same objective.

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

2026-06-22 `embed-m` frontier checkpoint: `target64/scifact192` is the current local dense and q8/fp16/o200 compact `embed-m` frontier, but it is evidence only. It is not the promoted default model, does not replace the CorkScrewDB/default `eos-embed-v1` q4 + fp16 sidecar rerank overfetch-250 path, and should be treated as an `embed-m` validation candidate rather than default alias promotion. Direct retrieval remains the gate; pairwise AUC is not sufficient. The `target80/scifact240` follow-up is a negative boundary: FiQA improved, but the SciFact guard and macro failed, so compact was skipped.

| Candidate | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 | Macro nDCG@10 | Macro recall@100 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| balanced Stage B `embed-m` baseline | 0.365649 | 0.152246 | 0.040103 | 0.185999 | 0.331780 | stronger staged baseline, still below anchor |
| protective replay `embed-m` continuation | 0.365697 | 0.152673 | 0.040820 | 0.186397 | 0.331876 | prior local `embed-m` benchmark |
| prior best local `embed-m` LR `0.0000025` | 0.365759 | 0.152676 | 0.040834 | 0.186423 | 0.331864 | comparison point for the dense local gate |
| half-frontier triple-SciFact guard `embed-m` | 0.366213 | 0.152950 | 0.040485 | 0.186549 | 0.333135 | superseded local dense `embed-m` frontier; still provenance for the previous smallfrontier gate |
| half-frontier compact q8/fp16 overfetch-200 | 0.366273 | 0.152950 | 0.040485 | 0.186569 | 0.333135 | superseded compact profile for that artifact only; `1.324138x` total compression |
| target64/scifact192 dense `embed-m` | 0.366464 | 0.153620 | 0.040733 | 0.186939 | 0.333283 | current local dense `embed-m` frontier |
| target64/scifact192 compact q8/fp16/o200 | 0.366512 | 0.153620 | 0.040733 | 0.186955 | 0.333283 | selected compact profile for current `embed-m` frontier; `1.324138x` compression |
| target80/scifact240 dense `embed-m` | 0.365767 | 0.153233 | 0.041101 | 0.186700 | 0.333605 | rejected boundary; FiQA up but macro/SciFact guard fail; compact skipped |
| target64 agreement-first dense `embed-m` | 0.366087 | 0.153902 | 0.041620 | 0.187203 | 0.333463 | rejected; macro up but SciFact guard failed; compact skipped |
| target64 agreement + scifact224 dense `embed-m` | 0.366854 | 0.154017 | 0.041111 | 0.187327 | 0.334545 | rejected near miss; macro/SciFact up but NFCorpus recall guard failed; compact skipped |
| target64 agreement + scifact208 + nfrecall16 dense `embed-m` | 0.365897 | 0.154026 | 0.041061 | 0.186995 | 0.333259 | rejected; macro up but SciFact and NFCorpus recall guards failed; compact skipped |
| target64 agreement + scifact224 + nfrecall16 dense `embed-m` | 0.365634 | 0.153456 | 0.041103 | 0.186731 | 0.333691 | rejected; macro/SciFact failed; recall guards passed; compact skipped |
| June 10 strict anchor | 0.482406 | 0.197733 | 0.117533 | 0.265891 | 0.452844 | previous strict dense anchor |
| targeted-v3 dense candidate | 0.562322 | 0.204117 | 0.120294 | 0.295578 | 0.462973 | previous default |
| nf005 dense candidate | 0.564538 | 0.205358 | 0.121109 | 0.297002 | 0.463390 | predecessor default |
| s40 dense candidate | 0.564538 | 0.205571 | 0.121261 | 0.297123 | 0.463394 | current promoted default |

The current `target64/scifact192` run root is `runs/eos-embed-m-smallfrontier-target64-scifact192-v1-20260622T010936Z/`, sealed SHA256 `7bef55013593780bee57dda59293ce5f8b267d4238fe3b3f9b6bb68701d390f4`. It starts from the balanced Stage B baseline and trains one LR `0.0000025`, HN3, no-teacher continuation on `320` rows: `64` FiQA dev-frontier rows, `64` NFCorpus dev-frontier rows, and `192` SciFact protective replay rows. It passes the dense local `embed-m` gate against the previous smallfrontier frontier and remains far below the s40 dense candidate macro nDCG `0.297123`. The selected compact profile for this frontier is q8/fp16 rerank overfetch-200, method `turboquant_ip_b8_overfetch200_fp16_rerank`, with total compression `1.324138x`.

The rejected `target80/scifact240` run root is `runs/eos-embed-m-smallfrontier-target80-scifact240-v1-20260622T013650Z/`, sealed SHA256 `ea0bf6cec8290b2c72d684424c973800958d83828d69e4819037d0e3ebc8c0ce`. It starts from the same balanced Stage B baseline and trains one LR `0.0000025`, HN3, no-teacher continuation on `400` rows: `80` FiQA dev-frontier rows, `80` NFCorpus dev-frontier rows, and `240` SciFact protective replay rows. FiQA improves versus target64, but macro nDCG and the SciFact guard fail, so q8/fp16/o200 compact evaluation was skipped.

Agreement-first target-row selection showed a real target-dataset signal and produced the best macro in this local `embed-m` branch (`0.187327` in the scifact224 repair), but every repair failed at least one promotion guard. The standalone agreement-first run raised macro while failing the SciFact guard; scifact224 repaired SciFact but missed the NFCorpus recall guard; scifact208 plus nfrecall16 failed both SciFact and NFCorpus recall; and additive scifact224 plus nfrecall16 passed recall but lost macro and SciFact. Therefore the agreement/SciFact/NFCorpus recall-control composition lane is locally exhausted. Future `embed-m` work should change the training signal or model objective, not keep adding or replacing small rows in this same recipe. Provenance: `.tiller/scratch/codex/eos-embed-m-target64-agreement-first-v1-report.md`, `.tiller/scratch/codex/eos-embed-m-target64-agreement-scifact224-v1-report.md`, `.tiller/scratch/codex/eos-embed-m-target64-agreement-scifact208-nfrecall16-v1-report.md`, and `.tiller/scratch/codex/eos-embed-m-target64-agreement-scifact224-nfrecall16-v1-report.md`.

The earlier half-frontier triple-SciFact guard run is `runs/eos-embed-m-half-frontier-triple-scifact-guard-20260615T000000Z/stage-c-half-frontier-triple-scifact-guard-lr25e-7-hn3-b16/`, sealed SHA256 `58b5b80a71520342062c6e6b7062b35ff95a425cccf9a683d23608192e2ac876`. It starts from the balanced Stage B baseline and trains one LR `0.0000025`, HN3, no-teacher continuation on `240` rows: `48` FiQA dev-frontier rows, `48` NFCorpus dev-frontier rows, and `144` SciFact protective replay rows. It is now provenance, not the current local frontier.

The prior protective replay continuation is `runs/eos-embed-m-fiqa-dev-toprank-protective-replay-probe-20260615T000000Z/`. It starts from the balanced Stage B baseline and trains one LR `0.000002`, HN3, no-teacher continuation on a 96-row blend: `48` FiQA dev top-rank rows, `24` SciFact replay rows, and `24` NFCorpus replay rows.

Negative findings for this branch: FiQA source oversampling regressed; the test-selected microrepair was diagnostic only; dev-heldout top-rank selection generalized directionally but was weaker without protective replay; the larger scale96 blend was worse than the 96-row protective blend on macro and FiQA nDCG; and target80/scifact240 shows that increasing all target row counts can improve FiQA while failing macro and the SciFact guard. Future `embed-m` work should change one data-quality lever around target64/scifact192 rather than increasing all row counts again.

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

SciFact-only mxbai s40 continuation status: rejected as a short-context model promotion candidate. The protected data/train path was valid (`4919` rows: `919` SciFact rows with teacher scores and `4000` NFCorpus/FiQA rows with teacher scores stripped), and the candidate moved embeddings slightly, with mean cosine around `0.99999955-0.99999966` and mean L2 delta around `0.00080-0.00091`. The strict dense gate still failed on nDCG@10 for SciFact (`-0.0009635494`), NFCorpus (`-0.0000655497`), and FiQA (`-0.0001069803`) while recall@100 was effectively unchanged. Do not promote this candidate or sweep the same SciFact-only mxbai arm; the next short-context quality attempt should change signal family/objective or use a more comparable external row.

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

Status: implementation is available, but the exact s40 Matryoshka 64/128 plus `128:4=0.05` TurboQuant prepared-IP no-clear recipe is quality-negative as of 2026-06-22. It moved embeddings but regressed repo-docs q4 child quality and failed dense short-set guards, so further Lane H work should change the training signal/data objective instead of repeating that recipe. The teacher-guided q4 child-rank continuation from s40 also moved and was mechanically valid, but failed dense short guards and stayed below the repo-docs q4 bridge; do not retune that same compact-child objective without changing signal family or protection strategy.

### Lane I: Sparse Lexical Head

Add an optional lexical-weight output trained from BM25/SPLADE-style teachers.

Minimum version:

- per-token vocabulary logits or hashed lexical bins
- sparse regularization
- teacher scores from BM25 and/or SPLADE-like external teacher
- hybrid retrieval scoreboard: dense only, sparse only, dense+sparse

Status: the first sparse lexical label exporter and capped-label evaluator are committed as non-default tooling. `eos export-sparse-lexical-labels` emits `manta.sparse_lexical_labels.v1` JSONL plus a manifest, and `eos eval-sparse-lexical-labels` evaluates sparse dot retrieval over the emitted capped labels. `eos fit-sparse-lexical-head` and `eos eval-sparse-lexical-head` now add a non-default experimental `manta.sparse_lexical_hash_head.v1` hashed-bin sidecar/evaluator scaffold over the same labels. `eos eval-sparse-lexical-head-vectors-hybrid` adds a vector-cache dense+sparse fusion evaluator for that sidecar using the existing minmax, zscore, RRF, and dense-prefix-protection hybrid methods. On SciFact train, the full exporter run used `5,183` documents and `809` queries with top-128 document labels and 65,536 hashed bins. Average/max nonzeros were `110.9085`/`128` for documents and `12.2905`/`32` for queries. The unbounded internal BM25 sparse oracle reached nDCG@10 `0.6638196190681942` and recall@100 `0.9012772970745775`; the capped top-128 exported-label evaluator reached nDCG@10 `0.657699123151907` and recall@100 `0.9012772970745775`, a capped-label delta of nDCG `-0.0061204959162872` and recall delta `0`.

First full-train real-data fusion evidence is reported at `runs/eos-lane-i-sparse-head-realdata-fusion-v1-20260622T063620Z/` and `.tiller/scratch/codex/eos-lane-i-sparse-head-realdata-fusion-v1-report.md`. The run exported the current s40 dense SciFact train vectors with CUDA (`5,183` docs, `809` queries, dim `256`) and evaluated the same experimental hashed sparse sidecar with zero missing query labels and zero missing document labels. Dense-only reached nDCG@10 `0.9406547706320765`, recall@100 `1.0`, and P@1 `0.8726823238566132`. Sparse-head-only reached nDCG@10 `0.6382200734082295` and recall@100 `0.8969509682735887`, so the sidecar remains weak alone. The best hybrid was `zscore_blend` alpha `0.25` with `dense_protect_top_k=0`: nDCG@10 `0.9558518532185711`, recall@10 `1.0`, recall@100 `1.0`, and P@1 `0.9035846724351051`. That is `+0.0151970825864946` nDCG@10 versus dense-only and `+0.3176317798103416` versus sparse-only. The second-best row was `minmax_blend` alpha `0.25`, nDCG@10 `0.9555320254340542`. Higher sparse weights regressed sharply, RRF was lower, and dense-protect top-k `1`/`3` preserved dense P@1 while giving up most of the nDCG lift.

Held-out SciFact test confirmation for the train-selected setting is reported at `runs/eos-lane-i-sparse-head-scifact-test-confirmation-v1-20260622T064847Z/` and `.tiller/scratch/codex/eos-lane-i-sparse-head-scifact-test-confirmation-v1-report.md`. The test run covered `5,183` documents, `300` queries, `339` relevant pairs, and zero missing query/doc sparse labels. Dense-only reached nDCG@10 `0.5645379155090131`, recall@100 `0.7964444444444444`, and P@1 `0.4766666666666667`; sparse-only hash-head scoring reached nDCG@10 `0.6406917246395614`, recall@100 `0.8752222222222222`, and P@1 `0.5133333333333333`. The primary train-selected hybrid, `zscore_blend` alpha `0.25` with `dense_protect_top_k=0`, reached nDCG@10 `0.6572165602412263`, recall@100 `0.9288888888888888`, and P@1 `0.57`, improving over dense-only by `+0.0926786447322132` nDCG@10, `+0.13244444444444436` recall@100, and `+0.09333333333333327` P@1. This is positive transfer evidence for low-weight sparse lexical hash-head fusion, not a test-selected claim; the `minmax_blend` alpha `0.25`, `zscore_blend` alpha `0.50`, `minmax_blend` alpha `0.50`, and RRF rows from the same run are diagnostic only.

Short-set NFCorpus evidence is reported at `runs/eos-lane-i-sparse-head-shortset-devtest-v1-20260622T065615Z/` and `.tiller/scratch/codex/eos-lane-i-sparse-head-shortset-devtest-v1-report.md`. The fixed transfer setting, `zscore_blend` alpha `0.25`, improved NFCorpus dense on dev from nDCG@10 `0.198337` / recall@100 `0.234948` to `0.228042` / `0.246808`, and on test from nDCG@10 `0.205571` / recall@100 `0.242059` / P@1 `0.263158` to `0.251123` / `0.270892` / `0.315789`. The dev-selected diagnostic row, `minmax_blend` alpha `0.75`, reached test nDCG@10 `0.307467`, recall@100 `0.284291`, and P@1 `0.399381`; treat it as per-dataset calibration only, not global promotion evidence.

FiQA evidence after the qrels-relevant empty-document coverage fix is reported at `runs/eos-lane-i-sparse-head-fiqa-devtest-after-emptydoc-fix-v1-20260622T071555Z/` and `.tiller/scratch/codex/eos-lane-i-sparse-head-fiqa-devtest-after-emptydoc-fix-v1-report.md`. Sparse lexical label export now keeps qrels-relevant empty corpus rows as zero-term labels; FiQA dev includes documents `33445` and `248226` with `nonzeros=0` and `terms=[]`, and dev/test sparse plus hybrid evals complete with `missing_doc_labels=0`. The fixed `zscore_blend` alpha `0.25` row improved FiQA dense on dev from nDCG@10 `0.103773` / recall@100 `0.341730` to `0.144906` / `0.455644`, and on test from nDCG@10 `0.121261` / recall@100 `0.351678` / P@1 `0.092593` to `0.153829` / `0.469370` / `0.132716`. The dev-selected diagnostic row, `minmax_blend` alpha `0.75`, reached test nDCG@10 `0.214599`, recall@100 `0.487663`, and P@1 `0.175926`; dense/hybrid FiQA still skips empty dense docs as expected (`2` dev, `1` test).

Rollup: fixed global `zscore_blend` alpha `0.25` is now positive versus dense on SciFact test, NFCorpus dev/test, and FiQA dev/test. Caveat: this remains split-specific hashed sparse-label scoring. It is not a trained neural sparse head, not dense model promotion, not a product/default quality claim, and not a default asset or alias change. The earlier oracle exactness comes from unbounded internal BM25 reconstruction, while exported top-128 labels truncate documents; `2,039` document records omitted `59,106` terms, so exported labels are intentionally not exact full-document BM25 state.

Next scaffold: productize a real calibration harness with reusable cross-split sparse-head evidence, or train an actual neural sparse head instead of relying on split-specific lexical-label sidecars. Keep these runs as separate experimental sidecar artifacts under `runs/`, keep `.mll` default output and shipped assets unchanged, and require quality proof before any alias/default discussion. Minimum gate: dense+sparse must beat dense-only on held-out data without violating per-dataset floors, and reports must state whether gains come from trained sparse output or from lexical labels/oracle baselines.

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

1. Change the model-quality signal family before the next guarded candidate: audit a stronger non-overlapping FIQA/SciFact-compatible teacher or synthetic signal, the trained Matryoshka/compact-head lane with movement-positive checks, or a larger-model bootstrap before spending a guarded training run.
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
