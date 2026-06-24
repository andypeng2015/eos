# eos-next-credible-version-packet-v1

## Outcome

This packet turns the current Eos embedder strategy into a concrete execution sequence for the next credible model version. It is grounded in the current promoted default, the fixed-trainer sentinel, the active scale-pilot state, and the existing repo surfaces for teacher/data scaling, `embed-m`, Matryoshka/compact heads, TurboQuant-aware training, and long-context validation.

No source files, defaults, assets, aliases, training runs, eval runs, commits, or pushes were performed for this packet.

## Distillation

- Current quality anchor remains `eos-embed-v1` / `corkscrewdb-default-embedder`, sealed artifact `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll`, SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`.
- Current short-set dense macro is about `0.297181608` nDCG@10 and `0.463396240` recall@100 across SciFact, NFCorpus, and FiQA. Dataset nDCG@10 is SciFact `0.564537916`, NFCorpus `0.205745968`, FiQA `0.121260941`.
- Fixed-trainer sentinel confirmed restore-best/per-epoch eval is active, but the recovered current-default recipe still produced `dense_reject`, macro nDCG delta `-0.0000763096`, and macro recall delta `-0.0000024571`. Do not spend more runs on tiny current-default replay justified only by trainer fix.
- Active scale pilot `eos-s40-fiqa-clean-scale-pilot-v1` has only a run-root marker at `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z` and no scratch report yet. Do not duplicate it; branch only after its report lands.
- Existing local data scale is limited. Local processed files reach `82,246` mixed rows and clean FiQA relabel rows top out around `20,265`; broader MS MARCO/NQ/Hotpot/full BEIR scale is not local and needs an acquisition descriptor with license checks.
- `embed-m` is not ready to replace the default. Accepted Stage C is `0.279050841` macro nDCG, still about `-0.01813` behind the current default, and the latest successors were dense rejects. Same-family tiny continuations should stop unless a boundary audit identifies a specific top-10 objective.
- Old nf005/s40 128d q4 Matryoshka lanes are exhausted for their exact recipes. One targeted prefix probe had no measurable repo-docs 128d movement; one teacher-guided q4 child-rank run moved weights but missed both dense preservation and compact-child targets.
- TurboQuant and packed-parent storage are strong product evidence, but they are not the binding constraint. Model quality is the blocker. Compact evaluation should follow dense acceptance, not rescue dense misses.
- Long-context product wedge is currently absent against required q4 external chunked baselines: official diagnostic gaps are about `-0.337066` on QMSum vs Qwen3 q4, `-0.267719` on QMSum vs mxbai q4, and `-0.255915` on 2WikiMQA vs required q4 external rows.

## Naming Recommendation

Do not public-version the next model yet. Keep execution runs under descriptive run IDs and keep the shipped model name `eos-embed-v1` until a candidate passes dense promotion, compact/TurboQuant promotion, package smoke, and release docs.

Use these names:

- Current shipped line: `eos-embed-v1` and alias `corkscrewdb-default-embedder`.
- Next dense release candidate, only after a dense exploration pass plus compact pass: `eos-embed-v1.1-rc1` in docs/release notes, with `EOS_MODEL_NAME=eos-embed-v1` unless package format or runtime contracts require a new model name.
- Capacity research line: keep `manta-embed-m` / `embed-m` as an experiment family, not a default alias, until it beats the current default or clearly earns a separate larger-model release.
- Compact-head candidate suffix: append profile evidence, for example `eos-embed-v1.1-rc1-q4fp16-o200` for the compact serving row, not for the dense artifact itself.

## Ordered Milestones

### M0. Close The Active Scale Pilot

Wait for `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1-report.md`. If no report exists, do not start another scale run. The only allowed work is status summarization from the active worker output.

Branching once the report lands:

- If dense exploration passes: run compact/TurboQuant q4/fp16/o200 and q2/q4/q8 frontier on that exact sealed candidate, then run strict dense promotion gate. If compact also passes, treat it as `eos-embed-v1.1-rc1` evidence.
- If dense is a near miss with bounded regressions: run a per-query boundary audit before more training. Decide whether losses are top-10 reranks, recall loss, or source/teacher conflict.
- If dense hard-regresses or macro does not move: stop local FiQA-only scaling and prioritize external teacher/data acquisition.
- If the pilot fails due to JSON shape, wrapper, disk, or package issues: fix only the infrastructure issue and rerun the same descriptor once. Do not change objective/data at the same time.

### M1. External Teacher And Data Scale

Goal: build a larger, cleaner, train-allowed data substrate before more model tuning.

Use the current primitives:

- `scripts/acquire_manta_embed_v1_datasets.fw` for current BEIR acquisition extension.
- `sample-corpus-negatives`, `export-teacher-score-requests`, `import-teacher-scores`, `score-teacher-hard-negatives`, `audit-teacher-scores`, `filter-teacher-scores`, and `relabel-teacher-negatives` from `cmd/eos`.
- `scripts/score_teacher_with_vector_cache.py`, `scripts/build_provenance_safe_agreement_teacher_prep.py`, and `scripts/combine_agreement_teacher_scores.py` for vector-cache and agreement lanes.

Minimum output before another big train:

- Acquisition manifest with dataset, license, split, corpus/query/qrel counts, SHA256s, and `train_allowed` flags.
- Teacher-cache manifest for Qwen3 0.6B and mxbai-large where local cache paths exist; do not cite older cache claims without exact files.
- Clean-negative train file with source accounting and no qrels-positive negative leaks.
- A held-out policy that keeps test rows out of training and selection.

Consumer-hardware boundary:

- Prefer Qwen3 0.6B and mxbai-large cached vectors before 4B/8B.
- Avoid hosted/API dependencies in release evidence.
- Treat MS MARCO license/commercial constraints explicitly before use.
- Cache external embeddings once; do not couple network export to training descriptors.

### M2. Scaled Dense S40/Eos-v1.1 Candidate

Goal: rerun the current-default family with enough clean signal to plausibly move dense quality.

Starting point:

- Initial package/tokenizer from `runs/eos-frontier-teacher-sentinel-balance-sweep-v1-s40-20260620T154736Z/`.
- Current promoted dense scoreboard from `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate-scoreboard/scoreboard.json`.

Training shape:

- Use `scripts/train_manta_embed_v1_candidate.fw` directly so `EOS_RESTORE_BEST=true` and per-epoch eval remain active.
- Use `EOS_MODEL_NAME=eos-embed-v1`, max sequence `256`, dim `128`, hidden `256`, repeats `2`, cached tokenizer, pretokenized JSONL, and a batch size that fits the local GPU.
- Target at least `90k` valid train rows for the mixed-data branch, or at least `30k` hard-negative rows if mixed pair/hard-negative shapes cannot be consumed together.
- Keep `EOS_TEACHER_LOSS_WEIGHT` low (`0.015` or lower) unless teacher filters prove positive-top1 and margin quality on the exact rows.

Exploration gate:

- Macro nDCG@10 delta versus current default `>= +0.0010`.
- Macro recall@100 delta `>= -0.0010`.
- No dataset nDCG@10 worse than `-0.0020`.
- No dataset recall@100 worse than `-0.0030`.
- BM25 comparison is diagnostic only for exploration.

Promotion gate:

- Strict selected-vs-anchor dense gate across SciFact, NFCorpus, and FiQA on `ndcg_at_10` and `recall_at_100`.
- Compact/TurboQuant cannot rescue a dense miss.
- No alias, asset, or docs promotion from an exploration-only run.

### M3. Capacity Scaling: `embed-m`

Goal: decide whether capacity is a useful path, without repeating low-signal Stage C continuations.

Current evidence:

- Accepted Stage C improved Stage B by only `+0.000609` macro nDCG and `+0.000870` macro recall, but remains about `-0.01813` macro nDCG behind the promoted default.
- Latest nfprotect replay preserved/improved recall but failed NFCorpus nDCG by `-0.000513` and did not beat Stage C macro.
- Random-start/current fine-tune recipes have failed historically; full 32768-vocab `embed-m` is blocked by tokenizer cost unless cached tokenizer or tokenizer improvements are used.

Next credible capacity work:

- First run read-only boundary/capacity decision if not already closed by the scale-pilot report.
- Do not run another same-line Stage C continuation unless the boundary audit names specific top-10 competitors and objective rows.
- If capacity proceeds, use a staged bootstrap/reset, not random-start fine-tune: cached 16k tokenizer, max sequence `512`, dim `192`, hidden `384`, repeats `3`, batch `16` to `64` depending on GPU memory, and explicit train/eval time budget.

Capacity gate:

- Improve over accepted Stage C by `>= +0.0020` macro nDCG.
- No dataset nDCG or recall delta below `-0.0005` versus Stage C.
- Shrink distance to current default on both macro nDCG and macro recall.
- Compact q8/fp16/o200 only after dense capacity acceptance.

Decision rule:

- If scaled `eos-embed-v1` data work moves quality, keep `embed-m` as a follow-up.
- If scaled `eos-embed-v1` data work is flat but teacher/external gaps remain large, run one bounded `embed-m` reset.
- If both data-scale and capacity reset fail, stop local architecture tuning and revisit data/teacher quality.

### M4. Real Matryoshka And Native Compact Head

Goal: train compact outputs as first-class objectives, not just prefix-truncated bridges.

Closed branches:

- Do not repeat the exact nf005 `128:4=0.05` targeted-prefix Matryoshka recipe; it had no measurable repo-docs 128d movement.
- Do not repeat the exact s40 teacher-guided q4 child-rank recipe; it was movement-positive but failed dense preservation and scored repo-docs 128d q4 `0.593812`, below bridge `0.603952` and target `0.608952`.

Next credible compact-head work:

- Start only from the current promoted default or a dense-accepted future candidate.
- Require `scripts/diagnose_eos_embedding_movement.fw` to be movement-positive at 64d, 128d, and full dim before expensive scoreboards.
- Use explicit `EOS_MATRYOSHKA_DIMS=64,128,256` and `EOS_MATRYOSHKA_WEIGHTS`, plus compact objectives such as `EOS_TURBOQUANT_PREFIX_OBJECTIVES` or `EOS_TURBOQUANT_RANK_MARGIN_OBJECTIVES` only when their source rows are train-safe.
- Keep full-dim dense preservation as a hard gate.

Compact-head gate:

- Full-dim dense strict pass versus its starting dense anchor.
- Repo-docs 128d q4 child nDCG@10 `>= 0.608952` if using the current repo-docs bridge target.
- Short-set 128d q8 and q4 rows reported with quantizer seed, vector bytes, compression, nDCG@10, recall@100, and storage boundary.
- No promotion from repo-docs alone; repo-docs is an early compact-head signal, not full release evidence.

### M5. Quantization-Aware Embeddings And TurboQuant Frontier

Goal: make compact retrieval survival an objective and a measured product surface.

Existing supports:

- `scripts/train_manta_embed_v1_candidate.fw` exposes `EOS_TURBOQUANT_PREFIX_OBJECTIVES`, `EOS_TURBOQUANT_COMPACT_OBJECTIVES`, `EOS_TURBOQUANT_RANK_MARGIN_OBJECTIVES`, `EOS_TURBOQUANT_PREFIX_SCORE_MODE`, and seeds.
- `scripts/score_manta_embed_v1_baselines.fw` emits dense, TurboQuant, TurboQuant rerank, external vector, external multivector, hybrid, and long rows.
- `scripts/smoke_eos_default_embedder_serving.fw` is the local serving proxy for q4/fp16/o200.
- `scripts/smoke_corkscrewdb_child_vectors.fw` covers local flat CorkScrewDB API packed/separate/single-parent layouts.

Order:

1. Dense pass.
2. q4/fp16/o200 compact non-regression versus current compact anchor.
3. q2/q4/q8 frontier for quality, compression, p50/p95/p99, docs/s, scores/s.
4. Packed-parent CorkScrewDB API smoke only for product storage/API claims.

Gate:

- Promoted compact profile remains q4/fp16/rerank-overfetch=200 unless q8 or q3 evidence is clearly better on quality/latency/storage.
- A direct low-bit profile is viable only if macro nDCG and recall deltas are `>= -0.001` and storage/latency wins are meaningful.
- q2 remains pressure testing unless a trained compact head closes its quality gap.
- q4/fp16 sidecar rerank and packed-parent direct child storage are different product surfaces; do not mix their storage claims.

### M6. Long-Context Wedge Validation

Goal: create a credible local long-context claim only after model quality or objective changes close the external q4 gap.

Current state:

- Synthetic late-needle chunking works, but that is not product-quality LongEmbed proof.
- Capped repo-docs and official qmsum/2wikimqa rows all carry `quality_claim=false`.
- Eos trails required q4 external chunked baselines on QMSum and 2WikiMQA by large nDCG margins.

Next credible long-context sequence:

1. Keep current long-context rows as diagnostic only.
2. Select external chunked q4 targets from Qwen3 0.6B and mxbai-large cache rows.
3. Build train-safe long-context hard negatives from non-test splits or synthetic tasks with explicit leakage checks.
4. Train only after short-set dense gates are protected.
5. Run official qmsum and 2wikimqa product-wedge summary with `quality_claim=false` until Eos q4 beats required external q4 rows.

Long-context gate:

- Eos preferred q4 row must beat all required q4 external chunked rows on nDCG@10 for both QMSum and 2WikiMQA.
- Short-set dense quality must not regress beyond the promotion gate.
- Retargeting, max observed document tokens, chunk policy, pooling, bit width, storage multiple, and quality-claim state must be recorded.

## Gates Summary

| Gate | Required pass condition |
| --- | --- |
| Scale exploration | macro nDCG delta `>= +0.0010`; macro recall delta `>= -0.0010`; per-dataset nDCG floor `-0.0020`; per-dataset recall floor `-0.0030` |
| Dense promotion | all selected SciFact/NFCorpus/FiQA `ndcg_at_10` and `recall_at_100` checks pass against current default with zero or explicitly accepted rounding tolerance |
| Compact promotion | dense promotion first; q4/fp16/o200 strict compact gate on `ndcg_at_10`, `recall_at_100`, and `total_compression_ratio` versus current compact anchor |
| `embed-m` capacity | `>= +0.0020` macro nDCG over accepted Stage C; no dataset metric below `-0.0005`; gap to current default shrinks |
| Native Matryoshka | movement-positive; full-dim dense strict pass; 128d q4 repo-docs nDCG `>= 0.608952` for the current bridge target |
| TurboQuant frontier | q2/q4/q8 rows report quality, compression, sidecar bytes, scoring latency, docs/s, scores/s; no low-bit default without measured quality survival |
| Long-context wedge | Eos preferred q4 beats Qwen3 and mxbai q4 chunked rows on both QMSum and 2WikiMQA; `quality_claim=false` until then |

## Resource Expectations

- Scale-pilot and dense s40 continuations: medium local training budget. Expect tens of GB of run-root churn for pretokenized JSONL, package copies, scoreboards, and logs. Require a disk preflight; prior runs ended with as little as `18G` free.
- `embed-m`: high local training budget. Use cached tokenizer and batch `16` to `64`; previous accepted Stage C took about `140s` for `369` examples and `24` steps, with `~421` train pairs/s. Full 32k-vocab tokenizer training is not a good default until cached or optimized.
- External vector caches: medium/high one-time cost. Qwen3 0.6B and mxbai-large are plausible local cache targets; 4B/8B should be aspirational unless hardware and disk are explicitly available.
- Matryoshka/compact-head: medium training plus medium scoring. Require movement diagnostic before full short-set and repo-docs scoreboards.
- TurboQuant frontier: medium eval/serving budget. It is cheaper than training but can multiply scoreboards by bits, rerank settings, datasets, and CorkScrewDB layouts.
- Long-context official wedge: medium/high eval and cache budget. Keep capped and diagnostic until a trained candidate exists.

## Task Descriptors

### D0. Scale Pilot Triage

- id/title: `eos-s40-fiqa-clean-scale-pilot-v1-triage`
- role/profile: `tiller-summary` or `tiller-worker` read-only summarizer
- objective: Consume the completed scale-pilot report when it appears and classify the outcome into `dense_exploration_pass`, `near_miss`, `hard_reject`, or `infrastructure_fail`.
- context paths: `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1-report.md`; `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1.run-root`; `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/`
- constraints: Do not train, score, compact-evaluate, edit, promote, or mutate the run. If no report exists, say so and stop.
- expected outputs: scratch triage note with dense metrics, train scale, restore-best evidence, compact status, and branch recommendation.
- verification target: report exists and any referenced JSON parses.
- budget tier/model ceiling: cheap.
- sandbox/permission needs: read plus optional scratch report write.
- dependencies/blockers: active worker report must exist.
- checkpoint criteria: no.
- report contract: Outcome; files inspected; verification; caveats; checkpoint candidate no; Arbiter next action.

### D1. External Data Acquisition And License Manifest

- id/title: `eos-external-retrieval-data-acquisition-manifest-v1`
- role/profile: `tiller-worker` execution, bounded acquisition/manifests only
- objective: Extend or wrap the current acquisition path to inventory additional public retrieval datasets and produce train/dev/test, license, and checksum manifests without training.
- context paths: `scripts/acquire_manta_embed_v1_datasets.fw`; `docs/production-embedding.md`; `datasets/manta-embed-v1/raw/`; `datasets/manta-embed-v1/processed/`
- constraints: No model training, no default movement, no use of test rows for train/selection, no hidden license assumptions.
- expected outputs: dataset manifest JSON/TSV, row-count table, license/terms notes, SHA256s, and explicit `train_allowed` flags.
- verification target: manifests parse; qrels/corpus/query counts match files; git status summarized.
- budget tier/model ceiling: medium execution.
- sandbox/permission needs: network/disk if acquisition is approved; otherwise read-only inventory.
- dependencies/blockers: storage budget and license acceptance.
- checkpoint criteria: yes only if tracked acquisition script/docs are intentionally changed in a separate reviewed slice.
- report contract: Outcome; files changed/inspected; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

### D2. Teacher Cache And Clean-Negative Builder

- id/title: `eos-qwen-mxbai-agreement-clean-negative-scale-v1`
- role/profile: `tiller-worker` execution
- objective: Build train-allowed Qwen3/mxbai agreement-scored hard negatives and random true-negative pools for SciFact/NFCorpus/FiQA plus any acquired datasets.
- context paths: `scripts/score_teacher_with_vector_cache.py`; `scripts/build_provenance_safe_agreement_teacher_prep.py`; `scripts/combine_agreement_teacher_scores.py`; `cmd/eos` teacher/relabel commands; `datasets/manta-embed-v1/processed/relabel/`
- constraints: Do not train. Preserve source tags. Do not assume vector caches exist; verify exact `doc-vectors.jsonl` and `query-vectors.jsonl` paths.
- expected outputs: clean-negative JSONL, teacher-score JSONL, audit/filter summaries, qrels-leak report, source counts, and train-allowed manifest.
- verification target: JSON parses; positive-top1 and margin audit reported; qrels-positive negative leaks are zero; row counts meet the planned scale floor.
- budget tier/model ceiling: medium/high execution depending on cache scoring.
- sandbox/permission needs: local disk and optional GPU/Python env for external cache scoring.
- dependencies/blockers: D1 for new datasets; local external vector caches or approved export.
- checkpoint criteria: no source checkpoint unless reusable helper scripts are added deliberately.
- report contract: Outcome; files generated/inspected; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

### D3. Scaled Dense Candidate Runner

- id/title: `eos-v1-1-scaled-dense-candidate-v1`
- role/profile: `tiller-worker` execution
- objective: Train exactly one scaled current-default-family candidate from clean/agreement data and evaluate dense exploration against the current promoted default.
- context paths: current default run root; D2 train file and manifest; `scripts/train_manta_embed_v1_candidate.fw`; `scripts/run_manta_embed_v1_guarded_candidate.fw`; `scripts/score_manta_embed_v1_baselines.fw`
- constraints: One train run plus dense scoring. No compact eval unless dense exploration passes. No alias/default/docs/assets changes.
- expected outputs: run root, package/sealed artifact, train/final/hard metrics, dense scoreboard, exploration summary, restore-best evidence.
- verification target: package inspect OK; eval-only optimizer updates `0`; `config.restore_best=true`; actual eval passes reported; scoreboard JSON parses; exploration gate verdict recorded.
- budget tier/model ceiling: medium/high execution.
- sandbox/permission needs: local GPU, disk preflight, scratch report write.
- dependencies/blockers: D0 branch or D2 clean data.
- checkpoint criteria: no source checkpoint; evidence report only.
- report contract: Outcome; dense table; train scale; restore-best evidence; compact skipped/run reason; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

### D4. Near-Miss Boundary Audit

- id/title: `eos-dense-nearmiss-boundary-audit-v1`
- role/profile: `tiller-investigator` read-only
- objective: If a scale or capacity run is a near miss, classify per-query losses into top-10 rerank, recall loss, source/teacher conflict, data leak risk, or unknown.
- context paths: candidate dense comparison JSON; per-query/ranking outputs if present; current default scoreboard; D3 report.
- constraints: Do not rerun eval unless root approves a bounded dump descriptor. Do not train.
- expected outputs: per-query delta table, top loss contributors, and a concrete next objective/data recommendation.
- verification target: emitted JSON/TSV parses; every claimed query loss points to an artifact.
- budget tier/model ceiling: medium/high read-only.
- sandbox/permission needs: read plus scratch writes.
- dependencies/blockers: requires per-query/ranking artifacts or a follow-up dump descriptor.
- checkpoint criteria: no.
- report contract: Outcome; files inspected/generated; verification; caveats; checkpoint candidate no; Arbiter next action.

### D5. `embed-m` Capacity Reset Probe

- id/title: `eos-embed-m-capacity-reset-probe-v1`
- role/profile: `tiller-worker` execution after architect approval
- objective: Run one staged/cached-tokenizer `embed-m` reset or continuation only if D4 or D3 justifies capacity work.
- context paths: accepted Stage C run; latest reject reports; `scripts/train_manta_embed_v1_candidate.fw`; `scripts/run_manta_embed_v1_guarded_candidate.fw`
- constraints: No random-start fine-tune. No same-line Stage C replay unless D4 names an objective. No default promotion.
- expected outputs: sealed `manta-embed-m` candidate, dense comparison versus Stage C and current default, compact q8/fp16/o200 only after dense capacity pass.
- verification target: package inspect OK; dense capacity gate verdict; pairwise and hard eval-only optimizer updates `0`; disk status.
- budget tier/model ceiling: high execution.
- sandbox/permission needs: local GPU, large disk, scratch write.
- dependencies/blockers: D4 or D3 capacity decision.
- checkpoint criteria: no source checkpoint.
- report contract: Outcome; train config; dense table; default gap; compact status; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

### D6. Native Matryoshka Compact-Head Candidate

- id/title: `eos-current-native-matryoshka-compact-head-v1`
- role/profile: `tiller-worker` execution
- objective: Train a movement-positive native compact-head candidate from current default or a dense-accepted future candidate with explicit 64/128/256 Matryoshka and q4 objectives.
- context paths: current default or D3 accepted candidate; `scripts/train_manta_embed_v1_candidate.fw`; `scripts/diagnose_eos_embedding_movement.fw`; `scripts/score_manta_embed_v1_baselines.fw`
- constraints: Do not use old nf005 exhausted recipe. Full-dim dense preservation is mandatory. No promotion from repo-docs alone.
- expected outputs: movement report, dense short-set scoreboard, repo-docs 128d child scoreboard, q2/q4/q8 compact metrics.
- verification target: movement-positive at 64/128/full; full-dim dense strict pass; repo-docs 128d q4 target checked; JSON parses.
- budget tier/model ceiling: medium/high execution.
- sandbox/permission needs: local GPU, scratch writes.
- dependencies/blockers: dense-accepted anchor or root approval to use current default.
- checkpoint criteria: no unless tracked docs are updated separately.
- report contract: Outcome; movement; dense/compact tables; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

### D7. TurboQuant Frontier And Serving Gate

- id/title: `eos-v1-1-turboquant-frontier-serving-v1`
- role/profile: `tiller-worker` execution
- objective: For a dense-accepted candidate, run q2/q4/q8 and q4/fp16/o200 scoring plus serving/CorkScrewDB-local smokes needed for compact release evidence.
- context paths: candidate run; current compact anchor; `scripts/score_manta_embed_v1_baselines.fw`; `scripts/smoke_eos_default_embedder_serving.fw`; `scripts/smoke_corkscrewdb_child_vectors.fw`; `docs/turboquant-multivector-frontier.md`
- constraints: Dense pass required. Keep q4/fp16 rerank and packed-parent direct child storage claims separate.
- expected outputs: frontier JSON/TSV, compact gate summary, serving p50/p95/p99, storage bytes, compression, CorkScrewDB layout manifest where applicable.
- verification target: compact gate passes or fails explicitly; JSON parses; latency/storage rows present; quantizer seed recorded.
- budget tier/model ceiling: medium execution.
- sandbox/permission needs: local disk, optional CorkScrewDB checkout for API smoke.
- dependencies/blockers: D3 dense pass.
- checkpoint criteria: yes only for tracked doc/status updates in a separate checkpoint.
- report contract: Outcome; files generated/inspected; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

### D8. Long-Context Target And Wedge Plan

- id/title: `eos-long-context-external-q4-target-plan-v1`
- role/profile: `tiller-architect` or `tiller-investigator` read-only
- objective: Select the exact long-context external q4 target rows and train-safe data needed for a future wedge attempt.
- context paths: `docs/local-long-context-embedder-wedge.md`; `scripts/run_long_context_product_wedge_pipeline.py`; `scripts/plan_guarded_longembed_repair_candidate.py`; `T3-current-long-context-decision-report.md`; official qmsum/2wikimqa comparison JSONs.
- constraints: No training or quality claims. Preserve `quality_claim=false` until gate passes.
- expected outputs: target table for QMSum and 2WikiMQA, storage ratios, gap sizes, train-safe data requirements, and a guarded run descriptor.
- verification target: target rows point to existing comparison artifacts; required external q4 rows are non-missing.
- budget tier/model ceiling: high read-only reasoning.
- sandbox/permission needs: read plus scratch write.
- dependencies/blockers: candidate model quality work if training is proposed.
- checkpoint criteria: no unless plan artifact is committed by root.
- report contract: Outcome; files inspected; verification; caveats; checkpoint candidate yes/no; Arbiter next action.

## Risks And Residual Decisions

- Data quality can dominate data scale. The teacher frontier reports show several rerankers had weak full-frontier positive-top1 rates and were marked `no_train`; do not convert request-only or audit-only artifacts into training rows.
- FiQA-heavy clean data can trade away SciFact/NFCorpus. Every mixed file needs source counts and per-source deltas.
- Disk is a live constraint. Training plus pretokenized files, vector caches, and scoreboards can exhaust the local root. Require `df -h` preflight and cleanup decisions before high-budget descriptors.
- `embed-m` may be capacity-positive but not quality-positive. It needs a larger, staged bootstrap or data reset, not blind continuation.
- Compact objectives can move weights without improving compact retrieval. Movement diagnostics are necessary but not sufficient.
- Long-context synthetic wins are easy to overstate. Official/capped rows are diagnostic until the required external q4 comparison gate passes.

## Evidence Paths Inspected

- `AGENTS.md`
- `CHANGELOG.md`
- `docs/default-corkscrew-embedder-plan.md`
- `docs/manta-embed-sota-avenues.md`
- `docs/production-embedding.md`
- `docs/local-long-context-embedder-wedge.md`
- `docs/turboquant-multivector-frontier.md`
- `scripts/train_manta_embed_v1_candidate.fw`
- `scripts/run_manta_embed_v1_guarded_candidate.fw`
- `scripts/score_manta_embed_v1_baselines.fw`
- `scripts/train_manta_embed_v1_shipping_pipeline.fw`
- `scripts/run_long_context_product_wedge_pipeline.py`
- `scripts/plan_guarded_longembed_repair_candidate.py`
- `.tiller/scratch/codex/eos-scaled-data-gate-plan-v1.md`
- `.tiller/scratch/codex/eos-performance-experiment-ladder-v1.md`
- `.tiller/scratch/codex/eos-fixed-trainer-sentinel-rerun-v1-report.md`
- `.tiller/scratch/codex/eos-embed-m-bootstrap-stageb-nontest-frontier-stagec-v1-report.md`
- `.tiller/scratch/codex/eos-embed-m-stagec-nfprotect-replay-v1-report.md`
- `.tiller/scratch/codex/eos-128q4-targeted-prefix-matryoshka-probe-v2-report.md`
- `.tiller/scratch/codex/eos-teacher-guided-128q4-child-rankmargin-v1-report.md`
- `.tiller/scratch/codex/T3-current-long-context-decision-report.md`
- `.tiller/scratch/codex/eos-current-default-long-context-wedge-rerun-v1-report.md`
- `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1.run-root`

## Verification

Packet verification target:

```bash
test -s .tiller/scratch/codex/eos-next-credible-version-packet-v1.md
git status --short --branch --ignored=no
```

Expected result: report exists; no source/default/asset changes.

## Checkpoint Candidate

Yes, report-only, if root chooses to checkpoint planning artifacts:

- `.tiller/scratch/codex/eos-next-credible-version-packet-v1.md`

No source/default/asset checkpoint is implied.

## Arbiter Next Action

Wait for and triage the active `eos-s40-fiqa-clean-scale-pilot-v1` report. If it passes dense exploration, compact-evaluate exactly that candidate and consider an `eos-embed-v1.1-rc1` release-candidate lane. If it rejects or stalls, prioritize external teacher/data acquisition and clean-negative scale before any capacity or compact-head training. Do not promote, rename, or long-context-claim a next model until dense, compact, package, and product-surface gates all pass.
