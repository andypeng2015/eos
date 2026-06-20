# eos-compact-nfcorpus-boundary-repair-v1 Report

## Outcome

Status: **NO-PROMOTE**.

Ran exactly one bounded continuation from the dense-pass/compact-fail candidate using four NFCorpus compact q4/fp16-overfetch200 boundary-loss hard-negative rows. Dense strict gate passed all six checks against nf005. Formal compact q4/fp16/rerank-overfetch200 gate failed one check: NFCorpus `recall_at_100`.

- Run: `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/`
- Prep: `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z-prep/`
- Report status: do not promote.

## Distillation

The targeted repair recovered compact NFCorpus `nDCG@10` above the formal compact anchor, but did not recover the strict compact `recall_at_100` gate. The same four previously lost relevant docs remain outside top-100, and one new relevant doc is gained into top-100. Net top-100 relevant count is still `-3`, producing NFCorpus compact recall delta `-0.001631`.

The probe is useful evidence that rank-margin plus boundary rows can improve dense/compact nDCG without breaking dense gates, but this exact candidate remains blocked by strict compact recall.

## Files Changed, Inspected, Generated

Source added:

- `scripts/mine_retrieval_boundary_losses.py`
- `scripts/test_mine_retrieval_boundary_losses.py`

Generated:

- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z-prep/data/train.compact-boundary-losses.jsonl`
- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z-prep/data/train.compact-boundary-losses.manifest.json`
- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/`
- `.tiller/scratch/codex/eos-compact-nfcorpus-boundary-repair-v1.run-id`
- `.tiller/scratch/codex/eos-compact-nfcorpus-boundary-repair-v1.prep-root`
- `.tiller/scratch/codex/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z.guarded-run.log`
- `.tiller/scratch/codex/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z.compact-scoreboard.log`

Inspected:

- Prior report: `.tiller/scratch/codex/eos-longembed-agreement-teacher-guarded-training-v1-report.md`
- Lost TSV and candidate/anchor per-query diagnostics under `runs/eos-longembed-agreement-teacher-guarded-training-v1-20260620T114058Z/`
- Dense anchor scoreboard under `runs/current-release-qwen3-nf005-continuation-20260616T224102Z/`
- Formal compact anchor scoreboard under `runs/eos-nf005-compact-anchor-provenance-repair-v1-20260619T091223Z/`

## Repair Data Counts

Manifest: `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z-prep/data/train.compact-boundary-losses.manifest.json`

- Lost rows read: `4`
- Hard-negative rows written: `4`
- Skipped rows: `0`
- Represented positives: `MED-2799`, `MED-3866`, `MED-4380`, `MED-2145`
- Represented queries: `PLAIN-2051`, `PLAIN-2770`, `PLAIN-551`, `PLAIN-660`
- Negative counts: `[8, 8, 8, 8]`
- Boundary window: ranks `90..105`
- Claim boundary: `quality_claim=false`; protected candidate repair data only, not benchmark-quality or promotion evidence.

## Exact Config

Training used only the compact-boundary rows, no teacher-score blend.

```bash
EOS_GUARD_RUN_ID=eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z
EOS_GUARD_RUN_DIR=$PWD/runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z
EOS_GUARD_CANDIDATE_RUN_ID=candidate
EOS_GUARD_ANCHOR_SCOREBOARD=$PWD/runs/current-release-qwen3-nf005-continuation-20260616T224102Z/candidate-scoreboard/scoreboard.json
EOS_GUARD_DATASETS=scifact,nfcorpus,fiqa
EOS_GUARD_METRICS=ndcg_at_10,recall_at_100
EOS_GUARD_CATEGORY=short_retrieval
EOS_GUARD_BASELINE=eos
EOS_GUARD_TOLERANCE=0
EOS_GUARD_FAIL_ON_GATE=0
EOS_GUARD_ALLOW_DIRTY=1
EOS_INITIAL_ARTIFACT=$PWD/runs/eos-longembed-agreement-teacher-guarded-training-v1-20260620T114058Z/candidate/eos-embed-v1.mll
EOS_TOKENIZER=$PWD/runs/eos-longembed-agreement-teacher-guarded-training-v1-20260620T114058Z/candidate/eos-embed-v1.tokenizer.mll
EOS_PACKAGE_TOKENIZER=$PWD/runs/eos-longembed-agreement-teacher-guarded-training-v1-20260620T114058Z/candidate/eos-embed-v1.tokenizer.mll
EOS_TRAIN_JSONL=$PWD/runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z-prep/data/train.compact-boundary-losses.jsonl
EOS_EVAL_JSONL=$PWD/datasets/manta-embed-v1/processed/eval.jsonl
EOS_HARD_EVAL_JSONL=$PWD/datasets/manta-embed-v1/processed/hard-eval.jsonl
EOS_PRETOKENIZE_JSONL=1
EOS_QUALITY_TARGET=pairwise
EOS_HARD_NEGATIVE_TRAIN=1
EOS_HARD_NEGATIVES_PER_QUERY=8
EOS_EPOCHS=1
EOS_BATCH_SIZE=4
EOS_LR=0.00000001
EOS_CONTRASTIVE_LOSS=infonce
EOS_TEMPERATURE=0.05
EOS_TEACHER_LOSS_WEIGHT=0
EOS_RESTORE_BEST=false
EOS_SELECT_METRIC=score_margin
EOS_EVAL_EVERY=1
EOS_EVAL_EVERY_STEPS=0
EOS_PATIENCE=3
EOS_PROGRESS_EVERY=1
EOS_SKIP_TESTS=1
EOS_MATRYOSHKA_DIMS=256
EOS_TURBOQUANT_PREFIX_OBJECTIVES=256:4=0.05
EOS_TURBOQUANT_PREFIX_SEED=5581486560434873699
EOS_TURBOQUANT_PREFIX_SCORE_MODE=prepared-ip
EOS_TURBOQUANT_RANK_MARGIN_OBJECTIVES=256:4=0.05
EOS_TURBOQUANT_RANK_MARGIN=0.01
```

Plan-only accepted the workload:

```text
train=4 hard_negative_contrastive examples batch=4 steps/epoch=1 train_pairs/epoch=436
```

Actual training reported `train_pairs/epoch=292`, `steps_run=1`, `optimizer_updates=7`.

## Train And Eval Metrics

| file | steps_run | steps_completed | optimizer_updates | eval loss | score_margin | pair_accuracy | top1 | mrr | pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train.metrics.json` | 1 | 2524 | 7 | 0.123572 | 0.243202 | 0.620958 | 0.946619 | 0.972717 | 1670 |
| `final-eval.metrics.json` | 0 | 2524 | 0 | 0.123572 | 0.243202 | 0.620958 | 0.946619 | 0.972717 | 1670 |
| `hard-eval.metrics.json` | 0 | 2524 | 0 | 0.121371 | 0.263014 | 0.627990 | 0.931507 | 0.964612 | 1672 |

Package evidence:

- trainable package inspect: `package verify: OK`
- sealed package inspect: `package verify: OK`
- sealed SHA256: `893a2b028aec10e27ca41250c90d0296ca628cd7ab33bf9cd0270737c13aa101`

## Dense Gate

Dense strict gate: **PASS checks=6**.

| dataset | metric | candidate | anchor | delta | result |
| --- | --- | ---: | ---: | ---: | --- |
| scifact | nDCG@10 | 0.564538 | 0.564538 | +0.000000 | PASS |
| scifact | recall@100 | 0.796444 | 0.796444 | +0.000000 | PASS |
| nfcorpus | nDCG@10 | 0.205627 | 0.205358 | +0.000269 | PASS |
| nfcorpus | recall@100 | 0.242048 | 0.242048 | +0.000000 | PASS |
| fiqa | nDCG@10 | 0.121137 | 0.121109 | +0.000028 | PASS |
| fiqa | recall@100 | 0.351678 | 0.351678 | +0.000000 | PASS |

Dense scoreboard: `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/candidate-scoreboard/scoreboard.json`

## Compact Gate

Compact q4/fp16/rerank-overfetch200 strict gate: **FAIL checks=6 failed=1**.

Gate log: `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/gates/compact-q4-fp16-overfetch200-vs-formal-anchor.gate.log`

| dataset | metric | candidate | anchor | delta | result |
| --- | --- | ---: | ---: | ---: | --- |
| scifact | nDCG@10 | 0.564538 | 0.564538 | +0.000000 | PASS |
| scifact | recall@100 | 0.796444 | 0.796444 | +0.000000 | PASS |
| nfcorpus | nDCG@10 | 0.205697 | 0.205519 | +0.000178 | PASS |
| nfcorpus | recall@100 | 0.242059 | 0.243690 | -0.001631 | FAIL |
| fiqa | nDCG@10 | 0.121137 | 0.121109 | +0.000028 | PASS |
| fiqa | recall@100 | 0.351678 | 0.351292 | +0.000386 | PASS |

Compact scoreboard: `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/compact-q4-fp16-overfetch200-scoreboard/scoreboard.json`

## Failure Diagnostics

Generated:

- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/diagnostics/candidate.nfcorpus.compact-q4-fp16-overfetch200.top130.per-query.jsonl`
- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/analysis/nfcorpus_compact_q4_fp16_overfetch200_diff.summary.json`
- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/analysis/nfcorpus_compact_q4_fp16_overfetch200_lost_top100_positives.tsv`
- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/analysis/nfcorpus_compact_q4_fp16_overfetch200_gained_top100_positives.tsv`
- `runs/eos-compact-nfcorpus-boundary-repair-v1-20260620T121150Z/analysis/nfcorpus_compact_q4_fp16_overfetch200_per_query_delta.tsv`

Summary:

- queries compared: `323`
- queries with recall delta: `5`
- queries with negative recall delta: `4`
- queries with positive recall delta: `1`
- relevant docs lost from top100: `4`
- relevant docs gained into top100: `1`
- net relevant top100 docs: `-3`
- mean recall delta: `-0.0016313667371140313`

Lost top-100 relevant docs:

| query | relevant_count | lost doc | anchor rank | candidate rank in top130 | query recall delta |
| --- | ---: | --- | ---: | ---: | ---: |
| `PLAIN-2051` | 355 | `MED-2799` | 100 | 106 | -0.002817 |
| `PLAIN-2770` | 41 | `MED-3866` | 100 | 101 | -0.024390 |
| `PLAIN-551` | 2 | `MED-4380` | 100 | 105 | -0.500000 |
| `PLAIN-660` | 475 | `MED-2145` | 99 | 102 | -0.002105 |

Gained top-100 relevant doc:

| query | relevant_count | gained doc | anchor rank in top130 | candidate rank | query recall delta |
| --- | ---: | --- | ---: | ---: | ---: |
| `PLAIN-1741` | 420 | `MED-1380` | 106 | 100 | +0.002381 |

## Verification Commands And Results

- `python3 scripts/test_mine_retrieval_boundary_losses.py`: PASS, `Ran 3 tests`.
- `python3 -m py_compile scripts/mine_retrieval_boundary_losses.py scripts/test_mine_retrieval_boundary_losses.py`: PASS.
- Repair data generation: PASS, `lost_rows=4 hard_negative_rows=4 skipped_rows=0`.
- Guarded training: PASS, completed exactly one continuation probe.
- Plan-only workload: PASS, small one-batch workload.
- Final eval-only gate: PASS, `optimizer_updates=0`.
- Hard eval-only gate: PASS, `optimizer_updates=0`.
- Package inspect/verify trainable and sealed: PASS.
- Dense gate: PASS, `scoreboard gate: PASS checks=6`.
- Compact gate: FAIL, `scoreboard gate: FAIL checks=6 failed=1`.
- Per-query diagnostic JSON/JSONL parse: PASS.
- `git diff --check`: PASS.

Current source status relevant to this descriptor:

```text
?? scripts/mine_retrieval_boundary_losses.py
?? scripts/test_mine_retrieval_boundary_losses.py
```

## Caveats And Residual Risk

- This is a NO-PROMOTE artifact because strict formal compact recall still fails.
- The hard-negative rows are protected repair data with `quality_claim=false`; they should not be treated as benchmark evidence.
- The repair used test-split boundary diagnostics by descriptor request. Keep the claim boundary narrow.
- `EOS_GUARD_ALLOW_DIRTY=1` was required because this descriptor added uncommitted source files.

## Checkpoint Candidate

Checkpoint candidate: **yes for the focused miner/test source and NO-PROMOTE report**, not for model promotion.

Do not checkpoint bulky `runs/` artifacts unless the project explicitly wants generated run evidence committed. The coherent source slice is:

- `scripts/mine_retrieval_boundary_losses.py`
- `scripts/test_mine_retrieval_boundary_losses.py`
- `.tiller/scratch/codex/eos-compact-nfcorpus-boundary-repair-v1-report.md`

## Arbiter Next Action

Record this descriptor as complete NO-PROMOTE. Do not promote the candidate. If another bounded probe is authorized, it should target the same four still-lost docs with a stronger compact recall objective or different ranking pressure; this probe’s exact q4 rank-margin/boundary-row setup is insufficient for strict compact recall.
