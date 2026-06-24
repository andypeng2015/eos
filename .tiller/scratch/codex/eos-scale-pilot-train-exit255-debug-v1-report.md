# eos-scale-pilot-train-exit255-debug-v1

## Outcome

Diagnosis: no deterministic `manta train-embed` code/config failure was found in bounded reproduction. The original 100k pilot failure is best classified as an abrupt external/process/resource interruption after workload planning, with the exact mechanism unproven from saved artifacts.

No source patch was applied by this debugger pass. I wrote bounded debug artifacts under the failed run root and this report only.

## Distillation

- Failed run evidence is silent after `planned workload`; no panic, Go error, trainer error, CUDA diagnostic, metrics write error, or `train.metrics.json` exists.
- The saved command shape is valid on current binaries/data shape: eval-only, 64-row hard-negative training, and 256-row hard-negative training all completed with the same restore-best, teacher loss, `--no-tokenizer`, source-weight, matryoshka, clear-prefix, and TurboQuant compact objective flags.
- The 64/256-row probes both restored best and wrote metrics with CUDA activation enabled.
- Current machine resource evidence shows 27 GiB RAM and 8 GiB swap with swap fully used during this investigation. That does not prove the old failure, but it makes a silent external termination/resource-pressure explanation plausible.
- The current worktree already has uncommitted training-progress hardening in the relevant files: wrapper default progress changed from `0` to `100`, `fit_start` progress was added, and eval phase progress was added. I did not create those source edits.

## Root Cause Evidence

Original failed artifacts:

- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/logs/train.log.cmd` saved the real command with `--progress-every 0`.
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/logs/train.log` contains only the workload line.
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/logs/train-driver.outer.log` shows preflight, build, tokenization, plan-only success, then the train step ending as `error: go run: exit status 255`.
- No status file was found under the run root/candidate, and no child diagnostic was available.

Bounded reproduction result:

- Direct eval-only on 8 tokenized eval rows completed and wrote metrics.
- Direct 64-row hard-negative train completed: `restored_best=true`, `steps_run=2`, `actual_eval_passes=3`, `activation=cuda`.
- Direct 256-row hard-negative train completed: `restored_best=true`, `steps_run=8`, `actual_eval_passes=3`, `activation=cuda`.

This rules out the obvious deterministic failures in argument parsing, package loading, `--no-tokenizer`, hard-negative token JSONL decoding, restore-best initial/final eval, CUDA activation env, teacher-score normalization, source weighting, and TurboQuant compact objective handling at small scale.

## Files Changed / Inspected

Changed by this pass:

- `.tiller/scratch/codex/eos-scale-pilot-train-exit255-debug-v1-report.md`

Generated debug artifacts:

- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/debug-exit255/eval/`
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/debug-exit255/train64/`
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/debug-exit255/train256/`

Inspected:

- `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1-report.md`
- `.tiller/scratch/codex/eos-debug-norestore-train-exit255-report.md`
- `.tiller/scratch/codex/eos-compact-aware-norestore-exit255-debug-report.md`
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/logs/train.log`
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/logs/train.log.cmd`
- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/logs/train-driver.outer.log`
- `cmd/eos/main.go`
- `runtime/embedding_train_job.go`
- `runtime/embedding_train_runner.go`
- `runtime/embedding_train_runner_test.go`
- `runtime/embedding_trainer.go`
- `scripts/train_manta_embed_v1_candidate.fw`

Pre-existing dirty source files observed, not changed by this pass:

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`
- `runtime/embedding_train_runner.go`
- `runtime/embedding_train_runner_test.go`
- `scripts/train_manta_embed_v1_candidate.fw`

## Verification Commands / Results

Eval-only smoke:

```bash
EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1 EOS_TRAIN_ENABLE_FAST_GELU=1 timeout 5m \
  runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/bin/manta \
  train-embed --eval-only --no-tokenizer --metrics-json .../debug-exit255/eval/eval.metrics.json \
  --seed 1 --progress-every 1 .../debug-exit255/eval/eos-embed-v1.mll .../debug-exit255/eval/eval8.tokens.jsonl
```

Result: passed; `final_eval.pair_count=8`, `accelerators.activation=cuda`.

64-row train smoke:

```bash
EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1 EOS_TRAIN_ENABLE_FAST_GELU=1 timeout 10m \
  .../candidate/bin/manta train-embed [saved objective flags] --epochs 1 --progress-every 1 \
  --metrics-json .../debug-exit255/train64/train.metrics.json --no-tokenizer \
  .../debug-exit255/train64/eos-embed-v1.mll .../debug-exit255/train64/train64.tokens.jsonl .../debug-exit255/train64/eval8.tokens.jsonl
```

Result: passed; `restored_best=true`, `steps_run=2`, `actual_train_pairs=5708`, `actual_eval_passes=3`, `final_eval.pair_count=8`, `activation=cuda`.

256-row train smoke:

```bash
EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1 EOS_TRAIN_ENABLE_FAST_GELU=1 timeout 15m \
  .../candidate/bin/manta train-embed [saved objective flags] --epochs 1 --progress-every 16 \
  --metrics-json .../debug-exit255/train256/train.metrics.json --no-tokenizer \
  .../debug-exit255/train256/eos-embed-v1.mll .../debug-exit255/train256/train256.tokens.jsonl .../debug-exit255/train256/eval16.tokens.jsonl
```

Result: passed; `restored_best=true`, `steps_run=8`, `actual_train_pairs=22812`, `actual_eval_passes=3`, `final_eval.pair_count=16`, `activation=cuda`.

Focused tests:

```bash
go test ./runtime -run 'TestEmbeddingTrainerFitHardNegatives|TestEstimateHardNegativeTrainWorkload' -count=1
go test ./cmd/eos -run 'TestRunTrainEmbed|TestTrainEmbed|TestTrainEmbedPlan|TestTrainEmbedHardNegative' -count=1
```

Result: both passed.

Resource checks:

```bash
free -h
df -h /home/draco/work/eos
```

Result during investigation: 27 GiB RAM, 8.0 GiB swap, swap fully used, 114 GiB free on `/dev/sdd`.

Kernel log checks did not find a clear OOM kill for the original run window; available logs mostly showed repeated WSL/dxg messages, so OOM/resource kill remains plausible but not proven.

## Caveats / Residual Risk

- The exact external interruption mechanism is still unproven. The saved logs do not include signal/exit-code detail from the child beyond the outer `go run: exit status 255`.
- I did not retry the full 100k run.
- The bounded train probes prove command health at 64/256 rows, not full-run completion.
- Pre-existing uncommitted progress/observability changes should be reviewed by the parent before treating them as a checkpointable fix.

## Checkpoint Candidate

No checkpoint candidate from this pass. No source patch was authored here.

If the pre-existing progress hardening is accepted and verified by its owner, it may be a separate checkpoint candidate, but it is not this debugger's change.

## Arbiter Next Action

Do not promote or score the failed candidate. Do not compact-evaluate it.

Next runnable descriptor: rerun the 100k scale pilot only after root/user approval, using a fresh run root and explicit observability/resource controls:

```bash
EOS_PROGRESS_EVERY=25 \
EOS_STATUS_EVERY_SECONDS=30 \
EOS_TRAIN_RECLAIM_EVERY=4 \
EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1 \
EOS_TRAIN_ENABLE_FAST_GELU=1 \
<same scale-pilot wrapper invocation that produced runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z, with the same 100k JSONL SHA eb5dad81341cba069d544b8deb91df4ee73dc3fcefe91128887704632b9d878a>
```

Recommended descriptor fields:

- Objective: fresh 100k rerun with explicit progress/status and resource snapshots.
- Constraints: one full train attempt only; abort if swap is already saturated or available RAM is materially below the prior preflight; do not promote unless `train.metrics.json`, restore-best evidence, final eval, hard eval, dense gates, and compact gates all pass.
- Verification: require `train.metrics.json` with `restore_best=true`, `actual_eval_passes>=3`, nonzero `steps_run`, changed train checkpoint hash, and retained progress/status logs.
