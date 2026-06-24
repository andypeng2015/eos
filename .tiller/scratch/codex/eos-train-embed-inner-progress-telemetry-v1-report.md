# eos-train-embed-inner-progress-telemetry-v1

## Outcome

Implemented lightweight inner progress telemetry for `manta train-embed` / `eos train-embed` using the existing `--progress-every` path.

Future wrapper-launched candidate runs now default `EOS_PROGRESS_EVERY` to `100`, so `scripts/train_manta_embed_v1_candidate.fw` passes `--progress-every 100` unless overridden. Operators can still set `EOS_PROGRESS_EVERY=0` to disable inner progress or another positive value for a different cadence.

## Distillation

- Existing runtime batch progress already existed, but the wrapper default disabled it with `EOS_PROGRESS_EVERY=0`.
- `EmbeddingTrainProgress` now includes `phase`, `eval_pass`, `eval_examples`, and `eval_pairs`.
- Train progress lines now include `phase=train`.
- Eval-only, per-epoch eval, step-triggered eval, and final eval passes emit `phase=eval_start` and `phase=eval_done` under the same `--progress-every` flag.
- CLI emits a `phase=fit_start` line before loading/fitting when `--progress-every > 0`, so future fatal exits after plan estimation have a clearer last-known command phase even if no optimizer step completes.
- No active run artifacts were touched, and no exit255 root-cause debugging was performed.

## Files Changed

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`
- `runtime/embedding_train_runner.go`
- `runtime/embedding_train_runner_test.go`
- `scripts/train_manta_embed_v1_candidate.fw`
- `.tiller/scratch/codex/eos-train-embed-inner-progress-telemetry-v1-report.md`

## Verification Commands / Results

- `go test ./runtime -run 'TestEmbeddingTrainerFitContrastiveReports(Progress|EvalProgress)$'`
  - Result: pass (`ok m31labs.dev/eos/runtime`).
- `go test ./cmd/eos -run 'TestRunTrainEmbedProgressEveryPrintsInnerProgress|TestRunTrainEmbedEvalOnlyUsesSingleContrastiveDataset'`
  - Result: pass (`ok m31labs.dev/eos/cmd/eos`).
- `go test ./runtime ./cmd/eos`
  - Result: pass (`runtime` 131.185s, `cmd/eos` 74.449s).
- `ferrous-wheel lint scripts/train_manta_embed_v1_candidate.fw`
  - Result: pass.
- `ferrous-wheel emit scripts/train_manta_embed_v1_candidate.fw >/tmp/train_manta_embed_v1_candidate.go && go test /tmp/train_manta_embed_v1_candidate.go`
  - Result: pass (`[no test files]`).
- `ferrous-wheel build scripts/train_manta_embed_v1_candidate.fw -o /tmp/train_manta_embed_v1_candidate_inner_progress_smoke`
  - Result: pass.
- `git diff --check -- runtime/embedding_train_runner.go runtime/embedding_train_runner_test.go cmd/eos/main.go cmd/eos/main_test.go scripts/train_manta_embed_v1_candidate.fw`
  - Result: pass.

## Caveats / Residual Risk

- This patch does not diagnose `blocked_training_exit255` for `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z`; a separate debugger owns that lane.
- If the process is killed by SIGKILL or exits fatally before stdout flushes, only progress emitted before the fatal event can help. The new `fit_start` line narrows that window after plan estimation, but it cannot make a hard process kill self-report.
- Batch progress still emits only after a completed optimizer step, by design. This preserves signal and avoids per-batch spam unless `--progress-every 1` is explicitly requested.

## Checkpoint Candidate

Yes. This is a scoped telemetry improvement with focused and broader package verification.

Suggested checkpoint paths:

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`
- `runtime/embedding_train_runner.go`
- `runtime/embedding_train_runner_test.go`
- `scripts/train_manta_embed_v1_candidate.fw`

## Arbiter Next Action

Checkpoint the verified telemetry patch. Future scale-pilot wrapper runs should leave `EOS_PROGRESS_EVERY` unset for the new default cadence, or set it explicitly when a different cadence is needed. Continue exit255 root-cause work in the separate debugger lane.
