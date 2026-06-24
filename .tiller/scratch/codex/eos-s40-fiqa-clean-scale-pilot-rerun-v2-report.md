# eos-s40-fiqa-clean-scale-pilot-rerun-v2 Report

## Outcome
Completed one controlled full rerun at `origin/main@efe4b02` using a fresh run root:

`/home/draco/work/eos/runs/eos-s40-fiqa-clean-scale-pilot-rerun-v2-20260624T073806Z`

The previous `train-embed` exit-255 failure did not reproduce. The wrapper completed data/tokenization, package tests/build, 2-epoch hard-negative training, final eval-only, hard eval-only, and full-corpus retrieval metrics. No model was promoted and no default was updated.

Dense exploration result: **FAIL**. Per-dataset nDCG and recall floors passed, and macro recall passed, but macro nDCG delta was `-0.000116608`, below the required `+0.0010` lift. Because dense exploration did not pass, I stopped after reporting and do not recommend compact eval for this candidate.

## Distillation
- Working tree was clean before launch and still clean after report generation.
- HEAD and `origin/main` were both `efe4b02 improve(embed): add train progress telemetry and capture eval evidence`.
- No obvious pre-existing `manta train-embed` process was running before launch.
- Requested controls were used: `EOS_PROGRESS_EVERY=25`, `EOS_STATUS_EVERY_SECONDS=30`, `EOS_TRAIN_RECLAIM_EVERY=4`.
- `EOS_TRAIN_RECLAIM_EVERY` is supported by `runtime/embedding_train_runner.go`; this run used reclaim interval `4`.
- New telemetry was present in `candidate/logs/train.log`: `phase=fit_start`, repeated `phase=train` every 25 batches, `phase=eval_start`, and `phase=eval_done` through epoch 2.
- Wrapper status sidecars were captured at `candidate/status.json` and `candidate/status.jsonl`; final phase was `Full-corpus retrieval metrics gate`, `state=completed`, `exit_code=0`.
- The outer zsh wrapper command returned code 1 after success because it referenced `PIPESTATUS[0]`, which is not set in zsh. Wrapper status, logs, sealed artifact, and final output show the candidate flow itself completed successfully.

## Files Changed / Generated
- Scratch report: `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-rerun-v2-report.md`
- Run-root pointer: `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-rerun-v2.run-root`
- Run root: `runs/eos-s40-fiqa-clean-scale-pilot-rerun-v2-20260624T073806Z/`
- Copied scale train data under run root: `data/train-mixed-clean-fiqa-hardneg-capped-100k.jsonl`
- Wrapper logs/status/metrics under: `candidate/`
- Final sealed artifact: `candidate/eos-embed-v1.sealed.mll`
- Sealed artifact sha256: `179d03f717a0ab3a9d19f6993e09add52a3ed29661fc2940c779d893680bb3dd`
- Dense retrieval leaderboard: `candidate/retrieval/eos-embed-v1-retrieval-20260624T110318Z/leaderboard.tsv`
- Scoreboard derived from wrapper retrieval metrics: `candidate-scoreboard-from-wrapper/scoreboard.json`
- Dense exploration gate evidence: `gates/dense-exploration-vs-current-default.json` and `.log`
- Native per-dataset floor gate logs: `gates/native-ndcg-floor-vs-current-default.log`, `gates/native-recall-floor-vs-current-default.log`
- Resource snapshots: `logs/preflight-resource.txt`, `logs/postflight-resource.txt`

## Verification Commands and Results
- `git status --short` before launch: clean.
- `git rev-parse --short HEAD origin/main`: both `efe4b02`.
- Existing train process check: no pre-existing `manta train-embed` found.
- Preflight resources: 27 GiB RAM, about 16 GiB available, swap already saturated at 8 GiB/8 GiB, root filesystem about 118 GiB free before launch. I proceeded because RAM headroom was adequate.
- Wrapper command saved in `logs/train-driver.outer.log`; run controls and full environment are preserved there.
- Wrapper final status: `candidate/status.json` has `phase=Full-corpus retrieval metrics gate`, `state=completed`, `exit_code=0`, `heartbeat_count=12`, `elapsed_seconds=380` for the final retrieval phase.
- Go/package gate: wrapper ran `go test ./cmd/eos ./runtime ./models -count=1` successfully.
- Tokenization/workload counts:
  - train hard-negative examples: `100000`
  - eval pair examples: `1670`
  - hard eval pair examples: `1672`
  - planned epochs: `2`; completed epochs: `2`
  - train batches per epoch: `3125`
  - actual eval passes: `4`
- Train metrics from `candidate/train.metrics.json`:
  - summary: `epochs_completed=2`, `steps_completed=8777`, `steps_run=6250`, `best_epoch=2`, `best_step=8777`, `restored_best=true`, `stopped_early=false`
  - final train loss: `1.5205898`
  - final eval: loss `0.12381401`, score margin `0.24434096`, pair accuracy `0.6257485`, threshold accuracy `0.76586825`, ROC AUC `0.8518097`, top1 `0.9466192`, MRR `0.9727165`, pair count `1670`
  - throughput: elapsed `11550.1179s`, examples/s `17.8942`, pairs/s `1561.9221`, train pairs/s `1585.7177`
- Hard eval-only metrics from `candidate/hard-eval.metrics.json`:
  - loss `0.1215277`, score margin `0.26435357`, pair accuracy `0.6267942`, threshold accuracy `0.7565789`, ROC AUC `0.83127743`, top1 `0.92808217`, MRR `0.9628995`, pair count `1672`
  - eval-only gate passed with optimizer updates `0`.
- Dense retrieval metrics from wrapper leaderboard:
  - SciFact cuda: nDCG@10 `0.564328`, recall@100 `0.796444`, MRR@10 `0.542263`
  - NFCorpus cuda: nDCG@10 `0.205484`, recall@100 `0.243763`, MRR@10 `0.334773`
  - FiQA cuda: nDCG@10 `0.121383`, recall@100 `0.350238`, MRR@10 `0.154676`
- Native per-dataset floor checks against current-default scoreboard:
  - `manta gate-scoreboard -datasets scifact,nfcorpus,fiqa -metrics ndcg_at_10 -tolerance 0.002 ...`: PASS, 3 checks.
  - `manta gate-scoreboard -datasets scifact,nfcorpus,fiqa -metrics recall_at_100 -tolerance 0.003 ...`: PASS, 3 checks.
- Custom exploration gate against current-default scoreboard:
  - macro nDCG candidate `0.297065000`, baseline `0.297181608`, delta `-0.000116608`; required `>= +0.001000000`; FAIL.
  - macro recall candidate `0.463481667`, baseline `0.463396240`, delta `+0.000085426`; required `>= -0.001000000`; PASS.
  - per-dataset floors all PASS:
    - SciFact nDCG delta `-0.000209916`, recall delta approximately `0.000000`
    - NFCorpus nDCG delta `-0.000261968`, recall delta `+0.001696933`
    - FiQA nDCG delta `+0.000122059`, recall delta `-0.001440209`
- Postflight resources: 27 GiB RAM, about 14 GiB available, swap still saturated, root filesystem about 103 GiB free. No persistent long training process remained.

## Caveats / Residual Risk
- The outer command's process exit was contaminated by the zsh `PIPESTATUS[0]` issue after wrapper success. Treat `candidate/status.json`, `status.jsonl`, final logs, metrics, and sealed artifact as authoritative for run completion.
- Swap was saturated before and after the run. The run completed despite that, but future long training descriptors should consider freeing swap/resource pressure first if reproducibility or wall time matters.
- The generated `candidate-scoreboard-from-wrapper` scoreboard is derived from the wrapper retrieval leaderboard and metrics files; it does not re-run retrieval. This avoids unnecessary duplicate dense eval while preserving gateable evidence.
- This candidate should not be promoted and should not proceed to compact eval because dense exploration failed the macro nDCG lift gate.

## Checkpoint Candidate
Yes, report/evidence only. No source checkpoint and no model promotion.

## Arbiter Next Action
Do not promote. Do not run compact eval for this candidate. Close this scale rerun as a completed diagnostic: the exit-255 failure did not recur, telemetry is sufficient, but the clean-scale FiQA candidate failed dense exploration on macro nDCG. Recommended next descriptor is a small postmortem/objective-or-data adjustment plan rather than compact evaluation, for example `eos-s40-fiqa-clean-scale-postmortem-v1` or an objective/data ablation descriptor targeting macro nDCG lift.
