# eos-s40-balanced-boundary-macro-ndcg-ablation-v1 Report

## Outcome

Completed two allowed one-epoch ablations from the current default s40 family artifact/tokenizer:

- Run root: `runs/eos-s40-balanced-boundary-macro-ndcg-ablation-v1-20260624T113003Z/`
- Arm A: `arm_a_balanced_clean`
- Arm B: `arm_b_balanced_boundary`

Dense exploration result: **FAIL / no-go**. Neither arm reached the continuation signal (`macro nDCG@10 delta >= +0.0004`), and neither reached the exploration pass gate (`macro nDCG@10 delta >= +0.0010`). Per-dataset nDCG/recall floors and macro recall floors passed, but macro nDCG did not lift.

No promotion was performed. No compact eval was run.

## Distillation

- Arm A isolated source balance: 30,000 rows, exactly 10,000 each for SciFact/NFCorpus/FiQA, teacher rows `0`.
- Arm B added train-only boundary pressure while preserving balance: 30,000 rows, exactly 10,000 each; 96 SciFact boundary rows and 96 NFCorpus boundary rows; teacher rows `0`.
- The descriptor-specified NFCorpus compact-mined file audited poorly for strict test doc-id exclusion: only 21 rows survived. Arm B therefore used the train-only fallback `runs/eos-nfcorpus-compact-top10-diff-and-non-test-repair-20260617T000000Z/data/nfcorpus-top10-competitor-repair.train.jsonl`, recorded in the data manifest.
- Balanced clean data gave a tiny macro nDCG lift (`+0.000056835`) and recall lift (`+0.000735574`), but below the `+0.0004` continuation threshold.
- Boundary pressure regressed macro nDCG (`-0.000103905`) while improving macro recall (`+0.000835572`).
- Recommendation: **no-go** for these arms; do not promote, do not compact-evaluate. Next useful ablation should change objective or data construction more materially rather than replaying this balance/boundary recipe.

## Files Changed / Generated

- Scratch report: `.tiller/scratch/codex/eos-s40-balanced-boundary-macro-ndcg-ablation-v1-report.md`
- Scratch helpers:
  - `.tiller/scratch/codex/eos_s40_balanced_boundary_ablation_builder.py`
  - `.tiller/scratch/codex/eos_s40_balanced_boundary_gate.py`
- Run-root pointer: `.tiller/scratch/codex/eos-s40-balanced-boundary-macro-ndcg-ablation-v1.run-root`
- Data manifest: `runs/eos-s40-balanced-boundary-macro-ndcg-ablation-v1-20260624T113003Z/data/balanced-boundary-data-manifest.json`
- Train JSONLs:
  - `data/arm_a_balanced_clean.train.jsonl`, sha256 `9a228d781423512f11b183acf1c092d36613ca0056d553ef9cda61abbb81af67`
  - `data/arm_b_balanced_boundary.train.jsonl`, sha256 `67dcac87b9c7b90d78482506965ce355c748cd6457e7a7933208a599ad7b027d`
- Dense gate artifacts:
  - `gates/arm_a_balanced_clean.dense-gate.json`
  - `gates/arm_a_balanced_clean.dense-gate.md`
  - `gates/arm_b_balanced_boundary.dense-gate.json`
  - `gates/arm_b_balanced_boundary.dense-gate.md`
- Arm outputs under:
  - `arm_a_balanced_clean/candidate/`
  - `arm_a_balanced_clean/candidate-scoreboard/`
  - `arm_b_balanced_boundary/candidate/`
  - `arm_b_balanced_boundary/candidate-scoreboard/`

## Data Manifest Summary

- Base audited candidate rows available: FiQA `60,789`, NFCorpus `13,160`, SciFact `13,106`.
- Exact test query-positive text pair rejections: NFCorpus `208`, SciFact `12`.
- Test audit set sizes:
  - FiQA: query IDs `648`, doc IDs `1,706`, query-positive text pairs `1,705`
  - NFCorpus: query IDs `323`, doc IDs `3,128`, query-positive text pairs `12,288`
  - SciFact: query IDs `300`, doc IDs `283`, query-positive text pairs `339`
- Arm A after filtering: `0` test-overlap rows, `0` teacher-score rows, negative-count distribution `{1: 27479, 3: 2521}`.
- Arm B after filtering: `0` test-overlap rows, `0` teacher-score rows, negative-count distribution `{1: 27307, 3: 2501, 4: 192}`.
- Arm B boundary audit: descriptor-specified NFCorpus file had `21` eligible rows; fallback train-only NFCorpus file had `99`, selected `96`; SciFact had `624`, selected `96`.

## Arm Results

Arm A train/eval:

- `restore_best=true`, `actual_eval_passes=3`, `epochs_completed=1`, `restored_best=true`
- final train loss `1.5229707`
- final eval loss `0.12377463`, score margin `0.24369821`, ROC AUC `0.8502306`
- train pairs/s `1559.29`
- sealed SHA256 `78b0723ff563b191501ed6284df753660257da6220e4cdd1a7efebb6a1f67d35`
- final eval-only optimizer updates `0`; hard eval-only optimizer updates `0`

Arm A dense:

| dataset | nDCG@10 | R@100 | nDCG delta | R@100 delta |
| --- | ---: | ---: | ---: | ---: |
| SciFact | `0.564804600` | `0.796444444` | `+0.000266685` | `+0.000000000` |
| NFCorpus | `0.205588484` | `0.244169909` | `-0.000157484` | `+0.002103842` |
| FiQA | `0.121322244` | `0.351781089` | `+0.000061303` | `+0.000102881` |

- macro nDCG delta `+0.000056835`
- macro recall delta `+0.000735574`
- exploration pass `false`; continuation signal `false`

Arm B train/eval:

- `restore_best=true`, `actual_eval_passes=3`, `epochs_completed=1`, `restored_best=true`
- final train loss `1.5162432`
- final eval loss `0.1237771`, score margin `0.24369496`, ROC AUC `0.8502463`
- train pairs/s `1573.77`
- sealed SHA256 `c6cb9bd42239b916c9d7bedf27d0e84c7baf59f11659b0a95b50e0c5efc49ed8`
- final eval-only optimizer updates `0`; hard eval-only optimizer updates `0`

Arm B dense:

| dataset | nDCG@10 | R@100 | nDCG delta | R@100 delta |
| --- | ---: | ---: | ---: | ---: |
| SciFact | `0.564328284` | `0.796444444` | `-0.000209632` | `+0.000000000` |
| NFCorpus | `0.205596240` | `0.243955500` | `-0.000149728` | `+0.001889432` |
| FiQA | `0.121308585` | `0.352295493` | `+0.000047645` | `+0.000617284` |

- macro nDCG delta `-0.000103905`
- macro recall delta `+0.000835572`
- exploration pass `false`; continuation signal `false`

## Verification Commands / Results

- `python3 -m json.tool <data-manifest>`: PASS.
- `python3 -m json.tool <arm_a dense-gate.json>` and `<arm_b dense-gate.json>`: PASS.
- `python3 -m py_compile .tiller/scratch/codex/eos_s40_balanced_boundary_ablation_builder.py`: PASS.
- `python3 -m py_compile .tiller/scratch/codex/eos_s40_balanced_boundary_gate.py`: PASS.
- Both guarded runs executed `go test ./cmd/eos ./runtime ./models -count=1`: PASS.
- Both guarded runs produced `train.metrics.json` with `config.restore_best=true` and `workload.actual_eval_passes=3`.
- Both guarded runs produced `final-eval.metrics.json` and `hard-eval.metrics.json` with `mode=eval` and `profile_delta.optimizer_updates=0`.
- Both scoreboards parsed and produced dense rows for SciFact, NFCorpus, and FiQA.
- Guarded scoreboard floor gate with tolerance `0.003` passed for both arms; custom exploration gate failed both arms on macro nDCG lift.

## Caveats / Residual Risk

- The working tree was dirty before the run, so guarded training used `EOS_GUARD_ALLOW_DIRTY=1` / `EOS_ALLOW_DIRTY=1`. No source/default/asset edits were made by the training commands.
- Two earlier data-only preflight roots were created while correcting the qrels audit parser: `runs/eos-s40-balanced-boundary-macro-ndcg-ablation-v1-20260624T112655Z/` and `runs/eos-s40-balanced-boundary-macro-ndcg-ablation-v1-20260624T112751Z/`. The reported experiment root is the corrected final root ending `T113003Z`.
- The strict NFCorpus boundary ID audit left only 21 rows from the descriptor-specified compact file; arm B used a recorded train-only fallback to satisfy the 64-128 row target.
- No compact/TurboQuant eval was run because dense exploration failed.

## Checkpoint Candidate

Yes, report/evidence only. Do not checkpoint model promotion or default asset movement.

## Arbiter Next Action

No-go. Do not promote and do not compact-evaluate either arm. Close this descriptor as a completed negative ablation. Recommended next action is a new ablation that changes objective/data pressure more substantially, because simple source balance plus small train-only boundary replay did not produce macro nDCG lift.
