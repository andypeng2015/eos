# eos-wrapper-pipestatus-zsh-fix-v1 Report

## Outcome
No source fix landed. The reported nonzero outer status is a one-off zsh invocation bug, not a bug in `scripts/train_manta_embed_v1_candidate.fw`.

The Ferrous wrapper runs child commands with `exec.Command` and `io.MultiWriter(os.Stdout, logFile)`; it does not use a shell pipeline, `tee`, `PIPESTATUS`, or zsh. The successful scale rerun's authoritative wrapper sidecars remain `candidate/status.json` and `candidate/status.jsonl`, whose final state is `completed` with `exit_code=0`.

## Distillation
- In bash, the first pipeline command status is `${PIPESTATUS[0]}`.
- In zsh, the pipeline status array is lowercase and 1-indexed: `${pipestatus[1]}`.
- `test ${PIPESTATUS[0]} -eq 0` under zsh expands the first operand to empty and returns `2` with `unknown condition: -eq`, contaminating an otherwise successful run.
- Pipeline status must be captured immediately after the pipeline because any subsequent command overwrites it.

Safe zsh pattern:

```zsh
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw 2>&1 | tee "$LOG"
st=${pipestatus[1]}
exit "$st"
```

Portable option when using bash semantics:

```bash
bash -lc 'ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw 2>&1 | tee "$LOG"; st=${PIPESTATUS[0]}; exit "$st"'
```

## Files Changed / Inspected
- Changed: `.tiller/scratch/codex/eos-wrapper-pipestatus-zsh-fix-v1-report.md`
- Inspected: `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-rerun-v2-report.md`
- Inspected: `runs/eos-s40-fiqa-clean-scale-pilot-rerun-v2-20260624T073806Z/logs/train-driver.outer.log`
- Inspected: `runs/eos-s40-fiqa-clean-scale-pilot-rerun-v2-20260624T073806Z/candidate/status.json`
- Inspected: `scripts/train_manta_embed_v1_candidate.fw`
- Searched: repo scripts, scratch reports, and rerun logs for `PIPESTATUS`, `pipestatus`, `tee`, and `pipefail`

## Verification Commands and Results
- `rg -n "PIPESTATUS|pipestatus|tee|pipefail|set -o pipefail" scripts .tiller/scratch/codex runs/eos-s40-fiqa-clean-scale-pilot-rerun-v2-20260624T073806Z/candidate runs/eos-s40-fiqa-clean-scale-pilot-rerun-v2-20260624T073806Z/logs -g '*.fw' -g '*.sh' -g '*.py' -g '*.md' -g '*.log' -g '*.json' -g '*.jsonl'`
  - Result: no `PIPESTATUS` usage in `scripts/`; hits are historical scratch reports or unrelated generated command snippets. The target `.fw` script contains no shell pipeline status handling.
- `zsh -fc 'true | tee /dev/null >/dev/null; test ${PIPESTATUS[0]} -eq 0'`
  - Result: exits `2`, prints `zsh:test:1: unknown condition: -eq`; reproduces the failure mode even though the pipeline command succeeds.
- `zsh -fc 'true | tee /dev/null >/dev/null; st=${pipestatus[1]}; print -r -- captured=$st; exit $st'`
  - Result: exits `0`, prints `captured=0`; proves successful inner command through `tee` returns zero with the correct zsh status array.
- `zsh -fc 'false | true; st=${pipestatus[1]}; print -r -- captured=$st all=${pipestatus[*]}; exit $st'`
  - Result: exits `1`, prints `captured=1 all=1 0`; proves the corrected zsh pattern preserves the first command's failure.
- `bash -lc 'false | true; st=${PIPESTATUS[0]}; printf "captured=%s now=%s\n" "$st" "${PIPESTATUS[0]}"; exit "$st"'`
  - Result: exits `1`, demonstrating the bash equivalent and also showing that later commands overwrite `PIPESTATUS`.

## Caveats / Residual Risk
- I did not run a full training job, per constraint.
- I did not run `ferrous-wheel lint/build` because no `.fw` source was changed.
- Historical scratch reports contain similar zsh `PIPESTATUS` mistakes; they are evidence notes, not active repo harness code.
- Any future parent/agent command that pipes a long run through `tee` should explicitly choose bash or use zsh's `${pipestatus[1]}` and snapshot it immediately.

## Checkpoint Candidate
No. Report-only diagnosis; no source behavior changed.

## Arbiter Next Action
Treat the scale rerun as completed based on wrapper status sidecars. For future long-run descriptors, use the safe zsh pattern above or run the outer command under bash when relying on `${PIPESTATUS[0]}`.
