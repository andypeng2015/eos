# eos-long-context-fusion-duplicate-recipe-fix-v1 Report

## Outcome

Fixed duplicate extra fusion recipe handling in `scripts/eval_eos_long_context_wedge.fw`.

`EOS_LC_WEDGE_PARENT_SPAN_FUSION_EXTRA_RECIPES` now treats exact duplicate recipes as idempotent and ignores them. Conflicting duplicates with the same method name but different recipe settings still fail deterministically. This prevents a rerun from failing during final fusion assembly after expensive Eos artifacts have already been generated when the extra recipe list repeats an existing default recipe exactly.

## Distillation

The failed QMSum run hit `parentSpanFusionRecipes()` after expensive direct/token-span/sparse artifacts were complete. The previous parser rejected any duplicate method name in extra recipes, including an exact duplicate of a default recipe. The new behavior is:

- default recipe table duplicates remain fatal;
- exact duplicate extra recipes are ignored;
- same method name with different `k`, `lambda`, or `protect` remains fatal.

A cheap `EOS_LC_WEDGE_RECIPE_SELF_TEST=1` path was added to verify this parser behavior without building Eos artifacts or running LongEmbed retrieval.

## Files Changed / Inspected

Changed:

- `scripts/eval_eos_long_context_wedge.fw`

Inspected:

- `.tiller/scratch/codex/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-report.md`
- `scripts/eval_eos_long_context_wedge.fw`
- `scripts/smoke_eos_sparse_long_context_retrieval.fw`
- `runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/manifest.json` was provided as context but did not need mutation.

## Verification Commands And Results

```bash
ferrous-wheel fmt scripts/eval_eos_long_context_wedge.fw
```

Result: exit 0. Ferrous Wheel printed formatted source to stdout; it does not rewrite the file in this invocation.

```bash
ferrous-wheel lint scripts/eval_eos_long_context_wedge.fw
```

Result: exit 0, no diagnostics.

```bash
EOS_LC_WEDGE_RECIPE_SELF_TEST=1 ferrous-wheel run scripts/eval_eos_long_context_wedge.fw
```

Result: exit 0, printed `parent-span fusion recipe self-test passed`.

```bash
ferrous-wheel build scripts/eval_eos_long_context_wedge.fw
```

Result: exit 0, built `eval_eos_long_context_wedge`; generated binary was removed after the check.

```bash
git diff --check
```

Result: exit 0.

## Caveats / Residual Risk

No expensive LongEmbed evaluation was rerun. Verification is parser-focused and build/lint-based, which matches the requested constraint to avoid expensive evals. The existing full harness path should now proceed past the exact duplicate extra recipe case; conflicting duplicate method definitions still intentionally stop the run.

## Checkpoint Candidate

Yes. This is a clean scoped source-only fix with focused self-test, lint/build success, and `git diff --check` passing.

## Arbiter Next Action

Checkpoint this source fix, then rerun only the previously failed final assembly or next bounded sparse-encoder q4 investigation when cost is acceptable.
