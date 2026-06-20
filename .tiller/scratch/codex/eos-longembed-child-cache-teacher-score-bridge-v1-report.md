# eos-longembed-child-cache-teacher-score-bridge-v1 Report

## Outcome

Implemented and validated a LongEmbed child-cache teacher-score bridge.

New script:

- `scripts/score_longembed_child_cache_teacher_bridge.py`

New focused tests:

- `scripts/test_score_longembed_child_cache_teacher_bridge.py`

Generated artifacts:

- `runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/`

The generated training-ready JSONL preserves all input examples and attaches
`teacher_scores` only when both Qwen3 and mxbai have full coverage and each
teacher ranks the labeled positive top-1. Examples that fail teacher agreement
are kept with `teacher_scores` omitted. `quality_claim=false` is preserved in
the bridge manifests.

Follow-up correction: regenerated the existing run manifests after renaming the
averaged scored-row keep metric to `averaged_scores.agreement_keep_rate`.
`averaged_scores.positive_top1_rate` now reports the positive-top1 rate among
averaged scored rows, which is `1.0` when any averaged rows exist because the
bridge agreement filter guarantees it.

## Distillation

- Cache child chunks are deterministically reconstructed from each cache
  manifest and dataset `corpus.jsonl`; chunk parameters are read from
  `manifest.json`, not hardcoded.
- Query vectors resolve by `metadata.query_id` first, with stable query text as
  fallback.
- Parent candidate documents resolve by stable corpus text, then score as the
  max cosine/dot score across the parent's child vectors.
- Mixed `qmsum` and `2wikimqa` inputs route by metadata/source hints and fail
  loudly if ambiguous.
- The real curated bundle had complete vector coverage for both teachers; rows
  were cleared only for teacher disagreement on labeled-positive top-1.

## Files Changed / Inspected

Changed:

- `scripts/score_longembed_child_cache_teacher_bridge.py`
- `scripts/test_score_longembed_child_cache_teacher_bridge.py`
- `.tiller/scratch/codex/eos-longembed-child-cache-teacher-score-bridge-v1-report.md`

Inspected:

- `scripts/score_teacher_with_vector_cache.py`
- `scripts/score_span_teacher_with_child_cache.py`
- `scripts/export_qwen3_retrieval_vectors.py`
- `scripts/test_export_qwen3_retrieval_vectors.py`
- `scripts/test_curate_longembed_frontier_hard_negatives.py`
- `cmd/eos/main.go`
- `runs/eos-longembed-fusion-topk-remine-v1-20260620T105437Z/curated/{train,eval}-hard-negatives.jsonl`
- `datasets/longembed-official/{qmsum,2wikimqa}/{corpus,queries}.jsonl`
- `runs/external-vector-caches/{qwen3-0.6b-longembed-real-doc20-128d,mxbai-large-longembed-real-doc20-128d}/{qmsum,2wikimqa}/`

## Generated Artifact Paths

Run root:

- `runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z`

Train:

- `train.agreement-scored.jsonl`
- `train.filtered.jsonl`
- `train.manifest.json`
- `train.audit-summary.json`
- `train.filter-summary.json`
- `train.average-scores.jsonl`
- `train.per-teacher-scores.jsonl`

Eval:

- `eval.agreement-scored.jsonl`
- `eval.filtered.jsonl`
- `eval.manifest.json`
- `eval.audit-summary.json`
- `eval.filter-summary.json`
- `eval.average-scores.jsonl`
- `eval.per-teacher-scores.jsonl`

## Counts

Train bridge manifest:

- examples: 21
- with averaged agreement `teacher_scores`: 17
- averaged `agreement_keep_rate`: 0.809524
- averaged `positive_top1_rate`: 1.0
- cleared: 4
- missing coverage: 0
- teacher disagreement: 4
- dataset split: `2wikimqa` 10/10 scored, `qmsum` 7/11 scored
- Qwen3 complete/top1: 21/19, mean margin 0.103322
- mxbai complete/top1: 21/17, mean margin 0.076820

Eval bridge manifest:

- examples: 5
- with averaged agreement `teacher_scores`: 4
- averaged `agreement_keep_rate`: 0.800000
- averaged `positive_top1_rate`: 1.0
- cleared: 1
- missing coverage: 0
- teacher disagreement: 1
- dataset split: `2wikimqa` 2/2 scored, `qmsum` 2/3 scored
- Qwen3 complete/top1: 5/4, mean margin 0.096316
- mxbai complete/top1: 5/5, mean margin 0.067053

Audit/filter:

- Train audit: examples 21, scored 17, missing 4, positive_top1_rate 1.0, mean margin 0.123581
- Train filter: examples 21, scored 17, missing 4, kept 17, cleared 0, dropped 0
- Eval audit: examples 5, scored 4, missing 1, positive_top1_rate 1.0, mean margin 0.110334
- Eval filter: examples 5, scored 4, missing 1, kept 4, cleared 0, dropped 0

JSONL line counts:

- `train.agreement-scored.jsonl`: 21
- `train.filtered.jsonl`: 21
- `train.average-scores.jsonl`: 153
- `train.per-teacher-scores.jsonl`: 378
- `eval.agreement-scored.jsonl`: 5
- `eval.filtered.jsonl`: 5
- `eval.average-scores.jsonl`: 36
- `eval.per-teacher-scores.jsonl`: 90

## Verification Commands / Results

Passed:

```bash
python3 scripts/test_score_longembed_child_cache_teacher_bridge.py
```

Result: `Ran 4 tests ... OK`. Re-run after the manifest semantic correction
also passed and now asserts that agreement keep-rate is separate from averaged
positive-top1 rate.

Passed:

```bash
python3 scripts/test_export_qwen3_retrieval_vectors.py
```

Result: `Ran 6 tests ... OK`.

Passed:

```bash
python3 scripts/test_curate_longembed_frontier_hard_negatives.py
```

Result: `Ran 3 tests ... OK`.

Passed:

```bash
python3 -m py_compile scripts/score_longembed_child_cache_teacher_bridge.py scripts/test_score_longembed_child_cache_teacher_bridge.py
```

Result: no output, exit 0. Re-run after the manifest semantic correction also
passed.

Scoring commands passed:

```bash
python3 scripts/score_longembed_child_cache_teacher_bridge.py --hard-negatives runs/eos-longembed-fusion-topk-remine-v1-20260620T105437Z/curated/train-hard-negatives.jsonl --dataset-dir qmsum=datasets/longembed-official/qmsum --dataset-dir 2wikimqa=datasets/longembed-official/2wikimqa --teacher-cache qwen3=runs/external-vector-caches/qwen3-0.6b-longembed-real-doc20-128d --teacher-cache mxbai=runs/external-vector-caches/mxbai-large-longembed-real-doc20-128d --output-jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.agreement-scored.jsonl --manifest runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.manifest.json --scores-jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.average-scores.jsonl --teacher-scores-jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.per-teacher-scores.jsonl
```

Result: examples 21, with_teacher_scores 17, cleared 4.

```bash
python3 scripts/score_longembed_child_cache_teacher_bridge.py --hard-negatives runs/eos-longembed-fusion-topk-remine-v1-20260620T105437Z/curated/eval-hard-negatives.jsonl --dataset-dir qmsum=datasets/longembed-official/qmsum --dataset-dir 2wikimqa=datasets/longembed-official/2wikimqa --teacher-cache qwen3=runs/external-vector-caches/qwen3-0.6b-longembed-real-doc20-128d --teacher-cache mxbai=runs/external-vector-caches/mxbai-large-longembed-real-doc20-128d --output-jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.agreement-scored.jsonl --manifest runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.manifest.json --scores-jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.average-scores.jsonl --teacher-scores-jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.per-teacher-scores.jsonl
```

Result: examples 5, with_teacher_scores 4, cleared 1.

The train/eval scoring commands were re-run in place after the manifest
semantic correction. The regenerated manifests now contain:

- train `averaged_scores.agreement_keep_rate=0.8095238095238095`
- train `averaged_scores.positive_top1_rate=1.0`
- eval `averaged_scores.agreement_keep_rate=0.8`
- eval `averaged_scores.positive_top1_rate=1.0`

Audit/filter commands passed:

```bash
go run ./cmd/eos audit-teacher-scores --mode text runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.agreement-scored.jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.audit-summary.json
go run ./cmd/eos audit-teacher-scores --mode text runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.agreement-scored.jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.audit-summary.json
go run ./cmd/eos filter-teacher-scores --mode text runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.agreement-scored.jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.filtered.jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.filter-summary.json
go run ./cmd/eos filter-teacher-scores --mode text runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.agreement-scored.jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.filtered.jsonl runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/eval.filter-summary.json
```

Result: train kept 17/17 scored teacher-score rows and dropped 0 examples; eval
kept 4/4 scored teacher-score rows and dropped 0 examples.

## Caveats / Residual Risk

- No training was run, per descriptor.
- These are protected candidate-training artifacts, not benchmark evidence;
  generated bridge manifests set `quality_claim=false`.
- Cleared examples are agreement failures, not coverage failures. They remain
  hard-negative examples without soft teacher scores.
- Candidate text-to-parent matching is strict. If future curated files mutate
  corpus text, the bridge will fail or clear rather than guess.
- The Go filter did not unexpectedly remove examples or clear any averaged
  agreement scores.

## Checkpoint Candidate

Yes. This is a coherent verified slice: new source script, focused tests,
generated training-ready artifacts, Eos audit/filter summaries, and this report.
Do not include unrelated files. The generated run artifacts may be ignored by
Git, but they are present in the workspace.

## Arbiter Next Action

Use `runs/eos-longembed-child-cache-teacher-scores-v1-20260620T112726Z/train.filtered.jsonl`
as the next guarded training probe input if the probe should consume the
post-filter copy, or `train.agreement-scored.jsonl` if the unmodified bridge
output is preferred. Both preserve all 21 train examples and carry 17
agreement-filtered averaged teacher-score rows.
