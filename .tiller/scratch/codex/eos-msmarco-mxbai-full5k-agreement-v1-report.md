# eos-msmarco-mxbai-full5k-agreement-v1 Report

## Outcome

Completed mxbai-large teacher scoring on the exact same audited 5,000-row MS MARCO substrate used by the Qwen3 full5k run, then produced Qwen3/mxbai agreement evidence. No training rows, release rows, commercial training rows, or promotion artifacts were generated.

Run root:

- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/`

Primary artifacts:

- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/manifest.json`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.teacher-scores.jsonl`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/qwen3-mxbai-agreement-report.json`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/eos-audit-teacher-scores.summary.json`

## Distillation

The run scored exactly the audited teacher-prep substrate:

- candidate rows: `5,000`
- candidate score rows: `20,000`
- unique query texts encoded: `4,668`
- unique candidate texts encoded: `19,982`
- mxbai teacher: `mixedbread-ai/mxbai-embed-large-v1`
- query prefix: `Represent this sentence for searching relevant passages: `
- document prefix: empty
- score scale: `cosine_normalized_dot`
- runtime path: `.venv-qwen3`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `--local-files-only`
- device: CUDA on `NVIDIA GeForce RTX 5070 Ti`
- scoring runtime recorded in manifest: `38.31559896469116` seconds

Legal/product gates preserved in manifest and score rows:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

## Exact Scoring Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1.py \
  --candidates runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl \
  --qwen-scored runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl \
  --source-manifest runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json \
  --source-requests runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl \
  --qwen-manifest runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/manifest.json \
  --run-root runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z \
  --max-rows 5000 \
  --model-id mixedbread-ai/mxbai-embed-large-v1 \
  --device cuda \
  --batch-size 32 \
  --local-files-only
```

## mxbai Metrics

Helper manifest metrics:

- positive top-1 rate: `1.0`
- positive top-1 count: `5,000 / 5,000`
- positive mean margin over best negative: `0.42697490575392055`
- positive score mean: `0.7905918504914876`
- negative score mean: `0.31547572584775563`
- margin min / median / max: `0.008173221323659163` / `0.4329873891372529` / `0.6744317041605328`
- missing score rows: `0`
- global-dev-positive negative refs flagged: `81`
- rows with global-dev-positive negatives: `80`

Repo audit summary:

- examples: `5,000`
- scored examples: `5,000`
- missing examples: `0`
- candidates: `20,000`
- scored candidates: `20,000`
- positive top-1 rate: `1.0`
- positive mean rank: `1`
- positive mean margin: `0.42697490586936476`
- mean normalized entropy: `0.9814576259546813`
- label policy: `selected_positive_and_any_qrels_positive`

## Qwen3/mxbai Agreement Metrics

- both-positive-top1 count/rate: `4,999 / 5,000`, `0.9998`
- Qwen3 positive top1 count: `4,999`
- mxbai positive top1 count: `5,000`
- either-failed count: `1`
- margin correlation Pearson: `0.7324485685155065`
- Qwen3 margin mean / median / min: `0.5312749150125786` / `0.5429375695057388` / `-0.03996013876894722`
- mxbai margin mean / median / min: `0.42697490575392055` / `0.4329873891372529` / `0.008173221323659163`
- mxbai-minus-Qwen margin delta mean / median: `-0.10430000925865807` / `-0.10791568112474437`
- agreement-filter candidate count/rate: `4,919 / 5,000`, `0.9838`
- agreement-filter exclusions: `80` global-dev-positive-negative rows, `1` Qwen non-top1/low-margin row, `0` mxbai non-top1/low-margin rows

Single Qwen3 non-top1 row:

- row index: `1326`
- row id: `933c61d37141a528a07b13f3f3cbadc80941f57552e88c6cb297683affb14be4`
- query id: `509174`
- Qwen3 margin: `-0.03996013876894722`
- mxbai top1: `true`
- mxbai margin: `0.039638386173010076`

## Files Changed Or Created

Created scratch wrapper and handoff:

- `.tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1.py`
- `.tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1-report.md`

Ignored/generated run artifacts:

- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/manifest.json`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.teacher-scores.jsonl`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/qwen3-mxbai-agreement-report.json`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/eos-audit-teacher-scores.summary.json`

Inspected context:

- `AGENTS.md`
- `docs/eos-distillation-compact-default-spec.md`
- `.tiller/scratch/codex/eos-msmarco-qwen3-full5k-teacher-scoring-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`
- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1-report.md`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/manifest.json`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/reports/eos-audit-teacher-scores.summary.json`

## Verification Commands And Results

Wrapper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1.py
```

Result: exited `0`.

Preflight:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3/bin/python - <<'PY'
from sentence_transformers import SentenceTransformer
import json, torch
model_id='mixedbread-ai/mxbai-embed-large-v1'
model=SentenceTransformer(model_id, device='cuda', local_files_only=True)
emb=model.encode(['Represent this sentence for searching relevant passages: test query','test passage'], batch_size=2, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
print(json.dumps({'model_id': model_id, 'dim': int(emb.shape[1]), 'dtype': str(emb.dtype), 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0)}, sort_keys=True))
PY
```

Result: local/offline mxbai cache loaded successfully; embedding dim `1024`; CUDA available on `NVIDIA GeForce RTX 5070 Ti`.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_mxbai_full5k_agreement.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .sample.candidate_rows_scored == 5000 and .counts.teacher_score_rows == 20000 and .counts.missing_score_rows == 0' \
  runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/manifest.json
```

Result: `true`.

Score/scored JSONL line counts:

```bash
wc -l \
  runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.teacher-scores.jsonl \
  runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.scored.teacher-candidates.jsonl
```

Result: `20000` score rows and `5000` scored candidate rows.

Independent JSONL/agreement recount:

```bash
python3 - <<'PY'
# parsed mxbai score/scored JSONL, asserted counts against manifest,
# and recomputed Qwen3/mxbai agreement counts from scored rows
PY
```

Result: `score_rows=20000`, `scored_rows=5000`, `flag_refs=81`, `flagged_rows=80`, `mxbai_top1=5000`, `qwen_top1=4999`, `both_top1=4999`, `agreement_candidates=4919`; single Qwen non-top1 row is row index `1326`, row id `933c61d37141a528a07b13f3f3cbadc80941f57552e88c6cb297683affb14be4`, query id `509174`.

Repo audit:

```bash
go run ./cmd/eos audit-teacher-scores \
  --qrels runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/dev.tsv \
  --corpus runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/corpus.jsonl \
  runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/artifacts/msmarco-passage.mxbai-large.scored.teacher-candidates.jsonl \
  runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/eos-audit-teacher-scores.summary.json
```

Result: exited `0`; `examples=5000`, `scored=5000`, `missing=0`, `positive_top1_rate=1.000000`, `mean_margin=0.426975`, `mean_normalized_entropy=0.981458`.

Final git status:

```text
## main...origin/main
```

With ignored-path status for this slice:

```text
## main...origin/main
!! .tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1.py
!! runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/
```

## Caveats Or Residual Risk

- This is teacher-scoring and agreement evidence only, not training data and not a promotion artifact.
- The mxbai wrapper is a scratch helper, not production source.
- Candidate negatives are deterministic corpus-ID negatives from the prep audit, not mined hard negatives.
- The `81` flagged global-dev-positive negative refs across `80` rows are not same-query leaks, but they are excluded from the agreement-filter candidate count and must be filtered, relabeled, or downweighted before any future training-row generation.
- The single Qwen3 non-top1 row is excluded from the agreement-filter candidate count. mxbai scored that same row as positive top1 with margin `0.039638386173010076`.
- The agreement-filter count uses low-margin threshold `0.0`; any stricter future margin gate will reduce the candidate count.
- MS MARCO gates remain restrictive: `release_train_allowed=false` and `commercial_use_allowed=false`.

## Checkpoint Candidate

Yes, but only for the tiny ignored scratch wrapper/report slice if root wants to preserve the procedure:

- `.tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1.py`
- `.tiller/scratch/codex/eos-msmarco-mxbai-full5k-agreement-v1-report.md`

Do not checkpoint score JSONL, run data, model cache, or raw/candidate corpus artifacts unless ignored-artifact policy changes explicitly.

## Recommended Next Action

Use `runs/eos-msmarco-mxbai-full5k-agreement-v1-20260624T181635Z/reports/qwen3-mxbai-agreement-report.json` as the gate for the next descriptor. The next bounded step can generate research-only agreement-filtered distillation candidates from the `4,919` rows only if it preserves `release_train_allowed=false`, `commercial_use_allowed=false`, excludes the `80` globally dev-positive-negative rows and the single Qwen3 non-top1 row, and records any stricter margin threshold.

## Arbiter Next Action

Proceed to agreement-filtered distillation-row planning/generation only under research-only gates, using the mxbai run root above, `missing_score_rows=0`, `release_train_allowed=false`, `commercial_use_allowed=false`, `agreement_filter_candidate_count=4919`, and explicit exclusion of global-dev-positive negatives plus the Qwen3 non-top1 row.
