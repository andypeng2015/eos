# eos-msmarco-qwen3-full5k-teacher-scoring-v1 Report

## Outcome

Completed the full audited 5,000-row MS MARCO Qwen3 teacher-scoring pass using local/offline `Qwen/Qwen3-Embedding-0.6B`. This produced agreement/margin evidence only. No training rows, release rows, commercial rows, or promotion artifacts were generated.

Run root:

- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/`

Primary artifacts:

- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/manifest.json`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.teacher-scores.jsonl`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/reports/eos-audit-teacher-scores.summary.json`

## Distillation

The run scored exactly the audited teacher-prep substrate:

- candidate rows: `5,000`
- candidate score rows: `20,000`
- unique query texts encoded: `4,668`
- unique candidate texts encoded: `19,982`
- teacher: `Qwen/Qwen3-Embedding-0.6B`
- score scale: `cosine_normalized_dot`
- runtime path: `.venv-qwen3`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `--local-files-only`
- device: CUDA on `NVIDIA GeForce RTX 5070 Ti`
- scoring runtime recorded in manifest: `107.55073499679565` seconds

Legal/product gates preserved in manifest and score rows:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

## Exact Scoring Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py \
  --candidates runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl \
  --source-manifest runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json \
  --source-requests runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl \
  --run-root runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z \
  --max-rows 5000 \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --device cuda \
  --batch-size 32 \
  --local-files-only \
  --schema eos.msmarco_qwen3_full5k_teacher_scoring.v1 \
  --report-title 'Qwen3 Full 5k Teacher Scoring'
```

## Metrics

Helper manifest metrics:

- positive top-1 rate: `0.9998`
- positive top-1 count: `4,999 / 5,000`
- positive mean margin over best negative: `0.5312749150125786`
- positive score mean: `0.6962603540551706`
- negative score mean: `0.11362707581656185`
- margin min / median / max: `-0.03996013876894722` / `0.5429375695057388` / `0.832475259225248`
- missing score rows: `0`
- global-dev-positive negative refs flagged: `81`
- rows with global-dev-positive negatives: `80`

Repo audit summary:

- examples: `5,000`
- scored examples: `5,000`
- missing examples: `0`
- candidates: `20,000`
- scored candidates: `20,000`
- positive top-1 rate: `0.9998`
- positive mean rank: `1.0002`
- positive mean margin: `0.5312749147973024`
- mean normalized entropy: `0.9714484335879819`
- label policy: `selected_positive_and_any_qrels_positive`

## Files Changed Or Created

Changed tracked scratch helper:

- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`

Created scratch handoff:

- `.tiller/scratch/codex/eos-msmarco-qwen3-full5k-teacher-scoring-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-qwen3-full5k-teacher-scoring-v1.run-root`

Ignored/generated run artifacts:

- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/manifest.json`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.teacher-scores.jsonl`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/reports/eos-audit-teacher-scores.summary.json`

Inspected context:

- `AGENTS.md`
- `docs/eos-distillation-compact-default-spec.md`
- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`
- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1-report.md`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/manifest.json`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/reports/teacher-agreement-margin-report.md`

## Verification Commands And Results

Helper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py
```

Result: exited `0`.

Preflight:

```bash
.venv-qwen3/bin/python - <<'PY'
import importlib.util, json, torch
print(json.dumps({m: importlib.util.find_spec(m) is not None for m in ["torch","sentence_transformers","transformers","numpy"]}, sort_keys=True))
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Result: dependencies present; CUDA available on `NVIDIA GeForce RTX 5070 Ti`; local Qwen3 model cache found under Hugging Face hub cache.

Input line counts:

```bash
wc -l \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl
```

Result: `5000` candidate rows and `20000` request rows.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_qwen3_full5k_teacher_scoring.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .sample.candidate_rows_scored == 5000 and .counts.teacher_score_rows == 20000 and .counts.missing_score_rows == 0' \
  runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/manifest.json
```

Result: `true`.

Score/scored JSONL parse and count consistency:

```bash
python3 - <<'PY'
# parsed score/scored JSONL and asserted counts against manifest
PY
```

Result: `score_rows=20000`, `scored_candidate_rows=5000`, `missing_score_rows=0`, `flagged_dev_positive_negative_refs=81`, `rows_with_flagged_dev_positive_negative=80`.

Repo audit:

```bash
go run ./cmd/eos audit-teacher-scores \
  --qrels runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/dev.tsv \
  --corpus runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/corpus.jsonl \
  runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl \
  runs/eos-msmarco-qwen3-full5k-teacher-scoring-v1-20260624T180226Z/reports/eos-audit-teacher-scores.summary.json
```

Result: exited `0`; `examples=5000`, `scored=5000`, `missing=0`, `positive_top1_rate=0.999800`, `mean_margin=0.531275`, `mean_normalized_entropy=0.971448`.

Final git status:

```text
## main...origin/main
 M .tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py
```

## Caveats Or Residual Risk

- This is teacher-scoring evidence only, not training data and not a promotion artifact.
- Only Qwen3 0.6B was scored. mxbai-large agreement remains a separate blocked/next descriptor.
- Candidate negatives are deterministic corpus-ID negatives from the prep audit, not mined hard negatives.
- There is one row where the positive was not top-1 under Qwen3; margin min is negative at `-0.03996013876894722`.
- The `81` flagged global-dev-positive negative refs are not same-query leaks, but they must be filtered, relabeled, or explicitly downweighted before any future training-row generation.
- MS MARCO gates remain restrictive: `release_train_allowed=false` and `commercial_use_allowed=false`.

## Checkpoint Candidate

Yes, but only for the tiny helper/report slice:

- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`
- `.tiller/scratch/codex/eos-msmarco-qwen3-full5k-teacher-scoring-v1-report.md`

Do not checkpoint score JSONL, run data, model cache, or raw/candidate corpus artifacts.

## Recommended Next Action

Run the mxbai-large teacher on the exact same 5,000 candidate rows, then produce a Qwen3/mxbai agreement report before any training-row generation.

## Arbiter Next Action

Proceed to mxbai agreement scoring/audit gated on this run root, `missing_score_rows=0`, `release_train_allowed=false`, `commercial_use_allowed=false`, and explicit handling of the `81` global-dev-positive negative refs plus the single Qwen3 non-top1 row.
