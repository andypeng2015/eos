# eos-msmarco-bounded-teacher-scoring-smoke-v1 Report

## Outcome

Completed the first bounded MS MARCO teacher-scoring smoke on the audited candidate substrate with local cached `Qwen/Qwen3-Embedding-0.6B`. This generated teacher agreement/margin evidence only. No training rows, release rows, commercial rows, or promotion artifacts were produced.

Run root:

- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/`

Primary artifacts:

- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/manifest.json`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/artifacts/msmarco-passage.qwen3-0.6b.teacher-scores.jsonl`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/reports/eos-audit-teacher-scores.summary.json`

Scratch helper:

- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`

## Distillation

The smoke scored a deterministic prefix of the audited MS MARCO candidate rows:

- sample bound: first `512` candidate rows from the 5,000-row substrate
- score rows: `2,048` total candidate scores
- unique query texts encoded: `474`
- unique candidate texts encoded: `2,047`
- teacher: `Qwen/Qwen3-Embedding-0.6B`
- score scale: `cosine_normalized_dot`
- runtime: local `.venv-qwen3`, `torch 2.12.0+cu130`, CUDA on `NVIDIA GeForce RTX 5070 Ti`
- model cache mode: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, helper `--local-files-only`

Legal gates preserved in manifest and score rows:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

## Metrics

Helper manifest metrics:

- positive top-1 rate: `1.000000`
- positive mean margin over best negative: `0.5235004064151636`
- positive score mean: `0.6902415440621961`
- negative score mean: `0.11472907514100572`
- missing score rows: `0`
- global-dev-positive negative refs in scored subset: `10`
- rows with global-dev-positive negatives: `10`

Repo audit summary:

- examples: `512`
- scored examples: `512`
- missing examples: `0`
- candidates: `2,048`
- scored candidates: `2,048`
- positive top-1 rate: `1`
- positive mean margin: `0.5235004068963462`
- mean normalized entropy: `0.9720946228025572`
- label policy: `selected_positive_and_any_qrels_positive`

## Files Changed Or Created

Ignored scratch files:

- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`
- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1-report.md`

Ignored/generated run artifacts:

- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/manifest.json`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/artifacts/msmarco-passage.qwen3-0.6b.teacher-scores.jsonl`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/reports/eos-audit-teacher-scores.summary.json`

Inspected context:

- `AGENTS.md`
- `docs/eos-distillation-compact-default-spec.md`
- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1-report.md`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl`
- `scripts/export_retrieval_vectors.py`
- `scripts/score_teacher_with_vector_cache.py`
- `scripts/test_score_teacher_with_vector_cache.py`
- `scripts/export_qwen3_retrieval_vectors.py`
- `scripts/test_export_qwen3_retrieval_vectors.py`
- `cmd/teacher-bridge/main.go`
- `cmd/eos/main.go`

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

Result: scoring dependencies present in `.venv-qwen3`; CUDA available on `NVIDIA GeForce RTX 5070 Ti`. Default `python3` lacked `torch`, `transformers`, and `sentence_transformers`, so the smoke used the existing venv.

Scoring:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py \
  --candidates runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl \
  --source-manifest runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json \
  --run-root runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z \
  --max-rows 512 \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --device cuda \
  --batch-size 32 \
  --local-files-only
```

Result: exited `0`; wrote manifest, score JSONL, scored candidate JSONL, and margin report. Elapsed scoring runtime recorded in manifest: `53.40611958503723` seconds.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_bounded_teacher_scoring_smoke.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .sample.candidate_rows_scored == 512 and .counts.teacher_score_rows == 2048 and .counts.missing_score_rows == 0' \
  runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/manifest.json
```

Result: `true`.

JSONL parse/count consistency:

```bash
python3 - <<'PY'
# parsed score/scored JSONL and asserted counts against manifest
PY
```

Result: `score_rows=2048`, `scored_candidate_rows=512`, `positive_top1_rate=1.0`, `mean_margin=0.5235004064151636`, `flagged_dev_positive_negative_refs=10`.

Repo audit:

```bash
go run ./cmd/eos audit-teacher-scores \
  --qrels runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/dev.tsv \
  --corpus runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/corpus.jsonl \
  runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/artifacts/msmarco-passage.qwen3-0.6b.scored.teacher-candidates.jsonl \
  runs/eos-msmarco-bounded-teacher-scoring-smoke-v1-20260624T172254Z/reports/eos-audit-teacher-scores.summary.json
```

Result: exited `0`; `examples=512`, `scored=512`, `missing=0`, `positive_top1_rate=1.000000`, `mean_margin=0.523500`, `mean_normalized_entropy=0.972095`.

Final git status:

```text
## main...origin/main
```

Scratch and run artifacts are ignored by existing `.gitignore` rules for `.tiller/` and `runs/`.

## Caveats Or Residual Risk

- This is a deterministic prefix smoke, not a full-substrate teacher run.
- Only Qwen3 0.6B was scored. mxbai-large is locally cached, but not run because one strong teacher satisfied the smoke and two-teacher scoring was not necessary.
- Candidate negatives are deterministic corpus-ID negatives from the prep audit, not mined hard negatives.
- The 10 flagged global-dev-positive negative refs are not same-query leaks, but they must not be silently treated as clean release negatives.
- MS MARCO legal/product gates remain restrictive: no release/commercial training claim.

## Checkpoint Candidate

Yes, but only for the tiny scratch helper/report if root wants to preserve the procedure:

- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1.py`
- `.tiller/scratch/codex/eos-msmarco-bounded-teacher-scoring-smoke-v1-report.md`

No checkpoint candidate for raw score/run artifacts.

## Recommended Next Action

Run the same helper at the full 5,000-row candidate bound for Qwen3 if the orchestrator accepts the extra runtime, then optionally run mxbai-large on the same bound for teacher agreement filtering. Keep both as evidence artifacts only until false-negative controls are specified.

## Arbiter Next Action

Proceed to bounded/full-substrate teacher agreement audit gated on `release_train_allowed=false`, `commercial_use_allowed=false`, `missing_score_rows=0`, and explicit handling of global-dev-positive negative flags; do not generate training rows yet.
