# eos-msmarco-hard-candidate-qwen3-scoring-v1 Report

## Outcome

Completed local/offline Qwen3 teacher scoring for the 5,000-row hard-candidate MS MARCO substrate. No model training, release/commercial rows, or distillation rows were generated.

Run root:

- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/`

Primary artifacts:

- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/manifest.json`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/artifacts/msmarco-passage.hard-candidates.qwen3-0.6b.teacher-scores.jsonl`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/artifacts/msmarco-passage.hard-candidates.qwen3-0.6b.scored.jsonl`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/reports/eos-audit-teacher-scores.summary.json`

## Distillation

The run scored exactly the mined hard-candidate substrate:

- candidate rows: `5,000`
- negative candidates: `135,565`
- score rows: `140,565`
- unique query texts encoded: `5,000`
- unique candidate texts encoded: `130,890`
- teacher: `Qwen/Qwen3-Embedding-0.6B`
- score scale: `cosine_normalized_dot`
- runtime path: `.venv-qwen3`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `--local-files-only`
- device: CUDA on `NVIDIA GeForce RTX 5070 Ti`
- scoring runtime recorded in manifest: `655.2820253372192` seconds

Legal gates were preserved in manifest and score/scored rows:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

## Exact Scoring Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1.py \
  --candidates runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/artifacts/msmarco-passage.hard-candidates.train.jsonl \
  --source-manifest runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/manifest.json \
  --run-root runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z \
  --max-rows 5000 \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --device cuda \
  --batch-size 32 \
  --local-files-only \
  --schema eos.msmarco_hard_candidate_qwen3_scoring.v1 \
  --report-title 'Qwen3 Hard-Candidate Teacher Scoring'
```

## Qwen3 Metrics

Helper manifest metrics:

- positive top-1 rate: `0.6108`
- positive top-1 count: `3,054 / 5,000`
- positive mean rank: `3.048`
- positive mean margin over best negative: `0.1330875060841441`
- positive score mean: `0.7017542249441147`
- negative score mean: `0.4183588338772414`
- margin min / median / max: `-0.43528786301612854` / `0.04434272646903992` / `0.7188933044672012`
- missing score rows: `0`
- candidate count distribution: `{"16": 1463, "17": 11, "18": 8, "19": 9, "20": 11, "21": 8, "22": 9, "23": 4, "24": 14, "25": 9, "26": 7, "27": 4, "28": 5, "29": 4, "30": 6, "31": 6, "32": 3422}`

Repo audit summary:

- examples: `5,000`
- scored examples: `5,000`
- missing examples: `0`
- candidates: `140,565`
- scored candidates: `140,565`
- positive top-1 rate: `0.6108`
- positive mean rank: `3.048`
- positive mean margin: `0.13308750608414413`
- mean normalized entropy: `0.9968321616359814`
- label policy: `selected_positive_and_any_qrels_positive`

Source-wise best negative primary source counts:

- `bm25_bounded`: `3,647`
- `random_fill_control`: `1,015`
- `random_tail_control`: `338`

For non-top1 rows, the best negative primary source counts were:

- `bm25_bounded`: `1,945`
- `random_tail_control`: `1`

## Dev-Positive Negative Flag Handling

The helper reloaded dev qrels for audit-only global-positive flagging and preserved row-embedded flags:

- global-dev-positive negative refs found during scoring: `952`
- rows with global-dev-positive negatives: `865`
- row-embedded global-dev-positive refs: `952`
- source manifest global-dev-positive refs: `952`

These are not same-query dev leaks, but they remain explicit audit flags and should not be silently treated as clean release negatives.

## Files Changed Or Created

Scratch files:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1.py`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1.run-root`

Ignored/generated run artifacts:

- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/manifest.json`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/artifacts/msmarco-passage.hard-candidates.qwen3-0.6b.teacher-scores.jsonl`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/artifacts/msmarco-passage.hard-candidates.qwen3-0.6b.scored.jsonl`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/reports/eos-audit-teacher-scores.summary.json`

Smoke run also created ignored artifacts under:

- `runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-smoke-20260624T192640Z/`

Input/source SHA256s recorded by manifest:

- source hard-candidate JSONL: `6109f4266c21b226d27397af4f7a617d9cc0dde971a31ee183dac53d885e83a8`
- source manifest: `a25cf8e6d45f9b7ca862394b67d06b2161c00545ee5d48cb0437b8fcb14e70bf`
- score JSONL: `5c0900baa2aa632ceaecec59de57d64b4c38f0207eb7fe96ecf5f724d078baca`
- scored candidate JSONL: `9570d35a70c2c414b841538448e8bafc3be76aaf24b1b9ebfd0ebd9f21fcc8d8`

## Verification Commands And Results

Helper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1.py
```

Result: exited `0`.

Preflight:

```bash
.venv-qwen3/bin/python - <<'PY'
import importlib.util, json, torch
print(json.dumps({m: importlib.util.find_spec(m) is not None for m in ["torch","sentence_transformers","transformers","numpy"]}, sort_keys=True))
print(json.dumps({"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(), "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}, sort_keys=True))
PY
```

Result: dependencies present; CUDA available on `NVIDIA GeForce RTX 5070 Ti`; local Qwen3 files found under Hugging Face hub cache.

Input preflight count:

```bash
python3 - <<'PY'
# parsed hard-candidate JSONL; counted rows, negatives, unique texts, candidate distribution, and dev-positive row flags
PY
```

Result: `rows=5000`, `negatives=135565`, `score_rows=140565`, `unique_queries=5000`, `unique_candidate_texts=130890`, `row_flags_in_file_refs=952`, `row_flags_in_file_rows=865`.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_hard_candidate_qwen3_scoring.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .legal_gate.train_allowed_for_research == true and .sample.candidate_rows_scored == 5000 and .counts.candidate_rows_scored == 5000 and .counts.missing_score_rows == 0' \
  runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/manifest.json
```

Result: `true`.

Score/scored JSONL parse and count consistency:

```bash
python3 - <<'PY'
# parsed score/scored JSONL and asserted counts/gates against manifest
PY
```

Result: `score_rows=140565`, `scored_candidate_rows=5000`, `negative_candidates=135565`, `missing_scored_rows=0`, `flagged_dev_positive_negative_refs=952`, `bad_legal_gate_rows=0`.

Repo audit:

```bash
go run ./cmd/eos audit-teacher-scores \
  --qrels runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/dev.tsv \
  --corpus runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/corpus.jsonl \
  runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/artifacts/msmarco-passage.hard-candidates.qwen3-0.6b.scored.jsonl \
  runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/reports/eos-audit-teacher-scores.summary.json
```

Result: exited `0`; `examples=5000`, `scored=5000`, `missing=0`, `positive_top1_rate=0.610800`, `mean_margin=0.133088`, `mean_normalized_entropy=0.996832`.

Final git status:

```text
## main...origin/main
```

## Caveats Or Residual Risk

- This is teacher-scoring evidence only, not training data and not a promotion artifact.
- Qwen3 alone was scored. mxbai hard-candidate agreement remains a separate next step.
- The hard-candidate substrate is much harder than the prior random-negative teacher-prep substrate: Qwen3 positive top-1 dropped from the prior random-negative `0.9998` to `0.6108` here.
- There are `1,946` rows where a negative beat the selected positive under Qwen3. The best-negative source is mostly `bm25_bounded`; one non-top1 best negative has `random_tail_control` as primary source.
- The `952` global-dev-positive negative refs must be filtered, relabeled, or explicitly downweighted before any future training-row generation.
- MS MARCO gates remain restrictive: `release_train_allowed=false` and `commercial_use_allowed=false`.

## Checkpoint Candidate

Yes, but only for the tiny scratch helper/report slice:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1.py`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-qwen3-scoring-v1.run-root`

Do not checkpoint score JSONL, run data, smoke run data, model cache, or raw/candidate corpus artifacts.

## Recommended Next Action

Run mxbai-large teacher scoring on the same hard-candidate substrate, then produce Qwen3/mxbai agreement and conflict evidence before any distillation-row generation.

## Arbiter Next Action

Proceed to mxbai hard-candidate agreement scoring gated on this run root, `missing_score_rows=0`, `candidate_rows_scored=5000`, `teacher_score_rows=140565`, legal gates false for release/commercial, and explicit handling of the `952` global-dev-positive negative refs plus Qwen3 non-top1 rows.
