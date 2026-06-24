# eos-msmarco-hard-candidate-mxbai-agreement-v1 Report

## Outcome

Completed local/offline mxbai-large teacher scoring for the exact 5,000-row hard-candidate MS MARCO substrate already scored by Qwen3, then produced Qwen3/mxbai agreement and conflict evidence. No model training, release/commercial rows, or distillation rows were generated.

Run root:

- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/`

Primary artifacts:

- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/manifest.json`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.teacher-scores.jsonl`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.scored.jsonl`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/qwen3-mxbai-hard-candidate-agreement-report.json`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/eos-audit-teacher-scores.summary.json`

## Distillation

The run scored exactly the hard-candidate substrate:

- candidate rows: `5,000`
- negative candidates: `135,565`
- score rows: `140,565`
- unique query texts encoded: `5,000`
- unique candidate texts encoded: `130,890`
- mxbai teacher: `mixedbread-ai/mxbai-embed-large-v1`
- query prefix: `Represent this sentence for searching relevant passages: `
- document prefix: empty
- score scale: `cosine_normalized_dot`
- runtime path: `.venv-qwen3`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `--local-files-only`
- device: CUDA on `NVIDIA GeForce RTX 5070 Ti`
- scoring runtime recorded in manifest: `230.14639377593994` seconds

Legal gates were preserved in manifest and score/scored rows:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

## Exact Scoring Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3/bin/python .tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1.py \
  --candidates runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/artifacts/msmarco-passage.hard-candidates.train.jsonl \
  --qwen-scored runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/artifacts/msmarco-passage.hard-candidates.qwen3-0.6b.scored.jsonl \
  --source-manifest runs/eos-msmarco-hard-candidate-mining-v1-20260624T183728Z/manifest.json \
  --qwen-manifest runs/eos-msmarco-hard-candidate-qwen3-scoring-v1-20260624T192721Z/manifest.json \
  --run-root runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z \
  --max-rows 5000 \
  --model-id mixedbread-ai/mxbai-embed-large-v1 \
  --device cuda \
  --batch-size 32 \
  --local-files-only
```

## mxbai Metrics

Helper manifest metrics:

- positive top-1 rate: `0.6626`
- positive top-1 count: `3,313 / 5,000`
- positive mean rank: `2.58`
- positive mean margin over best negative: `0.11090029296875`
- missing score rows: `0`
- global-dev-positive negative refs flagged: `952`
- rows with global-dev-positive negatives: `865`

Repo audit summary:

- examples: `5,000`
- scored examples: `5,000`
- missing examples: `0`
- candidates: `140,565`
- scored candidates: `140,565`
- positive top-1 rate: `0.6626000000000001`
- positive mean rank: `2.58`
- positive mean margin: `0.11090029296875001`
- mean normalized entropy: `0.9980262375797043`
- label policy: `selected_positive_and_any_qrels_positive`

## Qwen3/mxbai Agreement Metrics

- both-positive-top1 count/rate: `2,805 / 5,000`, `0.561`
- Qwen3 positive top1 count/rate: `3,054 / 5,000`, `0.6108`
- mxbai positive top1 count/rate: `3,313 / 5,000`, `0.6626`
- Qwen-only top1 / mxbai-only top1 / neither top1: `249 / 508 / 1,438`
- conflict/adjudication candidate rows/rate: `2,195 / 5,000`, `0.439`
- same top negative / different top negative: `2,427 / 2,573`
- margin correlation Pearson: `0.9557591477089961`
- clean rows without global-dev-positive negatives: `4,135`

Agreement-filter bands, excluding global-dev-positive-negative rows from clean counts:

- margin `>0.0`: clean `2,379`, flagged otherwise-agree `421`
- margin `>0.01`: clean `2,263`, flagged otherwise-agree `390`
- margin `>0.025`: clean `2,099`, flagged otherwise-agree `350`
- margin `>0.05`: clean `1,909`, flagged otherwise-agree `302`
- margin `>0.1`: clean `1,625`, flagged otherwise-agree `230`

Best mxbai negative primary source counts:

- `bm25_bounded`: `3,645`
- `random_fill_control`: `1,019`
- `random_tail_control`: `336`

## Dev-Positive Negative Flag Handling

The helper reloaded dev qrels for audit-only global-positive flagging and preserved row-embedded flags:

- global-dev-positive negative refs found during scoring: `952`
- rows with global-dev-positive negatives: `865`
- row-embedded global-dev-positive refs: `952`
- row-embedded global-dev-positive rows: `865`
- flagged otherwise-agree rows are reported separately by margin band and are not included in clean agreement-filter counts.

These are not same-query dev leaks, but they remain explicit audit flags and should not be treated as clean release negatives.

## Files Changed Or Created

Scratch files:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1.py`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1.run-root`

Ignored/generated run artifacts:

- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/manifest.json`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.teacher-scores.jsonl`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.scored.jsonl`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/qwen3-mxbai-hard-candidate-agreement-report.json`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/teacher-agreement-margin-report.md`
- `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/eos-audit-teacher-scores.summary.json`

Input/source SHA256s recorded by manifest:

- source hard-candidate JSONL: `6109f4266c21b226d27397af4f7a617d9cc0dde971a31ee183dac53d885e83a8`
- source manifest: `a25cf8e6d45f9b7ca862394b67d06b2161c00545ee5d48cb0437b8fcb14e70bf`
- Qwen3 manifest: `1b3612dd360ee5c0526fd9128798918254149dc3fcd217085175f3c54ea8651b`
- Qwen3 scored candidates: `9570d35a70c2c414b841538448e8bafc3be76aaf24b1b9ebfd0ebd9f21fcc8d8`
- mxbai score JSONL: `21a11926f11bcf97daf9a4fc4780ec94ad4e45bc0a9bd921954ecf1756d30294`
- mxbai scored candidate JSONL: `d58d20bd8e7c35727bf23c050cc6ef88b5f20a0e977315e6aa1ab200226d70e2`
- agreement report JSON: `245809ead4aa9138da826e5f7106752bfea19af7dcb92d86f2e3703de33e009f`

## Verification Commands And Results

Helper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1.py
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

Result: local/offline mxbai cache loaded successfully; embedding dim `1024`; dtype `float16`; CUDA available on `NVIDIA GeForce RTX 5070 Ti`.

Input preflight count:

```bash
python3 - <<'PY'
# parsed hard-candidate JSONL; counted rows, negatives, score rows, and embedded dev-positive flags
PY
```

Result: `rows=5000`, `negatives=135565`, `score_rows=140565`, `flag_refs=952`, `flag_rows=865`.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_hard_candidate_mxbai_agreement.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .legal_gate.train_allowed_for_research == true and .sample.candidate_rows_scored == 5000 and .counts.candidate_rows_scored == 5000 and .counts.teacher_score_rows == 140565 and .counts.missing_score_rows == 0' \
  runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/manifest.json
```

Result: `true`.

Score/scored JSONL line counts:

```bash
wc -l \
  runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.teacher-scores.jsonl \
  runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.scored.jsonl
```

Result: `140565` score rows and `5000` scored candidate rows.

Independent JSONL/agreement recount:

```bash
python3 - <<'PY'
# parsed mxbai score/scored JSONL and Qwen3 scored JSONL, asserted counts/gates, and recomputed agreement counts
PY
```

Result: `score_rows=140565`, `scored_rows=5000`, `bad_gate_rows=0`, `flag_refs=952`, `flag_rows=865`, `mxbai_top1=3313`, `qwen_top1=3054`, `both_top1=2805`, `qwen_only=249`, `mxbai_only=508`, `neither=1438`, `same_top_negative=2427`, `different_top_negative=2573`.

Repo audit:

```bash
go run ./cmd/eos audit-teacher-scores \
  --qrels runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/dev.tsv \
  --corpus runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/corpus.jsonl \
  runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/artifacts/msmarco-passage.hard-candidates.mxbai-large.scored.jsonl \
  runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/eos-audit-teacher-scores.summary.json
```

Result: exited `0`; `examples=5000`, `scored=5000`, `missing=0`, `positive_top1_rate=0.662600`, `mean_margin=0.110900`, `mean_normalized_entropy=0.998026`.

Final git status:

```text
## main...origin/main
```

## Caveats Or Residual Risk

- This is teacher-scoring and agreement evidence only, not training data and not a promotion artifact.
- The mxbai helper is a scratch helper, not production source.
- The hard-candidate substrate is materially harder than the prior deterministic 3-negative full5k substrate: mxbai positive top-1 is `0.6626` here versus `1.0` on the earlier full5k random-negative run.
- There are `2,195` rows where at least one teacher does not rank the selected positive top-1. These are the conflict/adjudication candidates for possible reranker yes/no use.
- The `952` global-dev-positive negative refs across `865` rows must be filtered, relabeled, or explicitly downweighted before any future training-row generation.
- MS MARCO gates remain restrictive: `release_train_allowed=false` and `commercial_use_allowed=false`.

## Checkpoint Candidate

Yes, but only for the tiny scratch helper/report/run-root marker slice:

- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1.py`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-hard-candidate-mxbai-agreement-v1.run-root`

Do not checkpoint score JSONL, run data, model cache, or raw/candidate corpus artifacts.

## Recommended Next Action

Use `runs/eos-msmarco-hard-candidate-mxbai-agreement-v1-20260624T194831Z/reports/qwen3-mxbai-hard-candidate-agreement-report.json` to plan agreement-filtered research-only distillation rows and a separate reranker/adjudication band. Start from the clean margin `>0.0` agreement count of `2,379`, keep the `421` flagged otherwise-agree rows separate, and preserve `release_train_allowed=false` plus `commercial_use_allowed=false`.

## Arbiter Next Action

Proceed to agreement-filtered distillation-row planning only under research-only gates, using this mxbai run root, `missing_score_rows=0`, `release_train_allowed=false`, `commercial_use_allowed=false`, clean agreement-filter count `2379` at margin `>0.0`, and explicit exclusion or separate accounting of all global-dev-positive negative rows plus the `2195` conflict/adjudication candidates.
