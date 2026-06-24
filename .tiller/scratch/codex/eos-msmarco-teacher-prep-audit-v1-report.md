# eos-msmarco-teacher-prep-audit-v1 Report

## Outcome

Completed a bounded MS MARCO teacher candidate prep/audit run after corpus acquisition. This produced provenance-safe candidate and teacher-request substrates for later Qwen3/mxbai scoring only. No model training, teacher scoring, release training rows, or commercial-use rows were produced.

Run root:

- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/`

Primary artifacts:

- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/reports/leak-split-audit.md`

Scratch helper:

- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1.py`

## Distillation

Built a deterministic bounded sample from `qrels/train.tsv`: first 5,000 unique train qrel pairs, with three deterministic corpus-ID negatives per row. Dev qrels were loaded for audit only and were not used to construct candidates.

Legal/product gate preserved in candidate rows, request rows, manifest, and report:

- `release_train_allowed=false`
- `commercial_use_allowed=false`
- `train_allowed_for_research=true`

The candidate JSONL keeps Eos text hard-negative compatibility fields:

- `source`
- `query`
- `positive`
- `negatives`

It also carries required provenance:

- `split`
- `query_id`
- `positive_doc_id`
- `negative_doc_ids`
- `row_id`
- `source_qrels_line`
- `qrel_score`

## Files Changed Or Created

Ignored scratch files:

- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1.py`
- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1-report.md`

Ignored/generated run artifacts:

- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.jsonl`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.manifest.json`
- `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/reports/leak-split-audit.md`

Inspected context:

- `AGENTS.md`
- `docs/eos-distillation-compact-default-spec.md`
- `.tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1-report.md`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/reports/corpus-resolvability.md`
- `scripts/build_provenance_safe_agreement_teacher_prep.py`
- `scripts/test_build_provenance_safe_agreement_teacher_prep.py`
- `cmd/eos/main.go`
- `runtime/embedding_hard_negative_dataset.go`

## Schema Compatibility

Candidate JSONL is compatible with `EmbeddingTextHardNegativeExample` because it includes `source`, `query`, `positive`, and `negatives`; extra provenance fields are preserved by the runtime JSON handling.

Teacher request support was verified with the repo command:

- `go run ./cmd/eos export-teacher-score-requests ...`

The helper-generated request JSONL carries extra `row_id`, `query_id`, `candidate_doc_id`, and legal-gate fields. The repo-exported request JSONL is also present for strict built-in schema compatibility, but it does not preserve row/doc provenance beyond source/query/candidate/example index.

## Counts

- Candidate rows: `5,000`
- Teacher request rows from helper: `20,000`
- Repo-exported teacher request rows: `20,000`
- Positives per row: `1`
- Negatives per row: `3`
- Unique candidate queries: `4,668`
- Unique positive doc IDs: `4,998`
- Unique negative doc IDs: `14,991`
- Source train qrel rows available: `532,761`
- Source dev qrel rows available: `59,273`

Artifact SHA256s from manifest:

- Candidate JSONL: `2166af2a4e03567ba012fb2563e1e8745ddf60152b71d84ee3470dd8edcad2f8`
- Helper teacher request JSONL: `6a6bdedbcae866fa1c108b96a82d655aac6cacaf5725306b8c13e7a8570c7acf`
- Leak/split audit report: `f4d5c6436dcd3fede565b1e8fa0b1e3d3699d47cf9a3380c1d00ac863b1404ba`

## Leak/Split Gates

Gate status: `passed`.

- Same-query negative positive leaks: `0`
- Same-query dev-positive negative leaks: `0`
- Dev qrels used for candidate construction: `0`
- Rows missing split provenance: `0`
- Rows missing row_id provenance: `0`
- Candidate rows not in train split: `0`
- Global train/dev positive doc overlap: `2,287`
- Candidate negative doc IDs that are positives in dev for other queries: `81`

The `81` dev-positive negative docs are global doc-overlap observations, not same-query leaks. They are reported so downstream teacher scoring can inspect or filter them if desired.

## Verification Commands And Results

Helper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1.py
```

Result: exited 0.

Teacher prep build:

```bash
python3 .tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1.py \
  --dataset-root runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only \
  --source-manifest runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json \
  --run-root runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z \
  --sample-qrels 5000 \
  --negatives-per-row 3
```

Result: exited 0; wrote 5,000 candidates and 20,000 request rows; leak gate passed.

Manifest gate:

```bash
jq -e '.schema == "eos.msmarco_teacher_prep_audit.v1" and .legal_gate.release_train_allowed == false and .legal_gate.commercial_use_allowed == false and .counts.candidate_rows == 5000 and .counts.teacher_request_rows == 20000 and .split_and_leak_audit.gate_status == "passed" and .split_and_leak_audit.candidate_negative_positive_leaks == 0 and .split_and_leak_audit.dev_qrels_used_for_candidate_construction == 0' \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/teacher-prep-manifest.json
```

Result: `true`.

JSONL parse/count check:

```bash
python3 - <<'PY'
# streamed candidate/request JSONL parse and count assertions
PY
```

Result: candidate rows `5000`; helper request rows `20000`; split and row_id provenance present.

Independent leak check against train/dev qrels:

```bash
python3 - <<'PY'
# reloaded train/dev qrels and checked every candidate negative
PY
```

Result: `leak gate verified: rows=5000 leaks=0 missing_provenance=0 dev_same_query=0`.

Repo teacher request exporter compatibility:

```bash
go run ./cmd/eos export-teacher-score-requests \
  --manifest runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.manifest.json \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.jsonl
```

Result: exited 0; exported examples `5000`, rows `20000`, positive rows `5000`, negative rows `15000`.

Repo-exported request manifest gate:

```bash
jq -e '.schema == "manta.teacher_score_requests.v1" and .examples == 5000 and .rows == 20000 and .positive_rows == 5000 and .negative_rows == 15000' \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.manifest.json
```

Result: `true`.

Line counts:

```bash
wc -l \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-candidates.train-hard-negatives.jsonl \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl \
  runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/eos-exported-teacher-score-requests.jsonl
```

Result: `5000`, `20000`, `20000`.

Final git status:

```text
## main...origin/main
```

The checkout reports clean because scratch and run artifacts are ignored.

## Caveats Or Residual Risk

- Negatives are deterministic corpus-ID negatives, not BM25/model-mined hard negatives. This is suitable for teacher request substrate/audit, not a claim of hard-negative quality.
- Candidate text resolution streamed the BEIR corpus and sampled only 5,000 train qrel rows. It is intentionally bounded and not representative of the full corpus.
- The repo-exported request JSONL validates compatibility but drops extra provenance fields. Use the helper-generated request JSONL when scoring needs row/doc provenance alongside request rows.
- `81` candidate negative doc IDs are positives in dev for other queries. Same-query leak checks pass; downstream scoring may choose to filter global dev-positive negatives more aggressively.
- License interpretation remains an engineering gate, not legal advice.

## Checkpoint Candidate

Yes, but only for tiny scratch helper/report artifacts if root wants to preserve the procedure:

- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1.py`
- `.tiller/scratch/codex/eos-msmarco-teacher-prep-audit-v1-report.md`

No checkpoint candidate for run-root candidate/request data unless root explicitly changes ignored artifact policy.

## Recommended Next Action

Use `runs/eos-msmarco-teacher-prep-audit-v1-20260624T171403Z/artifacts/msmarco-passage.teacher-score-requests.jsonl` as the provenance-rich input to a bounded Qwen3/mxbai teacher-cache/scoring descriptor. Keep scoring cache-only and report teacher agreement/margins before any training-row generation.

## Arbiter Next Action

Proceed to teacher-cache/scoring audit gated on `teacher-prep-manifest.json` with `release_train_allowed=false`, `commercial_use_allowed=false`, and `split_and_leak_audit.gate_status=passed`; do not train until teacher agreement, margin, and false-negative controls pass on this exact candidate set.
