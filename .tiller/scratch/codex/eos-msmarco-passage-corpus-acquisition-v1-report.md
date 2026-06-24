# eos-msmarco-passage-corpus-acquisition-v1 Report

## Outcome

Completed the critical-path MS MARCO passage corpus acquisition and resolvability audit. This was data acquisition only: no model training, teacher scoring, reranking, promotion, or release training rows were produced.

Run root:

- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/`

Primary artifacts:

- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/SHA256SUMS`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/reports/corpus-resolvability.json`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/reports/corpus-resolvability.md`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/`

Scratch helper:

- `.tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1.py`

## Distillation

The passage collection was downloaded from the official MS MARCO Azure URL, extracted, hashed, and streamed once to prove train/dev qrels resolve to both query text and corpus passage text.

Legal/product gate preserved:

- `train_allowed_for_research=true`
- `release_train_allowed=false`
- `commercial_use_allowed=false`
- requires independent legal review for product/service use

No MS MARCO triples, top1000 files, hard-negative files, teacher vectors, reranker scores, model outputs, or release training JSONL were downloaded or created.

## Source URLs And License Gate

Official source URLs recorded in the manifest:

- `https://microsoft.github.io/msmarco/`
- `https://microsoft.github.io/msmarco/Datasets.html`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv`

The manifest repeats the research-only/non-commercial engineering policy and does not imply release or commercial permission.

## Counts

Corpus:

- `collection.tar.gz`: 1,035,009,698 bytes, SHA256 `58ebb1088c84fb8f97d56c755fb2b237ef30822dcfee55ccb2e010a8079ee21e`
- `collection.tsv`: 3,061,567,852 bytes, SHA256 `86e0c5820be6b22280337a67e54cbb626645453d9ac3c377bac45abb81c5653d`
- corpus rows: `8,841,823`
- unique passage IDs: `8,841,823`
- malformed corpus rows: `0`
- duplicate corpus IDs: `0`

Qrels:

- train qrels: `532,761` rows, `502,939` unique queries, `516,472` unique positive passage IDs
- dev qrels: `59,273` rows, `55,578` unique queries, `59,096` unique positive passage IDs
- qrel malformed rows: `0` for train and dev

Queries:

- train queries: `808,731` rows
- dev queries: `101,093` rows
- eval queries retained only because they are part of `queries.tar.gz`: `101,092` rows
- BEIR-style `queries.jsonl`: `909,824` train+dev query rows; eval queries were not emitted

Resolvability:

- train qrels missing query text: `0`
- dev qrels missing query text: `0`
- train qrel unique doc IDs missing corpus text: `0`
- dev qrel unique doc IDs missing corpus text: `0`
- combined qrel unique doc IDs resolved in corpus: `573,281 / 573,281`
- train/dev query overlap: `0`
- train/dev positive-passage overlap: `2,287`

BEIR-style research-only root:

- `corpus.jsonl`: `8,841,823` rows, SHA256 `b32cc19267572a79d0825df2fd9ae26a8284d7e78449701f9e63d04754a13c5e`
- `queries.jsonl`: `909,824` rows, SHA256 `e28cb71d6875ccd8a08226f7dc19d9677b3925a1e24fcc48ef097d187eb733c3`
- `qrels/train.tsv`: `532,761` qrel rows plus header, SHA256 `76e65fbf067e2e21a74bf32d6c88c2807cba775d5a37718046accfd503ad73fb`
- `qrels/dev.tsv`: `59,273` qrel rows plus header, SHA256 `fc05231ef68f9970edda9884ef79654e61398337c4bcdf98131c26b428356906`

## Files Changed Or Created

Ignored/generated run artifacts:

- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/SHA256SUMS`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/collection.tar.gz`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/extracted/collection/collection.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/queries.tar.gz`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/queries/queries.train.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/queries/queries.dev.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/queries/queries.eval.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/qrels.train.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/raw/qrels.dev.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/reports/corpus-resolvability.json`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/reports/corpus-resolvability.md`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/corpus.jsonl`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/queries.jsonl`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/train.tsv`
- `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/beir-msmarco-passage-research-only/qrels/dev.tsv`

Ignored scratch files:

- `.tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1.py`
- `.tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1-report.md`

Inspected context:

- `AGENTS.md`
- `docs/eos-distillation-compact-default-spec.md`
- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1-report.md`
- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/acquisition-manifest.json`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/reports/split-safety.md`

## Verification Commands And Results

Helper syntax:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1.py
```

Result: exited 0.

Acquisition:

```bash
python3 .tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1.py \
  --run-root runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z
```

Result: exited 0; wrote manifest and corpus-resolvability report.

Manifest parse/gate:

```bash
jq -e '.schema == "eos.msmarco_passage_corpus_acquisition_manifest.v1" and .license_terms_summary.engineering_policy.release_train_allowed == false and .license_terms_summary.engineering_policy.commercial_use_allowed == false and .counts.corpus.rows == 8841823 and .split_safety.train_qrels_queries_missing_from_train_queries == 0 and .split_safety.dev_qrels_queries_missing_from_dev_queries == 0 and .split_safety.corpus_resolvability.train.missing_unique_doc_ids == 0 and .split_safety.corpus_resolvability.dev.missing_unique_doc_ids == 0' \
  runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json
```

Result: `true`.

SHA verification:

```bash
(cd runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z && sha256sum -c SHA256SUMS)
```

Result: all raw, extracted, and BEIR-style files reported `OK`.

Git status:

```bash
git status --short --branch
```

Result:

```text
## main...origin/main
```

The new run root and scratch files are ignored in this checkout.

## Caveats Or Residual Risk

- License interpretation remains an engineering gate, not legal advice.
- The `2,287` train/dev positive-passage overlap is retained in the report. It is expected for shared corpus documents and different queries, but downstream row builders must keep split provenance explicit.
- The BEIR-style qrels files include a standard header row; row counts above exclude the header.
- Eval queries are present in the reused upstream `queries.tar.gz` extraction but were not emitted to BEIR `queries.jsonl` and are not trainable.
- No hard negatives were acquired; teacher-cache/scoring work still needs an explicit descriptor and leak checks before any training rows are generated.

## Checkpoint Candidate

Yes, but only for tiny scratch/helper/report artifacts if root wants to preserve the local acquisition procedure:

- `.tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1.py`
- `.tiller/scratch/codex/eos-msmarco-passage-corpus-acquisition-v1-report.md`

No checkpoint candidate for raw corpus archives, extracted corpus, BEIR dataset files, checksums, or run-root artifacts unless root explicitly wants local ignored provenance committed through a separate policy decision.

## Recommended Next Action

Route the next descriptor to teacher-cache/scoring audit only after confirming this manifest as the source of truth. Candidate scope: build Qwen3 0.6B and mxbai-large cached scores over a bounded train-positive plus mined-negative candidate set, preserving the same `release_train_allowed=false` and no-commercial gates.

## Arbiter Next Action

Proceed to teacher-cache/scoring descriptor gated on `runs/eos-msmarco-passage-corpus-acquisition-v1-20260624T170110Z/acquisition-manifest.json`; do not create training rows or run training until teacher agreement, negative leak checks, and split provenance reports pass.
