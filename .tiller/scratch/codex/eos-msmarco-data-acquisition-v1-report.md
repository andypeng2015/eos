# eos-msmarco-data-acquisition-v1 Report

## Outcome

Completed a bounded MS MARCO Passage Ranking acquisition probe and license/split-safety audit. No model training, teacher scoring, promotion, or release training JSONL was produced.

Run root:

- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/`

Primary artifacts:

- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/acquisition-manifest.json`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/SHA256SUMS`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/reports/split-safety.json`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/reports/split-safety.md`

Scratch helper:

- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py`

## Distillation

Official MS MARCO pages record non-commercial research-only terms and no extension of license/IP rights. Engineering gate applied:

- `train_allowed_for_research=true` for train split rows only after corpus text acquisition and full qrel/query/corpus resolvability checks.
- `release_train_allowed=false`.
- `commercial_use_allowed=false`.
- Dev is sanity/eval only.
- Test rows are never training or selection inputs.

Because this host has about 82 GB free and several official artifacts are large or risky for an ambient probe, I downloaded only bounded qrels and query files. I did not download the passage corpus, triples, or top1000 candidate sets.

Downloaded/probed:

- `queries.tar.gz`: 18,882,551 bytes, SHA256 `05e4c62c9c8520cd695725340d4c990627fd1a92eb8a71b88e3c746031ba6de8`
- `qrels.train.tsv`: 10,589,532 bytes, SHA256 `641b3c9391ea19e4a3d9e3284299f07be6725ff2dd11591a3d4d3f293db17cf2`
- `qrels.dev.tsv`: 1,201,626 bytes, SHA256 `a1befe0a99974b166b9e09408ccccae471fda5df4953fd6dd86ce4caac691752`

Probe counts:

- Train qrels: 532,761 rows, 502,939 unique queries, 516,472 unique positive passage IDs.
- Dev qrels: 59,273 rows, 55,578 unique queries, 59,096 unique positive passage IDs.
- Queries: train 808,731 rows; dev 101,093 rows; eval 101,092 rows.

Split-safety results:

- Train/dev query overlap: 0.
- Train/dev positive-passage overlap: 2,287. This is not by itself leakage; corpus documents can be relevant to different train/dev queries, but training rows must still keep split provenance explicit.
- Train qrels missing query text: 0.
- Dev qrels missing query text: 0.
- Corpus/qrel resolvability: not run because the passage collection was intentionally not downloaded.
- Test boundary: safe for this probe; no test labels were downloaded.

## Files Changed Or Generated

Generated ignored run-local data/audit files:

- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/acquisition-manifest.json`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/SHA256SUMS`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/raw/queries.tar.gz`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/raw/qrels.train.tsv`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/raw/qrels.dev.tsv`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/raw/queries/queries.train.tsv`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/raw/queries/queries.dev.tsv`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/raw/queries/queries.eval.tsv`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/reports/split-safety.json`
- `runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/reports/split-safety.md`

Generated scratch files:

- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py`
- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1-report.md`

Inspected:

- `docs/eos-distillation-compact-default-spec.md`
- `docs/production-embedding.md`
- `docs/default-corkscrew-embedder-plan.md`
- `docs/manta-embed-sota-avenues.md`
- `scripts/acquire_manta_embed_v1_datasets.fw`

Official source URLs recorded in manifest:

- `https://microsoft.github.io/msmarco/`
- `https://microsoft.github.io/msmarco/Datasets.html`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv`
- `https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv`

## Verification Commands And Results

Start git status:

```text
## main...origin/main
```

Disk preflight:

```text
/dev/sdd 1007G 874G 82G 92% /
```

Acquisition/probe:

```bash
python3 .tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py \
  --run-root runs/eos-msmarco-data-acquisition-v1-20260624T165140Z
```

Result: exited 0; wrote manifest and split-safety JSON.

Manifest parse/gate:

```bash
jq -e '.schema == "eos.msmarco_acquisition_manifest.v1" and .license_terms_summary.engineering_policy.release_train_allowed == false and .counts.probe.qrels_train.rows > 0 and .counts.probe.qrels_dev.rows > 0' \
  runs/eos-msmarco-data-acquisition-v1-20260624T165140Z/acquisition-manifest.json
```

Result: `true`.

SHA verification:

```bash
(cd runs/eos-msmarco-data-acquisition-v1-20260624T165140Z && sha256sum -c SHA256SUMS)
```

Result: all six downloaded/extracted files `OK`.

Helper syntax check:

```bash
python3 -m py_compile .tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py
```

Result: exited 0.

End git status:

```text
## main...origin/main
```

The generated run and scratch files are ignored by Git in this checkout, so `git status --short --branch` remains clean.

## Caveats Or Residual Risk

- This is not a full acquisition. The passage corpus was not downloaded, so positive passage IDs are not yet resolved to text.
- No hard negatives were acquired. Official train triples/top1000 files remain planned or blocked for explicit large-download descriptors.
- The official page HEAD sizes observed during the probe differ from some human-readable table sizes; the manifest records both official table sizes and HEAD byte lengths.
- License terms were treated as an engineering gate, not a legal opinion.
- No processed training JSONL was emitted because corpus text is absent and `release_train_allowed=false`.

## Next Safe Command Or Descriptor

Recommended next descriptor: acquire the passage collection only, with the same license gate, then run corpus/qrel/query resolvability and emit a train/dev-safe positive-pair skeleton with `release_train_allowed=false`.

Safe next command shape:

```bash
python3 .tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py \
  --run-root runs/eos-msmarco-data-acquisition-v1-<new-timestamp>
```

That helper currently performs the bounded probe only. For full corpus acquisition, either extend it with an explicit `--download-corpus` flag and free-space floor, or create a second descriptor that downloads one of:

- `collection.tar.gz` plus existing `queries.tar.gz` and qrels, or
- `collectionandqueries.tar.gz` as the combined corpus/query/qrel source.

Do not download `triples.train.small.tar.gz`, `triples.train.full.tsv.gz`, `top1000.train.tar.gz`, or train from MS MARCO until the corpus-resolvability audit and legal/release gate are reviewed.

## Checkpoint Candidate

Yes, for the scratch helper/report and small manifest structure only:

- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1.py`
- `.tiller/scratch/codex/eos-msmarco-data-acquisition-v1-report.md`

No checkpoint candidate for ignored run data or raw downloaded MS MARCO files unless explicitly desired for local audit provenance.

## Arbiter Next Action

Route `eos-msmarco-passage-corpus-acquisition-v1`: download only the passage collection with a free-space floor, compute SHA256, resolve train/dev qrels to query and corpus text, verify train/dev/test boundary, and keep `release_train_allowed=false`.
