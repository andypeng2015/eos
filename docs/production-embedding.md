# Production Embedding Candidate

Use `scripts/train_manta_embed_v1_candidate.fw` to create a release-grade `manta-embed-v1` candidate. The workflow wraps the current Eos CLI primitives with production guardrails:

- refuses temporary input paths unless explicitly overridden
- refuses dirty repositories unless explicitly overridden
- records repo commit, Go version, selected environment, dataset SHA256, artifact SHA256, and run config
- trains from either a raw corpus or prepared JSONL
- runs eval-only verification on a copied package and requires `optimizer_updates=0`
- runs a separate hard holdout eval by default
- exports and verifies a sealed MLL
- supports metric gates through environment variables

## Preflight

Run the local preflight before spending trainer time on a candidate:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/verify_manta_production.fw
```

This uses generated tiny fixtures to verify the public Eos surface, build `cmd/eos`, initialize and train a `.mll` package, run eval-only checks with `optimizer_updates=0`, export a sealed `.mll`, and inspect both packages. It is a production path check, not a model quality result.

## Acquire Datasets

The default acquisition workflow downloads public BEIR retrieval datasets (`scifact`, `nfcorpus`, and `fiqa`), converts qrels to Eos text-pair JSONL, writes a tokenizer corpus, and emits initial threshold gates:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/manta-embed-v1 \
ferrous-wheel run scripts/acquire_manta_embed_v1_datasets.fw
```

The processed outputs are:

```text
/data/manta/datasets/manta-embed-v1/processed/train.jsonl
/data/manta/datasets/manta-embed-v1/processed/eval.jsonl
/data/manta/datasets/manta-embed-v1/processed/hard-eval.jsonl
/data/manta/datasets/manta-embed-v1/processed/tokenizer-corpus.txt
/data/manta/datasets/manta-embed-v1/processed/thresholds.env
```

Review dataset licenses before commercial use. The default avoids MS MARCO because it is commonly distributed with non-commercial research terms.

## Prepared JSONL Path

Prepared JSONL is the preferred path for production because train/eval splits are fixed before training starts.

```bash
EOS_RUN_ROOT=/data/manta/runs \
EOS_REPO_ROOT=$PWD \
EOS_RUN_ID=manta-embed-v1-20260412-a \
EOS_TRAIN_JSONL=/data/manta/datasets/manta-embed-v1/processed/train.jsonl \
EOS_EVAL_JSONL=/data/manta/datasets/manta-embed-v1/processed/eval.jsonl \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/manta-embed-v1/processed/hard-eval.jsonl \
EOS_TOKENIZER_CORPUS=/data/manta/datasets/manta-embed-v1/processed/tokenizer-corpus.txt \
EOS_THRESHOLDS_ENV=/data/manta/datasets/manta-embed-v1/processed/thresholds.env \
EOS_EPOCHS=3 \
EOS_BATCH_SIZE=1024 \
EOS_LR=0.005 \
EOS_TEMPERATURE=0.05 \
EOS_SELECT_METRIC=score_margin \
EOS_EVAL_EVERY_STEPS=0 \
EOS_MIN_AUC=0.70 \
EOS_MIN_THRESHOLD_ACCURACY=0.65 \
EOS_MIN_SCORE_MARGIN=0.05 \
EOS_MAX_LOSS=0.35 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

`EOS_THRESHOLDS_ENV` loads the acquisition workflow's current gate file and records its SHA256 in the run manifest. Explicitly exported `EOS_*` values still override values from that file. Set `EOS_TOKENIZER=/path/to/tokenizer.mll` when you want to reuse an existing tokenizer instead of training one from `EOS_TOKENIZER_CORPUS`.

When prepared JSONL is text and a tokenizer is available, the production workflow tokenizes train, validation, and hard-eval JSONL into run-local token files before training. Training and eval then read token JSONL directly, which front-loads BPE cost, makes the optimizer profile reflect model work, and records the generated token files in `datasets.sha256`.
Use `eos train-embed --no-tokenizer` when directly training token JSONL beside a sibling tokenizer; otherwise the CLI intentionally auto-discovers that tokenizer and treats the JSONL as text.
Use `eos diagnose-train-metrics /path/to/train.metrics.json` to explain backend use, transfer pressure, and suspicious training/eval counters from any run. Use `eos gate-train-metrics --thresholds /path/to/thresholds.env --scope quality /path/to/final-eval.metrics.json` to apply the same quality gate outside the production workflow. Use `--scope efficiency` for training throughput and backend counters, and `--scope eval-only` to enforce zero optimizer updates on validation runs.

Contrastive training uses pair-length-aware bucketing by default so batches reach larger exact-length matmul groups. `EOS_TRAIN_LENGTH_BUCKET_WINDOW` controls the shuffled sort window; larger values can improve grouping but must be profiled because they reduce local length randomness and can increase per-batch working-set pressure.

The production workflow defaults `EOS_EVAL_EVERY_STEPS=0`. Keep within-epoch eval disabled for full candidate runs unless you are debugging convergence; epoch eval, final validation eval, and hard holdout eval still run and are enough for release gating. On the acquired full split, step-level eval every 4 batches adds many full eval passes and dominates transfer without improving the optimizer update itself.

Eval-only candidate gates batch pairwise forward encodes by exact token length. `EOS_TRAIN_PAIR_EVAL_BATCH_SIZE` defaults to `256`; set `EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1` only for scalar A/B checks.

`EOS_TRAIN_ENABLE_FAST_GELU=1` is available for throughput experiments. It changes GELU math from precise tanh to a bounded rational tanh approximation, so use it only when the exact-GELU candidate and fast-GELU candidate are both evaluated against validation and hard holdout gates.

## Metric Thresholds

The default acquired eval files are pairwise positive/negative judgments with one deterministic sampled negative per positive. The initial release gates are:

```text
EOS_SELECT_METRIC=score_margin
EOS_MIN_AUC=0.70
EOS_MIN_THRESHOLD_ACCURACY=0.65
EOS_MIN_SCORE_MARGIN=0.05
EOS_MAX_LOSS=0.35
```

These gates are intentionally concrete rather than advisory. AUC must clear 0.70 as a threshold-free separability check, calibrated threshold accuracy must clear 0.65 against a 0.50 random baseline, positive scores must beat negative scores by at least 0.05 on average, and pairwise loss must stay at or below 0.35. Tighten them after the first stable full-size candidate establishes the project baseline.

## Training Efficiency Gates

Production candidate runs can also enforce hardware-specific training efficiency gates from `train.metrics.json`:

```text
EOS_MIN_TRAIN_PAIRS_PER_SEC=85000
EOS_MIN_OPTIMIZER_STEPS_PER_SEC=0.08
EOS_MAX_MATMUL_RUNS=400000
EOS_MAX_MATMUL_RUN_UPLOAD_MB=950000
EOS_MAX_MATMUL_RUN_DOWNLOAD_MB=485000
```

Set these from a known-good run on the target trainer host. Keep quality gates hardware-independent, and keep efficiency gates host-specific so a CPU fallback or data-transfer regression does not silently produce a slow candidate.

## Corpus Path

Use corpus mode only when you intentionally want this run to mine the train/eval pairs:

```bash
EOS_RUN_ROOT=/data/manta/runs \
EOS_REPO_ROOT=$PWD \
EOS_RUN_ID=manta-embed-v1-20260412-corpus-a \
EOS_CORPUS=/data/manta/corpus/prod-corpus.txt \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/manta-embed-v1/hard-eval.jsonl \
EOS_EPOCHS=3 \
EOS_BATCH_SIZE=1024 \
EOS_MAX_PAIRS=0 \
EOS_EVAL_PAIRS=512 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

Corpus mode writes mined pairs and the tokenizer into the run directory, then records their SHA256 values in `datasets.sha256`.

## JSONL Formats

Token contrastive JSONL:

```json
{"query_tokens":[1,2,3],"positive_tokens":[1,2,3],"query_mask":[1,1,1],"positive_mask":[1,1,1]}
```

Text-pair JSONL:

```json
{"query":"how to reset password","document":"reset your password from account settings","label":1}
{"left":"how to reset password","right":"billing invoice export","label":0}
```

Positive-only text pairs can train contrastively. Mixed positive/negative text pairs are valid for eval-only gates.

Token-pair eval JSONL:

```json
{"left_tokens":[1,2,3],"right_tokens":[1,2,3],"left_mask":[1,1,1],"right_mask":[1,1,1],"target":1}
{"left_tokens":[1,2,3],"right_tokens":[9,8],"left_mask":[1,1,1],"right_mask":[1,1],"target":0}
```

## Release Gate

A candidate is releasable only when:

- `manifest.json` status is `success`
- `logs/final-eval.log` and `logs/hard-eval.log` report `optimizer_updates=0`
- configured metric gates pass on hard eval
- `logs/inspect-package.log` reports `package verify: OK`
- `logs/inspect-sealed.log` reports `package verify: OK`
- `artifacts.sha256` contains the sealed MLL hash

The release artifact is:

```text
<run-dir>/manta-embed-v1.sealed.mll
```

Keep the full run directory with the released artifact. It is the audit trail for reproducing or rejecting the candidate later.
