# eos-compact-rank-diagnostics-and-miner

## Outcome

Implemented the first compact-aware default-model improvement slice:

- Added `eval-retrieval-turboquant --per-query-jsonl <path>` for compact per-query diagnostics on the artifact-backed default deployment surface.
- Added deterministic single-vector TurboQuant seed plumbing with `--quantizer-seed`; the smoke used `5581486560434873699`.
- Added `mine-retrieval-compact-hard-negatives` to consume compact per-query rows plus BEIR text/qrels and emit text hard-negative JSONL plus a manifest.
- Added leak guard behavior that rejects test split train-selection by default and supports explicit no-train validation smoke.
- Ran a plan-only prepared-IP compact-prefix smoke using existing training fields and explicit full-dim `256:4=0.05`; no training was run.

Review blockers fixed in follow-up:

- Test split mining now rejects `train_selection=true` unconditionally, including when `--allow-test-smoke` is present. Test-split smoke requires `--train-selection=false`.
- Compact miner negatives are filtered against qrels-authoritative positive docs, not just row-level relevance. The manifest can report `qrels_relevance_mismatches` when a per-query row marks a qrels-positive doc as non-relevant.
- Miner quantizer seed defaults to `DefaultTurboQuantMultiVectorQuantizerSeed`; matched rows with a different seed now fail with a seed mismatch error.
- Compact-reconstruct method derivation now shares evaluator naming and derives `..._reconstruct_rerank`.
- Miner corpus caps now use the evaluator's relevant-preserving corpus reader, so qrels positives past `--max-docs` are retained.
- Per-query JSONL scanner buffer now allows larger diagnostic rows.

No promoted assets, docs, default aliases, or model artifacts were modified.

## Distillation

This slice proves the diagnostic/mining plumbing can target the actual compact surface:

- method: `turboquant_ip_b4_overfetch200_fp16_rerank`
- bit width: `4`
- overfetch: `200`
- rerank storage: `fp16`
- quantizer seed: `5581486560434873699`
- artifact SHA256: `8074d2fce1842e232df2b4172d40463d82b57848c719b2d76fdd68aca682ac70`

The generated mining smoke is intentionally no-train because it used NFCorpus `test` qrels:

- `train_selection=false`
- `train_allowed=false`
- `leak_guard_status=validation_smoke_no_train_test_split`

The plan-only prepared-IP smoke accepted explicit full/default dim objectives without legacy prefix inheritance:

- `--matryoshka-dims 256`
- `--turboquant-prefix-objectives 256:4=0.05`
- `--turboquant-prefix-score-mode prepared-ip`
- no `--turboquant-prefix-bits`
- no training; output only planned workload.

## Files Changed

Source:

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`
- `runtime/retrieval_eval.go`
- `runtime/retrieval_turboquant.go`
- `runtime/retrieval_compact_hard_negative_mining.go`
- `runtime/retrieval_eval_test.go`

Generated run root:

- `runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/`

Key generated files:

- `diagnostics/nfcorpus-test-capped.turboquant.metrics.json`
- `diagnostics/nfcorpus-test-capped.turboquant.metrics.tsv`
- `diagnostics/nfcorpus-test-capped.turboquant.per-query.jsonl`
- `data/nfcorpus-test-capped.compact-hard-negatives.jsonl`
- `data/nfcorpus-test-capped.compact-hard-negatives.manifest.json`
- `logs/train-embed-prepared-ip-plan-only.log`
- `logs/train-embed-prepared-ip-plan-only.summary.json`

Inspected requested context:

- `.tiller/scratch/codex/eos-compact-aware-default-training-spec.md`
- `.tiller/scratch/codex/eos-compact-aware-training-hook-inventory-report.md`
- `.tiller/scratch/codex/eos-current-default-prod-release-readiness-smoke-report.md`
- `.tiller/scratch/codex/eos-nf005-nfcorpus-top10-teacher-pressure-candidate-report.md`
- `.tiller/scratch/codex/eos-current-nf005-nfcorpus-compact-rank-repair-report.md`
- `runtime/retrieval_turboquant.go`
- `runtime/retrieval_eval.go`
- `cmd/eos/main.go`
- `scripts/train_manta_embed_v1_candidate.fw`

## Verification Commands And Results

Passed:

```bash
gofmt -w runtime/retrieval_turboquant.go runtime/retrieval_compact_hard_negative_mining.go runtime/retrieval_eval.go runtime/retrieval_eval_test.go cmd/eos/main.go cmd/eos/main_test.go
go test ./runtime -run 'Retrieval|TurboQuant|PerQuery|Compact' -count=1
go test ./cmd/eos -run 'EvalRetrievalTurboQuant|Compact|PerQuery|TrainEmbed.*TurboQuant|MineRetrievalCompact' -count=1
git diff --check
jq empty \
  runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/diagnostics/nfcorpus-test-capped.turboquant.metrics.json \
  runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/data/nfcorpus-test-capped.compact-hard-negatives.manifest.json \
  runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/logs/train-embed-prepared-ip-plan-only.summary.json
```

Smoke command:

```bash
go run ./cmd/eos eval-retrieval-turboquant \
  --dataset nfcorpus \
  --split test \
  --bits 4 \
  --quantizer-seed 5581486560434873699 \
  --rerank-overfetch 200 \
  --rerank-storage fp16 \
  --top-k 100 \
  --max-docs 300 \
  --max-queries 5 \
  --metrics-json runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/diagnostics/nfcorpus-test-capped.turboquant.metrics.json \
  --metrics-tsv runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/diagnostics/nfcorpus-test-capped.turboquant.metrics.tsv \
  --per-query-jsonl runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/diagnostics/nfcorpus-test-capped.turboquant.per-query.jsonl \
  assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll \
  datasets/manta-embed-v1/raw/nfcorpus/nfcorpus
```

Result:

- backend: `cuda`
- docs: `3128` after relevant-doc cap protection
- queries: `5`
- dense nDCG@10 / recall@100: `0.274190` / `0.246288`
- q4 nDCG@10 / recall@100: `0.280581` / `0.235733`
- q4 fp16 rerank200 nDCG@10 / recall@100: `0.274190` / `0.246288`
- total compression for q4/fp16/rerank200: `1.59x`

Mining smoke:

```bash
go run ./cmd/eos mine-retrieval-compact-hard-negatives \
  --dataset nfcorpus \
  --split test \
  --per-query-jsonl runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/diagnostics/nfcorpus-test-capped.turboquant.per-query.jsonl \
  --manifest-json runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/data/nfcorpus-test-capped.compact-hard-negatives.manifest.json \
  --bits 4 \
  --overfetch 200 \
  --rerank-storage fp16 \
  --quantizer-seed 5581486560434873699 \
  --artifact-sha256 8074d2fce1842e232df2b4172d40463d82b57848c719b2d76fdd68aca682ac70 \
  --train-selection=false \
  --negatives 4 \
  --max-examples 12 \
  datasets/manta-embed-v1/raw/nfcorpus/nfcorpus \
  runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/data/nfcorpus-test-capped.compact-hard-negatives.jsonl
```

Result:

- examples: `12`
- positives: `12`
- negatives: `48`
- rows read / matched / emitted: `10 / 1 / 12`
- reason counts: `{"top10_competitor":12}`
- train allowed: `false`

This smoke was not rerun during the review-blocker follow-up because it already uses explicit no-train test mode and an explicit matching seed. The new test-split guard behavior is covered by runtime and CLI tests.

Plan-only prepared-IP smoke:

```bash
go run ./cmd/eos train-embed \
  --plan-only \
  --hard-negative-train \
  --hard-negatives-per-query 4 \
  --epochs 1 \
  --batch-size 4 \
  --contrastive-loss infonce \
  --matryoshka-dims 256 \
  --turboquant-prefix-objectives 256:4=0.05 \
  --turboquant-prefix-score-mode prepared-ip \
  runs/current-release-qwen3-nf005-continuation-20260616T224102Z/candidate/eos-embed-v1.mll \
  runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/data/nfcorpus-test-capped.compact-hard-negatives.jsonl
```

Result:

```text
planned workload: train=12 hard_negative_contrastive examples batch=4 steps/epoch=3 train_pairs/epoch=720 pairs(planned=720 actual=0)
```

`ferrous-wheel lint scripts/train_manta_embed_v1_candidate.fw` was not run because the script was not changed.

## Sample Output Schema

Per-query compact diagnostic row, abridged:

```json
{
  "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
  "dataset": "nfcorpus",
  "query_id": "PLAIN-2",
  "method": "turboquant_ip_b4_overfetch200_fp16_rerank",
  "bits": 4,
  "rerank_overfetch": 200,
  "rerank_storage": "fp16",
  "quantizer_seed": 5581486560434873699,
  "first_relevant_rank": 1,
  "top_k": [
    {
      "rank": 1,
      "doc_id": "MED-10",
      "score": 0.72633404,
      "relevance": 2,
      "dense_rank": 1,
      "dense_score": 0.7263274788856506,
      "compact_rank": 2,
      "compact_score": 0.7205060124397278
    }
  ]
}
```

Text hard-negative row uses the existing training-compatible schema:

```json
{
  "source": "nfcorpus:test:PLAIN-2:MED-10:top10_competitor",
  "query": "...",
  "positive": "...",
  "negatives": ["...", "...", "...", "..."]
}
```

## Leak Guard Behavior

- `mine-retrieval-compact-hard-negatives` defaults to `--train-selection=true`.
- With `--split test` and default train-selection, it errors: `refusing to mine train-selection rows from test split`.
- `--allow-test-smoke` does not bypass that guard; test split is only permitted by explicit no-train validation mode (`--train-selection=false`).
- Smoke manifest records `train_allowed=false`; generated NFCorpus test rows must not be used for training or selection.

## Caveats / Residual Risk

- The smoke used NFCorpus `test` qrels only in validation/no-train mode, because this descriptor asked for a small/capped proof path and no silent test training. Non-test train/dev mining should be run next before any training.
- `--max-docs 300` produced `3128` docs because existing evaluator logic preserves all qrels-relevant docs even under caps.
- The miner currently mines from per-query final top-k rows. For stronger rank-200 boundary mining, run diagnostics with a top-k/depth that includes the needed compact boundary evidence or extend the per-query row to emit separate candidate-boundary slices.
- The existing prepared-IP smoke is plan-only and proves config/plumbing acceptance, not objective quality.

## Checkpoint Candidate

Yes for source plus scratch report:

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`
- `runtime/retrieval_eval.go`
- `runtime/retrieval_turboquant.go`
- `runtime/retrieval_compact_hard_negative_mining.go`
- `runtime/retrieval_eval_test.go`
- `.tiller/scratch/codex/eos-compact-rank-diagnostics-and-miner-report.md`

Generated run artifacts under `runs/eos-compact-rank-diagnostics-and-miner-20260617T000000Z/` are useful evidence but should not be committed unless the project normally versions run artifacts.

## Arbiter Next Action

Run the same compact diagnostics/miner on non-test train/dev qrels for NFCorpus and FiQA, with `train_allowed=true`, then build the first compact-aware training dataset from those non-test rows plus dense preservation replay. Do not train from the generated test-split smoke rows.
