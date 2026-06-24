# eos-current-default-turboquant-frontier-v1

## Outcome

Completed a bounded current-default TurboQuant bit-depth frontier check using existing full 256d vector caches for the promoted narrow default. No model was trained or promoted.

q3 is supported by the current vector-cache TurboQuant evaluator and was measured on SciFact, NFCorpus, and FiQA. The evaluator help advertises `--bits` as comma-separated bit widths in supported range `2..8`; `--bits 2,3,4,8 --rerank-overfetch 200 --rerank-storage fp16` completed successfully.

The live sealed-artifact scoreboard harness was attempted first, but its first dense `eval-retrieval` command exited `255` with empty command logs before any TurboQuant/q3 work. I therefore used the existing current-default vector caches generated from `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll` to avoid rerunning live embedding.

## Distillation

Run artifact root:

```text
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/
```

Measured vector-cache inputs:

| dataset | docs | queries | vector dimension | cache source |
| --- | ---: | ---: | ---: | --- |
| SciFact | 5,183 | 300 | 256 | `runs/eos-narrow-default-hybrid-policy-full-scifact-q4-bm25dot-20260623T050356Z/vector-cache/` |
| NFCorpus | 3,633 | 323 | 256 | `runs/eos-narrow-default-hybrid-policy-full-nfcorpus-q4-bm25dot-20260623T050540Z/vector-cache/` |
| FiQA | 57,600 | 648 | 256 | `runs/eos-narrow-default-hybrid-policy-full-fiqa-q4-bm25dot-20260623T050722Z/vector-cache/` |

Dense macro reference from the same caches:

| row | macro nDCG@10 | macro recall@100 |
| --- | ---: | ---: |
| dense vector-cache | 0.297181608 | 0.463396240 |

Macro frontier:

| row | macro nDCG@10 | macro recall@100 | macro nDCG delta vs dense | macro recall delta vs dense | avg total compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| q2 direct | 0.255393100 | 0.437285086 | -0.041788508 | -0.026111154 | 15.058824x |
| q3 direct | 0.276546249 | 0.453805545 | -0.020635359 | -0.009590695 | 10.240000x |
| q4 direct | 0.287493121 | 0.459996975 | -0.009688487 | -0.003399265 | 7.757576x |
| q8 direct | 0.298375761 | 0.464414552 | +0.001194153 | +0.001018312 | 3.938462x |
| q2 + fp16/o200 | 0.297226349 | 0.458229944 | +0.000044741 | -0.005166297 | 1.765517x |
| q3 + fp16/o200 | 0.297203978 | 0.463995894 | +0.000022370 | +0.000599654 | 1.673203x |
| q4 + fp16/o200 | 0.297203978 | 0.464350586 | +0.000022370 | +0.000954345 | 1.590062x |
| q8 + fp16/o200 | 0.297203978 | 0.463396240 | +0.000022370 | +0.000000000 | 1.326425x |

Support matrix:

| bits | direct IP support | fp16 overfetch200 rerank support | measured current default? | summary |
| ---: | --- | --- | --- | --- |
| 2 | yes | yes | yes | Best direct storage pressure, but direct quality loss is large; fp16 rerank recovers nDCG but still loses macro recall. |
| 3 | yes | yes | yes | Middle point between q2 and q4: direct is materially better than q2 and more compact than q4; fp16 rerank nearly matches q4 quality with slightly better storage. |
| 4 | yes | yes | yes | Current compact default remains the best measured fp16/o200 storage-quality balance among q2/q3/q4/q8. |
| 8 | yes | yes | yes | Direct q8 is near-dense and slightly positive macro on this cache, but fp16 rerank total compression is worse than q4 because the fp16 sidecar dominates. |

## Per-Dataset Highlights

Direct q3:

| dataset | nDCG@10 | recall@100 | nDCG delta | recall delta | compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.526933198 | 0.792444444 | -0.037604717 | -0.004000000 | 10.240000x |
| NFCorpus | 0.197158565 | 0.235639808 | -0.008587403 | -0.006426260 | 10.240000x |
| FiQA | 0.105546985 | 0.333332383 | -0.015713955 | -0.018345826 | 10.240000x |

q3 + fp16 rerank overfetch200:

| dataset | nDCG@10 | recall@100 | nDCG delta | recall delta | total compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.564537916 | 0.796444444 | +0.000000000 | +0.000000000 | 1.673203x |
| NFCorpus | 0.205745968 | 0.244855255 | +0.000000000 | +0.002789188 | 1.673203x |
| FiQA | 0.121328052 | 0.350687982 | +0.000067111 | -0.000990226 | 1.673203x |

Current q4 + fp16/o200 reproduced the known macro values from the descriptor: macro nDCG@10 `0.297203978370`, macro recall@100 `0.464350585508`, total compression `1.590062111801x`.

## Storage / Quality Interpretation

For CorkScrewDB default embedder packaging, q4/fp16/o200 remains the best current default compact row. q3 is real and measured, but it is not a better default than q4 on this frontier:

- Direct q3 is useful as a diagnostic midpoint: it cuts q4 direct bytes by about 24% at the same dimensionality, but loses more quality than q4 on all three datasets.
- q3 + fp16/o200 is close to q4 + fp16/o200 quality, but q4 still has better macro recall and better total compression because the direct q-bit payload is only part of total storage once the fp16 sidecar is included.
- Direct q8 is the strongest no-sidecar single-vector row on this cache, but it gives up much of the compactness benefit and q8+fp16/o200 is the weakest total-compression rerank row.
- q2 remains a storage-pressure probe, not a default candidate: direct quality loss is substantial, and even fp16/o200 leaves macro recall below dense/q3/q4/q8 rerank.

## Files Changed Or Inspected

Generated:

```text
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/scifact.turboquant-frontier.metrics.json
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/scifact.turboquant-frontier.metrics.tsv
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/nfcorpus.turboquant-frontier.metrics.json
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/nfcorpus.turboquant-frontier.metrics.tsv
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/fiqa.turboquant-frontier.metrics.json
runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/fiqa.turboquant-frontier.metrics.tsv
.tiller/scratch/codex/eos-current-default-turboquant-frontier-v1.vectorcache-root
.tiller/scratch/codex/eos-current-default-turboquant-frontier-v1.run-root
.tiller/scratch/codex/eos-current-default-turboquant-frontier-v1-report.md
```

Failed/partial harness attempts:

```text
runs/eos-current-default-turboquant-frontier-v1-20260624T071156Z/
```

Inspected:

```text
docs/default-corkscrew-embedder-plan.md
docs/turboquant-multivector-frontier.md
scripts/score_manta_embed_v1_baselines.fw
cmd/eos/main.go
.tiller/scratch/codex/eos-turboquant-prefix-lowbit-probe-v1-report.md
.tiller/scratch/codex/eos-fp16-rerank-sidecar-report.md
```

The worktree currently has tracked modifications in `cmd/eos/main.go`, `cmd/eos/main_test.go`, `runtime/embedding_train_runner.go`, `runtime/embedding_train_runner_test.go`, and `scripts/train_manta_embed_v1_candidate.fw`. I did not edit those files during this descriptor and did not revert them.

## Verification Commands And Results

Help/support inspection:

```bash
runs/eos-current-default-turboquant-frontier-v1-20260624T071156Z/bin/manta eval-retrieval-vectors-turboquant --help
```

Result: command reports `--bits` as comma-separated TurboQuant IP bit widths, supported `2..8`, and supports `--rerank-storage fp16`.

Successful vector-cache eval, repeated for `scifact`, `nfcorpus`, and `fiqa`:

```bash
runs/eos-current-default-turboquant-frontier-v1-20260624T071156Z/bin/manta eval-retrieval-vectors-turboquant \
  --dataset <dataset> --split test --top-k 100 \
  --backend eos-vector-cache \
  --artifact assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll \
  --bits 2,3,4,8 \
  --rerank-overfetch 200 \
  --rerank-storage fp16 \
  --doc-vectors <cache>/doc-vectors.jsonl \
  --query-vectors <cache>/query-vectors.jsonl \
  --metrics-json runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/<dataset>.turboquant-frontier.metrics.json \
  --metrics-tsv runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/<dataset>.turboquant-frontier.metrics.tsv \
  datasets/manta-embed-v1/raw/<dataset>/<dataset>
```

Result: passed for all three datasets. The emitted rows include `turboquant_ip_b3` and `turboquant_ip_b3_overfetch200_fp16_rerank`.

Failed live scoreboard harness attempt:

```bash
EOS_SCOREBOARD_TURBOQUANT_BITS=2,3,4,8 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=200 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16 \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

Result: after setup fixes, the run built `bin/manta` but the first dense `eval-retrieval` command for SciFact exited `255` with empty stdout/stderr logs. This failure happened before TurboQuant/q3 evaluation.

JSON parse/aggregation:

```bash
jq 'keys, .rows[0]' runs/eos-current-default-turboquant-frontier-v1-vectorcache-20260624T071623Z/scifact.turboquant-frontier.metrics.json
python3 - <<'PY' ... aggregate macro rows ...
```

Result: JSON parsed and produced the macro table above.

## Caveats / Residual Risk

- The completed frontier uses vector caches generated from the durable default asset, not live sealed-artifact embedding from `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll`.
- The vector-cache dense quality matches the current default dense values, and q4/fp16/o200 reproduces the known macro compact values, so this is strong evaluator evidence for the current default vectors.
- The live scoreboard failure remains unexplained because the command exited `255` with empty logs. It was not a q3 support failure.
- No source tests were run because this was an eval/audit descriptor and no source edits were made.

## Checkpoint Candidate

Yes for generated evidence/report artifacts: this descriptor produced a meaningful report plus full short-set q2/q3/q4/q8 vector-cache metrics under the requested run prefix. No VCS commit was made.

## Arbiter Next Action

Record q3 as supported and measured for the current default vector-cache evaluator. Keep q4/fp16/o200 as the current compact default. Use q3 as a diagnostic frontier point only unless a later packaging decision explicitly values its small sidecar-inclusive storage gain over q4's better macro recall.
