# eos-frontier-delta-ledger-and-consensus-miner-v1 Report

## Outcome

Built a full test-split per-query frontier ledger for `scifact`, `nfcorpus`, and `fiqa` across:

- Eos dense current s40/default asset
- Eos compact q4/fp16/rerank-overfetch200 default policy
- BM25
- current-default hybrid `minmax_blend alpha=0.5`
- Qwen3 vector cache
- mxbai vector cache

No model training was run. No source files or tracked docs were edited.

Run root:

- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/`

Primary artifacts:

- `frontier-ledger-summary.json`
- `frontier-ledger-summary.tsv`
- `candidate-training-manifest.json`
- `{scifact,nfcorpus,fiqa}.consensus-relabel-candidates.jsonl`
- `{scifact,nfcorpus,fiqa}.disagreement-drop-cases.jsonl`
- `protection-query-outcomes.jsonl`

## Distillation

The ledger materially supports the next dense-quality move without repeating broad soft-teacher pressure:

| dataset | qrel queries | Eos miss@10 queries | Qwen3+mxbai both hit@10 | consensus relabel/protection candidates | drop/review cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 300 | 101 | 245 | 69 | 104 |
| NFCorpus | 323 | 167 | 227 | 80 | 202 |
| FiQA | 648 | 452 | 427 | 259 | 430 |
| total | 1271 | 720 | 899 | 408 | 736 |

The candidate manifest marks `eligible_for_later_candidate=true` because there are 408 Eos-miss/external-consensus rows. These should be used as silver relabel/protection/drop selection data for a later `eos-consensus-mined-no-soft-teacher-candidate` with `EOS_TEACHER_LOSS_WEIGHT=0`, not as direct soft teacher targets.

All six expected per-query sources have full qrel-query coverage for every dataset:

- SciFact: 300/300 rows per source
- NFCorpus: 323/323 rows per source
- FiQA: 648/648 rows per source

Selected aggregate per-query means from the generated ledger:

| dataset | source | nDCG@10 | recall@100 |
| --- | --- | ---: | ---: |
| SciFact | eos_dense | 0.564538 | 0.796444 |
| SciFact | eos_compact_q4_fp16_o200 | 0.564538 | 0.796444 |
| SciFact | hybrid_minmax_blend_a050 | 0.717645 | 0.932889 |
| SciFact | qwen3 | 0.702026 | 0.946667 |
| SciFact | mxbai | 0.738932 | 0.965000 |
| NFCorpus | eos_dense | 0.205571 | 0.242059 |
| NFCorpus | eos_compact_q4_fp16_o200 | 0.205571 | 0.244151 |
| NFCorpus | hybrid_minmax_blend_a050 | 0.311171 | 0.290150 |
| NFCorpus | qwen3 | 0.367229 | 0.344169 |
| NFCorpus | mxbai | 0.387018 | 0.372199 |
| FiQA | eos_dense | 0.121261 | 0.351678 |
| FiQA | eos_compact_q4_fp16_o200 | 0.121148 | 0.351678 |
| FiQA | hybrid_minmax_blend_a050 | 0.219416 | 0.500672 |
| FiQA | qwen3 | 0.449201 | 0.796879 |
| FiQA | mxbai | 0.452692 | 0.774386 |

`protection-query-outcomes.jsonl` includes the known FiQA blocker/protection IDs from prior hybrid reports (`9824`, `7345`, `6832`, `7529`, `7705`, `4605`) plus 20 NFCorpus Eos-miss rows where Qwen3 and mxbai both hit and agree on top1, for bounded protection review.

## Files Changed / Generated / Inspected

Tracked scratch report generated:

- `.tiller/scratch/codex/eos-frontier-delta-ledger-and-consensus-miner-v1-report.md`

Ignored run artifacts generated:

- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/eos`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/scripts/build_frontier_ledger.py`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/per-query/*.jsonl`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/metrics/*.json`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/metrics/*.tsv`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/logs/*.log`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/frontier-ledger-summary.json`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/frontier-ledger-summary.tsv`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/candidate-training-manifest.json`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/*.consensus-relabel-candidates.jsonl`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/*.disagreement-drop-cases.jsonl`
- `runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/protection-query-outcomes.jsonl`

Key inspected context:

- `AGENTS.md`
- `.tiller/scratch/codex/eos-qwen3-catchup-frontier-plan.md`
- `.tiller/scratch/codex/eos-embed-next-quality-lever-report.md`
- `.tiller/scratch/codex/eos-protected-scifact-teacher-lowpressure-report.md`
- `.tiller/scratch/codex/mxbai-shortset-scoreboard-report.md`
- `.tiller/scratch/codex/qwen3-leading-embedder-harness-report.md`
- `.tiller/scratch/codex/eos-official-longembed-external-qwen3-mxbai-compare-v2-report.md`
- `docs/production-embedding.md`
- `docs/benchmarks.md`
- `docs/manta-embed-sota-avenues.md`
- `docs/default-corkscrew-embedder-plan.md`
- `scripts/score_teacher_with_vector_cache.py`
- `scripts/score_manta_embed_v1_baselines.fw`
- `cmd/eos/main.go`
- `runtime/embedding_teacher_relabel.go`
- `runtime/retrieval_hard_negative_mining.go`

## Verification

Passed:

```bash
go build -o runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/eos ./cmd/eos
```

Passed evaluator commands, summarized:

- `eval-retrieval-vectors` for Qwen3 and mxbai on SciFact/NFCorpus/FiQA.
- `eval-retrieval-bm25` on SciFact/NFCorpus/FiQA.
- `eval-retrieval-hybrid --method minmax_blend --alpha 0.5` on SciFact/NFCorpus/FiQA.
- `eval-retrieval` with `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll` on SciFact/NFCorpus/FiQA.
- `eval-retrieval-turboquant --bits 4 --rerank-overfetch 200 --rerank-storage fp16` with the same asset on SciFact/NFCorpus/FiQA.

Passed JSON validation:

```bash
jq empty \
  runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/frontier-ledger-summary.json \
  runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/candidate-training-manifest.json \
  runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/metrics/*.json

for f in runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/*.jsonl \
         runs/eos-frontier-delta-ledger-and-consensus-miner-v1-20260621T193911Z/per-query/*.per-query.jsonl; do
  jq -c empty "$f"
done
```

Result: pass.

Row count verification:

- `frontier-ledger-summary.json` totals: `ledger_rows=1271`, `qrel_queries=1271`.
- No per-source row count mismatches in `candidate-training-manifest.json`.
- Extracted compact rows: SciFact `300`, NFCorpus `323`, FiQA `648`.

## Caveats / Residual Risk

- FiQA Qwen3 initially failed against `runs/external-vector-caches/qwen3-0.6b/fiqa/` because that shorter cache has no vectors for the test qrels. The final FiQA Qwen3 row uses `runs/external-vector-caches/qwen3-fiqa-full-20260615T033000Z/fiqa/`, so treat it as full exportable-text/sanitized-cache evidence, not raw-row-complete evidence.
- The compact Eos source is q4/fp16/rerank-overfetch200, matching current s40 docs. The older frontier-plan text mentions q4/fp16/o250; this run used the newer current default policy documented in `docs/default-corkscrew-embedder-plan.md`.
- The hybrid row is current-default `minmax_blend alpha=0.5` test evidence, not a dense model promotion.
- Consensus candidates are not sufficient by themselves to prove future training will improve dense Eos. They are a safer data-selection input than broad teacher loss.
- NFCorpus still needs caution: prior reports flagged anti-label teacher behavior. This ledger uses Qwen3+mxbai agreement with qrels/ranking outcomes to select review/relabel/drop candidates, but later training should keep NFCorpus teacher scores stripped unless a fresh audit explicitly clears them.

## Checkpoint Candidate

Yes, evidence checkpoint candidate for the tracked scratch report plus, if the project wants durable ignored evidence, the run root.

Do not treat this as a model/source promotion checkpoint. No commits were made.

## Arbiter Next Action

Queue `eos-consensus-mined-no-soft-teacher-candidate` using `candidate-training-manifest.json`, with `EOS_TEACHER_LOSS_WEIGHT=0`. Use consensus rows as supervised silver/protection selections and disagreement rows as drop/review filters. Keep NFCorpus/FiQA teacher scores stripped unless a new audit proves positive margin alignment.
