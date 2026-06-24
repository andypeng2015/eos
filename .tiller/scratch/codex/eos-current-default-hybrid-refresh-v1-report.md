# eos-current-default-hybrid-refresh-v1 Report

## Outcome

Completed a current-default fixed-policy hybrid retrieval refresh across SciFact, NFCorpus, and FiQA using the existing `cmd/eos` retrieval eval commands. No model was promoted and no core source was changed.

Run root:

- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/`

Current default artifact evaluated:

- `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll`

Policy evaluated:

- Dense + lexical BM25 hybrid
- `eval-retrieval-hybrid --method minmax --alpha 0.5 --top-k 100`
- Split: `test`
- Dataset root: `datasets/manta-embed-v1/raw`

The prior broad claim is directionally verified for the current default: fixed hybrid lifts dense nDCG@10 from about `0.56 -> 0.72` on SciFact, `0.21 -> 0.31` on NFCorpus, and `0.12 -> 0.22` on FiQA.

## Distillation

| dataset | docs | queries | dense nDCG@10 | BM25 nDCG@10 | hybrid nDCG@10 | hybrid delta vs dense | dense R@100 | BM25 R@100 | hybrid R@100 | hybrid recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 5,183 | 300 | 0.564537915509 | 0.661834847210 | 0.717644867485 | +0.153106951976 | 0.796444444444 | 0.885222222222 | 0.932888888889 | +0.136444444444 |
| NFCorpus | 3,633 | 323 | 0.205745967861 | 0.304513179332 | 0.311137772829 | +0.105391804969 | 0.242066067460 | 0.236735617312 | 0.290278895553 | +0.048212828093 |
| FiQA | 57,600 | 648 | 0.121260940614 | 0.232144918919 | 0.219415915378 | +0.098154974764 | 0.351678208623 | 0.493783060218 | 0.500980325402 | +0.149302116779 |
| macro | - | - | 0.297181607995 | 0.399497648487 | 0.416066185231 | +0.118884577236 | 0.463396240176 | 0.538580299918 | 0.574716036615 | +0.111319796439 |

Product gate recommendation:

- Pass for a first-class opt-in `eos-hybrid` retrieval mode over the current default dense artifact.
- Do not treat this as dense model promotion or as proof that a single fixed hybrid policy should silently replace all dense retrieval defaults.
- Before automatic/default selection, add or confirm product controls for policy identity, alpha/method selection, sparse-only fallback/diagnostics, and serving/API parity. FiQA is the main caution: fixed hybrid improves dense and recall, but BM25-only has higher nDCG@10 than fixed alpha-0.5 hybrid.

## Files Changed / Inspected

Changed:

- `.tiller/scratch/codex/eos-current-default-hybrid-refresh-v1-report.md`

Generated:

- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/`
- `.tiller/scratch/codex/eos-current-default-hybrid-refresh-v1.run-root`

Key generated artifacts:

- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/scifact.dense.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/scifact.bm25.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/scifact.hybrid-minmax-a0p5.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/nfcorpus.dense.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/nfcorpus.bm25.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/nfcorpus.hybrid-minmax-a0p5.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/fiqa.dense.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/fiqa.bm25.metrics.json`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/fiqa.hybrid-minmax-a0p5.metrics.json`

Inspected:

- `AGENTS.md`
- `CHANGELOG.md`
- `docs/default-corkscrew-embedder-plan.md`
- `scripts/calibrate_eos_embed_hybrid_retrieval.fw`
- `scripts/calibrate_eos_sparse_lexical_head_hybrid.fw`
- `scripts/smoke_eos_corkscrewdb_hybrid_policy.fw`
- `scripts/smoke_eos_hybrid_retrieval_serving.fw`
- `.tiller/scratch/codex/eos-performance-experiment-ladder-v1.md`
- `.tiller/scratch/codex/eos-current-default-hybrid-smoke-defaults-v1-report.md`
- `.tiller/scratch/codex/eos-sparse-head-hybrid-calibration-shortset-full-v1-report.md`
- `.tiller/scratch/codex/eos-corkscrewdb-hybrid-api-nf-fiqa-q4-v1-report.md`
- `.tiller/scratch/codex/eos-current-default-hybrid-policy-docs-buckley-v1-report.md`

## Verification Commands / Results

Help inspection:

```bash
go run ./cmd/eos eval-retrieval-hybrid --help
go run ./cmd/eos eval-retrieval --help
go run ./cmd/eos eval-retrieval-bm25 --help
```

Result: commands expose the needed `--dataset`, `--split`, `--top-k`, metrics, and hybrid `--method/--alpha` knobs. Help exits with status `1` after printing usage, as expected for Go flag help.

Full refresh command shape:

```bash
RUN_ID=eos-current-default-hybrid-refresh-v1-$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PWD/runs/$RUN_ID"
ARTIFACT="$PWD/runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll"
DATA_ROOT="$PWD/datasets/manta-embed-v1/raw"
mkdir -p "$RUN_DIR/bin" "$RUN_DIR/logs"
go build -trimpath -o "$RUN_DIR/bin/eos" ./cmd/eos
for dataset in scifact nfcorpus fiqa; do
  ds_dir="$DATA_ROOT/$dataset/$dataset"
  "$RUN_DIR/bin/eos" eval-retrieval --dataset "$dataset" --split test --top-k 100 --batch-size 64 --metrics-json "$RUN_DIR/${dataset}.dense.metrics.json" --per-query-jsonl "$RUN_DIR/${dataset}.dense.per-query.jsonl" "$ARTIFACT" "$ds_dir"
  "$RUN_DIR/bin/eos" eval-retrieval-bm25 --dataset "$dataset" --split test --top-k 100 --metrics-json "$RUN_DIR/${dataset}.bm25.metrics.json" --per-query-jsonl "$RUN_DIR/${dataset}.bm25.per-query.jsonl" "$ds_dir"
  "$RUN_DIR/bin/eos" eval-retrieval-hybrid --dataset "$dataset" --split test --top-k 100 --batch-size 64 --method minmax --alpha 0.5 --metrics-json "$RUN_DIR/${dataset}.hybrid-minmax-a0p5.metrics.json" --per-query-jsonl "$RUN_DIR/${dataset}.hybrid-minmax-a0p5.per-query.jsonl" "$ARTIFACT" "$ds_dir"
done
```

Result: completed for all three datasets.

JSON validation:

```bash
RUN_DIR=$(cat .tiller/scratch/codex/eos-current-default-hybrid-refresh-v1.run-root)
jq empty "$RUN_DIR"/*.metrics.json
```

Result: passed.

Git status after run:

```bash
git status --short
```

Result: pre-existing source modifications remain in `cmd/eos/main.go`, `cmd/eos/main_test.go`, `runtime/embedding_train_runner.go`, `runtime/embedding_train_runner_test.go`, and `scripts/train_manta_embed_v1_candidate.fw`. This task did not modify those files.

## Caveats / Residual Risk

- This is local command-level lexical BM25 + dense hybrid evidence, not CorkScrewDB remote/HNSW/federation/hosted parity evidence.
- The fixed alpha-0.5 hybrid policy was chosen to refresh current-default product-lane evidence with minimal sweep cost. It is not an exhaustive calibration run.
- The run overlapped with another current-default eval job in the workspace, so wall-clock runtime is not a service latency/SLO measurement.
- FiQA fixed hybrid improves dense nDCG@10 and recall@100, but BM25-only scored higher on nDCG@10. Product UI/API should keep the sparse contribution and policy identity explicit.
- Generated `runs/` artifacts are evidence artifacts only and remain outside source promotion.

## Checkpoint Candidate

Yes. Meaningful evidence/report artifact created without source changes.

Checkpoint paths:

- `.tiller/scratch/codex/eos-current-default-hybrid-refresh-v1-report.md`
- `.tiller/scratch/codex/eos-current-default-hybrid-refresh-v1.run-root`
- `runs/eos-current-default-hybrid-refresh-v1-20260624T071052Z/`

## Arbiter Next Action

Adopt `eos-hybrid` as a first-class opt-in product retrieval lane for the current default, with explicit policy metadata and caveats. Do not promote a model or silently replace dense defaults from this evidence alone; require a separate product/API decision for default selection and CorkScrewDB serving parity.
