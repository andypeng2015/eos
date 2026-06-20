# eos-vector-turboquant-perquery-harness-v1 Report

## Outcome

Implemented `--per-query-jsonl` and `--per-query-top-k` for `cmd/eos eval-retrieval-vectors-turboquant` by wiring the existing TurboQuant per-query runtime support through the vector-cache CLI path.

Added CLI/output test coverage in the existing tiny vector-cache TurboQuant command test.

Ran QMSum sparse-parent q4/q6/q7/q8 verification through the promoted flag. The run reproduced the prior aggregate pattern and wrote per-query JSONL directly from the main command:

- q4 nDCG@10: `0.496683742547263`
- q6 nDCG@10: `0.543815705447955`
- q7 nDCG@10: `0.546491679143015`
- q8 nDCG@10: `0.543815705447955`

## Distillation

The ad hoc QMSum sparse-parent sensitivity driver is no longer required for single-vector TurboQuant per-query diagnostics. The main vector-cache command now writes one JSONL row per evaluated query and method, using the existing `TurboQuantRetrievalPerQueryRow` schema (`manta.embedding_turboquant_retrieval_per_query.v1`).

The verification JSONL has `80` rows: `20` QMSum queries x `4` direct TurboQuant bit-width methods. Averaging per-query nDCG@10 by method matches the metrics JSON rows to `1e-12` absolute tolerance.

## Files Changed / Inspected / Generated

Changed:

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`

Inspected:

- `runtime/retrieval_turboquant.go`
- `runtime/retrieval_eval.go`
- `.tiller/scratch/codex/eos-qmsum-sparse-parent-tq-sensitivity-v1-report.md`

Generated:

- `.tiller/scratch/codex/eos-vector-turboquant-perquery-harness-v1-run-root.txt`
- `.tiller/scratch/codex/eos-vector-turboquant-perquery-harness-v1-report.md`
- `runs/eos-vector-turboquant-perquery-harness-v1-20260620T135444Z/eos-sparse-encoder-parent-turboquant-q4-q6-q7-q8.metrics.json`
- `runs/eos-vector-turboquant-perquery-harness-v1-20260620T135444Z/eos-sparse-encoder-parent-turboquant-q4-q6-q7-q8.metrics.tsv`
- `runs/eos-vector-turboquant-perquery-harness-v1-20260620T135444Z/eos-sparse-encoder-parent-turboquant-q4-q6-q7-q8.per-query.jsonl`
- `runs/eos-vector-turboquant-perquery-harness-v1-20260620T135444Z/logs/eval-qmsum-sparse-parent-q4-q6-q7-q8.log`

## Verification Commands and Results

```bash
gofmt -w cmd/eos/main.go cmd/eos/main_test.go
```

Result: completed.

```bash
go test ./cmd/eos
```

Result: `ok m31labs.dev/eos/cmd/eos 56.637s`.

```bash
RUN_DIR="runs/eos-vector-turboquant-perquery-harness-v1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR/logs"
printf '%s\n' "$RUN_DIR" > .tiller/scratch/codex/eos-vector-turboquant-perquery-harness-v1-run-root.txt
/usr/bin/time -p go run ./cmd/eos eval-retrieval-vectors-turboquant \
  --dataset qmsum \
  --backend eos-sparse-encoder-parent-128d \
  --artifact eos-long-context-wedge-maxseq4096.mll \
  --doc-vectors runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/vectors/eos-sparse-encoder-parent/doc-vectors.jsonl \
  --query-vectors runs/eos-qmsum-sparse-enabled-long-context-scoreboard-v1-20260620T124919Z/vectors/eos-sparse-encoder-parent/query-vectors.jsonl \
  --bits 4,6,7,8 \
  --metrics-json "$RUN_DIR/eos-sparse-encoder-parent-turboquant-q4-q6-q7-q8.metrics.json" \
  --metrics-tsv "$RUN_DIR/eos-sparse-encoder-parent-turboquant-q4-q6-q7-q8.metrics.tsv" \
  --per-query-jsonl "$RUN_DIR/eos-sparse-encoder-parent-turboquant-q4-q6-q7-q8.per-query.jsonl" \
  datasets/longembed-official/qmsum 2>&1 | tee "$RUN_DIR/logs/eval-qmsum-sparse-parent-q4-q6-q7-q8.log"
```

Result: completed in `real 42.01`. Output summary:

- dense nDCG@10 `0.546492`, recall@100 `1.000000`, vector bytes `10240`
- q4 nDCG@10 `0.496684`, delta `-0.049808`, recall@100 `1.000000`, compression `7.53x`
- q6 nDCG@10 `0.543816`, delta `-0.002676`, recall@100 `1.000000`, compression `5.12x`
- q7 nDCG@10 `0.546492`, delta `+0.000000`, recall@100 `1.000000`, compression `4.41x`
- q8 nDCG@10 `0.543816`, delta `-0.002676`, recall@100 `1.000000`, compression `3.88x`

```bash
RUN_DIR=$(cat .tiller/scratch/codex/eos-vector-turboquant-perquery-harness-v1-run-root.txt)
python3 - <<'PY' "$RUN_DIR"
# JSONL parse/schema/count validation and per-query aggregate comparison.
PY
git diff --check
```

Result:

- validated `80` JSONL rows across `4` methods
- each method has `20` query rows
- per-query average nDCG@10 matches metrics JSON rows to `1e-12`
- `git diff --check` passed with no output

## Caveats / Residual Risk

- QMSum verification is still the capped `20` doc / `20` query sparse-parent cache diagnostic; recall@100 remains trivial at this scale.
- `--per-query-top-k` is exposed for parity with `eval-retrieval-turboquant`, but the existing TurboQuant runtime keeps a metric-depth floor for per-query rows. In this path it is most useful for expanding diagnostics beyond the default floor, not reducing output depth.
- Generated `runs/` outputs are verification artifacts and should not be committed unless the parent intentionally wants evidence artifacts.

## Checkpoint Candidate

Yes. Source/test change is small and verified with `go test ./cmd/eos`, `git diff --check`, and QMSum sparse-parent JSONL validation.

Checkpoint source paths:

- `cmd/eos/main.go`
- `cmd/eos/main_test.go`

Optional report path:

- `.tiller/scratch/codex/eos-vector-turboquant-perquery-harness-v1-report.md`

## Arbiter Next Action

Checkpoint the source/test change. Keep QMSum q4 as **NO-PROMOTE**; q7 remains the first measured direct bit width preserving dense nDCG@10 on this capped sparse-parent diagnostic with material compression.
