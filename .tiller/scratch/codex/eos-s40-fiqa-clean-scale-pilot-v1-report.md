# eos-s40-fiqa-clean-scale-pilot-v1

## Outcome

Decision: `blocked_training_exit255`; dense eval not run; compact eval not run.

I ran the bounded exploration setup and started exactly one training run. Data validation and run-local mixed training JSONL construction succeeded, and the wrapper completed Go tests, build, artifact copy, train/eval/hard-eval pretokenization, and workload planning. The actual `manta train-embed` step then ran for about 1h50m and exited through the wrapper as `error: go run: exit status 255` before writing `train.metrics.json`, final eval metrics, hard eval metrics, or any trained/restored-best artifact evidence.

No source files, defaults, assets, docs, aliases, promoted artifacts, commits, pushes, or dataset files were changed. Generated artifacts were left in place.

## Distillation

- Mixed pilot data was hard-negative-compatible: all three inputs had non-empty `negatives` on every row.
- Run-local train file has `100000` rows, `quality_claim=false`, deterministic shuffle, source tags preserved, and zero exact test query-positive pair hits after selection.
- The only training attempt did not produce train metrics, restore-best evidence, or a trained candidate; therefore dense exploration and promotion gates are `not_evaluated`.
- Compact/TurboQuant eval was correctly skipped because dense exploration gate could not be evaluated.

## Run Paths

- Run root: `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z`
- Run-root pointer: `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1.run-root`
- Candidate wrapper dir: `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate`
- Train file: `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/data/train-mixed-clean-fiqa-hardneg-capped-100k.jsonl`
- Train manifest: `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/data/train-mixed-clean-fiqa-hardneg-capped-100k.manifest.json`
- Outer wrapper log: `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/logs/train-driver.outer.log`
- Train step log: `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/logs/train.log`

## Data Validation / Schema Choice

Schema decision: mixed source is hard-negative compatible; all input rows have `query`, `positive`, and non-empty `negatives`, so I used the capped mixed pilot rather than the fallback hard-negative-only pilot.

Input validation:

| source | input rows | rows with negatives | teacher-score rows | compatible |
| --- | ---: | ---: | ---: | --- |
| `shipping-mixed-pretrain-plus-beir.jsonl` | `78227` | `78227` | `0` | yes |
| `relabel/fiqa-hn-merged-train.jsonl` | `20265` | `20265` | `20265` | yes |
| `train-hard-negatives-plus-model.jsonl` | `9838` | `9838` | `0` | yes |

Selected train rows:

| source bucket | selected |
| --- | ---: |
| `shipping_mixed_pretrain_plus_beir` | `69971` |
| `clean_fiqa_relabel_hn_merged` | `20265` |
| `existing_mixed_hard_negatives_plus_model` | `9764` |
| total | `100000` |

The original target was approximately `72200/20265/9838`; exact test-pair filtering removed rows from the existing mixed hard-negative source, so I used all usable rows from that source (`9764`) and filled the 100k cap from the shipping mixed source. Selection used dedupe key `query + positive + negatives tuple + source` and deterministic `Python random.Random(2026062401).shuffle`.

Train JSONL SHA256:

```text
eb5dad81341cba069d544b8deb91df4ee73dc3fcefe91128887704632b9d878a  runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/data/train-mixed-clean-fiqa-hardneg-capped-100k.jsonl
```

Manifest has `quality_claim=false`.

## Training Attempt

The wrapper command used `scripts/train_manta_embed_v1_candidate.fw` directly and set the requested restore-best controls:

- `EOS_RESTORE_BEST=true`
- `EOS_EVAL_EVERY=1`
- `EOS_EVAL_EVERY_STEPS=0`
- `EOS_PATIENCE=2`

Planning succeeded:

```text
planned workload: train=100000 hard_negative_contrastive examples batch=32 steps/epoch=3125 train_pairs/epoch=33000000 eval=1670 pairwise examples eval_pairs/pass=1670 eval_passes(planned=4 actual=0) pairs(planned=66006680 actual=0)
```

Training failure:

```text
== Train eos-embed-v1 from prepared JSONL ==
planned workload: train=100000 hard_negative_contrastive examples batch=32 steps/epoch=3125 train_pairs/epoch=33000000 eval=1670 pairwise examples eval_pairs/pass=1670 eval_passes(planned=4 actual=0) pairs(planned=66006680 actual=0)
error: go run: exit status 255
```

The train log contains no stack trace or epoch/eval summary beyond the planned workload line. No `train.metrics.json`, `final-eval.metrics.json`, or `hard-eval.metrics.json` was produced.

## Restore-Best Evidence

Not available. `train.metrics.json` was not produced, so the required checks could not be satisfied:

- `config.restore_best=true`: not verifiable from train metrics
- `actual_eval_passes >= 3`: not verifiable
- `restored_best` / `best_epoch`: not present

## Artifact Status

The wrapper copied/packaged the starting artifact before training, but the train step failed before a trained candidate or post-train sealed package was produced. These files exist in the candidate dir but should be treated as pre-train/intermediate wrapper artifacts, not as a candidate result:

```text
188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf  runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/eos-embed-v1.mll
f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a  runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/eos-embed-v1.sealed.mll
```

Movement note: no artifact was promoted, installed, aliased, committed, or pushed.

## Dense Metrics / Gates

Dense metrics: not evaluated because training failed before producing a valid trained candidate.

Exploration gate: `not_evaluated`.

Promotion gate: `not_evaluated`; do not promote.

BM25 comparison: not run.

Compact/TurboQuant status: skipped because dense exploration gate was not evaluated/passed.

## Verification

Passed:

```bash
jq -e '.output_rows, .output_sha256, .quality_claim, .row_selection.postwrite_exact_test_query_positive_pairs' \
  runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/data/train-mixed-clean-fiqa-hardneg-capped-100k.manifest.json
```

Key outputs: `output_rows=100000`, `quality_claim=false`, `postwrite_exact_test_query_positive_pairs=0`.

Passed:

```bash
python3 - <<'PY'
# Parsed train JSONL, data manifest, candidate manifest, and tokenized train/eval/hard-eval JSONL.
PY
```

Key outputs:

- train JSONL: `100000` rows, SHA `eb5dad81341cba069d544b8deb91df4ee73dc3fcefe91128887704632b9d878a`
- tokenized train JSONL: `100000` rows
- tokenized eval JSONL: `1670` rows
- tokenized hard-eval JSONL: `1672` rows

Passed preflight/final disk checks:

- Preflight disk: `/dev/sdd` had `121G` available.
- Final disk: `/dev/sdd` had `115G` available.

Passed wrapper gates before train:

- `go test ./cmd/eos ./runtime ./models -count=1`
- `go build -trimpath -o .../candidate/bin/manta ./cmd/eos`
- `tokenize-embed --mode hard-negative`: `100000` examples
- `tokenize-embed --mode pair` eval: `1670` examples
- `tokenize-embed --mode pair` hard eval: `1672` examples
- plan-only workload emitted successfully

Failed:

- Actual train step exited through wrapper as `error: go run: exit status 255`.

Not run / not available:

- `train.metrics.json` restore-best checks
- final/hard eval-only gate checks
- package/sealed post-train inspect
- dense retrieval metrics
- compact/TurboQuant metrics

Final VCS status:

```text
## main...origin/main
```

## Caveats / Residual Risk

- The failed train command did not emit a useful stack trace or diagnostic beyond exit 255. It ran CPU-active for about 1h50m before returning.
- Because the wrapper only wrote the planned workload line to `candidate/logs/train.log`, root cause is unclear from artifacts alone.
- The candidate directory contains pre-train copied/packaged files; do not score or promote them as this experiment's trained output.
- I did not retry training, reduce batch size, change data scale, or start a second training run because the descriptor allowed exactly one training run after data validation.

## Files Changed / Generated / Inspected

Generated:

- `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/`
- `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1.run-root`
- `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-v1-report.md`

Inspected:

- `.tiller/scratch/codex/eos-scaled-data-gate-plan-v1.md`
- `.tiller/scratch/codex/eos-fixed-trainer-sentinel-rerun-v1-report.md`
- `scripts/train_manta_embed_v1_candidate.fw`
- `cmd/eos/main.go`
- `datasets/manta-embed-v1/processed/shipping-mixed-pretrain-plus-beir.jsonl`
- `datasets/manta-embed-v1/processed/relabel/fiqa-hn-merged-train.jsonl`
- `datasets/manta-embed-v1/processed/train-hard-negatives-plus-model.jsonl`
- `datasets/manta-embed-v1/processed/eval.jsonl`
- `datasets/manta-embed-v1/processed/hard-eval.jsonl`

## Checkpoint Candidate

No source checkpoint candidate. This is generated experiment evidence only, and the experiment did not complete training.

## Arbiter Next Action

Do not promote. Do not compact-evaluate. Next action should be a debugger/root-cause descriptor against `manta train-embed` exit 255 on the run-local 100k tokenized hard-negative file, using `runs/eos-s40-fiqa-clean-scale-pilot-v1-20260624T051401Z/candidate/logs/train.log.cmd` as the exact command source. A follow-up may run a diagnostic-only reproduction with progress enabled or a smaller row cap, but it should be a new descriptor because this one already consumed its single allowed training attempt.
