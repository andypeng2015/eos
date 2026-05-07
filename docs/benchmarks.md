# Benchmarks

Manta has three benchmark layers:

- Go microbenchmarks for isolated runtime kernels and trainer math.
- End-to-end training smokes for the native default embedding-model path.
- Embedder scoreboards for retrieval quality and serving efficiency claims.

Run the default microbenchmarks with Ferrous Wheel:

```bash
MANTA_BENCH_ROOT=$PWD ferrous-wheel run scripts/bench.fw
```

Run CUDA microbenchmarks:

```bash
MANTA_BENCH_ROOT=$PWD MANTA_BENCH_CUDA=1 ferrous-wheel run scripts/bench.fw
```

Run the sparse-attention x TurboQuant measurement layer:

```bash
MANTA_REPO_ROOT=$PWD ferrous-wheel run scripts/bench_sparse_attention.fw
```

The sparse-attention harness first writes a routed preflight plan as `sparse-attention-plan.tsv` and `sparse-attention-plan.json`, then records CUDA benchmark output as `sparse-attention-bench.jsonl`, `sparse-attention-bench.txt`, `sparse-attention-bench-summary.tsv`, and `sparse-attention-scaling.tsv` under `runs/<run-id>/`. The Go benchmark lines and summary table include selected-key count, candidate-key budget, estimated scores per query, score fraction, subquadratic-plan flag, TurboQuant K/V MiB, and logical K/V compression ratio. The scaling table fits log-log time alpha for exact f16 and routed TurboQuant rows, and the harness fails routed TurboQuant when alpha exceeds `MANTA_SPARSE_BENCH_MAX_ROUTED_TIME_ALPHA` (`0.95` by default; set `0` to disable during exploratory runs).

Run the default-model training smoke from a local asset package:

```bash
MANTA_BENCH_ROOT=$PWD MANTA_BENCH_MODEL_ASSETS=/path/to/assets/manta-embed-v1 ferrous-wheel run scripts/bench.fw
```

The model smoke copies the package into a temporary directory before training. It does not mutate the source asset directory.

Evaluate an existing candidate package against a token JSONL or text-pair JSONL eval file without running optimizer steps:

```bash
go run ./cmd/manta train-embed --eval-only /path/to/manta-embed-v1.mll /path/to/eval-mini.jsonl
```

When the package has a sibling `.tokenizer.mll`, text eval JSONL is tokenized automatically. Pass `--tokenizer /path/to/tokenizer.mll` to use an explicit tokenizer.

For a production candidate run with acquired BEIR-format datasets, provenance, metric gates, sealed export, and SHA256 manifests, use `scripts/acquire_manta_embed_v1_datasets.fw` followed by `scripts/train_manta_embed_v1_candidate.fw`. See `docs/production-embedding.md`.

Build the long-context embedder wedge scoreboard from an existing sealed candidate:

```bash
MANTA_REPO_ROOT=$PWD \
MANTA_SCOREBOARD_ARTIFACT=/path/to/manta-embed-v1.sealed.mll \
MANTA_SCOREBOARD_PAIRWISE_JSONL=/data/manta/datasets/manta-embed-v1/processed/eval.jsonl \
MANTA_SCOREBOARD_HARD_JSONL=/data/manta/datasets/manta-embed-v1/processed/hard-eval.jsonl \
MANTA_SCOREBOARD_RETRIEVAL_ROOT=/data/manta/datasets/manta-embed-v1 \
MANTA_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The scoreboard run writes `scoreboard.tsv`, `scoreboard.json`, per-task metrics JSON, command logs, and a run-local `manta` binary under `runs/<run-id>/`. Pairwise rows use `MANTA_SCOREBOARD_PAIRWISE_ARTIFACT` when set, or infer the sibling trainable package from a sealed artifact path. Add `MANTA_SCOREBOARD_LONG_ROOT` and `MANTA_SCOREBOARD_LONG_DATASETS` when long-document retrieval datasets are prepared.

Run a full retrieval-alignment round from an existing candidate when retrieval is behind the BM25 or open-model baselines:

```bash
MANTA_REPO_ROOT=$PWD \
MANTA_ALIGN_INITIAL_ARTIFACT=/path/to/manta-embed-v1.sealed.mll \
MANTA_ALIGN_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/run_manta_embed_v1_retrieval_alignment_round.fw
```

The alignment harness writes a baseline scoreboard, run-local model-hard negatives, a retrained candidate package, a candidate scoreboard, and `retrieval-alignment-summary.tsv/json` with per-dataset nDCG@10 and recall@100 deltas.
Candidate training defaults to batch `64` and one explicit hard negative per query. This keeps full mined rounds bounded; larger batches or more explicit negatives should be treated as throughput experiments because pair work grows quickly.
Set `MANTA_ALIGN_MODEL_HARD_DATASET_WEIGHTS=dataset=weight,...` to allocate more mined examples to weak datasets in the next mixed round.
Set `MANTA_ALIGN_CANDIDATE_SOURCE_WEIGHTS=source=weight,...` to source-balance hard-negative batches when the train JSONL has `source` fields. Family keys such as `fiqa` also apply to mined sources such as `fiqa:model` unless an exact key is present.
Set `MANTA_ALIGN_GATE_CANDIDATE=1` for promotion-style rounds. The gate records the summary, then fails when macro nDCG@10 is below `MANTA_ALIGN_MIN_MACRO_NDCG_DELTA` or any dataset regresses beyond `MANTA_ALIGN_MAX_DATASET_NDCG_REGRESSION`; `MANTA_ALIGN_MIN_DATASET_NDCG_RATIO` can enforce an additional per-dataset nDCG ratio floor. Use `MANTA_ALIGN_MAX_DATASET_RECALL_AT_100_REGRESSION` and `MANTA_ALIGN_MIN_DATASET_RECALL_AT_100_RATIO` to also guard recall@100.
Set `MANTA_ALIGN_CANDIDATE_CONTRASTIVE_LOSS=grouped_infonce` to test the query-grouped hard-negative objective. This counts only each query's own positive/negative candidate set in the training loss, which is useful as a retrieval-ranking ablation when corpus ranking regresses. The first grouped-only gated run rejected the candidate, so keep this flag as an experiment rather than a promotion path.
Set `MANTA_ALIGN_CANDIDATE_CONTRASTIVE_LOSS=hybrid_infonce` to keep the global hard-negative InfoNCE matrix and add a weighted grouped term. The first weight-`0.25` gated run improved FiQA but regressed SciFact/NFCorpus. The first strict pass used `MANTA_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.10`, `MANTA_ALIGN_CANDIDATE_LR=0.000025`, and `MANTA_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1`.
Model-hard and BM25 mining can now emit `teacher_scores` for each query's positive plus explicit negatives. Set `MANTA_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT` above zero to blend a soft teacher-distribution cross-entropy into hard-negative training, and tune `MANTA_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` when the teacher ranking is too sharp or too flat.
A lower-LR ratchet from that strict-pass checkpoint with `MANTA_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.05`, `MANTA_ALIGN_CANDIDATE_LR=0.0000125`, and `MANTA_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=3,fiqa=1` improved pairwise AUC but failed the retrieval gate. Do not treat repeated ratchets on the same mined blend as the default next step; remine model-hard negatives from the promoted artifact or add teacher-score distillation.
A fresh-mining round from the strict-pass checkpoint with `MANTA_ALIGN_MODEL_HARD_DATASET_WEIGHTS=scifact=1,nfcorpus=3,fiqa=1`, `MANTA_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.05`, `MANTA_ALIGN_CANDIDATE_LR=0.0000125`, and `MANTA_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1` passed the retrieval gate: macro nDCG@10 improved from `0.138397` to `0.145568`. Because recall@100 dipped, future promotion-style rounds should set an explicit recall floor when using this recipe.
A teacher-distilled follow-up from that checkpoint used fresh model-hard examples with `teacher_scores`, `MANTA_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT=0.20`, `MANTA_ALIGN_CANDIDATE_LR=0.000010`, and recall floors. The first full gated run with `MANTA_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1` improved macro nDCG@10 to `0.146301` but failed the NFCorpus nDCG floor. Reusing the same teacher-scored JSONL with `scifact=1,nfcorpus=3,fiqa=1` fixed that failure and raised macro nDCG@10 to `0.147862` while recall@100 stayed flat or improved on all three retrieval sets. Holding that recipe and softening `MANTA_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` to `1.5` raised macro nDCG@10 again to `0.148143`; the retrieval comparison gate passed with SciFact `0.331139`, NFCorpus `0.084325`, and FiQA `0.028967`. Temperature `1.25` and `2.0` also passed the gate but landed lower at macro `0.147645` and `0.148029`, respectively. LR `0.000008` at temperature `1.5` passed at macro `0.147625` and improved NFCorpus to `0.084927`, but lost enough SciFact to keep LR `0.000010` as the current best. Source weights `scifact=1,nfcorpus=4,fiqa=1` raised SciFact to `0.331362` but failed the NFCorpus nDCG floor and regressed FiQA. Source weights `scifact=2,nfcorpus=3,fiqa=1` passed the older fresh-baseline comparison at macro `0.146288`, but pairwise guardrails fell to validation AUC `0.815529` / hard AUC `0.808770` and macro stayed below the current best, so the local source-weight sweep is closed.
A deeper Lane B mining round from the temperature-`1.5` best requested `9000` model-hard examples with `MANTA_ALIGN_MODEL_HARD_NEGATIVES=5`, `MANTA_ALIGN_MODEL_HARD_CANDIDATE_TOP_K=400`, and `MANTA_ALIGN_CANDIDATE_HARD_NEGATIVES=2`. The run trained `13,038` blended hard-negative examples at `4262` train pairs/s with CUDA-backed forward/optimizer/activation/contrastive, but failed the promotion gate: macro nDCG@10 fell from `0.148144` to `0.143866`, SciFact fell to `0.320576`, and FiQA fell to `0.026453`. NFCorpus rose slightly to `0.084568`, so the next isolation run should reuse the same mined JSONL with one candidate hard negative.
The one-hard-negative reuse trained the same deep-mined JSONL at `5056` train pairs/s and improved NFCorpus to a new high-water mark of `0.087300`, but it still failed against the current best: SciFact `0.324630`, FiQA `0.025679`, macro `0.145870`, validation AUC `0.802976`, and hard AUC `0.800840`. Since the deep-mined JSONL already contains `5400` NFCorpus model-hard examples, the next reuse should drop the NF3 training source schedule and try balanced source weights.
Balanced source weights on that same deep-mined HN1 JSONL regressed further to macro `0.144915`: SciFact `0.322932`, NFCorpus `0.086364`, FiQA `0.025450`, validation AUC `0.794627`, and hard AUC `0.794320`. Do not continue source-sampler rescue on this file; the next local isolation should keep the NF3 HN1 shape but lower LR to test a smaller update.
Lowering the HN1 NF3 reuse to LR `0.000005` improved FiQA to `0.027508` and kept NFCorpus high at `0.086784`, but SciFact remained low at `0.323136`, macro landed at `0.145809`, validation AUC was `0.803438`, and hard AUC was `0.800508`. The remaining local check is reduced grouped pressure; otherwise move this lane to external-teacher work.
Reducing grouped pressure to `MANTA_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.025` at LR `0.000005` gave the best Lane B deep-mined balance but still failed promotion: SciFact `0.325439`, NFCorpus `0.086645`, FiQA `0.027204`, macro `0.146429`, validation AUC `0.804674`, and hard AUC `0.801615`. Close this deep-mined file for balanced promotion and move the next improvement path to external teacher import or a larger `embed-m` run.
The first `embed-m` capacity probes separated mechanical viability from quality. The full target shape (`32768` vocab, max sequence `512`, dim `192`, hidden `384`, `3` repeats) initialized, but full-corpus tokenizer training stayed CPU-bound for more than fifteen minutes before any optimizer step, so true `32768`-vocab iteration needs a cached tokenizer artifact or faster trainer first. A cached-tokenizer probe (`16384` vocab, max sequence `512`) trained and sealed at batch `64`, processing `1.393M` actual train pairs in `19m20s` with CUDA forward/optimizer/activation/contrastive and `1460.78` train pairs/s, but rejected hard: validation AUC `0.595854`, hard AUC `0.598887`, SciFact `0.160753`, NFCorpus `0.060778`, FiQA `0.012688`, macro `0.078073`. A scratch-style cached-tokenizer run with pure `infonce`, LR `0.002`, and one epoch rejected even earlier with validation AUC `0.495137` and hard AUC `0.498731`. Do not continue blind random-start `embed-m`; the next larger-model proof needs dimension-compatible bootstrapping, staged pretraining, or imported external teacher scores.

Compare a retrieval-only candidate scoreboard against a prior alignment summary without rerunning the full alignment harness:

```bash
MANTA_COMPARE_BASELINE_SUMMARY_JSON=/path/to/retrieval-alignment-summary.json \
MANTA_COMPARE_CANDIDATE_SCOREBOARD_JSON=/path/to/candidate-scoreboard/scoreboard.json \
ferrous-wheel run scripts/compare_manta_embed_v1_retrieval_candidate.fw
```

The comparison writes `retrieval-comparison-summary.tsv/json` beside the candidate scoreboard and applies the same macro nDCG@10, per-dataset nDCG, and recall@100 floors by default.

If you want a binary runner instead of `run` mode:

```bash
ferrous-wheel build scripts/bench.fw -o bin/manta-bench
bin/manta-bench
```

Capture CPU or heap profiles for any `manta` command with:

```bash
MANTA_CPU_PROFILE=/tmp/manta.cpu.pprof go run ./cmd/manta train-embed ...
MANTA_MEM_PROFILE=/tmp/manta.mem.pprof go run ./cmd/manta train-embed ...
```

Then inspect with `go tool pprof -top /tmp/manta.cpu.pprof`.

For repeatable GPU A/B profiles, use the Ferrous Wheel harness:

```bash
MANTA_REPO_ROOT=$PWD \
MANTA_GPU_PROFILE_ASSETS=/path/to/assets/manta-embed-v1 \
MANTA_GPU_PROFILE_TRAIN=/path/to/train-mini.jsonl \
MANTA_GPU_PROFILE_EVAL=/path/to/eval-mini.jsonl \
MANTA_GPU_PROFILE_VARIANTS=default,disable-batched-forward \
MANTA_GPU_PROFILE_ENV_DISABLE_BATCHED_FORWARD=MANTA_TRAIN_DISABLE_BATCHED_FORWARD=1 \
ferrous-wheel run scripts/profile_manta_gpu_efficiency.fw
```

The profile harness copies `.mll` package assets per variant, runs `train-embed`, writes each variant's `run.log`, `time.txt`, `cpu.pprof`, and `pprof-top.txt`, then writes a root `summary.tsv` with throughput and accelerator counters.
Set `MANTA_GPU_PROFILE_NO_TOKENIZER=1` when profiling already-tokenized JSONL so the copied sibling tokenizer does not force text mode.

## Current Default Model Smoke

The current reference smoke uses:

- model package: `manta-embed-v1`
- encoder repeats: `2`
- tokenizer vocab: `2454`
- max sequence length: `256`
- train set: `4096` contrastive examples
- eval set: `512` contrastive examples
- batch size: `1024`
- loss: InfoNCE
- temperature: `0.05`
- learning rate: `0.005`
- eval cadence: every `4` steps

Latest local CUDA result:

```text
throughput: elapsed=5.139s examples/s=896.74 pairs/s=867250.60 train_examples/s=845.15 train_pairs/s=865437.87 eval_examples/s=1752.73 eval_pairs/s=897395.54 optimizer_steps/s=0.83
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=30 matmul_runs=13644 matmul_run_upload_mb=3727.56 matmul_run_download_mb=2000.00 optimizer_updates=28 activation_calls=0 contrastive_calls=4
```

This is the promoted default benchmark path. It includes CUDA matmul scratch-buffer reuse, grouped batched backward, exact-length grouped contrastive forward for variable-length text, pair-length-aware bucketing across shuffled windows, strided-batched cuBLAS for grouped attention matmuls, rank-3 transpose support for grouped attention backward, batch-1024 contrastive training, sequence matmul bindings disabled by default, Q/K/V forward projection coalescing through a multi-bound-right CUDA path, Q/K/V attention-gradient accumulation through one concatenated shared-left GEMM, combined V/K attention backward gradients in one doubled-batch GEMM, Q/K/V input-gradient accumulation into one resident-right output download with one CUDA sync per accumulated group, and guarded grouped activation-backward helpers kept behind the activation accelerator flag. The larger batch keeps the full in-batch negative set intact, improves contrastive signal, cuts optimizer/contrastive calls on this smoke, and reduces per-pair orchestration overhead. Length bucketing is on by default for CLI contrastive training; set `--length-bucket-batches=false` to disable it or `MANTA_TRAIN_LENGTH_BUCKET_WINDOW` to tune the shuffled sort window. Disabling per-sequence matmul bindings trades a small upload increase for a large reduction in backend binding churn. Q/K/V coalescing preserves per-weight residency and quantization while uploading each shared left-hand activation once across query/key/value projections. Concatenated shared-left attention-gradient coalescing computes `input^T*[dQ|dK|dV]` as one standard GEMM, then splits the result back into the Q/K/V weight gradients. V/K gradient coalescing computes `scores^T*dMixed` and `dPreSoftmax^T*Q` in a single strided-batched dispatch because both use the same transpose shape. Input-gradient coalescing computes `dQ*Wq^T + dK*Wk^T + dV*Wv^T` against resident Q/K/V weights, synchronizes once after the accumulated cuBLAS calls, and downloads the accumulated result once.

Pairwise eval-only gates also use exact-length batched forward chunks by default. On the acquired hard eval set, the current default `MANTA_TRAIN_PAIR_EVAL_BATCH_SIZE=256` measured `9.85s`, `6704` matmul runs, and `4261.18 MB` uploaded versus `16.16s`, `53504` matmul runs, and `5135.84 MB` uploaded with `MANTA_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1`. Eval metrics matched within float tolerance.

Read the throughput line with both lenses:

- `train_examples/s` measures actual encoder training throughput.
- `train_pairs/s` measures in-batch contrastive work and grows roughly with `batch^2`.
- `optimizer_steps/s` protects the benchmark from hiding that very large batches reduce update count on small datasets.

## Recent Perf Delta

The training hot path moved as follows on the same mini smoke:

| Path | Train pairs/s | Matmul runs |
| --- | ---: | ---: |
| Instrumented baseline | `21975.83` | `409600` |
| CUDA scratch reuse | `22933.16` | `409600` |
| Grouped batched backward default | `33328.94` | `237568` |
| Exact-length grouped forward default | `35856.58` | `140056` |
| Strided-batched grouped attention | `42594.29` | `107552` |
| Rank-3 transpose batched attention backward | `82262.50` | `50208` |
| Batch-512 benchmark smoke | `177192.16` | `28848` |
| Batch-1024 default benchmark smoke | `407407.98` | `16464` |
| Disable sequence matmul bindings by default | `707265.87` | `16464` |
| Return grouped bound-right outputs as views | `742477.76` | `16464` |
| Concatenate shared-left Q/K/V gradient matmul | `762372.72` | `15336` |
| Combine attention V/K gradient matmuls | `803746.93` | `14772` |
| Accumulate Q/K/V input gradients with resident weights | `825766.14` | `13644` |
| Single-sync accumulated Q/K/V input gradients | `865437.87` | `13644` |

The main wins came from grouping real text batches by sequence length during backward, coalescing parameter-gradient matmuls into taller `X^T*dY` operations, grouping contrastive forward sequences by exact token length inside each original batch, promoting rank-3 x rank-3 CUDA matmul to `cublasSgemmStridedBatched`, allowing strided-batched matmul to handle transpose flags directly, and increasing the effective contrastive batch. The forward grouping keeps the full in-batch negative set intact and avoids padding, so attention math does not change.

The Q/K/V multi-bound-right path is transfer progress: it reduces matmul run uploads from `4173.15 MB` to `3727.56 MB` while preserving each weight's resident quantized form. The concatenated shared-left gradient path and combined V/K gradient path are dispatch-count wins on top of that. Accumulated input gradients keep the Q/K/V weights resident and reduce matmul run downloads from `2208.41 MB` to `2000.00 MB`. Single-sync accumulation removes the redundant per-term CUDA syncs inside that primitive while keeping the same transfer profile. Batch-1024 A/B measured `845.15 train_examples/s`, `865437.87 train_pairs/s`, and `13644` matmul runs by default versus `748.52 train_examples/s`, `766485.96 train_pairs/s`, and `13644` matmul runs with `MANTA_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1`.

Batched forward materialization now keeps downloaded backend outputs as per-sequence views and passes layer outputs forward by view instead of copying them through temporary buffers. On a real acquired-text 512 train / 128 eval A/B at batch 256, trainer elapsed moved from `9.557s` to `8.235s` and train examples/s moved from `57.91` to `70.67`; matmul upload/download counters stayed at `5838.20 MB` / `2990.88 MB`, as expected, because this cuts host copies rather than device transfer.

Attention backprop now also keeps batched Q/K/V gradient outputs as views instead of allocating zeroed per-sequence buffers and copying accelerator outputs into them. On the acquired-text 512 train / 128 eval CPU profile, `backpropAttentionSequences` moved from `4.69s` cumulative to `2.21s`, and `runtime.memclrNoHeapPointers` moved from `3.27s` to `1.26s`. Matmul counters stay unchanged; this is a host allocation and copy reduction inside the existing CUDA dispatch pattern.

The production candidate workflow keeps `MANTA_EVAL_EVERY_STEPS=0` by default. Step-level eval is still available for convergence debugging, but it inserts full eval passes into `train.log`; on the acquired full split, `--eval-every-steps 4` adds `21` extra contrastive eval passes across 3 epochs at batch 1024. Keep release gates in the final validation and hard holdout evals so training transfer reflects optimizer work first.

Ranked BPE tokenization removed a startup/data-ingest bottleneck before longer training runs. A direct batch-2048 `train-embed` CPU profile moved tokenizer encode time from `2.13s` cumulative (`BPETokenizer.Encode` -> `bpeMerge` -> `applyMerge`) to `0.25s` cumulative (`bpeMergeRanked`). End-to-end throughput remains dominated by training transfer/orchestration after tokenization, but corpus ingestion no longer burns a large fraction of host CPU.

Ranked BPE now compacts each selected merge in place instead of allocating a fresh token slice for every merge pass. On the acquired-text 512 train / 128 eval CPU profile, `applyRankedMerge` moved from `0.86s` cumulative to `0.45s`; total tokenizer encode time moved from `3.39s` to `3.10s`. The full run remains dominated by backend orchestration, memory clearing, and host-device traffic, so this is a data-ingest allocation cut rather than a matmul-counter change.

Prepared-text ingestion now keeps a trainer-local tokenization cache across train and eval loading. On the same acquired-text 512 train / 128 eval profile, with `1.38x` repeated text fields, `BPETokenizer.Encode` moved from `3.57s` cumulative to `1.98s`, train text tokenization moved from `2.82s` to `1.60s`, pair eval text tokenization moved from `0.75s` to `0.38s`, and trainer elapsed moved from `8.414s` to `7.577s`. Matmul counters stayed unchanged at `2788` runs, `5838.20 MB` uploaded, and `2990.88 MB` downloaded; this is an in-memory ingest optimization, and candidate packages, tokenizers, train profiles, checkpoints, and sealed outputs remain `.mll` artifacts.

Prepared JSONL production runs can now pretokenize once with `manta tokenize-embed` and train with `manta train-embed --no-tokenizer`. On the 512 train / 128 eval mini smoke, a same-code text JSONL profile still spent `3.61s` cumulative in `BPETokenizer.Encode` (`22.59%` of CPU samples), while the token JSONL profile removed tokenizer encode from the hot path. The GPU counters were identical for the two runs at `2712` matmul runs, `5838.20 MB` uploaded, and `2988.88 MB` downloaded; text measured `24948.46` train pairs/s and token JSONL measured `22414.32` train pairs/s in one noisy pair of local runs. Treat this as a training-profile cleanup and reproducibility win, not a claimed device-throughput gain.

Unbound activation acceleration now has a default shape ceiling so `MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL=1` does not route long-document activation groups through standalone upload/download kernels. On the tokenized acquired 4096 train / 512 eval split, fully unbounded CUDA activation regressed to `1m42.293s` and `42420.11` train pairs/s with `744` activation calls. Host activation measured `1m0.212s` and `73554.96` train pairs/s. Shape-limited opt-in measured `1m0.321s` and `73329.90` train pairs/s with `694` activation calls and identical train/eval metrics. On the smaller 512 / 128 tokenized smoke, the same shape-limited opt-in path measured `25239.78` train pairs/s versus `23997.21` for the host-activation default in the A/B run. Keep activation acceleration as an experiment until activation residency removes the extra transfers.

Fast GELU is available as an opt-in training math approximation for candidate-throughput experiments. `MANTA_TRAIN_ENABLE_FAST_GELU=1` replaces the precise tanh call in host GELU forward/backward with a bounded rational tanh approximation and keeps GELU backward on the host so forward/backward use matching math. On the tokenized 512 train / 128 eval smoke, precise GELU measured `24191.69` train pairs/s and fast GELU measured `32525.01` train pairs/s; eval AUC moved from `0.580322` to `0.581543`. On the tokenized 4096 train / 512 eval split, fast GELU measured `86779.56` train pairs/s versus the prior clean precise-GELU baseline of `73554.96`; eval AUC moved from `0.548126` to `0.547104`. Treat this as a speed/quality knob for candidate runs and keep the exact GELU default until larger validation clears the tradeoff.

Batched matmul upload staging now detects already-contiguous split views and uploads the backing span directly instead of copying those views into scratch first. On the same acquired-text 512 train / 128 eval profile, trainer elapsed moved from `7.577s` to `6.396s`, `runtime.memmove` moved from `1.07s` to `0.58s`, `runtime.memclrNoHeapPointers` moved from `0.78s` to `0.60s`, and `flattenFixedFloat32MatricesScratch` was down to `0.34s` cumulative. Matmul counters again stayed unchanged at `2788` runs, `5838.20 MB` uploaded, and `2990.88 MB` downloaded; the win is less host staging around the same CUDA work.

Trainer-owned scratch buffers now reuse transient float32 flattening inputs for batched matmul dispatches. On the same direct batch-2048 CPU profile, `flattenFixedFloat32Matrices` moved from roughly `0.72s` cumulative to `0.09s` cumulative. The remaining host-side allocation/copy profile is mostly broader activation/state materialization and unavoidable host-device transfer until full device residency lands.

## Batch Sweep

Batch size is now the largest exposed training knob. On the same 4096-example mini smoke:

| Batch | Run steps | Train examples/s | Train pairs/s | Matmul runs | Max RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `512` | `8` | `558.19` | `285791.12` | `28848` | `1.02 GB` |
| `1024` | `4` | `845.15` | `865437.87` | `13644` | `1.52 GB` |
| `2048` | `2` | `827.20` | `1694107.56` | `9408` | `2.50 GB` |
| `4096` | `1` | `741.49` | `3037152.08` | `5616` | `4.47 GB` |

Batch 2048 is a useful ceiling or large-corpus setting, but it halves optimizer steps on this mini smoke. Batch 4096 is one update and regresses example throughput despite very high pair throughput. Batch 1024 is the benchmark default because it keeps multiple updates in the smoke and captures most of the real throughput win.

## How Much Faster Can It Get?

The current profile still shows the next bottleneck is backend transfer/orchestration, not raw math:

```text
MatMulRuns ~= 13644 per 4096-example mini smoke at batch 1024
RunUploadedBytes ~= 3.73 GiB
RunDownloadedBytes ~= 2.00 GiB
```

Reasonable next targets:

- `850-1000 train_examples/s`: reduce host materialization and launch overhead inside same-length groups.
- `1000+ train_examples/s`: keep forward/backward intermediates device-resident across full layer groups and move optimizer/activation state through the same device-resident path.
- `1200+ train_examples/s`: persistent device-resident training steps with fused attention/FFN backward and fewer per-layer dispatches.

The practical ceiling for this tiny model is dominated by orchestration and host-device transfer, not raw GEMM throughput. Larger models will shift more time into actual math, but the same device-residency work is still required to get high GPU utilization.

## Relevant Flags

```bash
MANTA_TRAIN_DISABLE_BATCHED_BACKWARD=1
```

Disables the promoted grouped batched-backward path and returns to per-sequence backward.

```bash
MANTA_TRAIN_DISABLE_BATCHED_FORWARD=1
```

Disables the promoted batched forward path and returns to per-sequence forward encoding. Batched forward is enabled by default because the larger default-model run underfeeds the GPU unless forward work is coalesced aggressively.

```bash
MANTA_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1
```

Disables exact-length batched forward chunks for pairwise `train-embed --eval-only` runs. This is useful for A/B checks against the scalar pair encoder.

```bash
MANTA_TRAIN_PAIR_EVAL_BATCH_SIZE=256
```

Controls how many pair examples each pairwise eval chunk may contain before grouping by exact token length. The default is `256`; larger chunks reduce dispatch count further but can increase materialization pressure and wall time.

```bash
MANTA_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS=1
```

Re-enables per-sequence matmul bindings. These are disabled by default because batch-1024 grouped training spends more time binding and unbinding short-lived sequence tensors than it saves in uploads. Keep this for small-batch experiments and regression checks.

```bash
MANTA_TRAIN_DISABLE_QKV_MULTI_BOUND=1
```

Disables Q/K/V forward projection coalescing. By default CUDA uses one uploaded left-hand activation with three resident right-hand Q/K/V matrices for same-length forward groups. This cuts transfer bytes while preserving each right-hand weight's own quantization state.

```bash
MANTA_TRAIN_DISABLE_SHARED_LEFT_MATMUL=1
```

Disables shared-left matmul coalescing. By default Manta first tries the concatenated shared-left gradient path for attention backward Q/K/V weight-gradient matmuls, then falls back to the backend shared-left interface when concatenation is disabled or unavailable.

```bash
MANTA_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL=1
```

Disables only the concatenated shared-left path. This keeps the older backend shared-left fallback enabled for A/B testing.

```bash
MANTA_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD=1
```

Disables only the combined V/K attention backward gradient path. This returns to separate strided-batched GEMMs for `scores^T*dMixed` and `dPreSoftmax^T*Q`.

```bash
MANTA_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD=1
```

Disables only the accumulated Q/K/V attention input-gradient path. This returns to three resident right-hand matmuls for `dQ*Wq^T`, `dK*Wk^T`, and `dV*Wv^T`, followed by host accumulation.

```bash
MANTA_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1
```

Disables CUDA single-sync accumulation inside the backend accumulated resident-right matmul primitive. This keeps the same logical training path but synchronizes after each accumulated cuBLAS term, which is useful for A/B testing backend launch overhead.

```bash
MANTA_TRAIN_ENABLE_FAST_GELU=1
```

Uses a bounded rational tanh approximation for host GELU forward/backward. This is opt-in because it changes training math. It can materially improve CPU-bound trainer throughput while the model still uses host activation math, but production candidates should compare validation and hard-holdout metrics against exact GELU before promotion.

```bash
MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL=1
```

Enables CUDA/Metal activation backward acceleration. Activation backward batches GELU, softmax, and layernorm across same-length groups before dispatching. It is still opt-in because standalone activation kernels are upload/download-bound without broader activation residency. Large unbound groups fall back to host math by default through `MANTA_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS`.

```bash
MANTA_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS=1048576
```

Caps unbound activation accelerator calls by `rows * cols`. The default is `1048576`. Set it lower to force more host fallback for memory-transfer-heavy profiles; set it to `0` to remove the shape limit when profiling a fully unbounded activation path.

```bash
MANTA_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Enables only the attention softmax backward activation path while keeping GELU and layernorm backward on the host. This is a profiling seam, not a promoted default.

```bash
MANTA_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Disables the softmax backward activation path even when the broader activation accelerator is enabled. Use it to isolate GELU/layernorm activation experiments from attention softmax experiments.
