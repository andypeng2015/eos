# Consumer GPU Subquadratic Training And Inference Spec

This document defines what success means for Eos's long-context sparse attention direction: a single consumer-grade GPU should be able to train and serve models whose attention path scales below quadratic cost with context length.

The goal is not to claim frontier-model pretraining on a desktop card. The goal is to prove a direct, reproducible path where Eos-native kernels, training loops, and `.mll` artifacts can train the sparse policy and model components that make long-context inference practical on one local GPU.

The primary product wedge for this work is the local long-context embedder described in [local-long-context-embedder-wedge.md](local-long-context-embedder-wedge.md). This spec defines the lower-level GPU and attention gates needed to make that embedder credible.

## Success Statement

Eos succeeds when a user with one consumer GPU can:

- train or fine-tune a Eos-authored long-context model without dense `N x N` attention materialization
- run inference with TurboQuant-compressed K/V cache and routed sparse attention
- show measured attention scaling below quadratic as context length grows
- preserve task quality within explicit gates against dense or higher-budget teacher baselines
- export a sealed `.mll` artifact whose runtime metadata proves the sparse/compressed path was used

The claim must be supported by repeatable commands, recorded hardware metadata, GPU memory telemetry, quality metrics, and log-log scaling fits.

## Hardware Contract

Primary target: `C24`

- one consumer GPU
- `24 GB` VRAM target class
- CUDA first, Metal/WebGPU later
- no multi-GPU, tensor parallelism, NVLink, remote accelerator, or data-center card
- no CPU K/V offload in the measured inference path
- CPU RAM may stage datasets and artifacts, but not carry attention state for the measured run

Guardrail target: `C16`

- same constraints, but `16 GB` VRAM
- lower sequence/model targets are acceptable
- this tier prevents the design from becoming quietly workstation-only

Stretch target: `C12`

- used for smoke tests and small-model demos
- not sufficient for the main success claim

## Model And Workload Targets

The proof should use three workloads so kernel wins do not hide model failures.

| Workload | Purpose | Minimum pass |
| --- | --- | --- |
| Attention microbench | isolates attention scaling | exact and routed CUDA kernels match references at small `N`; routed path shows subquadratic slope at large `N` |
| Long-context training smoke | proves trainability | trains at least `100` optimizer steps on one GPU with sparse attention enabled and no dense attention fallback |
| Long-context inference demo | proves serving path | loads a sealed `.mll`, appends compressed K/V, and decodes with routed sparse attention |

Reference model sizes:

| Name | Use | Suggested shape |
| --- | --- | --- |
| `long-micro` | correctness and training bring-up | 4-8 layers, 256-512 width, 4-8 heads |
| `long-small` | primary consumer GPU proof | 8-12 layers, 512-768 width, 8-12 heads |
| `long-adapter` | stretch fine-tune path | frozen quantized base plus trained sparse router/adapters |

Full scratch training of a large foundation model is a non-goal. Training the sparse attention policy, router, summaries, adapters, and a small/medium reference model is in scope.

## Definition Of Subquadratic

Dense full-sequence attention for training/prefill costs:

```text
O(N^2 * D)
```

Dense autoregressive decode attention costs per generated token:

```text
O(N * D)
```

For routed block sparse attention:

```text
B = route block size
R = selected route blocks per query
candidate keys per query ~= R * B
route scores per query ~= N / B
per-query attention score cost ~= O((N / B + R * B) * D)
```

If `B = ceil(sqrt(N))` and `R` is bounded:

```text
per-query decode cost ~= O(sqrt(N) * D)
full prefill/training attention cost ~= O(N^1.5 * D)
```

That is the first subquadratic target. A later hierarchical router should move toward:

```text
per-query decode cost ~= O((log N + K) * D)
full prefill/training attention cost ~= O(N * (log N + K) * D)
```

## Measured Scaling Gates

Measure over at least four sequence lengths, preferably:

```text
4096, 8192, 16384, 32768, 65536
```

Fit `time = c * N^alpha` using log-log linear regression.

| Path | Required alpha | Stretch alpha |
| --- | ---: | ---: |
| full prefill/training attention | `<= 1.60` | `<= 1.25` |
| per-token decode attention | `<= 0.65` | `<= 0.35` |
| K/V memory growth | `<= 1.10` | `<= 1.00` |

If dense attention OOMs before the largest lengths, use the largest dense-comparable range for quality and speedup, then use sparse-only lengths for slope.

## Speed Gates

Speed gates are relative to the best dense or exact high-budget baseline on the same machine.

| Context | Minimum routed speedup | Stretch speedup |
| ---: | ---: | ---: |
| `16k` | `2x` | `4x` |
| `32k` | `4x` | `8x` |
| `64k` | `8x` | `16x` |

For contexts where dense OOMs, the pass condition becomes:

- routed path completes without OOM
- measured slope remains below the gate
- quality remains inside the quality gate
- metadata confirms dense K/V was not materialized

## Memory Gates

Inference on `C24`:

- `128k` context must fit with compressed K/V and routed sparse attention
- peak GPU memory must be reported
- dense K/V materialization must be `false`
- CPU K/V offload must be `false`

Training on `C24`:

- `32k` context training smoke must run at least `100` optimizer steps
- peak GPU memory must stay under `24 GB`
- dense attention matrix materialization must be absent
- gradient accumulation and activation checkpointing are allowed

Guardrail on `C16`:

- `64k` inference target
- `16k` training smoke target

## Quality Gates

Sparse attention can be approximate, but approximation must be explicit and bounded.

Kernel-level gates:

- exact fused TurboQuant sparse attention matches host reference within existing tensor tolerances
- routed sparse attention matches the routed host reference exactly within tolerance
- selected key count, routed block count, and candidate budget are logged

Teacher-comparison gates:

- average top-block/top-token recall against dense/high-budget teacher: `>= 0.95` for synthetic calibration cases
- per-query minimum top-block/top-token recall: explicit gate, target `>= 0.95` for robust sparse-routing promotion; record failures separately from average recall
- attention output cosine similarity to teacher: `>= 0.98` on calibration batches

Task-level gates:

- language-model validation loss no worse than `5%` versus dense/high-budget teacher on short contexts where dense is possible
- long-context retrieval/passkey accuracy no worse than `2 percentage points` versus high-budget teacher
- no regression on short-context smoke tasks when routing is disabled

The first production-quality claim requires all three gate layers. Microbench speed alone is not enough.

## Current Baseline

Already present in the repo:

- `sparse_attention(q, k, v, top_k)` for dense Q/K/V top-k attention
- `turbo_sparse_attention(q, kc, kn, vc, vn, top_k)` for TurboQuant-compressed K/V
- fused CUDA TurboQuant sparse attention with inline K decode while scoring and inline V decode for selected keys
- optional routed form:

```eos
turbo_sparse_attention(q, kc, kn, vc, vn, top_k, route_block_size, route_top_blocks)
```

Sparse embedding smoke now has CUDA backend evidence and a strict backend-vs-host TurboQuant parity gate. Routing calibration is separate: `eos calibrate-sparse-routing` sweeps `anchor`, `multiprobe`, `summary_mean`, `summary_mean_radius`, `summary_maxnorm`, `summary_blend_radius`, calibration-only `summary_multirep`, calibration-only `summary_hier_multirep`, calibration-only `learned_block_linear`, and teacher-only `oracle_block_max` policies, writes `calibration.json` and `calibration.tsv`, and can gate both average and per-query minimum recall. The deployable summary policies use one block-summary dot per block: `summary_mean_radius` adds a scalar radius bound, `summary_maxnorm` uses the largest-L2-norm key representative, and `summary_blend_radius` sweeps mean-to-maxnorm alpha plus scalar radius beta. `summary_multirep` precomputes deterministic farthest-point representatives per block and routes by the max query dot across them, but charges `block_count * route_summary_count` route scores plus candidate-key work, so it is a budget/geometry probe rather than a default. `summary_hier_multirep` tests whether coarse grouping can avoid paying K representatives for every fine block: it scores one coarse max-norm representative per group, then scores K fine representatives only inside selected groups before choosing final fine blocks; row output records group size, top groups, coarse group count, and fine summary-score accounting. `learned_block_linear` trains a tiny deterministic logistic linear scorer from `oracle_block_max` block labels on synthetic calibration seeds, then evaluates routed sparse attention from learned weights and cheap summary/radius/length features only; teacher score count is reported separately and is not included in evaluation route work. The current decision baseline is mixed: default-scale anchor and simple config/block-size sweeps do not pass under `score_count_fraction <= 0.2`, `recall_avg >= 0.95`, and `cosine >= 0.98`; best under-budget multiprobe at `4096x8x64`, block `64`, top blocks `8`, probes `4` reached recall avg `0.486328125`, cosine `0.98335072496811`, and score fraction `0.1875`. A learned-router same/held-out seed probe at `4096x8x64`, block `10`, top blocks `35..40` also failed strict gates; its best learned row reached recall avg/min `0.609375 / 0.46875`, cosine `0.9833500752094315`, and score fraction `0.19775390625`. Coarse oracle block-size sweeping with min recall had no robust pass; its best under-budget row, block `16` / top blocks `32`, reached recall avg/min `0.9921875 / 0.9375`, cosine `0.999981385545`, and score fraction `0.1875`, failing only the min recall `0.95` gate. A fine-grained teacher-only oracle boundary sweep found a narrow robust upper-bound region: block `10` / top blocks `38` reached recall avg/min `1 / 1`, cosine `1`, MSE `0`, and score fraction `0.19287109375`. Passing ranges were block `8` top blocks `36,37,38`, block `10` `35..40`, block `12` `35..39`, block `14` `34..37`, block `16` `33..35`, and block `24` none. This is calibration upper-bound evidence, not retrieval quality proof or a deployable selector.

Current routed implementation:

- scores one anchor key per block
- selects top route blocks
- scores only keys in selected blocks
- keeps exact top-k semantics inside the selected candidate set
- shares one attention-plan metadata path across host reference, CUDA sparse attention, and fused CUDA TurboQuant sparse attention, including selected keys, route blocks, candidate key budget, estimated per-query score work, and score-work fraction versus dense scoring

This proves the kernel path but not yet the full training/inference success claim.

## Experiment Layers

### Layer 0: Dense And Exact Baselines

Purpose: establish correctness and measurement references.

Implementation:

- dense attention reference for small `N`
- exact sparse top-k reference
- exact fused TurboQuant sparse CUDA path
- benchmark harness that records time, memory, selected keys, and metadata
- `eos plan-sparse-attention` preflight that sweeps context lengths and reports routed score-work fraction, estimated score-work alpha, and logical TurboQuant K/V memory before a GPU run
- `scripts/bench_sparse_attention.fw` harness that archives preflight TSV/JSON plus CUDA sparse-attention benchmark JSONL/text/summary TSV and measured scaling alpha TSV for exact f16 and routed TurboQuant paths
- configurable CUDA benchmark matrices that keep exact f16 key lengths small while sweeping deeper routed TurboQuant key lengths for measured alpha

Pass:

- exact CUDA output matches host reference
- no hidden dense K/V materialization in the TurboQuant fused path
- benchmarks produce machine-readable JSON/TSV
- preflight plans fail when routed score work is not actually subquadratic or when the TurboQuant K/V budget exceeds the target device envelope
- benchmark harness fails when routed TurboQuant measured time alpha exceeds the configured scaling gate

### Layer 1: Routed Block Sparse Attention

Purpose: get the first compute-skipping path.

Implementation:

- routed block attributes: `route_block_size`, `route_top_blocks`
- `B = ceil(sqrt(N))` auto mode, or an experiment harness that sweeps equivalent manual values
- CUDA kernel scores block anchors, then selected block members
- host reference mirrors the same approximation

Pass:

- routed CUDA output matches routed host reference
- scaling alpha clears the subquadratic gate when `B ~= sqrt(N)`
- metadata logs route config and candidate budget

### Layer 2: Cached Block Summaries

Purpose: replace raw anchor keys with better and cheaper routing signals.

Implementation:

- per-block summary vectors for K cache
- summaries updated on `kv_write`
- summary dtype can be `f16`, `q8`, or TurboQuant
- router scores summaries instead of decoding arbitrary anchor keys
- summaries support local, global, and learned block representatives

Pass:

- block recall beats anchor routing at the same candidate budget
- per-token decode cost improves or stays flat
- summary memory is less than `10%` of compressed K/V memory

### Layer 3: Learned Router

Purpose: move from heuristic routing to trained sparse policy.

Implementation:

- teacher labels from dense/high-budget attention on shorter contexts
- query-to-block router head
- losses:
  - top-block cross entropy
  - recall-weighted auxiliary loss
  - entropy/load-balance regularizer
  - optional distillation KL from dense attention over candidate blocks
- router budget schedule from high budget to target budget

Pass:

- average and per-query minimum top-block recall clear the selected calibration gates
- task loss clears the quality gate at target budget
- router remains stable when context length extrapolates beyond training length

### Layer 4: Sparse Training Backward

Purpose: make the sparse path trainable end to end.

Implementation:

- backward for selected sparse attention:
  - gradients for Q
  - gradients for selected K/V
  - gradients for router/summaries where differentiable
- straight-through or detached routing option for early experiments
- activation checkpointing over selected blocks
- no dense attention matrix allocation

Pass:

- gradient check on tiny tensors
- training smoke runs `100+` optimizer steps
- peak memory clears the C24/C16 gates
- loss decreases on synthetic and token LM tasks

### Layer 5: End-To-End Long-Context Training

Purpose: prove the model can learn with the sparse policy.

Implementation:

- train `long-micro` from scratch
- train or fine-tune `long-small`
- tasks:
  - copy/retrieval synthetic
  - passkey/needle
  - local language modeling
  - mixed short/long curriculum
- budget curriculum:
  - start dense/high-budget on short contexts
  - distill router
  - increase context
  - lower budget to target

Pass:

- no dense fallback in training logs
- validation loss within quality gate
- long-context task accuracy within gate
- training artifacts export as `.mll`

### Layer 6: Inference K/V Cache

Purpose: make the serving path real.

Implementation:

- compressed K/V cache stores TurboQuant coords and q-norms
- block summaries update incrementally
- decode uses routed TurboQuant sparse attention
- metadata proves:
  - `dense_kv_materialized=false`
  - `kv_decode=cuda_turboquant_inline`
  - `routing=block_summary` or stronger
  - CPU K/V offload disabled

Pass:

- `128k` C24 inference run completes
- per-token decode slope clears gate
- output quality clears long-context gates
- sealed `.mll` loads and runs through the normal runtime

### Layer 7: Hierarchical Routing

Purpose: push beyond `sqrt(N)` routing.

Implementation:

- multi-level block tree
- coarse-to-fine summaries
- fixed candidate budget independent of full context
- optional local window plus global memory tokens

Pass:

- decode alpha approaches `<= 0.35`
- prefill/training alpha approaches `<= 1.25`
- quality does not regress versus Layer 3/6 at equal memory

## Required Instrumentation

Every sparse attention run should be able to report:

```text
op
backend
device_name
device_vram_bytes
query_len
key_len
query_dim
value_dim
top_k
routing
route_block_size
route_top_blocks
candidate_key_budget
dense_kv_materialized
kv_decode
scratch_scope
gpu_peak_memory_bytes
kernel_time_ns
end_to_end_time_ns
```

Training runs should also report:

```text
optimizer_steps
tokens_per_second
examples_per_second
attention_alpha_fit
peak_gpu_memory_bytes
dense_attention_fallback_count
host_fallback_count
router_loss
teacher_kl
top_block_recall
```

## Benchmark Matrix

Minimum matrix for a success claim:

| Dimension | Values |
| --- | --- |
| Sequence length | `4k`, `8k`, `16k`, `32k`, `64k`; `128k` for inference |
| Head dim | `64`, `128` |
| Top-k | `16`, `32`, `64` |
| Routing | exact, block-anchor, multiprobe, block-summary, learned, oracle upper-bound |
| Route block size | fixed, `8..16` boundary candidates, `sqrt(N)`, learned/hierarchical |
| Route top blocks | powers of two for coarse scans plus boundary ranges near the score budget, including `33..40` where block sizes `8..16` currently pass teacher-only oracle gates |
| K/V dtype | dense f16, q8, q4 TurboQuant |
| Hardware tier | `C16`, `C24` |

## Go/No-Go Gates

Gate A: correctness

- exact fused path matches reference
- routed path matches routed reference
- no hidden dense K/V materialization

Gate B: scaling

- log-log alpha clears measured scaling gates
- speedup clears context gates or dense OOM path rules

Gate C: quality

- teacher recall average, teacher recall per-query minimum, and output similarity clear calibration gates
- task metrics clear long-context gates

Gate D: trainability

- sparse backward passes gradient checks
- `100+` optimizer steps on C24
- no dense fallback

Gate E: artifact

- training output exports as sealed `.mll`
- inference loads sealed artifact and reports sparse metadata

Gate F: consumer proof

- one C24 machine, no data-center accelerator
- commands and logs are reproducible from clean checkout
- full metrics bundle is archived

## Risk Register

Routing misses important keys.

- Mitigation: summary vectors, learned router, local window, global tokens, high-budget distillation.

Routing cost eats the savings.

- Mitigation: cached summaries, lower summary dimension, hierarchical routing, fused segmented kernels.

TurboQuant hurts attention quality.

- Mitigation: adaptive bits for summaries/K/V, q8 summaries with q4 values, per-layer budget tuning.

Training memory is dominated by activations, not attention.

- Mitigation: activation checkpointing, recompute selected blocks, adapter-first training, fused backward.

Subquadratic kernel wins do not translate to end-to-end wins.

- Mitigation: measure kernel and end-to-end separately, require both metadata and wall-clock gates.

Short-context quality regresses.

- Mitigation: exact mode default, route budget curriculum, disable routing below a context threshold.

## Direct Path

Milestone 1: measurement harness

- add sparse-attention benchmark command/script
- emit JSON/TSV metrics
- fit alpha automatically

Milestone 2: routed block sparse proof

- sweep `B ~= sqrt(N)` and `R`
- prove subquadratic slope on CUDA
- establish first quality/speed Pareto curve

Milestone 3: block summaries

- add summary cache representation
- update summaries on K/V append
- compare anchor, multiprobe, summary, and deployable approximations to the teacher-only oracle boundary region

Milestone 4: learned router

- build dense/high-budget teacher label generation
- train or approximate block labels around block sizes `8..16`, with block `10` / top blocks `38` as the first calibration target
- test extrapolation to longer contexts

Milestone 5: sparse backward

- implement backward for selected sparse attention
- add gradient checks
- run `long-micro` training smoke

Milestone 6: consumer training proof

- run `long-small` or adapter training on C24
- clear memory, quality, and no-fallback gates
- export sealed `.mll`

Milestone 7: consumer inference proof

- run `128k` C24 inference
- prove compressed K/V and routed sparse metadata
- clear decode scaling and task quality gates

Milestone 8: hierarchy and portability

- add hierarchical routing
- promote Metal/WebGPU reference paths
- tighten stretch alpha gates

## What We Can Claim At Each Stage

After Layer 1:

- "Eos has a fused TurboQuant sparse attention kernel and an experimental routed block sparse CUDA path."

After Layer 3:

- "Eos can learn a sparse routing policy that approximates dense attention on calibration workloads."

After Layer 5:

- "Eos can train a long-context sparse-attention model on one consumer GPU without dense attention materialization."

After Layer 6:

- "Eos can serve long-context inference on one consumer GPU with compressed K/V and subquadratic attention scaling."

After Layer 7:

- "Eos has a direct path toward near-linear long-context attention through hierarchical learned routing."
