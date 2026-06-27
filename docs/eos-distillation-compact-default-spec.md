# Eos Distillation And Compact Default Spec

This spec defines the next credible default-embedder push for Eos. The path is:

1. Lift dense base quality through external-teacher distillation on larger retrieval data.
2. Add quantization-aware embedding objectives only after dense quality moves.
3. Add native compact / Matryoshka heads as a differentiator only after dense preservation is proven.
4. Productize `eos-hybrid` now as an opt-in retrieval lane, not as dense model promotion.

No training or evaluation was run for this spec. Evidence is from the current docs and scratch reports listed in the source table.

## Source Evidence

| Path | Use |
| --- | --- |
| `docs/default-corkscrew-embedder-plan.md` | current default, promotion gates, external vector-cache lane, TurboQuant gates, hybrid boundary |
| `docs/manta-embed-sota-avenues.md` | current SOTA avenues, rejected local lanes, compact-head status, teacher-quality blockers |
| `docs/turboquant-multivector-frontier.md` | packed-parent and compact child-vector product boundaries |
| `docs/production-embedding.md` | production candidate workflow, dataset acquisition, promotion checks |
| `.tiller/scratch/codex/eos-next-credible-version-packet-v1.md` | ordered next-version strategy and gate definitions |
| `.tiller/scratch/codex/eos-scale-rerun-v2-dense-gate-postmortem.md` | failed 100k scale rerun postmortem |
| `.tiller/scratch/codex/eos-s40-fiqa-clean-scale-pilot-rerun-v2-report.md` | rerun completion, internal metrics, dense no-go evidence |
| `.tiller/scratch/codex/eos-s40-balanced-boundary-macro-ndcg-ablation-v1-report.md` | local balance/boundary no-go evidence |
| `.tiller/scratch/codex/eos-current-default-hybrid-refresh-v1-report.md` | current default hybrid product-lane evidence |
| `.tiller/scratch/codex/eos-current-default-turboquant-frontier-v1-report.md` | q2/q3/q4/q8 frontier and compact-default status |

## Decision Summary

The local-row ceiling is real. The current default dense macro across SciFact, NFCorpus, and FiQA is nDCG@10 `0.2971816079946877` and recall@100 `0.46339624017565995`. Recent local data scaling did not move dense macro nDCG enough:

| Evidence | Dense result |
| --- | --- |
| 100k FiQA-heavy rerun | Harness succeeded, but dense no-go: macro nDCG delta `-0.000116608`; macro recall delta `+0.000085426`. Data was skewed to FiQA `76,617`, SciFact `12,191`, NFCorpus `11,192`. Internal pair metrics were healthy but not predictive. |
| Balanced/boundary ablation | Arm A macro nDCG delta `+0.000056835`; arm B `-0.000103905`; both no-go. Simple source balance plus small train-only boundary replay is exhausted. |

Base quality is binding. Internal pair AUC, margin, top1, and InfoNCE health are necessary plumbing signals, not promotion evidence. Promotion depends on retrieval scoreboards.

Compactness is a multiplier on base quality. Current q4/fp16/o200 remains the compact default row with macro nDCG@10 `0.297203978370`, macro recall@100 `0.464350585508`, and compression `1.590062111801x`. q3 is supported and measured, but not a better default. q2 is a pressure test. Quant-aware work should reduce low-bit retrieval loss after the dense row improves.

Hybrid ships as product work now. Current fixed hybrid lifts dense macro nDCG from `0.297181607995` to `0.416066185231`, delta `+0.118884577236`, and macro recall from `0.463396240176` to `0.574716036615`, delta `+0.111319796439`. This is a first-class opt-in `eos-hybrid` retrieval surface for CorkScrewDB. It is not dense model promotion and must not silently replace dense defaults.

## Claim Boundaries

- No Qwen3 parity claim exists for Eos.
- No dense promotion may be inferred from `eos-hybrid`.
- No compact default promotion may happen before dense promotion.
- No long-context quality claim may be made from current rows. Existing long-context rows remain diagnostic and often carry `quality_claim=false`.
- No native Matryoshka claim may be made from prefix-truncated bridge caches.
- No q2 or q3 default claim may be made from the current frontier.
- No hosted-provider release claim may depend on uncached live API calls.

## Target Product Shape

| Surface | Near-term status | Promotion boundary |
| --- | --- | --- |
| `eos-embed-v1` dense | Current default remains promoted | Future candidate must pass strict dense selected-vs-anchor gate |
| `eos-turboquant-rerank` q4/fp16/o200 | Current compact default row | Candidate compact row is evaluated only after dense pass |
| `eos-hybrid` | Productize as opt-in | Requires policy metadata, calibration evidence, and API parity; not dense promotion |
| Quant-aware direct q4/q2 | Research after dense pass | Must shrink low-bit rank loss without dense regression |
| Native compact / Matryoshka heads | Research after dense pass | Must preserve full-dim dense and prove child-vector quality |

## Dense Base Distillation Design

### Data Staging

Use MS MARCO first because it is the nearest large-scale retrieval training substrate. Do not train until there is an acquisition manifest that records:

- dataset name, upstream source, license, and commercial-use interpretation;
- corpus/query/qrel counts and split names;
- SHA256s for raw and processed artifacts;
- exact train/dev/test separation;
- `train_allowed` flags per generated row family;
- known exclusions, skipped rows, and text-availability gaps.

Expand only after MS MARCO acquisition and split safety are clean:

| Stage | Data | Entry condition |
| --- | --- | --- |
| 1 | MS MARCO passage/document train plus dev sanity | Manifest, license note, split guard, raw counts, processed counts |
| 2 | Natural Questions retrieval-style pairs | Only after NQ manifest and held-out policy are reviewed |
| 3 | HotpotQA / multi-hop train rows | Only after positive/negative provenance and qrels mapping are safe |
| 4 | Full BEIR train/dev-safe expansion | Only after per-dataset train/test boundaries are encoded in the builder |

Do not use benchmark test rows for training, selection, teacher filtering, or policy selection. Treat local SciFact/NFCorpus/FiQA short-set test scoreboards as promotion gates, not training sources.

### Teacher Strategy

Use provider-boundary caches and scoring artifacts. Training descriptors should consume local JSONL/vector-cache/reranker-score files, not call providers inline.

Preferred teacher order:

| Teacher | Role | Reason |
| --- | --- | --- |
| Qwen3 0.6B embedding cache | first embedding teacher / retrieval scorer | local/cacheable, already part of external-cache lane |
| mxbai-large embedding cache | second embedding teacher / agreement scorer | strong local short-set evidence relative to Qwen3 |
| BGE reranker or cross-encoder | optional rerank teacher | only after teacher-quality gates pass on the exact candidate rows |
| Hosted embedding or reranking APIs | optional comparison only | avoid release dependency; cache first; license and privacy review required |

The cleaned NFCorpus reranker-frontier evidence blocks naive reranker use. Jina, BGE-large, mxbai-base subset, and mxbai-large full-frontier rows were teacher-quality blocked or no-train for that local frontier. Future reranker use needs fresh data and gates, not reuse of failed frontier assumptions.

### Agreement And Leakage Filtering

Every train row must preserve enough provenance to audit:

- query ID, text, source dataset, split, and generated/synthetic status;
- positive document ID/text/source;
- negative document IDs/text/source;
- qrels-positive negative leak checks by ID and exact query-positive text pair;
- teacher model ID, cache path, score version, score transform, and timestamp;
- whether the row was generated from train, dev, or test-visible material.

Filtering should use several gates together:

| Gate | Purpose |
| --- | --- |
| qrels-positive negative leak check | Remove negatives that are judged positive for the query or equivalent judged text pair |
| teacher margin | Keep examples where teacher score separates positive from negatives by a useful margin |
| teacher agreement | Prefer rows where Qwen3 and mxbai-large agree on positive-over-negative ordering |
| relabel-teacher-negatives | Convert likely false negatives instead of forcing them as negatives |
| false-negative control | Downweight or drop ambiguous negatives near the positive score or retrieved by multiple teachers |
| source balance | Avoid another FiQA-heavy macro failure by maintaining dataset/source accounting |

Rows that fail provenance or leak checks are not trainable. Rows that have weak teacher agreement are audit artifacts unless a descriptor explicitly chooses them for a bounded negative-control run.

### Training Objective

Use hard-negative InfoNCE as the base objective, but do not rely on pair separation alone. The 100k rerun had healthy internal metrics and still failed macro nDCG. Dense training should add a ranking-shaped teacher objective:

- hard-negative InfoNCE over query, positive, and mined negatives;
- soft teacher distribution distillation over candidate lists, with temperature and per-source normalization recorded;
- or margin distillation that preserves teacher positive-vs-hard-negative gaps;
- optional listwise loss over top-k candidates when the candidate set is trustworthy.

The target is top-10 and recall@100 retrieval movement, not just vector-pair AUC. Training reports must include internal metrics, but promotion decisions must cite retrieval scoreboards and per-dataset deltas.

### Staged Pilots

| Stage | Scope | Exit condition |
| --- | --- | --- |
| Acquisition audit | No training | manifests parse; licenses and split safety reviewed; row counts and SHA256s recorded |
| Teacher-cache audit | No training | Qwen3 0.6B and mxbai-large caches/scored rows exist; agreement and leak reports pass |
| 100k pilot | bounded training | dense macro nDCG moves upward; no dataset floor break; internal metrics reported only as diagnostics |
| 250k pilot | bounded training | only if 100k pilot gives credible dense movement |
| larger run | expensive training | only if dense gate trend is positive and data/teacher error analysis is clean |

### Dense Gates

Exploration gates:

- macro nDCG@10 delta versus current default `>= +0.0010`;
- macro recall@100 delta `>= -0.0010`;
- no dataset nDCG@10 delta below `-0.0020`;
- no dataset recall@100 delta below `-0.0030`;
- MS MARCO dev retrieval sanity is non-regressing versus the pilot baseline;
- no promotion or compact eval if dense exploration fails.

Promotion gates:

- strict selected-vs-current-default dense gate across SciFact, NFCorpus, and FiQA;
- metrics `ndcg_at_10` and `recall_at_100`;
- zero tolerance unless a rounding tolerance is explicitly accepted;
- package/sealed verification and eval-only optimizer-update checks still required;
- compact/TurboQuant cannot rescue a dense miss.

## Quant-Aware Embedding Objective Design

Quant-aware work starts after a dense candidate passes exploration and preferably after dense promotion. It is not a rescue lane for a dense no-go.

### Objective

Train retrieval-score survival under TurboQuant, not vector MSE alone:

- preserve dense top-k score ordering after q4 and q2 quantization;
- preserve positive-vs-hard-negative rank margins after direct TurboQuant IP scoring;
- optionally use a straight-through estimator for quantized document vectors if supported by the training path;
- otherwise use a score-preserving approximation that samples quantization noise or prepared TurboQuant scores;
- keep the full dense embedding loss active so dense quality does not drift.

MSE between dense and quantized vectors is insufficient because promotion is based on ranked retrieval. The q2/q3/q4 direct frontier shows quality loss is rank-shaped: q4 direct loses macro nDCG `-0.009688487`, q3 direct loses `-0.020635359`, and q2 direct loses `-0.041788508` versus dense on current caches.

### Gates

| Gate | Requirement |
| --- | --- |
| Dense preservation | Full dense short-set strict pass versus the starting dense candidate |
| q4 direct survival | q4 direct nDCG and recall deltas shrink meaningfully versus the same candidate without quant-aware training |
| q2 pressure test | q2 direct loss shrinks without using q2 as default unless quality is acceptable |
| q4/fp16/o200 | promoted compact profile still passes strict compact gate |
| Diagnostics | report bit width, quantizer seed, vector bytes, sidecar bytes, compression, nDCG, recall, latency summaries |

## Native Matryoshka Compact-Head Design

Native compact heads should train child vectors as first-class outputs. Prefix truncation plus L2 renormalization is a bridge measurement, not the target.

### Sequence

Run this with or after dense-pass and quant-aware work:

1. Start from a dense-accepted candidate.
2. Add native heads or explicit Matryoshka losses for `64d`, `128d`, and full dimension.
3. Run movement diagnostics before expensive scoreboards.
4. Preserve full-dim dense quality as a hard gate.
5. Score child vectors under direct q4/q8 and packed-parent CorkScrewDB layouts only after the above pass.

### Profile

| Head | Intended use | Gate |
| --- | --- | --- |
| `64d` | pressure test / extreme compact | diagnostic unless quality is unexpectedly strong |
| `128d` | primary compact child-vector target | q4 child quality plus packed-parent storage/API evidence |
| full `256d` | dense compatibility | strict dense preservation |

Do not implement this as only a prefix bridge. Native child vectors should have their own objective and movement checks. Existing no-clear Matryoshka and teacher-guided q4 child-rank recipes moved but failed quality; do not repeat those exact recipes.

### Gates

- movement-positive at `64d`, `128d`, and full dimension;
- full-dim dense strict pass versus starting dense anchor;
- `128d` q4 child-vector row beats the bridge target selected for the run;
- short-set `128d` q4/q8 rows include quality and storage accounting;
- packed-parent API evidence records layout, metadata mode, child-ID mode, DB bytes, vector payload multiple, p95, and parent-search mode;
- no promotion from repo-docs alone.

## Hybrid Product Lane

Productize `eos-hybrid` as a first-class opt-in retrieval mode for CorkScrewDB:

- dense artifact remains the current default embedder;
- hybrid policy must carry method, alpha, sparse method, top-k, dense-protect setting if any, calibration split, and score-normalization metadata;
- API responses and manifests should expose dense/sparse policy identity;
- dense-only, sparse-only, and fused diagnostics should be available for evaluation;
- FiQA caution remains: BM25-only nDCG@10 exceeded fixed alpha-0.5 hybrid in the refresh, even though hybrid beat dense.

The current refresh supports the lane:

| Row | macro nDCG@10 | macro recall@100 |
| --- | ---: | ---: |
| dense | 0.297181607995 | 0.463396240176 |
| BM25 | 0.399497648487 | 0.538580299918 |
| hybrid minmax alpha 0.5 | 0.416066185231 | 0.574716036615 |

Do not promote the dense model from hybrid evidence. Do not silently switch default retrieval to hybrid without a product/API decision.

## Descriptor-Backed Task List

### eos-msmarco-data-acquisition-v1

| Field | Value |
| --- | --- |
| id/title | `eos-msmarco-data-acquisition-v1` |
| role/profile | `tiller-worker` for data acquisition manifest and split-safety audit |
| objective | Acquire or stage MS MARCO retrieval data for Eos training, with license, split, qrels, and SHA256 manifests. Do not train. |
| context paths | `docs/production-embedding.md`; `docs/default-corkscrew-embedder-plan.md`; this spec |
| constraints | No test leakage; no model training; no promotion; record license/commercial-use caveat explicitly; keep raw and processed paths separate. |
| expected outputs | Acquisition manifest; processed row-count summary; train/dev/test split map; SHA256s; `train_allowed` policy; rejected/skipped-row report. |
| verification target | Manifests parse; counts match source files; qrels and corpus IDs resolve; no test rows appear in train manifests. |
| budget tier/model ceiling | medium, `gpt-5.5 medium` ceiling |
| sandbox/permission needs | filesystem and network if data is not already local; no VCS commit |
| dependencies/blockers | dataset license review; enough disk for raw and processed corpora |
| checkpoint criteria | yes for manifests and reusable acquisition docs/scripts only |
| report contract | Outcome; distillation; files changed/inspected; verification commands/results; caveats/residual risk; checkpoint candidate yes/no; Arbiter next action |

Maintained script: `scripts/acquire_msmarco_passage.py`. It defaults to a bounded query/qrels acquisition manifest and only audits passage text resolvability when `--download-corpus` or `--corpus-path` is explicitly supplied.

### eos-teacher-cache-and-score-v1

| Field | Value |
| --- | --- |
| id/title | `eos-teacher-cache-and-score-v1` |
| role/profile | `tiller-worker` or `tiller-debugger` for cache/scoring pipeline |
| objective | Build provider-boundary Qwen3 0.6B and mxbai-large embedding caches and teacher score files for MS MARCO pilot rows; optionally stage reranker scoring only after cache gates. |
| context paths | `docs/default-corkscrew-embedder-plan.md`; `docs/manta-embed-sota-avenues.md`; this spec |
| constraints | Prefer local/cacheable teachers; no inline provider dependency in training; no use of teacher rows without provenance; no reuse of known teacher-quality-blocked NFCorpus frontier as train data. |
| expected outputs | `doc-vectors.jsonl`, `query-vectors.jsonl`, score files, teacher-cache manifests, agreement/margin/leak audit, skipped-row report. |
| verification target | JSONL parses; vector dimensions consistent; ID coverage matches manifest; qrels-positive negative leaks are zero or explicitly removed; agreement summary generated. |
| budget tier/model ceiling | medium/high depending on cache size, `gpt-5.5 medium` ceiling |
| sandbox/permission needs | filesystem, GPU optional, network only for initial model/data fetch if absent |
| dependencies/blockers | `eos-msmarco-data-acquisition-v1`; local model availability; disk budget |
| checkpoint criteria | yes for manifests, cache metadata, and reusable scoring artifacts |
| report contract | Outcome; distillation; files changed/inspected; verification commands/results; caveats/residual risk; checkpoint candidate yes/no; Arbiter next action |

### eos-distilled-base-pilot-v1

| Field | Value |
| --- | --- |
| id/title | `eos-distilled-base-pilot-v1` |
| role/profile | `tiller-worker` for bounded training and dense gate evaluation |
| objective | Train a 100k distilled dense pilot using MS MARCO-stage train-safe rows, hard-negative InfoNCE, and listwise/soft or margin teacher distillation; evaluate dense retrieval gates only. |
| context paths | `docs/production-embedding.md`; `docs/default-corkscrew-embedder-plan.md`; `.tiller/scratch/codex/eos-scale-rerun-v2-dense-gate-postmortem.md`; this spec |
| constraints | No compact eval unless dense exploration passes; no promotion unless strict dense gate passes; preserve source balance; cite retrieval gates over internal AUC. |
| expected outputs | Run root; train/eval metrics; dense scoreboards; per-dataset deltas; MS MARCO dev sanity row; dense gate report; no-go/pass recommendation. |
| verification target | Wrapper completes; eval-only optimizer updates `0`; scoreboards parse; dense macro nDCG delta `>= +0.0010`; floors pass. |
| budget tier/model ceiling | medium/high training, `gpt-5.5 medium` ceiling; escalate only for root-cause debugging |
| sandbox/permission needs | filesystem, GPU, enough disk; no VCS commit |
| dependencies/blockers | `eos-msmarco-data-acquisition-v1`; `eos-teacher-cache-and-score-v1` |
| checkpoint criteria | yes for report/evidence; source checkpoint only for reusable harness fixes |
| report contract | Outcome; distillation; files changed/inspected; verification commands/results; caveats/residual risk; checkpoint candidate yes/no; Arbiter next action |

### eos-quant-aware-embedding-objective-v1

| Field | Value |
| --- | --- |
| id/title | `eos-quant-aware-embedding-objective-v1` |
| role/profile | `tiller-worker` for objective implementation/prototype or `tiller-debugger` if trainer changes are needed |
| objective | Add or run a quant-aware retrieval objective that preserves rank scores/margins after TurboQuant q4/q2, sequenced after a dense-pass candidate. |
| context paths | `docs/default-corkscrew-embedder-plan.md`; `docs/turboquant-multivector-frontier.md`; `.tiller/scratch/codex/eos-current-default-turboquant-frontier-v1-report.md`; this spec |
| constraints | Do not run before dense pass; optimize retrieval-score survival, not vector MSE alone; full dense preservation is a hard gate; q2 remains diagnostic. |
| expected outputs | Objective config; training/eval run if implementation exists; q2/q4/q8 dense and direct rows; q4/fp16/o200 compact rows; delta-vs-non-quant-aware comparison. |
| verification target | Dense strict pass; q4 direct loss shrinks; q2 loss shrinks as pressure test; compact q4/fp16/o200 gate passes. |
| budget tier/model ceiling | medium/high, `gpt-5.5 medium` ceiling |
| sandbox/permission needs | source edits only if objective plumbing is missing; GPU for training; no commit unless requested |
| dependencies/blockers | Dense pass from `eos-distilled-base-pilot-v1` or later accepted candidate |
| checkpoint criteria | yes if source objective/harness is added and tests pass; evidence-only checkpoint if no source changes |
| report contract | Outcome; distillation; files changed/inspected; verification commands/results; caveats/residual risk; checkpoint candidate yes/no; Arbiter next action |

### eos-native-matryoshka-compact-head-v1

| Field | Value |
| --- | --- |
| id/title | `eos-native-matryoshka-compact-head-v1` |
| role/profile | `tiller-worker` for compact-head experiment; `tiller-debugger` if movement or trainer behavior is suspect |
| objective | Train native `64d`/`128d`/full compact heads or Matryoshka objectives from a dense-pass candidate, then prove movement, dense preservation, and child-vector quality. |
| context paths | `docs/manta-embed-sota-avenues.md`; `docs/turboquant-multivector-frontier.md`; this spec |
| constraints | Do not repeat failed no-clear Matryoshka or teacher-guided q4 child-rank recipes; no prefix-bridge promotion; no repo-docs-only promotion. |
| expected outputs | Movement diagnostics; full-dim dense scoreboard; 64d/128d child-vector q4/q8 scoreboards; packed-parent storage/API report if quality moves. |
| verification target | Movement-positive at all target dims; full-dim strict dense pass; 128d q4 child quality beats selected bridge target; storage/API metadata complete. |
| budget tier/model ceiling | medium/high, `gpt-5.5 medium` ceiling |
| sandbox/permission needs | GPU for training; CorkScrewDB checkout optional for API smoke; no commit unless requested |
| dependencies/blockers | Dense pass; preferably quant-aware objective evidence |
| checkpoint criteria | yes for verified objective/harness/source changes or strong evidence report |
| report contract | Outcome; distillation; files changed/inspected; verification commands/results; caveats/residual risk; checkpoint candidate yes/no; Arbiter next action |

### corkscrewdb-eos-hybrid-productization-v1

| Field | Value |
| --- | --- |
| id/title | `corkscrewdb-eos-hybrid-productization-v1` |
| role/profile | `tiller-worker` in CorkScrewDB/Eos integration lane |
| objective | Productize opt-in `eos-hybrid` retrieval with dense+sparse fusion policy metadata, diagnostics, and CorkScrewDB API parity. |
| context paths | `docs/default-corkscrew-embedder-plan.md`; `.tiller/scratch/codex/eos-current-default-hybrid-refresh-v1-report.md`; this spec; CorkScrewDB integration docs if available |
| constraints | Do not promote dense model; do not silently replace dense default; preserve sparse/dense policy identity and calibration metadata. |
| expected outputs | Product/API design or implementation; policy manifest fields; calibration/gate report; local CorkScrewDB smoke evidence. |
| verification target | Dense-only, sparse-only, and hybrid rows parse; policy metadata is persisted; CorkScrewDB local API smoke passes for selected datasets. |
| budget tier/model ceiling | medium, `gpt-5.5 medium` ceiling |
| sandbox/permission needs | Eos and CorkScrewDB worktrees; build/test permission; no VCS commit unless requested |
| dependencies/blockers | Product decision on fixed vs calibrated policy; CorkScrewDB checkout state |
| checkpoint criteria | yes for verified product/API changes or docs |
| report contract | Outcome; distillation; files changed/inspected; verification commands/results; caveats/residual risk; checkpoint candidate yes/no; Arbiter next action |

## Arbiter Next Action

Route next execution to `eos-msmarco-data-acquisition-v1`, then `eos-teacher-cache-and-score-v1`. Do not run more local-row balance, boundary remix, compact, or Matryoshka training before the external data/teacher substrate exists. In parallel, route product work to `corkscrewdb-eos-hybrid-productization-v1` if CorkScrewDB wants the current opt-in hybrid lane.
