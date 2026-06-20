Outcome:
- Implemented guarded LongEmbed frontier hard-negative/teacher-pair miner.
- Generated a protected candidate-data run:
  `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z`
- Attached agreement-filtered Qwen3/mxbai `teacher_scores` for QMSum and
  2WikiMQA using existing child-cache bridge tooling.
- Preserved `quality_claim=false` and explicit claim boundary in the run
  manifest and row metadata. No training was run.

Distillation:
- New script reads `candidate-examples.tsv`, joins query/doc text from
  `datasets/longembed-official/{qmsum,2wikimqa,narrativeqa}`, maps winner/loser
  categories to rescue families, and uses loser-profile per-query `top_k`
  rankings to select hard nonrelevant negatives.
- Rescue family mapping:
  - `worst_compact_loss_vs_direct` -> `direct_rescue`
  - `direct_wins_token_span_loses` -> `direct_rescue`
  - `token_span_wins_direct_loses` -> `token_span_rescue`
  - `sparse_parent_compact_loses_dense_rank` -> `sparse_compact_preservation`
  - `external_wins_when_eos_loses` -> `external_teacher_win`
- Teacher scoring was applied only to QMSum and 2WikiMQA because available
  cache roots contain those datasets but not NarrativeQA.

Files changed:
- `scripts/mine_longembed_frontier_teacher_pairs.py`
- `scripts/test_mine_longembed_frontier_teacher_pairs.py`
- `.tiller/scratch/codex/eos-longembed-frontier-teacher-pair-miner-v1-report.md`

Files inspected:
- `scripts/curate_longembed_frontier_hard_negatives.py`
- `scripts/score_longembed_child_cache_teacher_bridge.py`
- `scripts/test_curate_longembed_frontier_hard_negatives.py`
- `scripts/test_score_longembed_child_cache_teacher_bridge.py`
- `runs/eos-longembed-profile-perquery-frontier-mining-v1-20260620T140238Z/candidate-examples.tsv`
- `runs/eos-longembed-profile-perquery-frontier-mining-v1-20260620T140238Z/frontier-report.json`
- `runs/eos-longembed-profile-perquery-frontier-mining-v1-20260620T140238Z/*.per-query.jsonl`
- `runs/eos-real-longembed-external-compare-v2-qmsum/`
- `runs/eos-real-longembed-external-compare-v2-2wikimqa/`
- `runs/eos-resumable-longembed-narrativeqa-doc20-span256-v1/`
- `datasets/longembed-official/{qmsum,2wikimqa,narrativeqa}/`

Generated:
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/all-hard-negatives.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/train-hard-negatives.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/eval-hard-negatives.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/teacher-eligible-train-hard-negatives.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/teacher-eligible-eval-hard-negatives.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/teacher-eligible-{train,eval}.agreement-scored.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/teacher-eligible-{train,eval}.filtered.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/teacher-eligible-{train,eval}.{teacher-manifest,audit-summary,filter-summary}.json`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/by-category/*.jsonl`
- `runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z/manifest.json`

Counts:
- Candidate TSV rows: 104
- Emitted rows: 104
- Skipped rows: none
- Split: train 93, eval 11
- Teacher-eligible split: train 66, eval 5
- By dataset: QMSum 37, 2WikiMQA 34, NarrativeQA 33
- By category:
  - `direct_wins_token_span_loses`: 19
  - `external_wins_when_eos_loses`: 15
  - `sparse_parent_compact_loses_dense_rank`: 29
  - `token_span_wins_direct_loses`: 17
  - `worst_compact_loss_vs_direct`: 24
- By rescue family:
  - `direct_rescue`: 43
  - `token_span_rescue`: 17
  - `sparse_compact_preservation`: 29
  - `external_teacher_win`: 15

Teacher coverage/agreement:
- Train teacher-eligible rows: 66
  - with agreement-filtered averaged `teacher_scores`: 49
  - cleared: 17
  - missing coverage: 0
  - teacher disagreement: 17
  - agreement keep rate: 0.7424242424
  - dataset split: 2WikiMQA 31/31 scored, QMSum 18/35 scored
- Eval teacher-eligible rows: 5
  - with agreement-filtered averaged `teacher_scores`: 4
  - cleared: 1
  - missing coverage: 0
  - teacher disagreement: 1
  - agreement keep rate: 0.8
  - dataset split: 2WikiMQA 3/3 scored, QMSum 1/2 scored
- Go audit/filter:
  - train audit: examples 66, scored 49, missing 17, positive_top1_rate 1.0,
    mean_margin 0.118903
  - train filter: kept 49/49 scored rows, cleared 0, dropped 0
  - eval audit: examples 5, scored 4, missing 1, positive_top1_rate 1.0,
    mean_margin 0.148915
  - eval filter: kept 4/4 scored rows, cleared 0, dropped 0

Missing cache paths:
- `runs/external-vector-caches/qwen3-0.6b-longembed-real-doc20-128d/narrativeqa/manifest.json`
- `runs/external-vector-caches/qwen3-0.6b-longembed-real-doc20-128d/narrativeqa/query-vectors.jsonl`
- `runs/external-vector-caches/qwen3-0.6b-longembed-real-doc20-128d/narrativeqa/child-doc-vectors.jsonl`
- `runs/external-vector-caches/mxbai-large-longembed-real-doc20-128d/narrativeqa/manifest.json`
- `runs/external-vector-caches/mxbai-large-longembed-real-doc20-128d/narrativeqa/query-vectors.jsonl`
- `runs/external-vector-caches/mxbai-large-longembed-real-doc20-128d/narrativeqa/child-doc-vectors.jsonl`

Verification commands/results:
- `python3 scripts/test_mine_longembed_frontier_teacher_pairs.py`
  - Passed, 2 tests.
- `python3 -m py_compile scripts/mine_longembed_frontier_teacher_pairs.py scripts/test_mine_longembed_frontier_teacher_pairs.py`
  - Passed.
- `python3 scripts/test_score_longembed_child_cache_teacher_bridge.py`
  - Passed, 4 tests.
- `python3 scripts/test_curate_longembed_frontier_hard_negatives.py`
  - Passed, 3 tests.
- Real mining command:
  - `python3 scripts/mine_longembed_frontier_teacher_pairs.py --candidate-tsv runs/eos-longembed-profile-perquery-frontier-mining-v1-20260620T140238Z/candidate-examples.tsv --frontier-report runs/eos-longembed-profile-perquery-frontier-mining-v1-20260620T140238Z/frontier-report.json --per-query-dir runs/eos-longembed-profile-perquery-frontier-mining-v1-20260620T140238Z --dataset-dir qmsum=datasets/longembed-official/qmsum --dataset-dir 2wikimqa=datasets/longembed-official/2wikimqa --dataset-dir narrativeqa=datasets/longembed-official/narrativeqa --run-dir qmsum=runs/eos-real-longembed-external-compare-v2-qmsum --run-dir 2wikimqa=runs/eos-real-longembed-external-compare-v2-2wikimqa --run-dir narrativeqa=runs/eos-resumable-longembed-narrativeqa-doc20-span256-v1 --output-dir runs/eos-longembed-frontier-teacher-pair-miner-v1-20260620T141733Z --seed 17 --max-negatives 8`
  - Passed: input 104, emitted 104, train 93, eval 11.
- Teacher bridge commands for `teacher-eligible-train-hard-negatives.jsonl` and
  `teacher-eligible-eval-hard-negatives.jsonl`
  - Passed: train 66 examples, 49 with teacher_scores; eval 5 examples, 4 with
    teacher_scores.
- `go run ./cmd/eos audit-teacher-scores --mode text ...`
  - Passed for train and eval scored files.
- `go run ./cmd/eos filter-teacher-scores --mode text ...`
  - Passed for train and eval scored files.
- JSON/JSONL parse validation over generated run files
  - Passed.
- `git diff --check`
  - Passed.

Caveats/residual risk:
- Generated data is protected candidate data from capped slices; it is not
  benchmark evidence. The manifest and rows explicitly set `quality_claim=false`.
- NarrativeQA rows remain unscored because Qwen3/mxbai child-vector caches are
  unavailable for NarrativeQA in the expected cache roots.
- `eos/token_span_dense` and `eos/token_span_q4` both route to the available
  token-span multivector per-query file for each dataset; separate q4 token-span
  per-query files were not present in the supplied context.
- Sparse parent q4/q6/q7/q8 profiles route to the available
  `*-sparse-parent-q4678.per-query.jsonl` files, matching the supplied frontier
  mining run.
- No training was run, per descriptor.

Checkpoint candidate: yes.
- Source-only checkpoint candidate: the new reusable script and test pass all
  focused checks. Generated run evidence is ignored by Git and can stay as
  report-only unless the parent/user explicitly wants to track it.

Recommended next action:
- Review and checkpoint `scripts/mine_longembed_frontier_teacher_pairs.py` and
  `scripts/test_mine_longembed_frontier_teacher_pairs.py`.
- Use the scored `teacher-eligible-*.filtered.jsonl` for guarded follow-up
  training only with the manifest claim boundary preserved.
