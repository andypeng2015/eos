# eos-scale-rerun-v2 Dense-Gate Postmortem

## Outcome
Do not promote `eos-s40-fiqa-clean-scale-pilot-rerun-v2`.

The dense gate failed because SciFact and NFCorpus nDCG regressed while FiQA nDCG improved only slightly. Macro nDCG delta was `-0.000116608`, which is `0.001116608` short of the `+0.0010` exploration threshold. Macro recall passed with `+0.000085426`, but the promotion-critical dense nDCG gate did not.

## Dense Gate Metrics
| Dataset | nDCG candidate | nDCG default | nDCG delta | R@100 candidate | R@100 default | R@100 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | `0.564328` | `0.564537916` | `-0.000209916` | `0.796444` | `0.796444444` | `-0.000000444` |
| NFCorpus | `0.205484` | `0.205745968` | `-0.000261968` | `0.243763` | `0.242066067` | `+0.001696933` |
| FiQA | `0.121383` | `0.121260941` | `+0.000122059` | `0.350238` | `0.351678209` | `-0.001440209` |

Macro recall passed despite the dense nDCG failure: `+0.000085426`.

## Internal Eval Health
Internal training/eval metrics were healthy and did not predict the dense gate failure:

- Final eval AUC: `0.851810`
- Final eval margin: `0.244341`
- Final eval top1: `0.946619`
- Hard eval AUC: `0.831277`
- Hard eval margin: `0.264354`
- Hard eval top1: `0.928082`

## Cause
The failure is an objective/data mismatch. Pair separation and hard-negative InfoNCE improved internal discrimination, but that did not translate into macro nDCG@10 rank lift on the dense gate.

The 100k data mix was skewed:

- FiQA: `76,617`
- SciFact: `12,191`
- NFCorpus: `11,192`
- Teacher-scored rows: `20,265`, all clean FiQA
- One-negative rows: `74,874`

The run therefore over-represented FiQA and clean teacher-scored FiQA rows while under-representing balanced boundary behavior across SciFact and NFCorpus. The result is consistent with a candidate that learned useful pair separation in the training objective without producing enough top-10 rank improvement across the macro dense gate.

## Recommendation
Do not promote this candidate. Do not run compact evaluation for this candidate.

Launch `eos-s40-balanced-boundary-macro-ndcg-ablation-v1` instead:

- Use balanced 30k arms across the gate datasets.
- Include train/dev-safe boundary-mined rows.
- Optimize for macro nDCG@10 lift rather than relying on internal AUC, pair margin, or hard-negative InfoNCE health as promotion proxies.

## Descriptor
Suggested follow-up descriptor:

```text
id/title: eos-s40-balanced-boundary-macro-ndcg-ablation-v1
role/profile: tiller-worker or tiller-debugger for bounded candidate construction, training, and dense gate evaluation
objective: Build and run balanced 30k-arm ablations with train/dev-safe boundary-mined rows to target macro nDCG@10 lift across SciFact, NFCorpus, and FiQA.
constraints: Do not promote unless dense macro nDCG clears the exploration gate; avoid compact evaluation until dense gate passes; keep dataset splits train/dev safe.
verification target: dense scoreboard macro nDCG delta >= +0.0010 with no unacceptable per-dataset regression; summarize recall separately.
checkpoint criteria: report-only checkpoint for setup/diagnosis, source checkpoint only for reusable harness or data-prep fixes.
report contract: Outcome; per-arm data mix; dense gate metrics; internal eval metrics; caveats; promotion recommendation; Arbiter next action.
```

## zsh PIPESTATUS Note
The outer `PIPESTATUS[0]` failure was a one-off zsh invocation bug, not evidence that the wrapper failed. Future zsh `tee` wrappers should snapshot zsh's lowercase 1-indexed array immediately:

```zsh
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw 2>&1 | tee "$LOG"
st=${pipestatus[1]}
exit "$st"
```

The existing report `.tiller/scratch/codex/eos-wrapper-pipestatus-zsh-fix-v1-report.md` contains the detailed PIPESTATUS diagnosis.

## Arbiter Next Action
Mark `eos-s40-fiqa-clean-scale-pilot-rerun-v2` as a dense-gate failure and route next execution to `eos-s40-balanced-boundary-macro-ndcg-ablation-v1`. Do not spend compact-eval budget on this candidate.
