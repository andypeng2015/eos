# compact-hardnegative-teacher-distill-v1 report

## Outcome

Implemented compact `compact_transformer_v1` hard-negative teacher-distribution distillation.

Compact hard-negative training no longer rejects `TeacherLossWeight > 0` or valid per-row `teacher_scores`. It now validates teacher score length against the local candidate count, accumulates teacher KL pooled gradients through the existing `accumulateTeacherDistributionHardNegativeGrads` helper, blends base and teacher losses/grads with the generic trainer's `1/(1+w)` and `w/(1+w)` scheme, and includes teacher pairs in returned `BatchSize`.

## Files Changed

- `runtime/embedding_trainer.go`
- `runtime/embedding_trainer_test.go`
- `.tiller/scratch/codex/compact-hardnegative-teacher-distill-v1-report.md`

## Exact Behavior Change

- `runCompactHardNegativeContrastiveBatchUpdate` still rejects compact hard-negative Matryoshka, TurboQuant prefix/compact/rank-margin objectives, hybrid loss, and unsupported contrastive losses.
- Valid `teacher_scores` are accepted for compact hard-negative InfoNCE and grouped InfoNCE.
- Malformed non-empty `teacher_scores` are rejected before optimizer mutation with the same error shape as the generic hard-negative path.
- When `TeacherLossWeight > 0` and usable teacher-score rows remain after source weighting, compact adds teacher-distribution KL gradients before compact backprop and reports base pairs plus teacher pairs.
- When `TeacherLossWeight == 0`, or source-specific teacher weight suppresses all teacher rows, behavior remains the base compact hard-negative update apart from teacher score shape validation.

## Verification Commands And Results

- `gofmt -w runtime/embedding_trainer.go runtime/embedding_trainer_test.go`: passed.
- `go test ./runtime -run 'CompactEmbeddingTrainerHardNegative|TeacherDistribution|HardNegativeTeacher|Compact' -count=1`: passed (`ok m31labs.dev/eos/runtime 36.790s`).
- `git diff --check`: passed.

`go test ./cmd/eos -run 'TrainEmbed|Teacher' -count=1` was not run because this patch does not change CLI parsing, train runner configuration, or teacher score normalization.

## Caveats / Residual Risk

- This enables the compact pooled-gradient teacher objective but does not add a quality recipe or BGE packet.
- Teacher score validation mirrors the generic trainer path in `runtime/embedding_trainer.go`; finite-value validation remains owned by dataset loading/conversion paths.

## Checkpoint Candidate

Yes. This is a coherent two-source-file implementation with focused runtime verification passing.

## Arbiter Next Action

Root should review the compact trainer diff and checkpoint if accepted, then unblock the BGE-guided compact student packet using `TeacherLossWeight > 0` and `teacher_scores`.
