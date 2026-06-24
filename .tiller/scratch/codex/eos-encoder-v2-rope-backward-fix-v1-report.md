# eos-encoder-v2-rope-backward-fix-v1 Report

## Outcome

Fixed the checkpoint-blocking RoPE backward issue in the pending encoder-v2 source diff.

The trainer now inverse/transposed-rotates row gradients before accumulating them into token embedding rows whenever `position_encoding=rope`.

## Root Cause

The encoder-v2 forward path applies fixed RoPE to gathered token rows before the encoder stack. Backward propagation returned gradients in RoPE-transformed input space, but the trainer accumulated those gradients directly into the unrotated token embedding table. For RoPE, the correct token embedding gradient is `R^T * grad_rope_input` per token position.

## Files Changed

- `runtime/embedding_trainer.go`
  - Added `applyRoPETransposeToRowsInPlace`.
  - Added `(*EmbeddingTrainer).accumulateInputTokenGrad`.
  - Routed every trainer `accumulateTokenGrad` call site through `accumulateInputTokenGrad`.
  - Covered single pairwise, batched pairwise fallback, contrastive, hard-negative, and batched contrastive/repeated-encoder backward paths.
- `runtime/embedding_trainer_test.go`
  - Added `TestEmbeddingTrainerRoPEBackwardRotatesTokenEmbeddingGradients`.

The worktree also still contains the broader pending v2 source/test edits from `eos-encoder-v2-mask-position-fix-v1`; those were not reverted or narrowed.

## Exact Gradient Fix

`accumulateInputTokenGrad` now checks `t.manifest.PositionEncoding`. For `rope`, it copies the row-gradient buffer, applies the inverse RoPE rotation using the same angle schedule as forward with the sine term negated, then calls the existing raw `accumulateTokenGrad` primitive.

`rg "accumulateTokenGrad" runtime/embedding_trainer.go` now shows all trainer call sites routed through `t.accumulateInputTokenGrad`; the raw `accumulateTokenGrad` function remains only as the low-level accumulator used by the helper and tests.

## Tests Added

- `TestEmbeddingTrainerRoPEBackwardRotatesTokenEmbeddingGradients`
  - Analytically verifies token embedding gradients equal raw row gradients after RoPE transpose/inverse rotation.
  - Confirms raw accumulation differs from the expected inverse-RoPE result, so it catches the original bug.
  - Confirms the helper does not mutate its input gradient buffer.

## Verification

Passed:

- `go test ./runtime -run 'TestEmbeddingTrainerRoPEBackwardRotatesTokenEmbeddingGradients|TestEmbeddingTrainerEncoderV2PaddingInvariantAndOrderSensitive|TestEmbeddingTrainerEncoderV1RemainsPermutationInvariantWithoutPositionEncoding|TestEmbeddingTrainerMaskedAttentionZerosPaddedKeyColumns' -count=1 -v`
- `go test ./runtime -run 'TestEmbeddingTrainerRoPEBackwardRotatesTokenEmbeddingGradients|TestEmbeddingTrainerEncoderV2PaddingInvariantAndOrderSensitive|TestEmbeddingTrainerEncoderV1RemainsPermutationInvariantWithoutPositionEncoding|TestEmbeddingTrainerMaskedAttentionZerosPaddedKeyColumns|TestEmbeddingTrainerTrainStepSupportsRepeatedEncoderAndExportsQuantizedWeights|TestEmbeddingTrainerTrainStepSupportsEncoderAndExportsQuantizedWeights|TestEmbeddingTrainerAttentionCheckpointRoundTrip|TestEmbeddingTrainerEncoderCheckpointRoundTrip|TestEmbeddingTrainManifestValidateModule|TestEmbeddingManifest|TestEmbeddingModel|TestEmbeddingTrainerRejects|TestEmbeddingTrainer.*Manifest' -count=1 -v`
- `go test ./runtime/backend -run 'TestMaskedSoftmaxRows|TestRoPERows' -count=1 -v`
- `go test ./compiler -run 'TestBuildEncoderTrainableQ8x2Preset|TestBuildEncoderTrainableQ4x2Preset|TestBuildTinyAttentionEmbedSource' -count=1 -v`
- `go test ./models -run 'TestInitDefaultEmbeddingPackageCreatesTrainablePackage|TestInitDefaultEmbeddingPackageQ4DeclaresQ4Params' -count=1 -v`
- `go test ./runtime ./runtime/backend ./compiler ./models`
- `git diff --check`

Final status:

```text
## main...origin/main
 M compiler/compiler.go
 M compiler/compiler_test.go
 M compiler/semantics.go
 M models/default_embedding.go
 M models/default_embedding_test.go
 M runtime/backend/tensor_ops.go
 M runtime/backend/tensor_ops_test.go
 M runtime/embedding_model.go
 M runtime/embedding_trainer.go
 M runtime/embedding_trainer_test.go
```

## Remaining Native Serving Caveats

- Host/symbolic `masked_softmax` and host rank-3 RoPE are covered by tests.
- Native CUDA/Metal specialized `masked_softmax` kernels were not added.
- I did not claim full device-native encoder-v2 serving. If native serving dispatch must execute `masked_softmax` directly, add dedicated CUDA/Metal kernels or an explicit host-fallback capability marker.

## Checkpoint Candidate

Yes. The RoPE backward blocker is fixed, all required package tests pass, and remaining native-serving gaps are explicitly documented.

## Arbiter Next Action

Checkpoint the source slice, then continue with the previously identified artifact-aware encoder-v2 package migration and native-serving capability decision.
