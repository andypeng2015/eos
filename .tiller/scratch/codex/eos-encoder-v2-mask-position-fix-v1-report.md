# eos-encoder-v2-mask-position-fix-v1 Report

## Outcome

Implemented an explicit encoder-v2 path for new/generated embedding packages:

- Mask fix landed: trainer attention now excludes padded key columns before attention softmax when `attention_mask_mode=key`.
- Positional v2 landed: trainer injects fixed RoPE into token rows when `position_encoding=rope`.
- Serving/source consistency landed for generated default packages: the compiler preset emits `rope(dequant(hidden_q))` plus `masked_softmax(scores, attention_mask)`, and `models.DefaultEmbeddingManifest` writes matching manifest fields.
- Old manifests remain v1 by default with `attention_mask_mode=none` and `position_encoding=none`.

No model artifacts, generated run roots, or checked-in binary example manifests were modified.

## Distillation

The root cause was two missing graph semantics:

1. `encodeLayer` and `encodeBatchedLayerStates` computed full attention scores over all token columns and normalized them before any attention mask was applied. Masking only happened at final mean pooling, so right-padding could change active token representations.
2. The repeated encoder had no positional input. Self-attention, FFN, and masked mean pooling were therefore permutation invariant over the active token multiset.

The fix is opt-in v2 behavior rather than silently changing old artifacts. `EmbeddingManifest.ValidateModule` requires serving graph support when v2 fields are set:

- `attention_mask_mode=key` requires a `masked_softmax` kernel op.
- `position_encoding=rope` requires a `rope` kernel op.

## Files Changed

- `runtime/embedding_trainer.go`
  - Applies manifest-gated RoPE to token embeddings before encoder layers.
  - Applies manifest-gated masked key-column attention softmax in single and batched trainer paths.
  - Stores masked probabilities, so masked key-column softmax backward gradients are zero through the existing softmax derivative.
- `runtime/embedding_model.go`
  - Adds manifest fields/constants for `attention_mask_mode` and `position_encoding`.
  - Adds v2 graph validation.
- `runtime/backend/tensor_ops.go`
  - Adds host/symbolic `masked_softmax` for rank-2 and rank-3 attention scores.
  - Extends batched RoPE to reset position per batch item.
- `compiler/semantics.go`, `compiler/compiler.go`
  - Adds `masked_softmax(scores, mask)` type checking/lowering.
  - Updates default trainable encoder preset to emit RoPE and masked softmax.
- `models/default_embedding.go`
  - Default generated package manifests now opt into v2 fields.
- Tests:
  - `runtime/embedding_trainer_test.go`
  - `runtime/backend/tensor_ops_test.go`
  - `compiler/compiler_test.go`
  - `models/default_embedding_test.go`

## Behavior

Trainer v2 diagnostics now pass:

- Right-padding invariance passes under `1e-6` max-abs/L2 tolerance.
- Reordered active tokens are materially different with RoPE.
- A v1 guard test confirms old no-position manifests remain permutation invariant by design.

The checked-in `examples/encoder_trainable_q8x2.eos` remains v1 because its sibling `.embedding.mll` manifest is a binary artifact and was not modified. Generated default packages use the new compiler preset plus v2 manifest, so their train/serve source contract is consistent.

## Verification

Passed:

- `go test ./runtime -run 'TestEmbeddingTrainerEncoderV2PaddingInvariantAndOrderSensitive|TestEmbeddingTrainerEncoderV1RemainsPermutationInvariantWithoutPositionEncoding|TestEmbeddingTrainerMaskedAttentionZerosPaddedKeyColumns|TestMaskedSoftmaxRows|TestRoPERows' -count=1 -v`
- `go test ./runtime/backend -run 'TestMaskedSoftmaxRows|TestRoPERows' -count=1 -v`
- `go test ./compiler -run 'TestBuildEncoderTrainableQ8x2Preset|TestBuildEncoderTrainableQ4x2Preset|TestBuildTinyAttentionEmbedSource' -count=1 -v`
- `go test ./models -run 'TestInitDefaultEmbeddingPackageCreatesTrainablePackage|TestInitDefaultEmbeddingPackageQ4DeclaresQ4Params' -count=1 -v`
- `go test ./runtime -run 'TestEmbeddingTrainerTrainStepSupportsRepeatedEncoderAndExportsQuantizedWeights|TestEmbeddingTrainerTrainStepSupportsEncoderAndExportsQuantizedWeights|TestEmbeddingTrainerAttentionCheckpointRoundTrip|TestEmbeddingTrainerEncoderCheckpointRoundTrip' -count=1 -v`
- `go test ./runtime -run 'TestEmbeddingTrainManifestValidateModule|TestEmbeddingManifest|TestEmbeddingModel|TestEmbeddingTrainerRejects|TestEmbeddingTrainer.*Manifest' -count=1 -v`
- `go test ./runtime ./runtime/backend ./compiler ./models`
- `go test ./cmd/eos -run 'TestRunInitModel|TestRunTrainCorpus|TestRunInspect|TestRunCompile' -count=1 -v`
- `go test ./cmd/eos -run 'TestRunTrainCorpusRepeatedEncoderExampleFlow|TestRunInspectShowsRepeatedEncoderEmbeddingDetails' -count=1 -v`
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

## Caveats And Residual Risk

- No training was run.
- No released/default artifacts were mutated. Existing artifacts/manifests keep v1 behavior unless regenerated with v2 fields and graph ops.
- The checked-in example `.eos` plus binary manifest remains v1; updating that example to v2 should be a separate artifact-aware slice.
- Host/symbolic serving and trainer paths are covered. Native CUDA/Metal specialization for `masked_softmax` was not added; if device-native serving dispatch starts executing that op directly, add dedicated native kernels or force host fallback for this op.

## Checkpoint Candidate

Yes. This is a coherent source/test slice with passing focused and package-level verification, and it preserves old artifact behavior through explicit manifest gating.

## Arbiter Next Action

Add an artifact-aware encoder-v2 package migration slice:

1. Regenerate or create a new v2 example bundle/manifest without overwriting old v1 artifacts.
2. Add packaged inference parity tests that load a v2 generated package and compare `EmbeddingModel.Embed`/`EmbedBatch` against the trainer forward path for padding invariance and order sensitivity.
3. Decide whether `masked_softmax` needs CUDA/Metal native kernels or an explicit host-fallback capability marker before any device-serving claim.
