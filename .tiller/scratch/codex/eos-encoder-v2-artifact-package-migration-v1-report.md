# eos-encoder-v2-artifact-package-migration-v1 Report

## Outcome

Implemented artifact-aware encoder-v2 package coverage without modifying checked-in v1 artifacts.

- Added a generated-package parity test that initializes a temporary v2 default embedder package, loads it through `Runtime.LoadEmbeddingPackage`, and exercises public `EmbeddingModel.Embed` / `EmbedBatch`.
- Fixed masked direct embedding input construction so trailing pad IDs are masked as padding. Leading and interior pad IDs remain active tokens, preserving existing masked-pool behavior.
- Explicitly asserted that the generated v2 package declares `attention_mask_mode=key`, `position_encoding=rope`, and contains serving graph ops `masked_softmax` and `rope`.
- Captured the native-serving boundary in test and report: this slice validates fallback-capable packaged serving, not full CUDA/Metal device-native `masked_softmax`.

## Distillation

The source v2 encoder path was already present after the prior mask/RoPE fixes. The missing artifact-ready coverage was the public package path: initialize a generated package, load sibling package artifacts/manifests/weights, and verify the serving behavior users actually call.

Direct `Embed` previously built an all-ones mask for every provided token, so `[4, 5]` and right-padded `[4, 5, 0, 0]` were not equivalent even under a v2 manifest. `buildMaskedTokenInputs` now trims only trailing `pad_id` values when constructing masks. This is scoped to masked package execution and leaves leading/interior pad IDs active.

## Files Changed

- `runtime/embedding_model.go`
  - `buildMaskedTokenInputs` now masks trailing `pad_id` tokens as padding for direct and batched masked package execution.
- `models/default_embedding_test.go`
  - Added `TestDefaultEmbeddingPackageV2PackagedInferenceParity`.
  - Added small local helpers for artifact op checks and vector comparison.

No `examples/*` source, manifest, or binary artifact files were changed.

## Artifact/Package Path Used

The test creates a deterministic temporary package:

- `${t.TempDir()}/eos-embed-v2-generated.mll`
- generated via `models.InitDefaultEmbeddingPackage`
- config: `Name=eos-embed-v2-generated`, `VocabSize=16`, `MaxSequence=8`, `EmbeddingDim=4`, `HiddenDim=8`, `Seed=7`
- loaded through `eosruntime.New(vulkan.New()).LoadEmbeddingPackage(...)`

The Vulkan backend is the portable host-fallback backend, avoiding CUDA/Metal hardware dependence while still exercising packaged runtime execution.

## Parity Assertions

`TestDefaultEmbeddingPackageV2PackagedInferenceParity` verifies:

- Manifest declares `attention_mask_mode=key`.
- Manifest declares `position_encoding=rope`.
- Artifact kernel bodies include `masked_softmax`.
- Artifact kernel bodies include `rope`.
- Artifact does not declare `device_execution` as a required capability for this v2 package.
- `EmbeddingModel.Embed([4, 5])` matches `EmbeddingModel.Embed([4, 5, 0, 0])` within `1e-5`.
- `EmbeddingModel.Embed([4, 5, 6])` differs from `EmbeddingModel.Embed([6, 5, 4])`, proving packaged v2 order sensitivity.
- `EmbeddingModel.EmbedBatch` rows match per-example `Embed` rows for equal-length padded examples, exercising the rank-3 batched RoPE path with position reset per batch item.

## Native-Serving/Fallback Decision

No CUDA/Metal native `masked_softmax` kernel was added in this slice.

Current boundary:

- v2 generated packages are artifact-ready for host/symbolic fallback-capable serving.
- The package parity test uses the fallback backend intentionally.
- The v2 artifact test rejects a device-native-only claim by checking that the generated module does not require `device_execution`.
- CUDA/Metal full device-native v2 serving remains blocked until `masked_softmax` has dedicated native kernels or an explicit per-op fallback marker/capability policy is added and tested.

## Verification

Passed:

- `go test ./models -run TestDefaultEmbeddingPackageV2PackagedInferenceParity -count=1 -v`
- `go test ./runtime -run 'TestEmbeddingModelEmbedMasked|TestEmbeddingModelEmbedBatchPadsRaggedWithMask|TestEmbeddingModelEmbedBatchGroupsRaggedAttentionInputs|TestLoadEmbeddingPackageAcceptsTrainingPackageManifest' -count=1 -v`
- `go test ./runtime ./compiler ./models`
- `git diff --check`

Final status:

```text
## main...origin/main
 M models/default_embedding_test.go
 M runtime/embedding_model.go
```

## Caveats And Residual Risk

- No model training was run.
- No checked-in v1 artifacts or manifests were regenerated or modified.
- The test uses a temporary generated package, not a committed v2 example bundle.
- The native capability boundary is still coarse: runtime output metadata does not yet expose a per-op `masked_softmax` fallback marker from `EmbeddingResult`. This report and test document fallback-capable packaged serving rather than full CUDA/Metal native serving.

## Checkpoint Candidate

Yes. This is a focused, verified source/test slice with no artifact mutation and passing requested package verification.

## Arbiter Next Action

Checkpoint this artifact-package migration slice. A follow-up native-serving descriptor should either implement CUDA/Metal native `masked_softmax` kernels or add explicit per-op fallback capability metadata that packaged v2 serving can assert directly.
