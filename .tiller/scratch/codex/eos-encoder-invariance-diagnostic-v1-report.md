# eos-encoder-invariance-diagnostic-v1 Report

## Outcome

Cheap encoder invariance diagnostics were added as a scratch runner and executed against the current trainable/repeated encoder path. Both diagnostics fail current behavior:

- Permutation sensitivity: **FAIL**. Same non-pad token multiset in different order produced exactly identical embeddings.
- Padding invariance: **FAIL**. Adding masked right-padding changed the embedding materially.

No source test was left in the tree because the behavior is failing and would not be appropriate as a CI assertion yet.

## Distillation

The smallest local deterministic path is `runtime`'s in-process `EmbeddingTrainer` forward path, using `newTinyTrainableRepeatedEncoderEmbeddingTrainer` from `runtime/embedding_trainer_test.go`. That helper matches the `examples/encoder_trainable_q8x2.eos` shape: q8 token embedding, tied q/k/v/o attention weights, FFN up/projection, attention residual/layernorm, FFN residual/layernorm, and `EncoderRepeats=2`.

The diagnostic points to two next fix surfaces:

1. **Attention mask**: masked padding is only excluded at final mean pooling; padded tokens still participate in attention softmax/value mixing for active rows.
2. **Positional signal**: the encoder has no positional input, so self-attention plus per-token FFN plus mean pooling is permutation invariant over the active token set.

Attention scaling, heads, untied layers, and role tokens may be later quality improvements, but they are not the direct cause of these two invariance failures. Fix attention masking first, then add a positional signal.

## Code Paths Inspected

- `examples/encoder_trainable_q8x2.eos`: trainable repeated encoder graph; `softmax(scores*)` has no attention mask; final `mean_pool(normalized, attention_mask)` applies the mask only at pooling.
- `runtime/embedding_trainer.go`: `encodeSequence`, `encodeLayer`, `softmaxRowsInPlace`, and final masked pooling. `encodeLayer` computes scores over all token positions, applies row softmax, mixes all values, then applies the mask only while accumulating `state.pooled`.
- `runtime/embedding_trainer_test.go`: tiny trainable encoder helpers, especially `newTinyTrainableRepeatedEncoderEmbeddingTrainer`.
- `runtime/embedding_model.go`: `EmbeddingManifest` fields for `EncoderRepeats`, attention params, residual/layernorm flags, and tokenizer max sequence validation.
- `docs/eos-distillation-compact-default-spec.md`, `docs/manta-embed-sota-avenues.md`, and `docs/production-embedding.md`: current model milestone context and default candidate boundaries.

## Diagnostic Method

Scratch helper:

```text
.tiller/scratch/codex/eos-encoder-invariance-diagnostic-v1.sh
```

The helper writes a temporary Go test at `runtime/eos_encoder_invariance_diagnostic_v1_test.go`, runs:

```text
go test ./runtime -run TestEosEncoderInvarianceDiagnosticV1 -count=1 -v
```

and removes the temporary test via shell trap.

Model path:

```text
runtime package, newTinyTrainableRepeatedEncoderEmbeddingTrainer
source shape equivalent: examples/encoder_trainable_q8x2.eos
encoder_repeats=2
embedding_dim=3
learning_rate=0.02
tokenizer max_sequence locally raised to 8 for this diagnostic only
```

No training step is run. The diagnostic uses the initialized deterministic tiny weights and calls `prepareForwardWeights`, `prepareMask`, and `encodeSequence`.

## Inputs And Tolerances

Padding invariance threshold:

```text
max_abs <= 1e-6
l2 <= 1e-6
```

Permutation material-difference threshold:

```text
max_abs > 1e-5 OR l2 > 1e-5
```

Permutation sensitivity inputs:

```text
tokens_a=[0 1 2], mask_a=[1 1 1]
tokens_b=[2 1 0], mask_b=[1 1 1]
```

Padding invariance inputs:

```text
tokens_a=[0 1],   mask_a=[1 1]
tokens_b=[0 1 2], mask_b=[1 1 0]
```

## Results

Permutation sensitivity:

```text
max_abs=0
l2=0
result=FAIL
embedding_a=[0.243702382 -0.193421215 -0.0502811484]
embedding_b=[0.243702382 -0.193421215 -0.0502811484]
```

Padding invariance:

```text
max_abs=0.0702058375
l2=0.0864272416
result=FAIL
embedding_base=[0.258297443 0.142381489 -0.400678933]
embedding_padded=[0.21701476 0.113458335 -0.330473095]
```

## Files Changed Or Created

Created scratch files:

```text
.tiller/scratch/codex/eos-encoder-invariance-diagnostic-v1.sh
.tiller/scratch/codex/eos-encoder-invariance-diagnostic-v1-report.md
```

Temporary source file created and removed during diagnostics:

```text
runtime/eos_encoder_invariance_diagnostic_v1_test.go
```

No persistent source edits and no model artifact modifications.

## Verification

Initial attempted diagnostic:

```text
go test ./runtime -run TestEosEncoderInvarianceDiagnosticV1 -count=1 -v
```

Result: failed before measuring invariance because the tiny helper's manifest had `max_sequence=2` and diagnostic sequences length `3`. The scratch runner now locally raises only `trainer.manifest.Tokenizer.MaxSequence` to `8`.

Final diagnostic command:

```text
sh .tiller/scratch/codex/eos-encoder-invariance-diagnostic-v1.sh
```

Result: command exited `0`; Go test passed as a logging diagnostic and reported both invariance checks as `FAIL`.

Cleanup verification:

```text
test ! -e runtime/eos_encoder_invariance_diagnostic_v1_test.go && echo removed || echo still-present
```

Result:

```text
removed
```

Git status:

```text
git status --short --branch
```

Result before writing this report:

```text
## main...origin/main
```

## Caveats And Residual Risk

- The diagnostic uses deterministic tiny initialized weights, not the promoted sealed default asset. It is intentionally a cheap graph-behavior diagnostic, not a quality benchmark.
- The runtime serving path should also be checked after fixes, because this diagnostic primarily exercises the trainer's native forward path. The graph source in `examples/encoder_trainable_q8x2.eos` shows the same mask/position issues.
- The scratch runner temporarily writes into `runtime/` and removes the file on exit. It should not be run concurrently with another Go test command that expects a stable package file list.

## Checkpoint Candidate

Yes, but scratch-only. This is a coherent diagnostic/report slice with no source or artifact changes.

## Arbiter Next Action

Proceed to an encoder-v2 design/fix descriptor with two concrete gates:

1. Apply attention masking before softmax in both trainer/runtime graph paths so right-padding is invariant under the recorded tolerance.
2. Add positional signal to the encoder so reordered active token sequences produce materially different pooled embeddings.

After those fixes, promote these diagnostics from scratch into narrow Go tests covering trainer forward and serving/batch paths.
