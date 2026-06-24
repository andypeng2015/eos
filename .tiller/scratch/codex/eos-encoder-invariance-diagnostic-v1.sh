#!/usr/bin/env sh
set -eu

test_file="runtime/eos_encoder_invariance_diagnostic_v1_test.go"
cleanup() {
	rm -f "$test_file"
}
trap cleanup EXIT INT TERM

cat > "$test_file" <<'GOEOF'
package eosruntime

import (
	"fmt"
	"math"
	"testing"
)

func TestEosEncoderInvarianceDiagnosticV1(t *testing.T) {
	const (
		maxAbsTolerance      = float32(1e-6)
		l2Tolerance          = float32(1e-6)
		materialMaxAbs       = float32(1e-5)
		materialL2           = float32(1e-5)
		learningRate float32 = 0.02
	)

	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, learningRate)
	trainer.manifest.Tokenizer.MaxSequence = 8
	forward := trainer.prepareForwardWeights()

	embed := func(tokens, mask []int32) []float32 {
		t.Helper()
		preparedMask, err := trainer.prepareMask(tokens, mask)
		if err != nil {
			t.Fatalf("prepare mask tokens=%v mask=%v: %v", tokens, mask, err)
		}
		seq, err := trainer.encodeSequence(tokens, preparedMask, forward.token, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, false)
		if err != nil {
			t.Fatalf("encode tokens=%v mask=%v: %v", tokens, mask, err)
		}
		return append([]float32(nil), seq.pooled...)
	}

	permutationA := []int32{0, 1, 2}
	permutationB := []int32{2, 1, 0}
	permutationMask := []int32{1, 1, 1}
	paddingBase := []int32{0, 1}
	paddingBaseMask := []int32{1, 1}
	paddingPadded := []int32{0, 1, 2}
	paddingPaddedMask := []int32{1, 1, 0}

	permAEmbedding := embed(permutationA, permutationMask)
	permBEmbedding := embed(permutationB, permutationMask)
	baseEmbedding := embed(paddingBase, paddingBaseMask)
	paddedEmbedding := embed(paddingPadded, paddingPaddedMask)

	permMaxAbs, permL2 := vectorDiffStats(permAEmbedding, permBEmbedding)
	padMaxAbs, padL2 := vectorDiffStats(baseEmbedding, paddedEmbedding)
	permPass := permMaxAbs > materialMaxAbs || permL2 > materialL2
	padPass := padMaxAbs <= maxAbsTolerance && padL2 <= l2Tolerance

	t.Logf("model_path=compiler.PresetEncoderTrainableQ8x2 equivalent via newTinyTrainableRepeatedEncoderEmbeddingTrainer")
	t.Logf("source_model=examples/encoder_trainable_q8x2.eos")
	t.Logf("package=runtime, encoder_repeats=%d, dim=%d, learning_rate=%g", trainer.encoderRepeats(), len(permAEmbedding), learningRate)
	t.Logf("tolerances padding max_abs<=%.8g l2<=%.8g; permutation material max_abs>%.8g or l2>%.8g", maxAbsTolerance, l2Tolerance, materialMaxAbs, materialL2)
	t.Logf("permutation tokens_a=%v mask_a=%v tokens_b=%v mask_b=%v max_abs=%.9g l2=%.9g result=%s", permutationA, permutationMask, permutationB, permutationMask, permMaxAbs, permL2, passFail(permPass))
	t.Logf("padding tokens_a=%v mask_a=%v tokens_b=%v mask_b=%v max_abs=%.9g l2=%.9g result=%s", paddingBase, paddingBaseMask, paddingPadded, paddingPaddedMask, padMaxAbs, padL2, passFail(padPass))
	t.Logf("permutation_embedding_a=%s", formatFloat32Vector(permAEmbedding))
	t.Logf("permutation_embedding_b=%s", formatFloat32Vector(permBEmbedding))
	t.Logf("padding_embedding_base=%s", formatFloat32Vector(baseEmbedding))
	t.Logf("padding_embedding_padded=%s", formatFloat32Vector(paddedEmbedding))
}

func vectorDiffStats(a, b []float32) (float32, float32) {
	maxAbs := float32(0)
	sumSquares := float64(0)
	for i := range a {
		diff := abs32(a[i] - b[i])
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSquares += float64(diff * diff)
	}
	return maxAbs, float32(math.Sqrt(sumSquares))
}

func passFail(pass bool) string {
	if pass {
		return "PASS"
	}
	return "FAIL"
}

func formatFloat32Vector(values []float32) string {
	return fmt.Sprintf("[%.9g %.9g %.9g]", values[0], values[1], values[2])
}
GOEOF

go test ./runtime -run TestEosEncoderInvarianceDiagnosticV1 -count=1 -v
