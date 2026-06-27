package backend

import (
	"math"
	"strings"
	"testing"
)

func TestBERTEmbedderReferenceComposesLayersPoolsAndNormalizes(t *testing.T) {
	layer0 := bertEncoderLayerFixture()
	layer1 := bertEncoderLayerFixture()
	inputIDs := NewTensorI32([]int{2, 3}, []int32{0, 1, 0, 1, 0, 1})
	attentionMask := NewTensorI32([]int{2, 3}, []int32{1, 1, 0, 0, 0, 0})
	tokenTypeIDs := NewTensorI32([]int{2, 3}, []int32{0, 1, 1, 1, 0, 1})
	layerWeights := append(bertLayerWeightTensors(layer0), bertLayerWeightTensors(layer1)...)

	out, err := BERTEmbedderReference(
		inputIDs, attentionMask, tokenTypeIDs,
		bertTokenEmbeddingsFixture(),
		NewTensorF32([]int{3, 2}, []float32{0, 1, 1, 0, -1, 0.5}),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{2, 3}),
		NewTensorF32([]int{2}, []float32{0.5, -0.5}),
		layerWeights,
		2, 1, 0.25, "gelu",
	)
	if err != nil {
		t.Fatalf("bert embedder: %v", err)
	}
	positionIDs := NewTensorI32([]int{2, 3}, []int32{0, 1, 2, 0, 1, 2})
	hidden, err := BERTEmbeddingsReference(
		bertTokenEmbeddingsFixture(),
		NewTensorF32([]int{3, 2}, []float32{0, 1, 1, 0, -1, 0.5}),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{2, 3}),
		NewTensorF32([]int{2}, []float32{0.5, -0.5}),
		inputIDs, positionIDs, tokenTypeIDs, 0.25,
	)
	if err != nil {
		t.Fatalf("manual embeddings: %v", err)
	}
	for i, layer := range []bertEncoderFixture{layer0, layer1} {
		hidden, err = BERTEncoderLayerReference(
			hidden, attentionMask,
			layer.queryWeight, layer.queryBias,
			layer.keyWeight, layer.keyBias,
			layer.valueWeight, layer.valueBias,
			layer.attentionOutputWeight, layer.attentionOutputBias,
			layer.attentionLayerNormWeight, layer.attentionLayerNormBias,
			layer.intermediateWeight, layer.intermediateBias,
			layer.outputWeight, layer.outputBias,
			layer.outputLayerNormWeight, layer.outputLayerNormBias,
			1, 0.25, "gelu",
		)
		if err != nil {
			t.Fatalf("manual layer %d: %v", i, err)
		}
	}
	pooled, err := meanPoolMaskedTensor(hidden, attentionMask)
	if err != nil {
		t.Fatalf("manual pool: %v", err)
	}
	want := normalizeRows(pooled)
	assertTensorClose(t, out, []int{2, 2}, want.F32)
	norm := math.Sqrt(float64(out.F32[0]*out.F32[0] + out.F32[1]*out.F32[1]))
	if math.Abs(norm-1) > 1e-5 {
		t.Fatalf("first row norm = %.8f, want 1", norm)
	}
	for i, value := range out.F32[2:] {
		if value != 0 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("all-masked row value %d = %v, want finite zero", i, value)
		}
	}
}

func TestBERTEmbedderReferenceSupportsCLSPooling(t *testing.T) {
	layer := bertEncoderLayerFixture()
	inputIDs := NewTensorI32([]int{1, 3}, []int32{0, 1, 0})
	attentionMask := NewTensorI32([]int{1, 3}, []int32{1, 1, 0})
	tokenTypeIDs := NewTensorI32([]int{1, 3}, []int32{0, 1, 1})
	layerWeights := bertLayerWeightTensors(layer)

	out, err := BERTEmbedderReferenceWithPooling(
		inputIDs, attentionMask, tokenTypeIDs,
		bertTokenEmbeddingsFixture(),
		NewTensorF32([]int{3, 2}, []float32{0, 1, 1, 0, -1, 0.5}),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{2, 3}),
		NewTensorF32([]int{2}, []float32{0.5, -0.5}),
		layerWeights,
		1, 1, 0.25, "gelu", "cls",
	)
	if err != nil {
		t.Fatalf("bert embedder cls: %v", err)
	}

	positionIDs := NewTensorI32([]int{1, 3}, []int32{0, 1, 2})
	hidden, err := BERTEmbeddingsReference(
		bertTokenEmbeddingsFixture(),
		NewTensorF32([]int{3, 2}, []float32{0, 1, 1, 0, -1, 0.5}),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{2, 3}),
		NewTensorF32([]int{2}, []float32{0.5, -0.5}),
		inputIDs, positionIDs, tokenTypeIDs, 0.25,
	)
	if err != nil {
		t.Fatalf("manual embeddings: %v", err)
	}
	hidden, err = BERTEncoderLayerReference(
		hidden, attentionMask,
		layer.queryWeight, layer.queryBias,
		layer.keyWeight, layer.keyBias,
		layer.valueWeight, layer.valueBias,
		layer.attentionOutputWeight, layer.attentionOutputBias,
		layer.attentionLayerNormWeight, layer.attentionLayerNormBias,
		layer.intermediateWeight, layer.intermediateBias,
		layer.outputWeight, layer.outputBias,
		layer.outputLayerNormWeight, layer.outputLayerNormBias,
		1, 0.25, "gelu",
	)
	if err != nil {
		t.Fatalf("manual layer: %v", err)
	}
	cls, err := clsPoolTensor(hidden)
	if err != nil {
		t.Fatalf("manual cls pool: %v", err)
	}
	want := normalizeRows(cls)
	assertTensorClose(t, out, []int{1, 2}, want.F32)
}

func TestBERTEmbedderReferenceValidationFailures(t *testing.T) {
	f := bertEncoderLayerFixture()
	_, err := bertEmbedderTensor([]*Tensor{
		NewTensorI32([]int{1, 2}, []int32{0, 1}),
		NewTensorI32([]int{1, 2}, []int32{1, 1}),
		NewTensorI32([]int{1, 2}, []int32{0, 0}),
		bertTokenEmbeddingsFixture(),
		bertPositionEmbeddingsFixture(),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{1, 1}),
		NewTensorF32([]int{2}, []float32{0, 0}),
	}, tensorValueType("f32", []string{"B", "2"}), map[string]string{"num_hidden_layers": "bad"})
	if err == nil || !strings.Contains(err.Error(), "num_hidden_layers") {
		t.Fatalf("error = %v, want num_hidden_layers parse error", err)
	}
	_, err = BERTEmbedderReference(
		NewTensorI32([]int{1, 2}, []int32{0, 1}),
		NewTensorI32([]int{1, 2}, []int32{1, 1}),
		NewTensorI32([]int{1, 2}, []int32{0, 0}),
		bertTokenEmbeddingsFixture(), bertPositionEmbeddingsFixture(), bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{1, 1}), NewTensorF32([]int{2}, []float32{0, 0}),
		bertLayerWeightTensors(f), 1, 1, 0.25, "relu",
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported hidden_act") {
		t.Fatalf("error = %v, want unsupported hidden_act", err)
	}
}

func bertLayerWeightTensors(f bertEncoderFixture) []*Tensor {
	return []*Tensor{
		f.queryWeight, f.queryBias,
		f.keyWeight, f.keyBias,
		f.valueWeight, f.valueBias,
		f.attentionOutputWeight, f.attentionOutputBias,
		f.attentionLayerNormWeight, f.attentionLayerNormBias,
		f.intermediateWeight, f.intermediateBias,
		f.outputWeight, f.outputBias,
		f.outputLayerNormWeight, f.outputLayerNormBias,
	}
}
