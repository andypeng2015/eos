package backend

import (
	"math"
	"strings"
	"testing"
)

func TestBERTEmbeddingReferenceRank1LayerNormAffine(t *testing.T) {
	out, err := BERTEmbeddingsReference(
		bertTokenEmbeddingsFixture(),
		bertPositionEmbeddingsFixture(),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{2, 3}),
		NewTensorF32([]int{2}, []float32{0.5, -0.5}),
		NewTensorI32([]int{2}, []int32{0, 1}),
		NewTensorI32([]int{2}, []int32{0, 1}),
		NewTensorI32([]int{2}, []int32{1, 0}),
		3,
	)
	if err != nil {
		t.Fatalf("bert embeddings: %v", err)
	}
	want := []float32{
		1.5, -2,
		float32(-9.0/math.Sqrt(23.25) + 0.5),
		float32(13.5/math.Sqrt(23.25) - 0.5),
	}
	assertTensorClose(t, out, []int{2, 2}, want)
}

func TestBERTEmbeddingReferenceRank2LayerNormAffine(t *testing.T) {
	out, err := BERTEmbeddingsReference(
		bertTokenEmbeddingsFixture(),
		bertPositionEmbeddingsFixture(),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{2, 3}),
		NewTensorF32([]int{2}, []float32{0.5, -0.5}),
		NewTensorI32([]int{2, 2}, []int32{0, 1, 0, 1}),
		NewTensorI32([]int{2, 2}, []int32{0, 1, 1, 0}),
		NewTensorI32([]int{2, 2}, []int32{1, 0, 0, 1}),
		3,
	)
	if err != nil {
		t.Fatalf("bert embeddings rank-2: %v", err)
	}
	want := []float32{
		1.5, -2,
		float32(-9.0/math.Sqrt(23.25) + 0.5),
		float32(13.5/math.Sqrt(23.25) - 0.5),
		0.5, -0.5,
		float32(-7.0/math.Sqrt(15.25) + 0.5),
		float32(10.5/math.Sqrt(15.25) - 0.5),
	}
	assertTensorClose(t, out, []int{2, 2, 2}, want)
}

func TestBERTEmbeddingReferenceValidationFailures(t *testing.T) {
	valid := func() []*Tensor {
		return []*Tensor{
			bertTokenEmbeddingsFixture(),
			bertPositionEmbeddingsFixture(),
			bertTokenTypeEmbeddingsFixture(),
			NewTensorF32([]int{2}, []float32{1, 1}),
			NewTensorF32([]int{2}, []float32{0, 0}),
			NewTensorI32([]int{2}, []int32{0, 1}),
			NewTensorI32([]int{2}, []int32{0, 1}),
			NewTensorI32([]int{2}, []int32{0, 1}),
		}
	}
	tests := []struct {
		name string
		edit func([]*Tensor)
		want string
	}{
		{
			name: "mismatched id shapes",
			edit: func(tensors []*Tensor) {
				tensors[6] = NewTensorI32([]int{1}, []int32{0})
			},
			want: "position_ids shape",
		},
		{
			name: "bad id dtype",
			edit: func(tensors []*Tensor) {
				tensors[5] = NewTensorI64([]int{2}, []int64{0, 1})
			},
			want: `input_ids dtype "i64" is not i32`,
		},
		{
			name: "out of range id",
			edit: func(tensors []*Tensor) {
				tensors[5] = NewTensorI32([]int{2}, []int32{0, 9})
			},
			want: "input_ids value 9 out of range",
		},
		{
			name: "bad gamma shape",
			edit: func(tensors []*Tensor) {
				tensors[3] = NewTensorF32([]int{3}, []float32{1, 1, 1})
			},
			want: "embedding_layernorm_weight shape",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tensors := valid()
			tc.edit(tensors)
			_, err := bertEmbeddingsTensor(tensors, tensorValueType("f32", []string{"T", "2"}), map[string]string{"epsilon": "1e-12"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestBERTEmbeddingReferenceRejectsBadEpsilon(t *testing.T) {
	_, err := bertEmbeddingsTensor([]*Tensor{
		bertTokenEmbeddingsFixture(),
		bertPositionEmbeddingsFixture(),
		bertTokenTypeEmbeddingsFixture(),
		NewTensorF32([]int{2}, []float32{1, 1}),
		NewTensorF32([]int{2}, []float32{0, 0}),
		NewTensorI32([]int{1}, []int32{0}),
		NewTensorI32([]int{1}, []int32{0}),
		NewTensorI32([]int{1}, []int32{0}),
	}, tensorValueType("f32", []string{"T", "2"}), map[string]string{"epsilon": "not-a-number"})
	if err == nil || !strings.Contains(err.Error(), "epsilon") {
		t.Fatalf("error = %v, want epsilon parse error", err)
	}
}

func bertTokenEmbeddingsFixture() *Tensor {
	return NewTensorF32([]int{2, 2}, []float32{
		1, 2,
		10, 20,
	})
}

func bertPositionEmbeddingsFixture() *Tensor {
	return NewTensorF32([]int{2, 2}, []float32{
		0, 1,
		1, 0,
	})
}

func bertTokenTypeEmbeddingsFixture() *Tensor {
	return NewTensorF32([]int{2, 2}, []float32{
		0, 0,
		2, -2,
	})
}
