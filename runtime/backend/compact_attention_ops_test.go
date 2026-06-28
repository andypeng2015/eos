package backend

import (
	"math"
	"testing"
)

func TestCompactMultiheadAttentionTensorMatchesReference(t *testing.T) {
	q := NewTensorF32([]int{2, 4}, []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
	})
	k := NewTensorF32([]int{2, 4}, []float32{
		1, 0, 1, 0,
		0, 1, 0, 1,
	})
	v := NewTensorF32([]int{2, 4}, []float32{
		1, 10, 100, 1000,
		2, 20, 200, 2000,
	})
	mask := NewTensorI32([]int{2}, []int32{1, 1})

	got, err := compactMultiheadAttentionTensor(q, k, v, mask, map[string]string{"num_attention_heads": "2"})
	if err != nil {
		t.Fatalf("compact_multihead_attention: %v", err)
	}
	want := referenceCompactMultiheadAttention(q.F32, k.F32, v.F32, mask.I32, 1, 2, 4, 2)
	assertBackendTensorClose(t, got, []int{2, 4}, want, 1e-6)
}

func TestCompactMultiheadAttentionTensorBatchedMasked(t *testing.T) {
	q := NewTensorF32([]int{2, 2, 4}, []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
		1, 1, 1, 1,
		1, 0, 1, 0,
	})
	k := NewTensorF32(q.Shape, append([]float32(nil), q.F32...))
	v := NewTensorF32(q.Shape, []float32{
		1, 10, 100, 1000,
		2, 20, 200, 2000,
		3, 30, 300, 3000,
		4, 40, 400, 4000,
	})
	mask := NewTensorI32([]int{2, 2}, []int32{1, 0, 1, 1})

	got, err := compactMultiheadAttentionTensor(q, k, v, mask, map[string]string{"num_attention_heads": "2"})
	if err != nil {
		t.Fatalf("compact_multihead_attention: %v", err)
	}
	want := referenceCompactMultiheadAttention(q.F32, k.F32, v.F32, mask.I32, 2, 2, 4, 2)
	assertBackendTensorClose(t, got, []int{2, 2, 4}, want, 1e-6)
}

func referenceCompactMultiheadAttention(q, k, v []float32, mask []int32, batch, tokens, hidden, heads int) []float32 {
	out := make([]float32, batch*tokens*hidden)
	headDim := hidden / heads
	scale := 1 / math.Sqrt(float64(headDim))
	logits := make([]float64, tokens)
	for b := 0; b < batch; b++ {
		for query := 0; query < tokens; query++ {
			queryRow := b*tokens + query
			for head := 0; head < heads; head++ {
				maxLogit := math.Inf(-1)
				active := false
				for key := 0; key < tokens; key++ {
					if mask[b*tokens+key] == 0 {
						continue
					}
					keyRow := b*tokens + key
					dot := 0.0
					for d := 0; d < headDim; d++ {
						col := head*headDim + d
						dot += float64(q[queryRow*hidden+col]) * float64(k[keyRow*hidden+col])
					}
					logit := dot * scale
					logits[key] = logit
					if logit > maxLogit {
						maxLogit = logit
					}
					active = true
				}
				if !active {
					continue
				}
				sum := 0.0
				for key := 0; key < tokens; key++ {
					if mask[b*tokens+key] == 0 {
						continue
					}
					ev := math.Exp(logits[key] - maxLogit)
					logits[key] = ev
					sum += ev
				}
				for key := 0; key < tokens; key++ {
					if mask[b*tokens+key] == 0 {
						continue
					}
					prob := logits[key] / sum
					valueRow := b*tokens + key
					for d := 0; d < headDim; d++ {
						col := head*headDim + d
						out[queryRow*hidden+col] += float32(prob * float64(v[valueRow*hidden+col]))
					}
				}
			}
		}
	}
	return out
}

func assertBackendTensorClose(t *testing.T, got *Tensor, wantShape []int, want []float32, tol float64) {
	t.Helper()
	if got == nil {
		t.Fatal("got nil tensor")
	}
	if len(got.Shape) != len(wantShape) {
		t.Fatalf("shape rank = %d, want %d", len(got.Shape), len(wantShape))
	}
	for i := range wantShape {
		if got.Shape[i] != wantShape[i] {
			t.Fatalf("shape = %v, want %v", got.Shape, wantShape)
		}
	}
	if len(got.F32) != len(want) {
		t.Fatalf("data length = %d, want %d", len(got.F32), len(want))
	}
	for i, value := range want {
		if diff := math.Abs(float64(got.F32[i] - value)); diff > tol {
			t.Fatalf("value[%d] = %.9g, want %.9g (diff %.3g)", i, got.F32[i], value, diff)
		}
	}
}
