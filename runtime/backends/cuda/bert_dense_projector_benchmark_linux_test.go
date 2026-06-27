//go:build linux

package cuda

import (
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func BenchmarkCUDABERTDenseProjectorBGE128Batch16(b *testing.B) {
	benchmarkCUDABERTDenseProjectorBGE(b, 16, 128)
}

func BenchmarkCUDABERTDenseProjectorBGE512Batch1(b *testing.B) {
	benchmarkCUDABERTDenseProjectorBGE(b, 1, 512)
}

func benchmarkCUDABERTDenseProjectorBGE(b *testing.B, batch, tokens int) {
	const (
		hidden = 384
		heads  = 12
		ffn    = 1536
	)
	fixture := cudaBGELikeEncoderLayerBenchmarkFixture(batch, tokens, hidden, ffn)
	out, err := runCUDABERTDenseProjectorBenchmarkLayer(fixture, heads)
	if err != nil {
		b.Fatal(err)
	}
	if len(out.F32) != batch*tokens*hidden {
		b.Fatalf("warmup output values = %d, want %d", len(out.F32), batch*tokens*hidden)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := runCUDABERTDenseProjectorBenchmarkLayer(fixture, heads)
		if err != nil {
			b.Fatal(err)
		}
		if len(out.F32) != batch*tokens*hidden {
			b.Fatalf("output values = %d, want %d", len(out.F32), batch*tokens*hidden)
		}
	}
}

func runCUDABERTDenseProjectorBenchmarkLayer(fixture cudaBGELikeEncoderLayerBenchmarkData, heads int) (*backend.Tensor, error) {
	return backend.BERTEncoderLayerReference(
		fixture.hiddenStates, fixture.attentionMask,
		fixture.queryWeight, fixture.queryBias,
		fixture.keyWeight, fixture.keyBias,
		fixture.valueWeight, fixture.valueBias,
		fixture.attentionOutputWeight, fixture.attentionOutputBias,
		fixture.attentionLayerNormWeight, fixture.attentionLayerNormBias,
		fixture.intermediateWeight, fixture.intermediateBias,
		fixture.outputWeight, fixture.outputBias,
		fixture.outputLayerNormWeight, fixture.outputLayerNormBias,
		heads, 1e-12, "gelu",
	)
}

type cudaBGELikeEncoderLayerBenchmarkData struct {
	hiddenStates             *backend.Tensor
	attentionMask            *backend.Tensor
	queryWeight              *backend.Tensor
	queryBias                *backend.Tensor
	keyWeight                *backend.Tensor
	keyBias                  *backend.Tensor
	valueWeight              *backend.Tensor
	valueBias                *backend.Tensor
	attentionOutputWeight    *backend.Tensor
	attentionOutputBias      *backend.Tensor
	attentionLayerNormWeight *backend.Tensor
	attentionLayerNormBias   *backend.Tensor
	intermediateWeight       *backend.Tensor
	intermediateBias         *backend.Tensor
	outputWeight             *backend.Tensor
	outputBias               *backend.Tensor
	outputLayerNormWeight    *backend.Tensor
	outputLayerNormBias      *backend.Tensor
}

func cudaBGELikeEncoderLayerBenchmarkFixture(batch, tokens, hidden, ffn int) cudaBGELikeEncoderLayerBenchmarkData {
	rows := batch * tokens
	hiddenValues := make([]float32, rows*hidden)
	for i := range hiddenValues {
		hiddenValues[i] = cudaBenchmarkValue(i, 97, 0.02)
	}
	mask := make([]int32, rows)
	for i := range mask {
		mask[i] = 1
	}
	return cudaBGELikeEncoderLayerBenchmarkData{
		hiddenStates:             backend.NewTensorF32([]int{batch, tokens, hidden}, hiddenValues),
		attentionMask:            backend.NewTensorI32([]int{batch, tokens}, mask),
		queryWeight:              cudaBenchmarkMatrix("query", hidden, hidden),
		queryBias:                cudaBenchmarkVector("query_bias", hidden, 0.001),
		keyWeight:                cudaBenchmarkMatrix("key", hidden, hidden),
		keyBias:                  cudaBenchmarkVector("key_bias", hidden, 0.001),
		valueWeight:              cudaBenchmarkMatrix("value", hidden, hidden),
		valueBias:                cudaBenchmarkVector("value_bias", hidden, 0.001),
		attentionOutputWeight:    cudaBenchmarkMatrix("attention_output", hidden, hidden),
		attentionOutputBias:      cudaBenchmarkVector("attention_output_bias", hidden, 0.001),
		attentionLayerNormWeight: cudaBenchmarkFilledVector(hidden, 1),
		attentionLayerNormBias:   cudaBenchmarkVector("attention_ln_bias", hidden, 0.001),
		intermediateWeight:       cudaBenchmarkMatrix("intermediate", ffn, hidden),
		intermediateBias:         cudaBenchmarkVector("intermediate_bias", ffn, 0.001),
		outputWeight:             cudaBenchmarkMatrix("output", hidden, ffn),
		outputBias:               cudaBenchmarkVector("output_bias", hidden, 0.001),
		outputLayerNormWeight:    cudaBenchmarkFilledVector(hidden, 1),
		outputLayerNormBias:      cudaBenchmarkVector("output_ln_bias", hidden, 0.001),
	}
}

func cudaBenchmarkMatrix(name string, rows, cols int) *backend.Tensor {
	values := make([]float32, rows*cols)
	nameOffset := len(name) * 17
	for i := range values {
		values[i] = cudaBenchmarkValue(i+nameOffset, 211, 0.015)
	}
	return backend.NewTensorF32([]int{rows, cols}, values)
}

func cudaBenchmarkVector(name string, size int, scale float32) *backend.Tensor {
	values := make([]float32, size)
	nameOffset := len(name) * 13
	for i := range values {
		values[i] = cudaBenchmarkValue(i+nameOffset, 53, scale)
	}
	return backend.NewTensorF32([]int{size}, values)
}

func cudaBenchmarkFilledVector(size int, value float32) *backend.Tensor {
	values := make([]float32, size)
	for i := range values {
		values[i] = value
	}
	return backend.NewTensorF32([]int{size}, values)
}

func cudaBenchmarkValue(index, period int, scale float32) float32 {
	centered := float32(index%period) - float32(period/2)
	return centered * scale / float32(period)
}
