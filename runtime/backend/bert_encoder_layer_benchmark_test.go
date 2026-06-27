package backend

import "testing"

func BenchmarkBERTEncoderLayerReferenceBGE128Batch1(b *testing.B) {
	benchmarkBERTEncoderLayerReferenceBGE(b, 1, 128)
}

func BenchmarkBERTEncoderLayerReferenceBGE128Batch16(b *testing.B) {
	benchmarkBERTEncoderLayerReferenceBGE(b, 16, 128)
}

func BenchmarkBERTEncoderLayerReferenceBGE512Batch1(b *testing.B) {
	benchmarkBERTEncoderLayerReferenceBGE(b, 1, 512)
}

func BenchmarkBERTEncoderLayerReferenceBGE512Batch16(b *testing.B) {
	benchmarkBERTEncoderLayerReferenceBGE(b, 16, 512)
}

func benchmarkBERTEncoderLayerReferenceBGE(b *testing.B, batch, tokens int) {
	const (
		hidden = 384
		heads  = 12
		ffn    = 1536
	)
	fixture := bgeLikeEncoderLayerBenchmarkFixture(batch, tokens, hidden, ffn)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := BERTEncoderLayerReference(
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
		if err != nil {
			b.Fatal(err)
		}
		if len(out.F32) != batch*tokens*hidden {
			b.Fatalf("output values = %d, want %d", len(out.F32), batch*tokens*hidden)
		}
	}
}

type bgeLikeEncoderLayerBenchmarkData struct {
	hiddenStates             *Tensor
	attentionMask            *Tensor
	queryWeight              *Tensor
	queryBias                *Tensor
	keyWeight                *Tensor
	keyBias                  *Tensor
	valueWeight              *Tensor
	valueBias                *Tensor
	attentionOutputWeight    *Tensor
	attentionOutputBias      *Tensor
	attentionLayerNormWeight *Tensor
	attentionLayerNormBias   *Tensor
	intermediateWeight       *Tensor
	intermediateBias         *Tensor
	outputWeight             *Tensor
	outputBias               *Tensor
	outputLayerNormWeight    *Tensor
	outputLayerNormBias      *Tensor
}

func bgeLikeEncoderLayerBenchmarkFixture(batch, tokens, hidden, ffn int) bgeLikeEncoderLayerBenchmarkData {
	rows := batch * tokens
	hiddenValues := make([]float32, rows*hidden)
	for i := range hiddenValues {
		hiddenValues[i] = benchmarkValue(i, 97, 0.02)
	}
	mask := make([]int32, rows)
	for i := range mask {
		mask[i] = 1
	}
	return bgeLikeEncoderLayerBenchmarkData{
		hiddenStates:             NewTensorF32([]int{batch, tokens, hidden}, hiddenValues),
		attentionMask:            NewTensorI32([]int{batch, tokens}, mask),
		queryWeight:              benchmarkMatrix("query", hidden, hidden),
		queryBias:                benchmarkVector("query_bias", hidden, 0.001),
		keyWeight:                benchmarkMatrix("key", hidden, hidden),
		keyBias:                  benchmarkVector("key_bias", hidden, 0.001),
		valueWeight:              benchmarkMatrix("value", hidden, hidden),
		valueBias:                benchmarkVector("value_bias", hidden, 0.001),
		attentionOutputWeight:    benchmarkMatrix("attention_output", hidden, hidden),
		attentionOutputBias:      benchmarkVector("attention_output_bias", hidden, 0.001),
		attentionLayerNormWeight: benchmarkFilledVector("attention_ln_weight", hidden, 1),
		attentionLayerNormBias:   benchmarkVector("attention_ln_bias", hidden, 0.001),
		intermediateWeight:       benchmarkMatrix("intermediate", ffn, hidden),
		intermediateBias:         benchmarkVector("intermediate_bias", ffn, 0.001),
		outputWeight:             benchmarkMatrix("output", hidden, ffn),
		outputBias:               benchmarkVector("output_bias", hidden, 0.001),
		outputLayerNormWeight:    benchmarkFilledVector("output_ln_weight", hidden, 1),
		outputLayerNormBias:      benchmarkVector("output_ln_bias", hidden, 0.001),
	}
}

func benchmarkMatrix(name string, rows, cols int) *Tensor {
	values := make([]float32, rows*cols)
	nameOffset := len(name) * 17
	for i := range values {
		values[i] = benchmarkValue(i+nameOffset, 211, 0.015)
	}
	return NewTensorF32([]int{rows, cols}, values)
}

func benchmarkVector(name string, size int, scale float32) *Tensor {
	values := make([]float32, size)
	nameOffset := len(name) * 13
	for i := range values {
		values[i] = benchmarkValue(i+nameOffset, 53, scale)
	}
	return NewTensorF32([]int{size}, values)
}

func benchmarkFilledVector(name string, size int, value float32) *Tensor {
	values := make([]float32, size)
	for i := range values {
		values[i] = value
	}
	return NewTensorF32([]int{size}, values)
}

func benchmarkValue(index, period int, scale float32) float32 {
	if period <= 0 {
		panic("invalid benchmark period")
	}
	centered := float32(index%period) - float32(period/2)
	return centered * scale / float32(period)
}
