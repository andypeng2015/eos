package backend

import (
	"math"
	"strings"
	"testing"
)

func TestBERTEncoderLayerReferenceTinyDeterministic(t *testing.T) {
	fixture := bertEncoderLayerFixture()
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
		1, 0.25, "gelu",
	)
	if err != nil {
		t.Fatalf("bert encoder layer: %v", err)
	}
	want := manualBERTEncoderLayer(t, fixture, 1, 0.25)
	assertTensorClose(t, out, []int{1, 2, 2}, want)
}

func TestBERTEncoderLayerReferenceMaskAllKeysProducesZeroContext(t *testing.T) {
	fixture := bertEncoderLayerFixture()
	fixture.attentionMask = NewTensorI32([]int{1, 2}, []int32{0, 0})
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
		1, 0.25, "",
	)
	if err != nil {
		t.Fatalf("bert encoder layer all masked: %v", err)
	}
	want := manualBERTEncoderLayer(t, fixture, 1, 0.25)
	assertTensorClose(t, out, []int{1, 2, 2}, want)
	for i, value := range out.F32 {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("out[%d] = %v, want finite", i, value)
		}
	}
}

func TestBERTEncoderLayerReferenceValidationFailures(t *testing.T) {
	valid := func() bertEncoderFixture {
		return bertEncoderLayerFixture()
	}
	tests := []struct {
		name string
		edit func(*bertEncoderFixture)
		want string
	}{
		{
			name: "hidden not divisible by heads",
			edit: func(f *bertEncoderFixture) {},
			want: "divisible",
		},
		{
			name: "bad mask dtype",
			edit: func(f *bertEncoderFixture) {
				f.attentionMask = NewTensorI64([]int{1, 2}, []int64{1, 1})
			},
			want: `attention_mask dtype "i64" is not i32`,
		},
		{
			name: "bad dense shape",
			edit: func(f *bertEncoderFixture) {
				f.queryWeight = NewTensorF32([]int{2, 3}, make([]float32, 6))
			},
			want: "query weight shape",
		},
		{
			name: "unsupported activation",
			edit: func(f *bertEncoderFixture) {},
			want: "unsupported hidden_act",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := valid()
			tc.edit(&f)
			heads := 1
			act := "gelu"
			if tc.name == "hidden not divisible by heads" {
				heads = 3
			}
			if tc.name == "unsupported activation" {
				act = "relu"
			}
			_, err := BERTEncoderLayerReference(
				f.hiddenStates, f.attentionMask,
				f.queryWeight, f.queryBias,
				f.keyWeight, f.keyBias,
				f.valueWeight, f.valueBias,
				f.attentionOutputWeight, f.attentionOutputBias,
				f.attentionLayerNormWeight, f.attentionLayerNormBias,
				f.intermediateWeight, f.intermediateBias,
				f.outputWeight, f.outputBias,
				f.outputLayerNormWeight, f.outputLayerNormBias,
				heads, 0.25, act,
			)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestBERTEncoderLayerStepRejectsBadAttributes(t *testing.T) {
	f := bertEncoderLayerFixture()
	inputs := []*Tensor{
		f.hiddenStates, f.attentionMask,
		f.queryWeight, f.queryBias,
		f.keyWeight, f.keyBias,
		f.valueWeight, f.valueBias,
		f.attentionOutputWeight, f.attentionOutputBias,
		f.attentionLayerNormWeight, f.attentionLayerNormBias,
		f.intermediateWeight, f.intermediateBias,
		f.outputWeight, f.outputBias,
		f.outputLayerNormWeight, f.outputLayerNormBias,
	}
	_, err := bertEncoderLayerTensor(inputs, tensorValueType("f32", []string{"B", "T", "2"}), map[string]string{"num_attention_heads": "nope"})
	if err == nil || !strings.Contains(err.Error(), "num_attention_heads") {
		t.Fatalf("error = %v, want num_attention_heads parse error", err)
	}
}

type bertEncoderFixture struct {
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

func bertEncoderLayerFixture() bertEncoderFixture {
	return bertEncoderFixture{
		hiddenStates:             NewTensorF32([]int{1, 2, 2}, []float32{2, -1, 1, 3}),
		attentionMask:            NewTensorI32([]int{1, 2}, []int32{1, 0}),
		queryWeight:              NewTensorF32([]int{2, 2}, []float32{2, -1, 0.5, 3}),
		queryBias:                NewTensorF32([]int{2}, []float32{0.25, -0.5}),
		keyWeight:                NewTensorF32([]int{2, 2}, []float32{1, 4, -2, 0.5}),
		keyBias:                  NewTensorF32([]int{2}, []float32{-0.25, 0.75}),
		valueWeight:              NewTensorF32([]int{2, 2}, []float32{1.5, -0.25, 0.75, 2}),
		valueBias:                NewTensorF32([]int{2}, []float32{0.1, -0.2}),
		attentionOutputWeight:    NewTensorF32([]int{2, 2}, []float32{0.5, -1.25, 1.5, 0.25}),
		attentionOutputBias:      NewTensorF32([]int{2}, []float32{0.3, -0.4}),
		attentionLayerNormWeight: NewTensorF32([]int{2}, []float32{1.2, -0.7}),
		attentionLayerNormBias:   NewTensorF32([]int{2}, []float32{0.05, 0.15}),
		intermediateWeight:       NewTensorF32([]int{3, 2}, []float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5}),
		intermediateBias:         NewTensorF32([]int{3}, []float32{0.1, -0.2, 0.3}),
		outputWeight:             NewTensorF32([]int{2, 3}, []float32{0.4, -0.8, 1.2, -1.1, 0.6, 0.2}),
		outputBias:               NewTensorF32([]int{2}, []float32{-0.05, 0.25}),
		outputLayerNormWeight:    NewTensorF32([]int{2}, []float32{0.9, 1.4}),
		outputLayerNormBias:      NewTensorF32([]int{2}, []float32{-0.2, 0.4}),
	}
}

func manualBERTEncoderLayer(t *testing.T, f bertEncoderFixture, heads int, epsilon float64) []float32 {
	t.Helper()
	b, tokens, hidden := f.hiddenStates.Shape[0], f.hiddenStates.Shape[1], f.hiddenStates.Shape[2]
	rows := b * tokens
	q := manualDenseHF(f.hiddenStates.F32, rows, hidden, f.queryWeight, f.queryBias)
	k := manualDenseHF(f.hiddenStates.F32, rows, hidden, f.keyWeight, f.keyBias)
	v := manualDenseHF(f.hiddenStates.F32, rows, hidden, f.valueWeight, f.valueBias)
	headDim := hidden / heads
	context := make([]float32, rows*hidden)
	for batch := 0; batch < b; batch++ {
		for i := 0; i < tokens; i++ {
			qrow := batch*tokens + i
			for head := 0; head < heads; head++ {
				logits := make([]float64, tokens)
				maxLogit := math.Inf(-1)
				active := 0
				for j := 0; j < tokens; j++ {
					if f.attentionMask.I32[batch*tokens+j] == 0 {
						continue
					}
					active++
					krow := batch*tokens + j
					dot := 0.0
					for d := 0; d < headDim; d++ {
						idx := head*headDim + d
						dot += float64(q[qrow*hidden+idx]) * float64(k[krow*hidden+idx])
					}
					logits[j] = dot / math.Sqrt(float64(headDim))
					maxLogit = math.Max(maxLogit, logits[j])
				}
				if active == 0 {
					continue
				}
				sum := 0.0
				for j := 0; j < tokens; j++ {
					if f.attentionMask.I32[batch*tokens+j] == 0 {
						continue
					}
					logits[j] = math.Exp(logits[j] - maxLogit)
					sum += logits[j]
				}
				for j := 0; j < tokens; j++ {
					if f.attentionMask.I32[batch*tokens+j] == 0 {
						continue
					}
					vrow := batch*tokens + j
					prob := logits[j] / sum
					for d := 0; d < headDim; d++ {
						idx := head*headDim + d
						context[qrow*hidden+idx] += float32(prob * float64(v[vrow*hidden+idx]))
					}
				}
			}
		}
	}
	attnProjected := manualDenseHF(context, rows, hidden, f.attentionOutputWeight, f.attentionOutputBias)
	attnResidual := make([]float32, len(attnProjected))
	for i := range attnResidual {
		attnResidual[i] = attnProjected[i] + f.hiddenStates.F32[i]
	}
	attnLayer := manualLayerNormAffine(attnResidual, rows, hidden, f.attentionLayerNormWeight.F32, f.attentionLayerNormBias.F32, epsilon)
	intermediate := manualDenseHF(attnLayer, rows, hidden, f.intermediateWeight, f.intermediateBias)
	for i, x := range intermediate {
		intermediate[i] = 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
	}
	projected := manualDenseHF(intermediate, rows, f.intermediateWeight.Shape[0], f.outputWeight, f.outputBias)
	outputResidual := make([]float32, len(projected))
	for i := range outputResidual {
		outputResidual[i] = projected[i] + attnLayer[i]
	}
	return manualLayerNormAffine(outputResidual, rows, hidden, f.outputLayerNormWeight.F32, f.outputLayerNormBias.F32, epsilon)
}

func manualDenseHF(input []float32, rows, in int, weight, bias *Tensor) []float32 {
	out := weight.Shape[0]
	result := make([]float32, rows*out)
	for r := 0; r < rows; r++ {
		for o := 0; o < out; o++ {
			sum := float64(bias.F32[o])
			for c := 0; c < in; c++ {
				sum += float64(input[r*in+c]) * float64(weight.F32[o*in+c])
			}
			result[r*out+o] = float32(sum)
		}
	}
	return result
}

func manualLayerNormAffine(input []float32, rows, dim int, gamma, beta []float32, epsilon float64) []float32 {
	out := make([]float32, len(input))
	for r := 0; r < rows; r++ {
		base := r * dim
		mean := 0.0
		for d := 0; d < dim; d++ {
			mean += float64(input[base+d])
		}
		mean /= float64(dim)
		variance := 0.0
		for d := 0; d < dim; d++ {
			centered := float64(input[base+d]) - mean
			variance += centered * centered
		}
		variance /= float64(dim)
		invStd := 1 / math.Sqrt(variance+epsilon)
		for d := 0; d < dim; d++ {
			out[base+d] = float32((float64(input[base+d])-mean)*invStd*float64(gamma[d]) + float64(beta[d]))
		}
	}
	return out
}
