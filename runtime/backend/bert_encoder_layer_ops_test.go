package backend

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
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

func TestBERTEncoderLayerProjectorMatchesHostDense(t *testing.T) {
	fixture := bertEncoderLayerFixture()
	want, err := bertEncoderLayerReferenceWithProjector(
		fixture.hiddenStates, fixture.attentionMask,
		fixture.queryWeight, fixture.queryBias,
		fixture.keyWeight, fixture.keyBias,
		fixture.valueWeight, fixture.valueBias,
		fixture.attentionOutputWeight, fixture.attentionOutputBias,
		fixture.attentionLayerNormWeight, fixture.attentionLayerNormBias,
		fixture.intermediateWeight, fixture.intermediateBias,
		fixture.outputWeight, fixture.outputBias,
		fixture.outputLayerNormWeight, fixture.outputLayerNormBias,
		1, 0.25, "gelu", nil,
	)
	if err != nil {
		t.Fatalf("host bert encoder layer: %v", err)
	}
	projector := &recordingBERTDenseProjector{}
	got, err := bertEncoderLayerReferenceWithProjector(
		fixture.hiddenStates, fixture.attentionMask,
		fixture.queryWeight, fixture.queryBias,
		fixture.keyWeight, fixture.keyBias,
		fixture.valueWeight, fixture.valueBias,
		fixture.attentionOutputWeight, fixture.attentionOutputBias,
		fixture.attentionLayerNormWeight, fixture.attentionLayerNormBias,
		fixture.intermediateWeight, fixture.intermediateBias,
		fixture.outputWeight, fixture.outputBias,
		fixture.outputLayerNormWeight, fixture.outputLayerNormBias,
		1, 0.25, "gelu", projector,
	)
	if err != nil {
		t.Fatalf("projected bert encoder layer: %v", err)
	}
	assertTensorClose(t, got, want.Shape, want.F32)
	if projector.manyCalls != 1 {
		t.Fatalf("DenseHFMany calls = %d, want 1", projector.manyCalls)
	}
	if projector.singleCalls != 3 {
		t.Fatalf("DenseHF single calls = %d, want 3", projector.singleCalls)
	}
}

func TestAcceleratedBERTDenseProjectorRebindsMutatedWeight(t *testing.T) {
	accel := newFakeBERTMatMulAccelerator()
	projector := &acceleratedBERTDenseProjector{
		accel:  accel,
		prefix: "test_bert_dense",
		max:    8,
		bound:  map[*Tensor]bertDenseBinding{},
	}
	input := []float32{2, 3}
	weight := NewTensorF32([]int{1, 2}, []float32{1, 1})
	bias := NewTensorF32([]int{1}, []float32{0})

	first, ok := projector.DenseHF(input, 1, 2, weight, bias)
	if !ok {
		t.Fatal("first accelerated dense did not run")
	}
	if len(first) != 1 || first[0] != 5 {
		t.Fatalf("first output = %v, want [5]", first)
	}

	weight.F32[1] = 10
	second, ok := projector.DenseHF(input, 1, 2, weight, bias)
	if !ok {
		t.Fatal("second accelerated dense did not run")
	}
	if len(second) != 1 || second[0] != 32 {
		t.Fatalf("second output = %v, want [32] after rebinding mutated weight", second)
	}
	if accel.bindCalls != 2 {
		t.Fatalf("bind calls = %d, want 2", accel.bindCalls)
	}
	if accel.unbindCalls != 1 {
		t.Fatalf("unbind calls = %d, want 1 stale binding cleanup", accel.unbindCalls)
	}
}

func TestAcceleratedBERTDenseProjectorCleansUpNewBindingsOnFallback(t *testing.T) {
	accel := newFakeBERTMatMulAccelerator()
	projector := &acceleratedBERTDenseProjector{
		accel:  accel,
		prefix: "test_bert_dense",
		max:    8,
		bound:  map[*Tensor]bertDenseBinding{},
	}
	input := []float32{1, 2}
	weights := []*Tensor{
		NewTensorF32([]int{1, 2}, []float32{1, 0}),
		NewTensorF32([]int{1, 2}, []float32{0, 1}),
	}
	biases := []*Tensor{
		NewTensorF32([]int{1}, []float32{0}),
		NewTensorF32([]int{1}, []float32{0}),
	}

	if _, ok := projector.DenseHFMany(input, 1, 2, weights, biases); ok {
		t.Fatal("DenseHFMany unexpectedly accelerated without multi-bound support")
	}
	if accel.bindCalls != 2 {
		t.Fatalf("bind calls = %d, want 2 before fallback", accel.bindCalls)
	}
	if accel.unbindCalls != 2 {
		t.Fatalf("unbind calls = %d, want cleanup of both newly-bound matrices", accel.unbindCalls)
	}
	if len(accel.bound) != 0 {
		t.Fatalf("fake accelerator resident bindings = %d, want 0", len(accel.bound))
	}
	if len(projector.bound) != 0 {
		t.Fatalf("projector bindings = %d, want 0", len(projector.bound))
	}
}

func TestAcceleratedBERTDenseProjectorCleansUpPartialBindFailure(t *testing.T) {
	accel := newFakeBERTMatMulAccelerator()
	accel.failBindAt = 2
	projector := &acceleratedBERTDenseProjector{
		accel:  accel,
		prefix: "test_bert_dense",
		max:    8,
		bound:  map[*Tensor]bertDenseBinding{},
	}
	input := []float32{1, 2}
	weights := []*Tensor{
		NewTensorF32([]int{1, 2}, []float32{1, 0}),
		NewTensorF32([]int{1, 2}, []float32{0, 1}),
	}
	biases := []*Tensor{
		NewTensorF32([]int{1}, []float32{0}),
		NewTensorF32([]int{1}, []float32{0}),
	}

	if _, ok := projector.DenseHFMany(input, 1, 2, weights, biases); ok {
		t.Fatal("DenseHFMany unexpectedly accelerated after bind failure")
	}
	if accel.bindCalls != 2 {
		t.Fatalf("bind calls = %d, want failed second bind attempt", accel.bindCalls)
	}
	if accel.unbindCalls != 1 {
		t.Fatalf("unbind calls = %d, want cleanup of first newly-bound matrix", accel.unbindCalls)
	}
	if len(accel.bound) != 0 {
		t.Fatalf("fake accelerator resident bindings = %d, want 0", len(accel.bound))
	}
	if len(projector.bound) != 0 {
		t.Fatalf("projector bindings = %d, want 0", len(projector.bound))
	}
}

func TestAcceleratedBERTDenseProjectorEvictsOldestBinding(t *testing.T) {
	accel := newFakeBERTMatMulAccelerator()
	projector := &acceleratedBERTDenseProjector{
		accel:  accel,
		prefix: "test_bert_dense",
		max:    2,
		bound:  map[*Tensor]bertDenseBinding{},
	}
	input := []float32{1, 2}
	bias := NewTensorF32([]int{1}, []float32{0})
	weights := []*Tensor{
		NewTensorF32([]int{1, 2}, []float32{1, 0}),
		NewTensorF32([]int{1, 2}, []float32{0, 1}),
		NewTensorF32([]int{1, 2}, []float32{1, 1}),
	}
	for _, weight := range weights {
		if _, ok := projector.DenseHF(input, 1, 2, weight, bias); !ok {
			t.Fatal("accelerated dense did not run")
		}
	}
	if len(projector.bound) != 2 {
		t.Fatalf("projector bindings = %d, want 2", len(projector.bound))
	}
	if len(accel.bound) != 2 {
		t.Fatalf("fake accelerator resident bindings = %d, want 2", len(accel.bound))
	}
	if accel.unbindCalls != 1 {
		t.Fatalf("unbind calls = %d, want one eviction", accel.unbindCalls)
	}
}

func TestBERTSelfAttentionContextParallelMatchesSerial(t *testing.T) {
	const (
		batch  = 2
		tokens = 17
		hidden = 8
		heads  = 2
	)
	headDim := hidden / heads
	values := func(name string) []float32 {
		out := make([]float32, batch*tokens*hidden)
		offset := len(name) * 11
		for i := range out {
			out[i] = benchmarkValue(i+offset, 37, 0.07)
		}
		return out
	}
	mask := make([]int32, batch*tokens)
	for b := 0; b < batch; b++ {
		for i := 0; i < tokens; i++ {
			if i%5 != 3 {
				mask[b*tokens+i] = 1
			}
		}
	}

	q := values("query")
	k := values("key")
	v := values("value")
	want := make([]float32, batch*tokens*hidden)
	bertSelfAttentionContextSerial(want, q, k, v, mask, batch, tokens, hidden, heads, headDim)

	oldProcs := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(oldProcs)
	got := bertSelfAttentionContext(q, k, v, mask, batch, tokens, hidden, heads, headDim)

	if len(got) != len(want) {
		t.Fatalf("context values = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("context[%d] = %.9g, want %.9g", i, got[i], want[i])
		}
	}
}

type fakeBERTMatMulAccelerator struct {
	bound       map[string]*Tensor
	bindCalls   int
	unbindCalls int
	runCalls    int
	failBindAt  int
}

func newFakeBERTMatMulAccelerator() *fakeBERTMatMulAccelerator {
	return &fakeBERTMatMulAccelerator{bound: map[string]*Tensor{}}
}

func (a *fakeBERTMatMulAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *fakeBERTMatMulAccelerator) RunMatMul(inputs []*Tensor, outputType eosartifact.ValueType) (StepDispatchResult, error) {
	return StepDispatchResult{}, fmt.Errorf("RunMatMul is not implemented")
}

func (a *fakeBERTMatMulAccelerator) RunMatMulWithTranspose(inputs []*Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error) {
	return StepDispatchResult{}, fmt.Errorf("RunMatMulWithTranspose is not implemented")
}

func (a *fakeBERTMatMulAccelerator) BindMatrix(name string, tensor *Tensor) error {
	a.bindCalls++
	if a.failBindAt > 0 && a.bindCalls == a.failBindAt {
		return fmt.Errorf("forced bind failure")
	}
	a.bound[name] = NewTensorF32(tensor.Shape, tensor.F32)
	return nil
}

func (a *fakeBERTMatMulAccelerator) UnbindMatrix(name string) error {
	a.unbindCalls++
	delete(a.bound, name)
	return nil
}

func (a *fakeBERTMatMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error) {
	return StepDispatchResult{}, fmt.Errorf("RunMatMulWithBoundLeft is not implemented")
}

func (a *fakeBERTMatMulAccelerator) RunMatMulWithBoundRight(lhs *Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error) {
	a.runCalls++
	weight, ok := a.bound[rightName]
	if !ok {
		return StepDispatchResult{}, fmt.Errorf("missing binding %q", rightName)
	}
	if !transposeRight {
		return StepDispatchResult{}, fmt.Errorf("fake accelerator expects transposeRight")
	}
	rows, in := lhs.Shape[0], lhs.Shape[1]
	out := weight.Shape[0]
	values := make([]float32, rows*out)
	denseHFRange(lhs.F32, values, 0, rows, in, out, weight.F32, make([]float32, out))
	return StepDispatchResult{Outputs: []*Tensor{NewTensorF32([]int{rows, out}, values)}}, nil
}

func (a *fakeBERTMatMulAccelerator) Stats() MatMulAcceleratorStats {
	return MatMulAcceleratorStats{BindCalls: int64(a.bindCalls), BoundMatrices: int64(len(a.bound)), RunCalls: int64(a.runCalls)}
}

func (a *fakeBERTMatMulAccelerator) Close() {}

type recordingBERTDenseProjector struct {
	singleCalls int
	manyCalls   int
}

func (p *recordingBERTDenseProjector) DenseHF(input []float32, rows, in int, weight, bias *Tensor) ([]float32, bool) {
	p.singleCalls++
	return denseHF(input, rows, in, weight, bias), true
}

func (p *recordingBERTDenseProjector) DenseHFMany(input []float32, rows, in int, weights, biases []*Tensor) ([][]float32, bool) {
	p.manyCalls++
	out := make([][]float32, len(weights))
	for i := range weights {
		out[i] = denseHF(input, rows, in, weights[i], biases[i])
	}
	return out, true
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
