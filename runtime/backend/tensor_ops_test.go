package backend

import (
	"context"
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

func TestMaterializeValueTensor(t *testing.T) {
	value, err := MaterializeValue(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		NewTensorF16([]int{2, 2}, []float32{1, 2, 3, 4}),
	)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	tensor, ok := value.(*Tensor)
	if !ok {
		t.Fatalf("value type = %T, want *Tensor", value)
	}
	if tensor.DType != "f16" || tensor.Shape[0] != 2 || tensor.Shape[1] != 2 {
		t.Fatalf("unexpected tensor: %+v", tensor)
	}
}

func TestMaterializeValueTensorI64(t *testing.T) {
	value, err := MaterializeValue(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "i64", Shape: []string{"T"}}},
		NewTensorI64([]int{3}, []int64{101, 202, 303}),
	)
	if err != nil {
		t.Fatalf("materialize i64: %v", err)
	}
	tensor, ok := value.(*Tensor)
	if !ok {
		t.Fatalf("value type = %T, want *Tensor", value)
	}
	if tensor.DType != "i64" || len(tensor.I64) != 3 {
		t.Fatalf("unexpected i64 tensor: %+v", tensor)
	}
}

func TestMaterializeValueCandidatePack(t *testing.T) {
	value, err := MaterializeValue(
		mantaartifact.ValueType{Kind: mantaartifact.ValueCandidatePack, CandidatePack: &mantaartifact.CandidatePackType{Shape: []string{"K", "D"}}},
		NewCandidatePack(
			NewTensorI64([]int{2}, []int64{1001, 3003}),
			NewTensorF32([]int{2}, []float32{1, 0.70710677}),
			NewTensorQ4([]int{2, 2}, []float32{1, 0, 1, 1}),
		),
	)
	if err != nil {
		t.Fatalf("materialize candidate pack: %v", err)
	}
	pack, ok := value.(*CandidatePack)
	if !ok {
		t.Fatalf("value type = %T, want *CandidatePack", value)
	}
	assertTensorI64(t, pack.IDs, []int{2}, []int64{1001, 3003})
	assertTensorClose(t, pack.Scores, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, pack.Docs, []int{2, 2}, []float32{1, 0, 1, 1})
}

func TestMaterializeValueWithBindingsBindsSymbols(t *testing.T) {
	bindings := map[string]int{}
	_, concrete, err := MaterializeValueWithBindings(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		NewTensorF16([]int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8}),
		bindings,
	)
	if err != nil {
		t.Fatalf("materialize with bindings: %v", err)
	}
	if bindings["T"] != 2 || bindings["D"] != 4 {
		t.Fatalf("unexpected bindings: %+v", bindings)
	}
	if got := concrete.Tensor.Shape[0] + "," + concrete.Tensor.Shape[1]; got != "2,4" {
		t.Fatalf("unexpected concrete shape: %v", concrete.Tensor.Shape)
	}
}

func TestMaterializeValueWithBindingsRejectsMismatch(t *testing.T) {
	bindings := map[string]int{"D": 2}
	_, _, err := MaterializeValueWithBindings(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		NewTensorF16([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		bindings,
	)
	if err == nil {
		t.Fatal("expected mismatch error")
	}
}

func TestPreviewValueWithBindingsReusesTensorPointer(t *testing.T) {
	bindings := map[string]int{}
	input := NewTensorF16([]int{2, 2}, []float32{1, 2, 3, 4})
	value, concrete, err := PreviewValueWithBindings(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		input,
		bindings,
	)
	if err != nil {
		t.Fatalf("preview with bindings: %v", err)
	}
	tensor, ok := value.(*Tensor)
	if !ok {
		t.Fatalf("preview type = %T, want *Tensor", value)
	}
	if tensor != input {
		t.Fatal("preview cloned tensor pointer")
	}
	if bindings["T"] != 2 || bindings["D"] != 2 {
		t.Fatalf("unexpected bindings after preview: %+v", bindings)
	}
	if got := concrete.Tensor.Shape[0] + "," + concrete.Tensor.Shape[1]; got != "2,2" {
		t.Fatalf("unexpected concrete shape: %v", concrete.Tensor.Shape)
	}
}

func TestGatherTensor(t *testing.T) {
	table := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		1, 1,
		1, 0,
	})
}

func TestMatmulTensor(t *testing.T) {
	lhs := NewTensorF16([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	rhs := NewTensorF16([]int{2, 2}, []float32{
		5, 6,
		7, 8,
	})
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		19, 22,
		43, 50,
	})
}

func TestMatmulTensorBatched(t *testing.T) {
	lhs := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	rhs := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		t.Fatalf("batched matmul: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
}

func TestMatmulTensorBatchedRHS(t *testing.T) {
	lhs := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 0,
		0, 1,
		1, 2,
		3, 4,
	})
	rhs := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		2, 0,
		1, 2,
	})
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		t.Fatalf("batched rhs matmul: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		4, 4,
		10, 8,
	})
}

func TestExecuteSymbolicDispatchesImageStep(t *testing.T) {
	mod := mantaartifact.NewModule("image_dispatch")
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "conv",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "x", Type: tensorValueType("f16", []string{"1", "1", "2", "2"})},
			{Name: "w", Type: tensorValueType("f16", []string{"1", "1", "2", "2"})},
			{Name: "b", Type: tensorValueType("f16", []string{"1"})},
		},
		Outputs: []mantaartifact.ValueBinding{
			{Name: "y", Type: tensorValueType("f16", []string{"1", "1", "1", "1"})},
		},
	}}
	mod.Buffers = []mantaartifact.Buffer{{Name: "y", DType: "f16", Shape: []string{"1", "1", "1", "1"}}}
	mod.Steps = []mantaartifact.Step{
		{Entry: "conv", Kind: mantaartifact.StepConv2D, Name: "conv2d", Inputs: []string{"x", "w", "b"}, Outputs: []string{"y"}},
		{Entry: "conv", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"y"}},
	}
	if err := mod.Validate(); err != nil {
		t.Fatal(err)
	}
	dispatch := func(_ context.Context, step mantaartifact.Step, outputType mantaartifact.ValueType, inputs []*Tensor) (StepDispatchResult, bool, error) {
		if step.Kind != mantaartifact.StepConv2D {
			return StepDispatchResult{}, false, nil
		}
		if len(inputs) != 3 {
			t.Fatalf("dispatch inputs = %d want 3", len(inputs))
		}
		if outputType.Tensor == nil || outputType.Tensor.DType != "f16" {
			t.Fatalf("unexpected output type: %+v", outputType)
		}
		return StepDispatchResult{
			Outputs:      []*Tensor{NewTensorF16([]int{1, 1, 1, 1}, []float32{42})},
			VariantEntry: "cuda_conv2d_test",
			Metadata:     map[string]any{"dispatch_mode": "test_device"},
		}, true, nil
	}
	result, err := ExecuteSymbolic(context.Background(), mod, nil, nil, nil, dispatch, mantaartifact.BackendCUDA, Request{
		Entry: "conv",
		Inputs: map[string]any{
			"x": NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4}),
			"w": NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 0, 0, 1}),
			"b": NewTensorF16([]int{1}, []float32{0}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	out := result.Outputs["y"].Data.(*Tensor)
	if out.F32[0] != 42 {
		t.Fatalf("dispatched output = %v", out.F32)
	}
	if got := result.Outputs["y"].Metadata["dispatch_mode"]; got != "test_device" {
		t.Fatalf("dispatch metadata = %v", got)
	}
	if len(result.Trace) == 0 || result.Trace[0].Variant != "cuda_conv2d_test" {
		t.Fatalf("trace variant not recorded: %+v", result.Trace)
	}
}

func TestExecuteSymbolicDispatchesMultiOutputImageStep(t *testing.T) {
	mod := mantaartifact.NewModule("turboquant_dispatch")
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "quantize",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "y", Type: tensorValueType("f16", []string{"1", "2", "1", "1"})},
		},
		Outputs: []mantaartifact.ValueBinding{
			{Name: "coords", Type: tensorValueType("q2", []string{"1", "2", "1", "1"})},
			{Name: "norms", Type: tensorValueType("q_norm", []string{"1", "1", "1"})},
		},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "coords", DType: "q2", Shape: []string{"1", "2", "1", "1"}},
		{Name: "norms", DType: "q_norm", Shape: []string{"1", "1", "1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "quantize", Kind: mantaartifact.StepTurboQEncode, Name: "quantize", Inputs: []string{"y"}, Outputs: []string{"coords", "norms"}, Attributes: map[string]string{"bits": "2"}},
		{Entry: "quantize", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"coords", "norms"}},
	}
	if err := mod.Validate(); err != nil {
		t.Fatal(err)
	}
	dispatch := func(_ context.Context, step mantaartifact.Step, _ mantaartifact.ValueType, inputs []*Tensor) (StepDispatchResult, bool, error) {
		if step.Kind != mantaartifact.StepTurboQEncode {
			return StepDispatchResult{}, false, nil
		}
		if len(inputs) != 1 {
			t.Fatalf("dispatch inputs = %d want 1", len(inputs))
		}
		return StepDispatchResult{
			Outputs: []*Tensor{
				NewTensorQ2([]int{1, 2, 1, 1}, []float32{1, 2}),
				NewTensorQNorm([]int{1, 1, 1}, []float32{128}),
			},
			VariantEntry: "cuda_turboquant_encode_test",
			Metadata:     map[string]any{"dispatch_mode": "test_device"},
		}, true, nil
	}
	result, err := ExecuteSymbolic(context.Background(), mod, nil, nil, nil, dispatch, mantaartifact.BackendCUDA, Request{
		Entry:  "quantize",
		Inputs: map[string]any{"y": NewTensorF16([]int{1, 2, 1, 1}, []float32{1, 2})},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Outputs["coords"].Data.(*Tensor).F32; got[0] != 1 || got[1] != 2 {
		t.Fatalf("coords = %v", got)
	}
	if got := result.Outputs["norms"].Data.(*Tensor).F32[0]; got != 128 {
		t.Fatalf("norm = %v", got)
	}
}

func TestTransposeTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	out, err := transposeTensor(in)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	assertTensorClose(t, out, []int{3, 2}, []float32{
		1, 4,
		2, 5,
		3, 6,
	})
}

func TestTransposeTensorBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	out, err := transposeTensor(in)
	if err != nil {
		t.Fatalf("batched transpose: %v", err)
	}
	assertTensorClose(t, out, []int{2, 3, 2}, []float32{
		1, 4,
		2, 5,
		3, 6,
		7, 10,
		8, 11,
		9, 12,
	})
}

func TestNormalizeRows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		3, 4,
		1, 0,
	})
	out := normalizeRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.6, 0.8,
		1, 0,
	})
}

func TestNormalizeRowsBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		3, 4,
		1, 0,
		0, 5,
		1, 1,
	})
	out := normalizeRows(in)
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		0.6, 0.8,
		1, 0,
		0, 1,
		0.70710677, 0.70710677,
	})
}

func TestLayerNormRows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		2, 4,
		6, 8,
	})
	out := layerNormRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		-0.999995, 0.999995,
		-0.999995, 0.999995,
	})
}

func TestLayerNormRowsBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		2, 4,
		6, 8,
		1, 3,
		5, 7,
	})
	out := layerNormRows(in)
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		-0.999995, 0.999995,
		-0.999995, 0.999995,
		-0.999995, 0.999995,
		-0.999995, 0.999995,
	})
}

func TestSoftmaxRowsBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		0, 1,
		1, 0,
		2, 0,
		0, 2,
	})
	out := softmaxRows(in)
	e := float32(math.Exp(1))
	e2 := float32(math.Exp(2))
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1 / (1 + e), e / (1 + e),
		e / (1 + e), 1 / (1 + e),
		e2 / (1 + e2), 1 / (1 + e2),
		1 / (1 + e2), e2 / (1 + e2),
	})
}

func TestMeanPoolTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	out, err := meanPoolTensor(in)
	if err != nil {
		t.Fatalf("mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestMeanPoolTensorBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
		0, 1,
		1, 0,
	})
	out, err := meanPoolTensor(in)
	if err != nil {
		t.Fatalf("batched mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0.5, 0.5,
	})
}

func TestMeanPoolMaskedTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	mask := NewTensorI32([]int{2}, []int32{1, 0})
	out, err := meanPoolMaskedTensor(in, mask)
	if err != nil {
		t.Fatalf("masked mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2}, []float32{
		1, 0,
	})
}

func TestMeanPoolMaskedTensorBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
		0, 1,
		1, 0,
	})
	mask := NewTensorI32([]int{2, 2}, []int32{
		1, 1,
		1, 0,
	})
	out, err := meanPoolMaskedTensor(in, mask)
	if err != nil {
		t.Fatalf("batched masked mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0, 1,
	})
}

func TestSoftmaxRows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		0, 0,
		1, 0,
	})
	out := softmaxRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.5, 0.5,
		0.7310586, 0.26894143,
	})
}

func TestGELUTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		-1, 0,
		1, 2,
	})
	out := geluTensor(in)
	want := make([]float32, len(in.F32))
	for i, x := range in.F32 {
		want[i] = approxGELU(x)
	}
	assertTensorClose(t, out, []int{2, 2}, want)
}

func TestRoPERows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	out := ropeRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		1, 0,
		-0.84147096, 0.5403023,
	})
}

func TestDotRows(t *testing.T) {
	query := NewTensorF16([]int{2}, []float32{1, 0})
	docs := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out, err := dotRows(query, docs)
	if err != nil {
		t.Fatalf("dot: %v", err)
	}
	assertTensorClose(t, out, []int{3}, []float32{1, 0, 1})
}

func TestCosineRows(t *testing.T) {
	query := NewTensorF16([]int{2}, []float32{1, 0})
	docs := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out, err := cosineRows(query, docs)
	if err != nil {
		t.Fatalf("cosine: %v", err)
	}
	assertTensorClose(t, out, []int{3}, []float32{1, 0, 0.70710677})
}

func TestCosineRowsBatched(t *testing.T) {
	query := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	docs := NewTensorQ4([]int{2, 3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
		0, 1,
		1, 0,
		1, 1,
	})
	out, err := cosineRows(query, docs)
	if err != nil {
		t.Fatalf("batched cosine: %v", err)
	}
	assertTensorClose(t, out, []int{2, 3}, []float32{
		1, 0, 0.70710677,
		1, 0, 0.70710677,
	})
}

func TestL2DistanceRows(t *testing.T) {
	query := NewTensorF16([]int{2}, []float32{1, 0})
	docs := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out, err := l2DistanceRows(query, docs)
	if err != nil {
		t.Fatalf("l2_distance: %v", err)
	}
	assertTensorClose(t, out, []int{3}, []float32{0, 1.4142135, 1})
}

func TestNewTensorQ4(t *testing.T) {
	tensor := NewTensorQ4([]int{2, 2}, []float32{1, 0, 0, 1})
	if tensor.DType != "q4" {
		t.Fatalf("dtype = %q, want q4", tensor.DType)
	}
	if len(tensor.F32) != 4 {
		t.Fatalf("len(F32) = %d, want 4", len(tensor.F32))
	}
}

func TestTopKTensor(t *testing.T) {
	in := NewTensorF32([]int{4}, []float32{0.25, 0.9, 0.5, 0.9})
	out, err := topKTensor(in, 3)
	if err != nil {
		t.Fatalf("topk: %v", err)
	}
	assertTensorI32(t, out, []int{3}, []int32{1, 3, 2})
}

func TestTopKTensorRank2(t *testing.T) {
	in := NewTensorF32([]int{2, 3}, []float32{
		0.25, 0.9, 0.5,
		0.9, 0.1, 0.9,
	})
	out, err := topKTensor(in, 2)
	if err != nil {
		t.Fatalf("topk rank2: %v", err)
	}
	assertTensorI32(t, out, []int{2, 2}, []int32{
		1, 2,
		0, 2,
	})
}

func TestGatherTensorRank1(t *testing.T) {
	table := NewTensorF32([]int{4}, []float32{0.25, 0.9, 0.5, 0.9})
	indices := NewTensorI32([]int{2}, []int32{3, 1})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank1: %v", err)
	}
	assertTensorClose(t, out, []int{2}, []float32{0.9, 0.9})
}

func TestGatherTensorRank1I32(t *testing.T) {
	table := NewTensorI32([]int{4}, []int32{101, 202, 303, 404})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank1 i32: %v", err)
	}
	assertTensorI32(t, out, []int{2}, []int32{303, 101})
}

func TestGatherTensorRank1I64(t *testing.T) {
	table := NewTensorI64([]int{4}, []int64{1001, 2002, 3003, 4004})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank1 i64: %v", err)
	}
	assertTensorI64(t, out, []int{2}, []int64{3003, 1001})
}

func TestGatherTensorRank2Q4(t *testing.T) {
	table := NewTensorQ4([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank2 q4: %v", err)
	}
	if out.DType != "q4" {
		t.Fatalf("dtype = %q, want q4", out.DType)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		1, 1,
		1, 0,
	})
}

func TestGatherTensorRank2WithRank2Indices(t *testing.T) {
	table := NewTensorI64([]int{2, 3}, []int64{
		1001, 2002, 3003,
		4004, 5005, 6006,
	})
	indices := NewTensorI32([]int{2, 2}, []int32{
		0, 2,
		0, 2,
	})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank2/rank2: %v", err)
	}
	assertTensorI64(t, out, []int{2, 2}, []int64{
		1001, 3003,
		4004, 6006,
	})
}

func TestGatherTensorRank2SharedWithRank2Indices(t *testing.T) {
	table := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	indices := NewTensorI32([]int{2, 2}, []int32{
		0, 2,
		1, 0,
	})
	out, err := gatherTensor(table, indices, mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f16",
			Shape: []string{"B", "T", "D"},
		},
	})
	if err != nil {
		t.Fatalf("gather shared rank2/rank2: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 0,
	})
}

func TestGatherTensorRank3Q4WithRank2Indices(t *testing.T) {
	table := NewTensorQ4([]int{2, 3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
		0, 1,
		1, 0,
		1, 1,
	})
	indices := NewTensorI32([]int{2, 2}, []int32{
		0, 2,
		0, 2,
	})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank3/rank2 q4: %v", err)
	}
	if out.DType != "q4" {
		t.Fatalf("dtype = %q, want q4", out.DType)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 1,
	})
}

func TestSparseAttentionReferenceSelectsTopKValues(t *testing.T) {
	query := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	key := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		-1, 0,
	})
	value := NewTensorF16([]int{3, 2}, []float32{
		10, 0,
		0, 20,
		-10, 0,
	})
	out, err := SparseAttentionReference(query, key, value, map[string]string{"top_k": "1"})
	if err != nil {
		t.Fatal(err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		10, 0,
		0, 20,
	})
}

func TestSparseAttentionReferenceRoutesThroughSelectedBlocks(t *testing.T) {
	query := NewTensorF16([]int{1, 2}, []float32{1, 0})
	key := NewTensorF16([]int{6, 2}, []float32{
		0, 10,
		0, 0,
		3, 0,
		2, 0,
		10, 0,
		0, 0,
	})
	value := NewTensorF16([]int{6, 1}, []float32{10, 20, 30, 40, 50, 60})
	out, err := SparseAttentionReference(query, key, value, map[string]string{
		"top_k":            "1",
		"route_block_size": "2",
		"route_top_blocks": "1",
	})
	if err != nil {
		t.Fatal(err)
	}
	assertTensorClose(t, out, []int{1, 1}, []float32{30})
}

func TestSparseAttentionPlanTracksRoutedBudget(t *testing.T) {
	plan := PlanSparseAttention(SparseAttentionPlanInput{
		QueryLen:       8,
		KeyLen:         64,
		QueryDim:       16,
		ValueDim:       32,
		TopK:           4,
		RouteBlockSize: 8,
		RouteTopBlocks: 2,
	})
	if plan.Routing != "block_anchor" || plan.RouteBlockCount != 8 || plan.SelectedRouteBlocks != 2 {
		t.Fatalf("routing plan = %+v", plan)
	}
	if plan.SelectedKeyCount != 4 || plan.CandidateKeyBudget != 16 || plan.EstimatedScoreCountPerQuery != 24 {
		t.Fatalf("budget plan = %+v", plan)
	}
	if math.Abs(plan.CandidateKeyFraction-0.25) > 0.000001 || math.Abs(plan.ScoreCountFraction-0.375) > 0.000001 {
		t.Fatalf("budget fractions = %+v", plan)
	}
	if !plan.SubquadraticScorePlan {
		t.Fatalf("subquadratic flag = false for %+v", plan)
	}
	clamped := PlanSparseAttention(SparseAttentionPlanInput{
		QueryLen:       1,
		KeyLen:         64,
		QueryDim:       16,
		ValueDim:       16,
		TopK:           32,
		RouteBlockSize: 8,
		RouteTopBlocks: 2,
	})
	if clamped.SelectedKeyCount != 16 {
		t.Fatalf("selected key count = %d, want route candidate budget clamp", clamped.SelectedKeyCount)
	}
}

func TestTurboQuantKVMemoryPlanEstimatesLogicalCompression(t *testing.T) {
	plan := PlanTurboQuantKVMemory(TurboQuantKVMemoryPlanInput{
		Batches:  1,
		KeyLen:   64,
		KeyDim:   16,
		ValueDim: 32,
		Bits:     4,
	})
	if plan.DenseKVBytes != 6144 {
		t.Fatalf("dense KV bytes = %d, want 6144", plan.DenseKVBytes)
	}
	if plan.KeyCoordBytes != 512 || plan.ValueCoordBytes != 1024 || plan.NormBytes != 512 {
		t.Fatalf("turboquant byte split = %+v", plan)
	}
	if plan.TurboQuantKVBytes != 2048 {
		t.Fatalf("turboquant KV bytes = %d, want 2048", plan.TurboQuantKVBytes)
	}
	if math.Abs(plan.CompressionRatio-3) > 0.000001 {
		t.Fatalf("compression ratio = %f, want 3", plan.CompressionRatio)
	}
}

func TestTurboSparseAttentionHostMetadataTracksBudget(t *testing.T) {
	query := NewTensorF16([]int{1, 16}, make([]float32, 16))
	keyCoords := NewTensorQ4([]int{1, 16, 64, 1}, make([]float32, 1*16*64))
	valueCoords := NewTensorQ4([]int{1, 32, 64, 1}, make([]float32, 1*32*64))
	meta := turboSparseAttentionMetadata("turbo_sparse_attention", query, keyCoords, valueCoords, map[string]string{
		"top_k":            "4",
		"route_block_size": "8",
		"route_top_blocks": "2",
	})
	if meta["kv_decode"] != "host_reference_decode" || meta["dense_kv_materialized"] != true {
		t.Fatalf("turbo sparse host metadata = %+v", meta)
	}
	if meta["routing"] != "block_anchor" || meta["candidate_key_budget"] != 16 || meta["selected_key_count"] != 4 {
		t.Fatalf("budget metadata = %+v", meta)
	}
	if meta["route_block_count"] != 8 || meta["estimated_score_count_per_query"] != 24 || meta["subquadratic_score_plan"] != true {
		t.Fatalf("routing metadata = %+v", meta)
	}
}

func TestTurboSparseAttentionReferenceMatchesDecodedSparseAttention(t *testing.T) {
	query := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	keyNCHW := NewTensorF16([]int{1, 2, 3, 1}, []float32{
		1, 0, -1,
		0, 1, 0,
	})
	valueNCHW := NewTensorF16([]int{1, 2, 3, 1}, []float32{
		10, 0, -10,
		0, 20, 0,
	})
	attrs := map[string]string{"bits": "4", "top_k": "1"}
	keyCoords, keyNorms, err := TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	out, err := TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatal(err)
	}
	decodedKey, err := TurboQuantDecodeReference(keyCoords, keyNorms, attrs)
	if err != nil {
		t.Fatal(err)
	}
	decodedValue, err := TurboQuantDecodeReference(valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatal(err)
	}
	keySeq, err := nchwToAttentionSequence(decodedKey, 2)
	if err != nil {
		t.Fatal(err)
	}
	valueSeq, err := nchwToAttentionSequence(decodedValue, 2)
	if err != nil {
		t.Fatal(err)
	}
	want, err := SparseAttentionReference(query, keySeq, valueSeq, attrs)
	if err != nil {
		t.Fatal(err)
	}
	assertTensorClose(t, out, want.Shape, want.F32)
}

func assertTensorClose(t *testing.T, tensor *Tensor, wantShape []int, want []float32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.F32) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.F32), len(want))
	}
	for i, got := range tensor.F32 {
		diff := got - want[i]
		if diff < -0.0005 || diff > 0.0005 {
			t.Fatalf("tensor[%d] = %f, want %f", i, got, want[i])
		}
	}
}

func assertTensorI32(t *testing.T, tensor *Tensor, wantShape []int, want []int32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.I32) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.I32), len(want))
	}
	for i, got := range tensor.I32 {
		if got != want[i] {
			t.Fatalf("tensor[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func assertTensorI64(t *testing.T, tensor *Tensor, wantShape []int, want []int64) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.I64) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.I64), len(want))
	}
	for i, got := range tensor.I64 {
		if got != want[i] {
			t.Fatalf("tensor[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func tensorValueType(dtype string, shape []string) mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
	}
}

func approxGELU(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}
