//go:build linux && cgo

package cuda

import (
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDASparseAttentionStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	key := backend.NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		-1, 0,
	})
	value := backend.NewTensorF16([]int{3, 2}, []float32{
		10, 0,
		0, 20,
		-10, 0,
	})
	step := mantaartifact.Step{Kind: mantaartifact.StepSparseAttention, Attributes: map[string]string{"top_k": "1"}}
	cfg, ok := planBuiltinSparseAttention(step, []*backend.Tensor{query, key, value})
	if !ok {
		t.Fatal("sparse_attention should be supported")
	}
	outputType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runSparseAttentionStep([]*backend.Tensor{query, key, value}, outputType, cfg)
	if err != nil {
		t.Fatalf("run sparse_attention: %v", err)
	}
	if got.VariantEntry != "__builtin_cuda_sparse_attention" {
		t.Fatalf("variant = %q", got.VariantEntry)
	}
	if got.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", got.Metadata["device_execution"])
	}
	want, err := backend.SparseAttentionReference(query, key, value, step.Attributes)
	if err != nil {
		t.Fatalf("reference sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}

func TestCUDATurboSparseAttentionStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	keyNCHW := backend.NewTensorF16([]int{1, 2, 3, 1}, []float32{
		1, 0, -1,
		0, 1, 0,
	})
	valueNCHW := backend.NewTensorF16([]int{1, 2, 3, 1}, []float32{
		10, 0, -10,
		0, 20, 0,
	})
	attrs := map[string]string{"bits": "4", "seed": "77", "top_k": "1"}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	step := mantaartifact.Step{Kind: mantaartifact.StepTurboSparseAttention, Attributes: attrs}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
	if !ok {
		t.Fatal("turbo_sparse_attention should be supported")
	}
	outputType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg)
	if err != nil {
		t.Fatalf("run turbo_sparse_attention: %v", err)
	}
	if got.VariantEntry != "__builtin_cuda_turbo_sparse_attention" {
		t.Fatalf("variant = %q", got.VariantEntry)
	}
	if got.Metadata["dense_kv_materialized"] != false {
		t.Fatalf("dense_kv_materialized = %v, want false", got.Metadata["dense_kv_materialized"])
	}
	if got.Metadata["kv_decode"] != "cuda_turboquant_inline" {
		t.Fatalf("kv_decode = %v, want cuda_turboquant_inline", got.Metadata["kv_decode"])
	}
	want, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatalf("reference turbo_sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}

func TestCUDATurboSparseAttentionBatchedStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{2, 1, 2}, []float32{
		1, 0,
		0, 1,
	})
	keyNCHW := backend.NewTensorF16([]int{2, 2, 3, 1}, []float32{
		1, 0, -1,
		0, 1, 0,
		0, 1, 0,
		1, 0, -1,
	})
	valueNCHW := backend.NewTensorF16([]int{2, 2, 3, 1}, []float32{
		10, 0, -10,
		0, 20, 0,
		1, 2, 3,
		100, 200, 300,
	})
	attrs := map[string]string{"bits": "4", "seed": "91", "top_k": "1"}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	step := mantaartifact.Step{Kind: mantaartifact.StepTurboSparseAttention, Attributes: attrs}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
	if !ok {
		t.Fatal("batched turbo_sparse_attention should be supported")
	}
	outputType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg)
	if err != nil {
		t.Fatalf("run batched turbo_sparse_attention: %v", err)
	}
	want, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatalf("reference batched turbo_sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}

func TestCUDATurboSparseAttentionRoutedStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{1, 2}, []float32{1, 0})
	keyNCHW := backend.NewTensorF16([]int{1, 2, 6, 1}, []float32{
		0, 0, 1, 2, 0, 0,
		1, 0, 0, 0, 0, 1,
	})
	valueNCHW := backend.NewTensorF16([]int{1, 2, 6, 1}, []float32{
		1, 2, 30, 40, 5, 6,
		10, 20, 300, 400, 50, 60,
	})
	attrs := map[string]string{
		"bits":             "4",
		"seed":             "119",
		"top_k":            "1",
		"route_block_size": "2",
		"route_top_blocks": "1",
	}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	step := mantaartifact.Step{Kind: mantaartifact.StepTurboSparseAttention, Attributes: attrs}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
	if !ok {
		t.Fatal("routed turbo_sparse_attention should be supported")
	}
	outputType := mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg)
	if err != nil {
		t.Fatalf("run routed turbo_sparse_attention: %v", err)
	}
	if got.Metadata["routing"] != "block_anchor" {
		t.Fatalf("routing = %v, want block_anchor", got.Metadata["routing"])
	}
	if got.Metadata["candidate_key_budget"] != 2 {
		t.Fatalf("candidate_key_budget = %v, want 2", got.Metadata["candidate_key_budget"])
	}
	want, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatalf("reference routed turbo_sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}
