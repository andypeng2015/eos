package metal

import (
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestBuiltinMatMulKeepsScaledAttributeEligible(t *testing.T) {
	lhs := backend.NewTensorF32([]int{1, 4}, []float32{1, 2, 3, 4})
	rhs := backend.NewTensorF32([]int{4, 1}, []float32{1, 1, 1, 1})
	inputs := []*backend.Tensor{lhs, rhs}
	if !supportsBuiltinMatMul(inputs) {
		t.Fatal("valid scaled attention matmul shape should be native eligible")
	}
	result := backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{backend.NewTensorF32([]int{1, 1}, []float32{10})},
		VariantEntry: "mps_matrix_multiplication",
		Metadata:     map[string]any{"dispatch_mode": "backend_native"},
	}
	scaled, err := backend.ApplyMatMulAttributesToResult(lhs, rhs, map[string]string{"scale": "rsqrt_rhs_rows"}, result)
	if err != nil {
		t.Fatalf("apply scaled attrs: %v", err)
	}
	if got := scaled.Outputs[0].F32[0]; got != 5 {
		t.Fatalf("scaled output = %v, want 5", got)
	}
	if scaled.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("metadata not preserved: %+v", scaled.Metadata)
	}
}
