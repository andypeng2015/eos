//go:build linux && cgo

package cuda

import (
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// TestCUDAGraphCaptureReplayMatchesDirect retires the cuBLAS-in-graph risk:
// a captured graph must reproduce the direct (synced) cuBLAS GEMM bit-for-bit
// and, on replay, recompute against fresh contents of the same stable input
// buffer. The numerical work lives in graphReplaySelfTest (native_linux.go),
// where the cgo C types are in scope.
func TestCUDAGraphCaptureReplayMatchesDirect(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer rt.close()

	if err := rt.graphReplaySelfTest(); err != nil {
		t.Fatalf("CUDA graph capture/replay self-test failed: %v", err)
	}
}

// TestCUDAGraphBoundRightMatMulMatchesNonGraph verifies the production wiring:
// the bound-right GEMM batch must produce identical results with CUDA-graph
// capture/replay enabled (EOS_CUDA_GRAPH) as without it. Two successive LHS
// inputs of the same shape exercise both the capture path (first call) and the
// replay path (second call, cache hit) against fresh input data.
func TestCUDAGraphBoundRightMatMulMatchesNonGraph(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer rt.close()

	w0 := &backend.Tensor{DType: "f32", Shape: []int{2, 2}, F32: []float32{0.9, -0.35, 0.2, 0.7}}
	w1 := &backend.Tensor{DType: "f32", Shape: []int{2, 2}, F32: []float32{0.1, 1.2, -0.8, 0.4}}
	if err := rt.bindMatMulRight("w0", w0); err != nil {
		t.Fatalf("bind w0: %v", err)
	}
	if err := rt.bindMatMulRight("w1", w1); err != nil {
		t.Fatalf("bind w1: %v", err)
	}

	outType := eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}}
	names := []string{"w0", "w1"}
	lhsA := &backend.Tensor{DType: "f32", Shape: []int{2, 2}, F32: []float32{1.0, -0.5, 0.25, 0.75}}
	lhsB := &backend.Tensor{DType: "f32", Shape: []int{2, 2}, F32: []float32{0.3, 0.6, -0.2, 0.9}}

	run := func() ([][]float32, error) {
		var outs [][]float32
		for _, lhs := range []*backend.Tensor{lhsA, lhsB} {
			res, err := rt.runMatMulWithBoundRights(lhs, names, outType, false, false)
			if err != nil {
				return nil, err
			}
			for _, r := range res {
				outs = append(outs, append([]float32(nil), r.Outputs[0].F32...))
			}
		}
		return outs, nil
	}

	prev := eosCudaGraphEnabled
	defer func() { eosCudaGraphEnabled = prev }()

	// Reference: graph disabled.
	eosCudaGraphEnabled = false
	ref, err := run()
	if err != nil {
		t.Fatalf("reference (non-graph) run: %v", err)
	}

	// Graph enabled: first LHS captures per shape, second LHS replays.
	eosCudaGraphEnabled = true
	got, err := run()
	if err != nil {
		t.Fatalf("graph run: %v", err)
	}

	if len(got) != len(ref) {
		t.Fatalf("output count mismatch: got %d, ref %d", len(got), len(ref))
	}
	for i := range ref {
		if !cudaFloat32SliceBitEqual(ref[i], got[i]) {
			t.Fatalf("output %d differs (graph vs non-graph): ref=%v got=%v", i, ref[i], got[i])
		}
	}
	if len(rt.graphCache) == 0 {
		t.Fatal("expected at least one captured graph in cache after graph-enabled run")
	}
}
