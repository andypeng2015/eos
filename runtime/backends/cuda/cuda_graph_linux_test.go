//go:build linux && cgo

package cuda

import "testing"

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
