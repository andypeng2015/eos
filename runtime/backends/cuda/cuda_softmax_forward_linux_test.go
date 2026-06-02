//go:build linux && cgo

package cuda

import "testing"

// TestCUDASoftmaxForwardRowsMatchesHost validates the on-device forward row
// softmax kernel against the CPU reference. It is the keystone of moving the
// forward pass to device-resident dataflow: with scores softmaxed on the
// device, the attention scores GEMM and the mixed GEMM no longer round-trip
// through the host, which both cuts host<->device transfer (manta's #1
// bottleneck) and is a prerequisite for whole-forward CUDA-graph capture.
func TestCUDASoftmaxForwardRowsMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer rt.close()

	if err := rt.softmaxForwardSelfTest(); err != nil {
		t.Fatalf("on-device forward softmax kernel: %v", err)
	}
}

// TestCUDAForwardActivationKernelsMatchHost validates the on-device forward
// GELU, layernorm, and residual-add kernels against host references — the
// remaining device-resident forward activations needed to keep the whole
// forward pass on the GPU (alongside the softmax kernel).
func TestCUDAForwardActivationKernelsMatchHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer rt.close()

	if err := rt.forwardActivationKernelsSelfTest(); err != nil {
		t.Fatalf("on-device forward activation kernels: %v", err)
	}
}
