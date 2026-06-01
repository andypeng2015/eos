//go:build !js || !wasm

package webgpu

import (
	"context"

	mantaartifact "m31labs.dev/manta/artifact/manta"
	"m31labs.dev/manta/runtime/backend"
)

type deviceRuntime struct{}

func newDeviceRuntime(context.Context) (*deviceRuntime, error) {
	return nil, nil
}

// adoptDeviceRuntime is a no-op off js/wasm: there is no WebGPU device runtime
// to share outside the browser, so inference continues via host fallback.
func adoptDeviceRuntime(any) (*deviceRuntime, error) {
	return nil, nil
}

func (*deviceRuntime) dispatchStep(context.Context, mantaartifact.Step, []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	return backend.StepDispatchResult{}, false, nil
}
