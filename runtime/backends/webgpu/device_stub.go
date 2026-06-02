//go:build !js || !wasm

package webgpu

import (
	"context"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
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

func (*deviceRuntime) dispatchStep(context.Context, eosartifact.Step, []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	return backend.StepDispatchResult{}, false, nil
}
