package eosruntime

import (
	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func newTrainerMatMulAccelerator() (backend.MatMulAccelerator, eosartifact.BackendKind, error) {
	return backend.NewPreferredMatMulAccelerator(eosartifact.BackendCUDA, eosartifact.BackendMetal)
}
