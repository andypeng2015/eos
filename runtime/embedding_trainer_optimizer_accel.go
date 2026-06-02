package eosruntime

import (
	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func newTrainerOptimizerAccelerator() (backend.OptimizerAccelerator, eosartifact.BackendKind, error) {
	return backend.NewPreferredOptimizerAccelerator(eosartifact.BackendCUDA, eosartifact.BackendMetal)
}
