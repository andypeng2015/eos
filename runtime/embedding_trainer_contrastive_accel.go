package eosruntime

import (
	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func newTrainerContrastiveAccelerator() (backend.ContrastiveAccelerator, eosartifact.BackendKind, error) {
	return backend.NewPreferredContrastiveAccelerator(eosartifact.BackendCUDA, eosartifact.BackendMetal)
}
