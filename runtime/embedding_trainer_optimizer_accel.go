package mantaruntime

import (
	mantaartifact "m31labs.dev/manta/artifact/manta"
	"m31labs.dev/manta/runtime/backend"
)

func newTrainerOptimizerAccelerator() (backend.OptimizerAccelerator, mantaartifact.BackendKind, error) {
	return backend.NewPreferredOptimizerAccelerator(mantaartifact.BackendCUDA, mantaartifact.BackendMetal)
}
