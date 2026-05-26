package mantaruntime

import (
	mantaartifact "m31labs.dev/manta/artifact/manta"
	"m31labs.dev/manta/runtime/backend"
)

func newTrainerMatMulAccelerator() (backend.MatMulAccelerator, mantaartifact.BackendKind, error) {
	return backend.NewPreferredMatMulAccelerator(mantaartifact.BackendCUDA, mantaartifact.BackendMetal)
}
