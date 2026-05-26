package mantaruntime

import (
	mantaartifact "m31labs.dev/manta/artifact/manta"
	"m31labs.dev/manta/runtime/backend"
)

func newTrainerContrastiveAccelerator() (backend.ContrastiveAccelerator, mantaartifact.BackendKind, error) {
	return backend.NewPreferredContrastiveAccelerator(mantaartifact.BackendCUDA, mantaartifact.BackendMetal)
}
