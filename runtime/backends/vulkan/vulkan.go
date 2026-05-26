package vulkan

import (
	mantaartifact "m31labs.dev/manta/artifact/manta"
	"m31labs.dev/manta/runtime/backends/internal/fallback"
)

// New returns the Vulkan backend surface. Device execution is added behind the
// same Backend contract; until then this backend executes through host fallback.
func New() *fallback.Backend {
	return fallback.New(mantaartifact.BackendVulkan, "Vulkan")
}
