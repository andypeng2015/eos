package vulkan

import (
	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backends/internal/fallback"
)

// New returns the Vulkan backend surface. Device execution is added behind the
// same Backend contract; until then this backend executes through host fallback.
func New() *fallback.Backend {
	return fallback.New(eosartifact.BackendVulkan, "Vulkan")
}
