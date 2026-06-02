package compiler

import (
	"strings"
	"testing"

	mantaartifact "m31labs.dev/manta/artifact/manta"
	"m31labs.dev/manta/ir/lir"
)

// TestWebGPUWorkgroupSizeFromHints verifies the generated WebGPU compute kernel
// derives @workgroup_size from the kernel's schedule hints (subgroup / tile)
// and never falls back to a single thread.
func TestWebGPUWorkgroupSizeFromHints(t *testing.T) {
	var webgpu kernelBackendEmitter
	found := false
	for _, e := range kernelBackendEmitters {
		if e.backend == mantaartifact.BackendWebGPU {
			webgpu, found = e, true
		}
	}
	if !found {
		t.Fatal("no WebGPU kernel emitter registered")
	}

	cases := []struct {
		name  string
		hints lir.ScheduleHints
		want  string
	}{
		{"default", lir.ScheduleHints{}, "@workgroup_size(64)"},
		{"tile-1d", lir.ScheduleHints{Tile: []int{128}}, "@workgroup_size(128)"},
		{"subgroup-2d", lir.ScheduleHints{Subgroup2D: []int{8, 8}}, "@workgroup_size(8, 8)"},
		{"tile-2d", lir.ScheduleHints{Tile2D: []int{16, 16}}, "@workgroup_size(16, 16)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src := emitGenericKernelSource(webgpu, lir.Kernel{Name: "k", Hints: c.hints})
			if !strings.Contains(src, c.want) {
				t.Errorf("want %q in:\n%s", c.want, src)
			}
			if strings.Contains(src, "@workgroup_size(1)") {
				t.Errorf("emitted single-thread workgroup:\n%s", src)
			}
		})
	}
}
