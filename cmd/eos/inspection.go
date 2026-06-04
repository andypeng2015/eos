package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"sort"
	"strings"
	"time"
	"unicode"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/directml"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/eos/runtime/backends/vulkan"
	"m31labs.dev/eos/runtime/backends/webgpu"
	prismvalidate "m31labs.dev/prism/validate"
)

type inspectionInput struct {
	Path         string
	Kind         string
	SourcePath   string
	ArtifactPath string
	Source       []byte
	Bundle       *compiler.Bundle
	Artifact     *eosartifact.Module
}

func loadInspectionInput(path string) (*inspectionInput, error) {
	if path == "" {
		return nil, fmt.Errorf("input path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if eosartifact.IsMLLBytes(data) {
		mod, err := eosartifact.DecodeMLL(data)
		if err != nil {
			return nil, err
		}
		return &inspectionInput{
			Path:         path,
			Kind:         "artifact",
			ArtifactPath: path,
			Artifact:     mod,
		}, nil
	}
	moduleName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	bundle, err := compiler.Build(data, compiler.Options{ModuleName: moduleName})
	if err != nil {
		return nil, err
	}
	return &inspectionInput{
		Path:       path,
		Kind:       "source",
		SourcePath: path,
		Source:     data,
		Bundle:     bundle,
		Artifact:   bundle.Artifact,
	}, nil
}

type graphReport struct {
	GraphVersion int              `json:"graph_version"`
	InputKind    string           `json:"input_kind"`
	SourcePath   string           `json:"source_path,omitempty"`
	ArtifactPath string           `json:"artifact_path,omitempty"`
	Module       string           `json:"module"`
	Counts       graphCounts      `json:"counts"`
	HIR          any              `json:"hir,omitempty"`
	MIR          any              `json:"mir,omitempty"`
	LIR          any              `json:"lir,omitempty"`
	Artifact     artifactSnapshot `json:"artifact"`
}

type graphCounts struct {
	SourceDecls      int `json:"source_decls,omitempty"`
	HIREntryPoints   int `json:"hir_entry_points,omitempty"`
	MIROps           int `json:"mir_ops,omitempty"`
	LIRBuffers       int `json:"lir_buffers,omitempty"`
	LIRKernels       int `json:"lir_kernels,omitempty"`
	LIRSteps         int `json:"lir_steps,omitempty"`
	ArtifactParams   int `json:"artifact_params"`
	ArtifactEntries  int `json:"artifact_entrypoints"`
	ArtifactBuffers  int `json:"artifact_buffers"`
	ArtifactKernels  int `json:"artifact_kernels"`
	ArtifactSteps    int `json:"artifact_steps"`
	KernelSourceVars int `json:"kernel_source_variants"`
}

type artifactSnapshot struct {
	Version      string                   `json:"version"`
	Name         string                   `json:"name"`
	Params       []eosartifact.Param      `json:"params,omitempty"`
	EntryPoints  []eosartifact.EntryPoint `json:"entry_points,omitempty"`
	Requirements eosartifact.Requirements `json:"requirements"`
	Buffers      []eosartifact.Buffer     `json:"buffers,omitempty"`
	Kernels      []kernelSnapshot         `json:"kernels,omitempty"`
	Steps        []eosartifact.Step       `json:"steps,omitempty"`
	Metadata     map[string]any           `json:"metadata,omitempty"`
}

type kernelSnapshot struct {
	Name     string                     `json:"name"`
	Inputs   []eosartifact.ValueBinding `json:"inputs,omitempty"`
	Outputs  []eosartifact.ValueBinding `json:"outputs,omitempty"`
	Hints    eosartifact.ScheduleHints  `json:"hints,omitempty"`
	Body     []eosartifact.KernelOp     `json:"body,omitempty"`
	Variants []variantSnapshot          `json:"variants,omitempty"`
}

type variantSnapshot struct {
	Backend     eosartifact.BackendKind `json:"backend"`
	Entry       string                  `json:"entry"`
	SourceBytes int                     `json:"source_bytes"`
	Meta        map[string]string       `json:"meta,omitempty"`
}

func newGraphReport(input *inspectionInput) graphReport {
	mod := input.Artifact
	report := graphReport{
		GraphVersion: 1,
		InputKind:    input.Kind,
		SourcePath:   input.SourcePath,
		ArtifactPath: input.ArtifactPath,
		Module:       mod.Name,
		Artifact:     snapshotArtifact(mod),
	}
	report.Counts.ArtifactParams = len(mod.Params)
	report.Counts.ArtifactEntries = len(mod.EntryPoints)
	report.Counts.ArtifactBuffers = len(mod.Buffers)
	report.Counts.ArtifactKernels = len(mod.Kernels)
	report.Counts.ArtifactSteps = len(mod.Steps)
	for _, kernel := range mod.Kernels {
		report.Counts.KernelSourceVars += len(kernel.Variants)
	}
	if input.Bundle != nil {
		report.HIR = input.Bundle.HIR
		report.MIR = input.Bundle.MIR
		report.LIR = input.Bundle.LIR
		report.Counts.SourceDecls = len(input.Bundle.Source.Decls)
		report.Counts.HIREntryPoints = len(input.Bundle.HIR.EntryPoints)
		report.Counts.MIROps = len(input.Bundle.MIR.Ops)
		report.Counts.LIRBuffers = len(input.Bundle.LIR.Buffers)
		report.Counts.LIRKernels = len(input.Bundle.LIR.Kernels)
		report.Counts.LIRSteps = len(input.Bundle.LIR.Steps)
	}
	return report
}

func snapshotArtifact(mod *eosartifact.Module) artifactSnapshot {
	if mod == nil {
		return artifactSnapshot{}
	}
	out := artifactSnapshot{
		Version:      mod.Version,
		Name:         mod.Name,
		Params:       mod.Params,
		EntryPoints:  mod.EntryPoints,
		Requirements: mod.Requirements,
		Buffers:      mod.Buffers,
		Steps:        mod.Steps,
		Metadata:     mod.Metadata,
	}
	for _, kernel := range mod.Kernels {
		snap := kernelSnapshot{
			Name:    kernel.Name,
			Inputs:  kernel.Inputs,
			Outputs: kernel.Outputs,
			Hints:   kernel.Hints,
			Body:    kernel.Body,
		}
		for _, variant := range kernel.Variants {
			snap.Variants = append(snap.Variants, variantSnapshot{
				Backend:     variant.Backend,
				Entry:       variant.Entry,
				SourceBytes: len(variant.Source),
				Meta:        variant.Meta,
			})
		}
		out.Kernels = append(out.Kernels, snap)
	}
	return out
}

func runGraph(args []string) error {
	fs := flag.NewFlagSet("graph", flag.ContinueOnError)
	format := fs.String("format", "json", "output format: json or dot")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: eos graph [--format json|dot] <source.eos|artifact.mll>")
	}
	input, err := loadInspectionInput(fs.Arg(0))
	if err != nil {
		return err
	}
	report := newGraphReport(input)
	switch *format {
	case "json":
		return writeJSON(os.Stdout, report)
	case "dot":
		fmt.Print(graphDOT(report))
		return nil
	default:
		return fmt.Errorf("unsupported graph format %q", *format)
	}
}

func graphDOT(report graphReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "digraph %s {\n", sanitizeIdent(report.Module))
	fmt.Fprintf(&b, "  module [label=%q, shape=box];\n", report.Module)
	for _, entry := range report.Artifact.EntryPoints {
		name := "entry_" + sanitizeIdent(entry.Name)
		fmt.Fprintf(&b, "  %s [label=%q, shape=oval];\n", name, "entry "+entry.Name)
		fmt.Fprintf(&b, "  module -> %s;\n", name)
	}
	for _, kernel := range report.Artifact.Kernels {
		name := "kernel_" + sanitizeIdent(kernel.Name)
		fmt.Fprintf(&b, "  %s [label=%q, shape=component];\n", name, "kernel "+kernel.Name)
		fmt.Fprintf(&b, "  module -> %s;\n", name)
	}
	for i, step := range report.Artifact.Steps {
		name := fmt.Sprintf("step_%03d", i)
		label := string(step.Kind)
		if step.Name != "" {
			label += "\\n" + step.Name
		}
		fmt.Fprintf(&b, "  %s [label=%q, shape=note];\n", name, label)
		if step.Entry != "" {
			fmt.Fprintf(&b, "  entry_%s -> %s;\n", sanitizeIdent(step.Entry), name)
		} else {
			fmt.Fprintf(&b, "  module -> %s;\n", name)
		}
		if step.Kernel != "" {
			fmt.Fprintf(&b, "  %s -> kernel_%s;\n", name, sanitizeIdent(step.Kernel))
		}
	}
	b.WriteString("}\n")
	return b.String()
}

func runKernels(args []string) error {
	fs := flag.NewFlagSet("kernels", flag.ContinueOnError)
	backendFilter := fs.String("backend", "", "backend to extract; empty extracts all")
	outDir := fs.String("out", "kernels", "directory for extracted kernel sources")
	validateSources := fs.Bool("validate", false, "record Prism kernel source validation status")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: eos kernels [--backend backend] [--out dir] [--validate] <source.eos|artifact.mll>")
	}
	input, err := loadInspectionInput(fs.Arg(0))
	if err != nil {
		return err
	}
	manifest, err := writeKernelSources(input.Artifact, *outDir, *backendFilter, "", *validateSources)
	if err != nil {
		return err
	}
	manifest.InputKind = input.Kind
	manifest.SourcePath = input.SourcePath
	manifest.ArtifactPath = input.ArtifactPath
	if err := writeJSONFile(filepath.Join(*outDir, "manifest.json"), manifest); err != nil {
		return err
	}
	fmt.Printf("wrote %d kernel source(s) -> %s\n", manifest.KernelSourceCount, *outDir)
	return nil
}

type kernelSourceManifest struct {
	ManifestVersion   int                 `json:"manifest_version"`
	CreatedAt         string              `json:"created_at,omitempty"`
	InputKind         string              `json:"input_kind,omitempty"`
	SourcePath        string              `json:"source_path,omitempty"`
	ArtifactPath      string              `json:"artifact_path,omitempty"`
	Module            string              `json:"module"`
	Backends          []string            `json:"backends"`
	KernelSourceCount int                 `json:"kernel_source_count"`
	Kernels           []kernelSourceEntry `json:"kernels"`
}

type kernelSourceEntry struct {
	Kernel      string                     `json:"kernel"`
	Backend     eosartifact.BackendKind    `json:"backend"`
	Entry       string                     `json:"entry"`
	SourceFile  string                     `json:"source_file"`
	SourceBytes int                        `json:"source_bytes"`
	Inputs      []eosartifact.ValueBinding `json:"inputs,omitempty"`
	Outputs     []eosartifact.ValueBinding `json:"outputs,omitempty"`
	Hints       eosartifact.ScheduleHints  `json:"hints,omitempty"`
	Validation  *kernelSourceValidation    `json:"validation,omitempty"`
}

type kernelSourceValidation struct {
	EntryChecked  bool   `json:"entry_checked"`
	ToolSkipped   bool   `json:"tool_skipped,omitempty"`
	ToolError     string `json:"tool_error,omitempty"`
	ToolOutputLen int    `json:"tool_output_len,omitempty"`
}

func writeKernelSources(mod *eosartifact.Module, outDir, backendFilter, manifestPrefix string, validateSources bool) (kernelSourceManifest, error) {
	if mod == nil {
		return kernelSourceManifest{}, fmt.Errorf("module is nil")
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return kernelSourceManifest{}, err
	}
	wantBackend := eosartifact.BackendKind(backendFilter)
	manifest := kernelSourceManifest{
		ManifestVersion:   1,
		Module:            mod.Name,
		KernelSourceCount: 0,
	}
	seenBackends := map[string]bool{}
	for _, kernel := range mod.Kernels {
		for _, variant := range kernel.Variants {
			if wantBackend != "" && variant.Backend != wantBackend {
				continue
			}
			filename := kernelSourceFilename(kernel.Name, variant.Backend)
			path := filepath.Join(outDir, filename)
			if err := os.WriteFile(path, []byte(variant.Source), 0o644); err != nil {
				return kernelSourceManifest{}, err
			}
			sourceFile := filename
			if manifestPrefix != "" {
				sourceFile = filepath.ToSlash(filepath.Join(manifestPrefix, filename))
			}
			validation, err := validateKernelSource(kernel.Name, variant, validateSources)
			if err != nil {
				return kernelSourceManifest{}, err
			}
			manifest.Kernels = append(manifest.Kernels, kernelSourceEntry{
				Kernel:      kernel.Name,
				Backend:     variant.Backend,
				Entry:       variant.Entry,
				SourceFile:  sourceFile,
				SourceBytes: len(variant.Source),
				Inputs:      kernel.Inputs,
				Outputs:     kernel.Outputs,
				Hints:       kernel.Hints,
				Validation:  validation,
			})
			manifest.KernelSourceCount++
			seenBackends[string(variant.Backend)] = true
		}
	}
	if wantBackend != "" && manifest.KernelSourceCount == 0 {
		return kernelSourceManifest{}, fmt.Errorf("no kernel variants found for backend %q", wantBackend)
	}
	for backendName := range seenBackends {
		manifest.Backends = append(manifest.Backends, backendName)
	}
	sort.Strings(manifest.Backends)
	return manifest, nil
}

func validateKernelSource(kernelName string, variant eosartifact.KernelVariant, runTool bool) (*kernelSourceValidation, error) {
	if !runTool {
		return nil, nil
	}
	src, err := eosartifact.PrismSourceForKernelVariant(kernelName, variant)
	if err != nil {
		return nil, err
	}
	if err := prismvalidate.CheckSource(src); err != nil {
		return nil, err
	}
	out := &kernelSourceValidation{EntryChecked: true}
	res, err := prismvalidate.RunSource(src)
	out.ToolSkipped = res.Skipped
	out.ToolOutputLen = len(res.Output)
	if err != nil {
		out.ToolError = err.Error()
	}
	return out, nil
}

func kernelSourceFilename(kernel string, backend eosartifact.BackendKind) string {
	return sanitizeFilename(kernel) + "." + sanitizeFilename(string(backend)) + "." + kernelSourceExt(backend)
}

func kernelSourceExt(backend eosartifact.BackendKind) string {
	switch backend {
	case eosartifact.BackendCUDA:
		return "cu"
	case eosartifact.BackendMetal:
		return "metal"
	case eosartifact.BackendVulkan:
		return "glsl"
	case eosartifact.BackendDirectML:
		return "hlsl"
	case eosartifact.BackendWebGPU:
		return "wgsl"
	default:
		return "txt"
	}
}

type compileBundleManifest struct {
	BundleVersion     int                 `json:"bundle_version"`
	CreatedAt         string              `json:"created_at"`
	Module            string              `json:"module"`
	SourcePath        string              `json:"source_path"`
	ArtifactPath      string              `json:"artifact_path"`
	EntryPoints       int                 `json:"entrypoints"`
	Steps             int                 `json:"steps"`
	Kernels           int                 `json:"kernels"`
	Backends          []string            `json:"backends"`
	KernelSourceCount int                 `json:"kernel_source_count"`
	KernelSources     []kernelSourceEntry `json:"kernel_sources"`
}

func writeCompileBundle(dir, srcPath string, src []byte, artifactPath string, bundle *compiler.Bundle, validateSources bool) error {
	if bundle == nil || bundle.Artifact == nil {
		return fmt.Errorf("compiler bundle is empty")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "source.eos"), src, 0o644); err != nil {
		return err
	}
	if err := eosartifact.WriteFile(filepath.Join(dir, "artifact.mll"), bundle.Artifact); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(dir, "graph.json"), newGraphReport(&inspectionInput{
		Kind:       "source",
		SourcePath: srcPath,
		Source:     src,
		Bundle:     bundle,
		Artifact:   bundle.Artifact,
	})); err != nil {
		return err
	}
	kernelManifest, err := writeKernelSources(bundle.Artifact, filepath.Join(dir, "kernels"), "", "kernels", validateSources)
	if err != nil {
		return err
	}
	manifest := compileBundleManifest{
		BundleVersion:     1,
		CreatedAt:         time.Now().UTC().Format(time.RFC3339),
		Module:            bundle.Artifact.Name,
		SourcePath:        srcPath,
		ArtifactPath:      artifactPath,
		EntryPoints:       len(bundle.Artifact.EntryPoints),
		Steps:             len(bundle.Artifact.Steps),
		Kernels:           len(bundle.Artifact.Kernels),
		Backends:          kernelManifest.Backends,
		KernelSourceCount: kernelManifest.KernelSourceCount,
		KernelSources:     kernelManifest.Kernels,
	}
	return writeJSONFile(filepath.Join(dir, "manifest.json"), manifest)
}

func runDoctor(args []string) error {
	if len(args) != 0 {
		return fmt.Errorf("usage: eos doctor")
	}
	fmt.Println("eos: dev")
	fmt.Printf("artifact schema: %s\n", eosartifact.Version)
	fmt.Printf("go: %s %s/%s\n", goruntime.Version(), goruntime.GOOS, goruntime.GOARCH)
	fmt.Println("backends:")
	for _, candidate := range []backend.Backend{cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New()} {
		fmt.Printf("  %s", candidate.Kind())
		if provider, ok := candidate.(backend.CapabilityProvider); ok {
			fmt.Printf(" capabilities=%s", strings.Join(provider.Capabilities(), ","))
		}
		fmt.Println()
	}
	fmt.Println("tools:")
	for _, tool := range []string{"nvidia-smi", "nvcc", "xcrun", "vulkaninfo"} {
		if path, err := exec.LookPath(tool); err == nil {
			fmt.Printf("  %s: %s\n", tool, path)
		} else {
			fmt.Printf("  %s: unavailable\n", tool)
		}
	}
	fmt.Println("env:")
	for _, name := range []string{"EOS_CPU_PROFILE", "EOS_MEM_PROFILE", "CUDA_VISIBLE_DEVICES", "VK_ICD_FILENAMES"} {
		value := os.Getenv(name)
		if value == "" {
			value = "-"
		}
		fmt.Printf("  %s=%s\n", name, value)
	}
	return nil
}

func writeJSONFile(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return writeJSON(file, value)
}

func writeJSON(w io.Writer, value any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}

func sanitizeFilename(value string) string {
	if value == "" {
		return "unnamed"
	}
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == '.' {
			return r
		}
		return '_'
	}, value)
}

func sanitizeIdent(value string) string {
	if value == "" {
		return "unnamed"
	}
	var b strings.Builder
	for i, r := range value {
		if unicode.IsLetter(r) || r == '_' || (i > 0 && unicode.IsDigit(r)) {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	return b.String()
}
