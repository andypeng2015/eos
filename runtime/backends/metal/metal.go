package metal

import (
	"context"
	"fmt"
	"sync"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

type cachedLoad struct {
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
	device   *deviceRuntime
}

// Backend is the Metal backend stub.
type Backend struct {
	mu          sync.Mutex
	loadCache   map[string]cachedLoad
	cacheHits   int
	cacheMisses int
}

// New returns a Metal backend.
func New() *Backend {
	return &Backend{loadCache: map[string]cachedLoad{}}
}

// Kind reports the backend identity.
func (b *Backend) Kind() eosartifact.BackendKind {
	return eosartifact.BackendMetal
}

// Capabilities reports the runtime features the Metal backend supports.
func (b *Backend) Capabilities() []string {
	return []string{
		eosartifact.CapabilityCandidatePack,
		eosartifact.CapabilityKVCache,
		eosartifact.CapabilityMaskedMeanPool,
		eosartifact.CapabilityHostFallback,
		eosartifact.CapabilityDeviceExecution,
		eosartifact.CapabilityImageOps,
		eosartifact.CapabilityTrainingLosses,
		eosartifact.CapabilityTurboQuant,
		eosartifact.CapabilitySparseAttention,
	}
}

// CanLoad reports whether the module allows Metal execution.
func (b *Backend) CanLoad(mod *eosartifact.Module) bool {
	return mod != nil && mod.SupportsBackend(eosartifact.BackendMetal)
}

// Load prepares a Metal executor stub.
func (b *Backend) Load(_ context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding) (backend.Executor, error) {
	return b.load(context.Background(), mod, weights, "")
}

func (b *Backend) LoadWithCacheKey(ctx context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	return b.load(ctx, mod, weights, cacheKey)
}

func (b *Backend) load(_ context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	if cacheKey != "" {
		if cached, ok := b.cachedLoad(cacheKey); ok {
			return &executor{module: mod, weights: weights, compiled: cached.compiled, native: cached.native, device: cached.device}, nil
		}
	}
	compiled, err := backend.CompileVariants(mod, eosartifact.BackendMetal)
	if err != nil {
		return nil, err
	}
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	native := map[string]backend.NativeKernelProgram{}
	for _, kernel := range mod.Kernels {
		prog, err := backend.CompileNativeKernelProgram(eosartifact.BackendMetal, kernel, compiled[kernel.Name])
		if err != nil {
			if device != nil {
				device.close()
			}
			return nil, err
		}
		if device != nil {
			if err := device.attachDeviceExecution(&prog, kernel); err != nil {
				device.close()
				return nil, err
			}
		}
		native[kernel.Name] = prog
	}
	if cacheKey != "" {
		b.storeCachedLoad(cacheKey, cachedLoad{compiled: compiled, native: native, device: device})
	}
	return &executor{module: mod, weights: weights, compiled: compiled, native: native, device: device}, nil
}

func (b *Backend) cachedLoad(cacheKey string) (cachedLoad, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	cached, ok := b.loadCache[cacheKey]
	if ok {
		b.cacheHits++
	} else {
		b.cacheMisses++
	}
	return cached, ok
}

func (b *Backend) storeCachedLoad(cacheKey string, cached cachedLoad) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.loadCache[cacheKey] = cached
}

type executor struct {
	module   *eosartifact.Module
	weights  map[string]backend.WeightBinding
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
	device   *deviceRuntime
}

func (e *executor) Backend() eosartifact.BackendKind {
	return eosartifact.BackendMetal
}

func (e *executor) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	return backend.ExecuteSymbolic(ctx, e.module, e.weights, e.compiled, e.dispatchKernel, e.dispatchStep, eosartifact.BackendMetal, req)
}

func (e *executor) dispatchKernel(_ context.Context, kernel eosartifact.Kernel, inputs []*backend.Tensor) (backend.KernelDispatchResult, error) {
	prog, ok := e.native[kernel.Name]
	if !ok {
		return backend.KernelDispatchResult{}, fmt.Errorf("Metal kernel %q is not compiled", kernel.Name)
	}
	meta := cloneLaunchConfig(prog.LaunchConfig)
	runner := prog.Run
	if shouldFallbackScoreKernel(kernel, inputs) && prog.Fallback != nil {
		runner = prog.Fallback
		meta["device_execution"] = false
		meta["dispatch_mode"] = "host_fallback"
		meta["execution_mode"] = "host_fallback"
		meta["launch_api"] = "host_reference"
		meta["fallback_reason"] = "unsupported_input_shape"
	}
	out, err := runner(inputs)
	if err != nil {
		return backend.KernelDispatchResult{}, err
	}
	return backend.KernelDispatchResult{
		Outputs:      out,
		VariantEntry: prog.Compiled.Entry,
		SourceHash:   prog.Compiled.SourceHash,
		Metadata:     meta,
	}, nil
}

func (e *executor) dispatchStep(_ context.Context, step eosartifact.Step, outputType eosartifact.ValueType, inputs []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	switch step.Kind {
	case eosartifact.StepMatMul:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinMatMul(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runMatMul(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	default:
		return backend.StepDispatchResult{}, false, nil
	}
}

func supportsBuiltinMatMul(inputs []*backend.Tensor) bool {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return false
	}
	lhs := inputs[0]
	rhs := inputs[1]
	switch len(lhs.Shape) {
	case 2:
		return len(rhs.Shape) == 2 && lhs.Shape[1] == rhs.Shape[0]
	case 3:
		switch len(rhs.Shape) {
		case 2:
			return lhs.Shape[2] == rhs.Shape[0]
		case 3:
			return lhs.Shape[0] == rhs.Shape[0] && lhs.Shape[2] == rhs.Shape[1]
		default:
			return false
		}
	default:
		return false
	}
}

func shouldFallbackScoreKernel(kernel eosartifact.Kernel, inputs []*backend.Tensor) bool {
	if len(kernel.Body) == 0 {
		return false
	}
	switch kernel.Body[0].Op {
	case "dot", "cosine", "l2_distance":
		if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
			return false
		}
		query := inputs[0]
		docs := inputs[1]
		return !(len(query.Shape) == 1 && len(docs.Shape) == 2 && query.Shape[0] == docs.Shape[1])
	default:
		return false
	}
}

func cloneLaunchConfig(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
