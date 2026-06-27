package backend

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strconv"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func executeStep(ctx context.Context, mod *eosartifact.Module, entry eosartifact.EntryPoint, step eosartifact.Step, env map[string]Value, compiled map[string]CompiledKernel, dispatch KernelDispatcher, dispatchStep StepDispatcher, bindings map[string]int, kind eosartifact.BackendKind) ([]Value, string, error) {
	switch step.Kind {
	case eosartifact.StepAlias:
		if len(step.Inputs) != 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("alias step %q expects 1 input and 1 output", step.Name)
		}
		in, ok := env[step.Inputs[0]]
		if !ok {
			return nil, "", fmt.Errorf("alias step %q missing input %q", step.Name, step.Inputs[0])
		}
		out := in
		if out.Metadata != nil {
			out.Metadata = cloneAnyMap(out.Metadata)
			out.Metadata["aliased_from"] = step.Inputs[0]
		} else {
			out.Metadata = map[string]any{"aliased_from": step.Inputs[0]}
		}
		return []Value{out}, "", nil
	case eosartifact.StepGather:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("gather step %q expects 2 inputs and 1 output", step.Name)
		}
		table, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		indices, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		out, err := gatherTensor(table, indices, resolveStepOutputType(mod, entry, step, 0, env), gatherModeOutputType(step.Attributes))
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", nil)}, "", nil
	case eosartifact.StepBERTEmbeddings:
		if len(step.Inputs) != 8 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("bert_embeddings step %q expects 8 inputs and 1 output", step.Name)
		}
		inputs := make([]*Tensor, 0, len(step.Inputs))
		for _, name := range step.Inputs {
			t, err := requireTensor(env[name], name)
			if err != nil {
				return nil, "", err
			}
			inputs = append(inputs, t)
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, inputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := bertEmbeddingsTensor(inputs, outputType, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("bert_embeddings"))}, "", nil
	case eosartifact.StepBERTEncoderLayer:
		if len(step.Inputs) != 18 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("bert_encoder_layer step %q expects 18 inputs and 1 output", step.Name)
		}
		inputs := make([]*Tensor, 0, len(step.Inputs))
		for _, name := range step.Inputs {
			t, err := requireTensor(env[name], name)
			if err != nil {
				return nil, "", err
			}
			inputs = append(inputs, t)
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, inputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := bertEncoderLayerTensor(inputs, outputType, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("bert_encoder_layer"))}, "", nil
	case eosartifact.StepBERTEmbedder:
		if len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("bert_embedder step %q expects 1 output", step.Name)
		}
		inputs := make([]*Tensor, 0, len(step.Inputs))
		for _, name := range step.Inputs {
			t, err := requireTensor(env[name], name)
			if err != nil {
				return nil, "", err
			}
			inputs = append(inputs, t)
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, inputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := bertEmbedderTensor(inputs, outputType, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		meta := hostReferenceMetadata("bert_embedder")
		pooling := step.Attributes["pooling"]
		if pooling == "" {
			pooling = "masked_mean"
		}
		meta["pooling"] = pooling
		meta["normalization"] = "l2"
		meta["execution_status"] = "host_reference_full_stack"
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", meta)}, "", nil
	case eosartifact.StepTranspose:
		if len(step.Inputs) != 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("transpose step %q expects 1 input and 1 output", step.Name)
		}
		in, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		out, err := transposeTensor(in)
		if err != nil {
			return nil, "", err
		}
		meta := map[string]any{
			"dispatch_mode":  "host_reference",
			"execution_mode": "host_reference",
			"launch_api":     "host_reference",
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", meta)}, "", nil
	case eosartifact.StepPack:
		if len(step.Inputs) != 3 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("pack_candidates step %q expects 3 inputs and 1 output", step.Name)
		}
		ids, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		scores, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		docs, err := requireTensor(env[step.Inputs[2]], step.Inputs[2])
		if err != nil {
			return nil, "", err
		}
		return []Value{makeCandidatePackValue(mod, entry, step, NewCandidatePack(ids, scores, docs), bindings, kind)}, "", nil
	case eosartifact.StepMatMul:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("matmul step %q expects 2 inputs and 1 output", step.Name)
		}
		lhs, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		rhs, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{lhs, rhs})
			if err != nil {
				return nil, "", err
			}
			if handled {
				values := make([]Value, 0, len(result.Outputs))
				for i, tensor := range result.Outputs {
					values = append(values, makeTensorValue(mod, entry, step, i, tensor, bindings, kind, result.VariantEntry, result.SourceHash, result.Metadata))
				}
				return values, result.VariantEntry, nil
			}
		}
		out, err := matmulTensorWithAttributes(lhs, rhs, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		meta := map[string]any{
			"dispatch_mode":  "host_reference",
			"execution_mode": "host_reference",
			"launch_api":     "host_reference",
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", meta)}, "", nil
	case eosartifact.StepTopK:
		if len(step.Inputs) != 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("topk step %q expects 1 input and 1 output", step.Name)
		}
		in, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		k, err := topKStepLimit(step)
		if err != nil {
			return nil, "", err
		}
		out, err := topKTensor(in, k)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", nil)}, "", nil
	case eosartifact.StepDequant:
		if len(step.Inputs) != 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("dequant step %q expects 1 input and 1 output", step.Name)
		}
		in, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		out := in.Clone()
		if out.DType != "f16" {
			out.DType = "f16"
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", nil)}, "", nil
	case eosartifact.StepConv2D:
		if len(step.Inputs) < 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("conv2d step %q expects at least 2 inputs and 1 output", step.Name)
		}
		input, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		weight, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		bias, err := optionalTensorInput(env, step.Inputs, 2)
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		dispatchInputs := optionalDispatchInputs(input, weight, bias)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, dispatchInputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := conv2DTensor(input, weight, bias, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("conv2d"))}, "", nil
	case eosartifact.StepConv2DTrans:
		if len(step.Inputs) < 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("conv2d_transpose step %q expects at least 2 inputs and 1 output", step.Name)
		}
		input, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		weight, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		bias, err := optionalTensorInput(env, step.Inputs, 2)
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		dispatchInputs := optionalDispatchInputs(input, weight, bias)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, dispatchInputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := conv2DTransposeTensor(input, weight, bias, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("conv2d_transpose"))}, "", nil
	case eosartifact.StepGDN, eosartifact.StepIGDN:
		if len(step.Inputs) < 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("%s step %q expects at least 1 input and 1 output", step.Kind, step.Name)
		}
		input, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		beta, err := optionalTensorInput(env, step.Inputs, 1)
		if err != nil {
			return nil, "", err
		}
		gamma, err := optionalTensorInput(env, step.Inputs, 2)
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		dispatchInputs := optionalDispatchInputs(input, beta, gamma)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, dispatchInputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := gdnTensor(input, beta, gamma, step.Kind == eosartifact.StepIGDN)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata(string(step.Kind)))}, "", nil
	case eosartifact.StepTurboQEncode:
		if len(step.Inputs) != 1 || len(step.Outputs) != 2 {
			return nil, "", fmt.Errorf("turboquant_encode step %q expects 1 input and 2 outputs", step.Name)
		}
		input, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{input})
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		coords, norms, err := turboQuantEncodeTensor(input, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		meta := hostReferenceMetadata("turboquant_encode")
		return []Value{
			makeTensorValue(mod, entry, step, 0, coords, bindings, kind, "", "", meta),
			makeTensorValue(mod, entry, step, 1, norms, bindings, kind, "", "", meta),
		}, "", nil
	case eosartifact.StepTurboQDecode:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("turboquant_decode step %q expects 2 inputs and 1 output", step.Name)
		}
		coords, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		norms, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{coords, norms})
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := turboQuantDecodeTensor(coords, norms, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("turboquant_decode"))}, "", nil
	case eosartifact.StepSparseAttention:
		if len(step.Inputs) != 3 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("sparse_attention step %q expects 3 inputs and 1 output", step.Name)
		}
		query, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		key, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		value, err := requireTensor(env[step.Inputs[2]], step.Inputs[2])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{query, key, value})
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := sparseAttentionTensor(query, key, value, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", sparseAttentionMetadata("sparse_attention", query, key, value, step.Attributes))}, "", nil
	case eosartifact.StepTurboSparseAttention:
		if len(step.Inputs) != 5 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("turbo_sparse_attention step %q expects 5 inputs and 1 output", step.Name)
		}
		query, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		keyCoords, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		keyNorms, err := requireTensor(env[step.Inputs[2]], step.Inputs[2])
		if err != nil {
			return nil, "", err
		}
		valueCoords, err := requireTensor(env[step.Inputs[3]], step.Inputs[3])
		if err != nil {
			return nil, "", err
		}
		valueNorms, err := requireTensor(env[step.Inputs[4]], step.Inputs[4])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms})
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := turboSparseAttentionTensor(query, keyCoords, keyNorms, valueCoords, valueNorms, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", turboSparseAttentionMetadata("turbo_sparse_attention", query, keyCoords, valueCoords, step.Attributes))}, "", nil
	case eosartifact.StepCrossEntropy:
		if len(step.Inputs) < 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("cross_entropy_factorized step %q expects at least 1 input and 1 output", step.Name)
		}
		codes, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		logits, err := optionalTensorInput(env, step.Inputs, 1)
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		dispatchInputs := optionalDispatchInputs(codes, logits)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, dispatchInputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := crossEntropyFactorizedTensor(codes, logits, step.Attributes)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("cross_entropy_factorized"))}, "", nil
	case eosartifact.StepMSELoss, eosartifact.StepMSSSIMLoss:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("%s step %q expects 2 inputs and 1 output", step.Kind, step.Name)
		}
		lhs, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		rhs, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{lhs, rhs})
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		var out *Tensor
		if step.Kind == eosartifact.StepMSELoss {
			out, err = mseLossTensor(lhs, rhs)
		} else {
			out, err = msSSIMLossTensor(lhs, rhs)
		}
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata(string(step.Kind)))}, "", nil
	case eosartifact.StepScalarAdd:
		if len(step.Inputs) < 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("scalar_add step %q expects at least 1 input and 1 output", step.Name)
		}
		inputs := make([]*Tensor, 0, len(step.Inputs))
		for _, name := range step.Inputs {
			t, err := requireTensor(env[name], name)
			if err != nil {
				return nil, "", err
			}
			inputs = append(inputs, t)
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, inputs)
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		out, err := scalarAddTensor(inputs...)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("scalar_add"))}, "", nil
	case eosartifact.StepRDLoss:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("rate_distortion_loss step %q expects distortion, rate, and 1 output", step.Name)
		}
		distortion, err := requireTensor(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		rate, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		outputType := resolveStepOutputType(mod, entry, step, 0, env)
		if dispatchStep != nil {
			result, handled, err := dispatchStep(ctx, step, outputType, []*Tensor{distortion, rate})
			if err != nil {
				return nil, "", err
			}
			if handled {
				return tensorValuesFromDispatch(mod, entry, step, result, bindings, kind), result.VariantEntry, nil
			}
		}
		rateWeight := attrFloat(step.Attributes, "lambda", 1) * attrFloat(step.Attributes, "rate_scale", 1)
		out, err := rateDistortionLossTensor(distortion, rate, rateWeight)
		if err != nil {
			return nil, "", err
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", hostReferenceMetadata("rate_distortion_loss"))}, "", nil
	case eosartifact.StepKVRead:
		if len(step.Inputs) != 1 || len(step.Outputs) != 1 {
			return nil, "", fmt.Errorf("kv_read step %q expects 1 input and 1 output", step.Name)
		}
		cache, err := requireKVCache(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		var out *Tensor
		if cache.Value != nil {
			out = cache.Value.Clone()
		} else {
			out = zeroTensorForType(resolveStepOutputType(mod, entry, step, 0, env), bindings)
		}
		return []Value{makeTensorValue(mod, entry, step, 0, out, bindings, kind, "", "", nil)}, "", nil
	case eosartifact.StepKVWrite:
		if len(step.Inputs) != 2 {
			return nil, "", fmt.Errorf("kv_write step %q expects 2 inputs", step.Name)
		}
		cache, err := requireKVCache(env[step.Inputs[0]], step.Inputs[0])
		if err != nil {
			return nil, "", err
		}
		value, err := requireTensor(env[step.Inputs[1]], step.Inputs[1])
		if err != nil {
			return nil, "", err
		}
		cache.Value = value.Clone()
		return nil, "", nil
	case eosartifact.StepLaunchKernel:
		kernel, ok := kernelByName(mod, step.Kernel)
		if !ok {
			return nil, "", fmt.Errorf("entrypoint %q step %q references unknown kernel %q", entry.Name, step.Name, step.Kernel)
		}
		if _, ok := compiled[step.Kernel]; !ok {
			return nil, "", fmt.Errorf("entrypoint %q kernel %q is not compiled for backend %q", entry.Name, step.Kernel, kind)
		}
		if dispatch == nil {
			return nil, "", fmt.Errorf("entrypoint %q kernel %q has no backend dispatcher", entry.Name, step.Kernel)
		}
		args := make([]*Tensor, 0, len(step.Inputs))
		for _, input := range step.Inputs {
			t, err := requireTensor(env[input], input)
			if err != nil {
				return nil, "", err
			}
			args = append(args, t)
		}
		result, err := dispatch(ctx, kernel, args)
		if err != nil {
			return nil, "", err
		}
		values := make([]Value, 0, len(result.Outputs))
		for i, tensor := range result.Outputs {
			values = append(values, makeTensorValue(mod, entry, step, i, tensor, bindings, kind, result.VariantEntry, result.SourceHash, result.Metadata))
		}
		return values, result.VariantEntry, nil
	default:
		return nil, "", nil
	}
}

func executeKernel(kernel eosartifact.Kernel, inputs []*Tensor) ([]*Tensor, error) {
	locals := map[string]*Tensor{}
	for i, input := range kernel.Inputs {
		if i >= len(inputs) {
			return nil, fmt.Errorf("kernel %q missing input %q", kernel.Name, input.Name)
		}
		locals[input.Name] = inputs[i].Clone()
	}
	returnNames := []string{}
	for _, op := range kernel.Body {
		switch op.Kind {
		case eosartifact.KernelOpReturn:
			returnNames = append([]string(nil), op.Outputs...)
		default:
			out, err := executeKernelOp(op, locals)
			if err != nil {
				return nil, fmt.Errorf("kernel %q op %q: %w", kernel.Name, op.Op, err)
			}
			if len(op.Outputs) == 0 {
				continue
			}
			locals[op.Outputs[0]] = out
		}
	}
	if len(returnNames) == 0 {
		for _, output := range kernel.Outputs {
			returnNames = append(returnNames, output.Name)
		}
	}
	results := make([]*Tensor, 0, len(returnNames))
	for _, name := range returnNames {
		t, ok := locals[name]
		if !ok {
			return nil, fmt.Errorf("missing kernel return value %q", name)
		}
		results = append(results, t.Clone())
	}
	return results, nil
}

func executeKernelOp(op eosartifact.KernelOp, locals map[string]*Tensor) (*Tensor, error) {
	switch op.Op {
	case "normalize", "rmsnorm":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		return normalizeRows(in), nil
	case "layernorm":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		return layerNormRows(in), nil
	case "softmax":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		return softmaxRows(in), nil
	case "masked_softmax":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		mask, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return maskedSoftmaxRows(in, mask)
	case "gelu":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		return geluTensor(in), nil
	case "rope":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		return ropeRows(in), nil
	case "binary_add":
		return binaryTensorOp(locals, op.Inputs, func(a, b float32) float32 { return a + b })
	case "binary_sub":
		return binaryTensorOp(locals, op.Inputs, func(a, b float32) float32 { return a - b })
	case "binary_mul":
		return binaryTensorOp(locals, op.Inputs, func(a, b float32) float32 { return a * b })
	case "binary_div":
		return binaryTensorOp(locals, op.Inputs, func(a, b float32) float32 { return a / b })
	case "dequant":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		out := in.Clone()
		out.DType = "f16"
		return out, nil
	case "conv2d":
		input, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		weight, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		bias, err := optionalLocalTensor(locals, op.Inputs, 2)
		if err != nil {
			return nil, err
		}
		return conv2DTensor(input, weight, bias, op.Attributes)
	case "conv2d_transpose":
		input, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		weight, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		bias, err := optionalLocalTensor(locals, op.Inputs, 2)
		if err != nil {
			return nil, err
		}
		return conv2DTransposeTensor(input, weight, bias, op.Attributes)
	case "gdn", "igdn":
		input, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		beta, err := optionalLocalTensor(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		gamma, err := optionalLocalTensor(locals, op.Inputs, 2)
		if err != nil {
			return nil, err
		}
		return gdnTensor(input, beta, gamma, op.Op == "igdn")
	case "turboquant_decode":
		coords, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		norms, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return turboQuantDecodeTensor(coords, norms, op.Attributes)
	case "cross_entropy_factorized":
		codes, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		logits, err := optionalLocalTensor(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return crossEntropyFactorizedTensor(codes, logits, op.Attributes)
	case "mse_loss":
		lhs, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		rhs, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return mseLossTensor(lhs, rhs)
	case "ms_ssim_loss":
		lhs, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		rhs, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return msSSIMLossTensor(lhs, rhs)
	case "mean_pool":
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		if len(op.Inputs) > 1 {
			mask, err := tensorByName(locals, op.Inputs, 1)
			if err != nil {
				return nil, err
			}
			return meanPoolMaskedTensor(in, mask)
		}
		return meanPoolTensor(in)
	case "dot":
		query, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		docs, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return dotRows(query, docs)
	case "cosine":
		query, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		docs, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return cosineRows(query, docs)
	case "l2_distance":
		query, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		docs, err := tensorByName(locals, op.Inputs, 1)
		if err != nil {
			return nil, err
		}
		return l2DistanceRows(query, docs)
	default:
		in, err := tensorByName(locals, op.Inputs, 0)
		if err != nil {
			return nil, err
		}
		return in.Clone(), nil
	}
}

func tensorByName(locals map[string]*Tensor, names []string, idx int) (*Tensor, error) {
	if idx >= len(names) {
		return nil, fmt.Errorf("missing input %d", idx)
	}
	t, ok := locals[names[idx]]
	if !ok || t == nil {
		return nil, fmt.Errorf("unknown local %q", names[idx])
	}
	return t, nil
}

func optionalTensorInput(env map[string]Value, names []string, idx int) (*Tensor, error) {
	if idx >= len(names) {
		return nil, nil
	}
	return requireTensor(env[names[idx]], names[idx])
}

func optionalLocalTensor(locals map[string]*Tensor, names []string, idx int) (*Tensor, error) {
	if idx >= len(names) {
		return nil, nil
	}
	return tensorByName(locals, names, idx)
}

func hostReferenceMetadata(op string) map[string]any {
	return map[string]any{
		"dispatch_mode":  "host_reference",
		"execution_mode": "host_reference",
		"launch_api":     "host_reference",
		"op":             op,
	}
}

func binaryTensorOp(locals map[string]*Tensor, names []string, fn func(float32, float32) float32) (*Tensor, error) {
	lhs, err := tensorByName(locals, names, 0)
	if err != nil {
		return nil, err
	}
	rhs, err := tensorByName(locals, names, 1)
	if err != nil {
		return nil, err
	}
	if !lhs.EqualShape(rhs) {
		return nil, fmt.Errorf("shape mismatch %v vs %v", lhs.Shape, rhs.Shape)
	}
	out := lhs.Clone()
	for i := range out.F32 {
		out.F32[i] = fn(lhs.F32[i], rhs.F32[i])
	}
	return out, nil
}

func gatherTensor(table, indices *Tensor, outTypes ...eosartifact.ValueType) (*Tensor, error) {
	if table == nil || indices == nil {
		return nil, fmt.Errorf("nil gather input")
	}
	if len(indices.Shape) != 1 && len(indices.Shape) != 2 {
		return nil, fmt.Errorf("gather expects indices rank 1 or 2")
	}
	if len(indices.I32) == 0 {
		return nil, fmt.Errorf("gather expects i32 indices")
	}
	if len(table.Shape) == 1 && len(indices.Shape) == 1 {
		out := &Tensor{
			DType: table.DType,
			Shape: []int{indices.Shape[0]},
		}
		switch table.DType {
		case "i32":
			out.I32 = make([]int32, indices.Shape[0])
			for i, idx := range indices.I32 {
				if idx < 0 || int(idx) >= len(table.I32) {
					return nil, fmt.Errorf("gather index %d out of range", idx)
				}
				out.I32[i] = table.I32[idx]
			}
		case "i64":
			out.I64 = make([]int64, indices.Shape[0])
			for i, idx := range indices.I32 {
				if idx < 0 || int(idx) >= len(table.I64) {
					return nil, fmt.Errorf("gather index %d out of range", idx)
				}
				out.I64[i] = table.I64[idx]
			}
		default:
			out.F32 = make([]float32, indices.Shape[0])
			for i, idx := range indices.I32 {
				if idx < 0 || int(idx) >= len(table.F32) {
					return nil, fmt.Errorf("gather index %d out of range", idx)
				}
				out.F32[i] = table.F32[idx]
			}
		}
		return out, nil
	}
	if len(table.Shape) == 2 && len(indices.Shape) == 1 {
		rows, cols := table.Shape[0], table.Shape[1]
		out := &Tensor{
			DType: table.DType,
			Shape: []int{indices.Shape[0], cols},
		}
		switch table.DType {
		case "i32":
			out.I32 = make([]int32, indices.Shape[0]*cols)
			for i, idx := range indices.I32 {
				if idx < 0 || int(idx) >= rows {
					return nil, fmt.Errorf("gather index %d out of range", idx)
				}
				copy(out.I32[i*cols:(i+1)*cols], table.I32[int(idx)*cols:(int(idx)+1)*cols])
			}
		case "i64":
			out.I64 = make([]int64, indices.Shape[0]*cols)
			for i, idx := range indices.I32 {
				if idx < 0 || int(idx) >= rows {
					return nil, fmt.Errorf("gather index %d out of range", idx)
				}
				copy(out.I64[i*cols:(i+1)*cols], table.I64[int(idx)*cols:(int(idx)+1)*cols])
			}
		default:
			out.F32 = make([]float32, indices.Shape[0]*cols)
			for i, idx := range indices.I32 {
				if idx < 0 || int(idx) >= rows {
					return nil, fmt.Errorf("gather index %d out of range", idx)
				}
				copy(out.F32[i*cols:(i+1)*cols], table.F32[int(idx)*cols:(int(idx)+1)*cols])
			}
		}
		return out, nil
	}
	if len(table.Shape) == 2 && len(indices.Shape) == 2 {
		if expectedTensorRank(outTypes...) == 3 || table.Shape[0] != indices.Shape[0] {
			rows, cols := table.Shape[0], table.Shape[1]
			batches, k := indices.Shape[0], indices.Shape[1]
			out := &Tensor{
				DType: table.DType,
				Shape: []int{batches, k, cols},
			}
			switch table.DType {
			case "i32":
				out.I32 = make([]int32, batches*k*cols)
				for b := 0; b < batches; b++ {
					for i := 0; i < k; i++ {
						idx := indices.I32[b*k+i]
						if idx < 0 || int(idx) >= rows {
							return nil, fmt.Errorf("gather index %d out of range", idx)
						}
						srcBase := int(idx) * cols
						dstBase := (b*k + i) * cols
						copy(out.I32[dstBase:dstBase+cols], table.I32[srcBase:srcBase+cols])
					}
				}
			case "i64":
				out.I64 = make([]int64, batches*k*cols)
				for b := 0; b < batches; b++ {
					for i := 0; i < k; i++ {
						idx := indices.I32[b*k+i]
						if idx < 0 || int(idx) >= rows {
							return nil, fmt.Errorf("gather index %d out of range", idx)
						}
						srcBase := int(idx) * cols
						dstBase := (b*k + i) * cols
						copy(out.I64[dstBase:dstBase+cols], table.I64[srcBase:srcBase+cols])
					}
				}
			default:
				out.F32 = make([]float32, batches*k*cols)
				for b := 0; b < batches; b++ {
					for i := 0; i < k; i++ {
						idx := indices.I32[b*k+i]
						if idx < 0 || int(idx) >= rows {
							return nil, fmt.Errorf("gather index %d out of range", idx)
						}
						srcBase := int(idx) * cols
						dstBase := (b*k + i) * cols
						copy(out.F32[dstBase:dstBase+cols], table.F32[srcBase:srcBase+cols])
					}
				}
			}
			return out, nil
		}
		batches, rows := table.Shape[0], table.Shape[1]
		if indices.Shape[0] != batches {
			return nil, fmt.Errorf("gather batch mismatch %v x %v", table.Shape, indices.Shape)
		}
		k := indices.Shape[1]
		out := &Tensor{
			DType: table.DType,
			Shape: []int{batches, k},
		}
		switch table.DType {
		case "i32":
			out.I32 = make([]int32, batches*k)
			for b := 0; b < batches; b++ {
				for i := 0; i < k; i++ {
					idx := indices.I32[b*k+i]
					if idx < 0 || int(idx) >= rows {
						return nil, fmt.Errorf("gather index %d out of range", idx)
					}
					out.I32[b*k+i] = table.I32[b*rows+int(idx)]
				}
			}
		case "i64":
			out.I64 = make([]int64, batches*k)
			for b := 0; b < batches; b++ {
				for i := 0; i < k; i++ {
					idx := indices.I32[b*k+i]
					if idx < 0 || int(idx) >= rows {
						return nil, fmt.Errorf("gather index %d out of range", idx)
					}
					out.I64[b*k+i] = table.I64[b*rows+int(idx)]
				}
			}
		default:
			out.F32 = make([]float32, batches*k)
			for b := 0; b < batches; b++ {
				for i := 0; i < k; i++ {
					idx := indices.I32[b*k+i]
					if idx < 0 || int(idx) >= rows {
						return nil, fmt.Errorf("gather index %d out of range", idx)
					}
					out.F32[b*k+i] = table.F32[b*rows+int(idx)]
				}
			}
		}
		return out, nil
	}
	if len(table.Shape) == 3 && len(indices.Shape) == 2 {
		batches, rows, cols := table.Shape[0], table.Shape[1], table.Shape[2]
		if indices.Shape[0] != batches {
			return nil, fmt.Errorf("gather batch mismatch %v x %v", table.Shape, indices.Shape)
		}
		k := indices.Shape[1]
		out := &Tensor{
			DType: table.DType,
			Shape: []int{batches, k, cols},
		}
		switch table.DType {
		case "i32":
			out.I32 = make([]int32, batches*k*cols)
			for b := 0; b < batches; b++ {
				for i := 0; i < k; i++ {
					idx := indices.I32[b*k+i]
					if idx < 0 || int(idx) >= rows {
						return nil, fmt.Errorf("gather index %d out of range", idx)
					}
					srcBase := (b*rows + int(idx)) * cols
					dstBase := (b*k + i) * cols
					copy(out.I32[dstBase:dstBase+cols], table.I32[srcBase:srcBase+cols])
				}
			}
		case "i64":
			out.I64 = make([]int64, batches*k*cols)
			for b := 0; b < batches; b++ {
				for i := 0; i < k; i++ {
					idx := indices.I32[b*k+i]
					if idx < 0 || int(idx) >= rows {
						return nil, fmt.Errorf("gather index %d out of range", idx)
					}
					srcBase := (b*rows + int(idx)) * cols
					dstBase := (b*k + i) * cols
					copy(out.I64[dstBase:dstBase+cols], table.I64[srcBase:srcBase+cols])
				}
			}
		default:
			out.F32 = make([]float32, batches*k*cols)
			for b := 0; b < batches; b++ {
				for i := 0; i < k; i++ {
					idx := indices.I32[b*k+i]
					if idx < 0 || int(idx) >= rows {
						return nil, fmt.Errorf("gather index %d out of range", idx)
					}
					srcBase := (b*rows + int(idx)) * cols
					dstBase := (b*k + i) * cols
					copy(out.F32[dstBase:dstBase+cols], table.F32[srcBase:srcBase+cols])
				}
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("gather expects table rank 1, 2, or 3 with compatible indices")
}

func matmulTensor(lhs, rhs *Tensor) (*Tensor, error) {
	if lhs == nil || rhs == nil {
		return nil, fmt.Errorf("nil matmul input")
	}
	if (len(lhs.Shape) != 2 && len(lhs.Shape) != 3) || (len(rhs.Shape) != 2 && len(rhs.Shape) != 3) {
		return nil, fmt.Errorf("matmul expects rank-2 or rank-3 tensor inputs")
	}
	if len(lhs.Shape) == 2 {
		if len(rhs.Shape) != 2 {
			return nil, fmt.Errorf("matmul rank-2 lhs requires rank-2 rhs")
		}
		m, k, n := lhs.Shape[0], lhs.Shape[1], rhs.Shape[1]
		if rhs.Shape[0] != k {
			return nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
		}
		out := &Tensor{
			DType: lhs.DType,
			Shape: []int{m, n},
			F32:   make([]float32, m*n),
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for kk := 0; kk < k; kk++ {
					sum += lhs.F32[i*k+kk] * rhs.F32[kk*n+j]
				}
				out.F32[i*n+j] = sum
			}
		}
		return out, nil
	}
	batches, rows, k := lhs.Shape[0], lhs.Shape[1], lhs.Shape[2]
	if len(rhs.Shape) == 2 {
		n := rhs.Shape[1]
		if rhs.Shape[0] != k {
			return nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
		}
		out := &Tensor{
			DType: lhs.DType,
			Shape: []int{batches, rows, n},
			F32:   make([]float32, batches*rows*n),
		}
		for b := 0; b < batches; b++ {
			for i := 0; i < rows; i++ {
				lhsBase := (b*rows + i) * k
				outBase := (b*rows + i) * n
				for j := 0; j < n; j++ {
					sum := float32(0)
					for kk := 0; kk < k; kk++ {
						sum += lhs.F32[lhsBase+kk] * rhs.F32[kk*n+j]
					}
					out.F32[outBase+j] = sum
				}
			}
		}
		return out, nil
	}
	if rhs.Shape[0] != batches || rhs.Shape[1] != k {
		return nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
	}
	n := rhs.Shape[2]
	out := &Tensor{
		DType: lhs.DType,
		Shape: []int{batches, rows, n},
		F32:   make([]float32, batches*rows*n),
	}
	for b := 0; b < batches; b++ {
		for i := 0; i < rows; i++ {
			lhsBase := (b*rows + i) * k
			outBase := (b*rows + i) * n
			rhsBatchBase := b * k * n
			for j := 0; j < n; j++ {
				sum := float32(0)
				for kk := 0; kk < k; kk++ {
					sum += lhs.F32[lhsBase+kk] * rhs.F32[rhsBatchBase+kk*n+j]
				}
				out.F32[outBase+j] = sum
			}
		}
	}
	return out, nil
}

func matmulTensorWithAttributes(lhs, rhs *Tensor, attrs map[string]string) (*Tensor, error) {
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		return nil, err
	}
	return ApplyMatMulAttributes(lhs, rhs, attrs, out)
}

// ApplyMatMulAttributes applies StepMatMul output attributes to an already
// computed matmul output tensor.
func ApplyMatMulAttributes(lhs, rhs *Tensor, attrs map[string]string, out *Tensor) (*Tensor, error) {
	if out == nil {
		return nil, fmt.Errorf("matmul output is nil")
	}
	scale, err := matmulScale(lhs, rhs, attrs)
	if err != nil {
		return nil, err
	}
	if scale != 1 {
		for i := range out.F32 {
			out.F32[i] *= scale
		}
	}
	return out, nil
}

// ApplyMatMulAttributesToResult applies StepMatMul output attributes to a
// backend-owned native matmul result.
func ApplyMatMulAttributesToResult(lhs, rhs *Tensor, attrs map[string]string, result StepDispatchResult) (StepDispatchResult, error) {
	if len(result.Outputs) != 1 {
		return StepDispatchResult{}, fmt.Errorf("matmul dispatch returned %d outputs, want 1", len(result.Outputs))
	}
	out, err := ApplyMatMulAttributes(lhs, rhs, attrs, result.Outputs[0])
	if err != nil {
		return StepDispatchResult{}, err
	}
	result.Outputs[0] = out
	return result, nil
}

func matmulScale(lhs, rhs *Tensor, attrs map[string]string) (float32, error) {
	if len(attrs) == 0 || attrs["scale"] == "" || attrs["scale"] == "none" {
		return 1, nil
	}
	switch raw := attrs["scale"]; raw {
	case "rsqrt_rhs_rows":
		if rhs == nil || len(rhs.Shape) < 2 {
			return 0, fmt.Errorf("matmul scale %q requires rank-2 or rank-3 rhs", raw)
		}
		keyWidth := rhs.Shape[len(rhs.Shape)-2]
		if keyWidth <= 0 {
			return 0, fmt.Errorf("matmul scale %q requires positive key width, got %d", raw, keyWidth)
		}
		return float32(1 / math.Sqrt(float64(keyWidth))), nil
	default:
		value, err := strconv.ParseFloat(raw, 32)
		if err != nil {
			return 0, fmt.Errorf("unsupported matmul scale %q", raw)
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("matmul scale must be finite, got %q", raw)
		}
		return float32(value), nil
	}
}

func transposeTensor(in *Tensor) (*Tensor, error) {
	if in == nil {
		return nil, fmt.Errorf("nil transpose input")
	}
	switch len(in.Shape) {
	case 2:
		rows, cols := in.Shape[0], in.Shape[1]
		out := tensorForDType(in.DType, []int{cols, rows}, rows*cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				src := r*cols + c
				dst := c*rows + r
				switch in.DType {
				case "i32":
					out.I32[dst] = in.I32[src]
				case "i64":
					out.I64[dst] = in.I64[src]
				default:
					out.F32[dst] = in.F32[src]
				}
			}
		}
		return out, nil
	case 3:
		batches, rows, cols := in.Shape[0], in.Shape[1], in.Shape[2]
		out := tensorForDType(in.DType, []int{batches, cols, rows}, batches*rows*cols)
		for b := 0; b < batches; b++ {
			srcBatchBase := b * rows * cols
			dstBatchBase := b * cols * rows
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					src := srcBatchBase + r*cols + c
					dst := dstBatchBase + c*rows + r
					switch in.DType {
					case "i32":
						out.I32[dst] = in.I32[src]
					case "i64":
						out.I64[dst] = in.I64[src]
					default:
						out.F32[dst] = in.F32[src]
					}
				}
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("transpose expects rank-2 or rank-3 tensor input")
	}
}

func normalizeRows(in *Tensor) *Tensor {
	out := in.Clone()
	if len(in.Shape) != 2 && len(in.Shape) != 3 {
		return out
	}
	rows, cols := in.Shape[0], in.Shape[len(in.Shape)-1]
	if len(in.Shape) == 3 {
		rows = in.Shape[0] * in.Shape[1]
	}
	for r := 0; r < rows; r++ {
		sum := float64(0)
		for c := 0; c < cols; c++ {
			v := float64(in.F32[r*cols+c])
			sum += v * v
		}
		norm := float32(math.Sqrt(sum))
		if norm == 0 {
			continue
		}
		for c := 0; c < cols; c++ {
			out.F32[r*cols+c] = in.F32[r*cols+c] / norm
		}
	}
	return out
}

func layerNormRows(in *Tensor) *Tensor {
	out := in.Clone()
	if len(in.Shape) != 2 && len(in.Shape) != 3 {
		return out
	}
	rows, cols := in.Shape[0], in.Shape[len(in.Shape)-1]
	if len(in.Shape) == 3 {
		rows = in.Shape[0] * in.Shape[1]
	}
	for r := 0; r < rows; r++ {
		mean := float32(0)
		base := r * cols
		for c := 0; c < cols; c++ {
			mean += in.F32[base+c]
		}
		mean /= float32(cols)
		variance := float32(0)
		for c := 0; c < cols; c++ {
			centered := in.F32[base+c] - mean
			variance += centered * centered
		}
		variance /= float32(cols)
		invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
		for c := 0; c < cols; c++ {
			out.F32[base+c] = (in.F32[base+c] - mean) * invStd
		}
	}
	return out
}

func bertEmbeddingsTensor(inputs []*Tensor, outputType eosartifact.ValueType, attrs map[string]string) (*Tensor, error) {
	if len(inputs) != 8 {
		return nil, fmt.Errorf("bert_embeddings expects 8 tensors")
	}
	epsilon, err := bertEmbeddingEpsilon(attrs)
	if err != nil {
		return nil, err
	}
	return BERTEmbeddingsReference(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], epsilon, outputType)
}

func bertEmbeddingEpsilon(attrs map[string]string) (float64, error) {
	const fallback = 1e-12
	if attrs == nil || attrs["epsilon"] == "" {
		return fallback, nil
	}
	epsilon, err := strconv.ParseFloat(attrs["epsilon"], 64)
	if err != nil {
		return 0, fmt.Errorf("bert_embeddings epsilon %q is invalid: %w", attrs["epsilon"], err)
	}
	if epsilon < 0 {
		return 0, fmt.Errorf("bert_embeddings epsilon must be non-negative, got %g", epsilon)
	}
	return epsilon, nil
}

// BERTEmbeddingsReference computes word + position + token-type embeddings
// followed by affine LayerNorm over the hidden dimension.
func BERTEmbeddingsReference(tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings, gamma, beta, inputIDs, positionIDs, tokenTypeIDs *Tensor, epsilon float64, outTypes ...eosartifact.ValueType) (*Tensor, error) {
	if tokenEmbeddings == nil || positionEmbeddings == nil || tokenTypeEmbeddings == nil || gamma == nil || beta == nil || inputIDs == nil || positionIDs == nil || tokenTypeIDs == nil {
		return nil, fmt.Errorf("bert_embeddings expects non-nil tensors")
	}
	dim, err := validateBERTEmbeddingTables(tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings)
	if err != nil {
		return nil, err
	}
	if err := validateBERTLayerNormParam("embedding_layernorm_weight", gamma, dim); err != nil {
		return nil, err
	}
	if err := validateBERTLayerNormParam("embedding_layernorm_bias", beta, dim); err != nil {
		return nil, err
	}
	if err := validateBERTEmbeddingIDs(inputIDs, positionIDs, tokenTypeIDs); err != nil {
		return nil, err
	}
	outShape := append([]int(nil), inputIDs.Shape...)
	outShape = append(outShape, dim)
	if err := validateBERTEmbeddingsOutputType(outShape, outTypes...); err != nil {
		return nil, err
	}
	rows := inputIDs.Elements()
	out := NewTensorF32(outShape, make([]float32, rows*dim))
	for row := 0; row < rows; row++ {
		tokenID := int(inputIDs.I32[row])
		positionID := int(positionIDs.I32[row])
		tokenTypeID := int(tokenTypeIDs.I32[row])
		if tokenID < 0 || tokenID >= tokenEmbeddings.Shape[0] {
			return nil, fmt.Errorf("bert_embeddings input_ids value %d out of range [0,%d)", tokenID, tokenEmbeddings.Shape[0])
		}
		if positionID < 0 || positionID >= positionEmbeddings.Shape[0] {
			return nil, fmt.Errorf("bert_embeddings position_ids value %d out of range [0,%d)", positionID, positionEmbeddings.Shape[0])
		}
		if tokenTypeID < 0 || tokenTypeID >= tokenTypeEmbeddings.Shape[0] {
			return nil, fmt.Errorf("bert_embeddings token_type_ids value %d out of range [0,%d)", tokenTypeID, tokenTypeEmbeddings.Shape[0])
		}
		base := row * dim
		tokenBase := tokenID * dim
		positionBase := positionID * dim
		tokenTypeBase := tokenTypeID * dim
		mean := float64(0)
		for d := 0; d < dim; d++ {
			sum := tokenEmbeddings.F32[tokenBase+d] + positionEmbeddings.F32[positionBase+d] + tokenTypeEmbeddings.F32[tokenTypeBase+d]
			out.F32[base+d] = sum
			mean += float64(sum)
		}
		mean /= float64(dim)
		variance := float64(0)
		for d := 0; d < dim; d++ {
			centered := float64(out.F32[base+d]) - mean
			variance += centered * centered
		}
		variance /= float64(dim)
		invStd := 1 / math.Sqrt(variance+epsilon)
		for d := 0; d < dim; d++ {
			normalized := (float64(out.F32[base+d]) - mean) * invStd
			out.F32[base+d] = float32(normalized*float64(gamma.F32[d]) + float64(beta.F32[d]))
		}
	}
	return out, nil
}

func validateBERTEmbeddingTables(tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings *Tensor) (int, error) {
	tables := []struct {
		name  string
		table *Tensor
	}{
		{"token_embeddings", tokenEmbeddings},
		{"position_embeddings", positionEmbeddings},
		{"token_type_embeddings", tokenTypeEmbeddings},
	}
	dim := 0
	for _, item := range tables {
		if item.table.DType != "f32" && item.table.DType != "f16" {
			return 0, fmt.Errorf("bert_embeddings %s dtype %q is not f32-compatible", item.name, item.table.DType)
		}
		if len(item.table.Shape) != 2 {
			return 0, fmt.Errorf("bert_embeddings %s must be rank-2, got shape %v", item.name, item.table.Shape)
		}
		if item.table.Shape[0] <= 0 || item.table.Shape[1] <= 0 {
			return 0, fmt.Errorf("bert_embeddings %s shape must be positive, got %v", item.name, item.table.Shape)
		}
		if item.table.Elements() != len(item.table.F32) {
			return 0, fmt.Errorf("bert_embeddings %s element count %d does not match backing data", item.name, item.table.Elements())
		}
		if dim == 0 {
			dim = item.table.Shape[1]
			continue
		}
		if item.table.Shape[1] != dim {
			return 0, fmt.Errorf("bert_embeddings %s hidden size %d does not match token_embeddings hidden size %d", item.name, item.table.Shape[1], dim)
		}
	}
	return dim, nil
}

func validateBERTLayerNormParam(name string, tensor *Tensor, dim int) error {
	if tensor.DType != "f32" && tensor.DType != "f16" {
		return fmt.Errorf("bert_embeddings %s dtype %q is not f32-compatible", name, tensor.DType)
	}
	if len(tensor.Shape) != 1 || tensor.Shape[0] != dim {
		return fmt.Errorf("bert_embeddings %s shape %v does not match hidden size [%d]", name, tensor.Shape, dim)
	}
	if tensor.Elements() != len(tensor.F32) {
		return fmt.Errorf("bert_embeddings %s element count %d does not match backing data", name, tensor.Elements())
	}
	return nil
}

func validateBERTEmbeddingIDs(inputIDs, positionIDs, tokenTypeIDs *Tensor) error {
	ids := []struct {
		name string
		t    *Tensor
	}{
		{"input_ids", inputIDs},
		{"position_ids", positionIDs},
		{"token_type_ids", tokenTypeIDs},
	}
	for _, item := range ids {
		if item.t.DType != "i32" {
			return fmt.Errorf("bert_embeddings %s dtype %q is not i32", item.name, item.t.DType)
		}
		if len(item.t.Shape) != 1 && len(item.t.Shape) != 2 {
			return fmt.Errorf("bert_embeddings %s rank must be 1 or 2, got shape %v", item.name, item.t.Shape)
		}
		if item.t.Elements() != len(item.t.I32) {
			return fmt.Errorf("bert_embeddings %s element count %d does not match backing data", item.name, item.t.Elements())
		}
	}
	if !inputIDs.EqualShape(positionIDs) {
		return fmt.Errorf("bert_embeddings position_ids shape %v does not match input_ids shape %v", positionIDs.Shape, inputIDs.Shape)
	}
	if !inputIDs.EqualShape(tokenTypeIDs) {
		return fmt.Errorf("bert_embeddings token_type_ids shape %v does not match input_ids shape %v", tokenTypeIDs.Shape, inputIDs.Shape)
	}
	return nil
}

func validateBERTEmbeddingsOutputType(shape []int, outTypes ...eosartifact.ValueType) error {
	if len(outTypes) == 0 || outTypes[0].Tensor == nil {
		return nil
	}
	tensorType := outTypes[0].Tensor
	if tensorType.DType != "" && tensorType.DType != "f32" && tensorType.DType != "f16" {
		return fmt.Errorf("bert_embeddings output dtype %q is not f32-compatible", tensorType.DType)
	}
	if len(tensorType.Shape) > 0 && len(tensorType.Shape) != len(shape) {
		return fmt.Errorf("bert_embeddings output rank %d does not match computed rank %d", len(tensorType.Shape), len(shape))
	}
	return nil
}

func meanPoolTensor(in *Tensor) (*Tensor, error) {
	if in == nil {
		return nil, fmt.Errorf("nil mean_pool input")
	}
	switch len(in.Shape) {
	case 2:
		rows, cols := in.Shape[0], in.Shape[1]
		out := tensorForDType(in.DType, []int{cols}, cols)
		if rows == 0 {
			return out, nil
		}
		for c := 0; c < cols; c++ {
			sum := float32(0)
			for r := 0; r < rows; r++ {
				sum += in.F32[r*cols+c]
			}
			out.F32[c] = sum / float32(rows)
		}
		return out, nil
	case 3:
		batches, rows, cols := in.Shape[0], in.Shape[1], in.Shape[2]
		out := tensorForDType(in.DType, []int{batches, cols}, batches*cols)
		if rows == 0 {
			return out, nil
		}
		for b := 0; b < batches; b++ {
			for c := 0; c < cols; c++ {
				sum := float32(0)
				for r := 0; r < rows; r++ {
					sum += in.F32[(b*rows+r)*cols+c]
				}
				out.F32[b*cols+c] = sum / float32(rows)
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("mean_pool expects rank-2 or rank-3 tensor input")
	}
}

func meanPoolMaskedTensor(in, mask *Tensor) (*Tensor, error) {
	if in == nil {
		return nil, fmt.Errorf("nil mean_pool input")
	}
	if mask == nil {
		return nil, fmt.Errorf("nil mean_pool mask")
	}
	if mask.DType != "i32" {
		return nil, fmt.Errorf("mean_pool mask dtype %q is not supported", mask.DType)
	}
	switch len(in.Shape) {
	case 2:
		if len(mask.Shape) != 1 || mask.Shape[0] != in.Shape[0] {
			return nil, fmt.Errorf("mean_pool mask shape %v does not match input %v", mask.Shape, in.Shape)
		}
		rows, cols := in.Shape[0], in.Shape[1]
		out := tensorForDType(in.DType, []int{cols}, cols)
		count := 0
		for r := 0; r < rows; r++ {
			if mask.I32[r] == 0 {
				continue
			}
			count++
			for c := 0; c < cols; c++ {
				out.F32[c] += in.F32[r*cols+c]
			}
		}
		if count == 0 {
			return out, nil
		}
		inv := 1 / float32(count)
		for c := 0; c < cols; c++ {
			out.F32[c] *= inv
		}
		return out, nil
	case 3:
		if len(mask.Shape) != 2 || mask.Shape[0] != in.Shape[0] || mask.Shape[1] != in.Shape[1] {
			return nil, fmt.Errorf("mean_pool mask shape %v does not match input %v", mask.Shape, in.Shape)
		}
		batches, rows, cols := in.Shape[0], in.Shape[1], in.Shape[2]
		out := tensorForDType(in.DType, []int{batches, cols}, batches*cols)
		for b := 0; b < batches; b++ {
			count := 0
			for r := 0; r < rows; r++ {
				if mask.I32[b*rows+r] == 0 {
					continue
				}
				count++
				base := (b*rows + r) * cols
				for c := 0; c < cols; c++ {
					out.F32[b*cols+c] += in.F32[base+c]
				}
			}
			if count == 0 {
				continue
			}
			inv := 1 / float32(count)
			for c := 0; c < cols; c++ {
				out.F32[b*cols+c] *= inv
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("mean_pool expects rank-2 or rank-3 tensor input")
	}
}

func expectedTensorRank(outTypes ...eosartifact.ValueType) int {
	rank := 0
	for _, outType := range outTypes {
		if outType.Tensor == nil {
			continue
		}
		if n := len(outType.Tensor.Shape); n > rank {
			rank = n
		}
	}
	return rank
}

func gatherModeOutputType(attrs map[string]string) eosartifact.ValueType {
	if attrs == nil {
		return eosartifact.ValueType{}
	}
	switch attrs["mode"] {
	case "shared":
		return eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{Shape: []string{"B", "T", "D"}}}
	case "batched":
		return eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{Shape: []string{"B", "T"}}}
	default:
		return eosartifact.ValueType{}
	}
}

func tensorForDType(dtype string, shape []int, elements int) *Tensor {
	switch dtype {
	case "f32":
		return NewTensorF32(shape, make([]float32, elements))
	case "f16":
		return NewTensorF16(shape, make([]float32, elements))
	case "q2":
		return NewTensorQ2(shape, make([]float32, elements))
	case "q4":
		return NewTensorQ4(shape, make([]float32, elements))
	case "q8":
		return NewTensorQ8(shape, make([]float32, elements))
	case "q_norm":
		return NewTensorQNorm(shape, make([]float32, elements))
	default:
		return &Tensor{DType: dtype, Shape: append([]int(nil), shape...), F32: make([]float32, elements)}
	}
}

func softmaxRows(in *Tensor) *Tensor {
	out := in.Clone()
	if len(in.Shape) != 2 && len(in.Shape) != 3 {
		return out
	}
	rows, cols := in.Shape[0], in.Shape[len(in.Shape)-1]
	if len(in.Shape) == 3 {
		rows = in.Shape[0] * in.Shape[1]
	}
	for r := 0; r < rows; r++ {
		maxV := in.F32[r*cols]
		for c := 1; c < cols; c++ {
			if v := in.F32[r*cols+c]; v > maxV {
				maxV = v
			}
		}
		sum := float64(0)
		for c := 0; c < cols; c++ {
			ev := math.Exp(float64(in.F32[r*cols+c] - maxV))
			out.F32[r*cols+c] = float32(ev)
			sum += ev
		}
		if sum == 0 {
			continue
		}
		inv := float32(1 / sum)
		for c := 0; c < cols; c++ {
			out.F32[r*cols+c] *= inv
		}
	}
	return out
}

func maskedSoftmaxRows(in, mask *Tensor) (*Tensor, error) {
	if in == nil {
		return nil, fmt.Errorf("nil masked_softmax input")
	}
	if mask == nil {
		return nil, fmt.Errorf("nil masked_softmax mask")
	}
	if mask.DType != "i32" {
		return nil, fmt.Errorf("masked_softmax mask dtype %q is not supported", mask.DType)
	}
	out := in.Clone()
	switch len(in.Shape) {
	case 2:
		if len(mask.Shape) != 1 || mask.Shape[0] != in.Shape[1] {
			return nil, fmt.Errorf("masked_softmax mask shape %v does not match input %v", mask.Shape, in.Shape)
		}
		maskedSoftmaxRowsF32(out.F32, in.F32, in.Shape[0], in.Shape[1], func(_ int, col int) bool {
			return mask.I32[col] != 0
		})
		return out, nil
	case 3:
		if len(mask.Shape) != 2 || mask.Shape[0] != in.Shape[0] || mask.Shape[1] != in.Shape[2] {
			return nil, fmt.Errorf("masked_softmax mask shape %v does not match input %v", mask.Shape, in.Shape)
		}
		queryRows, cols := in.Shape[1], in.Shape[2]
		maskedSoftmaxRowsF32(out.F32, in.F32, in.Shape[0]*queryRows, cols, func(row int, col int) bool {
			batch := row / queryRows
			return mask.I32[batch*cols+col] != 0
		})
		return out, nil
	default:
		return nil, fmt.Errorf("masked_softmax expects rank-2 or rank-3 tensor input")
	}
}

func maskedSoftmaxRowsF32(out, in []float32, rows, cols int, active func(row, col int) bool) {
	for r := 0; r < rows; r++ {
		base := r * cols
		maxV := float32(math.Inf(-1))
		anyActive := false
		for c := 0; c < cols; c++ {
			if !active(r, c) {
				continue
			}
			if !anyActive || in[base+c] > maxV {
				maxV = in[base+c]
			}
			anyActive = true
		}
		if !anyActive {
			for c := 0; c < cols; c++ {
				out[base+c] = 0
			}
			continue
		}
		sum := float64(0)
		for c := 0; c < cols; c++ {
			if !active(r, c) {
				out[base+c] = 0
				continue
			}
			ev := math.Exp(float64(in[base+c] - maxV))
			out[base+c] = float32(ev)
			sum += ev
		}
		if sum == 0 {
			continue
		}
		inv := float32(1 / sum)
		for c := 0; c < cols; c++ {
			if active(r, c) {
				out[base+c] *= inv
			}
		}
	}
}

func geluTensor(in *Tensor) *Tensor {
	out := in.Clone()
	for i, x := range in.F32 {
		cubic := x * x * x
		inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
		out.F32[i] = 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
	}
	return out
}

func ropeRows(in *Tensor) *Tensor {
	out := in.Clone()
	if len(in.Shape) != 2 && len(in.Shape) != 3 {
		return out
	}
	rows, cols := in.Shape[0], in.Shape[len(in.Shape)-1]
	if len(in.Shape) == 3 {
		rows = in.Shape[0] * in.Shape[1]
	}
	seqRows := rows
	if len(in.Shape) == 3 {
		seqRows = in.Shape[1]
	}
	for r := 0; r < rows; r++ {
		pos := r
		if len(in.Shape) == 3 {
			pos = r % seqRows
		}
		for c := 0; c+1 < cols; c += 2 {
			theta := float64(pos) / math.Pow(10000, float64(c)/float64(cols))
			cosTheta := float32(math.Cos(theta))
			sinTheta := float32(math.Sin(theta))
			x0 := in.F32[r*cols+c]
			x1 := in.F32[r*cols+c+1]
			out.F32[r*cols+c] = x0*cosTheta - x1*sinTheta
			out.F32[r*cols+c+1] = x0*sinTheta + x1*cosTheta
		}
	}
	return out
}

func dotRows(query, docs *Tensor) (*Tensor, error) {
	layout, err := scoreLayout(query, docs)
	if err != nil {
		return nil, err
	}
	if !layout.Batched {
		out := NewTensorF32([]int{layout.Rows}, make([]float32, layout.Rows))
		for r := 0; r < layout.Rows; r++ {
			sum := float32(0)
			base := r * layout.Cols
			for c := 0; c < layout.Cols; c++ {
				sum += query.F32[c] * docs.F32[base+c]
			}
			out.F32[r] = sum
		}
		return out, nil
	}
	out := NewTensorF32([]int{layout.Batches, layout.Rows}, make([]float32, layout.Batches*layout.Rows))
	for b := 0; b < layout.Batches; b++ {
		queryBase := b * layout.Cols
		for r := 0; r < layout.Rows; r++ {
			sum := float32(0)
			docBase := (b*layout.Rows + r) * layout.Cols
			for c := 0; c < layout.Cols; c++ {
				sum += query.F32[queryBase+c] * docs.F32[docBase+c]
			}
			out.F32[b*layout.Rows+r] = sum
		}
	}
	return out, nil
}

func cosineRows(query, docs *Tensor) (*Tensor, error) {
	layout, err := scoreLayout(query, docs)
	if err != nil {
		return nil, err
	}
	if !layout.Batched {
		queryNorm := float64(0)
		for c := 0; c < layout.Cols; c++ {
			v := float64(query.F32[c])
			queryNorm += v * v
		}
		out := NewTensorF32([]int{layout.Rows}, make([]float32, layout.Rows))
		for r := 0; r < layout.Rows; r++ {
			dot := float64(0)
			docNorm := float64(0)
			base := r * layout.Cols
			for c := 0; c < layout.Cols; c++ {
				q := float64(query.F32[c])
				d := float64(docs.F32[base+c])
				dot += q * d
				docNorm += d * d
			}
			denom := math.Sqrt(queryNorm) * math.Sqrt(docNorm)
			if denom == 0 {
				out.F32[r] = 0
				continue
			}
			out.F32[r] = float32(dot / denom)
		}
		return out, nil
	}
	out := NewTensorF32([]int{layout.Batches, layout.Rows}, make([]float32, layout.Batches*layout.Rows))
	for b := 0; b < layout.Batches; b++ {
		queryNorm := float64(0)
		queryBase := b * layout.Cols
		for c := 0; c < layout.Cols; c++ {
			v := float64(query.F32[queryBase+c])
			queryNorm += v * v
		}
		for r := 0; r < layout.Rows; r++ {
			dot := float64(0)
			docNorm := float64(0)
			docBase := (b*layout.Rows + r) * layout.Cols
			for c := 0; c < layout.Cols; c++ {
				q := float64(query.F32[queryBase+c])
				d := float64(docs.F32[docBase+c])
				dot += q * d
				docNorm += d * d
			}
			denom := math.Sqrt(queryNorm) * math.Sqrt(docNorm)
			if denom == 0 {
				out.F32[b*layout.Rows+r] = 0
				continue
			}
			out.F32[b*layout.Rows+r] = float32(dot / denom)
		}
	}
	return out, nil
}

func l2DistanceRows(query, docs *Tensor) (*Tensor, error) {
	layout, err := scoreLayout(query, docs)
	if err != nil {
		return nil, err
	}
	if !layout.Batched {
		out := NewTensorF32([]int{layout.Rows}, make([]float32, layout.Rows))
		for r := 0; r < layout.Rows; r++ {
			sum := float64(0)
			base := r * layout.Cols
			for c := 0; c < layout.Cols; c++ {
				diff := float64(query.F32[c] - docs.F32[base+c])
				sum += diff * diff
			}
			out.F32[r] = float32(math.Sqrt(sum))
		}
		return out, nil
	}
	out := NewTensorF32([]int{layout.Batches, layout.Rows}, make([]float32, layout.Batches*layout.Rows))
	for b := 0; b < layout.Batches; b++ {
		queryBase := b * layout.Cols
		for r := 0; r < layout.Rows; r++ {
			sum := float64(0)
			docBase := (b*layout.Rows + r) * layout.Cols
			for c := 0; c < layout.Cols; c++ {
				diff := float64(query.F32[queryBase+c] - docs.F32[docBase+c])
				sum += diff * diff
			}
			out.F32[b*layout.Rows+r] = float32(math.Sqrt(sum))
		}
	}
	return out, nil
}

type scoreKernelLayout struct {
	Batched bool
	Batches int
	Rows    int
	Cols    int
}

func scoreLayout(query, docs *Tensor) (scoreKernelLayout, error) {
	if query == nil || docs == nil {
		return scoreKernelLayout{}, fmt.Errorf("nil score input")
	}
	if len(query.Shape) == 1 && len(docs.Shape) == 2 {
		if query.Shape[0] != docs.Shape[1] {
			return scoreKernelLayout{}, fmt.Errorf("score mismatch %v x %v", query.Shape, docs.Shape)
		}
		return scoreKernelLayout{Rows: docs.Shape[0], Cols: docs.Shape[1]}, nil
	}
	if len(query.Shape) == 2 && len(docs.Shape) == 3 {
		if query.Shape[0] != docs.Shape[0] || query.Shape[1] != docs.Shape[2] {
			return scoreKernelLayout{}, fmt.Errorf("score mismatch %v x %v", query.Shape, docs.Shape)
		}
		return scoreKernelLayout{
			Batched: true,
			Batches: query.Shape[0],
			Rows:    docs.Shape[1],
			Cols:    docs.Shape[2],
		}, nil
	}
	return scoreKernelLayout{}, fmt.Errorf("score expects query/docs as [D] x [N,D] or [Q,D] x [Q,N,D]")
}

func topKTensor(in *Tensor, k int) (*Tensor, error) {
	if in == nil {
		return nil, fmt.Errorf("nil topk input")
	}
	type pair struct {
		idx int
		val float32
	}
	if len(in.Shape) == 1 {
		if k <= 0 || k > len(in.F32) {
			return nil, fmt.Errorf("topk limit %d out of range for %d scores", k, len(in.F32))
		}
		pairs := make([]pair, len(in.F32))
		for i, v := range in.F32 {
			pairs[i] = pair{idx: i, val: v}
		}
		slices.SortFunc(pairs, func(a, b pair) int {
			switch {
			case a.val > b.val:
				return -1
			case a.val < b.val:
				return 1
			case a.idx < b.idx:
				return -1
			case a.idx > b.idx:
				return 1
			default:
				return 0
			}
		})
		out := NewTensorI32([]int{k}, make([]int32, k))
		for i := 0; i < k; i++ {
			out.I32[i] = int32(pairs[i].idx)
		}
		return out, nil
	}
	if len(in.Shape) == 2 {
		rows, cols := in.Shape[0], in.Shape[1]
		if k <= 0 || k > cols {
			return nil, fmt.Errorf("topk limit %d out of range for row width %d", k, cols)
		}
		out := NewTensorI32([]int{rows, k}, make([]int32, rows*k))
		for r := 0; r < rows; r++ {
			pairs := make([]pair, cols)
			base := r * cols
			for c := 0; c < cols; c++ {
				pairs[c] = pair{idx: c, val: in.F32[base+c]}
			}
			slices.SortFunc(pairs, func(a, b pair) int {
				switch {
				case a.val > b.val:
					return -1
				case a.val < b.val:
					return 1
				case a.idx < b.idx:
					return -1
				case a.idx > b.idx:
					return 1
				default:
					return 0
				}
			})
			for i := 0; i < k; i++ {
				out.I32[r*k+i] = int32(pairs[i].idx)
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("topk expects rank-1 or rank-2 scores")
}

func topKStepLimit(step eosartifact.Step) (int, error) {
	if step.Attributes == nil {
		return 0, fmt.Errorf("topk step %q is missing limit attribute", step.Name)
	}
	raw := step.Attributes["k"]
	if raw == "" {
		return 0, fmt.Errorf("topk step %q is missing limit attribute", step.Name)
	}
	k, err := strconv.Atoi(raw)
	if err != nil || k <= 0 {
		return 0, fmt.Errorf("topk step %q has invalid limit %q", step.Name, raw)
	}
	return k, nil
}

func kernelByName(mod *eosartifact.Module, name string) (eosartifact.Kernel, bool) {
	for _, kernel := range mod.Kernels {
		if kernel.Name == name {
			return kernel, true
		}
	}
	return eosartifact.Kernel{}, false
}

func zeroTensorForType(typ eosartifact.ValueType, bindings map[string]int) *Tensor {
	if typ.Kind != eosartifact.ValueTensor || typ.Tensor == nil {
		return &Tensor{DType: "f32", Shape: []int{1}, F32: []float32{0}}
	}
	shape := make([]int, len(typ.Tensor.Shape))
	for i, dim := range typ.Tensor.Shape {
		if literal, err := strconv.Atoi(dim); err == nil {
			shape[i] = literal
			continue
		}
		if n, ok := bindings[dim]; ok {
			shape[i] = n
			continue
		}
		shape[i] = 2
	}
	n := 1
	if len(shape) == 0 {
		n = 1
	} else {
		for _, dim := range shape {
			n *= dim
		}
	}
	if typ.Tensor.DType == "i32" {
		return &Tensor{DType: "i32", Shape: shape, I32: make([]int32, n)}
	}
	if typ.Tensor.DType == "i64" {
		return &Tensor{DType: "i64", Shape: shape, I64: make([]int64, n)}
	}
	return &Tensor{DType: typ.Tensor.DType, Shape: shape, F32: make([]float32, n)}
}

func zeroCandidatePackForType(typ eosartifact.ValueType, bindings map[string]int) *CandidatePack {
	if typ.Kind != eosartifact.ValueCandidatePack || typ.CandidatePack == nil {
		return NewCandidatePack(NewTensorI64([]int{1}, []int64{0}), NewTensorF32([]int{1}, []float32{0}), NewTensorQ4([]int{1, 1}, []float32{0}))
	}
	shape := make([]int, len(typ.CandidatePack.Shape))
	for i, dim := range typ.CandidatePack.Shape {
		if literal, err := strconv.Atoi(dim); err == nil {
			shape[i] = literal
			continue
		}
		if n, ok := bindings[dim]; ok {
			shape[i] = n
			continue
		}
		shape[i] = 2
	}
	if len(shape) == 2 {
		k, d := shape[0], shape[1]
		return NewCandidatePack(
			NewTensorI64([]int{k}, make([]int64, k)),
			NewTensorF32([]int{k}, make([]float32, k)),
			NewTensorQ4([]int{k, d}, make([]float32, k*d)),
		)
	}
	if len(shape) == 3 {
		q, k, d := shape[0], shape[1], shape[2]
		return NewCandidatePack(
			NewTensorI64([]int{q, k}, make([]int64, q*k)),
			NewTensorF32([]int{q, k}, make([]float32, q*k)),
			NewTensorQ4([]int{q, k, d}, make([]float32, q*k*d)),
		)
	}
	return NewCandidatePack(NewTensorI64([]int{1}, []int64{0}), NewTensorF32([]int{1}, []float32{0}), NewTensorQ4([]int{1, 1}, []float32{0}))
}

func cloneAnyMap(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func makeTensorValue(mod *eosartifact.Module, entry eosartifact.EntryPoint, step eosartifact.Step, outputIndex int, tensor *Tensor, bindings map[string]int, kind eosartifact.BackendKind, variantEntry, sourceHash string, extra map[string]any) Value {
	valueType := resolveStepOutputType(mod, entry, step, outputIndex, nil)
	concreteType := valueType
	if bound, err := concretizeValueType(valueType, tensor, bindings); err == nil {
		concreteType = bound
	}
	meta := map[string]any{
		"backend":       string(kind),
		"step":          string(step.Kind),
		"kernel":        step.Kernel,
		"variant_entry": variantEntry,
		"source_hash":   sourceHash,
	}
	for k, v := range extra {
		meta[k] = v
	}
	return Value{
		Type:     concreteType,
		Data:     tensor,
		Producer: producerName(step),
		Inputs:   cloneStrings(step.Inputs),
		Metadata: meta,
	}
}

func tensorValuesFromDispatch(mod *eosartifact.Module, entry eosartifact.EntryPoint, step eosartifact.Step, result StepDispatchResult, bindings map[string]int, kind eosartifact.BackendKind) []Value {
	values := make([]Value, 0, len(result.Outputs))
	for i, tensor := range result.Outputs {
		values = append(values, makeTensorValue(mod, entry, step, i, tensor, bindings, kind, result.VariantEntry, result.SourceHash, result.Metadata))
	}
	return values
}

func optionalDispatchInputs(inputs ...*Tensor) []*Tensor {
	out := make([]*Tensor, 0, len(inputs))
	for _, input := range inputs {
		if input != nil {
			out = append(out, input)
		}
	}
	return out
}

func makeCandidatePackValue(mod *eosartifact.Module, entry eosartifact.EntryPoint, step eosartifact.Step, pack *CandidatePack, bindings map[string]int, kind eosartifact.BackendKind) Value {
	valueType := resolveStepOutputType(mod, entry, step, 0, nil)
	concreteType := valueType
	if bound, err := concretizeValueType(valueType, pack, bindings); err == nil {
		concreteType = bound
	}
	meta := map[string]any{
		"backend": string(kind),
		"step":    string(step.Kind),
	}
	return Value{
		Type:     concreteType,
		Data:     pack,
		Producer: producerName(step),
		Inputs:   cloneStrings(step.Inputs),
		Metadata: meta,
	}
}
