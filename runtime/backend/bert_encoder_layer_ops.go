package backend

import (
	"fmt"
	"math"
	"math/bits"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func bertEncoderLayerTensor(inputs []*Tensor, outputType eosartifact.ValueType, attrs map[string]string) (*Tensor, error) {
	if len(inputs) != 18 {
		return nil, fmt.Errorf("bert_encoder_layer expects 18 tensors")
	}
	cfg, err := bertEncoderLayerConfigFromAttrs(attrs)
	if err != nil {
		return nil, err
	}
	return BERTEncoderLayerReference(
		inputs[0], inputs[1],
		inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
		inputs[8], inputs[9], inputs[10], inputs[11],
		inputs[12], inputs[13], inputs[14], inputs[15],
		inputs[16], inputs[17],
		cfg.NumAttentionHeads, cfg.Epsilon, cfg.HiddenAct, outputType,
	)
}

type bertDenseProjector interface {
	DenseHF(input []float32, rows, in int, weight, bias *Tensor) ([]float32, bool)
	DenseHFMany(input []float32, rows, in int, weights, biases []*Tensor) ([][]float32, bool)
}

type bertEncoderLayerConfig struct {
	NumAttentionHeads int
	Epsilon           float64
	HiddenAct         string
}

func bertEncoderLayerConfigFromAttrs(attrs map[string]string) (bertEncoderLayerConfig, error) {
	cfg := bertEncoderLayerConfig{NumAttentionHeads: 1, Epsilon: 1e-12, HiddenAct: "gelu"}
	if attrs == nil {
		return cfg, nil
	}
	if attrs["num_attention_heads"] != "" {
		heads, err := strconv.Atoi(attrs["num_attention_heads"])
		if err != nil || heads <= 0 {
			return cfg, fmt.Errorf("bert_encoder_layer num_attention_heads %q is invalid", attrs["num_attention_heads"])
		}
		cfg.NumAttentionHeads = heads
	}
	if attrs["epsilon"] != "" {
		epsilon, err := strconv.ParseFloat(attrs["epsilon"], 64)
		if err != nil {
			return cfg, fmt.Errorf("bert_encoder_layer epsilon %q is invalid: %w", attrs["epsilon"], err)
		}
		if epsilon < 0 {
			return cfg, fmt.Errorf("bert_encoder_layer epsilon must be non-negative, got %g", epsilon)
		}
		cfg.Epsilon = epsilon
	}
	if attrs["hidden_act"] != "" {
		cfg.HiddenAct = attrs["hidden_act"]
	}
	if cfg.HiddenAct != "gelu" {
		return cfg, fmt.Errorf("bert_encoder_layer unsupported hidden_act %q; only gelu is supported", cfg.HiddenAct)
	}
	return cfg, nil
}

// BERTEncoderLayerReference computes one HF-compatible BERT Transformer
// encoder layer using imported role-named tensors. Dense weights use HF
// [out,in] layout, so every projection computes x @ W^T + b.
func BERTEncoderLayerReference(
	hiddenStates, attentionMask *Tensor,
	queryWeight, queryBias, keyWeight, keyBias, valueWeight, valueBias *Tensor,
	attentionOutputWeight, attentionOutputBias, attentionLayerNormWeight, attentionLayerNormBias *Tensor,
	intermediateWeight, intermediateBias, outputWeight, outputBias *Tensor,
	outputLayerNormWeight, outputLayerNormBias *Tensor,
	numAttentionHeads int,
	epsilon float64,
	hiddenAct string,
	outTypes ...eosartifact.ValueType,
) (*Tensor, error) {
	return bertEncoderLayerReferenceWithProjector(
		hiddenStates, attentionMask,
		queryWeight, queryBias, keyWeight, keyBias, valueWeight, valueBias,
		attentionOutputWeight, attentionOutputBias, attentionLayerNormWeight, attentionLayerNormBias,
		intermediateWeight, intermediateBias, outputWeight, outputBias,
		outputLayerNormWeight, outputLayerNormBias,
		numAttentionHeads, epsilon, hiddenAct, defaultBERTDenseProjector(), outTypes...,
	)
}

func bertEncoderLayerReferenceWithProjector(
	hiddenStates, attentionMask *Tensor,
	queryWeight, queryBias, keyWeight, keyBias, valueWeight, valueBias *Tensor,
	attentionOutputWeight, attentionOutputBias, attentionLayerNormWeight, attentionLayerNormBias *Tensor,
	intermediateWeight, intermediateBias, outputWeight, outputBias *Tensor,
	outputLayerNormWeight, outputLayerNormBias *Tensor,
	numAttentionHeads int,
	epsilon float64,
	hiddenAct string,
	projector bertDenseProjector,
	outTypes ...eosartifact.ValueType,
) (*Tensor, error) {
	if hiddenAct == "" {
		hiddenAct = "gelu"
	}
	if hiddenAct != "gelu" {
		return nil, fmt.Errorf("bert_encoder_layer unsupported hidden_act %q; only gelu is supported", hiddenAct)
	}
	if numAttentionHeads <= 0 {
		return nil, fmt.Errorf("bert_encoder_layer num_attention_heads must be positive, got %d", numAttentionHeads)
	}
	if epsilon < 0 {
		return nil, fmt.Errorf("bert_encoder_layer epsilon must be non-negative, got %g", epsilon)
	}
	if hiddenStates == nil || attentionMask == nil {
		return nil, fmt.Errorf("bert_encoder_layer expects non-nil hidden_states and attention_mask")
	}
	if hiddenStates.DType != "f32" && hiddenStates.DType != "f16" {
		return nil, fmt.Errorf("bert_encoder_layer hidden_states dtype %q is not f32-compatible", hiddenStates.DType)
	}
	if len(hiddenStates.Shape) != 3 {
		return nil, fmt.Errorf("bert_encoder_layer hidden_states must be rank-3 [B,T,H], got shape %v", hiddenStates.Shape)
	}
	batch, tokens, hidden := hiddenStates.Shape[0], hiddenStates.Shape[1], hiddenStates.Shape[2]
	if batch <= 0 || tokens <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("bert_encoder_layer hidden_states shape must be positive, got %v", hiddenStates.Shape)
	}
	if hiddenStates.Elements() != len(hiddenStates.F32) {
		return nil, fmt.Errorf("bert_encoder_layer hidden_states element count %d does not match backing data", hiddenStates.Elements())
	}
	if attentionMask.DType != "i32" {
		return nil, fmt.Errorf("bert_encoder_layer attention_mask dtype %q is not i32", attentionMask.DType)
	}
	if len(attentionMask.Shape) != 2 || attentionMask.Shape[0] != batch || attentionMask.Shape[1] != tokens {
		return nil, fmt.Errorf("bert_encoder_layer attention_mask shape %v does not match [B,T]=[%d,%d]", attentionMask.Shape, batch, tokens)
	}
	if attentionMask.Elements() != len(attentionMask.I32) {
		return nil, fmt.Errorf("bert_encoder_layer attention_mask element count %d does not match backing data", attentionMask.Elements())
	}
	if hidden%numAttentionHeads != 0 {
		return nil, fmt.Errorf("bert_encoder_layer hidden size %d must be divisible by num_attention_heads %d", hidden, numAttentionHeads)
	}
	headDim := hidden / numAttentionHeads
	if err := validateBERTDenseParam("encoder attention query", queryWeight, queryBias, hidden, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTDenseParam("encoder attention key", keyWeight, keyBias, hidden, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTDenseParam("encoder attention value", valueWeight, valueBias, hidden, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTDenseParam("encoder attention output", attentionOutputWeight, attentionOutputBias, hidden, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTLayerNormParam("encoder attention_layernorm_weight", attentionLayerNormWeight, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTLayerNormParam("encoder attention_layernorm_bias", attentionLayerNormBias, hidden); err != nil {
		return nil, err
	}
	intermediate := 0
	if intermediateWeight != nil && len(intermediateWeight.Shape) == 2 {
		intermediate = intermediateWeight.Shape[0]
	}
	if err := validateBERTDenseParam("encoder intermediate", intermediateWeight, intermediateBias, hidden, intermediate); err != nil {
		return nil, err
	}
	if err := validateBERTDenseParam("encoder output", outputWeight, outputBias, intermediate, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTLayerNormParam("encoder output_layernorm_weight", outputLayerNormWeight, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTLayerNormParam("encoder output_layernorm_bias", outputLayerNormBias, hidden); err != nil {
		return nil, err
	}
	if err := validateBERTEncoderLayerOutputType(hiddenStates.Shape, outTypes...); err != nil {
		return nil, err
	}

	rows := batch * tokens
	qkv, ok := denseHFManyWithProjector(projector, hiddenStates.F32, rows, hidden,
		[]*Tensor{queryWeight, keyWeight, valueWeight},
		[]*Tensor{queryBias, keyBias, valueBias},
	)
	if !ok {
		qkv = [][]float32{
			denseHF(hiddenStates.F32, rows, hidden, queryWeight, queryBias),
			denseHF(hiddenStates.F32, rows, hidden, keyWeight, keyBias),
			denseHF(hiddenStates.F32, rows, hidden, valueWeight, valueBias),
		}
	}
	q, k, v := qkv[0], qkv[1], qkv[2]
	context := bertSelfAttentionContext(q, k, v, attentionMask.I32, batch, tokens, hidden, numAttentionHeads, headDim)

	attentionProjected := denseHFWithProjector(projector, context, rows, hidden, attentionOutputWeight, attentionOutputBias)
	attentionNormInput := make([]float32, rows*hidden)
	for i := range attentionNormInput {
		attentionNormInput[i] = attentionProjected[i] + hiddenStates.F32[i]
	}
	attentionLayer := layerNormAffine(attentionNormInput, rows, hidden, attentionLayerNormWeight, attentionLayerNormBias, epsilon)

	intermediateValues := denseHFWithProjector(projector, attentionLayer, rows, hidden, intermediateWeight, intermediateBias)
	for i, value := range intermediateValues {
		intermediateValues[i] = exactGELU(value)
	}
	outputProjected := denseHFWithProjector(projector, intermediateValues, rows, intermediate, outputWeight, outputBias)
	outputNormInput := make([]float32, rows*hidden)
	for i := range outputNormInput {
		outputNormInput[i] = outputProjected[i] + attentionLayer[i]
	}
	out := NewTensorF32(hiddenStates.Shape, layerNormAffine(outputNormInput, rows, hidden, outputLayerNormWeight, outputLayerNormBias, epsilon))
	return out, nil
}

func denseHFWithProjector(projector bertDenseProjector, input []float32, rows, in int, weight, bias *Tensor) []float32 {
	if projector != nil {
		if out, ok := projector.DenseHF(input, rows, in, weight, bias); ok {
			return out
		}
	}
	return denseHF(input, rows, in, weight, bias)
}

func denseHFManyWithProjector(projector bertDenseProjector, input []float32, rows, in int, weights, biases []*Tensor) ([][]float32, bool) {
	if projector == nil {
		return nil, false
	}
	return projector.DenseHFMany(input, rows, in, weights, biases)
}

func bertSelfAttentionContext(q, k, v []float32, attentionMask []int32, batch, tokens, hidden, heads, headDim int) []float32 {
	context := make([]float32, batch*tokens*hidden)
	jobs := batch * tokens * heads
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || jobs < 64 || tokens < 16 {
		bertSelfAttentionContextSerial(context, q, k, v, attentionMask, batch, tokens, hidden, heads, headDim)
		return context
	}
	if workers > jobs {
		workers = jobs
	}
	chunk := (jobs + workers - 1) / workers
	var wg sync.WaitGroup
	for jobStart := 0; jobStart < jobs; jobStart += chunk {
		jobEnd := jobStart + chunk
		if jobEnd > jobs {
			jobEnd = jobs
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			logits := make([]float64, tokens)
			bertSelfAttentionContextRange(context, q, k, v, attentionMask, start, end, tokens, hidden, heads, headDim, logits)
		}(jobStart, jobEnd)
	}
	wg.Wait()
	return context
}

func bertSelfAttentionContextSerial(context, q, k, v []float32, attentionMask []int32, batch, tokens, hidden, heads, headDim int) {
	scale := 1 / math.Sqrt(float64(headDim))
	logits := make([]float64, tokens)
	for b := 0; b < batch; b++ {
		for i := 0; i < tokens; i++ {
			queryRow := b*tokens + i
			for head := 0; head < heads; head++ {
				activeKeys := 0
				maxLogit := math.Inf(-1)
				for j := 0; j < tokens; j++ {
					if attentionMask[b*tokens+j] == 0 {
						continue
					}
					activeKeys++
					keyRow := b*tokens + j
					dot := 0.0
					for d := 0; d < headDim; d++ {
						idx := head*headDim + d
						dot += float64(q[queryRow*hidden+idx]) * float64(k[keyRow*hidden+idx])
					}
					logit := dot * scale
					logits[j] = logit
					if logit > maxLogit {
						maxLogit = logit
					}
				}
				if activeKeys == 0 {
					continue
				}
				sumExp := 0.0
				for j := 0; j < tokens; j++ {
					if attentionMask[b*tokens+j] == 0 {
						continue
					}
					ev := math.Exp(logits[j] - maxLogit)
					logits[j] = ev
					sumExp += ev
				}
				for j := 0; j < tokens; j++ {
					if attentionMask[b*tokens+j] == 0 {
						continue
					}
					prob := logits[j] / sumExp
					valueRow := b*tokens + j
					for d := 0; d < headDim; d++ {
						idx := head*headDim + d
						context[queryRow*hidden+idx] += float32(prob * float64(v[valueRow*hidden+idx]))
					}
				}
			}
		}
	}
}

func bertSelfAttentionContextRange(context, q, k, v []float32, attentionMask []int32, jobStart, jobEnd, tokens, hidden, heads, headDim int, logits []float64) {
	scale := 1 / math.Sqrt(float64(headDim))
	for job := jobStart; job < jobEnd; job++ {
		head := job % heads
		queryIndex := job / heads
		batch := queryIndex / tokens
		queryToken := queryIndex % tokens
		queryRow := batch*tokens + queryToken

		activeKeys := 0
		maxLogit := math.Inf(-1)
		for keyToken := 0; keyToken < tokens; keyToken++ {
			if attentionMask[batch*tokens+keyToken] == 0 {
				continue
			}
			activeKeys++
			keyRow := batch*tokens + keyToken
			dot := 0.0
			for d := 0; d < headDim; d++ {
				idx := head*headDim + d
				dot += float64(q[queryRow*hidden+idx]) * float64(k[keyRow*hidden+idx])
			}
			logit := dot * scale
			logits[keyToken] = logit
			if logit > maxLogit {
				maxLogit = logit
			}
		}
		if activeKeys == 0 {
			continue
		}

		sumExp := 0.0
		for keyToken := 0; keyToken < tokens; keyToken++ {
			if attentionMask[batch*tokens+keyToken] == 0 {
				continue
			}
			ev := math.Exp(logits[keyToken] - maxLogit)
			logits[keyToken] = ev
			sumExp += ev
		}
		for keyToken := 0; keyToken < tokens; keyToken++ {
			if attentionMask[batch*tokens+keyToken] == 0 {
				continue
			}
			prob := logits[keyToken] / sumExp
			valueRow := batch*tokens + keyToken
			for d := 0; d < headDim; d++ {
				idx := head*headDim + d
				context[queryRow*hidden+idx] += float32(prob * float64(v[valueRow*hidden+idx]))
			}
		}
	}
}

func validateBERTDenseParam(name string, weight, bias *Tensor, in, out int) error {
	if weight == nil || bias == nil {
		return fmt.Errorf("bert_encoder_layer %s dense expects non-nil weight and bias", name)
	}
	if weight.DType != "f32" && weight.DType != "f16" {
		return fmt.Errorf("bert_encoder_layer %s weight dtype %q is not f32-compatible", name, weight.DType)
	}
	if bias.DType != "f32" && bias.DType != "f16" {
		return fmt.Errorf("bert_encoder_layer %s bias dtype %q is not f32-compatible", name, bias.DType)
	}
	if out <= 0 || in <= 0 {
		return fmt.Errorf("bert_encoder_layer %s dense shape must be positive, got out=%d in=%d", name, out, in)
	}
	if len(weight.Shape) != 2 || weight.Shape[0] != out || weight.Shape[1] != in {
		return fmt.Errorf("bert_encoder_layer %s weight shape %v does not match [out,in]=[%d,%d]", name, weight.Shape, out, in)
	}
	if len(bias.Shape) != 1 || bias.Shape[0] != out {
		return fmt.Errorf("bert_encoder_layer %s bias shape %v does not match [out]=[%d]", name, bias.Shape, out)
	}
	if weight.Elements() != len(weight.F32) {
		return fmt.Errorf("bert_encoder_layer %s weight element count %d does not match backing data", name, weight.Elements())
	}
	if bias.Elements() != len(bias.F32) {
		return fmt.Errorf("bert_encoder_layer %s bias element count %d does not match backing data", name, bias.Elements())
	}
	return nil
}

func validateBERTEncoderLayerOutputType(shape []int, outTypes ...eosartifact.ValueType) error {
	if len(outTypes) == 0 || outTypes[0].Tensor == nil {
		return nil
	}
	tensorType := outTypes[0].Tensor
	if tensorType.DType != "" && tensorType.DType != "f32" && tensorType.DType != "f16" {
		return fmt.Errorf("bert_encoder_layer output dtype %q is not f32-compatible", tensorType.DType)
	}
	if len(tensorType.Shape) != 0 && len(tensorType.Shape) != len(shape) {
		return fmt.Errorf("bert_encoder_layer output rank %d does not match shape %v", len(tensorType.Shape), shape)
	}
	return nil
}

func denseHF(input []float32, rows, in int, weight, bias *Tensor) []float32 {
	out := weight.Shape[0]
	result := make([]float32, rows*out)
	workItems := rows * in * out
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || rows < 16 || workItems < 1<<20 {
		denseHFRange(input, result, 0, rows, in, out, weight.F32, bias.F32)
		return result
	}
	if workers > rows {
		workers = rows
	}
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for rowStart := 0; rowStart < rows; rowStart += chunk {
		rowEnd := rowStart + chunk
		if rowEnd > rows {
			rowEnd = rows
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			denseHFRange(input, result, start, end, in, out, weight.F32, bias.F32)
		}(rowStart, rowEnd)
	}
	wg.Wait()
	return result
}

func denseHFRange(input, result []float32, rowStart, rowEnd, in, out int, weight, bias []float32) {
	for r := rowStart; r < rowEnd; r++ {
		for o := 0; o < out; o++ {
			sum := float64(bias[o])
			for c := 0; c < in; c++ {
				sum += float64(input[r*in+c]) * float64(weight[o*in+c])
			}
			result[r*out+o] = float32(sum)
		}
	}
}

type acceleratedBERTDenseProjector struct {
	accel   MatMulAccelerator
	backend eosartifact.BackendKind
	prefix  string
	nextID  atomic.Int64
	clock   uint64
	max     int

	mu    sync.Mutex
	runMu sync.Mutex
	bound map[*Tensor]bertDenseBinding
}

var (
	defaultBERTDenseProjectorOnce sync.Once
	defaultBERTDenseProjectorInst bertDenseProjector
)

const defaultBERTDenseMaxResidentBindings = 256

type bertDenseBinding struct {
	name     string
	fp       bertTensorFingerprint
	lastUsed uint64
}

type bertTensorFingerprint struct {
	dtype    string
	shape    [2]int
	elements int
	hash     uint64
}

func defaultBERTDenseProjector() bertDenseProjector {
	defaultBERTDenseProjectorOnce.Do(func() {
		if bertDenseAccelerationEnabled() {
			defaultBERTDenseProjectorInst = newAcceleratedBERTDenseProjector()
		}
	})
	return defaultBERTDenseProjectorInst
}

func bertDenseAccelerationEnabled() bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("EOS_BERT_DENSE_ACCEL")))
	return value != "0" && value != "false" && value != "off" && value != "disabled"
}

func newAcceleratedBERTDenseProjector() bertDenseProjector {
	accel, kind, err := NewPreferredMatMulAccelerator(eosartifact.BackendCUDA)
	if err != nil || accel == nil {
		return nil
	}
	return &acceleratedBERTDenseProjector{
		accel:   accel,
		backend: kind,
		prefix:  fmt.Sprintf("bert_dense_%s", kind),
		max:     defaultBERTDenseMaxResidentBindings,
		bound:   map[*Tensor]bertDenseBinding{},
	}
}

func (p *acceleratedBERTDenseProjector) DenseHF(input []float32, rows, in int, weight, bias *Tensor) ([]float32, bool) {
	outputs, ok := p.DenseHFMany(input, rows, in, []*Tensor{weight}, []*Tensor{bias})
	if !ok || len(outputs) != 1 {
		return nil, false
	}
	return outputs[0], true
}

func (p *acceleratedBERTDenseProjector) DenseHFMany(input []float32, rows, in int, weights, biases []*Tensor) ([][]float32, bool) {
	if p == nil || p.accel == nil || len(weights) == 0 || len(weights) != len(biases) {
		return nil, false
	}
	lhs := NewTensorF32([]int{rows, in}, input)
	outShapes := make([][]string, len(weights))
	names := make([]string, len(weights))
	newlyBound := make([]string, 0, len(weights))
	p.runMu.Lock()
	defer p.runMu.Unlock()
	succeeded := false
	defer func() {
		if !succeeded && len(newlyBound) > 0 {
			p.unbindNames(newlyBound)
		}
	}()
	for i := range weights {
		weight, bias := weights[i], biases[i]
		if !canAccelerateBERTDense(rows, in, input, weight, bias) {
			return nil, false
		}
		outShapes[i] = []string{strconv.Itoa(rows), strconv.Itoa(weight.Shape[0])}
		name, bound, ok := p.boundName(weight)
		if !ok {
			return nil, false
		}
		if bound {
			newlyBound = append(newlyBound, name)
		}
		names[i] = name
	}
	outType := eosartifact.ValueType{Tensor: &eosartifact.TensorType{DType: "f32", Shape: outShapes[0]}}
	var results []StepDispatchResult
	if len(names) > 1 {
		multi, ok := p.accel.(MultiBoundRightMatMulAccelerator)
		if !ok {
			return nil, false
		}
		got, err := multi.RunMatMulWithBoundRights(lhs, names, outType, false, true)
		if err != nil {
			return nil, false
		}
		results = got
	} else {
		result, err := p.accel.RunMatMulWithBoundRight(lhs, names[0], outType, false, true)
		if err != nil {
			return nil, false
		}
		results = []StepDispatchResult{result}
	}
	if len(results) != len(weights) {
		return nil, false
	}
	out := make([][]float32, len(results))
	for i, result := range results {
		if len(result.Outputs) != 1 || result.Outputs[0] == nil {
			return nil, false
		}
		values := append([]float32(nil), result.Outputs[0].F32...)
		addDenseBias(values, biases[i].F32, weights[i].Shape[0])
		out[i] = values
	}
	succeeded = true
	return out, true
}

func (p *acceleratedBERTDenseProjector) boundName(weight *Tensor) (string, bool, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fp := bertDenseWeightFingerprint(weight)
	p.clock++
	now := p.clock
	if binding, ok := p.bound[weight]; ok {
		if binding.fp == fp {
			binding.lastUsed = now
			p.bound[weight] = binding
			return binding.name, false, true
		}
		_ = p.accel.UnbindMatrix(binding.name)
		delete(p.bound, weight)
	}
	name := fmt.Sprintf("%s_%d", p.prefix, p.nextID.Add(1))
	if err := p.accel.BindMatrix(name, weight); err != nil {
		return "", false, false
	}
	p.bound[weight] = bertDenseBinding{name: name, fp: fp, lastUsed: now}
	p.evictLocked()
	return name, true, true
}

func (p *acceleratedBERTDenseProjector) unbindNames(names []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	remove := map[string]bool{}
	for _, name := range names {
		remove[name] = true
		_ = p.accel.UnbindMatrix(name)
	}
	for tensor, binding := range p.bound {
		if remove[binding.name] {
			delete(p.bound, tensor)
		}
	}
}

func (p *acceleratedBERTDenseProjector) evictLocked() {
	max := p.max
	if max <= 0 {
		max = defaultBERTDenseMaxResidentBindings
	}
	for len(p.bound) > max {
		var (
			victimTensor *Tensor
			victim       bertDenseBinding
			haveVictim   bool
		)
		for tensor, binding := range p.bound {
			if !haveVictim || binding.lastUsed < victim.lastUsed {
				victimTensor = tensor
				victim = binding
				haveVictim = true
			}
		}
		if !haveVictim {
			return
		}
		_ = p.accel.UnbindMatrix(victim.name)
		delete(p.bound, victimTensor)
	}
}

func bertDenseWeightFingerprint(weight *Tensor) bertTensorFingerprint {
	fp := bertTensorFingerprint{dtype: weight.DType, elements: len(weight.F32)}
	if len(weight.Shape) > 0 {
		fp.shape[0] = weight.Shape[0]
	}
	if len(weight.Shape) > 1 {
		fp.shape[1] = weight.Shape[1]
	}
	const offset64 = 1469598103934665603
	const prime64 = 1099511628211
	hash := uint64(offset64)
	mix := func(v uint64) {
		for i := 0; i < 8; i++ {
			hash ^= uint64(byte(v))
			hash *= prime64
			v >>= 8
		}
	}
	mix(uint64(len(weight.Shape)))
	for _, dim := range weight.Shape {
		mix(uint64(dim))
	}
	for _, value := range weight.F32 {
		mix(uint64(math.Float32bits(value)))
	}
	fp.hash = bits.RotateLeft64(hash, int(uint(fp.shape[0]+fp.shape[1]+fp.elements)&63))
	return fp
}

func canAccelerateBERTDense(rows, in int, input []float32, weight, bias *Tensor) bool {
	return rows > 0 &&
		in > 0 &&
		len(input) == rows*in &&
		weight != nil &&
		bias != nil &&
		weight.DType == "f32" &&
		bias.DType == "f32" &&
		len(weight.Shape) == 2 &&
		weight.Shape[1] == in &&
		weight.Elements() == len(weight.F32) &&
		len(bias.Shape) == 1 &&
		bias.Shape[0] == weight.Shape[0] &&
		bias.Elements() == len(bias.F32)
}

func addDenseBias(values, bias []float32, out int) {
	if out <= 0 {
		return
	}
	for i := range values {
		values[i] += bias[i%out]
	}
}

func layerNormAffine(input []float32, rows, dim int, gamma, beta *Tensor, epsilon float64) []float32 {
	out := make([]float32, len(input))
	for r := 0; r < rows; r++ {
		base := r * dim
		mean := 0.0
		for d := 0; d < dim; d++ {
			mean += float64(input[base+d])
		}
		mean /= float64(dim)
		variance := 0.0
		for d := 0; d < dim; d++ {
			centered := float64(input[base+d]) - mean
			variance += centered * centered
		}
		variance /= float64(dim)
		invStd := 1 / math.Sqrt(variance+epsilon)
		for d := 0; d < dim; d++ {
			normalized := (float64(input[base+d]) - mean) * invStd
			out[base+d] = float32(normalized*float64(gamma.F32[d]) + float64(beta.F32[d]))
		}
	}
	return out
}

func exactGELU(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}
