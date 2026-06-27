package backend

import (
	"fmt"
	"strconv"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func bertEmbedderTensor(inputs []*Tensor, outputType eosartifact.ValueType, attrs map[string]string) (*Tensor, error) {
	cfg, err := bertEmbedderConfigFromAttrs(attrs)
	if err != nil {
		return nil, err
	}
	expected := 8 + cfg.NumHiddenLayers*16
	if len(inputs) != expected {
		return nil, fmt.Errorf("bert_embedder expects %d tensors for %d layers, got %d", expected, cfg.NumHiddenLayers, len(inputs))
	}
	return BERTEmbedderReferenceWithPooling(
		inputs[0], inputs[1], inputs[2],
		inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
		inputs[8:],
		cfg.NumHiddenLayers, cfg.NumAttentionHeads, cfg.Epsilon, cfg.HiddenAct, cfg.Pooling, outputType,
	)
}

type bertEmbedderConfig struct {
	NumHiddenLayers   int
	NumAttentionHeads int
	Epsilon           float64
	HiddenAct         string
	Pooling           string
}

func bertEmbedderConfigFromAttrs(attrs map[string]string) (bertEmbedderConfig, error) {
	cfg := bertEmbedderConfig{NumHiddenLayers: 1, NumAttentionHeads: 1, Epsilon: 1e-12, HiddenAct: "gelu", Pooling: "masked_mean"}
	if attrs == nil {
		return cfg, nil
	}
	if attrs["num_hidden_layers"] != "" {
		layers, err := strconv.Atoi(attrs["num_hidden_layers"])
		if err != nil || layers <= 0 {
			return cfg, fmt.Errorf("bert_embedder num_hidden_layers %q is invalid", attrs["num_hidden_layers"])
		}
		cfg.NumHiddenLayers = layers
	}
	if attrs["num_attention_heads"] != "" {
		heads, err := strconv.Atoi(attrs["num_attention_heads"])
		if err != nil || heads <= 0 {
			return cfg, fmt.Errorf("bert_embedder num_attention_heads %q is invalid", attrs["num_attention_heads"])
		}
		cfg.NumAttentionHeads = heads
	}
	if attrs["epsilon"] != "" {
		epsilon, err := strconv.ParseFloat(attrs["epsilon"], 64)
		if err != nil {
			return cfg, fmt.Errorf("bert_embedder epsilon %q is invalid: %w", attrs["epsilon"], err)
		}
		if epsilon < 0 {
			return cfg, fmt.Errorf("bert_embedder epsilon must be non-negative, got %g", epsilon)
		}
		cfg.Epsilon = epsilon
	}
	if attrs["hidden_act"] != "" {
		cfg.HiddenAct = attrs["hidden_act"]
	}
	if attrs["pooling"] != "" {
		cfg.Pooling = attrs["pooling"]
	}
	if cfg.HiddenAct != "gelu" {
		return cfg, fmt.Errorf("bert_embedder unsupported hidden_act %q; only gelu is supported", cfg.HiddenAct)
	}
	if cfg.Pooling != "masked_mean" && cfg.Pooling != "cls" {
		return cfg, fmt.Errorf("bert_embedder unsupported pooling %q; supported values are masked_mean and cls", cfg.Pooling)
	}
	return cfg, nil
}

// BERTEmbedderReference composes BERT embeddings, every configured encoder
// layer, masked mean pooling, and row L2 normalization. Position IDs are
// generated as 0..T-1 for each batch row.
func BERTEmbedderReference(
	inputIDs, attentionMask, tokenTypeIDs *Tensor,
	tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings, embeddingLayerNormWeight, embeddingLayerNormBias *Tensor,
	layerWeights []*Tensor,
	numHiddenLayers, numAttentionHeads int,
	epsilon float64,
	hiddenAct string,
	outTypes ...eosartifact.ValueType,
) (*Tensor, error) {
	return BERTEmbedderReferenceWithPooling(
		inputIDs, attentionMask, tokenTypeIDs,
		tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings, embeddingLayerNormWeight, embeddingLayerNormBias,
		layerWeights,
		numHiddenLayers, numAttentionHeads, epsilon, hiddenAct, "masked_mean", outTypes...,
	)
}

// BERTEmbedderReferenceWithPooling composes BERT embeddings, every configured
// encoder layer, requested sentence pooling, and row L2 normalization.
func BERTEmbedderReferenceWithPooling(
	inputIDs, attentionMask, tokenTypeIDs *Tensor,
	tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings, embeddingLayerNormWeight, embeddingLayerNormBias *Tensor,
	layerWeights []*Tensor,
	numHiddenLayers, numAttentionHeads int,
	epsilon float64,
	hiddenAct string,
	pooling string,
	outTypes ...eosartifact.ValueType,
) (*Tensor, error) {
	if inputIDs == nil || attentionMask == nil || tokenTypeIDs == nil {
		return nil, fmt.Errorf("bert_embedder expects non-nil input_ids, attention_mask, and token_type_ids")
	}
	if inputIDs.DType != "i32" {
		return nil, fmt.Errorf("bert_embedder input_ids dtype %q is not i32", inputIDs.DType)
	}
	if len(inputIDs.Shape) != 2 {
		return nil, fmt.Errorf("bert_embedder input_ids must be rank-2 [B,T], got shape %v", inputIDs.Shape)
	}
	if inputIDs.Elements() != len(inputIDs.I32) {
		return nil, fmt.Errorf("bert_embedder input_ids element count %d does not match backing data", inputIDs.Elements())
	}
	if !inputIDs.EqualShape(attentionMask) {
		return nil, fmt.Errorf("bert_embedder attention_mask shape %v does not match input_ids shape %v", attentionMask.Shape, inputIDs.Shape)
	}
	if !inputIDs.EqualShape(tokenTypeIDs) {
		return nil, fmt.Errorf("bert_embedder token_type_ids shape %v does not match input_ids shape %v", tokenTypeIDs.Shape, inputIDs.Shape)
	}
	if attentionMask.DType != "i32" {
		return nil, fmt.Errorf("bert_embedder attention_mask dtype %q is not i32", attentionMask.DType)
	}
	if tokenTypeIDs.DType != "i32" {
		return nil, fmt.Errorf("bert_embedder token_type_ids dtype %q is not i32", tokenTypeIDs.DType)
	}
	if attentionMask.Elements() != len(attentionMask.I32) {
		return nil, fmt.Errorf("bert_embedder attention_mask element count %d does not match backing data", attentionMask.Elements())
	}
	if tokenTypeIDs.Elements() != len(tokenTypeIDs.I32) {
		return nil, fmt.Errorf("bert_embedder token_type_ids element count %d does not match backing data", tokenTypeIDs.Elements())
	}
	if numHiddenLayers <= 0 {
		return nil, fmt.Errorf("bert_embedder num_hidden_layers must be positive, got %d", numHiddenLayers)
	}
	if len(layerWeights) != numHiddenLayers*16 {
		return nil, fmt.Errorf("bert_embedder expects %d encoder layer tensors, got %d", numHiddenLayers*16, len(layerWeights))
	}
	if pooling == "" {
		pooling = "masked_mean"
	}
	if pooling != "masked_mean" && pooling != "cls" {
		return nil, fmt.Errorf("bert_embedder unsupported pooling %q; supported values are masked_mean and cls", pooling)
	}
	positionIDs := generatedBERTPositionIDs(inputIDs.Shape[0], inputIDs.Shape[1])
	hiddenStates, err := BERTEmbeddingsReference(
		tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings, embeddingLayerNormWeight, embeddingLayerNormBias,
		inputIDs, positionIDs, tokenTypeIDs, epsilon,
	)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < numHiddenLayers; layer++ {
		base := layer * 16
		hiddenStates, err = BERTEncoderLayerReference(
			hiddenStates, attentionMask,
			layerWeights[base+0], layerWeights[base+1],
			layerWeights[base+2], layerWeights[base+3],
			layerWeights[base+4], layerWeights[base+5],
			layerWeights[base+6], layerWeights[base+7],
			layerWeights[base+8], layerWeights[base+9],
			layerWeights[base+10], layerWeights[base+11],
			layerWeights[base+12], layerWeights[base+13],
			layerWeights[base+14], layerWeights[base+15],
			numAttentionHeads, epsilon, hiddenAct,
		)
		if err != nil {
			return nil, fmt.Errorf("bert_embedder encoder layer %d: %w", layer, err)
		}
	}
	var pooled *Tensor
	switch pooling {
	case "masked_mean":
		pooled, err = meanPoolMaskedTensor(hiddenStates, attentionMask)
		if err != nil {
			return nil, err
		}
	case "cls":
		pooled, err = clsPoolTensor(hiddenStates)
		if err != nil {
			return nil, err
		}
	}
	normalized := normalizeRows(pooled)
	if err := validateBERTEmbedderOutputType(normalized.Shape, outTypes...); err != nil {
		return nil, err
	}
	return normalized, nil
}

func clsPoolTensor(hiddenStates *Tensor) (*Tensor, error) {
	if hiddenStates == nil || hiddenStates.DType != "f32" || len(hiddenStates.Shape) != 3 {
		dtype := "<nil>"
		var shape []int
		if hiddenStates != nil {
			dtype = hiddenStates.DType
			shape = hiddenStates.Shape
		}
		return nil, fmt.Errorf("cls pooling expects f32 hidden_states [B,T,H], got dtype=%q shape=%v", dtype, shape)
	}
	batch, tokens, hidden := hiddenStates.Shape[0], hiddenStates.Shape[1], hiddenStates.Shape[2]
	if tokens <= 0 || hidden <= 0 || hiddenStates.Elements() != len(hiddenStates.F32) {
		return nil, fmt.Errorf("cls pooling hidden_states backing data length %d does not match shape %v", len(hiddenStates.F32), hiddenStates.Shape)
	}
	out := make([]float32, batch*hidden)
	for b := 0; b < batch; b++ {
		copy(out[b*hidden:(b+1)*hidden], hiddenStates.F32[b*tokens*hidden:b*tokens*hidden+hidden])
	}
	return NewTensorF32([]int{batch, hidden}, out), nil
}

func generatedBERTPositionIDs(batch, tokens int) *Tensor {
	ids := make([]int32, batch*tokens)
	for b := 0; b < batch; b++ {
		for t := 0; t < tokens; t++ {
			ids[b*tokens+t] = int32(t)
		}
	}
	return NewTensorI32([]int{batch, tokens}, ids)
}

func validateBERTEmbedderOutputType(shape []int, outTypes ...eosartifact.ValueType) error {
	if len(outTypes) == 0 || outTypes[0].Tensor == nil {
		return nil
	}
	tensorType := outTypes[0].Tensor
	if tensorType.DType != "" && tensorType.DType != "f32" && tensorType.DType != "f16" {
		return fmt.Errorf("bert_embedder output dtype %q is not f32-compatible", tensorType.DType)
	}
	if len(tensorType.Shape) != 0 && len(tensorType.Shape) != len(shape) {
		return fmt.Errorf("bert_embedder output rank %d does not match shape %v", len(tensorType.Shape), shape)
	}
	return nil
}
