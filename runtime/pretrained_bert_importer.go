package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const PretrainedBERTImporterPlanVersion = "manta/pretrained-bert-import-plan/v0alpha1"

type PretrainedBERTConfig struct {
	Architectures          []string `json:"architectures,omitempty"`
	ModelType              string   `json:"model_type,omitempty"`
	VocabSize              int      `json:"vocab_size"`
	HiddenSize             int      `json:"hidden_size"`
	NumHiddenLayers        int      `json:"num_hidden_layers"`
	NumAttentionHeads      int      `json:"num_attention_heads"`
	IntermediateSize       int      `json:"intermediate_size"`
	HiddenAct              string   `json:"hidden_act,omitempty"`
	HiddenDropoutProb      float64  `json:"hidden_dropout_prob,omitempty"`
	AttentionDropoutProb   float64  `json:"attention_probs_dropout_prob,omitempty"`
	MaxPositionEmbeddings  int      `json:"max_position_embeddings"`
	TypeVocabSize          int      `json:"type_vocab_size"`
	LayerNormEps           float64  `json:"layer_norm_eps,omitempty"`
	PositionEmbeddingType  string   `json:"position_embedding_type,omitempty"`
	PadTokenID             int      `json:"pad_token_id,omitempty"`
	BOSPositionEmbeddingID int      `json:"bos_token_id,omitempty"`
}

type PretrainedBERTTensorPlan struct {
	Name     string `json:"name"`
	Shape    []int  `json:"shape"`
	Required bool   `json:"required"`
	Role     string `json:"role"`
}

type PretrainedBERTImportPlan struct {
	Version                string                     `json:"version"`
	ModelName              string                     `json:"model_name,omitempty"`
	Architecture           string                     `json:"architecture"`
	Config                 PretrainedBERTConfig       `json:"config"`
	Tensors                []PretrainedBERTTensorPlan `json:"tensors"`
	PoolingPolicy          string                     `json:"pooling_policy"`
	OutputProjectionPolicy string                     `json:"output_projection_policy"`
	ExecutionStatus        string                     `json:"execution_status"`
}

func LoadPretrainedBERTConfig(path string) (PretrainedBERTConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PretrainedBERTConfig{}, err
	}
	var cfg PretrainedBERTConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return PretrainedBERTConfig{}, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := cfg.Validate(); err != nil {
		return PretrainedBERTConfig{}, fmt.Errorf("validate %s: %w", path, err)
	}
	return cfg, nil
}

func PlanPretrainedBERTImportFromDir(dir, modelName string) (PretrainedBERTImportPlan, error) {
	cfg, err := LoadPretrainedBERTConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return PretrainedBERTImportPlan{}, err
	}
	return PlanPretrainedBERTImport(cfg, modelName)
}

func PlanPretrainedBERTImport(cfg PretrainedBERTConfig, modelName string) (PretrainedBERTImportPlan, error) {
	if err := cfg.Validate(); err != nil {
		return PretrainedBERTImportPlan{}, err
	}
	architecture, err := supportedBERTArchitecture(cfg)
	if err != nil {
		return PretrainedBERTImportPlan{}, err
	}
	hidden := cfg.HiddenSize
	intermediate := cfg.IntermediateSize
	tensors := []PretrainedBERTTensorPlan{
		requiredBERTTensor("embeddings.word_embeddings.weight", "token_embeddings", cfg.VocabSize, hidden),
		requiredBERTTensor("embeddings.position_embeddings.weight", "position_embeddings", cfg.MaxPositionEmbeddings, hidden),
		requiredBERTTensor("embeddings.token_type_embeddings.weight", "token_type_embeddings", cfg.TypeVocabSize, hidden),
		requiredBERTTensor("embeddings.LayerNorm.weight", "embedding_layernorm_weight", hidden),
		requiredBERTTensor("embeddings.LayerNorm.bias", "embedding_layernorm_bias", hidden),
	}
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", layer)
		rolePrefix := fmt.Sprintf("encoder_layer_%d_", layer)
		tensors = append(tensors,
			requiredBERTTensor(prefix+"attention.self.query.weight", rolePrefix+"attention_query_weight", hidden, hidden),
			requiredBERTTensor(prefix+"attention.self.query.bias", rolePrefix+"attention_query_bias", hidden),
			requiredBERTTensor(prefix+"attention.self.key.weight", rolePrefix+"attention_key_weight", hidden, hidden),
			requiredBERTTensor(prefix+"attention.self.key.bias", rolePrefix+"attention_key_bias", hidden),
			requiredBERTTensor(prefix+"attention.self.value.weight", rolePrefix+"attention_value_weight", hidden, hidden),
			requiredBERTTensor(prefix+"attention.self.value.bias", rolePrefix+"attention_value_bias", hidden),
			requiredBERTTensor(prefix+"attention.output.dense.weight", rolePrefix+"attention_output_weight", hidden, hidden),
			requiredBERTTensor(prefix+"attention.output.dense.bias", rolePrefix+"attention_output_bias", hidden),
			requiredBERTTensor(prefix+"attention.output.LayerNorm.weight", rolePrefix+"attention_layernorm_weight", hidden),
			requiredBERTTensor(prefix+"attention.output.LayerNorm.bias", rolePrefix+"attention_layernorm_bias", hidden),
			requiredBERTTensor(prefix+"intermediate.dense.weight", rolePrefix+"intermediate_weight", intermediate, hidden),
			requiredBERTTensor(prefix+"intermediate.dense.bias", rolePrefix+"intermediate_bias", intermediate),
			requiredBERTTensor(prefix+"output.dense.weight", rolePrefix+"output_weight", hidden, intermediate),
			requiredBERTTensor(prefix+"output.dense.bias", rolePrefix+"output_bias", hidden),
			requiredBERTTensor(prefix+"output.LayerNorm.weight", rolePrefix+"output_layernorm_weight", hidden),
			requiredBERTTensor(prefix+"output.LayerNorm.bias", rolePrefix+"output_layernorm_bias", hidden),
		)
	}
	tensors = append(tensors,
		optionalBERTTensor("pooler.dense.weight", "pooler_weight", hidden, hidden),
		optionalBERTTensor("pooler.dense.bias", "pooler_bias", hidden),
	)
	return PretrainedBERTImportPlan{
		Version:                PretrainedBERTImporterPlanVersion,
		ModelName:              modelName,
		Architecture:           architecture,
		Config:                 cfg,
		Tensors:                tensors,
		PoolingPolicy:          "deferred: sentence-transformers pooling metadata is not imported yet; BGE/E5 typically use masked mean pooling plus L2 normalization",
		OutputProjectionPolicy: "deferred: dense/sparse projection heads and safetensors ingestion are outside this first planning slice",
		ExecutionStatus:        "plan_only: tokenizer/config/tensor metadata gate; no BERT graph execution or weight import",
	}, nil
}

func (cfg PretrainedBERTConfig) Validate() error {
	if cfg.ModelType != "" && cfg.ModelType != "bert" {
		return fmt.Errorf("unsupported model_type %q; only bert is supported", cfg.ModelType)
	}
	if cfg.VocabSize <= 0 {
		return fmt.Errorf("vocab_size must be positive")
	}
	if cfg.HiddenSize <= 0 {
		return fmt.Errorf("hidden_size must be positive")
	}
	if cfg.NumHiddenLayers <= 0 {
		return fmt.Errorf("num_hidden_layers must be positive")
	}
	if cfg.NumAttentionHeads <= 0 {
		return fmt.Errorf("num_attention_heads must be positive")
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return fmt.Errorf("hidden_size %d must be divisible by num_attention_heads %d", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	if cfg.IntermediateSize <= 0 {
		return fmt.Errorf("intermediate_size must be positive")
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("max_position_embeddings must be positive")
	}
	if cfg.TypeVocabSize <= 0 {
		return fmt.Errorf("type_vocab_size must be positive")
	}
	if cfg.HiddenAct != "" && !supportedBERTHiddenAct(cfg.HiddenAct) {
		return fmt.Errorf("unsupported hidden_act %q", cfg.HiddenAct)
	}
	if cfg.PositionEmbeddingType != "" && cfg.PositionEmbeddingType != "absolute" {
		return fmt.Errorf("unsupported position_embedding_type %q", cfg.PositionEmbeddingType)
	}
	return nil
}

func supportedBERTArchitecture(cfg PretrainedBERTConfig) (string, error) {
	if len(cfg.Architectures) == 0 {
		return "BertModel", nil
	}
	supported := []string{"BertModel", "BertForMaskedLM"}
	for _, architecture := range cfg.Architectures {
		if slices.Contains(supported, architecture) {
			return architecture, nil
		}
	}
	return "", fmt.Errorf("unsupported architectures %s; supported BERT planning architectures: %s", strings.Join(cfg.Architectures, ", "), strings.Join(supported, ", "))
}

func supportedBERTHiddenAct(name string) bool {
	switch name {
	case "gelu", "gelu_new", "gelu_fast", "relu":
		return true
	default:
		return false
	}
}

func requiredBERTTensor(name, role string, shape ...int) PretrainedBERTTensorPlan {
	return PretrainedBERTTensorPlan{Name: name, Shape: append([]int(nil), shape...), Required: true, Role: role}
}

func optionalBERTTensor(name, role string, shape ...int) PretrainedBERTTensorPlan {
	return PretrainedBERTTensorPlan{Name: name, Shape: append([]int(nil), shape...), Required: false, Role: role}
}
