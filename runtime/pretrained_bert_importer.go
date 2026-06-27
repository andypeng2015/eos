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
	Version                string                            `json:"version"`
	ModelName              string                            `json:"model_name,omitempty"`
	Architecture           string                            `json:"architecture"`
	Config                 PretrainedBERTConfig              `json:"config"`
	Tensors                []PretrainedBERTTensorPlan        `json:"tensors"`
	WeightVerification     *PretrainedBERTWeightVerification `json:"weight_verification,omitempty"`
	WeightLoadSmoke        *PretrainedBERTWeightLoadReport   `json:"weight_load_smoke,omitempty"`
	PoolingPolicy          string                            `json:"pooling_policy"`
	OutputProjectionPolicy string                            `json:"output_projection_policy"`
	ExecutionStatus        string                            `json:"execution_status"`
}

type PretrainedBERTWeightVerification struct {
	Status          string                        `json:"status"`
	Files           []string                      `json:"files,omitempty"`
	TensorCount     int                           `json:"tensor_count"`
	Missing         []string                      `json:"missing,omitempty"`
	Unexpected      []string                      `json:"unexpected,omitempty"`
	ShapeMismatches []PretrainedBERTShapeMismatch `json:"shape_mismatches,omitempty"`
	DTypeMismatches []PretrainedBERTDTypeMismatch `json:"dtype_mismatches,omitempty"`
}

type PretrainedBERTShapeMismatch struct {
	Name     string  `json:"name"`
	Expected []int   `json:"expected"`
	Actual   []int64 `json:"actual"`
}

type PretrainedBERTDTypeMismatch struct {
	Name       string   `json:"name"`
	Actual     string   `json:"actual"`
	Acceptable []string `json:"acceptable"`
}

type PretrainedBERTWeightTensor struct {
	Name       string  `json:"name"`
	Role       string  `json:"role"`
	DType      string  `json:"dtype"`
	Shape      []int64 `json:"shape"`
	SourceFile string  `json:"source_file"`
	ByteOffset int64   `json:"byte_offset"`
	ByteLength int64   `json:"byte_length"`
	Bytes      []byte  `json:"-"`
}

type PretrainedBERTWeightSet struct {
	Tensors []PretrainedBERTWeightTensor `json:"tensors"`
}

type PretrainedBERTWeightLoadReport struct {
	Status       string   `json:"status"`
	Files        []string `json:"files,omitempty"`
	TensorCount  int      `json:"tensor_count"`
	TotalBytes   int64    `json:"total_bytes"`
	Loaded       []string `json:"loaded,omitempty"`
	SkippedExtra []string `json:"skipped_extra,omitempty"`
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

func VerifyPretrainedBERTWeightsFromDir(dir string, plan PretrainedBERTImportPlan) (PretrainedBERTWeightVerification, error) {
	collection, err := ReadSafeTensorsCollectionFromDir(dir)
	if err != nil {
		return PretrainedBERTWeightVerification{}, err
	}
	report := VerifyPretrainedBERTWeights(plan, collection)
	return report, nil
}

func VerifyPretrainedBERTWeights(plan PretrainedBERTImportPlan, metadata SafeTensorsCollection) PretrainedBERTWeightVerification {
	report := PretrainedBERTWeightVerification{
		Status:      "ok",
		TensorCount: len(metadata.Tensors),
		Files:       append([]string(nil), metadata.Files...),
	}
	planned := make(map[string]PretrainedBERTTensorPlan, len(plan.Tensors))
	for _, tensor := range plan.Tensors {
		planned[tensor.Name] = tensor
		actual, ok := metadata.Tensors[tensor.Name]
		if !ok {
			if tensor.Required {
				report.Missing = append(report.Missing, tensor.Name)
			}
			continue
		}
		if !sameBERTTensorShape(tensor.Shape, actual.Shape) {
			report.ShapeMismatches = append(report.ShapeMismatches, PretrainedBERTShapeMismatch{
				Name:     tensor.Name,
				Expected: append([]int(nil), tensor.Shape...),
				Actual:   append([]int64(nil), actual.Shape...),
			})
		}
		if !acceptablePretrainedBERTDType(actual.DType) {
			report.DTypeMismatches = append(report.DTypeMismatches, PretrainedBERTDTypeMismatch{
				Name:       tensor.Name,
				Actual:     actual.DType,
				Acceptable: acceptablePretrainedBERTDTypes(),
			})
		}
	}
	for name := range metadata.Tensors {
		if _, ok := planned[name]; !ok {
			report.Unexpected = append(report.Unexpected, name)
		}
	}
	slices.Sort(report.Missing)
	slices.Sort(report.Unexpected)
	sortBERTShapeMismatches(report.ShapeMismatches)
	sortBERTDTypeMismatches(report.DTypeMismatches)
	if len(report.Missing) > 0 || len(report.Unexpected) > 0 || len(report.ShapeMismatches) > 0 || len(report.DTypeMismatches) > 0 {
		report.Status = "mismatch"
	}
	return report
}

func LoadPretrainedBERTWeightsFromDir(dir string, plan PretrainedBERTImportPlan) (PretrainedBERTWeightSet, PretrainedBERTWeightLoadReport, error) {
	collection, err := ReadSafeTensorsCollectionFromDir(dir)
	if err != nil {
		return PretrainedBERTWeightSet{}, PretrainedBERTWeightLoadReport{}, err
	}
	verification := VerifyPretrainedBERTWeights(plan, collection)
	if len(verification.Missing) > 0 || len(verification.ShapeMismatches) > 0 || len(verification.DTypeMismatches) > 0 {
		return PretrainedBERTWeightSet{}, PretrainedBERTWeightLoadReport{}, fmt.Errorf("pretrained BERT weight verification failed before byte ingestion: missing=%d shape_mismatches=%d dtype_mismatches=%d", len(verification.Missing), len(verification.ShapeMismatches), len(verification.DTypeMismatches))
	}
	requestedPlans := make([]PretrainedBERTTensorPlan, 0, len(plan.Tensors))
	plannedNames := make(map[string]struct{}, len(plan.Tensors))
	for _, tensor := range plan.Tensors {
		plannedNames[tensor.Name] = struct{}{}
		if tensor.Required {
			requestedPlans = append(requestedPlans, tensor)
			continue
		}
		if _, ok := collection.Tensors[tensor.Name]; ok {
			requestedPlans = append(requestedPlans, tensor)
		}
	}
	requestedNames := make([]string, 0, len(requestedPlans))
	roles := make(map[string]string, len(requestedPlans))
	for _, tensor := range requestedPlans {
		requestedNames = append(requestedNames, tensor.Name)
		roles[tensor.Name] = tensor.Role
	}
	slices.Sort(requestedNames)
	data, err := LoadSafeTensorData(collection, dir, requestedNames)
	if err != nil {
		return PretrainedBERTWeightSet{}, PretrainedBERTWeightLoadReport{}, err
	}
	set := PretrainedBERTWeightSet{Tensors: make([]PretrainedBERTWeightTensor, 0, len(requestedNames))}
	report := PretrainedBERTWeightLoadReport{
		Status:      "ok",
		Files:       append([]string(nil), collection.Files...),
		TensorCount: len(requestedNames),
		Loaded:      append([]string(nil), requestedNames...),
	}
	for _, name := range requestedNames {
		tensor := data[name]
		report.TotalBytes += tensor.ByteLength
		set.Tensors = append(set.Tensors, PretrainedBERTWeightTensor{
			Name:       name,
			Role:       roles[name],
			DType:      tensor.DType,
			Shape:      append([]int64(nil), tensor.Shape...),
			SourceFile: tensor.SourceFile,
			ByteOffset: tensor.ByteOffset,
			ByteLength: tensor.ByteLength,
			Bytes:      append([]byte(nil), tensor.Bytes...),
		})
	}
	for name := range collection.Tensors {
		if _, ok := plannedNames[name]; !ok {
			report.SkippedExtra = append(report.SkippedExtra, name)
		}
	}
	slices.Sort(report.SkippedExtra)
	return set, report, nil
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

func sameBERTTensorShape(expected []int, actual []int64) bool {
	if len(expected) != len(actual) {
		return false
	}
	for i := range expected {
		if int64(expected[i]) != actual[i] {
			return false
		}
	}
	return true
}

func acceptablePretrainedBERTDType(dtype string) bool {
	switch dtype {
	case "F32", "F16", "BF16":
		return true
	default:
		return false
	}
}

func acceptablePretrainedBERTDTypes() []string {
	return []string{"F32", "F16", "BF16"}
}

func sortBERTShapeMismatches(items []PretrainedBERTShapeMismatch) {
	slices.SortFunc(items, func(a, b PretrainedBERTShapeMismatch) int {
		return strings.Compare(a.Name, b.Name)
	})
}

func sortBERTDTypeMismatches(items []PretrainedBERTDTypeMismatch) {
	slices.SortFunc(items, func(a, b PretrainedBERTDTypeMismatch) int {
		return strings.Compare(a.Name, b.Name)
	})
}
