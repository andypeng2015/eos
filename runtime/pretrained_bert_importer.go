package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
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
	WeightDecodeSmoke      *PretrainedBERTWeightDecodeReport `json:"weight_decode_smoke,omitempty"`
	WeightFileExport       *PretrainedBERTWeightFileReport   `json:"weight_file_export,omitempty"`
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

type PretrainedBERTDecodedWeightTensor struct {
	Name        string    `json:"name"`
	Role        string    `json:"role"`
	SourceDType string    `json:"source_dtype"`
	Shape       []int64   `json:"shape"`
	SourceFile  string    `json:"source_file"`
	ByteOffset  int64     `json:"byte_offset"`
	ByteLength  int64     `json:"byte_length"`
	Values      []float32 `json:"-"`
}

type PretrainedBERTDecodedWeightSet struct {
	Tensors []PretrainedBERTDecodedWeightTensor `json:"tensors"`
}

type PretrainedBERTWeightLoadReport struct {
	Status       string   `json:"status"`
	Files        []string `json:"files,omitempty"`
	TensorCount  int      `json:"tensor_count"`
	TotalBytes   int64    `json:"total_bytes"`
	Loaded       []string `json:"loaded,omitempty"`
	SkippedExtra []string `json:"skipped_extra,omitempty"`
}

type PretrainedBERTWeightDecodeReport struct {
	Status        string         `json:"status"`
	Files         []string       `json:"files,omitempty"`
	TensorCount   int            `json:"tensor_count"`
	TotalElements int64          `json:"total_elements"`
	SourceDTypes  map[string]int `json:"source_dtypes,omitempty"`
	Loaded        []string       `json:"loaded,omitempty"`
	SkippedExtra  []string       `json:"skipped_extra,omitempty"`
}

type PretrainedBERTWeightFileReport struct {
	Status        string                                 `json:"status"`
	OutputPath    string                                 `json:"output_path,omitempty"`
	Files         []string                               `json:"files,omitempty"`
	TensorCount   int                                    `json:"tensor_count"`
	TotalElements int64                                  `json:"total_elements"`
	StorageDTypes map[string]int                         `json:"storage_dtypes,omitempty"`
	SourceDTypes  map[string]int                         `json:"source_dtypes,omitempty"`
	Loaded        []PretrainedBERTWeightFileTensorReport `json:"loaded,omitempty"`
	SkippedExtra  []string                               `json:"skipped_extra,omitempty"`
}

type PretrainedBERTWeightFileTensorReport struct {
	Role         string  `json:"role"`
	Name         string  `json:"name"`
	Shape        []int64 `json:"shape"`
	SourceDType  string  `json:"source_dtype"`
	StorageDType string  `json:"storage_dtype"`
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
	requestedPlans, plannedNames, err := requestedPretrainedBERTWeightPlans(plan, collection, "byte ingestion")
	if err != nil {
		return PretrainedBERTWeightSet{}, PretrainedBERTWeightLoadReport{}, err
	}
	requestedNames, roles := namesAndRolesForBERTWeightPlans(requestedPlans)
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

func LoadPretrainedBERTDecodedWeightsFromDir(dir string, plan PretrainedBERTImportPlan) (PretrainedBERTDecodedWeightSet, PretrainedBERTWeightDecodeReport, error) {
	collection, err := ReadSafeTensorsCollectionFromDir(dir)
	if err != nil {
		return PretrainedBERTDecodedWeightSet{}, PretrainedBERTWeightDecodeReport{}, err
	}
	requestedPlans, plannedNames, err := requestedPretrainedBERTWeightPlans(plan, collection, "float32 decode")
	if err != nil {
		return PretrainedBERTDecodedWeightSet{}, PretrainedBERTWeightDecodeReport{}, err
	}
	requestedNames, roles := namesAndRolesForBERTWeightPlans(requestedPlans)
	data, err := LoadSafeTensorFloat32Data(collection, dir, requestedNames)
	if err != nil {
		return PretrainedBERTDecodedWeightSet{}, PretrainedBERTWeightDecodeReport{}, err
	}
	set := PretrainedBERTDecodedWeightSet{Tensors: make([]PretrainedBERTDecodedWeightTensor, 0, len(requestedPlans))}
	report := PretrainedBERTWeightDecodeReport{
		Status:       "ok",
		Files:        append([]string(nil), collection.Files...),
		TensorCount:  len(requestedPlans),
		SourceDTypes: map[string]int{},
		Loaded:       make([]string, 0, len(requestedPlans)),
	}
	for _, planned := range requestedPlans {
		tensor := data[planned.Name]
		elementCount := int64(len(tensor.Values))
		report.TotalElements += elementCount
		report.SourceDTypes[tensor.SourceDType]++
		report.Loaded = append(report.Loaded, planned.Name)
		set.Tensors = append(set.Tensors, PretrainedBERTDecodedWeightTensor{
			Name:        planned.Name,
			Role:        roles[planned.Name],
			SourceDType: tensor.SourceDType,
			Shape:       append([]int64(nil), tensor.Shape...),
			SourceFile:  tensor.SourceFile,
			ByteOffset:  tensor.ByteOffset,
			ByteLength:  tensor.ByteLength,
			Values:      append([]float32(nil), tensor.Values...),
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

func BuildPretrainedBERTWeightFileFromDecoded(set PretrainedBERTDecodedWeightSet) (WeightFile, PretrainedBERTWeightFileReport, error) {
	if len(set.Tensors) == 0 {
		return WeightFile{}, PretrainedBERTWeightFileReport{}, fmt.Errorf("pretrained BERT decoded tensor set is empty")
	}
	weights := make(map[string]*backend.Tensor, len(set.Tensors))
	seenFiles := map[string]struct{}{}
	report := PretrainedBERTWeightFileReport{
		Status:        "ok",
		StorageDTypes: map[string]int{},
		SourceDTypes:  map[string]int{},
		Loaded:        make([]PretrainedBERTWeightFileTensorReport, 0, len(set.Tensors)),
	}
	for _, tensor := range set.Tensors {
		if tensor.Role == "" {
			return WeightFile{}, PretrainedBERTWeightFileReport{}, fmt.Errorf("pretrained BERT tensor %q has empty role", tensor.Name)
		}
		if _, exists := weights[tensor.Role]; exists {
			return WeightFile{}, PretrainedBERTWeightFileReport{}, fmt.Errorf("duplicate pretrained BERT tensor role %q", tensor.Role)
		}
		if !acceptablePretrainedBERTDType(tensor.SourceDType) {
			return WeightFile{}, PretrainedBERTWeightFileReport{}, fmt.Errorf("pretrained BERT tensor %q source dtype %q is not supported", tensor.Name, tensor.SourceDType)
		}
		shape, elements, err := backendShapeFromBERTDecodedTensor(tensor)
		if err != nil {
			return WeightFile{}, PretrainedBERTWeightFileReport{}, err
		}
		if int64(len(tensor.Values)) != elements {
			return WeightFile{}, PretrainedBERTWeightFileReport{}, fmt.Errorf("pretrained BERT tensor %q values=%d does not match shape elements=%d", tensor.Name, len(tensor.Values), elements)
		}
		weights[tensor.Role] = backend.NewTensorF32(shape, tensor.Values)
		report.TensorCount++
		report.TotalElements += elements
		report.StorageDTypes["f32"]++
		report.SourceDTypes[tensor.SourceDType]++
		if tensor.SourceFile != "" {
			seenFiles[tensor.SourceFile] = struct{}{}
		}
		report.Loaded = append(report.Loaded, PretrainedBERTWeightFileTensorReport{
			Role:         tensor.Role,
			Name:         tensor.Name,
			Shape:        append([]int64(nil), tensor.Shape...),
			SourceDType:  tensor.SourceDType,
			StorageDType: "f32",
		})
	}
	for file := range seenFiles {
		report.Files = append(report.Files, file)
	}
	slices.Sort(report.Files)
	slices.SortFunc(report.Loaded, func(a, b PretrainedBERTWeightFileTensorReport) int {
		if cmp := strings.Compare(a.Role, b.Role); cmp != 0 {
			return cmp
		}
		return strings.Compare(a.Name, b.Name)
	})
	return NewWeightFile(weights), report, nil
}

func ExportPretrainedBERTWeightFileFromDir(dir string, plan PretrainedBERTImportPlan, outPath string) (PretrainedBERTWeightFileReport, error) {
	if outPath == "" {
		return PretrainedBERTWeightFileReport{}, fmt.Errorf("weights output path is required")
	}
	set, decodeReport, err := LoadPretrainedBERTDecodedWeightsFromDir(dir, plan)
	if err != nil {
		return PretrainedBERTWeightFileReport{}, err
	}
	weightFile, report, err := BuildPretrainedBERTWeightFileFromDecoded(set)
	if err != nil {
		return PretrainedBERTWeightFileReport{}, err
	}
	report.OutputPath = outPath
	report.Files = append([]string(nil), decodeReport.Files...)
	report.SkippedExtra = append([]string(nil), decodeReport.SkippedExtra...)
	if err := weightFile.WriteFile(outPath); err != nil {
		return PretrainedBERTWeightFileReport{}, err
	}
	return report, nil
}

// BuildPretrainedBERTEmbeddingStageModule builds the first executable BERT
// parity slice: embedding-table lookup, embedding sum, and affine LayerNorm.
func BuildPretrainedBERTEmbeddingStageModule(plan PretrainedBERTImportPlan) (*eosartifact.Module, error) {
	if err := plan.Config.Validate(); err != nil {
		return nil, err
	}
	requiredRoles := map[string][]int{
		"token_embeddings":           {plan.Config.VocabSize, plan.Config.HiddenSize},
		"position_embeddings":        {plan.Config.MaxPositionEmbeddings, plan.Config.HiddenSize},
		"token_type_embeddings":      {plan.Config.TypeVocabSize, plan.Config.HiddenSize},
		"embedding_layernorm_weight": {plan.Config.HiddenSize},
		"embedding_layernorm_bias":   {plan.Config.HiddenSize},
	}
	plannedRoles := map[string]PretrainedBERTTensorPlan{}
	for _, tensor := range plan.Tensors {
		plannedRoles[tensor.Role] = tensor
	}
	for role, shape := range requiredRoles {
		tensor, ok := plannedRoles[role]
		if !ok {
			return nil, fmt.Errorf("pretrained BERT embedding module missing planned role %q", role)
		}
		if !slices.Equal(tensor.Shape, shape) {
			return nil, fmt.Errorf("pretrained BERT embedding module role %q shape %v does not match config shape %v", role, tensor.Shape, shape)
		}
	}

	mod := eosartifact.NewModule("pretrained_bert_embedding_stage")
	mod.Requirements.Capabilities = []string{eosartifact.CapabilityHostFallback}
	mod.Metadata = map[string]any{
		"source":           "pretrained_bert_import_plan",
		"model_name":       plan.ModelName,
		"architecture":     plan.Architecture,
		"execution_status": "embedding_stage_only: host reference; no encoder attention, pooling, tokenizer parity, or device execution claim",
	}
	hidden := strconv.Itoa(plan.Config.HiddenSize)
	mod.Params = []eosartifact.Param{
		bertEmbeddingParam("token_embeddings", "V", hidden),
		bertEmbeddingParam("position_embeddings", "P", hidden),
		bertEmbeddingParam("token_type_embeddings", "TT", hidden),
		bertEmbeddingParam("embedding_layernorm_weight", hidden),
		bertEmbeddingParam("embedding_layernorm_bias", hidden),
	}
	idType := bertTensorType("i32", "B", "T")
	embeddingType := bertTensorType("f32", "B", "T", hidden)
	mod.EntryPoints = []eosartifact.EntryPoint{{
		Name: "bert_embeddings",
		Kind: eosartifact.EntryPointPipeline,
		Inputs: []eosartifact.ValueBinding{
			{Name: "input_ids", Type: idType},
			{Name: "position_ids", Type: idType},
			{Name: "token_type_ids", Type: idType},
		},
		Outputs: []eosartifact.ValueBinding{
			{Name: "embeddings", Type: embeddingType},
		},
	}}
	mod.Buffers = []eosartifact.Buffer{{
		Name:  "embeddings",
		DType: "f32",
		Shape: []string{"B", "T", hidden},
	}}
	epsilon := plan.Config.LayerNormEps
	if epsilon == 0 {
		epsilon = 1e-12
	}
	mod.Steps = []eosartifact.Step{
		{
			Entry: "bert_embeddings",
			Kind:  eosartifact.StepBERTEmbeddings,
			Name:  "embedding_lookup_sum_layernorm",
			Inputs: []string{
				"token_embeddings",
				"position_embeddings",
				"token_type_embeddings",
				"embedding_layernorm_weight",
				"embedding_layernorm_bias",
				"input_ids",
				"position_ids",
				"token_type_ids",
			},
			Outputs:    []string{"embeddings"},
			Attributes: map[string]string{"epsilon": strconv.FormatFloat(epsilon, 'g', -1, 64)},
		},
		{
			Entry:   "bert_embeddings",
			Kind:    eosartifact.StepReturn,
			Name:    "return_embeddings",
			Outputs: []string{"embeddings"},
		},
	}
	if err := mod.Validate(); err != nil {
		return nil, err
	}
	return mod, nil
}

// BuildPretrainedBERTSingleLayerModule builds one executable host-reference
// BERT encoder layer backed by role-named imported weights.
func BuildPretrainedBERTSingleLayerModule(plan PretrainedBERTImportPlan, layer int) (*eosartifact.Module, error) {
	if err := plan.Config.Validate(); err != nil {
		return nil, err
	}
	if layer < 0 || layer >= plan.Config.NumHiddenLayers {
		return nil, fmt.Errorf("pretrained BERT encoder layer %d out of range [0,%d)", layer, plan.Config.NumHiddenLayers)
	}
	hiddenAct := plan.Config.HiddenAct
	if hiddenAct == "" {
		hiddenAct = "gelu"
	}
	if hiddenAct != "gelu" {
		return nil, fmt.Errorf("pretrained BERT encoder layer unsupported hidden_act %q; only gelu is supported", hiddenAct)
	}
	hidden := plan.Config.HiddenSize
	intermediate := plan.Config.IntermediateSize
	rolePrefix := fmt.Sprintf("encoder_layer_%d_", layer)
	requiredRoles := map[string][]int{
		rolePrefix + "attention_query_weight":     {hidden, hidden},
		rolePrefix + "attention_query_bias":       {hidden},
		rolePrefix + "attention_key_weight":       {hidden, hidden},
		rolePrefix + "attention_key_bias":         {hidden},
		rolePrefix + "attention_value_weight":     {hidden, hidden},
		rolePrefix + "attention_value_bias":       {hidden},
		rolePrefix + "attention_output_weight":    {hidden, hidden},
		rolePrefix + "attention_output_bias":      {hidden},
		rolePrefix + "attention_layernorm_weight": {hidden},
		rolePrefix + "attention_layernorm_bias":   {hidden},
		rolePrefix + "intermediate_weight":        {intermediate, hidden},
		rolePrefix + "intermediate_bias":          {intermediate},
		rolePrefix + "output_weight":              {hidden, intermediate},
		rolePrefix + "output_bias":                {hidden},
		rolePrefix + "output_layernorm_weight":    {hidden},
		rolePrefix + "output_layernorm_bias":      {hidden},
	}
	plannedRoles := map[string]PretrainedBERTTensorPlan{}
	for _, tensor := range plan.Tensors {
		plannedRoles[tensor.Role] = tensor
	}
	for role, shape := range requiredRoles {
		tensor, ok := plannedRoles[role]
		if !ok {
			return nil, fmt.Errorf("pretrained BERT encoder layer module missing planned role %q", role)
		}
		if !slices.Equal(tensor.Shape, shape) {
			return nil, fmt.Errorf("pretrained BERT encoder layer module role %q shape %v does not match config shape %v", role, tensor.Shape, shape)
		}
	}

	mod := eosartifact.NewModule("pretrained_bert_encoder_layer")
	mod.Requirements.Capabilities = []string{eosartifact.CapabilityHostFallback}
	mod.Metadata = map[string]any{
		"source":           "pretrained_bert_import_plan",
		"model_name":       plan.ModelName,
		"architecture":     plan.Architecture,
		"layer":            layer,
		"execution_status": "single_encoder_layer_only: host reference; no tokenizer, pooling, full encoder stack, package export, quantized execution, or device execution claim",
	}
	hiddenStr := strconv.Itoa(hidden)
	intermediateStr := strconv.Itoa(intermediate)
	paramRoles := []struct {
		role  string
		shape []string
	}{
		{rolePrefix + "attention_query_weight", []string{hiddenStr, hiddenStr}},
		{rolePrefix + "attention_query_bias", []string{hiddenStr}},
		{rolePrefix + "attention_key_weight", []string{hiddenStr, hiddenStr}},
		{rolePrefix + "attention_key_bias", []string{hiddenStr}},
		{rolePrefix + "attention_value_weight", []string{hiddenStr, hiddenStr}},
		{rolePrefix + "attention_value_bias", []string{hiddenStr}},
		{rolePrefix + "attention_output_weight", []string{hiddenStr, hiddenStr}},
		{rolePrefix + "attention_output_bias", []string{hiddenStr}},
		{rolePrefix + "attention_layernorm_weight", []string{hiddenStr}},
		{rolePrefix + "attention_layernorm_bias", []string{hiddenStr}},
		{rolePrefix + "intermediate_weight", []string{intermediateStr, hiddenStr}},
		{rolePrefix + "intermediate_bias", []string{intermediateStr}},
		{rolePrefix + "output_weight", []string{hiddenStr, intermediateStr}},
		{rolePrefix + "output_bias", []string{hiddenStr}},
		{rolePrefix + "output_layernorm_weight", []string{hiddenStr}},
		{rolePrefix + "output_layernorm_bias", []string{hiddenStr}},
	}
	mod.Params = make([]eosartifact.Param, 0, len(paramRoles))
	stepInputs := []string{"hidden_states", "attention_mask"}
	for _, param := range paramRoles {
		mod.Params = append(mod.Params, bertEmbeddingParam(param.role, param.shape...))
		stepInputs = append(stepInputs, param.role)
	}
	hiddenType := bertTensorType("f32", "B", "T", hiddenStr)
	maskType := bertTensorType("i32", "B", "T")
	mod.EntryPoints = []eosartifact.EntryPoint{{
		Name: "bert_encoder_layer",
		Kind: eosartifact.EntryPointPipeline,
		Inputs: []eosartifact.ValueBinding{
			{Name: "hidden_states", Type: hiddenType},
			{Name: "attention_mask", Type: maskType},
		},
		Outputs: []eosartifact.ValueBinding{
			{Name: "hidden_states_out", Type: hiddenType},
		},
	}}
	mod.Buffers = []eosartifact.Buffer{{
		Name:  "hidden_states_out",
		DType: "f32",
		Shape: []string{"B", "T", hiddenStr},
	}}
	epsilon := plan.Config.LayerNormEps
	if epsilon == 0 {
		epsilon = 1e-12
	}
	mod.Steps = []eosartifact.Step{
		{
			Entry:   "bert_encoder_layer",
			Kind:    eosartifact.StepBERTEncoderLayer,
			Name:    fmt.Sprintf("encoder_layer_%d_reference", layer),
			Inputs:  stepInputs,
			Outputs: []string{"hidden_states_out"},
			Attributes: map[string]string{
				"epsilon":             strconv.FormatFloat(epsilon, 'g', -1, 64),
				"hidden_act":          hiddenAct,
				"num_attention_heads": strconv.Itoa(plan.Config.NumAttentionHeads),
				"layer":               strconv.Itoa(layer),
			},
		},
		{
			Entry:   "bert_encoder_layer",
			Kind:    eosartifact.StepReturn,
			Name:    "return_hidden_states_out",
			Outputs: []string{"hidden_states_out"},
		},
	}
	if err := mod.Validate(); err != nil {
		return nil, err
	}
	return mod, nil
}

func bertEmbeddingParam(name string, shape ...string) eosartifact.Param {
	return eosartifact.Param{
		Name:    name,
		Type:    bertTensorType("f32", shape...),
		Binding: name,
	}
}

func bertTensorType(dtype string, shape ...string) eosartifact.ValueType {
	return eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: dtype,
			Shape: append([]string(nil), shape...),
		},
	}
}

func backendShapeFromBERTDecodedTensor(tensor PretrainedBERTDecodedWeightTensor) ([]int, int64, error) {
	if len(tensor.Shape) == 0 {
		return nil, 0, fmt.Errorf("pretrained BERT tensor %q shape is empty", tensor.Name)
	}
	maxInt := int64(int(^uint(0) >> 1))
	shape := make([]int, len(tensor.Shape))
	var elements int64 = 1
	for i, dim := range tensor.Shape {
		if dim <= 0 {
			return nil, 0, fmt.Errorf("pretrained BERT tensor %q shape dim %d must be positive, got %d", tensor.Name, i, dim)
		}
		if dim > maxInt {
			return nil, 0, fmt.Errorf("pretrained BERT tensor %q shape dim %d overflows int: %d", tensor.Name, i, dim)
		}
		if elements > maxInt/dim {
			return nil, 0, fmt.Errorf("pretrained BERT tensor %q element count overflows int", tensor.Name)
		}
		elements *= dim
		shape[i] = int(dim)
	}
	return shape, elements, nil
}

func requestedPretrainedBERTWeightPlans(plan PretrainedBERTImportPlan, collection SafeTensorsCollection, action string) ([]PretrainedBERTTensorPlan, map[string]struct{}, error) {
	verification := VerifyPretrainedBERTWeights(plan, collection)
	if len(verification.Missing) > 0 || len(verification.ShapeMismatches) > 0 || len(verification.DTypeMismatches) > 0 {
		return nil, nil, fmt.Errorf("pretrained BERT weight verification failed before %s: missing=%d shape_mismatches=%d dtype_mismatches=%d", action, len(verification.Missing), len(verification.ShapeMismatches), len(verification.DTypeMismatches))
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
	return requestedPlans, plannedNames, nil
}

func namesAndRolesForBERTWeightPlans(plans []PretrainedBERTTensorPlan) ([]string, map[string]string) {
	names := make([]string, 0, len(plans))
	roles := make(map[string]string, len(plans))
	for _, tensor := range plans {
		names = append(names, tensor.Name)
		roles[tensor.Name] = tensor.Role
	}
	return names, roles
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
