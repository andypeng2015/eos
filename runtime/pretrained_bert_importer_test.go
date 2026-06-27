package eosruntime

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestPlanPretrainedBERTImportFromConfig(t *testing.T) {
	cfg := PretrainedBERTConfig{
		Architectures:         []string{"BertModel"},
		ModelType:             "bert",
		VocabSize:             30522,
		HiddenSize:            384,
		NumHiddenLayers:       2,
		NumAttentionHeads:     12,
		IntermediateSize:      1536,
		HiddenAct:             "gelu",
		MaxPositionEmbeddings: 512,
		TypeVocabSize:         2,
		LayerNormEps:          1e-12,
		PositionEmbeddingType: "absolute",
	}
	plan, err := PlanPretrainedBERTImport(cfg, "BAAI/bge-small-en-v1.5")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	if plan.Version != PretrainedBERTImporterPlanVersion {
		t.Fatalf("version = %q", plan.Version)
	}
	if plan.Architecture != "BertModel" {
		t.Fatalf("architecture = %q", plan.Architecture)
	}
	assertBERTTensorPlan(t, plan, "embeddings.word_embeddings.weight", []int{30522, 384}, true)
	assertBERTTensorPlan(t, plan, "encoder.layer.1.attention.self.query.weight", []int{384, 384}, true)
	assertBERTTensorPlan(t, plan, "encoder.layer.1.intermediate.dense.weight", []int{1536, 384}, true)
	assertBERTTensorPlan(t, plan, "encoder.layer.1.output.dense.weight", []int{384, 1536}, true)
	assertBERTTensorPlan(t, plan, "pooler.dense.weight", []int{384, 384}, false)
	if !strings.Contains(plan.ExecutionStatus, "plan_only") {
		t.Fatalf("execution status = %q", plan.ExecutionStatus)
	}
}

func TestPlanPretrainedBERTImportFromDir(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures": ["BertForMaskedLM"],
		"model_type": "bert",
		"vocab_size": 100,
		"hidden_size": 32,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"intermediate_size": 64,
		"hidden_act": "gelu_new",
		"max_position_embeddings": 128,
		"type_vocab_size": 2
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	plan, err := PlanPretrainedBERTImportFromDir(dir, "fixture")
	if err != nil {
		t.Fatalf("plan from dir: %v", err)
	}
	if len(plan.Tensors) != 23 {
		t.Fatalf("tensor count = %d, want 23", len(plan.Tensors))
	}
	assertBERTTensorPlan(t, plan, "encoder.layer.0.attention.self.value.bias", []int{32}, true)
}

func TestBuildPretrainedBERTEmbeddingStageModuleExecutesWeightFile(t *testing.T) {
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             2,
		HiddenSize:            2,
		NumHiddenLayers:       1,
		NumAttentionHeads:     1,
		IntermediateSize:      4,
		MaxPositionEmbeddings: 2,
		TypeVocabSize:         2,
		LayerNormEps:          3,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	mod, err := BuildPretrainedBERTEmbeddingStageModule(plan)
	if err != nil {
		t.Fatalf("build embedding stage module: %v", err)
	}
	wantParamNames := []string{
		"token_embeddings",
		"position_embeddings",
		"token_type_embeddings",
		"embedding_layernorm_weight",
		"embedding_layernorm_bias",
	}
	if got := paramNames(mod.Params); !slices.Equal(got, wantParamNames) {
		t.Fatalf("params = %v, want %v", got, wantParamNames)
	}
	if len(mod.Steps) == 0 || mod.Steps[0].Kind != eosartifact.StepBERTEmbeddings {
		t.Fatalf("first step = %+v, want bert_embeddings", mod.Steps)
	}
	if mod.Steps[0].Attributes["epsilon"] != "3" {
		t.Fatalf("epsilon attr = %q, want 3", mod.Steps[0].Attributes["epsilon"])
	}

	weights, _, err := BuildPretrainedBERTWeightFileFromDecoded(PretrainedBERTDecodedWeightSet{Tensors: []PretrainedBERTDecodedWeightTensor{
		{Name: "embeddings.word_embeddings.weight", Role: "token_embeddings", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{1, 2, 10, 20}},
		{Name: "embeddings.position_embeddings.weight", Role: "position_embeddings", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{0, 1, 1, 0}},
		{Name: "embeddings.token_type_embeddings.weight", Role: "token_type_embeddings", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{0, 0, 2, -2}},
		{Name: "embeddings.LayerNorm.weight", Role: "embedding_layernorm_weight", SourceDType: "F32", Shape: []int64{2}, Values: []float32{2, 3}},
		{Name: "embeddings.LayerNorm.bias", Role: "embedding_layernorm_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.5, -0.5}},
	}})
	if err != nil {
		t.Fatalf("build weight file: %v", err)
	}

	rt := New(bertEmbeddingHostBackend{})
	prog, err := rt.Load(context.Background(), mod, weights.LoadOptions()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "bert_embeddings",
		Inputs: map[string]any{
			"input_ids":      backend.NewTensorI32([]int{2, 2}, []int32{0, 1, 0, 1}),
			"position_ids":   backend.NewTensorI32([]int{2, 2}, []int32{0, 1, 1, 0}),
			"token_type_ids": backend.NewTensorI32([]int{2, 2}, []int32{1, 0, 0, 1}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	value, ok := result.Outputs["embeddings"]
	if !ok {
		t.Fatalf("missing embeddings output: %+v", result.Outputs)
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", value.Data)
	}
	want := []float32{
		1.5, -2,
		float32(-9.0/math.Sqrt(23.25) + 0.5),
		float32(13.5/math.Sqrt(23.25) - 0.5),
		0.5, -0.5,
		float32(-7.0/math.Sqrt(15.25) + 0.5),
		float32(10.5/math.Sqrt(15.25) - 0.5),
	}
	assertTensorClose(t, tensor, []int{2, 2, 2}, want)
	if value.Metadata["dispatch_mode"] != "host_reference" {
		t.Fatalf("dispatch_mode = %v, want host_reference", value.Metadata["dispatch_mode"])
	}
}

func TestBuildPretrainedBERTSingleLayerModuleExecutesWeightFile(t *testing.T) {
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             2,
		HiddenSize:            2,
		NumHiddenLayers:       1,
		NumAttentionHeads:     1,
		IntermediateSize:      3,
		HiddenAct:             "gelu",
		MaxPositionEmbeddings: 2,
		TypeVocabSize:         2,
		LayerNormEps:          0.25,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	mod, err := BuildPretrainedBERTSingleLayerModule(plan, 0)
	if err != nil {
		t.Fatalf("build single-layer module: %v", err)
	}
	if got, want := mod.EntryPoints[0].Name, "bert_encoder_layer"; got != want {
		t.Fatalf("entrypoint = %q, want %q", got, want)
	}
	if len(mod.Steps) == 0 || mod.Steps[0].Kind != eosartifact.StepBERTEncoderLayer {
		t.Fatalf("first step = %+v, want bert_encoder_layer", mod.Steps)
	}
	if mod.Steps[0].Attributes["num_attention_heads"] != "1" || mod.Steps[0].Attributes["epsilon"] != "0.25" {
		t.Fatalf("step attrs = %+v", mod.Steps[0].Attributes)
	}

	decoded := pretrainedBERTSingleLayerDecodedWeights()
	weights, _, err := BuildPretrainedBERTWeightFileFromDecoded(PretrainedBERTDecodedWeightSet{Tensors: decoded})
	if err != nil {
		t.Fatalf("build weight file: %v", err)
	}
	rt := New(bertEmbeddingHostBackend{})
	prog, err := rt.Load(context.Background(), mod, weights.LoadOptions()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	hiddenStates := backend.NewTensorF32([]int{1, 2, 2}, []float32{2, -1, 1, 3})
	attentionMask := backend.NewTensorI32([]int{1, 2}, []int32{1, 0})
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "bert_encoder_layer",
		Inputs: map[string]any{
			"hidden_states":  hiddenStates,
			"attention_mask": attentionMask,
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	value, ok := result.Outputs["hidden_states_out"]
	if !ok {
		t.Fatalf("missing hidden_states_out output: %+v", result.Outputs)
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", value.Data)
	}
	expectedWeights := weights.Weights
	want, err := backend.BERTEncoderLayerReference(
		hiddenStates, attentionMask,
		expectedWeights["encoder_layer_0_attention_query_weight"], expectedWeights["encoder_layer_0_attention_query_bias"],
		expectedWeights["encoder_layer_0_attention_key_weight"], expectedWeights["encoder_layer_0_attention_key_bias"],
		expectedWeights["encoder_layer_0_attention_value_weight"], expectedWeights["encoder_layer_0_attention_value_bias"],
		expectedWeights["encoder_layer_0_attention_output_weight"], expectedWeights["encoder_layer_0_attention_output_bias"],
		expectedWeights["encoder_layer_0_attention_layernorm_weight"], expectedWeights["encoder_layer_0_attention_layernorm_bias"],
		expectedWeights["encoder_layer_0_intermediate_weight"], expectedWeights["encoder_layer_0_intermediate_bias"],
		expectedWeights["encoder_layer_0_output_weight"], expectedWeights["encoder_layer_0_output_bias"],
		expectedWeights["encoder_layer_0_output_layernorm_weight"], expectedWeights["encoder_layer_0_output_layernorm_bias"],
		1, 0.25, "gelu",
	)
	if err != nil {
		t.Fatalf("expected reference: %v", err)
	}
	assertTensorClose(t, tensor, want.Shape, want.F32)
	if value.Metadata["dispatch_mode"] != "host_reference" {
		t.Fatalf("dispatch_mode = %v, want host_reference", value.Metadata["dispatch_mode"])
	}
}

func TestBuildPretrainedBERTSingleLayerModuleValidation(t *testing.T) {
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             2,
		HiddenSize:            2,
		NumHiddenLayers:       1,
		NumAttentionHeads:     1,
		IntermediateSize:      3,
		HiddenAct:             "gelu",
		MaxPositionEmbeddings: 2,
		TypeVocabSize:         2,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	_, err = BuildPretrainedBERTSingleLayerModule(plan, 1)
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("expected out-of-range layer error, got %v", err)
	}
	missing := plan
	missing.Tensors = nil
	for _, tensor := range plan.Tensors {
		if tensor.Role == "encoder_layer_0_output_layernorm_bias" {
			continue
		}
		missing.Tensors = append(missing.Tensors, tensor)
	}
	_, err = BuildPretrainedBERTSingleLayerModule(missing, 0)
	if err == nil || !strings.Contains(err.Error(), "missing planned role") {
		t.Fatalf("expected missing role error, got %v", err)
	}
	unsupported := plan
	unsupported.Config.HiddenAct = "relu"
	_, err = BuildPretrainedBERTSingleLayerModule(unsupported, 0)
	if err == nil || !strings.Contains(err.Error(), "unsupported hidden_act") {
		t.Fatalf("expected unsupported activation error, got %v", err)
	}
}

func TestPlanPretrainedBERTImportRejectsUnsupportedArchitecture(t *testing.T) {
	cfg := PretrainedBERTConfig{
		Architectures:         []string{"RobertaModel"},
		ModelType:             "bert",
		VocabSize:             100,
		HiddenSize:            32,
		NumHiddenLayers:       1,
		NumAttentionHeads:     4,
		IntermediateSize:      64,
		MaxPositionEmbeddings: 128,
		TypeVocabSize:         2,
	}
	_, err := PlanPretrainedBERTImport(cfg, "bad")
	if err == nil || !strings.Contains(err.Error(), "unsupported architectures") {
		t.Fatalf("expected unsupported architecture error, got %v", err)
	}
}

func TestPlanPretrainedBERTImportRejectsInconsistentHeads(t *testing.T) {
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             100,
		HiddenSize:            30,
		NumHiddenLayers:       1,
		NumAttentionHeads:     8,
		IntermediateSize:      64,
		MaxPositionEmbeddings: 128,
		TypeVocabSize:         2,
	}
	_, err := PlanPretrainedBERTImport(cfg, "bad")
	if err == nil || !strings.Contains(err.Error(), "divisible") {
		t.Fatalf("expected divisibility error, got %v", err)
	}
}

func TestVerifyPretrainedBERTWeightsFromDirReportsMismatches(t *testing.T) {
	dir := t.TempDir()
	cfg := PretrainedBERTConfig{
		Architectures:         []string{"BertModel"},
		ModelType:             "bert",
		VocabSize:             7,
		HiddenSize:            8,
		NumHiddenLayers:       1,
		NumAttentionHeads:     2,
		IntermediateSize:      16,
		HiddenAct:             "gelu",
		MaxPositionEmbeddings: 16,
		TypeVocabSize:         2,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	header := safeTensorsHeaderForBERTPlan(plan)
	delete(header, "encoder.layer.0.output.LayerNorm.bias")
	header["embeddings.word_embeddings.weight"] = map[string]any{
		"dtype":        "F32",
		"shape":        []int64{8, 8},
		"data_offsets": []int64{0, 1},
	}
	header["encoder.layer.0.attention.self.query.bias"] = map[string]any{
		"dtype":        "I64",
		"shape":        []int64{8},
		"data_offsets": []int64{1, 2},
	}
	header["cls.predictions.decoder.weight"] = map[string]any{
		"dtype":        "F32",
		"shape":        []int64{7, 8},
		"data_offsets": []int64{0, 224},
	}
	payloadSize := renumberSafeTensorFixtureOffsets(header)
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model.safetensors"), header, make([]byte, payloadSize)); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	report, err := VerifyPretrainedBERTWeightsFromDir(dir, plan)
	if err != nil {
		t.Fatalf("verify weights: %v", err)
	}
	if report.Status != "mismatch" {
		t.Fatalf("status = %q", report.Status)
	}
	if !slices.Contains(report.Missing, "encoder.layer.0.output.LayerNorm.bias") {
		t.Fatalf("missing = %+v", report.Missing)
	}
	assertBERTShapeMismatch(t, report, "embeddings.word_embeddings.weight", []int{7, 8}, []int64{8, 8})
	assertBERTDTypeMismatch(t, report, "encoder.layer.0.attention.self.query.bias", "I64")
	if !slices.Contains(report.Unexpected, "cls.predictions.decoder.weight") {
		t.Fatalf("unexpected = %+v", report.Unexpected)
	}
	if slices.Contains(report.Missing, "pooler.dense.weight") {
		t.Fatalf("optional pooler should not be missing: %+v", report.Missing)
	}
}

func TestVerifyPretrainedBERTWeightsFromDirSupportsShardedIndex(t *testing.T) {
	dir := t.TempDir()
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             7,
		HiddenSize:            8,
		NumHiddenLayers:       1,
		NumAttentionHeads:     2,
		IntermediateSize:      16,
		MaxPositionEmbeddings: 16,
		TypeVocabSize:         2,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	header := safeTensorsHeaderForBERTPlan(plan)
	shard1 := map[string]any{}
	shard2 := map[string]any{}
	for i, tensor := range plan.Tensors {
		if !tensor.Required {
			continue
		}
		if i%2 == 0 {
			shard1[tensor.Name] = header[tensor.Name]
		} else {
			shard2[tensor.Name] = header[tensor.Name]
		}
	}
	payload1 := renumberSafeTensorFixtureOffsets(shard1)
	payload2 := renumberSafeTensorFixtureOffsets(shard2)
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model-00001-of-00002.safetensors"), shard1, make([]byte, payload1)); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model-00002-of-00002.safetensors"), shard2, make([]byte, payload2)); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	weightMap := map[string]string{}
	for name := range shard1 {
		weightMap[name] = "model-00001-of-00002.safetensors"
	}
	for name := range shard2 {
		weightMap[name] = "model-00002-of-00002.safetensors"
	}
	if err := writeJSON(filepath.Join(dir, "model.safetensors.index.json"), map[string]any{"metadata": map[string]any{"total_size": payload1 + payload2}, "weight_map": weightMap}); err != nil {
		t.Fatalf("write index: %v", err)
	}
	report, err := VerifyPretrainedBERTWeightsFromDir(dir, plan)
	if err != nil {
		t.Fatalf("verify sharded weights: %v", err)
	}
	if report.Status != "ok" {
		t.Fatalf("status = %q report = %+v", report.Status, report)
	}
	if !slices.Equal(report.Files, []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}) {
		t.Fatalf("files = %v", report.Files)
	}
}

func TestLoadPretrainedBERTWeightsLoadsPlannedAndPresentPoolerOnly(t *testing.T) {
	dir := t.TempDir()
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             3,
		HiddenSize:            2,
		NumHiddenLayers:       1,
		NumAttentionHeads:     1,
		IntermediateSize:      4,
		MaxPositionEmbeddings: 4,
		TypeVocabSize:         2,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	header := safeTensorsHeaderForBERTPlan(plan)
	header["pooler.dense.weight"] = map[string]any{"dtype": "F32", "shape": []int64{2, 2}, "data_offsets": []int64{0, 16}}
	header["pooler.dense.bias"] = map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}}
	header["cls.predictions.decoder.weight"] = map[string]any{"dtype": "F32", "shape": []int64{3, 2}, "data_offsets": []int64{0, 24}}
	payloadSize := renumberSafeTensorFixtureOffsets(header)
	payload := bytes.Repeat([]byte{0x5a}, payloadSize)
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model.safetensors"), header, payload); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	set, report, err := LoadPretrainedBERTWeightsFromDir(dir, plan)
	if err != nil {
		t.Fatalf("load weights: %v", err)
	}
	if report.Status != "ok" {
		t.Fatalf("status = %q", report.Status)
	}
	if !slices.Contains(report.Loaded, "pooler.dense.weight") || !slices.Contains(report.Loaded, "pooler.dense.bias") {
		t.Fatalf("expected pooler loaded, got %v", report.Loaded)
	}
	if slices.Contains(report.Loaded, "cls.predictions.decoder.weight") {
		t.Fatalf("unexpected classifier tensor loaded: %v", report.Loaded)
	}
	if !slices.Contains(report.SkippedExtra, "cls.predictions.decoder.weight") {
		t.Fatalf("expected classifier skipped, got %v", report.SkippedExtra)
	}
	for _, tensor := range set.Tensors {
		if tensor.Name == "embeddings.word_embeddings.weight" {
			if tensor.Role != "token_embeddings" || tensor.ByteLength != 24 || !bytes.Equal(tensor.Bytes, bytes.Repeat([]byte{0x5a}, 24)) {
				t.Fatalf("word embedding tensor = %+v bytes=%v", tensor, tensor.Bytes)
			}
			return
		}
	}
	t.Fatalf("word embedding tensor not loaded")
}

func TestLoadPretrainedBERTDecodedWeightsPreservesPlanOrderRolesDTypesAndValues(t *testing.T) {
	dir := t.TempDir()
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             3,
		HiddenSize:            2,
		NumHiddenLayers:       1,
		NumAttentionHeads:     1,
		IntermediateSize:      4,
		MaxPositionEmbeddings: 4,
		TypeVocabSize:         2,
	}
	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	header := safeTensorsHeaderForBERTPlan(plan)
	header["embeddings.position_embeddings.weight"].(map[string]any)["dtype"] = "F16"
	header["embeddings.token_type_embeddings.weight"].(map[string]any)["dtype"] = "BF16"
	header["pooler.dense.weight"] = map[string]any{"dtype": "F32", "shape": []int64{2, 2}, "data_offsets": []int64{0, 16}}
	header["pooler.dense.bias"] = map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}}
	header["cls.predictions.decoder.weight"] = map[string]any{"dtype": "F32", "shape": []int64{3, 2}, "data_offsets": []int64{0, 24}}
	payloadSize := renumberSafeTensorFixtureOffsets(header)
	payload := make([]byte, payloadSize)
	putF32Payload(t, header, payload, "embeddings.word_embeddings.weight", []float32{-1.25, 3.5, 0.5, -0.75, 2, -4})
	putU16Payload(t, header, payload, "embeddings.position_embeddings.weight", []uint16{0xc000, 0x3e00})
	putU16Payload(t, header, payload, "embeddings.token_type_embeddings.weight", []uint16{0xc020, 0x4050})
	putF32Payload(t, header, payload, "pooler.dense.bias", []float32{7, -8})
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model.safetensors"), header, payload); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	set, report, err := LoadPretrainedBERTDecodedWeightsFromDir(dir, plan)
	if err != nil {
		t.Fatalf("load decoded weights: %v", err)
	}
	if report.Status != "ok" {
		t.Fatalf("status = %q", report.Status)
	}
	if report.TensorCount != len(plan.Tensors) || len(set.Tensors) != len(plan.Tensors) {
		t.Fatalf("decoded tensor count report=%d set=%d plan=%d", report.TensorCount, len(set.Tensors), len(plan.Tensors))
	}
	if report.TotalElements == 0 {
		t.Fatalf("total elements = %d", report.TotalElements)
	}
	if report.SourceDTypes["F32"] == 0 || report.SourceDTypes["F16"] != 1 || report.SourceDTypes["BF16"] != 1 {
		t.Fatalf("source dtype counts = %+v", report.SourceDTypes)
	}
	if !slices.Contains(report.Loaded, "pooler.dense.weight") || !slices.Contains(report.Loaded, "pooler.dense.bias") {
		t.Fatalf("expected pooler loaded, got %v", report.Loaded)
	}
	if !slices.Contains(report.SkippedExtra, "cls.predictions.decoder.weight") {
		t.Fatalf("expected classifier skipped, got %v", report.SkippedExtra)
	}
	if set.Tensors[0].Name != plan.Tensors[0].Name || set.Tensors[1].Name != plan.Tensors[1].Name {
		t.Fatalf("decoded order = %s, %s; want %s, %s", set.Tensors[0].Name, set.Tensors[1].Name, plan.Tensors[0].Name, plan.Tensors[1].Name)
	}
	if set.Tensors[0].Role != "token_embeddings" || set.Tensors[0].SourceDType != "F32" {
		t.Fatalf("word embedding decoded tensor = %+v", set.Tensors[0])
	}
	assertFloat32Values(t, set.Tensors[0].Values[:3], []float32{-1.25, 3.5, 0.5})
	if set.Tensors[1].SourceDType != "F16" {
		t.Fatalf("position source dtype = %q", set.Tensors[1].SourceDType)
	}
	assertFloat32Values(t, set.Tensors[1].Values[:2], []float32{-2, 1.5})
	if set.Tensors[2].SourceDType != "BF16" {
		t.Fatalf("token type source dtype = %q", set.Tensors[2].SourceDType)
	}
	assertFloat32Values(t, set.Tensors[2].Values[:2], []float32{-2.5, 3.25})
}

func TestBuildPretrainedBERTWeightFileFromDecodedUsesRolesAndF32Storage(t *testing.T) {
	set := PretrainedBERTDecodedWeightSet{Tensors: []PretrainedBERTDecodedWeightTensor{
		{
			Name:        "embeddings.word_embeddings.weight",
			Role:        "token_embeddings",
			SourceDType: "BF16",
			Shape:       []int64{2, 2},
			SourceFile:  "model.safetensors",
			Values:      []float32{1, 2, 3, 4},
		},
		{
			Name:        "encoder.layer.0.attention.self.query.bias",
			Role:        "encoder_layer_0_attention_query_bias",
			SourceDType: "F32",
			Shape:       []int64{2},
			SourceFile:  "model.safetensors",
			Values:      []float32{-1, 0.5},
		},
	}}
	weightFile, report, err := BuildPretrainedBERTWeightFileFromDecoded(set)
	if err != nil {
		t.Fatalf("build weight file: %v", err)
	}
	if report.Status != "ok" || report.TensorCount != 2 || report.TotalElements != 6 {
		t.Fatalf("report = %+v", report)
	}
	if report.StorageDTypes["f32"] != 2 {
		t.Fatalf("storage dtype counts = %+v", report.StorageDTypes)
	}
	if report.SourceDTypes["BF16"] != 1 || report.SourceDTypes["F32"] != 1 {
		t.Fatalf("source dtype counts = %+v", report.SourceDTypes)
	}
	if len(report.Loaded) != 2 || report.Loaded[0].Role != "encoder_layer_0_attention_query_bias" || report.Loaded[1].Role != "token_embeddings" {
		t.Fatalf("loaded order = %+v", report.Loaded)
	}
	if _, ok := weightFile.Weights["embeddings.word_embeddings.weight"]; ok {
		t.Fatalf("weight file should use role names, got raw HF name")
	}
	token := weightFile.Weights["token_embeddings"]
	if token == nil {
		t.Fatalf("missing token_embeddings weight")
	}
	if token.DType != "f32" || !slices.Equal(token.Shape, []int{2, 2}) {
		t.Fatalf("token tensor dtype/shape = %s %v", token.DType, token.Shape)
	}
	assertFloat32Values(t, token.F32, []float32{1, 2, 3, 4})

	path := filepath.Join(t.TempDir(), "bert.weights.mll")
	if err := weightFile.WriteFile(path); err != nil {
		t.Fatalf("write weight file: %v", err)
	}
	roundTrip, err := ReadWeightFile(path)
	if err != nil {
		t.Fatalf("read weight file: %v", err)
	}
	roundToken := roundTrip.Weights["token_embeddings"]
	if roundToken == nil || roundToken.DType != "f32" || !slices.Equal(roundToken.Shape, []int{2, 2}) {
		t.Fatalf("round-trip token tensor = %+v", roundToken)
	}
	assertFloat32Values(t, roundToken.F32, []float32{1, 2, 3, 4})
}

func TestBuildPretrainedBERTWeightFileFromDecodedRejectsInvalidTensors(t *testing.T) {
	_, _, err := BuildPretrainedBERTWeightFileFromDecoded(PretrainedBERTDecodedWeightSet{Tensors: []PretrainedBERTDecodedWeightTensor{
		{Name: "a", Role: "dup", SourceDType: "F32", Shape: []int64{1}, Values: []float32{1}},
		{Name: "b", Role: "dup", SourceDType: "F32", Shape: []int64{1}, Values: []float32{2}},
	}})
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("expected duplicate role error, got %v", err)
	}

	_, _, err = BuildPretrainedBERTWeightFileFromDecoded(PretrainedBERTDecodedWeightSet{Tensors: []PretrainedBERTDecodedWeightTensor{
		{Name: "bad", Role: "bad", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{1, 2, 3}},
	}})
	if err == nil || !strings.Contains(err.Error(), "does not match shape elements") {
		t.Fatalf("expected element count error, got %v", err)
	}
}

func assertBERTTensorPlan(t *testing.T, plan PretrainedBERTImportPlan, name string, shape []int, required bool) {
	t.Helper()
	for _, tensor := range plan.Tensors {
		if tensor.Name != name {
			continue
		}
		if tensor.Required != required {
			t.Fatalf("%s required = %v, want %v", name, tensor.Required, required)
		}
		if len(tensor.Shape) != len(shape) {
			t.Fatalf("%s shape = %v, want %v", name, tensor.Shape, shape)
		}
		for i := range shape {
			if tensor.Shape[i] != shape[i] {
				t.Fatalf("%s shape = %v, want %v", name, tensor.Shape, shape)
			}
		}
		return
	}
	t.Fatalf("tensor %q not found in plan", name)
}

func paramNames(params []eosartifact.Param) []string {
	names := make([]string, 0, len(params))
	for _, param := range params {
		names = append(names, param.Name)
	}
	return names
}

func pretrainedBERTSingleLayerDecodedWeights() []PretrainedBERTDecodedWeightTensor {
	return []PretrainedBERTDecodedWeightTensor{
		{Name: "encoder.layer.0.attention.self.query.weight", Role: "encoder_layer_0_attention_query_weight", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{2, -1, 0.5, 3}},
		{Name: "encoder.layer.0.attention.self.query.bias", Role: "encoder_layer_0_attention_query_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.25, -0.5}},
		{Name: "encoder.layer.0.attention.self.key.weight", Role: "encoder_layer_0_attention_key_weight", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{1, 4, -2, 0.5}},
		{Name: "encoder.layer.0.attention.self.key.bias", Role: "encoder_layer_0_attention_key_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{-0.25, 0.75}},
		{Name: "encoder.layer.0.attention.self.value.weight", Role: "encoder_layer_0_attention_value_weight", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{1.5, -0.25, 0.75, 2}},
		{Name: "encoder.layer.0.attention.self.value.bias", Role: "encoder_layer_0_attention_value_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.1, -0.2}},
		{Name: "encoder.layer.0.attention.output.dense.weight", Role: "encoder_layer_0_attention_output_weight", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{0.5, -1.25, 1.5, 0.25}},
		{Name: "encoder.layer.0.attention.output.dense.bias", Role: "encoder_layer_0_attention_output_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.3, -0.4}},
		{Name: "encoder.layer.0.attention.output.LayerNorm.weight", Role: "encoder_layer_0_attention_layernorm_weight", SourceDType: "F32", Shape: []int64{2}, Values: []float32{1.2, -0.7}},
		{Name: "encoder.layer.0.attention.output.LayerNorm.bias", Role: "encoder_layer_0_attention_layernorm_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.05, 0.15}},
		{Name: "encoder.layer.0.intermediate.dense.weight", Role: "encoder_layer_0_intermediate_weight", SourceDType: "F32", Shape: []int64{3, 2}, Values: []float32{0.25, -0.5, 1.25, 0.75, -1.5, 0.5}},
		{Name: "encoder.layer.0.intermediate.dense.bias", Role: "encoder_layer_0_intermediate_bias", SourceDType: "F32", Shape: []int64{3}, Values: []float32{0.1, -0.2, 0.3}},
		{Name: "encoder.layer.0.output.dense.weight", Role: "encoder_layer_0_output_weight", SourceDType: "F32", Shape: []int64{2, 3}, Values: []float32{0.4, -0.8, 1.2, -1.1, 0.6, 0.2}},
		{Name: "encoder.layer.0.output.dense.bias", Role: "encoder_layer_0_output_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{-0.05, 0.25}},
		{Name: "encoder.layer.0.output.LayerNorm.weight", Role: "encoder_layer_0_output_layernorm_weight", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.9, 1.4}},
		{Name: "encoder.layer.0.output.LayerNorm.bias", Role: "encoder_layer_0_output_layernorm_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{-0.2, 0.4}},
	}
}

type bertEmbeddingHostBackend struct{}

func (bertEmbeddingHostBackend) Kind() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (bertEmbeddingHostBackend) Capabilities() []string {
	return []string{eosartifact.CapabilityHostFallback}
}

func (bertEmbeddingHostBackend) CanLoad(mod *eosartifact.Module) bool {
	return mod != nil && mod.SupportsBackend(eosartifact.BackendCUDA)
}

func (bertEmbeddingHostBackend) Load(_ context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding) (backend.Executor, error) {
	return bertEmbeddingHostExecutor{mod: mod, weights: weights}, nil
}

type bertEmbeddingHostExecutor struct {
	mod     *eosartifact.Module
	weights map[string]backend.WeightBinding
}

func (e bertEmbeddingHostExecutor) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (e bertEmbeddingHostExecutor) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	return backend.ExecuteSymbolic(ctx, e.mod, e.weights, nil, nil, nil, eosartifact.BackendCUDA, req)
}

func safeTensorsHeaderForBERTPlan(plan PretrainedBERTImportPlan) map[string]any {
	header := make(map[string]any)
	for _, tensor := range plan.Tensors {
		if !tensor.Required {
			continue
		}
		shape := make([]int64, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			shape[i] = int64(dim)
		}
		header[tensor.Name] = map[string]any{
			"dtype":        "F32",
			"shape":        shape,
			"data_offsets": []int64{0, 0},
		}
	}
	renumberSafeTensorFixtureOffsets(header)
	return header
}

func renumberSafeTensorFixtureOffsets(header map[string]any) int {
	var offset int64
	for _, raw := range header {
		tensor := raw.(map[string]any)
		span := safeTensorFixtureSpan(tensor)
		tensor["data_offsets"] = []int64{offset, offset + span}
		offset += span
	}
	return int(offset)
}

func safeTensorFixtureSpan(tensor map[string]any) int64 {
	dtype := tensor["dtype"].(string)
	var dtypeSize int64
	switch dtype {
	case "F32":
		dtypeSize = 4
	case "F16", "BF16":
		dtypeSize = 2
	case "I64":
		dtypeSize = 8
	default:
		dtypeSize = 1
	}
	var elements int64 = 1
	for _, dim := range tensor["shape"].([]int64) {
		elements *= dim
	}
	return elements * dtypeSize
}

func putF32Payload(t *testing.T, header map[string]any, payload []byte, name string, values []float32) {
	t.Helper()
	tensor := header[name].(map[string]any)
	offsets := tensor["data_offsets"].([]int64)
	span := int(offsets[1] - offsets[0])
	if len(values)*4 > span {
		t.Fatalf("%s values need %d bytes, span is %d", name, len(values)*4, span)
	}
	start := int(offsets[0])
	for i, value := range values {
		binary.LittleEndian.PutUint32(payload[start+i*4:], math.Float32bits(value))
	}
}

func putU16Payload(t *testing.T, header map[string]any, payload []byte, name string, values []uint16) {
	t.Helper()
	tensor := header[name].(map[string]any)
	offsets := tensor["data_offsets"].([]int64)
	span := int(offsets[1] - offsets[0])
	if len(values)*2 > span {
		t.Fatalf("%s values need %d bytes, span is %d", name, len(values)*2, span)
	}
	start := int(offsets[0])
	for i, value := range values {
		binary.LittleEndian.PutUint16(payload[start+i*2:], value)
	}
}

func assertBERTShapeMismatch(t *testing.T, report PretrainedBERTWeightVerification, name string, expected []int, actual []int64) {
	t.Helper()
	for _, mismatch := range report.ShapeMismatches {
		if mismatch.Name != name {
			continue
		}
		if !slices.Equal(mismatch.Expected, expected) || !slices.Equal(mismatch.Actual, actual) {
			t.Fatalf("shape mismatch = %+v, want expected=%v actual=%v", mismatch, expected, actual)
		}
		return
	}
	t.Fatalf("shape mismatch %q not found in %+v", name, report.ShapeMismatches)
}

func assertBERTDTypeMismatch(t *testing.T, report PretrainedBERTWeightVerification, name, actual string) {
	t.Helper()
	for _, mismatch := range report.DTypeMismatches {
		if mismatch.Name != name {
			continue
		}
		if mismatch.Actual != actual {
			t.Fatalf("dtype mismatch = %+v, want actual=%q", mismatch, actual)
		}
		return
	}
	t.Fatalf("dtype mismatch %q not found in %+v", name, report.DTypeMismatches)
}
