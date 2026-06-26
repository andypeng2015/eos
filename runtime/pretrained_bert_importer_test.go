package eosruntime

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
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
		"data_offsets": []int64{int64(len(header) + 1), int64(len(header) + 2)},
	}
	renumberSafeTensorFixtureOffsets(header)
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model.safetensors"), header, make([]byte, len(header))); err != nil {
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

func TestVerifyPretrainedBERTWeightsFromDirRejectsShardedIndex(t *testing.T) {
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
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"metadata":{},"weight_map":{}}`), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	_, err = VerifyPretrainedBERTWeightsFromDir(dir, plan)
	if err == nil || !strings.Contains(err.Error(), "sharded index is not supported") {
		t.Fatalf("expected sharded index error, got %v", err)
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

func safeTensorsHeaderForBERTPlan(plan PretrainedBERTImportPlan) map[string]any {
	header := make(map[string]any)
	var offset int64
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
			"data_offsets": []int64{offset, offset + 1},
		}
		offset++
	}
	return header
}

func renumberSafeTensorFixtureOffsets(header map[string]any) {
	var offset int64
	for _, raw := range header {
		tensor := raw.(map[string]any)
		tensor["data_offsets"] = []int64{offset, offset + 1}
		offset++
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
