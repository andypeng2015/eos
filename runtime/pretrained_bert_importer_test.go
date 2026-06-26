package eosruntime

import (
	"os"
	"path/filepath"
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
