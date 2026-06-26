package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosruntime "m31labs.dev/eos/runtime"
)

func TestRunImportPretrainedBERTWritesPlanJSON(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "hf-snapshot")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 7,
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"intermediate_size": 16,
		"hidden_act": "gelu",
		"max_position_embeddings": 16,
		"type_vocab_size": 2
	}`
	if err := os.WriteFile(filepath.Join(snapshot, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n"
	if err := os.WriteFile(filepath.Join(snapshot, "vocab.txt"), []byte(vocab), 0o644); err != nil {
		t.Fatalf("write vocab: %v", err)
	}
	header := map[string]any{
		"embeddings.word_embeddings.weight": map[string]any{
			"dtype":        "F32",
			"shape":        []int64{7, 8},
			"data_offsets": []int64{0, 1},
		},
	}
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, []byte{0}); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	planPath := filepath.Join(dir, "plan.json")
	if err := runImportPretrainedBERT([]string{
		"--source", snapshot,
		"--model-name", "fixture/bert",
		"--plan-json", planPath,
		"--tokenizer-smoke", "hello world",
		"--verify-weights",
	}); err != nil {
		t.Fatalf("run import-pretrained-bert: %v", err)
	}
	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("read plan json: %v", err)
	}
	var plan eosruntime.PretrainedBERTImportPlan
	if err := json.Unmarshal(data, &plan); err != nil {
		t.Fatalf("parse plan json: %v\n%s", err, data)
	}
	if plan.Version != eosruntime.PretrainedBERTImporterPlanVersion {
		t.Fatalf("version = %q", plan.Version)
	}
	if plan.ModelName != "fixture/bert" {
		t.Fatalf("model_name = %q", plan.ModelName)
	}
	if !strings.Contains(plan.ExecutionStatus, "plan_only") {
		t.Fatalf("execution_status = %q", plan.ExecutionStatus)
	}
	if plan.WeightVerification == nil {
		t.Fatal("expected weight verification report")
	}
	if plan.WeightVerification.Status != "mismatch" {
		t.Fatalf("weight verification status = %q", plan.WeightVerification.Status)
	}
	if !strings.Contains(strings.Join(plan.WeightVerification.Missing, ","), "embeddings.position_embeddings.weight") {
		t.Fatalf("expected missing required tensors, got %+v", plan.WeightVerification.Missing)
	}
	var foundWordEmbedding bool
	for _, tensor := range plan.Tensors {
		if tensor.Name == "embeddings.word_embeddings.weight" {
			foundWordEmbedding = true
			if len(tensor.Shape) != 2 || tensor.Shape[0] != 7 || tensor.Shape[1] != 8 {
				t.Fatalf("word embedding shape = %v", tensor.Shape)
			}
			if !tensor.Required {
				t.Fatal("word embedding tensor should be required")
			}
		}
	}
	if !foundWordEmbedding {
		t.Fatalf("plan missing embeddings.word_embeddings.weight: %+v", plan.Tensors)
	}
}

func writeCommandSafeTensorsFixture(path string, header map[string]any, payload []byte) error {
	data, err := json.Marshal(header)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	buf.Write(data)
	buf.Write(payload)
	return os.WriteFile(path, buf.Bytes(), 0o644)
}
