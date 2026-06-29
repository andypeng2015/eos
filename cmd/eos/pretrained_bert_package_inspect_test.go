package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosruntime "m31labs.dev/eos/runtime"
)

func TestRunInspectPretrainedBERTPackageText(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "fixture/bert", false)

	output := captureRunOutput(t, []string{"inspect-pretrained-bert-package", packagePath})

	if !strings.Contains(output, "model_name: fixture/bert") {
		t.Fatalf("missing model_name in output:\n%s", output)
	}
	if !strings.Contains(output, "architecture: BertModel") || !strings.Contains(output, "model_type: bert") {
		t.Fatalf("missing architecture/model_type in output:\n%s", output)
	}
	if !strings.Contains(output, "files: count=") || !strings.Contains(output, "roles=config") {
		t.Fatalf("missing file summary in output:\n%s", output)
	}
	if strings.Contains(output, "retrieval_role_contract:") {
		t.Fatalf("legacy package should not print retrieval role contract:\n%s", output)
	}
}

func TestRunInspectPretrainedBERTPackageJSON(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "intfloat/e5-small-v2", true)

	output := captureRunOutput(t, []string{"inspect-pretrained-bert-package", "--json", packagePath})

	var got struct {
		ModelName             string                                          `json:"model_name"`
		Architecture          string                                          `json:"architecture"`
		ModelType             string                                          `json:"model_type"`
		Pooling               string                                          `json:"pooling"`
		Normalization         string                                          `json:"normalization"`
		MaxLength             int                                             `json:"max_length"`
		NativeDim             int                                             `json:"native_dim"`
		IdentitySHA256        string                                          `json:"identity_sha256"`
		ModuleSHA256          string                                          `json:"module_sha256"`
		WeightsSHA256         string                                          `json:"weights_sha256"`
		FileCount             int                                             `json:"file_count"`
		FileRoles             []string                                        `json:"file_roles"`
		RetrievalRoleContract *eosruntime.PretrainedBERTRetrievalRoleContract `json:"retrieval_role_contract"`
	}
	if err := json.Unmarshal([]byte(output), &got); err != nil {
		t.Fatalf("parse inspect json: %v\n%s", err, output)
	}
	if got.ModelName != "intfloat/e5-small-v2" || got.Architecture != "BertModel" || got.ModelType != "bert" {
		t.Fatalf("identity fields = %+v", got)
	}
	if got.Pooling != "masked_mean" || got.Normalization != "l2" || got.MaxLength != 4 || got.NativeDim != 2 {
		t.Fatalf("shape/pooling fields = %+v", got)
	}
	if got.FileCount == 0 || len(got.FileRoles) == 0 {
		t.Fatalf("file summary = %+v", got)
	}
	if got.RetrievalRoleContract == nil {
		t.Fatalf("missing retrieval role contract: %+v", got)
	}
	if got.RetrievalRoleContract.QueryPrefix != "query: " || got.RetrievalRoleContract.DocumentPrefix != "passage: " {
		t.Fatalf("retrieval role contract = %+v", got.RetrievalRoleContract)
	}
}

func writeCommandPretrainedBERTPackageFixture(t *testing.T, modelName string, withST bool) string {
	t.Helper()
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "hf-snapshot")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 5,
		"hidden_size": 2,
		"num_hidden_layers": 1,
		"num_attention_heads": 1,
		"intermediate_size": 4,
		"hidden_act": "gelu",
		"max_position_embeddings": 4,
		"type_vocab_size": 2
	}`
	if err := os.WriteFile(filepath.Join(snapshot, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "vocab.txt"), []byte("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"), 0o644); err != nil {
		t.Fatalf("write vocab: %v", err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "tokenizer_config.json"), []byte(`{"do_lower_case":true}`+"\n"), 0o644); err != nil {
		t.Fatalf("write tokenizer config: %v", err)
	}
	if withST {
		if err := os.MkdirAll(filepath.Join(snapshot, "1_Pooling"), 0o755); err != nil {
			t.Fatalf("mkdir pooling: %v", err)
		}
		if err := os.WriteFile(filepath.Join(snapshot, "1_Pooling", "config.json"), []byte(`{"pooling_mode_mean_tokens":true}`+"\n"), 0o644); err != nil {
			t.Fatalf("write pooling config: %v", err)
		}
		if err := os.WriteFile(filepath.Join(snapshot, "sentence_bert_config.json"), []byte(`{"max_seq_length":4}`+"\n"), 0o644); err != nil {
			t.Fatalf("write sentence_bert_config: %v", err)
		}
	}
	plan, err := eosruntime.PlanPretrainedBERTImportFromDir(snapshot, modelName)
	if err != nil {
		t.Fatalf("plan fixture: %v", err)
	}
	header := commandSafeTensorsHeaderForBERTPlan(plan)
	payloadSize := renumberCommandSafeTensorFixtureOffsets(header)
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, make([]byte, payloadSize)); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	packagePath := filepath.Join(dir, "bert.imported.mll")
	planPath := filepath.Join(dir, "plan.json")
	if err := runImportPretrainedBERT([]string{
		"--source", snapshot,
		"--model-name", modelName,
		"--package-out", packagePath,
		"--plan-json", planPath,
	}); err != nil {
		t.Fatalf("run import-pretrained-bert: %v", err)
	}
	return packagePath
}
