package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
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
			"data_offsets": []int64{0, 224},
		},
	}
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, make([]byte, 224)); err != nil {
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

func TestRunImportPretrainedBERTLoadWeightsSmoke(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "hf-snapshot")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 3,
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
	plan, err := eosruntime.PlanPretrainedBERTImportFromDir(snapshot, "fixture")
	if err != nil {
		t.Fatalf("plan fixture: %v", err)
	}
	header := commandSafeTensorsHeaderForBERTPlan(plan)
	header["cls.predictions.decoder.weight"] = map[string]any{
		"dtype":        "F32",
		"shape":        []int64{3, 2},
		"data_offsets": []int64{0, 24},
	}
	payloadSize := renumberCommandSafeTensorFixtureOffsets(header)
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, bytes.Repeat([]byte{7}, payloadSize)); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	planPath := filepath.Join(dir, "plan.json")
	if err := runImportPretrainedBERT([]string{
		"--source", snapshot,
		"--plan-json", planPath,
		"--verify-weights",
		"--load-weights-smoke",
		"--decode-weights-smoke",
	}); err != nil {
		t.Fatalf("run import-pretrained-bert: %v", err)
	}
	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("read plan json: %v", err)
	}
	var loaded eosruntime.PretrainedBERTImportPlan
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("parse plan json: %v\n%s", err, data)
	}
	if loaded.WeightLoadSmoke == nil {
		t.Fatal("expected load smoke report")
	}
	if loaded.WeightLoadSmoke.Status != "ok" || loaded.WeightLoadSmoke.TotalBytes == 0 {
		t.Fatalf("load smoke = %+v", loaded.WeightLoadSmoke)
	}
	if !slices.Contains(loaded.WeightLoadSmoke.SkippedExtra, "cls.predictions.decoder.weight") {
		t.Fatalf("expected classifier skipped, got %+v", loaded.WeightLoadSmoke.SkippedExtra)
	}
	if loaded.WeightDecodeSmoke == nil {
		t.Fatal("expected decode smoke report")
	}
	if loaded.WeightDecodeSmoke.Status != "ok" || loaded.WeightDecodeSmoke.TotalElements == 0 {
		t.Fatalf("decode smoke = %+v", loaded.WeightDecodeSmoke)
	}
	if loaded.WeightDecodeSmoke.SourceDTypes["F32"] == 0 {
		t.Fatalf("expected F32 source dtype count, got %+v", loaded.WeightDecodeSmoke.SourceDTypes)
	}
	if !slices.Contains(loaded.WeightDecodeSmoke.SkippedExtra, "cls.predictions.decoder.weight") {
		t.Fatalf("expected classifier skipped by decode, got %+v", loaded.WeightDecodeSmoke.SkippedExtra)
	}
}

func TestRunImportPretrainedBERTWeightsOutWritesReadableWeightFile(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "hf-snapshot")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 3,
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
	plan, err := eosruntime.PlanPretrainedBERTImportFromDir(snapshot, "fixture")
	if err != nil {
		t.Fatalf("plan fixture: %v", err)
	}
	header := commandSafeTensorsHeaderForBERTPlan(plan)
	header["embeddings.token_type_embeddings.weight"].(map[string]any)["dtype"] = "BF16"
	payloadSize := renumberCommandSafeTensorFixtureOffsets(header)
	payload := make([]byte, payloadSize)
	putCommandF32Payload(t, header, payload, "embeddings.word_embeddings.weight", []float32{1, 2, 3, 4, 5, 6})
	putCommandU16Payload(t, header, payload, "embeddings.token_type_embeddings.weight", []uint16{0x3f80, 0xc000, 0x4020, 0x0000})
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, payload); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	planPath := filepath.Join(dir, "plan.json")
	weightsPath := filepath.Join(dir, "bert.weights.mll")
	modulePath := filepath.Join(dir, "bert.module.mll")
	packagePath := filepath.Join(dir, "bert.imported.mll")
	if err := runImportPretrainedBERT([]string{
		"--source", snapshot,
		"--plan-json", planPath,
		"--weights-out", weightsPath,
		"--module-out", modulePath,
		"--package-out", packagePath,
	}); err != nil {
		t.Fatalf("run import-pretrained-bert: %v", err)
	}
	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("read plan json: %v", err)
	}
	var loaded eosruntime.PretrainedBERTImportPlan
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("parse plan json: %v\n%s", err, data)
	}
	if loaded.WeightFileExport == nil {
		t.Fatal("expected weight file export report")
	}
	if loaded.WeightFileExport.OutputPath != weightsPath || loaded.WeightFileExport.TensorCount != len(plan.Tensors)-2 {
		t.Fatalf("weight file export = %+v", loaded.WeightFileExport)
	}
	if loaded.WeightFileExport.StorageDTypes["f32"] != loaded.WeightFileExport.TensorCount {
		t.Fatalf("storage dtype counts = %+v", loaded.WeightFileExport.StorageDTypes)
	}
	if loaded.WeightFileExport.SourceDTypes["BF16"] != 1 {
		t.Fatalf("source dtype counts = %+v", loaded.WeightFileExport.SourceDTypes)
	}
	if loaded.ModuleExport == nil {
		t.Fatal("expected module export report")
	}
	if loaded.ModuleExport.OutputPath != modulePath || loaded.ModuleExport.Entrypoint != "bert_embed" {
		t.Fatalf("module export = %+v", loaded.ModuleExport)
	}
	if loaded.ModuleExport.Pooling != "masked_mean" || loaded.ModuleExport.Normalization != "l2" {
		t.Fatalf("module export pooling/normalization = %+v", loaded.ModuleExport)
	}
	if loaded.ModuleExport.Layers != 1 || loaded.ModuleExport.HiddenSize != 2 || !strings.Contains(loaded.ModuleExport.ExecutionStatus, "host_reference_full_stack") {
		t.Fatalf("module export execution/config = %+v", loaded.ModuleExport)
	}

	weightFile, err := eosruntime.ReadWeightFile(weightsPath)
	if err != nil {
		t.Fatalf("read exported weight file: %v", err)
	}
	token := weightFile.Weights["token_embeddings"]
	if token == nil {
		t.Fatalf("missing token_embeddings in weight file")
	}
	if token.DType != "f32" || !slices.Equal(token.Shape, []int{3, 2}) {
		t.Fatalf("token tensor dtype/shape = %s %v", token.DType, token.Shape)
	}
	assertCommandFloat32Values(t, token.F32, []float32{1, 2, 3, 4, 5, 6})
	if _, ok := weightFile.Weights["embeddings.word_embeddings.weight"]; ok {
		t.Fatalf("weight file should use role names, got raw HF tensor key")
	}
	tokenTypes := weightFile.Weights["token_type_embeddings"]
	if tokenTypes == nil || tokenTypes.DType != "f32" || !slices.Equal(tokenTypes.Shape, []int{2, 2}) {
		t.Fatalf("token_type_embeddings = %+v", tokenTypes)
	}
	assertCommandFloat32Values(t, tokenTypes.F32, []float32{1, -2, 2.5, 0})

	mod, err := eosartifact.ReadFile(modulePath)
	if err != nil {
		t.Fatalf("read exported module: %v", err)
	}
	if mod.Name != "pretrained_bert_embedder" {
		t.Fatalf("module name = %q", mod.Name)
	}
	if len(mod.EntryPoints) == 0 || mod.EntryPoints[0].Name != "bert_embed" {
		t.Fatalf("entrypoints = %+v", mod.EntryPoints)
	}
	if len(mod.Steps) == 0 || mod.Steps[0].Kind != eosartifact.StepBERTEmbedder {
		t.Fatalf("first module step = %+v", mod.Steps)
	}
	if loaded.PackageExport == nil {
		t.Fatal("expected package export report")
	}
	if loaded.PackageExport.OutputPath != packagePath || loaded.PackageExport.NativeDim != 2 || loaded.PackageExport.MaxLength != 4 {
		t.Fatalf("package export = %+v", loaded.PackageExport)
	}
	if !strings.Contains(loaded.PackageExport.ExecutionStatus, "source_free_package_host_reference_full_stack") {
		t.Fatalf("package export execution_status = %q", loaded.PackageExport.ExecutionStatus)
	}
	if loaded.ExecutionStatus != loaded.PackageExport.ExecutionStatus {
		t.Fatalf("plan execution_status = %q, want package export status %q", loaded.ExecutionStatus, loaded.PackageExport.ExecutionStatus)
	}
	if !isCommandSHA256Hex(loaded.PackageExport.IdentitySHA256) || !isCommandSHA256Hex(loaded.PackageExport.ModuleSHA256) ||
		!isCommandSHA256Hex(loaded.PackageExport.WeightsSHA256) || !isCommandSHA256Hex(loaded.PackageExport.ConfigSHA256) ||
		!isCommandSHA256Hex(loaded.PackageExport.VocabSHA256) {
		t.Fatalf("package export hashes = %+v", loaded.PackageExport)
	}
	pkg, err := eosruntime.ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read package: %v", err)
	}
	if pkg.IdentitySHA256 != loaded.PackageExport.IdentitySHA256 || pkg.ModelName != "" || pkg.NativeDim != 2 {
		t.Fatalf("package identity/model/dim = %q/%q/%d", pkg.IdentitySHA256, pkg.ModelName, pkg.NativeDim)
	}
	if _, err := pkg.Tokenizer(); err != nil {
		t.Fatalf("package tokenizer: %v", err)
	}
}

func TestRunImportPretrainedBERTModuleOutUpdatesPlanExecutionStatus(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "hf-snapshot")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 3,
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
	plan, err := eosruntime.PlanPretrainedBERTImportFromDir(snapshot, "fixture")
	if err != nil {
		t.Fatalf("plan fixture: %v", err)
	}
	header := commandSafeTensorsHeaderForBERTPlan(plan)
	payloadSize := renumberCommandSafeTensorFixtureOffsets(header)
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, make([]byte, payloadSize)); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	planPath := filepath.Join(dir, "plan.json")
	modulePath := filepath.Join(dir, "bert.module.mll")
	if err := runImportPretrainedBERT([]string{
		"--source", snapshot,
		"--module-out", modulePath,
		"--plan-json", planPath,
	}); err != nil {
		t.Fatalf("run import-pretrained-bert: %v", err)
	}
	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("read plan json: %v", err)
	}
	var loaded eosruntime.PretrainedBERTImportPlan
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("parse plan json: %v\n%s", err, data)
	}
	if loaded.ModuleExport == nil {
		t.Fatal("expected module export report")
	}
	if strings.Contains(loaded.ExecutionStatus, "plan_only") {
		t.Fatalf("plan execution_status stayed stale: %q", loaded.ExecutionStatus)
	}
	if !strings.Contains(loaded.ExecutionStatus, "host_reference_full_stack") {
		t.Fatalf("plan execution_status = %q", loaded.ExecutionStatus)
	}
	if loaded.ExecutionStatus != loaded.ModuleExport.ExecutionStatus {
		t.Fatalf("plan execution_status = %q, want module export status %q", loaded.ExecutionStatus, loaded.ModuleExport.ExecutionStatus)
	}
}

func TestRunImportPretrainedBERTPackageReportIncludesRetrievalRoleContract(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "hf-snapshot")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatalf("mkdir snapshot: %v", err)
	}
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 3,
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
	if err := os.MkdirAll(filepath.Join(snapshot, "1_Pooling"), 0o755); err != nil {
		t.Fatalf("mkdir pooling: %v", err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "1_Pooling", "config.json"), []byte(`{"pooling_mode_mean_tokens":true}`+"\n"), 0o644); err != nil {
		t.Fatalf("write pooling config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "sentence_bert_config.json"), []byte(`{"max_seq_length":4}`+"\n"), 0o644); err != nil {
		t.Fatalf("write sentence_bert_config: %v", err)
	}
	plan, err := eosruntime.PlanPretrainedBERTImportFromDir(snapshot, "intfloat/e5-small-v2")
	if err != nil {
		t.Fatalf("plan fixture: %v", err)
	}
	header := commandSafeTensorsHeaderForBERTPlan(plan)
	payloadSize := renumberCommandSafeTensorFixtureOffsets(header)
	if err := writeCommandSafeTensorsFixture(filepath.Join(snapshot, "model.safetensors"), header, make([]byte, payloadSize)); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	planPath := filepath.Join(dir, "plan.json")
	packagePath := filepath.Join(dir, "e5.imported.mll")
	if err := runImportPretrainedBERT([]string{
		"--source", snapshot,
		"--model-name", "intfloat/e5-small-v2",
		"--package-out", packagePath,
		"--plan-json", planPath,
	}); err != nil {
		t.Fatalf("run import-pretrained-bert: %v", err)
	}
	data, err := os.ReadFile(planPath)
	if err != nil {
		t.Fatalf("read plan json: %v", err)
	}
	var loaded eosruntime.PretrainedBERTImportPlan
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("parse plan json: %v\n%s", err, data)
	}
	if loaded.PackageExport == nil || loaded.PackageExport.RetrievalRoleContract == nil {
		t.Fatalf("package export retrieval contract = %+v", loaded.PackageExport)
	}
	if loaded.PackageExport.RetrievalRoleContract.QueryPrefix != "query: " || loaded.PackageExport.RetrievalRoleContract.DocumentPrefix != "passage: " {
		t.Fatalf("package export prefixes = %+v", loaded.PackageExport.RetrievalRoleContract)
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

func commandSafeTensorsHeaderForBERTPlan(plan eosruntime.PretrainedBERTImportPlan) map[string]any {
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
	renumberCommandSafeTensorFixtureOffsets(header)
	return header
}

func renumberCommandSafeTensorFixtureOffsets(header map[string]any) int {
	var offset int64
	for _, raw := range header {
		tensor := raw.(map[string]any)
		dtype := tensor["dtype"].(string)
		var dtypeSize int64 = 4
		switch dtype {
		case "F16", "BF16":
			dtypeSize = 2
		case "I64":
			dtypeSize = 8
		}
		var elements int64 = 1
		for _, dim := range tensor["shape"].([]int64) {
			elements *= dim
		}
		span := elements * dtypeSize
		tensor["data_offsets"] = []int64{offset, offset + span}
		offset += span
	}
	return int(offset)
}

func putCommandF32Payload(t *testing.T, header map[string]any, payload []byte, name string, values []float32) {
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

func putCommandU16Payload(t *testing.T, header map[string]any, payload []byte, name string, values []uint16) {
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

func assertCommandFloat32Values(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want=%d got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%f want %f; got=%v want=%v", i, got[i], want[i], got, want)
		}
	}
}

func isCommandSHA256Hex(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, ch := range value {
		if (ch < '0' || ch > '9') && (ch < 'a' || ch > 'f') {
			return false
		}
	}
	return true
}
