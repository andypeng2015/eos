package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backends/cuda"
)

func TestPretrainedBERTRetrievalVectorExportWritesEvaluatorCompatibleCaches(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	rt := New(cuda.New())
	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        rt,
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors: %v", err)
	}
	if summary.Documents != 1 || summary.Queries != 1 || summary.NativeDim != 2 || summary.OutputDim != 2 {
		t.Fatalf("summary counts/dims = %+v", summary)
	}
	if summary.ExecutionMode != "pretrained_bert_host_reference" || summary.QualityClaim {
		t.Fatalf("summary mode/quality = %+v", summary)
	}
	if summary.QueryPrefix != "query " || summary.DocumentPrefix != "doc " || summary.MaxLength != 4 || summary.BatchSize != 1 {
		t.Fatalf("summary config = %+v", summary)
	}
	if summary.DocVectorPath != filepath.Join(outputDir, "doc-vectors.jsonl") ||
		summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary paths = %+v", summary)
	}

	docRows := readTinyVectorRows(t, summary.DocVectorPath)
	queryRows := readTinyVectorRows(t, summary.QueryVectorPath)
	if got := rowIDs(docRows); !slices.Equal(got, []string{"d1"}) {
		t.Fatalf("doc ids = %v", got)
	}
	if got := rowIDs(queryRows); !slices.Equal(got, []string{"q1"}) {
		t.Fatalf("query ids = %v", got)
	}
	assertFiniteUnitishVector(t, docRows[0].Embedding, 2)
	assertFiniteUnitishVector(t, queryRows[0].Embedding, 2)

	manifestData, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.Schema != PretrainedBERTRetrievalVectorExportManifestSchema || manifest.SourceDir != sourceDir ||
		manifest.ModulePath != modulePath || manifest.WeightsPath != weightsPath {
		t.Fatalf("manifest = %+v", manifest)
	}

	_, _, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")
	metrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:          "tiny-bert",
		CorpusPath:           filepath.Join(datasetDir, "corpus.jsonl"),
		QueriesPath:          filepath.Join(datasetDir, "queries.jsonl"),
		QrelsPath:            qrelsPath,
		DocVectorPath:        summary.DocVectorPath,
		QueryVectorPath:      summary.QueryVectorPath,
		BackendName:          summary.ExecutionMode,
		TopK:                 1,
		AllowMissingRelevant: false,
	})
	if err != nil {
		t.Fatalf("evaluate vector cache: %v", err)
	}
	if metrics.Inputs.Documents != 1 || metrics.Inputs.Queries != 1 || metrics.Quality.HitAt1 != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestPretrainedBERTTextEmbedderRejectsTooLargeMaxLength(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	_, err := LoadPretrainedBERTTextEmbedder(context.Background(), PretrainedBERTTextEmbedderConfig{
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		MaxLength:   5,
		Runtime:     New(cuda.New()),
	})
	if err == nil {
		t.Fatalf("expected max length error")
	}
}

func writeTinyPretrainedBERTExportFixture(t *testing.T) (sourceDir, modulePath, weightsPath string) {
	t.Helper()
	dir := t.TempDir()
	sourceDir = filepath.Join(dir, "source")
	if err := os.MkdirAll(sourceDir, 0o755); err != nil {
		t.Fatalf("mkdir source: %v", err)
	}
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             8,
		HiddenSize:            2,
		NumHiddenLayers:       2,
		NumAttentionHeads:     1,
		IntermediateSize:      3,
		HiddenAct:             "gelu",
		MaxPositionEmbeddings: 4,
		TypeVocabSize:         2,
		LayerNormEps:          0.25,
	}
	configData, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "config.json"), append(configData, '\n'), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "vocab.txt"), []byte("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nquery\ndoc\n"), 0o644); err != nil {
		t.Fatalf("write vocab: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "tokenizer_config.json"), []byte(`{"do_lower_case":true}`+"\n"), 0o644); err != nil {
		t.Fatalf("write tokenizer config: %v", err)
	}

	plan, err := PlanPretrainedBERTImport(cfg, "fixture")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	mod, err := BuildPretrainedBERTEmbedderModule(plan)
	if err != nil {
		t.Fatalf("build module: %v", err)
	}
	modulePath = filepath.Join(dir, "bert_embed.mll")
	if err := eosartifact.WriteFile(modulePath, mod); err != nil {
		t.Fatalf("write module: %v", err)
	}
	weights, _, err := BuildPretrainedBERTWeightFileFromDecoded(PretrainedBERTDecodedWeightSet{Tensors: tinyPretrainedBERTExportDecodedWeights()})
	if err != nil {
		t.Fatalf("build weights: %v", err)
	}
	weightsPath = filepath.Join(dir, "bert_weights.mll")
	if err := weights.WriteFile(weightsPath); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	return sourceDir, modulePath, weightsPath
}

func tinyPretrainedBERTExportDecodedWeights() []PretrainedBERTDecodedWeightTensor {
	decoded := []PretrainedBERTDecodedWeightTensor{
		{Name: "embeddings.word_embeddings.weight", Role: "token_embeddings", SourceDType: "F32", Shape: []int64{8, 2}, Values: []float32{
			0, 0,
			0.1, -0.1,
			1, 0,
			0, 1,
			-1, 0,
			1, 2,
			2, 1,
			-0.5, 1.5,
		}},
		{Name: "embeddings.position_embeddings.weight", Role: "position_embeddings", SourceDType: "F32", Shape: []int64{4, 2}, Values: []float32{0, 1, 1, 0, -1, 0.5, 0.25, -0.25}},
		{Name: "embeddings.token_type_embeddings.weight", Role: "token_type_embeddings", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{0, 0, 2, -2}},
		{Name: "embeddings.LayerNorm.weight", Role: "embedding_layernorm_weight", SourceDType: "F32", Shape: []int64{2}, Values: []float32{2, 3}},
		{Name: "embeddings.LayerNorm.bias", Role: "embedding_layernorm_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.5, -0.5}},
	}
	layer0 := pretrainedBERTSingleLayerDecodedWeights()
	decoded = append(decoded, layer0...)
	for _, tensor := range layer0 {
		tensor.Name = strings.Replace(tensor.Name, ".0.", ".1.", 1)
		tensor.Role = strings.Replace(tensor.Role, "_0_", "_1_", 1)
		decoded = append(decoded, tensor)
	}
	return decoded
}

func writeTinyPretrainedBERTBEIRFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "corpus.jsonl"), []byte(`{"_id":"d1","title":"", "text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir
}

func readTinyVectorRows(t *testing.T, path string) []retrievalVectorExportRow {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer file.Close()
	var rows []retrievalVectorExportRow
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var row retrievalVectorExportRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatalf("parse row: %v", err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan rows: %v", err)
	}
	return rows
}

func rowIDs(rows []retrievalVectorExportRow) []string {
	out := make([]string, len(rows))
	for i, row := range rows {
		out[i] = row.ID
	}
	return out
}

func assertFiniteUnitishVector(t *testing.T, vector []float32, dim int) {
	t.Helper()
	if len(vector) != dim {
		t.Fatalf("vector dim = %d, want %d", len(vector), dim)
	}
	var norm float64
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("vector[%d] = %v, want finite", i, value)
		}
		norm += float64(value * value)
	}
	if math.Sqrt(norm) < 0.9 || math.Sqrt(norm) > 1.1 {
		t.Fatalf("vector norm = %g, want near 1", math.Sqrt(norm))
	}
}
