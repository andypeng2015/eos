package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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
	if summary.QueryPrefix != "query " || summary.DocumentPrefix != "doc " || summary.LegacyDocPrefix != "doc " || summary.MaxLength != 4 || summary.BatchSize != 1 {
		t.Fatalf("summary config = %+v", summary)
	}
	if !summary.DocumentRoleApplied || !summary.QueryRoleApplied {
		t.Fatalf("summary role flags = doc:%v query:%v, want true/true", summary.DocumentRoleApplied, summary.QueryRoleApplied)
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
	if manifest.DocumentPrefix != "doc " || manifest.LegacyDocPrefix != "doc " || manifest.QueryPrefix != "query " {
		t.Fatalf("manifest prefixes = canonical:%q legacy:%q query:%q", manifest.DocumentPrefix, manifest.LegacyDocPrefix, manifest.QueryPrefix)
	}
	if !manifest.DocumentRoleApplied || !manifest.QueryRoleApplied {
		t.Fatalf("manifest role flags = doc:%v query:%v, want true/true", manifest.DocumentRoleApplied, manifest.QueryRoleApplied)
	}
	var manifestJSON map[string]any
	if err := json.Unmarshal(manifestData, &manifestJSON); err != nil {
		t.Fatalf("parse manifest json object: %v", err)
	}
	if manifestJSON["document_prefix"] != "doc " || manifestJSON["doc_prefix"] != "doc " {
		t.Fatalf("manifest json prefixes = document_prefix:%v doc_prefix:%v", manifestJSON["document_prefix"], manifestJSON["doc_prefix"])
	}
	if manifestJSON["document_role_applied"] != true || manifestJSON["query_role_applied"] != true {
		t.Fatalf("manifest json role flags = doc:%v query:%v", manifestJSON["document_role_applied"], manifestJSON["query_role_applied"])
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

func TestPretrainedBERTRetrievalVectorExportResumeAppendsPartialCaches(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 3)
	rt := New(cuda.New())
	seedOutput := filepath.Join(t.TempDir(), "seed")
	seed, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      seedOutput,
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
		t.Fatalf("seed export: %v", err)
	}
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	writeVectorRows(t, filepath.Join(outputDir, "doc-vectors.jsonl"), readTinyVectorRows(t, seed.DocVectorPath)[:1])
	writeVectorRows(t, filepath.Join(outputDir, "query-vectors.jsonl"), readTinyVectorRows(t, seed.QueryVectorPath)[:1])

	var progress []PretrainedBERTRetrievalVectorExportProgress
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
		Resume:         true,
		ProgressEvery:  1,
		Progress: func(p PretrainedBERTRetrievalVectorExportProgress) {
			progress = append(progress, p)
		},
	})
	if err != nil {
		t.Fatalf("resume export: %v", err)
	}
	if !summary.Resume || summary.ReusedDocuments != 1 || summary.ReusedQueries != 1 || summary.WrittenDocuments != 2 || summary.WrittenQueries != 2 {
		t.Fatalf("summary resume counters = %+v", summary)
	}
	if len(progress) == 0 {
		t.Fatalf("expected progress callbacks")
	}
	if got := rowIDs(readTinyVectorRows(t, summary.DocVectorPath)); !slices.Equal(got, []string{"d1", "d2", "d3"}) {
		t.Fatalf("doc ids = %v", got)
	}
	if got := rowIDs(readTinyVectorRows(t, summary.QueryVectorPath)); !slices.Equal(got, []string{"q1", "q2", "q3"}) {
		t.Fatalf("query ids = %v", got)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	data, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if !manifest.Resume || manifest.ReusedDocuments != 1 || manifest.ReusedQueries != 1 || manifest.WrittenDocuments != 2 || manifest.WrittenQueries != 2 {
		t.Fatalf("manifest resume counters = %+v", manifest)
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsMismatchedID(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "doc-vectors.jsonl"), []byte(`{"id":"not-d1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write stale doc vectors: %v", err)
	}

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "want prefix id") {
		t.Fatalf("err = %v, want mismatched id error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsDimensionMismatch(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	stale := `{"id":"d1","embedding":[1,0]}` + "\n" + `{"id":"d2","embedding":[1,0,0]}` + "\n"
	if err := os.WriteFile(filepath.Join(outputDir, "doc-vectors.jsonl"), []byte(stale), 0o644); err != nil {
		t.Fatalf("write stale doc vectors: %v", err)
	}

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "has dimension 3, want 2") {
		t.Fatalf("err = %v, want dimension mismatch error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsCompleteCacheWithWrongModelDimension(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	wrongDimDocs := []retrievalVectorExportRow{
		{ID: "d1", Embedding: []float32{1, 0, 0}},
		{ID: "d2", Embedding: []float32{0, 1, 0}},
	}
	wrongDimQueries := []retrievalVectorExportRow{
		{ID: "q1", Embedding: []float32{1, 0, 0}},
		{ID: "q2", Embedding: []float32{0, 1, 0}},
	}
	writeVectorRows(t, filepath.Join(outputDir, "doc-vectors.jsonl"), wrongDimDocs)
	writeVectorRows(t, filepath.Join(outputDir, "query-vectors.jsonl"), wrongDimQueries)

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "want current model output dimension 2") {
		t.Fatalf("err = %v, want current model dimension error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportWithoutResumeOverwritesStaleFiles(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "doc-vectors.jsonl"), []byte(`{"id":"stale","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write stale doc vectors: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "query-vectors.jsonl"), []byte(`{"id":"stale","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write stale query vectors: %v", err)
	}

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export without resume: %v", err)
	}
	if summary.Resume || summary.ReusedDocuments != 0 || summary.ReusedQueries != 0 || summary.WrittenDocuments != 2 || summary.WrittenQueries != 2 {
		t.Fatalf("summary counters = %+v", summary)
	}
	if got := rowIDs(readTinyVectorRows(t, summary.DocVectorPath)); !slices.Equal(got, []string{"d1", "d2"}) {
		t.Fatalf("doc ids = %v", got)
	}
	if got := rowIDs(readTinyVectorRows(t, summary.QueryVectorPath)); !slices.Equal(got, []string{"q1", "q2"}) {
		t.Fatalf("query ids = %v", got)
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
	return writeTinyPretrainedBERTBEIRFixtureN(t, 1)
}

func writeTinyPretrainedBERTBEIRFixtureN(t *testing.T, n int) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	var corpus, queries, qrels strings.Builder
	qrels.WriteString("query-id\tcorpus-id\tscore\n")
	for i := 1; i <= n; i++ {
		fmt.Fprintf(&corpus, `{"_id":"d%d","title":"", "text":"alpha"}`+"\n", i)
		fmt.Fprintf(&queries, `{"_id":"q%d","text":"alpha"}`+"\n", i)
		fmt.Fprintf(&qrels, "q%d\td%d\t1\n", i, i)
	}
	if err := os.WriteFile(filepath.Join(dir, "corpus.jsonl"), []byte(corpus.String()), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "queries.jsonl"), []byte(queries.String()), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "qrels", "test.tsv"), []byte(qrels.String()), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir
}

func writeVectorRows(t *testing.T, path string, rows []retrievalVectorExportRow) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			t.Fatalf("write row: %v", err)
		}
	}
	if err := writer.Flush(); err != nil {
		t.Fatalf("flush rows: %v", err)
	}
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
