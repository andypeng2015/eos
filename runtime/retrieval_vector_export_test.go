package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestRetrievalVectorExportWritesChildCachesAndManifest(t *testing.T) {
	model := loadTinyRetrievalExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "vectors")
	manifestPath := filepath.Join(dir, "export.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportEmbeddingRetrievalVectors(context.Background(), model, RetrievalVectorExportConfig{
		DatasetName:           "tiny-export",
		ArtifactPath:          "tiny.mll",
		CorpusPath:            corpusPath,
		QueriesPath:           queriesPath,
		QrelsPath:             qrelsPath,
		OutputDir:             outputDir,
		BatchSize:             1,
		DocumentChunkWords:    4,
		DocumentChunkOverlap:  1,
		DocumentChunkMinWords: 2,
		ManifestJSONPath:      manifestPath,
	})
	if err != nil {
		t.Fatalf("export vectors: %v", err)
	}
	if summary.Schema != RetrievalVectorExportManifestSchema || summary.Dataset != "tiny-export" || summary.Documents != 2 || summary.Queries != 1 || summary.ChildVectors != 4 || summary.Dimension != 2 {
		t.Fatalf("summary = %+v", summary)
	}
	if summary.ChildDocVectorPath != filepath.Join(outputDir, "child-doc-vectors.jsonl") || summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary paths = %+v", summary)
	}

	childRows := readJSONLRows(t, summary.ChildDocVectorPath)
	if len(childRows) != 4 {
		t.Fatalf("child row count = %d, want 4", len(childRows))
	}
	wantChildIDs := []string{"d1#chunk-0000", "d1#chunk-0001", "d1#chunk-0002", "d2#chunk-0000"}
	for i, row := range childRows {
		if row["child_id"] != wantChildIDs[i] {
			t.Fatalf("child row %d id = %v, want %s", i, row["child_id"], wantChildIDs[i])
		}
		if _, ok := row["embedding"].([]any); !ok {
			t.Fatalf("child row %d missing embedding array: %+v", i, row)
		}
	}
	if childRows[0]["parent_id"] != "d1" || childRows[3]["parent_id"] != "d2" {
		t.Fatalf("parent ids = %v / %v", childRows[0]["parent_id"], childRows[3]["parent_id"])
	}

	queryRows := readJSONLRows(t, summary.QueryVectorPath)
	if len(queryRows) != 1 || queryRows[0]["id"] != "q1" {
		t.Fatalf("query rows = %+v", queryRows)
	}
	queryEmbedding := queryRows[0]["embedding"].([]any)
	var norm float64
	for _, value := range queryEmbedding {
		v := value.(float64)
		norm += v * v
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
		t.Fatalf("query embedding norm = %.8f, want normalized", math.Sqrt(norm))
	}

	var manifest RetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.ChildVectors != summary.ChildVectors || manifest.Dimension != summary.Dimension {
		t.Fatalf("manifest = %+v, summary = %+v", manifest, summary)
	}
}

func TestRetrievalVectorExportRejectsInvalidChunkOverlap(t *testing.T) {
	_, err := ExportEmbeddingRetrievalVectors(context.Background(), loadTinyRetrievalExportModel(t), RetrievalVectorExportConfig{
		CorpusPath:            "corpus.jsonl",
		QueriesPath:           "queries.jsonl",
		OutputDir:             t.TempDir(),
		DocumentChunkWords:    8,
		DocumentChunkOverlap:  8,
		DocumentChunkMinWords: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "document-chunk-overlap must be smaller") {
		t.Fatalf("error = %v", err)
	}
}

func loadTinyRetrievalExportModel(t *testing.T) *EmbeddingModel {
	t.Helper()
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	if err := eosartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	})
	if err := weights.WriteFile(DefaultWeightFilePath(artifactPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	if err := tinyEmbeddingTokenizerFile().WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load package: %v", err)
	}
	return model
}

func writeTinyRetrievalExportDataset(t *testing.T, dir string) string {
	t.Helper()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","title":"one two","text":"three four five six seven eight nine ten"}`+"\n"+
			`{"_id":"d2","text":"alpha beta"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"one"}`+"\n"+
			`{"_id":"q2","text":"not selected"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return datasetDir
}

func readJSONLRows(t *testing.T, path string) []map[string]any {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open JSONL %q: %v", path, err)
	}
	defer file.Close()
	var rows []map[string]any
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var row map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatalf("decode row %q: %v", scanner.Text(), err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan JSONL %q: %v", path, err)
	}
	return rows
}
