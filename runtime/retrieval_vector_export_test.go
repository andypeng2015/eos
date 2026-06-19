package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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
	if summary.Schema != RetrievalVectorExportManifestSchema || summary.Dataset != "tiny-export" || summary.Documents != 2 || summary.Queries != 1 || summary.ChildVectors != 4 || summary.Dimension != 2 || summary.ModelDimension != 2 || summary.OutputDimension != 2 {
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
		if got := len(row["embedding"].([]any)); got != 2 {
			t.Fatalf("child row %d embedding dim = %d, want 2", i, got)
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
	if len(queryEmbedding) != 2 {
		t.Fatalf("query embedding dim = %d, want 2", len(queryEmbedding))
	}
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
	if manifest.ChildVectors != summary.ChildVectors || manifest.Dimension != summary.Dimension || manifest.ModelDimension != 2 || manifest.OutputDimension != 2 {
		t.Fatalf("manifest = %+v, summary = %+v", manifest, summary)
	}
}

func TestRetrievalVectorExportRejectsInvalidOutputDim(t *testing.T) {
	_, err := ExportEmbeddingRetrievalVectors(context.Background(), loadTinyRetrievalExportModel(t), RetrievalVectorExportConfig{
		CorpusPath:  "corpus.jsonl",
		QueriesPath: "queries.jsonl",
		OutputDir:   t.TempDir(),
		OutputDim:   -1,
		BatchSize:   1,
		MaxDocs:     1,
		MaxQueries:  1,
	})
	if err == nil || !strings.Contains(err.Error(), "output-dim must be non-negative") {
		t.Fatalf("error = %v", err)
	}

	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")
	_, err = ExportEmbeddingRetrievalVectors(context.Background(), loadTinyRetrievalExportModel(t), RetrievalVectorExportConfig{
		CorpusPath:  corpusPath,
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		OutputDir:   filepath.Join(dir, "vectors"),
		OutputDim:   3,
		BatchSize:   1,
	})
	if err == nil || !strings.Contains(err.Error(), "output-dim 3 exceeds encoded vector dimension 2") {
		t.Fatalf("error = %v", err)
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

func TestSparseTokenPoolRetrievalVectorExportWritesPrototypeManifest(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "sparse-vectors")
	manifestPath := filepath.Join(dir, "sparse.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:           "tiny-sparse",
		ArtifactPath:          artifactPath,
		CorpusPath:            corpusPath,
		QueriesPath:           queriesPath,
		QrelsPath:             qrelsPath,
		OutputDir:             outputDir,
		BatchSize:             1,
		MaxDocs:               1,
		MaxQueries:            1,
		OutputDim:             2,
		DocumentChunkWords:    4,
		DocumentChunkOverlap:  1,
		DocumentChunkMinWords: 2,
		ManifestJSONPath:      manifestPath,
		TopK:                  1,
		RouteBlockSize:        1,
		RouteTopBlocks:        1,
		Bits:                  2,
		Seed:                  17,
		MaxTokens:             4,
	})
	if err != nil {
		t.Fatalf("export sparse token pool vectors: %v", err)
	}
	if summary.Schema != SparseTokenPoolRetrievalVectorExportManifestSchema || summary.Method != "experimental_sparse_token_pool" || !summary.Experimental || summary.QualityClaim {
		t.Fatalf("summary prototype metadata = %+v", summary)
	}
	if summary.Documents != 1 || summary.Queries != 1 || summary.Dimension != 2 || summary.ModelDimension != 2 || summary.OutputDimension != 2 {
		t.Fatalf("summary counts/dims = %+v", summary)
	}
	if !summary.AttentionWeightsApplied || !summary.AttentionOutputApplied || !summary.ProjectionApplied || summary.HiddenProjectionApplied || summary.EncoderRepeatsApplied != 0 {
		t.Fatalf("expected attention/projection weights applied: %+v", summary)
	}
	if !summary.DenseKVMaterialized || summary.KVDecode != "host_reference_decode" || summary.Bits != 2 || summary.KeyBits != 2 || summary.ValueBits != 2 || summary.QuantizerSeed != 17 || summary.TopK != 1 {
		t.Fatalf("summary sparse metadata = %+v", summary)
	}
	if summary.DocumentTokenizerOutput.RecordCount != 3 || summary.DocumentTokenizerOutput.RecordCount != summary.ChildVectors || summary.DocumentTokenizerOutput.MaxObservedTokens != 4 || summary.DocumentTokenizerOutput.TotalTokens != 12 || summary.DocumentTokenizerOutput.TruncatedByMaxTokensCount != summary.ChildVectors {
		t.Fatalf("summary document tokenizer-output stats = %+v", summary.DocumentTokenizerOutput)
	}
	if math.Abs(summary.DocumentTokenizerOutput.MeanObservedTokens-4) > 1e-9 {
		t.Fatalf("summary document mean tokens = %.12f, want 4", summary.DocumentTokenizerOutput.MeanObservedTokens)
	}
	if summary.QueryTokenizerOutput.RecordCount != 1 || summary.QueryTokenizerOutput.MaxObservedTokens != 3 || summary.QueryTokenizerOutput.TotalTokens != 3 || summary.QueryTokenizerOutput.TruncatedByMaxTokensCount != 0 {
		t.Fatalf("summary query tokenizer-output stats = %+v", summary.QueryTokenizerOutput)
	}
	if summary.AttentionMode != SparseTokenPoolAttentionModeTurboQuantSparse || !summary.TurboQuantKVApplied {
		t.Fatalf("summary attention metadata = %+v", summary)
	}
	if summary.ChildDocVectorPath != filepath.Join(outputDir, "child-doc-vectors.jsonl") || summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary paths = %+v", summary)
	}

	childRows := readJSONLRows(t, summary.ChildDocVectorPath)
	if len(childRows) == 0 {
		t.Fatal("expected child document vector rows")
	}
	for i, row := range childRows {
		if row["parent_id"] != "d1" {
			t.Fatalf("child row %d parent_id = %v, want qrels-relevant d1", i, row["parent_id"])
		}
		embedding, ok := row["embedding"].([]any)
		if !ok || len(embedding) != 2 {
			t.Fatalf("child row %d embedding = %+v", i, row["embedding"])
		}
	}
	queryRows := readJSONLRows(t, summary.QueryVectorPath)
	if len(queryRows) != 1 || queryRows[0]["id"] != "q1" {
		t.Fatalf("query rows = %+v", queryRows)
	}
	queryEmbedding := queryRows[0]["embedding"].([]any)
	if len(queryEmbedding) != 2 {
		t.Fatalf("query embedding dim = %d, want 2", len(queryEmbedding))
	}
	var norm float64
	for _, value := range queryEmbedding {
		v := value.(float64)
		norm += v * v
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
		t.Fatalf("query embedding norm = %.8f, want normalized", math.Sqrt(norm))
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != summary.Schema || manifest.Method != summary.Method || manifest.QualityClaim || !manifest.AttentionWeightsApplied || manifest.ClaimBoundary == "" || len(manifest.Caveats) == 0 {
		t.Fatalf("manifest = %+v", manifest)
	}
	if manifest.Bits != 2 || manifest.KeyBits != 2 || manifest.ValueBits != 2 {
		t.Fatalf("manifest bits = bits:%d key:%d value:%d, want 2/2/2", manifest.Bits, manifest.KeyBits, manifest.ValueBits)
	}
	if manifest.DocumentTokenizerOutput != summary.DocumentTokenizerOutput || manifest.QueryTokenizerOutput != summary.QueryTokenizerOutput {
		t.Fatalf("manifest tokenizer-output stats = doc:%+v query:%+v, summary doc:%+v query:%+v", manifest.DocumentTokenizerOutput, manifest.QueryTokenizerOutput, summary.DocumentTokenizerOutput, summary.QueryTokenizerOutput)
	}
}

func TestSparseTokenPoolRetrievalVectorExportMinObservedDocTokensGuard(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	_, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:           "tiny-sparse-token-guard",
		ArtifactPath:          artifactPath,
		CorpusPath:            corpusPath,
		QueriesPath:           queriesPath,
		QrelsPath:             qrelsPath,
		OutputDir:             filepath.Join(dir, "sparse-token-guard-vectors"),
		BatchSize:             1,
		MaxDocs:               1,
		MaxQueries:            1,
		DocumentChunkWords:    4,
		DocumentChunkOverlap:  1,
		DocumentChunkMinWords: 2,
		TopK:                  1,
		Bits:                  2,
		Seed:                  17,
		MaxTokens:             4,
		MinObservedDocTokens:  5,
	})
	if err == nil || !strings.Contains(err.Error(), "observed document tokenizer-output max tokens 4 below --min-observed-doc-tokens 5") {
		t.Fatalf("guard error = %v", err)
	}
}

func TestSparseTokenPoolRetrievalVectorExportSparseEncoderManifest(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "sparse-encoder-vectors")
	manifestPath := filepath.Join(dir, "sparse-encoder.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:        "tiny-sparse-encoder",
		ArtifactPath:       artifactPath,
		CorpusPath:         corpusPath,
		QueriesPath:        queriesPath,
		QrelsPath:          qrelsPath,
		OutputDir:          outputDir,
		BatchSize:          1,
		MaxDocs:            1,
		MaxQueries:         1,
		OutputDim:          2,
		ManifestJSONPath:   manifestPath,
		TopK:               1,
		RouteBlockSize:     1,
		RouteTopBlocks:     1,
		Bits:               4,
		KeyBits:            4,
		ValueBits:          8,
		Seed:               99,
		MaxTokens:          4,
		RequireFullEncoder: true,
		Method:             "experimental_sparse_encoder_host_reference",
		EvidenceLevel:      "retrieval_cache_host_reference_sparse_encoder",
		ClaimBoundary:      "Prototype host-reference sparse encoder retrieval-cache evidence only; not a trained sparse/LongEmbed encoder, not sealed runtime inference, and not production quality evidence.",
	})
	if err != nil {
		t.Fatalf("export sparse encoder vectors: %v", err)
	}
	if summary.Method != "experimental_sparse_encoder_host_reference" || summary.EvidenceLevel != "retrieval_cache_host_reference_sparse_encoder" || summary.QualityClaim {
		t.Fatalf("summary identity = %+v", summary)
	}
	if !summary.RequireFullEncoder || !summary.FullEncoderApplied || !summary.AttentionWeightsApplied || !summary.AttentionOutputApplied || !summary.HiddenProjectionApplied || !summary.ProjectionApplied {
		t.Fatalf("summary full encoder metadata = %+v", summary)
	}
	if summary.ChildVectors != 0 || summary.ChildDocVectorPath != "" || summary.DocVectorPath != filepath.Join(outputDir, "doc-vectors.jsonl") || summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary parent-vector paths = %+v", summary)
	}
	if !summary.DenseKVMaterialized || summary.KVDecode != "host_reference_decode" || summary.TopK != 1 || summary.SparseTopK != 1 || summary.Bits != 4 || summary.KeyBits != 4 || summary.ValueBits != 8 || summary.QuantizerSeed != 99 {
		t.Fatalf("summary sparse audit metadata = %+v", summary)
	}
	if summary.TopKConfigured != 1 || summary.SparseTopKConfigured != 1 || summary.SparseTopKEffectiveMaxObservedDoc != 1 {
		t.Fatalf("summary sparse top-k provenance = %+v", summary)
	}
	if summary.MaxObservedDocPlan == nil || summary.MaxObservedDocPlan.KeyLen != summary.DocumentTokenizerOutput.MaxObservedTokens || summary.MaxObservedDocPlan.CandidateKeyBudget <= 0 || summary.MaxObservedDocPlan.ScoreCountFraction <= 0 {
		t.Fatalf("summary sparse plan = %+v for token stats %+v", summary.MaxObservedDocPlan, summary.DocumentTokenizerOutput)
	}
	if summary.CandidateKeyBudgetMaxObservedDoc != summary.MaxObservedDocPlan.CandidateKeyBudget || summary.ScoreFractionMaxObservedDoc != summary.MaxObservedDocPlan.ScoreCountFraction {
		t.Fatalf("summary sparse plan top-level fields = %+v plan=%+v", summary, summary.MaxObservedDocPlan)
	}
	if !strings.Contains(summary.ClaimBoundary, "not a trained sparse/LongEmbed encoder") {
		t.Fatalf("claim boundary = %q", summary.ClaimBoundary)
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Method != summary.Method || manifest.EvidenceLevel != summary.EvidenceLevel || !manifest.RequireFullEncoder || !manifest.FullEncoderApplied || manifest.MaxObservedDocPlan == nil {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestSparseTokenPoolRetrievalVectorExportSparseEncoderDefaultTopKProvenance(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "sparse-encoder-default-topk-vectors")
	manifestPath := filepath.Join(dir, "sparse-encoder-default-topk.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:        "tiny-sparse-encoder-default-topk",
		ArtifactPath:       artifactPath,
		CorpusPath:         corpusPath,
		QueriesPath:        queriesPath,
		QrelsPath:          qrelsPath,
		OutputDir:          outputDir,
		BatchSize:          1,
		MaxDocs:            1,
		MaxQueries:         1,
		OutputDim:          2,
		ManifestJSONPath:   manifestPath,
		RouteBlockSize:     1,
		RouteTopBlocks:     1,
		Bits:               4,
		Seed:               101,
		MaxTokens:          4,
		RequireFullEncoder: true,
		Method:             "experimental_sparse_encoder_host_reference",
		EvidenceLevel:      "retrieval_cache_host_reference_sparse_encoder",
		ClaimBoundary:      "Prototype host-reference sparse encoder retrieval-cache evidence only; not a trained sparse/LongEmbed encoder, not sealed runtime inference, and not production quality evidence.",
	})
	if err != nil {
		t.Fatalf("export sparse encoder vectors: %v", err)
	}
	if summary.MaxObservedDocPlan == nil {
		t.Fatalf("summary missing sparse plan: %+v", summary)
	}
	wantTopK := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen:       summary.DocumentTokenizerOutput.MaxObservedTokens,
		KeyLen:         summary.DocumentTokenizerOutput.MaxObservedTokens,
		QueryDim:       summary.ModelDimension,
		ValueDim:       summary.ModelDimension,
		RouteBlockSize: summary.RouteBlockSize,
		RouteTopBlocks: summary.RouteTopBlocks,
	}).TopK
	if wantTopK <= 0 {
		t.Fatalf("want effective top-k = %d", wantTopK)
	}
	if summary.TopKConfigured != 0 || summary.SparseTopKConfigured != 0 {
		t.Fatalf("summary configured top-k provenance = %+v", summary)
	}
	if summary.TopK != wantTopK || summary.SparseTopK != wantTopK || summary.SparseTopKEffectiveMaxObservedDoc != wantTopK || summary.MaxObservedDocPlan.TopK != wantTopK {
		t.Fatalf("summary effective top-k provenance = %+v want %d", summary, wantTopK)
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.TopKConfigured != 0 || manifest.SparseTopKConfigured != 0 || manifest.TopK != wantTopK || manifest.SparseTopK != wantTopK || manifest.SparseTopKEffectiveMaxObservedDoc != wantTopK {
		t.Fatalf("manifest sparse top-k provenance = %+v want %d", manifest, wantTopK)
	}
}

func TestSparseTokenPoolRetrievalVectorExportRequireFullEncoderRejectsFallback(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	_, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:        "tiny-sparse-encoder-requires-full",
		ArtifactPath:       artifactPath,
		CorpusPath:         corpusPath,
		QueriesPath:        queriesPath,
		QrelsPath:          qrelsPath,
		OutputDir:          filepath.Join(dir, "sparse-encoder-requires-full"),
		TopK:               1,
		Bits:               2,
		Seed:               17,
		RequireFullEncoder: true,
		Method:             "experimental_sparse_encoder_host_reference",
	})
	if err == nil || !strings.Contains(err.Error(), "requires full encoder weights") {
		t.Fatalf("error = %v, want full encoder failure", err)
	}
}

func TestSparseTokenPoolRetrievalVectorExportAppliesManifestEncoderFFN(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "sparse-ffn-vectors")
	manifestPath := filepath.Join(dir, "sparse-ffn.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:      "tiny-sparse-ffn",
		ArtifactPath:     artifactPath,
		CorpusPath:       corpusPath,
		QueriesPath:      queriesPath,
		QrelsPath:        qrelsPath,
		OutputDir:        outputDir,
		BatchSize:        1,
		MaxDocs:          1,
		MaxQueries:       1,
		OutputDim:        2,
		ManifestJSONPath: manifestPath,
		TopK:             2,
		RouteBlockSize:   2,
		RouteTopBlocks:   1,
		Bits:             4,
		Seed:             19,
		MaxTokens:        4,
	})
	if err != nil {
		t.Fatalf("export sparse token pool FFN vectors: %v", err)
	}
	if !summary.AttentionWeightsApplied || !summary.AttentionOutputApplied || !summary.HiddenProjectionApplied || !summary.ProjectionApplied {
		t.Fatalf("expected full encoder weights applied: %+v", summary)
	}
	if summary.EncoderRepeatsApplied != 2 || !summary.AttentionResidual || !summary.AttentionLayerNorm || !summary.FFNResidual || !summary.FFNLayerNorm {
		t.Fatalf("expected manifest encoder structure applied: %+v", summary)
	}
	if summary.HiddenProjectionParam != "ffn_up" || summary.ProjectionParam != "projection" || len(summary.SkippedWeights) != 0 {
		t.Fatalf("unexpected params/skips: %+v", summary)
	}
	rows := readJSONLRows(t, summary.DocVectorPath)
	if len(rows) != 1 {
		t.Fatalf("doc rows = %d, want 1", len(rows))
	}
	embedding, ok := rows[0]["embedding"].([]any)
	if !ok || len(embedding) != 2 {
		t.Fatalf("doc embedding = %+v", rows[0]["embedding"])
	}
	var norm float64
	for _, value := range embedding {
		v := value.(float64)
		norm += v * v
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
		t.Fatalf("doc embedding norm = %.8f, want normalized", math.Sqrt(norm))
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if !manifest.HiddenProjectionApplied || manifest.EncoderRepeatsApplied != 2 || manifest.HiddenProjectionParam != "ffn_up" {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestSparseTokenPoolRetrievalVectorExportTokenSpanChildren(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "sparse-token-span-vectors")
	manifestPath := filepath.Join(dir, "sparse-token-span.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:        "tiny-sparse-token-span",
		ArtifactPath:       artifactPath,
		CorpusPath:         corpusPath,
		QueriesPath:        queriesPath,
		QrelsPath:          qrelsPath,
		OutputDir:          outputDir,
		BatchSize:          1,
		MaxDocs:            1,
		MaxQueries:         1,
		OutputDim:          2,
		ManifestJSONPath:   manifestPath,
		TopK:               2,
		Bits:               4,
		Seed:               31,
		MaxTokens:          5,
		TokenSpanTokens:    2,
		TokenSpanOverlap:   1,
		TokenSpanMinTokens: 1,
	})
	if err != nil {
		t.Fatalf("export sparse token span vectors: %v", err)
	}
	if summary.DocVectorPath != "" || summary.ChildDocVectorPath != filepath.Join(outputDir, "child-doc-vectors.jsonl") {
		t.Fatalf("summary document paths = %+v", summary)
	}
	if summary.ChildVectors != 4 || summary.DocumentTokenizerOutput.RecordCount != 1 || summary.DocumentTokenizerOutput.MaxObservedTokens != 5 || summary.DocumentTokenizerOutput.TotalTokens != 5 {
		t.Fatalf("summary span counts/stats = %+v", summary)
	}
	if summary.TokenSpanTokens != 2 || summary.TokenSpanOverlap != 1 || summary.TokenSpanMinTokens != 1 {
		t.Fatalf("summary token span settings = %+v", summary)
	}
	if !summary.HiddenProjectionApplied || summary.EncoderRepeatsApplied != 2 {
		t.Fatalf("expected full encoder applied: %+v", summary)
	}
	rows := readJSONLRows(t, summary.ChildDocVectorPath)
	if len(rows) != 4 {
		t.Fatalf("child rows = %d, want 4", len(rows))
	}
	for i, row := range rows {
		wantChildID := fmt.Sprintf("d1#token-span-%04d", i)
		if row["parent_id"] != "d1" || row["child_id"] != wantChildID {
			t.Fatalf("child row %d ids = %+v, want parent d1 child %s", i, row, wantChildID)
		}
		embedding, ok := row["embedding"].([]any)
		if !ok || len(embedding) != 2 {
			t.Fatalf("child row %d embedding = %+v", i, row["embedding"])
		}
		var norm float64
		for _, value := range embedding {
			v := value.(float64)
			norm += v * v
		}
		if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
			t.Fatalf("child row %d norm = %.8f, want normalized", i, math.Sqrt(norm))
		}
	}
	queryRows := readJSONLRows(t, summary.QueryVectorPath)
	if len(queryRows) != 1 || queryRows[0]["id"] != "q1" {
		t.Fatalf("query rows = %+v", queryRows)
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.TokenSpanTokens != 2 || manifest.TokenSpanOverlap != 1 || manifest.TokenSpanMinTokens != 1 || manifest.DocumentTokenizerOutput != summary.DocumentTokenizerOutput {
		t.Fatalf("manifest token span fields/stats = %+v", manifest)
	}
}

func TestSparseTokenPoolRetrievalVectorExportTokenSpanRequiresFullEncoder(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	_, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:      "tiny-sparse-token-span-no-ffn",
		ArtifactPath:     artifactPath,
		CorpusPath:       corpusPath,
		QueriesPath:      queriesPath,
		QrelsPath:        qrelsPath,
		OutputDir:        filepath.Join(dir, "sparse-token-span-no-ffn"),
		TokenSpanTokens:  2,
		TokenSpanOverlap: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "token-span child vectors require full manifest encoder weights") {
		t.Fatalf("error = %v", err)
	}
}

func TestSparseTokenPoolRetrievalVectorExportRejectsTokenSpanChunkMix(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	_, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:        "tiny-sparse-token-span-invalid",
		ArtifactPath:       artifactPath,
		CorpusPath:         corpusPath,
		QueriesPath:        queriesPath,
		QrelsPath:          qrelsPath,
		OutputDir:          filepath.Join(dir, "sparse-token-span-invalid"),
		DocumentChunkWords: 4,
		TokenSpanTokens:    2,
	})
	if err == nil || !strings.Contains(err.Error(), "token-span-tokens is mutually exclusive with document-chunk-words") {
		t.Fatalf("error = %v", err)
	}
}

func TestSparseTokenPoolRetrievalVectorExportDenseAttentionMode(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "dense-attention-vectors")
	manifestPath := filepath.Join(dir, "dense-attention.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:      "tiny-sparse-dense-attention",
		ArtifactPath:     artifactPath,
		CorpusPath:       corpusPath,
		QueriesPath:      queriesPath,
		QrelsPath:        qrelsPath,
		OutputDir:        outputDir,
		BatchSize:        1,
		MaxDocs:          1,
		MaxQueries:       1,
		OutputDim:        2,
		ManifestJSONPath: manifestPath,
		TopK:             2,
		Bits:             4,
		Seed:             23,
		MaxTokens:        4,
		AttentionMode:    SparseTokenPoolAttentionModeDense,
	})
	if err != nil {
		t.Fatalf("export sparse token pool dense attention vectors: %v", err)
	}
	if summary.AttentionMode != SparseTokenPoolAttentionModeDense || summary.TurboQuantKVApplied || !summary.DenseKVMaterialized || summary.KVDecode != "not_applicable_dense_attention" {
		t.Fatalf("summary dense attention metadata = %+v", summary)
	}
	if !summary.AttentionWeightsApplied || !summary.AttentionOutputApplied || !summary.HiddenProjectionApplied || !summary.ProjectionApplied || summary.EncoderRepeatsApplied != 2 || len(summary.SkippedWeights) != 0 {
		t.Fatalf("expected full encoder with no skipped weights: %+v", summary)
	}
	rows := readJSONLRows(t, summary.DocVectorPath)
	if len(rows) != 1 {
		t.Fatalf("doc rows = %d, want 1", len(rows))
	}
	embedding, ok := rows[0]["embedding"].([]any)
	if !ok || len(embedding) != 2 {
		t.Fatalf("doc embedding = %+v", rows[0]["embedding"])
	}
	var norm float64
	for _, value := range embedding {
		v := value.(float64)
		norm += v * v
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
		t.Fatalf("doc embedding norm = %.8f, want normalized", math.Sqrt(norm))
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.AttentionMode != SparseTokenPoolAttentionModeDense || manifest.TurboQuantKVApplied || manifest.KVDecode != "not_applicable_dense_attention" || manifest.EncoderRepeatsApplied != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestSparseTokenPoolRetrievalVectorExportMixedKeyValueBits(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolFFNExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	outputDir := filepath.Join(dir, "mixed-kv-vectors")
	manifestPath := filepath.Join(dir, "mixed-kv.manifest.json")
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	summary, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:      "tiny-sparse-mixed-kv",
		ArtifactPath:     artifactPath,
		CorpusPath:       corpusPath,
		QueriesPath:      queriesPath,
		QrelsPath:        qrelsPath,
		OutputDir:        outputDir,
		BatchSize:        1,
		MaxDocs:          1,
		MaxQueries:       1,
		OutputDim:        2,
		ManifestJSONPath: manifestPath,
		TopK:             2,
		Bits:             4,
		KeyBits:          4,
		ValueBits:        8,
		Seed:             29,
		MaxTokens:        4,
	})
	if err != nil {
		t.Fatalf("export sparse token pool mixed K/V vectors: %v", err)
	}
	if summary.Bits != 4 || summary.KeyBits != 4 || summary.ValueBits != 8 || summary.AttentionMode != SparseTokenPoolAttentionModeTurboQuantSparse || !summary.TurboQuantKVApplied {
		t.Fatalf("summary mixed K/V metadata = %+v", summary)
	}
	rows := readJSONLRows(t, summary.DocVectorPath)
	if len(rows) != 1 {
		t.Fatalf("doc rows = %d, want 1", len(rows))
	}
	embedding, ok := rows[0]["embedding"].([]any)
	if !ok || len(embedding) != 2 {
		t.Fatalf("doc embedding = %+v", rows[0]["embedding"])
	}

	var manifest SparseTokenPoolRetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Bits != 4 || manifest.KeyBits != 4 || manifest.ValueBits != 8 || !manifest.TurboQuantKVApplied || manifest.KVDecode != "host_reference_decode" {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestSparseTokenPoolRetrievalVectorExportInvalidAttentionMode(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")

	_, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:   "tiny-invalid-attention",
		ArtifactPath:  artifactPath,
		CorpusPath:    corpusPath,
		QueriesPath:   queriesPath,
		QrelsPath:     qrelsPath,
		OutputDir:     filepath.Join(dir, "invalid-attention"),
		AttentionMode: "approx_dense",
	})
	if err == nil || !strings.Contains(err.Error(), "attention-mode must be") {
		t.Fatalf("error = %v", err)
	}
}

func TestSparseTokenPoolRetrievalVectorExportInvalidKeyValueBits(t *testing.T) {
	model, artifactPath := loadTinySparseTokenPoolExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")
	base := SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:  "tiny-invalid-kv-bits",
		ArtifactPath: artifactPath,
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    qrelsPath,
		OutputDir:    filepath.Join(dir, "invalid-kv-bits"),
	}

	keyCfg := base
	keyCfg.KeyBits = 3
	_, err := ExportSparseTokenPoolRetrievalVectors(context.Background(), model, keyCfg)
	if err == nil || !strings.Contains(err.Error(), "key-bits must be 0, 2, 4, or 8") {
		t.Fatalf("key-bits error = %v", err)
	}

	valueCfg := base
	valueCfg.ValueBits = 6
	_, err = ExportSparseTokenPoolRetrievalVectors(context.Background(), model, valueCfg)
	if err == nil || !strings.Contains(err.Error(), "value-bits must be 0, 2, 4, or 8") {
		t.Fatalf("value-bits error = %v", err)
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

func loadTinySparseTokenPoolExportModel(t *testing.T) (*EmbeddingModel, string) {
	t.Helper()
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param attn_q: f16[D, D] @weight("weights/attn_q")
param attn_k: f16[D, D] @weight("weights/attn_k")
param attn_v: f16[D, D] @weight("weights/attn_v")
param attn_o: f16[D, D] @weight("weights/attn_o")
param projection: f16[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden = gather(token_embedding, tokens)
    let q = @matmul(hidden, attn_q)
    let k = @matmul(hidden, attn_k)
    let v = @matmul(hidden, attn_v)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, attn_o)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden = gather(token_embedding, tokens)
    let q = @matmul(hidden, attn_q)
    let k = @matmul(hidden, attn_k)
    let v = @matmul(hidden, attn_v)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, attn_o)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_sparse_token_pool_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_sparse_token_pool_embed.mll")
	if err := eosartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                 "tiny_sparse_token_pool_embed",
		PooledEntry:          "embed_pooled",
		BatchEntry:           "embed_pooled_batch",
		TokenInput:           "tokens",
		MaskInput:            "attention_mask",
		OutputName:           "result",
		OutputDType:          "f16",
		TokenEmbeddingParam:  "token_embedding",
		AttentionQueryParam:  "attn_q",
		AttentionKeyParam:    "attn_k",
		AttentionValueParam:  "attn_v",
		AttentionOutputParam: "attn_o",
		ProjectionParam:      "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   5,
			MaxSequence: 8,
			PadID:       0,
			UnknownID:   1,
		},
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	identity := backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1})
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{5, 2}, []float32{
			0, 0,
			1, 0,
			0, 1,
			1, 1,
			1, -1,
		}),
		"attn_q":     identity,
		"attn_k":     identity,
		"attn_v":     identity,
		"attn_o":     identity,
		"projection": identity,
	})
	if err := weights.WriteFile(DefaultWeightFilePath(artifactPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	tokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "one", "alpha", "beta"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load package: %v", err)
	}
	return model, artifactPath
}

func loadTinySparseTokenPoolFFNExportModel(t *testing.T) (*EmbeddingModel, string) {
	t.Helper()
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param attn_q: f16[D, D] @weight("weights/attn_q")
param attn_k: f16[D, D] @weight("weights/attn_k")
param attn_v: f16[D, D] @weight("weights/attn_v")
param attn_o: f16[D, D] @weight("weights/attn_o")
param ffn_up: f16[D, H] @weight("weights/ffn_up")
param projection: f16[H, D] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[D] {
    let hidden = gather(token_embedding, tokens)
    let q1 = @matmul(hidden, attn_q)
    let k1 = @matmul(hidden, attn_k)
    let v1 = @matmul(hidden, attn_v)
    let kt1 = transpose(k1)
    let scores1 = @matmul(q1, kt1)
    let probs1 = softmax(scores1)
    let mixed1 = @matmul(probs1, v1)
    let attended1 = @matmul(mixed1, attn_o)
    let attn_hidden1 = layernorm(attended1 + hidden)
    let ffn_hidden1 = @matmul(attn_hidden1, ffn_up)
    let activated1 = gelu(ffn_hidden1)
    let projected1 = @matmul(activated1, projection)
    let encoded1 = layernorm(projected1 + attn_hidden1)
    let q2 = @matmul(encoded1, attn_q)
    let k2 = @matmul(encoded1, attn_k)
    let v2 = @matmul(encoded1, attn_v)
    let kt2 = transpose(k2)
    let scores2 = @matmul(q2, kt2)
    let probs2 = softmax(scores2)
    let mixed2 = @matmul(probs2, v2)
    let attended2 = @matmul(mixed2, attn_o)
    let attn_hidden2 = layernorm(attended2 + encoded1)
    let ffn_hidden2 = @matmul(attn_hidden2, ffn_up)
    let activated2 = gelu(ffn_hidden2)
    let projected2 = @matmul(activated2, projection)
    let encoded2 = layernorm(projected2 + attn_hidden2)
    let normalized = normalize(encoded2)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, D] {
    let hidden = gather(token_embedding, tokens)
    let q1 = @matmul(hidden, attn_q)
    let k1 = @matmul(hidden, attn_k)
    let v1 = @matmul(hidden, attn_v)
    let kt1 = transpose(k1)
    let scores1 = @matmul(q1, kt1)
    let probs1 = softmax(scores1)
    let mixed1 = @matmul(probs1, v1)
    let attended1 = @matmul(mixed1, attn_o)
    let attn_hidden1 = layernorm(attended1 + hidden)
    let ffn_hidden1 = @matmul(attn_hidden1, ffn_up)
    let activated1 = gelu(ffn_hidden1)
    let projected1 = @matmul(activated1, projection)
    let encoded1 = layernorm(projected1 + attn_hidden1)
    let q2 = @matmul(encoded1, attn_q)
    let k2 = @matmul(encoded1, attn_k)
    let v2 = @matmul(encoded1, attn_v)
    let kt2 = transpose(k2)
    let scores2 = @matmul(q2, kt2)
    let probs2 = softmax(scores2)
    let mixed2 = @matmul(probs2, v2)
    let attended2 = @matmul(mixed2, attn_o)
    let attn_hidden2 = layernorm(attended2 + encoded1)
    let ffn_hidden2 = @matmul(attn_hidden2, ffn_up)
    let activated2 = gelu(ffn_hidden2)
    let projected2 = @matmul(activated2, projection)
    let encoded2 = layernorm(projected2 + attn_hidden2)
    let normalized = normalize(encoded2)
    return mean_pool(normalized, attention_mask)
}
`)
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_sparse_token_pool_ffn_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_sparse_token_pool_ffn_embed.mll")
	if err := eosartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                  "tiny_sparse_token_pool_ffn_embed",
		PooledEntry:           "embed_pooled",
		BatchEntry:            "embed_pooled_batch",
		EncoderRepeats:        2,
		TokenInput:            "tokens",
		MaskInput:             "attention_mask",
		OutputName:            "result",
		OutputDType:           "f16",
		TokenEmbeddingParam:   "token_embedding",
		AttentionQueryParam:   "attn_q",
		AttentionKeyParam:     "attn_k",
		AttentionValueParam:   "attn_v",
		AttentionOutputParam:  "attn_o",
		AttentionResidual:     true,
		AttentionLayerNorm:    true,
		HiddenProjectionParam: "ffn_up",
		FFNResidual:           true,
		FFNLayerNorm:          true,
		ProjectionParam:       "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   5,
			MaxSequence: 8,
			PadID:       0,
			UnknownID:   1,
		},
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	identity := backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1})
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{5, 2}, []float32{
			0, 0,
			1, 0,
			0, 1,
			1, 1,
			1, -1,
		}),
		"attn_q": identity,
		"attn_k": identity,
		"attn_v": identity,
		"attn_o": identity,
		"ffn_up": backend.NewTensorF16([]int{2, 4}, []float32{
			1, 0, 0.5, -0.5,
			0, 1, -0.25, 0.75,
		}),
		"projection": backend.NewTensorF16([]int{4, 2}, []float32{
			1, 0,
			0, 1,
			0.5, 0.25,
			-0.25, 0.5,
		}),
	})
	if err := weights.WriteFile(DefaultWeightFilePath(artifactPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	tokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "one", "alpha", "beta"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load package: %v", err)
	}
	return model, artifactPath
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
