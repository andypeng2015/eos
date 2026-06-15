package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const RetrievalVectorExportManifestSchema = "manta.embedding_retrieval_vector_export.v1"

// RetrievalVectorExportConfig describes a BEIR vector-cache export from an
// Eos embedding model.
type RetrievalVectorExportConfig struct {
	DatasetName           string
	ArtifactPath          string
	CorpusPath            string
	QueriesPath           string
	QrelsPath             string
	OutputDir             string
	BatchSize             int
	MaxDocs               int
	MaxQueries            int
	OutputDim             int
	DocumentChunkWords    int
	DocumentChunkOverlap  int
	DocumentChunkMinWords int
	DocumentPrefix        string
	QueryPrefix           string
	ManifestJSONPath      string
}

// RetrievalVectorExportSummary is a compact manifest for generated vector caches.
type RetrievalVectorExportSummary struct {
	Schema                string    `json:"schema"`
	Dataset               string    `json:"dataset"`
	Artifact              string    `json:"artifact,omitempty"`
	Backend               string    `json:"backend,omitempty"`
	Documents             int       `json:"documents"`
	Queries               int       `json:"queries"`
	ChildVectors          int       `json:"child_vectors,omitempty"`
	Dimension             int       `json:"dimension"`
	ModelDimension        int       `json:"model_dimension,omitempty"`
	OutputDimension       int       `json:"output_dimension,omitempty"`
	DocVectorPath         string    `json:"doc_vector_path,omitempty"`
	ChildDocVectorPath    string    `json:"child_doc_vector_path,omitempty"`
	QueryVectorPath       string    `json:"query_vector_path"`
	DocumentChunkWords    int       `json:"document_chunk_words,omitempty"`
	DocumentChunkOverlap  int       `json:"document_chunk_overlap,omitempty"`
	DocumentChunkMinWords int       `json:"document_chunk_min_words,omitempty"`
	BatchSize             int       `json:"batch_size"`
	MaxDocs               int       `json:"max_docs,omitempty"`
	MaxQueries            int       `json:"max_queries,omitempty"`
	CorpusPath            string    `json:"corpus_path,omitempty"`
	QueriesPath           string    `json:"queries_path,omitempty"`
	QrelsPath             string    `json:"qrels_path,omitempty"`
	ElapsedSeconds        float64   `json:"elapsed_seconds"`
	CreatedAt             time.Time `json:"created_at"`
}

type retrievalDocumentChunk struct {
	ParentID string
	ChildID  string
	Text     string
}

type retrievalVectorExportRow struct {
	ID        string    `json:"id,omitempty"`
	ParentID  string    `json:"parent_id,omitempty"`
	ChildID   string    `json:"child_id,omitempty"`
	Embedding []float32 `json:"embedding"`
}

// ExportEmbeddingRetrievalVectors exports BEIR-compatible document/query vector
// cache JSONL files using the same tokenization, batching, and L2 normalization
// path as EvaluateEmbeddingRetrieval.
func ExportEmbeddingRetrievalVectors(ctx context.Context, model *EmbeddingModel, cfg RetrievalVectorExportConfig) (RetrievalVectorExportSummary, error) {
	if model == nil {
		return RetrievalVectorExportSummary{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeRetrievalVectorExportConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.OutputDir == "" {
		return RetrievalVectorExportSummary{}, fmt.Errorf("corpus path, queries path, and output dir are required")
	}
	if err := validateRetrievalVectorChunkConfig(cfg); err != nil {
		return RetrievalVectorExportSummary{}, err
	}

	start := time.Now()
	var qrels retrievalQrels
	var err error
	if cfg.QrelsPath != "" {
		qrels, err = readBEIRQrels(cfg.QrelsPath)
		if err != nil {
			return RetrievalVectorExportSummary{}, err
		}
	}
	corpus, err := readRetrievalExportCorpus(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return RetrievalVectorExportSummary{}, err
	}
	queries, _, err := readRetrievalExportQueries(cfg.QueriesPath, cfg.MaxQueries, qrels)
	if err != nil {
		return RetrievalVectorExportSummary{}, err
	}
	if len(corpus) == 0 {
		return RetrievalVectorExportSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return RetrievalVectorExportSummary{}, fmt.Errorf("queries are empty")
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return RetrievalVectorExportSummary{}, err
	}

	queryVectorPath := filepath.Join(cfg.OutputDir, "query-vectors.jsonl")
	var docVectorPath, childDocVectorPath string
	var dim, modelDim, childCount int
	if cfg.DocumentChunkWords > 0 {
		chunks := chunkRetrievalDocuments(corpus, cfg.DocumentChunkWords, cfg.DocumentChunkOverlap, cfg.DocumentChunkMinWords)
		if len(chunks) == 0 {
			return RetrievalVectorExportSummary{}, fmt.Errorf("document chunking selected no chunks")
		}
		childDocVectorPath = filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
		dim, modelDim, err = writeRetrievalChildVectorCache(ctx, model, chunks, childDocVectorPath, cfg.BatchSize, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return RetrievalVectorExportSummary{}, fmt.Errorf("write child document vectors: %w", err)
		}
		childCount = len(chunks)
	} else {
		docVectorPath = filepath.Join(cfg.OutputDir, "doc-vectors.jsonl")
		dim, modelDim, err = writeRetrievalVectorCache(ctx, model, corpus, docVectorPath, cfg.BatchSize, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return RetrievalVectorExportSummary{}, fmt.Errorf("write document vectors: %w", err)
		}
	}
	queryDim, queryModelDim, err := writeRetrievalVectorCache(ctx, model, queries, queryVectorPath, cfg.BatchSize, cfg.QueryPrefix, cfg.OutputDim)
	if err != nil {
		return RetrievalVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if dim != queryDim {
		return RetrievalVectorExportSummary{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", dim, queryDim)
	}
	if modelDim != queryModelDim {
		return RetrievalVectorExportSummary{}, fmt.Errorf("document vectors have encoded dimension %d but query vectors have encoded dimension %d", modelDim, queryModelDim)
	}

	summary := RetrievalVectorExportSummary{
		Schema:                RetrievalVectorExportManifestSchema,
		Dataset:               cfg.DatasetName,
		Artifact:              cfg.ArtifactPath,
		Backend:               string(model.Backend()),
		Documents:             len(corpus),
		Queries:               len(queries),
		ChildVectors:          childCount,
		Dimension:             dim,
		ModelDimension:        modelDim,
		OutputDimension:       dim,
		DocVectorPath:         docVectorPath,
		ChildDocVectorPath:    childDocVectorPath,
		QueryVectorPath:       queryVectorPath,
		DocumentChunkWords:    cfg.DocumentChunkWords,
		DocumentChunkOverlap:  cfg.DocumentChunkOverlap,
		DocumentChunkMinWords: cfg.DocumentChunkMinWords,
		BatchSize:             cfg.BatchSize,
		MaxDocs:               cfg.MaxDocs,
		MaxQueries:            cfg.MaxQueries,
		CorpusPath:            cfg.CorpusPath,
		QueriesPath:           cfg.QueriesPath,
		QrelsPath:             cfg.QrelsPath,
		ElapsedSeconds:        time.Since(start).Seconds(),
		CreatedAt:             time.Now().UTC(),
	}
	if cfg.ManifestJSONPath != "" {
		if err := WriteRetrievalVectorExportSummaryFile(cfg.ManifestJSONPath, summary); err != nil {
			return RetrievalVectorExportSummary{}, err
		}
	}
	return summary, nil
}

// WriteRetrievalVectorExportSummaryFile writes a JSON manifest for an export run.
func WriteRetrievalVectorExportSummaryFile(path string, summary RetrievalVectorExportSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func normalizeRetrievalVectorExportConfig(cfg RetrievalVectorExportConfig) RetrievalVectorExportConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.DocumentChunkMinWords == 0 {
		cfg.DocumentChunkMinWords = 1
	}
	return cfg
}

func validateRetrievalVectorChunkConfig(cfg RetrievalVectorExportConfig) error {
	if cfg.BatchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	if cfg.MaxDocs < 0 || cfg.MaxQueries < 0 {
		return fmt.Errorf("max-docs and max-queries must be non-negative")
	}
	if cfg.OutputDim < 0 {
		return fmt.Errorf("output-dim must be non-negative")
	}
	if cfg.DocumentChunkWords < 0 {
		return fmt.Errorf("document-chunk-words must be non-negative")
	}
	if cfg.DocumentChunkOverlap < 0 {
		return fmt.Errorf("document-chunk-overlap must be non-negative")
	}
	if cfg.DocumentChunkMinWords <= 0 {
		return fmt.Errorf("document-chunk-min-words must be positive")
	}
	if cfg.DocumentChunkWords == 0 && cfg.DocumentChunkOverlap != 0 {
		return fmt.Errorf("document-chunk-overlap requires document-chunk-words")
	}
	if cfg.DocumentChunkWords > 0 && cfg.DocumentChunkOverlap >= cfg.DocumentChunkWords {
		return fmt.Errorf("document-chunk-overlap must be smaller than document-chunk-words")
	}
	return nil
}

func readRetrievalExportCorpus(path string, limit int, qrels retrievalQrels) ([]retrievalTextRecord, error) {
	if qrels != nil {
		return readBEIRCorpusWithRelevant(path, limit, qrels)
	}
	return readBEIRCorpus(path, limit)
}

func readRetrievalExportQueries(path string, limit int, qrels retrievalQrels) ([]retrievalTextRecord, int, error) {
	if qrels != nil {
		return readBEIRQueries(path, qrels, limit)
	}
	records, err := readBEIRTextFile(path, nil, limit)
	return records, 0, err
}

func chunkRetrievalDocuments(docs []retrievalTextRecord, chunkWords, overlap, minWords int) []retrievalDocumentChunk {
	out := make([]retrievalDocumentChunk, 0, len(docs))
	for _, doc := range docs {
		out = append(out, chunkRetrievalDocumentText(doc.ID, doc.Text, chunkWords, overlap, minWords)...)
	}
	return out
}

func chunkRetrievalDocumentText(parentID, text string, chunkWords, overlap, minWords int) []retrievalDocumentChunk {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	if chunkWords <= 0 || len(words) <= chunkWords {
		return []retrievalDocumentChunk{{ParentID: parentID, ChildID: fmt.Sprintf("%s#chunk-0000", parentID), Text: strings.Join(words, " ")}}
	}
	chunks := []retrievalDocumentChunk{}
	step := chunkWords - overlap
	for start := 0; start < len(words); {
		end := start + chunkWords
		if end > len(words) {
			end = len(words)
		}
		chunk := words[start:end]
		if len(chunks) > 0 && len(chunk) < minWords {
			break
		}
		chunks = append(chunks, retrievalDocumentChunk{
			ParentID: parentID,
			ChildID:  fmt.Sprintf("%s#chunk-%04d", parentID, len(chunks)),
			Text:     strings.Join(chunk, " "),
		})
		if end >= len(words) {
			break
		}
		start += step
	}
	if len(chunks) == 0 {
		chunks = append(chunks, retrievalDocumentChunk{ParentID: parentID, ChildID: fmt.Sprintf("%s#chunk-0000", parentID), Text: strings.Join(words, " ")})
	}
	return chunks
}

func writeRetrievalVectorCache(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord, path string, batchSize int, prefix string, outputDim int) (int, int, error) {
	prefixed := prefixRetrievalRecords(records, prefix)
	vectors, err := embedRetrievalTexts(ctx, model, prefixed, batchSize)
	if err != nil {
		return 0, 0, err
	}
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim := 0, 0
	for _, vector := range vectors {
		if len(vector.Vector) == 0 {
			return 0, 0, fmt.Errorf("vector for %q is empty", vector.ID)
		}
		if modelDim == 0 {
			modelDim = len(vector.Vector)
		} else if len(vector.Vector) != modelDim {
			return 0, 0, fmt.Errorf("vector for %q has encoded dimension %d, want %d", vector.ID, len(vector.Vector), modelDim)
		}
		embedding, err := transformRetrievalExportVector(vector.Vector, outputDim)
		if err != nil {
			return 0, 0, fmt.Errorf("vector for %q: %w", vector.ID, err)
		}
		if dim == 0 {
			dim = len(embedding)
		} else if len(embedding) != dim {
			return 0, 0, fmt.Errorf("vector for %q has dimension %d, want %d", vector.ID, len(embedding), dim)
		}
		row := retrievalVectorExportRow{ID: vector.ID, Embedding: embedding}
		data, err := json.Marshal(row)
		if err != nil {
			return 0, 0, err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return 0, 0, err
		}
	}
	if err := writer.Flush(); err != nil {
		return 0, 0, err
	}
	return dim, modelDim, nil
}

func writeRetrievalChildVectorCache(ctx context.Context, model *EmbeddingModel, chunks []retrievalDocumentChunk, path string, batchSize int, prefix string, outputDim int) (int, int, error) {
	records := make([]retrievalTextRecord, len(chunks))
	for i, chunk := range chunks {
		records[i] = retrievalTextRecord{ID: chunk.ChildID, Text: chunk.Text}
	}
	vectors, err := embedRetrievalTexts(ctx, model, prefixRetrievalRecords(records, prefix), batchSize)
	if err != nil {
		return 0, 0, err
	}
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim := 0, 0
	for i, vector := range vectors {
		if len(vector.Vector) == 0 {
			return 0, 0, fmt.Errorf("vector for %q is empty", vector.ID)
		}
		if modelDim == 0 {
			modelDim = len(vector.Vector)
		} else if len(vector.Vector) != modelDim {
			return 0, 0, fmt.Errorf("vector for %q has encoded dimension %d, want %d", vector.ID, len(vector.Vector), modelDim)
		}
		embedding, err := transformRetrievalExportVector(vector.Vector, outputDim)
		if err != nil {
			return 0, 0, fmt.Errorf("vector for %q: %w", vector.ID, err)
		}
		if dim == 0 {
			dim = len(embedding)
		} else if len(embedding) != dim {
			return 0, 0, fmt.Errorf("vector for %q has dimension %d, want %d", vector.ID, len(embedding), dim)
		}
		row := retrievalVectorExportRow{
			ParentID:  chunks[i].ParentID,
			ChildID:   chunks[i].ChildID,
			Embedding: embedding,
		}
		data, err := json.Marshal(row)
		if err != nil {
			return 0, 0, err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return 0, 0, err
		}
	}
	if err := writer.Flush(); err != nil {
		return 0, 0, err
	}
	return dim, modelDim, nil
}

func transformRetrievalExportVector(vector []float32, outputDim int) ([]float32, error) {
	if outputDim == 0 || outputDim == len(vector) {
		return append([]float32(nil), vector...), nil
	}
	if outputDim > len(vector) {
		return nil, fmt.Errorf("output-dim %d exceeds encoded vector dimension %d", outputDim, len(vector))
	}
	if outputDim < 0 {
		return nil, fmt.Errorf("output-dim must be non-negative")
	}
	return normalizeRetrievalVector(vector[:outputDim]), nil
}

func prefixRetrievalRecords(records []retrievalTextRecord, prefix string) []retrievalTextRecord {
	if prefix == "" {
		return records
	}
	out := make([]retrievalTextRecord, len(records))
	for i, record := range records {
		out[i] = retrievalTextRecord{ID: record.ID, Text: prefix + record.Text}
	}
	return out
}
