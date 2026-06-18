package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"m31labs.dev/eos/runtime/backend"
)

const SparseTokenPoolRetrievalVectorExportManifestSchema = "manta.experimental_sparse_token_pool_retrieval_vector_export.v1"

// SparseTokenPoolRetrievalVectorExportConfig describes an experimental BEIR
// vector-cache export that pools token embeddings through routed TurboQuant
// sparse attention. It is a prototype path, not the production embedding model.
type SparseTokenPoolRetrievalVectorExportConfig struct {
	DatasetName           string
	ArtifactPath          string
	WeightFilePath        string
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
	TopK                  int
	RouteBlockSize        int
	RouteTopBlocks        int
	Bits                  int
	Seed                  int64
	MaxTokens             int
}

// SparseTokenPoolRetrievalVectorExportSummary is the manifest written beside
// experimental sparse-token pooled vector caches.
type SparseTokenPoolRetrievalVectorExportSummary struct {
	Schema                  string    `json:"schema"`
	Method                  string    `json:"method"`
	Experimental            bool      `json:"experimental"`
	QualityClaim            bool      `json:"quality_claim"`
	ClaimBoundary           string    `json:"claim_boundary"`
	Dataset                 string    `json:"dataset"`
	Artifact                string    `json:"artifact,omitempty"`
	WeightFile              string    `json:"weight_file,omitempty"`
	TokenizerPresent        bool      `json:"tokenizer_present"`
	Documents               int       `json:"documents"`
	Queries                 int       `json:"queries"`
	ChildVectors            int       `json:"child_vectors,omitempty"`
	Dimension               int       `json:"dimension"`
	ModelDimension          int       `json:"model_dimension,omitempty"`
	OutputDimension         int       `json:"output_dimension,omitempty"`
	DocVectorPath           string    `json:"doc_vector_path,omitempty"`
	ChildDocVectorPath      string    `json:"child_doc_vector_path,omitempty"`
	QueryVectorPath         string    `json:"query_vector_path"`
	DocumentChunkWords      int       `json:"document_chunk_words,omitempty"`
	DocumentChunkOverlap    int       `json:"document_chunk_overlap,omitempty"`
	DocumentChunkMinWords   int       `json:"document_chunk_min_words,omitempty"`
	BatchSize               int       `json:"batch_size"`
	MaxDocs                 int       `json:"max_docs,omitempty"`
	MaxQueries              int       `json:"max_queries,omitempty"`
	MaxTokens               int       `json:"max_tokens,omitempty"`
	CorpusPath              string    `json:"corpus_path,omitempty"`
	QueriesPath             string    `json:"queries_path,omitempty"`
	QrelsPath               string    `json:"qrels_path,omitempty"`
	TopK                    int       `json:"top_k"`
	RouteBlockSize          int       `json:"route_block_size,omitempty"`
	RouteTopBlocks          int       `json:"route_top_blocks,omitempty"`
	Bits                    int       `json:"bits"`
	QuantizerSeed           int64     `json:"quantizer_seed"`
	DenseKVMaterialized     bool      `json:"dense_kv_materialized"`
	KVDecode                string    `json:"kv_decode"`
	AttentionWeightsApplied bool      `json:"attention_weights_applied"`
	AttentionOutputApplied  bool      `json:"attention_output_applied"`
	ProjectionApplied       bool      `json:"projection_applied"`
	TokenEmbeddingParam     string    `json:"token_embedding_param"`
	AttentionQueryParam     string    `json:"attention_query_param,omitempty"`
	AttentionKeyParam       string    `json:"attention_key_param,omitempty"`
	AttentionValueParam     string    `json:"attention_value_param,omitempty"`
	AttentionOutputParam    string    `json:"attention_output_param,omitempty"`
	ProjectionParam         string    `json:"projection_param,omitempty"`
	SkippedWeights          []string  `json:"skipped_weights,omitempty"`
	Caveats                 []string  `json:"caveats"`
	ElapsedSeconds          float64   `json:"elapsed_seconds"`
	CreatedAt               time.Time `json:"created_at"`
}

type sparseTokenPoolEncoder struct {
	model              *EmbeddingModel
	weights            WeightFile
	tokenEmbedding     *backend.Tensor
	attentionQuery     *backend.Tensor
	attentionKey       *backend.Tensor
	attentionValue     *backend.Tensor
	attentionOutput    *backend.Tensor
	projection         *backend.Tensor
	cfg                SparseTokenPoolRetrievalVectorExportConfig
	manifest           EmbeddingManifest
	attentionWeightsOK bool
	attentionOutputOK  bool
	projectionOK       bool
	skippedWeights     []string
	modelDimension     int
	projectedDimension int
}

// ExportSparseTokenPoolRetrievalVectors exports BEIR-compatible vector caches
// using actual tokenizer ids and token_embedding rows plus routed TurboQuant
// sparse attention over the token sequence. The output is intentionally marked
// experimental and carries no retrieval-quality claim.
func ExportSparseTokenPoolRetrievalVectors(ctx context.Context, model *EmbeddingModel, cfg SparseTokenPoolRetrievalVectorExportConfig) (SparseTokenPoolRetrievalVectorExportSummary, error) {
	if model == nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeSparseTokenPoolExportConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.OutputDir == "" {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("corpus path, queries path, and output dir are required")
	}
	if cfg.WeightFilePath == "" {
		if cfg.ArtifactPath == "" {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("artifact path or weight file path is required")
		}
		cfg.WeightFilePath = DefaultWeightFilePath(cfg.ArtifactPath)
	}
	if err := validateSparseTokenPoolExportConfig(cfg); err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, err
	}
	weights, err := ReadWeightFile(cfg.WeightFilePath)
	if err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("read weights: %w", err)
	}
	encoder, err := newSparseTokenPoolEncoder(model, weights, cfg)
	if err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, err
	}

	start := time.Now()
	var qrels retrievalQrels
	if cfg.QrelsPath != "" {
		qrels, err = readBEIRQrels(cfg.QrelsPath)
		if err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, err
		}
	}
	corpus, err := readRetrievalExportCorpus(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, err
	}
	queries, _, err := readRetrievalExportQueries(cfg.QueriesPath, cfg.MaxQueries, qrels)
	if err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, err
	}
	if len(corpus) == 0 {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("queries are empty")
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, err
	}

	queryVectorPath := filepath.Join(cfg.OutputDir, "query-vectors.jsonl")
	var docVectorPath, childDocVectorPath string
	var dim, modelDim, childCount int
	if cfg.DocumentChunkWords > 0 {
		chunks := chunkRetrievalDocuments(corpus, cfg.DocumentChunkWords, cfg.DocumentChunkOverlap, cfg.DocumentChunkMinWords)
		if len(chunks) == 0 {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("document chunking selected no chunks")
		}
		childDocVectorPath = filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
		dim, modelDim, err = writeSparseTokenPoolChildVectorCache(ctx, encoder, chunks, childDocVectorPath, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("write child document vectors: %w", err)
		}
		childCount = len(chunks)
	} else {
		docVectorPath = filepath.Join(cfg.OutputDir, "doc-vectors.jsonl")
		dim, modelDim, err = writeSparseTokenPoolVectorCache(ctx, encoder, corpus, docVectorPath, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("write document vectors: %w", err)
		}
	}
	queryDim, queryModelDim, err := writeSparseTokenPoolVectorCache(ctx, encoder, queries, queryVectorPath, cfg.QueryPrefix, cfg.OutputDim)
	if err != nil {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if dim != queryDim {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", dim, queryDim)
	}
	if modelDim != queryModelDim {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("document vectors have encoded dimension %d but query vectors have encoded dimension %d", modelDim, queryModelDim)
	}

	summary := SparseTokenPoolRetrievalVectorExportSummary{
		Schema:                  SparseTokenPoolRetrievalVectorExportManifestSchema,
		Method:                  "experimental_sparse_token_pool",
		Experimental:            true,
		QualityClaim:            false,
		ClaimBoundary:           "Prototype sparse-token pooling retrieval cache. Not production dense embedder output, not trained sparse encoder evidence, and not LongEmbed proof.",
		Dataset:                 cfg.DatasetName,
		Artifact:                cfg.ArtifactPath,
		WeightFile:              cfg.WeightFilePath,
		TokenizerPresent:        model.HasTokenizer(),
		Documents:               len(corpus),
		Queries:                 len(queries),
		ChildVectors:            childCount,
		Dimension:               dim,
		ModelDimension:          modelDim,
		OutputDimension:         dim,
		DocVectorPath:           docVectorPath,
		ChildDocVectorPath:      childDocVectorPath,
		QueryVectorPath:         queryVectorPath,
		DocumentChunkWords:      cfg.DocumentChunkWords,
		DocumentChunkOverlap:    cfg.DocumentChunkOverlap,
		DocumentChunkMinWords:   cfg.DocumentChunkMinWords,
		BatchSize:               cfg.BatchSize,
		MaxDocs:                 cfg.MaxDocs,
		MaxQueries:              cfg.MaxQueries,
		MaxTokens:               cfg.MaxTokens,
		CorpusPath:              cfg.CorpusPath,
		QueriesPath:             cfg.QueriesPath,
		QrelsPath:               cfg.QrelsPath,
		TopK:                    cfg.TopK,
		RouteBlockSize:          cfg.RouteBlockSize,
		RouteTopBlocks:          cfg.RouteTopBlocks,
		Bits:                    cfg.Bits,
		QuantizerSeed:           cfg.Seed,
		DenseKVMaterialized:     true,
		KVDecode:                "host_reference_decode",
		AttentionWeightsApplied: encoder.attentionWeightsOK,
		AttentionOutputApplied:  encoder.attentionOutputOK,
		ProjectionApplied:       encoder.projectionOK,
		TokenEmbeddingParam:     encoder.manifest.TokenEmbeddingParam,
		AttentionQueryParam:     encoder.manifest.AttentionQueryParam,
		AttentionKeyParam:       encoder.manifest.AttentionKeyParam,
		AttentionValueParam:     encoder.manifest.AttentionValueParam,
		AttentionOutputParam:    encoder.manifest.AttentionOutputParam,
		ProjectionParam:         encoder.manifest.ProjectionParam,
		SkippedWeights:          encoder.skippedWeights,
		Caveats: []string{
			"experimental_sparse_token_pool uses one mean-derived global query row over token embeddings",
			"TurboSparseAttentionReference decodes quantized K/V on host in this prototype",
			"quality_claim=false; score generated caches before comparing and do not promote as a trained sparse encoder",
		},
		ElapsedSeconds: time.Since(start).Seconds(),
		CreatedAt:      time.Now().UTC(),
	}
	if cfg.ManifestJSONPath != "" {
		if err := WriteSparseTokenPoolRetrievalVectorExportSummaryFile(cfg.ManifestJSONPath, summary); err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, err
		}
	}
	return summary, nil
}

func WriteSparseTokenPoolRetrievalVectorExportSummaryFile(path string, summary SparseTokenPoolRetrievalVectorExportSummary) error {
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

func normalizeSparseTokenPoolExportConfig(cfg SparseTokenPoolRetrievalVectorExportConfig) SparseTokenPoolRetrievalVectorExportConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	if cfg.DocumentChunkMinWords == 0 {
		cfg.DocumentChunkMinWords = 1
	}
	if cfg.Bits == 0 {
		cfg.Bits = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = 0x4d697261
	}
	return cfg
}

func validateSparseTokenPoolExportConfig(cfg SparseTokenPoolRetrievalVectorExportConfig) error {
	if err := validateRetrievalVectorChunkConfig(RetrievalVectorExportConfig{
		BatchSize:             cfg.BatchSize,
		MaxDocs:               cfg.MaxDocs,
		MaxQueries:            cfg.MaxQueries,
		OutputDim:             cfg.OutputDim,
		DocumentChunkWords:    cfg.DocumentChunkWords,
		DocumentChunkOverlap:  cfg.DocumentChunkOverlap,
		DocumentChunkMinWords: cfg.DocumentChunkMinWords,
	}); err != nil {
		return err
	}
	if cfg.Bits != 2 && cfg.Bits != 4 && cfg.Bits != 8 {
		return fmt.Errorf("bits must be 2, 4, or 8")
	}
	if cfg.TopK < 0 || cfg.RouteBlockSize < 0 || cfg.RouteTopBlocks < 0 || cfg.MaxTokens < 0 {
		return fmt.Errorf("top-k, route-block-size, route-top-blocks, and max-tokens must be non-negative")
	}
	return nil
}

func newSparseTokenPoolEncoder(model *EmbeddingModel, weights WeightFile, cfg SparseTokenPoolRetrievalVectorExportConfig) (*sparseTokenPoolEncoder, error) {
	manifest := model.Manifest().normalized()
	token := weights.Weights[manifest.TokenEmbeddingParam]
	if token == nil {
		return nil, fmt.Errorf("missing token embedding weight %q", manifest.TokenEmbeddingParam)
	}
	if len(token.Shape) != 2 || token.Shape[0] <= 0 || token.Shape[1] <= 0 || len(token.F32) < token.Shape[0]*token.Shape[1] {
		return nil, fmt.Errorf("token embedding weight %q has invalid shape/data: %v", manifest.TokenEmbeddingParam, token.Shape)
	}
	enc := &sparseTokenPoolEncoder{
		model:          model,
		weights:        weights,
		tokenEmbedding: token,
		cfg:            cfg,
		manifest:       manifest,
		modelDimension: token.Shape[1],
	}
	enc.attentionQuery = enc.optionalMatrix(manifest.AttentionQueryParam, token.Shape[1], token.Shape[1])
	enc.attentionKey = enc.optionalMatrix(manifest.AttentionKeyParam, token.Shape[1], token.Shape[1])
	enc.attentionValue = enc.optionalMatrix(manifest.AttentionValueParam, token.Shape[1], token.Shape[1])
	if enc.attentionQuery != nil && enc.attentionKey != nil && enc.attentionValue != nil {
		enc.attentionWeightsOK = true
	}
	if enc.attentionWeightsOK {
		enc.attentionOutput = enc.optionalMatrix(manifest.AttentionOutputParam, token.Shape[1], token.Shape[1])
		enc.attentionOutputOK = enc.attentionOutput != nil
	}
	enc.projection = enc.optionalProjection(manifest.ProjectionParam, token.Shape[1])
	enc.projectionOK = enc.projection != nil
	if enc.projectionOK {
		enc.projectedDimension = enc.projection.Shape[1]
	} else {
		enc.projectedDimension = enc.modelDimension
	}
	return enc, nil
}

func (e *sparseTokenPoolEncoder) optionalMatrix(name string, in, out int) *backend.Tensor {
	if name == "" {
		return nil
	}
	t := e.weights.Weights[name]
	if t == nil {
		e.skippedWeights = append(e.skippedWeights, name+":missing")
		return nil
	}
	if len(t.Shape) != 2 || t.Shape[0] != in || t.Shape[1] != out || len(t.F32) < in*out {
		e.skippedWeights = append(e.skippedWeights, fmt.Sprintf("%s:shape_%v_want_%dx%d", name, t.Shape, in, out))
		return nil
	}
	return t
}

func (e *sparseTokenPoolEncoder) optionalProjection(name string, in int) *backend.Tensor {
	if name == "" {
		return nil
	}
	t := e.weights.Weights[name]
	if t == nil {
		e.skippedWeights = append(e.skippedWeights, name+":missing")
		return nil
	}
	if len(t.Shape) != 2 || t.Shape[0] != in || t.Shape[1] <= 0 || len(t.F32) < t.Shape[0]*t.Shape[1] {
		e.skippedWeights = append(e.skippedWeights, fmt.Sprintf("%s:shape_%v_want_%dxN", name, t.Shape, in))
		return nil
	}
	return t
}

func writeSparseTokenPoolVectorCache(ctx context.Context, encoder *sparseTokenPoolEncoder, records []retrievalTextRecord, path, prefix string, outputDim int) (int, int, error) {
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim := 0, 0
	for _, record := range prefixRetrievalRecords(records, prefix) {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		vector, err := encoder.embedText(record.Text)
		if err != nil {
			return 0, 0, fmt.Errorf("vector for %q: %w", record.ID, err)
		}
		if modelDim == 0 {
			modelDim = len(vector)
		} else if len(vector) != modelDim {
			return 0, 0, fmt.Errorf("vector for %q has encoded dimension %d, want %d", record.ID, len(vector), modelDim)
		}
		embedding, err := transformRetrievalExportVector(vector, outputDim)
		if err != nil {
			return 0, 0, fmt.Errorf("vector for %q: %w", record.ID, err)
		}
		if dim == 0 {
			dim = len(embedding)
		} else if len(embedding) != dim {
			return 0, 0, fmt.Errorf("vector for %q has dimension %d, want %d", record.ID, len(embedding), dim)
		}
		row := retrievalVectorExportRow{ID: record.ID, Embedding: embedding}
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

func writeSparseTokenPoolChildVectorCache(ctx context.Context, encoder *sparseTokenPoolEncoder, chunks []retrievalDocumentChunk, path, prefix string, outputDim int) (int, int, error) {
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim := 0, 0
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		vector, err := encoder.embedText(prefix + chunk.Text)
		if err != nil {
			return 0, 0, fmt.Errorf("vector for %q: %w", chunk.ChildID, err)
		}
		if modelDim == 0 {
			modelDim = len(vector)
		} else if len(vector) != modelDim {
			return 0, 0, fmt.Errorf("vector for %q has encoded dimension %d, want %d", chunk.ChildID, len(vector), modelDim)
		}
		embedding, err := transformRetrievalExportVector(vector, outputDim)
		if err != nil {
			return 0, 0, fmt.Errorf("vector for %q: %w", chunk.ChildID, err)
		}
		if dim == 0 {
			dim = len(embedding)
		} else if len(embedding) != dim {
			return 0, 0, fmt.Errorf("vector for %q has dimension %d, want %d", chunk.ChildID, len(embedding), dim)
		}
		row := retrievalVectorExportRow{ParentID: chunk.ParentID, ChildID: chunk.ChildID, Embedding: embedding}
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

func (e *sparseTokenPoolEncoder) embedText(text string) ([]float32, error) {
	tokens, _, err := e.model.TokenizeText(text)
	if err != nil {
		return nil, err
	}
	if e.cfg.MaxTokens > 0 && len(tokens) > e.cfg.MaxTokens {
		tokens = tokens[:e.cfg.MaxTokens]
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer produced no tokens")
	}
	hidden := e.gatherTokenRows(tokens)
	dim := e.tokenEmbedding.Shape[1]
	queryRow := meanRows(hidden, len(tokens), dim)
	keyRows := hidden
	valueRows := hidden
	if e.attentionWeightsOK {
		queryRow = matmulVectorRight(queryRow, e.attentionQuery)
		keyRows = matmulRowsRight(hidden, len(tokens), dim, e.attentionKey)
		valueRows = matmulRowsRight(hidden, len(tokens), dim, e.attentionValue)
	}
	query := backend.NewTensorF16([]int{1, dim}, queryRow)
	keyNCHW := attentionRowsToNCHW(keyRows, len(tokens), dim)
	valueNCHW := attentionRowsToNCHW(valueRows, len(tokens), dim)
	attrs := map[string]string{
		"bits": strconv.Itoa(e.cfg.Bits),
		"seed": strconv.FormatInt(e.cfg.Seed, 10),
	}
	if e.cfg.TopK > 0 {
		attrs["top_k"] = strconv.Itoa(e.cfg.TopK)
	}
	if e.cfg.RouteBlockSize > 0 {
		attrs["route_block_size"] = strconv.Itoa(e.cfg.RouteBlockSize)
	}
	if e.cfg.RouteTopBlocks > 0 {
		attrs["route_top_blocks"] = strconv.Itoa(e.cfg.RouteTopBlocks)
	}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		return nil, fmt.Errorf("encode key rows: %w", err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		return nil, fmt.Errorf("encode value rows: %w", err)
	}
	attended, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		return nil, fmt.Errorf("turbo sparse attention: %w", err)
	}
	if len(attended.Shape) != 2 || attended.Shape[0] != 1 || attended.Shape[1] != dim || len(attended.F32) < dim {
		return nil, fmt.Errorf("unexpected sparse attention output shape %v", attended.Shape)
	}
	vector := append([]float32(nil), attended.F32[:dim]...)
	if e.attentionOutputOK {
		vector = matmulVectorRight(vector, e.attentionOutput)
	}
	if e.projectionOK {
		vector = matmulVectorRight(vector, e.projection)
	}
	return normalizeRetrievalVector(vector), nil
}

func (e *sparseTokenPoolEncoder) gatherTokenRows(tokens []int32) []float32 {
	dim := e.tokenEmbedding.Shape[1]
	vocabRows := e.tokenEmbedding.Shape[0]
	out := make([]float32, len(tokens)*dim)
	for i, tok := range tokens {
		row := int(tok)
		if row < 0 || row >= vocabRows {
			continue
		}
		copy(out[i*dim:(i+1)*dim], e.tokenEmbedding.F32[row*dim:(row+1)*dim])
	}
	return out
}

func meanRows(rows []float32, count, dim int) []float32 {
	out := make([]float32, dim)
	if count <= 0 || dim <= 0 {
		return out
	}
	for i := 0; i < count; i++ {
		for d := 0; d < dim; d++ {
			out[d] += rows[i*dim+d]
		}
	}
	scale := float32(1.0 / float64(count))
	for d := range out {
		out[d] *= scale
	}
	return out
}

func matmulRowsRight(rows []float32, rowCount, inDim int, weight *backend.Tensor) []float32 {
	outDim := weight.Shape[1]
	out := make([]float32, rowCount*outDim)
	for r := 0; r < rowCount; r++ {
		for o := 0; o < outDim; o++ {
			sum := float32(0)
			for i := 0; i < inDim; i++ {
				sum += rows[r*inDim+i] * weight.F32[i*outDim+o]
			}
			out[r*outDim+o] = sum
		}
	}
	return out
}

func matmulVectorRight(vector []float32, weight *backend.Tensor) []float32 {
	outDim := weight.Shape[1]
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		sum := float32(0)
		for i := 0; i < len(vector); i++ {
			sum += vector[i] * weight.F32[i*outDim+o]
		}
		out[o] = sum
	}
	return out
}

func attentionRowsToNCHW(rows []float32, seqLen, dim int) *backend.Tensor {
	data := make([]float32, dim*seqLen)
	for t := 0; t < seqLen; t++ {
		for d := 0; d < dim; d++ {
			data[d*seqLen+t] = rows[t*dim+d]
		}
	}
	return backend.NewTensorF16([]int{1, dim, seqLen, 1}, data)
}
