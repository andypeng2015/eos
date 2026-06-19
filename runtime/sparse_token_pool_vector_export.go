package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"m31labs.dev/eos/runtime/backend"
)

const SparseTokenPoolRetrievalVectorExportManifestSchema = "manta.experimental_sparse_token_pool_retrieval_vector_export.v1"
const (
	SparseTokenPoolAttentionModeTurboQuantSparse = "turboquant_sparse"
	SparseTokenPoolAttentionModeDense            = "dense"
)

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
	TokenSpanTokens       int
	TokenSpanOverlap      int
	TokenSpanMinTokens    int
	DocumentPrefix        string
	QueryPrefix           string
	ManifestJSONPath      string
	TopK                  int
	RouteBlockSize        int
	RouteTopBlocks        int
	Bits                  int
	KeyBits               int
	ValueBits             int
	Seed                  int64
	MaxTokens             int
	MinObservedDocTokens  int
	AttentionMode         string
}

// SparseTokenPoolRetrievalVectorExportSummary is the manifest written beside
// experimental sparse-token pooled vector caches.
type SparseTokenPoolRetrievalVectorExportSummary struct {
	Schema                  string                    `json:"schema"`
	Method                  string                    `json:"method"`
	Experimental            bool                      `json:"experimental"`
	QualityClaim            bool                      `json:"quality_claim"`
	ClaimBoundary           string                    `json:"claim_boundary"`
	Dataset                 string                    `json:"dataset"`
	Artifact                string                    `json:"artifact,omitempty"`
	WeightFile              string                    `json:"weight_file,omitempty"`
	TokenizerPresent        bool                      `json:"tokenizer_present"`
	Documents               int                       `json:"documents"`
	Queries                 int                       `json:"queries"`
	ChildVectors            int                       `json:"child_vectors,omitempty"`
	Dimension               int                       `json:"dimension"`
	ModelDimension          int                       `json:"model_dimension,omitempty"`
	OutputDimension         int                       `json:"output_dimension,omitempty"`
	DocVectorPath           string                    `json:"doc_vector_path,omitempty"`
	ChildDocVectorPath      string                    `json:"child_doc_vector_path,omitempty"`
	QueryVectorPath         string                    `json:"query_vector_path"`
	DocumentChunkWords      int                       `json:"document_chunk_words,omitempty"`
	DocumentChunkOverlap    int                       `json:"document_chunk_overlap,omitempty"`
	DocumentChunkMinWords   int                       `json:"document_chunk_min_words,omitempty"`
	TokenSpanTokens         int                       `json:"token_span_tokens,omitempty"`
	TokenSpanOverlap        int                       `json:"token_span_overlap,omitempty"`
	TokenSpanMinTokens      int                       `json:"token_span_min_tokens,omitempty"`
	BatchSize               int                       `json:"batch_size"`
	MaxDocs                 int                       `json:"max_docs,omitempty"`
	MaxQueries              int                       `json:"max_queries,omitempty"`
	MaxTokens               int                       `json:"max_tokens,omitempty"`
	MinObservedDocTokens    int                       `json:"min_observed_doc_tokens,omitempty"`
	CorpusPath              string                    `json:"corpus_path,omitempty"`
	QueriesPath             string                    `json:"queries_path,omitempty"`
	QrelsPath               string                    `json:"qrels_path,omitempty"`
	TopK                    int                       `json:"top_k"`
	RouteBlockSize          int                       `json:"route_block_size,omitempty"`
	RouteTopBlocks          int                       `json:"route_top_blocks,omitempty"`
	Bits                    int                       `json:"bits"`
	KeyBits                 int                       `json:"key_bits"`
	ValueBits               int                       `json:"value_bits"`
	QuantizerSeed           int64                     `json:"quantizer_seed"`
	AttentionMode           string                    `json:"attention_mode"`
	TurboQuantKVApplied     bool                      `json:"turboquant_kv_applied"`
	DenseKVMaterialized     bool                      `json:"dense_kv_materialized"`
	KVDecode                string                    `json:"kv_decode"`
	AttentionWeightsApplied bool                      `json:"attention_weights_applied"`
	AttentionOutputApplied  bool                      `json:"attention_output_applied"`
	HiddenProjectionApplied bool                      `json:"hidden_projection_applied"`
	ProjectionApplied       bool                      `json:"projection_applied"`
	EncoderRepeatsApplied   int                       `json:"encoder_repeats_applied,omitempty"`
	AttentionResidual       bool                      `json:"attention_residual,omitempty"`
	AttentionLayerNorm      bool                      `json:"attention_layernorm,omitempty"`
	FFNResidual             bool                      `json:"ffn_residual,omitempty"`
	FFNLayerNorm            bool                      `json:"ffn_layernorm,omitempty"`
	TokenEmbeddingParam     string                    `json:"token_embedding_param"`
	AttentionQueryParam     string                    `json:"attention_query_param,omitempty"`
	AttentionKeyParam       string                    `json:"attention_key_param,omitempty"`
	AttentionValueParam     string                    `json:"attention_value_param,omitempty"`
	AttentionOutputParam    string                    `json:"attention_output_param,omitempty"`
	HiddenProjectionParam   string                    `json:"hidden_projection_param,omitempty"`
	ProjectionParam         string                    `json:"projection_param,omitempty"`
	SkippedWeights          []string                  `json:"skipped_weights,omitempty"`
	DocumentTokenizerOutput TokenizerOutputTokenStats `json:"document_tokenizer_output"`
	QueryTokenizerOutput    TokenizerOutputTokenStats `json:"query_tokenizer_output"`
	Caveats                 []string                  `json:"caveats"`
	ElapsedSeconds          float64                   `json:"elapsed_seconds"`
	CreatedAt               time.Time                 `json:"created_at"`
}

// TokenizerOutputTokenStats records token lengths after the packaged
// tokenizer and after the export-level --max-tokens cap, matching the sequence
// lengths consumed by the sparse-token-pool encoder.
type TokenizerOutputTokenStats struct {
	RecordCount               int     `json:"record_count"`
	TotalTokens               int64   `json:"total_tokens"`
	MaxObservedTokens         int     `json:"max_observed_tokens"`
	MeanObservedTokens        float64 `json:"mean_observed_tokens"`
	TruncatedByMaxTokensCount int     `json:"truncated_by_max_tokens_count"`
}

type sparseTokenPoolTokenLengthStats struct {
	RecordCount               int
	TotalTokens               int64
	MaxObservedTokens         int
	TruncatedByMaxTokensCount int
}

type sparseTokenPoolTokenObservation struct {
	ObservedTokens       int
	TruncatedByMaxTokens bool
}

type sparseTokenPoolTokenSpan struct {
	start int
	end   int
}

type sparseTokenPoolEncoder struct {
	model              *EmbeddingModel
	weights            WeightFile
	tokenEmbedding     *backend.Tensor
	attentionQuery     *backend.Tensor
	attentionKey       *backend.Tensor
	attentionValue     *backend.Tensor
	attentionOutput    *backend.Tensor
	hiddenProjection   *backend.Tensor
	projection         *backend.Tensor
	cfg                SparseTokenPoolRetrievalVectorExportConfig
	manifest           EmbeddingManifest
	attentionWeightsOK bool
	attentionOutputOK  bool
	hiddenProjectionOK bool
	projectionOK       bool
	fullEncoderOK      bool
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
	var docTokenStats, queryTokenStats sparseTokenPoolTokenLengthStats
	if cfg.TokenSpanTokens > 0 {
		childDocVectorPath = filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
		dim, modelDim, docTokenStats, childCount, err = writeSparseTokenPoolTokenSpanChildVectorCache(ctx, encoder, corpus, childDocVectorPath, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("write token-span child document vectors: %w", err)
		}
	} else if cfg.DocumentChunkWords > 0 {
		chunks := chunkRetrievalDocuments(corpus, cfg.DocumentChunkWords, cfg.DocumentChunkOverlap, cfg.DocumentChunkMinWords)
		if len(chunks) == 0 {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("document chunking selected no chunks")
		}
		childDocVectorPath = filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
		dim, modelDim, docTokenStats, err = writeSparseTokenPoolChildVectorCache(ctx, encoder, chunks, childDocVectorPath, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("write child document vectors: %w", err)
		}
		childCount = len(chunks)
	} else {
		docVectorPath = filepath.Join(cfg.OutputDir, "doc-vectors.jsonl")
		dim, modelDim, docTokenStats, err = writeSparseTokenPoolVectorCache(ctx, encoder, corpus, docVectorPath, cfg.DocumentPrefix, cfg.OutputDim)
		if err != nil {
			return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("write document vectors: %w", err)
		}
	}
	if cfg.MinObservedDocTokens > 0 && docTokenStats.MaxObservedTokens < cfg.MinObservedDocTokens {
		return SparseTokenPoolRetrievalVectorExportSummary{}, fmt.Errorf("sparse-token-pool observed document tokenizer-output max tokens %d below --min-observed-doc-tokens %d; check the tokenizer max sequence, input corpus length, --max-tokens, and qrels/max-docs filtering", docTokenStats.MaxObservedTokens, cfg.MinObservedDocTokens)
	}
	queryDim, queryModelDim, queryTokenStats, err := writeSparseTokenPoolVectorCache(ctx, encoder, queries, queryVectorPath, cfg.QueryPrefix, cfg.OutputDim)
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
		TokenSpanTokens:         cfg.TokenSpanTokens,
		TokenSpanOverlap:        cfg.TokenSpanOverlap,
		TokenSpanMinTokens:      cfg.TokenSpanMinTokens,
		BatchSize:               cfg.BatchSize,
		MaxDocs:                 cfg.MaxDocs,
		MaxQueries:              cfg.MaxQueries,
		MaxTokens:               cfg.MaxTokens,
		MinObservedDocTokens:    cfg.MinObservedDocTokens,
		CorpusPath:              cfg.CorpusPath,
		QueriesPath:             cfg.QueriesPath,
		QrelsPath:               cfg.QrelsPath,
		TopK:                    cfg.TopK,
		RouteBlockSize:          cfg.RouteBlockSize,
		RouteTopBlocks:          cfg.RouteTopBlocks,
		Bits:                    cfg.Bits,
		KeyBits:                 cfg.KeyBits,
		ValueBits:               cfg.ValueBits,
		QuantizerSeed:           cfg.Seed,
		AttentionMode:           cfg.AttentionMode,
		TurboQuantKVApplied:     cfg.AttentionMode == SparseTokenPoolAttentionModeTurboQuantSparse,
		DenseKVMaterialized:     true,
		KVDecode:                encoder.kvDecode(),
		AttentionWeightsApplied: encoder.attentionWeightsOK,
		AttentionOutputApplied:  encoder.attentionOutputOK,
		HiddenProjectionApplied: encoder.hiddenProjectionOK,
		ProjectionApplied:       encoder.projectionOK,
		EncoderRepeatsApplied:   encoder.encoderRepeatsApplied(),
		AttentionResidual:       encoder.manifest.AttentionResidual,
		AttentionLayerNorm:      encoder.manifest.AttentionLayerNorm,
		FFNResidual:             encoder.manifest.FFNResidual,
		FFNLayerNorm:            encoder.manifest.FFNLayerNorm,
		TokenEmbeddingParam:     encoder.manifest.TokenEmbeddingParam,
		AttentionQueryParam:     encoder.manifest.AttentionQueryParam,
		AttentionKeyParam:       encoder.manifest.AttentionKeyParam,
		AttentionValueParam:     encoder.manifest.AttentionValueParam,
		AttentionOutputParam:    encoder.manifest.AttentionOutputParam,
		HiddenProjectionParam:   encoder.manifest.HiddenProjectionParam,
		ProjectionParam:         encoder.manifest.ProjectionParam,
		SkippedWeights:          encoder.skippedWeights,
		DocumentTokenizerOutput: docTokenStats.summary(),
		QueryTokenizerOutput:    queryTokenStats.summary(),
		Caveats:                 encoder.caveats(),
		ElapsedSeconds:          time.Since(start).Seconds(),
		CreatedAt:               time.Now().UTC(),
	}
	if len(summary.Caveats) == 0 {
		summary.Caveats = []string{
			"TurboSparseAttentionReference decodes quantized K/V on host in this prototype",
			"quality_claim=false; score generated caches before comparing and do not promote as a trained sparse encoder",
		}
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
	if cfg.TokenSpanTokens > 0 && cfg.TokenSpanMinTokens == 0 {
		cfg.TokenSpanMinTokens = 1
	}
	if cfg.Bits == 0 {
		cfg.Bits = 4
	}
	if cfg.KeyBits == 0 {
		cfg.KeyBits = cfg.Bits
	}
	if cfg.ValueBits == 0 {
		cfg.ValueBits = cfg.Bits
	}
	if cfg.Seed == 0 {
		cfg.Seed = 0x4d697261
	}
	if cfg.AttentionMode == "" {
		cfg.AttentionMode = SparseTokenPoolAttentionModeTurboQuantSparse
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
	if cfg.TokenSpanTokens < 0 || cfg.TokenSpanOverlap < 0 || cfg.TokenSpanMinTokens < 0 {
		return fmt.Errorf("token-span-tokens, token-span-overlap, and token-span-min-tokens must be non-negative")
	}
	if cfg.TokenSpanTokens == 0 && (cfg.TokenSpanOverlap != 0 || cfg.TokenSpanMinTokens != 0) {
		return fmt.Errorf("token-span-overlap and token-span-min-tokens require token-span-tokens")
	}
	if cfg.TokenSpanTokens > 0 && cfg.DocumentChunkWords > 0 {
		return fmt.Errorf("token-span-tokens is mutually exclusive with document-chunk-words")
	}
	if cfg.TokenSpanTokens > 0 && cfg.TokenSpanOverlap >= cfg.TokenSpanTokens {
		return fmt.Errorf("token-span-overlap must be smaller than token-span-tokens")
	}
	if cfg.TokenSpanTokens > 0 && cfg.TokenSpanMinTokens > cfg.TokenSpanTokens {
		return fmt.Errorf("token-span-min-tokens must be less than or equal to token-span-tokens")
	}
	if cfg.Bits != 2 && cfg.Bits != 4 && cfg.Bits != 8 {
		return fmt.Errorf("bits must be 2, 4, or 8")
	}
	if cfg.KeyBits != 2 && cfg.KeyBits != 4 && cfg.KeyBits != 8 {
		return fmt.Errorf("key-bits must be 0, 2, 4, or 8")
	}
	if cfg.ValueBits != 2 && cfg.ValueBits != 4 && cfg.ValueBits != 8 {
		return fmt.Errorf("value-bits must be 0, 2, 4, or 8")
	}
	if cfg.TopK < 0 || cfg.RouteBlockSize < 0 || cfg.RouteTopBlocks < 0 || cfg.MaxTokens < 0 || cfg.MinObservedDocTokens < 0 {
		return fmt.Errorf("top-k, route-block-size, route-top-blocks, max-tokens, and min-observed-doc-tokens must be non-negative")
	}
	switch cfg.AttentionMode {
	case SparseTokenPoolAttentionModeTurboQuantSparse, SparseTokenPoolAttentionModeDense:
	default:
		return fmt.Errorf("attention-mode must be %q or %q, got %q", SparseTokenPoolAttentionModeTurboQuantSparse, SparseTokenPoolAttentionModeDense, cfg.AttentionMode)
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
	if manifest.HiddenProjectionParam != "" {
		enc.hiddenProjection = enc.optionalProjection(manifest.HiddenProjectionParam, token.Shape[1])
		enc.hiddenProjectionOK = enc.hiddenProjection != nil
	}
	projectionInput := token.Shape[1]
	if enc.hiddenProjectionOK {
		projectionInput = enc.hiddenProjection.Shape[1]
	}
	enc.projection = enc.optionalProjection(manifest.ProjectionParam, projectionInput)
	enc.projectionOK = enc.projection != nil
	enc.fullEncoderOK = enc.attentionWeightsOK && enc.attentionOutputOK && enc.hiddenProjectionOK && enc.projectionOK
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

func writeSparseTokenPoolVectorCache(ctx context.Context, encoder *sparseTokenPoolEncoder, records []retrievalTextRecord, path, prefix string, outputDim int) (int, int, sparseTokenPoolTokenLengthStats, error) {
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim := 0, 0
	var stats sparseTokenPoolTokenLengthStats
	for _, record := range prefixRetrievalRecords(records, prefix) {
		if err := ctx.Err(); err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, err
		}
		vector, observation, err := encoder.embedText(record.Text)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q: %w", record.ID, err)
		}
		stats.add(observation)
		if modelDim == 0 {
			modelDim = len(vector)
		} else if len(vector) != modelDim {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q has encoded dimension %d, want %d", record.ID, len(vector), modelDim)
		}
		embedding, err := transformRetrievalExportVector(vector, outputDim)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q: %w", record.ID, err)
		}
		if dim == 0 {
			dim = len(embedding)
		} else if len(embedding) != dim {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q has dimension %d, want %d", record.ID, len(embedding), dim)
		}
		row := retrievalVectorExportRow{ID: record.ID, Embedding: embedding}
		data, err := json.Marshal(row)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, err
		}
	}
	if err := writer.Flush(); err != nil {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, err
	}
	return dim, modelDim, stats, nil
}

func writeSparseTokenPoolTokenSpanChildVectorCache(ctx context.Context, encoder *sparseTokenPoolEncoder, records []retrievalTextRecord, path, prefix string, outputDim int) (int, int, sparseTokenPoolTokenLengthStats, int, error) {
	if !encoder.fullEncoderOK {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token-span child vectors require full manifest encoder weights; missing/invalid weights: %v", encoder.skippedWeights)
	}
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim, childCount := 0, 0, 0
	var stats sparseTokenPoolTokenLengthStats
	for _, record := range prefixRetrievalRecords(records, prefix) {
		if err := ctx.Err(); err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, err
		}
		rows, mask, observation, err := encoder.embedTextTokenRows(record.Text)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token rows for %q: %w", record.ID, err)
		}
		stats.add(observation)
		seqLen := observation.ObservedTokens
		rowDim := encoder.modelDimension
		if modelDim == 0 {
			modelDim = rowDim
		} else if rowDim != modelDim {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token rows for %q have encoded dimension %d, want %d", record.ID, rowDim, modelDim)
		}
		spans := sparseTokenPoolTokenSpans(seqLen, encoder.cfg.TokenSpanTokens, encoder.cfg.TokenSpanOverlap, encoder.cfg.TokenSpanMinTokens)
		if len(spans) == 0 {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token-span mode selected no spans for %q", record.ID)
		}
		for i, span := range spans {
			vector, err := poolSparseTokenRowsRange(rows, mask, span.start, span.end, seqLen, rowDim)
			if err != nil {
				return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token span %d for %q: %w", i, record.ID, err)
			}
			embedding, err := transformRetrievalExportVector(vector, outputDim)
			if err != nil {
				return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token span %d for %q: %w", i, record.ID, err)
			}
			if dim == 0 {
				dim = len(embedding)
			} else if len(embedding) != dim {
				return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, fmt.Errorf("token span %d for %q has dimension %d, want %d", i, record.ID, len(embedding), dim)
			}
			row := retrievalVectorExportRow{
				ParentID:  record.ID,
				ChildID:   fmt.Sprintf("%s#token-span-%04d", record.ID, i),
				Embedding: embedding,
			}
			data, err := json.Marshal(row)
			if err != nil {
				return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, err
			}
			if _, err := writer.Write(append(data, '\n')); err != nil {
				return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, err
			}
			childCount++
		}
	}
	if err := writer.Flush(); err != nil {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, 0, err
	}
	return dim, modelDim, stats, childCount, nil
}

func writeSparseTokenPoolChildVectorCache(ctx context.Context, encoder *sparseTokenPoolEncoder, chunks []retrievalDocumentChunk, path, prefix string, outputDim int) (int, int, sparseTokenPoolTokenLengthStats, error) {
	file, err := os.Create(path)
	if err != nil {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim, modelDim := 0, 0
	var stats sparseTokenPoolTokenLengthStats
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, err
		}
		vector, observation, err := encoder.embedText(prefix + chunk.Text)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q: %w", chunk.ChildID, err)
		}
		stats.add(observation)
		if modelDim == 0 {
			modelDim = len(vector)
		} else if len(vector) != modelDim {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q has encoded dimension %d, want %d", chunk.ChildID, len(vector), modelDim)
		}
		embedding, err := transformRetrievalExportVector(vector, outputDim)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q: %w", chunk.ChildID, err)
		}
		if dim == 0 {
			dim = len(embedding)
		} else if len(embedding) != dim {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, fmt.Errorf("vector for %q has dimension %d, want %d", chunk.ChildID, len(embedding), dim)
		}
		row := retrievalVectorExportRow{ParentID: chunk.ParentID, ChildID: chunk.ChildID, Embedding: embedding}
		data, err := json.Marshal(row)
		if err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return 0, 0, sparseTokenPoolTokenLengthStats{}, err
		}
	}
	if err := writer.Flush(); err != nil {
		return 0, 0, sparseTokenPoolTokenLengthStats{}, err
	}
	return dim, modelDim, stats, nil
}

func (e *sparseTokenPoolEncoder) embedText(text string) ([]float32, sparseTokenPoolTokenObservation, error) {
	tokens, mask, observation, err := e.tokenizeText(text)
	if err != nil {
		return nil, sparseTokenPoolTokenObservation{}, err
	}
	var vector []float32
	if e.fullEncoderOK {
		vector, err = e.embedTextFullEncoder(tokens, mask)
	} else {
		vector, err = e.embedTokenIDs(tokens)
	}
	if err != nil {
		return nil, sparseTokenPoolTokenObservation{}, err
	}
	return vector, observation, nil
}

func (e *sparseTokenPoolEncoder) embedTextTokenRows(text string) ([]float32, []int32, sparseTokenPoolTokenObservation, error) {
	tokens, mask, observation, err := e.tokenizeText(text)
	if err != nil {
		return nil, nil, sparseTokenPoolTokenObservation{}, err
	}
	if !e.fullEncoderOK {
		return nil, nil, sparseTokenPoolTokenObservation{}, fmt.Errorf("full manifest encoder weights are required; missing/invalid weights: %v", e.skippedWeights)
	}
	rows, err := e.embedTextFullEncoderRows(tokens)
	if err != nil {
		return nil, nil, sparseTokenPoolTokenObservation{}, err
	}
	return rows, mask, observation, nil
}

func (e *sparseTokenPoolEncoder) tokenizeText(text string) ([]int32, []int32, sparseTokenPoolTokenObservation, error) {
	tokens, mask, err := e.model.TokenizeText(text)
	if err != nil {
		return nil, nil, sparseTokenPoolTokenObservation{}, err
	}
	truncatedByMaxTokens := false
	if e.cfg.MaxTokens > 0 && len(tokens) > e.cfg.MaxTokens {
		tokens = tokens[:e.cfg.MaxTokens]
		if len(mask) > e.cfg.MaxTokens {
			mask = mask[:e.cfg.MaxTokens]
		}
		truncatedByMaxTokens = true
	}
	if len(tokens) == 0 {
		return nil, nil, sparseTokenPoolTokenObservation{}, fmt.Errorf("tokenizer produced no tokens")
	}
	observation := sparseTokenPoolTokenObservation{
		ObservedTokens:       len(tokens),
		TruncatedByMaxTokens: truncatedByMaxTokens,
	}
	return tokens, mask, observation, nil
}

func (e *sparseTokenPoolEncoder) embedTokenIDs(tokens []int32) ([]float32, error) {
	if e.cfg.AttentionMode == SparseTokenPoolAttentionModeDense {
		return nil, fmt.Errorf("attention-mode %q requires full manifest encoder weights; missing/invalid weights: %v", e.cfg.AttentionMode, e.skippedWeights)
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
		"seed": strconv.FormatInt(e.cfg.Seed, 10),
	}
	encodeKeyAttrs := map[string]string{
		"bits": strconv.Itoa(e.cfg.KeyBits),
		"seed": strconv.FormatInt(e.cfg.Seed, 10),
	}
	encodeValueAttrs := map[string]string{
		"bits": strconv.Itoa(e.cfg.ValueBits),
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
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, encodeKeyAttrs)
	if err != nil {
		return nil, fmt.Errorf("encode key rows: %w", err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, encodeValueAttrs)
	if err != nil {
		return nil, fmt.Errorf("encode value rows: %w", err)
	}
	attended, err := e.attendCompressed(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
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
	if e.hiddenProjectionOK && e.projectionOK {
		vector = matmulVectorRight(vector, e.hiddenProjection)
		for i, value := range vector {
			vector[i] = geluForward(value)
		}
		vector = matmulVectorRight(vector, e.projection)
		return normalizeRetrievalVector(vector), nil
	}
	if e.projectionOK {
		vector = matmulVectorRight(vector, e.projection)
	}
	return normalizeRetrievalVector(vector), nil
}

func (s *sparseTokenPoolTokenLengthStats) add(observation sparseTokenPoolTokenObservation) {
	s.RecordCount++
	s.TotalTokens += int64(observation.ObservedTokens)
	if observation.ObservedTokens > s.MaxObservedTokens {
		s.MaxObservedTokens = observation.ObservedTokens
	}
	if observation.TruncatedByMaxTokens {
		s.TruncatedByMaxTokensCount++
	}
}

func (s sparseTokenPoolTokenLengthStats) summary() TokenizerOutputTokenStats {
	out := TokenizerOutputTokenStats{
		RecordCount:               s.RecordCount,
		TotalTokens:               s.TotalTokens,
		MaxObservedTokens:         s.MaxObservedTokens,
		TruncatedByMaxTokensCount: s.TruncatedByMaxTokensCount,
	}
	if s.RecordCount > 0 {
		out.MeanObservedTokens = float64(s.TotalTokens) / float64(s.RecordCount)
	}
	return out
}

func (e *sparseTokenPoolEncoder) embedTextFullEncoder(tokens []int32, mask []int32) ([]float32, error) {
	current, err := e.embedTextFullEncoderRows(tokens)
	if err != nil {
		return nil, err
	}
	seqLen := len(tokens)
	dim := e.tokenEmbedding.Shape[1]
	return maskedMeanPoolNormalized(current, mask, seqLen, dim), nil
}

func (e *sparseTokenPoolEncoder) embedTextFullEncoderRows(tokens []int32) ([]float32, error) {
	current := e.gatherTokenRows(tokens)
	seqLen := len(tokens)
	dim := e.tokenEmbedding.Shape[1]
	attrs := e.sparseAttentionAttrs()
	for layer := 0; layer < e.manifest.EncoderRepeats; layer++ {
		q := matmulRowsRight(current, seqLen, dim, e.attentionQuery)
		k := matmulRowsRight(current, seqLen, dim, e.attentionKey)
		v := matmulRowsRight(current, seqLen, dim, e.attentionValue)
		attendedRows, err := e.attendRows(q, k, v, seqLen, dim, attrs)
		if err != nil {
			return nil, err
		}
		attnOutput := matmulRowsRight(attendedRows, seqLen, dim, e.attentionOutput)
		hidden := attnOutput
		if e.manifest.AttentionResidual {
			hidden = addRows(hidden, current)
		}
		if e.manifest.AttentionLayerNorm {
			hidden = layerNormRows(hidden, seqLen, dim)
		}
		ffnHidden := matmulRowsRight(hidden, seqLen, dim, e.hiddenProjection)
		for i, value := range ffnHidden {
			ffnHidden[i] = geluForward(value)
		}
		ffnOut := matmulRowsRight(ffnHidden, seqLen, e.hiddenProjection.Shape[1], e.projection)
		if e.manifest.FFNResidual {
			ffnOut = addRows(ffnOut, hidden)
		}
		if e.manifest.FFNLayerNorm {
			ffnOut = layerNormRows(ffnOut, seqLen, dim)
		}
		current = ffnOut
	}
	normalizeRowsInPlace(current, seqLen, dim)
	return current, nil
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

func (e *sparseTokenPoolEncoder) sparseAttentionAttrs() map[string]string {
	attrs := map[string]string{
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
	return attrs
}

func (e *sparseTokenPoolEncoder) attendRows(q, k, v []float32, seqLen, dim int, attrs map[string]string) ([]float32, error) {
	switch e.cfg.AttentionMode {
	case SparseTokenPoolAttentionModeDense:
		return denseSoftmaxAttentionRows(q, k, v, seqLen, dim), nil
	case SparseTokenPoolAttentionModeTurboQuantSparse:
		encodeKeyAttrs := map[string]string{
			"bits": strconv.Itoa(e.cfg.KeyBits),
			"seed": strconv.FormatInt(e.cfg.Seed, 10),
		}
		encodeValueAttrs := map[string]string{
			"bits": strconv.Itoa(e.cfg.ValueBits),
			"seed": strconv.FormatInt(e.cfg.Seed, 10),
		}
		keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(attentionRowsToNCHW(k, seqLen, dim), encodeKeyAttrs)
		if err != nil {
			return nil, fmt.Errorf("encode key rows: %w", err)
		}
		valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(attentionRowsToNCHW(v, seqLen, dim), encodeValueAttrs)
		if err != nil {
			return nil, fmt.Errorf("encode value rows: %w", err)
		}
		attended, err := e.attendCompressed(backend.NewTensorF16([]int{seqLen, dim}, q), keyCoords, keyNorms, valueCoords, valueNorms, attrs)
		if err != nil {
			return nil, fmt.Errorf("turbo sparse attention: %w", err)
		}
		if len(attended.Shape) != 2 || attended.Shape[0] != seqLen || attended.Shape[1] != dim || len(attended.F32) < seqLen*dim {
			return nil, fmt.Errorf("unexpected sparse attention output shape %v", attended.Shape)
		}
		return append([]float32(nil), attended.F32[:seqLen*dim]...), nil
	default:
		return nil, fmt.Errorf("unsupported attention mode %q", e.cfg.AttentionMode)
	}
}

func (e *sparseTokenPoolEncoder) attendCompressed(query, keyCoords, keyNorms, valueCoords, valueNorms *backend.Tensor, attrs map[string]string) (*backend.Tensor, error) {
	if e.cfg.KeyBits == e.cfg.ValueBits {
		equalAttrs := cloneStringMap(attrs)
		equalAttrs["bits"] = strconv.Itoa(e.cfg.KeyBits)
		return backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, equalAttrs)
	}
	decodeAttrs := map[string]string{"seed": strconv.FormatInt(e.cfg.Seed, 10)}
	keyDense, err := backend.TurboQuantDecodeReference(keyCoords, keyNorms, decodeAttrs)
	if err != nil {
		return nil, fmt.Errorf("key decode: %w", err)
	}
	valueDense, err := backend.TurboQuantDecodeReference(valueCoords, valueNorms, decodeAttrs)
	if err != nil {
		return nil, fmt.Errorf("value decode: %w", err)
	}
	queryRank := len(query.Shape)
	keyRows, err := attentionNCHWToRows(keyDense, queryRank)
	if err != nil {
		return nil, fmt.Errorf("key layout: %w", err)
	}
	valueRows, err := attentionNCHWToRows(valueDense, queryRank)
	if err != nil {
		return nil, fmt.Errorf("value layout: %w", err)
	}
	return backend.SparseAttentionReference(query, keyRows, valueRows, attrs)
}

func denseSoftmaxAttentionRows(q, k, v []float32, seqLen, dim int) []float32 {
	out := make([]float32, seqLen*dim)
	scores := make([]float32, seqLen)
	for row := 0; row < seqLen; row++ {
		maxScore := float32(math.Inf(-1))
		for col := 0; col < seqLen; col++ {
			score := float32(0)
			for d := 0; d < dim; d++ {
				score += q[row*dim+d] * k[col*dim+d]
			}
			scores[col] = score
			if score > maxScore {
				maxScore = score
			}
		}
		sum := float32(0)
		for col := 0; col < seqLen; col++ {
			prob := float32(math.Exp(float64(scores[col] - maxScore)))
			scores[col] = prob
			sum += prob
		}
		if sum == 0 {
			continue
		}
		for col := 0; col < seqLen; col++ {
			prob := scores[col] / sum
			for d := 0; d < dim; d++ {
				out[row*dim+d] += prob * v[col*dim+d]
			}
		}
	}
	return out
}

func (e *sparseTokenPoolEncoder) encoderRepeatsApplied() int {
	if e.fullEncoderOK {
		return e.manifest.EncoderRepeats
	}
	return 0
}

func (e *sparseTokenPoolEncoder) caveats() []string {
	caveats := []string{
		"quality_claim=false; score generated caches before comparing and do not promote as a trained sparse encoder",
	}
	if e.cfg.AttentionMode == SparseTokenPoolAttentionModeDense {
		caveats = append([]string{"dense attention diagnostic mode uses exact host softmax over materialized K/V and bypasses TurboQuant K/V encode/decode"}, caveats...)
	} else {
		caveats = append([]string{"TurboSparseAttentionReference decodes quantized K/V on host in this prototype"}, caveats...)
	}
	if !e.fullEncoderOK {
		caveats = append([]string{"experimental_sparse_token_pool used legacy fallback path because full encoder weights were unavailable"}, caveats...)
	}
	return caveats
}

func (e *sparseTokenPoolEncoder) kvDecode() string {
	if e.cfg.AttentionMode == SparseTokenPoolAttentionModeDense {
		return "not_applicable_dense_attention"
	}
	return "host_reference_decode"
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

func addRows(left, right []float32) []float32 {
	out := make([]float32, len(left))
	for i := range left {
		out[i] = left[i] + right[i]
	}
	return out
}

func layerNormRows(rows []float32, rowCount, dim int) []float32 {
	out := make([]float32, len(rows))
	for r := 0; r < rowCount; r++ {
		layerNormRow(out[r*dim:(r+1)*dim], rows[r*dim:(r+1)*dim])
	}
	return out
}

func normalizeRowsInPlace(rows []float32, rowCount, dim int) {
	for r := 0; r < rowCount; r++ {
		base := r * dim
		normalized := normalizeRetrievalVector(rows[base : base+dim])
		copy(rows[base:base+dim], normalized)
	}
}

func maskedMeanPoolNormalized(rows []float32, mask []int32, rowCount, dim int) []float32 {
	out := make([]float32, dim)
	active := 0
	for r := 0; r < rowCount; r++ {
		if r < len(mask) && mask[r] == 0 {
			continue
		}
		active++
		for d := 0; d < dim; d++ {
			out[d] += rows[r*dim+d]
		}
	}
	if active == 0 {
		active = rowCount
		for r := 0; r < rowCount; r++ {
			for d := 0; d < dim; d++ {
				out[d] += rows[r*dim+d]
			}
		}
	}
	scale := float32(1.0 / float64(active))
	for d := range out {
		out[d] *= scale
	}
	return normalizeRetrievalVector(out)
}

func sparseTokenPoolTokenSpans(tokenCount, spanTokens, overlap, minTokens int) []sparseTokenPoolTokenSpan {
	if tokenCount <= 0 || spanTokens <= 0 {
		return nil
	}
	if minTokens <= 0 {
		minTokens = 1
	}
	if tokenCount <= spanTokens {
		if tokenCount < minTokens {
			return nil
		}
		return []sparseTokenPoolTokenSpan{{start: 0, end: tokenCount}}
	}
	step := spanTokens - overlap
	if step <= 0 {
		return nil
	}
	spans := []sparseTokenPoolTokenSpan{}
	for start := 0; start < tokenCount; start += step {
		end := start + spanTokens
		if end > tokenCount {
			end = tokenCount
		}
		if end-start < minTokens {
			break
		}
		spans = append(spans, sparseTokenPoolTokenSpan{start: start, end: end})
		if end >= tokenCount {
			break
		}
	}
	return spans
}

func poolSparseTokenRowsRange(rows []float32, mask []int32, start, end, rowCount, dim int) ([]float32, error) {
	if start < 0 || end > rowCount || start >= end {
		return nil, fmt.Errorf("invalid token span [%d,%d) for token count %d", start, end, rowCount)
	}
	out := make([]float32, dim)
	active := 0
	for r := start; r < end; r++ {
		if r < len(mask) && mask[r] == 0 {
			continue
		}
		active++
		for d := 0; d < dim; d++ {
			out[d] += rows[r*dim+d]
		}
	}
	if active == 0 {
		return nil, fmt.Errorf("span [%d,%d) has no active tokens", start, end)
	}
	scale := float32(1.0 / float64(active))
	for d := range out {
		out[d] *= scale
	}
	return normalizeRetrievalVector(out), nil
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

func attentionNCHWToRows(input *backend.Tensor, queryRank int) (*backend.Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("nil tensor")
	}
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("expected NCHW tensor, got shape %v", input.Shape)
	}
	batches, channels, seqLen, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	if width != 1 {
		return nil, fmt.Errorf("expected width 1 for attention sequence layout, got %d", width)
	}
	switch queryRank {
	case 2:
		if batches != 1 {
			return nil, fmt.Errorf("rank-2 query expects compressed batch 1, got %d", batches)
		}
		out := backend.NewTensorF16([]int{seqLen, channels}, make([]float32, seqLen*channels))
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				out.F32[t*channels+c] = input.F32[(c*seqLen+t)*width]
			}
		}
		return out, nil
	case 3:
		out := backend.NewTensorF16([]int{batches, seqLen, channels}, make([]float32, batches*seqLen*channels))
		for b := 0; b < batches; b++ {
			for t := 0; t < seqLen; t++ {
				for c := 0; c < channels; c++ {
					out.F32[(b*seqLen+t)*channels+c] = input.F32[((b*channels+c)*seqLen+t)*width]
				}
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("query rank must be 2 or 3, got %d", queryRank)
	}
}

func cloneStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
