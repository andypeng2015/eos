package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const PretrainedBERTRetrievalVectorExportManifestSchema = "manta.pretrained_bert_retrieval_vector_export.v1"

// PretrainedBERTTextEmbedder runs an imported BERT-family embedder module with
// Hugging Face WordPiece tokenization and role-named weight files.
type PretrainedBERTTextEmbedder struct {
	config    PretrainedBERTConfig
	tokenizer *HFWordPieceTokenizer
	program   *Program
	maxLength int
}

type PretrainedBERTTextEmbedderConfig struct {
	SourceDir   string
	ModulePath  string
	WeightsPath string
	MaxLength   int
	Runtime     *Runtime
}

type PretrainedBERTRetrievalVectorExportConfig struct {
	DatasetName      string
	DatasetDir       string
	CorpusPath       string
	QueriesPath      string
	QrelsPath        string
	OutputDir        string
	SourceDir        string
	ModulePath       string
	WeightsPath      string
	QueryPrefix      string
	DocumentPrefix   string
	BatchSize        int
	MaxDocs          int
	MaxQueries       int
	MaxLength        int
	Split            string
	Runtime          *Runtime
	ManifestJSONPath string
}

type PretrainedBERTRetrievalVectorExportSummary struct {
	Schema              string    `json:"schema"`
	Dataset             string    `json:"dataset"`
	SourceDir           string    `json:"source_dir"`
	ModulePath          string    `json:"module_path"`
	WeightsPath         string    `json:"weights_path"`
	ExecutionMode       string    `json:"execution_mode"`
	QualityClaim        bool      `json:"quality_claim"`
	Documents           int       `json:"documents"`
	Queries             int       `json:"queries"`
	NativeDim           int       `json:"native_dim"`
	OutputDim           int       `json:"output_dim"`
	DocVectorPath       string    `json:"doc_vector_path"`
	QueryVectorPath     string    `json:"query_vector_path"`
	QueryPrefix         string    `json:"query_prefix"`
	DocumentPrefix      string    `json:"document_prefix"`
	LegacyDocPrefix     string    `json:"doc_prefix"`
	DocumentRoleApplied bool      `json:"document_role_applied"`
	QueryRoleApplied    bool      `json:"query_role_applied"`
	MaxLength           int       `json:"max_length"`
	BatchSize           int       `json:"batch_size"`
	MaxDocs             int       `json:"max_docs,omitempty"`
	MaxQueries          int       `json:"max_queries,omitempty"`
	CorpusPath          string    `json:"corpus_path,omitempty"`
	QueriesPath         string    `json:"queries_path,omitempty"`
	QrelsPath           string    `json:"qrels_path,omitempty"`
	ElapsedSeconds      float64   `json:"elapsed_seconds"`
	CreatedAt           time.Time `json:"created_at"`
}

func LoadPretrainedBERTTextEmbedder(ctx context.Context, cfg PretrainedBERTTextEmbedderConfig) (*PretrainedBERTTextEmbedder, error) {
	if cfg.SourceDir == "" || cfg.ModulePath == "" || cfg.WeightsPath == "" {
		return nil, fmt.Errorf("source dir, module path, and weights path are required")
	}
	if cfg.Runtime == nil {
		return nil, fmt.Errorf("runtime is required")
	}
	config, err := LoadPretrainedBERTConfig(filepath.Join(cfg.SourceDir, "config.json"))
	if err != nil {
		return nil, err
	}
	maxLength := cfg.MaxLength
	if maxLength <= 0 {
		maxLength = config.MaxPositionEmbeddings
	}
	if maxLength <= 0 || maxLength > config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("max length must be in [1,%d], got %d", config.MaxPositionEmbeddings, maxLength)
	}
	tokenizer, err := LoadHFWordPieceTokenizerFromDir(cfg.SourceDir)
	if err != nil {
		return nil, err
	}
	module, err := eosartifact.ReadFile(cfg.ModulePath)
	if err != nil {
		return nil, err
	}
	weights, err := ReadWeightFile(cfg.WeightsPath)
	if err != nil {
		return nil, err
	}
	program, err := cfg.Runtime.Load(ctx, module, weights.LoadOptions()...)
	if err != nil {
		return nil, err
	}
	return &PretrainedBERTTextEmbedder{
		config:    config,
		tokenizer: tokenizer,
		program:   program,
		maxLength: maxLength,
	}, nil
}

func (e *PretrainedBERTTextEmbedder) MaxLength() int {
	if e == nil {
		return 0
	}
	return e.maxLength
}

func (e *PretrainedBERTTextEmbedder) EmbedTextBatch(ctx context.Context, texts []string, prefix string) ([][]float32, error) {
	if e == nil || e.program == nil || e.tokenizer == nil {
		return nil, fmt.Errorf("pretrained BERT text embedder is not loaded")
	}
	if len(texts) == 0 {
		return nil, nil
	}
	inputIDs := make([]int32, 0, len(texts)*e.maxLength)
	attentionMask := make([]int32, 0, len(texts)*e.maxLength)
	tokenTypeIDs := make([]int32, 0, len(texts)*e.maxLength)
	for _, text := range texts {
		encoded, err := e.tokenizer.Encode(prefix+text, HFWordPieceEncodeOptions{
			MaxLength:      e.maxLength,
			PadToMaxLength: true,
		})
		if err != nil {
			return nil, err
		}
		inputIDs = append(inputIDs, encoded.IDs...)
		attentionMask = append(attentionMask, encoded.AttentionMask...)
		tokenTypeIDs = append(tokenTypeIDs, encoded.TokenTypeIDs...)
	}
	result, err := e.program.Run(ctx, backend.Request{
		Entry: "bert_embed",
		Inputs: map[string]any{
			"input_ids":      backend.NewTensorI32([]int{len(texts), e.maxLength}, inputIDs),
			"attention_mask": backend.NewTensorI32([]int{len(texts), e.maxLength}, attentionMask),
			"token_type_ids": backend.NewTensorI32([]int{len(texts), e.maxLength}, tokenTypeIDs),
		},
	})
	if err != nil {
		return nil, err
	}
	value, ok := result.Outputs["embeddings"]
	if !ok {
		return nil, fmt.Errorf("bert_embed output missing embeddings")
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok {
		return nil, fmt.Errorf("bert_embed embeddings output has data type %T, want *backend.Tensor", value.Data)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != len(texts) {
		return nil, fmt.Errorf("bert_embed embeddings shape = %v, want [%d,D]", tensor.Shape, len(texts))
	}
	dim := tensor.Shape[1]
	if dim <= 0 || len(tensor.F32) != len(texts)*dim {
		return nil, fmt.Errorf("bert_embed embeddings backing data length %d does not match shape %v", len(tensor.F32), tensor.Shape)
	}
	out := make([][]float32, len(texts))
	for row := range texts {
		start := row * dim
		vec := append([]float32(nil), tensor.F32[start:start+dim]...)
		for i, value := range vec {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("embedding row %d value %d is not finite: %v", row, i, value)
			}
		}
		out[row] = vec
	}
	return out, nil
}

func ExportPretrainedBERTRetrievalVectors(ctx context.Context, cfg PretrainedBERTRetrievalVectorExportConfig) (PretrainedBERTRetrievalVectorExportSummary, error) {
	cfg = normalizePretrainedBERTRetrievalVectorExportConfig(cfg)
	if cfg.OutputDir == "" || cfg.SourceDir == "" || cfg.ModulePath == "" || cfg.WeightsPath == "" {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("output dir, source dir, module path, and weights path are required")
	}
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("corpus path and queries path are required")
	}
	if cfg.BatchSize <= 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("batch-size must be positive")
	}
	if cfg.MaxDocs < 0 || cfg.MaxQueries < 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("max-docs and max-queries must be non-negative")
	}
	start := time.Now()
	embedder, err := LoadPretrainedBERTTextEmbedder(ctx, PretrainedBERTTextEmbedderConfig{
		SourceDir:   cfg.SourceDir,
		ModulePath:  cfg.ModulePath,
		WeightsPath: cfg.WeightsPath,
		MaxLength:   cfg.MaxLength,
		Runtime:     cfg.Runtime,
	})
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	var qrels retrievalQrels
	if cfg.QrelsPath != "" {
		qrels, err = readBEIRQrels(cfg.QrelsPath)
		if err != nil {
			return PretrainedBERTRetrievalVectorExportSummary{}, err
		}
	}
	corpus, err := readRetrievalExportCorpus(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	queries, _, err := readRetrievalExportQueries(cfg.QueriesPath, cfg.MaxQueries, qrels)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	if len(corpus) == 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("queries are empty")
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	docVectorPath := filepath.Join(cfg.OutputDir, "doc-vectors.jsonl")
	queryVectorPath := filepath.Join(cfg.OutputDir, "query-vectors.jsonl")
	docDim, err := writePretrainedBERTVectorCache(ctx, embedder, corpus, docVectorPath, cfg.BatchSize, cfg.DocumentPrefix)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("write document vectors: %w", err)
	}
	queryDim, err := writePretrainedBERTVectorCache(ctx, embedder, queries, queryVectorPath, cfg.BatchSize, cfg.QueryPrefix)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if docDim != queryDim {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}
	summary := PretrainedBERTRetrievalVectorExportSummary{
		Schema:              PretrainedBERTRetrievalVectorExportManifestSchema,
		Dataset:             cfg.DatasetName,
		SourceDir:           cfg.SourceDir,
		ModulePath:          cfg.ModulePath,
		WeightsPath:         cfg.WeightsPath,
		ExecutionMode:       "pretrained_bert_host_reference",
		QualityClaim:        false,
		Documents:           len(corpus),
		Queries:             len(queries),
		NativeDim:           docDim,
		OutputDim:           docDim,
		DocVectorPath:       docVectorPath,
		QueryVectorPath:     queryVectorPath,
		QueryPrefix:         cfg.QueryPrefix,
		DocumentPrefix:      cfg.DocumentPrefix,
		LegacyDocPrefix:     cfg.DocumentPrefix,
		DocumentRoleApplied: cfg.DocumentPrefix != "",
		QueryRoleApplied:    cfg.QueryPrefix != "",
		MaxLength:           embedder.MaxLength(),
		BatchSize:           cfg.BatchSize,
		MaxDocs:             cfg.MaxDocs,
		MaxQueries:          cfg.MaxQueries,
		CorpusPath:          cfg.CorpusPath,
		QueriesPath:         cfg.QueriesPath,
		QrelsPath:           cfg.QrelsPath,
		ElapsedSeconds:      time.Since(start).Seconds(),
		CreatedAt:           time.Now().UTC(),
	}
	if err := WritePretrainedBERTRetrievalVectorExportSummaryFile(cfg.ManifestJSONPath, summary); err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	return summary, nil
}

func WritePretrainedBERTRetrievalVectorExportSummaryFile(path string, summary PretrainedBERTRetrievalVectorExportSummary) error {
	if path == "" {
		path = filepath.Join(filepath.Dir(summary.DocVectorPath), "manifest.json")
	}
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

func normalizePretrainedBERTRetrievalVectorExportConfig(cfg PretrainedBERTRetrievalVectorExportConfig) PretrainedBERTRetrievalVectorExportConfig {
	if cfg.DatasetName == "" {
		if cfg.DatasetDir != "" {
			cfg.DatasetName = filepath.Base(cfg.DatasetDir)
		} else {
			cfg.DatasetName = "retrieval"
		}
	}
	if cfg.Split == "" {
		cfg.Split = "test"
	}
	if cfg.DatasetDir != "" {
		corpus, queries, qrels := BEIRRetrievalPaths(cfg.DatasetDir, cfg.Split)
		if cfg.CorpusPath == "" {
			cfg.CorpusPath = corpus
		}
		if cfg.QueriesPath == "" {
			cfg.QueriesPath = queries
		}
		if cfg.QrelsPath == "" {
			cfg.QrelsPath = qrels
		}
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.ManifestJSONPath == "" && cfg.OutputDir != "" {
		cfg.ManifestJSONPath = filepath.Join(cfg.OutputDir, "manifest.json")
	}
	return cfg
}

func writePretrainedBERTVectorCache(ctx context.Context, embedder *PretrainedBERTTextEmbedder, records []retrievalTextRecord, path string, batchSize int, prefix string) (int, error) {
	file, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	dim := 0
	for start := 0; start < len(records); start += batchSize {
		end := start + batchSize
		if end > len(records) {
			end = len(records)
		}
		texts := make([]string, end-start)
		for i, record := range records[start:end] {
			texts[i] = record.Text
		}
		vectors, err := embedder.EmbedTextBatch(ctx, texts, prefix)
		if err != nil {
			return 0, err
		}
		for i, vector := range vectors {
			record := records[start+i]
			if len(vector) == 0 {
				return 0, fmt.Errorf("vector for %q is empty", record.ID)
			}
			if dim == 0 {
				dim = len(vector)
			} else if len(vector) != dim {
				return 0, fmt.Errorf("vector for %q has dimension %d, want %d", record.ID, len(vector), dim)
			}
			row := retrievalVectorExportRow{ID: record.ID, Embedding: vector}
			data, err := json.Marshal(row)
			if err != nil {
				return 0, err
			}
			if _, err := writer.Write(append(data, '\n')); err != nil {
				return 0, err
			}
		}
		if err := writer.Flush(); err != nil {
			return 0, err
		}
	}
	if err := writer.Flush(); err != nil {
		return 0, err
	}
	return dim, nil
}
