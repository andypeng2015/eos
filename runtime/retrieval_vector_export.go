package eosruntime

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	RoleMode              string
	ManifestJSONPath      string
}

// RetrievalVectorExportSummary is a compact manifest for generated vector caches.
type RetrievalVectorExportSummary struct {
	Schema                string    `json:"schema"`
	Dataset               string    `json:"dataset"`
	Artifact              string    `json:"artifact,omitempty"`
	ArtifactSHA256        string    `json:"artifact_sha256,omitempty"`
	WeightsSHA256         string    `json:"weights_sha256,omitempty"`
	TokenizerSHA256       string    `json:"tokenizer_sha256,omitempty"`
	PackageManifestSHA256 string    `json:"package_manifest_sha256,omitempty"`
	PackageCacheKey       string    `json:"package_cache_key,omitempty"`
	EmbeddingSpaceID      string    `json:"embedding_space_id,omitempty"`
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
	DocumentRoleApplied   bool      `json:"document_role_applied"`
	QueryRoleApplied      bool      `json:"query_role_applied"`
	RoleMode              string    `json:"role_mode,omitempty"`
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
	docRole, queryRole, effectiveRoleMode, err := resolveEmbeddingRetrievalRoles(model, cfg.RoleMode)
	if err != nil {
		return RetrievalVectorExportSummary{}, err
	}
	if cfg.DocumentChunkWords > 0 {
		chunks := chunkRetrievalDocuments(corpus, cfg.DocumentChunkWords, cfg.DocumentChunkOverlap, cfg.DocumentChunkMinWords)
		if len(chunks) == 0 {
			return RetrievalVectorExportSummary{}, fmt.Errorf("document chunking selected no chunks")
		}
		childDocVectorPath = filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
		dim, modelDim, err = writeRetrievalChildVectorCache(ctx, model, chunks, childDocVectorPath, cfg.BatchSize, cfg.DocumentPrefix, cfg.OutputDim, docRole)
		if err != nil {
			return RetrievalVectorExportSummary{}, fmt.Errorf("write child document vectors: %w", err)
		}
		childCount = len(chunks)
	} else {
		docVectorPath = filepath.Join(cfg.OutputDir, "doc-vectors.jsonl")
		dim, modelDim, err = writeRetrievalVectorCache(ctx, model, corpus, docVectorPath, cfg.BatchSize, cfg.DocumentPrefix, cfg.OutputDim, docRole)
		if err != nil {
			return RetrievalVectorExportSummary{}, fmt.Errorf("write document vectors: %w", err)
		}
	}
	queryDim, queryModelDim, err := writeRetrievalVectorCache(ctx, model, queries, queryVectorPath, cfg.BatchSize, cfg.QueryPrefix, cfg.OutputDim, queryRole)
	if err != nil {
		return RetrievalVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if dim != queryDim {
		return RetrievalVectorExportSummary{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", dim, queryDim)
	}
	if modelDim != queryModelDim {
		return RetrievalVectorExportSummary{}, fmt.Errorf("document vectors have encoded dimension %d but query vectors have encoded dimension %d", modelDim, queryModelDim)
	}
	documentRoleApplied := cfg.DocumentPrefix != "" || effectiveRoleMode == EmbeddingRoleModeQueryDocument
	queryRoleApplied := cfg.QueryPrefix != "" || effectiveRoleMode == EmbeddingRoleModeQueryDocument
	identity, err := buildNativeRetrievalEmbeddingSpaceIdentity(model, cfg, dim, modelDim, effectiveRoleMode, documentRoleApplied, queryRoleApplied)
	if err != nil {
		return RetrievalVectorExportSummary{}, err
	}

	summary := RetrievalVectorExportSummary{
		Schema:                RetrievalVectorExportManifestSchema,
		Dataset:               cfg.DatasetName,
		Artifact:              cfg.ArtifactPath,
		ArtifactSHA256:        identity.ArtifactSHA256,
		WeightsSHA256:         identity.WeightsSHA256,
		TokenizerSHA256:       identity.TokenizerSHA256,
		PackageManifestSHA256: identity.PackageManifestSHA256,
		PackageCacheKey:       identity.PackageCacheKey,
		EmbeddingSpaceID:      identity.EmbeddingSpaceID,
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
		DocumentRoleApplied:   documentRoleApplied,
		QueryRoleApplied:      queryRoleApplied,
		RoleMode:              effectiveRoleMode,
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
	if cfg.RoleMode == "" {
		cfg.RoleMode = EmbeddingRoleModeAuto
	}
	return cfg
}

type nativeRetrievalEmbeddingSpaceIdentity struct {
	EmbeddingSpaceID      string `json:"-"`
	Schema                string `json:"schema"`
	ExecutionMode         string `json:"execution_mode"`
	ArtifactSHA256        string `json:"artifact_sha256,omitempty"`
	WeightsSHA256         string `json:"weights_sha256,omitempty"`
	TokenizerSHA256       string `json:"tokenizer_sha256,omitempty"`
	PackageManifestSHA256 string `json:"package_manifest_sha256,omitempty"`
	PackageCacheKey       string `json:"package_cache_key,omitempty"`
	ModelName             string `json:"model_name,omitempty"`
	ArchitectureVersion   string `json:"architecture_version,omitempty"`
	ManifestModelDim      int    `json:"manifest_model_dim,omitempty"`
	ManifestOutputDim     int    `json:"manifest_output_dim,omitempty"`
	NativeDim             int    `json:"native_dim"`
	RequestedOutputDim    int    `json:"requested_output_dim,omitempty"`
	EffectiveOutputDim    int    `json:"effective_output_dim"`
	TokenizerVocabSize    int    `json:"tokenizer_vocab_size,omitempty"`
	TokenizerMaxSequence  int    `json:"tokenizer_max_sequence,omitempty"`
	RoleConditioning      string `json:"role_conditioning,omitempty"`
	RawRoleIndex          int32  `json:"raw_role_index,omitempty"`
	QueryRoleIndex        int32  `json:"query_role_index,omitempty"`
	DocumentRoleIndex     int32  `json:"document_role_index,omitempty"`
	RoleMode              string `json:"role_mode"`
	DocumentRoleApplied   bool   `json:"document_role_applied"`
	QueryRoleApplied      bool   `json:"query_role_applied"`
	DocumentPrefix        string `json:"document_prefix"`
	QueryPrefix           string `json:"query_prefix"`
	DocumentChunkWords    int    `json:"document_chunk_words,omitempty"`
	DocumentChunkOverlap  int    `json:"document_chunk_overlap,omitempty"`
	DocumentChunkMinWords int    `json:"document_chunk_min_words,omitempty"`
}

func buildNativeRetrievalEmbeddingSpaceIdentity(model *EmbeddingModel, cfg RetrievalVectorExportConfig, effectiveOutputDim, nativeDim int, effectiveRoleMode string, documentRoleApplied, queryRoleApplied bool) (nativeRetrievalEmbeddingSpaceIdentity, error) {
	manifest := model.Manifest().normalized()
	identity := nativeRetrievalEmbeddingSpaceIdentity{
		Schema:                RetrievalVectorExportManifestSchema,
		ExecutionMode:         "native_eos_embedding",
		ModelName:             manifest.Name,
		ArchitectureVersion:   manifest.ArchitectureVersion,
		ManifestModelDim:      manifest.ModelDim,
		ManifestOutputDim:     manifest.OutputDim,
		NativeDim:             nativeDim,
		RequestedOutputDim:    cfg.OutputDim,
		EffectiveOutputDim:    effectiveOutputDim,
		TokenizerVocabSize:    manifest.Tokenizer.VocabSize,
		TokenizerMaxSequence:  manifest.Tokenizer.MaxSequence,
		RoleConditioning:      manifest.RoleConditioning,
		RawRoleIndex:          manifest.RawRoleIndex,
		QueryRoleIndex:        manifest.QueryRoleIndex,
		DocumentRoleIndex:     manifest.DocumentRoleIndex,
		RoleMode:              effectiveRoleMode,
		DocumentRoleApplied:   documentRoleApplied,
		QueryRoleApplied:      queryRoleApplied,
		DocumentPrefix:        cfg.DocumentPrefix,
		QueryPrefix:           cfg.QueryPrefix,
		DocumentChunkWords:    cfg.DocumentChunkWords,
		DocumentChunkOverlap:  cfg.DocumentChunkOverlap,
		DocumentChunkMinWords: cfg.DocumentChunkMinWords,
	}
	if cfg.ArtifactPath != "" {
		sum, err := optionalSHA256FileHex(cfg.ArtifactPath)
		if err != nil {
			return nativeRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash artifact: %w", err)
		}
		identity.ArtifactSHA256 = sum
		if identity.WeightsSHA256, err = optionalSHA256FileHex(DefaultWeightFilePath(cfg.ArtifactPath)); err != nil {
			return nativeRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash weights: %w", err)
		}
		if identity.TokenizerSHA256, err = optionalSHA256FileHex(DefaultTokenizerPath(cfg.ArtifactPath)); err != nil {
			return nativeRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash tokenizer: %w", err)
		}
		packageManifestPath := ResolvePackageManifestPath(cfg.ArtifactPath)
		if identity.PackageManifestSHA256, err = optionalSHA256FileHex(packageManifestPath); err != nil {
			return nativeRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash package manifest: %w", err)
		}
		if identity.PackageManifestSHA256 != "" {
			packageManifest, err := ReadPackageManifestFile(packageManifestPath)
			if err != nil {
				return nativeRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("read package manifest: %w", err)
			}
			identity.PackageCacheKey = packageManifest.CacheKey()
		}
	}
	data, err := json.Marshal(identity)
	if err != nil {
		return nativeRetrievalEmbeddingSpaceIdentity{}, err
	}
	sum := sha256.Sum256(data)
	identity.EmbeddingSpaceID = hex.EncodeToString(sum[:])
	return identity, nil
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

func writeRetrievalVectorCache(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord, path string, batchSize int, prefix string, outputDim int, role string) (int, int, error) {
	prefixed := prefixRetrievalRecords(records, prefix)
	vectors, err := embedRetrievalTexts(ctx, model, prefixed, batchSize, role)
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

func writeRetrievalChildVectorCache(ctx context.Context, model *EmbeddingModel, chunks []retrievalDocumentChunk, path string, batchSize int, prefix string, outputDim int, role string) (int, int, error) {
	records := make([]retrievalTextRecord, len(chunks))
	for i, chunk := range chunks {
		records[i] = retrievalTextRecord{ID: chunk.ChildID, Text: chunk.Text}
	}
	vectors, err := embedRetrievalTexts(ctx, model, prefixRetrievalRecords(records, prefix), batchSize, role)
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
