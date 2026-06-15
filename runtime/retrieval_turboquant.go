package eosruntime

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"m31labs.dev/turboquant"
)

const TurboQuantRetrievalEvalMetricsSchema = "manta.embedding_turboquant_retrieval_metrics.v1"

const (
	TurboQuantRerankStorageDense              = "dense"
	TurboQuantRerankStorageCompactReconstruct = "compact-reconstruct"
	TurboQuantRerankStorageFP16               = "fp16"
)

// TurboQuantRetrievalEvalMetrics compares dense retrieval with TurboQuant
// inner-product scoring over the same embedded BEIR-style corpus.
type TurboQuantRetrievalEvalMetrics struct {
	Schema        string                               `json:"schema"`
	Dataset       string                               `json:"dataset"`
	Artifact      string                               `json:"artifact,omitempty"`
	Backend       string                               `json:"backend,omitempty"`
	Inputs        RetrievalEvalInputMetrics            `json:"inputs"`
	Config        TurboQuantRetrievalEvalConfigMetrics `json:"config"`
	Dense         TurboQuantDenseRetrievalMetrics      `json:"dense"`
	Rows          []TurboQuantRetrievalBitMetrics      `json:"rows"`
	SkippedCounts RetrievalEvalSkippedCounts           `json:"skipped_counts,omitempty"`
}

type TurboQuantRetrievalEvalConfigMetrics struct {
	BatchSize       int    `json:"batch_size"`
	TopK            int    `json:"top_k"`
	MaxDocs         int    `json:"max_docs,omitempty"`
	MaxQueries      int    `json:"max_queries,omitempty"`
	Bits            []int  `json:"bits"`
	RerankOverfetch []int  `json:"rerank_overfetch,omitempty"`
	RerankStorage   string `json:"rerank_storage,omitempty"`
}

type TurboQuantDenseRetrievalMetrics struct {
	Quality         RetrievalEvalQualityMetrics `json:"quality"`
	VectorBytes     int64                       `json:"vector_bytes"`
	ScoreSeconds    float64                     `json:"score_seconds"`
	ScoresPerSecond float64                     `json:"scores_per_second"`
	QueryLatency    RetrievalEvalLatencyMetrics `json:"query_latency"`
}

type TurboQuantRetrievalBitMetrics struct {
	Bits                int                         `json:"bits"`
	Method              string                      `json:"method"`
	RerankOverfetch     int                         `json:"rerank_overfetch,omitempty"`
	Quality             RetrievalEvalQualityMetrics `json:"quality"`
	NDCGAt10Delta       float64                     `json:"ndcg_at_10_delta"`
	RecallAt100Delta    float64                     `json:"recall_at_100_delta"`
	VectorBytes         int64                       `json:"vector_bytes"`
	DenseVectorBytes    int64                       `json:"dense_vector_bytes"`
	CompressionRatio    float64                     `json:"compression_ratio"`
	RerankStorage       string                      `json:"rerank_storage,omitempty"`
	RerankSidecarBytes  int64                       `json:"rerank_sidecar_bytes,omitempty"`
	TotalVectorBytes    int64                       `json:"total_vector_bytes,omitempty"`
	TotalCompression    float64                     `json:"total_compression_ratio,omitempty"`
	QuantizeSeconds     float64                     `json:"quantize_seconds"`
	ScoreSeconds        float64                     `json:"score_seconds"`
	RerankScoreSeconds  float64                     `json:"rerank_score_seconds,omitempty"`
	DocsPerSecond       float64                     `json:"docs_per_second"`
	ScoresPerSecond     float64                     `json:"scores_per_second"`
	QueryLatency        RetrievalEvalLatencyMetrics `json:"query_latency"`
	RerankScores        int64                       `json:"rerank_scores,omitempty"`
	SkippedRelevantDocs int                         `json:"skipped_relevant_docs,omitempty"`
	SkippedQueries      int                         `json:"skipped_queries_without_relevant_docs,omitempty"`
}

type RetrievalEvalLatencyMetrics struct {
	Count  int     `json:"count"`
	MinMS  float64 `json:"min_ms"`
	MeanMS float64 `json:"mean_ms"`
	P50MS  float64 `json:"p50_ms"`
	P95MS  float64 `json:"p95_ms"`
	P99MS  float64 `json:"p99_ms"`
	MaxMS  float64 `json:"max_ms"`
}

// EvaluateTurboQuantRetrieval embeds a BEIR-style split once, then evaluates
// dense float32 retrieval against TurboQuant IP-preserving document vectors.
func EvaluateTurboQuantRetrieval(ctx context.Context, model *EmbeddingModel, cfg RetrievalEvalConfig, bits []int) (TurboQuantRetrievalEvalMetrics, error) {
	return EvaluateTurboQuantRetrievalWithRerank(ctx, model, cfg, bits, nil)
}

// EvaluateTurboQuantRetrievalWithRerank embeds a BEIR-style split once, then
// evaluates dense retrieval, direct TurboQuant IP scoring, and optional
// TurboQuant-overfetch plus exact dense reranking rows.
func EvaluateTurboQuantRetrievalWithRerank(ctx context.Context, model *EmbeddingModel, cfg RetrievalEvalConfig, bits, rerankOverfetch []int) (TurboQuantRetrievalEvalMetrics, error) {
	return EvaluateTurboQuantRetrievalWithRerankStorage(ctx, model, cfg, bits, rerankOverfetch, TurboQuantRerankStorageDense)
}

// EvaluateTurboQuantRetrievalWithRerankStorage embeds a BEIR-style split once,
// then evaluates dense retrieval, direct TurboQuant IP scoring, and optional
// TurboQuant-overfetch reranking with the requested rerank storage.
func EvaluateTurboQuantRetrievalWithRerankStorage(ctx context.Context, model *EmbeddingModel, cfg RetrievalEvalConfig, bits, rerankOverfetch []int, rerankStorage string) (TurboQuantRetrievalEvalMetrics, error) {
	if model == nil {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	rerankOverfetch = normalizeTurboQuantRerankOverfetch(rerankOverfetch)
	var err error
	rerankStorage, err = normalizeTurboQuantRerankStorage(rerankStorage)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}

	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	docVectors, err := embedRetrievalTexts(ctx, model, corpus, cfg.BatchSize)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("embed corpus: %w", err)
	}
	queryVectors, err := embedRetrievalTexts(ctx, model, queries, cfg.BatchSize)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("embed queries: %w", err)
	}
	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(ctx, cfg, bits, rerankOverfetch, rerankStorage, docVectors, queryVectors, qrels)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	metrics.Schema = TurboQuantRetrievalEvalMetricsSchema
	metrics.Dataset = cfg.DatasetName
	metrics.Artifact = cfg.ArtifactPath
	metrics.Backend = string(model.Backend())
	metrics.Inputs.CorpusPath = cfg.CorpusPath
	metrics.Inputs.QueriesPath = cfg.QueriesPath
	metrics.Inputs.QrelsPath = cfg.QrelsPath
	metrics.SkippedCounts.QueriesWithoutText = skippedQueries
	return metrics, nil
}

// EvaluateTurboQuantVectorCacheRetrieval evaluates precomputed document/query
// vectors against BEIR-style qrels, then compares dense scoring with
// TurboQuant IP-preserving document-vector scoring.
func EvaluateTurboQuantVectorCacheRetrieval(ctx context.Context, cfg RetrievalEvalConfig, bits []int) (TurboQuantRetrievalEvalMetrics, error) {
	return EvaluateTurboQuantVectorCacheRetrievalWithRerank(ctx, cfg, bits, nil)
}

// EvaluateTurboQuantVectorCacheRetrievalWithRerank evaluates precomputed vectors
// with direct TurboQuant rows and optional exact dense reranking after
// TurboQuant overfetch.
func EvaluateTurboQuantVectorCacheRetrievalWithRerank(ctx context.Context, cfg RetrievalEvalConfig, bits, rerankOverfetch []int) (TurboQuantRetrievalEvalMetrics, error) {
	return EvaluateTurboQuantVectorCacheRetrievalWithRerankStorage(ctx, cfg, bits, rerankOverfetch, TurboQuantRerankStorageDense)
}

// EvaluateTurboQuantVectorCacheRetrievalWithRerankStorage evaluates precomputed
// vectors with direct TurboQuant rows and optional reranking after TurboQuant
// overfetch using the requested rerank storage.
func EvaluateTurboQuantVectorCacheRetrievalWithRerankStorage(ctx context.Context, cfg RetrievalEvalConfig, bits, rerankOverfetch []int, rerankStorage string) (TurboQuantRetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	if cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("document and query vector paths are required")
	}
	if cfg.BackendName == "" {
		cfg.BackendName = "vectors"
	}
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	rerankOverfetch = normalizeTurboQuantRerankOverfetch(rerankOverfetch)
	rerankStorage, err := normalizeTurboQuantRerankStorage(rerankStorage)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}

	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("read document vectors: %w", err)
	}
	if len(docVectors) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no vectors for the evaluated corpus")
	}
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, retrievalIDs(queries))
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(queryVectors) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("query vector cache has no vectors for qrels queries")
	}
	if docDim != queryDim {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(ctx, cfg, bits, rerankOverfetch, rerankStorage, docVectors, queryVectors, qrels)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	metrics.Artifact = cfg.ArtifactPath
	metrics.Backend = cfg.BackendName
	metrics.Inputs.CorpusPath = cfg.CorpusPath
	metrics.Inputs.QueriesPath = cfg.QueriesPath
	metrics.Inputs.QrelsPath = cfg.QrelsPath
	metrics.Inputs.DocVectorPath = cfg.DocVectorPath
	metrics.Inputs.QueryVectorPath = cfg.QueryVectorPath
	metrics.SkippedCounts.QueriesWithoutText = skippedQueries
	metrics.SkippedCounts.QueriesWithoutVector = missingQueryVectors
	metrics.SkippedCounts.DocumentsWithoutVector = missingDocVectors
	return metrics, nil
}

func evaluateTurboQuantVectorRetrieval(ctx context.Context, cfg RetrievalEvalConfig, bits []int, docs, queries []retrievalVectorRecord, qrels retrievalQrels) (TurboQuantRetrievalEvalMetrics, error) {
	return evaluateTurboQuantVectorRetrievalWithRerank(ctx, cfg, bits, nil, docs, queries, qrels)
}

func evaluateTurboQuantVectorRetrievalWithRerank(ctx context.Context, cfg RetrievalEvalConfig, bits, rerankOverfetch []int, docs, queries []retrievalVectorRecord, qrels retrievalQrels) (TurboQuantRetrievalEvalMetrics, error) {
	return evaluateTurboQuantVectorRetrievalWithRerankStorage(ctx, cfg, bits, rerankOverfetch, TurboQuantRerankStorageDense, docs, queries, qrels)
}

func evaluateTurboQuantVectorRetrievalWithRerankStorage(ctx context.Context, cfg RetrievalEvalConfig, bits, rerankOverfetch []int, rerankStorage string, docs, queries []retrievalVectorRecord, qrels retrievalQrels) (TurboQuantRetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	rerankOverfetch = normalizeTurboQuantRerankOverfetch(rerankOverfetch)
	rerankStorage, err := normalizeTurboQuantRerankStorage(rerankStorage)
	if err != nil {
		return TurboQuantRetrievalEvalMetrics{}, err
	}
	if len(docs) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("corpus vectors are empty")
	}
	if len(queries) == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("query vectors are empty")
	}
	dim := len(docs[0].Vector)
	if dim == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("document vector dimension is zero")
	}
	for _, doc := range docs {
		if len(doc.Vector) != dim {
			return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("document %q vector dimension = %d, want %d", doc.ID, len(doc.Vector), dim)
		}
	}
	for _, query := range queries {
		if len(query.Vector) != dim {
			return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("query %q vector dimension = %d, want %d", query.ID, len(query.Vector), dim)
		}
	}

	scoreStart := time.Now()
	denseQuality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, denseLatency := computeDenseRetrievalQualityWithLatency(queries, docs, qrels, cfg.TopK)
	denseScoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return TurboQuantRetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}

	scoredPairs := int64(evaluatedQueries) * int64(len(docs))
	denseVectorBytes := int64(len(docs) * dim * 4)
	out := TurboQuantRetrievalEvalMetrics{
		Schema:  TurboQuantRetrievalEvalMetricsSchema,
		Dataset: cfg.DatasetName,
		Inputs: RetrievalEvalInputMetrics{
			Documents:     len(docs),
			Queries:       evaluatedQueries,
			RelevantPairs: relevantPairs,
			ScoredPairs:   scoredPairs,
		},
		Config: TurboQuantRetrievalEvalConfigMetrics{
			BatchSize:       cfg.BatchSize,
			TopK:            cfg.TopK,
			MaxDocs:         cfg.MaxDocs,
			MaxQueries:      cfg.MaxQueries,
			Bits:            append([]int(nil), bits...),
			RerankOverfetch: append([]int(nil), rerankOverfetch...),
			RerankStorage:   rerankStorageMetricsValue(rerankOverfetch, rerankStorage),
		},
		Dense: TurboQuantDenseRetrievalMetrics{
			Quality:         denseQuality,
			VectorBytes:     denseVectorBytes,
			ScoreSeconds:    denseScoreDuration.Seconds(),
			ScoresPerSecond: ratePerSecond(float64(scoredPairs), denseScoreDuration),
			QueryLatency:    denseLatency,
		},
		Rows: make([]TurboQuantRetrievalBitMetrics, 0, len(bits)),
	}

	for _, bitWidth := range bits {
		if err := ctx.Err(); err != nil {
			return TurboQuantRetrievalEvalMetrics{}, err
		}
		rows, err := evaluateTurboQuantRetrievalBits(ctx, dim, bitWidth, cfg.TopK, rerankOverfetch, rerankStorage, docs, queries, qrels, denseQuality, denseVectorBytes, scoredPairs)
		if err != nil {
			return TurboQuantRetrievalEvalMetrics{}, err
		}
		for _, row := range rows {
			row.SkippedRelevantDocs = skippedRelevantDocs
			row.SkippedQueries = skippedNoRelevant
			out.Rows = append(out.Rows, row)
		}
	}
	return out, nil
}

func evaluateTurboQuantRetrievalBits(ctx context.Context, dim, bitWidth, topK int, rerankOverfetch []int, rerankStorage string, docs, queries []retrievalVectorRecord, qrels retrievalQrels, denseQuality RetrievalEvalQualityMetrics, denseVectorBytes, scoredPairs int64) ([]TurboQuantRetrievalBitMetrics, error) {
	q := turboquant.NewIP(dim, bitWidth)
	quantizeStart := time.Now()
	qdocs := make([]turboQuantRetrievalDoc, len(docs))
	var quantizedBytes int64
	for i, doc := range docs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		qx := q.Quantize(doc.Vector)
		qdocs[i] = turboQuantRetrievalDoc{ID: doc.ID, Vector: qx}
		quantizedBytes += int64(len(qx.MSE) + len(qx.Signs) + 4) // ResNorm is a float32 sidecar.
	}
	quantizeDuration := time.Since(quantizeStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, _, skippedRelevantDocs, skippedNoRelevant, queryLatency := computeTurboQuantRetrievalQuality(ctx, q, queries, qdocs, qrels, topK)
	if evaluatedQueries == 0 {
		return nil, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}
	scoreDuration := time.Since(scoreStart)
	rows := []TurboQuantRetrievalBitMetrics{{
		Bits:                bitWidth,
		Method:              fmt.Sprintf("turboquant_ip_b%d", bitWidth),
		Quality:             quality,
		NDCGAt10Delta:       quality.NDCGAt10 - denseQuality.NDCGAt10,
		RecallAt100Delta:    quality.RecallAt100 - denseQuality.RecallAt100,
		VectorBytes:         quantizedBytes,
		DenseVectorBytes:    denseVectorBytes,
		CompressionRatio:    ratioFloat64(float64(denseVectorBytes), float64(quantizedBytes)),
		TotalVectorBytes:    quantizedBytes,
		TotalCompression:    ratioFloat64(float64(denseVectorBytes), float64(quantizedBytes)),
		QuantizeSeconds:     quantizeDuration.Seconds(),
		ScoreSeconds:        scoreDuration.Seconds(),
		DocsPerSecond:       ratePerSecond(float64(len(docs)), quantizeDuration),
		ScoresPerSecond:     ratePerSecond(float64(scoredPairs), scoreDuration),
		QueryLatency:        queryLatency,
		SkippedRelevantDocs: skippedRelevantDocs,
		SkippedQueries:      skippedNoRelevant,
	}}
	for _, overfetch := range rerankOverfetch {
		finalTopK := topK
		if finalTopK < 100 {
			finalTopK = 100
		}
		if overfetch <= finalTopK || overfetch > len(docs) {
			continue
		}
		rerankStart := time.Now()
		var rerankQuality RetrievalEvalQualityMetrics
		var evaluatedQueries, skippedRelevantDocs, skippedNoRelevant int
		var rerankScores int64
		var rerankLatency RetrievalEvalLatencyMetrics
		var method string
		var rerankSidecarBytes int64
		switch rerankStorage {
		case TurboQuantRerankStorageDense:
			rerankQuality, evaluatedQueries, _, skippedRelevantDocs, skippedNoRelevant, rerankScores, rerankLatency = computeTurboQuantDenseRerankRetrievalQuality(ctx, q, queries, docs, qdocs, qrels, topK, overfetch)
			method = fmt.Sprintf("turboquant_ip_b%d_overfetch%d_dense_rerank", bitWidth, overfetch)
			rerankSidecarBytes = denseVectorBytes
		case TurboQuantRerankStorageCompactReconstruct:
			rerankQuality, evaluatedQueries, _, skippedRelevantDocs, skippedNoRelevant, rerankScores, rerankLatency = computeTurboQuantReconstructRerankRetrievalQuality(ctx, q, queries, qdocs, qrels, topK, overfetch)
			method = fmt.Sprintf("turboquant_ip_b%d_overfetch%d_reconstruct_rerank", bitWidth, overfetch)
		case TurboQuantRerankStorageFP16:
			rerankQuality, evaluatedQueries, _, skippedRelevantDocs, skippedNoRelevant, rerankScores, rerankLatency = computeTurboQuantFP16RerankRetrievalQuality(ctx, q, queries, docs, qdocs, qrels, topK, overfetch)
			method = fmt.Sprintf("turboquant_ip_b%d_overfetch%d_fp16_rerank", bitWidth, overfetch)
			rerankSidecarBytes = int64(len(docs) * dim * 2)
		default:
			return nil, fmt.Errorf("unsupported TurboQuant rerank storage %q", rerankStorage)
		}
		if evaluatedQueries == 0 {
			return nil, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
		}
		rerankDuration := time.Since(rerankStart)
		totalVectorBytes := quantizedBytes + rerankSidecarBytes
		rows = append(rows, TurboQuantRetrievalBitMetrics{
			Bits:                bitWidth,
			Method:              method,
			RerankOverfetch:     overfetch,
			Quality:             rerankQuality,
			NDCGAt10Delta:       rerankQuality.NDCGAt10 - denseQuality.NDCGAt10,
			RecallAt100Delta:    rerankQuality.RecallAt100 - denseQuality.RecallAt100,
			VectorBytes:         quantizedBytes,
			DenseVectorBytes:    denseVectorBytes,
			CompressionRatio:    ratioFloat64(float64(denseVectorBytes), float64(quantizedBytes)),
			RerankStorage:       rerankStorage,
			RerankSidecarBytes:  rerankSidecarBytes,
			TotalVectorBytes:    totalVectorBytes,
			TotalCompression:    ratioFloat64(float64(denseVectorBytes), float64(totalVectorBytes)),
			QuantizeSeconds:     quantizeDuration.Seconds(),
			ScoreSeconds:        scoreDuration.Seconds() + rerankDuration.Seconds(),
			RerankScoreSeconds:  rerankDuration.Seconds(),
			DocsPerSecond:       ratePerSecond(float64(len(docs)), quantizeDuration),
			ScoresPerSecond:     ratePerSecond(float64(scoredPairs+rerankScores), scoreDuration+rerankDuration),
			QueryLatency:        rerankLatency,
			RerankScores:        rerankScores,
			SkippedRelevantDocs: skippedRelevantDocs,
			SkippedQueries:      skippedNoRelevant,
		})
	}
	return rows, nil
}

type turboQuantRetrievalDoc struct {
	ID     string
	Vector turboquant.IPQuantized
}

func computeDenseRetrievalQualityWithLatency(queries, docs []retrievalVectorRecord, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int, RetrievalEvalLatencyMetrics) {
	docIDSet := make(map[string]bool, len(docs))
	for _, doc := range docs {
		docIDSet[doc.ID] = true
	}
	if topK < 100 {
		topK = 100
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		scores := topRetrievalScores(query.Vector, docs, topK)
		latencies = append(latencies, time.Since(queryStart))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, scores, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, summarizeRetrievalEvalLatencies(latencies)
}

func computeTurboQuantRetrievalQuality(ctx context.Context, q *turboquant.IPQuantizer, queries []retrievalVectorRecord, docs []turboQuantRetrievalDoc, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int, RetrievalEvalLatencyMetrics) {
	docIDSet := make(map[string]bool, len(docs))
	for _, doc := range docs {
		docIDSet[doc.ID] = true
	}
	if topK < 100 {
		topK = 100
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		prepared := q.PrepareQuery(query.Vector)
		scores := topTurboQuantRetrievalScores(q, prepared, docs, topK)
		latencies = append(latencies, time.Since(queryStart))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, scores, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, summarizeRetrievalEvalLatencies(latencies)
}

func computeTurboQuantDenseRerankRetrievalQuality(ctx context.Context, q *turboquant.IPQuantizer, queries []retrievalVectorRecord, denseDocs []retrievalVectorRecord, qdocs []turboQuantRetrievalDoc, qrels retrievalQrels, topK, overfetchK int) (RetrievalEvalQualityMetrics, int, int, int, int, int64, RetrievalEvalLatencyMetrics) {
	docIDSet := make(map[string]bool, len(denseDocs))
	denseByID := make(map[string][]float32, len(denseDocs))
	for _, doc := range denseDocs {
		docIDSet[doc.ID] = true
		denseByID[doc.ID] = doc.Vector
	}
	if topK < 100 {
		topK = 100
	}
	if overfetchK < topK {
		overfetchK = topK
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	var rerankScores int64
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		prepared := q.PrepareQuery(query.Vector)
		candidates := topTurboQuantRetrievalScores(q, prepared, qdocs, overfetchK)
		reranked := topDenseRerankScores(query.Vector, candidates, denseByID, topK)
		latencies = append(latencies, time.Since(queryStart))
		rerankScores += int64(len(candidates))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, reranked, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, rerankScores, summarizeRetrievalEvalLatencies(latencies)
}

func computeTurboQuantReconstructRerankRetrievalQuality(ctx context.Context, q *turboquant.IPQuantizer, queries []retrievalVectorRecord, qdocs []turboQuantRetrievalDoc, qrels retrievalQrels, topK, overfetchK int) (RetrievalEvalQualityMetrics, int, int, int, int, int64, RetrievalEvalLatencyMetrics) {
	docIDSet := make(map[string]bool, len(qdocs))
	quantizedByID := make(map[string]turboquant.IPQuantized, len(qdocs))
	for _, doc := range qdocs {
		docIDSet[doc.ID] = true
		quantizedByID[doc.ID] = doc.Vector
	}
	if topK < 100 {
		topK = 100
	}
	if overfetchK < topK {
		overfetchK = topK
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	var rerankScores int64
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		prepared := q.PrepareQuery(query.Vector)
		candidates := topTurboQuantRetrievalScores(q, prepared, qdocs, overfetchK)
		reranked := topTurboQuantReconstructRerankScores(q, query.Vector, candidates, quantizedByID, topK)
		latencies = append(latencies, time.Since(queryStart))
		rerankScores += int64(len(candidates))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, reranked, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, rerankScores, summarizeRetrievalEvalLatencies(latencies)
}

func computeTurboQuantFP16RerankRetrievalQuality(ctx context.Context, q *turboquant.IPQuantizer, queries []retrievalVectorRecord, denseDocs []retrievalVectorRecord, qdocs []turboQuantRetrievalDoc, qrels retrievalQrels, topK, overfetchK int) (RetrievalEvalQualityMetrics, int, int, int, int, int64, RetrievalEvalLatencyMetrics) {
	docIDSet := make(map[string]bool, len(denseDocs))
	halfByID := make(map[string][]uint16, len(denseDocs))
	for _, doc := range denseDocs {
		docIDSet[doc.ID] = true
		halfVec := make([]uint16, len(doc.Vector))
		for i, value := range doc.Vector {
			halfVec[i] = float32ToHalf(value)
		}
		halfByID[doc.ID] = halfVec
	}
	if topK < 100 {
		topK = 100
	}
	if overfetchK < topK {
		overfetchK = topK
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	var rerankScores int64
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		prepared := q.PrepareQuery(query.Vector)
		candidates := topTurboQuantRetrievalScores(q, prepared, qdocs, overfetchK)
		reranked := topFP16RerankScores(query.Vector, candidates, halfByID, topK)
		latencies = append(latencies, time.Since(queryStart))
		rerankScores += int64(len(candidates))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, reranked, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, rerankScores, summarizeRetrievalEvalLatencies(latencies)
}

func summarizeRetrievalEvalLatencies(latencies []time.Duration) RetrievalEvalLatencyMetrics {
	if len(latencies) == 0 {
		return RetrievalEvalLatencyMetrics{}
	}
	values := append([]time.Duration(nil), latencies...)
	sort.Slice(values, func(i, j int) bool {
		return values[i] < values[j]
	})
	var total time.Duration
	for _, value := range values {
		total += value
	}
	return RetrievalEvalLatencyMetrics{
		Count:  len(values),
		MinMS:  durationMilliseconds(values[0]),
		MeanMS: durationMilliseconds(total) / float64(len(values)),
		P50MS:  durationMilliseconds(percentileDuration(values, 0.50)),
		P95MS:  durationMilliseconds(percentileDuration(values, 0.95)),
		P99MS:  durationMilliseconds(percentileDuration(values, 0.99)),
		MaxMS:  durationMilliseconds(values[len(values)-1]),
	}
}

func percentileDuration(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	index := int(math.Ceil(p*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func durationMilliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func topDenseRerankScores(query []float32, candidates []retrievalScoredDoc, docs map[string][]float32, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(candidates) {
		topK = len(candidates)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, candidate := range candidates {
		vec, ok := docs[candidate.ID]
		if !ok {
			continue
		}
		score := retrievalScoredDoc{ID: candidate.ID, Score: dotRetrievalVectors(query, vec)}
		if len(h) < topK {
			heap.Push(&h, score)
			continue
		}
		if retrievalScoreBetter(score, h[0]) {
			h[0] = score
			heap.Fix(&h, 0)
		}
	}
	scores := []retrievalScoredDoc(h)
	slicesSortRetrievalScores(scores)
	return scores
}

func topFP16RerankScores(query []float32, candidates []retrievalScoredDoc, docs map[string][]uint16, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(candidates) {
		topK = len(candidates)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, candidate := range candidates {
		vec, ok := docs[candidate.ID]
		if !ok {
			continue
		}
		var scoreValue float32
		for i, half := range vec {
			scoreValue += query[i] * halfToFloat32(half)
		}
		score := retrievalScoredDoc{ID: candidate.ID, Score: scoreValue}
		if len(h) < topK {
			heap.Push(&h, score)
			continue
		}
		if retrievalScoreBetter(score, h[0]) {
			h[0] = score
			heap.Fix(&h, 0)
		}
	}
	scores := []retrievalScoredDoc(h)
	slicesSortRetrievalScores(scores)
	return scores
}

func topTurboQuantReconstructRerankScores(q *turboquant.IPQuantizer, query []float32, candidates []retrievalScoredDoc, docs map[string]turboquant.IPQuantized, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(candidates) {
		topK = len(candidates)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, candidate := range candidates {
		qx, ok := docs[candidate.ID]
		if !ok {
			continue
		}
		score := retrievalScoredDoc{ID: candidate.ID, Score: dotRetrievalVectors(query, q.Dequantize(qx))}
		if len(h) < topK {
			heap.Push(&h, score)
			continue
		}
		if retrievalScoreBetter(score, h[0]) {
			h[0] = score
			heap.Fix(&h, 0)
		}
	}
	scores := []retrievalScoredDoc(h)
	slicesSortRetrievalScores(scores)
	return scores
}

func topTurboQuantRetrievalScores(q *turboquant.IPQuantizer, prepared turboquant.PreparedQuery, docs []turboQuantRetrievalDoc, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, doc := range docs {
		score := retrievalScoredDoc{ID: doc.ID, Score: q.InnerProductPrepared(doc.Vector, prepared)}
		if len(h) < topK {
			heap.Push(&h, score)
			continue
		}
		if retrievalScoreBetter(score, h[0]) {
			h[0] = score
			heap.Fix(&h, 0)
		}
	}
	scores := []retrievalScoredDoc(h)
	slicesSortRetrievalScores(scores)
	return scores
}

func slicesSortRetrievalScores(scores []retrievalScoredDoc) {
	for i := 1; i < len(scores); i++ {
		for j := i; j > 0 && retrievalScoreBetter(scores[j], scores[j-1]); j-- {
			scores[j], scores[j-1] = scores[j-1], scores[j]
		}
	}
}

func normalizeTurboQuantRetrievalBits(bits []int) []int {
	if len(bits) == 0 {
		return []int{2, 4, 8}
	}
	seen := map[int]bool{}
	out := make([]int, 0, len(bits))
	for _, bit := range bits {
		if seen[bit] {
			continue
		}
		seen[bit] = true
		out = append(out, bit)
	}
	return out
}

func normalizeTurboQuantRerankOverfetch(overfetch []int) []int {
	seen := map[int]bool{}
	out := make([]int, 0, len(overfetch))
	for _, k := range overfetch {
		if k <= 0 || seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, k)
	}
	return out
}

func normalizeTurboQuantRerankStorage(storage string) (string, error) {
	switch storage {
	case "", TurboQuantRerankStorageDense, "dense-f32", "exact", "exact-dense":
		return TurboQuantRerankStorageDense, nil
	case TurboQuantRerankStorageCompactReconstruct, "compact", "reconstruct", "turboquant-reconstruct":
		return TurboQuantRerankStorageCompactReconstruct, nil
	case TurboQuantRerankStorageFP16, "half", "f16", "dense-f16":
		return TurboQuantRerankStorageFP16, nil
	default:
		return "", fmt.Errorf("unsupported TurboQuant rerank storage %q; use %q, %q, or %q", storage, TurboQuantRerankStorageDense, TurboQuantRerankStorageCompactReconstruct, TurboQuantRerankStorageFP16)
	}
}

func rerankStorageMetricsValue(overfetch []int, storage string) string {
	if len(overfetch) == 0 {
		return ""
	}
	return storage
}

func validateTurboQuantRetrievalBits(bits []int) error {
	for _, bit := range bits {
		if bit < 2 || bit > 8 {
			return fmt.Errorf("TurboQuant IP bit width %d is unsupported; use 2..8", bit)
		}
	}
	return nil
}

func ratioFloat64(num, denom float64) float64 {
	if denom == 0 {
		return 0
	}
	return num / denom
}
