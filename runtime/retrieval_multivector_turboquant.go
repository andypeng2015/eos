package eosruntime

import (
	"container/heap"
	"context"
	"fmt"
	"time"

	"m31labs.dev/turboquant"
)

const TurboQuantMultiVectorRetrievalEvalMetricsSchema = "manta.embedding_turboquant_multivector_retrieval_metrics.v1"
const DefaultTurboQuantMultiVectorQuantizerSeed int64 = 0x4d756c7469566563

// TurboQuantMultiVectorRetrievalEvalMetrics compares parent-child dense
// max-child retrieval with direct TurboQuant child-vector scoring.
type TurboQuantMultiVectorRetrievalEvalMetrics struct {
	Schema        string                                      `json:"schema"`
	Dataset       string                                      `json:"dataset"`
	Artifact      string                                      `json:"artifact,omitempty"`
	Backend       string                                      `json:"backend,omitempty"`
	Inputs        TurboQuantMultiVectorRetrievalInputMetrics  `json:"inputs"`
	Config        TurboQuantMultiVectorRetrievalConfigMetrics `json:"config"`
	Dense         TurboQuantMultiVectorDenseMetrics           `json:"dense"`
	Rows          []TurboQuantMultiVectorBitMetrics           `json:"rows"`
	SkippedCounts RetrievalEvalSkippedCounts                  `json:"skipped_counts,omitempty"`
}

type TurboQuantMultiVectorRetrievalInputMetrics struct {
	CorpusPath               string  `json:"corpus_path,omitempty"`
	QueriesPath              string  `json:"queries_path,omitempty"`
	QrelsPath                string  `json:"qrels_path,omitempty"`
	DocVectorPath            string  `json:"doc_vector_path,omitempty"`
	QueryVectorPath          string  `json:"query_vector_path,omitempty"`
	Parents                  int     `json:"parents"`
	ParentCount              int     `json:"parent_count"`
	ChildVectors             int     `json:"child_vectors"`
	ChildCount               int     `json:"child_count"`
	AverageChildrenPerParent float64 `json:"average_children_per_parent"`
	AvgChildrenPerParent     float64 `json:"avg_children_per_parent"`
	MaxChildrenPerParent     int     `json:"max_children_per_parent"`
	Queries                  int     `json:"queries"`
	RelevantPairs            int     `json:"relevant_pairs"`
	ScoredChildPairs         int64   `json:"scored_child_pairs"`
}

type TurboQuantMultiVectorRetrievalConfigMetrics struct {
	TopK                 int   `json:"top_k"`
	MaxDocs              int   `json:"max_docs,omitempty"`
	MaxQueries           int   `json:"max_queries,omitempty"`
	Bits                 []int `json:"bits"`
	AllowMissingRelevant bool  `json:"allow_missing_relevant"`
	QuantizerSeed        int64 `json:"quantizer_seed"`
	BaselineDim          int   `json:"baseline_dim"`
}

type TurboQuantMultiVectorDenseMetrics struct {
	Quality                          RetrievalEvalQualityMetrics `json:"quality"`
	BaselineDim                      int                         `json:"baseline_dim"`
	ParentCount                      int                         `json:"parent_count"`
	ChildCount                       int                         `json:"child_count"`
	AvgChildrenPerParent             float64                     `json:"avg_children_per_parent"`
	MaxChildrenPerParent             int                         `json:"max_children_per_parent"`
	DenseBaselineBytes               int64                       `json:"dense_baseline_bytes"`
	DenseBaselineTotalBytes          int64                       `json:"dense_baseline_total_bytes"`
	ChildVectorBytes                 int64                       `json:"child_vector_bytes"`
	QuantizedVectorBytes             int64                       `json:"quantized_vector_bytes"`
	DenseParentBytes                 int64                       `json:"dense_parent_bytes"`
	DenseChildBytes                  int64                       `json:"dense_child_bytes"`
	VectorsThatFitInOneDenseBaseline int64                       `json:"vectors_that_fit_in_one_dense_baseline"`
	StorageMultipleOfDenseBaseline   float64                     `json:"storage_multiple_of_dense_baseline"`
	ParentBudgetStorageMultiple      float64                     `json:"parent_budget_storage_multiple"`
	ScoreSeconds                     float64                     `json:"score_seconds"`
	ScoresPerSecond                  float64                     `json:"scores_per_second"`
	QueryLatency                     RetrievalEvalLatencyMetrics `json:"query_latency"`
}

type TurboQuantMultiVectorBitMetrics struct {
	Bits                             int                         `json:"bits"`
	Method                           string                      `json:"method"`
	QuantizerSeed                    int64                       `json:"quantizer_seed"`
	Quality                          RetrievalEvalQualityMetrics `json:"quality"`
	NDCGAt10Delta                    float64                     `json:"ndcg_at_10_delta"`
	RecallAt100Delta                 float64                     `json:"recall_at_100_delta"`
	BaselineDim                      int                         `json:"baseline_dim"`
	ParentCount                      int                         `json:"parent_count"`
	ChildCount                       int                         `json:"child_count"`
	AvgChildrenPerParent             float64                     `json:"avg_children_per_parent"`
	MaxChildrenPerParent             int                         `json:"max_children_per_parent"`
	QuantizedVectorBytes             int64                       `json:"quantized_vector_bytes"`
	QuantizedChildBytes              int64                       `json:"quantized_child_bytes"`
	DenseBaselineBytes               int64                       `json:"dense_baseline_bytes"`
	DenseBaselineTotalBytes          int64                       `json:"dense_baseline_total_bytes"`
	DenseParentBytes                 int64                       `json:"dense_parent_bytes"`
	DenseChildBytes                  int64                       `json:"dense_child_bytes"`
	DenseChildCompression            float64                     `json:"dense_child_compression_ratio"`
	VectorsThatFitInOneDenseBaseline int64                       `json:"vectors_that_fit_in_one_dense_baseline"`
	StorageMultipleOfDenseBaseline   float64                     `json:"storage_multiple_of_dense_baseline"`
	ParentBudgetStorageMultiple      float64                     `json:"parent_budget_storage_multiple"`
	QuantizeSeconds                  float64                     `json:"quantize_seconds"`
	ScoreSeconds                     float64                     `json:"score_seconds"`
	ChildrenPerSecond                float64                     `json:"children_per_second"`
	ScoresPerSecond                  float64                     `json:"scores_per_second"`
	QueryLatency                     RetrievalEvalLatencyMetrics `json:"query_latency"`
	SkippedRelevantDocs              int                         `json:"skipped_relevant_docs,omitempty"`
	SkippedQueries                   int                         `json:"skipped_queries_without_relevant_docs,omitempty"`
}

// EvaluateTurboQuantMultiVectorCacheRetrieval evaluates precomputed child
// document vectors and query vectors against parent-document qrels. Every child
// is scored, scores are aggregated by max child score per parent, and parent IDs
// are evaluated against BEIR-style qrels.
func EvaluateTurboQuantMultiVectorCacheRetrieval(ctx context.Context, cfg RetrievalEvalConfig, bits []int) (TurboQuantMultiVectorRetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	if cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("document and query vector paths are required")
	}
	if cfg.BackendName == "" {
		cfg.BackendName = "vectors"
	}
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}

	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	childVectors, missingParents, docDim, err := readRetrievalChildVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("read document child vectors: %w", err)
	}
	if len(childVectors) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no child vectors for the evaluated corpus")
	}
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, retrievalIDs(queries))
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(queryVectors) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("query vector cache has no vectors for qrels queries")
	}
	if docDim != queryDim {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("document child vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(ctx, cfg, bits, childVectors, queryVectors, qrels)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
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
	metrics.SkippedCounts.DocumentsWithoutVector = missingParents
	return metrics, nil
}

func evaluateTurboQuantMultiVectorRetrieval(ctx context.Context, cfg RetrievalEvalConfig, bits []int, children []retrievalChildVectorRecord, queries []retrievalVectorRecord, qrels retrievalQrels) (TurboQuantMultiVectorRetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	if len(children) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child vectors are empty")
	}
	if len(queries) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("query vectors are empty")
	}
	if cfg.QuantizerSeed == 0 {
		cfg.QuantizerSeed = DefaultTurboQuantMultiVectorQuantizerSeed
	}
	dim := len(children[0].Vector)
	if dim == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child vector dimension is zero")
	}
	baselineDim := cfg.BaselineDim
	if baselineDim == 0 {
		baselineDim = dim
	}
	if baselineDim < 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("baseline dim must be positive or zero to use child vector dimension")
	}
	for _, child := range children {
		if child.ParentID == "" {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child vector has empty parent id")
		}
		if len(child.Vector) != dim {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("parent %q child %q vector dimension = %d, want %d", child.ParentID, child.ChildID, len(child.Vector), dim)
		}
	}
	for _, query := range queries {
		if len(query.Vector) != dim {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("query %q vector dimension = %d, want %d", query.ID, len(query.Vector), dim)
		}
	}

	parentCount := countRetrievalParents(children)
	maxChildrenPerParent := maxRetrievalChildrenPerParent(children)
	avgChildrenPerParent := ratioFloat64(float64(len(children)), float64(parentCount))
	if !cfg.AllowMissingRelevant {
		missingRelevantDocs, missingRelevantQueries := countMissingRelevantParents(queries, children, qrels)
		if missingRelevantDocs > 0 {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child-vector cache is missing %d qrels-relevant parent documents across %d queries; rerun with --allow-missing-relevant only for diagnostic smoke metrics", missingRelevantDocs, missingRelevantQueries)
		}
	}
	scoreStart := time.Now()
	denseQuality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, denseLatency := computeDenseMultiVectorRetrievalQuality(ctx, queries, children, qrels, cfg.TopK)
	denseScoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant parent documents in the evaluated vector cache")
	}

	scoredPairs := int64(evaluatedQueries) * int64(len(children))
	denseBaselineBytes := int64(baselineDim * 4)
	denseParentBytes := int64(parentCount) * denseBaselineBytes
	denseChildBytes := int64(len(children) * dim * 4)
	denseChildVectorBytes := int64(dim * 4)
	denseStorageMultiple := ratioFloat64(float64(denseChildBytes), float64(denseParentBytes))
	out := TurboQuantMultiVectorRetrievalEvalMetrics{
		Schema:  TurboQuantMultiVectorRetrievalEvalMetricsSchema,
		Dataset: cfg.DatasetName,
		Inputs: TurboQuantMultiVectorRetrievalInputMetrics{
			Parents:                  parentCount,
			ParentCount:              parentCount,
			ChildVectors:             len(children),
			ChildCount:               len(children),
			AverageChildrenPerParent: avgChildrenPerParent,
			AvgChildrenPerParent:     avgChildrenPerParent,
			MaxChildrenPerParent:     maxChildrenPerParent,
			Queries:                  evaluatedQueries,
			RelevantPairs:            relevantPairs,
			ScoredChildPairs:         scoredPairs,
		},
		Config: TurboQuantMultiVectorRetrievalConfigMetrics{
			TopK:                 cfg.TopK,
			MaxDocs:              cfg.MaxDocs,
			MaxQueries:           cfg.MaxQueries,
			Bits:                 append([]int(nil), bits...),
			AllowMissingRelevant: cfg.AllowMissingRelevant,
			QuantizerSeed:        cfg.QuantizerSeed,
			BaselineDim:          baselineDim,
		},
		Dense: TurboQuantMultiVectorDenseMetrics{
			Quality:                          denseQuality,
			BaselineDim:                      baselineDim,
			ParentCount:                      parentCount,
			ChildCount:                       len(children),
			AvgChildrenPerParent:             avgChildrenPerParent,
			MaxChildrenPerParent:             maxChildrenPerParent,
			DenseBaselineBytes:               denseBaselineBytes,
			DenseBaselineTotalBytes:          denseParentBytes,
			ChildVectorBytes:                 denseChildVectorBytes,
			QuantizedVectorBytes:             denseChildVectorBytes,
			DenseParentBytes:                 denseParentBytes,
			DenseChildBytes:                  denseChildBytes,
			VectorsThatFitInOneDenseBaseline: denseBaselineBytes / denseChildVectorBytes,
			StorageMultipleOfDenseBaseline:   denseStorageMultiple,
			ParentBudgetStorageMultiple:      denseStorageMultiple,
			ScoreSeconds:                     denseScoreDuration.Seconds(),
			ScoresPerSecond:                  ratePerSecond(float64(scoredPairs), denseScoreDuration),
			QueryLatency:                     denseLatency,
		},
		Rows: make([]TurboQuantMultiVectorBitMetrics, 0, len(bits)),
	}
	for _, bitWidth := range bits {
		row, err := evaluateTurboQuantMultiVectorBits(ctx, dim, baselineDim, bitWidth, cfg.TopK, cfg.QuantizerSeed, children, queries, qrels, denseQuality, parentCount, maxChildrenPerParent, denseBaselineBytes, denseParentBytes, denseChildBytes, scoredPairs)
		if err != nil {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
		}
		row.SkippedRelevantDocs = skippedRelevantDocs
		row.SkippedQueries = skippedNoRelevant
		out.Rows = append(out.Rows, row)
	}
	return out, nil
}

type turboQuantMultiVectorChild struct {
	ParentID string
	ChildID  string
	Vector   turboquant.IPQuantized
}

func evaluateTurboQuantMultiVectorBits(ctx context.Context, dim, baselineDim, bitWidth, topK int, quantizerSeed int64, children []retrievalChildVectorRecord, queries []retrievalVectorRecord, qrels retrievalQrels, denseQuality RetrievalEvalQualityMetrics, parentCount, maxChildrenPerParent int, denseBaselineBytes, denseParentBytes, denseChildBytes, scoredPairs int64) (TurboQuantMultiVectorBitMetrics, error) {
	q := turboquant.NewIPWithSeed(dim, bitWidth, quantizerSeed)
	quantizeStart := time.Now()
	qchildren := make([]turboQuantMultiVectorChild, len(children))
	mseBytes, signBytes := turboquant.IPQuantizedSizes(dim, bitWidth)
	quantizedVectorBytes := int64(mseBytes + signBytes + 4)
	var quantizedBytes int64
	for i, child := range children {
		if err := ctx.Err(); err != nil {
			return TurboQuantMultiVectorBitMetrics{}, err
		}
		qx := q.Quantize(child.Vector)
		qchildren[i] = turboQuantMultiVectorChild{ParentID: child.ParentID, ChildID: child.ChildID, Vector: qx}
		quantizedBytes += quantizedVectorBytes
	}
	quantizeDuration := time.Since(quantizeStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, _, skippedRelevantDocs, skippedNoRelevant, queryLatency := computeTurboQuantMultiVectorRetrievalQuality(ctx, q, queries, qchildren, qrels, topK)
	if evaluatedQueries == 0 {
		return TurboQuantMultiVectorBitMetrics{}, fmt.Errorf("no queries had relevant parent documents in the evaluated vector cache")
	}
	scoreDuration := time.Since(scoreStart)
	return TurboQuantMultiVectorBitMetrics{
		Bits:                             bitWidth,
		Method:                           fmt.Sprintf("turboquant_ip_b%d_child_max", bitWidth),
		QuantizerSeed:                    q.Seed(),
		Quality:                          quality,
		NDCGAt10Delta:                    quality.NDCGAt10 - denseQuality.NDCGAt10,
		RecallAt100Delta:                 quality.RecallAt100 - denseQuality.RecallAt100,
		BaselineDim:                      baselineDim,
		ParentCount:                      parentCount,
		ChildCount:                       len(children),
		AvgChildrenPerParent:             ratioFloat64(float64(len(children)), float64(parentCount)),
		MaxChildrenPerParent:             maxChildrenPerParent,
		QuantizedVectorBytes:             quantizedVectorBytes,
		QuantizedChildBytes:              quantizedBytes,
		DenseBaselineBytes:               denseBaselineBytes,
		DenseBaselineTotalBytes:          denseParentBytes,
		DenseParentBytes:                 denseParentBytes,
		DenseChildBytes:                  denseChildBytes,
		DenseChildCompression:            ratioFloat64(float64(denseChildBytes), float64(quantizedBytes)),
		VectorsThatFitInOneDenseBaseline: denseBaselineBytes / quantizedVectorBytes,
		StorageMultipleOfDenseBaseline:   ratioFloat64(float64(quantizedBytes), float64(denseParentBytes)),
		ParentBudgetStorageMultiple:      ratioFloat64(float64(quantizedBytes), float64(denseParentBytes)),
		QuantizeSeconds:                  quantizeDuration.Seconds(),
		ScoreSeconds:                     scoreDuration.Seconds(),
		ChildrenPerSecond:                ratePerSecond(float64(len(children)), quantizeDuration),
		ScoresPerSecond:                  ratePerSecond(float64(scoredPairs), scoreDuration),
		QueryLatency:                     queryLatency,
		SkippedRelevantDocs:              skippedRelevantDocs,
		SkippedQueries:                   skippedNoRelevant,
	}, nil
}

func countMissingRelevantParents(queries []retrievalVectorRecord, children []retrievalChildVectorRecord, qrels retrievalQrels) (int, int) {
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
	}
	missingDocs := 0
	missingQueries := 0
	for _, query := range queries {
		queryMissing := 0
		for parentID := range qrels[query.ID] {
			if !parentIDSet[parentID] {
				queryMissing++
			}
		}
		if queryMissing > 0 {
			missingQueries++
			missingDocs += queryMissing
		}
	}
	return missingDocs, missingQueries
}

func countRetrievalParents(children []retrievalChildVectorRecord) int {
	seen := make(map[string]bool, len(children))
	for _, child := range children {
		seen[child.ParentID] = true
	}
	return len(seen)
}

func maxRetrievalChildrenPerParent(children []retrievalChildVectorRecord) int {
	counts := make(map[string]int, len(children))
	maxCount := 0
	for _, child := range children {
		counts[child.ParentID]++
		if counts[child.ParentID] > maxCount {
			maxCount = counts[child.ParentID]
		}
	}
	return maxCount
}

func computeDenseMultiVectorRetrievalQuality(ctx context.Context, queries []retrievalVectorRecord, children []retrievalChildVectorRecord, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int, RetrievalEvalLatencyMetrics) {
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
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
		filteredRels := filterParentRels(rels, parentIDSet, &skippedRelevantDocs)
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		scores := topDenseMultiVectorParentScores(query.Vector, children, topK)
		latencies = append(latencies, time.Since(queryStart))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, scores, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, summarizeRetrievalEvalLatencies(latencies)
}

func computeTurboQuantMultiVectorRetrievalQuality(ctx context.Context, q *turboquant.IPQuantizer, queries []retrievalVectorRecord, children []turboQuantMultiVectorChild, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int, RetrievalEvalLatencyMetrics) {
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
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
		filteredRels := filterParentRels(rels, parentIDSet, &skippedRelevantDocs)
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		prepared := q.PrepareQuery(query.Vector)
		scores := topTurboQuantMultiVectorParentScores(q, prepared, children, topK)
		latencies = append(latencies, time.Since(queryStart))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, scores, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, summarizeRetrievalEvalLatencies(latencies)
}

func filterParentRels(rels map[string]float64, parentIDSet map[string]bool, skipped *int) map[string]float64 {
	filtered := make(map[string]float64, len(rels))
	for parentID, rel := range rels {
		if parentIDSet[parentID] {
			filtered[parentID] = rel
		} else {
			(*skipped)++
		}
	}
	return filtered
}

func topDenseMultiVectorParentScores(query []float32, children []retrievalChildVectorRecord, topK int) []retrievalScoredDoc {
	best := make(map[string]float32, len(children))
	for _, child := range children {
		score := dotRetrievalVectors(query, child.Vector)
		if prior, ok := best[child.ParentID]; !ok || score > prior {
			best[child.ParentID] = score
		}
	}
	return topParentScoresFromMap(best, topK)
}

func topTurboQuantMultiVectorParentScores(q *turboquant.IPQuantizer, prepared turboquant.PreparedQuery, children []turboQuantMultiVectorChild, topK int) []retrievalScoredDoc {
	best := make(map[string]float32, len(children))
	for _, child := range children {
		score := q.InnerProductPrepared(child.Vector, prepared)
		if prior, ok := best[child.ParentID]; !ok || score > prior {
			best[child.ParentID] = score
		}
	}
	return topParentScoresFromMap(best, topK)
}

func topParentScoresFromMap(best map[string]float32, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(best) {
		topK = len(best)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for id, scoreValue := range best {
		score := retrievalScoredDoc{ID: id, Score: scoreValue}
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
