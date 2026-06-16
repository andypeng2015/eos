package eosruntime

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"
)

const (
	defaultHybridMethod    = "minmax"
	defaultHybridAlpha     = 0.75
	defaultHybridRRFK      = 60
	defaultHybridRRFLambda = 1.0
)

// EvaluateHybridRetrieval evaluates dense embedding retrieval fused with BM25 over
// each query's dense/BM25 top-k union.
func EvaluateHybridRetrieval(ctx context.Context, model *EmbeddingModel, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	if model == nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeRetrievalEvalConfig(cfg)
	hybridCfg, err := normalizeRetrievalEvalHybridConfig(cfg.Hybrid)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	start := time.Now()
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	docStart := time.Now()
	docVectors, err := embedRetrievalTexts(ctx, model, corpus, cfg.BatchSize)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embed corpus: %w", err)
	}
	index, err := buildBM25Index(ctx, corpus)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("build BM25 index: %w", err)
	}
	docDuration := time.Since(docStart)

	queryStart := time.Now()
	queryVectors, err := embedRetrievalTexts(ctx, model, queries, cfg.BatchSize)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embed queries: %w", err)
	}
	tokenizedQueries, err := tokenizeBM25Queries(ctx, queries)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("tokenize BM25 queries: %w", err)
	}
	queryDuration := time.Since(queryStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeHybridRetrievalQuality(ctx, queryVectors, docVectors, tokenizedQueriesByID(tokenizedQueries), index, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath, hybridCfg)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docVectors))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  "hybrid",
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:    cfg.CorpusPath,
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			Documents:     len(docVectors),
			Queries:       evaluatedQueries,
			RelevantPairs: relevantPairs,
			ScoredPairs:   scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			BatchSize:  cfg.BatchSize,
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
			Hybrid:     retrievalEvalHybridMetrics(hybridCfg),
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: docDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docVectors)), docDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(queryVectors)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
		},
	}, nil
}

// EvaluateVectorCacheHybridRetrieval evaluates precomputed dense vectors fused
// with BM25 text from the same BEIR corpus/query/qrels files.
func EvaluateVectorCacheHybridRetrieval(ctx context.Context, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	hybridCfg, err := normalizeRetrievalEvalHybridConfig(cfg.Hybrid)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	if cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("document and query vector paths are required")
	}
	if cfg.BackendName == "" {
		cfg.BackendName = "vectors-hybrid"
	}
	start := time.Now()
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	docStart := time.Now()
	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read document vectors: %w", err)
	}
	if len(docVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no vectors for the evaluated corpus")
	}
	vectorCorpus := filterRetrievalTextsByVectorOrder(corpus, docVectors)
	index, err := buildBM25Index(ctx, vectorCorpus)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("build BM25 index: %w", err)
	}
	docDuration := time.Since(docStart)

	queryStart := time.Now()
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, retrievalIDs(queries))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(queryVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("query vector cache has no vectors for qrels queries")
	}
	if docDim != queryDim {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}
	vectorQueries := filterRetrievalTextsByVectorOrder(queries, queryVectors)
	tokenizedQueries, err := tokenizeBM25Queries(ctx, vectorQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("tokenize BM25 queries: %w", err)
	}
	queryDuration := time.Since(queryStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeHybridRetrievalQuality(ctx, queryVectors, docVectors, tokenizedQueriesByID(tokenizedQueries), index, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath, hybridCfg)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated vector cache")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docVectors))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  cfg.BackendName,
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:      cfg.CorpusPath,
			QueriesPath:     cfg.QueriesPath,
			QrelsPath:       cfg.QrelsPath,
			DocVectorPath:   cfg.DocVectorPath,
			QueryVectorPath: cfg.QueryVectorPath,
			Documents:       len(docVectors),
			Queries:         evaluatedQueries,
			RelevantPairs:   relevantPairs,
			ScoredPairs:     scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			BatchSize:  cfg.BatchSize,
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
			Hybrid:     retrievalEvalHybridMetrics(hybridCfg),
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: docDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docVectors)), docDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(queryVectors)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
			QueriesWithoutVector:       missingQueryVectors,
			DocumentsWithoutVector:     missingDocVectors,
		},
	}, nil
}

func normalizeRetrievalEvalHybridConfig(cfg RetrievalEvalHybridConfig) (RetrievalEvalHybridConfig, error) {
	method := strings.ToLower(strings.TrimSpace(cfg.Method))
	if method == "" {
		method = defaultHybridMethod
	}
	switch method {
	case "minmax":
		method = "minmax_blend"
	case "zscore":
		method = "zscore_blend"
	}
	switch method {
	case "minmax_blend", "zscore_blend", "rrf":
	default:
		return RetrievalEvalHybridConfig{}, fmt.Errorf("unknown hybrid method %q", cfg.Method)
	}
	cfg.Method = method
	if !cfg.AlphaSet && cfg.Alpha == 0 {
		cfg.Alpha = defaultHybridAlpha
	}
	if cfg.Alpha < 0 || cfg.Alpha > 1 {
		return RetrievalEvalHybridConfig{}, fmt.Errorf("hybrid alpha must be in [0,1], got %g", cfg.Alpha)
	}
	if cfg.RRFK == 0 {
		cfg.RRFK = defaultHybridRRFK
	}
	if cfg.RRFK <= 0 {
		return RetrievalEvalHybridConfig{}, fmt.Errorf("hybrid rrf-k must be positive, got %g", cfg.RRFK)
	}
	if cfg.RRFLambda == 0 {
		cfg.RRFLambda = defaultHybridRRFLambda
	}
	if cfg.RRFLambda < 0 {
		return RetrievalEvalHybridConfig{}, fmt.Errorf("hybrid rrf-lambda must be non-negative, got %g", cfg.RRFLambda)
	}
	if cfg.DenseProtectTopK < 0 {
		return RetrievalEvalHybridConfig{}, fmt.Errorf("hybrid dense-protect-top-k must be non-negative, got %d", cfg.DenseProtectTopK)
	}
	return cfg, nil
}

func retrievalEvalHybridMetrics(cfg RetrievalEvalHybridConfig) *RetrievalEvalHybridMetrics {
	return &RetrievalEvalHybridMetrics{
		Method:           cfg.Method,
		Alpha:            cfg.Alpha,
		RRFK:             cfg.RRFK,
		RRFLambda:        cfg.RRFLambda,
		DenseProtectTopK: cfg.DenseProtectTopK,
	}
}

func tokenizedQueriesByID(queries []bm25Query) map[string][]string {
	out := make(map[string][]string, len(queries))
	for _, query := range queries {
		out[query.ID] = query.Tokens
	}
	return out
}

func filterRetrievalTextsByVectorOrder(records []retrievalTextRecord, vectors []retrievalVectorRecord) []retrievalTextRecord {
	byID := make(map[string]string, len(records))
	for _, record := range records {
		byID[record.ID] = record.Text
	}
	out := make([]retrievalTextRecord, 0, len(vectors))
	for _, vector := range vectors {
		if text := byID[vector.ID]; text != "" {
			out = append(out, retrievalTextRecord{ID: vector.ID, Text: text})
		}
	}
	return out
}

func computeHybridRetrievalQuality(ctx context.Context, queries, docs []retrievalVectorRecord, bm25Queries map[string][]string, index bm25Index, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string, cfg RetrievalEvalHybridConfig) (RetrievalEvalQualityMetrics, int, int, int, int, error) {
	docIDSet := make(map[string]bool, len(docs))
	for _, doc := range docs {
		docIDSet[doc.ID] = true
	}
	if topK < 100 {
		topK = 100
	}
	writer, err := newRetrievalPerQueryWriter(perQueryJSONLPath)
	if err != nil {
		return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
	}
	defer writer.Close()
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
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
		denseScores := topRetrievalScores(query.Vector, docs, topK)
		bm25Scores := topBM25Scores(bm25Queries[query.ID], index, topK)
		scores := fuseHybridScores(denseScores, bm25Scores, topK, cfg)
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		row := buildRetrievalPerQueryRow(datasetName, query.ID, scores, filteredRels)
		addRetrievalPerQueryQuality(&totals, row.Quality)
		if err := writer.Write(row); err != nil {
			return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
		}
	}
	if err := writer.Close(); err != nil {
		return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, nil
}

func fuseHybridScores(denseScores, bm25Scores []retrievalScoredDoc, topK int, cfg RetrievalEvalHybridConfig) []retrievalScoredDoc {
	if topK <= 0 {
		topK = 100
	}
	denseRank, denseRaw := retrievalScoresByID(denseScores)
	bm25Rank, bm25Raw := retrievalScoresByID(bm25Scores)
	candidateSet := map[string]bool{}
	candidates := []string{}
	addCandidate := func(id string) {
		if id == "" || candidateSet[id] {
			return
		}
		candidateSet[id] = true
		candidates = append(candidates, id)
	}
	for _, score := range denseScores {
		addCandidate(score.ID)
	}
	for _, score := range bm25Scores {
		addCandidate(score.ID)
	}

	denseNorm := map[string]float64{}
	bm25Norm := map[string]float64{}
	if cfg.Method == "minmax_blend" {
		denseNorm = minmaxHybridScores(denseRaw)
		bm25Norm = minmaxHybridScores(bm25Raw)
	} else if cfg.Method == "zscore_blend" {
		denseNorm = zscoreHybridScores(denseRaw)
		bm25Norm = zscoreHybridScores(bm25Raw)
	}

	fused := make([]retrievalScoredDoc, 0, len(candidates))
	for _, id := range candidates {
		score := 0.0
		switch cfg.Method {
		case "rrf":
			if rank, ok := denseRank[id]; ok {
				score += 1 / (cfg.RRFK + float64(rank))
			}
			if rank, ok := bm25Rank[id]; ok {
				score += cfg.RRFLambda / (cfg.RRFK + float64(rank))
			}
		case "minmax_blend", "zscore_blend":
			score = (1-cfg.Alpha)*denseNorm[id] + cfg.Alpha*bm25Norm[id]
		}
		fused = append(fused, retrievalScoredDoc{ID: id, Score: float32(score)})
	}
	const missingRank = int(^uint(0)>>1) / 4
	slices.SortFunc(fused, func(a, b retrievalScoredDoc) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		adr := denseRankOrDefault(denseRank, a.ID, missingRank)
		bdr := denseRankOrDefault(denseRank, b.ID, missingRank)
		if adr != bdr {
			return adr - bdr
		}
		abr := denseRankOrDefault(bm25Rank, a.ID, missingRank)
		bbr := denseRankOrDefault(bm25Rank, b.ID, missingRank)
		if abr != bbr {
			return abr - bbr
		}
		return strings.Compare(a.ID, b.ID)
	})
	if cfg.DenseProtectTopK > 0 {
		fused = protectDensePrefix(fused, denseScores, cfg.DenseProtectTopK, topK)
	}
	if len(fused) > topK {
		return fused[:topK]
	}
	return fused
}

func protectDensePrefix(fused, denseScores []retrievalScoredDoc, protectTopK, topK int) []retrievalScoredDoc {
	if protectTopK <= 0 || len(fused) == 0 {
		return fused
	}
	if topK <= 0 {
		topK = 100
	}
	fusedByID := make(map[string]retrievalScoredDoc, len(fused))
	for _, score := range fused {
		fusedByID[score.ID] = score
	}
	protected := make(map[string]bool, min(protectTopK, len(denseScores)))
	out := make([]retrievalScoredDoc, 0, min(topK, len(fused)))
	for _, dense := range denseScores {
		if len(out) >= protectTopK || len(out) >= topK {
			break
		}
		score, ok := fusedByID[dense.ID]
		if !ok || protected[dense.ID] {
			continue
		}
		out = append(out, score)
		protected[dense.ID] = true
	}
	for _, score := range fused {
		if len(out) >= topK {
			break
		}
		if protected[score.ID] {
			continue
		}
		out = append(out, score)
	}
	return out
}

func retrievalScoresByID(scores []retrievalScoredDoc) (map[string]int, map[string]float64) {
	ranks := make(map[string]int, len(scores))
	values := make(map[string]float64, len(scores))
	for i, score := range scores {
		if _, exists := ranks[score.ID]; exists {
			continue
		}
		ranks[score.ID] = i + 1
		values[score.ID] = float64(score.Score)
	}
	return ranks, values
}

func denseRankOrDefault(ranks map[string]int, id string, fallback int) int {
	if rank, ok := ranks[id]; ok {
		return rank
	}
	return fallback
}

func minmaxHybridScores(values map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(values))
	if len(values) == 0 {
		return out
	}
	lo := math.Inf(1)
	hi := math.Inf(-1)
	for _, value := range values {
		lo = min(lo, value)
		hi = max(hi, value)
	}
	if hi == lo {
		for id := range values {
			out[id] = 1
		}
		return out
	}
	for id, value := range values {
		out[id] = (value - lo) / (hi - lo)
	}
	return out
}

func zscoreHybridScores(values map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(values))
	if len(values) == 0 {
		return out
	}
	var sum float64
	for _, value := range values {
		sum += value
	}
	mean := sum / float64(len(values))
	var variance float64
	for _, value := range values {
		delta := value - mean
		variance += delta * delta
	}
	sigma := math.Sqrt(variance / float64(len(values)))
	if sigma == 0 {
		return out
	}
	for id, value := range values {
		out[id] = (value - mean) / sigma
	}
	return out
}
