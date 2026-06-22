package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"time"
)

const SparseLexicalLinearHeadSchema = "manta.sparse_lexical_linear_head.v1"

const (
	SparseLexicalLinearHeadTargetTransformIdentity = "identity"
	SparseLexicalLinearHeadTargetTransformLog1p    = "log1p"
)

type SparseLexicalLinearHeadFitConfig struct {
	DatasetName       string
	Split             string
	LabelsPath        string
	DocVectorPath     string
	QueryVectorPath   string
	HeadPath          string
	HashBins          int
	MaxBins           int
	MaxPredictedTerms int
	BinRank           string
	Epochs            int
	LearningRate      float64
	NegativeRatio     int
	TargetTransform   string
}

type SparseLexicalLinearHeadEvalConfig struct {
	DatasetName       string
	Split             string
	CorpusPath        string
	QueriesPath       string
	QrelsPath         string
	HeadPath          string
	DocVectorPath     string
	QueryVectorPath   string
	ArtifactPath      string
	TopK              int
	MaxDocs           int
	MaxQueries        int
	PerQueryJSONLPath string
	DocMaxTerms       int
	QueryMaxTerms     int
	ScoreThreshold    float64
	Hybrid            RetrievalEvalHybridConfig
}

type SparseLexicalLinearHead struct {
	Schema         string                        `json:"schema"`
	Experimental   bool                          `json:"experimental"`
	Dataset        string                        `json:"dataset,omitempty"`
	Split          string                        `json:"split,omitempty"`
	Representation string                        `json:"representation"`
	Inputs         SparseLexicalLinearHeadInputs `json:"inputs"`
	Hashing        SparseLexicalHashing          `json:"hashing"`
	Config         SparseLexicalLinearHeadParams `json:"config"`
	Stats          SparseLexicalLinearHeadStats  `json:"stats"`
	Bins           []SparseLexicalLinearHeadBin  `json:"bins"`
	CreatedAt      time.Time                     `json:"created_at"`
}

type SparseLexicalLinearHeadInputs struct {
	LabelsPath      string `json:"labels_path"`
	DocVectorPath   string `json:"doc_vector_path"`
	QueryVectorPath string `json:"query_vector_path"`
}

type SparseLexicalLinearHeadParams struct {
	Dimension         int     `json:"dimension"`
	HashBins          int     `json:"hash_bins"`
	MaxBins           int     `json:"max_bins"`
	MaxPredictedTerms int     `json:"max_predicted_terms"`
	BinRank           string  `json:"bin_rank"`
	Epochs            int     `json:"epochs"`
	LearningRate      float64 `json:"learning_rate"`
	NegativeRatio     int     `json:"negative_ratio"`
	Loss              string  `json:"loss"`
	TargetTransform   string  `json:"target_transform"`
	Normalization     string  `json:"normalization"`
}

type SparseLexicalLinearHeadStats struct {
	DocumentLabels      int     `json:"document_labels"`
	QueryLabels         int     `json:"query_labels"`
	DocumentVectors     int     `json:"document_vectors"`
	QueryVectors        int     `json:"query_vectors"`
	MissingDocVectors   int     `json:"missing_doc_vectors,omitempty"`
	MissingQueryVectors int     `json:"missing_query_vectors,omitempty"`
	CandidateBins       int     `json:"candidate_bins"`
	StoredBins          int     `json:"stored_bins"`
	TrainingExamples    int     `json:"training_examples"`
	TrainingUpdates     int     `json:"training_updates"`
	FinalMSE            float64 `json:"final_mse"`
}

type SparseLexicalLinearHeadBin struct {
	Bin         uint32    `json:"bin"`
	Bias        float32   `json:"bias"`
	Weights     []float32 `json:"weights"`
	Support     int       `json:"support"`
	TotalWeight float64   `json:"total_weight"`
}

type sparseLexicalLinearExample struct {
	ID      string
	Vector  []float32
	Targets map[uint32]float64
}

func FitSparseLexicalLinearHead(cfg SparseLexicalLinearHeadFitConfig) (SparseLexicalLinearHead, error) {
	if cfg.LabelsPath == "" || cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" || cfg.HeadPath == "" {
		return SparseLexicalLinearHead{}, fmt.Errorf("labels, doc-vectors, query-vectors, and head-json paths are required")
	}
	binRank, err := normalizeSparseLexicalProjectionPrototypeRank(cfg.BinRank)
	if err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("bin rank must be one of support, total_weight, avg_weight; got %q", cfg.BinRank)
	}
	cfg.BinRank = binRank
	if cfg.Epochs == 0 {
		cfg.Epochs = 3
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.05
	}
	if cfg.NegativeRatio == 0 {
		cfg.NegativeRatio = 2
	}
	if cfg.HashBins <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("hash bins must be positive, got %d", cfg.HashBins)
	}
	if uint64(cfg.HashBins) > uint64(math.MaxUint32) {
		return SparseLexicalLinearHead{}, fmt.Errorf("hash bins must be <= %d, got %d", uint64(math.MaxUint32), cfg.HashBins)
	}
	if cfg.MaxBins <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("max bins must be positive, got %d", cfg.MaxBins)
	}
	if cfg.MaxBins > cfg.HashBins {
		return SparseLexicalLinearHead{}, fmt.Errorf("max bins must be <= hash bins, got %d > %d", cfg.MaxBins, cfg.HashBins)
	}
	if cfg.MaxPredictedTerms <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("max predicted terms must be positive, got %d", cfg.MaxPredictedTerms)
	}
	if cfg.Epochs <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("epochs must be positive, got %d", cfg.Epochs)
	}
	if cfg.LearningRate <= 0 || math.IsNaN(cfg.LearningRate) || math.IsInf(cfg.LearningRate, 0) {
		return SparseLexicalLinearHead{}, fmt.Errorf("learning rate must be finite and positive, got %g", cfg.LearningRate)
	}
	if cfg.NegativeRatio < 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("negative ratio must be non-negative, got %d", cfg.NegativeRatio)
	}
	targetTransform, err := normalizeSparseLexicalLinearHeadTargetTransform(cfg.TargetTransform)
	if err != nil {
		return SparseLexicalLinearHead{}, err
	}
	cfg.TargetTransform = targetTransform
	if sameSparseLexicalOutputPath(cfg.LabelsPath, cfg.HeadPath) {
		return SparseLexicalLinearHead{}, fmt.Errorf("labels path and head-json path must differ: %s", cfg.LabelsPath)
	}

	labels, labelStats, err := readSparseLexicalLabelFile(cfg.LabelsPath, cfg.DatasetName, cfg.Split)
	if err != nil {
		return SparseLexicalLinearHead{}, err
	}
	if len(labels.documentsOrdered) == 0 || len(labels.queries) == 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical labels must include at least one document and query")
	}
	docIDs := make([]string, 0, len(labels.documentsOrdered))
	for _, doc := range labels.documentsOrdered {
		docIDs = append(docIDs, doc.ID)
	}
	queryIDs := sparseLexicalProjectionQueryIDs(labels)
	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, docIDs)
	if err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("read document vectors: %w", err)
	}
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, queryIDs)
	if err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(docVectors) == 0 && len(queryVectors) == 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("no train vectors matched sparse lexical labels")
	}
	if docDim > 0 && queryDim > 0 && docDim != queryDim {
		return SparseLexicalLinearHead{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}
	dim := docDim
	if dim == 0 {
		dim = queryDim
	}

	accumulators := map[uint32]*sparseLexicalProjectionAccumulator{}
	docVectorMap := retrievalVectorMap(docVectors)
	for _, label := range labels.documentsOrdered {
		if _, ok := docVectorMap[label.ID]; ok {
			if err := addSparseLexicalLinearLabelAccumulator(accumulators, label, cfg.HashBins); err != nil {
				return SparseLexicalLinearHead{}, err
			}
		}
	}
	queryVectorMap := retrievalVectorMap(queryVectors)
	for _, id := range queryIDs {
		if _, ok := queryVectorMap[id]; ok {
			if err := addSparseLexicalLinearLabelAccumulator(accumulators, labels.queries[id], cfg.HashBins); err != nil {
				return SparseLexicalLinearHead{}, err
			}
		}
	}
	retained := sparseLexicalLinearRetainedBins(accumulators, cfg.MaxBins, cfg.BinRank)
	if len(retained) == 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("no non-zero sparse lexical linear bins were retained")
	}
	retainedSet := make(map[uint32]bool, len(retained))
	for _, item := range retained {
		retainedSet[item.Bin] = true
	}
	examples, err := sparseLexicalLinearExamples(labels, docVectors, queryVectors, queryIDs, cfg.HashBins, retainedSet, cfg.TargetTransform)
	if err != nil {
		return SparseLexicalLinearHead{}, err
	}
	if len(examples) == 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("no train examples matched retained sparse lexical bins")
	}
	bins, updates, finalMSE := trainSparseLexicalLinearBins(retained, examples, dim, cfg.Epochs, cfg.LearningRate, cfg.NegativeRatio)
	if len(bins) == 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("no sparse lexical linear weights were learned")
	}

	head := SparseLexicalLinearHead{
		Schema:         SparseLexicalLinearHeadSchema,
		Experimental:   true,
		Dataset:        cfg.DatasetName,
		Split:          cfg.Split,
		Representation: "experimental_sparse_lexical_linear_head",
		Inputs: SparseLexicalLinearHeadInputs{
			LabelsPath:      cfg.LabelsPath,
			DocVectorPath:   cfg.DocVectorPath,
			QueryVectorPath: cfg.QueryVectorPath,
		},
		Hashing: SparseLexicalHashing{Algorithm: "fnv1a32", Bins: cfg.HashBins, Seed: "none"},
		Config: SparseLexicalLinearHeadParams{
			Dimension:         dim,
			HashBins:          cfg.HashBins,
			MaxBins:           cfg.MaxBins,
			MaxPredictedTerms: cfg.MaxPredictedTerms,
			BinRank:           cfg.BinRank,
			Epochs:            cfg.Epochs,
			LearningRate:      cfg.LearningRate,
			NegativeRatio:     cfg.NegativeRatio,
			Loss:              "per_bin_mse_sgd",
			TargetTransform:   cfg.TargetTransform,
			Normalization:     "input_l2",
		},
		Stats: SparseLexicalLinearHeadStats{
			DocumentLabels:      labelStats.DocumentLabels,
			QueryLabels:         labelStats.QueryLabels,
			DocumentVectors:     len(docVectors),
			QueryVectors:        len(queryVectors),
			MissingDocVectors:   missingDocVectors,
			MissingQueryVectors: missingQueryVectors,
			CandidateBins:       len(accumulators),
			StoredBins:          len(bins),
			TrainingExamples:    len(examples),
			TrainingUpdates:     updates,
			FinalMSE:            finalMSE,
		},
		Bins:      bins,
		CreatedAt: time.Now().UTC(),
	}
	data, err := json.MarshalIndent(head, "", "  ")
	if err != nil {
		return SparseLexicalLinearHead{}, err
	}
	data = append(data, '\n')
	if err := os.WriteFile(cfg.HeadPath, data, 0o644); err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("write sparse lexical linear head: %w", err)
	}
	return head, nil
}

func EvaluateSparseLexicalLinearHeadVectorHybrid(ctx context.Context, cfg SparseLexicalLinearHeadEvalConfig) (RetrievalEvalMetrics, error) {
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" || cfg.HeadPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, qrels, and head-json paths are required")
	}
	if cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("document and query vector paths are required")
	}
	if cfg.TopK == 0 {
		cfg.TopK = 100
	}
	if cfg.TopK < 100 {
		cfg.TopK = 100
	}
	if cfg.DocMaxTerms < 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("doc max terms must be non-negative, got %d", cfg.DocMaxTerms)
	}
	if cfg.QueryMaxTerms < 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("query max terms must be non-negative, got %d", cfg.QueryMaxTerms)
	}
	if math.IsNaN(cfg.ScoreThreshold) || math.IsInf(cfg.ScoreThreshold, 0) {
		return RetrievalEvalMetrics{}, fmt.Errorf("score threshold must be finite, got %g", cfg.ScoreThreshold)
	}
	sparseOnly := sparseLexicalProjectionSparseOnlyMethod(cfg.Hybrid.Method)
	hybridCfg := cfg.Hybrid
	var err error
	if !sparseOnly {
		hybridCfg, err = normalizeRetrievalEvalHybridConfig(cfg.Hybrid)
		if err != nil {
			return RetrievalEvalMetrics{}, err
		}
	}

	start := time.Now()
	head, err := ReadSparseLexicalLinearHead(cfg.HeadPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if cfg.DatasetName != "" && head.Dataset != "" && head.Dataset != cfg.DatasetName {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical linear head dataset=%q, want %q", head.Dataset, cfg.DatasetName)
	}
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

	loadStart := time.Now()
	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read document vectors: %w", err)
	}
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, retrievalIDs(queries))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(docVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no vectors for the evaluated corpus")
	}
	if len(queryVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("query vector cache has no vectors for qrels queries")
	}
	if docDim != queryDim {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}
	if docDim != head.Config.Dimension {
		return RetrievalEvalMetrics{}, fmt.Errorf("vector dimension %d does not match linear head dimension %d", docDim, head.Config.Dimension)
	}
	docMaxTerms := cfg.DocMaxTerms
	if docMaxTerms == 0 {
		docMaxTerms = head.Config.MaxPredictedTerms
	}
	queryMaxTerms := cfg.QueryMaxTerms
	if queryMaxTerms == 0 {
		queryMaxTerms = head.Config.MaxPredictedTerms
	}
	sparseDocs := make([]sparseLexicalHashedVector, 0, len(docVectors))
	for _, doc := range docVectors {
		sparseDocs = append(sparseDocs, predictSparseLexicalLinearVector(doc.ID, doc.Vector, head, docMaxTerms, cfg.ScoreThreshold))
	}
	sparseQueries := make(map[string]sparseLexicalHashedVector, len(queryVectors))
	for _, query := range queryVectors {
		sparseQueries[query.ID] = predictSparseLexicalLinearVector(query.ID, query.Vector, head, queryMaxTerms, cfg.ScoreThreshold)
	}
	loadDuration := time.Since(loadStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeSparseLexicalProjectionVectorQuality(ctx, queryVectors, docVectors, sparseQueries, sparseDocs, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath, hybridCfg, sparseOnly)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated vector cache")
	}
	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docVectors))
	stats := SparseLexicalLabelEvalStats{
		Representation:     "experimental_sparse_lexical_linear_head",
		HeadPath:           cfg.HeadPath,
		HashBins:           head.Hashing.Bins,
		DocumentLabels:     len(sparseDocs),
		QueryLabels:        len(sparseQueries),
		DocumentMaxHashNNZ: docMaxTerms,
		QueryMaxHashNNZ:    queryMaxTerms,
		ScoreThreshold:     cfg.ScoreThreshold,
	}
	metricsHybrid := retrievalEvalHybridMetrics(hybridCfg)
	if sparseOnly {
		metricsHybrid = &RetrievalEvalHybridMetrics{Method: "sparse_only"}
	}
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  "sparse_lexical_linear_head_vectors_hybrid",
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:      cfg.CorpusPath,
			QueriesPath:     cfg.QueriesPath,
			QrelsPath:       cfg.QrelsPath,
			HeadPath:        cfg.HeadPath,
			DocVectorPath:   cfg.DocVectorPath,
			QueryVectorPath: cfg.QueryVectorPath,
			Documents:       len(docVectors),
			Queries:         evaluatedQueries,
			RelevantPairs:   relevantPairs,
			ScoredPairs:     scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
			Hybrid:     metricsHybrid,
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: loadDuration.Seconds(),
			QueryEmbedSeconds:    loadDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docVectors)), loadDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(queryVectors)), loadDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
			QueriesWithoutVector:       missingQueryVectors,
			DocumentsWithoutVector:     missingDocVectors,
		},
		SparseLexical: &stats,
	}, nil
}

func ReadSparseLexicalLinearHead(path string) (SparseLexicalLinearHead, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("read sparse lexical linear head: %w", err)
	}
	var head SparseLexicalLinearHead
	if err := json.Unmarshal(data, &head); err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("decode sparse lexical linear head: %w", err)
	}
	if head.Schema != SparseLexicalLinearHeadSchema {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head has unsupported schema %q", head.Schema)
	}
	if head.Hashing.Algorithm != "fnv1a32" {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head has unsupported hashing algorithm %q", head.Hashing.Algorithm)
	}
	if head.Hashing.Bins <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head hash bins must be positive, got %d", head.Hashing.Bins)
	}
	if head.Config.Dimension <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head dimension must be positive, got %d", head.Config.Dimension)
	}
	if head.Config.MaxPredictedTerms <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head max predicted terms must be positive, got %d", head.Config.MaxPredictedTerms)
	}
	if head.Config.Epochs <= 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head epochs must be positive, got %d", head.Config.Epochs)
	}
	if head.Config.LearningRate <= 0 || math.IsNaN(head.Config.LearningRate) || math.IsInf(head.Config.LearningRate, 0) {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head learning rate must be finite and positive, got %g", head.Config.LearningRate)
	}
	if head.Config.NegativeRatio < 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head negative ratio must be non-negative, got %d", head.Config.NegativeRatio)
	}
	targetTransform, err := normalizeSparseLexicalLinearHeadTargetTransform(head.Config.TargetTransform)
	if err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head has invalid target_transform: %w", err)
	}
	head.Config.TargetTransform = targetTransform
	binRank, err := normalizeSparseLexicalProjectionPrototypeRank(head.Config.BinRank)
	if err != nil {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head has invalid bin_rank: %w", err)
	}
	head.Config.BinRank = binRank
	if len(head.Bins) == 0 {
		return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head has no bins")
	}
	for i, bin := range head.Bins {
		if bin.Bin >= uint32(head.Hashing.Bins) {
			return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head bin %d id=%d outside hash_bins=%d", i, bin.Bin, head.Hashing.Bins)
		}
		if len(bin.Weights) != head.Config.Dimension {
			return SparseLexicalLinearHead{}, fmt.Errorf("sparse lexical linear head bin %d dimension=%d, want %d", i, len(bin.Weights), head.Config.Dimension)
		}
	}
	return head, nil
}

func normalizeSparseLexicalLinearHeadTargetTransform(transform string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(transform)) {
	case "", SparseLexicalLinearHeadTargetTransformIdentity:
		return SparseLexicalLinearHeadTargetTransformIdentity, nil
	case SparseLexicalLinearHeadTargetTransformLog1p:
		return SparseLexicalLinearHeadTargetTransformLog1p, nil
	default:
		return "", fmt.Errorf("target transform must be one of identity, log1p; got %q", transform)
	}
}

func sparseLexicalLinearTransformTarget(weight float64, transform string) float64 {
	switch transform {
	case SparseLexicalLinearHeadTargetTransformLog1p:
		return math.Log1p(weight)
	default:
		return weight
	}
}

func sparseLexicalLinearRetainedBins(acc map[uint32]*sparseLexicalProjectionAccumulator, maxBins int, binRank string) []*sparseLexicalProjectionAccumulator {
	items := make([]*sparseLexicalProjectionAccumulator, 0, len(acc))
	for _, item := range acc {
		items = append(items, item)
	}
	slices.SortFunc(items, func(a, b *sparseLexicalProjectionAccumulator) int {
		switch binRank {
		case SparseLexicalProjectionPrototypeRankTotalWeight:
			if cmp := compareSparseLexicalProjectionTotalWeight(a, b); cmp != 0 {
				return cmp
			}
			if cmp := compareSparseLexicalProjectionSupport(a, b); cmp != 0 {
				return cmp
			}
		case SparseLexicalProjectionPrototypeRankAvgWeight:
			if cmp := compareSparseLexicalProjectionAvgWeight(a, b); cmp != 0 {
				return cmp
			}
			if cmp := compareSparseLexicalProjectionSupport(a, b); cmp != 0 {
				return cmp
			}
			if cmp := compareSparseLexicalProjectionTotalWeight(a, b); cmp != 0 {
				return cmp
			}
		default:
			if cmp := compareSparseLexicalProjectionSupport(a, b); cmp != 0 {
				return cmp
			}
			if cmp := compareSparseLexicalProjectionTotalWeight(a, b); cmp != 0 {
				return cmp
			}
		}
		return compareSparseLexicalProjectionBin(a, b)
	})
	if len(items) > maxBins {
		items = items[:maxBins]
	}
	return items
}

func addSparseLexicalLinearLabelAccumulator(acc map[uint32]*sparseLexicalProjectionAccumulator, label SparseLexicalLabelRecord, hashBins int) error {
	hashed, _, err := sparseLexicalHashRecord(label, hashBins)
	if err != nil {
		return err
	}
	for _, term := range hashed.Terms {
		if term.Weight <= 0 {
			continue
		}
		item := acc[term.Bin]
		if item == nil {
			item = &sparseLexicalProjectionAccumulator{Bin: term.Bin}
			acc[term.Bin] = item
		}
		item.Support++
		item.TotalWeight += term.Weight
	}
	return nil
}

func sparseLexicalLinearExamples(labels sparseLexicalLabelSet, docVectors, queryVectors []retrievalVectorRecord, queryIDs []string, hashBins int, retained map[uint32]bool, targetTransform string) ([]sparseLexicalLinearExample, error) {
	examples := make([]sparseLexicalLinearExample, 0, len(docVectors)+len(queryVectors))
	docVectorMap := retrievalVectorMap(docVectors)
	for _, label := range labels.documentsOrdered {
		vector, ok := docVectorMap[label.ID]
		if !ok {
			continue
		}
		targets, err := sparseLexicalLinearTargets(label, hashBins, retained, targetTransform)
		if err != nil {
			return nil, err
		}
		examples = append(examples, sparseLexicalLinearExample{ID: label.ID, Vector: vector.Vector, Targets: targets})
	}
	queryVectorMap := retrievalVectorMap(queryVectors)
	for _, id := range queryIDs {
		vector, ok := queryVectorMap[id]
		if !ok {
			continue
		}
		targets, err := sparseLexicalLinearTargets(labels.queries[id], hashBins, retained, targetTransform)
		if err != nil {
			return nil, err
		}
		examples = append(examples, sparseLexicalLinearExample{ID: id, Vector: vector.Vector, Targets: targets})
	}
	return examples, nil
}

func sparseLexicalLinearTargets(label SparseLexicalLabelRecord, hashBins int, retained map[uint32]bool, targetTransform string) (map[uint32]float64, error) {
	hashed, _, err := sparseLexicalHashRecord(label, hashBins)
	if err != nil {
		return nil, err
	}
	targets := make(map[uint32]float64)
	for _, term := range hashed.Terms {
		if term.Weight > 0 && retained[term.Bin] {
			targets[term.Bin] = sparseLexicalLinearTransformTarget(term.Weight, targetTransform)
		}
	}
	return targets, nil
}

func trainSparseLexicalLinearBins(retained []*sparseLexicalProjectionAccumulator, examples []sparseLexicalLinearExample, dim, epochs int, learningRate float64, negativeRatio int) ([]SparseLexicalLinearHeadBin, int, float64) {
	slices.SortFunc(retained, func(a, b *sparseLexicalProjectionAccumulator) int {
		return compareSparseLexicalProjectionBin(a, b)
	})
	bins := make([]SparseLexicalLinearHeadBin, 0, len(retained))
	for _, item := range retained {
		bins = append(bins, SparseLexicalLinearHeadBin{
			Bin:         item.Bin,
			Weights:     make([]float32, dim),
			Support:     item.Support,
			TotalWeight: item.TotalWeight,
		})
	}
	indexByBin := make(map[uint32]int, len(bins))
	for i, bin := range bins {
		indexByBin[bin.Bin] = i
	}
	updates := 0
	for epoch := 0; epoch < epochs; epoch++ {
		for exampleIndex, example := range examples {
			for bin, target := range example.Targets {
				idx, ok := indexByBin[bin]
				if !ok {
					continue
				}
				sparseLexicalLinearSGDUpdate(&bins[idx], example.Vector, target, learningRate)
				updates++
			}
			if negativeRatio == 0 || len(bins) == 0 {
				continue
			}
			seenNegatives := 0
			for offset := 0; seenNegatives < negativeRatio && offset < len(bins)*2; offset++ {
				idx := ((epoch + 1) * (exampleIndex + 1 + offset)) % len(bins)
				if _, positive := example.Targets[bins[idx].Bin]; positive {
					continue
				}
				sparseLexicalLinearSGDUpdate(&bins[idx], example.Vector, 0, learningRate)
				updates++
				seenNegatives++
			}
		}
	}
	return bins, updates, sparseLexicalLinearMSE(bins, examples, indexByBin)
}

func sparseLexicalLinearSGDUpdate(bin *SparseLexicalLinearHeadBin, vector []float32, target, learningRate float64) {
	prediction := float64(bin.Bias)
	for i, value := range vector {
		prediction += float64(bin.Weights[i]) * float64(value)
	}
	gradient := prediction - target
	bin.Bias -= float32(learningRate * gradient)
	for i, value := range vector {
		bin.Weights[i] -= float32(learningRate * gradient * float64(value))
	}
}

func sparseLexicalLinearMSE(bins []SparseLexicalLinearHeadBin, examples []sparseLexicalLinearExample, indexByBin map[uint32]int) float64 {
	var sum float64
	count := 0
	for _, example := range examples {
		for bin, target := range example.Targets {
			idx, ok := indexByBin[bin]
			if !ok {
				continue
			}
			pred := sparseLexicalLinearScore(example.Vector, bins[idx])
			diff := pred - target
			sum += diff * diff
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func predictSparseLexicalLinearVector(id string, vector []float32, head SparseLexicalLinearHead, maxTerms int, scoreThreshold float64) sparseLexicalHashedVector {
	candidates := make([]sparseLexicalHashedTerm, 0, len(head.Bins))
	for _, bin := range head.Bins {
		score := sparseLexicalLinearScore(vector, bin)
		if score > scoreThreshold {
			candidates = append(candidates, sparseLexicalHashedTerm{Bin: bin.Bin, Weight: score})
		}
	}
	slices.SortFunc(candidates, func(a, b sparseLexicalHashedTerm) int {
		if a.Weight > b.Weight {
			return -1
		}
		if a.Weight < b.Weight {
			return 1
		}
		if a.Bin < b.Bin {
			return -1
		}
		if a.Bin > b.Bin {
			return 1
		}
		return 0
	})
	if len(candidates) > maxTerms {
		candidates = candidates[:maxTerms]
	}
	slices.SortFunc(candidates, func(a, b sparseLexicalHashedTerm) int {
		if a.Bin < b.Bin {
			return -1
		}
		if a.Bin > b.Bin {
			return 1
		}
		return 0
	})
	return sparseLexicalHashedVector{ID: id, Terms: candidates}
}

func sparseLexicalLinearScore(vector []float32, bin SparseLexicalLinearHeadBin) float64 {
	score := float64(bin.Bias)
	for i, value := range vector {
		score += float64(bin.Weights[i]) * float64(value)
	}
	return score
}
