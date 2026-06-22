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

const SparseLexicalProjectionHeadSchema = "manta.sparse_lexical_projection_head.v1"

type SparseLexicalProjectionHeadFitConfig struct {
	DatasetName       string
	Split             string
	LabelsPath        string
	DocVectorPath     string
	QueryVectorPath   string
	HeadPath          string
	HashBins          int
	MaxPrototypes     int
	MaxPredictedTerms int
}

type SparseLexicalProjectionHeadEvalConfig struct {
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
	Hybrid            RetrievalEvalHybridConfig
}

type SparseLexicalProjectionHead struct {
	Schema         string                             `json:"schema"`
	Experimental   bool                               `json:"experimental"`
	Dataset        string                             `json:"dataset,omitempty"`
	Split          string                             `json:"split,omitempty"`
	Representation string                             `json:"representation"`
	Inputs         SparseLexicalProjectionHeadInputs  `json:"inputs"`
	Hashing        SparseLexicalHashing               `json:"hashing"`
	Config         SparseLexicalProjectionHeadParams  `json:"config"`
	Stats          SparseLexicalProjectionHeadStats   `json:"stats"`
	Prototypes     []SparseLexicalProjectionPrototype `json:"prototypes"`
	CreatedAt      time.Time                          `json:"created_at"`
}

type SparseLexicalProjectionHeadInputs struct {
	LabelsPath      string `json:"labels_path"`
	DocVectorPath   string `json:"doc_vector_path"`
	QueryVectorPath string `json:"query_vector_path"`
}

type SparseLexicalProjectionHeadParams struct {
	Dimension         int    `json:"dimension"`
	HashBins          int    `json:"hash_bins"`
	MaxPrototypes     int    `json:"max_prototypes"`
	MaxPredictedTerms int    `json:"max_predicted_terms"`
	PrototypeSource   string `json:"prototype_source"`
	Normalization     string `json:"normalization"`
}

type SparseLexicalProjectionHeadStats struct {
	DocumentLabels      int `json:"document_labels"`
	QueryLabels         int `json:"query_labels"`
	DocumentVectors     int `json:"document_vectors"`
	QueryVectors        int `json:"query_vectors"`
	MissingDocVectors   int `json:"missing_doc_vectors,omitempty"`
	MissingQueryVectors int `json:"missing_query_vectors,omitempty"`
	CandidatePrototypes int `json:"candidate_prototypes"`
	StoredPrototypes    int `json:"stored_prototypes"`
}

type SparseLexicalProjectionPrototype struct {
	Bin         uint32    `json:"bin"`
	Vector      []float32 `json:"vector"`
	Support     int       `json:"support"`
	TotalWeight float64   `json:"total_weight"`
}

type sparseLexicalProjectionAccumulator struct {
	Bin         uint32
	Vector      []float64
	Support     int
	TotalWeight float64
}

func FitSparseLexicalProjectionHead(cfg SparseLexicalProjectionHeadFitConfig) (SparseLexicalProjectionHead, error) {
	if cfg.LabelsPath == "" || cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" || cfg.HeadPath == "" {
		return SparseLexicalProjectionHead{}, fmt.Errorf("labels, doc-vectors, query-vectors, and head-json paths are required")
	}
	if cfg.HashBins <= 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("hash bins must be positive, got %d", cfg.HashBins)
	}
	if uint64(cfg.HashBins) > uint64(math.MaxUint32) {
		return SparseLexicalProjectionHead{}, fmt.Errorf("hash bins must be <= %d, got %d", uint64(math.MaxUint32), cfg.HashBins)
	}
	if cfg.MaxPrototypes <= 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("max prototypes must be positive, got %d", cfg.MaxPrototypes)
	}
	if cfg.MaxPredictedTerms <= 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("max predicted terms must be positive, got %d", cfg.MaxPredictedTerms)
	}
	if cfg.MaxPrototypes > cfg.HashBins {
		return SparseLexicalProjectionHead{}, fmt.Errorf("max prototypes must be <= hash bins, got %d > %d", cfg.MaxPrototypes, cfg.HashBins)
	}
	if sameSparseLexicalOutputPath(cfg.LabelsPath, cfg.HeadPath) {
		return SparseLexicalProjectionHead{}, fmt.Errorf("labels path and head-json path must differ: %s", cfg.LabelsPath)
	}

	labels, labelStats, err := readSparseLexicalLabelFile(cfg.LabelsPath, cfg.DatasetName, cfg.Split)
	if err != nil {
		return SparseLexicalProjectionHead{}, err
	}
	if len(labels.documentsOrdered) == 0 || len(labels.queries) == 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical labels must include at least one document and query")
	}
	docIDs := make([]string, 0, len(labels.documentsOrdered))
	for _, doc := range labels.documentsOrdered {
		docIDs = append(docIDs, doc.ID)
	}
	queryIDs := sparseLexicalProjectionQueryIDs(labels)
	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, docIDs)
	if err != nil {
		return SparseLexicalProjectionHead{}, fmt.Errorf("read document vectors: %w", err)
	}
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, queryIDs)
	if err != nil {
		return SparseLexicalProjectionHead{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(docVectors) == 0 && len(queryVectors) == 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("no train vectors matched sparse lexical labels")
	}
	if docDim > 0 && queryDim > 0 && docDim != queryDim {
		return SparseLexicalProjectionHead{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}
	dim := docDim
	if dim == 0 {
		dim = queryDim
	}

	accumulators := map[uint32]*sparseLexicalProjectionAccumulator{}
	docVectorMap := retrievalVectorMap(docVectors)
	for _, label := range labels.documentsOrdered {
		if vector, ok := docVectorMap[label.ID]; ok {
			if err := addSparseLexicalProjectionExample(accumulators, vector.Vector, label, cfg.HashBins); err != nil {
				return SparseLexicalProjectionHead{}, err
			}
		}
	}
	queryVectorMap := retrievalVectorMap(queryVectors)
	for _, id := range queryIDs {
		if vector, ok := queryVectorMap[id]; ok {
			if err := addSparseLexicalProjectionExample(accumulators, vector.Vector, labels.queries[id], cfg.HashBins); err != nil {
				return SparseLexicalProjectionHead{}, err
			}
		}
	}
	prototypes := sparseLexicalProjectionPrototypes(accumulators, cfg.MaxPrototypes)
	if len(prototypes) == 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("no non-zero sparse lexical projection prototypes were learned")
	}

	head := SparseLexicalProjectionHead{
		Schema:         SparseLexicalProjectionHeadSchema,
		Experimental:   true,
		Dataset:        cfg.DatasetName,
		Split:          cfg.Split,
		Representation: "experimental_sparse_lexical_projection_head",
		Inputs: SparseLexicalProjectionHeadInputs{
			LabelsPath:      cfg.LabelsPath,
			DocVectorPath:   cfg.DocVectorPath,
			QueryVectorPath: cfg.QueryVectorPath,
		},
		Hashing: SparseLexicalHashing{Algorithm: "fnv1a32", Bins: cfg.HashBins, Seed: "none"},
		Config: SparseLexicalProjectionHeadParams{
			Dimension:         dim,
			HashBins:          cfg.HashBins,
			MaxPrototypes:     cfg.MaxPrototypes,
			MaxPredictedTerms: cfg.MaxPredictedTerms,
			PrototypeSource:   "label_weighted_dense_doc_query_centroids",
			Normalization:     "input_l2_and_prototype_l2",
		},
		Stats: SparseLexicalProjectionHeadStats{
			DocumentLabels:      labelStats.DocumentLabels,
			QueryLabels:         labelStats.QueryLabels,
			DocumentVectors:     len(docVectors),
			QueryVectors:        len(queryVectors),
			MissingDocVectors:   missingDocVectors,
			MissingQueryVectors: missingQueryVectors,
			CandidatePrototypes: len(accumulators),
			StoredPrototypes:    len(prototypes),
		},
		Prototypes: prototypes,
		CreatedAt:  time.Now().UTC(),
	}
	data, err := json.MarshalIndent(head, "", "  ")
	if err != nil {
		return SparseLexicalProjectionHead{}, err
	}
	data = append(data, '\n')
	if err := os.WriteFile(cfg.HeadPath, data, 0o644); err != nil {
		return SparseLexicalProjectionHead{}, fmt.Errorf("write sparse lexical projection head: %w", err)
	}
	return head, nil
}

func EvaluateSparseLexicalProjectionHeadVectorHybrid(ctx context.Context, cfg SparseLexicalProjectionHeadEvalConfig) (RetrievalEvalMetrics, error) {
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
	head, err := ReadSparseLexicalProjectionHead(cfg.HeadPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if cfg.DatasetName != "" && head.Dataset != "" && head.Dataset != cfg.DatasetName {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical projection head dataset=%q, want %q", head.Dataset, cfg.DatasetName)
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
		return RetrievalEvalMetrics{}, fmt.Errorf("vector dimension %d does not match projection head dimension %d", docDim, head.Config.Dimension)
	}
	sparseDocs := make([]sparseLexicalHashedVector, 0, len(docVectors))
	for _, doc := range docVectors {
		sparseDocs = append(sparseDocs, predictSparseLexicalProjectionVector(doc.ID, doc.Vector, head))
	}
	sparseQueries := make(map[string]sparseLexicalHashedVector, len(queryVectors))
	for _, query := range queryVectors {
		sparseQueries[query.ID] = predictSparseLexicalProjectionVector(query.ID, query.Vector, head)
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
		Representation:     "experimental_sparse_lexical_projection_head",
		HeadPath:           cfg.HeadPath,
		HashBins:           head.Hashing.Bins,
		DocumentLabels:     len(sparseDocs),
		QueryLabels:        len(sparseQueries),
		DocumentMaxHashNNZ: head.Config.MaxPredictedTerms,
		QueryMaxHashNNZ:    head.Config.MaxPredictedTerms,
	}
	metricsHybrid := retrievalEvalHybridMetrics(hybridCfg)
	if sparseOnly {
		metricsHybrid = &RetrievalEvalHybridMetrics{Method: "sparse_only"}
	}
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  "sparse_lexical_projection_head_vectors_hybrid",
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

func ReadSparseLexicalProjectionHead(path string) (SparseLexicalProjectionHead, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SparseLexicalProjectionHead{}, fmt.Errorf("read sparse lexical projection head: %w", err)
	}
	var head SparseLexicalProjectionHead
	if err := json.Unmarshal(data, &head); err != nil {
		return SparseLexicalProjectionHead{}, fmt.Errorf("decode sparse lexical projection head: %w", err)
	}
	if head.Schema != SparseLexicalProjectionHeadSchema {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head has unsupported schema %q", head.Schema)
	}
	if head.Hashing.Algorithm != "fnv1a32" {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head has unsupported hashing algorithm %q", head.Hashing.Algorithm)
	}
	if head.Hashing.Bins <= 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head hash bins must be positive, got %d", head.Hashing.Bins)
	}
	if head.Config.Dimension <= 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head dimension must be positive, got %d", head.Config.Dimension)
	}
	if head.Config.MaxPredictedTerms <= 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head max predicted terms must be positive, got %d", head.Config.MaxPredictedTerms)
	}
	if len(head.Prototypes) == 0 {
		return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head has no prototypes")
	}
	for i, proto := range head.Prototypes {
		if proto.Bin >= uint32(head.Hashing.Bins) {
			return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head prototype %d bin=%d outside hash_bins=%d", i, proto.Bin, head.Hashing.Bins)
		}
		if len(proto.Vector) != head.Config.Dimension {
			return SparseLexicalProjectionHead{}, fmt.Errorf("sparse lexical projection head prototype %d dimension=%d, want %d", i, len(proto.Vector), head.Config.Dimension)
		}
	}
	return head, nil
}

func sparseLexicalProjectionQueryIDs(labels sparseLexicalLabelSet) []string {
	ids := make([]string, 0, len(labels.queries))
	for id := range labels.queries {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func retrievalVectorMap(vectors []retrievalVectorRecord) map[string]retrievalVectorRecord {
	out := make(map[string]retrievalVectorRecord, len(vectors))
	for _, vector := range vectors {
		out[vector.ID] = vector
	}
	return out
}

func addSparseLexicalProjectionExample(acc map[uint32]*sparseLexicalProjectionAccumulator, vector []float32, label SparseLexicalLabelRecord, hashBins int) error {
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
			item = &sparseLexicalProjectionAccumulator{Bin: term.Bin, Vector: make([]float64, len(vector))}
			acc[term.Bin] = item
		}
		item.Support++
		item.TotalWeight += term.Weight
		for i, value := range vector {
			item.Vector[i] += float64(value) * term.Weight
		}
	}
	return nil
}

func sparseLexicalProjectionPrototypes(acc map[uint32]*sparseLexicalProjectionAccumulator, maxPrototypes int) []SparseLexicalProjectionPrototype {
	items := make([]*sparseLexicalProjectionAccumulator, 0, len(acc))
	for _, item := range acc {
		items = append(items, item)
	}
	slices.SortFunc(items, func(a, b *sparseLexicalProjectionAccumulator) int {
		if a.Support != b.Support {
			return b.Support - a.Support
		}
		if a.TotalWeight > b.TotalWeight {
			return -1
		}
		if a.TotalWeight < b.TotalWeight {
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
	if len(items) > maxPrototypes {
		items = items[:maxPrototypes]
	}
	prototypes := make([]SparseLexicalProjectionPrototype, 0, len(items))
	for _, item := range items {
		vec := normalizeSparseLexicalProjectionPrototype(item.Vector)
		if len(vec) == 0 {
			continue
		}
		prototypes = append(prototypes, SparseLexicalProjectionPrototype{
			Bin:         item.Bin,
			Vector:      vec,
			Support:     item.Support,
			TotalWeight: item.TotalWeight,
		})
	}
	slices.SortFunc(prototypes, func(a, b SparseLexicalProjectionPrototype) int {
		if a.Bin < b.Bin {
			return -1
		}
		if a.Bin > b.Bin {
			return 1
		}
		return 0
	})
	return prototypes
}

func normalizeSparseLexicalProjectionPrototype(vector []float64) []float32 {
	var norm float64
	for _, value := range vector {
		norm += value * value
	}
	if norm <= 0 {
		return nil
	}
	scale := 1 / math.Sqrt(norm)
	out := make([]float32, len(vector))
	for i, value := range vector {
		out[i] = float32(value * scale)
	}
	return out
}

func predictSparseLexicalProjectionVector(id string, vector []float32, head SparseLexicalProjectionHead) sparseLexicalHashedVector {
	candidates := make([]sparseLexicalHashedTerm, 0, len(head.Prototypes))
	for _, proto := range head.Prototypes {
		score := float64(dotRetrievalVectors(vector, proto.Vector))
		if score > 0 {
			candidates = append(candidates, sparseLexicalHashedTerm{Bin: proto.Bin, Weight: score})
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
	if len(candidates) > head.Config.MaxPredictedTerms {
		candidates = candidates[:head.Config.MaxPredictedTerms]
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

func sparseLexicalProjectionSparseOnlyMethod(method string) bool {
	switch strings.ToLower(strings.TrimSpace(method)) {
	case "sparse", "sparse_only", "sparse-only":
		return true
	default:
		return false
	}
}

func computeSparseLexicalProjectionVectorQuality(ctx context.Context, queries, docs []retrievalVectorRecord, sparseQueries map[string]sparseLexicalHashedVector, sparseDocs []sparseLexicalHashedVector, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string, cfg RetrievalEvalHybridConfig, sparseOnly bool) (RetrievalEvalQualityMetrics, int, int, int, int, error) {
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
		sparseQuery := sparseQueries[query.ID]
		sparseScores := topSparseLexicalHashScores(sparseQuery.Terms, sparseDocs, topK)
		scores := sparseScores
		if !sparseOnly {
			denseScores := topRetrievalScores(query.Vector, docs, topK)
			scores = fuseHybridScores(denseScores, sparseScores, topK, cfg)
		}
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
