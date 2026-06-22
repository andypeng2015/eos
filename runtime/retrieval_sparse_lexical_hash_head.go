package eosruntime

import (
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"time"
)

const SparseLexicalHashHeadSchema = "manta.sparse_lexical_hash_head.v1"

type SparseLexicalHashHeadFitConfig struct {
	DatasetName string
	Split       string
	LabelsPath  string
	HeadPath    string
	HashBins    int
}

type SparseLexicalHashHeadEvalConfig struct {
	DatasetName       string
	Split             string
	CorpusPath        string
	QueriesPath       string
	QrelsPath         string
	LabelsPath        string
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

type SparseLexicalHashHead struct {
	Schema         string                      `json:"schema"`
	Experimental   bool                        `json:"experimental"`
	Dataset        string                      `json:"dataset,omitempty"`
	Split          string                      `json:"split,omitempty"`
	Representation string                      `json:"representation"`
	Inputs         SparseLexicalHashHeadInputs `json:"inputs"`
	Hashing        SparseLexicalHashing        `json:"hashing"`
	Stats          SparseLexicalLabelEvalStats `json:"stats"`
	CreatedAt      time.Time                   `json:"created_at"`
}

type SparseLexicalHashHeadInputs struct {
	LabelsPath string `json:"labels_path"`
}

type sparseLexicalHashedVector struct {
	ID    string
	Terms []sparseLexicalHashedTerm
}

type sparseLexicalHashedTerm struct {
	Bin    uint32
	Weight float64
}

func FitSparseLexicalHashHead(cfg SparseLexicalHashHeadFitConfig) (SparseLexicalHashHead, error) {
	if cfg.LabelsPath == "" || cfg.HeadPath == "" {
		return SparseLexicalHashHead{}, fmt.Errorf("labels and head-json paths are required")
	}
	if cfg.HashBins <= 0 {
		return SparseLexicalHashHead{}, fmt.Errorf("hash bins must be positive, got %d", cfg.HashBins)
	}
	if uint64(cfg.HashBins) > uint64(math.MaxUint32) {
		return SparseLexicalHashHead{}, fmt.Errorf("hash bins must be <= %d, got %d", uint64(math.MaxUint32), cfg.HashBins)
	}
	if sameSparseLexicalOutputPath(cfg.LabelsPath, cfg.HeadPath) {
		return SparseLexicalHashHead{}, fmt.Errorf("labels path and head-json path must differ: %s", cfg.LabelsPath)
	}
	labels, stats, err := readSparseLexicalLabelFile(cfg.LabelsPath, cfg.DatasetName, cfg.Split)
	if err != nil {
		return SparseLexicalHashHead{}, err
	}
	if len(labels.documentsOrdered) == 0 || len(labels.queries) == 0 {
		return SparseLexicalHashHead{}, fmt.Errorf("sparse lexical labels must include at least one document and query")
	}
	stats.Representation = "experimental_hashed_sparse_lexical_head"
	stats.LabelsPath = cfg.LabelsPath
	stats.HeadPath = cfg.HeadPath
	stats.HashBins = cfg.HashBins
	if err := fillSparseLexicalHashStats(labels, cfg.HashBins, &stats); err != nil {
		return SparseLexicalHashHead{}, err
	}
	head := SparseLexicalHashHead{
		Schema:         SparseLexicalHashHeadSchema,
		Experimental:   true,
		Dataset:        cfg.DatasetName,
		Split:          cfg.Split,
		Representation: "experimental_hashed_sparse_lexical_head",
		Inputs:         SparseLexicalHashHeadInputs{LabelsPath: cfg.LabelsPath},
		Hashing: SparseLexicalHashing{
			Algorithm: "fnv1a32",
			Bins:      cfg.HashBins,
			Seed:      "none",
		},
		Stats:     stats,
		CreatedAt: time.Now().UTC(),
	}
	data, err := json.MarshalIndent(head, "", "  ")
	if err != nil {
		return SparseLexicalHashHead{}, err
	}
	data = append(data, '\n')
	if err := os.WriteFile(cfg.HeadPath, data, 0o644); err != nil {
		return SparseLexicalHashHead{}, fmt.Errorf("write sparse lexical hash head: %w", err)
	}
	return head, nil
}

func EvaluateSparseLexicalHashHead(ctx context.Context, cfg SparseLexicalHashHeadEvalConfig) (RetrievalEvalMetrics, error) {
	if cfg.QueriesPath == "" || cfg.QrelsPath == "" || cfg.LabelsPath == "" || cfg.HeadPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("queries, qrels, labels, and head-json paths are required")
	}
	if cfg.TopK == 0 {
		cfg.TopK = 100
	}
	if cfg.TopK < 100 {
		return RetrievalEvalMetrics{}, fmt.Errorf("top-k must be at least 100 for recall_at_100, got %d", cfg.TopK)
	}
	start := time.Now()
	head, err := ReadSparseLexicalHashHead(cfg.HeadPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if cfg.DatasetName != "" && head.Dataset != "" && head.Dataset != cfg.DatasetName {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical hash head dataset=%q, want %q", head.Dataset, cfg.DatasetName)
	}
	if cfg.Split != "" && head.Split != "" && head.Split != cfg.Split {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical hash head split=%q, want %q", head.Split, cfg.Split)
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}
	loadStart := time.Now()
	labels, stats, err := readSparseLexicalLabelFile(cfg.LabelsPath, cfg.DatasetName, cfg.Split)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	docs, queryVectors, err := sparseLexicalHashVectors(labels, head.Hashing.Bins, &stats)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	loadDuration := time.Since(loadStart)
	if len(docs) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no document sparse lexical labels found")
	}
	if len(queryVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no query sparse lexical labels found")
	}
	stats.Representation = "experimental_hashed_sparse_lexical_head"
	stats.LabelsPath = cfg.LabelsPath
	stats.HeadPath = cfg.HeadPath
	stats.HashBins = head.Hashing.Bins
	evalQueries := make([]sparseLexicalHashedVector, 0, len(queries))
	missingQueryLabels := 0
	for _, query := range queries {
		label, ok := queryVectors[query.ID]
		if !ok {
			missingQueryLabels++
			continue
		}
		evalQueries = append(evalQueries, label)
	}
	stats.MissingQueryLabels = missingQueryLabels
	docMap := make(map[string]sparseLexicalHashedVector, len(docs))
	for _, doc := range docs {
		docMap[doc.ID] = doc
	}
	missingDocLabels := countMissingSparseLexicalHashedRelevantDocs(evalQueries, qrels, docMap)
	stats.MissingDocLabels = missingDocLabels
	if missingQueryLabels > 0 || missingDocLabels > 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical hash head labels missing required qrels coverage: query_labels=%d relevant_doc_labels=%d", missingQueryLabels, missingDocLabels)
	}
	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, err := computeSparseLexicalHashRetrievalQuality(ctx, evalQueries, docs, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated labels")
	}
	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docs))
	return RetrievalEvalMetrics{
		Schema:  RetrievalEvalMetricsSchema,
		Dataset: cfg.DatasetName,
		Backend: "sparse_lexical_hash_head",
		Inputs: RetrievalEvalInputMetrics{
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			LabelPath:     cfg.LabelsPath,
			HeadPath:      cfg.HeadPath,
			Documents:     len(docs),
			Queries:       evaluatedQueries,
			RelevantPairs: relevantPairs,
			ScoredPairs:   scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			TopK:       cfg.TopK,
			MaxQueries: cfg.MaxQueries,
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: loadDuration.Seconds(),
			QueryEmbedSeconds:    loadDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docs)), loadDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(evalQueries)), loadDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText: skippedQueries,
		},
		SparseLexical: &stats,
	}, nil
}

// EvaluateSparseLexicalHashHeadVectorHybrid evaluates precomputed dense vectors
// fused with the experimental hashed-bin sparse lexical sidecar.
func EvaluateSparseLexicalHashHeadVectorHybrid(ctx context.Context, cfg SparseLexicalHashHeadEvalConfig) (RetrievalEvalMetrics, error) {
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" || cfg.LabelsPath == "" || cfg.HeadPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, qrels, labels, and head-json paths are required")
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
	hybridCfg, err := normalizeRetrievalEvalHybridConfig(cfg.Hybrid)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	start := time.Now()
	head, err := ReadSparseLexicalHashHead(cfg.HeadPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if cfg.DatasetName != "" && head.Dataset != "" && head.Dataset != cfg.DatasetName {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical hash head dataset=%q, want %q", head.Dataset, cfg.DatasetName)
	}
	if cfg.Split != "" && head.Split != "" && head.Split != cfg.Split {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical hash head split=%q, want %q", head.Split, cfg.Split)
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
	labels, stats, err := readSparseLexicalLabelFile(cfg.LabelsPath, cfg.DatasetName, cfg.Split)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	sparseDocs, sparseQueryMap, err := sparseLexicalHashVectors(labels, head.Hashing.Bins, &stats)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(sparseDocs) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no document sparse lexical labels found")
	}
	if len(sparseQueryMap) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no query sparse lexical labels found")
	}
	stats.Representation = "experimental_hashed_sparse_lexical_head"
	stats.LabelsPath = cfg.LabelsPath
	stats.HeadPath = cfg.HeadPath
	stats.HashBins = head.Hashing.Bins

	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read document vectors: %w", err)
	}
	if len(docVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no vectors for the evaluated corpus")
	}
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

	sparseDocMap := make(map[string]sparseLexicalHashedVector, len(sparseDocs))
	for _, doc := range sparseDocs {
		sparseDocMap[doc.ID] = doc
	}
	evalSparseDocs := make([]sparseLexicalHashedVector, 0, len(docVectors))
	for _, doc := range docVectors {
		sparseDoc, ok := sparseDocMap[doc.ID]
		if !ok {
			continue
		}
		evalSparseDocs = append(evalSparseDocs, sparseDoc)
	}
	evalSparseQueries := make(map[string]sparseLexicalHashedVector, len(queryVectors))
	for _, query := range queryVectors {
		sparseQuery, ok := sparseQueryMap[query.ID]
		if !ok {
			continue
		}
		evalSparseQueries[query.ID] = sparseQuery
	}
	requiredSparseQueries := make([]sparseLexicalHashedVector, 0, len(queries))
	missingQueryLabels := 0
	for _, query := range queries {
		if sparseQuery, ok := sparseQueryMap[query.ID]; ok {
			requiredSparseQueries = append(requiredSparseQueries, sparseQuery)
		} else {
			missingQueryLabels++
		}
	}
	stats.MissingQueryLabels = missingQueryLabels
	stats.MissingDocLabels = countMissingSparseLexicalHashedRelevantDocs(requiredSparseQueries, qrels, sparseDocMap)
	if stats.MissingQueryLabels > 0 || stats.MissingDocLabels > 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical hash head labels missing required qrels/vector coverage: query_labels=%d relevant_doc_labels=%d", stats.MissingQueryLabels, stats.MissingDocLabels)
	}
	loadDuration := time.Since(loadStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeSparseLexicalHashVectorHybridQuality(ctx, queryVectors, docVectors, evalSparseQueries, evalSparseDocs, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath, hybridCfg)
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
		Backend:  "sparse_lexical_hash_head_vectors_hybrid",
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:      cfg.CorpusPath,
			QueriesPath:     cfg.QueriesPath,
			QrelsPath:       cfg.QrelsPath,
			LabelPath:       cfg.LabelsPath,
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
			Hybrid:     retrievalEvalHybridMetrics(hybridCfg),
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

func ReadSparseLexicalHashHead(path string) (SparseLexicalHashHead, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SparseLexicalHashHead{}, fmt.Errorf("read sparse lexical hash head: %w", err)
	}
	var head SparseLexicalHashHead
	if err := json.Unmarshal(data, &head); err != nil {
		return SparseLexicalHashHead{}, fmt.Errorf("decode sparse lexical hash head: %w", err)
	}
	if head.Schema != SparseLexicalHashHeadSchema {
		return SparseLexicalHashHead{}, fmt.Errorf("sparse lexical hash head has unsupported schema %q", head.Schema)
	}
	if head.Hashing.Algorithm != "fnv1a32" {
		return SparseLexicalHashHead{}, fmt.Errorf("sparse lexical hash head has unsupported hashing algorithm %q", head.Hashing.Algorithm)
	}
	if head.Hashing.Bins <= 0 {
		return SparseLexicalHashHead{}, fmt.Errorf("sparse lexical hash head hash bins must be positive, got %d", head.Hashing.Bins)
	}
	if uint64(head.Hashing.Bins) > uint64(math.MaxUint32) {
		return SparseLexicalHashHead{}, fmt.Errorf("sparse lexical hash head hash bins must be <= %d, got %d", uint64(math.MaxUint32), head.Hashing.Bins)
	}
	return head, nil
}

func fillSparseLexicalHashStats(labels sparseLexicalLabelSet, hashBins int, stats *SparseLexicalLabelEvalStats) error {
	_, _, err := sparseLexicalHashVectors(labels, hashBins, stats)
	return err
}

func sparseLexicalHashVectors(labels sparseLexicalLabelSet, hashBins int, stats *SparseLexicalLabelEvalStats) ([]sparseLexicalHashedVector, map[string]sparseLexicalHashedVector, error) {
	docs := make([]sparseLexicalHashedVector, 0, len(labels.documentsOrdered))
	queries := make(map[string]sparseLexicalHashedVector, len(labels.queries))
	var docHashNNZ, queryHashNNZ int
	stats.DocumentMergedBins = 0
	stats.QueryMergedBins = 0
	stats.DocumentMaxHashNNZ = 0
	stats.QueryMaxHashNNZ = 0
	for _, record := range labels.documentsOrdered {
		vec, merged, err := sparseLexicalHashRecord(record, hashBins)
		if err != nil {
			return nil, nil, err
		}
		docs = append(docs, vec)
		docHashNNZ += len(vec.Terms)
		stats.DocumentMergedBins += merged
		if len(vec.Terms) > stats.DocumentMaxHashNNZ {
			stats.DocumentMaxHashNNZ = len(vec.Terms)
		}
	}
	queryIDs := make([]string, 0, len(labels.queries))
	for id := range labels.queries {
		queryIDs = append(queryIDs, id)
	}
	slices.Sort(queryIDs)
	for _, id := range queryIDs {
		vec, merged, err := sparseLexicalHashRecord(labels.queries[id], hashBins)
		if err != nil {
			return nil, nil, err
		}
		queries[id] = vec
		queryHashNNZ += len(vec.Terms)
		stats.QueryMergedBins += merged
		if len(vec.Terms) > stats.QueryMaxHashNNZ {
			stats.QueryMaxHashNNZ = len(vec.Terms)
		}
	}
	if len(docs) > 0 {
		stats.DocumentAvgHashNNZ = float64(docHashNNZ) / float64(len(docs))
	}
	if len(queries) > 0 {
		stats.QueryAvgHashNNZ = float64(queryHashNNZ) / float64(len(queries))
	}
	return docs, queries, nil
}

func sparseLexicalHashRecord(record SparseLexicalLabelRecord, hashBins int) (sparseLexicalHashedVector, int, error) {
	if hashBins <= 0 {
		return sparseLexicalHashedVector{}, 0, fmt.Errorf("hash bins must be positive, got %d", hashBins)
	}
	weights := make(map[uint32]float64, len(record.Terms))
	for i, term := range record.Terms {
		bin, err := sparseLexicalTermHashBin(term, hashBins)
		if err != nil {
			return sparseLexicalHashedVector{}, 0, fmt.Errorf("sparse lexical label %s %q term %d: %w", record.RecordType, record.ID, i, err)
		}
		weights[bin] += term.Weight
	}
	terms := make([]sparseLexicalHashedTerm, 0, len(weights))
	for bin, weight := range weights {
		terms = append(terms, sparseLexicalHashedTerm{Bin: bin, Weight: weight})
	}
	slices.SortFunc(terms, func(a, b sparseLexicalHashedTerm) int {
		if a.Bin < b.Bin {
			return -1
		}
		if a.Bin > b.Bin {
			return 1
		}
		return 0
	})
	return sparseLexicalHashedVector{ID: record.ID, Terms: terms}, len(record.Terms) - len(terms), nil
}

func sparseLexicalTermHashBin(term SparseLexicalLabelTerm, hashBins int) (uint32, error) {
	expectedPtr := sparseLexicalHashBin(term.Term, hashBins)
	if expectedPtr == nil {
		return 0, fmt.Errorf("hash bins must be positive")
	}
	expected := *expectedPtr
	if term.HashBin == nil {
		if term.Term == "" {
			return 0, fmt.Errorf("term text is required when hash_bin is absent")
		}
		return expected, nil
	}
	if *term.HashBin >= uint32(hashBins) {
		return 0, fmt.Errorf("hash_bin=%d outside hash_bins=%d", *term.HashBin, hashBins)
	}
	if term.Term != "" && *term.HashBin != expected {
		return 0, fmt.Errorf("hash_bin=%d incompatible with fnv1a32(%q) %% %d = %d", *term.HashBin, term.Term, hashBins, expected)
	}
	return *term.HashBin, nil
}

func countMissingSparseLexicalHashedRelevantDocs(queries []sparseLexicalHashedVector, qrels retrievalQrels, documents map[string]sparseLexicalHashedVector) int {
	missing := 0
	for _, query := range queries {
		for docID := range qrels[query.ID] {
			if _, ok := documents[docID]; !ok {
				missing++
			}
		}
	}
	return missing
}

func computeSparseLexicalHashRetrievalQuality(ctx context.Context, queries, docs []sparseLexicalHashedVector, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string) (RetrievalEvalQualityMetrics, int, int, error) {
	if topK < 100 {
		topK = 100
	}
	writer, err := newRetrievalPerQueryWriter(perQueryJSONLPath)
	if err != nil {
		return RetrievalEvalQualityMetrics{}, 0, 0, err
	}
	defer writer.Close()
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return RetrievalEvalQualityMetrics{}, 0, 0, err
		}
		rels := qrels[query.ID]
		if len(rels) == 0 {
			continue
		}
		scores := topSparseLexicalHashScores(query.Terms, docs, topK)
		evaluatedQueries++
		relevantPairs += len(rels)
		row := buildRetrievalPerQueryRow(datasetName, query.ID, scores, rels)
		addRetrievalPerQueryQuality(&totals, row.Quality)
		if err := writer.Write(row); err != nil {
			return RetrievalEvalQualityMetrics{}, 0, 0, err
		}
	}
	if err := writer.Close(); err != nil {
		return RetrievalEvalQualityMetrics{}, 0, 0, err
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, nil
}

func computeSparseLexicalHashVectorHybridQuality(ctx context.Context, queries, docs []retrievalVectorRecord, sparseQueries map[string]sparseLexicalHashedVector, sparseDocs []sparseLexicalHashedVector, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string, cfg RetrievalEvalHybridConfig) (RetrievalEvalQualityMetrics, int, int, int, int, error) {
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
		denseScores := topRetrievalScores(query.Vector, docs, topK)
		sparseScores := topSparseLexicalHashScores(sparseQuery.Terms, sparseDocs, topK)
		scores := fuseHybridScores(denseScores, sparseScores, topK, cfg)
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

func topSparseLexicalHashScores(query []sparseLexicalHashedTerm, docs []sparseLexicalHashedVector, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, doc := range docs {
		score := retrievalScoredDoc{ID: doc.ID, Score: float32(SparseLexicalHashDot(query, doc.Terms))}
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
	slices.SortFunc(scores, func(a, b retrievalScoredDoc) int {
		if retrievalScoreBetter(a, b) {
			return -1
		}
		if retrievalScoreBetter(b, a) {
			return 1
		}
		return 0
	})
	return scores
}

func SparseLexicalHashDot(query, document []sparseLexicalHashedTerm) float64 {
	docWeights := make(map[uint32]float64, len(document))
	for _, term := range document {
		docWeights[term.Bin] += term.Weight
	}
	var score float64
	for _, term := range query {
		score += term.Weight * docWeights[term.Bin]
	}
	return score
}
