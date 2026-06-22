package eosruntime

import (
	"bufio"
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"
)

const SparseLexicalLabelsSchema = "manta.sparse_lexical_labels.v1"

type SparseLexicalLabelExportConfig struct {
	DatasetName      string
	Split            string
	CorpusPath       string
	QueriesPath      string
	QrelsPath        string
	OutputPath       string
	ManifestPath     string
	TopTerms         int
	HashBins         int
	MaxDocs          int
	MaxQueries       int
	OracleTopK       int
	OracleMaxQueries int
}

type SparseLexicalLabelEvalConfig struct {
	DatasetName       string
	Split             string
	QueriesPath       string
	QrelsPath         string
	LabelsPath        string
	TopK              int
	MaxQueries        int
	PerQueryJSONLPath string
}

type SparseLexicalLabelExportSummary struct {
	Schema       string                         `json:"schema"`
	Dataset      string                         `json:"dataset"`
	Split        string                         `json:"split"`
	Inputs       SparseLexicalLabelExportInputs `json:"inputs"`
	Config       SparseLexicalLabelExportParams `json:"config"`
	Stats        SparseLexicalLabelExportStats  `json:"stats"`
	Tokenizer    SparseLexicalBM25Tokenizer     `json:"tokenizer"`
	BM25         SparseLexicalBM25Params        `json:"bm25"`
	Ordering     string                         `json:"ordering"`
	Hashing      SparseLexicalHashing           `json:"hashing,omitempty"`
	Oracle       SparseLexicalOracleSummary     `json:"oracle"`
	OutputPath   string                         `json:"output_path"`
	ManifestPath string                         `json:"manifest_path,omitempty"`
}

type SparseLexicalLabelExportInputs struct {
	CorpusPath  string `json:"corpus_path"`
	QueriesPath string `json:"queries_path"`
	QrelsPath   string `json:"qrels_path"`
}

type SparseLexicalLabelExportParams struct {
	TopTerms         int `json:"top_terms"`
	HashBins         int `json:"hash_bins,omitempty"`
	MaxDocs          int `json:"max_docs,omitempty"`
	MaxQueries       int `json:"max_queries,omitempty"`
	OracleTopK       int `json:"oracle_top_k"`
	OracleMaxQueries int `json:"oracle_max_queries,omitempty"`
}

type SparseLexicalLabelExportStats struct {
	Documents          int     `json:"documents"`
	Queries            int     `json:"queries"`
	DocumentAvgNNZ     float64 `json:"document_avg_nonzeros"`
	DocumentMaxNNZ     int     `json:"document_max_nonzeros"`
	QueryAvgNNZ        float64 `json:"query_avg_nonzeros"`
	QueryMaxNNZ        int     `json:"query_max_nonzeros"`
	DocumentTruncated  int     `json:"document_truncated_records"`
	QueryTruncated     int     `json:"query_truncated_records"`
	DocumentOmitted    int     `json:"document_omitted_terms"`
	QueryOmitted       int     `json:"query_omitted_terms"`
	VocabularyTerms    int     `json:"vocabulary_terms"`
	HashedCollisions   int     `json:"hashed_collisions,omitempty"`
	HashedCollisionPct float64 `json:"hashed_collision_pct,omitempty"`
}

type SparseLexicalLabelEvalStats struct {
	Representation     string  `json:"representation"`
	LabelsPath         string  `json:"labels_path"`
	HeadPath           string  `json:"head_path,omitempty"`
	HashBins           int     `json:"hash_bins,omitempty"`
	DocumentLabels     int     `json:"document_labels"`
	QueryLabels        int     `json:"query_labels"`
	DocumentAvgNNZ     float64 `json:"document_avg_nonzeros"`
	DocumentMaxNNZ     int     `json:"document_max_nonzeros"`
	QueryAvgNNZ        float64 `json:"query_avg_nonzeros"`
	QueryMaxNNZ        int     `json:"query_max_nonzeros"`
	DocumentAvgHashNNZ float64 `json:"document_avg_hashed_nonzeros,omitempty"`
	DocumentMaxHashNNZ int     `json:"document_max_hashed_nonzeros,omitempty"`
	QueryAvgHashNNZ    float64 `json:"query_avg_hashed_nonzeros,omitempty"`
	QueryMaxHashNNZ    int     `json:"query_max_hashed_nonzeros,omitempty"`
	DocumentMergedBins int     `json:"document_merged_bins,omitempty"`
	QueryMergedBins    int     `json:"query_merged_bins,omitempty"`
	ScoreThreshold     float64 `json:"score_threshold,omitempty"`
	MissingQueryLabels int     `json:"missing_query_labels"`
	MissingDocLabels   int     `json:"missing_doc_labels"`
}

type SparseLexicalBM25Tokenizer struct {
	Name        string `json:"name"`
	Lowercase   bool   `json:"lowercase"`
	TokenChars  string `json:"token_chars"`
	EmptyDocPad bool   `json:"empty_doc_pad"`
}

type SparseLexicalBM25Params struct {
	K1        float64 `json:"k1"`
	B         float64 `json:"b"`
	AvgLength float64 `json:"avg_length"`
	Documents int     `json:"documents"`
}

type SparseLexicalHashing struct {
	Algorithm string `json:"algorithm"`
	Bins      int    `json:"bins"`
	Seed      string `json:"seed"`
}

type SparseLexicalOracleSummary struct {
	Queries                  int     `json:"queries"`
	RelevantPairs            int     `json:"relevant_pairs"`
	TopK                     int     `json:"top_k"`
	ReconstructionTerms      string  `json:"reconstruction_terms"`
	ExportedTermsExact       bool    `json:"exported_terms_exact"`
	MaxAbsScoreDelta         float64 `json:"max_abs_score_delta"`
	ExactScoreReconstruction bool    `json:"exact_score_reconstruction"`
	NDCGAt10                 float64 `json:"ndcg_at_10"`
	RecallAt100              float64 `json:"recall_at_100"`
}

type SparseLexicalLabelRecord struct {
	Schema     string                   `json:"schema"`
	RecordType string                   `json:"record_type"`
	Dataset    string                   `json:"dataset"`
	Split      string                   `json:"split"`
	ID         string                   `json:"id"`
	NonZeros   int                      `json:"nonzeros"`
	Terms      []SparseLexicalLabelTerm `json:"terms"`
}

type SparseLexicalLabelTerm struct {
	Term    string  `json:"term"`
	Weight  float64 `json:"weight"`
	HashBin *uint32 `json:"hash_bin,omitempty"`
}

// EvaluateSparseLexicalLabels evaluates exported capped sparse lexical labels over BEIR qrels.
func EvaluateSparseLexicalLabels(ctx context.Context, cfg SparseLexicalLabelEvalConfig) (RetrievalEvalMetrics, error) {
	if cfg.QueriesPath == "" || cfg.QrelsPath == "" || cfg.LabelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("queries, qrels, and labels paths are required")
	}
	if cfg.TopK == 0 {
		cfg.TopK = 100
	}
	if cfg.TopK < 100 {
		return RetrievalEvalMetrics{}, fmt.Errorf("top-k must be at least 100 for recall_at_100, got %d", cfg.TopK)
	}
	start := time.Now()
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
	loadDuration := time.Since(loadStart)
	if len(labels.documents) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no document sparse lexical labels found")
	}
	if len(labels.queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no query sparse lexical labels found")
	}
	stats.Representation = "capped_exported_sparse_lexical_labels"
	stats.LabelsPath = cfg.LabelsPath

	evalQueries := make([]SparseLexicalLabelRecord, 0, len(queries))
	missingQueryLabels := 0
	for _, query := range queries {
		label, ok := labels.queries[query.ID]
		if !ok {
			missingQueryLabels++
			continue
		}
		evalQueries = append(evalQueries, label)
	}
	stats.MissingQueryLabels = missingQueryLabels
	missingDocLabels := countMissingSparseLexicalRelevantDocs(evalQueries, qrels, labels.documents)
	stats.MissingDocLabels = missingDocLabels
	if missingQueryLabels > 0 || missingDocLabels > 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("sparse lexical labels missing required qrels coverage: query_labels=%d relevant_doc_labels=%d", missingQueryLabels, missingDocLabels)
	}

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, err := computeSparseLexicalLabelRetrievalQuality(ctx, evalQueries, labels.documentsOrdered, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated labels")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(labels.documentsOrdered))
	return RetrievalEvalMetrics{
		Schema:  RetrievalEvalMetricsSchema,
		Dataset: cfg.DatasetName,
		Backend: "sparse_lexical_labels_capped",
		Inputs: RetrievalEvalInputMetrics{
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			LabelPath:     cfg.LabelsPath,
			Documents:     len(labels.documentsOrdered),
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
			DocumentsPerSecond:   ratePerSecond(float64(len(labels.documentsOrdered)), loadDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(evalQueries)), loadDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText: skippedQueries,
		},
		SparseLexical: &stats,
	}, nil
}

func ExportSparseLexicalLabels(ctx context.Context, cfg SparseLexicalLabelExportConfig) (SparseLexicalLabelExportSummary, error) {
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" || cfg.OutputPath == "" {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("corpus, queries, qrels, and output paths are required")
	}
	if cfg.TopTerms <= 0 {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("top terms must be positive, got %d", cfg.TopTerms)
	}
	if cfg.HashBins < 0 {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("hash bins must be non-negative, got %d", cfg.HashBins)
	}
	if cfg.HashBins > math.MaxUint32 {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("hash bins must be <= %d, got %d", uint64(math.MaxUint32), cfg.HashBins)
	}
	if cfg.ManifestPath != "" && sameSparseLexicalOutputPath(cfg.OutputPath, cfg.ManifestPath) {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("labels output path and manifest path must differ: %s", cfg.OutputPath)
	}
	if cfg.OracleTopK == 0 {
		cfg.OracleTopK = 100
	}
	if cfg.OracleTopK < 100 {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("oracle top-k must be at least 100 for recall_at_100, got %d", cfg.OracleTopK)
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return SparseLexicalLabelExportSummary{}, err
	}
	corpus, err := readBEIRCorpusWithRelevantIncludingEmpty(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return SparseLexicalLabelExportSummary{}, err
	}
	queries, _, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return SparseLexicalLabelExportSummary{}, err
	}
	if len(corpus) == 0 {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("no qrels queries found in queries file")
	}
	index, err := buildBM25Index(ctx, corpus)
	if err != nil {
		return SparseLexicalLabelExportSummary{}, err
	}

	f, err := os.Create(cfg.OutputPath)
	if err != nil {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("create sparse lexical labels: %w", err)
	}
	writer := bufio.NewWriter(f)
	enc := json.NewEncoder(writer)
	stats := sparseLexicalStatsAccumulator{}
	for _, doc := range index.Documents {
		if err := ctx.Err(); err != nil {
			_ = f.Close()
			return SparseLexicalLabelExportSummary{}, err
		}
		record := SparseLexicalLabelRecord{
			Schema:     SparseLexicalLabelsSchema,
			RecordType: "document",
			Dataset:    cfg.DatasetName,
			Split:      cfg.Split,
			ID:         doc.ID,
			Terms:      sparseLexicalDocumentTerms(doc, index, cfg.TopTerms, cfg.HashBins),
		}
		record.NonZeros = len(record.Terms)
		stats.addDocument(record.NonZeros, sparseLexicalDocumentTermCount(doc))
		if err := enc.Encode(record); err != nil {
			_ = f.Close()
			return SparseLexicalLabelExportSummary{}, fmt.Errorf("write document sparse labels: %w", err)
		}
	}
	tokenizedQueries, err := tokenizeBM25Queries(ctx, queries)
	if err != nil {
		_ = f.Close()
		return SparseLexicalLabelExportSummary{}, err
	}
	for _, query := range tokenizedQueries {
		if err := ctx.Err(); err != nil {
			_ = f.Close()
			return SparseLexicalLabelExportSummary{}, err
		}
		record := SparseLexicalLabelRecord{
			Schema:     SparseLexicalLabelsSchema,
			RecordType: "query",
			Dataset:    cfg.DatasetName,
			Split:      cfg.Split,
			ID:         query.ID,
			Terms:      sparseLexicalQueryTerms(query.Tokens, cfg.TopTerms, cfg.HashBins),
		}
		record.NonZeros = len(record.Terms)
		stats.addQuery(record.NonZeros, sparseLexicalQueryTermCount(query.Tokens))
		if err := enc.Encode(record); err != nil {
			_ = f.Close()
			return SparseLexicalLabelExportSummary{}, fmt.Errorf("write query sparse labels: %w", err)
		}
	}
	if err := writer.Flush(); err != nil {
		_ = f.Close()
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("flush sparse lexical labels: %w", err)
	}
	if err := f.Close(); err != nil {
		return SparseLexicalLabelExportSummary{}, fmt.Errorf("close sparse lexical labels: %w", err)
	}

	exportStats := stats.summary(len(index.DocFreq), sparseLexicalHashCollisionCount(index.DocFreq, cfg.HashBins))
	summary := SparseLexicalLabelExportSummary{
		Schema:  SparseLexicalLabelsSchema,
		Dataset: cfg.DatasetName,
		Split:   cfg.Split,
		Inputs: SparseLexicalLabelExportInputs{
			CorpusPath:  cfg.CorpusPath,
			QueriesPath: cfg.QueriesPath,
			QrelsPath:   cfg.QrelsPath,
		},
		Config: SparseLexicalLabelExportParams{
			TopTerms:         cfg.TopTerms,
			HashBins:         cfg.HashBins,
			MaxDocs:          cfg.MaxDocs,
			MaxQueries:       cfg.MaxQueries,
			OracleTopK:       cfg.OracleTopK,
			OracleMaxQueries: cfg.OracleMaxQueries,
		},
		Stats: exportStats,
		Tokenizer: SparseLexicalBM25Tokenizer{
			Name:        "eos_bm25_unicode_letter_digit_lowercase",
			Lowercase:   true,
			TokenChars:  "unicode_letter_or_digit",
			EmptyDocPad: true,
		},
		BM25: SparseLexicalBM25Params{
			K1:        index.K1,
			B:         index.B,
			AvgLength: index.AvgLength,
			Documents: len(index.Documents),
		},
		Ordering:     "documents then qrels-filtered queries in input order; terms by descending absolute weight, then term ascending; JSON encoder order is stable for structs",
		Oracle:       sparseLexicalOracle(ctx, tokenizedQueries, index, qrels, cfg.OracleTopK, cfg.OracleMaxQueries, exportStats),
		OutputPath:   cfg.OutputPath,
		ManifestPath: cfg.ManifestPath,
	}
	if cfg.HashBins > 0 {
		summary.Hashing = SparseLexicalHashing{
			Algorithm: "fnv32a(term) % hash_bins",
			Bins:      cfg.HashBins,
			Seed:      "go/hash/fnv offset basis",
		}
	}
	if cfg.ManifestPath != "" {
		data, err := json.MarshalIndent(summary, "", "  ")
		if err != nil {
			return SparseLexicalLabelExportSummary{}, err
		}
		data = append(data, '\n')
		if err := os.WriteFile(cfg.ManifestPath, data, 0o644); err != nil {
			return SparseLexicalLabelExportSummary{}, fmt.Errorf("write sparse lexical manifest: %w", err)
		}
	}
	return summary, nil
}

type sparseLexicalLabelSet struct {
	documents        map[string]SparseLexicalLabelRecord
	documentsOrdered []SparseLexicalLabelRecord
	queries          map[string]SparseLexicalLabelRecord
}

func readSparseLexicalLabelFile(path, datasetName, split string) (sparseLexicalLabelSet, SparseLexicalLabelEvalStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("open sparse lexical labels: %w", err)
	}
	defer f.Close()
	out := sparseLexicalLabelSet{
		documents: map[string]SparseLexicalLabelRecord{},
		queries:   map[string]SparseLexicalLabelRecord{},
	}
	var docNNZ, queryNNZ int
	stats := SparseLexicalLabelEvalStats{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		var record SparseLexicalLabelRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("decode sparse lexical labels line %d: %w", line, err)
		}
		if record.Schema != SparseLexicalLabelsSchema {
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("sparse lexical labels line %d has unsupported schema %q", line, record.Schema)
		}
		if record.ID == "" {
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("sparse lexical labels line %d is missing id", line)
		}
		if datasetName != "" && record.Dataset != "" && record.Dataset != datasetName {
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("sparse lexical labels line %d dataset=%q, want %q", line, record.Dataset, datasetName)
		}
		if split != "" && record.Split != "" && record.Split != split {
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("sparse lexical labels line %d split=%q, want %q", line, record.Split, split)
		}
		if record.NonZeros != len(record.Terms) {
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("sparse lexical labels line %d nonzeros=%d, terms=%d", line, record.NonZeros, len(record.Terms))
		}
		switch record.RecordType {
		case "document":
			if _, ok := out.documents[record.ID]; ok {
				return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("duplicate document sparse lexical label id %q", record.ID)
			}
			out.documents[record.ID] = record
			out.documentsOrdered = append(out.documentsOrdered, record)
			stats.DocumentLabels++
			docNNZ += record.NonZeros
			if record.NonZeros > stats.DocumentMaxNNZ {
				stats.DocumentMaxNNZ = record.NonZeros
			}
		case "query":
			if _, ok := out.queries[record.ID]; ok {
				return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("duplicate query sparse lexical label id %q", record.ID)
			}
			out.queries[record.ID] = record
			stats.QueryLabels++
			queryNNZ += record.NonZeros
			if record.NonZeros > stats.QueryMaxNNZ {
				stats.QueryMaxNNZ = record.NonZeros
			}
		default:
			return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("sparse lexical labels line %d has unsupported record_type %q", line, record.RecordType)
		}
	}
	if err := scanner.Err(); err != nil {
		return sparseLexicalLabelSet{}, SparseLexicalLabelEvalStats{}, fmt.Errorf("scan sparse lexical labels: %w", err)
	}
	if stats.DocumentLabels > 0 {
		stats.DocumentAvgNNZ = float64(docNNZ) / float64(stats.DocumentLabels)
	}
	if stats.QueryLabels > 0 {
		stats.QueryAvgNNZ = float64(queryNNZ) / float64(stats.QueryLabels)
	}
	return out, stats, nil
}

func countMissingSparseLexicalRelevantDocs(queries []SparseLexicalLabelRecord, qrels retrievalQrels, documents map[string]SparseLexicalLabelRecord) int {
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

func computeSparseLexicalLabelRetrievalQuality(ctx context.Context, queries, docs []SparseLexicalLabelRecord, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string) (RetrievalEvalQualityMetrics, int, int, error) {
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
		scores := topSparseLexicalLabelScores(query.Terms, docs, topK)
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

func topSparseLexicalLabelScores(query []SparseLexicalLabelTerm, docs []SparseLexicalLabelRecord, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, doc := range docs {
		score := retrievalScoredDoc{ID: doc.ID, Score: float32(SparseLexicalDot(query, doc.Terms))}
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

func sparseLexicalDocumentTerms(doc bm25Document, index bm25Index, topTerms, hashBins int) []SparseLexicalLabelTerm {
	terms := make([]SparseLexicalLabelTerm, 0, len(doc.TermFreq))
	for term, tf := range doc.TermFreq {
		if term == "" {
			continue
		}
		terms = append(terms, SparseLexicalLabelTerm{
			Term:    term,
			Weight:  bm25DocumentTermWeight(term, tf, doc, index),
			HashBin: sparseLexicalHashBin(term, hashBins),
		})
	}
	return boundSparseLexicalTerms(terms, topTerms)
}

func sparseLexicalQueryTerms(tokens []string, topTerms, hashBins int) []SparseLexicalLabelTerm {
	tf := make(map[string]int, len(tokens))
	for _, token := range tokens {
		if token != "" {
			tf[token]++
		}
	}
	terms := make([]SparseLexicalLabelTerm, 0, len(tf))
	for term, count := range tf {
		terms = append(terms, SparseLexicalLabelTerm{
			Term:    term,
			Weight:  float64(count),
			HashBin: sparseLexicalHashBin(term, hashBins),
		})
	}
	return boundSparseLexicalTerms(terms, topTerms)
}

func sparseLexicalDocumentTermCount(doc bm25Document) int {
	count := 0
	for term := range doc.TermFreq {
		if term != "" {
			count++
		}
	}
	return count
}

func sparseLexicalQueryTermCount(tokens []string) int {
	tf := make(map[string]bool, len(tokens))
	for _, token := range tokens {
		if token != "" {
			tf[token] = true
		}
	}
	return len(tf)
}

func bm25DocumentTermWeight(term string, tf int, doc bm25Document, index bm25Index) float64 {
	if tf == 0 || len(index.Documents) == 0 || index.AvgLength == 0 {
		return 0
	}
	df := float64(index.DocFreq[term])
	if df == 0 {
		return 0
	}
	nDocs := float64(len(index.Documents))
	idf := math.Log(1 + (nDocs-df+0.5)/(df+0.5))
	lengthNorm := index.K1 * (1 - index.B + index.B*float64(doc.Length)/index.AvgLength)
	tfWeight := (float64(tf) * (index.K1 + 1)) / (float64(tf) + lengthNorm)
	return idf * tfWeight
}

func SparseLexicalDot(query, document []SparseLexicalLabelTerm) float64 {
	docWeights := make(map[string]float64, len(document))
	for _, term := range document {
		docWeights[term.Term] += term.Weight
	}
	var score float64
	for _, term := range query {
		score += term.Weight * docWeights[term.Term]
	}
	return score
}

func boundSparseLexicalTerms(terms []SparseLexicalLabelTerm, topTerms int) []SparseLexicalLabelTerm {
	slices.SortFunc(terms, func(a, b SparseLexicalLabelTerm) int {
		aw := math.Abs(a.Weight)
		bw := math.Abs(b.Weight)
		if aw > bw {
			return -1
		}
		if aw < bw {
			return 1
		}
		if a.Term < b.Term {
			return -1
		}
		if a.Term > b.Term {
			return 1
		}
		return 0
	})
	if topTerms > 0 && len(terms) > topTerms {
		terms = terms[:topTerms]
	}
	return terms
}

func sparseLexicalHashBin(term string, hashBins int) *uint32 {
	if hashBins <= 0 {
		return nil
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(term))
	bin := h.Sum32() % uint32(hashBins)
	return &bin
}

func sparseLexicalHashCollisionCount(docFreq map[string]int, hashBins int) int {
	if hashBins <= 0 {
		return 0
	}
	seen := map[uint32]string{}
	collisions := 0
	for term := range docFreq {
		bin := *sparseLexicalHashBin(term, hashBins)
		if prev, ok := seen[bin]; ok && prev != term {
			collisions++
			continue
		}
		seen[bin] = term
	}
	return collisions
}

func sameSparseLexicalOutputPath(a, b string) bool {
	aa := filepath.Clean(a)
	bb := filepath.Clean(b)
	if abs, err := filepath.Abs(aa); err == nil {
		aa = abs
	}
	if abs, err := filepath.Abs(bb); err == nil {
		bb = abs
	}
	return aa == bb
}

func sparseLexicalOracle(ctx context.Context, queries []bm25Query, index bm25Index, qrels retrievalQrels, topK, maxQueries int, stats SparseLexicalLabelExportStats) SparseLexicalOracleSummary {
	docTerms := make(map[string][]SparseLexicalLabelTerm, len(index.Documents))
	for _, doc := range index.Documents {
		docTerms[doc.ID] = sparseLexicalDocumentTerms(doc, index, maxSparseLexicalTerms(), 0)
	}
	docIDSet := make(map[string]bool, len(index.Documents))
	for _, doc := range index.Documents {
		docIDSet[doc.ID] = true
	}
	var totals RetrievalEvalQualityMetrics
	var maxDelta float64
	evaluated := 0
	relevantPairs := 0
	for _, query := range queries {
		if maxQueries > 0 && evaluated >= maxQueries {
			break
		}
		if ctx.Err() != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			}
		}
		if len(filteredRels) == 0 {
			continue
		}
		queryTerms := sparseLexicalQueryTerms(query.Tokens, maxSparseLexicalTerms(), 0)
		scores := make([]retrievalScoredDoc, 0, len(index.Documents))
		for _, doc := range index.Documents {
			exact := scoreBM25Document(query.Tokens, doc, index)
			reconstructed := SparseLexicalDot(queryTerms, docTerms[doc.ID])
			if delta := math.Abs(exact - reconstructed); delta > maxDelta {
				maxDelta = delta
			}
			scores = append(scores, retrievalScoredDoc{ID: doc.ID, Score: float32(reconstructed)})
		}
		slices.SortFunc(scores, func(a, b retrievalScoredDoc) int {
			if retrievalScoreBetter(a, b) {
				return -1
			}
			if retrievalScoreBetter(b, a) {
				return 1
			}
			return 0
		})
		if topK > 0 && len(scores) > topK {
			scores = scores[:topK]
		}
		row := buildRetrievalPerQueryRow("", query.ID, scores, filteredRels)
		addRetrievalPerQueryQuality(&totals, row.Quality)
		evaluated++
		relevantPairs += len(filteredRels)
	}
	if evaluated > 0 {
		averageRetrievalQuality(&totals, evaluated)
	}
	return SparseLexicalOracleSummary{
		Queries:                  evaluated,
		RelevantPairs:            relevantPairs,
		TopK:                     topK,
		ReconstructionTerms:      "unbounded_internal",
		ExportedTermsExact:       stats.DocumentTruncated == 0 && stats.QueryTruncated == 0,
		MaxAbsScoreDelta:         maxDelta,
		ExactScoreReconstruction: maxDelta < 1e-9,
		NDCGAt10:                 totals.NDCGAt10,
		RecallAt100:              totals.RecallAt100,
	}
}

func maxSparseLexicalTerms() int {
	return int(^uint(0) >> 1)
}

type sparseLexicalStatsAccumulator struct {
	documents         int
	queries           int
	documentNNZ       int
	documentMax       int
	documentTruncated int
	documentOmitted   int
	queryNNZ          int
	queryMax          int
	queryTruncated    int
	queryOmitted      int
}

func (s *sparseLexicalStatsAccumulator) addDocument(nnz, unbounded int) {
	s.documents++
	s.documentNNZ += nnz
	if nnz > s.documentMax {
		s.documentMax = nnz
	}
	if unbounded > nnz {
		s.documentTruncated++
		s.documentOmitted += unbounded - nnz
	}
}

func (s *sparseLexicalStatsAccumulator) addQuery(nnz, unbounded int) {
	s.queries++
	s.queryNNZ += nnz
	if nnz > s.queryMax {
		s.queryMax = nnz
	}
	if unbounded > nnz {
		s.queryTruncated++
		s.queryOmitted += unbounded - nnz
	}
}

func (s sparseLexicalStatsAccumulator) summary(vocabularyTerms, collisions int) SparseLexicalLabelExportStats {
	stats := SparseLexicalLabelExportStats{
		Documents:         s.documents,
		Queries:           s.queries,
		DocumentMaxNNZ:    s.documentMax,
		QueryMaxNNZ:       s.queryMax,
		DocumentTruncated: s.documentTruncated,
		QueryTruncated:    s.queryTruncated,
		DocumentOmitted:   s.documentOmitted,
		QueryOmitted:      s.queryOmitted,
		VocabularyTerms:   vocabularyTerms,
		HashedCollisions:  collisions,
	}
	if s.documents > 0 {
		stats.DocumentAvgNNZ = float64(s.documentNNZ) / float64(s.documents)
	}
	if s.queries > 0 {
		stats.QueryAvgNNZ = float64(s.queryNNZ) / float64(s.queries)
	}
	if vocabularyTerms > 0 && collisions > 0 {
		stats.HashedCollisionPct = float64(collisions) / float64(vocabularyTerms)
	}
	return stats
}
