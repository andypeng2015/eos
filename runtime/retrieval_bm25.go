package eosruntime

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"
	"unicode"
)

const (
	defaultBM25K1 = 0.9
	defaultBM25B  = 0.4
)

type bm25Document struct {
	ID       string
	Length   int
	TermFreq map[string]int
}

type bm25Index struct {
	Documents []bm25Document
	DocFreq   map[string]int
	Postings  map[string][]int
	AvgLength float64
	K1        float64
	B         float64
	// DFPruneThreshold, when in (0, 1), causes candidate generation
	// (bm25CandidateDocIndices) to skip query terms whose document
	// frequency exceeds DFPruneThreshold * len(Documents) -- "stopword
	// like" terms whose postings lists dominate candidate-generation cost
	// against a large corpus without being selective. <= 0 or >= 1 means
	// "off" (unpruned, exhaustive candidate generation, the historical
	// behavior). Scoring itself is unaffected: pruning only changes which
	// documents are considered as candidates, not the BM25 formula applied
	// to whichever documents are scored.
	DFPruneThreshold float64
}

// EvaluateBM25Retrieval evaluates a BEIR-style split with a lexical BM25 baseline.
func EvaluateBM25Retrieval(ctx context.Context, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	start := time.Now()
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpus(cfg.CorpusPath, cfg.MaxDocs)
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

	indexStart := time.Now()
	index, err := buildBM25Index(ctx, corpus)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	indexDuration := time.Since(indexStart)

	queryStart := time.Now()
	tokenizedQueries, err := tokenizeBM25Queries(ctx, queries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queryDuration := time.Since(queryStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeBM25RetrievalQuality(ctx, tokenizedQueries, index, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(index.Documents))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  "bm25",
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:    cfg.CorpusPath,
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			Documents:     len(index.Documents),
			Queries:       evaluatedQueries,
			RelevantPairs: relevantPairs,
			ScoredPairs:   scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			BatchSize:  cfg.BatchSize,
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: indexDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(index.Documents)), indexDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(tokenizedQueries)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
		},
	}, nil
}

func buildBM25Index(ctx context.Context, records []retrievalTextRecord) (bm25Index, error) {
	index := bm25Index{
		Documents: make([]bm25Document, 0, len(records)),
		DocFreq:   map[string]int{},
		Postings:  map[string][]int{},
		K1:        defaultBM25K1,
		B:         defaultBM25B,
	}
	var totalLength int
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return bm25Index{}, err
		}
		tokens := tokenizeBM25Text(record.Text)
		if len(tokens) == 0 {
			tokens = []string{""}
		}
		tf := make(map[string]int, len(tokens))
		seen := map[string]bool{}
		for _, token := range tokens {
			tf[token]++
			if !seen[token] {
				index.DocFreq[token]++
				index.Postings[token] = append(index.Postings[token], len(index.Documents))
				seen[token] = true
			}
		}
		index.Documents = append(index.Documents, bm25Document{
			ID:       record.ID,
			Length:   len(tokens),
			TermFreq: tf,
		})
		totalLength += len(tokens)
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return bm25Index{}, err
			}
		}
	}
	if len(index.Documents) > 0 {
		index.AvgLength = float64(totalLength) / float64(len(index.Documents))
	}
	return index, nil
}

type bm25Query struct {
	ID     string
	Tokens []string
}

func tokenizeBM25Queries(ctx context.Context, records []retrievalTextRecord) ([]bm25Query, error) {
	out := make([]bm25Query, len(records))
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		out[i] = bm25Query{ID: record.ID, Tokens: tokenizeBM25Text(record.Text)}
	}
	return out, nil
}

func computeBM25RetrievalQuality(ctx context.Context, queries []bm25Query, index bm25Index, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string) (RetrievalEvalQualityMetrics, int, int, int, int, error) {
	docIDSet := make(map[string]bool, len(index.Documents))
	for _, doc := range index.Documents {
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
	// Reused across every query in this (sequential) loop instead of
	// allocating a fresh map[int]bool candidate-dedup set per query.
	scratch := newBM25CandidateScratch(len(index.Documents))
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
		scores := topBM25Scores(query.Tokens, index, topK, scratch)
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

// bm25CandidateScratch is a reusable, epoch-stamped dense accumulator used
// to deduplicate BM25 candidate document indices while unioning query-term
// postings lists. It replaces a fresh map[int]bool allocated on every query
// -- at hundreds-of-thousands-of-documents scale that's real hashing and
// allocation cost paid on every single query -- with one slice sized to the
// corpus that is "cleared" in O(1) by bumping an epoch counter instead of
// being reallocated or zeroed. A scratch instance is NOT safe for
// concurrent use: each parallel mining worker (mineBM25QueriesParallel)
// owns its own instance and reuses it across every query that worker
// processes; sequential callers (e.g. computeBM25RetrievalQuality) allocate
// one and reuse it across their whole query loop.
type bm25CandidateScratch struct {
	stamp []uint32
	epoch uint32
}

func newBM25CandidateScratch(numDocs int) *bm25CandidateScratch {
	if numDocs < 0 {
		numDocs = 0
	}
	return &bm25CandidateScratch{stamp: make([]uint32, numDocs)}
}

// reset lazily invalidates every previous mark in O(1) by advancing the
// epoch counter. It grows the backing slice if the corpus is larger than
// what this scratch was created for (defensive; corpus size is fixed
// within a single index/mining run) and only falls back to a real O(n)
// clear on the extremely unlikely uint32 epoch wraparound (~4.29 billion
// queries against one scratch instance).
func (s *bm25CandidateScratch) reset(numDocs int) {
	if len(s.stamp) < numDocs {
		grown := make([]uint32, numDocs)
		copy(grown, s.stamp)
		s.stamp = grown
	}
	s.epoch++
	if s.epoch == 0 {
		for i := range s.stamp {
			s.stamp[i] = 0
		}
		s.epoch = 1
	}
}

// markIfNew marks docIndex as a candidate for the current epoch, returning
// true the first time it is marked (i.e. it was not already a candidate).
//
// The out-of-range branch is not expected to be reachable in practice: every
// scratch is sized to (and reset()-grown to at least) len(index.Documents),
// and every docIndex callers pass in comes straight out of index.Postings,
// whose values are always valid indices into index.Documents. It is kept as
// a defensive fallback (report "new"/"not contained" rather than panicking)
// so a future caller that mismatches a scratch against a different index has
// a signpost here instead of a slice-bounds panic.
func (s *bm25CandidateScratch) markIfNew(docIndex int) bool {
	if docIndex < 0 || docIndex >= len(s.stamp) {
		return true
	}
	if s.stamp[docIndex] == s.epoch {
		return false
	}
	s.stamp[docIndex] = s.epoch
	return true
}

// contains reports whether docIndex was marked during the current epoch. See
// markIfNew's comment: the out-of-range branch is a defensive fallback for a
// scratch/index mismatch, not an expected path.
func (s *bm25CandidateScratch) contains(docIndex int) bool {
	if docIndex < 0 || docIndex >= len(s.stamp) {
		return false
	}
	return s.stamp[docIndex] == s.epoch
}

func topBM25Scores(queryTokens []string, index bm25Index, topK int, scratch *bm25CandidateScratch) []retrievalScoredDoc {
	if topK <= 0 || topK > len(index.Documents) {
		topK = len(index.Documents)
	}
	h := make(retrievalScoreHeap, 0, topK)
	qf := newBM25QueryFreq(queryTokens)
	candidates := bm25CandidateDocIndices(queryTokens, index, scratch)
	for _, docIndex := range candidates {
		doc := index.Documents[docIndex]
		score := retrievalScoredDoc{ID: doc.ID, Score: float32(scoreBM25DocumentQueryFreq(qf, doc, index))}
		pushBM25Score(&h, score, topK)
	}
	if len(h) < topK {
		for docIndex, doc := range index.Documents {
			if scratch.contains(docIndex) {
				continue
			}
			pushBM25Score(&h, retrievalScoredDoc{ID: doc.ID}, topK)
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

// dfPruneQueryTerms drops query tokens whose document frequency exceeds
// threshold * len(index.Documents) -- "stopword-like" terms that dominate
// candidate-generation cost (their postings lists are enormous against a
// large corpus) without being very selective, since their near-flat IDF
// weight rarely changes which documents end up ranked highest. threshold
// <= 0 or >= 1 disables pruning (queryTokens is returned unchanged). If
// every token would be pruned (e.g. a query made entirely of common
// terms), pruning is skipped for that query entirely so it never loses all
// of its candidates to this optimization.
func dfPruneQueryTerms(queryTokens []string, index bm25Index, threshold float64) []string {
	if threshold <= 0 || threshold >= 1 || len(index.Documents) == 0 {
		return queryTokens
	}
	maxDF := threshold * float64(len(index.Documents))
	kept := make([]string, 0, len(queryTokens))
	for _, token := range queryTokens {
		if token == "" {
			continue
		}
		if float64(index.DocFreq[token]) > maxDF {
			continue
		}
		kept = append(kept, token)
	}
	if len(kept) == 0 {
		return queryTokens
	}
	return kept
}

func bm25CandidateDocIndices(queryTokens []string, index bm25Index, scratch *bm25CandidateScratch) []int {
	scratch.reset(len(index.Documents))
	terms := dfPruneQueryTerms(queryTokens, index, index.DFPruneThreshold)
	candidates := make([]int, 0, len(terms))
	for _, token := range terms {
		if token == "" {
			continue
		}
		for _, docIndex := range index.Postings[token] {
			if scratch.markIfNew(docIndex) {
				candidates = append(candidates, docIndex)
			}
		}
	}
	return candidates
}

func pushBM25Score(h *retrievalScoreHeap, score retrievalScoredDoc, topK int) {
	if topK <= 0 {
		return
	}
	if len(*h) < topK {
		heap.Push(h, score)
		return
	}
	if retrievalScoreBetter(score, (*h)[0]) {
		(*h)[0] = score
		heap.Fix(h, 0)
	}
}

func topBM25NonPositiveScores(queryTokens []string, positiveIDs map[string]bool, index bm25Index, topK int, scratch *bm25CandidateScratch) []retrievalScoredDoc {
	if topK <= 0 || topK > len(index.Documents) {
		topK = len(index.Documents)
	}
	h := make(retrievalScoreHeap, 0, topK)
	qf := newBM25QueryFreq(queryTokens)
	candidates := bm25CandidateDocIndices(queryTokens, index, scratch)
	for _, docIndex := range candidates {
		doc := index.Documents[docIndex]
		if positiveIDs[doc.ID] {
			continue
		}
		score := retrievalScoredDoc{ID: doc.ID, Score: float32(scoreBM25DocumentQueryFreq(qf, doc, index))}
		pushBM25Score(&h, score, topK)
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

func topBM25NonPositiveTexts(queryTokens []string, positiveIDs map[string]bool, index bm25Index, docText map[string]string, topK int) []string {
	scratch := newBM25CandidateScratch(len(index.Documents))
	scores := topBM25NonPositiveScores(queryTokens, positiveIDs, index, topK, scratch)
	texts := make([]string, 0, len(scores))
	seen := map[string]bool{}
	for _, score := range scores {
		text := strings.TrimSpace(docText[score.ID])
		if text == "" || seen[text] {
			continue
		}
		seen[text] = true
		texts = append(texts, text)
	}
	return texts
}

// bm25QueryFreq is a query's tokens pre-aggregated into unique terms and
// their within-query counts, computed once per query. scoreBM25Document
// used to rebuild this as a fresh map on every call -- and it is called
// once per (query, candidate-document) pair, so for a query scored against
// thousands of BM25 candidates that was thousands of redundant map
// allocations of the exact same content. Hoisting it out to be built once
// per query (newBM25QueryFreq) and reused across every candidate document
// (scoreBM25DocumentQueryFreq) removes that redundant per-document work.
type bm25QueryFreq struct {
	terms []string
	freqs []int
}

func newBM25QueryFreq(queryTokens []string) bm25QueryFreq {
	if len(queryTokens) == 0 {
		return bm25QueryFreq{}
	}
	counts := make(map[string]int, len(queryTokens))
	terms := make([]string, 0, len(queryTokens))
	for _, token := range queryTokens {
		if token == "" {
			continue
		}
		if counts[token] == 0 {
			terms = append(terms, token)
		}
		counts[token]++
	}
	freqs := make([]int, len(terms))
	for i, token := range terms {
		freqs[i] = counts[token]
	}
	return bm25QueryFreq{terms: terms, freqs: freqs}
}

func scoreBM25DocumentQueryFreq(qf bm25QueryFreq, doc bm25Document, index bm25Index) float64 {
	if len(qf.terms) == 0 || len(index.Documents) == 0 || index.AvgLength == 0 {
		return 0
	}
	var score float64
	nDocs := float64(len(index.Documents))
	lengthNorm := index.K1 * (1 - index.B + index.B*float64(doc.Length)/index.AvgLength)
	for i, token := range qf.terms {
		tf := doc.TermFreq[token]
		if tf == 0 {
			continue
		}
		df := float64(index.DocFreq[token])
		idf := math.Log(1 + (nDocs-df+0.5)/(df+0.5))
		tfWeight := (float64(tf) * (index.K1 + 1)) / (float64(tf) + lengthNorm)
		score += float64(qf.freqs[i]) * idf * tfWeight
	}
	return score
}

// scoreBM25Document scores a single document against queryTokens. It is the
// stable, simple entry point kept for one-off scoring calls (e.g. scoring a
// query's own positive document, or the sparse-lexical-label oracle in
// retrieval_sparse_lexical_labels.go); hot loops that score many documents
// per query should build a bm25QueryFreq once via newBM25QueryFreq and call
// scoreBM25DocumentQueryFreq directly instead of calling this per document.
func scoreBM25Document(queryTokens []string, doc bm25Document, index bm25Index) float64 {
	return scoreBM25DocumentQueryFreq(newBM25QueryFreq(queryTokens), doc, index)
}

func tokenizeBM25Text(text string) []string {
	text = strings.ToLower(text)
	tokens := []string{}
	var b strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			continue
		}
		if b.Len() > 0 {
			tokens = append(tokens, b.String())
			b.Reset()
		}
	}
	if b.Len() > 0 {
		tokens = append(tokens, b.String())
	}
	return tokens
}
