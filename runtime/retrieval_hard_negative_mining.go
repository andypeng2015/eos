package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"sync"
)

// DefaultBM25MiningDFPruneThreshold is the CLI-facing default for
// --df-prune-threshold: the literature-typical 0.10 (query terms whose
// document frequency exceeds 10% of the corpus are skipped during BM25
// candidate generation). This was validated, not assumed: a quality check
// mining all 500 real queries in the shared benchmark pool
// (runs/bm25-mining-benchmark-v1-20260710T192830Z/bench-dataset, a real
// 300k-document MS MARCO sample) with --negatives 8 found 500/500 (100%)
// of mined top-8 hard-negative sets identical between exhaustive
// (threshold=0, unpruned) and 0.10-pruned mining -- exact ID-set matches,
// and in fact byte-identical rank order too -- comfortably clearing the
// >=95% bar this change required before flipping the default on. See
// runs/bm25-miner-optimization-v1-20260710T205739Z/quality/ for the raw
// per-query outputs this was computed from. Operators can still force the
// old exhaustive behavior with --df-prune-threshold 0 (or any value >= 1).
const DefaultBM25MiningDFPruneThreshold = 0.10

// DefaultBM25MiningWorkers caps the automatic --mining-workers default at 8
// even on much wider machines: BM25 mining is CPU-bound per query but each
// query's own work is small, so beyond a handful of workers the shared
// read-only index's cache behavior and Go's scheduler overhead dominate
// any further parallel gain.
const DefaultBM25MiningWorkers = 8

// RetrievalHardNegativeMiningConfig describes BEIR hard-negative mining.
type RetrievalHardNegativeMiningConfig struct {
	DatasetName          string
	CorpusPath           string
	QueriesPath          string
	QrelsPath            string
	NegativesPerPositive int
	CandidateTopK        int
	BatchSize            int
	MaxExamples          int
	MaxDocs              int
	MaxQueries           int
	RoleMode             string
	// DFPruneThreshold is honored by MineBM25TextHardNegatives only (the
	// lexical BM25 miner). It is a fraction of the corpus (0, 1); query
	// terms whose document frequency exceeds that fraction are skipped
	// during candidate generation. The Go zero value (0) means "off"
	// (exhaustive candidate generation, identical to the pre-pruning
	// behavior) so existing callers that do not set this field are
	// unaffected. Values <= 0 or >= 1 also mean "off". See
	// DefaultBM25MiningDFPruneThreshold for the CLI's own default and the
	// measurement backing it.
	DFPruneThreshold float64
	// MiningWorkers is honored by MineBM25TextHardNegatives only. It sets
	// how many queries are mined concurrently against the shared read-only
	// BM25 index. <= 0 means "auto": min(DefaultBM25MiningWorkers,
	// GOMAXPROCS). Output ordering and summary counts are identical
	// regardless of this value (results are collected by query index, not
	// completion order).
	MiningWorkers int
}

type RetrievalHardNegativeMiningSummary struct {
	DatasetName                           string
	RoleMode                              string `json:"role_mode,omitempty"`
	Queries                               int
	PositivePairs                         int
	Examples                              int
	Negatives                             int
	SkippedQueriesNoText                  int
	SkippedPositiveDocs                   int
	SkippedQueriesNoNegative              int
	DuplicatePositiveTextNegativesSkipped int
	// DFPruneThreshold and MiningWorkers echo the effective (post-default)
	// config MineBM25TextHardNegatives actually ran with, for provenance
	// on multi-hour production mining jobs. Zero value for both fields on
	// summaries returned by MineModelTextHardNegatives, which does not use
	// either setting.
	DFPruneThreshold float64 `json:"df_prune_threshold,omitempty"`
	MiningWorkers    int     `json:"mining_workers,omitempty"`
}

type retrievalPositiveDoc struct {
	ID    string
	Score float64
	Text  string
}

type retrievalScoredText struct {
	ID    string
	Score float32
	Text  string
}

type retrievalMiningCandidateResult struct {
	Candidates                            []retrievalScoredText
	DuplicatePositiveTextNegativesSkipped int
}

// MineBM25TextHardNegatives mines text hard negatives from BEIR data using the same BM25 scorer as the lexical baseline.
func MineBM25TextHardNegatives(ctx context.Context, cfg RetrievalHardNegativeMiningConfig) ([]EmbeddingTextHardNegativeExample, RetrievalHardNegativeMiningSummary, error) {
	cfg = normalizeRetrievalHardNegativeMiningConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	corpus, err := readBEIRCorpus(cfg.CorpusPath, cfg.MaxDocs)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	if len(corpus) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("no qrels queries found in queries file")
	}
	index, err := buildBM25Index(ctx, corpus)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	index.DFPruneThreshold = cfg.DFPruneThreshold
	docText := make(map[string]string, len(corpus))
	for _, doc := range corpus {
		docText[doc.ID] = doc.Text
	}
	indexDocs := make(map[string]bm25Document, len(index.Documents))
	for _, doc := range index.Documents {
		indexDocs[doc.ID] = doc
	}
	// summary.MiningWorkers must reflect what mining actually ran with, not
	// the requested/defaulted cfg.MiningWorkers: mineBM25QueriesParallel
	// clamps its worker pool to len(queries) (a worker with no query to
	// process would be wasted), so for small runs (queries < MiningWorkers)
	// the requested value overstates the concurrency that was actually
	// used. effectiveBM25MiningWorkers is the single source of truth both
	// the worker pool and this summary (and, via the CLI's config echo,
	// operators) derive from, so they can never diverge.
	effectiveWorkers := effectiveBM25MiningWorkers(cfg.MiningWorkers, len(queries))
	summary := RetrievalHardNegativeMiningSummary{
		DatasetName:          cfg.DatasetName,
		RoleMode:             cfg.RoleMode,
		Queries:              len(queries),
		SkippedQueriesNoText: skippedQueries,
		DFPruneThreshold:     cfg.DFPruneThreshold,
		MiningWorkers:        effectiveWorkers,
	}

	results, err := mineBM25QueriesParallel(ctx, queries, qrels, docText, indexDocs, index, cfg)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}

	// Sequential, index-ordered merge: this is the one place that decides
	// what actually lands in the output, so behavior (including the exact
	// point cfg.MaxExamples stops accepting further queries/positives) is
	// identical no matter how many workers computed the per-query results
	// above or in what order they finished.
	out := []EmbeddingTextHardNegativeExample{}
	for _, result := range results {
		if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
			break
		}
		summary.SkippedPositiveDocs += result.SkippedPositiveDocs
		summary.DuplicatePositiveTextNegativesSkipped += result.DuplicatePositiveTextNegativesSkipped
		summary.SkippedQueriesNoNegative += result.SkippedQueriesNoNegative
		for _, example := range result.Examples {
			if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
				break
			}
			out = append(out, example)
			summary.PositivePairs++
			summary.Negatives += len(example.Negatives)
		}
	}
	summary.Examples = len(out)
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("BM25 hard-negative mining produced no examples")
	}
	return out, summary, nil
}

// bm25MiningQueryResult is the self-contained output of mining a single
// query, computed by one worker in mineBM25QueriesParallel against the
// shared read-only index. It intentionally mirrors every side effect the
// old sequential per-query loop body had (skip counters plus emitted
// examples) so the sequential merge in MineBM25TextHardNegatives can
// reproduce the exact old output byte-for-byte regardless of worker count.
type bm25MiningQueryResult struct {
	Examples                              []EmbeddingTextHardNegativeExample
	SkippedPositiveDocs                   int
	DuplicatePositiveTextNegativesSkipped int
	SkippedQueriesNoNegative              int
}

// mineBM25Query mines one query's hard negatives against the shared
// read-only index/docText/indexDocs/qrels. scratch is private,
// caller-owned, mutable per-worker state; everything else this function
// touches is read-only after MineBM25TextHardNegatives finished building
// it, so concurrent calls (one per worker, never sharing a scratch) are
// safe.
func mineBM25Query(query retrievalTextRecord, rels map[string]float64, docText map[string]string, indexDocs map[string]bm25Document, index bm25Index, cfg RetrievalHardNegativeMiningConfig, scratch *bm25CandidateScratch) bm25MiningQueryResult {
	var result bm25MiningQueryResult
	positives, skippedPositiveDocs := bm25MiningPositiveDocs(rels, docText)
	result.SkippedPositiveDocs = skippedPositiveDocs
	if len(positives) == 0 {
		return result
	}
	positiveIDs := make(map[string]bool, len(positives))
	for _, positive := range positives {
		positiveIDs[positive.ID] = true
	}
	positiveTextFingerprints := retrievalPositiveTextFingerprints(rels, docText)
	queryTokens := tokenizeBM25Text(query.Text)
	candidateResult := bm25MiningNegativeCandidates(queryTokens, positiveIDs, positiveTextFingerprints, index, docText, cfg, scratch)
	result.DuplicatePositiveTextNegativesSkipped = candidateResult.DuplicatePositiveTextNegativesSkipped
	negativeCandidates := candidateResult.Candidates
	if len(negativeCandidates) == 0 {
		result.SkippedQueriesNoNegative = 1
		return result
	}
	qf := newBM25QueryFreq(queryTokens)
	result.Examples = make([]EmbeddingTextHardNegativeExample, 0, len(positives))
	for _, positive := range positives {
		exampleNegatives := negativeCandidates
		if len(exampleNegatives) > cfg.NegativesPerPositive {
			exampleNegatives = exampleNegatives[:cfg.NegativesPerPositive]
		}
		positiveScore := float32(positive.Score)
		if doc, ok := indexDocs[positive.ID]; ok {
			positiveScore = float32(scoreBM25DocumentQueryFreq(qf, doc, index))
		}
		result.Examples = append(result.Examples, EmbeddingTextHardNegativeExample{
			Query:         query.Text,
			Positive:      positive.Text,
			Negatives:     scoredTextValues(exampleNegatives),
			TeacherScores: teacherScoresFromScoredTexts(positiveScore, exampleNegatives),
			ExtraFields:   retrievalHardNegativeProvenanceFields(query.ID, positive.ID, scoredTextIDs(exampleNegatives)),
		})
	}
	return result
}

// effectiveBM25MiningWorkers clamps workers to [1, numQueries] (when
// numQueries > 0): a worker with no query to process would be wasted, so
// requesting more workers than there are queries silently mines with fewer.
// Both mineBM25QueriesParallel's actual worker pool and
// MineBM25TextHardNegatives' summary/CLI provenance echo must derive this
// value the same way, or the reported concurrency can overstate what
// mining actually ran with (see summary.MiningWorkers' doc comment).
func effectiveBM25MiningWorkers(workers, numQueries int) int {
	if workers < 1 {
		workers = 1
	}
	if numQueries > 0 && workers > numQueries {
		workers = numQueries
	}
	return workers
}

// mineBM25QueriesParallel mines every query's hard negatives with a worker
// pool of cfg.MiningWorkers goroutines sharing the read-only index, docText
// and indexDocs (built once by the caller, never written to again). Each
// worker owns a private bm25CandidateScratch reused across every query it
// is assigned, so concurrent workers never share mutable state. Results are
// written to results[i] for query queries[i], so the returned slice is in
// the same order as queries regardless of which worker processed which
// query or the order in which they finished -- ordering is therefore
// entirely up to the caller's later, sequential consumption of the slice.
func mineBM25QueriesParallel(ctx context.Context, queries []retrievalTextRecord, qrels retrievalQrels, docText map[string]string, indexDocs map[string]bm25Document, index bm25Index, cfg RetrievalHardNegativeMiningConfig) ([]bm25MiningQueryResult, error) {
	results := make([]bm25MiningQueryResult, len(queries))
	if len(queries) == 0 {
		return results, nil
	}
	workers := effectiveBM25MiningWorkers(cfg.MiningWorkers, len(queries))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error
	setErr := func(err error) {
		if err == nil {
			return
		}
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}
	hasErr := func() bool {
		errMu.Lock()
		ok := firstErr != nil
		errMu.Unlock()
		return ok
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			scratch := newBM25CandidateScratch(len(index.Documents))
			for i := range jobs {
				if err := ctx.Err(); err != nil {
					setErr(err)
					continue
				}
				query := queries[i]
				results[i] = mineBM25Query(query, qrels[query.ID], docText, indexDocs, index, cfg, scratch)
			}
		}()
	}
	for i := range queries {
		if err := ctx.Err(); err != nil {
			setErr(err)
			break
		}
		if hasErr() {
			break
		}
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	errMu.Lock()
	err := firstErr
	errMu.Unlock()
	if err != nil {
		return nil, err
	}
	return results, nil
}

// MineModelTextHardNegatives mines text hard negatives from BEIR data using the embedding model's own retrieval ranking.
func MineModelTextHardNegatives(ctx context.Context, model *EmbeddingModel, cfg RetrievalHardNegativeMiningConfig) ([]EmbeddingTextHardNegativeExample, RetrievalHardNegativeMiningSummary, error) {
	if model == nil {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeRetrievalHardNegativeMiningConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	corpus, err := readBEIRCorpus(cfg.CorpusPath, cfg.MaxDocs)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	if len(corpus) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("no qrels queries found in queries file")
	}
	docRole, queryRole, effectiveRoleMode, err := resolveEmbeddingRetrievalRoles(model, cfg.RoleMode)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	docVectors, err := embedRetrievalTexts(ctx, model, corpus, cfg.BatchSize, docRole)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("embed corpus: %w", err)
	}
	queryVectors, err := embedRetrievalTexts(ctx, model, queries, cfg.BatchSize, queryRole)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("embed queries: %w", err)
	}
	queryText := make(map[string]string, len(queries))
	for _, query := range queries {
		queryText[query.ID] = query.Text
	}
	docText := make(map[string]string, len(corpus))
	for _, doc := range corpus {
		docText[doc.ID] = doc.Text
	}
	docVectorByID := make(map[string][]float32, len(docVectors))
	for _, doc := range docVectors {
		docVectorByID[doc.ID] = doc.Vector
	}
	summary := RetrievalHardNegativeMiningSummary{
		DatasetName:          cfg.DatasetName,
		RoleMode:             effectiveRoleMode,
		Queries:              len(queries),
		SkippedQueriesNoText: skippedQueries,
	}
	out := []EmbeddingTextHardNegativeExample{}
	for _, query := range queryVectors {
		if err := ctx.Err(); err != nil {
			return nil, RetrievalHardNegativeMiningSummary{}, err
		}
		positives, skippedPositiveDocs := bm25MiningPositiveDocs(qrels[query.ID], docText)
		summary.SkippedPositiveDocs += skippedPositiveDocs
		if len(positives) == 0 {
			continue
		}
		positiveIDs := make(map[string]bool, len(positives))
		for _, positive := range positives {
			positiveIDs[positive.ID] = true
		}
		positiveTextFingerprints := retrievalPositiveTextFingerprints(qrels[query.ID], docText)
		candidateDepth := cfg.CandidateTopK + len(positiveIDs)
		scores := topRetrievalScores(query.Vector, docVectors, candidateDepth)
		candidateResult := modelMiningNegativeCandidates(scores, positiveIDs, positiveTextFingerprints, docText, cfg)
		summary.DuplicatePositiveTextNegativesSkipped += candidateResult.DuplicatePositiveTextNegativesSkipped
		negativeCandidates := candidateResult.Candidates
		if len(negativeCandidates) == 0 {
			summary.SkippedQueriesNoNegative++
			continue
		}
		for _, positive := range positives {
			if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
				break
			}
			exampleNegatives := negativeCandidates
			if len(exampleNegatives) > cfg.NegativesPerPositive {
				exampleNegatives = exampleNegatives[:cfg.NegativesPerPositive]
			}
			positiveScore := float32(positive.Score)
			if vector, ok := docVectorByID[positive.ID]; ok {
				positiveScore = dotRetrievalVectors(query.Vector, vector)
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Query:         queryText[query.ID],
				Positive:      positive.Text,
				Negatives:     scoredTextValues(exampleNegatives),
				TeacherScores: teacherScoresFromScoredTexts(positiveScore, exampleNegatives),
				ExtraFields:   retrievalHardNegativeProvenanceFields(query.ID, positive.ID, scoredTextIDs(exampleNegatives)),
			})
			summary.PositivePairs++
			summary.Negatives += len(exampleNegatives)
		}
		if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
			break
		}
	}
	summary.Examples = len(out)
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("model hard-negative mining produced no examples")
	}
	return out, summary, nil
}

func normalizeRetrievalHardNegativeMiningConfig(cfg RetrievalHardNegativeMiningConfig) RetrievalHardNegativeMiningConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.NegativesPerPositive <= 0 {
		cfg.NegativesPerPositive = 1
	}
	if cfg.CandidateTopK <= 0 {
		cfg.CandidateTopK = 100
	}
	if cfg.CandidateTopK < cfg.NegativesPerPositive {
		cfg.CandidateTopK = cfg.NegativesPerPositive
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.RoleMode == "" {
		cfg.RoleMode = EmbeddingRoleModeAuto
	}
	if cfg.MiningWorkers <= 0 {
		cfg.MiningWorkers = min(DefaultBM25MiningWorkers, runtime.GOMAXPROCS(0))
	}
	if cfg.MiningWorkers < 1 {
		cfg.MiningWorkers = 1
	}
	return cfg
}

func bm25MiningPositiveDocs(rels map[string]float64, docText map[string]string) ([]retrievalPositiveDoc, int) {
	positives := make([]retrievalPositiveDoc, 0, len(rels))
	skipped := 0
	for docID, rel := range rels {
		text := strings.TrimSpace(docText[docID])
		if text == "" {
			skipped++
			continue
		}
		positives = append(positives, retrievalPositiveDoc{ID: docID, Score: rel, Text: text})
	}
	slices.SortFunc(positives, func(a, b retrievalPositiveDoc) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})
	return positives, skipped
}

func bm25MiningNegativeCandidates(queryTokens []string, positiveIDs map[string]bool, positiveTextFingerprints map[string]bool, index bm25Index, docText map[string]string, cfg RetrievalHardNegativeMiningConfig, scratch *bm25CandidateScratch) retrievalMiningCandidateResult {
	result := topBM25NonPositiveScoredTexts(queryTokens, positiveIDs, positiveTextFingerprints, index, docText, cfg.CandidateTopK, scratch)
	negatives := result.Candidates
	if len(negatives) > cfg.NegativesPerPositive {
		negatives = negatives[:cfg.NegativesPerPositive]
	}
	result.Candidates = negatives
	return result
}

func modelMiningNegativeCandidates(scores []retrievalScoredDoc, positiveIDs map[string]bool, positiveTextFingerprints map[string]bool, docText map[string]string, cfg RetrievalHardNegativeMiningConfig) retrievalMiningCandidateResult {
	limit := cfg.NegativesPerPositive
	if cfg.CandidateTopK > 0 && cfg.CandidateTopK < limit {
		limit = cfg.CandidateTopK
	}
	result := retrievalMiningCandidateResult{}
	negatives := make([]retrievalScoredText, 0, limit)
	seen := map[string]bool{}
	candidates := 0
	for _, score := range scores {
		if positiveIDs[score.ID] {
			continue
		}
		text := strings.TrimSpace(docText[score.ID])
		fingerprint := normalizeRetrievalHardNegativeTextFingerprint(text)
		if text == "" || fingerprint == "" || seen[fingerprint] {
			continue
		}
		if positiveTextFingerprints[fingerprint] {
			result.DuplicatePositiveTextNegativesSkipped++
			continue
		}
		candidates++
		if cfg.CandidateTopK > 0 && candidates > cfg.CandidateTopK {
			break
		}
		seen[fingerprint] = true
		negatives = append(negatives, retrievalScoredText{ID: score.ID, Score: score.Score, Text: text})
		if len(negatives) >= limit {
			break
		}
	}
	result.Candidates = negatives
	return result
}

func modelMiningNegativeTexts(scores []retrievalScoredDoc, positiveIDs map[string]bool, docText map[string]string, cfg RetrievalHardNegativeMiningConfig) []string {
	return scoredTextValues(modelMiningNegativeCandidates(scores, positiveIDs, nil, docText, cfg).Candidates)
}

func topBM25NonPositiveScoredTexts(queryTokens []string, positiveIDs map[string]bool, positiveTextFingerprints map[string]bool, index bm25Index, docText map[string]string, topK int, scratch *bm25CandidateScratch) retrievalMiningCandidateResult {
	scores := topBM25NonPositiveScores(queryTokens, positiveIDs, index, topK, scratch)
	result := retrievalMiningCandidateResult{Candidates: make([]retrievalScoredText, 0, len(scores))}
	seen := map[string]bool{}
	for _, score := range scores {
		text := strings.TrimSpace(docText[score.ID])
		fingerprint := normalizeRetrievalHardNegativeTextFingerprint(text)
		if text == "" || fingerprint == "" || seen[fingerprint] {
			continue
		}
		if positiveTextFingerprints[fingerprint] {
			result.DuplicatePositiveTextNegativesSkipped++
			continue
		}
		seen[fingerprint] = true
		result.Candidates = append(result.Candidates, retrievalScoredText{ID: score.ID, Score: score.Score, Text: text})
	}
	return result
}

func scoredTextValues(candidates []retrievalScoredText) []string {
	out := make([]string, len(candidates))
	for i, candidate := range candidates {
		out[i] = candidate.Text
	}
	return out
}

func scoredTextIDs(candidates []retrievalScoredText) []string {
	out := make([]string, len(candidates))
	for i, candidate := range candidates {
		out[i] = candidate.ID
	}
	return out
}

func retrievalPositiveTextFingerprints(rels map[string]float64, docText map[string]string) map[string]bool {
	out := map[string]bool{}
	for docID := range rels {
		fingerprint := normalizeRetrievalHardNegativeTextFingerprint(docText[docID])
		if fingerprint != "" {
			out[fingerprint] = true
		}
	}
	return out
}

func normalizeRetrievalHardNegativeTextFingerprint(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

func retrievalHardNegativeProvenanceFields(queryID, positiveDocID string, negativeDocIDs []string) map[string]json.RawMessage {
	fields := map[string]json.RawMessage{}
	put := func(key string, value any) {
		data, err := json.Marshal(value)
		if err == nil {
			fields[key] = data
		}
	}
	put("query_id", queryID)
	put("positive_doc_id", positiveDocID)
	put("negative_doc_ids", append([]string(nil), negativeDocIDs...))
	return fields
}

func teacherScoresFromScoredTexts(positiveScore float32, negatives []retrievalScoredText) []float32 {
	out := make([]float32, 1, 1+len(negatives))
	out[0] = positiveScore
	for _, negative := range negatives {
		out = append(out, negative.Score)
	}
	return out
}
