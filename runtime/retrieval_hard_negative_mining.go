package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

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
	docText := make(map[string]string, len(corpus))
	for _, doc := range corpus {
		docText[doc.ID] = doc.Text
	}
	indexDocs := make(map[string]bm25Document, len(index.Documents))
	for _, doc := range index.Documents {
		indexDocs[doc.ID] = doc
	}
	summary := RetrievalHardNegativeMiningSummary{
		DatasetName:          cfg.DatasetName,
		RoleMode:             cfg.RoleMode,
		Queries:              len(queries),
		SkippedQueriesNoText: skippedQueries,
	}
	out := []EmbeddingTextHardNegativeExample{}
	for _, query := range queries {
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
		queryTokens := tokenizeBM25Text(query.Text)
		candidateResult := bm25MiningNegativeCandidates(queryTokens, positiveIDs, positiveTextFingerprints, index, docText, cfg)
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
			if doc, ok := indexDocs[positive.ID]; ok {
				positiveScore = float32(scoreBM25Document(queryTokens, doc, index))
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Query:         query.Text,
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
		return nil, summary, fmt.Errorf("BM25 hard-negative mining produced no examples")
	}
	return out, summary, nil
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

func bm25MiningNegativeCandidates(queryTokens []string, positiveIDs map[string]bool, positiveTextFingerprints map[string]bool, index bm25Index, docText map[string]string, cfg RetrievalHardNegativeMiningConfig) retrievalMiningCandidateResult {
	result := topBM25NonPositiveScoredTexts(queryTokens, positiveIDs, positiveTextFingerprints, index, docText, cfg.CandidateTopK)
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

func topBM25NonPositiveScoredTexts(queryTokens []string, positiveIDs map[string]bool, positiveTextFingerprints map[string]bool, index bm25Index, docText map[string]string, topK int) retrievalMiningCandidateResult {
	scores := topBM25NonPositiveScores(queryTokens, positiveIDs, index, topK)
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
