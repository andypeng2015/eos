package eosruntime

import (
	"fmt"
	"math/rand"
	"sort"
)

// SampleCorpusNegativesConfig controls random-corpus negative sampling for
// teacher scoring. Random non-qrel documents are the source of trustworthy
// true negatives on sparse-label corpora, where model-mined candidates are
// dominated by unlabeled relevant documents.
type SampleCorpusNegativesConfig struct {
	DatasetDir string
	// Split selects the qrels file under <dataset-dir>/qrels; defaults to train.
	Split string
	// PerQuery is how many random non-qrel documents to sample per query.
	PerQuery int
	// Seed makes sampling reproducible.
	Seed int64
	// MaxQueries caps sampled queries for smoke runs; 0 samples all.
	MaxQueries int
	// Source tags emitted rows for source-aware training and score imports.
	Source string
}

// SampleCorpusNegativesSummary reports what was sampled.
type SampleCorpusNegativesSummary struct {
	CorpusDocuments  int `json:"corpus_documents"`
	QrelsQueries     int `json:"qrels_queries"`
	SampledQueries   int `json:"sampled_queries"`
	SkippedNoPositive int `json:"skipped_no_positive"`
	EmittedNegatives int `json:"emitted_negatives"`
}

// SampleCorpusNegatives emits one row per qrels query whose positive is the
// query's first labeled document and whose negatives are PerQuery random
// non-qrel corpus documents. Rows carry no teacher scores; score them with
// export-teacher-score-requests and import-teacher-scores, then feed the
// scored file to relabel-teacher-negatives as the true-negative pool.
func SampleCorpusNegatives(cfg SampleCorpusNegativesConfig) ([]EmbeddingTextHardNegativeExample, SampleCorpusNegativesSummary, error) {
	summary := SampleCorpusNegativesSummary{}
	if cfg.DatasetDir == "" {
		return nil, summary, fmt.Errorf("dataset directory is required")
	}
	if cfg.PerQuery <= 0 {
		return nil, summary, fmt.Errorf("per-query must be positive")
	}
	split := cfg.Split
	if split == "" {
		split = "train"
	}
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(cfg.DatasetDir, split)
	qrels, err := readBEIRQrels(qrelsPath)
	if err != nil {
		return nil, summary, err
	}
	if len(qrels) == 0 {
		return nil, summary, fmt.Errorf("qrels %s is empty", qrelsPath)
	}
	summary.QrelsQueries = len(qrels)
	corpus, err := readBEIRCorpus(corpusPath, 0)
	if err != nil {
		return nil, summary, err
	}
	if len(corpus) == 0 {
		return nil, summary, fmt.Errorf("corpus %s is empty", corpusPath)
	}
	summary.CorpusDocuments = len(corpus)
	corpusText := make(map[string]string, len(corpus))
	for _, record := range corpus {
		corpusText[record.ID] = record.Text
	}
	queries, _, err := readBEIRQueries(queriesPath, qrels, 0)
	if err != nil {
		return nil, summary, err
	}
	sort.Slice(queries, func(i, j int) bool { return queries[i].ID < queries[j].ID })
	if cfg.MaxQueries > 0 && len(queries) > cfg.MaxQueries {
		queries = queries[:cfg.MaxQueries]
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	var out []EmbeddingTextHardNegativeExample
	for _, query := range queries {
		labeled := qrels[query.ID]
		positive := ""
		labeledIDs := make([]string, 0, len(labeled))
		for docID := range labeled {
			labeledIDs = append(labeledIDs, docID)
		}
		sort.Strings(labeledIDs)
		for _, docID := range labeledIDs {
			if text, ok := corpusText[docID]; ok && text != "" {
				positive = text
				break
			}
		}
		if positive == "" {
			summary.SkippedNoPositive++
			continue
		}
		negatives := make([]string, 0, cfg.PerQuery)
		seen := make(map[int]bool, cfg.PerQuery)
		attempts := 0
		for len(negatives) < cfg.PerQuery && attempts < cfg.PerQuery*50 {
			attempts++
			idx := rng.Intn(len(corpus))
			if seen[idx] {
				continue
			}
			seen[idx] = true
			record := corpus[idx]
			if labeled[record.ID] != 0 || record.Text == "" || record.Text == positive {
				continue
			}
			negatives = append(negatives, record.Text)
		}
		if len(negatives) == 0 {
			continue
		}
		out = append(out, EmbeddingTextHardNegativeExample{
			Source:    cfg.Source,
			Query:     query.Text,
			Positive:  positive,
			Negatives: negatives,
		})
		summary.SampledQueries++
		summary.EmittedNegatives += len(negatives)
	}
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("no queries produced sampled negatives")
	}
	return out, summary, nil
}
