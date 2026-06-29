package eosruntime

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These tests drive the retrieval selection gate: train-embed must be able to
// select/restore-best on held-out retrieval metrics instead of the saturated
// pairwise gate (top1 ~0.957 while retrieval nDCG ~0.148).

func TestValidTrainSelectionMetricAcceptsRetrievalNDCG(t *testing.T) {
	for _, metric := range []string{"retrieval_ndcg", "retrieval_ndcg_at_10"} {
		if !validTrainSelectionMetric(metric) {
			t.Fatalf("%s must be a valid selection metric", metric)
		}
	}
}

func TestValidTrainSelectionMetricAcceptsRetrievalMAPAndRecall(t *testing.T) {
	for _, metric := range []string{"retrieval_map", "retrieval_map_at_100", "retrieval_recall", "retrieval_recall_at_100"} {
		if !validTrainSelectionMetric(metric) {
			t.Fatalf("%s must be a valid selection metric", metric)
		}
	}
}

func TestEvalRankMetricReturnsRetrievalNDCG(t *testing.T) {
	m := EmbeddingEvalMetrics{RetrievalNDCGAt10: 0.42}
	if got := evalRankMetric(m, "retrieval_ndcg"); got != 0.42 {
		t.Fatalf("evalRankMetric(retrieval_ndcg) = %v, want 0.42", got)
	}
	if got := evalRankMetric(m, "retrieval_ndcg_at_10"); got != 0.42 {
		t.Fatalf("evalRankMetric(retrieval_ndcg_at_10) = %v, want 0.42", got)
	}
}

func TestEvalRankMetricReturnsRetrievalMAPAndRecall(t *testing.T) {
	m := EmbeddingEvalMetrics{RetrievalMAPAt100: 0.37, RetrievalRecallAt100: 0.81}
	if got := evalRankMetric(m, "retrieval_map"); got != 0.37 {
		t.Fatalf("evalRankMetric(retrieval_map) = %v, want 0.37", got)
	}
	if got := evalRankMetric(m, "retrieval_map_at_100"); got != 0.37 {
		t.Fatalf("evalRankMetric(retrieval_map_at_100) = %v, want 0.37", got)
	}
	if got := evalRankMetric(m, "retrieval_recall"); got != 0.81 {
		t.Fatalf("evalRankMetric(retrieval_recall) = %v, want 0.81", got)
	}
	if got := evalRankMetric(m, "retrieval_recall_at_100"); got != 0.81 {
		t.Fatalf("evalRankMetric(retrieval_recall_at_100) = %v, want 0.81", got)
	}
}

func TestBetterEvalMetricsSelectsHigherRetrievalNDCG(t *testing.T) {
	best := EmbeddingEvalMetrics{RetrievalNDCGAt10: 0.30}
	improved := EmbeddingEvalMetrics{RetrievalNDCGAt10: 0.35}
	if !betterEvalMetrics(improved, best, "retrieval_ndcg", 0) {
		t.Fatal("higher retrieval nDCG must be considered better")
	}
	if betterEvalMetrics(best, improved, "retrieval_ndcg", 0) {
		t.Fatal("lower retrieval nDCG must not be considered better")
	}
}

func TestBetterEvalMetricsSelectsHigherRetrievalRecall(t *testing.T) {
	best := EmbeddingEvalMetrics{RetrievalRecallAt100: 0.45}
	improved := EmbeddingEvalMetrics{RetrievalRecallAt100: 0.52}
	if !betterEvalMetrics(improved, best, "retrieval_recall", 0) {
		t.Fatal("higher retrieval recall must be considered better")
	}
	if betterEvalMetrics(best, improved, "retrieval_recall_at_100", 0) {
		t.Fatal("lower retrieval recall must not be considered better")
	}
}

// A capped corpus must still include the qrels-relevant docs, otherwise nDCG is
// meaningless (the gate silently reads 0 when the cap drops all relevant docs —
// e.g. -retrieval-eval-max-docs 2000 on fiqa, whose relevant docs are late in
// file order).
func TestReadBEIRCorpusWithRelevantIncludesRelevantUnderCap(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	var b strings.Builder
	for i := 0; i < 20; i++ {
		b.WriteString(fmt.Sprintf(`{"_id":"d%d","text":"document %d filler body text"}`+"\n", i, i))
	}
	if err := os.WriteFile(corpusPath, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	// relevant doc d18 sits well past a cap of 3
	qrels := retrievalQrels{"q1": map[string]float64{"d18": 1}}
	recs, err := readBEIRCorpusWithRelevant(corpusPath, 3, qrels)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range recs {
		if r.ID == "d18" {
			found = true
		}
	}
	if !found {
		t.Fatalf("relevant doc d18 must be included under the cap; got %d docs without it", len(recs))
	}
	// non-relevant fill still respects the cap as a floor of distractors
	if len(recs) < 3 {
		t.Fatalf("expected at least the cap of 3 docs, got %d", len(recs))
	}
}
