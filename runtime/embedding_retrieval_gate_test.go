package eosruntime

import "testing"

// These tests drive the retrieval-nDCG selection gate: train-embed must be able
// to select/restore-best on retrieval nDCG (the deployment metric) instead of
// the saturated pairwise gate (top1 ~0.957 while retrieval nDCG ~0.148).

func TestValidTrainSelectionMetricAcceptsRetrievalNDCG(t *testing.T) {
	if !validTrainSelectionMetric("retrieval_ndcg") {
		t.Fatal("retrieval_ndcg must be a valid selection metric")
	}
}

func TestEvalRankMetricReturnsRetrievalNDCG(t *testing.T) {
	m := EmbeddingEvalMetrics{RetrievalNDCGAt10: 0.42}
	if got := evalRankMetric(m, "retrieval_ndcg"); got != 0.42 {
		t.Fatalf("evalRankMetric(retrieval_ndcg) = %v, want 0.42", got)
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
