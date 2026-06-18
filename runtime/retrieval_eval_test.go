package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestComputeRetrievalQualityPerfectRanking(t *testing.T) {
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1})},
	}
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0.2, 0.1})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries := computeRetrievalQuality(queries, docs, qrels, 100)
	if queriesCount != 2 || relevantPairs != 2 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if quality.NDCGAt10 != 1 || quality.MRRAt10 != 1 || quality.RecallAt10 != 1 || quality.RecallAt100 != 1 {
		t.Fatalf("quality = %+v, want perfect ranking", quality)
	}
	if quality.NDCGAt100 != 1 || quality.PrecisionAt1 != 1 || quality.PrecisionAt5 != 0.2 || quality.PrecisionAt10 != 0.1 {
		t.Fatalf("precision/ndcg@100 = %+v, want perfect top hit with fixed-k precision", quality)
	}
	if quality.HitAt1 != 1 || quality.HitAt5 != 1 || quality.HitAt10 != 1 || quality.MAPAt10 != 1 || quality.MAPAt100 != 1 {
		t.Fatalf("hit/map quality = %+v, want perfect ranking", quality)
	}
}

func TestComputeHybridRetrievalQualityMinmaxAlphaWeightsBM25(t *testing.T) {
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0})},
	}
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{0, 1})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0.5, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
	}
	corpus := []retrievalTextRecord{
		{ID: "d1", Text: "alpha exact target"},
		{ID: "d2", Text: "beta dense distractor"},
		{ID: "d3", Text: "gamma fallback"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build BM25 index: %v", err)
	}
	denseQuality, _, _, _, _ := computeRetrievalQuality(queries, docs, qrels, 100)

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries, err := computeHybridRetrievalQuality(
		context.Background(),
		queries,
		docs,
		map[string][]string{"q1": tokenizeBM25Text("alpha")},
		index,
		qrels,
		100,
		"tiny",
		"",
		RetrievalEvalHybridConfig{Method: "minmax_blend", Alpha: 0.75, RRFK: 60, RRFLambda: 1},
	)
	if err != nil {
		t.Fatalf("compute hybrid quality: %v", err)
	}
	if queriesCount != 1 || relevantPairs != 1 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if denseQuality.NDCGAt10 >= 1 {
		t.Fatalf("dense quality = %+v, want imperfect dense baseline", denseQuality)
	}
	if quality.NDCGAt10 != 1 || quality.MRRAt10 != 1 {
		t.Fatalf("hybrid quality = %+v, want BM25-weighted top hit", quality)
	}
}

func TestFuseHybridScoresDefaultLeavesFusedOrder(t *testing.T) {
	denseScores := []retrievalScoredDoc{
		{ID: "dense-winner", Score: 1},
		{ID: "tail", Score: 0.5},
		{ID: "lexical-winner", Score: 0},
	}
	bm25Scores := []retrievalScoredDoc{
		{ID: "lexical-winner", Score: 10},
		{ID: "tail", Score: 5},
		{ID: "dense-winner", Score: 0},
	}

	got := fuseHybridScores(denseScores, bm25Scores, 100, RetrievalEvalHybridConfig{
		Method: "minmax_blend",
		Alpha:  0.75,
	})
	if len(got) < 3 {
		t.Fatalf("fused length = %d, want at least 3", len(got))
	}
	want := []string{"lexical-winner", "tail", "dense-winner"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("fused[%d] = %q, want %q; got=%+v", i, got[i].ID, id, got[:3])
		}
	}
	if got[0].DenseRank != 3 || got[0].BM25Rank != 1 {
		t.Fatalf("lexical winner component ranks = dense:%d bm25:%d, want 3/1", got[0].DenseRank, got[0].BM25Rank)
	}
	if got[0].DenseScore == nil || *got[0].DenseScore != 0 || got[0].BM25Score == nil || *got[0].BM25Score != 10 {
		t.Fatalf("lexical winner component raw scores = dense:%v bm25:%v, want 0/10", got[0].DenseScore, got[0].BM25Score)
	}
	if got[0].DenseNormalizedScore == nil || *got[0].DenseNormalizedScore != 0 || got[0].BM25NormalizedScore == nil || *got[0].BM25NormalizedScore != 1 {
		t.Fatalf("lexical winner component normalized scores = dense:%v bm25:%v, want 0/1", got[0].DenseNormalizedScore, got[0].BM25NormalizedScore)
	}
}

func TestFuseHybridScoresDenseProtectTopKPreservesDensePrefix(t *testing.T) {
	denseScores := []retrievalScoredDoc{
		{ID: "dense-winner", Score: 1},
		{ID: "dense-second", Score: 0.9},
		{ID: "lexical-winner", Score: 0},
	}
	bm25Scores := []retrievalScoredDoc{
		{ID: "lexical-winner", Score: 10},
		{ID: "dense-second", Score: 5},
		{ID: "dense-winner", Score: 0},
	}

	got := fuseHybridScores(denseScores, bm25Scores, 100, RetrievalEvalHybridConfig{
		Method:           "minmax_blend",
		Alpha:            0.75,
		DenseProtectTopK: 2,
	})
	if len(got) < 3 {
		t.Fatalf("fused length = %d, want at least 3", len(got))
	}
	want := []string{"dense-winner", "dense-second", "lexical-winner"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("protected fused[%d] = %q, want %q; got=%+v", i, got[i].ID, id, got[:3])
		}
	}
	if got[0].DenseRank != 1 || got[0].BM25Rank != 3 {
		t.Fatalf("protected dense winner component ranks = dense:%d bm25:%d, want 1/3", got[0].DenseRank, got[0].BM25Rank)
	}
	if got[0].DenseScore == nil || *got[0].DenseScore != 1 || got[0].BM25Score == nil || *got[0].BM25Score != 0 {
		t.Fatalf("protected dense winner component raw scores = dense:%v bm25:%v, want 1/0", got[0].DenseScore, got[0].BM25Score)
	}
	if got[0].DenseNormalizedScore == nil || *got[0].DenseNormalizedScore != 1 || got[0].BM25NormalizedScore == nil || *got[0].BM25NormalizedScore != 0 {
		t.Fatalf("protected dense winner component normalized scores = dense:%v bm25:%v, want 1/0", got[0].DenseNormalizedScore, got[0].BM25NormalizedScore)
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsQualityAndCost(t *testing.T) {
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0, 0, 1, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq",
		TopK:        100,
	}, []int{8}, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval: %v", err)
	}
	if metrics.Schema != TurboQuantRetrievalEvalMetricsSchema || metrics.Dataset != "tiny-tq" {
		t.Fatalf("metrics identity = schema:%q dataset:%q", metrics.Schema, metrics.Dataset)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || metrics.Dense.Quality.RecallAt100 != 1 {
		t.Fatalf("dense quality = %+v, want perfect", metrics.Dense.Quality)
	}
	if metrics.Dense.VectorBytes != int64(len(docs)*len(docs[0].Vector)*4) {
		t.Fatalf("dense vector bytes = %d", metrics.Dense.VectorBytes)
	}
	if metrics.Dense.QueryLatency.Count != len(queries) || metrics.Dense.QueryLatency.P95MS < 0 {
		t.Fatalf("dense query latency = %+v, want populated latency metrics", metrics.Dense.QueryLatency)
	}
	if len(metrics.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(metrics.Rows))
	}
	row := metrics.Rows[0]
	if row.Bits != 8 || row.Method != "turboquant_ip_b8" {
		t.Fatalf("row identity = bits:%d method:%q", row.Bits, row.Method)
	}
	if row.VectorBytes <= 0 || row.VectorBytes >= row.DenseVectorBytes {
		t.Fatalf("quantized bytes = %d dense bytes = %d", row.VectorBytes, row.DenseVectorBytes)
	}
	if row.CompressionRatio <= 1 {
		t.Fatalf("compression ratio = %v, want > 1", row.CompressionRatio)
	}
	if row.QueryLatency.Count != len(queries) || row.QueryLatency.P95MS < 0 {
		t.Fatalf("quantized query latency = %+v, want populated latency metrics", row.QueryLatency)
	}
	if row.Quality.NDCGAt10 < 0.99 || row.Quality.RecallAt100 != 1 {
		t.Fatalf("quantized quality = %+v, want near-perfect", row.Quality)
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsDenseRerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerank(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-rerank",
		TopK:        100,
	}, []int{8}, []int{110}, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with rerank: %v", err)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_dense_rerank" || rerank.RerankOverfetch != 110 {
		t.Fatalf("rerank row identity = method:%q overfetch:%d", rerank.Method, rerank.RerankOverfetch)
	}
	if rerank.RerankStorage != TurboQuantRerankStorageDense || rerank.RerankSidecarBytes != rerank.DenseVectorBytes {
		t.Fatalf("dense rerank storage = storage:%q sidecar:%d dense:%d", rerank.RerankStorage, rerank.RerankSidecarBytes, rerank.DenseVectorBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes+rerank.RerankSidecarBytes || rerank.TotalCompression <= 0 {
		t.Fatalf("dense rerank total accounting = %+v", rerank)
	}
	if rerank.RerankScores != int64(len(queries)*110) || rerank.RerankScoreSeconds <= 0 {
		t.Fatalf("rerank accounting = scores:%d seconds:%f", rerank.RerankScores, rerank.RerankScoreSeconds)
	}
	if rerank.VectorBytes != metrics.Rows[0].VectorBytes || rerank.CompressionRatio != metrics.Rows[0].CompressionRatio {
		t.Fatalf("rerank storage = %+v direct = %+v", rerank, metrics.Rows[0])
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsCompactReconstructRerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-compact-rerank",
		TopK:        100,
	}, []int{8}, []int{110}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with compact rerank: %v", err)
	}
	if metrics.Config.RerankStorage != TurboQuantRerankStorageCompactReconstruct {
		t.Fatalf("config rerank storage = %q", metrics.Config.RerankStorage)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and compact rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_reconstruct_rerank" || rerank.RerankOverfetch != 110 {
		t.Fatalf("compact rerank row identity = method:%q overfetch:%d", rerank.Method, rerank.RerankOverfetch)
	}
	if rerank.RerankStorage != TurboQuantRerankStorageCompactReconstruct || rerank.RerankSidecarBytes != 0 {
		t.Fatalf("compact rerank storage = storage:%q sidecar:%d", rerank.RerankStorage, rerank.RerankSidecarBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes || rerank.TotalCompression != rerank.CompressionRatio {
		t.Fatalf("compact rerank total accounting = %+v", rerank)
	}
	if rerank.RerankScores != int64(len(queries)*110) || rerank.RerankScoreSeconds <= 0 {
		t.Fatalf("compact rerank accounting = scores:%d seconds:%f", rerank.RerankScores, rerank.RerankScoreSeconds)
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsFP16RerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-fp16-rerank",
		TopK:        100,
	}, []int{8}, []int{110}, "half", docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with fp16 rerank: %v", err)
	}
	if metrics.Config.RerankStorage != TurboQuantRerankStorageFP16 {
		t.Fatalf("config rerank storage = %q", metrics.Config.RerankStorage)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and fp16 rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_fp16_rerank" || rerank.RerankOverfetch != 110 {
		t.Fatalf("fp16 rerank row identity = method:%q overfetch:%d", rerank.Method, rerank.RerankOverfetch)
	}
	wantSidecarBytes := int64(len(docs) * len(docs[0].Vector) * 2)
	if rerank.RerankStorage != TurboQuantRerankStorageFP16 || rerank.RerankSidecarBytes != wantSidecarBytes {
		t.Fatalf("fp16 rerank storage = storage:%q sidecar:%d want:%d", rerank.RerankStorage, rerank.RerankSidecarBytes, wantSidecarBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes+wantSidecarBytes {
		t.Fatalf("fp16 rerank total bytes = %d, want %d", rerank.TotalVectorBytes, rerank.VectorBytes+wantSidecarBytes)
	}
	if rerank.RerankSidecarBytes == rerank.DenseVectorBytes {
		t.Fatalf("fp16 rerank sidecar unexpectedly matches dense f32 sidecar bytes: %+v", rerank)
	}
	if rerank.TotalCompression >= rerank.CompressionRatio || rerank.TotalCompression >= 2 {
		t.Fatalf("fp16 total compression = %.6f quant compression = %.6f", rerank.TotalCompression, rerank.CompressionRatio)
	}
	if rerank.RerankScores != int64(len(queries)*110) || rerank.RerankScoreSeconds <= 0 {
		t.Fatalf("fp16 rerank accounting = scores:%d seconds:%f", rerank.RerankScores, rerank.RerankScoreSeconds)
	}
	if rerank.QueryLatency.Count != len(queries) || rerank.QueryLatency.P95MS < 0 {
		t.Fatalf("fp16 rerank query latency = %+v, want populated latency metrics", rerank.QueryLatency)
	}
	if rerank.Quality.NDCGAt10 < 0.99 || rerank.Quality.RecallAt100 != 1 {
		t.Fatalf("fp16 rerank quality = %+v, want near-perfect", rerank.Quality)
	}
}

func TestTurboQuantFP16RerankRescueLeavesIncompleteWindowUnchanged(t *testing.T) {
	fp16Ranked, compactScores, compactRanks := turboQuantRescueFixture(119)
	got := applyTurboQuantFP16RerankRescue(fp16Ranked, compactScores, compactRanks)
	if len(got) != len(fp16Ranked) {
		t.Fatalf("rescued length = %d, want %d", len(got), len(fp16Ranked))
	}
	for i := range fp16Ranked {
		if got[i].ID != fp16Ranked[i].ID {
			t.Fatalf("rescued rank %d = %q, want unchanged %q", i+1, got[i].ID, fp16Ranked[i].ID)
		}
	}
}

func TestTurboQuantFP16RerankRescuePromotesStrongCompactBoundaryCandidate(t *testing.T) {
	fp16Ranked, compactScores, compactRanks := turboQuantRescueFixture(121)
	compactScores["d120"] = 10_000
	compactRanks["d120"] = 1

	got := applyTurboQuantFP16RerankRescue(fp16Ranked, compactScores, compactRanks)
	top100 := map[string]bool{}
	for _, doc := range got[:100] {
		top100[doc.ID] = true
	}
	if !top100["d120"] {
		t.Fatalf("d120 was not rescued into top100")
	}
	if top100["d100"] {
		t.Fatalf("d100 remained in top100 after higher-priority d120 rescue")
	}
	if got[99].ID != "d120" {
		t.Fatalf("rescued rank 100 = %q, want d120", got[99].ID)
	}
}

func TestTurboQuantFP16RerankRescueCompactTieBreakMatchesSimulation(t *testing.T) {
	fp16Ranked, compactScores, compactRanks := turboQuantRescueFixture(121)
	compactScores["d110"] = 10_000
	compactScores["d111"] = 10_000
	compactRanks["d110"] = 7
	compactRanks["d111"] = 7

	got := applyTurboQuantFP16RerankRescue(fp16Ranked, compactScores, compactRanks)
	if got[99].ID != "d111" {
		t.Fatalf("rescued compact/doc tie rank 100 = %q, want d111", got[99].ID)
	}
}

func TestEvaluateTurboQuantVectorRetrievalWritesCompactPerQueryJSONL(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		if i == 0 {
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})}}
	qrels := retrievalQrels{"q1": {"d1": 1}}
	perQueryPath := filepath.Join(t.TempDir(), "compact.per-query.jsonl")

	_, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-compact",
		TopK:              100,
		PerQueryJSONLPath: perQueryPath,
	}, []int{8}, []int{110}, TurboQuantRerankStorageFP16, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with per-query: %v", err)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query lines = %d, want direct and rerank rows\n%s", len(lines), data)
	}
	var direct, rerank TurboQuantRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &direct); err != nil {
		t.Fatalf("decode direct row: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &rerank); err != nil {
		t.Fatalf("decode rerank row: %v", err)
	}
	if direct.Schema != TurboQuantRetrievalPerQuerySchema || direct.Method != "turboquant_ip_b8" || direct.Bits != 8 {
		t.Fatalf("direct row = %+v", direct)
	}
	if direct.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed || rerank.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("per-query seeds = direct:%d rerank:%d", direct.QuantizerSeed, rerank.QuantizerSeed)
	}
	if rerank.Method != "turboquant_ip_b8_overfetch110_fp16_rerank" || rerank.RerankOverfetch != 110 || rerank.RerankStorage != TurboQuantRerankStorageFP16 {
		t.Fatalf("rerank row = %+v", rerank)
	}
	if len(rerank.TopK) == 0 || rerank.TopK[0].DenseRank == nil || rerank.TopK[0].CompactRank == nil || rerank.TopK[0].DenseScore == nil || rerank.TopK[0].CompactScore == nil {
		t.Fatalf("rerank top doc missing dense/compact evidence: %+v", rerank.TopK)
	}
}

func turboQuantRescueFixture(n int) ([]retrievalScoredDoc, map[string]float32, map[string]int) {
	fp16Ranked := make([]retrievalScoredDoc, n)
	compactScores := make(map[string]float32, n)
	compactRanks := make(map[string]int, n)
	for i := range fp16Ranked {
		id := fmt.Sprintf("d%03d", i+1)
		fp16Ranked[i] = retrievalScoredDoc{ID: id, Score: float32(n - i)}
		compactScores[id] = float32(n - i)
		compactRanks[id] = i + 1
	}
	return fp16Ranked, compactScores, compactRanks
}

func TestMineCompactTextHardNegativesWritesManifestAndGuardsTestSplit(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "tiny")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels", "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","title":"positive","text":"alpha positive document"}`+"\n"+
			`{"_id":"d2","title":"negative","text":"alpha hard negative"}`+"\n"+
			`{"_id":"d3","title":"negative","text":"alpha boundary negative"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	perQueryPath := filepath.Join(dir, "compact.per-query.jsonl")
	row := TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
			{Rank: 2, DocID: "d1", Score: 0.8, Relevance: 1},
			{Rank: 3, DocID: "d3", Score: 0.7, Relevance: 0},
		},
	}
	data, _ := json.Marshal(row)
	if err := os.WriteFile(perQueryPath, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write per-query: %v", err)
	}
	blockedOutput := filepath.Join(dir, "blocked.jsonl")
	_, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        blockedOutput,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		TrainSelection:    true,
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to mine train-selection rows from test split") {
		t.Fatalf("expected test split guard error, got %v", err)
	}
	_, err = MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        blockedOutput,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		TrainSelection:    true,
		AllowTestSmoke:    true,
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to mine train-selection rows from test split") {
		t.Fatalf("allow-test-smoke must not bypass train-selection guard, got %v", err)
	}
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifestPath := filepath.Join(dir, "manifest.json")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		ManifestPath:      manifestPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		TrainSelection:    false,
		NegativesPerRow:   2,
	})
	if err != nil {
		t.Fatalf("mine compact hard negatives: %v", err)
	}
	if manifest.TrainAllowed || manifest.LeakGuardStatus != "validation_smoke_no_train_test_split" || manifest.RowsEmitted != 1 || manifest.Negatives != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
	if manifest.PerQuerySHA256 == "" || manifest.HardNegativesSHA256 == "" || manifest.ReasonCounts["top10_competitor"] != 1 {
		t.Fatalf("manifest hashes/reasons = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || examples[0].Source == "" || len(examples[0].Negatives) != 2 {
		t.Fatalf("examples = %+v", examples)
	}
}

func TestMineCompactTextHardNegativesFiltersQrelsPositiveNegatives(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"positive one","text":"alpha first positive"}`,
			`{"_id":"d2","title":"positive two","text":"alpha stale row positive"}`,
			`{"_id":"d3","title":"negative","text":"alpha true negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.95, Relevance: 0},
			{Rank: 2, DocID: "d3", Score: 0.90, Relevance: 0},
		},
	})
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		NegativesPerRow:   2,
		MaxExamples:       1,
	})
	if err != nil {
		t.Fatalf("mine compact hard negatives: %v", err)
	}
	if manifest.QrelsRelevanceMismatches != 1 || manifest.Negatives != 1 {
		t.Fatalf("manifest mismatch/negative counts = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || len(examples[0].Negatives) != 1 || !strings.Contains(examples[0].Negatives[0], "true negative") {
		t.Fatalf("qrels-positive doc leaked or expected negative missing: %+v", examples)
	}
	if strings.Contains(strings.Join(examples[0].Negatives, "\n"), "stale row positive") {
		t.Fatalf("qrels-positive document emitted as negative: %+v", examples[0].Negatives)
	}
}

func TestMineCompactTextHardNegativesQuantizerSeedDefaultAndMismatch(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"positive","text":"alpha positive"}`,
			`{"_id":"d2","title":"negative","text":"alpha negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
	)
	row := TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed + 1,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
		},
	}
	mismatchPath := writeCompactMiningRows(t, dir, row)
	_, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: mismatchPath,
		OutputPath:        filepath.Join(dir, "mismatch.jsonl"),
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
	})
	if err == nil || !strings.Contains(err.Error(), "quantizer seed mismatch") {
		t.Fatalf("expected quantizer seed mismatch, got %v", err)
	}

	row.QuantizerSeed = DefaultTurboQuantMultiVectorQuantizerSeed
	matchPath := writeCompactMiningRows(t, dir, row)
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: matchPath,
		OutputPath:        filepath.Join(dir, "matched.jsonl"),
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
	})
	if err != nil {
		t.Fatalf("mine with default seed: %v", err)
	}
	if manifest.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("manifest quantizer seed = %d, want default %d", manifest.QuantizerSeed, DefaultTurboQuantMultiVectorQuantizerSeed)
	}
}

func TestMineCompactTextHardNegativesDerivesCompactReconstructMethod(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"positive","text":"alpha positive"}`,
			`{"_id":"d2","title":"negative","text":"alpha negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_reconstruct_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageCompactReconstruct,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
		},
	})
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        filepath.Join(dir, "hard-negatives.jsonl"),
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageCompactReconstruct,
	})
	if err != nil {
		t.Fatalf("mine compact reconstruct defaults: %v", err)
	}
	if manifest.Method != "turboquant_ip_b4_overfetch200_reconstruct_rerank" || manifest.RowsMatched != 1 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestMineCompactTextHardNegativesMaxDocsPreservesQrelsPositive(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"negative","text":"alpha early negative"}`,
			`{"_id":"d2","title":"filler","text":"alpha early filler"}`,
			`{"_id":"d3","title":"filler","text":"alpha later filler"}`,
			`{"_id":"d4","title":"positive","text":"alpha late positive"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td4\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d1", Score: 0.9, Relevance: 0},
		},
	})
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		MaxDocs:           2,
	})
	if err != nil {
		t.Fatalf("mine with capped corpus: %v", err)
	}
	if manifest.RowsEmitted != 1 || manifest.SkippedNoPositive != 0 {
		t.Fatalf("manifest = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || !strings.Contains(examples[0].Positive, "late positive") {
		t.Fatalf("positive beyond cap was not preserved: %+v", examples)
	}
}

func writeCompactMiningDataset(t *testing.T, corpusLines []string, qrels string) (dir, corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir = t.TempDir()
	datasetDir := filepath.Join(dir, "tiny")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath = filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath = filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath = filepath.Join(datasetDir, "qrels", "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(strings.Join(corpusLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte(qrels), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir, corpusPath, queriesPath, qrelsPath
}

func writeCompactMiningRows(t *testing.T, dir string, rows ...TurboQuantRetrievalPerQueryRow) string {
	t.Helper()
	path := filepath.Join(dir, fmt.Sprintf("compact-%d.per-query.jsonl", time.Now().UnixNano()))
	var b strings.Builder
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal per-query row: %v", err)
		}
		b.Write(data)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write per-query: %v", err)
	}
	return path
}

func TestPercentileDurationUsesConservativeNearestRankForSmallSamples(t *testing.T) {
	durations := []time.Duration{
		1 * time.Millisecond,
		2 * time.Millisecond,
		3 * time.Millisecond,
		4 * time.Millisecond,
		5 * time.Millisecond,
	}
	tests := []struct {
		name string
		p    float64
		want time.Duration
	}{
		{name: "min", p: 0, want: 1 * time.Millisecond},
		{name: "median", p: 0.50, want: 3 * time.Millisecond},
		{name: "p95", p: 0.95, want: 5 * time.Millisecond},
		{name: "p99", p: 0.99, want: 5 * time.Millisecond},
		{name: "max", p: 1, want: 5 * time.Millisecond},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := percentileDuration(durations, tt.p); got != tt.want {
				t.Fatalf("percentileDuration(p=%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestComputeRetrievalQualityUsesBoundedTopK(t *testing.T) {
	queries := []retrievalVectorRecord{{ID: "q", Vector: []float32{1}}}
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		docs[i] = retrievalVectorRecord{
			ID:     fmt.Sprintf("d%03d", i),
			Vector: []float32{float32(200 - i)},
		}
	}
	qrels := retrievalQrels{
		"q": {
			"d000": 1,
			"d009": 1,
			"d099": 1,
			"d100": 1,
		},
	}

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries := computeRetrievalQuality(queries, docs, qrels, 100)
	if queriesCount != 1 || relevantPairs != 4 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if quality.MRRAt10 != 1 {
		t.Fatalf("mrr@10 = %v, want 1", quality.MRRAt10)
	}
	if quality.PrecisionAt1 != 1 {
		t.Fatalf("precision@1 = %v, want 1", quality.PrecisionAt1)
	}
	if quality.PrecisionAt5 != 0.2 {
		t.Fatalf("precision@5 = %v, want 0.2", quality.PrecisionAt5)
	}
	if quality.PrecisionAt10 != 0.2 {
		t.Fatalf("precision@10 = %v, want 0.2", quality.PrecisionAt10)
	}
	if quality.HitAt1 != 1 || quality.HitAt5 != 1 || quality.HitAt10 != 1 {
		t.Fatalf("hit quality = %+v, want hits at 1/5/10", quality)
	}
	if quality.RecallAt10 != 0.5 {
		t.Fatalf("recall@10 = %v, want 0.5", quality.RecallAt10)
	}
	if quality.RecallAt100 != 0.75 {
		t.Fatalf("recall@100 = %v, want 0.75", quality.RecallAt100)
	}
	if math.Abs(quality.MAPAt10-0.3) > 1e-12 {
		t.Fatalf("map@10 = %.12f, want 0.300000000000", quality.MAPAt10)
	}
	if math.Abs(quality.MAPAt100-0.3075) > 1e-12 {
		t.Fatalf("map@100 = %.12f, want 0.307500000000", quality.MAPAt100)
	}
}

func TestEmbedRetrievalTextsGroupsByTokenLengthAndPreservesOrder(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}
	if err := model.attachTokenizer(tinyEmbeddingTokenizerFile()); err != nil {
		t.Fatalf("attach tokenizer: %v", err)
	}
	records := []retrievalTextRecord{
		{ID: "long-1", Text: "aa"},
		{ID: "short", Text: "a"},
		{ID: "long-2", Text: "aa"},
	}

	got, err := embedRetrievalTexts(context.Background(), model, records, 2)
	if err != nil {
		t.Fatalf("embed retrieval texts: %v", err)
	}
	if len(got) != len(records) {
		t.Fatalf("embedded rows = %d, want %d", len(got), len(records))
	}
	for i, record := range records {
		if got[i].ID != record.ID {
			t.Fatalf("row %d id = %q, want %q", i, got[i].ID, record.ID)
		}
		want, err := model.EmbedText(context.Background(), record.Text)
		if err != nil {
			t.Fatalf("embed text %q: %v", record.Text, err)
		}
		wantRows, err := embeddingRows(want.Embeddings, 1)
		if err != nil {
			t.Fatalf("embedding rows: %v", err)
		}
		wantVector := normalizeRetrievalVector(wantRows[0])
		for j, wantValue := range wantVector {
			if diff := math.Abs(float64(got[i].Vector[j] - wantValue)); diff > 1e-5 {
				t.Fatalf("row %d vector[%d] = %v, want %v", i, j, got[i].Vector[j], wantValue)
			}
		}
	}
}

func TestReadBEIRRetrievalFiles(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","title":"Title","text":"Document body"}`+"\n"+
			`{"_id":"d2","text":"Other document"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"document query"}`+"\n"+
			`{"_id":"q2","text":"unused query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	corpusPath, queriesPath, gotQrelsPath := BEIRRetrievalPaths(dir, "test")
	if gotQrelsPath != qrelsPath {
		t.Fatalf("qrels path = %q, want %q", gotQrelsPath, qrelsPath)
	}
	qrels, err := readBEIRQrels(gotQrelsPath)
	if err != nil {
		t.Fatalf("read qrels: %v", err)
	}
	corpus, err := readBEIRCorpus(corpusPath, 0)
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	queries, skipped, err := readBEIRQueries(queriesPath, qrels, 0)
	if err != nil {
		t.Fatalf("read queries: %v", err)
	}
	if len(corpus) != 2 || corpus[0].Text != "Title\nDocument body" {
		t.Fatalf("corpus = %+v", corpus)
	}
	if len(queries) != 1 || queries[0].ID != "q1" || skipped != 0 {
		t.Fatalf("queries = %+v skipped=%d", queries, skipped)
	}
}

func TestReadBEIRQrelsAcceptsTRECFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.tsv")
	if err := os.WriteFile(path, []byte("q1\tQ0\td1\t2\nq1\tQ0\td2\t0\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	qrels, err := readBEIRQrels(path)
	if err != nil {
		t.Fatalf("read qrels: %v", err)
	}
	if got := qrels["q1"]["d1"]; got != 2 {
		t.Fatalf("qrels[q1][d1] = %v, want 2", got)
	}
	if _, ok := qrels["q1"]["d2"]; ok {
		t.Fatalf("non-positive qrel was retained: %+v", qrels)
	}
}

func TestEvaluateBM25RetrievalRanksLexicalMatch(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha finance"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	metrics, err := EvaluateBM25Retrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny",
		CorpusPath:  corpusPath,
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		TopK:        100,
	})
	if err != nil {
		t.Fatalf("evaluate bm25: %v", err)
	}
	if metrics.Backend != "bm25" || metrics.Dataset != "tiny" {
		t.Fatalf("metrics identity = backend:%q dataset:%q", metrics.Backend, metrics.Dataset)
	}
	if metrics.Inputs.Documents != 2 || metrics.Inputs.Queries != 1 || metrics.Inputs.ScoredPairs != 2 {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 || metrics.Quality.RecallAt10 != 1 || metrics.Quality.RecallAt100 != 1 || metrics.Quality.MAPAt10 != 1 {
		t.Fatalf("quality = %+v, want perfect lexical ranking", metrics.Quality)
	}
}

func TestEvaluateVectorCacheRetrievalUsesBEIRQualityMetrics(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"alpha query"}`+"\n"+
			`{"_id":"q2","text":"beta query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"id":"d1","vector":[1,0]}`+"\n"+
			`{"id":"d2","vector":[0,1]}`+"\n"+
			`{"id":"d3","vector":[0.8,0.6]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"id":"q1","vector":[0.7,0.7]}`+"\n"+
			`{"id":"q2","vector":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	metrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny-vectors",
		ArtifactPath:    "external-model",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
		BackendName:     "external",
		TopK:            100,
	})
	if err != nil {
		t.Fatalf("evaluate vector cache retrieval: %v", err)
	}
	wantNDCG := (1/math.Log2(3) + 1) / 2
	if math.Abs(metrics.Quality.NDCGAt10-wantNDCG) > 1e-12 {
		t.Fatalf("ndcg@10 = %.12f, want %.12f", metrics.Quality.NDCGAt10, wantNDCG)
	}
	if metrics.Quality.MRRAt10 != 0.75 || metrics.Quality.RecallAt10 != 1 || metrics.Quality.RecallAt100 != 1 || metrics.Quality.HitAt1 != 0.5 || metrics.Quality.HitAt5 != 1 {
		t.Fatalf("quality = %+v", metrics.Quality)
	}
	if metrics.Schema != RetrievalEvalMetricsSchema || metrics.Dataset != "tiny-vectors" || metrics.Artifact != "external-model" || metrics.Backend != "external" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != queryVectorsPath {
		t.Fatalf("vector paths = %+v", metrics.Inputs)
	}
	if metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 2 || metrics.Inputs.RelevantPairs != 2 || metrics.Inputs.ScoredPairs != 6 {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
}

func TestEvaluateVectorCacheRetrievalWritesPerQueryJSONL(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	perQueryPath := filepath.Join(dir, "per-query.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t2\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"id":"d1","vector":[1,0]}`+"\n"+
			`{"id":"d2","vector":[0,1]}`+"\n"+
			`{"id":"d3","vector":[0.8,0.6]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"id":"q1","vector":[0.7,0.7]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	metrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-vectors",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   queryVectorsPath,
		TopK:              100,
		PerQueryJSONLPath: perQueryPath,
	})
	if err != nil {
		t.Fatalf("evaluate vector cache retrieval: %v", err)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("per-query lines = %d, want 1\n%s", len(lines), data)
	}
	var row RetrievalEvalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if row.Schema != RetrievalEvalPerQuerySchema || row.Dataset != "tiny-vectors" || row.QueryID != "q1" {
		t.Fatalf("row identity = %+v", row)
	}
	if row.RelevantCount != 1 || row.FirstRelevantRank != 2 {
		t.Fatalf("row relevant summary = count:%d first:%d, want count 1 first rank 2", row.RelevantCount, row.FirstRelevantRank)
	}
	if len(row.TopK) != 3 {
		t.Fatalf("top_k len = %d, want 3", len(row.TopK))
	}
	if row.TopK[0].Rank != 1 || row.TopK[0].DocID != "d3" || row.TopK[0].Relevance != 0 {
		t.Fatalf("top_k[0] = %+v, want d3 non-relevant", row.TopK[0])
	}
	if row.TopK[1].Rank != 2 || row.TopK[1].DocID != "d1" || row.TopK[1].Relevance != 2 {
		t.Fatalf("top_k[1] = %+v, want d1 relevance 2", row.TopK[1])
	}
	if math.Abs(row.Quality.NDCGAt10-metrics.Quality.NDCGAt10) > 1e-12 || row.Quality.MRRAt10 != 0.5 {
		t.Fatalf("row quality = %+v metrics quality = %+v", row.Quality, metrics.Quality)
	}
}

func TestEvaluateTurboQuantVectorCacheRetrievalUsesExternalCaches(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"alpha query"}`+"\n"+
			`{"_id":"q2","text":"beta query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"d2","embedding":[0,1,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"d3","embedding":[0,0,1,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"_id":"q1","embedding":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"q2","embedding":[0,1,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	metrics, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny-cache-tq",
		ArtifactPath:    "bge-cache",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
		BackendName:     "bge",
		TopK:            100,
	}, []int{8})
	if err != nil {
		t.Fatalf("evaluate turboquant vector cache retrieval: %v", err)
	}
	if metrics.Schema != TurboQuantRetrievalEvalMetricsSchema || metrics.Dataset != "tiny-cache-tq" || metrics.Artifact != "bge-cache" || metrics.Backend != "bge" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != queryVectorsPath {
		t.Fatalf("vector paths = %+v", metrics.Inputs)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || metrics.Dense.Quality.RecallAt100 != 1 {
		t.Fatalf("dense quality = %+v, want perfect", metrics.Dense.Quality)
	}
	if len(metrics.Rows) != 1 || metrics.Rows[0].Bits != 8 || metrics.Rows[0].CompressionRatio <= 1 {
		t.Fatalf("rows = %+v", metrics.Rows)
	}
	if metrics.Rows[0].Quality.NDCGAt10 < 0.99 || metrics.Rows[0].Quality.RecallAt100 != 1 {
		t.Fatalf("quantized quality = %+v, want near-perfect", metrics.Rows[0].Quality)
	}
}

func TestReadRetrievalChildVectorCachePreservesMultipleChildrenPerParent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "child-vectors.jsonl")
	if err := os.WriteFile(path, []byte(
		`{"parent_id":"p2","child_id":"p2-c1","vector":[0,1]}`+"\n"+
			`{"parent_id":"p1","child_id":"p1-c1","embedding":[1,0]}`+"\n"+
			`{"parent_id":"p1","child_id":"p1-c2","values":[0.8,0.6]}`+"\n"+
			`{"id":"p3","vector":[0,0.5]}`+"\n"+
			`{"parent_id":"skip","child_id":"skip-c1","vector":[1,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write child vectors: %v", err)
	}

	children, missing, dim, err := readRetrievalChildVectorCache(path, []string{"p1", "p2", "p3", "missing"})
	if err != nil {
		t.Fatalf("read child vectors: %v", err)
	}
	if missing != 1 || dim != 2 || len(children) != 4 {
		t.Fatalf("missing=%d dim=%d children=%d", missing, dim, len(children))
	}
	got := []string{}
	for _, child := range children {
		got = append(got, child.ParentID+"/"+child.ChildID)
	}
	want := []string{"p1/p1-c1", "p1/p1-c2", "p2/p2-c1", "p3/p3"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("child order = %v, want %v", got, want)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalAggregatesChildrenByParent(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p1", ChildID: "p1-b", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"p1": 1}}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-multivector",
		TopK:        100,
		BaselineDim: 32,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate multivector turboquant retrieval: %v", err)
	}
	if metrics.Schema != TurboQuantMultiVectorRetrievalEvalMetricsSchema || metrics.Dataset != "tiny-multivector" {
		t.Fatalf("metrics identity = schema:%q dataset:%q", metrics.Schema, metrics.Dataset)
	}
	if metrics.Inputs.Parents != 2 || metrics.Inputs.ParentCount != 2 || metrics.Inputs.ChildVectors != 3 || metrics.Inputs.ChildCount != 3 || metrics.Inputs.AverageChildrenPerParent != 1.5 || metrics.Inputs.AvgChildrenPerParent != 1.5 || metrics.Inputs.MaxChildrenPerParent != 2 || metrics.Inputs.ScoredChildPairs != 3 {
		t.Fatalf("input accounting = %+v", metrics.Inputs)
	}
	if metrics.Config.BaselineDim != 32 {
		t.Fatalf("baseline dim = %d", metrics.Config.BaselineDim)
	}
	if metrics.Dense.BaselineDim != 32 || metrics.Dense.DenseBaselineBytes != int64(32*4) || metrics.Dense.DenseParentBytes != int64(2*32*4) || metrics.Dense.DenseChildBytes != int64(3*8*4) {
		t.Fatalf("dense bytes = %+v", metrics.Dense)
	}
	if metrics.Dense.VectorsThatFitInOneDenseBaseline != 4 || metrics.Dense.StorageMultipleOfDenseBaseline != 0.375 {
		t.Fatalf("dense baseline accounting = %+v", metrics.Dense)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || metrics.Dense.Quality.MRRAt10 != 1 || metrics.Dense.Quality.RecallAt100 != 1 {
		t.Fatalf("dense quality = %+v, want parent p1 top-ranked by second child", metrics.Dense.Quality)
	}
	if len(metrics.Rows) != 1 || metrics.Rows[0].Bits != 8 {
		t.Fatalf("rows = %+v", metrics.Rows)
	}
	row := metrics.Rows[0]
	if row.Method != "turboquant_ip_b8_child_max" || row.QuantizedChildBytes <= 0 || row.DenseChildCompression <= 1 {
		t.Fatalf("row accounting = %+v", row)
	}
	if metrics.Config.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed || row.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("quantizer seeds = config:%d row:%d", metrics.Config.QuantizerSeed, row.QuantizerSeed)
	}
	if row.ParentBudgetStorageMultiple <= 0 || row.StorageMultipleOfDenseBaseline != row.ParentBudgetStorageMultiple || row.DenseParentBytes != metrics.Dense.DenseParentBytes || row.DenseBaselineBytes != metrics.Dense.DenseBaselineBytes || row.DenseChildBytes != metrics.Dense.DenseChildBytes {
		t.Fatalf("row storage = %+v", row)
	}
	if row.BaselineDim != 32 || row.ParentCount != 2 || row.ChildCount != 3 || row.AvgChildrenPerParent != 1.5 || row.MaxChildrenPerParent != 2 || row.QuantizedVectorBytes <= 0 || row.VectorsThatFitInOneDenseBaseline <= 0 {
		t.Fatalf("row baseline accounting = %+v", row)
	}
	if row.Quality.NDCGAt10 < 0.99 || row.Quality.RecallAt100 != 1 {
		t.Fatalf("quantized quality = %+v, want near-perfect", row.Quality)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalFailsMissingRelevantParentsByDefault(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"p1": 1, "missing-parent": 1}}

	_, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "missing-relevant",
		TopK:        100,
	}, []int{8}, children, queries, qrels)
	if err == nil || !strings.Contains(err.Error(), "missing 1 qrels-relevant parent documents") || !strings.Contains(err.Error(), "--allow-missing-relevant") {
		t.Fatalf("strict coverage error = %v", err)
	}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:          "missing-relevant",
		TopK:                 100,
		AllowMissingRelevant: true,
		QuantizerSeed:        17,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("allow missing relevant: %v", err)
	}
	if !metrics.Config.AllowMissingRelevant || metrics.Config.QuantizerSeed != 17 {
		t.Fatalf("config = %+v", metrics.Config)
	}
	if metrics.Rows[0].SkippedRelevantDocs != 1 || metrics.Rows[0].SkippedQueries != 0 {
		t.Fatalf("skipped counts = row:%+v skipped:%+v", metrics.Rows[0], metrics.SkippedCounts)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalSeededRowsRepeat(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{0.9, 0.1, 0.2, 0.3, 0.05, 0.01, 0.4, 0.2})},
		{ParentID: "p1", ChildID: "p1-b", Vector: normalizeRetrievalVector([]float32{0.1, 0.8, 0.1, 0.2, 0.3, 0.2, 0.1, 0.4})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0.2, 0.1, 0.9, 0.1, 0.4, 0.3, 0.2, 0.1})},
		{ParentID: "p3", ChildID: "p3-a", Vector: normalizeRetrievalVector([]float32{0.1, 0.2, 0.2, 0.9, 0.1, 0.4, 0.3, 0.2})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0.1, 0.2, 0.3, 0.1, 0, 0.4, 0.2})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0.1, 0.2, 1, 0.1, 0.4, 0.2, 0.1, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"p1": 1},
		"q2": {"p2": 1},
	}
	cfg := RetrievalEvalConfig{DatasetName: "seeded", TopK: 100, QuantizerSeed: 12345}

	first, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), cfg, []int{2, 4}, children, queries, qrels)
	if err != nil {
		t.Fatalf("first evaluation: %v", err)
	}
	second, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), cfg, []int{2, 4}, children, queries, qrels)
	if err != nil {
		t.Fatalf("second evaluation: %v", err)
	}
	if first.Config.QuantizerSeed != 12345 || second.Config.QuantizerSeed != 12345 {
		t.Fatalf("config seeds = %d/%d", first.Config.QuantizerSeed, second.Config.QuantizerSeed)
	}
	for i := range first.Rows {
		a, b := first.Rows[i], second.Rows[i]
		if a.Bits != b.Bits || a.Method != b.Method || a.QuantizerSeed != b.QuantizerSeed || a.QuantizedChildBytes != b.QuantizedChildBytes {
			t.Fatalf("row identity mismatch %d: %+v vs %+v", i, a, b)
		}
		if a.Quality != b.Quality || a.NDCGAt10Delta != b.NDCGAt10Delta || a.RecallAt100Delta != b.RecallAt100Delta {
			t.Fatalf("quality mismatch %d: %+v vs %+v", i, a, b)
		}
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalWritesPerQueryJSONL(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p1", ChildID: "p1-b", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"p1": 1}}
	perQueryPath := filepath.Join(t.TempDir(), "multivector.per-query.jsonl")

	_, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-multivector",
		TopK:              1,
		QuantizerSeed:     99,
		PerQueryJSONLPath: perQueryPath,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate multivector turboquant retrieval: %v", err)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query lines = %d, want dense and q8 rows\n%s", len(lines), data)
	}
	var dense, compact TurboQuantMultiVectorRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &dense); err != nil {
		t.Fatalf("decode dense row: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &compact); err != nil {
		t.Fatalf("decode compact row: %v", err)
	}
	if dense.Schema != TurboQuantMultiVectorRetrievalPerQuerySchema || dense.Dataset != "tiny-multivector" || dense.Method != "float32_child_max" || dense.QueryID != "q1" {
		t.Fatalf("dense row identity = %+v", dense)
	}
	if dense.FirstRelevantRank != 1 || dense.RelevantCount != 1 || len(dense.TopK) == 0 || dense.TopK[0].DocID != "p1" || dense.TopK[0].ChildID != "p1-b" || dense.TopK[0].ChildScore == nil {
		t.Fatalf("dense row evidence = %+v", dense)
	}
	if compact.Method != "turboquant_ip_b8_child_max" || compact.Bits != 8 || compact.QuantizerSeed != 99 || compact.ScoringSurface != "turboquant_ip_prepared_child_max" {
		t.Fatalf("compact row identity = %+v", compact)
	}
	if len(compact.TopK) == 0 || compact.TopK[0].CompactRank == nil || *compact.TopK[0].CompactRank != 1 || compact.TopK[0].CompactChildID == "" || compact.TopK[0].DenseRank == nil || compact.TopK[0].DenseChildID != "p1-b" {
		t.Fatalf("compact row evidence = %+v", compact)
	}
}

func TestEvaluateVectorCacheRetrievalRejectsDimensionMismatch(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(corpusPath, []byte(`{"_id":"d1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(`{"id":"d1","vector":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"id":"q1","vector":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	_, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
	})
	if err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("error = %v, want dimension mismatch", err)
	}
}

func TestMineBM25TextHardNegativesUsesTopLexicalNonPositive(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega unrelated"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	examples, summary, err := MineBM25TextHardNegatives(context.Background(), RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        2,
		MaxExamples:          1,
	})
	if err != nil {
		t.Fatalf("mine hard negatives: %v", err)
	}
	if summary.DatasetName != "tiny" || summary.Examples != 1 || summary.PositivePairs != 1 || summary.Negatives != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(examples) != 1 || examples[0].Query != "alpha" || examples[0].Positive != "alpha target" || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] != "alpha distractor" {
		t.Fatalf("examples = %+v", examples)
	}
}

func TestModelMiningNegativeTextsUsesTopModelNonPositive(t *testing.T) {
	scores := []retrievalScoredDoc{
		{ID: "positive", Score: 0.99},
		{ID: "hard", Score: 0.98},
		{ID: "duplicate", Score: 0.97},
		{ID: "blank", Score: 0.96},
		{ID: "easy", Score: 0.10},
	}
	positives := map[string]bool{"positive": true}
	docText := map[string]string{
		"positive":  "target",
		"hard":      "hard negative",
		"duplicate": "hard negative",
		"blank":     " ",
		"easy":      "easy negative",
	}

	negatives := modelMiningNegativeTexts(scores, positives, docText, RetrievalHardNegativeMiningConfig{
		NegativesPerPositive: 2,
		CandidateTopK:        4,
	})

	if len(negatives) != 2 || negatives[0] != "hard negative" || negatives[1] != "easy negative" {
		t.Fatalf("negatives = %+v", negatives)
	}
}
