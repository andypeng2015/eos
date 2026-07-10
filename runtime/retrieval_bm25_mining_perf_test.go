package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// --- DF-threshold pruning: unit-level correctness -------------------------

func TestDFPruneQueryTermsDropsHighDFTermsKeepsRareOnes(t *testing.T) {
	corpus := []retrievalTextRecord{
		{ID: "d1", Text: "rare common"},
		{ID: "d2", Text: "common filler"},
		{ID: "d3", Text: "common filler"},
		{ID: "d4", Text: "common filler"},
		{ID: "d5", Text: "common filler"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	// "common" DF = 5/5 = 1.0 (every doc); "rare" DF = 1/5 = 0.2.
	queryTokens := []string{"rare", "common"}

	pruned := dfPruneQueryTerms(queryTokens, index, 0.5)
	if !slices.Equal(pruned, []string{"rare"}) {
		t.Fatalf("pruned terms at threshold=0.5 = %v, want [rare]", pruned)
	}

	// threshold <= 0 or >= 1 must be a no-op (exhaustive, historical behavior).
	if got := dfPruneQueryTerms(queryTokens, index, 0); !slices.Equal(got, queryTokens) {
		t.Fatalf("threshold=0 pruned = %v, want unchanged %v", got, queryTokens)
	}
	if got := dfPruneQueryTerms(queryTokens, index, -0.3); !slices.Equal(got, queryTokens) {
		t.Fatalf("threshold=-0.3 pruned = %v, want unchanged %v", got, queryTokens)
	}
	if got := dfPruneQueryTerms(queryTokens, index, 1); !slices.Equal(got, queryTokens) {
		t.Fatalf("threshold=1 pruned = %v, want unchanged %v", got, queryTokens)
	}
}

func TestDFPruneQueryTermsFallsBackWhenEveryTermWouldBePruned(t *testing.T) {
	corpus := []retrievalTextRecord{
		{ID: "d1", Text: "common"},
		{ID: "d2", Text: "common"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	// "common" is the only query token and appears in every document: a
	// query made entirely of stopword-like terms must never be pruned down
	// to zero candidate-generating terms.
	queryTokens := []string{"common"}
	got := dfPruneQueryTerms(queryTokens, index, 0.1)
	if !slices.Equal(got, queryTokens) {
		t.Fatalf("all-terms-pruned fallback = %v, want unpruned %v", got, queryTokens)
	}
}

// bm25CandidateDocIndicesPruningFixture builds a corpus where "common"
// appears in every document (DF fraction 1.0) and "rare" appears only in
// two documents (the positive and one genuine hard negative). It returns
// the built index and the query tokens ["rare" "common"].
func bm25CandidateDocIndicesPruningFixture(t *testing.T) (bm25Index, []string) {
	t.Helper()
	corpus := []retrievalTextRecord{
		{ID: "positive", Text: "rare target passage common"},
		{ID: "hard-negative", Text: "rare distractor common"},
	}
	for i := 0; i < 10; i++ {
		corpus = append(corpus, retrievalTextRecord{ID: fmt.Sprintf("chaff-%d", i), Text: "common filler"})
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	return index, tokenizeBM25Text("rare common")
}

func TestBM25CandidateDocIndicesPruningShrinksCandidateSet(t *testing.T) {
	index, queryTokens := bm25CandidateDocIndicesPruningFixture(t)

	exhaustive := index
	exhaustive.DFPruneThreshold = 0
	exhaustiveCandidates := bm25CandidateDocIndices(queryTokens, exhaustive, newBM25CandidateScratch(len(index.Documents)))
	if len(exhaustiveCandidates) != len(index.Documents) {
		t.Fatalf("exhaustive candidates = %d, want all %d docs ('common' is in every doc)", len(exhaustiveCandidates), len(index.Documents))
	}

	pruned := index
	pruned.DFPruneThreshold = 0.5
	prunedCandidates := bm25CandidateDocIndices(queryTokens, pruned, newBM25CandidateScratch(len(index.Documents)))
	if len(prunedCandidates) != 2 {
		t.Fatalf("pruned candidates = %d, want 2 (positive + hard-negative, reached via 'rare' only)", len(prunedCandidates))
	}
}

func TestBM25CandidateDocIndicesPruningPreservesTopNegative(t *testing.T) {
	index, queryTokens := bm25CandidateDocIndicesPruningFixture(t)
	positiveIDs := map[string]bool{"positive": true}

	exhaustive := index
	exhaustive.DFPruneThreshold = 0
	exhaustiveTop := topBM25NonPositiveScores(queryTokens, positiveIDs, exhaustive, 1, newBM25CandidateScratch(len(index.Documents)))

	pruned := index
	pruned.DFPruneThreshold = 0.5
	prunedTop := topBM25NonPositiveScores(queryTokens, positiveIDs, pruned, 1, newBM25CandidateScratch(len(index.Documents)))

	if len(exhaustiveTop) != 1 || len(prunedTop) != 1 {
		t.Fatalf("top-1 negative counts = exhaustive:%d pruned:%d, want 1/1", len(exhaustiveTop), len(prunedTop))
	}
	if exhaustiveTop[0].ID != "hard-negative" {
		t.Fatalf("exhaustive top-1 negative = %q, want hard-negative", exhaustiveTop[0].ID)
	}
	if prunedTop[0].ID != "hard-negative" {
		t.Fatalf("pruned top-1 negative = %q, want hard-negative (pruning must not change the mined result here)", prunedTop[0].ID)
	}
}

// TestMineBM25TextHardNegativesDFPruneThresholdMatchesExhaustiveOnStopwordFixture
// exercises the same equivalence at the public MineBM25TextHardNegatives
// level (BEIR files on disk, full pipeline) rather than the internal
// candidate-generation helpers directly.
func TestMineBM25TextHardNegativesDFPruneThresholdMatchesExhaustiveOnStopwordFixture(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusLines := []string{
		`{"_id":"positive","text":"rare target passage common"}`,
		`{"_id":"hard-negative","text":"rare distractor common"}`,
	}
	for i := 0; i < 10; i++ {
		corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":"chaff-%d","text":"common filler"}`, i))
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	if err := os.WriteFile(corpusPath, []byte(strings.Join(corpusLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	queriesPath := filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"rare common"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	qrelsPath := filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\tpositive\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	baseCfg := RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        20,
		MiningWorkers:        1,
	}

	exhaustiveCfg := baseCfg
	exhaustiveCfg.DFPruneThreshold = 0
	exhaustiveExamples, _, err := MineBM25TextHardNegatives(context.Background(), exhaustiveCfg)
	if err != nil {
		t.Fatalf("exhaustive mine: %v", err)
	}

	prunedCfg := baseCfg
	prunedCfg.DFPruneThreshold = 0.5
	prunedExamples, _, err := MineBM25TextHardNegatives(context.Background(), prunedCfg)
	if err != nil {
		t.Fatalf("pruned mine: %v", err)
	}

	if len(exhaustiveExamples) != 1 || len(prunedExamples) != 1 {
		t.Fatalf("example counts = exhaustive:%d pruned:%d, want 1/1", len(exhaustiveExamples), len(prunedExamples))
	}
	want := []string{"rare distractor common"}
	if !slices.Equal(exhaustiveExamples[0].Negatives, want) {
		t.Fatalf("exhaustive negatives = %v, want %v", exhaustiveExamples[0].Negatives, want)
	}
	if !slices.Equal(prunedExamples[0].Negatives, want) {
		t.Fatalf("pruned negatives = %v, want %v (pruning must not change the mined result here)", prunedExamples[0].Negatives, want)
	}
}

// writeBM25MiningPruningParallelismFixture writes a corpus where a
// stopword-like term ("filler") appears in every one of numTopics*8 + 4*
// numTopics documents (100% DF, comfortably above both the >50%-of-corpus
// bar and the 0.10 DF-prune default) while each topic's own "signatureN"
// term appears only in that topic's 4 documents (a positive plus 3
// candidate negatives with strictly decreasing signature-term frequency, so
// BM25 ranks them in a fixed, non-tied order): well under the 0.10
// threshold, so DF pruning at the default genuinely drops "filler" from
// every query's candidate-generation term set while keeping "signatureN",
// without ever tripping the dfPruneQueryTerms "every term pruned" fallback.
// A pool of filler-only chaff documents pads the corpus past 30 documents
// and gives exhaustive (unpruned) candidate generation real additional
// (low-scoring) candidates that pruning must still correctly out-rank away
// -- i.e. pruning and exhaustive mining must reach the identical top-k here.
func writeBM25MiningPruningParallelismFixture(t *testing.T, numTopics int) (corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	var corpusLines, queryLines []string
	qrelLines := []string{"query-id\tcorpus-id\tscore"}
	for topic := 0; topic < numTopics; topic++ {
		signature := fmt.Sprintf("signature%d", topic)
		positiveID := fmt.Sprintf("t%d-positive", topic)
		corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":%q,"text":%q}`, positiveID, signature+" target filler"))
		for k, repeats := range []int{3, 2, 1} {
			negID := fmt.Sprintf("t%d-negative%d", topic, k)
			text := strings.TrimSpace(strings.Repeat(signature+" ", repeats)) + " filler"
			corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":%q,"text":%q}`, negID, text))
		}
		queryLines = append(queryLines, fmt.Sprintf(`{"_id":"query%d","text":%q}`, topic, signature+" filler"))
		qrelLines = append(qrelLines, fmt.Sprintf("query%d\t%s\t1", topic, positiveID))
	}
	// Filler-only chaff: shares the corpus-dominating "filler" term with
	// every on-topic document but none of the per-topic signature terms, so
	// it only competes as a (weak) BM25 candidate when "filler" itself
	// participates in candidate generation -- i.e. only under exhaustive
	// (threshold=0) mining, never under the pruned default.
	numChaff := 4 * numTopics
	for c := 0; c < numChaff; c++ {
		corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":"chaff-%d","text":"filler unrelated%d"}`, c, c))
	}
	corpusPath = filepath.Join(dir, "corpus.jsonl")
	if err := os.WriteFile(corpusPath, []byte(strings.Join(corpusLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	queriesPath = filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(queriesPath, []byte(strings.Join(queryLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	qrelsPath = filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(qrelsPath, []byte(strings.Join(qrelLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return corpusPath, queriesPath, qrelsPath
}

// TestMineBM25TextHardNegativesDFPruneThresholdMatchesExhaustiveWithRealPruningAndParallelism
// closes the coverage gap between DF pruning and query-parallel mining:
// every other pruning test uses a single worker, and every other
// parallelism test disables pruning (DFPruneThreshold left at its Go zero
// value, 0). Here both are exercised together on a fixture where pruning
// genuinely shrinks the candidate set (not the all-terms-pruned fallback).
func TestMineBM25TextHardNegativesDFPruneThresholdMatchesExhaustiveWithRealPruningAndParallelism(t *testing.T) {
	const numTopics = 6
	corpusPath, queriesPath, qrelsPath := writeBM25MiningPruningParallelismFixture(t, numTopics)

	corpus, err := readBEIRCorpus(corpusPath, 0)
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	if len(index.Documents) < 30 {
		t.Fatalf("fixture sanity: corpus = %d documents, want >= 30", len(index.Documents))
	}

	// Fixture sanity: DF pruning at the CLI default must genuinely shrink
	// the candidate set for these queries (drop "filler", keep
	// "signatureN") rather than no-op via the all-terms-pruned fallback.
	queryTokens := tokenizeBM25Text("signature0 filler")
	prunedTerms := dfPruneQueryTerms(queryTokens, index, DefaultBM25MiningDFPruneThreshold)
	if !slices.Equal(prunedTerms, []string{"signature0"}) {
		t.Fatalf("fixture sanity: pruned query terms = %v, want [signature0] (filler must be pruned, signature0 must survive)", prunedTerms)
	}
	exhaustiveIndex := index
	exhaustiveIndex.DFPruneThreshold = 0
	exhaustiveCandidates := bm25CandidateDocIndices(queryTokens, exhaustiveIndex, newBM25CandidateScratch(len(index.Documents)))
	prunedIndex := index
	prunedIndex.DFPruneThreshold = DefaultBM25MiningDFPruneThreshold
	prunedCandidates := bm25CandidateDocIndices(queryTokens, prunedIndex, newBM25CandidateScratch(len(index.Documents)))
	if len(prunedCandidates) >= len(exhaustiveCandidates) {
		t.Fatalf("fixture sanity: pruned candidates = %d, exhaustive candidates = %d; pruning must genuinely shrink the candidate set", len(prunedCandidates), len(exhaustiveCandidates))
	}

	baseCfg := RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 2,
		CandidateTopK:        50,
	}

	exhaustiveCfg := baseCfg
	exhaustiveCfg.DFPruneThreshold = 0
	exhaustiveCfg.MiningWorkers = 1
	exhaustive, _, err := MineBM25TextHardNegatives(context.Background(), exhaustiveCfg)
	if err != nil {
		t.Fatalf("exhaustive mine: %v", err)
	}
	if len(exhaustive) != numTopics {
		t.Fatalf("exhaustive examples = %d, want %d", len(exhaustive), numTopics)
	}

	prunedReferenceCfg := baseCfg
	prunedReferenceCfg.DFPruneThreshold = DefaultBM25MiningDFPruneThreshold
	prunedReferenceCfg.MiningWorkers = 1
	prunedReference, prunedReferenceSummary, err := MineBM25TextHardNegatives(context.Background(), prunedReferenceCfg)
	if err != nil {
		t.Fatalf("pruned reference (workers=1) mine: %v", err)
	}

	// (b) DF-pruned mining (the CLI default) matches exhaustive (threshold=0)
	// top-k on this fixture: pruning must not change WHICH hard negatives
	// are mined, only how cheaply candidate generation finds them.
	assertHardNegativeExamplesEqual(t, "pruned(workers=1) vs exhaustive", prunedReference, exhaustive)

	// (a) DF-pruned mining is byte-identical across worker counts.
	for _, workers := range []int{1, 4} {
		cfg := baseCfg
		cfg.DFPruneThreshold = DefaultBM25MiningDFPruneThreshold
		cfg.MiningWorkers = workers
		label := fmt.Sprintf("workers=%d", workers)
		got, gotSummary, err := MineBM25TextHardNegatives(context.Background(), cfg)
		if err != nil {
			t.Fatalf("%s: pruned mine: %v", label, err)
		}
		assertHardNegativeExamplesEqual(t, label, got, prunedReference)
		assertHardNegativeSummariesEqualModuloWorkers(t, label, gotSummary, prunedReferenceSummary)
	}
}

// --- Query-parallel mining: determinism -----------------------------------

// writeBM25MiningDeterminismFixture writes numQueries independent topics,
// each with one positive doc and numDocsPerQuery candidate docs, so that
// mining genuinely has per-query work to distribute across workers (as
// opposed to a single trivial query, which would not exercise a worker
// pool meaningfully).
func writeBM25MiningDeterminismFixture(t *testing.T, numQueries int) (corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	const numDocsPerQuery = 6
	var corpusLines, queryLines []string
	qrelLines := []string{"query-id\tcorpus-id\tscore"}
	for q := 0; q < numQueries; q++ {
		topic := fmt.Sprintf("topic%d", q)
		positiveID := fmt.Sprintf("q%d-positive", q)
		corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":%q,"text":%q}`, positiveID, topic+" alpha target"))
		for d := 0; d < numDocsPerQuery; d++ {
			docID := fmt.Sprintf("q%d-doc%d", q, d)
			text := fmt.Sprintf("%s alpha candidate%d filler%d", topic, d, (d*7+q*3)%5)
			corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":%q,"text":%q}`, docID, text))
		}
		queryLines = append(queryLines, fmt.Sprintf(`{"_id":"query%d","text":%q}`, q, topic+" alpha"))
		qrelLines = append(qrelLines, fmt.Sprintf("query%d\t%s\t1", q, positiveID))
	}
	corpusPath = filepath.Join(dir, "corpus.jsonl")
	if err := os.WriteFile(corpusPath, []byte(strings.Join(corpusLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	queriesPath = filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(queriesPath, []byte(strings.Join(queryLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	qrelsPath = filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(qrelsPath, []byte(strings.Join(qrelLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return corpusPath, queriesPath, qrelsPath
}

func assertHardNegativeExamplesEqual(t *testing.T, label string, got, want []EmbeddingTextHardNegativeExample) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: example count = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i].Query != want[i].Query || got[i].Positive != want[i].Positive {
			t.Fatalf("%s: example[%d] query/positive = (%q,%q), want (%q,%q)", label, i, got[i].Query, got[i].Positive, want[i].Query, want[i].Positive)
		}
		if !slices.Equal(got[i].Negatives, want[i].Negatives) {
			t.Fatalf("%s: example[%d] negatives = %v, want %v", label, i, got[i].Negatives, want[i].Negatives)
		}
		if !slices.Equal(got[i].TeacherScores, want[i].TeacherScores) {
			t.Fatalf("%s: example[%d] teacher scores = %v, want %v", label, i, got[i].TeacherScores, want[i].TeacherScores)
		}
	}
}

func assertHardNegativeSummariesEqualModuloWorkers(t *testing.T, label string, got, want RetrievalHardNegativeMiningSummary) {
	t.Helper()
	if got.Examples != want.Examples ||
		got.PositivePairs != want.PositivePairs ||
		got.Negatives != want.Negatives ||
		got.SkippedQueriesNoText != want.SkippedQueriesNoText ||
		got.SkippedPositiveDocs != want.SkippedPositiveDocs ||
		got.SkippedQueriesNoNegative != want.SkippedQueriesNoNegative ||
		got.DuplicatePositiveTextNegativesSkipped != want.DuplicatePositiveTextNegativesSkipped {
		t.Fatalf("%s: summary = %+v, want (modulo MiningWorkers) %+v", label, got, want)
	}
}

func TestMineBM25TextHardNegativesParallelWorkersAreDeterministic(t *testing.T) {
	const numQueries = 24
	corpusPath, queriesPath, qrelsPath := writeBM25MiningDeterminismFixture(t, numQueries)

	baseCfg := RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 3,
		CandidateTopK:        10,
	}

	referenceCfg := baseCfg
	referenceCfg.MiningWorkers = 1
	reference, referenceSummary, err := MineBM25TextHardNegatives(context.Background(), referenceCfg)
	if err != nil {
		t.Fatalf("reference (workers=1) mine: %v", err)
	}
	if len(reference) != numQueries {
		t.Fatalf("reference examples = %d, want %d", len(reference), numQueries)
	}

	for _, workers := range []int{2, 4, 8, 16} {
		for repeat := 0; repeat < 3; repeat++ {
			cfg := baseCfg
			cfg.MiningWorkers = workers
			label := fmt.Sprintf("workers=%d repeat=%d", workers, repeat)
			got, gotSummary, err := MineBM25TextHardNegatives(context.Background(), cfg)
			if err != nil {
				t.Fatalf("%s: mine: %v", label, err)
			}
			assertHardNegativeExamplesEqual(t, label, got, reference)
			assertHardNegativeSummariesEqualModuloWorkers(t, label, gotSummary, referenceSummary)
		}
	}
}

// TestMineBM25TextHardNegativesParallelWorkersRespectMaxExamplesDeterministically
// exercises the trickiest part of the parallel merge: cfg.MaxExamples must
// cut off the output at exactly the same point (same examples, same skip
// counters accrued only up to the cutoff) no matter how many workers
// computed the (necessarily eager, since workers do not know about each
// other's progress) per-query results feeding that merge.
func TestMineBM25TextHardNegativesParallelWorkersRespectMaxExamplesDeterministically(t *testing.T) {
	const numQueries = 24
	corpusPath, queriesPath, qrelsPath := writeBM25MiningDeterminismFixture(t, numQueries)

	baseCfg := RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 3,
		CandidateTopK:        10,
		MaxExamples:          numQueries / 2,
	}

	referenceCfg := baseCfg
	referenceCfg.MiningWorkers = 1
	reference, referenceSummary, err := MineBM25TextHardNegatives(context.Background(), referenceCfg)
	if err != nil {
		t.Fatalf("reference (workers=1) mine: %v", err)
	}
	if len(reference) != numQueries/2 {
		t.Fatalf("reference examples = %d, want %d (MaxExamples cutoff)", len(reference), numQueries/2)
	}
	if referenceSummary.SkippedPositiveDocs != 0 && referenceSummary.SkippedPositiveDocs >= numQueries {
		t.Fatalf("reference SkippedPositiveDocs = %d looks like it counted queries past the MaxExamples cutoff", referenceSummary.SkippedPositiveDocs)
	}

	for _, workers := range []int{2, 4, 8} {
		for repeat := 0; repeat < 3; repeat++ {
			cfg := baseCfg
			cfg.MiningWorkers = workers
			label := fmt.Sprintf("workers=%d repeat=%d", workers, repeat)
			got, gotSummary, err := MineBM25TextHardNegatives(context.Background(), cfg)
			if err != nil {
				t.Fatalf("%s: mine: %v", label, err)
			}
			assertHardNegativeExamplesEqual(t, label, got, reference)
			assertHardNegativeSummariesEqualModuloWorkers(t, label, gotSummary, referenceSummary)
		}
	}
}

// writeBM25MiningMultiPositiveDeterminismFixture is like
// writeBM25MiningDeterminismFixture but gives each of numQueries topics
// positivesPerQuery qrel-relevant positive documents (multiple qrel lines
// per query) instead of one, so MineBM25TextHardNegatives emits
// positivesPerQuery examples per query -- one per positive, all sharing that
// query's own negative-candidate pool -- instead of the single-example
// -per-query shape every other fixture in this file produces. That is what
// makes a MaxExamples cutoff land mid-query (partway through one query's own
// multiple examples, not just at a query boundary) meaningful to test.
func writeBM25MiningMultiPositiveDeterminismFixture(t *testing.T, numQueries, positivesPerQuery int) (corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	const numDocsPerQuery = 6
	var corpusLines, queryLines []string
	qrelLines := []string{"query-id\tcorpus-id\tscore"}
	for q := 0; q < numQueries; q++ {
		topic := fmt.Sprintf("topic%d", q)
		for p := 0; p < positivesPerQuery; p++ {
			positiveID := fmt.Sprintf("q%d-positive%d", q, p)
			corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":%q,"text":%q}`, positiveID, fmt.Sprintf("%s alpha target%d", topic, p)))
			qrelLines = append(qrelLines, fmt.Sprintf("query%d\t%s\t1", q, positiveID))
		}
		for d := 0; d < numDocsPerQuery; d++ {
			docID := fmt.Sprintf("q%d-doc%d", q, d)
			text := fmt.Sprintf("%s alpha candidate%d filler%d", topic, d, (d*7+q*3)%5)
			corpusLines = append(corpusLines, fmt.Sprintf(`{"_id":%q,"text":%q}`, docID, text))
		}
		queryLines = append(queryLines, fmt.Sprintf(`{"_id":"query%d","text":%q}`, q, topic+" alpha"))
	}
	corpusPath = filepath.Join(dir, "corpus.jsonl")
	if err := os.WriteFile(corpusPath, []byte(strings.Join(corpusLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	queriesPath = filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(queriesPath, []byte(strings.Join(queryLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	qrelsPath = filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(qrelsPath, []byte(strings.Join(qrelLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return corpusPath, queriesPath, qrelsPath
}

// TestMineBM25TextHardNegativesParallelWorkersRespectMaxExamplesMidQueryCutoffWithMultiplePositives
// extends the MaxExamples-cutoff coverage to queries with MULTIPLE
// positives: TestMineBM25TextHardNegativesParallelWorkersRespectMaxExamplesDeterministically
// only ever lands its cutoff on a query boundary (one example per query).
// With multiple positives per query, the inner per-example break in
// MineBM25TextHardNegatives' sequential merge loop can fire mid-query --
// including only some of one query's own examples -- which is a materially
// different code path that must still be reproduced byte-for-byte
// regardless of worker count.
func TestMineBM25TextHardNegativesParallelWorkersRespectMaxExamplesMidQueryCutoffWithMultiplePositives(t *testing.T) {
	const numQueries = 6
	const positivesPerQuery = 3
	corpusPath, queriesPath, qrelsPath := writeBM25MiningMultiPositiveDeterminismFixture(t, numQueries, positivesPerQuery)

	baseCfg := RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 2,
		CandidateTopK:        10,
		// 2 full queries (3 examples each = 6) plus 2 of the third query's 3
		// examples: a genuine mid-query cutoff, not a query-boundary one.
		MaxExamples: 8,
	}

	referenceCfg := baseCfg
	referenceCfg.MiningWorkers = 1
	reference, referenceSummary, err := MineBM25TextHardNegatives(context.Background(), referenceCfg)
	if err != nil {
		t.Fatalf("reference (workers=1) mine: %v", err)
	}
	if len(reference) != baseCfg.MaxExamples {
		t.Fatalf("reference examples = %d, want MaxExamples cutoff %d", len(reference), baseCfg.MaxExamples)
	}
	if referenceSummary.PositivePairs != baseCfg.MaxExamples || referenceSummary.Examples != baseCfg.MaxExamples {
		t.Fatalf("reference summary = %+v, want PositivePairs/Examples == MaxExamples %d", referenceSummary, baseCfg.MaxExamples)
	}

	// Fixture sanity: confirm the cutoff genuinely lands mid-query. The last
	// accepted example's query must NOT have contributed all
	// positivesPerQuery of its own examples, or this run never exercises
	// the inner (mid-query) break at all -- it would degenerate into the
	// already-covered query-boundary case.
	lastQuery := reference[len(reference)-1].Query
	sameQueryCount := 0
	for _, example := range reference {
		if example.Query == lastQuery {
			sameQueryCount++
		}
	}
	if sameQueryCount == positivesPerQuery {
		t.Fatalf("fixture sanity: MaxExamples=%d cutoff landed on a query boundary (last query %q kept all %d examples); want a mid-query cutoff", baseCfg.MaxExamples, lastQuery, positivesPerQuery)
	}

	for _, workers := range []int{1, 2, 8} {
		for repeat := 0; repeat < 3; repeat++ {
			cfg := baseCfg
			cfg.MiningWorkers = workers
			label := fmt.Sprintf("workers=%d repeat=%d", workers, repeat)
			got, gotSummary, err := MineBM25TextHardNegatives(context.Background(), cfg)
			if err != nil {
				t.Fatalf("%s: mine: %v", label, err)
			}
			assertHardNegativeExamplesEqual(t, label, got, reference)
			assertHardNegativeSummariesEqualModuloWorkers(t, label, gotSummary, referenceSummary)
		}
	}
}

func TestMineBM25TextHardNegativesMiningWorkersDefaultsWithoutError(t *testing.T) {
	corpusPath, queriesPath, qrelsPath := writeBM25MiningDeterminismFixture(t, 4)
	cfg := RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        5,
		// MiningWorkers left unset (0): must default rather than mine with
		// zero workers.
	}
	examples, summary, err := MineBM25TextHardNegatives(context.Background(), cfg)
	if err != nil {
		t.Fatalf("mine with default workers: %v", err)
	}
	if len(examples) != 4 {
		t.Fatalf("examples = %d, want 4", len(examples))
	}
	if summary.MiningWorkers < 1 {
		t.Fatalf("summary.MiningWorkers = %d, want >= 1 (auto default)", summary.MiningWorkers)
	}
}

// TestMineBM25TextHardNegativesSummaryMiningWorkersClampedToQueryCount pins
// summary.MiningWorkers to the effective (post-clamp) worker count
// mineBM25QueriesParallel actually ran with, not the requested/defaulted
// cfg.MiningWorkers: on a 1-query dataset, mining only ever uses 1 worker
// (a worker with no query to process would be wasted) regardless of the
// auto default (normally > 1 on a multi-core machine) or an explicit
// request for more, so the summary -- and the CLI's "mining_workers="
// provenance echo built from it -- must report 1, not the pre-clamp value.
func TestMineBM25TextHardNegativesSummaryMiningWorkersClampedToQueryCount(t *testing.T) {
	corpusPath, queriesPath, qrelsPath := writeBM25MiningDeterminismFixture(t, 1)
	for _, requested := range []int{0, 1, 5} {
		label := fmt.Sprintf("requested=%d", requested)
		cfg := RetrievalHardNegativeMiningConfig{
			DatasetName:          "tiny",
			CorpusPath:           corpusPath,
			QueriesPath:          queriesPath,
			QrelsPath:            qrelsPath,
			NegativesPerPositive: 1,
			CandidateTopK:        5,
			MiningWorkers:        requested, // 0 means "auto default".
		}
		examples, summary, err := MineBM25TextHardNegatives(context.Background(), cfg)
		if err != nil {
			t.Fatalf("%s: mine: %v", label, err)
		}
		if len(examples) != 1 {
			t.Fatalf("%s: examples = %d, want 1", label, len(examples))
		}
		if summary.MiningWorkers != 1 {
			t.Fatalf("%s: summary.MiningWorkers = %d, want 1 (effective workers clamped to the single query, not the requested/default value)", label, summary.MiningWorkers)
		}
	}
}

// --- bm25CandidateScratch: epoch-stamped accumulator semantics ------------

func TestBM25CandidateScratchEpochResetIsolatesQueries(t *testing.T) {
	scratch := newBM25CandidateScratch(5)
	scratch.reset(5)
	if !scratch.markIfNew(2) {
		t.Fatalf("first mark of 2 should report new")
	}
	if scratch.markIfNew(2) {
		t.Fatalf("second mark of 2 in the same epoch should report not-new")
	}
	if !scratch.contains(2) {
		t.Fatalf("contains(2) should be true after marking")
	}
	if scratch.contains(3) {
		t.Fatalf("contains(3) should be false; 3 was never marked")
	}

	scratch.reset(5) // simulate moving on to the next query
	if scratch.contains(2) {
		t.Fatalf("contains(2) should be false after reset (new epoch)")
	}
	if !scratch.markIfNew(2) {
		t.Fatalf("2 should be markable again as new after reset")
	}
}

func TestBM25CandidateScratchGrowsForLargerCorpus(t *testing.T) {
	scratch := newBM25CandidateScratch(2)
	scratch.reset(2)
	if !scratch.markIfNew(1) {
		t.Fatalf("mark of 1 should report new")
	}
	// Grow to a larger corpus mid-lifetime (defensive path; corpus size is
	// normally fixed within a run, but reset must not panic or lose
	// previously marked low indices when growing).
	scratch.reset(10)
	if scratch.contains(1) {
		t.Fatalf("contains(1) should be false: reset always advances the epoch even when growing")
	}
	if !scratch.markIfNew(9) {
		t.Fatalf("mark of 9 (within grown bounds) should report new")
	}
	if !scratch.contains(9) {
		t.Fatalf("contains(9) should be true after marking")
	}
}

// --- Multi-query eval call sites: reused-scratch epoch isolation ----------
//
// computeBM25RetrievalQuality (runtime/retrieval_bm25.go) and
// computeHybridRetrievalQuality (runtime/retrieval_hybrid.go) each allocate
// one bm25CandidateScratch and reuse it across their whole query loop
// instead of a fresh candidate-dedup set per query. Every existing test that
// reaches either function uses exactly one query, so the reused scratch
// never actually crosses a query boundary in any prior test. The tests below
// pin that a query processed as part of a multi-query batch sharing that
// reused scratch produces byte-identical output to the same query computed
// alone (a fresh, single-use scratch) -- i.e. the epoch stamping in
// bm25CandidateScratch never lets one query's marks leak into another's.

// readRetrievalPerQueryRows decodes every line of a per-query JSONL file
// written by newRetrievalPerQueryWriter into RetrievalEvalPerQueryRow, in
// file order (== query-processing order).
func readRetrievalPerQueryRows(t *testing.T, path string) []RetrievalEvalPerQueryRow {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read per-query JSONL %s: %v", path, err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	rows := make([]RetrievalEvalPerQueryRow, len(lines))
	for i, line := range lines {
		if err := json.Unmarshal([]byte(line), &rows[i]); err != nil {
			t.Fatalf("decode per-query row %d from %s: %v", i, path, err)
		}
	}
	return rows
}

// assertRetrievalPerQueryRowsIdentical fails unless got and want marshal to
// exactly the same JSON.
func assertRetrievalPerQueryRowsIdentical(t *testing.T, label string, got, want RetrievalEvalPerQueryRow) {
	t.Helper()
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("%s: marshal batched row: %v", label, err)
	}
	wantJSON, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("%s: marshal fresh row: %v", label, err)
	}
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("%s: per-query row mismatch\nbatched=%s\nfresh=  %s", label, gotJSON, wantJSON)
	}
}

func TestComputeBM25RetrievalQualityMultiQueryMatchesFreshSingleQueryComputation(t *testing.T) {
	// Three independent topics against one shared corpus/index: each
	// query's own positive/negative pair is reachable only through that
	// topic's own token (alpha/beta/gamma), so scoring every query pulls in
	// the OTHER topics' documents too (via computeBM25RetrievalQuality's
	// topBM25Scores "backfill" pass, which scores every remaining
	// non-candidate document at 0 so the ranking still covers the whole
	// corpus) -- exactly the scratch.contains() path that would wrongly
	// drop a previously-marked document from a later query's ranking if
	// epoch isolation were broken.
	corpus := []retrievalTextRecord{
		{ID: "q1-positive", Text: "alpha target"},
		{ID: "q1-negative", Text: "filler"},
		{ID: "q2-positive", Text: "beta target"},
		{ID: "q2-negative", Text: "filler"},
		{ID: "q3-positive", Text: "gamma target"},
		{ID: "q3-negative", Text: "filler"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	queries, err := tokenizeBM25Queries(context.Background(), []retrievalTextRecord{
		{ID: "q1", Text: "alpha"},
		{ID: "q2", Text: "beta"},
		{ID: "q3", Text: "gamma"},
	})
	if err != nil {
		t.Fatalf("tokenize queries: %v", err)
	}
	qrels := retrievalQrels{
		"q1": {"q1-positive": 1},
		"q2": {"q2-positive": 1},
		"q3": {"q3-positive": 1},
	}

	dir := t.TempDir()
	batchedPath := filepath.Join(dir, "batched.jsonl")
	batchedQuality, batchedQueries, batchedRelevant, _, _, err := computeBM25RetrievalQuality(context.Background(), queries, index, qrels, 100, "tiny", batchedPath)
	if err != nil {
		t.Fatalf("batched compute: %v", err)
	}
	if batchedQueries != 3 || batchedRelevant != 3 {
		t.Fatalf("batched counts = queries:%d relevant:%d, want 3/3", batchedQueries, batchedRelevant)
	}
	if batchedQuality.NDCGAt10 != 1 || batchedQuality.MRRAt10 != 1 {
		t.Fatalf("batched quality = %+v, want perfect top hit across all 3 queries", batchedQuality)
	}
	batchedRows := readRetrievalPerQueryRows(t, batchedPath)
	if len(batchedRows) != 3 {
		t.Fatalf("batched rows = %d, want 3", len(batchedRows))
	}

	for i, query := range queries {
		freshPath := filepath.Join(dir, fmt.Sprintf("fresh-%s.jsonl", query.ID))
		freshQuality, freshQueries, freshRelevant, _, _, err := computeBM25RetrievalQuality(context.Background(), []bm25Query{query}, index, qrels, 100, "tiny", freshPath)
		if err != nil {
			t.Fatalf("fresh compute %s: %v", query.ID, err)
		}
		if freshQueries != 1 || freshRelevant != 1 {
			t.Fatalf("fresh counts %s = queries:%d relevant:%d, want 1/1", query.ID, freshQueries, freshRelevant)
		}
		if freshQuality.NDCGAt10 != 1 || freshQuality.MRRAt10 != 1 {
			t.Fatalf("fresh quality %s = %+v, want perfect top hit", query.ID, freshQuality)
		}
		freshRows := readRetrievalPerQueryRows(t, freshPath)
		if len(freshRows) != 1 {
			t.Fatalf("fresh rows %s = %d, want 1", query.ID, len(freshRows))
		}
		if batchedRows[i].QueryID != query.ID || freshRows[0].QueryID != query.ID {
			t.Fatalf("row identity mismatch at %d: batched=%q fresh=%q want %q", i, batchedRows[i].QueryID, freshRows[0].QueryID, query.ID)
		}
		if len(batchedRows[i].TopK) != len(freshRows[0].TopK) {
			t.Fatalf("query=%s: batched top_k len=%d fresh top_k len=%d, want equal (a stale scratch mark would silently drop a document from a later query's ranking)", query.ID, len(batchedRows[i].TopK), len(freshRows[0].TopK))
		}
		assertRetrievalPerQueryRowsIdentical(t, fmt.Sprintf("query=%s", query.ID), batchedRows[i], freshRows[0])
	}
}

func TestComputeHybridRetrievalQualityMultiQueryMatchesFreshSingleQueryComputation(t *testing.T) {
	// Mirrors TestComputeBM25RetrievalQualityMultiQueryMatchesFreshSingleQueryComputation
	// for computeHybridRetrievalQuality (runtime/retrieval_hybrid.go), which
	// reuses its own bm25CandidateScratch (bm25Scratch) across the same kind
	// of multi-query loop while also fusing in per-query dense scores.
	corpus := []retrievalTextRecord{
		{ID: "q1-positive", Text: "alpha target"},
		{ID: "q1-negative", Text: "filler"},
		{ID: "q2-positive", Text: "beta target"},
		{ID: "q2-negative", Text: "filler"},
		{ID: "q3-positive", Text: "gamma target"},
		{ID: "q3-negative", Text: "filler"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	docs := []retrievalVectorRecord{
		{ID: "q1-positive", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0})},
		{ID: "q1-negative", Vector: normalizeRetrievalVector([]float32{0, 0, 0, 1})},
		{ID: "q2-positive", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0})},
		{ID: "q2-negative", Vector: normalizeRetrievalVector([]float32{0, 0, 0, 1})},
		{ID: "q3-positive", Vector: normalizeRetrievalVector([]float32{0, 0, 1, 0})},
		{ID: "q3-negative", Vector: normalizeRetrievalVector([]float32{0, 0, 0, 1})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0})},
		{ID: "q3", Vector: normalizeRetrievalVector([]float32{0, 0, 1, 0})},
	}
	bm25Queries := map[string][]string{
		"q1": tokenizeBM25Text("alpha"),
		"q2": tokenizeBM25Text("beta"),
		"q3": tokenizeBM25Text("gamma"),
	}
	qrels := retrievalQrels{
		"q1": {"q1-positive": 1},
		"q2": {"q2-positive": 1},
		"q3": {"q3-positive": 1},
	}
	hybridCfg := RetrievalEvalHybridConfig{Method: "minmax_blend", Alpha: 0.75, RRFK: 60, RRFLambda: 1}

	dir := t.TempDir()
	batchedPath := filepath.Join(dir, "batched.jsonl")
	batchedQuality, batchedQueries, batchedRelevant, _, _, err := computeHybridRetrievalQuality(context.Background(), queries, docs, bm25Queries, index, qrels, 100, "tiny", batchedPath, hybridCfg)
	if err != nil {
		t.Fatalf("batched compute: %v", err)
	}
	if batchedQueries != 3 || batchedRelevant != 3 {
		t.Fatalf("batched counts = queries:%d relevant:%d, want 3/3", batchedQueries, batchedRelevant)
	}
	if batchedQuality.NDCGAt10 != 1 || batchedQuality.MRRAt10 != 1 {
		t.Fatalf("batched quality = %+v, want perfect top hit across all 3 queries", batchedQuality)
	}
	batchedRows := readRetrievalPerQueryRows(t, batchedPath)
	if len(batchedRows) != 3 {
		t.Fatalf("batched rows = %d, want 3", len(batchedRows))
	}

	for i, query := range queries {
		freshPath := filepath.Join(dir, fmt.Sprintf("fresh-%s.jsonl", query.ID))
		freshQuality, freshQueries, freshRelevant, _, _, err := computeHybridRetrievalQuality(context.Background(), []retrievalVectorRecord{query}, docs, bm25Queries, index, qrels, 100, "tiny", freshPath, hybridCfg)
		if err != nil {
			t.Fatalf("fresh compute %s: %v", query.ID, err)
		}
		if freshQueries != 1 || freshRelevant != 1 {
			t.Fatalf("fresh counts %s = queries:%d relevant:%d, want 1/1", query.ID, freshQueries, freshRelevant)
		}
		if freshQuality.NDCGAt10 != 1 || freshQuality.MRRAt10 != 1 {
			t.Fatalf("fresh quality %s = %+v, want perfect top hit", query.ID, freshQuality)
		}
		freshRows := readRetrievalPerQueryRows(t, freshPath)
		if len(freshRows) != 1 {
			t.Fatalf("fresh rows %s = %d, want 1", query.ID, len(freshRows))
		}
		if batchedRows[i].QueryID != query.ID || freshRows[0].QueryID != query.ID {
			t.Fatalf("row identity mismatch at %d: batched=%q fresh=%q want %q", i, batchedRows[i].QueryID, freshRows[0].QueryID, query.ID)
		}
		if len(batchedRows[i].TopK) != len(freshRows[0].TopK) {
			t.Fatalf("query=%s: batched top_k len=%d fresh top_k len=%d, want equal (a stale scratch mark would silently drop a document from a later query's ranking)", query.ID, len(batchedRows[i].TopK), len(freshRows[0].TopK))
		}
		assertRetrievalPerQueryRowsIdentical(t, fmt.Sprintf("query=%s", query.ID), batchedRows[i], freshRows[0])
	}
}
