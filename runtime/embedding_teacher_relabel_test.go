package eosruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func relabelConfigForTest() RelabelTeacherNegativesConfig {
	return RelabelTeacherNegativesConfig{
		PromoteMin:           0.92,
		PromoteSlack:         0.05,
		PromoteCap:           2,
		NegativeMax:          0.80,
		NegativesPerRow:      2,
		PromotedSourceSuffix: "promoted",
	}
}

func TestRelabelTeacherNegativesPromotesAndKeeps(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"hot", "cold", "warm"},
			TeacherScores: []float32{0.90, 0.95, 0.40, 0.85},
		},
	}
	out, summary, err := RelabelTeacherNegatives(examples, nil, relabelConfigForTest())
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if summary.Promoted != 1 || summary.TrueNegativesKept != 1 || summary.DroppedAmbiguous != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(out) != 2 {
		t.Fatalf("rows = %d, want 2 (original + promoted)", len(out))
	}
	original := out[0]
	if original.Positive != "pos1" || len(original.Negatives) != 1 || original.Negatives[0] != "cold" {
		t.Fatalf("original row = %+v", original)
	}
	if len(original.TeacherScores) != 2 || original.TeacherScores[0] != 0.90 || original.TeacherScores[1] != 0.40 {
		t.Fatalf("original teacher scores = %v", original.TeacherScores)
	}
	promoted := out[1]
	if promoted.Positive != "hot" || promoted.Source != "fiqa:promoted" {
		t.Fatalf("promoted row = %+v", promoted)
	}
	if len(promoted.TeacherScores) != 2 || promoted.TeacherScores[0] != 0.95 {
		t.Fatalf("promoted teacher scores = %v", promoted.TeacherScores)
	}
}

func TestRelabelTeacherNegativesPromoteSlackBlocksWeakCandidates(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"close-but-weaker"},
			TeacherScores: []float32{0.99, 0.93},
		},
	}
	out, summary, err := RelabelTeacherNegatives(examples, nil, relabelConfigForTest())
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if summary.Promoted != 0 || summary.DroppedAmbiguous != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(out) != 1 {
		t.Fatalf("rows = %d, want 1", len(out))
	}
}

func TestRelabelTeacherNegativesPromoteCapKeepsHighestScores(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"a", "b", "c"},
			TeacherScores: []float32{0.80, 0.93, 0.97, 0.95},
		},
	}
	out, summary, err := RelabelTeacherNegatives(examples, nil, relabelConfigForTest())
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if summary.Promoted != 2 || summary.PromotedCapSkipped != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(out) != 3 {
		t.Fatalf("rows = %d, want 3", len(out))
	}
	if out[1].Positive != "b" || out[2].Positive != "c" {
		t.Fatalf("promoted order = %q, %q (want b then c)", out[1].Positive, out[2].Positive)
	}
}

func TestRelabelTeacherNegativesAttachesPoolNegatives(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"promoteme"},
			TeacherScores: []float32{0.90, 0.96},
		},
	}
	poolSource := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa:random",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"rand1", "rand2", "too-relevant"},
			TeacherScores: []float32{0.90, 0.30, 0.50, 0.95},
		},
	}
	pool := BuildTeacherNegativePool(poolSource, 0.80)
	if len(pool["q1"]) != 2 {
		t.Fatalf("pool = %+v", pool)
	}
	out, summary, err := RelabelTeacherNegatives(examples, pool, relabelConfigForTest())
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if summary.PoolQueries != 1 || summary.PoolNegatives != 2 {
		t.Fatalf("summary = %+v", summary)
	}
	for _, row := range out {
		if len(row.Negatives) != 2 {
			t.Fatalf("row %q negatives = %v, want 2 pool negatives", row.Positive, row.Negatives)
		}
		if row.Negatives[0] != "rand1" || row.Negatives[1] != "rand2" {
			t.Fatalf("row %q negatives = %v (want lowest-score-first pool order)", row.Positive, row.Negatives)
		}
		if len(row.TeacherScores) != 3 {
			t.Fatalf("row %q teacher scores = %v", row.Positive, row.TeacherScores)
		}
	}
}

func TestRelabelTeacherNegativesUnscoredRowsKeepPairOnly(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{Source: "fiqa", Query: "q1", Positive: "pos1", Negatives: []string{"polluted"}},
	}
	out, summary, err := RelabelTeacherNegatives(examples, nil, relabelConfigForTest())
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if summary.UnscoredExamples != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(out) != 1 || len(out[0].Negatives) != 0 || len(out[0].TeacherScores) != 0 {
		t.Fatalf("rows = %+v", out)
	}
}

func TestRelabelTeacherNegativesDeduplicatesQueryPositivePairs(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"shared-doc"},
			TeacherScores: []float32{0.90, 0.96},
		},
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "shared-doc",
			Negatives:     []string{"low"},
			TeacherScores: []float32{0.95, 0.20},
		},
	}
	out, summary, err := RelabelTeacherNegatives(examples, nil, relabelConfigForTest())
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if summary.DuplicateRowsSkipped != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	count := 0
	for _, row := range out {
		if row.Query == "q1" && row.Positive == "shared-doc" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("shared-doc rows = %d, want 1", count)
	}
}

func TestRelabelTeacherNegativesEmitPairs(t *testing.T) {
	cfg := relabelConfigForTest()
	cfg.EmitPairs = true
	examples := []EmbeddingTextHardNegativeExample{
		{
			Source:        "fiqa",
			Query:         "q1",
			Positive:      "pos1",
			Negatives:     []string{"hot", "cold"},
			TeacherScores: []float32{0.90, 0.95, 0.40},
		},
	}
	out, _, err := RelabelTeacherNegatives(examples, nil, cfg)
	if err != nil {
		t.Fatalf("relabel: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("rows = %d, want 2", len(out))
	}
	for _, row := range out {
		if len(row.Negatives) != 0 || len(row.TeacherScores) != 0 {
			t.Fatalf("pairs row carries negatives: %+v", row)
		}
	}
}

func TestRelabelTeacherNegativesValidatesThresholds(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{Source: "fiqa", Query: "q1", Positive: "p", Negatives: []string{"n"}, TeacherScores: []float32{0.9, 0.5}},
	}
	cfg := relabelConfigForTest()
	cfg.NegativeMax = 0.95
	if _, _, err := RelabelTeacherNegatives(examples, nil, cfg); err == nil {
		t.Fatal("expected error when negative-max >= promote-min")
	}
}

func TestSummarizeTeacherScores(t *testing.T) {
	examples := []EmbeddingTextHardNegativeExample{
		{Query: "q1", Positive: "p", Negatives: []string{"a", "b"}, TeacherScores: []float32{0.9, 0.5, 0.7}},
		{Query: "q2", Positive: "p2", Negatives: []string{"c"}},
	}
	stats := SummarizeTeacherScores(examples)
	if stats.ScoredExamples != 1 || stats.UnscoredExamples != 1 || stats.Positives != 1 || stats.Negatives != 2 {
		t.Fatalf("stats = %+v", stats)
	}
	rendered := FormatTeacherScoreQuantiles(stats)
	if !strings.Contains(rendered, "positive_scores") || !strings.Contains(rendered, "q50") {
		t.Fatalf("rendered = %q", rendered)
	}
}

func TestSampleCorpusNegatives(t *testing.T) {
	dir := t.TempDir()
	var corpus strings.Builder
	corpus.WriteString(`{"_id":"d1","title":"T1","text":"relevant doc"}` + "\n")
	for i := 2; i <= 30; i++ {
		corpus.WriteString(`{"_id":"d` + string(rune('0'+i/10)) + string(rune('0'+i%10)) + `","text":"filler doc"}` + "\n")
	}
	if err := os.WriteFile(filepath.Join(dir, "corpus.jsonl"), []byte(corpus.String()), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"first query"}`+"\n"+
			`{"_id":"q2","text":"second query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	cfg := SampleCorpusNegativesConfig{DatasetDir: dir, Split: "train", PerQuery: 3, Seed: 7, Source: "fiqa:random"}
	rows, summary, err := SampleCorpusNegatives(cfg)
	if err != nil {
		t.Fatalf("sample: %v", err)
	}
	if summary.SampledQueries != 2 || summary.EmittedNegatives != 6 {
		t.Fatalf("summary = %+v", summary)
	}
	for _, row := range rows {
		if row.Positive != "T1\nrelevant doc" {
			t.Fatalf("positive = %q", row.Positive)
		}
		if len(row.Negatives) != 3 {
			t.Fatalf("negatives = %v", row.Negatives)
		}
		for _, negative := range row.Negatives {
			if negative == row.Positive {
				t.Fatal("sampled a qrel document as negative")
			}
		}
		if row.Source != "fiqa:random" {
			t.Fatalf("source = %q", row.Source)
		}
	}

	again, _, err := SampleCorpusNegatives(cfg)
	if err != nil {
		t.Fatalf("resample: %v", err)
	}
	if len(again) != len(rows) || again[0].Negatives[0] != rows[0].Negatives[0] {
		t.Fatal("sampling is not deterministic for a fixed seed")
	}
}
