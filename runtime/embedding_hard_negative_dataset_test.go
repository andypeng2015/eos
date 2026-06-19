package eosruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbeddingHardNegativeExamplesFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hard-negatives.jsonl")
	examples := []EmbeddingHardNegativeExample{
		{
			Source:         "scifact",
			QueryTokens:    []int32{1, 2},
			PositiveTokens: []int32{3, 4},
			NegativeTokens: [][]int32{{5, 6}, {7, 8}},
			QueryMask:      []int32{1, 1},
			PositiveMask:   []int32{1, 1},
			NegativeMasks:  [][]int32{{1, 1}, {1, 1}},
			TeacherScores:  []float32{1.5, 0.25, 0.1},
		},
	}
	if err := WriteEmbeddingHardNegativeExamplesFile(path, examples); err != nil {
		t.Fatalf("write hard-negative dataset: %v", err)
	}
	got, err := ReadEmbeddingHardNegativeExamplesFile(path)
	if err != nil {
		t.Fatalf("read hard-negative dataset: %v", err)
	}
	if len(got) != 1 || len(got[0].NegativeTokens) != 2 {
		t.Fatalf("round trip = %+v, want one example with two negatives", got)
	}
	if got[0].Source != "scifact" {
		t.Fatalf("source = %q, want scifact", got[0].Source)
	}
	if len(got[0].TeacherScores) != 3 || got[0].TeacherScores[0] != 1.5 || got[0].TeacherScores[2] != 0.1 {
		t.Fatalf("teacher scores = %+v, want preserved", got[0].TeacherScores)
	}
	got[0].NegativeTokens[0][0] = 99
	if examples[0].NegativeTokens[0][0] == 99 {
		t.Fatal("round trip did not clone negative token slices")
	}
	got[0].TeacherScores[0] = 99
	if examples[0].TeacherScores[0] == 99 {
		t.Fatal("round trip did not clone teacher scores")
	}
}

func TestEmbeddingHardNegativeExamplesRejectTeacherScoreCountMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hard-negatives.jsonl")
	examples := []EmbeddingHardNegativeExample{
		{
			QueryTokens:    []int32{1},
			PositiveTokens: []int32{2},
			NegativeTokens: [][]int32{{3}},
			TeacherScores:  []float32{1},
		},
	}
	if err := WriteEmbeddingHardNegativeExamplesFile(path, examples); err == nil {
		t.Fatal("write hard-negative dataset succeeded with mismatched teacher scores")
	}
}

func TestBuildEmbeddingHardNegativeExamplesFromPairsGroupsByQuery(t *testing.T) {
	pairs := []EmbeddingPairExample{
		{LeftTokens: []int32{1}, RightTokens: []int32{10}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{11}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{2}, RightTokens: []int32{20}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{2}, RightTokens: []int32{21}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 0},
		{LeftTokens: []int32{2}, RightTokens: []int32{22}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}
	got, err := BuildEmbeddingHardNegativeExamplesFromPairs(pairs, 1)
	if err != nil {
		t.Fatalf("build hard negatives: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("examples = %d, want 2", len(got))
	}
	for i, example := range got {
		if len(example.NegativeTokens) != 1 {
			t.Fatalf("example %d negatives = %d, want 1", i, len(example.NegativeTokens))
		}
	}
	if got[0].QueryTokens[0] != 1 || got[1].QueryTokens[0] != 2 {
		t.Fatalf("query order = %+v", got)
	}
}

func TestBuildEmbeddingTextHardNegativeExamplesFromPairs(t *testing.T) {
	pairs := []EmbeddingTextPairExample{
		{Source: "fiqa", Query: "q1", Right: "p1", Target: 1},
		{Source: "fiqa", Query: "q1", Right: "n1", Target: -1},
		{Source: "fiqa", Query: "q1", Right: "n2", Target: 0},
	}
	got, err := BuildEmbeddingTextHardNegativeExamplesFromPairs(pairs, 2)
	if err != nil {
		t.Fatalf("build text hard negatives: %v", err)
	}
	if len(got) != 1 || got[0].Query != "q1" || got[0].Positive != "p1" || len(got[0].Negatives) != 2 {
		t.Fatalf("text hard negatives = %+v", got)
	}
	if got[0].Source != "fiqa" {
		t.Fatalf("source = %q, want fiqa", got[0].Source)
	}
}

func TestEmbeddingTextHardNegativeExamplesFileRoundTripPreservesSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "text-hard-negatives.jsonl")
	examples := []EmbeddingTextHardNegativeExample{
		{Source: "nfcorpus:model", Query: "q", Positive: "p", Negatives: []string{"n1", "n2"}, TeacherScores: []float32{0.9, 0.8, 0.7}},
	}
	if err := WriteEmbeddingTextHardNegativeExamplesFile(path, examples); err != nil {
		t.Fatalf("write text hard-negative dataset: %v", err)
	}
	got, err := ReadEmbeddingTextHardNegativeExamplesFile(path)
	if err != nil {
		t.Fatalf("read text hard-negative dataset: %v", err)
	}
	if len(got) != 1 || got[0].Source != "nfcorpus:model" {
		t.Fatalf("round trip = %+v, want source nfcorpus:model", got)
	}
	if len(got[0].TeacherScores) != 3 || got[0].TeacherScores[1] != 0.8 {
		t.Fatalf("teacher scores = %+v, want preserved", got[0].TeacherScores)
	}
}

func TestReadEmbeddingTextHardNegativeExamplesFileAcceptsLongJSONLRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "text-hard-negatives.jsonl")
	longPositive := strings.Repeat("p", embeddingJSONLScannerInitialBuffer+4096)
	data := `{"source":"longembed","query":"q","positive":"` + longPositive + `","negatives":["n"]}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write long text hard-negative dataset: %v", err)
	}
	got, err := ReadEmbeddingTextHardNegativeExamplesFile(path)
	if err != nil {
		t.Fatalf("read long text hard-negative dataset: %v", err)
	}
	if len(got) != 1 || len(got[0].Positive) != len(longPositive) {
		t.Fatalf("long record = %+v, want one preserved positive of length %d", got, len(longPositive))
	}
}

func TestBuildEmbeddingTextHardNegativeEvalPairsExpandsPositiveAndNegatives(t *testing.T) {
	got, err := BuildEmbeddingTextHardNegativeEvalPairs([]EmbeddingTextHardNegativeExample{
		{Source: "longembed", Query: "q", Positive: "p", Negatives: []string{"n1", "n2"}},
	}, 1)
	if err != nil {
		t.Fatalf("build text hard-negative eval pairs: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("eval pairs = %d, want 2", len(got))
	}
	if got[0].Query != "q" || got[0].Right != "p" || got[0].Target != 1 {
		t.Fatalf("positive pair = %+v", got[0])
	}
	if got[1].Query != "q" || got[1].Right != "n1" || got[1].Target != 0 {
		t.Fatalf("negative pair = %+v", got[1])
	}
}

func TestReadEmbeddingTextHardNegativeEvalPairsFileKeepsOrdinaryPairEval(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eval-pairs.jsonl")
	data := "" +
		"{\"query\":\"q\",\"document\":\"p\",\"label\":1}\n" +
		"{\"left\":\"q\",\"right\":\"n\",\"label\":0}\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write eval pairs: %v", err)
	}
	got, err := ReadEmbeddingTextHardNegativeEvalPairsFile(path, 1)
	if err != nil {
		t.Fatalf("read hard-negative eval pairs: %v", err)
	}
	if len(got) != 2 || got[0].Target != 1 || got[1].Target != 0 {
		t.Fatalf("eval pairs = %+v, want ordinary pair eval preserved", got)
	}
}

func TestBuildEmbeddingHardNegativeEvalPairsExpandsTokenizedGroupedRows(t *testing.T) {
	got, err := BuildEmbeddingHardNegativeEvalPairs([]EmbeddingHardNegativeExample{
		{
			Source:         "tokenized",
			QueryTokens:    []int32{1},
			PositiveTokens: []int32{2},
			NegativeTokens: [][]int32{{3}, {4}},
			QueryMask:      []int32{1},
			PositiveMask:   []int32{1},
			NegativeMasks:  [][]int32{{1}, {1}},
		},
	}, 2)
	if err != nil {
		t.Fatalf("build hard-negative eval pairs: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("eval pairs = %d, want 4", len(got))
	}
	if got[0].Target != 1 || got[1].Target != 0 || got[2].Target != 1 || got[3].Target != 0 {
		t.Fatalf("targets = %v %v %v %v, want positive/negative pairs", got[0].Target, got[1].Target, got[2].Target, got[3].Target)
	}
	assertInt32SliceEqual(t, got[1].RightTokens, []int32{3})
	assertInt32SliceEqual(t, got[3].RightTokens, []int32{4})
}

func TestTokenizeEmbeddingTextHardNegativeExamplesPreservesSource(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	got, err := TokenizeEmbeddingTextHardNegativeExamples([]EmbeddingTextHardNegativeExample{
		{Source: "fiqa:model", Query: "ab", Positive: "cd", Negatives: []string{"ab"}, TeacherScores: []float32{0.7, 0.2}},
	}, tokenizer)
	if err != nil {
		t.Fatalf("tokenize hard-negative examples: %v", err)
	}
	if len(got) != 1 || got[0].Source != "fiqa:model" {
		t.Fatalf("tokenized hard negatives = %+v, want source fiqa:model", got)
	}
	if len(got[0].TeacherScores) != 2 || got[0].TeacherScores[0] != 0.7 {
		t.Fatalf("tokenized teacher scores = %+v, want preserved", got[0].TeacherScores)
	}
}
