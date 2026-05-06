package mantaruntime

import (
	"path/filepath"
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
	got[0].NegativeTokens[0][0] = 99
	if examples[0].NegativeTokens[0][0] == 99 {
		t.Fatal("round trip did not clone negative token slices")
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
		{Source: "nfcorpus:model", Query: "q", Positive: "p", Negatives: []string{"n1", "n2"}},
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
}

func TestTokenizeEmbeddingTextHardNegativeExamplesPreservesSource(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	got, err := TokenizeEmbeddingTextHardNegativeExamples([]EmbeddingTextHardNegativeExample{
		{Source: "fiqa:model", Query: "ab", Positive: "cd", Negatives: []string{"ab"}},
	}, tokenizer)
	if err != nil {
		t.Fatalf("tokenize hard-negative examples: %v", err)
	}
	if len(got) != 1 || got[0].Source != "fiqa:model" {
		t.Fatalf("tokenized hard negatives = %+v, want source fiqa:model", got)
	}
}
