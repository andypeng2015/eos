package eosruntime

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbeddingTextScoreSpectrumResearchOnlyRequiresExplicitAllow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"row_id":"r1","source":"msmarco","query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["positive","negative"],"selected_positive_index":0,"hard_negative_doc_ids":["n"],"combined_soft_targets":[2,1],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	if _, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path); err == nil {
		t.Fatal("read score-spectrum dataset succeeded without AllowResearchOnly")
	}
	got, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true})
	if err != nil {
		t.Fatalf("read score-spectrum dataset with AllowResearchOnly: %v", err)
	}
	if len(got) != 1 || got[0].RowID != "r1" || got[0].PositiveIndexes[0] != 0 {
		t.Fatalf("score-spectrum rows = %+v, want one canonical row", got)
	}
	if got[0].HardNegativeEligible[0] || !got[0].HardNegativeEligible[1] {
		t.Fatalf("hard eligibility = %+v, want positive false and negative true", got[0].HardNegativeEligible)
	}
	assertFloat32SumNearOne(t, got[0].TargetProbabilities)
	if got[0].TargetProbabilities[0] <= got[0].TargetProbabilities[1] {
		t.Fatalf("target probabilities = %+v, want normalized 2:1 order", got[0].TargetProbabilities)
	}
}

func TestEmbeddingTextScoreSpectrumRejectsBadResearchGateWhenAllowed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["positive","negative"],"positive_indexes":[0],"target_probabilities":[0.8,0.2],"release_train_allowed":true,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	if _, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}); err == nil {
		t.Fatal("read score-spectrum dataset succeeded with non-research-only gates under AllowResearchOnly")
	}
}

func TestEmbeddingTextScoreSpectrumRoundTripPreservesExtraFieldsAndGates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	examples := []EmbeddingTextScoreSpectrumExample{
		{
			RowID:                   "r1",
			Source:                  "msmarco",
			Query:                   "query",
			CandidateIDs:            []string{"p", "n"},
			Candidates:              []string{"positive", "negative"},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.75, 0.25},
			HardLossWeight:          0.4,
			SoftLossWeight:          0.6,
			TrainAllowedForResearch: true,
			SourceArtifactHash:      "sha256:abc",
			ExtraFields: map[string]json.RawMessage{
				"artifact_name": json.RawMessage(`"score-spectrum"`),
				"raw_scores":    json.RawMessage(`{"teacher":[1,0]}`),
			},
		},
	}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	if err := WriteEmbeddingTextScoreSpectrumExamplesFile(path, examples, opts); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	got, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, opts)
	if err != nil {
		t.Fatalf("read score-spectrum dataset: %v", err)
	}
	if len(got) != 1 || got[0].SourceArtifactHash != "sha256:abc" || got[0].HardLossWeight != 0.4 || got[0].SoftLossWeight != 0.6 {
		t.Fatalf("round trip = %+v, want weights and hash preserved", got)
	}
	if len(got[0].ExtraFields["artifact_name"]) == 0 || len(got[0].ExtraFields["raw_scores"]) == 0 {
		t.Fatalf("extra fields = %+v, want preserved provenance", got[0].ExtraFields)
	}
	out, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	var row map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(out))), &row); err != nil {
		t.Fatalf("decode output row: %v\n%s", err, out)
	}
	if row["artifact_name"] != "score-spectrum" || row["release_train_allowed"] != false || row["commercial_use_allowed"] != false || row["train_allowed_for_research"] != true {
		t.Fatalf("output row = %+v, want extra provenance and research gates", row)
	}
}

func TestEmbeddingTextScoreSpectrumMergesDuplicateCandidates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"query":"q","candidate_doc_ids":["p1","n1","n2"],"candidate_texts":["Positive text"," negative   text ","Negative Text"],"positive_indexes":[0],"hard_negative_eligible":[false,true,true],"target_probabilities":[0.5,0.2,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	got, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true})
	if err != nil {
		t.Fatalf("read score-spectrum dataset: %v", err)
	}
	row := got[0]
	if len(row.Candidates) != 2 {
		t.Fatalf("candidates = %+v, want duplicate merged", row.Candidates)
	}
	if row.CandidateIDs[1] != "n1" || !row.HardNegativeEligible[1] {
		t.Fatalf("merged candidate ids/eligibility = %+v %+v, want first id and hard eligible", row.CandidateIDs, row.HardNegativeEligible)
	}
	if math.Abs(float64(row.TargetProbabilities[1]-0.5)) > 1e-6 {
		t.Fatalf("target probabilities = %+v, want duplicate mass combined", row.TargetProbabilities)
	}
}

func TestEmbeddingTextScoreSpectrumMapsPositiveDocIDsBeforeDuplicateMerge(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"query":"q","candidate_doc_ids":["p1","p2","p3","n1"],"candidate_texts":["Positive text"," positive  text ","Other positive","negative"],"selected_positive_index":0,"positive_doc_ids":["p2","p3"],"hard_negative_doc_ids":["n1"],"combined_soft_targets":[0.2,0.3,0.4,0.1],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	got, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true})
	if err != nil {
		t.Fatalf("read score-spectrum dataset: %v", err)
	}
	row := got[0]
	if len(row.Candidates) != 3 {
		t.Fatalf("candidates = %+v, want duplicate positives merged", row.Candidates)
	}
	if len(row.PositiveIndexes) != 2 || row.PositiveIndexes[0] != 0 || row.PositiveIndexes[1] != 1 {
		t.Fatalf("positive indexes = %+v, want selected positive plus doc-id positives after merge", row.PositiveIndexes)
	}
	if row.HardNegativeEligible[0] || row.HardNegativeEligible[1] || !row.HardNegativeEligible[2] {
		t.Fatalf("hard eligibility = %+v, want positives excluded and n1 eligible", row.HardNegativeEligible)
	}
	if row.CandidateIDs[0] != "p1" || row.CandidateIDs[1] != "p3" || row.CandidateIDs[2] != "n1" {
		t.Fatalf("candidate ids = %+v, want first duplicate id retained and alignment preserved", row.CandidateIDs)
	}
	if math.Abs(float64(row.TargetProbabilities[0]-0.5)) > 1e-6 || math.Abs(float64(row.TargetProbabilities[1]-0.4)) > 1e-6 || math.Abs(float64(row.TargetProbabilities[2]-0.1)) > 1e-6 {
		t.Fatalf("target probabilities = %+v, want duplicate positive mass combined", row.TargetProbabilities)
	}
}

func TestEmbeddingTextScoreSpectrumRejectsDuplicatePositiveHardConflict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["same text"," Same  Text "],"positive_indexes":[0],"hard_negative_eligible":[false,true],"target_probabilities":[0.6,0.4],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	if _, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}); err == nil {
		t.Fatal("read score-spectrum dataset succeeded with duplicate positive/hard conflict")
	}
}

func TestEmbeddingTextScoreSpectrumValidationRejectsAlignmentAndProbabilityErrors(t *testing.T) {
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	cases := []struct {
		name string
		row  string
	}{
		{
			name: "candidate id mismatch",
			row:  `{"query":"q","candidate_doc_ids":["p"],"candidate_texts":["p","n"],"positive_indexes":[0],"target_probabilities":[0.7,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
		{
			name: "negative probability",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_indexes":[0],"target_probabilities":[1,-1],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
		{
			name: "positive out of range",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_indexes":[2],"target_probabilities":[0.7,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
		{
			name: "positive hard eligible",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_indexes":[0],"hard_negative_eligible":[true,true],"target_probabilities":[0.7,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
		{
			name: "unknown positive doc id",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_doc_ids":["missing"],"target_probabilities":[0.7,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
			if err := os.WriteFile(path, []byte(tc.row+"\n"), 0o644); err != nil {
				t.Fatalf("write score-spectrum dataset: %v", err)
			}
			if _, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, opts); err == nil {
				t.Fatal("read score-spectrum dataset succeeded, want validation error")
			}
		})
	}
}

func TestTokenizeEmbeddingTextScoreSpectrumExamplesPreservesAlignment(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	got, err := TokenizeEmbeddingTextScoreSpectrumExamples([]EmbeddingTextScoreSpectrumExample{
		{
			RowID:                   "r1",
			Source:                  "msmarco",
			Query:                   "ab",
			CandidateIDs:            []string{"p", "n"},
			Candidates:              []string{"cd", "ab"},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.8, 0.2},
			TrainAllowedForResearch: true,
		},
	}, tokenizer, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true})
	if err != nil {
		t.Fatalf("tokenize score-spectrum examples: %v", err)
	}
	if len(got) != 1 || got[0].RowID != "r1" || got[0].Source != "msmarco" {
		t.Fatalf("tokenized rows = %+v, want row metadata", got)
	}
	if len(got[0].CandidateTokens) != 2 || len(got[0].CandidateMasks) != 2 || got[0].CandidateIDs[1] != "n" {
		t.Fatalf("candidate alignment = %+v ids=%+v masks=%+v", got[0].CandidateTokens, got[0].CandidateIDs, got[0].CandidateMasks)
	}
	if got[0].PositiveIndexes[0] != 0 || got[0].HardNegativeEligible[0] || !got[0].HardNegativeEligible[1] {
		t.Fatalf("labels = positives %+v hard %+v, want aligned labels", got[0].PositiveIndexes, got[0].HardNegativeEligible)
	}
	if math.Abs(float64(got[0].TargetProbabilities[0]-0.8)) > 1e-6 || math.Abs(float64(got[0].TargetProbabilities[1]-0.2)) > 1e-6 {
		t.Fatalf("target probabilities = %+v, want preserved", got[0].TargetProbabilities)
	}
}

func TestEmbeddingScoreSpectrumTokenizedRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-tokenized.jsonl")
	examples := []EmbeddingScoreSpectrumExample{
		{
			RowID:                   "r1",
			Source:                  "tokenized",
			QueryTokens:             []int32{1, 2},
			QueryMask:               []int32{1, 1},
			CandidateIDs:            []string{"p", "n"},
			CandidateTokens:         [][]int32{{3}, {4}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.9, 0.1},
			TrainAllowedForResearch: true,
			SourceArtifactHash:      "sha256:def",
		},
	}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	if err := WriteEmbeddingScoreSpectrumExamplesFile(path, examples, opts); err != nil {
		t.Fatalf("write tokenized score-spectrum dataset: %v", err)
	}
	got, err := ReadEmbeddingScoreSpectrumExamplesFile(path, opts)
	if err != nil {
		t.Fatalf("read tokenized score-spectrum dataset: %v", err)
	}
	if len(got) != 1 || got[0].CandidateIDs[0] != "p" || got[0].SourceArtifactHash != "sha256:def" {
		t.Fatalf("tokenized round trip = %+v, want metadata preserved", got)
	}
	got[0].CandidateTokens[0][0] = 99
	if examples[0].CandidateTokens[0][0] == 99 {
		t.Fatal("round trip did not clone candidate token slices")
	}
}

func assertFloat32SumNearOne(t *testing.T, values []float32) {
	t.Helper()
	sum := float32(0)
	for _, value := range values {
		sum += value
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum(%+v) = %v, want 1", values, sum)
	}
}
