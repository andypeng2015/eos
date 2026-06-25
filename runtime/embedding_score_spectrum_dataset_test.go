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
	if got[0].SelectedPositiveIndex == nil || *got[0].SelectedPositiveIndex != 0 {
		t.Fatalf("selected_positive_index = %v, want 0", got[0].SelectedPositiveIndex)
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
	selected := 0
	examples := []EmbeddingTextScoreSpectrumExample{
		{
			RowID:                   "r1",
			Source:                  "msmarco",
			Query:                   "query",
			CandidateIDs:            []string{"p", "n"},
			Candidates:              []string{"positive", "negative"},
			PositiveIndexes:         []int{0},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.75, 0.25},
			HardLossWeight:          0.4,
			SoftLossWeight:          0.6,
			RecoveryLossWeight:      0.7,
			TrainPolicy:             "score-spectrum-native-eval-test",
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
	if len(got) != 1 || got[0].SourceArtifactHash != "sha256:abc" || got[0].HardLossWeight != 0.4 || got[0].SoftLossWeight != 0.6 || got[0].RecoveryLossWeight != 0.7 || got[0].TrainPolicy != "score-spectrum-native-eval-test" {
		t.Fatalf("round trip = %+v, want weights and hash preserved", got)
	}
	if got[0].SelectedPositiveIndex == nil || *got[0].SelectedPositiveIndex != 0 {
		t.Fatalf("selected_positive_index = %v, want 0", got[0].SelectedPositiveIndex)
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
	if row["selected_positive_index"] != float64(0) || row["recovery_loss_weight"] != 0.7 || row["train_policy"] != "score-spectrum-native-eval-test" {
		t.Fatalf("output row = %+v, want selected positive and recovery provenance", row)
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
	if row.SelectedPositiveIndex == nil || *row.SelectedPositiveIndex != 0 {
		t.Fatalf("selected positive index = %v, want remapped to merged candidate 0", row.SelectedPositiveIndex)
	}
}

func TestEmbeddingTextScoreSpectrumFoldsSelectedPositiveIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"query":"q","candidate_doc_ids":["original","alt","n"],"candidate_texts":["original positive","alternate positive","negative"],"positive_indexes":[1],"selected_positive_index":0,"hard_negative_eligible":[false,false,true],"target_probabilities":[0.4,0.5,0.1],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	got, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true})
	if err != nil {
		t.Fatalf("read score-spectrum dataset: %v", err)
	}
	row := got[0]
	if row.SelectedPositiveIndex == nil || *row.SelectedPositiveIndex != 0 {
		t.Fatalf("selected positive index = %v, want 0", row.SelectedPositiveIndex)
	}
	if len(row.PositiveIndexes) != 2 || row.PositiveIndexes[0] != 0 || row.PositiveIndexes[1] != 1 {
		t.Fatalf("positive indexes = %+v, want selected folded into canonical positives [0 1]", row.PositiveIndexes)
	}
	if row.HardNegativeEligible[0] || row.HardNegativeEligible[1] || !row.HardNegativeEligible[2] {
		t.Fatalf("hard eligibility = %+v, want folded positives excluded", row.HardNegativeEligible)
	}
}

func TestEmbeddingTextScoreSpectrumDuplicateMergeRemapsSelectedPositiveIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	data := `{"query":"q","candidate_doc_ids":["p1","p2","n1"],"candidate_texts":["same positive"," Same  Positive ","negative"],"positive_indexes":[1],"selected_positive_index":1,"hard_negative_eligible":[false,false,true],"target_probabilities":[0.2,0.6,0.2],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	got, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true})
	if err != nil {
		t.Fatalf("read score-spectrum dataset: %v", err)
	}
	row := got[0]
	if len(row.Candidates) != 2 {
		t.Fatalf("candidates = %+v, want duplicate selected positive merged", row.Candidates)
	}
	if row.SelectedPositiveIndex == nil || *row.SelectedPositiveIndex != 0 {
		t.Fatalf("selected positive index = %v, want remapped to surviving candidate 0", row.SelectedPositiveIndex)
	}
	if len(row.PositiveIndexes) != 1 || row.PositiveIndexes[0] != 0 {
		t.Fatalf("positive indexes = %+v, want remapped positive 0", row.PositiveIndexes)
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
		{
			name: "selected positive out of range",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_indexes":[0],"selected_positive_index":2,"target_probabilities":[0.7,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
		{
			name: "selected positive hard eligible",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_indexes":[0],"selected_positive_index":1,"hard_negative_eligible":[false,true],"target_probabilities":[0.7,0.3],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		},
		{
			name: "negative recovery weight",
			row:  `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["p","n"],"positive_indexes":[0],"target_probabilities":[0.7,0.3],"recovery_loss_weight":-0.1,"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
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
	selected := 0
	got, err := TokenizeEmbeddingTextScoreSpectrumExamples([]EmbeddingTextScoreSpectrumExample{
		{
			RowID:                   "r1",
			Source:                  "msmarco",
			Query:                   "ab",
			CandidateIDs:            []string{"p", "n"},
			Candidates:              []string{"cd", "ab"},
			PositiveIndexes:         []int{0},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.8, 0.2},
			RecoveryLossWeight:      0.25,
			TrainPolicy:             "policy-a",
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
	if got[0].SelectedPositiveIndex == nil || *got[0].SelectedPositiveIndex != 0 || got[0].RecoveryLossWeight != 0.25 || got[0].TrainPolicy != "policy-a" {
		t.Fatalf("selected/recovery/policy = %v/%v/%q, want preserved", got[0].SelectedPositiveIndex, got[0].RecoveryLossWeight, got[0].TrainPolicy)
	}
	if math.Abs(float64(got[0].TargetProbabilities[0]-0.8)) > 1e-6 || math.Abs(float64(got[0].TargetProbabilities[1]-0.2)) > 1e-6 {
		t.Fatalf("target probabilities = %+v, want preserved", got[0].TargetProbabilities)
	}
}

func TestEmbeddingScoreSpectrumTokenizedRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-tokenized.jsonl")
	selected := 0
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
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.9, 0.1},
			RecoveryLossWeight:      0.35,
			TrainPolicy:             "tokenized-policy",
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
	if len(got) != 1 || got[0].CandidateIDs[0] != "p" || got[0].SourceArtifactHash != "sha256:def" || got[0].RecoveryLossWeight != 0.35 || got[0].TrainPolicy != "tokenized-policy" {
		t.Fatalf("tokenized round trip = %+v, want metadata preserved", got)
	}
	if got[0].SelectedPositiveIndex == nil || *got[0].SelectedPositiveIndex != 0 {
		t.Fatalf("selected_positive_index = %v, want 0", got[0].SelectedPositiveIndex)
	}
	got[0].CandidateTokens[0][0] = 99
	if examples[0].CandidateTokens[0][0] == 99 {
		t.Fatal("round trip did not clone candidate token slices")
	}
}

func TestEmbeddingScoreSpectrumTokenizedFoldsSelectedOnlyPositive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-tokenized.jsonl")
	selected := 0
	examples := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{1},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{2}, {3}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.8, 0.2},
			TrainAllowedForResearch: true,
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
	if len(got) != 1 || got[0].SelectedPositiveIndex == nil || *got[0].SelectedPositiveIndex != 0 {
		t.Fatalf("tokenized rows = %+v, want selected positive 0 preserved", got)
	}
	if len(got[0].PositiveIndexes) != 1 || got[0].PositiveIndexes[0] != 0 {
		t.Fatalf("positive indexes = %+v, want selected-only row canonicalized to [0]", got[0].PositiveIndexes)
	}
}

func TestEmbeddingScoreSpectrumTokenizedFoldsSelectedPositiveIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-tokenized.jsonl")
	selected := 0
	examples := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{1},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{2}, {3}, {4}},
			CandidateMasks:          [][]int32{{1}, {1}, {1}},
			PositiveIndexes:         []int{1},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, false, true},
			TargetProbabilities:     []float32{0.4, 0.5, 0.1},
			TrainAllowedForResearch: true,
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
	if len(got) != 1 || got[0].SelectedPositiveIndex == nil || *got[0].SelectedPositiveIndex != 0 {
		t.Fatalf("tokenized rows = %+v, want selected positive 0 preserved", got)
	}
	if len(got[0].PositiveIndexes) != 2 || got[0].PositiveIndexes[0] != 0 || got[0].PositiveIndexes[1] != 1 {
		t.Fatalf("positive indexes = %+v, want selected folded into canonical positives [0 1]", got[0].PositiveIndexes)
	}
}

func TestEmbeddingScoreSpectrumTokenizedValidationRejectsSelectedPositiveAndRecoveryWeight(t *testing.T) {
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	cases := []struct {
		name   string
		mutate func(*EmbeddingScoreSpectrumExample)
	}{
		{
			name: "selected positive out of range",
			mutate: func(example *EmbeddingScoreSpectrumExample) {
				selected := 2
				example.SelectedPositiveIndex = &selected
			},
		},
		{
			name: "selected positive hard eligible",
			mutate: func(example *EmbeddingScoreSpectrumExample) {
				selected := 1
				example.SelectedPositiveIndex = &selected
			},
		},
		{
			name: "negative recovery weight",
			mutate: func(example *EmbeddingScoreSpectrumExample) {
				example.RecoveryLossWeight = -0.1
			},
		},
		{
			name: "nonfinite recovery weight",
			mutate: func(example *EmbeddingScoreSpectrumExample) {
				example.RecoveryLossWeight = float32(math.Inf(1))
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			example := EmbeddingScoreSpectrumExample{
				QueryTokens:             []int32{1},
				QueryMask:               []int32{1},
				CandidateTokens:         [][]int32{{1}, {2}},
				CandidateMasks:          [][]int32{{1}, {1}},
				PositiveIndexes:         []int{0},
				HardNegativeEligible:    []bool{false, true},
				TargetProbabilities:     []float32{0.8, 0.2},
				TrainAllowedForResearch: true,
			}
			tc.mutate(&example)
			path := filepath.Join(t.TempDir(), "score-spectrum-tokenized.jsonl")
			if err := WriteEmbeddingScoreSpectrumExamplesFile(path, []EmbeddingScoreSpectrumExample{example}, opts); err == nil {
				t.Fatal("write tokenized score-spectrum dataset succeeded, want validation error")
			}
		})
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
