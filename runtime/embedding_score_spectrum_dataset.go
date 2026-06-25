package eosruntime

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// EmbeddingScoreSpectrumExample is one tokenized query with ranked candidates and soft targets.
type EmbeddingScoreSpectrumExample struct {
	RowID                   string
	Source                  string
	QueryTokens             []int32
	QueryMask               []int32
	CandidateIDs            []string
	CandidateTokens         [][]int32
	CandidateMasks          [][]int32
	PositiveIndexes         []int
	HardNegativeEligible    []bool
	TargetProbabilities     []float32
	HardLossWeight          float32
	SoftLossWeight          float32
	ReleaseTrainAllowed     bool
	CommercialUseAllowed    bool
	TrainAllowedForResearch bool
	SourceArtifactHash      string
	ExtraFields             map[string]json.RawMessage
}

// EmbeddingTextScoreSpectrumExample is the text JSONL form for ranked-candidate score-spectrum data.
type EmbeddingTextScoreSpectrumExample struct {
	RowID                   string
	Source                  string
	Query                   string
	CandidateIDs            []string
	Candidates              []string
	PositiveIndexes         []int
	HardNegativeEligible    []bool
	TargetProbabilities     []float32
	HardLossWeight          float32
	SoftLossWeight          float32
	ReleaseTrainAllowed     bool
	CommercialUseAllowed    bool
	TrainAllowedForResearch bool
	SourceArtifactHash      string
	ExtraFields             map[string]json.RawMessage
}

type EmbeddingScoreSpectrumReadOptions struct {
	AllowResearchOnly bool
}

type embeddingScoreSpectrumRecord struct {
	RowID                   string                      `json:"row_id,omitempty"`
	Source                  string                      `json:"source,omitempty"`
	QueryTokens             []int32                     `json:"query_tokens"`
	QueryMask               []int32                     `json:"query_mask,omitempty"`
	CandidateIDs            []string                    `json:"candidate_ids,omitempty"`
	CandidateTokens         [][]int32                   `json:"candidate_tokens"`
	CandidateMasks          [][]int32                   `json:"candidate_masks,omitempty"`
	PositiveIndexes         []int                       `json:"positive_indexes"`
	HardNegativeEligible    []bool                      `json:"hard_negative_eligible"`
	TargetProbabilities     []float32                   `json:"target_probabilities"`
	HardLossWeight          float32                     `json:"hard_loss_weight,omitempty"`
	SoftLossWeight          float32                     `json:"soft_loss_weight,omitempty"`
	LegalGates              embeddingScoreSpectrumGates `json:"legal_gates,omitempty"`
	ReleaseTrainAllowed     bool                        `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed    bool                        `json:"commercial_use_allowed,omitempty"`
	TrainAllowedForResearch bool                        `json:"train_allowed_for_research,omitempty"`
	SourceArtifactHash      string                      `json:"source_artifact_hash,omitempty"`
	ExtraFields             map[string]json.RawMessage
}

type embeddingTextScoreSpectrumRecord struct {
	RowID                   string                          `json:"row_id,omitempty"`
	Source                  string                          `json:"source,omitempty"`
	Query                   string                          `json:"query"`
	CandidateIDs            []string                        `json:"candidate_doc_ids,omitempty"`
	CandidateTexts          []string                        `json:"candidate_texts,omitempty"`
	Candidates              []embeddingScoreCandidateRecord `json:"candidates,omitempty"`
	PositiveIndexes         []int                           `json:"positive_indexes,omitempty"`
	PositiveDocIDs          []string                        `json:"positive_doc_ids,omitempty"`
	SelectedPositiveIndex   *int                            `json:"selected_positive_index,omitempty"`
	HardNegativeEligible    []bool                          `json:"hard_negative_eligible,omitempty"`
	HardNegativeDocIDs      []string                        `json:"hard_negative_doc_ids,omitempty"`
	TargetProbabilities     []float32                       `json:"target_probabilities,omitempty"`
	CombinedSoftTargets     []float32                       `json:"combined_soft_targets,omitempty"`
	HardLossWeight          float32                         `json:"hard_loss_weight,omitempty"`
	SoftLossWeight          float32                         `json:"soft_loss_weight,omitempty"`
	LegalGates              embeddingScoreSpectrumGates     `json:"legal_gates,omitempty"`
	ReleaseTrainAllowed     bool                            `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed    bool                            `json:"commercial_use_allowed,omitempty"`
	TrainAllowedForResearch bool                            `json:"train_allowed_for_research,omitempty"`
	SourceArtifactHash      string                          `json:"source_artifact_hash,omitempty"`
	ExtraFields             map[string]json.RawMessage
}

type embeddingScoreCandidateRecord struct {
	CandidateIndex int    `json:"candidate_index,omitempty"`
	DocID          string `json:"doc_id,omitempty"`
	Text           string `json:"text"`
}

type embeddingScoreSpectrumGates struct {
	ReleaseTrainAllowed     bool `json:"release_train_allowed"`
	CommercialUseAllowed    bool `json:"commercial_use_allowed"`
	TrainAllowedForResearch bool `json:"train_allowed_for_research"`
}

type embeddingTextScoreSpectrumKnownRecord embeddingTextScoreSpectrumRecord

var embeddingTextScoreSpectrumKnownFields = map[string]struct{}{
	"row_id":                     {},
	"source":                     {},
	"query":                      {},
	"candidate_doc_ids":          {},
	"candidate_texts":            {},
	"candidates":                 {},
	"positive_indexes":           {},
	"positive_doc_ids":           {},
	"selected_positive_index":    {},
	"hard_negative_eligible":     {},
	"hard_negative_doc_ids":      {},
	"target_probabilities":       {},
	"combined_soft_targets":      {},
	"hard_loss_weight":           {},
	"soft_loss_weight":           {},
	"legal_gates":                {},
	"release_train_allowed":      {},
	"commercial_use_allowed":     {},
	"train_allowed_for_research": {},
	"source_artifact_hash":       {},
}

func (r *embeddingTextScoreSpectrumRecord) UnmarshalJSON(data []byte) error {
	var known embeddingTextScoreSpectrumKnownRecord
	if err := json.Unmarshal(data, &known); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for key := range embeddingTextScoreSpectrumKnownFields {
		delete(fields, key)
	}
	*r = embeddingTextScoreSpectrumRecord(known)
	r.ExtraFields = cloneRawMessageMap(fields)
	return nil
}

func (r embeddingTextScoreSpectrumRecord) MarshalJSON() ([]byte, error) {
	fields := cloneRawMessageMap(r.ExtraFields)
	put := func(key string, value any, omit bool) error {
		if omit {
			delete(fields, key)
			return nil
		}
		if fields == nil {
			fields = map[string]json.RawMessage{}
		}
		data, err := json.Marshal(value)
		if err != nil {
			return err
		}
		fields[key] = data
		return nil
	}
	if err := put("row_id", r.RowID, r.RowID == ""); err != nil {
		return nil, err
	}
	if err := put("source", r.Source, r.Source == ""); err != nil {
		return nil, err
	}
	if err := put("query", r.Query, false); err != nil {
		return nil, err
	}
	if err := put("candidate_doc_ids", r.CandidateIDs, len(r.CandidateIDs) == 0); err != nil {
		return nil, err
	}
	if err := put("candidate_texts", r.CandidateTexts, len(r.CandidateTexts) == 0); err != nil {
		return nil, err
	}
	if err := put("positive_indexes", r.PositiveIndexes, len(r.PositiveIndexes) == 0); err != nil {
		return nil, err
	}
	if err := put("hard_negative_eligible", r.HardNegativeEligible, len(r.HardNegativeEligible) == 0); err != nil {
		return nil, err
	}
	if err := put("target_probabilities", r.TargetProbabilities, false); err != nil {
		return nil, err
	}
	if err := put("hard_loss_weight", r.HardLossWeight, r.HardLossWeight == 0); err != nil {
		return nil, err
	}
	if err := put("soft_loss_weight", r.SoftLossWeight, r.SoftLossWeight == 0); err != nil {
		return nil, err
	}
	if err := put("legal_gates", r.LegalGates, false); err != nil {
		return nil, err
	}
	if err := put("release_train_allowed", r.ReleaseTrainAllowed, false); err != nil {
		return nil, err
	}
	if err := put("commercial_use_allowed", r.CommercialUseAllowed, false); err != nil {
		return nil, err
	}
	if err := put("train_allowed_for_research", r.TrainAllowedForResearch, false); err != nil {
		return nil, err
	}
	if err := put("source_artifact_hash", r.SourceArtifactHash, r.SourceArtifactHash == ""); err != nil {
		return nil, err
	}
	return json.Marshal(fields)
}

func ReadEmbeddingTextScoreSpectrumExamplesFile(path string, opts ...EmbeddingScoreSpectrumReadOptions) ([]EmbeddingTextScoreSpectrumExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	allowResearch := scoreSpectrumAllowResearch(opts)
	var out []EmbeddingTextScoreSpectrumExample
	if err := scanEmbeddingJSONLLines(f, func(lineNo int, line string) error {
		var record embeddingTextScoreSpectrumRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		example, err := record.example(allowResearch)
		if err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, example)
		return nil
	}); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text score-spectrum dataset is empty")
	}
	return out, nil
}

func WriteEmbeddingTextScoreSpectrumExamplesFile(path string, examples []EmbeddingTextScoreSpectrumExample, opts ...EmbeddingScoreSpectrumReadOptions) error {
	if len(examples) == 0 {
		return fmt.Errorf("text score-spectrum dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingTextScoreSpectrumRecord(example, scoreSpectrumAllowResearch(opts))
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func ReadEmbeddingScoreSpectrumExamplesFile(path string, opts ...EmbeddingScoreSpectrumReadOptions) ([]EmbeddingScoreSpectrumExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	allowResearch := scoreSpectrumAllowResearch(opts)
	var out []EmbeddingScoreSpectrumExample
	if err := scanEmbeddingJSONLLines(f, func(lineNo int, line string) error {
		var record embeddingScoreSpectrumRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		example, err := record.example(allowResearch)
		if err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, example)
		return nil
	}); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("score-spectrum dataset is empty")
	}
	return out, nil
}

func WriteEmbeddingScoreSpectrumExamplesFile(path string, examples []EmbeddingScoreSpectrumExample, opts ...EmbeddingScoreSpectrumReadOptions) error {
	if len(examples) == 0 {
		return fmt.Errorf("score-spectrum dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingScoreSpectrumRecord(example, scoreSpectrumAllowResearch(opts))
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func TokenizeEmbeddingTextScoreSpectrumExamples(examples []EmbeddingTextScoreSpectrumExample, tokenizer *BPETokenizer, opts ...EmbeddingScoreSpectrumReadOptions) ([]EmbeddingScoreSpectrumExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("text score-spectrum dataset is empty")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	cache := embeddingTextTokenCache{}
	allowResearch := scoreSpectrumAllowResearch(opts)
	out := make([]EmbeddingScoreSpectrumExample, 0, len(examples))
	for i, example := range examples {
		record, err := newEmbeddingTextScoreSpectrumRecord(example, allowResearch)
		if err != nil {
			return nil, fmt.Errorf("example %d: %w", i, err)
		}
		clean, err := record.example(allowResearch)
		if err != nil {
			return nil, fmt.Errorf("example %d: %w", i, err)
		}
		query, err := cache.encode(clean.Query, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d query: %w", i, err)
		}
		candidateTokens := make([][]int32, 0, len(clean.Candidates))
		candidateMasks := make([][]int32, 0, len(clean.Candidates))
		for j, text := range clean.Candidates {
			candidate, err := cache.encode(text, tokenizer)
			if err != nil {
				return nil, fmt.Errorf("example %d candidate %d: %w", i, j, err)
			}
			candidate = cloneTokenizedText(candidate)
			candidateTokens = append(candidateTokens, candidate.tokens)
			candidateMasks = append(candidateMasks, candidate.mask)
		}
		query = cloneTokenizedText(query)
		out = append(out, EmbeddingScoreSpectrumExample{
			RowID:                   clean.RowID,
			Source:                  clean.Source,
			QueryTokens:             query.tokens,
			QueryMask:               query.mask,
			CandidateIDs:            append([]string(nil), clean.CandidateIDs...),
			CandidateTokens:         candidateTokens,
			CandidateMasks:          candidateMasks,
			PositiveIndexes:         append([]int(nil), clean.PositiveIndexes...),
			HardNegativeEligible:    append([]bool(nil), clean.HardNegativeEligible...),
			TargetProbabilities:     append([]float32(nil), clean.TargetProbabilities...),
			HardLossWeight:          clean.HardLossWeight,
			SoftLossWeight:          clean.SoftLossWeight,
			ReleaseTrainAllowed:     clean.ReleaseTrainAllowed,
			CommercialUseAllowed:    clean.CommercialUseAllowed,
			TrainAllowedForResearch: clean.TrainAllowedForResearch,
			SourceArtifactHash:      clean.SourceArtifactHash,
			ExtraFields:             cloneRawMessageMap(clean.ExtraFields),
		})
	}
	return out, nil
}

func newEmbeddingTextScoreSpectrumRecord(example EmbeddingTextScoreSpectrumExample, allowResearch bool) (embeddingTextScoreSpectrumRecord, error) {
	clean, err := validateAndCanonicalizeTextScoreSpectrum(example, allowResearch)
	if err != nil {
		return embeddingTextScoreSpectrumRecord{}, err
	}
	gates := embeddingScoreSpectrumGates{clean.ReleaseTrainAllowed, clean.CommercialUseAllowed, clean.TrainAllowedForResearch}
	return embeddingTextScoreSpectrumRecord{
		RowID:                   clean.RowID,
		Source:                  clean.Source,
		Query:                   clean.Query,
		CandidateIDs:            append([]string(nil), clean.CandidateIDs...),
		CandidateTexts:          append([]string(nil), clean.Candidates...),
		PositiveIndexes:         append([]int(nil), clean.PositiveIndexes...),
		HardNegativeEligible:    append([]bool(nil), clean.HardNegativeEligible...),
		TargetProbabilities:     append([]float32(nil), clean.TargetProbabilities...),
		HardLossWeight:          clean.HardLossWeight,
		SoftLossWeight:          clean.SoftLossWeight,
		LegalGates:              gates,
		ReleaseTrainAllowed:     clean.ReleaseTrainAllowed,
		CommercialUseAllowed:    clean.CommercialUseAllowed,
		TrainAllowedForResearch: clean.TrainAllowedForResearch,
		SourceArtifactHash:      clean.SourceArtifactHash,
		ExtraFields:             cloneRawMessageMap(clean.ExtraFields),
	}, nil
}

func (r embeddingTextScoreSpectrumRecord) example(allowResearch bool) (EmbeddingTextScoreSpectrumExample, error) {
	candidateIDs := append([]string(nil), r.CandidateIDs...)
	candidateTexts := append([]string(nil), r.CandidateTexts...)
	if len(candidateTexts) == 0 && len(r.Candidates) > 0 {
		candidateIDs = make([]string, 0, len(r.Candidates))
		candidateTexts = make([]string, 0, len(r.Candidates))
		for _, candidate := range r.Candidates {
			candidateIDs = append(candidateIDs, candidate.DocID)
			candidateTexts = append(candidateTexts, candidate.Text)
		}
	}
	positiveIndexes := append([]int(nil), r.PositiveIndexes...)
	if len(positiveIndexes) == 0 && r.SelectedPositiveIndex != nil {
		positiveIndexes = []int{*r.SelectedPositiveIndex}
	} else if r.SelectedPositiveIndex != nil {
		positiveIndexes = append(positiveIndexes, *r.SelectedPositiveIndex)
	}
	if len(r.PositiveDocIDs) > 0 {
		docPositiveIndexes, err := scoreSpectrumPositiveDocIndexes(candidateIDs, r.PositiveDocIDs)
		if err != nil {
			return EmbeddingTextScoreSpectrumExample{}, err
		}
		positiveIndexes = append(positiveIndexes, docPositiveIndexes...)
	}
	targets := r.TargetProbabilities
	if len(targets) == 0 {
		targets = r.CombinedSoftTargets
	}
	hardEligible := append([]bool(nil), r.HardNegativeEligible...)
	if len(hardEligible) == 0 && len(r.HardNegativeDocIDs) > 0 {
		hardSet := map[string]bool{}
		for _, id := range r.HardNegativeDocIDs {
			hardSet[id] = true
		}
		hardEligible = make([]bool, len(candidateIDs))
		for i, id := range candidateIDs {
			hardEligible[i] = hardSet[id]
		}
	}
	example := EmbeddingTextScoreSpectrumExample{
		RowID:                   r.RowID,
		Source:                  r.Source,
		Query:                   r.Query,
		CandidateIDs:            candidateIDs,
		Candidates:              candidateTexts,
		PositiveIndexes:         positiveIndexes,
		HardNegativeEligible:    hardEligible,
		TargetProbabilities:     append([]float32(nil), targets...),
		HardLossWeight:          r.HardLossWeight,
		SoftLossWeight:          r.SoftLossWeight,
		ReleaseTrainAllowed:     r.ReleaseTrainAllowed || r.LegalGates.ReleaseTrainAllowed,
		CommercialUseAllowed:    r.CommercialUseAllowed || r.LegalGates.CommercialUseAllowed,
		TrainAllowedForResearch: r.TrainAllowedForResearch || r.LegalGates.TrainAllowedForResearch,
		SourceArtifactHash:      r.SourceArtifactHash,
		ExtraFields:             cloneRawMessageMap(r.ExtraFields),
	}
	return validateAndCanonicalizeTextScoreSpectrum(example, allowResearch)
}

func newEmbeddingScoreSpectrumRecord(example EmbeddingScoreSpectrumExample, allowResearch bool) (embeddingScoreSpectrumRecord, error) {
	if err := validateTokenizedScoreSpectrum(example, allowResearch); err != nil {
		return embeddingScoreSpectrumRecord{}, err
	}
	probabilities, err := normalizeScoreSpectrumProbabilities(example.TargetProbabilities, len(example.CandidateTokens))
	if err != nil {
		return embeddingScoreSpectrumRecord{}, err
	}
	gates := embeddingScoreSpectrumGates{example.ReleaseTrainAllowed, example.CommercialUseAllowed, example.TrainAllowedForResearch}
	return embeddingScoreSpectrumRecord{
		RowID:                   example.RowID,
		Source:                  example.Source,
		QueryTokens:             append([]int32(nil), example.QueryTokens...),
		QueryMask:               append([]int32(nil), example.QueryMask...),
		CandidateIDs:            append([]string(nil), example.CandidateIDs...),
		CandidateTokens:         cloneInt32Matrix(example.CandidateTokens),
		CandidateMasks:          cloneInt32Matrix(example.CandidateMasks),
		PositiveIndexes:         append([]int(nil), example.PositiveIndexes...),
		HardNegativeEligible:    append([]bool(nil), example.HardNegativeEligible...),
		TargetProbabilities:     probabilities,
		HardLossWeight:          example.HardLossWeight,
		SoftLossWeight:          example.SoftLossWeight,
		LegalGates:              gates,
		ReleaseTrainAllowed:     example.ReleaseTrainAllowed,
		CommercialUseAllowed:    example.CommercialUseAllowed,
		TrainAllowedForResearch: example.TrainAllowedForResearch,
		SourceArtifactHash:      example.SourceArtifactHash,
		ExtraFields:             cloneRawMessageMap(example.ExtraFields),
	}, nil
}

func (r embeddingScoreSpectrumRecord) example(allowResearch bool) (EmbeddingScoreSpectrumExample, error) {
	example := EmbeddingScoreSpectrumExample{
		RowID:                   r.RowID,
		Source:                  r.Source,
		QueryTokens:             r.QueryTokens,
		QueryMask:               r.QueryMask,
		CandidateIDs:            r.CandidateIDs,
		CandidateTokens:         r.CandidateTokens,
		CandidateMasks:          r.CandidateMasks,
		PositiveIndexes:         r.PositiveIndexes,
		HardNegativeEligible:    r.HardNegativeEligible,
		TargetProbabilities:     r.TargetProbabilities,
		HardLossWeight:          r.HardLossWeight,
		SoftLossWeight:          r.SoftLossWeight,
		ReleaseTrainAllowed:     r.ReleaseTrainAllowed || r.LegalGates.ReleaseTrainAllowed,
		CommercialUseAllowed:    r.CommercialUseAllowed || r.LegalGates.CommercialUseAllowed,
		TrainAllowedForResearch: r.TrainAllowedForResearch || r.LegalGates.TrainAllowedForResearch,
		SourceArtifactHash:      r.SourceArtifactHash,
		ExtraFields:             r.ExtraFields,
	}
	if err := validateTokenizedScoreSpectrum(example, allowResearch); err != nil {
		return EmbeddingScoreSpectrumExample{}, err
	}
	probabilities, err := normalizeScoreSpectrumProbabilities(r.TargetProbabilities, len(r.CandidateTokens))
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, err
	}
	example.QueryTokens = append([]int32(nil), r.QueryTokens...)
	example.QueryMask = append([]int32(nil), r.QueryMask...)
	example.CandidateIDs = append([]string(nil), r.CandidateIDs...)
	example.CandidateTokens = cloneInt32Matrix(r.CandidateTokens)
	example.CandidateMasks = cloneInt32Matrix(r.CandidateMasks)
	example.PositiveIndexes = append([]int(nil), r.PositiveIndexes...)
	example.HardNegativeEligible = append([]bool(nil), r.HardNegativeEligible...)
	example.TargetProbabilities = probabilities
	example.ExtraFields = cloneRawMessageMap(r.ExtraFields)
	return example, nil
}

func validateAndCanonicalizeTextScoreSpectrum(example EmbeddingTextScoreSpectrumExample, allowResearch bool) (EmbeddingTextScoreSpectrumExample, error) {
	if strings.TrimSpace(example.Query) == "" {
		return EmbeddingTextScoreSpectrumExample{}, fmt.Errorf("query is empty")
	}
	if len(example.Candidates) == 0 {
		return EmbeddingTextScoreSpectrumExample{}, fmt.Errorf("candidates are empty")
	}
	if len(example.CandidateIDs) > 0 && len(example.CandidateIDs) != len(example.Candidates) {
		return EmbeddingTextScoreSpectrumExample{}, fmt.Errorf("candidate_ids length %d does not match candidates length %d", len(example.CandidateIDs), len(example.Candidates))
	}
	if len(example.CandidateIDs) == 0 {
		example.CandidateIDs = make([]string, len(example.Candidates))
	}
	if len(example.HardNegativeEligible) == 0 {
		example.HardNegativeEligible = make([]bool, len(example.Candidates))
	}
	if len(example.HardNegativeEligible) != len(example.Candidates) {
		return EmbeddingTextScoreSpectrumExample{}, fmt.Errorf("hard_negative_eligible length %d does not match candidates length %d", len(example.HardNegativeEligible), len(example.Candidates))
	}
	if err := validateScoreSpectrumLegalGates(example.ReleaseTrainAllowed, example.CommercialUseAllowed, example.TrainAllowedForResearch, allowResearch); err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	if err := validateScoreSpectrumLabelsAndProbabilities(len(example.Candidates), example.PositiveIndexes, example.HardNegativeEligible, example.TargetProbabilities); err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	clean, err := mergeDuplicateTextScoreSpectrumCandidates(example)
	if err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	if err := validateScoreSpectrumLabelsAndProbabilities(len(clean.Candidates), clean.PositiveIndexes, clean.HardNegativeEligible, clean.TargetProbabilities); err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	return clean, nil
}

func validateTokenizedScoreSpectrum(example EmbeddingScoreSpectrumExample, allowResearch bool) error {
	if len(example.QueryTokens) == 0 {
		return fmt.Errorf("query_tokens are empty")
	}
	if len(example.QueryMask) > 0 && len(example.QueryMask) != len(example.QueryTokens) {
		return fmt.Errorf("query_mask length %d does not match query_tokens length %d", len(example.QueryMask), len(example.QueryTokens))
	}
	if len(example.CandidateTokens) == 0 {
		return fmt.Errorf("candidate_tokens are empty")
	}
	if len(example.CandidateIDs) > 0 && len(example.CandidateIDs) != len(example.CandidateTokens) {
		return fmt.Errorf("candidate_ids length %d does not match candidate_tokens length %d", len(example.CandidateIDs), len(example.CandidateTokens))
	}
	if len(example.CandidateMasks) > 0 && len(example.CandidateMasks) != len(example.CandidateTokens) {
		return fmt.Errorf("candidate_masks length %d does not match candidate_tokens length %d", len(example.CandidateMasks), len(example.CandidateTokens))
	}
	for i, tokens := range example.CandidateTokens {
		if len(tokens) == 0 {
			return fmt.Errorf("candidate_tokens[%d] are empty", i)
		}
		if len(example.CandidateMasks) > i && len(example.CandidateMasks[i]) > 0 && len(example.CandidateMasks[i]) != len(tokens) {
			return fmt.Errorf("candidate_masks[%d] length %d does not match candidate_tokens[%d] length %d", i, len(example.CandidateMasks[i]), i, len(tokens))
		}
	}
	if err := validateScoreSpectrumLegalGates(example.ReleaseTrainAllowed, example.CommercialUseAllowed, example.TrainAllowedForResearch, allowResearch); err != nil {
		return err
	}
	return validateScoreSpectrumLabelsAndProbabilities(len(example.CandidateTokens), example.PositiveIndexes, example.HardNegativeEligible, example.TargetProbabilities)
}

type scoreSpectrumMergeCandidate struct {
	id       string
	text     string
	prob     float32
	positive bool
	hard     bool
}

func mergeDuplicateTextScoreSpectrumCandidates(example EmbeddingTextScoreSpectrumExample) (EmbeddingTextScoreSpectrumExample, error) {
	positiveSet, err := scoreSpectrumPositiveSet(len(example.Candidates), example.PositiveIndexes)
	if err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	probs, err := normalizeScoreSpectrumProbabilities(example.TargetProbabilities, len(example.Candidates))
	if err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	indexByKey := map[string]int{}
	merged := []scoreSpectrumMergeCandidate{}
	for i, text := range example.Candidates {
		if strings.TrimSpace(text) == "" {
			return EmbeddingTextScoreSpectrumExample{}, fmt.Errorf("candidate %d is empty", i)
		}
		key := normalizeScoreSpectrumCandidateText(text)
		if j, ok := indexByKey[key]; ok {
			if (merged[j].positive || positiveSet[i]) && (merged[j].hard || example.HardNegativeEligible[i]) {
				return EmbeddingTextScoreSpectrumExample{}, fmt.Errorf("duplicate candidate %d conflicts between positive and hard-negative labels", i)
			}
			merged[j].prob += probs[i]
			merged[j].positive = merged[j].positive || positiveSet[i]
			merged[j].hard = merged[j].hard || example.HardNegativeEligible[i]
			continue
		}
		indexByKey[key] = len(merged)
		merged = append(merged, scoreSpectrumMergeCandidate{
			id:       example.CandidateIDs[i],
			text:     text,
			prob:     probs[i],
			positive: positiveSet[i],
			hard:     example.HardNegativeEligible[i],
		})
	}
	out := example
	out.CandidateIDs = make([]string, len(merged))
	out.Candidates = make([]string, len(merged))
	out.HardNegativeEligible = make([]bool, len(merged))
	out.TargetProbabilities = make([]float32, len(merged))
	out.PositiveIndexes = out.PositiveIndexes[:0]
	for i, candidate := range merged {
		out.CandidateIDs[i] = candidate.id
		out.Candidates[i] = candidate.text
		out.HardNegativeEligible[i] = candidate.hard
		out.TargetProbabilities[i] = candidate.prob
		if candidate.positive {
			out.PositiveIndexes = append(out.PositiveIndexes, i)
			out.HardNegativeEligible[i] = false
		}
	}
	out.TargetProbabilities, err = normalizeScoreSpectrumProbabilities(out.TargetProbabilities, len(out.Candidates))
	if err != nil {
		return EmbeddingTextScoreSpectrumExample{}, err
	}
	return out, nil
}

func validateScoreSpectrumLabelsAndProbabilities(candidateCount int, positiveIndexes []int, hardEligible []bool, probabilities []float32) error {
	if candidateCount == 0 {
		return fmt.Errorf("candidates are empty")
	}
	positiveSet, err := scoreSpectrumPositiveSet(candidateCount, positiveIndexes)
	if err != nil {
		return err
	}
	if len(hardEligible) != candidateCount {
		return fmt.Errorf("hard_negative_eligible length %d does not match candidate count %d", len(hardEligible), candidateCount)
	}
	for i := range hardEligible {
		if positiveSet[i] && hardEligible[i] {
			return fmt.Errorf("positive candidate %d cannot be hard-negative eligible", i)
		}
	}
	_, err = normalizeScoreSpectrumProbabilities(probabilities, candidateCount)
	return err
}

func normalizeScoreSpectrumProbabilities(probabilities []float32, want int) ([]float32, error) {
	if len(probabilities) != want {
		return nil, fmt.Errorf("target_probabilities length %d does not match candidate count %d", len(probabilities), want)
	}
	sum := 0.0
	for i, p := range probabilities {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
			return nil, fmt.Errorf("target_probabilities[%d] must be finite", i)
		}
		if p < 0 {
			return nil, fmt.Errorf("target_probabilities[%d] must be nonnegative", i)
		}
		sum += float64(p)
	}
	if sum <= 0 {
		return nil, fmt.Errorf("target_probabilities sum must be positive")
	}
	out := make([]float32, want)
	running := float32(0)
	for i := 0; i < want; i++ {
		if i == want-1 {
			last := float32(1) - running
			if last < 0 {
				last = 0
			}
			out[i] = last
			break
		}
		out[i] = float32(float64(probabilities[i]) / sum)
		running += out[i]
	}
	return out, nil
}

func scoreSpectrumPositiveSet(candidateCount int, positiveIndexes []int) ([]bool, error) {
	if len(positiveIndexes) == 0 {
		return nil, fmt.Errorf("positive_indexes are empty")
	}
	out := make([]bool, candidateCount)
	for _, idx := range positiveIndexes {
		if idx < 0 || idx >= candidateCount {
			return nil, fmt.Errorf("positive index %d out of range for %d candidates", idx, candidateCount)
		}
		out[idx] = true
	}
	return out, nil
}

func scoreSpectrumPositiveDocIndexes(candidateIDs, positiveDocIDs []string) ([]int, error) {
	if len(candidateIDs) == 0 {
		return nil, fmt.Errorf("positive_doc_ids require candidate_doc_ids")
	}
	indexByID := make(map[string]int, len(candidateIDs))
	for i, id := range candidateIDs {
		if strings.TrimSpace(id) == "" {
			continue
		}
		if _, exists := indexByID[id]; exists {
			return nil, fmt.Errorf("candidate_doc_ids contains duplicate id %q", id)
		}
		indexByID[id] = i
	}
	out := make([]int, 0, len(positiveDocIDs))
	for _, id := range positiveDocIDs {
		idx, ok := indexByID[id]
		if !ok {
			return nil, fmt.Errorf("positive_doc_ids contains unknown candidate id %q", id)
		}
		out = append(out, idx)
	}
	return out, nil
}

func validateScoreSpectrumLegalGates(releaseTrainAllowed, commercialUseAllowed, trainAllowedForResearch, allowResearch bool) error {
	researchOnly := trainAllowedForResearch && !releaseTrainAllowed && !commercialUseAllowed
	if researchOnly && !allowResearch {
		return fmt.Errorf("score-spectrum row is research-only; set AllowResearchOnly to read it")
	}
	if allowResearch {
		if !trainAllowedForResearch || releaseTrainAllowed || commercialUseAllowed {
			return fmt.Errorf("research-only score-spectrum rows require train_allowed_for_research=true, release_train_allowed=false, commercial_use_allowed=false")
		}
	}
	return nil
}

func normalizeScoreSpectrumCandidateText(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

func scoreSpectrumAllowResearch(opts []EmbeddingScoreSpectrumReadOptions) bool {
	return len(opts) > 0 && opts[0].AllowResearchOnly
}
