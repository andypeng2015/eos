package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const embeddingVectorDistillExampleSchemaV1 = "eos.vector_distill_example.v1"

// EmbeddingVectorDistillExample is one JSONL row of a vector-distillation dataset.
// Each row pairs raw text with a pre-computed teacher embedding vector.
type EmbeddingVectorDistillExample struct {
	Schema                  string
	ID                      string
	Text                    string
	TeacherVector           []float32
	TrainAllowedForResearch bool
	ReleaseTrainAllowed     bool
	CommercialUseAllowed    bool
}

// EmbeddingTokenizedVectorDistillExample is the tokenizer-adapted form used
// by the trainer; TeacherVector is preserved verbatim.
type EmbeddingTokenizedVectorDistillExample struct {
	ID                      string
	Tokens                  []int32
	Mask                    []int32
	TeacherVector           []float32
	TrainAllowedForResearch bool
	ReleaseTrainAllowed     bool
	CommercialUseAllowed    bool
}

type embeddingVectorDistillRecord struct {
	Schema                  string    `json:"schema"`
	ID                      string    `json:"id"`
	Text                    string    `json:"text"`
	TeacherVector           []float32 `json:"teacher_vector"`
	TrainAllowedForResearch bool      `json:"train_allowed_for_research,omitempty"`
	ReleaseTrainAllowed     bool      `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed    bool      `json:"commercial_use_allowed,omitempty"`
}

// ReadEmbeddingVectorDistillExamplesFile reads a vector-distillation JSONL file.
func ReadEmbeddingVectorDistillExamplesFile(path string) ([]EmbeddingVectorDistillExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingVectorDistillExample
	if err := scanEmbeddingJSONLLines(f, func(lineNo int, line string) error {
		var rec embeddingVectorDistillRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		ex, err := rec.example()
		if err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, ex)
		return nil
	}); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("vector distillation dataset is empty")
	}
	return out, nil
}

// TokenizeEmbeddingVectorDistillExamples tokenizes text from each example.
func TokenizeEmbeddingVectorDistillExamples(examples []EmbeddingVectorDistillExample, tokenizer *BPETokenizer) ([]EmbeddingTokenizedVectorDistillExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("vector distillation dataset is empty")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	cache := embeddingTextTokenCache{}
	out := make([]EmbeddingTokenizedVectorDistillExample, 0, len(examples))
	for i, ex := range examples {
		encoded, err := cache.encode(ex.Text, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d (%s): %w", i, ex.ID, err)
		}
		encoded = cloneTokenizedText(encoded)
		out = append(out, EmbeddingTokenizedVectorDistillExample{
			ID:                      ex.ID,
			Tokens:                  encoded.tokens,
			Mask:                    encoded.mask,
			TeacherVector:           append([]float32(nil), ex.TeacherVector...),
			TrainAllowedForResearch: ex.TrainAllowedForResearch,
			ReleaseTrainAllowed:     ex.ReleaseTrainAllowed,
			CommercialUseAllowed:    ex.CommercialUseAllowed,
		})
	}
	return out, nil
}

func (r embeddingVectorDistillRecord) example() (EmbeddingVectorDistillExample, error) {
	if r.Schema != embeddingVectorDistillExampleSchemaV1 {
		return EmbeddingVectorDistillExample{}, fmt.Errorf("schema %q does not match %q", r.Schema, embeddingVectorDistillExampleSchemaV1)
	}
	if strings.TrimSpace(r.ID) == "" {
		return EmbeddingVectorDistillExample{}, fmt.Errorf("id is empty")
	}
	if strings.TrimSpace(r.Text) == "" {
		return EmbeddingVectorDistillExample{}, fmt.Errorf("example %q text is empty", r.ID)
	}
	if len(r.TeacherVector) == 0 {
		return EmbeddingVectorDistillExample{}, fmt.Errorf("example %q teacher_vector is empty", r.ID)
	}
	for i, v := range r.TeacherVector {
		if !isFinite32(v) {
			return EmbeddingVectorDistillExample{}, fmt.Errorf("example %q teacher_vector[%d] must be finite", r.ID, i)
		}
	}
	return EmbeddingVectorDistillExample{
		Schema:                  r.Schema,
		ID:                      r.ID,
		Text:                    r.Text,
		TeacherVector:           append([]float32(nil), r.TeacherVector...),
		TrainAllowedForResearch: r.TrainAllowedForResearch,
		ReleaseTrainAllowed:     r.ReleaseTrainAllowed,
		CommercialUseAllowed:    r.CommercialUseAllowed,
	}, nil
}

// validateVectorDistillResearchGates returns an error if any training example
// is research-only and allowResearchOnly is false.
func validateVectorDistillResearchGates(examples []EmbeddingTokenizedVectorDistillExample, allowResearchOnly bool) error {
	for i, ex := range examples {
		if ex.TrainAllowedForResearch && !ex.ReleaseTrainAllowed && !allowResearchOnly {
			return fmt.Errorf("vector distillation example %d (%s) is research-only (train_allowed_for_research=true, release_train_allowed=false); pass --allow-research-only-vector-distill to allow", i, ex.ID)
		}
	}
	return nil
}
