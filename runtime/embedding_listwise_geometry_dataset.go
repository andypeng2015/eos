package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const embeddingListwiseGeometryBatchSchemaV1 = "eos.listwise_geometry_batch.v1"

// EmbeddingListwiseGeometryBatch is one JSONL row of listwise query/document teacher geometry.
type EmbeddingListwiseGeometryBatch struct {
	Schema                  string
	BatchID                 string
	SourceCounts            map[string]int
	Examples                []EmbeddingListwiseGeometryExample
	Queries                 []EmbeddingListwiseGeometryQuery
	Documents               []EmbeddingListwiseGeometryDocument
	TeacherSimilarity       [][]float32
	TeacherModelID          string
	Score                   string
	Normalized              bool
	TrainPolicy             string
	ReleaseTrainAllowed     bool
	CommercialUseAllowed    bool
	TrainAllowedForResearch bool
	SourceArtifactHash      string
	ExtraFields             map[string]json.RawMessage
}

type EmbeddingListwiseGeometryExample struct {
	RowID          string   `json:"row_id"`
	Source         string   `json:"source"`
	QueryID        string   `json:"query_id"`
	PositiveDocID  string   `json:"positive_doc_id"`
	NegativeDocIDs []string `json:"negative_doc_ids,omitempty"`
}

type EmbeddingListwiseGeometryQuery struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type EmbeddingListwiseGeometryDocument struct {
	ID   string `json:"id"`
	Text string `json:"text"`
	Role string `json:"role,omitempty"`
}

type embeddingListwiseGeometryBatchRecord struct {
	Schema                  string                              `json:"schema"`
	BatchID                 string                              `json:"batch_id,omitempty"`
	SourceCounts            map[string]int                      `json:"source_counts,omitempty"`
	Examples                []EmbeddingListwiseGeometryExample  `json:"examples"`
	Queries                 []EmbeddingListwiseGeometryQuery    `json:"queries"`
	Documents               []EmbeddingListwiseGeometryDocument `json:"documents"`
	TeacherSimilarity       [][]float32                         `json:"teacher_similarity"`
	TeacherModelID          string                              `json:"teacher_model_id,omitempty"`
	Score                   string                              `json:"score,omitempty"`
	Normalized              bool                                `json:"normalized,omitempty"`
	TrainPolicy             string                              `json:"train_policy,omitempty"`
	LegalGates              embeddingScoreSpectrumGates         `json:"legal_gates,omitempty"`
	ReleaseTrainAllowed     bool                                `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed    bool                                `json:"commercial_use_allowed,omitempty"`
	TrainAllowedForResearch bool                                `json:"train_allowed_for_research,omitempty"`
	SourceArtifactHash      string                              `json:"source_artifact_hash,omitempty"`
	ExtraFields             map[string]json.RawMessage
}

type embeddingListwiseGeometryBatchKnownRecord embeddingListwiseGeometryBatchRecord

var embeddingListwiseGeometryBatchKnownFields = map[string]struct{}{
	"schema":                     {},
	"batch_id":                   {},
	"source_counts":              {},
	"examples":                   {},
	"queries":                    {},
	"documents":                  {},
	"teacher_similarity":         {},
	"teacher_model_id":           {},
	"score":                      {},
	"normalized":                 {},
	"train_policy":               {},
	"legal_gates":                {},
	"release_train_allowed":      {},
	"commercial_use_allowed":     {},
	"train_allowed_for_research": {},
	"source_artifact_hash":       {},
}

func (r *embeddingListwiseGeometryBatchRecord) UnmarshalJSON(data []byte) error {
	var known embeddingListwiseGeometryBatchKnownRecord
	if err := json.Unmarshal(data, &known); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for key := range embeddingListwiseGeometryBatchKnownFields {
		delete(fields, key)
	}
	*r = embeddingListwiseGeometryBatchRecord(known)
	r.ExtraFields = cloneRawMessageMap(fields)
	return nil
}

// ReadEmbeddingListwiseGeometryBatchesFile reads listwise geometry JSONL batch rows.
func ReadEmbeddingListwiseGeometryBatchesFile(path string) ([]EmbeddingListwiseGeometryBatch, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingListwiseGeometryBatch
	if err := scanEmbeddingJSONLLines(f, func(lineNo int, line string) error {
		var record embeddingListwiseGeometryBatchRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		batch, err := record.batch()
		if err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, batch)
		return nil
	}); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("listwise geometry dataset is empty")
	}
	return out, nil
}

func (r embeddingListwiseGeometryBatchRecord) batch() (EmbeddingListwiseGeometryBatch, error) {
	if r.Schema != embeddingListwiseGeometryBatchSchemaV1 {
		return EmbeddingListwiseGeometryBatch{}, fmt.Errorf("schema %q does not match %q", r.Schema, embeddingListwiseGeometryBatchSchemaV1)
	}
	if strings.TrimSpace(r.BatchID) == "" {
		return EmbeddingListwiseGeometryBatch{}, fmt.Errorf("batch_id is empty")
	}
	if len(r.Examples) == 0 {
		return EmbeddingListwiseGeometryBatch{}, fmt.Errorf("examples are empty")
	}
	if len(r.Queries) == 0 {
		return EmbeddingListwiseGeometryBatch{}, fmt.Errorf("queries are empty")
	}
	if len(r.Documents) == 0 {
		return EmbeddingListwiseGeometryBatch{}, fmt.Errorf("documents are empty")
	}
	if err := validateEmbeddingListwiseGeometryExamples(r.Examples); err != nil {
		return EmbeddingListwiseGeometryBatch{}, err
	}
	if err := validateEmbeddingListwiseGeometryQueries(r.Queries); err != nil {
		return EmbeddingListwiseGeometryBatch{}, err
	}
	if err := validateEmbeddingListwiseGeometryDocuments(r.Documents); err != nil {
		return EmbeddingListwiseGeometryBatch{}, err
	}
	if err := validateEmbeddingListwiseGeometryMatrix(r.TeacherSimilarity, len(r.Queries), len(r.Documents), "teacher_similarity"); err != nil {
		return EmbeddingListwiseGeometryBatch{}, err
	}
	if err := validateEmbeddingListwiseGeometryScore(r.Score); err != nil {
		return EmbeddingListwiseGeometryBatch{}, err
	}
	if r.LegalGates != (embeddingScoreSpectrumGates{}) {
		r.ReleaseTrainAllowed = r.LegalGates.ReleaseTrainAllowed
		r.CommercialUseAllowed = r.LegalGates.CommercialUseAllowed
		r.TrainAllowedForResearch = r.LegalGates.TrainAllowedForResearch
	}
	return EmbeddingListwiseGeometryBatch{
		Schema:                  r.Schema,
		BatchID:                 r.BatchID,
		SourceCounts:            cloneStringIntMap(r.SourceCounts),
		Examples:                cloneEmbeddingListwiseGeometryExamples(r.Examples),
		Queries:                 append([]EmbeddingListwiseGeometryQuery(nil), r.Queries...),
		Documents:               append([]EmbeddingListwiseGeometryDocument(nil), r.Documents...),
		TeacherSimilarity:       cloneFloat32Matrix(r.TeacherSimilarity),
		TeacherModelID:          r.TeacherModelID,
		Score:                   r.Score,
		Normalized:              r.Normalized,
		TrainPolicy:             r.TrainPolicy,
		ReleaseTrainAllowed:     r.ReleaseTrainAllowed,
		CommercialUseAllowed:    r.CommercialUseAllowed,
		TrainAllowedForResearch: r.TrainAllowedForResearch,
		SourceArtifactHash:      r.SourceArtifactHash,
		ExtraFields:             cloneRawMessageMap(r.ExtraFields),
	}, nil
}

func validateEmbeddingListwiseGeometryScore(score string) error {
	switch score {
	case "", "cosine", "dot":
		return nil
	default:
		return fmt.Errorf("score %q is unsupported; want empty, cosine, or dot", score)
	}
}

func validateEmbeddingListwiseGeometryExamples(examples []EmbeddingListwiseGeometryExample) error {
	for i, example := range examples {
		if strings.TrimSpace(example.RowID) == "" {
			return fmt.Errorf("examples[%d].row_id is empty", i)
		}
		if strings.TrimSpace(example.Source) == "" {
			return fmt.Errorf("examples[%d].source is empty", i)
		}
		if strings.TrimSpace(example.QueryID) == "" {
			return fmt.Errorf("examples[%d].query_id is empty", i)
		}
		if strings.TrimSpace(example.PositiveDocID) == "" {
			return fmt.Errorf("examples[%d].positive_doc_id is empty", i)
		}
		for j, id := range example.NegativeDocIDs {
			if strings.TrimSpace(id) == "" {
				return fmt.Errorf("examples[%d].negative_doc_ids[%d] is empty", i, j)
			}
		}
	}
	return nil
}

func validateEmbeddingListwiseGeometryQueries(queries []EmbeddingListwiseGeometryQuery) error {
	seen := map[string]bool{}
	for i, query := range queries {
		if strings.TrimSpace(query.ID) == "" {
			return fmt.Errorf("queries[%d].id is empty", i)
		}
		if strings.TrimSpace(query.Text) == "" {
			return fmt.Errorf("queries[%d].text is empty", i)
		}
		if seen[query.ID] {
			return fmt.Errorf("queries[%d].id %q is duplicated", i, query.ID)
		}
		seen[query.ID] = true
	}
	return nil
}

func validateEmbeddingListwiseGeometryDocuments(documents []EmbeddingListwiseGeometryDocument) error {
	seen := map[string]bool{}
	for i, document := range documents {
		if strings.TrimSpace(document.ID) == "" {
			return fmt.Errorf("documents[%d].id is empty", i)
		}
		if strings.TrimSpace(document.Text) == "" {
			return fmt.Errorf("documents[%d].text is empty", i)
		}
		if seen[document.ID] {
			return fmt.Errorf("documents[%d].id %q is duplicated", i, document.ID)
		}
		seen[document.ID] = true
	}
	return nil
}

func validateEmbeddingListwiseGeometryMatrix(matrix [][]float32, rows, cols int, name string) error {
	if len(matrix) != rows {
		return fmt.Errorf("%s row count %d does not match query count %d", name, len(matrix), rows)
	}
	for i, row := range matrix {
		if len(row) != cols {
			return fmt.Errorf("%s[%d] length %d does not match document count %d", name, i, len(row), cols)
		}
		for j, value := range row {
			if !isFinite32(value) {
				return fmt.Errorf("%s[%d][%d] must be finite", name, i, j)
			}
		}
	}
	return nil
}

func cloneEmbeddingListwiseGeometryExamples(in []EmbeddingListwiseGeometryExample) []EmbeddingListwiseGeometryExample {
	if len(in) == 0 {
		return nil
	}
	out := make([]EmbeddingListwiseGeometryExample, len(in))
	for i, example := range in {
		out[i] = example
		out[i].NegativeDocIDs = append([]string(nil), example.NegativeDocIDs...)
	}
	return out
}

func cloneFloat32Matrix(in [][]float32) [][]float32 {
	if len(in) == 0 {
		return nil
	}
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func cloneStringIntMap(in map[string]int) map[string]int {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]int, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
