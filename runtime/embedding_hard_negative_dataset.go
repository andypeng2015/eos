package eosruntime

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// EmbeddingHardNegativeExample is one query-positive example with explicit hard negatives.
type EmbeddingHardNegativeExample struct {
	Source         string
	QueryTokens    []int32
	PositiveTokens []int32
	NegativeTokens [][]int32
	QueryMask      []int32
	PositiveMask   []int32
	NegativeMasks  [][]int32
	TeacherScores  []float32
}

type EmbeddingTextHardNegativeExample struct {
	Source        string
	Query         string
	Positive      string
	Negatives     []string
	TeacherScores []float32
}

type embeddingHardNegativeRecord struct {
	Source         string    `json:"source,omitempty"`
	QueryTokens    []int32   `json:"query_tokens"`
	PositiveTokens []int32   `json:"positive_tokens"`
	NegativeTokens [][]int32 `json:"negative_tokens,omitempty"`
	QueryMask      []int32   `json:"query_mask,omitempty"`
	PositiveMask   []int32   `json:"positive_mask,omitempty"`
	NegativeMasks  [][]int32 `json:"negative_masks,omitempty"`
	TeacherScores  []float32 `json:"teacher_scores,omitempty"`
}

type embeddingTextHardNegativeRecord struct {
	Source        string    `json:"source,omitempty"`
	Query         string    `json:"query"`
	Positive      string    `json:"positive"`
	Document      string    `json:"document,omitempty"`
	Negatives     []string  `json:"negatives,omitempty"`
	TeacherScores []float32 `json:"teacher_scores,omitempty"`
}

// ReadEmbeddingHardNegativeExamplesFile reads tokenized hard-negative JSONL.
func ReadEmbeddingHardNegativeExamplesFile(path string) ([]EmbeddingHardNegativeExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingHardNegativeExample
	if err := scanEmbeddingJSONLLines(f, func(lineNo int, line string) error {
		var record embeddingHardNegativeRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		example, err := record.example()
		if err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, example)
		return nil
	}); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("hard-negative dataset is empty")
	}
	return out, nil
}

// WriteEmbeddingHardNegativeExamplesFile writes tokenized hard-negative JSONL.
func WriteEmbeddingHardNegativeExamplesFile(path string, examples []EmbeddingHardNegativeExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("hard-negative dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingHardNegativeRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

// BuildEmbeddingHardNegativeExamplesFromPairs groups labeled pair examples by query.
func BuildEmbeddingHardNegativeExamplesFromPairs(pairs []EmbeddingPairExample, maxNegatives int) ([]EmbeddingHardNegativeExample, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("pair dataset is empty")
	}
	type queryGroup struct {
		source      string
		queryTokens []int32
		queryMask   []int32
		positives   []embeddingTokenizedText
		negatives   []embeddingTokenizedText
	}
	groups := map[string]*queryGroup{}
	order := []string{}
	for _, pair := range pairs {
		key := embeddingBatchSequenceKey(pair.LeftTokens, pair.LeftMask)
		group := groups[key]
		if group == nil {
			group = &queryGroup{
				source:      pair.Source,
				queryTokens: append([]int32(nil), pair.LeftTokens...),
				queryMask:   append([]int32(nil), pair.LeftMask...),
			}
			groups[key] = group
			order = append(order, key)
		} else if group.source == "" {
			group.source = pair.Source
		}
		item := embeddingTokenizedText{
			tokens: append([]int32(nil), pair.RightTokens...),
			mask:   append([]int32(nil), pair.RightMask...),
		}
		if pair.Target > 0 {
			group.positives = append(group.positives, item)
		} else {
			group.negatives = append(group.negatives, item)
		}
	}
	out := []EmbeddingHardNegativeExample{}
	for _, key := range order {
		group := groups[key]
		if len(group.positives) == 0 || len(group.negatives) == 0 {
			continue
		}
		limit := maxNegatives
		if limit <= 0 || limit > len(group.negatives) {
			limit = len(group.negatives)
		}
		for i, positive := range group.positives {
			negatives := make([][]int32, 0, limit)
			negativeMasks := make([][]int32, 0, limit)
			for j := 0; j < limit; j++ {
				negative := group.negatives[(i+j)%len(group.negatives)]
				negatives = append(negatives, append([]int32(nil), negative.tokens...))
				negativeMasks = append(negativeMasks, append([]int32(nil), negative.mask...))
			}
			out = append(out, EmbeddingHardNegativeExample{
				Source:         group.source,
				QueryTokens:    append([]int32(nil), group.queryTokens...),
				PositiveTokens: append([]int32(nil), positive.tokens...),
				NegativeTokens: negatives,
				QueryMask:      append([]int32(nil), group.queryMask...),
				PositiveMask:   append([]int32(nil), positive.mask...),
				NegativeMasks:  negativeMasks,
			})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("hard-negative dataset has no positive query groups with negatives")
	}
	return out, nil
}

// ReadEmbeddingTextHardNegativeExamplesFile reads grouped text hard-negative JSONL.
func ReadEmbeddingTextHardNegativeExamplesFile(path string) ([]EmbeddingTextHardNegativeExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingTextHardNegativeExample
	if err := scanEmbeddingJSONLLines(f, func(lineNo int, line string) error {
		var record embeddingTextHardNegativeRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		example, err := record.example()
		if err != nil {
			return fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, example)
		return nil
	}); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text hard-negative dataset is empty")
	}
	return out, nil
}

func WriteEmbeddingTextHardNegativeExamplesFile(path string, examples []EmbeddingTextHardNegativeExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("text hard-negative dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingTextHardNegativeRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func BuildEmbeddingTextHardNegativeExamplesFromPairs(pairs []EmbeddingTextPairExample, maxNegatives int) ([]EmbeddingTextHardNegativeExample, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("text pair dataset is empty")
	}
	type queryGroup struct {
		source    string
		positives []string
		negatives []string
	}
	groups := map[string]*queryGroup{}
	order := []string{}
	for _, pair := range pairs {
		key := pair.Query
		group := groups[key]
		if group == nil {
			group = &queryGroup{source: pair.Source}
			groups[key] = group
			order = append(order, key)
		} else if group.source == "" {
			group.source = pair.Source
		}
		if pair.Target > 0 {
			group.positives = append(group.positives, pair.Right)
		} else {
			group.negatives = append(group.negatives, pair.Right)
		}
	}
	out := []EmbeddingTextHardNegativeExample{}
	for _, query := range order {
		group := groups[query]
		if len(group.positives) == 0 || len(group.negatives) == 0 {
			continue
		}
		limit := maxNegatives
		if limit <= 0 || limit > len(group.negatives) {
			limit = len(group.negatives)
		}
		for i, positive := range group.positives {
			negatives := make([]string, 0, limit)
			for j := 0; j < limit; j++ {
				negatives = append(negatives, group.negatives[(i+j)%len(group.negatives)])
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Source:    group.source,
				Query:     query,
				Positive:  positive,
				Negatives: negatives,
			})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text hard-negative dataset has no positive query groups with negatives")
	}
	return out, nil
}

func ReadEmbeddingHardNegativeEvalPairsFile(path string, maxNegatives int) ([]EmbeddingPairExample, error) {
	grouped, err := ReadEmbeddingHardNegativeExamplesFile(path)
	if err == nil {
		return BuildEmbeddingHardNegativeEvalPairs(grouped, maxNegatives)
	}
	pairs, pairErr := ReadEmbeddingPairExamplesFile(path)
	if pairErr != nil {
		return nil, fmt.Errorf("read eval hard-negative or pair dataset: %w", err)
	}
	return pairs, nil
}

func ReadEmbeddingTextHardNegativeEvalPairsFile(path string, maxNegatives int) ([]EmbeddingTextPairExample, error) {
	grouped, err := ReadEmbeddingTextHardNegativeExamplesFile(path)
	if err == nil {
		return BuildEmbeddingTextHardNegativeEvalPairs(grouped, maxNegatives)
	}
	pairs, pairErr := ReadEmbeddingTextPairExamplesFile(path)
	if pairErr != nil {
		return nil, fmt.Errorf("read eval text hard-negative or pair dataset: %w", err)
	}
	return pairs, nil
}

func BuildEmbeddingHardNegativeEvalPairs(examples []EmbeddingHardNegativeExample, maxNegatives int) ([]EmbeddingPairExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("hard-negative eval dataset is empty")
	}
	out := []EmbeddingPairExample{}
	for _, example := range examples {
		limit := hardNegativeEvalNegativeLimit(len(example.NegativeTokens), maxNegatives)
		for i := 0; i < limit; i++ {
			pair := EmbeddingPairExample{
				Source:      example.Source,
				LeftTokens:  append([]int32(nil), example.QueryTokens...),
				LeftMask:    append([]int32(nil), example.QueryMask...),
				RightTokens: append([]int32(nil), example.PositiveTokens...),
				RightMask:   append([]int32(nil), example.PositiveMask...),
				Target:      1,
			}
			out = append(out, pair)
			negative := EmbeddingPairExample{
				Source:      example.Source,
				LeftTokens:  append([]int32(nil), example.QueryTokens...),
				LeftMask:    append([]int32(nil), example.QueryMask...),
				RightTokens: append([]int32(nil), example.NegativeTokens[i]...),
				Target:      0,
			}
			if len(example.NegativeMasks) > i {
				negative.RightMask = append([]int32(nil), example.NegativeMasks[i]...)
			}
			out = append(out, negative)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("hard-negative eval dataset has no negatives")
	}
	return out, nil
}

func BuildEmbeddingTextHardNegativeEvalPairs(examples []EmbeddingTextHardNegativeExample, maxNegatives int) ([]EmbeddingTextPairExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("text hard-negative eval dataset is empty")
	}
	out := []EmbeddingTextPairExample{}
	for _, example := range examples {
		limit := hardNegativeEvalNegativeLimit(len(example.Negatives), maxNegatives)
		for i := 0; i < limit; i++ {
			out = append(out,
				EmbeddingTextPairExample{
					Source: example.Source,
					Query:  example.Query,
					Right:  example.Positive,
					Target: 1,
				},
				EmbeddingTextPairExample{
					Source: example.Source,
					Query:  example.Query,
					Right:  example.Negatives[i],
					Target: 0,
				},
			)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text hard-negative eval dataset has no negatives")
	}
	return out, nil
}

func hardNegativeEvalNegativeLimit(total, maxNegatives int) int {
	if maxNegatives <= 0 || maxNegatives > total {
		return total
	}
	return maxNegatives
}

func TokenizeEmbeddingTextHardNegativeExamples(examples []EmbeddingTextHardNegativeExample, tokenizer *BPETokenizer) ([]EmbeddingHardNegativeExample, error) {
	return tokenizeEmbeddingTextHardNegativeExamples(examples, tokenizer, embeddingTextTokenCache{}, true)
}

func tokenizeEmbeddingTextHardNegativeExamples(examples []EmbeddingTextHardNegativeExample, tokenizer *BPETokenizer, cache embeddingTextTokenCache, cloneOutput bool) ([]EmbeddingHardNegativeExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("text hard-negative dataset is empty")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if cache == nil {
		cache = embeddingTextTokenCache{}
	}
	out := make([]EmbeddingHardNegativeExample, 0, len(examples))
	for i, example := range examples {
		query, err := cache.encode(example.Query, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d query: %w", i, err)
		}
		positive, err := cache.encode(example.Positive, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d positive: %w", i, err)
		}
		negatives := make([][]int32, 0, len(example.Negatives))
		negativeMasks := make([][]int32, 0, len(example.Negatives))
		for j, rawNegative := range example.Negatives {
			negative, err := cache.encode(rawNegative, tokenizer)
			if err != nil {
				return nil, fmt.Errorf("example %d negative %d: %w", i, j, err)
			}
			if cloneOutput {
				negative = cloneTokenizedText(negative)
			}
			negatives = append(negatives, negative.tokens)
			negativeMasks = append(negativeMasks, negative.mask)
		}
		if cloneOutput {
			query = cloneTokenizedText(query)
			positive = cloneTokenizedText(positive)
		}
		out = append(out, EmbeddingHardNegativeExample{
			Source:         example.Source,
			QueryTokens:    query.tokens,
			PositiveTokens: positive.tokens,
			NegativeTokens: negatives,
			QueryMask:      query.mask,
			PositiveMask:   positive.mask,
			NegativeMasks:  negativeMasks,
			TeacherScores:  append([]float32(nil), example.TeacherScores...),
		})
	}
	return out, nil
}

func limitHardNegativeExamples(examples []EmbeddingHardNegativeExample, maxNegatives int) []EmbeddingHardNegativeExample {
	if maxNegatives <= 0 {
		return examples
	}
	out := make([]EmbeddingHardNegativeExample, len(examples))
	for i, example := range examples {
		out[i] = example
		if len(out[i].NegativeTokens) > maxNegatives {
			out[i].NegativeTokens = out[i].NegativeTokens[:maxNegatives]
		}
		if len(out[i].NegativeMasks) > maxNegatives {
			out[i].NegativeMasks = out[i].NegativeMasks[:maxNegatives]
		}
		if len(out[i].TeacherScores) > maxNegatives+1 {
			out[i].TeacherScores = out[i].TeacherScores[:maxNegatives+1]
		}
	}
	return out
}

func newEmbeddingHardNegativeRecord(example EmbeddingHardNegativeExample) (embeddingHardNegativeRecord, error) {
	if len(example.QueryTokens) == 0 {
		return embeddingHardNegativeRecord{}, fmt.Errorf("query_tokens are empty")
	}
	if len(example.PositiveTokens) == 0 {
		return embeddingHardNegativeRecord{}, fmt.Errorf("positive_tokens are empty")
	}
	if len(example.QueryMask) > 0 && len(example.QueryMask) != len(example.QueryTokens) {
		return embeddingHardNegativeRecord{}, fmt.Errorf("query_mask length %d does not match query_tokens length %d", len(example.QueryMask), len(example.QueryTokens))
	}
	if len(example.PositiveMask) > 0 && len(example.PositiveMask) != len(example.PositiveTokens) {
		return embeddingHardNegativeRecord{}, fmt.Errorf("positive_mask length %d does not match positive_tokens length %d", len(example.PositiveMask), len(example.PositiveTokens))
	}
	if len(example.NegativeTokens) == 0 {
		return embeddingHardNegativeRecord{}, fmt.Errorf("negative_tokens are empty")
	}
	if len(example.NegativeMasks) > 0 && len(example.NegativeMasks) != len(example.NegativeTokens) {
		return embeddingHardNegativeRecord{}, fmt.Errorf("negative_masks length %d does not match negative_tokens length %d", len(example.NegativeMasks), len(example.NegativeTokens))
	}
	for i, tokens := range example.NegativeTokens {
		if len(tokens) == 0 {
			return embeddingHardNegativeRecord{}, fmt.Errorf("negative_tokens[%d] are empty", i)
		}
		if len(example.NegativeMasks) > i && len(example.NegativeMasks[i]) > 0 && len(example.NegativeMasks[i]) != len(tokens) {
			return embeddingHardNegativeRecord{}, fmt.Errorf("negative_masks[%d] length %d does not match negative_tokens[%d] length %d", i, len(example.NegativeMasks[i]), i, len(tokens))
		}
	}
	if err := validateHardNegativeTeacherScores(example.TeacherScores, 1+len(example.NegativeTokens)); err != nil {
		return embeddingHardNegativeRecord{}, err
	}
	return embeddingHardNegativeRecord{
		Source:         example.Source,
		QueryTokens:    append([]int32(nil), example.QueryTokens...),
		PositiveTokens: append([]int32(nil), example.PositiveTokens...),
		NegativeTokens: cloneInt32Matrix(example.NegativeTokens),
		QueryMask:      append([]int32(nil), example.QueryMask...),
		PositiveMask:   append([]int32(nil), example.PositiveMask...),
		NegativeMasks:  cloneInt32Matrix(example.NegativeMasks),
		TeacherScores:  append([]float32(nil), example.TeacherScores...),
	}, nil
}

func (r embeddingHardNegativeRecord) example() (EmbeddingHardNegativeExample, error) {
	record, err := newEmbeddingHardNegativeRecord(EmbeddingHardNegativeExample{
		Source:         r.Source,
		QueryTokens:    r.QueryTokens,
		PositiveTokens: r.PositiveTokens,
		NegativeTokens: r.NegativeTokens,
		QueryMask:      r.QueryMask,
		PositiveMask:   r.PositiveMask,
		NegativeMasks:  r.NegativeMasks,
		TeacherScores:  r.TeacherScores,
	})
	if err != nil {
		return EmbeddingHardNegativeExample{}, err
	}
	return EmbeddingHardNegativeExample{
		Source:         record.Source,
		QueryTokens:    record.QueryTokens,
		PositiveTokens: record.PositiveTokens,
		NegativeTokens: record.NegativeTokens,
		QueryMask:      record.QueryMask,
		PositiveMask:   record.PositiveMask,
		NegativeMasks:  record.NegativeMasks,
		TeacherScores:  record.TeacherScores,
	}, nil
}

func newEmbeddingTextHardNegativeRecord(example EmbeddingTextHardNegativeExample) (embeddingTextHardNegativeRecord, error) {
	if strings.TrimSpace(example.Query) == "" {
		return embeddingTextHardNegativeRecord{}, fmt.Errorf("query is empty")
	}
	if strings.TrimSpace(example.Positive) == "" {
		return embeddingTextHardNegativeRecord{}, fmt.Errorf("positive is empty")
	}
	if len(example.Negatives) == 0 {
		return embeddingTextHardNegativeRecord{}, fmt.Errorf("negatives are empty")
	}
	for i, negative := range example.Negatives {
		if strings.TrimSpace(negative) == "" {
			return embeddingTextHardNegativeRecord{}, fmt.Errorf("negative %d is empty", i)
		}
	}
	if err := validateHardNegativeTeacherScores(example.TeacherScores, 1+len(example.Negatives)); err != nil {
		return embeddingTextHardNegativeRecord{}, err
	}
	return embeddingTextHardNegativeRecord{
		Source:        example.Source,
		Query:         example.Query,
		Positive:      example.Positive,
		Negatives:     append([]string(nil), example.Negatives...),
		TeacherScores: append([]float32(nil), example.TeacherScores...),
	}, nil
}

func (r embeddingTextHardNegativeRecord) example() (EmbeddingTextHardNegativeExample, error) {
	positive := firstNonEmpty(r.Positive, r.Document)
	record, err := newEmbeddingTextHardNegativeRecord(EmbeddingTextHardNegativeExample{
		Source:        r.Source,
		Query:         r.Query,
		Positive:      positive,
		Negatives:     r.Negatives,
		TeacherScores: r.TeacherScores,
	})
	if err != nil {
		return EmbeddingTextHardNegativeExample{}, err
	}
	return EmbeddingTextHardNegativeExample{
		Source:        record.Source,
		Query:         record.Query,
		Positive:      record.Positive,
		Negatives:     record.Negatives,
		TeacherScores: record.TeacherScores,
	}, nil
}

func validateHardNegativeTeacherScores(scores []float32, want int) error {
	if len(scores) == 0 {
		return nil
	}
	if len(scores) != want {
		return fmt.Errorf("teacher_scores length %d does not match candidate count %d", len(scores), want)
	}
	for i, score := range scores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			return fmt.Errorf("teacher_scores[%d] must be finite", i)
		}
	}
	return nil
}

func cloneInt32Matrix(in [][]int32) [][]int32 {
	if len(in) == 0 {
		return nil
	}
	out := make([][]int32, len(in))
	for i := range in {
		out[i] = append([]int32(nil), in[i]...)
	}
	return out
}
