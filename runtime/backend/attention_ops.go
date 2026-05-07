package backend

import (
	"fmt"
	"math"
	"sort"
)

type sparseAttentionCandidate struct {
	index int
	score float32
}

// SparseAttentionPlan describes the per-query work budget for exact or routed
// sparse attention. It is used for host/CUDA metadata, not for kernel dispatch.
type SparseAttentionPlan struct {
	QueryLen                    int
	KeyLen                      int
	QueryDim                    int
	ValueDim                    int
	TopK                        int
	Routing                     string
	RouteBlockSize              int
	RouteTopBlocks              int
	RouteBlockCount             int
	SelectedRouteBlocks         int
	SelectedKeyCount            int
	CandidateKeyBudget          int
	DenseScoreCountPerQuery     int
	RoutedAnchorScoresPerQuery  int
	EstimatedScoreCountPerQuery int
	CandidateKeyFraction        float64
	ScoreCountFraction          float64
	SubquadraticScorePlan       bool
}

// SparseAttentionPlanInput is the shape/config input to PlanSparseAttention.
type SparseAttentionPlanInput struct {
	QueryLen       int
	KeyLen         int
	QueryDim       int
	ValueDim       int
	TopK           int
	RouteBlockSize int
	RouteTopBlocks int
}

// TurboQuantKVMemoryPlanInput describes logical dense-vs-TurboQuant KV-cache
// storage for one attention layer.
type TurboQuantKVMemoryPlanInput struct {
	Batches            int
	KeyLen             int
	KeyDim             int
	ValueDim           int
	Bits               int
	DenseBytesPerValue int
	NormBytesPerVector int
}

// TurboQuantKVMemoryPlan is a logical memory estimate. Runtime tensors may use
// host-friendly layouts; this reports the compact K/V target budget.
type TurboQuantKVMemoryPlan struct {
	Batches            int
	KeyLen             int
	KeyDim             int
	ValueDim           int
	Bits               int
	DenseBytesPerValue int
	NormBytesPerVector int
	DenseKVBytes       int64
	TurboQuantKVBytes  int64
	KeyCoordBytes      int64
	ValueCoordBytes    int64
	NormBytes          int64
	CompressionRatio   float64
}

// PlanSparseAttention normalizes sparse attention routing knobs and derives
// auditable budget metadata for one query row.
func PlanSparseAttention(in SparseAttentionPlanInput) SparseAttentionPlan {
	topK := in.TopK
	if in.KeyLen > 0 {
		if topK <= 0 {
			topK = int(math.Ceil(math.Sqrt(float64(in.KeyLen))))
		}
		if topK < 1 {
			topK = 1
		}
		if topK > in.KeyLen {
			topK = in.KeyLen
		}
	}
	plan := SparseAttentionPlan{
		QueryLen:                    in.QueryLen,
		KeyLen:                      in.KeyLen,
		QueryDim:                    in.QueryDim,
		ValueDim:                    in.ValueDim,
		TopK:                        topK,
		Routing:                     "exact",
		SelectedKeyCount:            topK,
		CandidateKeyBudget:          in.KeyLen,
		DenseScoreCountPerQuery:     in.KeyLen,
		EstimatedScoreCountPerQuery: in.KeyLen,
		CandidateKeyFraction:        1,
		ScoreCountFraction:          1,
	}
	if in.KeyLen <= 0 {
		plan.CandidateKeyBudget = 0
		plan.DenseScoreCountPerQuery = 0
		plan.EstimatedScoreCountPerQuery = 0
		plan.CandidateKeyFraction = 0
		plan.ScoreCountFraction = 0
		return plan
	}
	blockSize, topBlocks := normalizeSparseAttentionRoute(in.KeyLen, in.RouteBlockSize, in.RouteTopBlocks)
	if blockSize <= 0 || topBlocks <= 0 {
		return plan
	}
	blockCount := (in.KeyLen + blockSize - 1) / blockSize
	candidateBudget := topBlocks * blockSize
	if candidateBudget > in.KeyLen {
		candidateBudget = in.KeyLen
	}
	estimatedScores := blockCount + candidateBudget
	plan.Routing = "block_anchor"
	plan.RouteBlockSize = blockSize
	plan.RouteTopBlocks = topBlocks
	plan.RouteBlockCount = blockCount
	plan.SelectedRouteBlocks = topBlocks
	plan.SelectedKeyCount = topK
	if plan.SelectedKeyCount > candidateBudget {
		plan.SelectedKeyCount = candidateBudget
	}
	plan.CandidateKeyBudget = candidateBudget
	plan.RoutedAnchorScoresPerQuery = blockCount
	plan.EstimatedScoreCountPerQuery = estimatedScores
	plan.CandidateKeyFraction = float64(candidateBudget) / float64(in.KeyLen)
	plan.ScoreCountFraction = float64(estimatedScores) / float64(in.KeyLen)
	plan.SubquadraticScorePlan = estimatedScores < in.KeyLen
	return plan
}

// PlanTurboQuantKVMemory estimates logical dense and TurboQuant-compressed K/V
// cache footprint.
func PlanTurboQuantKVMemory(in TurboQuantKVMemoryPlanInput) TurboQuantKVMemoryPlan {
	if in.Batches <= 0 {
		in.Batches = 1
	}
	if in.Bits <= 0 {
		in.Bits = 4
	}
	if in.DenseBytesPerValue <= 0 {
		in.DenseBytesPerValue = 2
	}
	if in.NormBytesPerVector <= 0 {
		in.NormBytesPerVector = 4
	}
	plan := TurboQuantKVMemoryPlan{
		Batches:            in.Batches,
		KeyLen:             in.KeyLen,
		KeyDim:             in.KeyDim,
		ValueDim:           in.ValueDim,
		Bits:               in.Bits,
		DenseBytesPerValue: in.DenseBytesPerValue,
		NormBytesPerVector: in.NormBytesPerVector,
	}
	if in.KeyLen <= 0 || in.KeyDim <= 0 || in.ValueDim <= 0 {
		return plan
	}
	batches := int64(in.Batches)
	keyLen := int64(in.KeyLen)
	keyDim := int64(in.KeyDim)
	valueDim := int64(in.ValueDim)
	bits := int64(in.Bits)
	plan.DenseKVBytes = batches * keyLen * (keyDim + valueDim) * int64(in.DenseBytesPerValue)
	plan.KeyCoordBytes = ceilDivInt64(batches*keyLen*keyDim*bits, 8)
	plan.ValueCoordBytes = ceilDivInt64(batches*keyLen*valueDim*bits, 8)
	plan.NormBytes = batches * keyLen * 2 * int64(in.NormBytesPerVector)
	plan.TurboQuantKVBytes = plan.KeyCoordBytes + plan.ValueCoordBytes + plan.NormBytes
	if plan.TurboQuantKVBytes > 0 {
		plan.CompressionRatio = float64(plan.DenseKVBytes) / float64(plan.TurboQuantKVBytes)
	}
	return plan
}

func ceilDivInt64(n, d int64) int64 {
	if n <= 0 || d <= 0 {
		return 0
	}
	return (n + d - 1) / d
}

// Metadata returns stable, machine-readable budget fields for traces and
// manifests.
func (p SparseAttentionPlan) Metadata() map[string]any {
	return map[string]any{
		"query_len":                       p.QueryLen,
		"key_len":                         p.KeyLen,
		"query_dim":                       p.QueryDim,
		"value_dim":                       p.ValueDim,
		"top_k":                           p.TopK,
		"routing":                         p.Routing,
		"route_block_size":                p.RouteBlockSize,
		"route_top_blocks":                p.RouteTopBlocks,
		"route_block_count":               p.RouteBlockCount,
		"selected_route_blocks":           p.SelectedRouteBlocks,
		"selected_key_count":              p.SelectedKeyCount,
		"candidate_key_budget":            p.CandidateKeyBudget,
		"dense_score_count_per_query":     p.DenseScoreCountPerQuery,
		"routed_anchor_scores_per_query":  p.RoutedAnchorScoresPerQuery,
		"estimated_score_count_per_query": p.EstimatedScoreCountPerQuery,
		"candidate_key_fraction":          p.CandidateKeyFraction,
		"score_count_fraction":            p.ScoreCountFraction,
		"subquadratic_score_plan":         p.SubquadraticScorePlan,
	}
}

func sparseAttentionTensor(query, key, value *Tensor, attrs map[string]string) (*Tensor, error) {
	if query == nil || key == nil || value == nil {
		return nil, fmt.Errorf("sparse_attention expects query, key, and value")
	}
	if len(query.Shape) == 2 && len(key.Shape) == 2 && len(value.Shape) == 2 {
		qLen, dim := query.Shape[0], query.Shape[1]
		kLen, keyDim := key.Shape[0], key.Shape[1]
		vLen, valueDim := value.Shape[0], value.Shape[1]
		if dim != keyDim {
			return nil, fmt.Errorf("sparse_attention query dim %d does not match key dim %d", dim, keyDim)
		}
		if kLen != vLen {
			return nil, fmt.Errorf("sparse_attention key length %d does not match value length %d", kLen, vLen)
		}
		out := tensorForDType(value.DType, []int{qLen, valueDim}, qLen*valueDim)
		for q := 0; q < qLen; q++ {
			selected := selectSparseAttentionKeysWithRouting(kLen, sparseAttentionBudget(attrs, kLen), attrs, func(k int) float32 {
				sum := float32(0)
				for d := 0; d < dim; d++ {
					sum += query.F32[q*dim+d] * key.F32[k*dim+d]
				}
				return sum
			})
			writeSparseAttentionValue(out.F32[q*valueDim:(q+1)*valueDim], selected, valueDim, func(k, d int) float32 {
				return value.F32[k*valueDim+d]
			})
		}
		return out, nil
	}
	if len(query.Shape) == 3 && len(key.Shape) == 3 && len(value.Shape) == 3 {
		batches, qLen, dim := query.Shape[0], query.Shape[1], query.Shape[2]
		keyBatches, kLen, keyDim := key.Shape[0], key.Shape[1], key.Shape[2]
		valueBatches, vLen, valueDim := value.Shape[0], value.Shape[1], value.Shape[2]
		if batches != keyBatches || batches != valueBatches {
			return nil, fmt.Errorf("sparse_attention batch mismatch: query %d key %d value %d", batches, keyBatches, valueBatches)
		}
		if dim != keyDim {
			return nil, fmt.Errorf("sparse_attention query dim %d does not match key dim %d", dim, keyDim)
		}
		if kLen != vLen {
			return nil, fmt.Errorf("sparse_attention key length %d does not match value length %d", kLen, vLen)
		}
		out := tensorForDType(value.DType, []int{batches, qLen, valueDim}, batches*qLen*valueDim)
		budget := sparseAttentionBudget(attrs, kLen)
		for b := 0; b < batches; b++ {
			for q := 0; q < qLen; q++ {
				selected := selectSparseAttentionKeysWithRouting(kLen, budget, attrs, func(k int) float32 {
					sum := float32(0)
					for d := 0; d < dim; d++ {
						sum += query.F32[(b*qLen+q)*dim+d] * key.F32[(b*kLen+k)*dim+d]
					}
					return sum
				})
				row := out.F32[((b*qLen + q) * valueDim):((b*qLen + q + 1) * valueDim)]
				writeSparseAttentionValue(row, selected, valueDim, func(k, d int) float32 {
					return value.F32[(b*vLen+k)*valueDim+d]
				})
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("sparse_attention expects rank-2 or rank-3 query/key/value tensors")
}

// SparseAttentionReference runs top-k sparse attention over dense Q/K/V tensors.
func SparseAttentionReference(query, key, value *Tensor, attrs map[string]string) (*Tensor, error) {
	return sparseAttentionTensor(query, key, value, attrs)
}

func turboSparseAttentionTensor(query, keyCoords, keyNorms, valueCoords, valueNorms *Tensor, attrs map[string]string) (*Tensor, error) {
	if query == nil {
		return nil, fmt.Errorf("turbo_sparse_attention expects query")
	}
	keyDense, err := turboQuantDecodeTensor(keyCoords, keyNorms, attrs)
	if err != nil {
		return nil, fmt.Errorf("turbo_sparse_attention key decode: %w", err)
	}
	valueDense, err := turboQuantDecodeTensor(valueCoords, valueNorms, attrs)
	if err != nil {
		return nil, fmt.Errorf("turbo_sparse_attention value decode: %w", err)
	}
	keySeq, err := nchwToAttentionSequence(keyDense, len(query.Shape))
	if err != nil {
		return nil, fmt.Errorf("turbo_sparse_attention key layout: %w", err)
	}
	valueSeq, err := nchwToAttentionSequence(valueDense, len(query.Shape))
	if err != nil {
		return nil, fmt.Errorf("turbo_sparse_attention value layout: %w", err)
	}
	return sparseAttentionTensor(query, keySeq, valueSeq, attrs)
}

// TurboSparseAttentionReference runs sparse attention with TurboQuant-compressed
// K/V tensors. The compressed K/V layout is NCHW, interpreted as B,D,T,1.
func TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms *Tensor, attrs map[string]string) (*Tensor, error) {
	return turboSparseAttentionTensor(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
}

func sparseAttentionMetadata(op string, query, key, value *Tensor, attrs map[string]string) map[string]any {
	meta := hostReferenceMetadata(op)
	if plan, ok := sparseAttentionPlanFromDenseTensors(query, key, value, attrs); ok {
		for key, value := range plan.Metadata() {
			meta[key] = value
		}
	}
	return meta
}

func turboSparseAttentionMetadata(op string, query, keyCoords, valueCoords *Tensor, attrs map[string]string) map[string]any {
	meta := hostReferenceMetadata(op)
	if plan, ok := sparseAttentionPlanFromCompressedTensors(query, keyCoords, valueCoords, attrs); ok {
		for key, value := range plan.Metadata() {
			meta[key] = value
		}
	}
	meta["kv_decode"] = "host_reference_decode"
	meta["dense_kv_materialized"] = true
	return meta
}

func sparseAttentionPlanFromDenseTensors(query, key, value *Tensor, attrs map[string]string) (SparseAttentionPlan, bool) {
	if query == nil || key == nil || value == nil {
		return SparseAttentionPlan{}, false
	}
	var queryLen, keyLen, queryDim, valueDim int
	if len(query.Shape) == 2 && len(key.Shape) == 2 && len(value.Shape) == 2 {
		queryLen, queryDim = query.Shape[0], query.Shape[1]
		keyLen = key.Shape[0]
		valueDim = value.Shape[1]
	} else if len(query.Shape) == 3 && len(key.Shape) == 3 && len(value.Shape) == 3 {
		queryLen, queryDim = query.Shape[1], query.Shape[2]
		keyLen = key.Shape[1]
		valueDim = value.Shape[2]
	} else {
		return SparseAttentionPlan{}, false
	}
	return PlanSparseAttention(SparseAttentionPlanInput{
		QueryLen:       queryLen,
		KeyLen:         keyLen,
		QueryDim:       queryDim,
		ValueDim:       valueDim,
		TopK:           attrInt(attrs, "top_k", 0),
		RouteBlockSize: attrInt(attrs, "route_block_size", 0),
		RouteTopBlocks: attrInt(attrs, "route_top_blocks", 0),
	}), true
}

func sparseAttentionPlanFromCompressedTensors(query, keyCoords, valueCoords *Tensor, attrs map[string]string) (SparseAttentionPlan, bool) {
	if query == nil || keyCoords == nil || valueCoords == nil || len(keyCoords.Shape) != 4 || len(valueCoords.Shape) != 4 {
		return SparseAttentionPlan{}, false
	}
	keyLen := keyCoords.Shape[2]
	queryDim := keyCoords.Shape[1]
	valueDim := valueCoords.Shape[1]
	var queryLen int
	switch len(query.Shape) {
	case 2:
		queryLen = query.Shape[0]
	case 3:
		queryLen = query.Shape[1]
	default:
		return SparseAttentionPlan{}, false
	}
	return PlanSparseAttention(SparseAttentionPlanInput{
		QueryLen:       queryLen,
		KeyLen:         keyLen,
		QueryDim:       queryDim,
		ValueDim:       valueDim,
		TopK:           attrInt(attrs, "top_k", 0),
		RouteBlockSize: attrInt(attrs, "route_block_size", 0),
		RouteTopBlocks: attrInt(attrs, "route_top_blocks", 0),
	}), true
}

func nchwToAttentionSequence(input *Tensor, queryRank int) (*Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("nil tensor")
	}
	if len(input.Shape) != 4 {
		return nil, fmt.Errorf("expected NCHW tensor, got shape %v", input.Shape)
	}
	batches, channels, seqLen, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	if width != 1 {
		return nil, fmt.Errorf("expected width 1 for attention sequence layout, got %d", width)
	}
	switch queryRank {
	case 2:
		if batches != 1 {
			return nil, fmt.Errorf("rank-2 query expects compressed batch 1, got %d", batches)
		}
		out := tensorForDType(input.DType, []int{seqLen, channels}, seqLen*channels)
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				out.F32[t*channels+c] = input.F32[offset4(input.Shape, 0, c, t, 0)]
			}
		}
		return out, nil
	case 3:
		out := tensorForDType(input.DType, []int{batches, seqLen, channels}, batches*seqLen*channels)
		for b := 0; b < batches; b++ {
			for t := 0; t < seqLen; t++ {
				for c := 0; c < channels; c++ {
					out.F32[(b*seqLen+t)*channels+c] = input.F32[offset4(input.Shape, b, c, t, 0)]
				}
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("query rank must be 2 or 3, got %d", queryRank)
	}
}

func sparseAttentionBudget(attrs map[string]string, keyLen int) int {
	if keyLen <= 0 {
		return 0
	}
	budget := attrInt(attrs, "top_k", 0)
	if budget <= 0 {
		budget = int(math.Ceil(math.Sqrt(float64(keyLen))))
	}
	if budget < 1 {
		budget = 1
	}
	if budget > keyLen {
		budget = keyLen
	}
	return budget
}

func selectSparseAttentionKeys(keyLen, budget int, scoreAt func(int) float32) []sparseAttentionCandidate {
	candidates := make([]sparseAttentionCandidate, 0, keyLen)
	for k := 0; k < keyLen; k++ {
		candidates = append(candidates, sparseAttentionCandidate{index: k, score: scoreAt(k)})
	}
	return selectTopSparseAttentionCandidates(candidates, budget)
}

func selectSparseAttentionKeysWithRouting(keyLen, budget int, attrs map[string]string, scoreAt func(int) float32) []sparseAttentionCandidate {
	routeBlockSize, routeTopBlocks := sparseAttentionRouteConfig(attrs, keyLen)
	if routeBlockSize <= 0 || routeTopBlocks <= 0 {
		return selectSparseAttentionKeys(keyLen, budget, scoreAt)
	}
	blockCount := (keyLen + routeBlockSize - 1) / routeBlockSize
	blocks := make([]sparseAttentionCandidate, 0, blockCount)
	for block := 0; block < blockCount; block++ {
		start := block * routeBlockSize
		end := start + routeBlockSize
		if end > keyLen {
			end = keyLen
		}
		anchor := start + (end-start)/2
		blocks = append(blocks, sparseAttentionCandidate{index: block, score: scoreAt(anchor)})
	}
	blocks = selectTopSparseAttentionCandidates(blocks, routeTopBlocks)
	candidates := make([]sparseAttentionCandidate, 0, len(blocks)*routeBlockSize)
	for _, block := range blocks {
		start := block.index * routeBlockSize
		end := start + routeBlockSize
		if end > keyLen {
			end = keyLen
		}
		for k := start; k < end; k++ {
			candidates = append(candidates, sparseAttentionCandidate{index: k, score: scoreAt(k)})
		}
	}
	return selectTopSparseAttentionCandidates(candidates, budget)
}

func sparseAttentionRouteConfig(attrs map[string]string, keyLen int) (int, int) {
	return normalizeSparseAttentionRoute(keyLen, attrInt(attrs, "route_block_size", 0), attrInt(attrs, "route_top_blocks", 0))
}

func normalizeSparseAttentionRoute(keyLen, blockSize, topBlocks int) (int, int) {
	if keyLen <= 0 || blockSize <= 0 || topBlocks <= 0 {
		return 0, 0
	}
	if blockSize > keyLen {
		blockSize = keyLen
	}
	blockCount := (keyLen + blockSize - 1) / blockSize
	if topBlocks > blockCount {
		topBlocks = blockCount
	}
	return blockSize, topBlocks
}

func selectTopSparseAttentionCandidates(candidates []sparseAttentionCandidate, budget int) []sparseAttentionCandidate {
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].score == candidates[j].score {
			return candidates[i].index < candidates[j].index
		}
		return candidates[i].score > candidates[j].score
	})
	if budget < len(candidates) {
		candidates = candidates[:budget]
	}
	return candidates
}

func writeSparseAttentionValue(out []float32, selected []sparseAttentionCandidate, valueDim int, valueAt func(int, int) float32) {
	if len(selected) == 0 {
		return
	}
	maxScore := selected[0].score
	for _, candidate := range selected[1:] {
		if candidate.score > maxScore {
			maxScore = candidate.score
		}
	}
	weights := make([]float64, len(selected))
	denom := float64(0)
	for i, candidate := range selected {
		weight := math.Exp(float64(candidate.score - maxScore))
		weights[i] = weight
		denom += weight
	}
	if denom == 0 || math.IsNaN(denom) {
		return
	}
	for i, candidate := range selected {
		scale := float32(weights[i] / denom)
		for d := 0; d < valueDim; d++ {
			out[d] += scale * valueAt(candidate.index, d)
		}
	}
}
