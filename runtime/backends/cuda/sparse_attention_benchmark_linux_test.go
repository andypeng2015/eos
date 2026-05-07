//go:build linux && cgo

package cuda

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func BenchmarkCUDASparseAttentionSweep(b *testing.B) {
	rt, err := newDeviceRuntime()
	if err != nil {
		b.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		b.Skip("no cuda runtime available")
	}
	defer rt.close()

	cases, err := sparseAttentionBenchCasesFromEnv("MANTA_CUDA_SPARSE_BENCH_EXACT_KEY_LENS", "1024,4096")
	if err != nil {
		b.Fatal(err)
	}
	for _, tc := range cases {
		b.Run(tc.name("exact-f16"), func(b *testing.B) {
			query, key, value := syntheticSparseAttentionTensors(1, tc.KeyLen, tc.QueryDim, tc.ValueDim)
			attrs := map[string]string{"top_k": strconv.Itoa(tc.TopK)}
			step := mantaartifact.Step{Kind: mantaartifact.StepSparseAttention, Attributes: attrs}
			cfg, ok := planBuiltinSparseAttention(step, []*backend.Tensor{query, key, value})
			if !ok {
				b.Fatalf("sparse_attention benchmark config rejected: %+v", tc)
			}
			plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
				QueryLen: 1,
				KeyLen:   tc.KeyLen,
				QueryDim: tc.QueryDim,
				ValueDim: tc.ValueDim,
				TopK:     tc.TopK,
			})
			outputType := sparseAttentionBenchOutputType()
			if _, err := rt.runSparseAttentionStep([]*backend.Tensor{query, key, value}, outputType, cfg); err != nil {
				b.Fatalf("warm up sparse_attention: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := rt.runSparseAttentionStep([]*backend.Tensor{query, key, value}, outputType, cfg); err != nil {
					b.Fatalf("run sparse_attention: %v", err)
				}
			}
			b.StopTimer()
			reportSparseAttentionBenchPlan(b, plan)
		})
	}
}

func BenchmarkCUDATurboSparseAttentionSweep(b *testing.B) {
	rt, err := newDeviceRuntime()
	if err != nil {
		b.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		b.Skip("no cuda runtime available")
	}
	defer rt.close()

	cases, err := sparseAttentionBenchCasesFromEnv("MANTA_CUDA_SPARSE_BENCH_ROUTED_KEY_LENS", "1024,4096,16384")
	if err != nil {
		b.Fatal(err)
	}
	for _, tc := range cases {
		b.Run(tc.name("routed-tq"), func(b *testing.B) {
			routeBlockSize := int(math.Ceil(math.Sqrt(float64(tc.KeyLen))))
			query, keyNCHW, valueNCHW := syntheticTurboSparseAttentionTensors(1, tc.KeyLen, tc.QueryDim, tc.ValueDim)
			attrs := map[string]string{
				"bits":             strconv.Itoa(tc.Bits),
				"seed":             "20260507",
				"top_k":            strconv.Itoa(tc.TopK),
				"route_block_size": strconv.Itoa(routeBlockSize),
				"route_top_blocks": strconv.Itoa(tc.RouteTopBlocks),
			}
			keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
			if err != nil {
				b.Fatalf("encode key: %v", err)
			}
			valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
			if err != nil {
				b.Fatalf("encode value: %v", err)
			}
			inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
			step := mantaartifact.Step{Kind: mantaartifact.StepTurboSparseAttention, Attributes: attrs}
			cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
			if !ok {
				b.Fatalf("turbo_sparse_attention benchmark config rejected: %+v", tc)
			}
			plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
				QueryLen:       1,
				KeyLen:         tc.KeyLen,
				QueryDim:       tc.QueryDim,
				ValueDim:       tc.ValueDim,
				TopK:           tc.TopK,
				RouteBlockSize: routeBlockSize,
				RouteTopBlocks: tc.RouteTopBlocks,
			})
			kv := backend.PlanTurboQuantKVMemory(backend.TurboQuantKVMemoryPlanInput{
				Batches:  1,
				KeyLen:   tc.KeyLen,
				KeyDim:   tc.QueryDim,
				ValueDim: tc.ValueDim,
				Bits:     tc.Bits,
			})
			outputType := sparseAttentionBenchOutputType()
			if _, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg); err != nil {
				b.Fatalf("warm up turbo_sparse_attention: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg); err != nil {
					b.Fatalf("run turbo_sparse_attention: %v", err)
				}
			}
			b.StopTimer()
			reportSparseAttentionBenchPlan(b, plan)
			b.ReportMetric(float64(kv.TurboQuantKVBytes)/(1024*1024), "turbo_kv_mib")
			b.ReportMetric(kv.CompressionRatio, "kv_compression")
		})
	}
}

type sparseAttentionBenchCase struct {
	KeyLen         int
	QueryDim       int
	ValueDim       int
	TopK           int
	Bits           int
	RouteTopBlocks int
}

func TestSparseAttentionBenchCasesFromEnv(t *testing.T) {
	t.Setenv("MANTA_CUDA_SPARSE_BENCH_TEST_KEY_LENS", "256,1024")
	t.Setenv("MANTA_CUDA_SPARSE_BENCH_QUERY_DIM", "32")
	t.Setenv("MANTA_CUDA_SPARSE_BENCH_VALUE_DIM", "48")
	t.Setenv("MANTA_CUDA_SPARSE_BENCH_TOP_K", "0")
	t.Setenv("MANTA_CUDA_SPARSE_BENCH_BITS", "8")
	t.Setenv("MANTA_CUDA_SPARSE_BENCH_ROUTE_TOP_BLOCKS", "4")

	cases, err := sparseAttentionBenchCasesFromEnv("MANTA_CUDA_SPARSE_BENCH_TEST_KEY_LENS", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) != 2 {
		t.Fatalf("cases = %d, want 2", len(cases))
	}
	if cases[0].KeyLen != 256 || cases[0].QueryDim != 32 || cases[0].ValueDim != 48 || cases[0].TopK != 16 {
		t.Fatalf("first case = %+v", cases[0])
	}
	if cases[1].KeyLen != 1024 || cases[1].TopK != 32 || cases[1].Bits != 8 || cases[1].RouteTopBlocks != 4 {
		t.Fatalf("second case = %+v", cases[1])
	}
}

func (tc sparseAttentionBenchCase) name(prefix string) string {
	return fmt.Sprintf("%s/n%d/d%d/v%d/k%d", prefix, tc.KeyLen, tc.QueryDim, tc.ValueDim, tc.TopK)
}

func sparseAttentionBenchCasesFromEnv(keyLensEnv, fallbackKeyLens string) ([]sparseAttentionBenchCase, error) {
	keyLens, err := sparseAttentionBenchIntListEnv(keyLensEnv, fallbackKeyLens)
	if err != nil {
		return nil, err
	}
	queryDim, err := sparseAttentionBenchIntEnv("MANTA_CUDA_SPARSE_BENCH_QUERY_DIM", 64)
	if err != nil {
		return nil, err
	}
	valueDim, err := sparseAttentionBenchIntEnv("MANTA_CUDA_SPARSE_BENCH_VALUE_DIM", 64)
	if err != nil {
		return nil, err
	}
	topK, err := sparseAttentionBenchIntEnv("MANTA_CUDA_SPARSE_BENCH_TOP_K", 0)
	if err != nil {
		return nil, err
	}
	bits, err := sparseAttentionBenchIntEnv("MANTA_CUDA_SPARSE_BENCH_BITS", 4)
	if err != nil {
		return nil, err
	}
	if bits != 2 && bits != 4 && bits != 8 {
		return nil, fmt.Errorf("MANTA_CUDA_SPARSE_BENCH_BITS must be 2, 4, or 8")
	}
	routeTopBlocks, err := sparseAttentionBenchIntEnv("MANTA_CUDA_SPARSE_BENCH_ROUTE_TOP_BLOCKS", 2)
	if err != nil {
		return nil, err
	}
	if queryDim <= 0 || valueDim <= 0 {
		return nil, fmt.Errorf("MANTA_CUDA_SPARSE_BENCH_QUERY_DIM and MANTA_CUDA_SPARSE_BENCH_VALUE_DIM must be positive")
	}
	if routeTopBlocks <= 0 {
		return nil, fmt.Errorf("MANTA_CUDA_SPARSE_BENCH_ROUTE_TOP_BLOCKS must be positive")
	}
	cases := make([]sparseAttentionBenchCase, 0, len(keyLens))
	for _, keyLen := range keyLens {
		selectedTopK := topK
		if selectedTopK <= 0 {
			selectedTopK = int(math.Ceil(math.Sqrt(float64(keyLen))))
		}
		if selectedTopK < 1 {
			selectedTopK = 1
		}
		if selectedTopK > keyLen {
			selectedTopK = keyLen
		}
		if selectedTopK > cudaSparseAttentionMaxTopK {
			selectedTopK = cudaSparseAttentionMaxTopK
		}
		cases = append(cases, sparseAttentionBenchCase{
			KeyLen:         keyLen,
			QueryDim:       queryDim,
			ValueDim:       valueDim,
			TopK:           selectedTopK,
			Bits:           bits,
			RouteTopBlocks: routeTopBlocks,
		})
	}
	return cases, nil
}

func sparseAttentionBenchIntEnv(name string, fallback int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer: %w", name, err)
	}
	return value, nil
}

func sparseAttentionBenchIntListEnv(name, fallback string) ([]int, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		raw = fallback
	}
	parts := strings.Split(raw, ",")
	values := make([]int, 0, len(parts))
	for i, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("%s[%d] must be an integer: %w", name, i, err)
		}
		if value <= 0 {
			return nil, fmt.Errorf("%s[%d] must be positive", name, i)
		}
		values = append(values, value)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("%s must include at least one key length", name)
	}
	return values, nil
}

func sparseAttentionBenchOutputType() mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: "f16"},
	}
}

func reportSparseAttentionBenchPlan(b *testing.B, plan backend.SparseAttentionPlan) {
	b.ReportMetric(float64(plan.SelectedKeyCount), "selected_keys")
	b.ReportMetric(float64(plan.CandidateKeyBudget), "candidate_keys")
	b.ReportMetric(float64(plan.EstimatedScoreCountPerQuery), "scores_per_query")
	b.ReportMetric(plan.ScoreCountFraction, "score_fraction")
	if plan.SubquadraticScorePlan {
		b.ReportMetric(1, "subq_plan")
	} else {
		b.ReportMetric(0, "subq_plan")
	}
}

func syntheticSparseAttentionTensors(queryLen, keyLen, queryDim, valueDim int) (*backend.Tensor, *backend.Tensor, *backend.Tensor) {
	query := make([]float32, queryLen*queryDim)
	key := make([]float32, keyLen*queryDim)
	value := make([]float32, keyLen*valueDim)
	fillSparseAttentionQuery(query, queryLen, queryDim)
	fillSparseAttentionMatrix(key, keyLen, queryDim, 29)
	fillSparseAttentionMatrix(value, keyLen, valueDim, 37)
	return backend.NewTensorF16([]int{queryLen, queryDim}, query),
		backend.NewTensorF16([]int{keyLen, queryDim}, key),
		backend.NewTensorF16([]int{keyLen, valueDim}, value)
}

func syntheticTurboSparseAttentionTensors(queryLen, keyLen, queryDim, valueDim int) (*backend.Tensor, *backend.Tensor, *backend.Tensor) {
	query := make([]float32, queryLen*queryDim)
	key := make([]float32, queryDim*keyLen)
	value := make([]float32, valueDim*keyLen)
	fillSparseAttentionQuery(query, queryLen, queryDim)
	fillSparseAttentionNCHW(key, queryDim, keyLen, 31)
	fillSparseAttentionNCHW(value, valueDim, keyLen, 41)
	return backend.NewTensorF16([]int{queryLen, queryDim}, query),
		backend.NewTensorF16([]int{1, queryDim, keyLen, 1}, key),
		backend.NewTensorF16([]int{1, valueDim, keyLen, 1}, value)
}

func fillSparseAttentionQuery(data []float32, rows, width int) {
	for row := 0; row < rows; row++ {
		for col := 0; col < width; col++ {
			idx := row*width + col
			data[idx] = float32(((row+3)*(col+5))%23)/23 - 0.5
		}
	}
}

func fillSparseAttentionMatrix(data []float32, rows, width, mod int) {
	for row := 0; row < rows; row++ {
		for col := 0; col < width; col++ {
			idx := row*width + col
			data[idx] = float32(((row+1)*(col+7))%mod)/float32(mod) - 0.5
		}
	}
}

func fillSparseAttentionNCHW(data []float32, channels, height, mod int) {
	for channel := 0; channel < channels; channel++ {
		for pos := 0; pos < height; pos++ {
			idx := channel*height + pos
			data[idx] = float32(((channel+1)*(pos+11))%mod)/float32(mod) - 0.5
		}
	}
}
