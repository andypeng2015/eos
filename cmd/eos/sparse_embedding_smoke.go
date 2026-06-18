package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"m31labs.dev/eos/runtime/backend"
)

type sparseEmbeddingSmokeConfig struct {
	RunRoot          string  `json:"run_root"`
	RunDir           string  `json:"run_dir"`
	RuntimeSeqLen    int     `json:"runtime_seq_len"`
	QueryLen         int     `json:"query_len"`
	Dim              int     `json:"dim"`
	ValueDim         int     `json:"value_dim"`
	TopK             int     `json:"top_k"`
	RouteBlockSize   int     `json:"route_block_size"`
	RouteTopBlocks   int     `json:"route_top_blocks"`
	Bits             int     `json:"bits"`
	Seed             int64   `json:"seed"`
	PreflightKeyLens []int   `json:"preflight_key_lens"`
	MaxScoreFraction float64 `json:"max_score_fraction"`
	MaxTurboKVMiB    float64 `json:"max_turbo_kv_mib"`
	RequireSubq      bool    `json:"require_subquadratic"`
}

type sparseEmbeddingSmokeManifest struct {
	Schema                  string                     `json:"schema"`
	GeneratedAt             string                     `json:"generated_at"`
	Config                  sparseEmbeddingSmokeConfig `json:"config"`
	Preflight               sparseAttentionPlanReport  `json:"preflight"`
	Runtime                 sparseEmbeddingRuntime     `json:"runtime"`
	Embedding               sparseEmbeddingVector      `json:"embedding"`
	Artifacts               map[string]string          `json:"artifacts"`
	ThirtyTwoKPreflight     sparseEmbedding32KStatus   `json:"thirty_two_k_preflight"`
	ThirtyTwoKPreflightOnly bool                       `json:"32k_preflight_only"`
}

type sparseEmbedding32KStatus struct {
	Present bool   `json:"present"`
	Passed  bool   `json:"passed"`
	Status  string `json:"status"`
}

type sparseEmbeddingRuntime struct {
	Backend                   string         `json:"backend"`
	DeviceExecution           bool           `json:"device_execution"`
	Status                    string         `json:"status"`
	OutputShape               []int          `json:"output_shape"`
	AttentionMetadata         map[string]any `json:"attention_metadata"`
	DenseKVMaterialized       bool           `json:"dense_kv_materialized"`
	KVDecode                  string         `json:"kv_decode"`
	TurboQuantKeyCoordShape   []int          `json:"turboquant_key_coord_shape"`
	TurboQuantValueCoordShape []int          `json:"turboquant_value_coord_shape"`
	TurboQuantKeyNormShape    []int          `json:"turboquant_key_norm_shape"`
	TurboQuantValueNormShape  []int          `json:"turboquant_value_norm_shape"`
}

type sparseEmbeddingVector struct {
	Dimension int       `json:"dimension"`
	L2Norm    float64   `json:"l2_norm"`
	SHA256    string    `json:"sha256"`
	Vector    []float32 `json:"vector"`
	Preview   []float32 `json:"preview"`
}

func runSmokeSparseEmbeddingEncoder(args []string) error {
	cfg, err := parseSparseEmbeddingSmokeConfig(args)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(cfg.RunDir, "logs"), 0o755); err != nil {
		return err
	}
	logPath := filepath.Join(cfg.RunDir, "logs", "smoke.log")
	logLines := []string{
		"smoke=sparse_embedding_encoder",
		"created_utc=" + time.Now().UTC().Format(time.RFC3339),
		"runtime_seq_len=" + strconv.Itoa(cfg.RuntimeSeqLen),
		"preflight_key_lens=" + joinInts(cfg.PreflightKeyLens),
	}

	preflight := buildSparseEmbeddingPreflight(cfg)
	preflightPath := filepath.Join(cfg.RunDir, "sparse-attention-preflight.json")
	if err := writeSparseAttentionPlanReport(preflightPath, preflight); err != nil {
		return err
	}

	manifest, err := executeSparseEmbeddingSmoke(cfg, preflight)
	if err != nil {
		return err
	}
	manifest.Artifacts = map[string]string{
		"manifest_json":              filepath.Join(cfg.RunDir, "manifest.json"),
		"summary_tsv":                filepath.Join(cfg.RunDir, "summary.tsv"),
		"sparse_attention_preflight": preflightPath,
		"log":                        logPath,
	}
	if err := writeSparseEmbeddingManifest(manifest.Artifacts["manifest_json"], manifest); err != nil {
		return err
	}
	if err := writeSparseEmbeddingSummary(manifest.Artifacts["summary_tsv"], manifest); err != nil {
		return err
	}
	logLines = append(logLines,
		"status="+manifest.Runtime.Status,
		"embedding_dim="+strconv.Itoa(manifest.Embedding.Dimension),
		"embedding_sha256="+manifest.Embedding.SHA256,
		"preflight_gate="+passFail(preflight.Gate.Passed),
		"32k_preflight_status="+manifest.ThirtyTwoKPreflight.Status,
	)
	if err := os.WriteFile(logPath, []byte(strings.Join(logLines, "\n")+"\n"), 0o644); err != nil {
		return err
	}
	fmt.Printf("run_dir: %s\n", cfg.RunDir)
	fmt.Printf("manifest: %s\n", manifest.Artifacts["manifest_json"])
	fmt.Printf("summary_tsv: %s\n", manifest.Artifacts["summary_tsv"])
	fmt.Printf("preflight_json: %s\n", preflightPath)
	fmt.Printf("embedding_dim=%d runtime_seq_len=%d 32k_preflight=%s gate=%s\n",
		manifest.Embedding.Dimension, cfg.RuntimeSeqLen, manifest.ThirtyTwoKPreflight.Status, passFail(preflight.Gate.Passed))
	if !preflight.Gate.Passed {
		return fmt.Errorf("sparse embedding smoke preflight failed: %s", strings.Join(preflight.Gate.FailureReasons, "; "))
	}
	return nil
}

func parseSparseEmbeddingSmokeConfig(args []string) (sparseEmbeddingSmokeConfig, error) {
	defaultRunRoot := filepath.Join(".", "runs")
	stamp := time.Now().UTC().Format("20060102T150405Z")
	fs := flag.NewFlagSet("smoke-sparse-embedding-encoder", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	runRoot := fs.String("run-root", smokeEnvString("EOS_SPARSE_EMBED_SMOKE_RUN_ROOT", defaultRunRoot), "run artifact root")
	runDir := fs.String("run-dir", smokeEnvString("EOS_SPARSE_EMBED_SMOKE_RUN_DIR", ""), "exact run artifact directory")
	seqLen := fs.Int("seq-len", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_SEQ_LEN", 4096), "runtime sequence length for executable smoke")
	queryLen := fs.Int("query-len", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_QUERY_LEN", 8), "synthetic encoder query rows to pool")
	dim := fs.Int("dim", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_DIM", 64), "synthetic query/key dimension")
	valueDim := fs.Int("value-dim", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_VALUE_DIM", 0), "synthetic value and embedding dimension; 0 uses --dim")
	topK := fs.Int("top-k", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_TOP_K", 64), "sparse selected keys per query; 0 uses ceil(sqrt(seq_len))")
	routeBlockSize := fs.Int("route-block-size", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_ROUTE_BLOCK_SIZE", 0), "route block size; 0 uses ceil(sqrt(length))")
	routeTopBlocks := fs.Int("route-top-blocks", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_ROUTE_TOP_BLOCKS", 2), "route blocks selected per query")
	bits := fs.Int("bits", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_BITS", 4), "TurboQuant K/V bits: 2, 4, or 8")
	seed := fs.Int64("seed", smokeEnvInt64("EOS_SPARSE_EMBED_SMOKE_SEED", 5581486560434873699), "synthetic data and TurboQuant seed")
	preflightRaw := fs.String("preflight-key-lens", smokeEnvString("EOS_SPARSE_EMBED_SMOKE_PREFLIGHT_KEY_LENS", "4096,8192,16384,32768"), "comma-separated sparse-attention preflight key lengths")
	maxScoreFraction := fs.Float64("max-score-fraction", smokeEnvFloat("EOS_SPARSE_EMBED_SMOKE_MAX_SCORE_FRACTION", 0.2), "fail when preflight score fraction exceeds this value")
	maxTurboKVMiB := fs.Float64("max-turbo-kv-mib", smokeEnvFloat("EOS_SPARSE_EMBED_SMOKE_MAX_TURBO_KV_MIB", 512), "fail when preflight TurboQuant K/V MiB exceeds this value")
	requireSubq := fs.Bool("require-subquadratic", smokeEnvBool("EOS_SPARSE_EMBED_SMOKE_REQUIRE_SUBQUADRATIC", true), "require preflight rows to reduce score work versus dense scoring")
	if err := fs.Parse(args); err != nil {
		return sparseEmbeddingSmokeConfig{}, err
	}
	if fs.NArg() != 0 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("usage: eos smoke-sparse-embedding-encoder [flags]")
	}
	keyLens, err := parsePositiveIntCSV(*preflightRaw, "preflight-key-lens")
	if err != nil {
		return sparseEmbeddingSmokeConfig{}, err
	}
	if !containsInt(keyLens, 32768) {
		keyLens = append(keyLens, 32768)
	}
	cfg := sparseEmbeddingSmokeConfig{
		RunRoot:          *runRoot,
		RunDir:           *runDir,
		RuntimeSeqLen:    *seqLen,
		QueryLen:         *queryLen,
		Dim:              *dim,
		ValueDim:         *valueDim,
		TopK:             *topK,
		RouteBlockSize:   *routeBlockSize,
		RouteTopBlocks:   *routeTopBlocks,
		Bits:             *bits,
		Seed:             *seed,
		PreflightKeyLens: keyLens,
		MaxScoreFraction: *maxScoreFraction,
		MaxTurboKVMiB:    *maxTurboKVMiB,
		RequireSubq:      *requireSubq,
	}
	if cfg.ValueDim == 0 {
		cfg.ValueDim = cfg.Dim
	}
	if cfg.RunDir == "" {
		cfg.RunDir = filepath.Join(cfg.RunRoot, "eos-sparse-embedding-encoder-smoke-"+stamp)
	}
	if cfg.RuntimeSeqLen <= 0 || cfg.QueryLen <= 0 || cfg.Dim <= 0 || cfg.ValueDim <= 0 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("seq-len, query-len, dim, and value-dim must be positive")
	}
	if cfg.TopK < 0 || cfg.RouteBlockSize < 0 || cfg.RouteTopBlocks < 0 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("top-k, route-block-size, and route-top-blocks must be non-negative")
	}
	if cfg.RouteTopBlocks == 0 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("route-top-blocks must be positive for routed sparse embedding smoke")
	}
	if cfg.Bits != 2 && cfg.Bits != 4 && cfg.Bits != 8 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("bits must be 2, 4, or 8")
	}
	if cfg.MaxScoreFraction <= 0 || cfg.MaxTurboKVMiB < 0 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("max-score-fraction must be positive and max-turbo-kv-mib non-negative")
	}
	return cfg, nil
}

func buildSparseEmbeddingPreflight(cfg sparseEmbeddingSmokeConfig) sparseAttentionPlanReport {
	report := sparseAttentionPlanReport{
		Schema:     "manta.sparse_attention_plan.v1",
		CreatedUTC: time.Now().UTC().Format(time.RFC3339),
		Config: sparseAttentionPlanConfig{
			KeyLens:          append([]int(nil), cfg.PreflightKeyLens...),
			QueryLen:         cfg.QueryLen,
			QueryDim:         cfg.Dim,
			ValueDim:         cfg.ValueDim,
			TopK:             cfg.TopK,
			RouteBlockSize:   cfg.RouteBlockSize,
			RouteTopBlocks:   cfg.RouteTopBlocks,
			Bits:             cfg.Bits,
			Batches:          1,
			MaxScoreFraction: cfg.MaxScoreFraction,
			MaxTurboKVMiB:    cfg.MaxTurboKVMiB,
			RequireSubq:      cfg.RequireSubq,
		},
	}
	for _, keyLen := range cfg.PreflightKeyLens {
		blockSize := cfg.RouteBlockSize
		if blockSize == 0 {
			blockSize = int(math.Ceil(math.Sqrt(float64(keyLen))))
		}
		plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
			QueryLen:       cfg.QueryLen,
			KeyLen:         keyLen,
			QueryDim:       cfg.Dim,
			ValueDim:       cfg.ValueDim,
			TopK:           cfg.TopK,
			RouteBlockSize: blockSize,
			RouteTopBlocks: cfg.RouteTopBlocks,
		})
		kv := backend.PlanTurboQuantKVMemory(backend.TurboQuantKVMemoryPlanInput{
			Batches:            1,
			KeyLen:             keyLen,
			KeyDim:             cfg.Dim,
			ValueDim:           cfg.ValueDim,
			Bits:               cfg.Bits,
			DenseBytesPerValue: 2,
		})
		report.Rows = append(report.Rows, sparseAttentionPlanRow{
			KeyLen:                      keyLen,
			QueryLen:                    plan.QueryLen,
			QueryDim:                    plan.QueryDim,
			ValueDim:                    plan.ValueDim,
			TopK:                        plan.TopK,
			Routing:                     plan.Routing,
			RouteBlockSize:              plan.RouteBlockSize,
			RouteTopBlocks:              plan.RouteTopBlocks,
			RouteBlockCount:             plan.RouteBlockCount,
			SelectedRouteBlocks:         plan.SelectedRouteBlocks,
			SelectedKeyCount:            plan.SelectedKeyCount,
			CandidateKeyBudget:          plan.CandidateKeyBudget,
			DenseScoreCountPerQuery:     plan.DenseScoreCountPerQuery,
			EstimatedScoreCountPerQuery: plan.EstimatedScoreCountPerQuery,
			ScoreCountFraction:          plan.ScoreCountFraction,
			CandidateKeyFraction:        plan.CandidateKeyFraction,
			SubquadraticScorePlan:       plan.SubquadraticScorePlan,
			DenseTotalScoreCount:        int64(plan.QueryLen) * int64(plan.DenseScoreCountPerQuery),
			EstimatedTotalScoreCount:    int64(plan.QueryLen) * int64(plan.EstimatedScoreCountPerQuery),
			Bits:                        kv.Bits,
			DenseKVBytes:                kv.DenseKVBytes,
			TurboQuantKVBytes:           kv.TurboQuantKVBytes,
			TurboQuantKVMiB:             float64(kv.TurboQuantKVBytes) / (1024 * 1024),
			TurboQuantCompressionRatio:  kv.CompressionRatio,
		})
	}
	report.Gate = evaluateSparseAttentionPlanGate(report.Rows, cfg.MaxScoreFraction, cfg.MaxTurboKVMiB, cfg.RequireSubq)
	return report
}

func executeSparseEmbeddingSmoke(cfg sparseEmbeddingSmokeConfig, preflight sparseAttentionPlanReport) (sparseEmbeddingSmokeManifest, error) {
	blockSize := cfg.RouteBlockSize
	if blockSize == 0 {
		blockSize = int(math.Ceil(math.Sqrt(float64(cfg.RuntimeSeqLen))))
	}
	attrs := map[string]string{
		"bits":             strconv.Itoa(cfg.Bits),
		"seed":             strconv.FormatInt(cfg.Seed, 10),
		"top_k":            strconv.Itoa(cfg.TopK),
		"route_block_size": strconv.Itoa(blockSize),
		"route_top_blocks": strconv.Itoa(cfg.RouteTopBlocks),
	}
	query := backend.NewTensorF16([]int{cfg.QueryLen, cfg.Dim}, syntheticQuery(cfg.QueryLen, cfg.Dim, cfg.Seed))
	key := backend.NewTensorF16([]int{1, cfg.Dim, cfg.RuntimeSeqLen, 1}, syntheticNCHW(1, cfg.Dim, cfg.RuntimeSeqLen, cfg.Seed, 17))
	value := backend.NewTensorF16([]int{1, cfg.ValueDim, cfg.RuntimeSeqLen, 1}, syntheticNCHW(1, cfg.ValueDim, cfg.RuntimeSeqLen, cfg.Seed, 53))
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(key, attrs)
	if err != nil {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("turboquant encode key: %w", err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(value, attrs)
	if err != nil {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("turboquant encode value: %w", err)
	}
	out, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("turbo sparse attention reference: %w", err)
	}
	embedding := normalizeVector(meanPoolRows(out, cfg.QueryLen, cfg.ValueDim))
	plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen:       cfg.QueryLen,
		KeyLen:         cfg.RuntimeSeqLen,
		QueryDim:       cfg.Dim,
		ValueDim:       cfg.ValueDim,
		TopK:           cfg.TopK,
		RouteBlockSize: blockSize,
		RouteTopBlocks: cfg.RouteTopBlocks,
	})
	meta := plan.Metadata()
	meta["bits"] = cfg.Bits
	meta["seed"] = cfg.Seed
	meta["kv_decode"] = "host_reference_decode"
	meta["dense_kv_materialized"] = true
	meta["device_execution"] = false
	meta["runtime_seq_len"] = cfg.RuntimeSeqLen
	status32k := sparseEmbedding32KStatus{Status: "missing"}
	for _, row := range preflight.Rows {
		if row.KeyLen == 32768 {
			status32k.Present = true
			status32k.Passed = row.SubquadraticScorePlan && row.ScoreCountFraction <= cfg.MaxScoreFraction
			status32k.Status = passFail(status32k.Passed)
			break
		}
	}
	return sparseEmbeddingSmokeManifest{
		Schema:      "manta.sparse_embedding_encoder_smoke.v1",
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Config:      cfg,
		Preflight:   preflight,
		Runtime: sparseEmbeddingRuntime{
			Backend:                   "host_reference",
			DeviceExecution:           false,
			Status:                    "pass",
			OutputShape:               append([]int(nil), out.Shape...),
			AttentionMetadata:         meta,
			DenseKVMaterialized:       true,
			KVDecode:                  "host_reference_decode",
			TurboQuantKeyCoordShape:   append([]int(nil), keyCoords.Shape...),
			TurboQuantValueCoordShape: append([]int(nil), valueCoords.Shape...),
			TurboQuantKeyNormShape:    append([]int(nil), keyNorms.Shape...),
			TurboQuantValueNormShape:  append([]int(nil), valueNorms.Shape...),
		},
		Embedding: sparseEmbeddingVector{
			Dimension: len(embedding),
			L2Norm:    vectorNorm(embedding),
			SHA256:    vectorSHA256(embedding),
			Vector:    embedding,
			Preview:   previewVector(embedding, 16),
		},
		ThirtyTwoKPreflight:     status32k,
		ThirtyTwoKPreflightOnly: cfg.RuntimeSeqLen < 32768,
	}, nil
}

func syntheticQuery(rows, dim int, seed int64) []float32 {
	out := make([]float32, rows*dim)
	phase := float64(seed%997) / 997
	for r := 0; r < rows; r++ {
		for d := 0; d < dim; d++ {
			x := float64((r+1)*(d+3)) * 0.071
			out[r*dim+d] = float32(0.5*math.Sin(x+phase) + 0.5*math.Cos(x*0.37+phase))
		}
	}
	return out
}

func syntheticNCHW(batches, channels, seqLen int, seed int64, salt int) []float32 {
	out := make([]float32, batches*channels*seqLen)
	phase := float64((seed+int64(salt))%1543) / 1543
	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for t := 0; t < seqLen; t++ {
				x := float64((t+1)*(c+salt+1)) * 0.013
				out[(b*channels+c)*seqLen+t] = float32(0.6*math.Sin(x+phase) + 0.4*math.Cos(float64(t%127)*0.021+float64(c)*0.11))
			}
		}
	}
	return out
}

func meanPoolRows(t *backend.Tensor, rows, dim int) []float32 {
	out := make([]float32, dim)
	if t == nil || rows <= 0 || dim <= 0 {
		return out
	}
	for r := 0; r < rows; r++ {
		for d := 0; d < dim; d++ {
			out[d] += t.F32[r*dim+d]
		}
	}
	scale := float32(1.0 / float64(rows))
	for d := range out {
		out[d] *= scale
	}
	return out
}

func normalizeVector(v []float32) []float32 {
	norm := vectorNorm(v)
	if norm == 0 {
		return v
	}
	out := append([]float32(nil), v...)
	scale := float32(1 / norm)
	for i := range out {
		out[i] *= scale
	}
	return out
}

func vectorNorm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func vectorSHA256(v []float32) string {
	h := sha256.New()
	for _, x := range v {
		_, _ = fmt.Fprintf(h, "%.9g\n", x)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func previewVector(v []float32, n int) []float32 {
	if len(v) < n {
		n = len(v)
	}
	return append([]float32(nil), v[:n]...)
}

func writeSparseEmbeddingManifest(path string, manifest sparseEmbeddingSmokeManifest) error {
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func writeSparseEmbeddingSummary(path string, manifest sparseEmbeddingSmokeManifest) error {
	meta := manifest.Runtime.AttentionMetadata
	columns := []string{
		"status",
		"runtime_seq_len",
		"32k_preflight_only",
		"embedding_dim",
		"route_block_size",
		"route_top_blocks",
		"top_k",
		"bits",
		"seed",
		"selected_key_count",
		"candidate_key_budget",
		"score_count_fraction",
		"dense_kv_materialized",
		"kv_decode",
		"backend",
		"device_execution",
		"preflight_32768",
		"embedding_sha256",
	}
	row := []string{
		manifest.Runtime.Status,
		strconv.Itoa(manifest.Config.RuntimeSeqLen),
		strconv.FormatBool(manifest.ThirtyTwoKPreflightOnly),
		strconv.Itoa(manifest.Embedding.Dimension),
		fmt.Sprint(meta["route_block_size"]),
		fmt.Sprint(meta["route_top_blocks"]),
		fmt.Sprint(meta["top_k"]),
		strconv.Itoa(manifest.Config.Bits),
		strconv.FormatInt(manifest.Config.Seed, 10),
		fmt.Sprint(meta["selected_key_count"]),
		fmt.Sprint(meta["candidate_key_budget"]),
		fmt.Sprintf("%.6f", metaFloat64(meta["score_count_fraction"])),
		strconv.FormatBool(manifest.Runtime.DenseKVMaterialized),
		manifest.Runtime.KVDecode,
		manifest.Runtime.Backend,
		strconv.FormatBool(manifest.Runtime.DeviceExecution),
		manifest.ThirtyTwoKPreflight.Status,
		manifest.Embedding.SHA256,
	}
	return os.WriteFile(path, []byte(strings.Join(columns, "\t")+"\n"+strings.Join(row, "\t")+"\n"), 0o644)
}

func metaFloat64(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	default:
		return 0
	}
}

func smokeEnvString(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func smokeEnvInt(name string, fallback int) int {
	if value := os.Getenv(name); value != "" {
		if n, err := strconv.Atoi(value); err == nil {
			return n
		}
	}
	return fallback
}

func smokeEnvInt64(name string, fallback int64) int64 {
	if value := os.Getenv(name); value != "" {
		if n, err := strconv.ParseInt(value, 10, 64); err == nil {
			return n
		}
	}
	return fallback
}

func smokeEnvFloat(name string, fallback float64) float64 {
	if value := os.Getenv(name); value != "" {
		if n, err := strconv.ParseFloat(value, 64); err == nil {
			return n
		}
	}
	return fallback
}

func smokeEnvBool(name string, fallback bool) bool {
	if value := os.Getenv(name); value != "" {
		if b, err := strconv.ParseBool(value); err == nil {
			return b
		}
	}
	return fallback
}

func containsInt(values []int, want int) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func joinInts(values []int) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return strings.Join(parts, ",")
}
