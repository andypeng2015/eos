package main

import (
	"context"
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

	"m31labs.dev/eos/compiler"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
)

type sparseEmbeddingSmokeConfig struct {
	RunRoot          string  `json:"run_root"`
	RunDir           string  `json:"run_dir"`
	Backend          string  `json:"backend"`
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
	MaxParitySeqLen  int     `json:"max_parity_seq_len"`
	RequireSubq      bool    `json:"require_subquadratic"`
}

type sparseEmbeddingSmokeManifest struct {
	Schema                  string                     `json:"schema"`
	GeneratedAt             string                     `json:"generated_at"`
	Config                  sparseEmbeddingSmokeConfig `json:"config"`
	Preflight               sparseAttentionPlanReport  `json:"preflight"`
	Runtime                 sparseEmbeddingRuntime     `json:"runtime"`
	Parity                  sparseEmbeddingParity      `json:"parity"`
	Embedding               sparseEmbeddingVector      `json:"embedding"`
	Artifacts               map[string]string          `json:"artifacts"`
	ThirtyTwoKPreflight     sparseEmbedding32KStatus   `json:"thirty_two_k_preflight"`
	ThirtyTwoKPreflightOnly bool                       `json:"32k_preflight_only"`
}

type sparseEmbeddingSmokeScorecard struct {
	Schema      string                             `json:"schema"`
	GeneratedAt string                             `json:"generated_at"`
	Rows        []sparseEmbeddingSmokeScorecardRow `json:"rows"`
}

type sparseEmbeddingSmokeScorecardRow struct {
	Category                            string   `json:"category"`
	Dataset                             string   `json:"dataset"`
	Baseline                            string   `json:"baseline"`
	Status                              string   `json:"status"`
	Method                              string   `json:"method"`
	EvidenceLevel                       string   `json:"evidence_level"`
	QualityClaim                        bool     `json:"quality_claim"`
	ClaimBoundary                       string   `json:"claim_boundary"`
	SourceManifest                      string   `json:"source_manifest"`
	SourceArtifacts                     []string `json:"source_artifacts"`
	RuntimeSeqLen                       int      `json:"runtime_seq_len"`
	ThirtyTwoKPreflightStatus           string   `json:"thirty_two_k_preflight_status"`
	ThirtyTwoKPreflightOnly             bool     `json:"32k_preflight_only"`
	Preflight32768ScoreFraction         float64  `json:"preflight_32768_score_fraction"`
	Preflight32768Subquadratic          bool     `json:"preflight_32768_subquadratic"`
	PreflightGateStatus                 string   `json:"preflight_gate_status"`
	RequestedBackend                    string   `json:"requested_backend"`
	ActualBackend                       string   `json:"actual_backend"`
	RuntimeBackend                      string   `json:"runtime_backend"`
	CUDAAvailable                       bool     `json:"cuda_available"`
	CUDAEvidenceStatus                  string   `json:"cuda_evidence_status"`
	FallbackReason                      string   `json:"fallback_reason,omitempty"`
	DeviceExecution                     bool     `json:"device_execution"`
	DenseKVMaterialized                 bool     `json:"dense_kv_materialized"`
	KVDecode                            string   `json:"kv_decode"`
	TraceVariant                        string   `json:"trace_variant,omitempty"`
	Bits                                int      `json:"bits"`
	QuantizerSeed                       int64    `json:"quantizer_seed"`
	EmbeddingDim                        int      `json:"embedding_dim"`
	EmbeddingSHA256                     string   `json:"embedding_sha256"`
	ParityStatus                        string   `json:"parity_status"`
	ParityBackendVsHostPassed           bool     `json:"parity_backend_vs_host_passed"`
	ParityBackendVsHostMaxAbsError      float64  `json:"parity_backend_vs_host_max_abs_error"`
	ParityBackendVsHostMSE              float64  `json:"parity_backend_vs_host_mse"`
	ParityBackendVsHostCosineSimilarity float64  `json:"parity_backend_vs_host_cosine_similarity"`
	ParityDiagnosticsStatus             string   `json:"parity_diagnostics_status"`
}

type sparseEmbedding32KStatus struct {
	Present bool   `json:"present"`
	Passed  bool   `json:"passed"`
	Status  string `json:"status"`
}

type sparseEmbeddingRuntime struct {
	Backend                   string         `json:"backend"`
	RequestedBackend          string         `json:"requested_backend"`
	ActualBackend             string         `json:"actual_backend"`
	CUDAAvailable             bool           `json:"cuda_available"`
	CUDAEvidenceStatus        string         `json:"cuda_evidence_status"`
	FallbackReason            string         `json:"fallback_reason,omitempty"`
	TraceVariant              string         `json:"trace_variant,omitempty"`
	TraceEntry                string         `json:"trace_entry,omitempty"`
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

type sparseEmbeddingParity struct {
	Status                    string                       `json:"status"`
	StrictGate                bool                         `json:"strict_gate"`
	MaxAbsErrorTolerance      float64                      `json:"max_abs_error_tolerance"`
	MSETolerance              float64                      `json:"mse_tolerance"`
	CosineSimilarityTolerance float64                      `json:"cosine_similarity_tolerance"`
	BackendVsHostTurboQuant   sparseEmbeddingParityCompare `json:"backend_vs_host_turboquant"`
	Diagnostics               sparseEmbeddingParityDiag    `json:"diagnostics"`
}

type sparseEmbeddingParityDiag struct {
	Status                         string                       `json:"status"`
	SkippedReason                  string                       `json:"skipped_reason,omitempty"`
	DenseFullVsExactSparse         sparseEmbeddingParityCompare `json:"dense_full_vs_exact_sparse"`
	ExactSparseVsRoutedSparse      sparseEmbeddingParityCompare `json:"exact_sparse_vs_routed_sparse"`
	RoutedDenseVsTurboQuantRouted  sparseEmbeddingParityCompare `json:"routed_dense_vs_turboquant_routed"`
	DenseFullSHA256                string                       `json:"dense_full_sha256,omitempty"`
	ExactSparseSHA256              string                       `json:"exact_sparse_sha256,omitempty"`
	RoutedSparseDenseSHA256        string                       `json:"routed_sparse_dense_sha256,omitempty"`
	TurboQuantRoutedHostSHA256     string                       `json:"turboquant_routed_host_sha256,omitempty"`
	DenseDiagnosticRuntimeSeqLen   int                          `json:"dense_diagnostic_runtime_seq_len,omitempty"`
	DenseDiagnosticMaxParitySeqLen int                          `json:"dense_diagnostic_max_parity_seq_len,omitempty"`
}

type sparseEmbeddingParityCompare struct {
	Status           string  `json:"status"`
	Passed           bool    `json:"passed"`
	MaxAbsError      float64 `json:"max_abs_error"`
	MSE              float64 `json:"mse"`
	CosineSimilarity float64 `json:"cosine_similarity"`
	ActualSHA256     string  `json:"actual_sha256"`
	ExpectedSHA256   string  `json:"expected_sha256"`
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
		"requested_backend=" + cfg.Backend,
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
		"scorecard_json":             filepath.Join(cfg.RunDir, "scorecard.json"),
		"scorecard_tsv":              filepath.Join(cfg.RunDir, "scorecard.tsv"),
		"sparse_attention_preflight": preflightPath,
		"log":                        logPath,
	}
	if err := writeSparseEmbeddingManifest(manifest.Artifacts["manifest_json"], manifest); err != nil {
		return err
	}
	if err := writeSparseEmbeddingSummary(manifest.Artifacts["summary_tsv"], manifest); err != nil {
		return err
	}
	scorecard := buildSparseEmbeddingSmokeScorecard(manifest)
	if err := writeSparseEmbeddingSmokeScorecard(manifest.Artifacts["scorecard_json"], scorecard); err != nil {
		return err
	}
	if err := writeSparseEmbeddingSmokeScorecardTSV(manifest.Artifacts["scorecard_tsv"], scorecard); err != nil {
		return err
	}
	logLines = append(logLines,
		"status="+manifest.Runtime.Status,
		"actual_backend="+manifest.Runtime.ActualBackend,
		"cuda_available="+strconv.FormatBool(manifest.Runtime.CUDAAvailable),
		"cuda_evidence_status="+manifest.Runtime.CUDAEvidenceStatus,
		"fallback_reason="+manifest.Runtime.FallbackReason,
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
	fmt.Printf("scorecard_json: %s\n", manifest.Artifacts["scorecard_json"])
	fmt.Printf("scorecard_tsv: %s\n", manifest.Artifacts["scorecard_tsv"])
	fmt.Printf("preflight_json: %s\n", preflightPath)
	fmt.Printf("embedding_dim=%d runtime_seq_len=%d requested_backend=%s actual_backend=%s cuda_available=%t 32k_preflight=%s gate=%s\n",
		manifest.Embedding.Dimension, cfg.RuntimeSeqLen, manifest.Runtime.RequestedBackend, manifest.Runtime.ActualBackend, manifest.Runtime.CUDAAvailable, manifest.ThirtyTwoKPreflight.Status, passFail(preflight.Gate.Passed))
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
	backendFlag := fs.String("backend", smokeEnvString("EOS_SPARSE_EMBED_SMOKE_BACKEND", "auto"), "execution backend: auto, host, or cuda")
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
	maxParitySeqLen := fs.Int("max-parity-seq-len", smokeEnvInt("EOS_SPARSE_EMBED_SMOKE_MAX_PARITY_SEQ_LEN", 4096), "maximum runtime sequence length for dense parity diagnostics; 0 disables dense diagnostics")
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
		Backend:          strings.ToLower(strings.TrimSpace(*backendFlag)),
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
		MaxParitySeqLen:  *maxParitySeqLen,
		RequireSubq:      *requireSubq,
	}
	if cfg.ValueDim == 0 {
		cfg.ValueDim = cfg.Dim
	}
	if cfg.Backend == "" {
		cfg.Backend = "auto"
	}
	if cfg.RunDir == "" {
		cfg.RunDir = filepath.Join(cfg.RunRoot, "eos-sparse-embedding-encoder-smoke-"+stamp)
	}
	switch cfg.Backend {
	case "auto", "host", "cuda":
	default:
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("backend must be auto, host, or cuda")
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
	if cfg.MaxParitySeqLen < 0 {
		return sparseEmbeddingSmokeConfig{}, fmt.Errorf("max-parity-seq-len must be non-negative")
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
	plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen:       cfg.QueryLen,
		KeyLen:         cfg.RuntimeSeqLen,
		QueryDim:       cfg.Dim,
		ValueDim:       cfg.ValueDim,
		TopK:           cfg.TopK,
		RouteBlockSize: blockSize,
		RouteTopBlocks: cfg.RouteTopBlocks,
	})
	attrs := map[string]string{
		"bits":             strconv.Itoa(cfg.Bits),
		"seed":             strconv.FormatInt(cfg.Seed, 10),
		"top_k":            strconv.Itoa(plan.TopK),
		"route_block_size": strconv.Itoa(blockSize),
		"route_top_blocks": strconv.Itoa(cfg.RouteTopBlocks),
	}
	query := backend.NewTensorF16([]int{cfg.QueryLen, cfg.Dim}, syntheticQuery(cfg.QueryLen, cfg.Dim, cfg.Seed))
	key := backend.NewTensorF16([]int{1, cfg.Dim, cfg.RuntimeSeqLen, 1}, syntheticNCHW(1, cfg.Dim, cfg.RuntimeSeqLen, cfg.Seed, 17))
	value := backend.NewTensorF16([]int{1, cfg.ValueDim, cfg.RuntimeSeqLen, 1}, syntheticNCHW(1, cfg.ValueDim, cfg.RuntimeSeqLen, cfg.Seed, 53))
	keySeq := nchwSmokeAttentionSequence(key)
	valueSeq := nchwSmokeAttentionSequence(value)
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(key, attrs)
	if err != nil {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("turboquant encode key: %w", err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(value, attrs)
	if err != nil {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("turboquant encode value: %w", err)
	}

	var out *backend.Tensor
	var runtimeMeta sparseEmbeddingRuntime
	switch cfg.Backend {
	case "host":
		var err error
		out, runtimeMeta, err = executeSparseEmbeddingHost(cfg, query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
		if err != nil {
			return sparseEmbeddingSmokeManifest{}, err
		}
	case "cuda", "auto":
		cudaOut, cudaRuntime, err := executeSparseEmbeddingCUDA(cfg, query, keyCoords, keyNorms, valueCoords, valueNorms, attrs, plan)
		if err == nil {
			out = cudaOut
			runtimeMeta = cudaRuntime
		} else if cfg.Backend == "cuda" {
			return sparseEmbeddingSmokeManifest{}, fmt.Errorf("cuda sparse embedding smoke unavailable: %w", err)
		} else {
			var hostErr error
			out, runtimeMeta, hostErr = executeSparseEmbeddingHost(cfg, query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
			if hostErr != nil {
				return sparseEmbeddingSmokeManifest{}, hostErr
			}
			runtimeMeta.FallbackReason = err.Error()
			runtimeMeta.CUDAEvidenceStatus = "fallback_unavailable"
		}
	default:
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("unsupported backend %q", cfg.Backend)
	}
	if out == nil {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("sparse embedding smoke produced no output tensor")
	}
	parity, err := evaluateSparseEmbeddingParity(cfg, out, query, keySeq, valueSeq, keyCoords, keyNorms, valueCoords, valueNorms, attrs, plan)
	if err != nil {
		return sparseEmbeddingSmokeManifest{}, err
	}
	if !parity.StrictGate {
		return sparseEmbeddingSmokeManifest{}, fmt.Errorf("sparse embedding backend parity failed: max_abs_error=%.9g mse=%.9g cosine_similarity=%.9g",
			parity.BackendVsHostTurboQuant.MaxAbsError,
			parity.BackendVsHostTurboQuant.MSE,
			parity.BackendVsHostTurboQuant.CosineSimilarity)
	}
	runtimeMeta.OutputShape = append([]int(nil), out.Shape...)
	runtimeMeta.TurboQuantKeyCoordShape = append([]int(nil), keyCoords.Shape...)
	runtimeMeta.TurboQuantValueCoordShape = append([]int(nil), valueCoords.Shape...)
	runtimeMeta.TurboQuantKeyNormShape = append([]int(nil), keyNorms.Shape...)
	runtimeMeta.TurboQuantValueNormShape = append([]int(nil), valueNorms.Shape...)

	if runtimeMeta.AttentionMetadata == nil {
		runtimeMeta.AttentionMetadata = plan.Metadata()
	}
	runtimeMeta.AttentionMetadata["bits"] = cfg.Bits
	runtimeMeta.AttentionMetadata["seed"] = cfg.Seed
	runtimeMeta.AttentionMetadata["kv_decode"] = runtimeMeta.KVDecode
	runtimeMeta.AttentionMetadata["dense_kv_materialized"] = runtimeMeta.DenseKVMaterialized
	runtimeMeta.AttentionMetadata["device_execution"] = runtimeMeta.DeviceExecution
	runtimeMeta.AttentionMetadata["runtime_seq_len"] = cfg.RuntimeSeqLen

	embedding := normalizeVector(meanPoolRows(out, cfg.QueryLen, cfg.ValueDim))
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
		Runtime:     runtimeMeta,
		Parity:      parity,
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

func evaluateSparseEmbeddingParity(cfg sparseEmbeddingSmokeConfig, backendOut, query, keySeq, valueSeq, keyCoords, keyNorms, valueCoords, valueNorms *backend.Tensor, attrs map[string]string, plan backend.SparseAttentionPlan) (sparseEmbeddingParity, error) {
	const (
		maxAbsTol = 1e-4
		mseTol    = 1e-8
		cosTol    = 0.999999
	)
	hostTurbo, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		return sparseEmbeddingParity{}, fmt.Errorf("host turboquant parity reference: %w", err)
	}
	backendVsHost := compareSparseEmbeddingTensors(backendOut, hostTurbo)
	backendVsHost.Passed = backendVsHost.MaxAbsError <= maxAbsTol && backendVsHost.MSE <= mseTol && backendVsHost.CosineSimilarity >= cosTol
	backendVsHost.Status = passFail(backendVsHost.Passed)

	parity := sparseEmbeddingParity{
		Status:                    backendVsHost.Status,
		StrictGate:                backendVsHost.Passed,
		MaxAbsErrorTolerance:      maxAbsTol,
		MSETolerance:              mseTol,
		CosineSimilarityTolerance: cosTol,
		BackendVsHostTurboQuant:   backendVsHost,
		Diagnostics: sparseEmbeddingParityDiag{
			Status:                         "skipped",
			DenseDiagnosticRuntimeSeqLen:   cfg.RuntimeSeqLen,
			DenseDiagnosticMaxParitySeqLen: cfg.MaxParitySeqLen,
		},
	}
	if cfg.MaxParitySeqLen == 0 {
		parity.Diagnostics.SkippedReason = "disabled_by_max_parity_seq_len"
		return parity, nil
	}
	if cfg.RuntimeSeqLen > cfg.MaxParitySeqLen {
		parity.Diagnostics.SkippedReason = fmt.Sprintf("runtime_seq_len %d exceeds max_parity_seq_len %d", cfg.RuntimeSeqLen, cfg.MaxParitySeqLen)
		return parity, nil
	}

	denseFullAttrs := cloneStringMap(attrs)
	denseFullAttrs["top_k"] = strconv.Itoa(cfg.RuntimeSeqLen)
	delete(denseFullAttrs, "route_block_size")
	delete(denseFullAttrs, "route_top_blocks")
	denseFull, err := backend.SparseAttentionReference(query, keySeq, valueSeq, denseFullAttrs)
	if err != nil {
		return sparseEmbeddingParity{}, fmt.Errorf("dense full parity reference: %w", err)
	}

	exactSparseAttrs := cloneStringMap(attrs)
	exactSparseAttrs["top_k"] = strconv.Itoa(plan.TopK)
	delete(exactSparseAttrs, "route_block_size")
	delete(exactSparseAttrs, "route_top_blocks")
	exactSparse, err := backend.SparseAttentionReference(query, keySeq, valueSeq, exactSparseAttrs)
	if err != nil {
		return sparseEmbeddingParity{}, fmt.Errorf("exact sparse parity reference: %w", err)
	}

	routedSparse, err := backend.SparseAttentionReference(query, keySeq, valueSeq, attrs)
	if err != nil {
		return sparseEmbeddingParity{}, fmt.Errorf("routed sparse parity reference: %w", err)
	}

	parity.Diagnostics.Status = "computed"
	parity.Diagnostics.SkippedReason = ""
	parity.Diagnostics.DenseFullVsExactSparse = diagnosticSparseEmbeddingCompare(denseFull, exactSparse)
	parity.Diagnostics.ExactSparseVsRoutedSparse = diagnosticSparseEmbeddingCompare(exactSparse, routedSparse)
	parity.Diagnostics.RoutedDenseVsTurboQuantRouted = diagnosticSparseEmbeddingCompare(routedSparse, hostTurbo)
	parity.Diagnostics.DenseFullSHA256 = tensorSHA256(denseFull)
	parity.Diagnostics.ExactSparseSHA256 = tensorSHA256(exactSparse)
	parity.Diagnostics.RoutedSparseDenseSHA256 = tensorSHA256(routedSparse)
	parity.Diagnostics.TurboQuantRoutedHostSHA256 = tensorSHA256(hostTurbo)
	return parity, nil
}

func diagnosticSparseEmbeddingCompare(actual, expected *backend.Tensor) sparseEmbeddingParityCompare {
	cmp := compareSparseEmbeddingTensors(actual, expected)
	cmp.Status = "computed"
	cmp.Passed = actual != nil && expected != nil && len(actual.F32) == len(expected.F32) && sameInts(actual.Shape, expected.Shape)
	return cmp
}

func compareSparseEmbeddingTensors(actual, expected *backend.Tensor) sparseEmbeddingParityCompare {
	cmp := sparseEmbeddingParityCompare{
		Status:         "fail",
		ActualSHA256:   tensorSHA256(actual),
		ExpectedSHA256: tensorSHA256(expected),
	}
	if actual == nil || expected == nil || len(actual.F32) != len(expected.F32) || !sameInts(actual.Shape, expected.Shape) {
		return cmp
	}
	var maxAbs, sumSq, dot, actualNorm, expectedNorm float64
	for i := range actual.F32 {
		a := float64(actual.F32[i])
		e := float64(expected.F32[i])
		diff := math.Abs(a - e)
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSq += (a - e) * (a - e)
		dot += a * e
		actualNorm += a * a
		expectedNorm += e * e
	}
	cmp.MaxAbsError = maxAbs
	if len(actual.F32) > 0 {
		cmp.MSE = sumSq / float64(len(actual.F32))
	}
	if actualNorm == 0 && expectedNorm == 0 {
		cmp.CosineSimilarity = 1
	} else if actualNorm != 0 && expectedNorm != 0 {
		cmp.CosineSimilarity = dot / (math.Sqrt(actualNorm) * math.Sqrt(expectedNorm))
	}
	return cmp
}

func nchwSmokeAttentionSequence(input *backend.Tensor) *backend.Tensor {
	if input == nil || len(input.Shape) != 4 {
		return nil
	}
	batches, channels, seqLen, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	if batches != 1 || width != 1 {
		return nil
	}
	out := backend.NewTensorF16([]int{seqLen, channels}, make([]float32, seqLen*channels))
	for t := 0; t < seqLen; t++ {
		for c := 0; c < channels; c++ {
			out.F32[t*channels+c] = input.F32[(c*seqLen+t)*width]
		}
	}
	return out
}

func tensorSHA256(t *backend.Tensor) string {
	if t == nil {
		return ""
	}
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "shape=%v\n", t.Shape)
	for _, x := range t.F32 {
		_, _ = fmt.Fprintf(h, "%.9g\n", x)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func cloneStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func executeSparseEmbeddingHost(cfg sparseEmbeddingSmokeConfig, query, keyCoords, keyNorms, valueCoords, valueNorms *backend.Tensor, attrs map[string]string) (*backend.Tensor, sparseEmbeddingRuntime, error) {
	out, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		return nil, sparseEmbeddingRuntime{}, fmt.Errorf("turbo sparse attention reference: %w", err)
	}
	blockSize := cfg.RouteBlockSize
	if blockSize == 0 {
		blockSize = int(math.Ceil(math.Sqrt(float64(cfg.RuntimeSeqLen))))
	}
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
	return out, sparseEmbeddingRuntime{
		Backend:             "host_reference",
		RequestedBackend:    cfg.Backend,
		ActualBackend:       "host_reference",
		CUDAAvailable:       false,
		CUDAEvidenceStatus:  "not_requested",
		DeviceExecution:     false,
		Status:              "pass",
		AttentionMetadata:   meta,
		DenseKVMaterialized: true,
		KVDecode:            "host_reference_decode",
	}, nil
}

func executeSparseEmbeddingCUDA(cfg sparseEmbeddingSmokeConfig, query, keyCoords, keyNorms, valueCoords, valueNorms *backend.Tensor, attrs map[string]string, plan backend.SparseAttentionPlan) (*backend.Tensor, sparseEmbeddingRuntime, error) {
	src := []byte(fmt.Sprintf(`
pipeline attend(q: f16[Q, D], kc: q%d[1, D, T, 1], kn: q_norm[1, T, 1], vc: q%d[1, V, T, 1], vn: q_norm[1, T, 1]) -> f16[Q, V] {
    return turbo_sparse_attention(q, kc, kn, vc, vn, %d, %d, %d)
}
`, cfg.Bits, cfg.Bits, plan.TopK, plan.RouteBlockSize, plan.RouteTopBlocks))
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "sparse_embedding_cuda_smoke"})
	if err != nil {
		return nil, sparseEmbeddingRuntime{}, fmt.Errorf("build cuda smoke module: %w", err)
	}
	for i := range bundle.Artifact.Steps {
		if bundle.Artifact.Steps[i].Kind == "turbo_sparse_attention" {
			if bundle.Artifact.Steps[i].Attributes == nil {
				bundle.Artifact.Steps[i].Attributes = map[string]string{}
			}
			for k, v := range attrs {
				bundle.Artifact.Steps[i].Attributes[k] = v
			}
		}
	}
	rt := eosruntime.New(cuda.New())
	program, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		return nil, sparseEmbeddingRuntime{}, fmt.Errorf("load cuda runtime: %w", err)
	}
	raw, err := program.Run(context.Background(), backend.Request{
		Entry: "attend",
		Inputs: map[string]any{
			"q":  query,
			"kc": keyCoords,
			"kn": keyNorms,
			"vc": valueCoords,
			"vn": valueNorms,
		},
	})
	if err != nil {
		return nil, sparseEmbeddingRuntime{}, fmt.Errorf("run cuda runtime: %w", err)
	}
	outputName, out, meta, err := singleTensorOutputWithMetadata(raw)
	if err != nil {
		return nil, sparseEmbeddingRuntime{}, err
	}
	traceVariant, traceEntry := turboSparseAttentionTrace(raw)
	deviceExecution := metaBool(meta["device_execution"])
	denseKVMaterialized := metaBoolDefault(meta["dense_kv_materialized"], true)
	kvDecode := fmt.Sprint(meta["kv_decode"])
	cudaOK := deviceExecution && !denseKVMaterialized && kvDecode == "cuda_turboquant_inline" && traceVariant == "__builtin_cuda_turbo_sparse_attention"
	if !cudaOK {
		return nil, sparseEmbeddingRuntime{}, fmt.Errorf("cuda turbo_sparse_attention evidence missing: output=%s variant=%q device_execution=%v dense_kv_materialized=%v kv_decode=%q", outputName, traceVariant, deviceExecution, denseKVMaterialized, kvDecode)
	}
	meta["runtime_seq_len"] = cfg.RuntimeSeqLen
	return out, sparseEmbeddingRuntime{
		Backend:             "cuda",
		RequestedBackend:    cfg.Backend,
		ActualBackend:       "cuda",
		CUDAAvailable:       true,
		CUDAEvidenceStatus:  "executed",
		TraceVariant:        traceVariant,
		TraceEntry:          traceEntry,
		DeviceExecution:     true,
		Status:              "pass",
		AttentionMetadata:   meta,
		DenseKVMaterialized: false,
		KVDecode:            kvDecode,
	}, nil
}

func singleTensorOutputWithMetadata(raw backend.Result) (string, *backend.Tensor, map[string]any, error) {
	outputName := ""
	var tensor *backend.Tensor
	var meta map[string]any
	for name, value := range raw.Outputs {
		t, ok := value.Data.(*backend.Tensor)
		if !ok || t == nil {
			continue
		}
		if outputName != "" {
			return "", nil, nil, fmt.Errorf("cuda smoke produced multiple tensor outputs: %q and %q", outputName, name)
		}
		outputName = name
		tensor = t
		meta = cloneAnyMap(value.Metadata)
	}
	if outputName == "" {
		return "", nil, nil, fmt.Errorf("cuda smoke produced no tensor output")
	}
	return outputName, tensor, meta, nil
}

func turboSparseAttentionTrace(raw backend.Result) (variant, entry string) {
	for _, step := range raw.Trace {
		if step.Kind == "turbo_sparse_attention" {
			return step.Variant, step.Entry
		}
	}
	return "", ""
}

func cloneAnyMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func metaBool(v any) bool {
	b, _ := v.(bool)
	return b
}

func metaBoolDefault(v any, fallback bool) bool {
	if b, ok := v.(bool); ok {
		return b
	}
	return fallback
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
		"requested_backend",
		"actual_backend",
		"cuda_available",
		"cuda_evidence_status",
		"fallback_reason",
		"trace_variant",
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
		"parity_status",
		"parity_backend_vs_host_passed",
		"parity_backend_vs_host_max_abs_error",
		"parity_backend_vs_host_mse",
		"parity_backend_vs_host_cosine_similarity",
		"parity_backend_actual_sha256",
		"parity_host_turboquant_sha256",
		"parity_diagnostics_status",
		"dense_full_vs_exact_sparse_max_abs_error",
		"dense_full_vs_exact_sparse_mse",
		"dense_full_vs_exact_sparse_cosine_similarity",
		"exact_sparse_vs_routed_sparse_max_abs_error",
		"exact_sparse_vs_routed_sparse_mse",
		"exact_sparse_vs_routed_sparse_cosine_similarity",
		"routed_dense_vs_turboquant_routed_max_abs_error",
		"routed_dense_vs_turboquant_routed_mse",
		"routed_dense_vs_turboquant_routed_cosine_similarity",
	}
	row := []string{
		manifest.Runtime.Status,
		strconv.Itoa(manifest.Config.RuntimeSeqLen),
		strconv.FormatBool(manifest.ThirtyTwoKPreflightOnly),
		strconv.Itoa(manifest.Embedding.Dimension),
		manifest.Runtime.RequestedBackend,
		manifest.Runtime.ActualBackend,
		strconv.FormatBool(manifest.Runtime.CUDAAvailable),
		manifest.Runtime.CUDAEvidenceStatus,
		manifest.Runtime.FallbackReason,
		manifest.Runtime.TraceVariant,
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
		manifest.Parity.Status,
		strconv.FormatBool(manifest.Parity.BackendVsHostTurboQuant.Passed),
		formatParityFloat(manifest.Parity.BackendVsHostTurboQuant.MaxAbsError),
		formatParityFloat(manifest.Parity.BackendVsHostTurboQuant.MSE),
		formatParityFloat(manifest.Parity.BackendVsHostTurboQuant.CosineSimilarity),
		manifest.Parity.BackendVsHostTurboQuant.ActualSHA256,
		manifest.Parity.BackendVsHostTurboQuant.ExpectedSHA256,
		manifest.Parity.Diagnostics.Status,
		formatParityFloat(manifest.Parity.Diagnostics.DenseFullVsExactSparse.MaxAbsError),
		formatParityFloat(manifest.Parity.Diagnostics.DenseFullVsExactSparse.MSE),
		formatParityFloat(manifest.Parity.Diagnostics.DenseFullVsExactSparse.CosineSimilarity),
		formatParityFloat(manifest.Parity.Diagnostics.ExactSparseVsRoutedSparse.MaxAbsError),
		formatParityFloat(manifest.Parity.Diagnostics.ExactSparseVsRoutedSparse.MSE),
		formatParityFloat(manifest.Parity.Diagnostics.ExactSparseVsRoutedSparse.CosineSimilarity),
		formatParityFloat(manifest.Parity.Diagnostics.RoutedDenseVsTurboQuantRouted.MaxAbsError),
		formatParityFloat(manifest.Parity.Diagnostics.RoutedDenseVsTurboQuantRouted.MSE),
		formatParityFloat(manifest.Parity.Diagnostics.RoutedDenseVsTurboQuantRouted.CosineSimilarity),
	}
	return os.WriteFile(path, []byte(strings.Join(columns, "\t")+"\n"+strings.Join(row, "\t")+"\n"), 0o644)
}

func buildSparseEmbeddingSmokeScorecard(manifest sparseEmbeddingSmokeManifest) sparseEmbeddingSmokeScorecard {
	row32768, has32768 := sparseEmbeddingPreflightRow(manifest.Preflight.Rows, 32768)
	scoreFraction := 0.0
	subquadratic := false
	if has32768 {
		scoreFraction = row32768.ScoreCountFraction
		subquadratic = row32768.SubquadraticScorePlan
	}
	sourceArtifacts := []string{}
	for _, key := range []string{"manifest_json", "summary_tsv", "sparse_attention_preflight"} {
		if path := manifest.Artifacts[key]; path != "" {
			sourceArtifacts = append(sourceArtifacts, path)
		}
	}
	row := sparseEmbeddingSmokeScorecardRow{
		Category:                            "long_context_sparse_smoke",
		Dataset:                             "synthetic_sparse_embedding_encoder_smoke",
		Baseline:                            "eos-sparse-embedding-encoder",
		Status:                              manifest.Runtime.Status,
		Method:                              "routed_turboquant_sparse_attention_encoder_smoke",
		EvidenceLevel:                       "smoke_synthetic_kernel_evidence",
		QualityClaim:                        false,
		ClaimBoundary:                       "synthetic routed TurboQuant sparse-attention encoder smoke; validates preflight/runtime/parity evidence only, not retrieval quality or LongEmbed performance",
		SourceManifest:                      manifest.Artifacts["manifest_json"],
		SourceArtifacts:                     sourceArtifacts,
		RuntimeSeqLen:                       manifest.Config.RuntimeSeqLen,
		ThirtyTwoKPreflightStatus:           manifest.ThirtyTwoKPreflight.Status,
		ThirtyTwoKPreflightOnly:             manifest.ThirtyTwoKPreflightOnly,
		Preflight32768ScoreFraction:         scoreFraction,
		Preflight32768Subquadratic:          subquadratic,
		PreflightGateStatus:                 passFail(manifest.Preflight.Gate.Passed),
		RequestedBackend:                    manifest.Runtime.RequestedBackend,
		ActualBackend:                       manifest.Runtime.ActualBackend,
		RuntimeBackend:                      manifest.Runtime.Backend,
		CUDAAvailable:                       manifest.Runtime.CUDAAvailable,
		CUDAEvidenceStatus:                  manifest.Runtime.CUDAEvidenceStatus,
		FallbackReason:                      manifest.Runtime.FallbackReason,
		DeviceExecution:                     manifest.Runtime.DeviceExecution,
		DenseKVMaterialized:                 manifest.Runtime.DenseKVMaterialized,
		KVDecode:                            manifest.Runtime.KVDecode,
		TraceVariant:                        manifest.Runtime.TraceVariant,
		Bits:                                manifest.Config.Bits,
		QuantizerSeed:                       manifest.Config.Seed,
		EmbeddingDim:                        manifest.Embedding.Dimension,
		EmbeddingSHA256:                     manifest.Embedding.SHA256,
		ParityStatus:                        manifest.Parity.Status,
		ParityBackendVsHostPassed:           manifest.Parity.BackendVsHostTurboQuant.Passed,
		ParityBackendVsHostMaxAbsError:      manifest.Parity.BackendVsHostTurboQuant.MaxAbsError,
		ParityBackendVsHostMSE:              manifest.Parity.BackendVsHostTurboQuant.MSE,
		ParityBackendVsHostCosineSimilarity: manifest.Parity.BackendVsHostTurboQuant.CosineSimilarity,
		ParityDiagnosticsStatus:             manifest.Parity.Diagnostics.Status,
	}
	return sparseEmbeddingSmokeScorecard{
		Schema:      "eos.sparse_embedding_encoder_smoke_scorecard.v1",
		GeneratedAt: manifest.GeneratedAt,
		Rows:        []sparseEmbeddingSmokeScorecardRow{row},
	}
}

func sparseEmbeddingPreflightRow(rows []sparseAttentionPlanRow, keyLen int) (sparseAttentionPlanRow, bool) {
	for _, row := range rows {
		if row.KeyLen == keyLen {
			return row, true
		}
	}
	return sparseAttentionPlanRow{}, false
}

func writeSparseEmbeddingSmokeScorecard(path string, scorecard sparseEmbeddingSmokeScorecard) error {
	data, err := json.MarshalIndent(scorecard, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func writeSparseEmbeddingSmokeScorecardTSV(path string, scorecard sparseEmbeddingSmokeScorecard) error {
	columns := []string{
		"category",
		"dataset",
		"baseline",
		"status",
		"method",
		"evidence_level",
		"quality_claim",
		"runtime_seq_len",
		"32k_preflight_status",
		"32k_preflight_only",
		"preflight_32768_score_fraction",
		"preflight_32768_subquadratic",
		"preflight_gate_status",
		"requested_backend",
		"actual_backend",
		"runtime_backend",
		"cuda_available",
		"cuda_evidence_status",
		"device_execution",
		"dense_kv_materialized",
		"kv_decode",
		"bits",
		"quantizer_seed",
		"embedding_dim",
		"embedding_sha256",
		"parity_status",
		"parity_backend_vs_host_passed",
		"parity_backend_vs_host_max_abs_error",
		"parity_backend_vs_host_mse",
		"parity_backend_vs_host_cosine_similarity",
		"parity_diagnostics_status",
		"source_manifest",
	}
	lines := []string{strings.Join(columns, "\t")}
	for _, row := range scorecard.Rows {
		values := []string{
			row.Category,
			row.Dataset,
			row.Baseline,
			row.Status,
			row.Method,
			row.EvidenceLevel,
			strconv.FormatBool(row.QualityClaim),
			strconv.Itoa(row.RuntimeSeqLen),
			row.ThirtyTwoKPreflightStatus,
			strconv.FormatBool(row.ThirtyTwoKPreflightOnly),
			fmt.Sprintf("%.6f", row.Preflight32768ScoreFraction),
			strconv.FormatBool(row.Preflight32768Subquadratic),
			row.PreflightGateStatus,
			row.RequestedBackend,
			row.ActualBackend,
			row.RuntimeBackend,
			strconv.FormatBool(row.CUDAAvailable),
			row.CUDAEvidenceStatus,
			strconv.FormatBool(row.DeviceExecution),
			strconv.FormatBool(row.DenseKVMaterialized),
			row.KVDecode,
			strconv.Itoa(row.Bits),
			strconv.FormatInt(row.QuantizerSeed, 10),
			strconv.Itoa(row.EmbeddingDim),
			row.EmbeddingSHA256,
			row.ParityStatus,
			strconv.FormatBool(row.ParityBackendVsHostPassed),
			formatParityFloat(row.ParityBackendVsHostMaxAbsError),
			formatParityFloat(row.ParityBackendVsHostMSE),
			formatParityFloat(row.ParityBackendVsHostCosineSimilarity),
			row.ParityDiagnosticsStatus,
			row.SourceManifest,
		}
		lines = append(lines, strings.Join(values, "\t"))
	}
	return os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644)
}

func formatParityFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', 9, 64)
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
