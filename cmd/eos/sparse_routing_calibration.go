package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"m31labs.dev/eos/runtime/backend"
)

type sparseRoutingCalibrationConfig struct {
	RunRoot                 string    `json:"run_root"`
	RunDir                  string    `json:"run_dir"`
	JSONPath                string    `json:"json_path"`
	TSVPath                 string    `json:"tsv_path"`
	SeqLen                  int       `json:"seq_len"`
	QueryLen                int       `json:"query_len"`
	Dim                     int       `json:"dim"`
	ValueDim                int       `json:"value_dim"`
	TopK                    int       `json:"top_k"`
	RouteBlockSize          int       `json:"route_block_size"`
	RouteTopBlocks          []int     `json:"route_top_blocks"`
	RouteModes              []string  `json:"route_modes"`
	RouteProbes             []int     `json:"route_probes"`
	RouteSummaryCounts      []int     `json:"route_summary_counts"`
	RouteSummaryAlphas      []float64 `json:"route_summary_alphas"`
	RouteRadiusWeights      []float64 `json:"route_radius_weights"`
	Seed                    int64     `json:"seed"`
	LearnedRouterTrainSeeds []int64   `json:"learned_router_train_seeds"`
	LearnedRouterEvalSeeds  []int64   `json:"learned_router_eval_seeds"`
	LearnedRouterSteps      int       `json:"learned_router_steps"`
	LearnedRouterLR         float64   `json:"learned_router_lr"`
	LearnedRouterL2         float64   `json:"learned_router_l2"`
	MaxScoreFraction        float64   `json:"max_score_fraction"`
	MinExactTopKRecall      float64   `json:"min_exact_topk_recall"`
	MinExactTopKRecallMin   float64   `json:"min_exact_topk_recall_min"`
	MinOutputCosine         float64   `json:"min_output_cosine"`
	RequirePass             bool      `json:"require_pass"`
	CreatedUTC              time.Time `json:"-"`
}

type sparseRoutingCalibrationReport struct {
	Schema        string                           `json:"schema"`
	CreatedUTC    string                           `json:"created_utc"`
	Config        sparseRoutingCalibrationConfig   `json:"config"`
	LearnedRouter *sparseRoutingLearnedBlockRouter `json:"learned_router,omitempty"`
	Rows          []sparseRoutingCalibrationRow    `json:"rows"`
	Summary       sparseRoutingCalibrationSummary  `json:"summary"`
	Artifacts     map[string]string                `json:"artifacts"`
	Description   string                           `json:"description"`
}

type sparseRoutingCalibrationSummary struct {
	Rows          int                             `json:"rows"`
	PassingRows   int                             `json:"passing_rows"`
	BestPassing   *sparseRoutingCalibrationRowRef `json:"best_passing,omitempty"`
	BestFallback  *sparseRoutingCalibrationRowRef `json:"best_fallback,omitempty"`
	RequirePass   bool                            `json:"require_pass"`
	Status        string                          `json:"status"`
	FailureReason string                          `json:"failure_reason,omitempty"`
}

type sparseRoutingCalibrationRowRef struct {
	Seed               int64   `json:"seed"`
	RouteMode          string  `json:"route_mode"`
	RouteProbes        int     `json:"route_probes"`
	RouteSummaryCount  int     `json:"route_summary_count"`
	RouteSummaryAlpha  float64 `json:"route_summary_alpha"`
	RouteRadiusWeight  float64 `json:"route_radius_weight"`
	RouteTopBlocks     int     `json:"route_top_blocks"`
	ExactTopKRecallAvg float64 `json:"exact_topk_recall_avg"`
	ExactTopKRecallMin float64 `json:"exact_topk_recall_min"`
	OutputCosine       float64 `json:"output_cosine_similarity"`
	ScoreCountFraction float64 `json:"score_count_fraction"`
	Pass               bool    `json:"pass"`
}

type sparseRoutingCalibrationRow struct {
	Seed                        int64    `json:"seed"`
	RouteMode                   string   `json:"route_mode"`
	RouteProbes                 int      `json:"route_probes"`
	RouteSummaryCount           int      `json:"route_summary_count"`
	RouteSummaryAlpha           float64  `json:"route_summary_alpha"`
	RouteRadiusWeight           float64  `json:"route_radius_weight"`
	RouteTopBlocks              int      `json:"route_top_blocks"`
	RouteBlockSize              int      `json:"route_block_size"`
	RouteBlockCount             int      `json:"route_block_count"`
	SelectedRouteBlocks         int      `json:"selected_route_blocks"`
	TopK                        int      `json:"top_k"`
	SelectedKeyCount            int      `json:"selected_key_count"`
	CandidateKeyBudget          int      `json:"candidate_key_budget"`
	DenseScoreCountPerQuery     int      `json:"dense_score_count_per_query"`
	RoutedAnchorScoresPerQuery  int      `json:"routed_anchor_scores_per_query"`
	EstimatedScoreCountPerQuery int      `json:"estimated_score_count_per_query"`
	TeacherOnly                 bool     `json:"teacher_only"`
	OraclePolicy                bool     `json:"oracle_policy"`
	TeacherScoreCountPerQuery   int      `json:"teacher_score_count_per_query"`
	CandidateKeyFraction        float64  `json:"candidate_key_fraction"`
	ScoreCountFraction          float64  `json:"score_count_fraction"`
	SubquadraticScorePlan       bool     `json:"subquadratic_score_plan"`
	ExactTopKRecallAvg          float64  `json:"exact_topk_recall_avg"`
	ExactTopKRecallMin          float64  `json:"exact_topk_recall_min"`
	MaxAbsError                 float64  `json:"max_abs_error"`
	MSE                         float64  `json:"mse"`
	OutputCosineSimilarity      float64  `json:"output_cosine_similarity"`
	ExactSparseSHA256           string   `json:"exact_sparse_sha256"`
	RoutedSparseSHA256          string   `json:"routed_sparse_sha256"`
	Pass                        bool     `json:"pass"`
	Status                      string   `json:"status"`
	FailureReasons              []string `json:"failure_reasons,omitempty"`
}

type sparseRoutingCandidate struct {
	index int
	score float32
}

type sparseRoutingLearnedBlockRouter struct {
	Weights        []float64 `json:"weights"`
	FeatureMeans   []float64 `json:"feature_means"`
	FeatureInvStd  []float64 `json:"feature_inv_std"`
	TrainExamples  int       `json:"train_examples"`
	TrainPositives int       `json:"train_positives"`
	TrainSeeds     []int64   `json:"train_seeds"`
	TrainTopBlocks int       `json:"train_top_blocks"`
	Steps          int       `json:"steps"`
	LR             float64   `json:"lr"`
	L2             float64   `json:"l2"`
}

func runCalibrateSparseRouting(args []string) error {
	cfg, err := parseSparseRoutingCalibrationConfig(args)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(cfg.RunDir, 0o755); err != nil {
		return err
	}
	report, err := executeSparseRoutingCalibration(cfg)
	if err != nil {
		return err
	}
	if err := writeSparseRoutingCalibrationJSON(cfg.JSONPath, report); err != nil {
		return err
	}
	if err := writeSparseRoutingCalibrationTSV(cfg.TSVPath, report); err != nil {
		return err
	}
	fmt.Printf("run_dir: %s\n", cfg.RunDir)
	fmt.Printf("calibration_json: %s\n", cfg.JSONPath)
	fmt.Printf("calibration_tsv: %s\n", cfg.TSVPath)
	fmt.Printf("summary: rows=%d passing_rows=%d gate=%s\n", report.Summary.Rows, report.Summary.PassingRows, report.Summary.Status)
	if report.Summary.BestPassing != nil {
		ref := report.Summary.BestPassing
		fmt.Printf("best_passing: route_mode=%s route_probes=%d route_summary_count=%d route_summary_alpha=%.6g route_radius_weight=%.6g route_top_blocks=%d recall_avg=%.6f recall_min=%.6f cosine=%.9g score_fraction=%.6f\n",
			ref.RouteMode, ref.RouteProbes, ref.RouteSummaryCount, ref.RouteSummaryAlpha, ref.RouteRadiusWeight, ref.RouteTopBlocks, ref.ExactTopKRecallAvg, ref.ExactTopKRecallMin, ref.OutputCosine, ref.ScoreCountFraction)
	} else if report.Summary.BestFallback != nil {
		ref := report.Summary.BestFallback
		fmt.Printf("best_fallback: route_mode=%s route_probes=%d route_summary_count=%d route_summary_alpha=%.6g route_radius_weight=%.6g route_top_blocks=%d recall_avg=%.6f recall_min=%.6f cosine=%.9g score_fraction=%.6f\n",
			ref.RouteMode, ref.RouteProbes, ref.RouteSummaryCount, ref.RouteSummaryAlpha, ref.RouteRadiusWeight, ref.RouteTopBlocks, ref.ExactTopKRecallAvg, ref.ExactTopKRecallMin, ref.OutputCosine, ref.ScoreCountFraction)
	}
	if cfg.RequirePass && report.Summary.PassingRows == 0 {
		return fmt.Errorf("sparse routing calibration failed: %s", report.Summary.FailureReason)
	}
	return nil
}

func parseSparseRoutingCalibrationConfig(args []string) (sparseRoutingCalibrationConfig, error) {
	defaultRunRoot := filepath.Join(".", "runs")
	stamp := time.Now().UTC().Format("20060102T150405Z")
	fs := flag.NewFlagSet("calibrate-sparse-routing", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	runRoot := fs.String("run-root", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_RUN_ROOT", defaultRunRoot), "run artifact root")
	runDir := fs.String("run-dir", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_RUN_DIR", ""), "exact run artifact directory")
	jsonPath := fs.String("json", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_JSON", ""), "write calibration JSON path; default is <run-dir>/calibration.json")
	tsvPath := fs.String("tsv", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_TSV", ""), "write calibration TSV path; default is <run-dir>/calibration.tsv")
	seqLen := fs.Int("seq-len", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_SEQ_LEN", 512), "synthetic key/value sequence length")
	queryLen := fs.Int("query-len", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_QUERY_LEN", 4), "synthetic query rows")
	dim := fs.Int("dim", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_DIM", 32), "synthetic query/key dimension")
	valueDim := fs.Int("value-dim", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_VALUE_DIM", 0), "synthetic value dimension; 0 uses --dim")
	topK := fs.Int("top-k", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_TOP_K", 16), "exact sparse top-k selected keys; 0 uses ceil(sqrt(seq_len))")
	routeBlockSize := fs.Int("route-block-size", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_BLOCK_SIZE", 32), "route block size; 0 uses ceil(sqrt(seq_len))")
	routeTopBlocksRaw := fs.String("route-top-blocks", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_TOP_BLOCKS", "1,2,4,8"), "comma-separated route_top_blocks sweep")
	routeModesRaw := fs.String("route-modes", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_MODES", "anchor"), "comma-separated routing policy sweep: anchor,summary_mean,summary_mean_radius,summary_maxnorm,summary_blend_radius,summary_multirep,learned_block_linear,multiprobe,oracle_block_max")
	routeProbesRaw := fs.String("route-probes", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_PROBES", "1"), "comma-separated multiprobe probe counts")
	routeSummaryCountsRaw := fs.String("route-summary-counts", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_SUMMARY_COUNTS", "2,4,8"), "comma-separated summary_multirep representative counts")
	routeSummaryAlphasRaw := fs.String("route-summary-alphas", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_SUMMARY_ALPHAS", "0,0.25,0.5,0.75,1"), "comma-separated summary_blend_radius alpha sweep values; representative = mean + alpha*(maxnorm_rep-mean)")
	routeRadiusWeightsRaw := fs.String("route-radius-weights", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_ROUTE_RADIUS_WEIGHTS", "-0.25,0,0.25,0.5"), "comma-separated summary_blend_radius beta sweep values for beta*||query||_2*radius")
	seed := fs.Int64("seed", smokeEnvInt64("EOS_SPARSE_ROUTING_CALIBRATION_SEED", 5581486560434873699), "synthetic data seed")
	learnedRouterTrainSeedsRaw := fs.String("learned-router-train-seeds", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_LEARNED_ROUTER_TRAIN_SEEDS", ""), "comma-separated synthetic seeds for learned_block_linear training; default is --seed")
	learnedRouterEvalSeedsRaw := fs.String("learned-router-eval-seeds", smokeEnvString("EOS_SPARSE_ROUTING_CALIBRATION_LEARNED_ROUTER_EVAL_SEEDS", ""), "comma-separated synthetic eval seeds for learned_block_linear and other requested modes; default is --seed")
	learnedRouterSteps := fs.Int("learned-router-steps", smokeEnvInt("EOS_SPARSE_ROUTING_CALIBRATION_LEARNED_ROUTER_STEPS", 12), "deterministic SGD passes for learned_block_linear")
	learnedRouterLR := fs.Float64("learned-router-lr", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_LEARNED_ROUTER_LR", 0.05), "learning rate for learned_block_linear logistic SGD")
	learnedRouterL2 := fs.Float64("learned-router-l2", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_LEARNED_ROUTER_L2", 0.0001), "L2 penalty for learned_block_linear logistic SGD")
	maxScoreFraction := fs.Float64("max-score-fraction", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_MAX_SCORE_FRACTION", 0.5), "passing row maximum score-work fraction versus exact dense scoring")
	minExactTopKRecall := fs.Float64("min-exact-topk-recall", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_MIN_EXACT_TOPK_RECALL", 0.9), "passing row minimum mean exact top-k recall in routed candidate set")
	minExactTopKRecallMin := fs.Float64("min-exact-topk-recall-min", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_MIN_EXACT_TOPK_RECALL_MIN", 0), "passing row minimum per-query exact top-k recall in routed candidate set")
	minOutputCosine := fs.Float64("min-output-cosine", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_MIN_OUTPUT_COSINE", 0.95), "passing row minimum routed-vs-exact sparse output cosine")
	requirePass := fs.Bool("require-pass", smokeEnvBool("EOS_SPARSE_ROUTING_CALIBRATION_REQUIRE_PASS", false), "return non-zero when no sweep row meets all thresholds")
	if err := fs.Parse(args); err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	if fs.NArg() != 0 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("usage: eos calibrate-sparse-routing [flags]")
	}
	routeTopBlocks, err := parsePositiveIntCSV(*routeTopBlocksRaw, "route-top-blocks")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	routeModes, err := parseSparseRoutingRouteModes(*routeModesRaw)
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	routeProbes, err := parsePositiveIntCSV(*routeProbesRaw, "route-probes")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	routeSummaryCounts, err := parsePositiveIntCSV(*routeSummaryCountsRaw, "route-summary-counts")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	routeSummaryAlphas, err := parseFloatCSV(*routeSummaryAlphasRaw, "route-summary-alphas")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	routeRadiusWeights, err := parseFloatCSV(*routeRadiusWeightsRaw, "route-radius-weights")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	trainSeeds, err := parseOptionalInt64CSV(*learnedRouterTrainSeedsRaw, "learned-router-train-seeds")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	evalSeeds, err := parseOptionalInt64CSV(*learnedRouterEvalSeedsRaw, "learned-router-eval-seeds")
	if err != nil {
		return sparseRoutingCalibrationConfig{}, err
	}
	if len(trainSeeds) == 0 {
		trainSeeds = []int64{*seed}
	}
	if len(evalSeeds) == 0 {
		evalSeeds = []int64{*seed}
	}
	cfg := sparseRoutingCalibrationConfig{
		RunRoot:                 *runRoot,
		RunDir:                  *runDir,
		JSONPath:                *jsonPath,
		TSVPath:                 *tsvPath,
		SeqLen:                  *seqLen,
		QueryLen:                *queryLen,
		Dim:                     *dim,
		ValueDim:                *valueDim,
		TopK:                    *topK,
		RouteBlockSize:          *routeBlockSize,
		RouteTopBlocks:          routeTopBlocks,
		RouteModes:              routeModes,
		RouteProbes:             routeProbes,
		RouteSummaryCounts:      routeSummaryCounts,
		RouteSummaryAlphas:      routeSummaryAlphas,
		RouteRadiusWeights:      routeRadiusWeights,
		Seed:                    *seed,
		LearnedRouterTrainSeeds: trainSeeds,
		LearnedRouterEvalSeeds:  evalSeeds,
		LearnedRouterSteps:      *learnedRouterSteps,
		LearnedRouterLR:         *learnedRouterLR,
		LearnedRouterL2:         *learnedRouterL2,
		MaxScoreFraction:        *maxScoreFraction,
		MinExactTopKRecall:      *minExactTopKRecall,
		MinExactTopKRecallMin:   *minExactTopKRecallMin,
		MinOutputCosine:         *minOutputCosine,
		RequirePass:             *requirePass,
		CreatedUTC:              time.Now().UTC(),
	}
	if cfg.ValueDim == 0 {
		cfg.ValueDim = cfg.Dim
	}
	if cfg.RunDir == "" {
		cfg.RunDir = filepath.Join(cfg.RunRoot, "eos-sparse-routing-calibration-"+stamp)
	}
	if cfg.JSONPath == "" {
		cfg.JSONPath = filepath.Join(cfg.RunDir, "calibration.json")
	}
	if cfg.TSVPath == "" {
		cfg.TSVPath = filepath.Join(cfg.RunDir, "calibration.tsv")
	}
	if cfg.SeqLen <= 0 || cfg.QueryLen <= 0 || cfg.Dim <= 0 || cfg.ValueDim <= 0 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("seq-len, query-len, dim, and value-dim must be positive")
	}
	if cfg.TopK < 0 || cfg.RouteBlockSize < 0 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("top-k and route-block-size must be non-negative")
	}
	if cfg.MaxScoreFraction <= 0 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("max-score-fraction must be positive")
	}
	if cfg.LearnedRouterSteps <= 0 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("learned-router-steps must be positive")
	}
	if cfg.LearnedRouterLR <= 0 || math.IsNaN(cfg.LearnedRouterLR) || math.IsInf(cfg.LearnedRouterLR, 0) {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("learned-router-lr must be positive and finite")
	}
	if cfg.LearnedRouterL2 < 0 || math.IsNaN(cfg.LearnedRouterL2) || math.IsInf(cfg.LearnedRouterL2, 0) {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("learned-router-l2 must be non-negative and finite")
	}
	if cfg.MinExactTopKRecall < 0 || cfg.MinExactTopKRecall > 1 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("min-exact-topk-recall must be between 0 and 1")
	}
	if cfg.MinExactTopKRecallMin < 0 || cfg.MinExactTopKRecallMin > 1 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("min-exact-topk-recall-min must be between 0 and 1")
	}
	if cfg.MinOutputCosine < -1 || cfg.MinOutputCosine > 1 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("min-output-cosine must be between -1 and 1")
	}
	return cfg, nil
}

func parseSparseRoutingRouteModes(raw string) ([]string, error) {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	seen := map[string]struct{}{}
	for i, part := range parts {
		mode := strings.TrimSpace(part)
		if mode == "" {
			continue
		}
		switch mode {
		case "anchor", "summary_mean", "summary_mean_radius", "summary_maxnorm", "summary_blend_radius", "summary_multirep", "learned_block_linear", "multiprobe", "oracle_block_max":
		default:
			return nil, fmt.Errorf("route-modes[%d] %q must be one of anchor, summary_mean, summary_mean_radius, summary_maxnorm, summary_blend_radius, summary_multirep, learned_block_linear, multiprobe, oracle_block_max", i, mode)
		}
		if _, ok := seen[mode]; ok {
			continue
		}
		seen[mode] = struct{}{}
		out = append(out, mode)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("route-modes must contain at least one routing policy")
	}
	return out, nil
}

func parseOptionalInt64CSV(raw, label string) ([]int64, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	out := make([]int64, 0, len(parts))
	for i, part := range parts {
		item := strings.TrimSpace(part)
		if item == "" {
			continue
		}
		value, err := strconv.ParseInt(item, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%s[%d] %q must be an int64", label, i, item)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s must contain at least one int64 when set", label)
	}
	return out, nil
}

func parseFloatCSV(raw, label string) ([]float64, error) {
	parts := strings.Split(raw, ",")
	out := make([]float64, 0, len(parts))
	for i, part := range parts {
		item := strings.TrimSpace(part)
		if item == "" {
			continue
		}
		value, err := strconv.ParseFloat(item, 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("%s[%d] %q must be a finite float", label, i, item)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s must contain at least one finite float", label)
	}
	return out, nil
}

func executeSparseRoutingCalibration(cfg sparseRoutingCalibrationConfig) (sparseRoutingCalibrationReport, error) {
	blockSize := cfg.RouteBlockSize
	if blockSize == 0 {
		blockSize = int(math.Ceil(math.Sqrt(float64(cfg.SeqLen))))
	}
	exactPlan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen: cfg.QueryLen,
		KeyLen:   cfg.SeqLen,
		QueryDim: cfg.Dim,
		ValueDim: cfg.ValueDim,
		TopK:     cfg.TopK,
	})
	report := sparseRoutingCalibrationReport{
		Schema:     "manta.sparse_routing_calibration.v1",
		CreatedUTC: cfg.CreatedUTC.Format(time.RFC3339),
		Config:     cfg,
		Artifacts: map[string]string{
			"calibration_json": cfg.JSONPath,
			"calibration_tsv":  cfg.TSVPath,
		},
		Description: "Synthetic sparse attention router policy calibration only; not retrieval quality evidence.",
	}
	report.Config.CreatedUTC = time.Time{}
	maxSummaryCount := 1
	if sparseRoutingModeEnabled(cfg.RouteModes, "summary_multirep") {
		maxSummaryCount = maxPositiveInt(cfg.RouteSummaryCounts)
	}
	var learnedRouter *sparseRoutingLearnedBlockRouter
	if sparseRoutingModeEnabled(cfg.RouteModes, "learned_block_linear") {
		var err error
		learnedRouter, err = trainSparseRoutingLearnedBlockRouter(cfg, blockSize, maxPositiveInt(cfg.RouteTopBlocks))
		if err != nil {
			return sparseRoutingCalibrationReport{}, err
		}
		report.LearnedRouter = learnedRouter
	}
	for _, evalSeed := range cfg.LearnedRouterEvalSeeds {
		query := backend.NewTensorF16([]int{cfg.QueryLen, cfg.Dim}, syntheticQuery(cfg.QueryLen, cfg.Dim, evalSeed))
		keyNCHW := backend.NewTensorF16([]int{1, cfg.Dim, cfg.SeqLen, 1}, syntheticNCHW(1, cfg.Dim, cfg.SeqLen, evalSeed, 17))
		valueNCHW := backend.NewTensorF16([]int{1, cfg.ValueDim, cfg.SeqLen, 1}, syntheticNCHW(1, cfg.ValueDim, cfg.SeqLen, evalSeed, 53))
		keySeq := nchwSmokeAttentionSequence(keyNCHW)
		valueSeq := nchwSmokeAttentionSequence(valueNCHW)
		exactAttrs := map[string]string{"top_k": strconv.Itoa(exactPlan.TopK)}
		exactSparse, err := backend.SparseAttentionReference(query, keySeq, valueSeq, exactAttrs)
		if err != nil {
			return sparseRoutingCalibrationReport{}, fmt.Errorf("exact sparse reference seed=%d: %w", evalSeed, err)
		}
		exactSelections := exactTopKSelections(query, keySeq, exactPlan.TopK)
		blockSummaries := sparseRoutingBlockSummariesForKey(keySeq, blockSize, maxSummaryCount)
		for _, mode := range cfg.RouteModes {
			probesForMode := []int{1}
			if mode == "multiprobe" {
				probesForMode = cfg.RouteProbes
			}
			summaryCountsForMode := []int{1}
			if mode == "summary_multirep" {
				summaryCountsForMode = cfg.RouteSummaryCounts
			}
			for _, probes := range probesForMode {
				for _, summaryCount := range summaryCountsForMode {
					alphaSweep := []float64{0}
					betaSweep := []float64{0}
					if mode == "summary_blend_radius" {
						alphaSweep = cfg.RouteSummaryAlphas
						betaSweep = cfg.RouteRadiusWeights
					}
					for _, alpha := range alphaSweep {
						for _, beta := range betaSweep {
							for _, topBlocks := range cfg.RouteTopBlocks {
								plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
									QueryLen:       cfg.QueryLen,
									KeyLen:         cfg.SeqLen,
									QueryDim:       cfg.Dim,
									ValueDim:       cfg.ValueDim,
									TopK:           cfg.TopK,
									RouteBlockSize: blockSize,
									RouteTopBlocks: topBlocks,
								})
								params := sparseRoutingPolicyParams{
									Mode:          mode,
									Probes:        probes,
									SummaryCount:  summaryCount,
									SummaryAlpha:  alpha,
									RadiusWeight:  beta,
									LearnedRouter: learnedRouter,
								}
								routedSparse, candidateSets, err := sparseRoutingPolicyOutput(query, keySeq, valueSeq, plan.RouteBlockSize, plan.RouteTopBlocks, plan.TopK, params, blockSummaries)
								if err != nil {
									return sparseRoutingCalibrationReport{}, fmt.Errorf("routed sparse reference seed=%d route_mode=%s route_probes=%d route_summary_count=%d route_summary_alpha=%g route_radius_weight=%g route_top_blocks=%d: %w", evalSeed, mode, probes, summaryCount, alpha, beta, topBlocks, err)
								}
								recallAvg, recallMin := sparseRoutingCandidateRecall(exactSelections, candidateSets)
								cmp := compareSparseEmbeddingTensors(routedSparse, exactSparse)
								routeScoreCount := sparseRoutingRouteScoreCount(plan.RouteBlockCount, mode, probes, summaryCount)
								row := sparseRoutingCalibrationRow{
									Seed:                        evalSeed,
									RouteMode:                   mode,
									RouteProbes:                 probes,
									RouteSummaryCount:           summaryCount,
									RouteSummaryAlpha:           alpha,
									RouteRadiusWeight:           beta,
									RouteTopBlocks:              plan.RouteTopBlocks,
									RouteBlockSize:              plan.RouteBlockSize,
									RouteBlockCount:             plan.RouteBlockCount,
									SelectedRouteBlocks:         plan.SelectedRouteBlocks,
									TopK:                        plan.TopK,
									SelectedKeyCount:            plan.SelectedKeyCount,
									CandidateKeyBudget:          plan.CandidateKeyBudget,
									DenseScoreCountPerQuery:     plan.DenseScoreCountPerQuery,
									RoutedAnchorScoresPerQuery:  routeScoreCount,
									EstimatedScoreCountPerQuery: routeScoreCount + plan.CandidateKeyBudget,
									TeacherOnly:                 sparseRoutingTeacherOnlyPolicy(mode),
									OraclePolicy:                sparseRoutingOraclePolicy(mode),
									TeacherScoreCountPerQuery:   sparseRoutingTeacherScoreCount(plan.DenseScoreCountPerQuery, mode),
									CandidateKeyFraction:        plan.CandidateKeyFraction,
									SubquadraticScorePlan:       plan.SubquadraticScorePlan,
									ExactTopKRecallAvg:          recallAvg,
									ExactTopKRecallMin:          recallMin,
									MaxAbsError:                 cmp.MaxAbsError,
									MSE:                         cmp.MSE,
									OutputCosineSimilarity:      cmp.CosineSimilarity,
									ExactSparseSHA256:           tensorSHA256(exactSparse),
									RoutedSparseSHA256:          tensorSHA256(routedSparse),
								}
								if row.DenseScoreCountPerQuery > 0 {
									row.ScoreCountFraction = float64(row.EstimatedScoreCountPerQuery) / float64(row.DenseScoreCountPerQuery)
									row.SubquadraticScorePlan = row.EstimatedScoreCountPerQuery < row.DenseScoreCountPerQuery
								}
								if row.ExactTopKRecallAvg < cfg.MinExactTopKRecall {
									row.FailureReasons = append(row.FailureReasons, fmt.Sprintf("exact_topk_recall_avg %.6f below %.6f", row.ExactTopKRecallAvg, cfg.MinExactTopKRecall))
								}
								if row.ExactTopKRecallMin < cfg.MinExactTopKRecallMin {
									row.FailureReasons = append(row.FailureReasons, fmt.Sprintf("exact_topk_recall_min %.6f below %.6f", row.ExactTopKRecallMin, cfg.MinExactTopKRecallMin))
								}
								if row.OutputCosineSimilarity < cfg.MinOutputCosine {
									row.FailureReasons = append(row.FailureReasons, fmt.Sprintf("output_cosine_similarity %.9g below %.9g", row.OutputCosineSimilarity, cfg.MinOutputCosine))
								}
								if row.ScoreCountFraction > cfg.MaxScoreFraction {
									row.FailureReasons = append(row.FailureReasons, fmt.Sprintf("score_count_fraction %.6f exceeds %.6f", row.ScoreCountFraction, cfg.MaxScoreFraction))
								}
								row.Pass = len(row.FailureReasons) == 0
								row.Status = passFail(row.Pass)
								report.Rows = append(report.Rows, row)
							}
						}
					}
				}
			}
		}
	}
	report.Summary = summarizeSparseRoutingCalibration(report.Rows, cfg.RequirePass)
	return report, nil
}

func exactTopKSelections(query, key *backend.Tensor, topK int) [][]int {
	out := make([][]int, query.Shape[0])
	qLen, dim, keyLen := query.Shape[0], query.Shape[1], key.Shape[0]
	for q := 0; q < qLen; q++ {
		selected := sparseRoutingSelectTop(keyLen, topK, func(k int) float32 {
			var sum float32
			for d := 0; d < dim; d++ {
				sum += query.F32[q*dim+d] * key.F32[k*dim+d]
			}
			return sum
		})
		out[q] = make([]int, len(selected))
		for i, candidate := range selected {
			out[q][i] = candidate.index
		}
	}
	return out
}

type sparseRoutingBlockSummaries struct {
	Mean       [][]float32
	MaxNormRep [][]float32
	MultiRep   [][][]float32
	Radius     []float32
}

type sparseRoutingPolicyParams struct {
	Mode          string
	Probes        int
	SummaryCount  int
	SummaryAlpha  float64
	RadiusWeight  float64
	LearnedRouter *sparseRoutingLearnedBlockRouter
}

func sparseRoutingBlockSummariesForKey(key *backend.Tensor, blockSize, maxMultiRep int) sparseRoutingBlockSummaries {
	if key == nil || len(key.Shape) != 2 || blockSize <= 0 {
		return sparseRoutingBlockSummaries{}
	}
	if maxMultiRep <= 0 {
		maxMultiRep = 1
	}
	keyLen, dim := key.Shape[0], key.Shape[1]
	if blockSize > keyLen {
		blockSize = keyLen
	}
	blockCount := (keyLen + blockSize - 1) / blockSize
	out := sparseRoutingBlockSummaries{
		Mean:       make([][]float32, blockCount),
		MaxNormRep: make([][]float32, blockCount),
		MultiRep:   make([][][]float32, blockCount),
		Radius:     make([]float32, blockCount),
	}
	for block := 0; block < blockCount; block++ {
		start := block * blockSize
		end := start + blockSize
		if end > keyLen {
			end = keyLen
		}
		mean := make([]float32, dim)
		for k := start; k < end; k++ {
			for d := 0; d < dim; d++ {
				mean[d] += key.F32[k*dim+d]
			}
		}
		scale := float32(1.0 / float64(end-start))
		for d := 0; d < dim; d++ {
			mean[d] *= scale
		}
		var radiusSq float32
		maxNormSq := float32(math.Inf(-1))
		maxNormIndex := start
		for k := start; k < end; k++ {
			var normSq float32
			for d := 0; d < dim; d++ {
				v := key.F32[k*dim+d]
				normSq += v * v
			}
			if normSq > maxNormSq {
				maxNormSq = normSq
				maxNormIndex = k
			}
		}
		for k := start; k < end; k++ {
			var distSq float32
			for d := 0; d < dim; d++ {
				diff := key.F32[k*dim+d] - mean[d]
				distSq += diff * diff
			}
			if distSq > radiusSq {
				radiusSq = distSq
			}
		}
		maxNormRep := make([]float32, dim)
		copy(maxNormRep, key.F32[maxNormIndex*dim:(maxNormIndex+1)*dim])
		out.Mean[block] = mean
		out.MaxNormRep[block] = maxNormRep
		out.MultiRep[block] = sparseRoutingSelectMultiReps(key, start, end, dim, maxMultiRep, maxNormIndex)
		out.Radius[block] = float32(math.Sqrt(float64(radiusSq)))
	}
	return out
}

func sparseRoutingSelectMultiReps(key *backend.Tensor, start, end, dim, count, firstIndex int) [][]float32 {
	if key == nil || start >= end || dim <= 0 || count <= 0 {
		return nil
	}
	if count > end-start {
		count = end - start
	}
	if firstIndex < start || firstIndex >= end {
		firstIndex = start
	}
	reps := make([][]float32, 0, count)
	selected := map[int]struct{}{}
	addRep := func(index int) {
		rep := make([]float32, dim)
		copy(rep, key.F32[index*dim:(index+1)*dim])
		reps = append(reps, rep)
		selected[index] = struct{}{}
	}
	addRep(firstIndex)
	for len(reps) < count {
		bestIndex := -1
		bestMinDistSq := float32(math.Inf(-1))
		for k := start; k < end; k++ {
			if _, ok := selected[k]; ok {
				continue
			}
			minDistSq := float32(math.Inf(1))
			for _, rep := range reps {
				var distSq float32
				for d := 0; d < dim; d++ {
					diff := key.F32[k*dim+d] - rep[d]
					distSq += diff * diff
				}
				if distSq < minDistSq {
					minDistSq = distSq
				}
			}
			if bestIndex < 0 || minDistSq > bestMinDistSq || (minDistSq == bestMinDistSq && k < bestIndex) {
				bestIndex = k
				bestMinDistSq = minDistSq
			}
		}
		if bestIndex < 0 {
			break
		}
		addRep(bestIndex)
	}
	return reps
}

func trainSparseRoutingLearnedBlockRouter(cfg sparseRoutingCalibrationConfig, blockSize, trainTopBlocks int) (*sparseRoutingLearnedBlockRouter, error) {
	if trainTopBlocks <= 0 {
		return nil, fmt.Errorf("learned_block_linear training requires at least one positive route-top-blocks value")
	}
	type example struct {
		features []float64
		label    float64
	}
	var examples []example
	var positives int
	for _, seed := range cfg.LearnedRouterTrainSeeds {
		query := backend.NewTensorF16([]int{cfg.QueryLen, cfg.Dim}, syntheticQuery(cfg.QueryLen, cfg.Dim, seed))
		keyNCHW := backend.NewTensorF16([]int{1, cfg.Dim, cfg.SeqLen, 1}, syntheticNCHW(1, cfg.Dim, cfg.SeqLen, seed, 17))
		keySeq := nchwSmokeAttentionSequence(keyNCHW)
		summaries := sparseRoutingBlockSummariesForKey(keySeq, blockSize, 1)
		keyLen := keySeq.Shape[0]
		blockCount := (keyLen + blockSize - 1) / blockSize
		if trainTopBlocks > blockCount {
			trainTopBlocks = blockCount
		}
		for q := 0; q < cfg.QueryLen; q++ {
			scoreAt := func(k int) float32 {
				var sum float32
				for d := 0; d < cfg.Dim; d++ {
					sum += query.F32[q*cfg.Dim+d] * keySeq.F32[k*cfg.Dim+d]
				}
				return sum
			}
			oracleBlocks := sparseRoutingSelectTop(blockCount, trainTopBlocks, func(block int) float32 {
				start := block * blockSize
				end := start + blockSize
				if end > keyLen {
					end = keyLen
				}
				best := float32(math.Inf(-1))
				for k := start; k < end; k++ {
					if score := scoreAt(k); score > best {
						best = score
					}
				}
				return best
			})
			positiveBlocks := map[int]struct{}{}
			for _, candidate := range oracleBlocks {
				positiveBlocks[candidate.index] = struct{}{}
			}
			for block := 0; block < blockCount; block++ {
				start := block * blockSize
				end := start + blockSize
				if end > keyLen {
					end = keyLen
				}
				label := 0.0
				if _, ok := positiveBlocks[block]; ok {
					label = 1
					positives++
				}
				examples = append(examples, example{
					features: sparseRoutingLearnedFeatures(query, q, cfg.Dim, start, end, block, summaries),
					label:    label,
				})
			}
		}
	}
	if len(examples) == 0 || positives == 0 || positives == len(examples) {
		return nil, fmt.Errorf("learned_block_linear training produced degenerate labels: examples=%d positives=%d", len(examples), positives)
	}
	featureCount := len(examples[0].features)
	means := make([]float64, featureCount)
	for _, ex := range examples {
		for i, value := range ex.features {
			means[i] += value
		}
	}
	for i := range means {
		means[i] /= float64(len(examples))
	}
	invStd := make([]float64, featureCount)
	for _, ex := range examples {
		for i, value := range ex.features {
			diff := value - means[i]
			invStd[i] += diff * diff
		}
	}
	for i := range invStd {
		std := math.Sqrt(invStd[i]/float64(len(examples)) + 1e-12)
		invStd[i] = 1 / std
	}
	if featureCount > 0 {
		means[0] = 0
		invStd[0] = 1
	}
	weights := make([]float64, featureCount)
	posWeight := float64(len(examples)-positives) / float64(positives)
	for step := 0; step < cfg.LearnedRouterSteps; step++ {
		for _, ex := range examples {
			var z float64
			for i, value := range ex.features {
				x := (value - means[i]) * invStd[i]
				z += weights[i] * x
			}
			p := 1 / (1 + math.Exp(-clampFloat64(z, -40, 40)))
			weight := 1.0
			if ex.label > 0 {
				weight = posWeight
			}
			err := (p - ex.label) * weight
			for i, value := range ex.features {
				x := (value - means[i]) * invStd[i]
				weights[i] -= cfg.LearnedRouterLR * (err*x + cfg.LearnedRouterL2*weights[i])
			}
		}
	}
	return &sparseRoutingLearnedBlockRouter{
		Weights:        weights,
		FeatureMeans:   means,
		FeatureInvStd:  invStd,
		TrainExamples:  len(examples),
		TrainPositives: positives,
		TrainSeeds:     append([]int64{}, cfg.LearnedRouterTrainSeeds...),
		TrainTopBlocks: trainTopBlocks,
		Steps:          cfg.LearnedRouterSteps,
		LR:             cfg.LearnedRouterLR,
		L2:             cfg.LearnedRouterL2,
	}, nil
}

func sparseRoutingLearnedFeatures(query *backend.Tensor, queryRow, dim, start, end, block int, blockSummaries sparseRoutingBlockSummaries) []float64 {
	meanScore := 0.0
	maxNormScore := 0.0
	if mean, ok := sparseRoutingBlockMean(blockSummaries, block, dim); ok {
		meanScore = float64(sparseRoutingDotQuerySummary(query, queryRow, dim, mean))
	}
	if rep, ok := sparseRoutingBlockMaxNormRep(blockSummaries, block, dim); ok {
		maxNormScore = float64(sparseRoutingDotQuerySummary(query, queryRow, dim, rep))
	}
	radiusFeature := 0.0
	if block < len(blockSummaries.Radius) {
		radiusFeature = float64(sparseRoutingQueryNorm(query, queryRow, dim) * blockSummaries.Radius[block])
	}
	blockLen := float64(end - start)
	return []float64{
		1,
		meanScore,
		maxNormScore,
		radiusFeature,
		blockLen,
	}
}

func sparseRoutingLearnedBlockScore(query *backend.Tensor, queryRow, dim, start, end, block int, blockSummaries sparseRoutingBlockSummaries, model *sparseRoutingLearnedBlockRouter) float32 {
	if model == nil || len(model.Weights) == 0 {
		return float32(math.Inf(-1))
	}
	features := sparseRoutingLearnedFeatures(query, queryRow, dim, start, end, block, blockSummaries)
	var score float64
	for i, value := range features {
		if i >= len(model.Weights) || i >= len(model.FeatureMeans) || i >= len(model.FeatureInvStd) {
			break
		}
		score += model.Weights[i] * (value - model.FeatureMeans[i]) * model.FeatureInvStd[i]
	}
	return float32(score)
}

func sparseRoutingModeEnabled(modes []string, want string) bool {
	for _, mode := range modes {
		if mode == want {
			return true
		}
	}
	return false
}

func maxPositiveInt(values []int) int {
	var out int
	for _, value := range values {
		if value > out {
			out = value
		}
	}
	return out
}

func clampFloat64(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func sparseRoutingRouteScoreCount(blockCount int, mode string, probes, summaryCount int) int {
	if mode == "multiprobe" {
		return blockCount * probes
	}
	if mode == "summary_multirep" {
		if summaryCount < 1 {
			summaryCount = 1
		}
		return blockCount * summaryCount
	}
	return blockCount
}

func sparseRoutingEstimatedScoreCount(blockCount, candidateBudget int, mode string, probes, summaryCount int) int {
	return sparseRoutingRouteScoreCount(blockCount, mode, probes, summaryCount) + candidateBudget
}

func sparseRoutingTeacherOnlyPolicy(mode string) bool {
	return mode == "oracle_block_max"
}

func sparseRoutingOraclePolicy(mode string) bool {
	return mode == "oracle_block_max"
}

func sparseRoutingTeacherScoreCount(denseScoreCount int, mode string) int {
	if sparseRoutingTeacherOnlyPolicy(mode) {
		return denseScoreCount
	}
	return 0
}

func sparseRoutingPolicyOutput(query, key, value *backend.Tensor, blockSize, topBlocks, topK int, params sparseRoutingPolicyParams, blockSummaries sparseRoutingBlockSummaries) (*backend.Tensor, []map[int]struct{}, error) {
	if query == nil || key == nil || value == nil {
		return nil, nil, fmt.Errorf("policy sparse attention expects query, key, and value")
	}
	if len(query.Shape) != 2 || len(key.Shape) != 2 || len(value.Shape) != 2 {
		return nil, nil, fmt.Errorf("policy sparse attention expects rank-2 query/key/value tensors")
	}
	qLen, dim := query.Shape[0], query.Shape[1]
	keyLen, keyDim := key.Shape[0], key.Shape[1]
	valueLen, valueDim := value.Shape[0], value.Shape[1]
	if dim != keyDim {
		return nil, nil, fmt.Errorf("query dim %d does not match key dim %d", dim, keyDim)
	}
	if keyLen != valueLen {
		return nil, nil, fmt.Errorf("key length %d does not match value length %d", keyLen, valueLen)
	}
	out := backend.NewTensorF16([]int{qLen, valueDim}, make([]float32, qLen*valueDim))
	candidateSets := make([]map[int]struct{}, qLen)
	for q := 0; q < qLen; q++ {
		scoreAt := func(k int) float32 {
			var sum float32
			for d := 0; d < dim; d++ {
				sum += query.F32[q*dim+d] * key.F32[k*dim+d]
			}
			return sum
		}
		candidates, candidateSet := sparseRoutingPolicyCandidates(query, keyLen, dim, q, blockSize, topBlocks, params, blockSummaries, scoreAt)
		selected := sparseRoutingSelectTopCandidates(candidates, topK)
		sparseRoutingWriteValue(out.F32[q*valueDim:(q+1)*valueDim], selected, valueDim, func(k, d int) float32 {
			return value.F32[k*valueDim+d]
		})
		candidateSets[q] = candidateSet
	}
	return out, candidateSets, nil
}

func sparseRoutingPolicyCandidates(query *backend.Tensor, keyLen, dim, queryRow, blockSize, topBlocks int, params sparseRoutingPolicyParams, blockSummaries sparseRoutingBlockSummaries, scoreAt func(int) float32) ([]sparseRoutingCandidate, map[int]struct{}) {
	candidateSet := map[int]struct{}{}
	if keyLen <= 0 || blockSize <= 0 || topBlocks <= 0 {
		candidates := sparseRoutingSelectTop(keyLen, keyLen, scoreAt)
		for _, candidate := range candidates {
			candidateSet[candidate.index] = struct{}{}
		}
		return candidates, candidateSet
	}
	if blockSize > keyLen {
		blockSize = keyLen
	}
	blockCount := (keyLen + blockSize - 1) / blockSize
	if topBlocks > blockCount {
		topBlocks = blockCount
	}
	blocks := make([]sparseRoutingCandidate, 0, blockCount)
	for block := 0; block < blockCount; block++ {
		start := block * blockSize
		end := start + blockSize
		if end > keyLen {
			end = keyLen
		}
		blocks = append(blocks, sparseRoutingCandidate{
			index: block,
			score: sparseRoutingBlockScore(query, queryRow, dim, start, end, block,
				params, blockSummaries, scoreAt),
		})
	}
	blocks = sparseRoutingSelectTopCandidates(blocks, topBlocks)
	candidates := make([]sparseRoutingCandidate, 0, topBlocks*blockSize)
	for _, block := range blocks {
		start := block.index * blockSize
		end := start + blockSize
		if end > keyLen {
			end = keyLen
		}
		for k := start; k < end; k++ {
			candidates = append(candidates, sparseRoutingCandidate{index: k, score: scoreAt(k)})
			candidateSet[k] = struct{}{}
		}
	}
	return candidates, candidateSet
}

func sparseRoutingBlockScore(query *backend.Tensor, queryRow, dim, start, end, block int, params sparseRoutingPolicyParams, blockSummaries sparseRoutingBlockSummaries, scoreAt func(int) float32) float32 {
	switch params.Mode {
	case "summary_mean":
		if mean, ok := sparseRoutingBlockMean(blockSummaries, block, dim); ok {
			return sparseRoutingDotQuerySummary(query, queryRow, dim, mean)
		}
	case "summary_mean_radius":
		if mean, ok := sparseRoutingBlockMean(blockSummaries, block, dim); ok {
			// Deployable Cauchy-style upper bound:
			// dot(q, mean(block)) + ||q||_2 * max_k_in_block ||key_k - mean(block)||_2.
			// The scalar radius is precomputed per block, so routing uses one summary dot
			// per block plus this scalar correction before candidate-key scoring.
			score := sparseRoutingDotQuerySummary(query, queryRow, dim, mean)
			if block < len(blockSummaries.Radius) {
				score += sparseRoutingQueryNorm(query, queryRow, dim) * blockSummaries.Radius[block]
			}
			return score
		}
	case "summary_maxnorm":
		if rep, ok := sparseRoutingBlockMaxNormRep(blockSummaries, block, dim); ok {
			return sparseRoutingDotQuerySummary(query, queryRow, dim, rep)
		}
	case "summary_blend_radius":
		if mean, ok := sparseRoutingBlockMean(blockSummaries, block, dim); ok {
			if maxNormRep, ok := sparseRoutingBlockMaxNormRep(blockSummaries, block, dim); ok {
				score := sparseRoutingDotQueryBlendedSummary(query, queryRow, dim, mean, maxNormRep, float32(params.SummaryAlpha))
				if block < len(blockSummaries.Radius) && params.RadiusWeight != 0 {
					score += float32(params.RadiusWeight) * sparseRoutingQueryNorm(query, queryRow, dim) * blockSummaries.Radius[block]
				}
				return score
			}
		}
	case "summary_multirep":
		if reps, ok := sparseRoutingBlockMultiReps(blockSummaries, block, dim, params.SummaryCount); ok {
			best := float32(math.Inf(-1))
			for _, rep := range reps {
				if score := sparseRoutingDotQuerySummary(query, queryRow, dim, rep); score > best {
					best = score
				}
			}
			return best
		}
	case "learned_block_linear":
		return sparseRoutingLearnedBlockScore(query, queryRow, dim, start, end, block, blockSummaries, params.LearnedRouter)
	case "multiprobe":
		best := float32(math.Inf(-1))
		for _, k := range sparseRoutingProbeIndices(start, end, params.Probes) {
			if score := scoreAt(k); score > best {
				best = score
			}
		}
		return best
	case "oracle_block_max":
		best := float32(math.Inf(-1))
		for k := start; k < end; k++ {
			if score := scoreAt(k); score > best {
				best = score
			}
		}
		return best
	}
	anchor := start + (end-start)/2
	return scoreAt(anchor)
}

func sparseRoutingBlockMean(blockSummaries sparseRoutingBlockSummaries, block, dim int) ([]float32, bool) {
	if block < len(blockSummaries.Mean) && len(blockSummaries.Mean[block]) == dim {
		return blockSummaries.Mean[block], true
	}
	return nil, false
}

func sparseRoutingBlockMaxNormRep(blockSummaries sparseRoutingBlockSummaries, block, dim int) ([]float32, bool) {
	if block < len(blockSummaries.MaxNormRep) && len(blockSummaries.MaxNormRep[block]) == dim {
		return blockSummaries.MaxNormRep[block], true
	}
	return nil, false
}

func sparseRoutingBlockMultiReps(blockSummaries sparseRoutingBlockSummaries, block, dim, count int) ([][]float32, bool) {
	if count <= 0 {
		count = 1
	}
	if block >= len(blockSummaries.MultiRep) || len(blockSummaries.MultiRep[block]) == 0 {
		return nil, false
	}
	reps := blockSummaries.MultiRep[block]
	if count > len(reps) {
		count = len(reps)
	}
	for i := 0; i < count; i++ {
		if len(reps[i]) != dim {
			return nil, false
		}
	}
	return reps[:count], true
}

func sparseRoutingDotQuerySummary(query *backend.Tensor, queryRow, dim int, summary []float32) float32 {
	var sum float32
	for d := 0; d < dim; d++ {
		sum += query.F32[queryRow*dim+d] * summary[d]
	}
	return sum
}

func sparseRoutingDotQueryBlendedSummary(query *backend.Tensor, queryRow, dim int, mean, maxNormRep []float32, alpha float32) float32 {
	var sum float32
	for d := 0; d < dim; d++ {
		rep := mean[d] + alpha*(maxNormRep[d]-mean[d])
		sum += query.F32[queryRow*dim+d] * rep
	}
	return sum
}

func sparseRoutingQueryNorm(query *backend.Tensor, queryRow, dim int) float32 {
	var sumSq float32
	for d := 0; d < dim; d++ {
		v := query.F32[queryRow*dim+d]
		sumSq += v * v
	}
	return float32(math.Sqrt(float64(sumSq)))
}

func sparseRoutingProbeIndices(start, end, probes int) []int {
	if probes <= 1 || end-start <= 1 {
		return []int{start + (end-start)/2}
	}
	out := make([]int, 0, probes)
	last := end - 1
	seen := map[int]struct{}{}
	for i := 0; i < probes; i++ {
		k := start + int(math.Round(float64(i)*float64(last-start)/float64(probes-1)))
		if k < start {
			k = start
		}
		if k > last {
			k = last
		}
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, k)
	}
	if len(out) == 0 {
		out = append(out, start+(end-start)/2)
	}
	return out
}

func sparseRoutingWriteValue(out []float32, selected []sparseRoutingCandidate, valueDim int, valueAt func(int, int) float32) {
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
	var denom float64
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

func sparseRoutingCandidateRecall(exactSelections [][]int, candidateSets []map[int]struct{}) (float64, float64) {
	if len(exactSelections) == 0 {
		return 0, 0
	}
	minRecall := 1.0
	var sumRecall float64
	for q, exact := range exactSelections {
		candidates := map[int]struct{}{}
		if q < len(candidateSets) && candidateSets[q] != nil {
			candidates = candidateSets[q]
		}
		recall := sparseRoutingRecall(exact, candidates)
		sumRecall += recall
		if recall < minRecall {
			minRecall = recall
		}
	}
	return sumRecall / float64(len(exactSelections)), minRecall
}

func routedCandidateRecall(query, key *backend.Tensor, exactSelections [][]int, blockSize, topBlocks int) (float64, float64) {
	qLen, dim, keyLen := query.Shape[0], query.Shape[1], key.Shape[0]
	if qLen == 0 {
		return 0, 0
	}
	minRecall := 1.0
	var sumRecall float64
	for q := 0; q < qLen; q++ {
		candidates := sparseRoutingCandidateSet(keyLen, blockSize, topBlocks, func(k int) float32 {
			var sum float32
			for d := 0; d < dim; d++ {
				sum += query.F32[q*dim+d] * key.F32[k*dim+d]
			}
			return sum
		})
		recall := sparseRoutingRecall(exactSelections[q], candidates)
		sumRecall += recall
		if recall < minRecall {
			minRecall = recall
		}
	}
	return sumRecall / float64(qLen), minRecall
}

func sparseRoutingCandidateSet(keyLen, blockSize, topBlocks int, scoreAt func(int) float32) map[int]struct{} {
	out := map[int]struct{}{}
	if keyLen <= 0 || blockSize <= 0 || topBlocks <= 0 {
		for k := 0; k < keyLen; k++ {
			out[k] = struct{}{}
		}
		return out
	}
	if blockSize > keyLen {
		blockSize = keyLen
	}
	blockCount := (keyLen + blockSize - 1) / blockSize
	if topBlocks > blockCount {
		topBlocks = blockCount
	}
	blocks := make([]sparseRoutingCandidate, 0, blockCount)
	for block := 0; block < blockCount; block++ {
		start := block * blockSize
		end := start + blockSize
		if end > keyLen {
			end = keyLen
		}
		anchor := start + (end-start)/2
		blocks = append(blocks, sparseRoutingCandidate{index: block, score: scoreAt(anchor)})
	}
	blocks = sparseRoutingSelectTopCandidates(blocks, topBlocks)
	for _, block := range blocks {
		start := block.index * blockSize
		end := start + blockSize
		if end > keyLen {
			end = keyLen
		}
		for k := start; k < end; k++ {
			out[k] = struct{}{}
		}
	}
	return out
}

func sparseRoutingRecall(exact []int, candidates map[int]struct{}) float64 {
	if len(exact) == 0 {
		return 1
	}
	var hits int
	for _, k := range exact {
		if _, ok := candidates[k]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(exact))
}

func sparseRoutingSelectTop(keyLen, budget int, scoreAt func(int) float32) []sparseRoutingCandidate {
	candidates := make([]sparseRoutingCandidate, 0, keyLen)
	for k := 0; k < keyLen; k++ {
		candidates = append(candidates, sparseRoutingCandidate{index: k, score: scoreAt(k)})
	}
	return sparseRoutingSelectTopCandidates(candidates, budget)
}

func sparseRoutingSelectTopCandidates(candidates []sparseRoutingCandidate, budget int) []sparseRoutingCandidate {
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

func summarizeSparseRoutingCalibration(rows []sparseRoutingCalibrationRow, requirePass bool) sparseRoutingCalibrationSummary {
	summary := sparseRoutingCalibrationSummary{
		Rows:        len(rows),
		RequirePass: requirePass,
		Status:      "pass",
	}
	bestPassing := -1
	bestFallback := -1
	for i, row := range rows {
		if row.Pass {
			summary.PassingRows++
			if bestPassing < 0 || sparseRoutingRowBetter(row, rows[bestPassing]) {
				bestPassing = i
			}
		}
		if bestFallback < 0 || sparseRoutingRowBetter(row, rows[bestFallback]) {
			bestFallback = i
		}
	}
	if bestPassing >= 0 {
		ref := sparseRoutingRowRef(rows[bestPassing])
		summary.BestPassing = &ref
	}
	if bestFallback >= 0 && bestPassing < 0 {
		ref := sparseRoutingRowRef(rows[bestFallback])
		summary.BestFallback = &ref
	}
	if requirePass && summary.PassingRows == 0 {
		summary.Status = "fail"
		summary.FailureReason = "no routing policy row met recall, per-query recall, cosine, and score-fraction thresholds"
	}
	return summary
}

func sparseRoutingRowBetter(a, b sparseRoutingCalibrationRow) bool {
	if a.ExactTopKRecallAvg != b.ExactTopKRecallAvg {
		return a.ExactTopKRecallAvg > b.ExactTopKRecallAvg
	}
	if a.OutputCosineSimilarity != b.OutputCosineSimilarity {
		return a.OutputCosineSimilarity > b.OutputCosineSimilarity
	}
	if a.ScoreCountFraction != b.ScoreCountFraction {
		return a.ScoreCountFraction < b.ScoreCountFraction
	}
	if a.RouteMode != b.RouteMode {
		return a.RouteMode < b.RouteMode
	}
	if a.RouteProbes != b.RouteProbes {
		return a.RouteProbes < b.RouteProbes
	}
	if a.RouteSummaryCount != b.RouteSummaryCount {
		return a.RouteSummaryCount < b.RouteSummaryCount
	}
	return a.RouteTopBlocks < b.RouteTopBlocks
}

func sparseRoutingRowRef(row sparseRoutingCalibrationRow) sparseRoutingCalibrationRowRef {
	return sparseRoutingCalibrationRowRef{
		Seed:               row.Seed,
		RouteMode:          row.RouteMode,
		RouteProbes:        row.RouteProbes,
		RouteSummaryCount:  row.RouteSummaryCount,
		RouteSummaryAlpha:  row.RouteSummaryAlpha,
		RouteRadiusWeight:  row.RouteRadiusWeight,
		RouteTopBlocks:     row.RouteTopBlocks,
		ExactTopKRecallAvg: row.ExactTopKRecallAvg,
		ExactTopKRecallMin: row.ExactTopKRecallMin,
		OutputCosine:       row.OutputCosineSimilarity,
		ScoreCountFraction: row.ScoreCountFraction,
		Pass:               row.Pass,
	}
}

func writeSparseRoutingCalibrationJSON(path string, report sparseRoutingCalibrationReport) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func writeSparseRoutingCalibrationTSV(path string, report sparseRoutingCalibrationReport) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	columns := []string{
		"seed",
		"route_mode",
		"route_probes",
		"route_summary_count",
		"route_summary_alpha",
		"route_radius_weight",
		"route_top_blocks",
		"route_block_size",
		"route_block_count",
		"selected_route_blocks",
		"top_k",
		"selected_key_count",
		"candidate_key_budget",
		"dense_score_count_per_query",
		"routed_anchor_scores_per_query",
		"estimated_score_count_per_query",
		"teacher_only",
		"oracle_policy",
		"teacher_score_count_per_query",
		"candidate_key_fraction",
		"score_count_fraction",
		"subquadratic_score_plan",
		"exact_topk_recall_avg",
		"exact_topk_recall_min",
		"max_abs_error",
		"mse",
		"output_cosine_similarity",
		"exact_sparse_sha256",
		"routed_sparse_sha256",
		"pass",
		"status",
		"failure_reasons",
	}
	lines := []string{strings.Join(columns, "\t")}
	for _, row := range report.Rows {
		failureReasons := strings.Join(row.FailureReasons, "; ")
		if failureReasons == "" {
			failureReasons = "-"
		}
		fields := []string{
			strconv.FormatInt(row.Seed, 10),
			row.RouteMode,
			strconv.Itoa(row.RouteProbes),
			strconv.Itoa(row.RouteSummaryCount),
			formatParityFloat(row.RouteSummaryAlpha),
			formatParityFloat(row.RouteRadiusWeight),
			strconv.Itoa(row.RouteTopBlocks),
			strconv.Itoa(row.RouteBlockSize),
			strconv.Itoa(row.RouteBlockCount),
			strconv.Itoa(row.SelectedRouteBlocks),
			strconv.Itoa(row.TopK),
			strconv.Itoa(row.SelectedKeyCount),
			strconv.Itoa(row.CandidateKeyBudget),
			strconv.Itoa(row.DenseScoreCountPerQuery),
			strconv.Itoa(row.RoutedAnchorScoresPerQuery),
			strconv.Itoa(row.EstimatedScoreCountPerQuery),
			strconv.FormatBool(row.TeacherOnly),
			strconv.FormatBool(row.OraclePolicy),
			strconv.Itoa(row.TeacherScoreCountPerQuery),
			fmt.Sprintf("%.6f", row.CandidateKeyFraction),
			fmt.Sprintf("%.6f", row.ScoreCountFraction),
			strconv.FormatBool(row.SubquadraticScorePlan),
			fmt.Sprintf("%.6f", row.ExactTopKRecallAvg),
			fmt.Sprintf("%.6f", row.ExactTopKRecallMin),
			formatParityFloat(row.MaxAbsError),
			formatParityFloat(row.MSE),
			formatParityFloat(row.OutputCosineSimilarity),
			row.ExactSparseSHA256,
			row.RoutedSparseSHA256,
			strconv.FormatBool(row.Pass),
			row.Status,
			failureReasons,
		}
		lines = append(lines, strings.Join(fields, "\t"))
	}
	return os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644)
}
