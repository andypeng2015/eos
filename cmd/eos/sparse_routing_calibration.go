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
	RunRoot            string    `json:"run_root"`
	RunDir             string    `json:"run_dir"`
	JSONPath           string    `json:"json_path"`
	TSVPath            string    `json:"tsv_path"`
	SeqLen             int       `json:"seq_len"`
	QueryLen           int       `json:"query_len"`
	Dim                int       `json:"dim"`
	ValueDim           int       `json:"value_dim"`
	TopK               int       `json:"top_k"`
	RouteBlockSize     int       `json:"route_block_size"`
	RouteTopBlocks     []int     `json:"route_top_blocks"`
	Seed               int64     `json:"seed"`
	MaxScoreFraction   float64   `json:"max_score_fraction"`
	MinExactTopKRecall float64   `json:"min_exact_topk_recall"`
	MinOutputCosine    float64   `json:"min_output_cosine"`
	RequirePass        bool      `json:"require_pass"`
	CreatedUTC         time.Time `json:"-"`
}

type sparseRoutingCalibrationReport struct {
	Schema      string                          `json:"schema"`
	CreatedUTC  string                          `json:"created_utc"`
	Config      sparseRoutingCalibrationConfig  `json:"config"`
	Rows        []sparseRoutingCalibrationRow   `json:"rows"`
	Summary     sparseRoutingCalibrationSummary `json:"summary"`
	Artifacts   map[string]string               `json:"artifacts"`
	Description string                          `json:"description"`
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
	RouteTopBlocks     int     `json:"route_top_blocks"`
	ExactTopKRecallAvg float64 `json:"exact_topk_recall_avg"`
	OutputCosine       float64 `json:"output_cosine_similarity"`
	ScoreCountFraction float64 `json:"score_count_fraction"`
	Pass               bool    `json:"pass"`
}

type sparseRoutingCalibrationRow struct {
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
		fmt.Printf("best_passing: route_top_blocks=%d recall_avg=%.6f cosine=%.9g score_fraction=%.6f\n",
			ref.RouteTopBlocks, ref.ExactTopKRecallAvg, ref.OutputCosine, ref.ScoreCountFraction)
	} else if report.Summary.BestFallback != nil {
		ref := report.Summary.BestFallback
		fmt.Printf("best_fallback: route_top_blocks=%d recall_avg=%.6f cosine=%.9g score_fraction=%.6f\n",
			ref.RouteTopBlocks, ref.ExactTopKRecallAvg, ref.OutputCosine, ref.ScoreCountFraction)
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
	seed := fs.Int64("seed", smokeEnvInt64("EOS_SPARSE_ROUTING_CALIBRATION_SEED", 5581486560434873699), "synthetic data seed")
	maxScoreFraction := fs.Float64("max-score-fraction", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_MAX_SCORE_FRACTION", 0.5), "passing row maximum score-work fraction versus exact dense scoring")
	minExactTopKRecall := fs.Float64("min-exact-topk-recall", smokeEnvFloat("EOS_SPARSE_ROUTING_CALIBRATION_MIN_EXACT_TOPK_RECALL", 0.9), "passing row minimum mean exact top-k recall in routed candidate set")
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
	cfg := sparseRoutingCalibrationConfig{
		RunRoot:            *runRoot,
		RunDir:             *runDir,
		JSONPath:           *jsonPath,
		TSVPath:            *tsvPath,
		SeqLen:             *seqLen,
		QueryLen:           *queryLen,
		Dim:                *dim,
		ValueDim:           *valueDim,
		TopK:               *topK,
		RouteBlockSize:     *routeBlockSize,
		RouteTopBlocks:     routeTopBlocks,
		Seed:               *seed,
		MaxScoreFraction:   *maxScoreFraction,
		MinExactTopKRecall: *minExactTopKRecall,
		MinOutputCosine:    *minOutputCosine,
		RequirePass:        *requirePass,
		CreatedUTC:         time.Now().UTC(),
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
	if cfg.MinExactTopKRecall < 0 || cfg.MinExactTopKRecall > 1 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("min-exact-topk-recall must be between 0 and 1")
	}
	if cfg.MinOutputCosine < -1 || cfg.MinOutputCosine > 1 {
		return sparseRoutingCalibrationConfig{}, fmt.Errorf("min-output-cosine must be between -1 and 1")
	}
	return cfg, nil
}

func executeSparseRoutingCalibration(cfg sparseRoutingCalibrationConfig) (sparseRoutingCalibrationReport, error) {
	blockSize := cfg.RouteBlockSize
	if blockSize == 0 {
		blockSize = int(math.Ceil(math.Sqrt(float64(cfg.SeqLen))))
	}
	query := backend.NewTensorF16([]int{cfg.QueryLen, cfg.Dim}, syntheticQuery(cfg.QueryLen, cfg.Dim, cfg.Seed))
	keyNCHW := backend.NewTensorF16([]int{1, cfg.Dim, cfg.SeqLen, 1}, syntheticNCHW(1, cfg.Dim, cfg.SeqLen, cfg.Seed, 17))
	valueNCHW := backend.NewTensorF16([]int{1, cfg.ValueDim, cfg.SeqLen, 1}, syntheticNCHW(1, cfg.ValueDim, cfg.SeqLen, cfg.Seed, 53))
	keySeq := nchwSmokeAttentionSequence(keyNCHW)
	valueSeq := nchwSmokeAttentionSequence(valueNCHW)
	exactPlan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen: cfg.QueryLen,
		KeyLen:   cfg.SeqLen,
		QueryDim: cfg.Dim,
		ValueDim: cfg.ValueDim,
		TopK:     cfg.TopK,
	})
	exactAttrs := map[string]string{"top_k": strconv.Itoa(exactPlan.TopK)}
	exactSparse, err := backend.SparseAttentionReference(query, keySeq, valueSeq, exactAttrs)
	if err != nil {
		return sparseRoutingCalibrationReport{}, fmt.Errorf("exact sparse reference: %w", err)
	}
	exactSelections := exactTopKSelections(query, keySeq, exactPlan.TopK)
	report := sparseRoutingCalibrationReport{
		Schema:     "manta.sparse_routing_calibration.v1",
		CreatedUTC: cfg.CreatedUTC.Format(time.RFC3339),
		Config:     cfg,
		Artifacts: map[string]string{
			"calibration_json": cfg.JSONPath,
			"calibration_tsv":  cfg.TSVPath,
		},
		Description: "Synthetic block-anchor sparse attention router calibration only; not retrieval quality evidence.",
	}
	report.Config.CreatedUTC = time.Time{}
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
		attrs := map[string]string{
			"top_k":            strconv.Itoa(plan.TopK),
			"route_block_size": strconv.Itoa(plan.RouteBlockSize),
			"route_top_blocks": strconv.Itoa(plan.RouteTopBlocks),
		}
		routedSparse, err := backend.SparseAttentionReference(query, keySeq, valueSeq, attrs)
		if err != nil {
			return sparseRoutingCalibrationReport{}, fmt.Errorf("routed sparse reference route_top_blocks=%d: %w", topBlocks, err)
		}
		recallAvg, recallMin := routedCandidateRecall(query, keySeq, exactSelections, plan.RouteBlockSize, plan.RouteTopBlocks)
		cmp := compareSparseEmbeddingTensors(routedSparse, exactSparse)
		row := sparseRoutingCalibrationRow{
			RouteTopBlocks:              plan.RouteTopBlocks,
			RouteBlockSize:              plan.RouteBlockSize,
			RouteBlockCount:             plan.RouteBlockCount,
			SelectedRouteBlocks:         plan.SelectedRouteBlocks,
			TopK:                        plan.TopK,
			SelectedKeyCount:            plan.SelectedKeyCount,
			CandidateKeyBudget:          plan.CandidateKeyBudget,
			DenseScoreCountPerQuery:     plan.DenseScoreCountPerQuery,
			RoutedAnchorScoresPerQuery:  plan.RoutedAnchorScoresPerQuery,
			EstimatedScoreCountPerQuery: plan.EstimatedScoreCountPerQuery,
			CandidateKeyFraction:        plan.CandidateKeyFraction,
			ScoreCountFraction:          plan.ScoreCountFraction,
			SubquadraticScorePlan:       plan.SubquadraticScorePlan,
			ExactTopKRecallAvg:          recallAvg,
			ExactTopKRecallMin:          recallMin,
			MaxAbsError:                 cmp.MaxAbsError,
			MSE:                         cmp.MSE,
			OutputCosineSimilarity:      cmp.CosineSimilarity,
			ExactSparseSHA256:           tensorSHA256(exactSparse),
			RoutedSparseSHA256:          tensorSHA256(routedSparse),
		}
		if row.ExactTopKRecallAvg < cfg.MinExactTopKRecall {
			row.FailureReasons = append(row.FailureReasons, fmt.Sprintf("exact_topk_recall_avg %.6f below %.6f", row.ExactTopKRecallAvg, cfg.MinExactTopKRecall))
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
		summary.FailureReason = "no route_top_blocks row met recall, cosine, and score-fraction thresholds"
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
	return a.RouteTopBlocks < b.RouteTopBlocks
}

func sparseRoutingRowRef(row sparseRoutingCalibrationRow) sparseRoutingCalibrationRowRef {
	return sparseRoutingCalibrationRowRef{
		RouteTopBlocks:     row.RouteTopBlocks,
		ExactTopKRecallAvg: row.ExactTopKRecallAvg,
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
