package mantaruntime

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
)

// EmbeddingTrainRunConfig controls dataset-level native training.
type EmbeddingTrainRunConfig struct {
	Epochs                    int
	BatchSize                 int
	Shuffle                   bool
	Seed                      int64
	EvalEveryEpoch            int
	EvalEverySteps            int
	EarlyStoppingPatience     int
	SelectMetric              string
	MinDelta                  float32
	RestoreBest               bool
	EvalOnly                  bool
	PairwiseTrain             bool
	HardNegativeTrain         bool
	HardNegativesPerQuery     int
	HardNegativeSourceWeights map[string]int
	LengthBucketBatches       bool
	LearningRate              float32
	ContrastiveLoss           string
	Temperature               float32
	GroupedLossWeight         float32
	TeacherLossWeight         float32
	TeacherTemperature        float32
	TeacherSourceTemperatures map[string]float32
	ProgressEverySteps        int
	Progress                  EmbeddingTrainProgressFunc
}

// EmbeddingTrainProgressFunc receives incremental training progress updates.
type EmbeddingTrainProgressFunc func(EmbeddingTrainProgress)

// EmbeddingTrainProgress reports one completed optimizer step.
type EmbeddingTrainProgress struct {
	Epoch              int
	Batch              int
	Batches            int
	Step               int
	BatchExamples      int
	BatchPairs         int64
	EpochTrainExamples int64
	EpochTrainPairs    int64
	PlannedEpochPairs  int64
	Loss               float32
	AverageScore       float32
	Elapsed            time.Duration
}

// EmbeddingTrainEpochSummary records one epoch of training progress.
type EmbeddingTrainEpochSummary struct {
	Epoch    int
	Step     int
	Train    EmbeddingTrainMetrics
	Eval     *EmbeddingEvalMetrics
	Improved bool
}

// EmbeddingTrainWorkload summarizes planned and actual pairwise work for a run.
type EmbeddingTrainWorkload struct {
	TrainMode            string
	EvalMode             string
	TrainExamples        int
	EvalExamples         int
	BatchSize            int
	PlannedEpochs        int
	CompletedEpochs      int
	TrainBatchesPerEpoch int
	TrainPairsPerEpoch   int64
	EvalPairsPerPass     int64
	PlannedEvalPasses    int
	ActualEvalPasses     int
	PlannedTrainPairs    int64
	ActualTrainPairs     int64
	ActualTrainExamples  int64
	PlannedEvalPairs     int64
	ActualEvalPairs      int64
	ActualEvalExamples   int64
	PlannedTotalPairs    int64
	ActualTotalPairs     int64
	ActualTotalExamples  int64
}

// EmbeddingTrainRunSummary summarizes a full train/eval run.
type EmbeddingTrainRunSummary struct {
	Config          EmbeddingTrainRunConfig
	Workload        EmbeddingTrainWorkload
	EpochsCompleted int
	StepsCompleted  int
	StepsRun        int
	BestEpoch       int
	BestStep        int
	FinalTrain      EmbeddingTrainMetrics
	LastEval        *EmbeddingEvalMetrics
	BestEval        *EmbeddingEvalMetrics
	FinalEval       *EmbeddingEvalMetrics
	RestoredBest    bool
	StoppedEarly    bool
	History         []EmbeddingTrainEpochSummary
	StartProfile    EmbeddingTrainProfile
	EndProfile      EmbeddingTrainProfile
	DeltaProfile    EmbeddingTrainProfile
	Elapsed         time.Duration
	TrainDuration   time.Duration
	EvalDuration    time.Duration
}

// Fit trains over a dataset, periodically evaluates, and can restore the best checkpoint.
func (t *EmbeddingTrainer) Fit(trainSet, evalSet []EmbeddingPairExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if cfg.EvalOnly {
		if len(evalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:       cfg,
		StartProfile: t.TrainProfile(),
		Workload:     EstimatePairwiseTrainWorkload(len(trainSet), len(evalSet), cfg),
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.FinalEval = cloneEvalMetrics(finalEval)
		summary.LastEval = cloneEvalMetrics(finalEval)
		summary.BestEval = cloneEvalMetrics(finalEval)
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = int64(len(evalSet))
		summary.Workload.ActualEvalExamples = int64(len(evalSet))
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))

	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		trainStart := time.Now()
		trainMetrics, err := t.runEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(trainMetrics.BatchSize)
		summary.Workload.ActualTrainExamples += int64(trainMetrics.BatchSize)

		if len(evalSet) > 0 && epoch%cfg.EvalEveryEpoch == 0 {
			evalStart := time.Now()
			evalMetrics, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			summary.EvalDuration += time.Since(evalStart)
			summary.Workload.ActualEvalPasses++
			summary.Workload.ActualEvalPairs += int64(len(evalSet))
			summary.Workload.ActualEvalExamples += int64(len(evalSet))
			record.Eval = cloneEvalMetrics(evalMetrics)
			summary.LastEval = cloneEvalMetrics(evalMetrics)
			if !haveBest || betterEvalMetrics(evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
				bestCheckpoint, err = t.Checkpoint()
				if err != nil {
					return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d checkpoint: %w", epoch, err)
				}
				haveBest = true
				record.Improved = true
				summary.BestEval = cloneEvalMetrics(evalMetrics)
				summary.BestEpoch = epoch
				summary.BestStep = t.step
				noImproveEvals = 0
			} else {
				noImproveEvals++
				if cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
					summary.StoppedEarly = true
					summary.History = append(summary.History, record)
					break
				}
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 {
		evalStart := time.Now()
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.FinalEval = cloneEvalMetrics(finalEval)
		if summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(finalEval)
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// FitContrastive trains over query-positive examples using in-batch negatives.
func (t *EmbeddingTrainer) FitContrastive(trainSet, evalSet []EmbeddingContrastiveExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if cfg.EvalOnly {
		if len(evalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 1 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be at least 2 for contrastive training")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:       cfg,
		StartProfile: t.TrainProfile(),
		Workload:     EstimateContrastiveTrainWorkload(len(trainSet), len(evalSet), cfg),
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		finalEval, err := t.EvaluateContrastive(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.FinalEval = cloneEvalMetrics(finalEval)
		summary.LastEval = cloneEvalMetrics(finalEval)
		summary.BestEval = cloneEvalMetrics(finalEval)
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = int64(len(evalSet) * len(evalSet))
		summary.Workload.ActualEvalExamples = int64(len(evalSet))
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)
	recordEval := func(epoch int) (*EmbeddingEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalMetrics, err := t.EvaluateContrastive(evalSet)
		if err != nil {
			return nil, false, err
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet) * len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.LastEval = cloneEvalMetrics(evalMetrics)
		improved := false
		if !haveBest || betterEvalMetrics(evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
			bestCheckpoint, err = t.Checkpoint()
			if err != nil {
				return nil, false, err
			}
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(evalMetrics)
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else {
			noImproveEvals++
		}
		return cloneEvalMetrics(evalMetrics), improved, nil
	}
	if len(evalSet) > 0 && cfg.RestoreBest {
		if _, _, err := recordEval(0); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
		if cfg.LengthBucketBatches {
			bucketContrastiveOrderByLength(trainSet, indices, cfg.BatchSize)
		}
		trainStart := time.Now()
		var afterBatch contrastiveEpochBatchHook
		if len(evalSet) > 0 && cfg.EvalEverySteps > 0 {
			afterBatch = func(progress EmbeddingTrainProgress) error {
				if progress.Batch <= 0 || progress.Batch%cfg.EvalEverySteps != 0 {
					return nil
				}
				if _, _, err := recordEval(epoch); err != nil {
					return fmt.Errorf("step %d eval: %w", progress.Step, err)
				}
				return nil
			}
		}
		trainMetrics, err := t.runContrastiveEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart, afterBatch)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(trainMetrics.BatchSize)
		summary.Workload.ActualTrainExamples += int64(contrastiveUsableExampleCount(len(indices), cfg.BatchSize))

		if len(evalSet) > 0 && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, improved, err := recordEval(epoch)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.Improved = improved
			if !improved {
				if cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
					summary.StoppedEarly = true
					summary.History = append(summary.History, record)
					break
				}
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 {
		evalStart := time.Now()
		finalEval, err := t.EvaluateContrastive(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet) * len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.FinalEval = cloneEvalMetrics(finalEval)
		if summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(finalEval)
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// FitHardNegatives trains over query-positive examples with explicit hard negatives.
func (t *EmbeddingTrainer) FitHardNegatives(trainSet []EmbeddingHardNegativeExample, evalSet []EmbeddingPairExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if cfg.EvalOnly {
		if len(evalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:       cfg,
		StartProfile: t.TrainProfile(),
		Workload:     EstimateHardNegativeTrainWorkload(len(trainSet), cfg.HardNegativesPerQuery, len(evalSet), cfg),
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.FinalEval = cloneEvalMetrics(finalEval)
		summary.LastEval = cloneEvalMetrics(finalEval)
		summary.BestEval = cloneEvalMetrics(finalEval)
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = int64(len(evalSet))
		summary.Workload.ActualEvalExamples = int64(len(evalSet))
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)
	recordEval := func(epoch int) (*EmbeddingEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalMetrics, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return nil, false, err
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.LastEval = cloneEvalMetrics(evalMetrics)
		improved := false
		if !haveBest || betterEvalMetrics(evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
			bestCheckpoint, err = t.Checkpoint()
			if err != nil {
				return nil, false, err
			}
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(evalMetrics)
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else {
			noImproveEvals++
		}
		return cloneEvalMetrics(evalMetrics), improved, nil
	}
	if len(evalSet) > 0 && cfg.RestoreBest {
		if _, _, err := recordEval(0); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
		if len(cfg.HardNegativeSourceWeights) > 0 {
			indices = hardNegativeSourceWeightedOrder(trainSet, indices, cfg.BatchSize, cfg.HardNegativeSourceWeights, cfg.LengthBucketBatches)
		} else if cfg.LengthBucketBatches {
			bucketHardNegativeOrderByLength(trainSet, indices, cfg.BatchSize)
		}
		trainStart := time.Now()
		var afterBatch contrastiveEpochBatchHook
		if len(evalSet) > 0 && cfg.EvalEverySteps > 0 {
			afterBatch = func(progress EmbeddingTrainProgress) error {
				if progress.Batch <= 0 || progress.Batch%cfg.EvalEverySteps != 0 {
					return nil
				}
				if _, _, err := recordEval(epoch); err != nil {
					return fmt.Errorf("step %d eval: %w", progress.Step, err)
				}
				return nil
			}
		}
		trainMetrics, err := t.runHardNegativeEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart, afterBatch)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(trainMetrics.BatchSize)
		summary.Workload.ActualTrainExamples += int64(len(indices))

		if len(evalSet) > 0 && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, improved, err := recordEval(epoch)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.Improved = improved
			if !improved && cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
				summary.StoppedEarly = true
				summary.History = append(summary.History, record)
				break
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 {
		evalStart := time.Now()
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.FinalEval = cloneEvalMetrics(finalEval)
		if summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(finalEval)
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// EstimatePairwiseTrainWorkload returns planned pairwise work for supervised pair training.
func EstimatePairwiseTrainWorkload(trainExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches := batchCount(trainExamples, cfg.BatchSize, 1)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	trainPairsPerEpoch := int64(trainExamples)
	evalPairsPerPass := int64(evalExamples)
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	return EmbeddingTrainWorkload{
		TrainMode:            "pairwise",
		EvalMode:             workloadEvalMode(evalExamples, "pairwise"),
		TrainExamples:        trainExamples,
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

// EstimateContrastiveTrainWorkload returns planned pairwise work for contrastive training with in-batch negatives.
func EstimateContrastiveTrainWorkload(trainExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches, trainPairsPerEpoch := contrastiveBatchWork(trainExamples, cfg.BatchSize)
	evalPairsPerPass := contrastiveEvalPairs(evalExamples)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalExamples > 0 {
		evalPasses++
	}
	if cfg.EvalEverySteps > 0 && evalExamples > 0 {
		evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
	}
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	return EmbeddingTrainWorkload{
		TrainMode:            "contrastive",
		EvalMode:             workloadEvalMode(evalExamples, "contrastive"),
		TrainExamples:        trainExamples,
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

// EstimateHardNegativeTrainWorkload returns planned work for explicit hard-negative contrastive training.
func EstimateHardNegativeTrainWorkload(trainExamples, negativesPerExample, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	if negativesPerExample < 0 {
		negativesPerExample = 0
	}
	batches, trainPairsPerEpoch := hardNegativeBatchWork(trainExamples, cfg.BatchSize, negativesPerExample, cfg.ContrastiveLoss, cfg.TeacherLossWeight)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalExamples > 0 {
		evalPasses++
	}
	if cfg.EvalEverySteps > 0 && evalExamples > 0 {
		evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
	}
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	evalPairsPerPass := int64(evalExamples)
	return EmbeddingTrainWorkload{
		TrainMode:            hardNegativeTrainMode(cfg.ContrastiveLoss),
		EvalMode:             workloadEvalMode(evalExamples, "pairwise"),
		TrainExamples:        trainExamples,
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

func batchCount(total, batchSize, minBatch int) int {
	if total <= 0 || batchSize <= 0 {
		return 0
	}
	count := 0
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		if end-start < minBatch {
			break
		}
		count++
	}
	return count
}

func contrastiveUsableExampleCount(total, batchSize int) int {
	if total <= 0 || batchSize <= 1 {
		return 0
	}
	used := 0
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		if end-start < 2 {
			break
		}
		used += end - start
	}
	return used
}

func contrastiveBatchWork(total, batchSize int) (int, int64) {
	if total <= 0 || batchSize <= 1 {
		return 0, 0
	}
	var pairs int64
	batches := 0
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		n := end - start
		if n < 2 {
			break
		}
		batches++
		pairs += int64(n) * int64(n)
	}
	return batches, pairs
}

func contrastiveEvalPairs(total int) int64 {
	if total <= 0 {
		return 0
	}
	return int64(total) * int64(total)
}

func hardNegativeBatchWork(total, batchSize, negativesPerExample int, loss string, teacherLossWeight float32) (int, int64) {
	if total <= 0 || batchSize <= 0 {
		return 0, 0
	}
	if negativesPerExample < 0 {
		negativesPerExample = 0
	}
	var pairs int64
	batches := 0
	candidatesPerExample := int64(1 + negativesPerExample)
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		n := end - start
		if n <= 0 {
			break
		}
		candidates := int64(n) * candidatesPerExample
		if loss == "grouped_infonce" {
			if candidatesPerExample < 2 {
				break
			}
			batches++
			pairs += int64(n) * candidatesPerExample
		} else if loss == "hybrid_infonce" {
			if candidates < 2 {
				break
			}
			batches++
			pairs += int64(n) * candidates
			if candidatesPerExample >= 2 {
				pairs += int64(n) * candidatesPerExample
			}
		} else {
			if candidates < 2 {
				break
			}
			batches++
			pairs += int64(n) * candidates
		}
		if teacherLossWeight > 0 && candidatesPerExample >= 2 {
			pairs += int64(n) * candidatesPerExample
		}
	}
	return batches, pairs
}

func hardNegativeTrainMode(loss string) string {
	switch loss {
	case "grouped_infonce":
		return "hard_negative_grouped_infonce"
	case "hybrid_infonce":
		return "hard_negative_hybrid_infonce"
	default:
		return "hard_negative_contrastive"
	}
}

func hardNegativeBatchPairCount(batch []EmbeddingHardNegativeExample) int64 {
	if len(batch) == 0 {
		return 0
	}
	candidates := 0
	for _, example := range batch {
		candidates += 1 + len(example.NegativeTokens)
	}
	return int64(len(batch)) * int64(candidates)
}

func spreadHardNegativeOrderByQuery(trainSet []EmbeddingHardNegativeExample, order []int) []int {
	if len(trainSet) == 0 || len(order) < 2 {
		return order
	}
	type queryBucket struct {
		indexes []int
		next    int
	}
	buckets := []queryBucket{}
	bucketByKey := map[string]int{}
	for _, idx := range order {
		if idx < 0 || idx >= len(trainSet) {
			return order
		}
		example := trainSet[idx]
		key := embeddingBatchSequenceKey(example.QueryTokens, example.QueryMask)
		bucketIndex, ok := bucketByKey[key]
		if !ok {
			bucketIndex = len(buckets)
			bucketByKey[key] = bucketIndex
			buckets = append(buckets, queryBucket{})
		}
		buckets[bucketIndex].indexes = append(buckets[bucketIndex].indexes, idx)
	}
	if len(buckets) == len(order) {
		return order
	}
	active := make([]int, 0, len(buckets))
	for i := range buckets {
		active = append(active, i)
	}
	out := make([]int, 0, len(order))
	for len(active) > 0 {
		nextActive := make([]int, 0, len(active))
		for _, bucketIndex := range active {
			bucket := &buckets[bucketIndex]
			out = append(out, bucket.indexes[bucket.next])
			bucket.next++
			if bucket.next < len(bucket.indexes) {
				nextActive = append(nextActive, bucketIndex)
			}
		}
		active = nextActive
	}
	if len(out) != len(order) {
		return order
	}
	return out
}

func hardNegativeSourceWeightedOrder(trainSet []EmbeddingHardNegativeExample, order []int, batchSize int, weights map[string]int, lengthBucket bool) []int {
	weights = normalizeHardNegativeSourceWeights(weights)
	if len(trainSet) == 0 || len(order) < 2 || batchSize <= 0 || len(weights) == 0 {
		return order
	}
	type sourceQueue struct {
		indexes []int
		next    int
		weight  int
	}
	groups := []sourceQueue{}
	groupByKey := map[string]int{}
	for _, idx := range order {
		if idx < 0 || idx >= len(trainSet) {
			return order
		}
		key := hardNegativeSourceGroupKey(weights, trainSet[idx].Source)
		groupIndex, ok := groupByKey[key]
		if !ok {
			groupIndex = len(groups)
			groupByKey[key] = groupIndex
			groups = append(groups, sourceQueue{
				weight: hardNegativeSourceWeight(weights, trainSet[idx].Source),
			})
		}
		groups[groupIndex].indexes = append(groups[groupIndex].indexes, idx)
	}
	if len(groups) == 0 {
		return order
	}
	for i := range groups {
		if lengthBucket {
			bucketHardNegativeOrderByLength(trainSet, groups[i].indexes, batchSize)
		}
		groups[i].indexes = spreadHardNegativeOrderByQuery(trainSet, groups[i].indexes)
	}
	out := make([]int, 0, len(order))
	remaining := len(order)
	for remaining > 0 {
		batchStart := len(out)
		for len(out)-batchStart < batchSize && remaining > 0 {
			progressed := false
			for i := range groups {
				group := &groups[i]
				for repeat := 0; repeat < group.weight && len(out)-batchStart < batchSize; repeat++ {
					if group.next >= len(group.indexes) {
						break
					}
					out = append(out, group.indexes[group.next])
					group.next++
					remaining--
					progressed = true
				}
			}
			if !progressed {
				break
			}
		}
		if len(out) == batchStart {
			break
		}
	}
	if len(out) != len(order) {
		return order
	}
	return out
}

func hardNegativeSourceGroupKey(weights map[string]int, source string) string {
	exact := normalizedHardNegativeSource(source)
	if _, ok := weights[exact]; ok {
		return exact
	}
	return hardNegativeSourceFamily(source)
}

func hardNegativeSourceWeight(weights map[string]int, source string) int {
	if len(weights) == 0 {
		return 1
	}
	exact := normalizedHardNegativeSource(source)
	if weight := weights[exact]; weight > 0 {
		return weight
	}
	family := hardNegativeSourceFamily(source)
	if weight := weights[family]; weight > 0 {
		return weight
	}
	if weight := weights["*"]; weight > 0 {
		return weight
	}
	return 1
}

func hardNegativeTeacherTemperature(temperatures map[string]float32, source string, fallback float32) float32 {
	if fallback <= 0 {
		fallback = 1
	}
	if len(temperatures) == 0 {
		return fallback
	}
	exact := normalizedHardNegativeSource(source)
	if temp := temperatures[exact]; temp > 0 {
		return temp
	}
	family := hardNegativeSourceFamily(source)
	if temp := temperatures[family]; temp > 0 {
		return temp
	}
	if temp := temperatures["*"]; temp > 0 {
		return temp
	}
	return fallback
}

func hardNegativeSourceFamily(source string) string {
	source = normalizedHardNegativeSource(source)
	if idx := strings.IndexByte(source, ':'); idx > 0 {
		return source[:idx]
	}
	return source
}

func normalizedHardNegativeSource(source string) string {
	source = strings.ToLower(strings.TrimSpace(source))
	if source == "" {
		return "unknown"
	}
	return source
}

func normalizeHardNegativeSourceWeights(weights map[string]int) map[string]int {
	if len(weights) == 0 {
		return nil
	}
	out := map[string]int{}
	for source, weight := range weights {
		key := normalizedHardNegativeSource(source)
		if weight <= 0 {
			continue
		}
		out[key] = weight
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizeHardNegativeTeacherTemperatures(temperatures map[string]float32) map[string]float32 {
	if len(temperatures) == 0 {
		return nil
	}
	out := map[string]float32{}
	for source, temp := range temperatures {
		key := normalizedHardNegativeSource(source)
		if temp <= 0 {
			continue
		}
		out[key] = temp
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func plannedEvalPassCount(evalExamples, epochs, evalEvery int) int {
	if evalExamples <= 0 || epochs <= 0 {
		return 0
	}
	if evalEvery <= 0 {
		evalEvery = 1
	}
	return epochs/evalEvery + 1
}

func workloadEvalMode(evalExamples int, mode string) string {
	if evalExamples <= 0 {
		return ""
	}
	return mode
}

func (t *EmbeddingTrainer) runEpoch(trainSet []EmbeddingPairExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalExamples := 0
	batchIndex := 0
	totalBatches := batchCount(len(order), batchSize, 1)
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingPairExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		metrics, err := t.TrainStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(metrics.BatchSize)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalExamples += metrics.BatchSize
		batchIndex++
		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      metrics.BatchSize,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalExamples),
			EpochTrainPairs:    int64(totalExamples),
			PlannedEpochPairs:  int64(len(order)),
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		})
	}
	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no examples")
	}
	inv := float32(1) / float32(totalExamples)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * inv,
		AverageScore: totalScore * inv,
		BatchSize:    totalExamples,
	}, nil
}

type contrastiveEpochBatchHook func(EmbeddingTrainProgress) error

func (t *EmbeddingTrainer) runContrastiveEpoch(trainSet []EmbeddingContrastiveExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time, afterBatch contrastiveEpochBatchHook) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalExamples := 0
	totalTrainExamples := 0
	var totalPairs int64
	batchIndex := 0
	totalBatches, plannedEpochPairs := contrastiveBatchWork(len(order), batchSize)
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		if end-start < 2 {
			break
		}
		batch := make([]EmbeddingContrastiveExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		metrics, err := t.TrainContrastiveStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(metrics.BatchSize)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalExamples += metrics.BatchSize
		totalTrainExamples += end - start
		totalPairs += int64(metrics.BatchSize)
		batchIndex++
		progress := EmbeddingTrainProgress{
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		}
		maybeReportTrainProgress(cfg, progress)
		if afterBatch != nil {
			if err := afterBatch(progress); err != nil {
				return EmbeddingTrainMetrics{}, err
			}
		}
	}
	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable contrastive batches")
	}
	inv := float32(1) / float32(totalExamples)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * inv,
		AverageScore: totalScore * inv,
		BatchSize:    totalExamples,
	}, nil
}

func (t *EmbeddingTrainer) runHardNegativeEpoch(trainSet []EmbeddingHardNegativeExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time, afterBatch contrastiveEpochBatchHook) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalExamples := 0
	totalTrainExamples := 0
	var totalPairs int64
	batchIndex := 0
	if len(cfg.HardNegativeSourceWeights) == 0 {
		order = spreadHardNegativeOrderByQuery(trainSet, order)
	}
	totalBatches, plannedEpochPairs := hardNegativeBatchWork(len(order), batchSize, cfg.HardNegativesPerQuery, cfg.ContrastiveLoss, cfg.TeacherLossWeight)
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingHardNegativeExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		if hardNegativeBatchPairCount(batch) < 2 {
			break
		}
		metrics, err := t.TrainHardNegativeContrastiveStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(metrics.BatchSize)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalExamples += metrics.BatchSize
		totalTrainExamples += end - start
		totalPairs += int64(metrics.BatchSize)
		batchIndex++
		progress := EmbeddingTrainProgress{
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		}
		maybeReportTrainProgress(cfg, progress)
		if afterBatch != nil {
			if err := afterBatch(progress); err != nil {
				return EmbeddingTrainMetrics{}, err
			}
		}
	}
	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable hard-negative batches")
	}
	inv := float32(1) / float32(totalExamples)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * inv,
		AverageScore: totalScore * inv,
		BatchSize:    totalExamples,
	}, nil
}

func (t *EmbeddingTrainer) restoreCheckpoint(checkpoint EmbeddingTrainCheckpoint) error {
	restored, err := NewEmbeddingTrainerFromCheckpoint(t.module, checkpoint)
	if err != nil {
		return err
	}
	*t = *restored
	return nil
}

func maybeReportTrainProgress(cfg EmbeddingTrainRunConfig, progress EmbeddingTrainProgress) {
	if cfg.Progress == nil || cfg.ProgressEverySteps <= 0 || progress.Batch <= 0 {
		return
	}
	if progress.Batch%cfg.ProgressEverySteps != 0 && progress.Batch != progress.Batches {
		return
	}
	cfg.Progress(progress)
}

func bucketContrastiveOrderByLength(trainSet []EmbeddingContrastiveExample, order []int, batchSize int) {
	if len(trainSet) == 0 || len(order) < 2 || batchSize <= 1 {
		return
	}
	windowSize := contrastiveLengthBucketWindow(batchSize, len(order))
	for start := 0; start < len(order); start += windowSize {
		end := start + windowSize
		if end > len(order) {
			end = len(order)
		}
		window := order[start:end]
		sort.SliceStable(window, func(i, j int) bool {
			left := contrastiveExampleSortLength(trainSet[window[i]])
			right := contrastiveExampleSortLength(trainSet[window[j]])
			return left < right
		})
	}
}

func bucketHardNegativeOrderByLength(trainSet []EmbeddingHardNegativeExample, order []int, batchSize int) {
	if len(trainSet) == 0 || len(order) == 0 || batchSize <= 0 {
		return
	}
	windowSize := contrastiveLengthBucketWindow(batchSize, len(order))
	for start := 0; start < len(order); start += windowSize {
		end := start + windowSize
		if end > len(order) {
			end = len(order)
		}
		window := order[start:end]
		sort.SliceStable(window, func(i, j int) bool {
			left := hardNegativeExampleSortLength(trainSet[window[i]])
			right := hardNegativeExampleSortLength(trainSet[window[j]])
			return left < right
		})
	}
}

func contrastiveLengthBucketWindow(batchSize, total int) int {
	if batchSize <= 1 || total <= 0 {
		return 0
	}
	windowSize := batchSize * 4
	if raw := trainEnv("MANTA_TRAIN_LENGTH_BUCKET_WINDOW"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			windowSize = parsed
		}
	}
	if windowSize < batchSize {
		windowSize = batchSize
	}
	if windowSize > total {
		windowSize = total
	}
	return windowSize
}

func contrastiveExampleSortLength(example EmbeddingContrastiveExample) int {
	length := len(example.QueryTokens)
	if len(example.PositiveTokens) > length {
		length = len(example.PositiveTokens)
	}
	return length
}

func hardNegativeExampleSortLength(example EmbeddingHardNegativeExample) int {
	length := len(example.QueryTokens)
	if len(example.PositiveTokens) > length {
		length = len(example.PositiveTokens)
	}
	for _, tokens := range example.NegativeTokens {
		if len(tokens) > length {
			length = len(tokens)
		}
	}
	return length
}

func expandContrastiveExamples(examples []EmbeddingContrastiveExample) []EmbeddingPairExample {
	if len(examples) == 0 {
		return nil
	}
	out := make([]EmbeddingPairExample, 0, len(examples)*len(examples))
	for i, example := range examples {
		out = append(out, EmbeddingPairExample{
			Source:      example.Source,
			LeftTokens:  append([]int32(nil), example.QueryTokens...),
			RightTokens: append([]int32(nil), example.PositiveTokens...),
			LeftMask:    append([]int32(nil), example.QueryMask...),
			RightMask:   append([]int32(nil), example.PositiveMask...),
			Target:      1,
		})
		for j, negative := range examples {
			if i == j {
				continue
			}
			out = append(out, EmbeddingPairExample{
				Source:      example.Source,
				LeftTokens:  append([]int32(nil), example.QueryTokens...),
				RightTokens: append([]int32(nil), negative.PositiveTokens...),
				LeftMask:    append([]int32(nil), example.QueryMask...),
				RightMask:   append([]int32(nil), negative.PositiveMask...),
				Target:      -1,
			})
		}
	}
	return out
}

func normalizedTrainRunConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if cfg.EvalOnly {
		cfg.Epochs = 0
	} else if cfg.Epochs == 0 {
		cfg.Epochs = 1
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 8
	}
	if cfg.EvalEveryEpoch == 0 {
		cfg.EvalEveryEpoch = 1
	}
	if cfg.SelectMetric == "" {
		cfg.SelectMetric = "score_margin"
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	if cfg.HardNegativesPerQuery == 0 {
		cfg.HardNegativesPerQuery = 1
	}
	cfg.GroupedLossWeight = effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
	cfg.HardNegativeSourceWeights = normalizeHardNegativeSourceWeights(cfg.HardNegativeSourceWeights)
	cfg.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
	return cfg
}

func (t *EmbeddingTrainer) applyTrainRunOverrides(cfg EmbeddingTrainRunConfig) error {
	if t == nil {
		return nil
	}
	next := t.config
	changed := false
	if cfg.LearningRate > 0 {
		next.LearningRate = cfg.LearningRate
		changed = true
	}
	if cfg.ContrastiveLoss != "" {
		next.ContrastiveLoss = cfg.ContrastiveLoss
		changed = true
	}
	if cfg.Temperature > 0 {
		next.Temperature = cfg.Temperature
		changed = true
	}
	if cfg.GroupedLossWeight > 0 {
		next.GroupedLossWeight = cfg.GroupedLossWeight
		changed = true
	}
	if cfg.TeacherLossWeight > 0 {
		next.TeacherLossWeight = cfg.TeacherLossWeight
		changed = true
	}
	if cfg.TeacherTemperature > 0 {
		next.TeacherTemperature = cfg.TeacherTemperature
		changed = true
	}
	if len(cfg.TeacherSourceTemperatures) > 0 {
		next.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
		changed = true
	}
	if !changed {
		return nil
	}
	next = normalizedTrainConfig(next, t.tokenParam, t.attnQParam, t.attnKParam, t.attnVParam, t.attnOParam, t.hiddenParam, t.projParam)
	if err := validateTrainConfig(next); err != nil {
		return err
	}
	t.config = next
	return nil
}

func (t *EmbeddingTrainer) syncTrainRunObjectiveConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if t == nil {
		return cfg
	}
	if cfg.ContrastiveLoss == "" {
		cfg.ContrastiveLoss = t.config.ContrastiveLoss
	}
	if cfg.GroupedLossWeight == 0 {
		cfg.GroupedLossWeight = t.config.GroupedLossWeight
	}
	if cfg.TeacherLossWeight == 0 {
		cfg.TeacherLossWeight = t.config.TeacherLossWeight
	}
	if cfg.TeacherTemperature == 0 {
		cfg.TeacherTemperature = t.config.TeacherTemperature
	}
	if len(cfg.TeacherSourceTemperatures) == 0 && len(t.config.TeacherSourceTemperatures) > 0 {
		cfg.TeacherSourceTemperatures = t.config.TeacherSourceTemperatures
	}
	cfg.GroupedLossWeight = effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
	cfg.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
	return cfg
}

func validTrainSelectionMetric(metric string) bool {
	switch metric {
	case "loss", "pair_accuracy", "threshold_accuracy", "score_margin", "auc", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr", "mean_positive_rank", "mean_rank":
		return true
	default:
		return false
	}
}

func betterEvalMetrics(current, best EmbeddingEvalMetrics, metric string, minDelta float32) bool {
	const eps = 1e-6
	primaryDelta := float32(math.Max(float64(minDelta), eps))
	switch metric {
	case "loss":
		if current.Loss < best.Loss-primaryDelta {
			return true
		}
		if math.Abs(float64(current.Loss-best.Loss)) <= eps {
			if current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps && current.PairAccuracy > best.PairAccuracy+float32(eps) {
				return true
			}
		}
	case "pair_accuracy":
		if current.PairAccuracy > best.PairAccuracy+primaryDelta {
			return true
		}
		if math.Abs(float64(current.PairAccuracy-best.PairAccuracy)) <= eps {
			if current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps && current.Loss < best.Loss-float32(eps) {
				return true
			}
		}
	case "threshold_accuracy":
		if current.ThresholdAccuracy > best.ThresholdAccuracy+primaryDelta {
			return true
		}
		if math.Abs(float64(current.ThresholdAccuracy-best.ThresholdAccuracy)) <= eps {
			if current.ROCAUC > best.ROCAUC+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ROCAUC-best.ROCAUC)) <= eps && current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
		}
	case "auc", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr":
		currentRankMetric := evalRankMetric(current, metric)
		bestRankMetric := evalRankMetric(best, metric)
		if currentRankMetric > bestRankMetric+primaryDelta {
			return true
		}
		if math.Abs(float64(currentRankMetric-bestRankMetric)) <= eps {
			if current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps && current.Loss < best.Loss-float32(eps) {
				return true
			}
		}
	case "mean_positive_rank", "mean_rank":
		if current.MeanPositiveRank < best.MeanPositiveRank-primaryDelta {
			return true
		}
		if math.Abs(float64(current.MeanPositiveRank-best.MeanPositiveRank)) <= eps {
			if current.Top1Accuracy > best.Top1Accuracy+float32(eps) {
				return true
			}
			if math.Abs(float64(current.Top1Accuracy-best.Top1Accuracy)) <= eps && current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
		}
	default:
		if current.ScoreMargin > best.ScoreMargin+primaryDelta {
			return true
		}
		if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps {
			if current.PairAccuracy > best.PairAccuracy+float32(eps) {
				return true
			}
			if math.Abs(float64(current.PairAccuracy-best.PairAccuracy)) <= eps && current.Loss < best.Loss-float32(eps) {
				return true
			}
		}
	}
	return false
}

func evalRankMetric(metrics EmbeddingEvalMetrics, metric string) float32 {
	switch metric {
	case "auc":
		return metrics.ROCAUC
	case "top5_accuracy":
		return metrics.Top5Accuracy
	case "top10_accuracy":
		return metrics.Top10Accuracy
	case "mrr":
		return metrics.MeanReciprocalRank
	default:
		return metrics.Top1Accuracy
	}
}

func cloneEvalMetrics(metrics EmbeddingEvalMetrics) *EmbeddingEvalMetrics {
	out := metrics
	return &out
}
