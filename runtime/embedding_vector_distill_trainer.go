package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
	"runtime/debug"
	"time"
)

// vectorDistillProjectionState holds the train-time-only 128→384 distillation
// projection matrix and its AdamW moments.  It is NEVER persisted to disk;
// the served model uses the native 128-d student output exclusively.
type vectorDistillProjectionState struct {
	W        []float32 // shape [inputDim * teacherDim], row-major
	Mom1     []float32 // AdamW first moment
	Mom2     []float32 // AdamW second moment
	InputDim int
	OutDim   int
	Step     int // AdamW step counter (separate from student step)
}

// newVectorDistillProjectionState allocates and Kaiming-initialises a fresh
// projection matrix.
func newVectorDistillProjectionState(inputDim, outDim int, rng *rand.Rand) *vectorDistillProjectionState {
	n := inputDim * outDim
	W := make([]float32, n)
	// Kaiming uniform: fan_in = inputDim, bound = sqrt(1/inputDim)
	bound := float32(math.Sqrt(1.0 / float64(inputDim)))
	for i := range W {
		W[i] = (rng.Float32()*2 - 1) * bound
	}
	return &vectorDistillProjectionState{
		W:        W,
		Mom1:     make([]float32, n),
		Mom2:     make([]float32, n),
		InputDim: inputDim,
		OutDim:   outDim,
	}
}

// FitVectorDistill trains the compact student using MSE+cosine distillation
// from a BGE-style teacher's 384-d embeddings.  The distillation projection
// (128→384) lives only in this function's scope and is never written to disk.
// The retrieval eval gate embeds via the native 128-d student path.
func (t *EmbeddingTrainer) FitVectorDistill(
	trainSet []EmbeddingTokenizedVectorDistillExample,
	cfg EmbeddingTrainRunConfig,
) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if cfg.EvalOnly {
		// eval-only path: retrieval eval only — no training data needed
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("vector distillation training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
		if err := validateVectorDistillResearchGates(trainSet, cfg.AllowResearchOnlyVectorDistill); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
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
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              EstimateVectorDistillTrainWorkload(len(trainSet), 0, cfg),
	}

	// --- eval-only path ---
	if cfg.EvalOnly {
		evalStart := time.Now()
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, 0, 0, runStart)
		if retrievalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("retrieval eval: %w", err)
		} else if retrievalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalMetrics)
			summary.LastEval = cloneEvalMetrics(*retrievalMetrics)
			summary.BestEval = cloneEvalMetrics(*retrievalMetrics)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		appendTrainEvalHistory(&summary, 0, t.step, "eval_only", true, summary.FinalEval, nil, nil)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	// determine teacher dim from the first example
	if len(trainSet[0].TeacherVector) == 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("vector distillation: teacher_vector is empty in first example")
	}
	teacherDim := len(trainSet[0].TeacherVector)

	// distillation projection state — persists across epochs, discarded on return
	var proj *vectorDistillProjectionState
	rng := rand.New(rand.NewSource(cfg.Seed))

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)

	// recordEval runs the retrieval gate eval and returns updated metrics.
	recordEval := func(epoch int, trigger string) (*EmbeddingEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, 0, 0, runStart)
		var evalMetrics *EmbeddingEvalMetrics
		if retrievalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return nil, false, err
		} else if retrievalMetrics != nil {
			evalMetrics = retrievalMetrics
			summary.LastEval = cloneEvalMetrics(*retrievalMetrics)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, evalPass, 0, 0, runStart)
		improved := false
		selectable := evalMetrics != nil && retrievalSelectionMetric(cfg.SelectMetric)
		if selectable && (!haveBest || betterEvalMetrics(*evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta)) {
			checkpoint, err := t.Checkpoint()
			if err != nil {
				return nil, false, fmt.Errorf("checkpoint: %w", err)
			}
			bestCheckpoint = checkpoint
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(*evalMetrics)
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else if selectable {
			noImproveEvals++
		}
		appendTrainEvalHistory(&summary, epoch, t.step, trigger, improved, evalMetrics, nil, nil)
		return evalMetrics, improved, nil
	}

	// initial eval before any training (when retrieval eval is configured and restore-best is on)
	if t.retrievalEvalEnabled && retrievalSelectionMetric(cfg.SelectMetric) && cfg.RestoreBest {
		if _, _, err := recordEval(0, "initial"); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		}
		trainStart := time.Now()
		trainMetrics, updatedProj, err := t.runVectorDistillEpoch(trainSet, indices, cfg.BatchSize, cfg, teacherDim, proj, epoch, runStart)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		proj = updatedProj
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(len(indices))
		summary.Workload.ActualTrainExamples += int64(len(indices))

		if t.retrievalEvalEnabled && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, improved, err := recordEval(epoch, "epoch")
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.Improved = improved
			if t.retrievalEvalEnabled && retrievalSelectionMetric(cfg.SelectMetric) && !improved &&
				cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
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

	// final eval
	if t.retrievalEvalEnabled {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, 0, 0, runStart)
		if retrievalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final retrieval eval: %w", err)
		} else if retrievalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalMetrics)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, 0, 0, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, nil, nil)
		if summary.BestEval == nil {
			if summary.FinalEval != nil {
				summary.BestEval = cloneEvalMetrics(*summary.FinalEval)
			}
		}
	}

	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	if restored {
		summary.EndProfile = diffTrainProfile(preRestoreEndProfile, restoreStartProfile)
	} else {
		summary.EndProfile = t.TrainProfile()
	}
	summary.DeltaProfile = diffTrainProfile(summary.StartProfile, t.TrainProfile())
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// runVectorDistillEpoch runs one epoch of vector-distillation training.
// It returns the epoch metrics and the (possibly newly-created) projection state.
func (t *EmbeddingTrainer) runVectorDistillEpoch(
	trainSet []EmbeddingTokenizedVectorDistillExample,
	order []int,
	batchSize int,
	cfg EmbeddingTrainRunConfig,
	teacherDim int,
	proj *vectorDistillProjectionState,
	epoch int,
	runStart time.Time,
) (EmbeddingTrainMetrics, *vectorDistillProjectionState, error) {
	totalLoss := float32(0)
	totalExamples := 0
	totalBatches := batchCount(len(order), batchSize, 1)
	batchIndex := 0

	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingTokenizedVectorDistillExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}

		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Phase:              "train_start",
			Epoch:              epoch,
			Batch:              batchIndex + 1,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      len(batch),
			BatchPairs:         int64(len(batch)),
			EpochTrainExamples: int64(totalExamples),
			EpochTrainPairs:    int64(totalExamples),
			PlannedEpochPairs:  int64(len(order)),
			Elapsed:            time.Since(runStart),
		})

		metrics, updatedProj, err := t.trainVectorDistillBatch(batch, proj, teacherDim)
		if err != nil {
			return EmbeddingTrainMetrics{}, proj, err
		}
		proj = updatedProj

		totalLoss += metrics.Loss * float32(len(batch))
		totalExamples += len(batch)
		batchIndex++

		if n := trainMemReclaimEvery(); n > 0 && batchIndex%n == 0 {
			debug.FreeOSMemory()
		}

		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      len(batch),
			BatchPairs:         int64(len(batch)),
			EpochTrainExamples: int64(totalExamples),
			EpochTrainPairs:    int64(totalExamples),
			PlannedEpochPairs:  int64(len(order)),
			Loss:               metrics.Loss,
			Elapsed:            time.Since(runStart),
		})
	}

	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("vector distillation epoch has no usable examples")
	}
	return EmbeddingTrainMetrics{
		Loss:      totalLoss / float32(totalExamples),
		BatchSize: totalExamples,
	}, proj, nil
}

// trainVectorDistillBatch runs one optimizer step over a batch of
// (text, teacher_vector) examples.  The distillation projection state is
// returned (created on first call) so the caller can persist it across batches.
func (t *EmbeddingTrainer) trainVectorDistillBatch(
	batch []EmbeddingTokenizedVectorDistillExample,
	proj *vectorDistillProjectionState,
	teacherDim int,
) (EmbeddingTrainMetrics, *vectorDistillProjectionState, error) {
	if !t.isCompactTrainer() {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("vector distillation requires compact_transformer_v1")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("vector distillation batch is empty")
	}

	// Build sequence inputs
	inputs := make([]embeddingSequenceInput, len(batch))
	for i, ex := range batch {
		inputs[i] = embeddingSequenceInput{
			tokens: ex.Tokens,
			mask:   ex.Mask,
			role:   t.queryRoleIndex(),
			label:  fmt.Sprintf("batch %d", i),
		}
	}
	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("missing compact forward weights")
	}
	encoded, err := t.encodeSequenceInputs(inputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}
	defer t.releaseEncodedSequences(encoded)

	// Lazily initialise the projection on first call (student dim known now)
	if proj == nil {
		if len(encoded) == 0 || len(encoded[0].pooled) == 0 {
			return EmbeddingTrainMetrics{}, nil, fmt.Errorf("vector distillation: encoded sequence has empty pooled output")
		}
		studentDim := len(encoded[0].pooled)
		proj = newVectorDistillProjectionState(studentDim, teacherDim, rand.New(rand.NewSource(42)))
	}

	grads := newCompactEmbeddingGradState(t.compactState)
	gradW := make([]float32, proj.InputDim*proj.OutDim)
	totalLoss := float32(0)

	for i, enc := range encoded {
		student := enc.pooled
		teacher := batch[i].TeacherVector

		// Forward through projection: proj_vec = W @ student
		projVec := make([]float32, proj.OutDim)
		for si := 0; si < proj.InputDim; si++ {
			base := si * proj.OutDim
			sv := student[si]
			for k := 0; k < proj.OutDim; k++ {
				projVec[k] += proj.W[base+k] * sv
			}
		}

		// Loss and gradient w.r.t. projected vector
		lossResult, lerr := VectorDistillLossAndGrad(projVec, teacher)
		if lerr != nil {
			return EmbeddingTrainMetrics{}, proj, fmt.Errorf("example %d: %w", i, lerr)
		}
		totalLoss += lossResult.Loss

		// Backprop through projection → gradStudent and gradW
		gradStudent := make([]float32, proj.InputDim)
		accumulateVectorDistillProjectionGrads(student, lossResult.GradProj, proj.W, gradStudent, gradW)

		// Backprop gradStudent through the encoder
		if berr := t.backpropCompactEncodedSequence(enc, gradStudent, forward.compact, grads); berr != nil {
			return EmbeddingTrainMetrics{}, proj, fmt.Errorf("example %d backprop: %w", i, berr)
		}
	}

	// Scale: average over batch
	batchScale := float32(1) / float32(len(batch))

	// Update student parameters
	t.applyCompactOptimizerUpdates(grads, batchScale)

	// Update projection with its own AdamW (same LR/hyper-params as student)
	proj.Step++
	applyVectorDistillProjectionAdamW(
		proj.W, proj.Mom1, proj.Mom2, gradW,
		t.config.LearningRate, t.config.Beta1, t.config.Beta2, t.config.Epsilon, t.config.WeightDecay,
		proj.Step,
	)

	return EmbeddingTrainMetrics{
		Loss:      totalLoss * batchScale,
		BatchSize: len(batch),
	}, proj, nil
}

// EstimateVectorDistillTrainWorkload returns a planned-work summary for
// vector-distillation runs (one example = one training pair).
func EstimateVectorDistillTrainWorkload(trainExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches := batchCount(trainExamples, cfg.BatchSize, 1)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	trainPairsPerEpoch := int64(trainExamples)
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		evalPasses = 1 // retrieval-only eval
	} else {
		if cfg.RestoreBest && trainExamples > 0 {
			evalPasses++
		}
		if cfg.EvalEverySteps > 0 && trainExamples > 0 {
			evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
		}
		// initial eval pass if restoring best
		if cfg.RestoreBest && trainExamples > 0 {
			evalPasses++
		}
	}
	evalPairsPerPass := int64(evalExamples)
	return EmbeddingTrainWorkload{
		TrainMode:            "vector_distill",
		EvalMode:             workloadEvalModeVectorDistill(evalExamples, cfg),
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

func workloadEvalModeVectorDistill(evalExamples int, cfg EmbeddingTrainRunConfig) string {
	if cfg.RetrievalEval.CorpusPath != "" {
		return "retrieval"
	}
	if evalExamples > 0 {
		return "pairwise"
	}
	return ""
}
