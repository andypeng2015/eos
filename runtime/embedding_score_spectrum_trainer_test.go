package eosruntime

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbeddingTrainerTrainScoreSpectrumStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.Temperature = 0.05

	metrics, err := trainer.TrainScoreSpectrumStep(tinyEmbeddingScoreSpectrumDataset())
	if err != nil {
		t.Fatalf("train score-spectrum step: %v", err)
	}
	if metrics.BatchSize != 4 {
		t.Fatalf("batch size = %d, want 4 row-local query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 {
		t.Fatalf("loss = %f, want non-negative", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
}

func TestEmbeddingTrainerTrainScoreSpectrumStepFoldsSelectedOnlyPositive(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.Temperature = 0.05

	selected := 0
	batch := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("train score-spectrum selected-only step: %v", err)
	}
	if metrics.BatchSize != 2 || metrics.Loss < 0 {
		t.Fatalf("metrics = %+v, want one row with two candidates and non-negative loss", metrics)
	}
	if len(batch[0].PositiveIndexes) != 0 {
		t.Fatalf("caller batch positive indexes mutated to %+v, want unchanged empty slice", batch[0].PositiveIndexes)
	}
}

func TestEmbeddingTrainerEvaluateScoreSpectrumNativeMetricsAndNoOptimizerUpdate(t *testing.T) {
	selected0 := 0
	examples := []EmbeddingScoreSpectrumExample{
		{
			PositiveIndexes:       []int{0},
			SelectedPositiveIndex: &selected0,
			HardNegativeEligible:  []bool{false, true},
			TargetProbabilities:   []float32{1, 0},
		},
		{
			PositiveIndexes:       []int{0, 1},
			SelectedPositiveIndex: &selected0,
			HardNegativeEligible:  []bool{false, false, true},
			TargetProbabilities:   []float32{0.2, 0.7, 0.1},
		},
	}

	metrics, err := evaluateScoreSpectrumEncodings(
		[]*embeddingEncodedSequence{
			{pooled: []float32{1, 0}},
			{pooled: []float32{1, 1}},
		},
		[]*embeddingEncodedSequence{
			{pooled: []float32{1, 0}},
			{pooled: []float32{0, 1}},
			{pooled: []float32{1, 0}},
			{pooled: []float32{1, 1}},
			{pooled: []float32{0, 1}},
		},
		[]embeddingCandidateSpan{{Start: 0, End: 2}, {Start: 2, End: 5}},
		examples,
		1,
	)
	if err != nil {
		t.Fatalf("evaluate score-spectrum: %v", err)
	}
	if metrics.RowCount != 2 || metrics.CandidateCount != 5 {
		t.Fatalf("row/candidate counts = %d/%d, want 2/5", metrics.RowCount, metrics.CandidateCount)
	}
	if metrics.AnyPositiveTop1 != 1 || metrics.AnyPositiveRowCount != 2 {
		t.Fatalf("any-positive top1/count = %v/%d, want 1/2", metrics.AnyPositiveTop1, metrics.AnyPositiveRowCount)
	}
	if metrics.OriginalPositiveTop1 != 0.5 || metrics.OriginalPositiveRowCount != 2 {
		t.Fatalf("original-positive top1/count = %v/%d, want 0.5/2", metrics.OriginalPositiveTop1, metrics.OriginalPositiveRowCount)
	}
	if metrics.AlternateRelevantRecovery != 1 || metrics.AlternateRecoveryRowCount != 1 {
		t.Fatalf("alternate recovery/count = %v/%d, want 1/1", metrics.AlternateRelevantRecovery, metrics.AlternateRecoveryRowCount)
	}
	wantMargin := float32((1 + (1 - 1/math.Sqrt2)) / 2)
	if math.Abs(float64(metrics.BestPositiveHardestNegativeMargin-wantMargin)) > 1e-5 || metrics.MarginRowCount != 2 {
		t.Fatalf("margin/count = %v/%d, want %v/2", metrics.BestPositiveHardestNegativeMargin, metrics.MarginRowCount, wantMargin)
	}
	if metrics.TargetCrossEntropy <= 0 || metrics.TargetKL < 0 || metrics.Loss < 0 {
		t.Fatalf("loss/ce/kl = %v/%v/%v, want valid positive metrics", metrics.Loss, metrics.TargetCrossEntropy, metrics.TargetKL)
	}

	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	startProfile := trainer.TrainProfile()
	startStep := trainer.step
	if _, err := trainer.EvaluateScoreSpectrum(tinyEmbeddingScoreSpectrumDataset()); err != nil {
		t.Fatalf("evaluate score-spectrum through trainer: %v", err)
	}
	if trainer.step != startStep {
		t.Fatalf("step = %d, want unchanged %d", trainer.step, startStep)
	}
	endProfile := trainer.TrainProfile()
	if endProfile.Step != startProfile.Step || endProfile.Optimizer.UpdateCalls != startProfile.Optimizer.UpdateCalls {
		t.Fatalf("optimizer state changed after eval: start step/update=%d/%d end step/update=%d/%d", startProfile.Step, startProfile.Optimizer.UpdateCalls, endProfile.Step, endProfile.Optimizer.UpdateCalls)
	}
}

func TestEmbeddingTrainerEvaluateScoreSpectrumFoldsSelectedOnlyPositive(t *testing.T) {
	selected := 0
	examples := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	metrics, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).EvaluateScoreSpectrum(examples)
	if err != nil {
		t.Fatalf("evaluate score-spectrum selected-only row: %v", err)
	}
	if metrics.RowCount != 1 || metrics.CandidateCount != 2 {
		t.Fatalf("row/candidate counts = %d/%d, want 1/2", metrics.RowCount, metrics.CandidateCount)
	}
	if metrics.AnyPositiveRowCount != 1 || metrics.OriginalPositiveRowCount != 1 {
		t.Fatalf("positive denominators = any %d original %d, want 1/1", metrics.AnyPositiveRowCount, metrics.OriginalPositiveRowCount)
	}
	if len(examples[0].PositiveIndexes) != 0 {
		t.Fatalf("caller examples positive indexes mutated to %+v, want unchanged empty slice", examples[0].PositiveIndexes)
	}
}

func TestEmbeddingTrainerTrainScoreSpectrumRejectsPositiveMarkedHardEligible(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	batch := tinyEmbeddingScoreSpectrumDataset()
	batch[0].HardNegativeEligible[0] = true

	_, err := trainer.TrainScoreSpectrumStep(batch)
	if err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("error = %v, want positive hard-eligible rejection", err)
	}
	if trainer.step != 0 {
		t.Fatalf("step = %d, want no optimizer update", trainer.step)
	}
}

func TestEmbeddingTrainerScoreSpectrumRejectsSelectedPositiveMarkedHardEligible(t *testing.T) {
	selected := 0
	batch := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{true, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	if _, err := trainer.TrainScoreSpectrumStep(batch); err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("train error = %v, want selected-positive hard-eligible rejection", err)
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).EvaluateScoreSpectrum(batch); err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("eval error = %v, want selected-positive hard-eligible rejection", err)
	}
}

func TestEmbeddingTrainerScoreSpectrumExcludedHardCandidatesHaveZeroGradient(t *testing.T) {
	queries := []*embeddingEncodedSequence{{pooled: []float32{1, 0}}}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{-1, 0}},
	}
	queryGrads := [][]float32{{0, 0}}
	candidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	examples := []EmbeddingScoreSpectrumExample{{
		PositiveIndexes:      []int{0},
		HardNegativeEligible: []bool{false, true, false},
		TargetProbabilities:  []float32{1, 0, 0},
		HardLossWeight:       1,
		SoftLossWeight:       0,
	}}

	_, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, EmbeddingTrainConfig{Temperature: 1, ScoreSpectrumLossMode: ScoreSpectrumLossModeHardSoft, ScoreSpectrumRecoveryTopK: 4, ScoreSpectrumRecoveryTau: 1}, queryGrads, candidateGrads)
	if err != nil {
		t.Fatalf("accumulate score-spectrum grads: %v", err)
	}
	if candidateGrads[2][0] != 0 || candidateGrads[2][1] != 0 {
		t.Fatalf("excluded candidate grad = %v, want zero", candidateGrads[2])
	}
}

func TestEmbeddingTrainerScoreSpectrumRecoveryChangesLossAndGradients(t *testing.T) {
	queries := []*embeddingEncodedSequence{{pooled: []float32{1, 0}}}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{-1, 0}},
	}
	examples := []EmbeddingScoreSpectrumExample{{
		PositiveIndexes:      []int{0},
		HardNegativeEligible: []bool{false, true, true},
		TargetProbabilities:  []float32{1, 0, 0},
	}}
	baseQueryGrads := [][]float32{{0, 0}}
	baseCandidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	baseLoss, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, EmbeddingTrainConfig{
		Temperature:               0.5,
		ScoreSpectrumLossMode:     ScoreSpectrumLossModeHardSoft,
		ScoreSpectrumRecoveryTopK: 4,
		ScoreSpectrumRecoveryTau:  0.5,
	}, baseQueryGrads, baseCandidateGrads)
	if err != nil {
		t.Fatalf("base score-spectrum grads: %v", err)
	}

	recoveryQueryGrads := [][]float32{{0, 0}}
	recoveryCandidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	recoveryLoss, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, EmbeddingTrainConfig{
		Temperature:                 0.5,
		ScoreSpectrumLossMode:       ScoreSpectrumLossModeHardSoftRecovery,
		ScoreSpectrumRecoveryWeight: 1,
		ScoreSpectrumRecoveryMargin: 0.1,
		ScoreSpectrumRecoveryTopK:   1,
		ScoreSpectrumRecoveryTau:    0.25,
	}, recoveryQueryGrads, recoveryCandidateGrads)
	if err != nil {
		t.Fatalf("recovery score-spectrum grads: %v", err)
	}
	if recoveryLoss <= baseLoss {
		t.Fatalf("recovery loss = %f, want above base %f", recoveryLoss, baseLoss)
	}
	if recoveryCandidateGrads[1][0] == baseCandidateGrads[1][0] && recoveryCandidateGrads[1][1] == baseCandidateGrads[1][1] {
		t.Fatalf("selected hard-negative gradient did not change: base=%v recovery=%v", baseCandidateGrads[1], recoveryCandidateGrads[1])
	}
	if recoveryCandidateGrads[2][0] != baseCandidateGrads[2][0] || recoveryCandidateGrads[2][1] != baseCandidateGrads[2][1] {
		t.Fatalf("non-topK hard-negative recovery gradient changed: base=%v recovery=%v", baseCandidateGrads[2], recoveryCandidateGrads[2])
	}
}

func TestEmbeddingTrainerScoreSpectrumRecoveryFoldsSelectedOnlyPositive(t *testing.T) {
	selected := 0
	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateTokens:         [][]int32{{0}, {1}},
		CandidateMasks:          [][]int32{{1}, {1}},
		SelectedPositiveIndex:   &selected,
		HardNegativeEligible:    []bool{false, true},
		TargetProbabilities:     []float32{1, 0},
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
		RecoveryLossWeight:      2,
	}}
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ScoreSpectrumLossMode = ScoreSpectrumLossModeRecovery
	trainer.config.ScoreSpectrumRecoveryWeight = 1
	trainer.config.ScoreSpectrumRecoveryTopK = 1
	trainer.config.ScoreSpectrumRecoveryTau = 0.05
	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("train recovery selected-only score-spectrum row: %v", err)
	}
	if metrics.Loss <= 0 || trainer.step != 1 {
		t.Fatalf("metrics/step = %+v/%d, want positive loss and one update", metrics, trainer.step)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumEvalOnlyDoesNotUpdate(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	startStep := trainer.step

	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), tinyEncoderPairDataset(), EmbeddingTrainRunConfig{
		EvalOnly: true,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum eval-only: %v", err)
	}
	if trainer.step != startStep || summary.StepsRun != 0 || summary.DeltaProfile.Step != 0 {
		t.Fatalf("step changed trainer=%d start=%d stepsRun=%d delta=%d", trainer.step, startStep, summary.StepsRun, summary.DeltaProfile.Step)
	}
	if summary.FinalEval == nil || summary.Workload.ActualEvalPairs == 0 {
		t.Fatalf("missing eval metrics/workload: final=%v actualEvalPairs=%d", summary.FinalEval, summary.Workload.ActualEvalPairs)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumPreservesInheritedRecoveryWeight(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.ScoreSpectrumLossMode = ScoreSpectrumLossModeRecovery
	trainer.config.ScoreSpectrumRecoveryWeight = 1.5
	trainer.config.ScoreSpectrumRecoveryTopK = 1
	trainer.config.ScoreSpectrumRecoveryTau = 0.05

	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum with inherited recovery weight: %v", err)
	}
	if summary.Config.ScoreSpectrumLossMode != ScoreSpectrumLossModeRecovery {
		t.Fatalf("summary loss mode = %q, want %q", summary.Config.ScoreSpectrumLossMode, ScoreSpectrumLossModeRecovery)
	}
	if summary.Config.ScoreSpectrumRecoveryWeight != 1.5 {
		t.Fatalf("summary recovery weight = %v, want 1.5", summary.Config.ScoreSpectrumRecoveryWeight)
	}
	if trainer.config.ScoreSpectrumRecoveryWeight != 1.5 {
		t.Fatalf("trainer recovery weight = %v, want 1.5", trainer.config.ScoreSpectrumRecoveryWeight)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumResearchOnlyRequiresExplicitFlag(t *testing.T) {
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyResearchOnlyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "AllowResearchOnlyScoreSpectrum") {
		t.Fatalf("error = %v, want explicit research flag rejection", err)
	}

	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyResearchOnlyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:                         1,
		BatchSize:                      2,
		AllowResearchOnlyScoreSpectrum: true,
	})
	if err != nil {
		t.Fatalf("fit research-only score-spectrum: %v", err)
	}
	if summary.StepsRun != 1 || summary.Workload.ActualTrainExamples != 2 || summary.Workload.ActualTrainPairs != 4 {
		t.Fatalf("steps/examples/pairs = %d/%d/%d, want 1/2/4", summary.StepsRun, summary.Workload.ActualTrainExamples, summary.Workload.ActualTrainPairs)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumUsesQueryAverageLossAndPairWorkload(t *testing.T) {
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum: %v", err)
	}
	if summary.FinalTrain.BatchSize != 2 {
		t.Fatalf("final train batch size = %d, want 2 query examples", summary.FinalTrain.BatchSize)
	}
	if summary.Workload.ActualTrainPairs != 4 {
		t.Fatalf("actual train pairs = %d, want 4 row-local candidate scores", summary.Workload.ActualTrainPairs)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumRejectsSingleTargetObjectives(t *testing.T) {
	tests := []struct {
		name string
		cfg  EmbeddingTrainRunConfig
		want string
	}{
		{
			name: "matryoshka",
			cfg:  EmbeddingTrainRunConfig{MatryoshkaDims: []int{2}, MatryoshkaWeights: []float32{1}},
			want: "matryoshka",
		},
		{
			name: "turboquant prefix",
			cfg:  EmbeddingTrainRunConfig{TurboQuantPrefixBits: []int{4}},
			want: "turboquant prefix",
		},
		{
			name: "turboquant compact",
			cfg:  EmbeddingTrainRunConfig{TurboQuantCompactObjectives: []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 4, Weight: 1}}},
			want: "turboquant compact",
		},
		{
			name: "turboquant rank margin",
			cfg:  EmbeddingTrainRunConfig{TurboQuantRankMarginObjectives: []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 4, Weight: 1}}},
			want: "turboquant rank-margin",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			cfg.Epochs = 1
			cfg.BatchSize = 2
			_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, cfg)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestEmbeddingTrainerFitScoreSpectrumIsolatesInheritedCompactPackageObjectives(t *testing.T) {
	source := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	source.config.MatryoshkaDims = []int{2}
	source.config.MatryoshkaWeights = []float32{1}
	source.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
	source.config.TurboQuantPrefixSeed = 11
	artifactPath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	if _, err := source.WriteTrainingPackage(artifactPath); err != nil {
		t.Fatalf("write source package: %v", err)
	}

	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		t.Fatalf("load source package: %v", err)
	}
	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum from compact package: %v", err)
	}
	if len(summary.Config.TurboQuantCompactObjectives) != 0 || len(trainer.config.TurboQuantCompactObjectives) != 0 {
		t.Fatalf("compact objectives were not isolated: summary=%+v trainer=%+v", summary.Config.TurboQuantCompactObjectives, trainer.config.TurboQuantCompactObjectives)
	}
	if len(summary.Config.MatryoshkaDims) != 0 || len(trainer.config.MatryoshkaDims) != 0 {
		t.Fatalf("matryoshka objectives were not isolated: summary=%v trainer=%v", summary.Config.MatryoshkaDims, trainer.config.MatryoshkaDims)
	}
	if trainer.config.TurboQuantPrefixSeed != 0 {
		t.Fatalf("prefix seed = %d, want cleared", trainer.config.TurboQuantPrefixSeed)
	}

	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		t.Fatalf("rewrite isolated package: %v", err)
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		t.Fatalf("read train manifest: %v", err)
	}
	if len(trainManifest.Config.TurboQuantCompactObjectives) != 0 {
		t.Fatalf("train manifest compact objectives = %+v, want cleared", trainManifest.Config.TurboQuantCompactObjectives)
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if len(checkpoint.Config.TurboQuantCompactObjectives) != 0 {
		t.Fatalf("checkpoint compact objectives = %+v, want cleared", checkpoint.Config.TurboQuantCompactObjectives)
	}
	packageManifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if err := packageManifest.VerifyFiles(map[string]string{
		"artifact":           paths.ArtifactPath,
		"embedding_manifest": paths.EmbeddingManifestPath,
		"weights":            paths.WeightFilePath,
		"memory_plan":        paths.MemoryPlanPath,
		"train_manifest":     paths.TrainManifestPath,
		"checkpoint":         paths.CheckpointPath,
		"train_profile":      paths.TrainProfilePath,
	}); err != nil {
		t.Fatalf("verify package manifest: %v", err)
	}
	for _, want := range []string{"matryoshka", "turboquant_compact_objectives", "turboquant_prefix_seed"} {
		if !hasScoreSpectrumObjectiveName(packageManifest.ScoreSpectrum.AutoClearedObjectives, want) {
			t.Fatalf("auto-cleared objectives = %v, missing %q", packageManifest.ScoreSpectrum.AutoClearedObjectives, want)
		}
		if !hasScoreSpectrumObjectiveName(trainManifest.ScoreSpectrum.IsolatedInheritedObjectives, want) {
			t.Fatalf("isolated objectives = %v, missing %q", trainManifest.ScoreSpectrum.IsolatedInheritedObjectives, want)
		}
	}
}

func TestEmbeddingTrainerFitScoreSpectrumIsolatesInheritedPrefixRankAndMatryoshkaObjectives(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.MatryoshkaDims = []int{2}
	trainer.config.MatryoshkaWeights = []float32{0.5}
	trainer.config.TurboQuantPrefixBits = []int{2}
	trainer.config.TurboQuantPrefixObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
	trainer.config.TurboQuantPrefixWeight = 0.75
	trainer.config.TurboQuantPrefixSeed = 17
	trainer.config.TurboQuantPrefixScoreMode = TurboQuantPrefixScoreModePreparedIP
	trainer.config.TurboQuantRankMarginObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.5}}
	trainer.config.TurboQuantRankMargin = 0.03

	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum with inherited objectives: %v", err)
	}
	if err := validateScoreSpectrumTrainerConfig(trainer.config); err != nil {
		t.Fatalf("trainer config still has incompatible objectives: %v", err)
	}
	if err := validateScoreSpectrumRunConfig(summary.Config); err != nil {
		t.Fatalf("summary config still has incompatible objectives: %v", err)
	}
	for _, want := range []string{"matryoshka", "turboquant_prefix_bits", "turboquant_prefix_objectives", "turboquant_prefix_weight", "turboquant_prefix_seed", "turboquant_prefix_score_mode", "turboquant_rank_margin_objectives", "turboquant_rank_margin"} {
		if !hasScoreSpectrumObjectiveName(trainer.scoreSpectrumLineage.AutoClearedObjectives, want) {
			t.Fatalf("auto-cleared objectives = %v, missing %q", trainer.scoreSpectrumLineage.AutoClearedObjectives, want)
		}
	}
}

func hasScoreSpectrumObjectiveName(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestEstimateScoreSpectrumTrainWorkloadCountsRowLocalCandidates(t *testing.T) {
	workload := EstimateScoreSpectrumTrainWorkload(tinyEmbeddingScoreSpectrumDataset(), 3, EmbeddingTrainRunConfig{
		Epochs:    2,
		BatchSize: 2,
	})
	if workload.TrainMode != "score_spectrum_grouped" {
		t.Fatalf("train mode = %q", workload.TrainMode)
	}
	if workload.TrainPairsPerEpoch != 4 || workload.PlannedTrainPairs != 8 {
		t.Fatalf("train pairs per epoch/planned = %d/%d, want 4/8", workload.TrainPairsPerEpoch, workload.PlannedTrainPairs)
	}
}

func TestEstimateScoreSpectrumTrainWorkloadMixedEvalModeUsesActualEvalSets(t *testing.T) {
	workload := estimateScoreSpectrumTrainWorkload(tinyEmbeddingScoreSpectrumDataset(), 2, 1, 4, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if workload.EvalMode != "mixed" {
		t.Fatalf("eval mode = %q, want mixed", workload.EvalMode)
	}

	workload = estimateScoreSpectrumTrainWorkload(tinyEmbeddingScoreSpectrumDataset(), 0, 1, 2, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if workload.EvalMode != "score_spectrum_grouped" {
		t.Fatalf("eval mode = %q, want score_spectrum_grouped", workload.EvalMode)
	}
}

func tinyEmbeddingScoreSpectrumDataset() []EmbeddingScoreSpectrumExample {
	return []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
		{
			QueryTokens:             []int32{1},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{1}, {0}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}
}

func tinyResearchOnlyEmbeddingScoreSpectrumDataset() []EmbeddingScoreSpectrumExample {
	out := tinyEmbeddingScoreSpectrumDataset()
	for i := range out {
		out[i].ReleaseTrainAllowed = false
		out[i].CommercialUseAllowed = false
		out[i].TrainAllowedForResearch = true
	}
	return out
}
