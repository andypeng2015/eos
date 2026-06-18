package eosruntime

import (
	"math"
	"strings"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/turboquant"
)

func TestEstimateContrastiveTrainWorkload(t *testing.T) {
	workload := EstimateContrastiveTrainWorkload(512, 128, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      64,
		EvalEveryEpoch: 4,
	})
	if workload.TrainMode != "contrastive" {
		t.Fatalf("train mode = %q, want contrastive", workload.TrainMode)
	}
	if workload.TrainBatchesPerEpoch != 8 {
		t.Fatalf("train batches/epoch = %d, want 8", workload.TrainBatchesPerEpoch)
	}
	if workload.TrainPairsPerEpoch != 32768 {
		t.Fatalf("train pairs/epoch = %d, want 32768", workload.TrainPairsPerEpoch)
	}
	if workload.EvalPairsPerPass != 16384 {
		t.Fatalf("eval pairs/pass = %d, want 16384", workload.EvalPairsPerPass)
	}
	if workload.PlannedEvalPasses != 1 {
		t.Fatalf("planned eval passes = %d, want 1", workload.PlannedEvalPasses)
	}
	if workload.PlannedTotalPairs != 49152 {
		t.Fatalf("planned total pairs = %d, want 49152", workload.PlannedTotalPairs)
	}
}

func TestEstimateContrastiveTrainWorkloadWithMatryoshkaPrefixes(t *testing.T) {
	workload := EstimateContrastiveTrainWorkload(512, 128, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      64,
		EvalEveryEpoch: 4,
		MatryoshkaDims: []int{64, 128},
	})
	if workload.TrainPairsPerEpoch != 98304 {
		t.Fatalf("train pairs/epoch = %d, want 98304", workload.TrainPairsPerEpoch)
	}
	if workload.EvalPairsPerPass != 16384 {
		t.Fatalf("eval pairs/pass = %d, want unchanged 16384", workload.EvalPairsPerPass)
	}
	if workload.PlannedTotalPairs != 114688 {
		t.Fatalf("planned total pairs = %d, want 114688", workload.PlannedTotalPairs)
	}
}

func TestEstimateContrastiveTrainWorkloadWithTurboQuantPrefixes(t *testing.T) {
	workload := EstimateContrastiveTrainWorkload(512, 128, EmbeddingTrainRunConfig{
		Epochs:               1,
		BatchSize:            64,
		EvalEveryEpoch:       4,
		MatryoshkaDims:       []int{64, 128},
		TurboQuantPrefixBits: []int{2, 4},
	})
	if workload.TrainPairsPerEpoch != 229376 {
		t.Fatalf("train pairs/epoch = %d, want 229376", workload.TrainPairsPerEpoch)
	}
	if workload.EvalPairsPerPass != 16384 {
		t.Fatalf("eval pairs/pass = %d, want unchanged 16384", workload.EvalPairsPerPass)
	}
	if workload.PlannedTotalPairs != 245760 {
		t.Fatalf("planned total pairs = %d, want 245760", workload.PlannedTotalPairs)
	}
}

func TestEstimateContrastiveTrainWorkloadWithTurboQuantPrefixObjectives(t *testing.T) {
	workload := EstimateContrastiveTrainWorkload(512, 128, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      64,
		EvalEveryEpoch: 4,
		MatryoshkaDims: []int{64, 128},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 128, BitWidth: 4, Weight: 0.5},
			{Dim: 64, BitWidth: 2, Weight: 0},
		},
	})
	if workload.TrainPairsPerEpoch != 131072 {
		t.Fatalf("train pairs/epoch = %d, want 131072", workload.TrainPairsPerEpoch)
	}
	if workload.PlannedTotalPairs != 147456 {
		t.Fatalf("planned total pairs = %d, want 147456", workload.PlannedTotalPairs)
	}
}

func TestEstimateGroupedHardNegativeTrainWorkload(t *testing.T) {
	workload := EstimateHardNegativeTrainWorkload(128, 1, 0, EmbeddingTrainRunConfig{
		Epochs:          1,
		BatchSize:       64,
		ContrastiveLoss: "grouped_infonce",
	})
	if workload.TrainMode != "hard_negative_grouped_infonce" {
		t.Fatalf("train mode = %q, want hard_negative_grouped_infonce", workload.TrainMode)
	}
	if workload.TrainBatchesPerEpoch != 2 {
		t.Fatalf("train batches/epoch = %d, want 2", workload.TrainBatchesPerEpoch)
	}
	if workload.TrainPairsPerEpoch != 256 {
		t.Fatalf("train pairs/epoch = %d, want 256", workload.TrainPairsPerEpoch)
	}
}

func TestEstimateHybridHardNegativeTrainWorkload(t *testing.T) {
	workload := EstimateHardNegativeTrainWorkload(128, 1, 0, EmbeddingTrainRunConfig{
		Epochs:          1,
		BatchSize:       64,
		ContrastiveLoss: "hybrid_infonce",
	})
	if workload.TrainMode != "hard_negative_hybrid_infonce" {
		t.Fatalf("train mode = %q, want hard_negative_hybrid_infonce", workload.TrainMode)
	}
	if workload.TrainBatchesPerEpoch != 2 {
		t.Fatalf("train batches/epoch = %d, want 2", workload.TrainBatchesPerEpoch)
	}
	if workload.TrainPairsPerEpoch != 16640 {
		t.Fatalf("train pairs/epoch = %d, want 16640", workload.TrainPairsPerEpoch)
	}
}

func TestEstimateHardNegativeTrainWorkloadWithTurboQuantPrefixObjectives(t *testing.T) {
	grouped := EstimateHardNegativeTrainWorkload(128, 1, 0, EmbeddingTrainRunConfig{
		Epochs:          1,
		BatchSize:       64,
		ContrastiveLoss: "grouped_infonce",
		MatryoshkaDims:  []int{64, 128},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 64, BitWidth: 4, Weight: 0},
			{Dim: 128, BitWidth: 4, Weight: 0.5},
		},
	})
	if grouped.TrainPairsPerEpoch != 1024 {
		t.Fatalf("grouped train pairs/epoch = %d, want 1024", grouped.TrainPairsPerEpoch)
	}
	hybrid := EstimateHardNegativeTrainWorkload(128, 1, 0, EmbeddingTrainRunConfig{
		Epochs:          1,
		BatchSize:       64,
		ContrastiveLoss: "hybrid_infonce",
		MatryoshkaDims:  []int{64, 128},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 64, BitWidth: 4, Weight: 0},
			{Dim: 128, BitWidth: 4, Weight: 0.5},
		},
	})
	if hybrid.TrainPairsPerEpoch != 66560 {
		t.Fatalf("hybrid train pairs/epoch = %d, want 66560", hybrid.TrainPairsPerEpoch)
	}
}

func TestEmbeddingTrainerFitImprovesEvalAndTracksBest(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingPairDataset()

	before, err := trainer.EvaluatePairs(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.Fit(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit: %v", err)
	}
	if summary.EpochsCompleted != 6 {
		t.Fatalf("epochs completed = %d, want 6", summary.EpochsCompleted)
	}
	if summary.StepsCompleted != 12 {
		t.Fatalf("steps completed = %d, want 12", summary.StepsCompleted)
	}
	if len(summary.History) != 6 {
		t.Fatalf("history len = %d, want 6", len(summary.History))
	}
	if summary.BestEval == nil {
		t.Fatal("expected best eval metrics")
	}
	if summary.LastEval == nil {
		t.Fatal("expected last eval metrics")
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.BestEpoch <= 0 || summary.BestStep <= 0 {
		t.Fatalf("best epoch/step = %d/%d, want both positive", summary.BestEpoch, summary.BestStep)
	}
	if !summary.RestoredBest {
		t.Fatal("expected best checkpoint restore")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}

	finalEval, err := trainer.EvaluatePairs(trainSet)
	if err != nil {
		t.Fatalf("eval final: %v", err)
	}
	assertClose(t, finalEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, finalEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestEmbeddingTrainerFitStopsEarlyOnPlateau(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0)
	trainSet := tinyEmbeddingPairDataset()

	summary, err := trainer.Fit(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:                8,
		BatchSize:             2,
		Shuffle:               false,
		Seed:                  1,
		EarlyStoppingPatience: 1,
		MinDelta:              2,
		RestoreBest:           true,
	})
	if err != nil {
		t.Fatalf("fit: %v", err)
	}
	if !summary.StoppedEarly {
		t.Fatal("expected early stopping")
	}
	if summary.EpochsCompleted != 2 {
		t.Fatalf("epochs completed = %d, want 2", summary.EpochsCompleted)
	}
	if summary.StepsCompleted != 4 {
		t.Fatalf("steps completed = %d, want 4", summary.StepsCompleted)
	}
	if summary.BestEval == nil || summary.FinalEval == nil {
		t.Fatal("expected best and final eval metrics")
	}
	assertClose(t, summary.FinalEval.ScoreMargin, summary.BestEval.ScoreMargin, 0.000001)
	assertClose(t, summary.FinalEval.Loss, summary.BestEval.Loss, 0.000001)
}

func TestEmbeddingTrainerFitRejectsInvalidRunConfig(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingPairDataset()

	if _, err := trainer.Fit(nil, nil, EmbeddingTrainRunConfig{}); err == nil {
		t.Fatal("expected empty training dataset error")
	}
	if _, err := trainer.Fit(trainSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: 1, EvalEveryEpoch: 1, SelectMetric: "unknown"}); err == nil {
		t.Fatal("expected select metric error")
	}
	if _, err := trainer.Fit(trainSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: -1}); err == nil {
		t.Fatal("expected batch size error")
	}
	if _, err := trainer.Fit(trainSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: 2, ProgressEverySteps: -1}); err == nil {
		t.Fatal("expected progress interval error")
	}
}

func TestEmbeddingTrainerFitContrastiveImprovesEval(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveRejectsInvalidMatryoshkaConfig(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	if _, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		MatryoshkaDims: []int{3},
	}); err == nil {
		t.Fatal("expected matryoshka dim exceeding embedding dimension error")
	}
	if _, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:            1,
		BatchSize:         2,
		MatryoshkaDims:    []int{1},
		MatryoshkaWeights: []float32{1, 0.5},
	}); err == nil {
		t.Fatal("expected matryoshka weight length error")
	}
}

func TestEmbeddingTrainerFitContrastiveRejectsInvalidTurboQuantPrefixConfig(t *testing.T) {
	trainSet := tinyEmbeddingContrastiveDataset()
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:               1,
		BatchSize:            2,
		MatryoshkaDims:       []int{2},
		TurboQuantPrefixBits: []int{1},
	}); err == nil {
		t.Fatal("expected invalid turboquant prefix bit error")
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:               1,
		BatchSize:            2,
		TurboQuantPrefixBits: []int{2},
	}); err == nil {
		t.Fatal("expected turboquant prefix bits to require matryoshka dims")
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:               1,
		BatchSize:            2,
		MatryoshkaDims:       []int{1},
		TurboQuantPrefixBits: []int{2},
	}); err == nil {
		t.Fatal("expected turboquant prefix dims to require dim >= 2")
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                    1,
		BatchSize:                 2,
		MatryoshkaDims:            []int{2},
		TurboQuantPrefixBits:      []int{2},
		TurboQuantPrefixScoreMode: "bogus",
	}); err == nil {
		t.Fatal("expected invalid turboquant prefix score mode error")
	}
	for _, tt := range []struct {
		name string
		cfg  EmbeddingTrainRunConfig
	}{
		{
			name: "missing dim",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 4, BitWidth: 2, Weight: 0.5},
				},
			},
		},
		{
			name: "duplicate",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 2, Weight: 0.5},
					{Dim: 2, BitWidth: 2, Weight: 0.25},
				},
			},
		},
		{
			name: "invalid bit",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 9, Weight: 0.5},
				},
			},
		},
		{
			name: "negative weight",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 2, Weight: -0.5},
				},
			},
		},
		{
			name: "nan weight",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 2, Weight: float32(math.NaN())},
				},
			},
		},
		{
			name: "inf weight",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 2, Weight: float32(math.Inf(1))},
				},
			},
		},
		{
			name: "mix bits",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims:       []int{2},
				TurboQuantPrefixBits: []int{2},
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 4, Weight: 0.5},
				},
			},
		},
		{
			name: "mix weight",
			cfg: EmbeddingTrainRunConfig{
				MatryoshkaDims:         []int{2},
				TurboQuantPrefixWeight: 0.25,
				TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
					{Dim: 2, BitWidth: 4, Weight: 0.5},
				},
			},
		},
	} {
		tt.cfg.Epochs = 1
		tt.cfg.BatchSize = 2
		if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, tt.cfg); err == nil {
			t.Fatalf("%s: expected invalid turboquant prefix objectives error", tt.name)
		}
	}
	for _, tt := range []struct {
		in   string
		want string
	}{
		{"", TurboQuantPrefixScoreModeReconstructCosine},
		{"reconstruct_cosine", TurboQuantPrefixScoreModeReconstructCosine},
		{"reconstruct-cosine", TurboQuantPrefixScoreModeReconstructCosine},
		{"prepared_ip", TurboQuantPrefixScoreModePreparedIP},
		{"prepared-ip", TurboQuantPrefixScoreModePreparedIP},
	} {
		if mode, err := NormalizeTurboQuantPrefixScoreModeForCLI(tt.in); err != nil || mode != tt.want {
			t.Fatalf("mode %q = %q, err=%v; want %q", tt.in, mode, err, tt.want)
		}
	}
}

func TestParseTurboQuantPrefixObjectives(t *testing.T) {
	objectives, err := ParseTurboQuantPrefixObjectives("128:4=0.5,64:2=0,64:4=0.25")
	if err != nil {
		t.Fatalf("parse objectives: %v", err)
	}
	if got, want := FormatTurboQuantPrefixObjectives(objectives), "64:2=0,64:4=0.25,128:4=0.5"; got != want {
		t.Fatalf("formatted objectives = %q, want %q", got, want)
	}
	for _, raw := range []string{
		"128:4=0.5,",
		"128:4",
		"128=0.5",
		"128:1=0.5",
		"128:4=-0.5",
		"128:4=NaN",
		"128:4=+Inf",
		"128:4=0.5,128:4=0.25",
	} {
		if _, err := ParseTurboQuantPrefixObjectives(raw); err == nil {
			t.Fatalf("ParseTurboQuantPrefixObjectives(%q) succeeded, want error", raw)
		}
	}
}

func TestEmbeddingTrainerFitContrastiveMatryoshkaPrefixRunsAndTracksWork(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:            1,
		BatchSize:         2,
		Shuffle:           false,
		MatryoshkaDims:    []int{1, 2},
		MatryoshkaWeights: []float32{0.5, 1},
	})
	if err != nil {
		t.Fatalf("fit contrastive matryoshka: %v", err)
	}
	if got, want := summary.Config.MatryoshkaDims, []int{1}; len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("normalized matryoshka dims = %v, want %v", got, want)
	}
	if summary.FinalTrain.BatchSize != 8 {
		t.Fatalf("final train batch size = %d, want base+prefix pair count 8", summary.FinalTrain.BatchSize)
	}
	if summary.Workload.PlannedTrainPairs != 8 || summary.Workload.ActualTrainPairs != 8 {
		t.Fatalf("train pairs planned/actual = %d/%d, want 8/8", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if summary.FinalTrain.Loss <= 0 {
		t.Fatalf("final train loss = %f, want positive", summary.FinalTrain.Loss)
	}
}

func TestEmbeddingTrainerFitContrastiveTurboQuantPrefixRunsAndTracksWork(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                 1,
		BatchSize:              2,
		Shuffle:                false,
		MatryoshkaDims:         []int{2, 3},
		MatryoshkaWeights:      []float32{0.5, 1},
		TurboQuantPrefixBits:   []int{4, 2},
		TurboQuantPrefixWeight: 0.25,
	})
	if err != nil {
		t.Fatalf("fit contrastive turboquant prefix: %v", err)
	}
	if got, want := summary.Config.MatryoshkaDims, []int{2}; len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("normalized matryoshka dims = %v, want %v", got, want)
	}
	if got, want := summary.Config.TurboQuantPrefixBits, []int{2, 4}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("normalized turboquant prefix bits = %v, want %v", got, want)
	}
	if summary.Config.TurboQuantPrefixSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("turboquant prefix seed = %d, want default %d", summary.Config.TurboQuantPrefixSeed, DefaultTurboQuantMultiVectorQuantizerSeed)
	}
	if summary.FinalTrain.BatchSize != 16 {
		t.Fatalf("final train batch size = %d, want base+dense-prefix+2 turboquant prefixes pair count 16", summary.FinalTrain.BatchSize)
	}
	if summary.Workload.PlannedTrainPairs != 16 || summary.Workload.ActualTrainPairs != 16 {
		t.Fatalf("train pairs planned/actual = %d/%d, want 16/16", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if summary.FinalTrain.Loss <= 0 {
		t.Fatalf("final train loss = %f, want positive", summary.FinalTrain.Loss)
	}
}

func TestEmbeddingTrainerFitContrastiveTurboQuantPrefixObjectivesRunAndTrackWork(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:            1,
		BatchSize:         2,
		Shuffle:           false,
		MatryoshkaDims:    []int{2, 3},
		MatryoshkaWeights: []float32{0.5, 1},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 2, BitWidth: 4, Weight: 0.25},
			{Dim: 2, BitWidth: 2, Weight: 0},
		},
	})
	if err != nil {
		t.Fatalf("fit contrastive turboquant prefix objectives: %v", err)
	}
	if got, want := summary.Config.MatryoshkaDims, []int{2}; len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("normalized matryoshka dims = %v, want %v", got, want)
	}
	if got := summary.Config.TurboQuantPrefixObjectives; len(got) != 2 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0 || got[1].Dim != 2 || got[1].BitWidth != 4 || got[1].Weight != 0.25 {
		t.Fatalf("normalized turboquant prefix objectives = %+v", got)
	}
	if summary.Config.TurboQuantPrefixWeight != 0 {
		t.Fatalf("turboquant prefix weight = %f, want 0 in explicit mode", summary.Config.TurboQuantPrefixWeight)
	}
	if summary.Config.TurboQuantPrefixSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("turboquant prefix seed = %d, want default %d", summary.Config.TurboQuantPrefixSeed, DefaultTurboQuantMultiVectorQuantizerSeed)
	}
	if summary.FinalTrain.BatchSize != 12 {
		t.Fatalf("final train batch size = %d, want base+dense-prefix+1 active turboquant prefix pair count 12", summary.FinalTrain.BatchSize)
	}
	if summary.Workload.PlannedTrainPairs != 12 || summary.Workload.ActualTrainPairs != 12 {
		t.Fatalf("train pairs planned/actual = %d/%d, want 12/12", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
}

func TestEmbeddingTrainerFitContrastiveTurboQuantPrefixObjectiveAllowsBaseDim(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		MatryoshkaDims: []int{3},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 3, BitWidth: 2, Weight: 0.25},
		},
		TurboQuantPrefixScoreMode: TurboQuantPrefixScoreModePreparedIP,
	})
	if err != nil {
		t.Fatalf("fit contrastive base-dim turboquant prefix objective: %v", err)
	}
	if len(summary.Config.MatryoshkaDims) != 0 {
		t.Fatalf("normalized matryoshka dims = %v, want base dim omitted", summary.Config.MatryoshkaDims)
	}
	if got := summary.Config.TurboQuantPrefixObjectives; len(got) != 1 || got[0].Dim != 3 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("normalized turboquant prefix objectives = %+v", got)
	}
	if summary.FinalTrain.BatchSize != 8 {
		t.Fatalf("final train batch size = %d, want base+base-dim turboquant prefix pair count 8", summary.FinalTrain.BatchSize)
	}
	if summary.Workload.PlannedTrainPairs != 8 || summary.Workload.ActualTrainPairs != 8 {
		t.Fatalf("train pairs planned/actual = %d/%d, want 8/8", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
}

func TestEmbeddingTrainerFitContrastiveContinuesLegacyTurboQuantWithExplicitObjectives(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	_, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                 1,
		BatchSize:              2,
		Shuffle:                false,
		MatryoshkaDims:         []int{2},
		TurboQuantPrefixBits:   []int{2},
		TurboQuantPrefixWeight: 0.25,
	})
	if err != nil {
		t.Fatalf("initial legacy turboquant prefix fit: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		MatryoshkaDims: []int{2},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 2, BitWidth: 4, Weight: 0.5},
		},
	})
	if err != nil {
		t.Fatalf("continue legacy turboquant with explicit objectives: %v", err)
	}
	if len(summary.Config.TurboQuantPrefixBits) != 0 {
		t.Fatalf("turboquant prefix bits = %v, want cleared", summary.Config.TurboQuantPrefixBits)
	}
	if got := summary.Config.TurboQuantPrefixObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 4 || got[0].Weight != 0.5 {
		t.Fatalf("turboquant prefix objectives = %+v, want explicit 2:4=0.5", got)
	}
	if summary.Config.TurboQuantPrefixWeight != 0 {
		t.Fatalf("turboquant prefix weight = %f, want 0 in explicit mode", summary.Config.TurboQuantPrefixWeight)
	}
}

func TestEmbeddingTrainerFitContrastiveContinuesExplicitObjectivesWithLegacyTurboQuant(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	_, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		MatryoshkaDims: []int{2},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 2, BitWidth: 4, Weight: 0.5},
		},
	})
	if err != nil {
		t.Fatalf("initial explicit turboquant prefix fit: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:               1,
		BatchSize:            2,
		Shuffle:              false,
		MatryoshkaDims:       []int{2},
		TurboQuantPrefixBits: []int{2},
	})
	if err != nil {
		t.Fatalf("continue explicit objectives with legacy turboquant: %v", err)
	}
	if len(summary.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("turboquant prefix objectives = %+v, want cleared", summary.Config.TurboQuantPrefixObjectives)
	}
	if got := summary.Config.TurboQuantPrefixBits; len(got) != 1 || got[0] != 2 {
		t.Fatalf("turboquant prefix bits = %v, want [2]", got)
	}
	if summary.Config.TurboQuantPrefixWeight != 1 {
		t.Fatalf("turboquant prefix weight = %f, want legacy default 1", summary.Config.TurboQuantPrefixWeight)
	}
}

func TestEmbeddingTrainerFitContrastiveInheritsTurboQuantPrefixObjectivesByDefault(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	_, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		MatryoshkaDims: []int{2},
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 2, BitWidth: 4, Weight: 0.5},
		},
		TurboQuantPrefixSeed:      123,
		TurboQuantPrefixScoreMode: TurboQuantPrefixScoreModePreparedIP,
	})
	if err != nil {
		t.Fatalf("initial explicit turboquant prefix fit: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
		Shuffle:   false,
	})
	if err != nil {
		t.Fatalf("continue without turboquant prefix flags: %v", err)
	}
	if got := summary.Config.TurboQuantPrefixObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 4 || got[0].Weight != 0.5 {
		t.Fatalf("inherited turboquant prefix objectives = %+v, want 2:4=0.5", got)
	}
	if summary.Config.TurboQuantPrefixSeed != 123 {
		t.Fatalf("inherited turboquant prefix seed = %d, want 123", summary.Config.TurboQuantPrefixSeed)
	}
	if summary.Config.TurboQuantPrefixScoreMode != TurboQuantPrefixScoreModePreparedIP {
		t.Fatalf("inherited turboquant prefix score mode = %q, want %q", summary.Config.TurboQuantPrefixScoreMode, TurboQuantPrefixScoreModePreparedIP)
	}
}

func TestEmbeddingTrainerFitContrastiveClearTurboQuantPrefixDisablesInheritedObjectives(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	_, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                 1,
		BatchSize:              2,
		Shuffle:                false,
		MatryoshkaDims:         []int{2},
		TurboQuantPrefixBits:   []int{2},
		TurboQuantPrefixWeight: 0.25,
		TurboQuantPrefixSeed:   123,
	})
	if err != nil {
		t.Fatalf("initial legacy turboquant prefix fit: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                1,
		BatchSize:             2,
		Shuffle:               false,
		ClearTurboQuantPrefix: true,
	})
	if err != nil {
		t.Fatalf("continue with clear turboquant prefix: %v", err)
	}
	if len(summary.Config.TurboQuantPrefixBits) != 0 {
		t.Fatalf("turboquant prefix bits = %v, want cleared", summary.Config.TurboQuantPrefixBits)
	}
	if len(summary.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("turboquant prefix objectives = %+v, want cleared", summary.Config.TurboQuantPrefixObjectives)
	}
	if summary.Config.TurboQuantPrefixWeight != 0 || summary.Config.TurboQuantPrefixSeed != 0 || summary.Config.TurboQuantPrefixScoreMode != "" {
		t.Fatalf("turboquant prefix associated config = weight:%f seed:%d mode:%q, want cleared", summary.Config.TurboQuantPrefixWeight, summary.Config.TurboQuantPrefixSeed, summary.Config.TurboQuantPrefixScoreMode)
	}
	if len(trainer.config.TurboQuantPrefixBits) != 0 || len(trainer.config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("trainer turboquant prefix config = bits:%v objectives:%+v, want cleared", trainer.config.TurboQuantPrefixBits, trainer.config.TurboQuantPrefixObjectives)
	}
}

func TestEmbeddingTrainerFitContrastiveRejectsClearTurboQuantPrefixWithNewObjectives(t *testing.T) {
	trainSet := tinyEmbeddingContrastiveDataset()
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                1,
		BatchSize:             2,
		Shuffle:               false,
		ClearTurboQuantPrefix: true,
		TurboQuantPrefixBits:  []int{2},
	}); err == nil || !strings.Contains(err.Error(), "clear_turboquant_prefix") {
		t.Fatalf("clear with prefix bits error = %v, want clear_turboquant_prefix conflict", err)
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                1,
		BatchSize:             2,
		Shuffle:               false,
		ClearTurboQuantPrefix: true,
		TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
			{Dim: 2, BitWidth: 4, Weight: 0.5},
		},
	}); err == nil || !strings.Contains(err.Error(), "clear_turboquant_prefix") {
		t.Fatalf("clear with prefix objectives error = %v, want clear_turboquant_prefix conflict", err)
	}
}

func TestTurboQuantPreparedIPPrefixScoreMatchesQuantizer(t *testing.T) {
	q := turboquant.NewIPWithSeed(4, 3, DefaultTurboQuantMultiVectorQuantizerSeed)
	rawCandidate := []float32{2, -1, 0.5, 1.5}
	rawQuery := []float32{-0.25, 1.25, 0.75, 0.5}
	candidate := normalizedCopy(rawCandidate)
	query := normalizedCopy(rawQuery)

	got := turboQuantPreparedPrefixScore(q, candidate, query)
	want := q.InnerProductPrepared(q.Quantize(candidate), q.PrepareQuery(query))
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("prepared score = %.9f, want %.9f", got, want)
	}
}

func TestTurboQuantPreparedPrefixMatrixBuildsValidZeroRows(t *testing.T) {
	seqs := []*embeddingEncodedSequence{
		nil,
		{pooled: []float32{0, 0, 0}},
		nil,
		{pooled: []float32{1}},
	}
	queryMatrix := newTurboQuantPreparedPrefixMatrix(seqs, 3, 2, DefaultTurboQuantMultiVectorQuantizerSeed, true)
	candidateMatrix := newTurboQuantPreparedPrefixMatrix(seqs, 3, 2, DefaultTurboQuantMultiVectorQuantizerSeed, false)
	if queryMatrix.width != 3 || candidateMatrix.width != 3 {
		t.Fatalf("matrix widths = query:%d candidate:%d, want 3", queryMatrix.width, candidateMatrix.width)
	}
	q := turboquant.NewIPWithSeed(3, 2, DefaultTurboQuantMultiVectorQuantizerSeed)
	for i := range seqs {
		score := q.InnerProductPrepared(candidateMatrix.quantized[i], queryMatrix.prepared[i])
		if !finite32(score) {
			t.Fatalf("row %d prepared score must be finite, got %f", i, score)
		}
		if queryMatrix.rawNorms[i] != 0 || candidateMatrix.rawNorms[i] != 0 {
			t.Fatalf("row %d raw norms = query:%f candidate:%f, want zero", i, queryMatrix.rawNorms[i], candidateMatrix.rawNorms[i])
		}
	}
}

func TestEmbeddingTrainerFitContrastiveTurboQuantPreparedIPPrefixRunsAndTracksWork(t *testing.T) {
	trainSet := tinyEmbeddingContrastiveDataset()
	base := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	baseSummary, err := base.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                 1,
		BatchSize:              2,
		Shuffle:                false,
		MatryoshkaDims:         []int{2, 3},
		MatryoshkaWeights:      []float32{0.5, 1},
		TurboQuantPrefixBits:   []int{4, 2},
		TurboQuantPrefixWeight: 0.25,
	})
	if err != nil {
		t.Fatalf("fit contrastive default turboquant prefix: %v", err)
	}

	prepared := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	preparedSummary, err := prepared.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:                    1,
		BatchSize:                 2,
		Shuffle:                   false,
		MatryoshkaDims:            []int{2, 3},
		MatryoshkaWeights:         []float32{0.5, 1},
		TurboQuantPrefixBits:      []int{4, 2},
		TurboQuantPrefixWeight:    0.25,
		TurboQuantPrefixScoreMode: "prepared-ip",
	})
	if err != nil {
		t.Fatalf("fit contrastive prepared-ip turboquant prefix: %v", err)
	}
	if preparedSummary.Config.TurboQuantPrefixScoreMode != TurboQuantPrefixScoreModePreparedIP {
		t.Fatalf("score mode = %q, want %q", preparedSummary.Config.TurboQuantPrefixScoreMode, TurboQuantPrefixScoreModePreparedIP)
	}
	if preparedSummary.Workload.PlannedTrainPairs != baseSummary.Workload.PlannedTrainPairs || preparedSummary.Workload.ActualTrainPairs != baseSummary.Workload.ActualTrainPairs {
		t.Fatalf("prepared-ip workload planned/actual = %d/%d, want %d/%d", preparedSummary.Workload.PlannedTrainPairs, preparedSummary.Workload.ActualTrainPairs, baseSummary.Workload.PlannedTrainPairs, baseSummary.Workload.ActualTrainPairs)
	}
	if preparedSummary.FinalTrain.BatchSize != baseSummary.FinalTrain.BatchSize {
		t.Fatalf("prepared-ip batch size = %d, want %d", preparedSummary.FinalTrain.BatchSize, baseSummary.FinalTrain.BatchSize)
	}
}

func TestTurboQuantPreparedIPPrefixGradientsAreFiniteAndTangent(t *testing.T) {
	queries := []*embeddingEncodedSequence{
		{pooled: []float32{1, 2, -1}},
		{pooled: []float32{-1, 0.5, 2}},
	}
	positives := []*embeddingEncodedSequence{
		{pooled: []float32{0.5, -1, 2}},
		{pooled: []float32{2, 1, -0.5}},
	}
	queryGrads := newEmbeddingPooledGradBuffers(queries)
	positiveGrads := newEmbeddingPooledGradBuffers(positives)
	loss, score := accumulateTurboQuantPreparedIPPrefixInfoNCEContrastiveGrads(queries, positives, 3, 2, DefaultTurboQuantMultiVectorQuantizerSeed, 0.05, queryGrads, positiveGrads)
	if !finite32(loss) || !finite32(score) {
		t.Fatalf("loss/score must be finite: loss=%f score=%f", loss, score)
	}
	for i, grad := range queryGrads {
		assertFiniteAndTangent(t, "query", i, queries[i].pooled[:3], grad[:3])
	}
	for i, grad := range positiveGrads {
		assertFiniteAndTangent(t, "positive", i, positives[i].pooled[:3], grad[:3])
	}
}

func TestEmbeddingTrainerTurboQuantPrefixDisablesContrastiveAccelerator(t *testing.T) {
	batch := tinyEmbeddingContrastiveDataset()
	accelerated := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	accelerated.config.ContrastiveLoss = "infonce"
	accelerated.config.Temperature = 0.05
	accel := &countingContrastiveAccelerator{}
	accelerated.contrastiveAccel = accel
	if _, err := accelerated.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train accelerated contrastive: %v", err)
	}
	if accel.squareCalls != 1 {
		t.Fatalf("accelerator square calls = %d, want 1 when compact-prefix objective unset", accel.squareCalls)
	}

	host := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	host.config.ContrastiveLoss = "infonce"
	host.config.Temperature = 0.05
	host.config.MatryoshkaDims = []int{2}
	host.config.MatryoshkaWeights = []float32{1}
	host.config.TurboQuantPrefixBits = []int{2}
	host.config.TurboQuantPrefixWeight = 1
	host.config.TurboQuantPrefixSeed = DefaultTurboQuantMultiVectorQuantizerSeed
	hostAccel := &countingContrastiveAccelerator{}
	host.contrastiveAccel = hostAccel
	if _, err := host.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train turboquant prefix contrastive: %v", err)
	}
	if hostAccel.squareCalls != 0 || hostAccel.rectCalls != 0 {
		t.Fatalf("accelerator calls square=%d rect=%d, want host path when turboquant prefix objective is enabled", hostAccel.squareCalls, hostAccel.rectCalls)
	}
}

func normalizedCopy(values []float32) []float32 {
	out := append([]float32(nil), values...)
	norm := vectorNorm(out)
	if norm == 0 {
		return out
	}
	inv := 1 / norm
	for i := range out {
		out[i] *= inv
	}
	return out
}

func finite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func assertFiniteAndTangent(t *testing.T, label string, index int, raw, grad []float32) {
	t.Helper()
	if len(raw) != len(grad) {
		t.Fatalf("%s[%d] len raw=%d grad=%d", label, index, len(raw), len(grad))
	}
	dot := float32(0)
	norm := float32(0)
	gradNorm := float32(0)
	for i := range raw {
		if !finite32(grad[i]) {
			t.Fatalf("%s[%d] grad[%d]=%f, want finite", label, index, i, grad[i])
		}
		dot += raw[i] * grad[i]
		norm += raw[i] * raw[i]
		gradNorm += grad[i] * grad[i]
	}
	if norm > 0 && gradNorm > 0 && math.Abs(float64(dot)) > 1e-4 {
		t.Fatalf("%s[%d] raw dot grad = %.9f, want tangent", label, index, dot)
	}
}

func TestEmbeddingTrainerFitContrastiveTracksWorkloadAndTiming(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	cfg := EmbeddingTrainRunConfig{
		Epochs:      1,
		BatchSize:   2,
		Shuffle:     false,
		Seed:        1,
		RestoreBest: true,
	}

	summary, err := trainer.FitContrastive(trainSet, trainSet, cfg)
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}

	expected := EstimateContrastiveTrainWorkload(len(trainSet), len(trainSet), cfg)
	if summary.Workload.TrainPairsPerEpoch != expected.TrainPairsPerEpoch {
		t.Fatalf("train pairs/epoch = %d, want %d", summary.Workload.TrainPairsPerEpoch, expected.TrainPairsPerEpoch)
	}
	if summary.Workload.PlannedTotalPairs != expected.PlannedTotalPairs {
		t.Fatalf("planned total pairs = %d, want %d", summary.Workload.PlannedTotalPairs, expected.PlannedTotalPairs)
	}
	if summary.Workload.ActualTrainPairs != expected.PlannedTrainPairs {
		t.Fatalf("actual train pairs = %d, want %d", summary.Workload.ActualTrainPairs, expected.PlannedTrainPairs)
	}
	if summary.Workload.ActualTrainExamples != int64(len(trainSet)) {
		t.Fatalf("actual train examples = %d, want %d", summary.Workload.ActualTrainExamples, len(trainSet))
	}
	if summary.Workload.ActualEvalPairs != expected.PlannedEvalPairs {
		t.Fatalf("actual eval pairs = %d, want %d", summary.Workload.ActualEvalPairs, expected.PlannedEvalPairs)
	}
	expectedEvalExamples := int64(len(trainSet) * expected.PlannedEvalPasses)
	if summary.Workload.ActualEvalExamples != expectedEvalExamples {
		t.Fatalf("actual eval examples = %d, want %d", summary.Workload.ActualEvalExamples, expectedEvalExamples)
	}
	if summary.Workload.ActualTotalPairs != summary.Workload.ActualTrainPairs+summary.Workload.ActualEvalPairs {
		t.Fatalf("actual total pairs = %d, want %d", summary.Workload.ActualTotalPairs, summary.Workload.ActualTrainPairs+summary.Workload.ActualEvalPairs)
	}
	if summary.Workload.ActualTotalExamples != summary.Workload.ActualTrainExamples+summary.Workload.ActualEvalExamples {
		t.Fatalf("actual total examples = %d, want %d", summary.Workload.ActualTotalExamples, summary.Workload.ActualTrainExamples+summary.Workload.ActualEvalExamples)
	}
	if summary.Workload.ActualEvalPasses != expected.PlannedEvalPasses {
		t.Fatalf("actual eval passes = %d, want %d", summary.Workload.ActualEvalPasses, expected.PlannedEvalPasses)
	}
	if summary.StepsRun != expected.TrainBatchesPerEpoch {
		t.Fatalf("run steps = %d, want %d", summary.StepsRun, expected.TrainBatchesPerEpoch)
	}
	if summary.Elapsed <= 0 {
		t.Fatalf("elapsed = %s, want > 0", summary.Elapsed)
	}
	if summary.TrainDuration <= 0 {
		t.Fatalf("train duration = %s, want > 0", summary.TrainDuration)
	}
	if summary.EvalDuration <= 0 {
		t.Fatalf("eval duration = %s, want > 0", summary.EvalDuration)
	}
	if summary.Elapsed+time.Millisecond < summary.TrainDuration+summary.EvalDuration {
		t.Fatalf("elapsed = %s, train+eval = %s, want elapsed >= train+eval", summary.Elapsed, summary.TrainDuration+summary.EvalDuration)
	}
}

func TestEmbeddingTrainerFitContrastiveReportsProgress(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	var reports []EmbeddingTrainProgress

	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:             1,
		BatchSize:          2,
		Shuffle:            false,
		ProgressEverySteps: 1,
		Progress: func(progress EmbeddingTrainProgress) {
			reports = append(reports, progress)
		},
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if len(reports) != summary.StepsRun {
		t.Fatalf("progress reports = %d, want %d", len(reports), summary.StepsRun)
	}
	got := reports[0]
	if got.Epoch != 1 || got.Batch != 1 || got.Batches != 1 {
		t.Fatalf("progress position = epoch %d batch %d/%d, want 1 1/1", got.Epoch, got.Batch, got.Batches)
	}
	if got.BatchExamples != len(trainSet) {
		t.Fatalf("batch examples = %d, want %d", got.BatchExamples, len(trainSet))
	}
	if got.BatchPairs != int64(len(trainSet)*len(trainSet)) {
		t.Fatalf("batch pairs = %d, want %d", got.BatchPairs, len(trainSet)*len(trainSet))
	}
	if got.Step != trainer.step {
		t.Fatalf("progress step = %d, want trainer step %d", got.Step, trainer.step)
	}
	if got.Elapsed <= 0 {
		t.Fatalf("progress elapsed = %s, want > 0", got.Elapsed)
	}
}

func TestEmbeddingTrainerFitContrastiveEvalOnlyDoesNotTrain(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	evalSet := tinyEmbeddingContrastiveDataset()

	summary, err := trainer.FitContrastive(nil, evalSet, EmbeddingTrainRunConfig{EvalOnly: true})
	if err != nil {
		t.Fatalf("fit contrastive eval-only: %v", err)
	}
	if summary.EpochsCompleted != 0 {
		t.Fatalf("epochs completed = %d, want 0", summary.EpochsCompleted)
	}
	if summary.StepsRun != 0 {
		t.Fatalf("steps run = %d, want 0", summary.StepsRun)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.Workload.PlannedTrainPairs != 0 || summary.Workload.ActualTrainPairs != 0 {
		t.Fatalf("train pairs planned/actual = %d/%d, want 0/0", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if summary.Workload.PlannedEvalPasses != 1 || summary.Workload.ActualEvalPasses != 1 {
		t.Fatalf("eval passes planned/actual = %d/%d, want 1/1", summary.Workload.PlannedEvalPasses, summary.Workload.ActualEvalPasses)
	}
}

func TestEmbeddingTrainerFitContrastiveEvaluatesWithinEpoch(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		EvalEveryEpoch: 99,
		EvalEverySteps: 1,
		SelectMetric:   "mrr",
		RestoreBest:    true,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.Workload.PlannedEvalPasses != 3 || summary.Workload.ActualEvalPasses != 3 {
		t.Fatalf("eval passes planned/actual = %d/%d, want 3/3", summary.Workload.PlannedEvalPasses, summary.Workload.ActualEvalPasses)
	}
	if summary.BestEval == nil || summary.FinalEval == nil {
		t.Fatal("expected best and final eval metrics")
	}
	if !summary.RestoredBest {
		t.Fatal("expected restore-best to use the best checkpoint")
	}
}

func TestBucketContrastiveOrderByLengthSortsWithinDefaultWindows(t *testing.T) {
	t.Setenv("EOS_TRAIN_LENGTH_BUCKET_WINDOW", "")
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: make([]int32, 8), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 2), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 7), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 3), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 6), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 4), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 5), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 1), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 1), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 9), PositiveTokens: make([]int32, 1)},
	}
	order := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	bucketContrastiveOrderByLength(trainSet, order, 2)

	want := []int{7, 1, 3, 5, 6, 4, 2, 0, 8, 9}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order[%d] = %d, want %d (full order %v)", i, order[i], want[i], order)
		}
	}
}

func TestBucketContrastiveOrderByLengthPreservesEqualLengthShuffleOrder(t *testing.T) {
	t.Setenv("EOS_TRAIN_LENGTH_BUCKET_WINDOW", "")
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: make([]int32, 8), PositiveTokens: make([]int32, 10)},
		{QueryTokens: make([]int32, 2), PositiveTokens: make([]int32, 10)},
		{QueryTokens: make([]int32, 5), PositiveTokens: make([]int32, 10)},
		{QueryTokens: make([]int32, 10), PositiveTokens: make([]int32, 3)},
	}
	order := []int{0, 1, 2, 3}

	bucketContrastiveOrderByLength(trainSet, order, 2)

	want := []int{0, 1, 2, 3}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order[%d] = %d, want %d (full order %v)", i, order[i], want[i], order)
		}
	}
}

func TestBucketContrastiveOrderByLengthHonorsWindowOverride(t *testing.T) {
	t.Setenv("EOS_TRAIN_LENGTH_BUCKET_WINDOW", "4")
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: make([]int32, 8), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 1), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 7), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 2), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 6), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 3), PositiveTokens: make([]int32, 1)},
	}
	order := []int{0, 1, 2, 3, 4, 5}

	bucketContrastiveOrderByLength(trainSet, order, 2)

	want := []int{1, 3, 2, 0, 5, 4}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order[%d] = %d, want %d (full order %v)", i, order[i], want[i], order)
		}
	}
}

func TestEmbeddingTrainerFitContrastiveUsesFinalEvalAsBestWhenNoEpochEvalRuns(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	cfg := EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		Seed:           1,
		EvalEveryEpoch: 4,
		RestoreBest:    false,
	}

	summary, err := trainer.FitContrastive(trainSet, trainSet, cfg)
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.BestEval == nil {
		t.Fatal("expected best eval metrics")
	}
	if summary.BestEpoch != 1 {
		t.Fatalf("best epoch = %d, want 1", summary.BestEpoch)
	}
	if summary.BestStep != summary.StepsCompleted {
		t.Fatalf("best step = %d, want %d", summary.BestStep, summary.StepsCompleted)
	}
	assertClose(t, summary.BestEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, summary.BestEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestEmbeddingTrainerFitHardNegativesTracksPairwiseEval(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainSet := tinyEmbeddingHardNegativeDataset()
	evalSet := tinyEmbeddingPairDataset()
	summary, err := trainer.FitHardNegatives(trainSet, evalSet, EmbeddingTrainRunConfig{
		Epochs:                2,
		BatchSize:             2,
		EvalEveryEpoch:        1,
		SelectMetric:          "mrr",
		RestoreBest:           true,
		HardNegativeTrain:     true,
		HardNegativesPerQuery: 1,
	})
	if err != nil {
		t.Fatalf("fit hard negatives: %v", err)
	}
	if summary.Workload.TrainMode != "hard_negative_contrastive" {
		t.Fatalf("train mode = %q", summary.Workload.TrainMode)
	}
	if summary.FinalEval == nil || summary.FinalEval.PairCount != len(evalSet) {
		t.Fatalf("final eval = %+v, want %d pairs", summary.FinalEval, len(evalSet))
	}
	if summary.StepsRun == 0 {
		t.Fatal("expected optimizer steps")
	}
}

func TestEmbeddingTrainerFitHardNegativesExplicitZeroTeacherLossOverridesCheckpoint(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.TeacherLossWeight = 0.1
	trainer.config.TeacherTemperature = 2
	trainer.config.TeacherSourceWeights = map[string]float32{"fiqa": 0.25, "nfcorpus": 0.05, "scifact": 1}

	summary, err := trainer.FitHardNegatives(tinyEmbeddingHardNegativeDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:                1,
		BatchSize:             2,
		RestoreBest:           true,
		HardNegativeTrain:     true,
		HardNegativesPerQuery: 1,
		TeacherLossWeight:     0,
		TeacherLossWeightSet:  true,
	})
	if err != nil {
		t.Fatalf("fit hard negatives: %v", err)
	}
	if summary.Config.TeacherLossWeight != 0 {
		t.Fatalf("summary teacher loss weight = %f, want explicit zero", summary.Config.TeacherLossWeight)
	}
	if len(summary.Config.TeacherSourceWeights) != 0 {
		t.Fatalf("summary teacher source weights = %v, want cleared when teacher loss is disabled", summary.Config.TeacherSourceWeights)
	}
	if trainer.config.TeacherLossWeight != 0 {
		t.Fatalf("trainer teacher loss weight = %f, want explicit zero", trainer.config.TeacherLossWeight)
	}
	if len(trainer.config.TeacherSourceWeights) != 0 {
		t.Fatalf("trainer teacher source weights = %v, want cleared when teacher loss is disabled", trainer.config.TeacherSourceWeights)
	}
	if summary.Workload.TrainPairsPerEpoch != 8 {
		t.Fatalf("train pairs/epoch = %d, want 8 without inherited teacher pairs", summary.Workload.TrainPairsPerEpoch)
	}
	if summary.FinalTrain.BatchSize != 8 {
		t.Fatalf("final train batch size = %d, want 8 without inherited teacher pairs", summary.FinalTrain.BatchSize)
	}
}

func TestEmbeddingTrainerFitContrastiveFFNImprovesEval(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive ffn: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("ffn score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("ffn pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveAttentionImprovesEval(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive attention: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("attention score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("attention pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveEncoderProducesStableEval(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)
	trainSet := tinyEncoderContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive encoder: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected final and best eval metrics")
	}
	if summary.StepsCompleted == 0 || len(summary.History) == 0 {
		t.Fatalf("expected encoder fit to execute training steps, got summary %+v", summary)
	}
	if summary.FinalEval.ScoreMargin+0.000001 < before.ScoreMargin {
		t.Fatalf("encoder score margin regressed: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy+0.000001 < before.PairAccuracy {
		t.Fatalf("encoder pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveRepeatedEncoderProducesStableEval(t *testing.T) {
	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	trainSet := tinyEncoderContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive repeated encoder: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected final and best eval metrics")
	}
	if summary.StepsCompleted == 0 || len(summary.History) == 0 {
		t.Fatalf("expected repeated encoder fit to execute training steps, got summary %+v", summary)
	}
	if summary.FinalEval.ScoreMargin+0.000001 < before.ScoreMargin {
		t.Fatalf("repeated encoder score margin regressed: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy+0.000001 < before.PairAccuracy {
		t.Fatalf("repeated encoder pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerEvaluateContrastiveMatchesExpandedPairs(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	contrastive := tinyEmbeddingContrastiveDataset()

	got, err := trainer.EvaluateContrastive(contrastive)
	if err != nil {
		t.Fatalf("evaluate contrastive: %v", err)
	}
	want, err := trainer.EvaluatePairs(expandContrastiveExamples(contrastive))
	if err != nil {
		t.Fatalf("evaluate expanded pairs: %v", err)
	}

	assertClose(t, got.Loss, want.Loss, 0.000001)
	assertClose(t, got.AverageScore, want.AverageScore, 0.000001)
	assertClose(t, got.PositiveMeanScore, want.PositiveMeanScore, 0.000001)
	assertClose(t, got.NegativeMeanScore, want.NegativeMeanScore, 0.000001)
	assertClose(t, got.PairAccuracy, want.PairAccuracy, 0.000001)
	assertClose(t, got.ThresholdAccuracy, want.ThresholdAccuracy, 0.000001)
	assertClose(t, got.ScoreThreshold, want.ScoreThreshold, 0.000001)
	assertClose(t, got.ROCAUC, want.ROCAUC, 0.000001)
	assertClose(t, got.ScoreMargin, want.ScoreMargin, 0.000001)
	if got.PairCount != want.PairCount {
		t.Fatalf("pair count = %d, want %d", got.PairCount, want.PairCount)
	}
	if got.PositiveCount != want.PositiveCount {
		t.Fatalf("positive count = %d, want %d", got.PositiveCount, want.PositiveCount)
	}
	if got.NegativeCount != want.NegativeCount {
		t.Fatalf("negative count = %d, want %d", got.NegativeCount, want.NegativeCount)
	}
}

func TestEvaluateContrastiveEncodingsTracksRankingMetrics(t *testing.T) {
	queries := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0, 0}},
		{pooled: []float32{0, 1, 0}},
		{pooled: []float32{0, 0, 1}},
	}
	positives := []*embeddingEncodedSequence{
		{pooled: []float32{0, 1, 0}},
		{pooled: []float32{1, 0, 0}},
		{pooled: []float32{0, 0, 1}},
	}

	got := evaluateContrastiveEncodings(queries, positives, EmbeddingTrainConfig{})

	assertClose(t, got.Top1Accuracy, 1.0/3.0, 0.000001)
	assertClose(t, got.Top5Accuracy, 1, 0.000001)
	assertClose(t, got.Top10Accuracy, 1, 0.000001)
	assertClose(t, got.MeanReciprocalRank, 2.0/3.0, 0.000001)
	assertClose(t, got.MeanPositiveRank, 5.0/3.0, 0.000001)
}

func TestEmbeddingTrainerTrainContrastiveStepMatchesExpandedPairStep(t *testing.T) {
	left := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	checkpoint, err := left.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	right, err := NewEmbeddingTrainerFromCheckpoint(left.module, checkpoint)
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	left.Close()
	right.Close()

	batch := tinyEmbeddingContrastiveDataset()
	got, err := left.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	want, err := right.TrainStep(expandContrastiveExamples(batch))
	if err != nil {
		t.Fatalf("train expanded pairs: %v", err)
	}

	assertClose(t, got.Loss, want.Loss, 0.000001)
	assertClose(t, got.AverageScore, want.AverageScore, 0.000001)
	if got.BatchSize != want.BatchSize {
		t.Fatalf("batch size = %d, want %d", got.BatchSize, want.BatchSize)
	}

	leftEval, err := left.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval left: %v", err)
	}
	rightEval, err := right.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval right: %v", err)
	}
	assertClose(t, leftEval.Loss, rightEval.Loss, 0.000001)
	assertClose(t, leftEval.ScoreMargin, rightEval.ScoreMargin, 0.000001)
	assertClose(t, leftEval.PairAccuracy, rightEval.PairAccuracy, 0.000001)
}

func TestEmbeddingTrainerTrainContrastiveStepSupportsInfoNCE(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	batch := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	metrics, err := trainer.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train infonce: %v", err)
	}
	if metrics.BatchSize != len(batch)*len(batch) {
		t.Fatalf("batch size = %d, want %d", metrics.BatchSize, len(batch)*len(batch))
	}
	after, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.PairCount != len(batch)*len(batch) {
		t.Fatalf("pair count = %d, want %d", after.PairCount, len(batch)*len(batch))
	}
	if loss := infoNCERowLoss([]float32{0.2, 0.1}, 0, 0.05); loss <= 0 {
		t.Fatalf("infonce row loss = %f, want positive", loss)
	}
	if after.Loss > before.Loss+0.000001 {
		t.Fatalf("infonce eval loss regressed: before=%f after=%f", before.Loss, after.Loss)
	}
}

func TestEmbeddingTrainerTrainHardNegativeContrastiveStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	batch := tinyEmbeddingHardNegativeDataset()

	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train hard-negative step: %v", err)
	}
	if metrics.BatchSize != 8 {
		t.Fatalf("batch size = %d, want 8 rectangular query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 {
		t.Fatalf("loss = %f, want non-negative", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
}

func TestEmbeddingTrainerTrainGroupedHardNegativeContrastiveStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "grouped_infonce"
	trainer.config.Temperature = 0.05
	accel := &countingContrastiveAccelerator{}
	trainer.contrastiveAccel = accel

	metrics, err := trainer.TrainHardNegativeContrastiveStep(tinyEmbeddingHardNegativeDataset())
	if err != nil {
		t.Fatalf("train grouped hard-negative step: %v", err)
	}
	if metrics.BatchSize != 4 {
		t.Fatalf("batch size = %d, want 4 grouped query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 {
		t.Fatalf("loss = %f, want non-negative", metrics.Loss)
	}
	if accel.rectCalls != 0 || accel.squareCalls != 0 {
		t.Fatalf("accelerator calls square=%d rect=%d, want host grouped path", accel.squareCalls, accel.rectCalls)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
}

func TestEmbeddingTrainerTrainHybridHardNegativeContrastiveStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "hybrid_infonce"
	trainer.config.Temperature = 0.05
	trainer.config.GroupedLossWeight = 0.25
	accel := &countingContrastiveAccelerator{}
	trainer.contrastiveAccel = accel

	metrics, err := trainer.TrainHardNegativeContrastiveStep(tinyEmbeddingHardNegativeDataset())
	if err != nil {
		t.Fatalf("train hybrid hard-negative step: %v", err)
	}
	if metrics.BatchSize != 12 {
		t.Fatalf("batch size = %d, want 12 hybrid query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 {
		t.Fatalf("loss = %f, want non-negative", metrics.Loss)
	}
	if accel.rectCalls != 1 || accel.squareCalls != 0 {
		t.Fatalf("accelerator calls square=%d rect=%d, want one rectangular global InfoNCE call", accel.squareCalls, accel.rectCalls)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
}

func TestEmbeddingTrainerTrainTeacherDistilledHardNegativeStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	trainer.config.TeacherLossWeight = 0.5
	trainer.config.TeacherTemperature = 1
	batch := tinyEmbeddingHardNegativeDataset()
	batch[0].TeacherScores = []float32{0.9, 0.7}
	batch[1].TeacherScores = []float32{0.8, 0.6}

	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train teacher-distilled hard-negative step: %v", err)
	}
	if metrics.BatchSize != 12 {
		t.Fatalf("batch size = %d, want 12 rectangular plus teacher query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 {
		t.Fatalf("loss = %f, want non-negative", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
}

func TestEmbeddingTrainerTrainHardNegativeContrastiveStepUsesRectangularAccelerator(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	accel := &countingContrastiveAccelerator{}
	trainer.contrastiveAccel = accel

	_, err := trainer.TrainHardNegativeContrastiveStep(tinyEmbeddingHardNegativeDataset())
	if err != nil {
		t.Fatalf("train hard-negative step: %v", err)
	}
	if accel.rectCalls != 1 {
		t.Fatalf("rectangular calls = %d, want 1", accel.rectCalls)
	}
	if accel.squareCalls != 0 {
		t.Fatalf("square calls = %d, want 0", accel.squareCalls)
	}
	if accel.queryRows != 2 || accel.candidateRows != 4 || accel.width == 0 {
		t.Fatalf("accelerator shape query=%d candidate=%d width=%d, want 2x4xwidth", accel.queryRows, accel.candidateRows, accel.width)
	}
	if len(accel.targetIndexes) != 2 || accel.targetIndexes[0] != 0 || accel.targetIndexes[1] != 2 {
		t.Fatalf("target indexes = %v, want [0 2]", accel.targetIndexes)
	}
}

func TestSpreadHardNegativeOrderByQuerySeparatesRepeatedQueries(t *testing.T) {
	trainSet := []EmbeddingHardNegativeExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{10}, NegativeTokens: [][]int32{{90}}, QueryMask: []int32{1}},
		{QueryTokens: []int32{0}, PositiveTokens: []int32{11}, NegativeTokens: [][]int32{{91}}, QueryMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{12}, NegativeTokens: [][]int32{{92}}, QueryMask: []int32{1}},
		{QueryTokens: []int32{2}, PositiveTokens: []int32{13}, NegativeTokens: [][]int32{{93}}, QueryMask: []int32{1}},
		{QueryTokens: []int32{3}, PositiveTokens: []int32{14}, NegativeTokens: [][]int32{{94}}, QueryMask: []int32{1}},
		{QueryTokens: []int32{4}, PositiveTokens: []int32{15}, NegativeTokens: [][]int32{{95}}, QueryMask: []int32{1}},
	}
	got := spreadHardNegativeOrderByQuery(trainSet, []int{0, 1, 2, 3, 4, 5})
	if len(got) != len(trainSet) {
		t.Fatalf("spread order length = %d, want %d", len(got), len(trainSet))
	}
	seenIndexes := map[int]bool{}
	for _, idx := range got {
		if seenIndexes[idx] {
			t.Fatalf("spread order repeats index %d: %v", idx, got)
		}
		seenIndexes[idx] = true
	}
	for start := 0; start < len(got); start += 3 {
		end := start + 3
		if end > len(got) {
			end = len(got)
		}
		seenQueries := map[string]bool{}
		for _, idx := range got[start:end] {
			key := embeddingBatchSequenceKey(trainSet[idx].QueryTokens, trainSet[idx].QueryMask)
			if seenQueries[key] {
				t.Fatalf("chunk %v has repeated query in order %v", got[start:end], got)
			}
			seenQueries[key] = true
		}
	}
}

func TestHardNegativeSourceWeightedOrderBuildsBalancedBatches(t *testing.T) {
	trainSet := []EmbeddingHardNegativeExample{
		{Source: "scifact", QueryTokens: []int32{0}, PositiveTokens: []int32{10}, NegativeTokens: [][]int32{{90}}, QueryMask: []int32{1}},
		{Source: "nfcorpus", QueryTokens: []int32{1}, PositiveTokens: []int32{11}, NegativeTokens: [][]int32{{91}}, QueryMask: []int32{1}},
		{Source: "nfcorpus", QueryTokens: []int32{2}, PositiveTokens: []int32{12}, NegativeTokens: [][]int32{{92}}, QueryMask: []int32{1}},
		{Source: "fiqa:model", QueryTokens: []int32{3}, PositiveTokens: []int32{13}, NegativeTokens: [][]int32{{93}}, QueryMask: []int32{1}},
		{Source: "fiqa", QueryTokens: []int32{4}, PositiveTokens: []int32{14}, NegativeTokens: [][]int32{{94}}, QueryMask: []int32{1}},
		{Source: "scifact", QueryTokens: []int32{5}, PositiveTokens: []int32{15}, NegativeTokens: [][]int32{{95}}, QueryMask: []int32{1}},
	}
	got := hardNegativeSourceWeightedOrder(trainSet, []int{0, 1, 2, 3, 4, 5}, 4, map[string]int{
		"scifact":  1,
		"nfcorpus": 2,
		"fiqa":     1,
	}, false)
	want := []int{0, 1, 2, 3, 5, 4}
	if len(got) != len(want) {
		t.Fatalf("weighted order length = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weighted order = %v, want %v", got, want)
		}
	}
}

func TestHardNegativeSourceWeightUsesFamilyUnlessExactWeightExists(t *testing.T) {
	weights := normalizeHardNegativeSourceWeights(map[string]int{
		"fiqa":       2,
		"fiqa:model": 4,
	})
	if got := hardNegativeSourceWeight(weights, "fiqa:bm25"); got != 2 {
		t.Fatalf("fiqa:bm25 weight = %d, want 2", got)
	}
	if got := hardNegativeSourceWeight(weights, "fiqa:model"); got != 4 {
		t.Fatalf("fiqa:model weight = %d, want 4", got)
	}
	if got := hardNegativeSourceGroupKey(weights, "fiqa:model"); got != "fiqa:model" {
		t.Fatalf("fiqa:model group key = %q, want fiqa:model", got)
	}
}

func TestHardNegativeTeacherTemperatureUsesFamilyUnlessExactExists(t *testing.T) {
	temperatures := normalizeHardNegativeTeacherTemperatures(map[string]float32{
		"fiqa":       20,
		"fiqa:model": 1.5,
		"*":          7,
	})
	if got := hardNegativeTeacherTemperature(temperatures, "fiqa:bm25", 3); got != 20 {
		t.Fatalf("fiqa:bm25 temperature = %f, want 20", got)
	}
	if got := hardNegativeTeacherTemperature(temperatures, "fiqa:model", 3); got != 1.5 {
		t.Fatalf("fiqa:model temperature = %f, want 1.5", got)
	}
	if got := hardNegativeTeacherTemperature(temperatures, "scifact", 3); got != 7 {
		t.Fatalf("scifact temperature = %f, want wildcard temperature", got)
	}
	if got := hardNegativeTeacherTemperature(nil, "scifact", 3); got != 3 {
		t.Fatalf("fallback temperature = %f, want fallback", got)
	}
}

func TestHardNegativeTeacherWeightUsesExactFamilyWildcardAndDefault(t *testing.T) {
	weights := normalizeHardNegativeTeacherWeights(map[string]float32{
		"fiqa":       0.25,
		"fiqa:model": 0,
		"*":          0.75,
	})
	if got := hardNegativeTeacherWeight(weights, "fiqa:bm25"); got != 0.25 {
		t.Fatalf("fiqa:bm25 teacher weight = %f, want 0.25", got)
	}
	if got := hardNegativeTeacherWeight(weights, "fiqa:model"); got != 0 {
		t.Fatalf("fiqa:model teacher weight = %f, want exact zero", got)
	}
	if got := hardNegativeTeacherWeight(weights, "scifact"); got != 0.75 {
		t.Fatalf("scifact teacher weight = %f, want wildcard", got)
	}
	if got := hardNegativeTeacherWeight(nil, "nfcorpus"); got != 1 {
		t.Fatalf("default teacher weight = %f, want 1", got)
	}
}

func TestAccumulateTeacherDistributionHardNegativeGradsSkipsZeroWeightedSource(t *testing.T) {
	queries := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{1, 0}},
	}
	spans := []embeddingCandidateSpan{
		{Start: 0, End: 2},
		{Start: 2, End: 4},
	}
	teacherScores := [][]float32{
		{1, 0},
		{0, 1},
	}
	teacherTemperatures := []float32{1, 1}
	teacherSourceWeights := []float32{1, 0}
	queryGrads := newEmbeddingPooledGradBuffers(queries)
	candidateGrads := newEmbeddingPooledGradBuffers(candidates)

	loss, score, pairs := accumulateTeacherDistributionHardNegativeGrads(queries, candidates, spans, teacherScores, teacherTemperatures, teacherSourceWeights, 0.05, 1, queryGrads, candidateGrads)

	if pairs != 2 {
		t.Fatalf("teacher pairs = %d, want only first source's two candidates", pairs)
	}
	if loss <= 0 {
		t.Fatalf("teacher loss = %f, want positive contribution from weighted source", loss)
	}
	if score == 0 {
		t.Fatalf("teacher score = %f, want weighted source model scores counted", score)
	}
	for i, grad := range queryGrads[1] {
		if grad != 0 {
			t.Fatalf("zero-weighted query grad[%d] = %f, want 0", i, grad)
		}
	}
	for row := 2; row < 4; row++ {
		for col, grad := range candidateGrads[row] {
			if grad != 0 {
				t.Fatalf("zero-weighted candidate %d grad[%d] = %f, want 0", row, col, grad)
			}
		}
	}
}

func TestNormalizeHardNegativeTeacherScoresForRunSourceZScore(t *testing.T) {
	trainSet := []EmbeddingHardNegativeExample{
		{Source: "scifact", TeacherScores: []float32{10, 20}},
		{Source: "scifact", TeacherScores: []float32{30, 40}},
		{Source: "fiqa:model", TeacherScores: []float32{0.1, 0.3}},
		{Source: "fiqa", TeacherScores: []float32{100, 300}},
		{Source: "missing"},
	}
	got, err := normalizeHardNegativeTeacherScoresForRun(trainSet, "source-zscore")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(trainSet) {
		t.Fatalf("normalized examples = %d, want %d", len(got), len(trainSet))
	}
	wantFirst := []float32{-1.3416408, -0.4472136}
	for i, want := range wantFirst {
		if math.Abs(float64(got[0].TeacherScores[i]-want)) > 1e-5 {
			t.Fatalf("scifact score %d = %f, want %f", i, got[0].TeacherScores[i], want)
		}
	}
	if got[2].TeacherScores[0] >= got[2].TeacherScores[1] {
		t.Fatalf("fiqa:model ordering not preserved: %v", got[2].TeacherScores)
	}
	if len(got[4].TeacherScores) != 0 {
		t.Fatalf("missing teacher scores = %v, want empty", got[4].TeacherScores)
	}
	if trainSet[0].TeacherScores[0] != 10 {
		t.Fatalf("input teacher scores mutated: %v", trainSet[0].TeacherScores)
	}
}

func TestNormalizeHardNegativeTeacherScoresForRunExampleZScore(t *testing.T) {
	got, err := normalizeHardNegativeTeacherScoresForRun([]EmbeddingHardNegativeExample{
		{Source: "scifact", TeacherScores: []float32{2, 4, 6}},
	}, "example_zscore")
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-1.2247449, 0, 1.2247449}
	for i := range want {
		if math.Abs(float64(got[0].TeacherScores[i]-want[i])) > 1e-5 {
			t.Fatalf("example zscore %d = %f, want %f", i, got[0].TeacherScores[i], want[i])
		}
	}
}

type countingContrastiveAccelerator struct {
	squareCalls   int
	rectCalls     int
	queryRows     int
	candidateRows int
	width         int
	targetIndexes []int
}

func (a *countingContrastiveAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *countingContrastiveAccelerator) RunInfoNCE(query, positive *backend.Tensor, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	a.squareCalls++
	if query == nil || positive == nil || query.Rank() != 2 || positive.Rank() != 2 {
		return backend.ContrastiveGradResult{}, nil
	}
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32(query.Shape, make([]float32, len(query.F32))),
		PositiveGrads: backend.NewTensorF32(positive.Shape, make([]float32, len(positive.F32))),
		LossSum:       1,
		ScoreSum:      1,
	}, nil
}

func (a *countingContrastiveAccelerator) RunInfoNCEWithTargets(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	a.rectCalls++
	a.targetIndexes = append([]int(nil), targetIndexes...)
	if query == nil || candidates == nil || query.Rank() != 2 || candidates.Rank() != 2 {
		return backend.ContrastiveGradResult{}, nil
	}
	a.queryRows = query.Shape[0]
	a.candidateRows = candidates.Shape[0]
	a.width = query.Shape[1]
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32(query.Shape, make([]float32, len(query.F32))),
		PositiveGrads: backend.NewTensorF32(candidates.Shape, make([]float32, len(candidates.F32))),
		LossSum:       1,
		ScoreSum:      float32(query.Shape[0] * candidates.Shape[0]),
	}, nil
}

func (a *countingContrastiveAccelerator) Stats() backend.ContrastiveAcceleratorStats {
	return backend.ContrastiveAcceleratorStats{RunCalls: int64(a.squareCalls + a.rectCalls)}
}

func (a *countingContrastiveAccelerator) Close() {}

func TestContrastiveCosineFastPathMatchesCosineGrad(t *testing.T) {
	left := []float32{0.25, -0.5, 0.75, 1.25}
	right := []float32{-0.75, 0.5, 0.125, 1.5}
	score, wantLeft, wantRight := cosineGrad(left, right)
	leftNorm := vectorNorm(left)
	rightNorm := vectorNorm(right)

	gotScore := cosineScoreWithNorms(left, right, leftNorm, rightNorm)
	assertClose(t, gotScore, score, 0.000001)

	gotLeft := make([]float32, len(left))
	gotRight := make([]float32, len(right))
	accumulateCosineGradFromScore(left, right, leftNorm, rightNorm, gotScore, 1, gotLeft, gotRight)
	for i := range wantLeft {
		assertClose(t, gotLeft[i], wantLeft[i], 0.000001)
		assertClose(t, gotRight[i], wantRight[i], 0.000001)
	}
}

func BenchmarkAccumulateInfoNCEContrastiveGrads128x64(b *testing.B) {
	queries, positives := syntheticContrastiveEncodings(128, 64)
	queryGrads := make([][]float32, len(queries))
	positiveGrads := make([][]float32, len(positives))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
		positiveGrads[i] = make([]float32, len(positives[i].pooled))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := range queryGrads {
			clear(queryGrads[row])
			clear(positiveGrads[row])
		}
		accumulateInfoNCEContrastiveGrads(queries, positives, 0.05, queryGrads, positiveGrads)
	}
}

func syntheticContrastiveEncodings(rows, width int) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence) {
	queries := make([]*embeddingEncodedSequence, rows)
	positives := make([]*embeddingEncodedSequence, rows)
	for row := 0; row < rows; row++ {
		query := make([]float32, width)
		positive := make([]float32, width)
		for col := 0; col < width; col++ {
			value := float32(((row+1)*(col+3))%23) / 23
			query[col] = value - 0.5
			positive[col] = value*0.75 + float32((row+col)%7)/17 - 0.35
		}
		queries[row] = &embeddingEncodedSequence{pooled: query}
		positives[row] = &embeddingEncodedSequence{pooled: positive}
	}
	return queries, positives
}

func newTinyTrainableEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	return trainer
}

func newTinyTrainable3DEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_3d_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_3d_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 3}, []float32{
			1, 0, 0.5,
			0, 1, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	return trainer
}

func tinyEmbeddingPairDataset() []EmbeddingPairExample {
	return []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}
}

func tinyEmbeddingContrastiveDataset() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
}

func tinyEmbeddingHardNegativeDataset() []EmbeddingHardNegativeExample {
	return []EmbeddingHardNegativeExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, NegativeTokens: [][]int32{{1}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, NegativeTokens: [][]int32{{0}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}},
	}
}

func tinyEncoderPairDataset() []EmbeddingPairExample {
	return []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{0, 0}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: -0.5},
		{LeftTokens: []int32{1, 1}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: -0.5},
	}
}

func tinyEncoderContrastiveDataset() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
}
