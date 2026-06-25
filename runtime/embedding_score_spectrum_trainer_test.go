package eosruntime

import (
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

	_, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, 1, queryGrads, candidateGrads)
	if err != nil {
		t.Fatalf("accumulate score-spectrum grads: %v", err)
	}
	if candidateGrads[2][0] != 0 || candidateGrads[2][1] != 0 {
		t.Fatalf("excluded candidate grad = %v, want zero", candidateGrads[2])
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
