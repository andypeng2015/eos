package eosruntime

import (
	"math"
	"strings"
	"testing"
)

func TestScoreSpectrumLossOnePositiveOneHotMatchesGroupedInfoNCERowCrossEntropy(t *testing.T) {
	scores := []float32{0.25, -0.5, 1.1}
	target := []float32{0, 1, 0}
	hardEligible := []bool{true, false, true}

	got, err := scoreSpectrumLossAndGrad(scores, []int{1}, hardEligible, target, 1, 1, 0)
	if err != nil {
		t.Fatalf("score spectrum loss: %v", err)
	}

	// With a single positive and every other candidate active as a hard negative,
	// the multi-positive loss is the same single-target cross-entropy as grouped InfoNCE.
	want := singleTargetCrossEntropy(scores, 1)
	assertClose32(t, got.Loss, want, 1e-6, "loss")
	wantGrad := singleTargetCrossEntropyGrad(scores, 1)
	for i := range wantGrad {
		assertClose32(t, got.Grad[i], wantGrad[i], 1e-6, "grad")
	}
	assertFiniteProbabilityVector(t, got.HardProbs, 1e-6, "hard probs")
}

func TestScoreSpectrumLossCandidateOrderPermutationInvariant(t *testing.T) {
	scores := []float32{0.7, -0.2, 1.4, 0.1}
	positives := []int{0, 2}
	hardEligible := []bool{false, true, false, true}
	probs := []float32{0.55, 0.05, 0.35, 0.05}
	base, err := scoreSpectrumLossAndGrad(scores, positives, hardEligible, probs, 1, 0.75, 0.25)
	if err != nil {
		t.Fatalf("base loss: %v", err)
	}

	perm := []int{2, 0, 3, 1}
	permutedScores := permuteFloat32(scores, perm)
	permutedHardEligible := permuteBool(hardEligible, perm)
	permutedProbs := permuteFloat32(probs, perm)
	permutedPositives := []int{0, 1}
	got, err := scoreSpectrumLossAndGrad(permutedScores, permutedPositives, permutedHardEligible, permutedProbs, 1, 0.75, 0.25)
	if err != nil {
		t.Fatalf("permuted loss: %v", err)
	}

	assertClose32(t, got.Loss, base.Loss, 1e-6, "loss")
	for newIndex, oldIndex := range perm {
		assertClose32(t, got.Grad[newIndex], base.Grad[oldIndex], 1e-6, "grad")
	}
}

func TestScoreSpectrumLossIncreasingEitherPositiveLowersHardLoss(t *testing.T) {
	scores := []float32{0, 0.5, -0.25, 1}
	positives := []int{0, 2}
	hardEligible := []bool{false, true, false, true}
	base, err := scoreSpectrumLossAndGrad(scores, positives, hardEligible, nil, 1, 1, 0)
	if err != nil {
		t.Fatalf("base loss: %v", err)
	}

	firstHigher := append([]float32(nil), scores...)
	firstHigher[0] += 0.8
	first, err := scoreSpectrumLossAndGrad(firstHigher, positives, hardEligible, nil, 1, 1, 0)
	if err != nil {
		t.Fatalf("first positive higher loss: %v", err)
	}
	if first.Loss >= base.Loss {
		t.Fatalf("raising first positive loss = %f, want below base %f", first.Loss, base.Loss)
	}

	secondHigher := append([]float32(nil), scores...)
	secondHigher[2] += 0.8
	second, err := scoreSpectrumLossAndGrad(secondHigher, positives, hardEligible, nil, 1, 1, 0)
	if err != nil {
		t.Fatalf("second positive higher loss: %v", err)
	}
	if second.Loss >= base.Loss {
		t.Fatalf("raising second positive loss = %f, want below base %f", second.Loss, base.Loss)
	}
}

func TestScoreSpectrumLossRejectsPositiveMarkedHardEligible(t *testing.T) {
	_, err := scoreSpectrumLossAndGrad(
		[]float32{0.1, 0.2},
		[]int{0},
		[]bool{true, true},
		nil,
		1,
		1,
		0,
	)
	if err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("error = %v, want positive hard-eligible rejection", err)
	}
}

func TestScoreSpectrumLossExcludedCandidatesHaveZeroHardGradient(t *testing.T) {
	got, err := scoreSpectrumLossAndGrad(
		[]float32{0.6, 1.2, -0.3, 0.4},
		[]int{0},
		[]bool{false, true, false, false},
		nil,
		1,
		1,
		0,
	)
	if err != nil {
		t.Fatalf("hard loss: %v", err)
	}
	if got.Grad[2] != 0 || got.Grad[3] != 0 {
		t.Fatalf("excluded hard gradients = [%f %f], want zero", got.Grad[2], got.Grad[3])
	}
	if got.HardProbs[2] != 0 || got.HardProbs[3] != 0 {
		t.Fatalf("excluded hard probabilities = [%f %f], want zero", got.HardProbs[2], got.HardProbs[3])
	}
}

func TestScoreSpectrumLossSoftProbabilitiesFiniteNonnegativeAlignedAndSumOne(t *testing.T) {
	got, err := scoreSpectrumLossAndGrad(
		[]float32{0.2, 0.4, -0.1},
		nil,
		nil,
		[]float32{0.2, 0.3, 0.5},
		1,
		0,
		1,
	)
	if err != nil {
		t.Fatalf("soft loss: %v", err)
	}
	if len(got.SoftProbs) != 3 {
		t.Fatalf("soft probs length = %d, want 3", len(got.SoftProbs))
	}
	sum := float32(0)
	for i, prob := range got.SoftProbs {
		if prob < 0 || math.IsNaN(float64(prob)) || math.IsInf(float64(prob), 0) {
			t.Fatalf("soft probs[%d] = %f, want finite non-negative", i, prob)
		}
		sum += prob
	}
	assertClose32(t, sum, 1, 1e-6, "soft probability sum")

	invalid := []struct {
		name  string
		probs []float32
	}{
		{name: "length", probs: []float32{1, 0}},
		{name: "negative", probs: []float32{0.7, -0.1, 0.4}},
		{name: "nan", probs: []float32{float32(math.NaN()), 0.5, 0.5}},
		{name: "sum", probs: []float32{0.2, 0.2, 0.2}},
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := scoreSpectrumLossAndGrad([]float32{0.2, 0.4, -0.1}, nil, nil, tc.probs, 1, 0, 1); err == nil {
				t.Fatal("expected invalid probability error")
			}
		})
	}
}

func TestScoreSpectrumLossTemperatureMatchesGroupedInfoNCE(t *testing.T) {
	scores := []float32{0.25, -0.5, 1.1}
	target := []float32{0, 1, 0}
	hardEligible := []bool{true, false, true}
	temperature := float32(0.2)
	got, err := scoreSpectrumLossAndGrad(scores, []int{1}, hardEligible, target, temperature, 1, 0)
	if err != nil {
		t.Fatalf("score spectrum loss: %v", err)
	}
	probs := make([]float32, len(scores))
	wantLoss := infoNCERowProbsAndLossInto(scores, 1, temperature, probs)
	assertClose32(t, got.Loss, wantLoss, 1e-6, "loss")
	for i, prob := range probs {
		targetProb := float32(0)
		if i == 1 {
			targetProb = 1
		}
		wantGrad := (prob - targetProb) / temperature
		assertClose32(t, got.Grad[i], wantGrad, 1e-6, "grad")
	}
}

func TestScoreSpectrumRecoveryLossGradientSignsAndTopK(t *testing.T) {
	got, err := scoreSpectrumLossAndGrad(
		[]float32{0.2, 0.7, -0.3, 0.5},
		[]int{0},
		[]bool{false, true, true, true},
		nil,
		0.5,
		0,
		0,
		scoreSpectrumRecoveryLossOptions{Enabled: true, Weight: 1, Margin: 0.1, TopK: 2, Tau: 0.25},
	)
	if err != nil {
		t.Fatalf("recovery loss: %v", err)
	}
	if got.Loss <= 0 {
		t.Fatalf("recovery loss = %f, want positive", got.Loss)
	}
	if got.Grad[0] >= 0 {
		t.Fatalf("positive grad = %f, want negative", got.Grad[0])
	}
	if got.Grad[1] <= 0 || got.Grad[3] <= 0 {
		t.Fatalf("selected hard-negative grads = [%f %f], want positive", got.Grad[1], got.Grad[3])
	}
	if got.Grad[2] != 0 {
		t.Fatalf("non-topK hard-negative grad = %f, want zero", got.Grad[2])
	}
}

func TestScoreSpectrumRecoveryLossValidation(t *testing.T) {
	tests := []struct {
		name         string
		scores       []float32
		positives    []int
		hardEligible []bool
		margin       float32
		topK         int
		tau          float32
		want         string
	}{
		{name: "missing positive", scores: []float32{0, 1}, positives: nil, hardEligible: []bool{false, true}, topK: 1, tau: 0.1, want: "positive"},
		{name: "missing hard", scores: []float32{0, 1}, positives: []int{0}, hardEligible: []bool{false, false}, topK: 1, tau: 0.1, want: "hard-negative"},
		{name: "bad topk", scores: []float32{0, 1}, positives: []int{0}, hardEligible: []bool{false, true}, topK: 0, tau: 0.1, want: "topK"},
		{name: "bad tau", scores: []float32{0, 1}, positives: []int{0}, hardEligible: []bool{false, true}, topK: 1, tau: 0, want: "tau"},
		{name: "bad margin", scores: []float32{0, 1}, positives: []int{0}, hardEligible: []bool{false, true}, margin: -0.1, topK: 1, tau: 0.1, want: "margin"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := scoreSpectrumLossAndGrad(
				tc.scores,
				tc.positives,
				tc.hardEligible,
				nil,
				0.5,
				0,
				0,
				scoreSpectrumRecoveryLossOptions{Enabled: true, Weight: 1, Margin: tc.margin, TopK: tc.topK, Tau: tc.tau},
			)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want containing %q", err, tc.want)
			}
		})
	}
}

func assertFiniteProbabilityVector(t *testing.T, probs []float32, tol float32, name string) {
	t.Helper()
	sum := float32(0)
	for i, prob := range probs {
		if prob < 0 || math.IsNaN(float64(prob)) || math.IsInf(float64(prob), 0) {
			t.Fatalf("%s[%d] = %f, want finite non-negative", name, i, prob)
		}
		sum += prob
	}
	assertClose32(t, sum, 1, tol, name+" sum")
}

func singleTargetCrossEntropy(scores []float32, target int) float32 {
	probs := make([]float32, len(scores))
	softmaxScoresInto(scores, 1, probs)
	prob := probs[target]
	if prob < 1e-12 {
		prob = 1e-12
	}
	return -float32(math.Log(float64(prob)))
}

func singleTargetCrossEntropyGrad(scores []float32, target int) []float32 {
	probs := make([]float32, len(scores))
	softmaxScoresInto(scores, 1, probs)
	probs[target]--
	return probs
}

func permuteFloat32(values []float32, perm []int) []float32 {
	out := make([]float32, len(perm))
	for i, oldIndex := range perm {
		out[i] = values[oldIndex]
	}
	return out
}

func permuteBool(values []bool, perm []int) []bool {
	out := make([]bool, len(perm))
	for i, oldIndex := range perm {
		out[i] = values[oldIndex]
	}
	return out
}

func assertClose32(t *testing.T, got, want, tol float32, name string) {
	t.Helper()
	if float32(math.Abs(float64(got-want))) > tol {
		t.Fatalf("%s = %.9f, want %.9f", name, got, want)
	}
}
