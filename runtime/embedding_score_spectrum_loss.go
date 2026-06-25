package eosruntime

import (
	"fmt"
	"math"
)

type scoreSpectrumLossResult struct {
	Loss      float32
	Grad      []float32
	HardProbs []float32
	SoftProbs []float32
}

func scoreSpectrumLossAndGrad(scores []float32, positives []int, hardEligible []bool, probs []float32, temperature, hardWeight, softWeight float32) (scoreSpectrumLossResult, error) {
	var result scoreSpectrumLossResult
	if len(scores) == 0 {
		return result, fmt.Errorf("score-spectrum loss requires at least one score")
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	if !isFinite32(temperature) {
		return result, fmt.Errorf("temperature must be finite and positive")
	}
	if !isFiniteNonNegative(hardWeight) {
		return result, fmt.Errorf("hardWeight must be finite and non-negative")
	}
	if !isFiniteNonNegative(softWeight) {
		return result, fmt.Errorf("softWeight must be finite and non-negative")
	}
	for i, score := range scores {
		if !isFinite32(score) {
			return result, fmt.Errorf("scores[%d] must be finite", i)
		}
	}
	if len(hardEligible) != 0 && len(hardEligible) != len(scores) {
		return result, fmt.Errorf("hardEligible length %d does not match scores length %d", len(hardEligible), len(scores))
	}
	if softWeight > 0 || len(probs) > 0 {
		if err := validateScoreSpectrumProbabilities(probs, len(scores)); err != nil {
			return result, err
		}
	}

	scaledScores := make([]float32, len(scores))
	for i, score := range scores {
		scaledScores[i] = score / temperature
	}
	result.Grad = make([]float32, len(scores))
	if hardWeight > 0 {
		hardLoss, hardGrad, hardProbs, err := scoreSpectrumHardLossAndGrad(scaledScores, positives, hardEligible)
		if err != nil {
			return result, err
		}
		result.Loss += hardWeight * hardLoss
		result.HardProbs = hardProbs
		for i, grad := range hardGrad {
			result.Grad[i] += hardWeight * grad / temperature
		}
	}
	if softWeight > 0 {
		softLoss, softGrad, softProbs := scoreSpectrumSoftLossAndGrad(scaledScores, probs)
		result.Loss += softWeight * softLoss
		result.SoftProbs = softProbs
		for i, grad := range softGrad {
			result.Grad[i] += softWeight * grad / temperature
		}
	}
	return result, nil
}

func scoreSpectrumHardLossAndGrad(scores []float32, positives []int, hardEligible []bool) (float32, []float32, []float32, error) {
	positive := make([]bool, len(scores))
	for _, idx := range positives {
		if idx < 0 || idx >= len(scores) {
			return 0, nil, nil, fmt.Errorf("positive index %d out of range", idx)
		}
		if positive[idx] {
			return 0, nil, nil, fmt.Errorf("positive index %d is duplicated", idx)
		}
		positive[idx] = true
	}
	if len(positives) == 0 {
		return 0, nil, nil, fmt.Errorf("hard loss requires at least one positive")
	}

	activeCount := 0
	for i := range scores {
		eligible := i < len(hardEligible) && hardEligible[i]
		if positive[i] && eligible {
			return 0, nil, nil, fmt.Errorf("positive index %d cannot be hard-negative eligible", i)
		}
		if positive[i] || eligible {
			activeCount++
		}
	}
	if activeCount == 0 {
		return 0, nil, nil, fmt.Errorf("hard loss requires at least one active candidate")
	}

	maxActive := float32(math.Inf(-1))
	maxPositive := float32(math.Inf(-1))
	for i, score := range scores {
		eligible := i < len(hardEligible) && hardEligible[i]
		if positive[i] || eligible {
			if score > maxActive {
				maxActive = score
			}
		}
		if positive[i] && score > maxPositive {
			maxPositive = score
		}
	}

	sumActive := float32(0)
	sumPositive := float32(0)
	activeExp := make([]float32, len(scores))
	positiveExp := make([]float32, len(scores))
	for i, score := range scores {
		eligible := i < len(hardEligible) && hardEligible[i]
		if positive[i] || eligible {
			value := float32(math.Exp(float64(score - maxActive)))
			activeExp[i] = value
			sumActive += value
		}
		if positive[i] {
			value := float32(math.Exp(float64(score - maxPositive)))
			positiveExp[i] = value
			sumPositive += value
		}
	}
	if sumActive == 0 || sumPositive == 0 {
		return 0, nil, nil, fmt.Errorf("hard loss encountered zero probability mass")
	}

	grad := make([]float32, len(scores))
	hardProbs := make([]float32, len(scores))
	for i := range scores {
		if activeExp[i] > 0 {
			hardProbs[i] = activeExp[i] / sumActive
			grad[i] += hardProbs[i]
		}
		if positiveExp[i] > 0 {
			grad[i] -= positiveExp[i] / sumPositive
		}
	}
	loss := float32(math.Log(float64(sumActive)) + float64(maxActive) - math.Log(float64(sumPositive)) - float64(maxPositive))
	return loss, grad, hardProbs, nil
}

func scoreSpectrumSoftLossAndGrad(scores []float32, target []float32) (float32, []float32, []float32) {
	model := make([]float32, len(scores))
	softmaxScoresInto(scores, 1, model)
	grad := make([]float32, len(scores))
	loss := float32(0)
	for i, prob := range model {
		clamped := prob
		if clamped < 1e-12 {
			clamped = 1e-12
		}
		loss -= target[i] * float32(math.Log(float64(clamped)))
		grad[i] = prob - target[i]
	}
	return loss, grad, model
}

func validateScoreSpectrumProbabilities(probs []float32, want int) error {
	if len(probs) != want {
		return fmt.Errorf("probabilities length %d does not match scores length %d", len(probs), want)
	}
	sum := float32(0)
	for i, prob := range probs {
		if !isFiniteNonNegative(prob) {
			return fmt.Errorf("probabilities[%d] must be finite and non-negative", i)
		}
		sum += prob
	}
	if float32(math.Abs(float64(sum-1))) > 1e-4 {
		return fmt.Errorf("probabilities must sum to 1, got %g", sum)
	}
	return nil
}

func isFiniteNonNegative(value float32) bool {
	return isFinite32(value) && value >= 0
}

func isFinite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
