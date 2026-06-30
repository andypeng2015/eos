package eosruntime

import (
	"fmt"
	"math"
)

// VectorDistillLossResult holds the scalar loss and d_loss/d_proj gradient.
type VectorDistillLossResult struct {
	Loss     float32
	GradProj []float32 // gradient w.r.t. the projected 384-d vector
}

// VectorDistillLossAndGrad computes the combined MSE + cosine distillation loss
// between a projected student embedding (proj, teacherDim-d) and a teacher
// embedding (teacher, teacherDim-d):
//
//	loss = 0.5 * MSE(proj, teacher) + 0.5 * (1 − cosine(proj, teacher))
//
// Returns the scalar loss and d_loss/d_proj so the caller can back-propagate
// through the linear distillation projection.
func VectorDistillLossAndGrad(proj, teacher []float32) (VectorDistillLossResult, error) {
	if len(proj) == 0 || len(teacher) == 0 {
		return VectorDistillLossResult{}, fmt.Errorf("vector_distill: embeddings must be non-empty")
	}
	if len(proj) != len(teacher) {
		return VectorDistillLossResult{}, fmt.Errorf("vector_distill: proj length %d != teacher length %d", len(proj), len(teacher))
	}
	d := len(proj)

	// --- MSE ---
	var mseLoss float32
	for i := range proj {
		diff := proj[i] - teacher[i]
		mseLoss += diff * diff
	}
	mseLoss /= float32(d)

	// --- cosine ---
	var projSqNorm, teacherSqNorm, dot float64
	for i := range proj {
		projSqNorm += float64(proj[i]) * float64(proj[i])
		teacherSqNorm += float64(teacher[i]) * float64(teacher[i])
		dot += float64(proj[i]) * float64(teacher[i])
	}
	projNorm := float32(math.Sqrt(projSqNorm))
	teacherNorm := float32(math.Sqrt(teacherSqNorm))

	var cosine float32
	var cosineGradProj []float32
	if projNorm > 0 && teacherNorm > 0 {
		denom := projNorm * teacherNorm
		cosine = float32(dot) / denom
		// d(cosine)/d(proj[k]) = teacher[k]/denom − proj[k]*cosine/projNorm²
		projScale := cosine / (projNorm * projNorm)
		cosineGradProj = make([]float32, d)
		for k := range proj {
			cosineGradProj[k] = teacher[k]/denom - proj[k]*projScale
		}
	}

	// --- combined ---
	// total = 0.5*mse + 0.5*(1−cosine)
	// d_total/d_proj[k]
	//   = 0.5 * d_mse/d_proj[k]  + 0.5 * d(1−cosine)/d_proj[k]
	//   = 0.5 * (2/d)*(proj[k]−teacher[k]) − 0.5 * d_cosine/d_proj[k]
	totalLoss := 0.5*mseLoss + 0.5*(1-cosine)
	gradProj := make([]float32, d)
	cosineLossScale := float32(0.5)
	for k := range proj {
		mseGrad := float32(1)/float32(d) * (proj[k] - teacher[k]) // 0.5 * 2/d * diff
		var cosGrad float32
		if cosineGradProj != nil {
			cosGrad = -cosineLossScale * cosineGradProj[k] // -0.5 * d_cosine/d_proj
		}
		gradProj[k] = mseGrad + cosGrad
	}

	return VectorDistillLossResult{Loss: totalLoss, GradProj: gradProj}, nil
}

// accumulateVectorDistillProjectionGrads back-propagates through the linear
// distillation projection W (shape [inputDim, teacherDim], stored row-major)
// and accumulates into gradStudent (inputDim) and gradW (inputDim*teacherDim).
//
// forward: proj[k] = Σ_i W[i*teacherDim + k] * student[i]
// backward:
//
//	gradStudent[i] += Σ_k W[i*teacherDim + k] * gradProj[k]
//	gradW[i*teacherDim + k] += student[i] * gradProj[k]
func accumulateVectorDistillProjectionGrads(
	student []float32, // inputDim
	gradProj []float32, // teacherDim — d_loss/d_proj
	W []float32, // inputDim * teacherDim
	gradStudent []float32, // accumulate in-place
	gradW []float32, // accumulate in-place
) {
	if len(gradProj) == 0 || len(student) == 0 {
		return
	}
	inputDim := len(student)
	teacherDim := len(gradProj)
	for i := 0; i < inputDim; i++ {
		si := student[i]
		base := i * teacherDim
		var gs float32
		for k := 0; k < teacherDim; k++ {
			gs += W[base+k] * gradProj[k]
			gradW[base+k] += si * gradProj[k]
		}
		gradStudent[i] += gs
	}
}

// applyVectorDistillProjectionAdamW applies an AdamW step to the distillation
// projection weights using the same hyperparams as the student encoder.
func applyVectorDistillProjectionAdamW(
	W, m1, m2, gradW []float32,
	lr, beta1, beta2, eps, weightDecay float32,
	step int,
) {
	if len(W) == 0 || step <= 0 {
		return
	}
	b1 := float64(beta1)
	b2 := float64(beta2)
	corr1 := float32(1 - math.Pow(b1, float64(step)))
	corr2 := float32(1 - math.Pow(b2, float64(step)))
	for i := range W {
		g := gradW[i]
		if weightDecay != 0 {
			g += weightDecay * W[i]
		}
		m1[i] = beta1*m1[i] + (1-beta1)*g
		m2[i] = beta2*m2[i] + (1-beta2)*g*g
		mHat := m1[i]
		vHat := m2[i]
		if corr1 != 0 {
			mHat /= corr1
		}
		if corr2 != 0 {
			vHat /= corr2
		}
		W[i] -= lr * (mHat / (float32(math.Sqrt(float64(vHat))) + eps))
	}
}
