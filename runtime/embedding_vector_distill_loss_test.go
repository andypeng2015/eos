package eosruntime

import (
	"math"
	"testing"
)

// TestVectorDistillLossAndGradMSEPlusCosineCombined verifies the combined
// MSE+cosine loss on a fixed 2-d input against analytically computed values.
func TestVectorDistillLossAndGradMSEPlusCosineCombined(t *testing.T) {
	// proj = [1, 0], teacher = [0, 1]
	// MSE = (1/2)*[(1-0)^2 + (0-1)^2] = 1.0
	// cosine(proj,teacher) = dot/|proj||teacher| = 0/(1*1) = 0
	// cosine_loss = 1 - 0 = 1.0
	// total = 0.5*1.0 + 0.5*1.0 = 1.0
	proj := []float32{1, 0}
	teacher := []float32{0, 1}
	result, err := VectorDistillLossAndGrad(proj, teacher)
	if err != nil {
		t.Fatalf("VectorDistillLossAndGrad error: %v", err)
	}
	wantLoss := float32(1.0)
	if math.Abs(float64(result.Loss-wantLoss)) > 1e-5 {
		t.Errorf("loss = %f, want %f", result.Loss, wantLoss)
	}

	// Gradient w.r.t. proj[k]:
	// d_total/d_proj[k] = 0.5*(2/d)*(proj[k]-teacher[k]) - 0.5*d_cosine/d_proj[k]
	// For proj=[1,0], teacher=[0,1]:
	//   projNorm=1, teacherNorm=1, denom=1, cosine=0
	//   d_cosine/d_proj[0] = teacher[0]/denom - proj[0]*cosine/projNorm^2 = 0/1 - 1*0/1 = 0
	//   d_cosine/d_proj[1] = teacher[1]/denom - proj[1]*cosine/projNorm^2 = 1/1 - 0*0/1 = 1
	//
	// d_total/d_proj[0] = 0.5*(2/2)*(1-0) - 0.5*0 = 0.5
	// d_total/d_proj[1] = 0.5*(2/2)*(0-1) - 0.5*1 = -0.5 - 0.5 = -1.0
	wantGrad := []float32{0.5, -1.0}
	for i, g := range result.GradProj {
		if math.Abs(float64(g-wantGrad[i])) > 1e-5 {
			t.Errorf("GradProj[%d] = %f, want %f", i, g, wantGrad[i])
		}
	}
}

// TestVectorDistillLossAndGradPerfectMatchIsZero confirms that when the student
// projection exactly matches the teacher, the loss is zero.
func TestVectorDistillLossAndGradPerfectMatchIsZero(t *testing.T) {
	v := []float32{0.6, 0.8} // already unit-norm
	result, err := VectorDistillLossAndGrad(v, v)
	if err != nil {
		t.Fatalf("VectorDistillLossAndGrad error: %v", err)
	}
	if result.Loss > 1e-5 {
		t.Errorf("perfect-match loss = %f, want ~0", result.Loss)
	}
	for i, g := range result.GradProj {
		if math.Abs(float64(g)) > 1e-5 {
			t.Errorf("perfect-match GradProj[%d] = %f, want ~0", i, g)
		}
	}
}

// TestVectorDistillLossAndGradLossDecreasesWithGradientStep confirms that
// taking a gradient step reduces the loss.
func TestVectorDistillLossAndGradLossDecreasesWithGradientStep(t *testing.T) {
	proj := []float32{1, 0, 0}
	teacher := []float32{0, 1, 0}

	result1, err := VectorDistillLossAndGrad(proj, teacher)
	if err != nil {
		t.Fatalf("first loss: %v", err)
	}

	// Gradient descent step
	lr := float32(0.1)
	proj2 := make([]float32, len(proj))
	for i := range proj {
		proj2[i] = proj[i] - lr*result1.GradProj[i]
	}

	result2, err := VectorDistillLossAndGrad(proj2, teacher)
	if err != nil {
		t.Fatalf("second loss: %v", err)
	}
	if result2.Loss >= result1.Loss {
		t.Errorf("loss did not decrease: before=%f after=%f", result1.Loss, result2.Loss)
	}
}

// TestVectorDistillLossAndGradRejectsLengthMismatch verifies the dimension check.
func TestVectorDistillLossAndGradRejectsLengthMismatch(t *testing.T) {
	_, err := VectorDistillLossAndGrad([]float32{1, 0}, []float32{0, 1, 0})
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
}

// TestAccumulateVectorDistillProjectionGradsBackpropIsCorrect verifies that
// the projection back-propagation computes correct d_loss/d_student and d_loss/d_W
// against a finite-difference approximation.
func TestAccumulateVectorDistillProjectionGradsBackpropIsCorrect(t *testing.T) {
	inputDim := 3
	outDim := 4
	// Fixed W, student, teacher
	W := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 0.1, 0.2, 0.3,
	}
	student := []float32{0.6, -0.5, 0.3}
	teacher := []float32{0.2, 0.8, -0.1, 0.5}

	// Forward
	proj := make([]float32, outDim)
	for i := 0; i < inputDim; i++ {
		base := i * outDim
		for k := 0; k < outDim; k++ {
			proj[k] += W[base+k] * student[i]
		}
	}

	lossResult, err := VectorDistillLossAndGrad(proj, teacher)
	if err != nil {
		t.Fatalf("loss: %v", err)
	}

	// Analytic gradients
	gradStudent := make([]float32, inputDim)
	gradW := make([]float32, inputDim*outDim)
	accumulateVectorDistillProjectionGrads(student, lossResult.GradProj, W, gradStudent, gradW)

	// Finite-difference check for gradStudent
	eps := float32(1e-4)
	for i := 0; i < inputDim; i++ {
		studentPlus := append([]float32(nil), student...)
		studentPlus[i] += eps
		projPlus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for k := 0; k < outDim; k++ {
				projPlus[k] += W[base+k] * studentPlus[si]
			}
		}
		rPlus, _ := VectorDistillLossAndGrad(projPlus, teacher)

		studentMinus := append([]float32(nil), student...)
		studentMinus[i] -= eps
		projMinus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for k := 0; k < outDim; k++ {
				projMinus[k] += W[base+k] * studentMinus[si]
			}
		}
		rMinus, _ := VectorDistillLossAndGrad(projMinus, teacher)

		fdGrad := (rPlus.Loss - rMinus.Loss) / (2 * eps)
		if math.Abs(float64(gradStudent[i]-fdGrad)) > 1e-3 {
			t.Errorf("gradStudent[%d] analytic=%f fd=%f", i, gradStudent[i], fdGrad)
		}
	}

	// Finite-difference check for gradW[0] (first row)
	for k := 0; k < outDim; k++ {
		Wplus := append([]float32(nil), W...)
		Wplus[k] += eps
		projPlus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for kk := 0; kk < outDim; kk++ {
				projPlus[kk] += Wplus[base+kk] * student[si]
			}
		}
		rPlus, _ := VectorDistillLossAndGrad(projPlus, teacher)

		Wminus := append([]float32(nil), W...)
		Wminus[k] -= eps
		projMinus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for kk := 0; kk < outDim; kk++ {
				projMinus[kk] += Wminus[base+kk] * student[si]
			}
		}
		rMinus, _ := VectorDistillLossAndGrad(projMinus, teacher)

		fdGrad := (rPlus.Loss - rMinus.Loss) / (2 * eps)
		if math.Abs(float64(gradW[k]-fdGrad)) > 1e-3 {
			t.Errorf("gradW[0][%d] analytic=%f fd=%f", k, gradW[k], fdGrad)
		}
	}
}
