package eosruntime

import (
	"fmt"
	"math"
)

type EmbeddingListwiseGeometryLossResult struct {
	Loss         float32
	Grad         [][]float32
	TeacherProbs [][]float32
	StudentProbs [][]float32
}

// EmbeddingListwiseGeometryLossAndGrad computes row-wise teacher-to-student cross-entropy.
func EmbeddingListwiseGeometryLossAndGrad(studentSimilarity, teacherSimilarity [][]float32, temperature float32) (EmbeddingListwiseGeometryLossResult, error) {
	var result EmbeddingListwiseGeometryLossResult
	if temperature <= 0 || !isFinite32(temperature) {
		return result, fmt.Errorf("temperature must be finite and positive")
	}
	rows, cols, err := validateEmbeddingListwiseGeometryLossMatrix(studentSimilarity, "student_similarity")
	if err != nil {
		return result, err
	}
	if rows == 0 || cols == 0 {
		return result, fmt.Errorf("similarity matrices must be non-empty")
	}
	teacherRows, teacherCols, err := validateEmbeddingListwiseGeometryLossMatrix(teacherSimilarity, "teacher_similarity")
	if err != nil {
		return result, err
	}
	if teacherRows != rows || teacherCols != cols {
		return result, fmt.Errorf("teacher shape %dx%d does not match student shape %dx%d", teacherRows, teacherCols, rows, cols)
	}

	result.Grad = make([][]float32, rows)
	result.TeacherProbs = make([][]float32, rows)
	result.StudentProbs = make([][]float32, rows)
	rowScale := float32(1) / float32(rows)
	for i := 0; i < rows; i++ {
		teacherProbs := make([]float32, cols)
		studentProbs := make([]float32, cols)
		softmaxScoresInto(teacherSimilarity[i], temperature, teacherProbs)
		softmaxScoresInto(studentSimilarity[i], temperature, studentProbs)
		grad := make([]float32, cols)
		for j := 0; j < cols; j++ {
			prob := studentProbs[j]
			if prob < 1e-12 {
				prob = 1e-12
			}
			result.Loss -= rowScale * teacherProbs[j] * float32(math.Log(float64(prob)))
			grad[j] = rowScale * (studentProbs[j] - teacherProbs[j]) / temperature
		}
		result.Grad[i] = grad
		result.TeacherProbs[i] = teacherProbs
		result.StudentProbs[i] = studentProbs
	}
	return result, nil
}

func validateEmbeddingListwiseGeometryLossMatrix(matrix [][]float32, name string) (int, int, error) {
	if len(matrix) == 0 {
		return 0, 0, nil
	}
	cols := len(matrix[0])
	if cols == 0 {
		return 0, 0, fmt.Errorf("%s[0] is empty", name)
	}
	for i, row := range matrix {
		if len(row) != cols {
			return 0, 0, fmt.Errorf("%s[%d] length %d does not match row width %d", name, i, len(row), cols)
		}
		for j, value := range row {
			if !isFinite32(value) {
				return 0, 0, fmt.Errorf("%s[%d][%d] must be finite", name, i, j)
			}
		}
	}
	return len(matrix), cols, nil
}
