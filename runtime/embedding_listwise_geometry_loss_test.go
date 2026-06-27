package eosruntime

import (
	"math"
	"strings"
	"testing"
)

func TestEmbeddingListwiseGeometryLossAndGradDimensionsAndRowSums(t *testing.T) {
	teacher := [][]float32{{2, 0, -1}, {-1, 0.5, 2}}
	student := [][]float32{{1.8, 0.1, -0.7}, {-0.8, 0.3, 1.7}}

	got, err := EmbeddingListwiseGeometryLossAndGrad(student, teacher, 0.5)
	if err != nil {
		t.Fatalf("listwise geometry loss: %v", err)
	}
	if got.Loss <= 0 || !isFinite32(got.Loss) {
		t.Fatalf("loss = %f, want positive finite", got.Loss)
	}
	if len(got.Grad) != 2 || len(got.Grad[0]) != 3 || len(got.Grad[1]) != 3 {
		t.Fatalf("grad shape = %+v, want 2x3", got.Grad)
	}
	for i, row := range got.Grad {
		sum := float32(0)
		for _, value := range row {
			sum += value
		}
		assertClose32(t, sum, 0, 1e-6, "grad row sum")
		if len(got.TeacherProbs[i]) != 3 || len(got.StudentProbs[i]) != 3 {
			t.Fatalf("prob shape row %d = teacher %d student %d, want 3/3", i, len(got.TeacherProbs[i]), len(got.StudentProbs[i]))
		}
	}
}

func TestEmbeddingListwiseGeometryLossLowerWhenStudentMatchesTeacher(t *testing.T) {
	teacher := [][]float32{{2, 0, -1}, {-1, 0.5, 2}}
	inverted := [][]float32{{-2, 0, 1}, {1, -0.5, -2}}

	matched, err := EmbeddingListwiseGeometryLossAndGrad(teacher, teacher, 1)
	if err != nil {
		t.Fatalf("matched loss: %v", err)
	}
	bad, err := EmbeddingListwiseGeometryLossAndGrad(inverted, teacher, 1)
	if err != nil {
		t.Fatalf("inverted loss: %v", err)
	}
	if matched.Loss >= bad.Loss {
		t.Fatalf("matched loss = %f, inverted loss = %f, want matched lower", matched.Loss, bad.Loss)
	}
}

func TestEmbeddingListwiseGeometryLossValidation(t *testing.T) {
	valid := [][]float32{{1, 0}, {0, 1}}
	tests := []struct {
		name        string
		student     [][]float32
		teacher     [][]float32
		temperature float32
		want        string
	}{
		{name: "temperature", student: valid, teacher: valid, temperature: 0, want: "temperature"},
		{name: "ragged", student: [][]float32{{1, 0}, {1}}, teacher: valid, temperature: 1, want: "length"},
		{name: "shape", student: [][]float32{{1, 0}}, teacher: valid, temperature: 1, want: "shape"},
		{name: "non finite", student: [][]float32{{float32(math.NaN()), 0}}, teacher: [][]float32{{1, 0}}, temperature: 1, want: "finite"},
		{name: "empty", student: nil, teacher: nil, temperature: 1, want: "non-empty"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := EmbeddingListwiseGeometryLossAndGrad(tc.student, tc.teacher, tc.temperature)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want containing %q", err, tc.want)
			}
		})
	}
}
