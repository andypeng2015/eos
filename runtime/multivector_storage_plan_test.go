package eosruntime

import "testing"

func TestPlanMultiVectorStorageUsesTurboQuantIPPayloadBytes(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		Bits:             []int{2, 4, 8},
		Objects:          1000,
		VectorsPerObject: []int{1, 16},
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.Schema != MultiVectorStoragePlanSchema || len(plan.Rows) != 6 {
		t.Fatalf("unexpected plan identity: schema=%q rows=%d", plan.Schema, len(plan.Rows))
	}
	row := plan.Rows[0]
	if row.Bits != 2 || row.VectorsPerObject != 1 {
		t.Fatalf("unexpected first row: %+v", row)
	}
	if row.DenseParentBytes != 512 || row.DenseParentTotalBytes != 512000 {
		t.Fatalf("dense bytes = parent:%d total:%d", row.DenseParentBytes, row.DenseParentTotalBytes)
	}
	if row.QuantizedPayloadBytes != 36 || row.QuantizedVectorBytes != 36 {
		t.Fatalf("q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.TotalQuantizedBytes != 36000 {
		t.Fatalf("total quantized bytes = %d", row.TotalQuantizedBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 14 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if got, want := plan.Rows[1].FitsInOneDenseVectorStorage, false; got != want {
		t.Fatalf("16 q2 child vectors should exceed one dense parent budget")
	}
}

func TestPlanMultiVectorStorageAccountsForFP16Sidecar(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		Bits:             []int{4},
		Objects:          1,
		VectorsPerObject: []int{1, 2},
		SidecarStorage:   MultiVectorSidecarFP16,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	row := plan.Rows[0]
	if row.SidecarStorage != MultiVectorSidecarFP16 || row.SidecarBytesPerVector != 256 {
		t.Fatalf("sidecar = storage:%q bytes:%d", row.SidecarStorage, row.SidecarBytesPerVector)
	}
	if row.QuantizedPayloadBytes != 68 || row.QuantizedVectorBytes != 324 {
		t.Fatalf("q4 fp16 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 1 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("one q4 fp16 row fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if plan.Rows[1].FitsInOneDenseVectorStorage {
		t.Fatalf("two q4 fp16 sidecar vectors should exceed one dense parent budget")
	}
}
