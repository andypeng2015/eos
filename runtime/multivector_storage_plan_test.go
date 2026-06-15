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
	if plan.Config.BaselineDim != 128 || row.BaselineDim != 128 {
		t.Fatalf("baseline dim = config:%d row:%d", plan.Config.BaselineDim, row.BaselineDim)
	}
	if row.DenseParentBytes != 512 || row.DenseParentTotalBytes != 512000 {
		t.Fatalf("dense bytes = parent:%d total:%d", row.DenseParentBytes, row.DenseParentTotalBytes)
	}
	if row.DenseBaselineBytes != row.DenseParentBytes || row.DenseBaselineTotalBytes != row.DenseParentTotalBytes {
		t.Fatalf("dense baseline aliases = bytes:%d total:%d", row.DenseBaselineBytes, row.DenseBaselineTotalBytes)
	}
	if row.QuantizedPayloadBytes != 36 || row.QuantizedVectorBytes != 36 {
		t.Fatalf("q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorOverheadBytes != 0 || row.DenseVectorStorageBytes != 512 || row.QuantizedVectorStorageBytes != 36 {
		t.Fatalf("storage bytes = overhead:%d dense:%d quantized:%d", row.VectorOverheadBytes, row.DenseVectorStorageBytes, row.QuantizedVectorStorageBytes)
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

func TestPlanMultiVectorStorageUsesLargerBaselineDimForDenseBudget(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		BaselineDim:      3072,
		Bits:             []int{2},
		Objects:          1,
		VectorsPerObject: []int{128},
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	row := plan.Rows[0]
	if plan.Config.BaselineDim != 3072 || row.BaselineDim != 3072 {
		t.Fatalf("baseline dim = config:%d row:%d", plan.Config.BaselineDim, row.BaselineDim)
	}
	if row.DenseParentBytes != 12288 || row.DenseBaselineBytes != 12288 {
		t.Fatalf("dense baseline bytes = parent:%d baseline:%d", row.DenseParentBytes, row.DenseBaselineBytes)
	}
	if row.QuantizedPayloadBytes != 36 || row.QuantizedVectorBytes != 36 {
		t.Fatalf("q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 341 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if row.StorageMultipleOfDenseParentCost < 0.3749 || row.StorageMultipleOfDenseParentCost > 0.3751 {
		t.Fatalf("storage multiple = %.6f", row.StorageMultipleOfDenseParentCost)
	}
}

func TestPlanMultiVectorStorageAccountsForPerVectorOverhead(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                 128,
		BaselineDim:         3072,
		Bits:                []int{2},
		Objects:             1000,
		VectorsPerObject:    []int{64, 128, 256},
		VectorOverheadBytes: 32,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.Config.VectorOverheadBytes != 32 {
		t.Fatalf("config overhead = %d", plan.Config.VectorOverheadBytes)
	}
	row := plan.Rows[0]
	if row.QuantizedPayloadBytes != 36 || row.QuantizedVectorBytes != 36 {
		t.Fatalf("raw q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorOverheadBytes != 32 || row.DenseVectorStorageBytes != 12320 || row.QuantizedVectorStorageBytes != 68 {
		t.Fatalf("storage bytes = overhead:%d dense:%d quantized:%d", row.VectorOverheadBytes, row.DenseVectorStorageBytes, row.QuantizedVectorStorageBytes)
	}
	if row.DenseBaselineTotalBytes != 12320000 || row.TotalQuantizedBytes != 4352000 {
		t.Fatalf("totals = dense:%d quantized:%d", row.DenseBaselineTotalBytes, row.TotalQuantizedBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 181 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if !plan.Rows[1].FitsInOneDenseVectorStorage {
		t.Fatalf("128 q2 child vectors with 32-byte overhead should fit in one dense baseline storage budget")
	}
	if plan.Rows[2].StorageMultipleOfDenseParentCost < 1.4129 || plan.Rows[2].StorageMultipleOfDenseParentCost > 1.4130 {
		t.Fatalf("storage multiple = %.6f", plan.Rows[2].StorageMultipleOfDenseParentCost)
	}
}

func TestPlanMultiVectorStorageRejectsNegativePerVectorOverhead(t *testing.T) {
	_, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                 128,
		Bits:                []int{2},
		Objects:             1,
		VectorOverheadBytes: -1,
	})
	if err == nil {
		t.Fatal("plan succeeded with negative vector overhead")
	}
}
