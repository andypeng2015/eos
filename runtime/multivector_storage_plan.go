package eosruntime

import (
	"fmt"

	"m31labs.dev/turboquant"
)

const MultiVectorStoragePlanSchema = "manta.multivector_storage_plan.v1"

const (
	MultiVectorSidecarNone  = "none"
	MultiVectorSidecarFP16  = "fp16"
	MultiVectorSidecarDense = "dense"
)

type MultiVectorStoragePlanInput struct {
	Dim              int
	Bits             []int
	Objects          int
	VectorsPerObject []int
	SidecarStorage   string
}

type MultiVectorStoragePlan struct {
	Schema string                       `json:"schema"`
	Config MultiVectorStoragePlanConfig `json:"config"`
	Rows   []MultiVectorStoragePlanRow  `json:"rows"`
}

type MultiVectorStoragePlanConfig struct {
	Dim              int    `json:"dim"`
	Bits             []int  `json:"bits"`
	Objects          int    `json:"objects"`
	VectorsPerObject []int  `json:"vectors_per_object"`
	SidecarStorage   string `json:"sidecar_storage"`
}

type MultiVectorStoragePlanRow struct {
	Dim                              int     `json:"dim"`
	Bits                             int     `json:"bits"`
	Objects                          int     `json:"objects"`
	VectorsPerObject                 int     `json:"vectors_per_object"`
	DenseParentBytes                 int64   `json:"dense_parent_bytes"`
	DenseParentTotalBytes            int64   `json:"dense_parent_total_bytes"`
	QuantizedPayloadBytes            int64   `json:"quantized_payload_bytes"`
	SidecarStorage                   string  `json:"sidecar_storage"`
	SidecarBytesPerVector            int64   `json:"sidecar_bytes_per_vector"`
	QuantizedVectorBytes             int64   `json:"quantized_vector_bytes"`
	TotalQuantizedBytes              int64   `json:"total_quantized_bytes"`
	DenseToQuantizedVectorRatio      float64 `json:"dense_to_quantized_vector_ratio"`
	TotalCompressionRatio            float64 `json:"total_compression_ratio"`
	VectorsThatFitInOneDenseVector   int64   `json:"vectors_that_fit_in_one_dense_vector"`
	FitsInOneDenseVectorStorage      bool    `json:"fits_in_one_dense_vector_storage"`
	StorageMultipleOfDenseParentCost float64 `json:"storage_multiple_of_dense_parent_cost"`
}

func PlanMultiVectorStorage(in MultiVectorStoragePlanInput) (MultiVectorStoragePlan, error) {
	if in.Dim <= 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("dim must be positive")
	}
	if in.Objects <= 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("objects must be positive")
	}
	bits := normalizeTurboQuantRetrievalBits(in.Bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return MultiVectorStoragePlan{}, err
	}
	vectorsPerObject := normalizeMultiVectorStorageCounts(in.VectorsPerObject)
	if len(vectorsPerObject) == 0 {
		if len(in.VectorsPerObject) > 0 {
			return MultiVectorStoragePlan{}, fmt.Errorf("vectors-per-object must contain at least one positive integer")
		}
		vectorsPerObject = []int{1}
	}
	sidecarStorage, err := normalizeMultiVectorSidecarStorage(in.SidecarStorage)
	if err != nil {
		return MultiVectorStoragePlan{}, err
	}

	plan := MultiVectorStoragePlan{
		Schema: MultiVectorStoragePlanSchema,
		Config: MultiVectorStoragePlanConfig{
			Dim:              in.Dim,
			Bits:             append([]int(nil), bits...),
			Objects:          in.Objects,
			VectorsPerObject: append([]int(nil), vectorsPerObject...),
			SidecarStorage:   sidecarStorage,
		},
	}
	denseParentBytes := int64(in.Dim) * 4
	denseParentTotalBytes := denseParentBytes * int64(in.Objects)
	sidecarBytesPerVector := multiVectorSidecarBytes(in.Dim, sidecarStorage)
	for _, bitWidth := range bits {
		mseBytes, signBytes := turboquant.IPQuantizedSizes(in.Dim, bitWidth)
		quantizedPayloadBytes := int64(mseBytes + signBytes + 4)
		quantizedVectorBytes := quantizedPayloadBytes + sidecarBytesPerVector
		for _, count := range vectorsPerObject {
			totalQuantizedBytes := int64(in.Objects) * int64(count) * quantizedVectorBytes
			row := MultiVectorStoragePlanRow{
				Dim:                              in.Dim,
				Bits:                             bitWidth,
				Objects:                          in.Objects,
				VectorsPerObject:                 count,
				DenseParentBytes:                 denseParentBytes,
				DenseParentTotalBytes:            denseParentTotalBytes,
				QuantizedPayloadBytes:            quantizedPayloadBytes,
				SidecarStorage:                   sidecarStorage,
				SidecarBytesPerVector:            sidecarBytesPerVector,
				QuantizedVectorBytes:             quantizedVectorBytes,
				TotalQuantizedBytes:              totalQuantizedBytes,
				DenseToQuantizedVectorRatio:      ratioFloat64(float64(denseParentBytes), float64(quantizedVectorBytes)),
				TotalCompressionRatio:            ratioFloat64(float64(denseParentTotalBytes), float64(totalQuantizedBytes)),
				VectorsThatFitInOneDenseVector:   denseParentBytes / quantizedVectorBytes,
				FitsInOneDenseVectorStorage:      int64(count)*quantizedVectorBytes <= denseParentBytes,
				StorageMultipleOfDenseParentCost: ratioFloat64(float64(int64(count)*quantizedVectorBytes), float64(denseParentBytes)),
			}
			plan.Rows = append(plan.Rows, row)
		}
	}
	return plan, nil
}

func normalizeMultiVectorStorageCounts(values []int) []int {
	seen := map[int]bool{}
	out := make([]int, 0, len(values))
	for _, value := range values {
		if value <= 0 || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func normalizeMultiVectorSidecarStorage(storage string) (string, error) {
	switch storage {
	case "", MultiVectorSidecarNone, "direct", "quantized", "quantized-only", TurboQuantRerankStorageCompactReconstruct, "compact", "reconstruct":
		return MultiVectorSidecarNone, nil
	case MultiVectorSidecarFP16, "f16", "half", "dense-f16":
		return MultiVectorSidecarFP16, nil
	case MultiVectorSidecarDense, "dense-f32", "fp32":
		return MultiVectorSidecarDense, nil
	default:
		return "", fmt.Errorf("unsupported sidecar storage %q; use %q, %q, or %q", storage, MultiVectorSidecarNone, MultiVectorSidecarFP16, MultiVectorSidecarDense)
	}
}

func multiVectorSidecarBytes(dim int, storage string) int64 {
	switch storage {
	case MultiVectorSidecarFP16:
		return int64(dim) * 2
	case MultiVectorSidecarDense:
		return int64(dim) * 4
	default:
		return 0
	}
}
