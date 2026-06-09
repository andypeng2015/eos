package eosruntime

import (
	"testing"

	"m31labs.dev/eos/runtime/backend"
	mll "m31labs.dev/mll"
)

func TestPackQuantizedTensorStorageQ8MatchesFakeQuantGrid(t *testing.T) {
	values := []float32{0.013, -0.92, 0.5, 0.0001, -0.31, 1.7, -1.7, 0.0}
	tensor := backend.NewTensorQ8([]int{2, 4}, append([]float32(nil), values...))

	storage, raw, scale, err := packQuantizedTensorStorage(tensor)
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	if storage != mll.DTypeQ8 {
		t.Fatalf("storage = %v, want DTypeQ8", storage)
	}
	if len(raw) != len(values) {
		t.Fatalf("payload bytes = %d, want %d", len(raw), len(values))
	}
	if scale <= 0 {
		t.Fatalf("scale = %v", scale)
	}

	decoded, err := decodeTensorEntry(mll.TensorEntry{
		DType: mll.DTypeQ8,
		Shape: []uint64{2, 4},
		Data:  raw,
	}, "q8", scale)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.DType != "q8" {
		t.Fatalf("decoded dtype = %q", decoded.DType)
	}

	want := append([]float32(nil), values...)
	fakeQuantizeDataInPlace(want, 8)
	for i := range want {
		if decoded.F32[i] != want[i] {
			t.Fatalf("value[%d] = %v, want fake-quant grid value %v", i, decoded.F32[i], want[i])
		}
	}
}

func TestPackQuantizedTensorStorageQ4RoundTrip(t *testing.T) {
	values := []float32{0.7, -0.7, 0.1, -0.05, 0.35, 0.0, -0.21} // odd length exercises nibble padding
	tensor := backend.NewTensorQ4([]int{7}, append([]float32(nil), values...))

	storage, raw, scale, err := packQuantizedTensorStorage(tensor)
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	if storage != mll.DTypeQ4 {
		t.Fatalf("storage = %v, want DTypeQ4", storage)
	}
	if len(raw) != 4 {
		t.Fatalf("payload bytes = %d, want 4", len(raw))
	}

	decoded, err := decodeTensorEntry(mll.TensorEntry{
		DType: mll.DTypeQ4,
		Shape: []uint64{7},
		Data:  raw,
	}, "q4", scale)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	want := append([]float32(nil), values...)
	fakeQuantizeDataInPlace(want, 4)
	for i := range want {
		if decoded.F32[i] != want[i] {
			t.Fatalf("value[%d] = %v, want fake-quant grid value %v", i, decoded.F32[i], want[i])
		}
	}
}

func TestPackQuantizedTensorStorageZeroTensor(t *testing.T) {
	tensor := backend.NewTensorQ8([]int{3}, []float32{0, 0, 0})
	storage, raw, scale, err := packQuantizedTensorStorage(tensor)
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	if scale != 0 {
		t.Fatalf("scale = %v, want 0 for all-zero tensor", scale)
	}
	decoded, err := decodeTensorEntry(mll.TensorEntry{DType: storage, Shape: []uint64{3}, Data: raw}, "q8", scale)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i, value := range decoded.F32 {
		if value != 0 {
			t.Fatalf("value[%d] = %v, want 0", i, value)
		}
	}
}

func TestDecodePackedTensorWithoutScaleFailsLoudly(t *testing.T) {
	if _, err := decodeTensorEntry(mll.TensorEntry{
		DType: mll.DTypeQ8,
		Shape: []uint64{2},
		Data:  []byte{5, 0},
	}, "q8", 0); err == nil {
		t.Fatal("expected error for non-zero packed payload without scale")
	}
}

func TestDecodeQuantizedDataInPlaceIdempotent(t *testing.T) {
	// Values already on the fake-quant grid must re-quantize to themselves:
	// the QAT forward applies fake-quant again on load, so a packed seal must
	// be a fixed point of the quantizer.
	values := []float32{0.4, -1.2, 0.9, 0.0, 1.2}
	fakeQuantizeDataInPlace(values, 8)
	again := append([]float32(nil), values...)
	fakeQuantizeDataInPlace(again, 8)
	for i := range values {
		if values[i] != again[i] {
			t.Fatalf("value[%d] not idempotent: %v vs %v", i, values[i], again[i])
		}
	}
}
