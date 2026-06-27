package eosruntime

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPretrainedBERTProjectionHeadRoundTripApplyAndNormalize(t *testing.T) {
	head, err := NewPretrainedBERTProjectionHead(3, 2, []float32{
		1, 0,
		0, 2,
		1, 0,
	})
	if err != nil {
		t.Fatalf("new head: %v", err)
	}
	head.SourceModel = "fixture-bge"
	head.Initialization = "unit-test"
	head.Loss = "listwise_kl_softmax_dot"
	head.DataProvenance = "unit-test qrels=train"
	path := filepath.Join(t.TempDir(), "head.mll")
	if err := WritePretrainedBERTProjectionHeadFile(path, head); err != nil {
		t.Fatalf("write head: %v", err)
	}
	loaded, err := ReadPretrainedBERTProjectionHeadFile(path)
	if err != nil {
		t.Fatalf("read head: %v", err)
	}
	if loaded.Schema != PretrainedBERTProjectionHeadSchema || loaded.InputDim != 3 || loaded.OutputDim != 2 || loaded.SourceModel != "fixture-bge" {
		t.Fatalf("loaded head = %+v", loaded)
	}
	got, err := loaded.Apply([]float32{1, 1, 1})
	if err != nil {
		t.Fatalf("apply head: %v", err)
	}
	want := []float32{float32(1 / math.Sqrt2), float32(1 / math.Sqrt2)}
	if math.Abs(float64(got[0]-want[0])) > 1e-6 || math.Abs(float64(got[1]-want[1])) > 1e-6 {
		t.Fatalf("projection = %v, want %v", got, want)
	}
	norm := math.Hypot(float64(got[0]), float64(got[1]))
	if math.Abs(norm-1) > 1e-6 {
		t.Fatalf("norm = %.9f, want 1", norm)
	}
}

func TestPretrainedBERTProjectionHeadRejectsShapeMismatch(t *testing.T) {
	_, err := NewPretrainedBERTProjectionHead(3, 2, []float32{1, 2, 3})
	if err == nil || !strings.Contains(err.Error(), "weights length") {
		t.Fatalf("err = %v, want weights length error", err)
	}
	head, err := NewPretrainedBERTProjectionHead(3, 2, []float32{1, 0, 0, 1, 1, 1})
	if err != nil {
		t.Fatalf("new head: %v", err)
	}
	_, err = head.Apply([]float32{1, 2})
	if err == nil || !strings.Contains(err.Error(), "input dimension mismatch") {
		t.Fatalf("err = %v, want input dimension mismatch", err)
	}

	bad := head
	bad.InputDim = 4
	data, err := encodePretrainedBERTProjectionHeadMLL(bad)
	if err != nil {
		t.Fatalf("encode malformed head: %v", err)
	}
	path := filepath.Join(t.TempDir(), "bad-shape.mll")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write malformed head: %v", err)
	}
	_, err = ReadPretrainedBERTProjectionHeadFile(path)
	if err == nil || !strings.Contains(err.Error(), "weights length") {
		t.Fatalf("err = %v, want weights length error", err)
	}
}

func TestPretrainedBERTProjectionHeadReadRejectsStaleWeightsHash(t *testing.T) {
	head, err := NewPretrainedBERTProjectionHead(2, 1, []float32{1, 0})
	if err != nil {
		t.Fatalf("new head: %v", err)
	}
	head.WeightsSHA256 = strings.Repeat("0", 64)
	head.ContentSHA256 = ""
	data, err := encodePretrainedBERTProjectionHeadMLL(head)
	if err != nil {
		t.Fatalf("encode stale-hash head: %v", err)
	}
	path := filepath.Join(t.TempDir(), "stale-hash.mll")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write stale-hash head: %v", err)
	}
	_, err = ReadPretrainedBERTProjectionHeadFile(path)
	if err == nil || !strings.Contains(err.Error(), "weights_sha256") {
		t.Fatalf("err = %v, want weights_sha256 mismatch", err)
	}
}

func TestPretrainedBERTProjectionHeadIdentityIsDeterministic(t *testing.T) {
	head, err := NewPretrainedBERTProjectionHead(2, 1, []float32{2, 0})
	if err != nil {
		t.Fatalf("new head: %v", err)
	}
	first := head.IdentityHash()
	second := head.IdentityHash()
	if first == "" || first != second {
		t.Fatalf("identity hashes = %q %q", first, second)
	}
	head.Weights[0] = 3
	if first == head.IdentityHash() {
		t.Fatalf("identity did not change after weight mutation")
	}
	head.Weights[0] = 2
	head.SourceModel = "different path provenance"
	head.Loss = "different loss provenance"
	head.DataProvenance = "different data provenance"
	if first != head.IdentityHash() {
		t.Fatalf("identity changed after non-behavior provenance update: before=%q after=%q", first, head.IdentityHash())
	}
}
