package eosruntime

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestReadSafeTensorsMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	payload := append(bytes.Repeat([]byte{1}, 224), bytes.Repeat([]byte{2}, 16)...)
	header := map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"embeddings.word_embeddings.weight": map[string]any{
			"dtype":        "F32",
			"shape":        []int64{7, 8},
			"data_offsets": []int64{0, 224},
		},
		"encoder.layer.0.attention.self.query.bias": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{8},
			"data_offsets": []int64{224, 240},
		},
	}
	if err := writeSafeTensorsFixture(path, header, payload); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	metadata, err := ReadSafeTensorsMetadata(path)
	if err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	if metadata.FileSize <= 0 || metadata.HeaderLength == 0 {
		t.Fatalf("missing file/header sizes: %+v", metadata)
	}
	if metadata.Metadata["format"] != "pt" {
		t.Fatalf("metadata = %+v", metadata.Metadata)
	}
	tensor := metadata.Tensors["embeddings.word_embeddings.weight"]
	if tensor.DType != "F32" || len(tensor.Shape) != 2 || tensor.Shape[0] != 7 || tensor.Shape[1] != 8 {
		t.Fatalf("tensor metadata = %+v", tensor)
	}
}

func TestReadSafeTensorsMetadataRejectsOverlappingOffsets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	header := map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
		"b": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{4, 12}},
	}
	if err := writeSafeTensorsFixture(path, header, make([]byte, 12)); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, err := ReadSafeTensorsMetadata(path)
	if err == nil || !strings.Contains(err.Error(), "overlapping") {
		t.Fatalf("expected overlap error, got %v", err)
	}
}

func TestReadSafeTensorsMetadataRejectsOutOfBoundsOffsets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	header := map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
	}
	if err := writeSafeTensorsFixture(path, header, make([]byte, 8)); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, err := ReadSafeTensorsMetadata(path)
	if err == nil || !strings.Contains(err.Error(), "exceed payload") {
		t.Fatalf("expected bounds error, got %v", err)
	}
}

func TestReadSafeTensorsCollectionFromShardedIndex(t *testing.T) {
	dir := t.TempDir()
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model-00001-of-00002.safetensors"), map[string]any{
		"embeddings.word_embeddings.weight": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
	}, []byte{1, 2, 3, 4, 5, 6, 7, 8}); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := writeSafeTensorsFixture(filepath.Join(dir, "model-00002-of-00002.safetensors"), map[string]any{
		"encoder.layer.0.attention.self.query.bias": map[string]any{"dtype": "BF16", "shape": []int64{2}, "data_offsets": []int64{0, 4}},
	}, []byte{9, 10, 11, 12}); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	index := map[string]any{
		"metadata": map[string]any{"total_size": 12},
		"weight_map": map[string]string{
			"embeddings.word_embeddings.weight":         "model-00001-of-00002.safetensors",
			"encoder.layer.0.attention.self.query.bias": "model-00002-of-00002.safetensors",
		},
	}
	if err := writeJSON(filepath.Join(dir, "model.safetensors.index.json"), index); err != nil {
		t.Fatalf("write index: %v", err)
	}
	collection, err := ReadSafeTensorsCollectionFromDir(dir)
	if err != nil {
		t.Fatalf("read sharded collection: %v", err)
	}
	if !slices.Equal(collection.Files, []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}) {
		t.Fatalf("files = %v", collection.Files)
	}
	if string(collection.Metadata["total_size"]) != "12" {
		t.Fatalf("metadata total_size = %s", collection.Metadata["total_size"])
	}
	tensor := collection.Tensors["encoder.layer.0.attention.self.query.bias"]
	if tensor.SourceFile != "model-00002-of-00002.safetensors" {
		t.Fatalf("source file = %q", tensor.SourceFile)
	}
}

func TestReadSafeTensorsCollectionRejectsMissingShard(t *testing.T) {
	dir := t.TempDir()
	index := map[string]any{
		"metadata": map[string]string{},
		"weight_map": map[string]string{
			"a": "missing.safetensors",
		},
	}
	if err := writeJSON(filepath.Join(dir, "model.safetensors.index.json"), index); err != nil {
		t.Fatalf("write index: %v", err)
	}
	_, err := ReadSafeTensorsCollectionFromDir(dir)
	if err == nil || !strings.Contains(err.Error(), "missing safetensors shard") {
		t.Fatalf("expected missing shard error, got %v", err)
	}
}

func TestReadSafeTensorsCollectionRejectsUnsafeShardPath(t *testing.T) {
	cases := []struct {
		name    string
		shard   string
		wantErr string
	}{
		{name: "empty", shard: "", wantErr: "empty shard file"},
		{name: "absolute", shard: filepath.Join(t.TempDir(), "outside.safetensors"), wantErr: "absolute shard path"},
		{name: "escaping", shard: "../outside.safetensors", wantErr: "escaping shard path"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			index := map[string]any{
				"metadata": map[string]any{"total_size": 4},
				"weight_map": map[string]string{
					"a": tc.shard,
				},
			}
			if err := writeJSON(filepath.Join(dir, "model.safetensors.index.json"), index); err != nil {
				t.Fatalf("write index: %v", err)
			}
			_, err := ReadSafeTensorsCollectionFromDir(dir)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected %q error, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestReadSafeTensorsIndexRejectsDuplicateWeightMapKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors.index.json")
	data := []byte(`{"metadata":{},"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}`)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	_, err := ReadSafeTensorsIndex(path)
	if err == nil || !strings.Contains(err.Error(), "duplicate tensor names") {
		t.Fatalf("expected duplicate index tensor error, got %v", err)
	}
}

func TestReadSafeTensorsCollectionRejectsDuplicateTensorAcrossShards(t *testing.T) {
	dir := t.TempDir()
	headerOne := map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 4}},
	}
	headerTwo := map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 4}},
		"b": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{4, 8}},
	}
	if err := writeSafeTensorsFixture(filepath.Join(dir, "one.safetensors"), headerOne, []byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("write shard one: %v", err)
	}
	if err := writeSafeTensorsFixture(filepath.Join(dir, "two.safetensors"), headerTwo, []byte{5, 6, 7, 8, 9, 10, 11, 12}); err != nil {
		t.Fatalf("write shard two: %v", err)
	}
	index := map[string]any{
		"metadata": map[string]string{},
		"weight_map": map[string]string{
			"a": "one.safetensors",
			"b": "two.safetensors",
		},
	}
	if err := writeJSON(filepath.Join(dir, "model.safetensors.index.json"), index); err != nil {
		t.Fatalf("write index: %v", err)
	}
	_, err := ReadSafeTensorsCollectionFromDir(dir)
	if err == nil || !strings.Contains(err.Error(), "duplicate tensor") {
		t.Fatalf("expected duplicate shard tensor error, got %v", err)
	}
}

func TestReadSafeTensorsMetadataRejectsDuplicateTensorHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dup.safetensors")
	header := []byte(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatalf("write length: %v", err)
	}
	buf.Write(header)
	buf.Write([]byte{1, 2, 3, 4})
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, err := ReadSafeTensorsMetadata(path)
	if err == nil || !strings.Contains(err.Error(), "duplicate tensor names") {
		t.Fatalf("expected duplicate tensor error, got %v", err)
	}
}

func TestReadSafeTensorsMetadataRejectsByteCountMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	header := map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 4}},
	}
	if err := writeSafeTensorsFixture(path, header, []byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, err := ReadSafeTensorsMetadata(path)
	if err == nil || !strings.Contains(err.Error(), "does not match shape elements") {
		t.Fatalf("expected byte-count mismatch error, got %v", err)
	}
}

func TestLoadSafeTensorDataRejectsUnsupportedDType(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	header := map[string]any{
		"a": map[string]any{"dtype": "QUANT4", "shape": []int64{4}, "data_offsets": []int64{0, 2}},
	}
	if err := writeSafeTensorsFixture(path, header, []byte{1, 2}); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, _, err := LoadSafeTensorDataFromDir(dir, []string{"a"})
	if err == nil || !strings.Contains(err.Error(), "unsupported dtype") {
		t.Fatalf("expected unsupported dtype error, got %v", err)
	}
}

func TestLoadSafeTensorDataCopiesExactBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	want := []byte{9, 8, 7, 6}
	header := map[string]any{
		"a": map[string]any{"dtype": "BF16", "shape": []int64{2}, "data_offsets": []int64{0, 4}},
	}
	if err := writeSafeTensorsFixture(path, header, want); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	data, _, err := LoadSafeTensorDataFromDir(dir, []string{"a"})
	if err != nil {
		t.Fatalf("load tensor bytes: %v", err)
	}
	if !bytes.Equal(data["a"].Bytes, want) {
		t.Fatalf("bytes = %v, want %v", data["a"].Bytes, want)
	}
}

func writeSafeTensorsFixture(path string, header map[string]any, payload []byte) error {
	data, err := json.Marshal(header)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	buf.Write(data)
	buf.Write(payload)
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func writeJSON(path string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
