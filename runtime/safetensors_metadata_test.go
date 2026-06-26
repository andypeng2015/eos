package eosruntime

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadSafeTensorsMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
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
	if err := writeSafeTensorsFixture(path, header, make([]byte, 240)); err != nil {
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
		"a": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 8}},
		"b": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{4, 12}},
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
		"a": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 16}},
	}
	if err := writeSafeTensorsFixture(path, header, make([]byte, 8)); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, err := ReadSafeTensorsMetadata(path)
	if err == nil || !strings.Contains(err.Error(), "exceed payload") {
		t.Fatalf("expected bounds error, got %v", err)
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
