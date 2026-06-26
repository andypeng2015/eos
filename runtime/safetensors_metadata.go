package eosruntime

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

const safeTensorsHeaderLengthBytes = 8

type SafeTensorsMetadata struct {
	Path         string                    `json:"path,omitempty"`
	FileSize     int64                     `json:"file_size"`
	HeaderLength uint64                    `json:"header_length"`
	Metadata     map[string]string         `json:"metadata,omitempty"`
	Tensors      map[string]SafeTensorInfo `json:"tensors"`
}

type SafeTensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type safeTensorInfoJSON struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

func ReadSafeTensorsMetadata(path string) (SafeTensorsMetadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return SafeTensorsMetadata{}, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return SafeTensorsMetadata{}, err
	}
	if stat.Size() < safeTensorsHeaderLengthBytes {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: file too small for safetensors header", path)
	}
	var headerLength uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLength); err != nil {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: read safetensors header length: %w", path, err)
	}
	if headerLength == 0 {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: safetensors header length is zero", path)
	}
	if headerLength > uint64(stat.Size()-safeTensorsHeaderLengthBytes) {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: safetensors header length %d exceeds file payload %d", path, headerLength, stat.Size()-safeTensorsHeaderLengthBytes)
	}
	header := make([]byte, headerLength)
	if _, err := io.ReadFull(file, header); err != nil {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: read safetensors header: %w", path, err)
	}
	meta, err := parseSafeTensorsHeader(path, header, stat.Size(), headerLength)
	if err != nil {
		return SafeTensorsMetadata{}, err
	}
	meta.Path = path
	meta.FileSize = stat.Size()
	meta.HeaderLength = headerLength
	return meta, nil
}

func parseSafeTensorsHeader(path string, header []byte, fileSize int64, headerLength uint64) (SafeTensorsMetadata, error) {
	if !json.Valid(header) {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: invalid safetensors JSON header", path)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: parse safetensors JSON header: %w", path, err)
	}
	tensors := make(map[string]SafeTensorInfo, len(raw))
	var metadata map[string]string
	for name, payload := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(payload, &metadata); err != nil {
				return SafeTensorsMetadata{}, fmt.Errorf("%s: parse safetensors __metadata__: %w", path, err)
			}
			continue
		}
		var rawTensor safeTensorInfoJSON
		if err := json.Unmarshal(payload, &rawTensor); err != nil {
			return SafeTensorsMetadata{}, fmt.Errorf("%s: parse safetensors tensor %q: %w", path, name, err)
		}
		tensor := SafeTensorInfo{
			DType: rawTensor.DType,
			Shape: append([]int64(nil), rawTensor.Shape...),
		}
		if len(rawTensor.DataOffsets) != 2 {
			return SafeTensorsMetadata{}, fmt.Errorf("%s: safetensors tensor %q must have two data_offsets", path, name)
		}
		tensor.DataOffsets = [2]int64{rawTensor.DataOffsets[0], rawTensor.DataOffsets[1]}
		if tensor.DType == "" {
			return SafeTensorsMetadata{}, fmt.Errorf("%s: safetensors tensor %q missing dtype", path, name)
		}
		if tensor.Shape == nil {
			return SafeTensorsMetadata{}, fmt.Errorf("%s: safetensors tensor %q missing shape", path, name)
		}
		for _, dim := range tensor.Shape {
			if dim < 0 {
				return SafeTensorsMetadata{}, fmt.Errorf("%s: safetensors tensor %q has negative shape dimension %d", path, name, dim)
			}
		}
		tensors[name] = tensor
	}
	meta := SafeTensorsMetadata{Metadata: metadata, Tensors: tensors}
	if err := validateSafeTensorOffsets(path, meta, fileSize, headerLength); err != nil {
		return SafeTensorsMetadata{}, err
	}
	return meta, nil
}

func validateSafeTensorOffsets(path string, meta SafeTensorsMetadata, fileSize int64, headerLength uint64) error {
	payloadSize := fileSize - safeTensorsHeaderLengthBytes - int64(headerLength)
	if payloadSize < 0 {
		return fmt.Errorf("%s: safetensors payload size is negative", path)
	}
	spans := make([]safeTensorSpan, 0, len(meta.Tensors))
	for name, tensor := range meta.Tensors {
		start, end := tensor.DataOffsets[0], tensor.DataOffsets[1]
		if start < 0 || end < start {
			return fmt.Errorf("%s: safetensors tensor %q has invalid data_offsets %v", path, name, tensor.DataOffsets)
		}
		if end > payloadSize {
			return fmt.Errorf("%s: safetensors tensor %q data_offsets %v exceed payload size %d", path, name, tensor.DataOffsets, payloadSize)
		}
		spans = append(spans, safeTensorSpan{name: name, start: start, end: end})
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start == spans[j].start {
			return spans[i].end < spans[j].end
		}
		return spans[i].start < spans[j].start
	})
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return fmt.Errorf("%s: safetensors tensors %q and %q have overlapping data_offsets", path, spans[i-1].name, spans[i].name)
		}
	}
	return nil
}

type safeTensorSpan struct {
	name       string
	start, end int64
}
