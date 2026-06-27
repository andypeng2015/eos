package eosruntime

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
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
	SourceFile  string   `json:"source_file,omitempty"`
}

type SafeTensorsIndex struct {
	Path      string                     `json:"path,omitempty"`
	Metadata  map[string]json.RawMessage `json:"metadata,omitempty"`
	WeightMap map[string]string          `json:"weight_map"`
}

type SafeTensorsShardMetadata struct {
	File     string              `json:"file"`
	Metadata SafeTensorsMetadata `json:"metadata"`
}

type SafeTensorsCollection struct {
	Path     string                         `json:"path,omitempty"`
	Metadata map[string]json.RawMessage     `json:"metadata,omitempty"`
	Files    []string                       `json:"files"`
	Tensors  map[string]SafeTensorInfo      `json:"tensors"`
	Shards   map[string]SafeTensorsMetadata `json:"-"`
}

type SafeTensorData struct {
	Name       string  `json:"name"`
	DType      string  `json:"dtype"`
	Shape      []int64 `json:"shape"`
	SourceFile string  `json:"source_file"`
	ByteOffset int64   `json:"byte_offset"`
	ByteLength int64   `json:"byte_length"`
	Bytes      []byte  `json:"-"`
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
	for name, tensor := range meta.Tensors {
		tensor.SourceFile = filepath.Base(path)
		meta.Tensors[name] = tensor
	}
	meta.FileSize = stat.Size()
	meta.HeaderLength = headerLength
	return meta, nil
}

func ReadSafeTensorsIndex(path string) (SafeTensorsIndex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SafeTensorsIndex{}, err
	}
	if dups, err := duplicateObjectKeys(data, "weight_map"); err != nil {
		return SafeTensorsIndex{}, fmt.Errorf("%s: parse safetensors index: %w", path, err)
	} else if len(dups) > 0 {
		return SafeTensorsIndex{}, fmt.Errorf("%s: duplicate tensor names in safetensors index weight_map: %s", path, strings.Join(dups, ", "))
	}
	var index SafeTensorsIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return SafeTensorsIndex{}, fmt.Errorf("%s: parse safetensors index: %w", path, err)
	}
	if index.WeightMap == nil {
		return SafeTensorsIndex{}, fmt.Errorf("%s: safetensors index missing weight_map", path)
	}
	if index.Metadata == nil {
		index.Metadata = map[string]json.RawMessage{}
	}
	for key, value := range index.Metadata {
		if !safeTensorsIndexMetadataValueIsScalar(value) {
			return SafeTensorsIndex{}, fmt.Errorf("%s: safetensors index metadata %q must be a JSON scalar", path, key)
		}
	}
	index.Path = path
	return index, nil
}

func ReadSafeTensorsCollectionFromDir(dir string) (SafeTensorsCollection, error) {
	if matches, err := filepath.Glob(filepath.Join(dir, "*.safetensors.index.json")); err != nil {
		return SafeTensorsCollection{}, err
	} else if len(matches) > 0 {
		sort.Strings(matches)
		return ReadSafeTensorsCollectionFromIndex(matches[0])
	}
	path := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return SafeTensorsCollection{}, fmt.Errorf("model.safetensors not found in %s", dir)
		}
		return SafeTensorsCollection{}, err
	}
	metadata, err := ReadSafeTensorsMetadata(path)
	if err != nil {
		return SafeTensorsCollection{}, err
	}
	return SafeTensorsCollection{
		Path:     path,
		Metadata: cloneSafeTensorsStringMap(metadata.Metadata),
		Files:    []string{filepath.Base(path)},
		Tensors:  cloneSafeTensorInfoMap(metadata.Tensors),
		Shards:   map[string]SafeTensorsMetadata{filepath.Base(path): metadata},
	}, nil
}

func ReadSafeTensorsCollectionFromIndex(indexPath string) (SafeTensorsCollection, error) {
	index, err := ReadSafeTensorsIndex(indexPath)
	if err != nil {
		return SafeTensorsCollection{}, err
	}
	dir := filepath.Dir(indexPath)
	shardSet := make(map[string]struct{})
	for name, shard := range index.WeightMap {
		if name == "" {
			return SafeTensorsCollection{}, fmt.Errorf("%s: safetensors index has empty tensor name", indexPath)
		}
		if err := validateSafeTensorsShardPath(indexPath, name, shard); err != nil {
			return SafeTensorsCollection{}, err
		}
		shardSet[shard] = struct{}{}
	}
	files := make([]string, 0, len(shardSet))
	for file := range shardSet {
		files = append(files, file)
	}
	sort.Strings(files)
	collection := SafeTensorsCollection{
		Path:     indexPath,
		Metadata: cloneSafeTensorsIndexMetadata(index.Metadata),
		Files:    append([]string(nil), files...),
		Tensors:  make(map[string]SafeTensorInfo, len(index.WeightMap)),
		Shards:   make(map[string]SafeTensorsMetadata, len(files)),
	}
	headerOwners := make(map[string]string, len(index.WeightMap))
	for _, file := range files {
		shardPath := filepath.Join(dir, file)
		if _, err := os.Stat(shardPath); err != nil {
			if os.IsNotExist(err) {
				return SafeTensorsCollection{}, fmt.Errorf("%s: missing safetensors shard %s", indexPath, file)
			}
			return SafeTensorsCollection{}, err
		}
		meta, err := ReadSafeTensorsMetadata(shardPath)
		if err != nil {
			return SafeTensorsCollection{}, err
		}
		for key, value := range meta.Metadata {
			rawValue, err := json.Marshal(value)
			if err != nil {
				return SafeTensorsCollection{}, err
			}
			if existing, ok := collection.Metadata[key]; ok && string(existing) != string(rawValue) {
				return SafeTensorsCollection{}, fmt.Errorf("%s: conflicting safetensors metadata %q: index/shard values %s vs %s", indexPath, key, existing, rawValue)
			}
			collection.Metadata[key] = rawValue
		}
		for name, tensor := range meta.Tensors {
			if previous, ok := headerOwners[name]; ok {
				return SafeTensorsCollection{}, fmt.Errorf("%s: duplicate tensor %q in shards %s and %s", indexPath, name, previous, file)
			}
			headerOwners[name] = file
			mappedFile, ok := index.WeightMap[name]
			if !ok {
				return SafeTensorsCollection{}, fmt.Errorf("%s: tensor %q present in shard %s but absent from index", indexPath, name, file)
			}
			if mappedFile != file {
				return SafeTensorsCollection{}, fmt.Errorf("%s: tensor %q index maps to %s but shard header is %s", indexPath, name, mappedFile, file)
			}
			tensor.SourceFile = file
			collection.Tensors[name] = tensor
		}
		collection.Shards[file] = meta
	}
	for name, file := range index.WeightMap {
		if _, ok := collection.Tensors[name]; !ok {
			return SafeTensorsCollection{}, fmt.Errorf("%s: tensor %q mapped to %s but absent from shard metadata", indexPath, name, file)
		}
	}
	return collection, nil
}

func LoadSafeTensorDataFromDir(dir string, names []string) (map[string]SafeTensorData, SafeTensorsCollection, error) {
	collection, err := ReadSafeTensorsCollectionFromDir(dir)
	if err != nil {
		return nil, SafeTensorsCollection{}, err
	}
	data, err := LoadSafeTensorData(collection, filepath.Dir(collection.Path), names)
	if err != nil {
		return nil, SafeTensorsCollection{}, err
	}
	return data, collection, nil
}

func LoadSafeTensorData(collection SafeTensorsCollection, dir string, names []string) (map[string]SafeTensorData, error) {
	result := make(map[string]SafeTensorData, len(names))
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		tensor, ok := collection.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("safetensors tensor %q not found", name)
		}
		span, err := safeTensorByteSpan(name, tensor)
		if err != nil {
			return nil, err
		}
		shard, ok := collection.Shards[tensor.SourceFile]
		if !ok {
			return nil, fmt.Errorf("safetensors tensor %q references unknown source file %q", name, tensor.SourceFile)
		}
		absoluteOffset := int64(safeTensorsHeaderLengthBytes) + int64(shard.HeaderLength) + tensor.DataOffsets[0]
		file, err := os.Open(filepath.Join(dir, tensor.SourceFile))
		if err != nil {
			return nil, err
		}
		raw := make([]byte, span)
		_, readErr := file.ReadAt(raw, absoluteOffset)
		closeErr := file.Close()
		if readErr != nil {
			return nil, fmt.Errorf("%s: read tensor %q bytes: %w", tensor.SourceFile, name, readErr)
		}
		if closeErr != nil {
			return nil, closeErr
		}
		result[name] = SafeTensorData{
			Name:       name,
			DType:      tensor.DType,
			Shape:      append([]int64(nil), tensor.Shape...),
			SourceFile: tensor.SourceFile,
			ByteOffset: absoluteOffset,
			ByteLength: span,
			Bytes:      raw,
		}
	}
	return result, nil
}

func parseSafeTensorsHeader(path string, header []byte, fileSize int64, headerLength uint64) (SafeTensorsMetadata, error) {
	if !json.Valid(header) {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: invalid safetensors JSON header", path)
	}
	if dups, err := duplicateObjectKeys(header, ""); err != nil {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: parse safetensors JSON header: %w", path, err)
	} else if len(dups) > 0 {
		return SafeTensorsMetadata{}, fmt.Errorf("%s: duplicate tensor names in safetensors header: %s", path, strings.Join(dups, ", "))
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
		if _, ok := safeTensorDTypeSize(tensor.DType); ok {
			if _, err := safeTensorByteSpan(name, tensor); err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
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

func safeTensorByteSpan(name string, tensor SafeTensorInfo) (int64, error) {
	start, end := tensor.DataOffsets[0], tensor.DataOffsets[1]
	span := end - start
	if span < 0 {
		return 0, fmt.Errorf("safetensors tensor %q has invalid data_offsets %v", name, tensor.DataOffsets)
	}
	dtypeSize, ok := safeTensorDTypeSize(tensor.DType)
	if !ok {
		return 0, fmt.Errorf("safetensors tensor %q has unsupported dtype %q for byte ingestion", name, tensor.DType)
	}
	elements, err := safeTensorNumElements(name, tensor.Shape)
	if err != nil {
		return 0, err
	}
	if elements != 0 && dtypeSize > int64(^uint64(0)>>1)/elements {
		return 0, fmt.Errorf("safetensors tensor %q byte count overflows int64", name)
	}
	expected := elements * dtypeSize
	if span != expected {
		return 0, fmt.Errorf("safetensors tensor %q data_offsets span %d does not match shape elements %d * dtype size %d = %d", name, span, elements, dtypeSize, expected)
	}
	return span, nil
}

func safeTensorDTypeSize(dtype string) (int64, bool) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3":
		return 1, true
	case "F16", "BF16", "I16", "U16":
		return 2, true
	case "F32", "I32", "U32":
		return 4, true
	case "F64", "I64", "U64":
		return 8, true
	default:
		return 0, false
	}
}

func safeTensorNumElements(name string, shape []int64) (int64, error) {
	var elements int64 = 1
	for _, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("safetensors tensor %q has negative shape dimension %d", name, dim)
		}
		if dim != 0 && elements > int64(^uint64(0)>>1)/dim {
			return 0, fmt.Errorf("safetensors tensor %q element count overflows int64", name)
		}
		elements *= dim
	}
	return elements, nil
}

func duplicateObjectKeys(data []byte, objectKey string) ([]string, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	target := data
	if objectKey != "" {
		payload, ok := raw[objectKey]
		if !ok {
			return nil, nil
		}
		target = payload
	}
	decoder := json.NewDecoder(strings.NewReader(string(target)))
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := token.(json.Delim)
	if !ok || delim != '{' {
		return nil, fmt.Errorf("expected JSON object")
	}
	seen := map[string]struct{}{}
	dups := map[string]struct{}{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		key, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("expected JSON object key")
		}
		if _, ok := seen[key]; ok && key != "__metadata__" {
			dups[key] = struct{}{}
		}
		seen[key] = struct{}{}
		var discard json.RawMessage
		if err := decoder.Decode(&discard); err != nil {
			return nil, err
		}
	}
	out := make([]string, 0, len(dups))
	for key := range dups {
		out = append(out, key)
	}
	sort.Strings(out)
	return out, nil
}

func safeTensorsIndexMetadataValueIsScalar(value json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(value))
	if trimmed == "" {
		return false
	}
	switch trimmed[0] {
	case '{', '[':
		return false
	default:
		return json.Valid(value)
	}
}

func validateSafeTensorsShardPath(indexPath, tensorName, shard string) error {
	if shard == "" {
		return fmt.Errorf("%s: safetensors index tensor %q has empty shard file", indexPath, tensorName)
	}
	if filepath.IsAbs(shard) {
		return fmt.Errorf("%s: safetensors index tensor %q has absolute shard path %q", indexPath, tensorName, shard)
	}
	clean := filepath.Clean(shard)
	if clean == "." {
		return fmt.Errorf("%s: safetensors index tensor %q has invalid shard path %q", indexPath, tensorName, shard)
	}
	for _, part := range strings.Split(filepath.ToSlash(shard), "/") {
		if part == ".." {
			return fmt.Errorf("%s: safetensors index tensor %q has escaping shard path %q", indexPath, tensorName, shard)
		}
	}
	return nil
}

func cloneSafeTensorsStringMap(in map[string]string) map[string]json.RawMessage {
	if in == nil {
		return nil
	}
	out := make(map[string]json.RawMessage, len(in))
	for key, value := range in {
		rawValue, err := json.Marshal(value)
		if err != nil {
			continue
		}
		out[key] = rawValue
	}
	return out
}

func cloneSafeTensorsIndexMetadata(in map[string]json.RawMessage) map[string]json.RawMessage {
	if in == nil {
		return nil
	}
	out := make(map[string]json.RawMessage, len(in))
	for key, value := range in {
		out[key] = append(json.RawMessage(nil), value...)
	}
	return out
}

func cloneSafeTensorInfoMap(in map[string]SafeTensorInfo) map[string]SafeTensorInfo {
	out := make(map[string]SafeTensorInfo, len(in))
	for key, value := range in {
		value.Shape = append([]int64(nil), value.Shape...)
		out[key] = value
	}
	return out
}

type safeTensorSpan struct {
	name       string
	start, end int64
}
