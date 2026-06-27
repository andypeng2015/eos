package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"m31labs.dev/eos/runtime/backend"
	mll "m31labs.dev/mll"
)

const (
	PretrainedBERTProjectionHeadSchema = "manta.pretrained_bert_projection_head.v1"
	pretrainedBERTProjectionTensorName = "projection"
)

var tagXPHD = [4]byte{'X', 'P', 'H', 'D'}

type PretrainedBERTProjectionHead struct {
	Schema         string            `json:"schema"`
	SourceModel    string            `json:"source_model,omitempty"`
	InputDim       int               `json:"input_dim"`
	OutputDim      int               `json:"output_dim"`
	Seed           int64             `json:"seed,omitempty"`
	Loss           string            `json:"loss,omitempty"`
	DataProvenance string            `json:"data_provenance,omitempty"`
	Initialization string            `json:"initialization,omitempty"`
	Provenance     map[string]string `json:"provenance,omitempty"`
	CreatedAt      time.Time         `json:"created_at,omitempty"`
	Weights        []float32         `json:"-"`
	WeightsSHA256  string            `json:"weights_sha256,omitempty"`
	ContentSHA256  string            `json:"content_sha256,omitempty"`
}

type pretrainedBERTProjectionHeadBehaviorIdentity struct {
	Schema        string `json:"schema"`
	InputDim      int    `json:"input_dim"`
	OutputDim     int    `json:"output_dim"`
	WeightsSHA256 string `json:"weights_sha256"`
}

func NewPretrainedBERTProjectionHead(inputDim, outputDim int, weights []float32) (PretrainedBERTProjectionHead, error) {
	head := PretrainedBERTProjectionHead{
		Schema:    PretrainedBERTProjectionHeadSchema,
		InputDim:  inputDim,
		OutputDim: outputDim,
		Weights:   append([]float32(nil), weights...),
		CreatedAt: time.Now().UTC(),
	}
	if err := head.Validate(); err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	head.WeightsSHA256 = sha256Float32Hex(head.Weights)
	head.ContentSHA256 = head.IdentityHash()
	return head, nil
}

func (h PretrainedBERTProjectionHead) Validate() error {
	if h.Schema == "" {
		return fmt.Errorf("projection head schema is required")
	}
	if h.Schema != PretrainedBERTProjectionHeadSchema {
		return fmt.Errorf("projection head schema %q is not supported, want %q", h.Schema, PretrainedBERTProjectionHeadSchema)
	}
	if h.InputDim <= 0 || h.OutputDim <= 0 {
		return fmt.Errorf("projection head dimensions must be positive, got input=%d output=%d", h.InputDim, h.OutputDim)
	}
	if len(h.Weights) != h.InputDim*h.OutputDim {
		return fmt.Errorf("projection head weights length %d does not match input_dim*output_dim %d", len(h.Weights), h.InputDim*h.OutputDim)
	}
	for i, value := range h.Weights {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("projection head weight %d is not finite: %v", i, value)
		}
	}
	return nil
}

func (h PretrainedBERTProjectionHead) Apply(vector []float32) ([]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	if len(vector) != h.InputDim {
		return nil, fmt.Errorf("projection head input dimension mismatch: got %d want %d", len(vector), h.InputDim)
	}
	out := make([]float32, h.OutputDim)
	for j := 0; j < h.OutputDim; j++ {
		var sum float64
		for i, value := range vector {
			sum += float64(value) * float64(h.Weights[i*h.OutputDim+j])
		}
		out[j] = float32(sum)
	}
	return normalizeRetrievalVector(out), nil
}

func (h PretrainedBERTProjectionHead) IdentityHash() string {
	canonical := pretrainedBERTProjectionHeadBehaviorIdentity{
		Schema:        h.Schema,
		InputDim:      h.InputDim,
		OutputDim:     h.OutputDim,
		WeightsSHA256: sha256Float32Hex(h.Weights),
	}
	data, err := json.Marshal(canonical)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func WritePretrainedBERTProjectionHeadFile(path string, head PretrainedBERTProjectionHead) error {
	if err := head.Validate(); err != nil {
		return err
	}
	head.WeightsSHA256 = sha256Float32Hex(head.Weights)
	head.ContentSHA256 = head.IdentityHash()
	data, err := encodePretrainedBERTProjectionHeadMLL(head)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func ReadPretrainedBERTProjectionHeadFile(path string) (PretrainedBERTProjectionHead, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	head, err := decodePretrainedBERTProjectionHeadMLL(data)
	if err != nil {
		return PretrainedBERTProjectionHead{}, fmt.Errorf("read projection head %s: %w", path, err)
	}
	if err := head.Validate(); err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	return head, nil
}

func encodePretrainedBERTProjectionHeadMLL(head PretrainedBERTProjectionHead) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	typeBuilder := mll.NewTypeBuilder()
	parmBuilder := mll.NewParmBuilder()
	tnsrBuilder := mll.NewTnsrBuilder()
	nameIdx := strg.Intern(pretrainedBERTProjectionTensorName)
	shape := []int{head.InputDim, head.OutputDim}
	mllShape, err := intShapeToMLLShape(strg, shape)
	if err != nil {
		return nil, err
	}
	typeIndex := typeBuilder.AddTensorType(nameIdx, mll.DTypeF32, mllShape)
	parmBuilder.Add(mll.ParmDecl{NameIdx: nameIdx, TypeRef: mll.Ref{Tag: mll.TagTYPE, Index: typeIndex}})
	tnsrBuilder.Add(mll.TensorEntry{
		NameIdx: nameIdx,
		DType:   mll.DTypeF32,
		Shape:   []uint64{uint64(head.InputDim), uint64(head.OutputDim)},
		Data:    encodeFloat32Bytes(head.Weights),
	})
	meta := head
	meta.Weights = nil
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = time.Now().UTC()
	}
	metaBody, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	headSection := mll.HeadSection{
		Name:        strg.Intern("manta-pretrained-bert-projection-head"),
		Description: strg.Intern("Eos imported BERT compact projection head"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "projection_head_schema", head.Schema),
			headIntMeta(strg, "input_dim", int64(head.InputDim)),
			headIntMeta(strg, "output_dim", int64(head.OutputDim)),
			headStringMeta(strg, "weights_sha256", head.WeightsSHA256),
		},
	}
	sections := make([]mll.SectionInput, 0, 6)
	if body, digestBody, err := encodeHeadSection(headSection, mll.ProfileWeightsOnly); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagHEAD, Body: body, DigestBody: digestBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSTRG, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagPARM, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTNSR, Body: body, Flags: mll.SectionFlagRequired | mll.SectionFlagAligned, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{Tag: tagXPHD, Body: metaBody, Flags: mll.SectionFlagSkippable | mll.SectionFlagSchemaless, SchemaVersion: 1})
	return mll.WriteToBytes(mll.ProfileWeightsOnly, mll.V1_0, sections)
}

func decodePretrainedBERTProjectionHeadMLL(data []byte) (PretrainedBERTProjectionHead, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	if reader.Profile() != mll.ProfileWeightsOnly {
		return PretrainedBERTProjectionHead{}, fmt.Errorf("projection head profile = %d, want %d", reader.Profile(), mll.ProfileWeightsOnly)
	}
	body, ok := reader.Section(tagXPHD)
	if !ok {
		return PretrainedBERTProjectionHead{}, fmt.Errorf("projection head missing XPHD metadata section")
	}
	var head PretrainedBERTProjectionHead
	if err := json.Unmarshal(body, &head); err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	if head.Schema == "" {
		head.Schema = PretrainedBERTProjectionHeadSchema
	}
	weights, err := decodeProjectionHeadWeights(reader, head.InputDim, head.OutputDim)
	if err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	head.Weights = weights
	actualWeightsSHA256 := sha256Float32Hex(weights)
	if head.WeightsSHA256 != "" && head.WeightsSHA256 != actualWeightsSHA256 {
		return PretrainedBERTProjectionHead{}, fmt.Errorf("projection head weights_sha256 %q does not match tensor weights %q", head.WeightsSHA256, actualWeightsSHA256)
	}
	head.WeightsSHA256 = actualWeightsSHA256
	if err := head.Validate(); err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	actualContentSHA256 := head.IdentityHash()
	if head.ContentSHA256 != "" && head.ContentSHA256 != actualContentSHA256 {
		return PretrainedBERTProjectionHead{}, fmt.Errorf("projection head content_sha256 %q does not match behavior content %q", head.ContentSHA256, actualContentSHA256)
	}
	head.ContentSHA256 = actualContentSHA256
	return head, nil
}

func decodeProjectionHeadWeights(reader *mll.Reader, inputDim, outputDim int) ([]float32, error) {
	weightFile, err := decodeWeightFileFromMLLReader(reader, WeightFileVersion, nil, nil)
	if err != nil {
		return nil, err
	}
	tensor := weightFile.Weights[pretrainedBERTProjectionTensorName]
	if tensor == nil {
		names := make([]string, 0, len(weightFile.Weights))
		for name := range weightFile.Weights {
			names = append(names, name)
		}
		sort.Strings(names)
		return nil, fmt.Errorf("projection tensor %q missing; found %v", pretrainedBERTProjectionTensorName, names)
	}
	if tensor.DType != "f32" {
		return nil, fmt.Errorf("projection tensor dtype = %q, want f32", tensor.DType)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != inputDim || tensor.Shape[1] != outputDim {
		return nil, fmt.Errorf("projection tensor shape = %v, want [%d %d]", tensor.Shape, inputDim, outputDim)
	}
	return append([]float32(nil), tensor.F32...), nil
}

func sha256Float32Hex(values []float32) string {
	return sha256BytesHex(encodeFloat32Bytes(values))
}

func sha256BytesHex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func projectionHeadFromRows(inputDim, outputDim int, weights []float32, sourceModel, initialization, loss, dataProvenance string, seed int64, provenance map[string]string) (PretrainedBERTProjectionHead, error) {
	head, err := NewPretrainedBERTProjectionHead(inputDim, outputDim, weights)
	if err != nil {
		return PretrainedBERTProjectionHead{}, err
	}
	head.SourceModel = sourceModel
	head.Initialization = initialization
	head.Loss = loss
	head.DataProvenance = dataProvenance
	head.Seed = seed
	if len(provenance) > 0 {
		head.Provenance = map[string]string{}
		for k, v := range provenance {
			head.Provenance[k] = v
		}
	}
	head.WeightsSHA256 = sha256Float32Hex(head.Weights)
	head.ContentSHA256 = head.IdentityHash()
	return head, nil
}

func tensorFromProjectionHead(head PretrainedBERTProjectionHead) *backend.Tensor {
	return backend.NewTensorF32([]int{head.InputDim, head.OutputDim}, append([]float32(nil), head.Weights...))
}
