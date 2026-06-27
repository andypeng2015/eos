package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"

	eosartifact "m31labs.dev/eos/artifact/eos"
	mll "m31labs.dev/mll"
)

const PretrainedBERTPackageVersion = "manta/pretrained-bert-package/v0alpha1"

var tagXPBT = [4]byte{'X', 'P', 'B', 'T'}

type PretrainedBERTPackageFile struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int    `json:"bytes"`
}

type PretrainedBERTPackage struct {
	Version                string                      `json:"version"`
	ModelName              string                      `json:"model_name,omitempty"`
	Architecture           string                      `json:"architecture,omitempty"`
	Config                 PretrainedBERTConfig        `json:"config"`
	STMetadata             *PretrainedBERTSTMetadata   `json:"sentence_transformers,omitempty"`
	Pooling                string                      `json:"pooling,omitempty"`
	Normalization          string                      `json:"normalization,omitempty"`
	MaxLength              int                         `json:"max_length"`
	NativeDim              int                         `json:"native_dim"`
	ModuleSHA256           string                      `json:"module_sha256"`
	WeightsSHA256          string                      `json:"weights_sha256"`
	Files                  []PretrainedBERTPackageFile `json:"files"`
	ModuleBytes            []byte                      `json:"module_bytes"`
	WeightsBytes           []byte                      `json:"weights_bytes"`
	ConfigJSON             []byte                      `json:"config_json"`
	Vocab                  []byte                      `json:"vocab"`
	TokenizerJSON          []byte                      `json:"tokenizer_json,omitempty"`
	TokenizerConfigJSON    []byte                      `json:"tokenizer_config_json,omitempty"`
	SpecialTokensMapJSON   []byte                      `json:"special_tokens_map_json,omitempty"`
	STPoolingConfigJSON    []byte                      `json:"sentence_transformers_pooling_config_json,omitempty"`
	SentenceBERTConfigJSON []byte                      `json:"sentence_bert_config_json,omitempty"`
	IdentitySHA256         string                      `json:"identity_sha256"`
}

type PretrainedBERTPackageExportReport struct {
	Status         string `json:"status"`
	OutputPath     string `json:"output_path,omitempty"`
	IdentitySHA256 string `json:"identity_sha256,omitempty"`
	ModelName      string `json:"model_name,omitempty"`
	ModuleSHA256   string `json:"module_sha256,omitempty"`
	WeightsSHA256  string `json:"weights_sha256,omitempty"`
	ConfigSHA256   string `json:"config_sha256,omitempty"`
	VocabSHA256    string `json:"vocab_sha256,omitempty"`
	Pooling        string `json:"pooling,omitempty"`
	Normalization  string `json:"normalization,omitempty"`
	MaxLength      int    `json:"max_length,omitempty"`
	NativeDim      int    `json:"native_dim,omitempty"`
	FileCount      int    `json:"file_count"`
	PackageBytes   int64  `json:"package_bytes,omitempty"`
}

func ExportPretrainedBERTPackageFromDir(dir string, plan PretrainedBERTImportPlan, outPath string) (PretrainedBERTPackageExportReport, error) {
	if outPath == "" {
		return PretrainedBERTPackageExportReport{}, fmt.Errorf("package output path is required")
	}
	pkg, err := BuildPretrainedBERTPackageFromDir(dir, plan)
	if err != nil {
		return PretrainedBERTPackageExportReport{}, err
	}
	data, err := encodePretrainedBERTPackageMLL(pkg)
	if err != nil {
		return PretrainedBERTPackageExportReport{}, err
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return PretrainedBERTPackageExportReport{}, err
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		return PretrainedBERTPackageExportReport{}, err
	}
	return PretrainedBERTPackageExportReport{
		Status:         "ok",
		OutputPath:     outPath,
		IdentitySHA256: pkg.IdentitySHA256,
		ModelName:      pkg.ModelName,
		ModuleSHA256:   pkg.ModuleSHA256,
		WeightsSHA256:  pkg.WeightsSHA256,
		ConfigSHA256:   packageFileHash(pkg.Files, "config"),
		VocabSHA256:    packageFileHash(pkg.Files, "vocab"),
		Pooling:        pkg.Pooling,
		Normalization:  pkg.Normalization,
		MaxLength:      pkg.MaxLength,
		NativeDim:      pkg.NativeDim,
		FileCount:      len(pkg.Files),
		PackageBytes:   int64(len(data)),
	}, nil
}

func BuildPretrainedBERTPackageFromDir(dir string, plan PretrainedBERTImportPlan) (PretrainedBERTPackage, error) {
	if dir == "" {
		return PretrainedBERTPackage{}, fmt.Errorf("source dir is required")
	}
	if err := plan.Config.Validate(); err != nil {
		return PretrainedBERTPackage{}, err
	}
	module, err := BuildPretrainedBERTEmbedderModule(plan)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	moduleBytes, err := eosartifact.EncodeMLL(module)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	decoded, _, err := LoadPretrainedBERTDecodedWeightsFromDir(dir, plan)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	weightFile, _, err := BuildPretrainedBERTWeightFileFromDecoded(decoded)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	weightsBytes, err := encodeWeightFileMLL(weightFile)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	configJSON, err := readRequiredPretrainedBERTPackageFile(dir, "config.json")
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	vocab, err := readRequiredPretrainedBERTPackageFile(dir, "vocab.txt")
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	tokenizerJSON, err := readOptionalPretrainedBERTPackageFile(dir, "tokenizer.json")
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	tokenizerConfig, err := readOptionalPretrainedBERTPackageFile(dir, "tokenizer_config.json")
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	specialTokens, err := readOptionalPretrainedBERTPackageFile(dir, "special_tokens_map.json")
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	stPooling, err := readOptionalPretrainedBERTPackageFile(dir, filepath.Join("1_Pooling", "config.json"))
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	stConfig, err := readOptionalPretrainedBERTPackageFile(dir, "sentence_bert_config.json")
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	stMetadata, err := parsePretrainedBERTPackageSTMetadata(stPooling, stConfig)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	pooling, _ := module.Metadata["pooling"].(string)
	normalization, _ := module.Metadata["normalization"].(string)
	maxLength, _ := resolvePretrainedBERTMaxLength(0, plan.Config, stMetadata)
	pkg := PretrainedBERTPackage{
		Version:                PretrainedBERTPackageVersion,
		ModelName:              plan.ModelName,
		Architecture:           plan.Architecture,
		Config:                 plan.Config,
		STMetadata:             stMetadata,
		Pooling:                pooling,
		Normalization:          normalization,
		MaxLength:              maxLength,
		NativeDim:              plan.Config.HiddenSize,
		ModuleSHA256:           sha256BytesHex(moduleBytes),
		WeightsSHA256:          sha256BytesHex(weightsBytes),
		ModuleBytes:            moduleBytes,
		WeightsBytes:           weightsBytes,
		ConfigJSON:             configJSON,
		Vocab:                  vocab,
		TokenizerJSON:          tokenizerJSON,
		TokenizerConfigJSON:    tokenizerConfig,
		SpecialTokensMapJSON:   specialTokens,
		STPoolingConfigJSON:    stPooling,
		SentenceBERTConfigJSON: stConfig,
	}
	pkg.Files = pretrainedBERTPackageFiles(pkg)
	pkg.IdentitySHA256 = pkg.IdentityHash()
	return pkg, nil
}

func ReadPretrainedBERTPackageFile(path string) (PretrainedBERTPackage, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	if !eosartifact.IsMLLBytes(data) {
		return PretrainedBERTPackage{}, fmt.Errorf("pretrained BERT package %q is not an MLL file", path)
	}
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return PretrainedBERTPackage{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return PretrainedBERTPackage{}, fmt.Errorf("pretrained BERT package profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	body, ok := reader.Section(tagXPBT)
	if !ok {
		return PretrainedBERTPackage{}, fmt.Errorf("pretrained BERT package missing XPBT section")
	}
	var pkg PretrainedBERTPackage
	if err := json.Unmarshal(body, &pkg); err != nil {
		return PretrainedBERTPackage{}, err
	}
	if err := pkg.Validate(); err != nil {
		return PretrainedBERTPackage{}, err
	}
	return pkg, nil
}

func (p PretrainedBERTPackage) Module() (*eosartifact.Module, error) {
	if len(p.ModuleBytes) == 0 {
		return nil, fmt.Errorf("pretrained BERT package missing module bytes")
	}
	return eosartifact.DecodeMLL(p.ModuleBytes)
}

func (p PretrainedBERTPackage) Weights() (WeightFile, error) {
	if len(p.WeightsBytes) == 0 {
		return WeightFile{}, fmt.Errorf("pretrained BERT package missing weights bytes")
	}
	return decodeWeightFileMLL(p.WeightsBytes)
}

func (p PretrainedBERTPackage) Tokenizer() (*HFWordPieceTokenizer, error) {
	return LoadHFWordPieceTokenizerFromBytes(p.Vocab, p.TokenizerConfigJSON, p.SpecialTokensMapJSON)
}

func (p PretrainedBERTPackage) Validate() error {
	if p.Version != PretrainedBERTPackageVersion {
		return fmt.Errorf("pretrained BERT package version %q is not supported, want %q", p.Version, PretrainedBERTPackageVersion)
	}
	if err := p.Config.Validate(); err != nil {
		return err
	}
	if p.NativeDim != p.Config.HiddenSize {
		return fmt.Errorf("pretrained BERT package native_dim %d does not match config hidden_size %d", p.NativeDim, p.Config.HiddenSize)
	}
	if len(p.ConfigJSON) == 0 || len(p.Vocab) == 0 || len(p.ModuleBytes) == 0 || len(p.WeightsBytes) == 0 {
		return fmt.Errorf("pretrained BERT package requires config, vocab, module, and weights bytes")
	}
	embeddedConfig, err := parsePretrainedBERTPackageConfig(p.ConfigJSON)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(p.Config, embeddedConfig) {
		return fmt.Errorf("pretrained BERT package config does not match embedded config.json")
	}
	embeddedSTMetadata, err := parsePretrainedBERTPackageSTMetadata(p.STPoolingConfigJSON, p.SentenceBERTConfigJSON)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(emptyPretrainedBERTSTMetadataToNil(p.STMetadata), embeddedSTMetadata) {
		return fmt.Errorf("pretrained BERT package sentence-transformers metadata does not match embedded metadata files")
	}
	resolvedMaxLength, _ := resolvePretrainedBERTMaxLength(0, p.Config, embeddedSTMetadata)
	if p.MaxLength != resolvedMaxLength {
		return fmt.Errorf("pretrained BERT package max_length %d does not match embedded config/sentence-transformers max length %d", p.MaxLength, resolvedMaxLength)
	}
	if p.ModuleSHA256 != sha256BytesHex(p.ModuleBytes) {
		return fmt.Errorf("pretrained BERT package module sha256 mismatch")
	}
	if p.WeightsSHA256 != sha256BytesHex(p.WeightsBytes) {
		return fmt.Errorf("pretrained BERT package weights sha256 mismatch")
	}
	computedFiles := pretrainedBERTPackageFiles(p)
	storedFiles := canonicalPretrainedBERTPackageFiles(p.Files)
	if !reflect.DeepEqual(storedFiles, computedFiles) {
		return fmt.Errorf("pretrained BERT package file table does not match embedded bytes")
	}
	if p.IdentitySHA256 != p.IdentityHash() {
		return fmt.Errorf("pretrained BERT package identity sha256 mismatch")
	}
	if _, err := p.Module(); err != nil {
		return fmt.Errorf("decode package module: %w", err)
	}
	if _, err := p.Weights(); err != nil {
		return fmt.Errorf("decode package weights: %w", err)
	}
	if _, err := p.Tokenizer(); err != nil {
		return fmt.Errorf("decode package tokenizer: %w", err)
	}
	return nil
}

func parsePretrainedBERTPackageConfig(data []byte) (PretrainedBERTConfig, error) {
	var cfg PretrainedBERTConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return PretrainedBERTConfig{}, fmt.Errorf("parse embedded config.json: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return PretrainedBERTConfig{}, fmt.Errorf("validate embedded config.json: %w", err)
	}
	return cfg, nil
}

func parsePretrainedBERTPackageSTMetadata(poolingConfigJSON, sentenceBERTConfigJSON []byte) (*PretrainedBERTSTMetadata, error) {
	meta := &PretrainedBERTSTMetadata{}
	if len(poolingConfigJSON) > 0 {
		pooling, source, err := parsePretrainedBERTSTPoolingBytes(poolingConfigJSON, filepath.Join("1_Pooling", "config.json"))
		if err != nil {
			return nil, err
		}
		if pooling != "" {
			meta.Pooling = pooling
			meta.PoolingSource = source
		}
	}
	if len(sentenceBERTConfigJSON) > 0 {
		maxSeqLength, source, err := parsePretrainedBERTSTMaxSeqLengthBytes(sentenceBERTConfigJSON, "sentence_bert_config.json")
		if err != nil {
			return nil, err
		}
		if maxSeqLength > 0 {
			meta.MaxSeqLength = maxSeqLength
			meta.MaxLengthSource = source
		}
	}
	return emptyPretrainedBERTSTMetadataToNil(meta), nil
}

func parsePretrainedBERTSTPoolingBytes(data []byte, source string) (string, string, error) {
	var raw struct {
		PoolingModeCLSToken    bool `json:"pooling_mode_cls_token"`
		PoolingModeMeanTokens  bool `json:"pooling_mode_mean_tokens"`
		PoolingModeMaxTokens   bool `json:"pooling_mode_max_tokens"`
		PoolingModeMeanSqrtLen bool `json:"pooling_mode_mean_sqrt_len_tokens"`
		PoolingModeWeighted    bool `json:"pooling_mode_weightedmean_tokens"`
		PoolingModeLastToken   bool `json:"pooling_mode_lasttoken"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return "", "", fmt.Errorf("parse embedded %s: %w", source, err)
	}
	switch {
	case raw.PoolingModeCLSToken && !raw.PoolingModeMeanTokens && !raw.PoolingModeMaxTokens && !raw.PoolingModeMeanSqrtLen && !raw.PoolingModeWeighted && !raw.PoolingModeLastToken:
		return "cls", source, nil
	case raw.PoolingModeMeanTokens && !raw.PoolingModeCLSToken && !raw.PoolingModeMaxTokens && !raw.PoolingModeMeanSqrtLen && !raw.PoolingModeWeighted && !raw.PoolingModeLastToken:
		return "masked_mean", source, nil
	case !raw.PoolingModeCLSToken && !raw.PoolingModeMeanTokens && !raw.PoolingModeMaxTokens && !raw.PoolingModeMeanSqrtLen && !raw.PoolingModeWeighted && !raw.PoolingModeLastToken:
		return "", source, nil
	default:
		return "", "", fmt.Errorf("unsupported embedded SentenceTransformers pooling config %s: only CLS-only and mean-only pooling are supported", source)
	}
}

func parsePretrainedBERTSTMaxSeqLengthBytes(data []byte, source string) (int, string, error) {
	var raw struct {
		MaxSeqLength int `json:"max_seq_length"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return 0, "", fmt.Errorf("parse embedded %s: %w", source, err)
	}
	if raw.MaxSeqLength < 0 {
		return 0, "", fmt.Errorf("sentence-transformers max_seq_length must be non-negative, got %d", raw.MaxSeqLength)
	}
	if raw.MaxSeqLength == 0 {
		return 0, source, nil
	}
	return raw.MaxSeqLength, source, nil
}

func emptyPretrainedBERTSTMetadataToNil(meta *PretrainedBERTSTMetadata) *PretrainedBERTSTMetadata {
	if meta == nil {
		return nil
	}
	if meta.Pooling == "" && meta.PoolingSource == "" && meta.MaxSeqLength == 0 && meta.MaxLengthSource == "" {
		return nil
	}
	return meta
}

func (p PretrainedBERTPackage) IdentityHash() string {
	type identity struct {
		Version       string                      `json:"version"`
		ModelName     string                      `json:"model_name,omitempty"`
		Architecture  string                      `json:"architecture,omitempty"`
		ModelType     string                      `json:"model_type,omitempty"`
		Pooling       string                      `json:"pooling,omitempty"`
		Normalization string                      `json:"normalization,omitempty"`
		MaxLength     int                         `json:"max_length"`
		NativeDim     int                         `json:"native_dim"`
		ModuleSHA256  string                      `json:"module_sha256"`
		WeightsSHA256 string                      `json:"weights_sha256"`
		Files         []PretrainedBERTPackageFile `json:"files"`
	}
	files := pretrainedBERTPackageFiles(p)
	moduleSHA256 := sha256BytesHex(p.ModuleBytes)
	weightsSHA256 := sha256BytesHex(p.WeightsBytes)
	data, _ := json.Marshal(identity{
		Version:       p.Version,
		ModelName:     p.ModelName,
		Architecture:  p.Architecture,
		ModelType:     p.Config.ModelType,
		Pooling:       p.Pooling,
		Normalization: p.Normalization,
		MaxLength:     p.MaxLength,
		NativeDim:     p.NativeDim,
		ModuleSHA256:  moduleSHA256,
		WeightsSHA256: weightsSHA256,
		Files:         files,
	})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func encodePretrainedBERTPackageMLL(pkg PretrainedBERTPackage) ([]byte, error) {
	if err := pkg.Validate(); err != nil {
		return nil, err
	}
	body, err := json.Marshal(pkg)
	if err != nil {
		return nil, err
	}
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	var (
		dimsBuilder mll.DimsBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)
	head := mll.HeadSection{
		Name:        strg.Intern("pretrained-bert-package"),
		Description: strg.Intern("Eos source-free imported BERT package"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "pretrained_bert_package_version", pkg.Version),
			headStringMeta(strg, "model_name", pkg.ModelName),
			headStringMeta(strg, "identity_sha256", pkg.IdentitySHA256),
			headStringMeta(strg, "module_sha256", pkg.ModuleSHA256),
			headStringMeta(strg, "weights_sha256", pkg.WeightsSHA256),
			headIntMeta(strg, "native_dim", int64(pkg.NativeDim)),
			headIntMeta(strg, "max_length", int64(pkg.MaxLength)),
		},
	}
	sections := make([]mll.SectionInput, 0, 7)
	if headBody, digestBody, err := encodeHeadSection(head, mll.ProfileSealed); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagHEAD, Body: headBody, DigestBody: digestBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if strgBody, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSTRG, Body: strgBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if dimsBody, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagDIMS, Body: dimsBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if parmBody, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagPARM, Body: parmBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if entrBody, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagENTR, Body: entrBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if tnsrBody, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTNSR, Body: tnsrBody, Flags: mll.SectionFlagRequired | mll.SectionFlagAligned, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXPBT,
		Body:          body,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func pretrainedBERTPackageFiles(pkg PretrainedBERTPackage) []PretrainedBERTPackageFile {
	files := []PretrainedBERTPackageFile{
		packageFile("module", "module.mll", pkg.ModuleBytes),
		packageFile("weights", "weights.mll", pkg.WeightsBytes),
		packageFile("config", "config.json", pkg.ConfigJSON),
		packageFile("vocab", "vocab.txt", pkg.Vocab),
	}
	addOptional := func(role, path string, data []byte) {
		if len(data) > 0 {
			files = append(files, packageFile(role, path, data))
		}
	}
	addOptional("tokenizer_json", "tokenizer.json", pkg.TokenizerJSON)
	addOptional("tokenizer_config", "tokenizer_config.json", pkg.TokenizerConfigJSON)
	addOptional("special_tokens_map", "special_tokens_map.json", pkg.SpecialTokensMapJSON)
	addOptional("sentence_transformers_pooling", "1_Pooling/config.json", pkg.STPoolingConfigJSON)
	addOptional("sentence_transformers_config", "sentence_bert_config.json", pkg.SentenceBERTConfigJSON)
	sort.Slice(files, func(i, j int) bool { return files[i].Role < files[j].Role })
	return files
}

func canonicalPretrainedBERTPackageFiles(files []PretrainedBERTPackageFile) []PretrainedBERTPackageFile {
	canonical := append([]PretrainedBERTPackageFile(nil), files...)
	sort.Slice(canonical, func(i, j int) bool {
		if canonical[i].Role != canonical[j].Role {
			return canonical[i].Role < canonical[j].Role
		}
		return canonical[i].Path < canonical[j].Path
	})
	return canonical
}

func packageFile(role, path string, data []byte) PretrainedBERTPackageFile {
	return PretrainedBERTPackageFile{Role: role, Path: path, SHA256: sha256BytesHex(data), Bytes: len(data)}
}

func packageFileHash(files []PretrainedBERTPackageFile, role string) string {
	for _, file := range files {
		if file.Role == role {
			return file.SHA256
		}
	}
	return ""
}

func readRequiredPretrainedBERTPackageFile(dir, rel string) ([]byte, error) {
	data, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		return nil, err
	}
	return data, nil
}

func readOptionalPretrainedBERTPackageFile(dir, rel string) ([]byte, error) {
	data, err := os.ReadFile(filepath.Join(dir, rel))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return data, nil
}
