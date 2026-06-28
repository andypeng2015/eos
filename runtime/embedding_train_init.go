package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// EmbeddingTrainInitOptions controls training-package initialization from an artifact.
type EmbeddingTrainInitOptions struct {
	Seed                    int64
	ShapeSizes              map[string]int
	BootstrapArtifactPath   string
	BootstrapCheckpointPath string
}

// InitializeEmbeddingTrainerPackage reads an artifact plus its sibling embedding manifest, initializes trainable weights, and writes a training package.
func InitializeEmbeddingTrainerPackage(artifactPath string, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	manifest, err := ReadEmbeddingManifestFile(ResolveEmbeddingManifestPath(artifactPath))
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return InitializeEmbeddingTrainerPackageWithManifest(artifactPath, manifest, EmbeddingTrainConfig{}, opts)
}

// InitializeEmbeddingTrainerPackageWithManifest initializes a training package from an explicit embedding manifest and trainer config.
func InitializeEmbeddingTrainerPackageWithManifest(artifactPath string, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	mod, err := eosartifact.ReadFile(artifactPath)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	manifest = manifest.normalized()
	if manifest.ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1 {
		return initializeGenericEmbeddingTrainerPackage(artifactPath, mod, manifest, cfg, opts)
	}
	trainManifest := EmbeddingTrainManifest{
		Name:      manifest.Name,
		Embedding: manifest,
		Config:    cfg,
	}
	if err := trainManifest.ValidateModule(mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weights, err := initializedTrainingWeights(mod, manifest, opts)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if opts.BootstrapArtifactPath != "" || opts.BootstrapCheckpointPath != "" {
		if err := bootstrapTrainingWeights(weights, manifest, opts); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	trainer, err := NewEmbeddingTrainer(mod, manifest, weights, cfg)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return trainer.WriteTrainingPackage(artifactPath)
}

func initializeGenericEmbeddingTrainerPackage(artifactPath string, mod *eosartifact.Module, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	manifest = manifest.normalizedForModule(mod)
	if err := manifest.ValidateModule(mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if err := validateGenericTrainableEmbeddingModule(mod, manifest); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weights, err := initializedTrainingWeights(mod, manifest, opts)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if opts.BootstrapArtifactPath != "" || opts.BootstrapCheckpointPath != "" {
		if err := bootstrapTrainingWeights(weights, manifest, opts); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	return writeGenericEmbeddingTrainingPackage(artifactPath, mod, manifest, cfg, weights)
}

func validateGenericTrainableEmbeddingModule(mod *eosartifact.Module, manifest EmbeddingManifest) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if manifest.ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 {
		return fmt.Errorf("generic embedding package initialization requires architecture_version=%q", EmbeddingArchitectureCompactTransformerV1)
	}
	if manifest.ParameterTying != EmbeddingParameterTyingUntied {
		return fmt.Errorf("%s package initialization requires parameter_tying=%q", manifest.ArchitectureVersion, EmbeddingParameterTyingUntied)
	}
	for _, param := range mod.Params {
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return fmt.Errorf("param %q is not a tensor weight", param.Name)
		}
		if !param.Trainable {
			return fmt.Errorf("param %q is not trainable", param.Name)
		}
	}
	if manifest.OutputProjectionParam != "" {
		if err := validateEmbeddingParam(mod, manifest.OutputProjectionParam); err != nil {
			return err
		}
	}
	return nil
}

func writeGenericEmbeddingTrainingPackage(artifactPath string, mod *eosartifact.Module, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, weights map[string]*backend.Tensor) (EmbeddingTrainPackagePaths, error) {
	if err := eosartifact.WriteFile(artifactPath, mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	embeddingManifestPath := DefaultEmbeddingManifestPath(artifactPath)
	if err := manifest.WriteFile(embeddingManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weightPath := DefaultWeightFilePath(artifactPath)
	if err := NewWeightFile(weights).WriteFile(weightPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	memoryPlan := NewMemoryPlan(mod, weights, MemoryPlanOptions{})
	memoryPlanPath := DefaultMemoryPlanPath(artifactPath)
	if err := memoryPlan.WriteFile(memoryPlanPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainManifest := EmbeddingTrainManifest{
		Name:      manifest.Name,
		Embedding: manifest,
		Config:    cfg,
	}
	trainManifestPath := DefaultEmbeddingTrainManifestPath(artifactPath)
	if err := trainManifest.WriteFile(trainManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	checkpoint := genericEmbeddingTrainCheckpoint(manifest, cfg, weights)
	checkpointPath := DefaultEmbeddingCheckpointPath(artifactPath)
	if err := checkpoint.WriteFile(checkpointPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainProfilePath := DefaultEmbeddingTrainProfilePath(artifactPath)
	if err := (EmbeddingTrainProfile{Version: EmbeddingTrainProfileVersion}).WriteFile(trainProfilePath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageFiles := map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": embeddingManifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
		"train_manifest":     trainManifestPath,
		"checkpoint":         checkpointPath,
		"train_profile":      trainProfilePath,
	}
	tokenizerPath := DefaultTokenizerPath(artifactPath)
	if _, err := os.Stat(tokenizerPath); err == nil {
		packageFiles["tokenizer"] = tokenizerPath
	}
	packageManifest, err := BuildPackageManifest(PackageTraining, mod, packageFiles)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageManifestPath := DefaultPackageManifestPath(artifactPath)
	if err := packageManifest.WriteFile(packageManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return EmbeddingTrainPackagePaths{
		ArtifactPath:          artifactPath,
		EmbeddingManifestPath: embeddingManifestPath,
		TokenizerPath:         tokenizerPath,
		WeightFilePath:        weightPath,
		MemoryPlanPath:        memoryPlanPath,
		TrainManifestPath:     trainManifestPath,
		CheckpointPath:        checkpointPath,
		TrainProfilePath:      trainProfilePath,
		PackageManifestPath:   packageManifestPath,
	}, nil
}

func genericEmbeddingTrainCheckpoint(manifest EmbeddingManifest, cfg EmbeddingTrainConfig, weights map[string]*backend.Tensor) EmbeddingTrainCheckpoint {
	tensors := make(map[string]*backend.Tensor, len(weights))
	moments := make(map[string]*backend.Tensor, len(weights)*2)
	for name, tensor := range weights {
		if tensor == nil {
			continue
		}
		master := tensorAsMasterF32(tensor)
		tensors[name] = master
		moments[name+"_moment_1"] = zeroLikeMaster(master)
		moments[name+"_moment_2"] = zeroLikeMaster(master)
	}
	return EmbeddingTrainCheckpoint{
		Version:       EmbeddingTrainCheckpointVersion,
		Manifest:      manifest,
		Config:        cfg,
		Tensors:       tensors,
		MomentTensors: moments,
	}
}

func initializedTrainingWeights(mod *eosartifact.Module, manifest EmbeddingManifest, opts EmbeddingTrainInitOptions) (map[string]*backend.Tensor, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	seed := opts.Seed
	if seed == 0 {
		seed = 1
	}
	rng := rand.New(rand.NewSource(seed))
	weights := make(map[string]*backend.Tensor, len(mod.Params))
	for _, param := range mod.Params {
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return nil, fmt.Errorf("param %q is not a tensor weight", param.Name)
		}
		shape, err := resolveTrainingInitShape(param.Type.Tensor.Shape, manifest.Tokenizer, opts.ShapeSizes)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		tensor, err := initializedWeightTensor(param.Type.Tensor.DType, shape, rng)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		if manifest.roleConditioned() && param.Name == manifest.RoleEmbeddingParam {
			tensor = zeroInitializedTensor(param.Type.Tensor.DType, shape)
		}
		weights[param.Name] = tensor
	}
	return weights, nil
}

func zeroInitializedTensor(dtype string, shape []int) *backend.Tensor {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	data := make([]float32, n)
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, data)
	case "q4":
		return backend.NewTensorQ4(shape, data)
	case "q8":
		return backend.NewTensorQ8(shape, data)
	default:
		return backend.NewTensorF32(shape, data)
	}
}

func resolveTrainingInitShape(shape []string, tokenizer TokenizerManifest, sizes map[string]int) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		if n, err := strconv.Atoi(dim); err == nil {
			if n <= 0 {
				return nil, fmt.Errorf("shape dim %q must be positive", dim)
			}
			out[i] = n
			continue
		}
		if sizes != nil && sizes[dim] > 0 {
			out[i] = sizes[dim]
			continue
		}
		switch dim {
		case "V":
			if tokenizer.VocabSize > 0 {
				out[i] = tokenizer.VocabSize
				continue
			}
		case "T":
			if tokenizer.MaxSequence > 0 {
				out[i] = tokenizer.MaxSequence
				continue
			}
		}
		return nil, fmt.Errorf("unresolved symbolic dim %q", dim)
	}
	return out, nil
}

func initializedWeightTensor(dtype string, shape []int, rng *rand.Rand) (*backend.Tensor, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	n := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("shape %v is invalid", shape)
		}
		n *= dim
	}
	scale := initializerScale(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = (rng.Float32()*2 - 1) * scale
	}
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, data), nil
	case "f32":
		return backend.NewTensorF32(shape, data), nil
	case "q4":
		return backend.NewTensorQ4(shape, data), nil
	case "q8":
		return backend.NewTensorQ8(shape, data), nil
	default:
		return nil, fmt.Errorf("unsupported weight dtype %q", dtype)
	}
}

func initializerScale(shape []int) float32 {
	if len(shape) >= 2 {
		fanIn := shape[len(shape)-2]
		fanOut := shape[len(shape)-1]
		if fanIn > 0 && fanOut > 0 {
			return float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))
		}
	}
	if len(shape) == 1 && shape[0] > 0 {
		return float32(1.0 / math.Sqrt(float64(shape[0])))
	}
	return 0.02
}

func bootstrapTrainingWeights(weights map[string]*backend.Tensor, targetManifest EmbeddingManifest, opts EmbeddingTrainInitOptions) error {
	checkpointPath, err := resolveBootstrapCheckpointPath(opts)
	if err != nil {
		return err
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(checkpointPath)
	if err != nil {
		return fmt.Errorf("read bootstrap checkpoint %q: %w", checkpointPath, err)
	}
	copies := []struct {
		role       string
		targetName string
		source     *backend.Tensor
	}{
		{role: "token_embedding", targetName: targetManifest.TokenEmbeddingParam, source: checkpoint.TokenEmbedding},
		{role: "role_embedding", targetName: targetManifest.RoleEmbeddingParam, source: checkpoint.RoleEmbedding},
		{role: "attention_query", targetName: targetManifest.AttentionQueryParam, source: checkpoint.AttentionQuery},
		{role: "attention_key", targetName: targetManifest.AttentionKeyParam, source: checkpoint.AttentionKey},
		{role: "attention_value", targetName: targetManifest.AttentionValueParam, source: checkpoint.AttentionValue},
		{role: "attention_output", targetName: targetManifest.AttentionOutputParam, source: checkpoint.AttentionOutput},
		{role: "hidden_projection", targetName: targetManifest.HiddenProjectionParam, source: checkpoint.HiddenProjection},
		{role: "projection", targetName: targetManifest.ProjectionParam, source: checkpoint.Projection},
	}
	copied := map[string]struct{}{}
	for _, copySpec := range copies {
		if copySpec.targetName == "" || copySpec.source == nil {
			continue
		}
		target := weights[copySpec.targetName]
		if target == nil {
			return fmt.Errorf("bootstrap target tensor %q for role %s is missing", copySpec.targetName, copySpec.role)
		}
		if err := copyOverlappingTensor(target, copySpec.source); err != nil {
			return fmt.Errorf("bootstrap %s: %w", copySpec.role, err)
		}
		copied[copySpec.targetName] = struct{}{}
	}
	targetNames := make([]string, 0, len(weights))
	for name := range weights {
		targetNames = append(targetNames, name)
	}
	sort.Strings(targetNames)
	for _, name := range targetNames {
		if _, ok := copied[name]; ok {
			continue
		}
		source := checkpoint.Tensors[name]
		if source == nil {
			continue
		}
		if err := copyOverlappingTensor(weights[name], source); err != nil {
			return fmt.Errorf("bootstrap generic tensor %q into target %q: %w", name, name, err)
		}
	}
	return nil
}

func resolveBootstrapCheckpointPath(opts EmbeddingTrainInitOptions) (string, error) {
	if opts.BootstrapArtifactPath != "" && opts.BootstrapCheckpointPath != "" {
		return "", fmt.Errorf("bootstrap artifact path and checkpoint path are mutually exclusive")
	}
	if opts.BootstrapCheckpointPath != "" {
		if _, err := os.Stat(opts.BootstrapCheckpointPath); err != nil {
			return "", fmt.Errorf("bootstrap checkpoint %q: %w", opts.BootstrapCheckpointPath, err)
		}
		return opts.BootstrapCheckpointPath, nil
	}
	if opts.BootstrapArtifactPath == "" {
		return "", fmt.Errorf("bootstrap artifact path is required")
	}
	if _, err := os.Stat(opts.BootstrapArtifactPath); err != nil {
		return "", fmt.Errorf("bootstrap artifact %q: %w", opts.BootstrapArtifactPath, err)
	}
	if strings.HasSuffix(opts.BootstrapArtifactPath, ".embed-train.mll") {
		return opts.BootstrapArtifactPath, nil
	}
	checkpointPath := DefaultEmbeddingCheckpointPath(opts.BootstrapArtifactPath)
	if _, err := os.Stat(checkpointPath); err != nil {
		return "", fmt.Errorf("bootstrap checkpoint %q derived from artifact %q: %w", checkpointPath, opts.BootstrapArtifactPath, err)
	}
	return checkpointPath, nil
}

func copyOverlappingTensor(target, source *backend.Tensor) error {
	if target == nil || source == nil {
		return fmt.Errorf("nil tensor")
	}
	if len(target.Shape) != len(source.Shape) {
		return fmt.Errorf("rank mismatch target=%d source=%d", len(target.Shape), len(source.Shape))
	}
	if len(target.F32) != target.Elements() {
		return fmt.Errorf("target tensor shape %v has %d f32 values, want %d", target.Shape, len(target.F32), target.Elements())
	}
	if len(source.F32) != source.Elements() {
		return fmt.Errorf("source tensor shape %v has %d f32 values, want %d", source.Shape, len(source.F32), source.Elements())
	}
	limits := make([]int, len(target.Shape))
	for i := range target.Shape {
		if target.Shape[i] <= 0 || source.Shape[i] <= 0 {
			return fmt.Errorf("invalid shapes target=%v source=%v", target.Shape, source.Shape)
		}
		limits[i] = min(target.Shape[i], source.Shape[i])
	}
	copyOverlappingTensorAtRank(target.F32, target.Shape, source.F32, source.Shape, limits, 0, 0, 0)
	return nil
}

func copyOverlappingTensorAtRank(targetData []float32, targetShape []int, sourceData []float32, sourceShape []int, limits []int, rank int, targetOffset int, sourceOffset int) {
	if rank == len(limits)-1 {
		copy(targetData[targetOffset:targetOffset+limits[rank]], sourceData[sourceOffset:sourceOffset+limits[rank]])
		return
	}
	targetStride := rowMajorStride(targetShape, rank)
	sourceStride := rowMajorStride(sourceShape, rank)
	for i := 0; i < limits[rank]; i++ {
		copyOverlappingTensorAtRank(targetData, targetShape, sourceData, sourceShape, limits, rank+1, targetOffset+i*targetStride, sourceOffset+i*sourceStride)
	}
}

func rowMajorStride(shape []int, dim int) int {
	stride := 1
	for i := dim + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	return stride
}
