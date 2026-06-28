package eosruntime

import (
	"fmt"
	"math/rand"
	"os"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func SyncEmbeddingTokenizerVocab(artifactPath string, vocabSize int) error {
	if vocabSize <= 0 {
		return fmt.Errorf("tokenizer vocab size must be positive")
	}
	var tokenParam string
	var embeddingManifest *EmbeddingManifest
	if manifestPath := ResolveEmbeddingManifestPath(artifactPath); fileExists(manifestPath) {
		manifest, err := ReadEmbeddingManifestFile(manifestPath)
		if err != nil {
			return err
		}
		embeddingManifest = &manifest
		tokenParam = manifest.TokenEmbeddingParam
	}
	var trainManifest *EmbeddingTrainManifest
	if trainManifestPath := ResolveEmbeddingTrainManifestPath(artifactPath); fileExists(trainManifestPath) {
		manifest, err := ReadEmbeddingTrainManifestFile(trainManifestPath)
		if err != nil {
			return err
		}
		trainManifest = &manifest
		if tokenParam == "" {
			tokenParam = manifest.Embedding.TokenEmbeddingParam
		}
	}
	if tokenParam == "" {
		return fmt.Errorf("token embedding param is not available for tokenizer sync")
	}
	var weights *WeightFile
	if weightPath := DefaultWeightFilePath(artifactPath); fileExists(weightPath) {
		readWeights, err := ReadWeightFile(weightPath)
		if err != nil {
			return err
		}
		weights = &readWeights
		tensor, ok := readWeights.Weights[tokenParam]
		if !ok {
			return fmt.Errorf("weight file missing token embedding %q", tokenParam)
		}
		if rows := tensorLeadingRows(tensor); rows > vocabSize {
			return fmt.Errorf("tokenizer vocab size %d would shrink weight token embedding rows %d; tensor resize is not supported by tokenizer sync", vocabSize, rows)
		}
	}
	var checkpoint *EmbeddingTrainCheckpoint
	if checkpointPath := DefaultEmbeddingCheckpointPath(artifactPath); fileExists(checkpointPath) {
		readCheckpoint, err := ReadEmbeddingTrainCheckpointFile(checkpointPath)
		if err != nil {
			return err
		}
		checkpoint = &readCheckpoint
		if rows := tensorLeadingRows(readCheckpoint.TokenEmbedding); rows > vocabSize {
			return fmt.Errorf("tokenizer vocab size %d would shrink checkpoint token embedding rows %d; tensor resize is not supported by tokenizer sync", vocabSize, rows)
		}
	}
	if embeddingManifest != nil {
		embeddingManifest.Tokenizer.VocabSize = vocabSize
		if err := embeddingManifest.WriteFile(ResolveEmbeddingManifestPath(artifactPath)); err != nil {
			return err
		}
	}
	if trainManifest != nil {
		trainManifest.Embedding.Tokenizer.VocabSize = vocabSize
		if err := trainManifest.WriteFile(ResolveEmbeddingTrainManifestPath(artifactPath)); err != nil {
			return err
		}
	}
	if weights != nil {
		weightPath := DefaultWeightFilePath(artifactPath)
		tensor := weights.Weights[tokenParam]
		weights.Weights[tokenParam] = expandTensorLeadingRows(tensor, vocabSize, false)
		if err := weights.WriteFile(weightPath); err != nil {
			return err
		}
		if memoryPlanPath := DefaultMemoryPlanPath(artifactPath); fileExists(memoryPlanPath) {
			mod, err := eosartifact.ReadFile(artifactPath)
			if err != nil {
				return err
			}
			plan := NewMemoryPlan(mod, weights.Weights, MemoryPlanOptions{})
			if err := plan.WriteFile(memoryPlanPath); err != nil {
				return err
			}
		}
	}
	if checkpoint != nil {
		checkpointPath := DefaultEmbeddingCheckpointPath(artifactPath)
		checkpoint.Manifest.Tokenizer.VocabSize = vocabSize
		checkpoint.TokenEmbedding = expandTensorLeadingRows(checkpoint.TokenEmbedding, vocabSize, false)
		checkpoint.TokenMoment1 = expandTensorLeadingRows(checkpoint.TokenMoment1, vocabSize, true)
		checkpoint.TokenMoment2 = expandTensorLeadingRows(checkpoint.TokenMoment2, vocabSize, true)
		if err := checkpoint.WriteFile(checkpointPath); err != nil {
			return err
		}
	}
	if packageManifestPath := ResolvePackageManifestPath(artifactPath); fileExists(packageManifestPath) {
		if _, _, err := RebuildSiblingPackageManifest(artifactPath); err != nil {
			return err
		}
	}
	return nil
}

func tensorLeadingRows(t *backend.Tensor) int {
	if t == nil || len(t.Shape) == 0 {
		return 0
	}
	return t.Shape[0]
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func expandTensorLeadingRows(t *backend.Tensor, rows int, zeroFill bool) *backend.Tensor {
	if t == nil || len(t.Shape) == 0 || t.Shape[0] >= rows {
		return t
	}
	rowWidth := 1
	for _, dim := range t.Shape[1:] {
		if dim <= 0 {
			return t
		}
		rowWidth *= dim
	}
	newShape := append([]int(nil), t.Shape...)
	oldRows := newShape[0]
	newShape[0] = rows
	switch t.DType {
	case "f16", "f32", "q4", "q8":
		data := make([]float32, rows*rowWidth)
		copy(data, t.F32)
		if !zeroFill {
			rng := rand.New(rand.NewSource(int64(rows*rowWidth + oldRows)))
			scale := initializerScale(newShape)
			for i := oldRows * rowWidth; i < len(data); i++ {
				data[i] = (rng.Float32()*2 - 1) * scale
			}
		}
		switch t.DType {
		case "f16":
			return backend.NewTensorF16(newShape, data)
		case "f32":
			return backend.NewTensorF32(newShape, data)
		case "q4":
			return backend.NewTensorQ4(newShape, data)
		case "q8":
			return backend.NewTensorQ8(newShape, data)
		}
	case "i32":
		data := make([]int32, rows*rowWidth)
		copy(data, t.I32)
		return backend.NewTensorI32(newShape, data)
	case "i64":
		data := make([]int64, rows*rowWidth)
		copy(data, t.I64)
		return backend.NewTensorI64(newShape, data)
	}
	return t
}
