package eosruntime

import (
	"os"
	"path/filepath"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
)

func TestInitializeEmbeddingTrainerPackageWithManifestCreatesPackage(t *testing.T) {
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	paths, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       7,
		ShapeSizes: map[string]int{"D": 4, "E": 3},
	})
	if err != nil {
		t.Fatalf("initialize training package: %v", err)
	}
	for _, candidate := range []string{
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	trainer, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	if got := trainer.tokenEmbed.Shape; len(got) != 2 || got[0] != 8 || got[1] != 4 {
		t.Fatalf("token embedding shape = %v, want [8 4]", got)
	}
	if got := trainer.projection.Shape; len(got) != 2 || got[0] != 4 || got[1] != 3 {
		t.Fatalf("projection shape = %v, want [4 3]", got)
	}
}

func TestInitializeEmbeddingTrainerPackageRejectsUnresolvedShapes(t *testing.T) {
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer:           TokenizerManifest{VocabSize: 8},
	}
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{}); err == nil {
		t.Fatal("expected unresolved symbolic dim error")
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapCopiesOverlap(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyTrainInitArtifact(t, dir, "source_embed.mll")
	sourcePaths, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "E": 2},
	})
	if err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sourceCheckpoint, err := ReadEmbeddingTrainCheckpointFile(sourcePaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read source checkpoint: %v", err)
	}
	sourceCheckpoint.Step = 17
	sourceCheckpoint.TokenEmbedding = backend.NewTensorF32([]int{8, 2}, []float32{
		101, 102,
		103, 104,
		105, 106,
		107, 108,
		109, 110,
		111, 112,
		113, 114,
		115, 116,
	})
	sourceCheckpoint.Projection = backend.NewTensorF32([]int{2, 2}, []float32{
		201, 202,
		203, 204,
	})
	sourceCheckpoint.TokenMoment1 = backend.NewTensorF32([]int{8, 2}, filledFloat32(16, 9))
	sourceCheckpoint.TokenMoment2 = backend.NewTensorF32([]int{8, 2}, filledFloat32(16, 8))
	sourceCheckpoint.ProjMoment1 = backend.NewTensorF32([]int{2, 2}, filledFloat32(4, 7))
	sourceCheckpoint.ProjMoment2 = backend.NewTensorF32([]int{2, 2}, filledFloat32(4, 6))
	if err := sourceCheckpoint.WriteFile(sourcePaths.CheckpointPath); err != nil {
		t.Fatalf("rewrite source checkpoint: %v", err)
	}

	targetPath, targetManifest := buildTinyTrainInitArtifact(t, dir, "target_embed.mll")
	baselinePath, baselineManifest := buildTinyTrainInitArtifact(t, dir, "baseline_embed.mll")
	baselinePaths, err := InitializeEmbeddingTrainerPackageWithManifest(baselinePath, baselineManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       11,
		ShapeSizes: map[string]int{"D": 3, "E": 4},
	})
	if err != nil {
		t.Fatalf("initialize baseline package: %v", err)
	}
	targetPaths, err := InitializeEmbeddingTrainerPackageWithManifest(targetPath, targetManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:                  11,
		BootstrapArtifactPath: sourcePath,
		ShapeSizes:            map[string]int{"D": 3, "E": 4},
	})
	if err != nil {
		t.Fatalf("initialize target package with bootstrap: %v", err)
	}

	baseline, err := ReadEmbeddingTrainCheckpointFile(baselinePaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read baseline checkpoint: %v", err)
	}
	target, err := ReadEmbeddingTrainCheckpointFile(targetPaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read target checkpoint: %v", err)
	}
	if target.Step != 0 {
		t.Fatalf("target step = %d, want 0", target.Step)
	}
	assertFloat32SliceEqual(t, target.TokenEmbedding.Shape, []int{8, 3})
	assertFloat32SliceEqual(t, target.Projection.Shape, []int{3, 4})

	for row := 0; row < 8; row++ {
		for col := 0; col < 2; col++ {
			got := target.TokenEmbedding.F32[row*3+col]
			want := sourceCheckpoint.TokenEmbedding.F32[row*2+col]
			if got != want {
				t.Fatalf("token overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
		if got, want := target.TokenEmbedding.F32[row*3+2], baseline.TokenEmbedding.F32[row*3+2]; got != want {
			t.Fatalf("token non-overlap[%d,2] = %f, want initialized baseline %f", row, got, want)
		}
	}
	for row := 0; row < 2; row++ {
		for col := 0; col < 2; col++ {
			got := target.Projection.F32[row*4+col]
			want := sourceCheckpoint.Projection.F32[row*2+col]
			if got != want {
				t.Fatalf("projection overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
		for col := 2; col < 4; col++ {
			got := target.Projection.F32[row*4+col]
			want := baseline.Projection.F32[row*4+col]
			if got != want {
				t.Fatalf("projection non-overlap[%d,%d] = %f, want initialized baseline %f", row, col, got, want)
			}
		}
	}
	for col := 0; col < 4; col++ {
		got := target.Projection.F32[2*4+col]
		want := baseline.Projection.F32[2*4+col]
		if got != want {
			t.Fatalf("projection extra row[2,%d] = %f, want initialized baseline %f", col, got, want)
		}
	}
	for i, v := range target.TokenMoment1.F32 {
		if v != 0 {
			t.Fatalf("target token moment 1[%d] = %f, want zero", i, v)
		}
	}
	for i, v := range target.ProjMoment2.F32 {
		if v != 0 {
			t.Fatalf("target projection moment 2[%d] = %f, want zero", i, v)
		}
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapRejectsRankMismatch(t *testing.T) {
	target := backend.NewTensorF32([]int{2, 2}, []float32{1, 2, 3, 4})
	source := backend.NewTensorF32([]int{2, 1, 2}, []float32{1, 2, 3, 4})
	if err := copyOverlappingTensor(target, source); err == nil {
		t.Fatal("expected rank mismatch error")
	}
}

func buildTinyTrainInitArtifact(t *testing.T, dir, name string) (string, EmbeddingManifest) {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(dir, name)
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	return path, manifest
}

func filledFloat32(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func assertFloat32SliceEqual(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape rank = %d, want %d: got %v want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape = %v, want %v", got, want)
		}
	}
}
