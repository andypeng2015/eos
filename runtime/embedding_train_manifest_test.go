package eosruntime

import (
	"path/filepath"
	"testing"

	"m31labs.dev/eos/compiler"
)

func TestDefaultEmbeddingTrainManifestPath(t *testing.T) {
	got := DefaultEmbeddingTrainManifestPath("/tmp/tiny_train_embed_q8.mll")
	if want := "/tmp/tiny_train_embed_q8.train.mll"; got != want {
		t.Fatalf("train manifest path = %q, want %q", got, want)
	}
}

func TestEmbeddingTrainManifestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.train.mll")
	want := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config: EmbeddingTrainConfig{
			LearningRate:              0.05,
			WeightBits:                8,
			Optimizer:                 "adamw",
			Beta1:                     0.9,
			Beta2:                     0.999,
			Epsilon:                   1e-8,
			ContrastiveLoss:           "infonce",
			Temperature:               0.05,
			MatryoshkaDims:            []int{64, 128},
			MatryoshkaWeights:         []float32{0.5, 1},
			TurboQuantPrefixBits:      []int{2, 4},
			TurboQuantPrefixWeight:    0.25,
			TurboQuantPrefixSeed:      DefaultTurboQuantMultiVectorQuantizerSeed,
			TurboQuantPrefixScoreMode: TurboQuantPrefixScoreModePreparedIP,
		},
	}
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if got.Name != want.Name || got.Embedding.Name != want.Embedding.Name || got.Config.Optimizer != want.Config.Optimizer || got.Config.ContrastiveLoss != want.Config.ContrastiveLoss || got.Config.Temperature != want.Config.Temperature {
		t.Fatalf("manifest mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
	if len(got.Config.TurboQuantPrefixBits) != 2 || got.Config.TurboQuantPrefixBits[0] != 2 || got.Config.TurboQuantPrefixBits[1] != 4 {
		t.Fatalf("turboquant prefix bits = %v, want [2 4]", got.Config.TurboQuantPrefixBits)
	}
	if got.Config.TurboQuantPrefixWeight != 0.25 {
		t.Fatalf("turboquant prefix weight = %f, want 0.25", got.Config.TurboQuantPrefixWeight)
	}
	if got.Config.TurboQuantPrefixSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("turboquant prefix seed = %d, want %d", got.Config.TurboQuantPrefixSeed, DefaultTurboQuantMultiVectorQuantizerSeed)
	}
	if got.Config.TurboQuantPrefixScoreMode != TurboQuantPrefixScoreModePreparedIP {
		t.Fatalf("turboquant prefix score mode = %q, want %q", got.Config.TurboQuantPrefixScoreMode, TurboQuantPrefixScoreModePreparedIP)
	}
}

func TestEmbeddingTrainManifestRoundTripTurboQuantPrefixObjectives(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.train.mll")
	want := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config: EmbeddingTrainConfig{
			LearningRate:      0.05,
			WeightBits:        8,
			Optimizer:         "adamw",
			Beta1:             0.9,
			Beta2:             0.999,
			Epsilon:           1e-8,
			ContrastiveLoss:   "infonce",
			Temperature:       0.05,
			MatryoshkaDims:    []int{64, 128},
			MatryoshkaWeights: []float32{0.5, 1},
			TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
				{Dim: 128, BitWidth: 4, Weight: 0.5},
				{Dim: 64, BitWidth: 2, Weight: 0},
			},
			TurboQuantPrefixSeed:      DefaultTurboQuantMultiVectorQuantizerSeed,
			TurboQuantPrefixScoreMode: TurboQuantPrefixScoreModePreparedIP,
		},
	}
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if formatted := FormatTurboQuantPrefixObjectives(got.Config.TurboQuantPrefixObjectives); formatted != "64:2=0,128:4=0.5" {
		t.Fatalf("turboquant prefix objectives = %q", formatted)
	}
}

func TestEmbeddingTrainManifestValidateModule(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config:    EmbeddingTrainConfig{LearningRate: 0.05},
	}
	if err := manifest.ValidateModule(bundle.Artifact); err != nil {
		t.Fatalf("validate train manifest: %v", err)
	}
}
