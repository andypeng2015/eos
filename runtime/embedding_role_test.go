package eosruntime

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestEmbeddingManifestRoleConditioningRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "role.embedding.mll")
	manifest := tinyRoleEmbeddingManifest()
	if err := manifest.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	got = got.normalized()
	if got.RoleConditioning != EmbeddingRoleConditioningAdditiveV1 || got.RoleEmbeddingParam != "role_embedding" || got.RoleInput != "role_ids" || got.BatchRoleInput != "role_ids" {
		t.Fatalf("role manifest fields not preserved: %+v", got)
	}
	if got.RawRoleIndex != 0 || got.QueryRoleIndex != 1 || got.DocumentRoleIndex != 2 {
		t.Fatalf("role indexes = raw:%d query:%d document:%d", got.RawRoleIndex, got.QueryRoleIndex, got.DocumentRoleIndex)
	}
}

func TestEmbeddingModelRoleConditioningEffectAndLegacyRejection(t *testing.T) {
	bundle, err := compiler.Build([]byte(tinyRoleEmbeddingSource()), compiler.Options{ModuleName: "role_embed"})
	if err != nil {
		t.Fatalf("build role source: %v", err)
	}
	rt := New(metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyRoleEmbeddingManifest(),
		WithWeight("token_embedding", backend.NewTensorF16([]int{4, 2}, make([]float32, 8))),
		WithWeight("role_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			0, 0,
			1, 0,
			0, 2,
		})),
		WithWeight("projection", backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1})),
	)
	if err != nil {
		t.Fatalf("load role model: %v", err)
	}
	raw, err := model.EmbedWithRole(context.Background(), []int32{1, 2}, EmbeddingRoleRaw)
	if err != nil {
		t.Fatalf("embed raw: %v", err)
	}
	query, err := model.EmbedWithRole(context.Background(), []int32{1, 2}, EmbeddingRoleQuery)
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if raw.Embeddings.F32[0] != 0 || raw.Embeddings.F32[1] != 0 {
		t.Fatalf("raw role output = %v, want zero parity", raw.Embeddings.F32)
	}
	if query.Embeddings.F32[0] <= raw.Embeddings.F32[0] {
		t.Fatalf("query role did not affect embedding: raw=%v query=%v", raw.Embeddings.F32, query.Embeddings.F32)
	}

	legacy, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyRoleEmbeddingManifest().legacyWithoutRole(),
		WithWeight("token_embedding", backend.NewTensorF16([]int{4, 2}, make([]float32, 8))),
		WithWeight("role_embedding", backend.NewTensorF16([]int{3, 2}, make([]float32, 6))),
		WithWeight("projection", backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1})),
	)
	if err != nil {
		t.Fatalf("load legacy-shaped model: %v", err)
	}
	if _, err := legacy.EmbedWithRole(context.Background(), []int32{1, 2}, EmbeddingRoleQuery); err == nil || !strings.Contains(err.Error(), "query role") {
		t.Fatalf("legacy query role error = %v", err)
	}
}

func TestEmbeddingModelMasklessRoleConditioning(t *testing.T) {
	bundle, err := compiler.Build([]byte(tinyMasklessRoleEmbeddingSource()), compiler.Options{ModuleName: "role_embed_maskless"})
	if err != nil {
		t.Fatalf("build maskless role source: %v", err)
	}
	manifest := tinyRoleEmbeddingManifest()
	manifest.Name = "role_embed_maskless"
	manifest.MaskInput = ""
	rt := New(metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, manifest,
		WithWeight("token_embedding", backend.NewTensorF16([]int{4, 2}, make([]float32, 8))),
		WithWeight("role_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			0, 0,
			1, 0,
			0, 2,
		})),
		WithWeight("projection", backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1})),
	)
	if err != nil {
		t.Fatalf("load maskless role model: %v", err)
	}
	query, err := model.EmbedWithRole(context.Background(), []int32{1, 2}, EmbeddingRoleQuery)
	if err != nil {
		t.Fatalf("embed maskless query role: %v", err)
	}
	if query.Embeddings.F32[0] <= 0 || query.Embeddings.F32[1] != 0 {
		t.Fatalf("maskless query role output = %v, want query role contribution", query.Embeddings.F32)
	}
	docs, err := model.EmbedBatchWithRole(context.Background(), [][]int32{{1, 2}, {2, 3}}, EmbeddingRoleDocument)
	if err != nil {
		t.Fatalf("embed maskless document batch role: %v", err)
	}
	if len(docs.Embeddings.F32) != 4 || docs.Embeddings.F32[0] != 0 || docs.Embeddings.F32[1] <= 0 || docs.Embeddings.F32[2] != 0 || docs.Embeddings.F32[3] <= 0 {
		t.Fatalf("maskless document role batch output = %v, want document role contribution in both rows", docs.Embeddings.F32)
	}
}

func TestEmbeddingTrainerRoleCheckpointRoundTrip(t *testing.T) {
	bundle, err := compiler.Build([]byte(tinyRoleEmbeddingSource()), compiler.Options{ModuleName: "role_train"})
	if err != nil {
		t.Fatalf("build role source: %v", err)
	}
	manifest := tinyRoleEmbeddingManifest()
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{4, 2}, make([]float32, 8)),
		"role_embedding":  backend.NewTensorF16([]int{3, 2}, []float32{0, 0, 0.25, 0.5, 0.75, 1}),
		"projection":      backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1}),
	}, EmbeddingTrainConfig{LearningRate: 0.01})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.RoleEmbedding == nil || checkpoint.RoleMoment1 == nil || checkpoint.RoleMoment2 == nil {
		t.Fatalf("checkpoint missing role tensors")
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(bundle.Artifact, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}
	if restored.roleEmbed == nil || len(restored.roleEmbed.F32) != len(trainer.roleEmbed.F32) {
		t.Fatalf("restored role tensor = %+v", restored.roleEmbed)
	}
}

func TestEmbeddingTrainerContrastiveRoleGradientsAreNotDoubleCounted(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newTinyRoleEmbeddingTrainer(t, 0.1)
	before := append([]float32(nil), trainer.roleEmbed.F32...)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train contrastive role step: %v", err)
	}
	queryDelta := roleRowDeltaNorm(before, trainer.roleEmbed.F32, int(trainer.queryRoleIndex()), trainer.roleEmbed.Shape[1])
	docDelta := roleRowDeltaNorm(before, trainer.roleEmbed.F32, int(trainer.documentRoleIndex()), trainer.roleEmbed.Shape[1])
	if queryDelta <= 0 || docDelta <= 0 {
		t.Fatalf("role row movement query=%g document=%g, want both nonzero", queryDelta, docDelta)
	}
	ratio := queryDelta / docDelta
	if ratio < 0.8 || ratio > 1.25 {
		t.Fatalf("query/document role movement ratio = %g (query=%g document=%g), want near 1 without duplicate query accumulation", ratio, queryDelta, docDelta)
	}
}

func TestEmbeddingTrainerListwiseGeometryUpdatesQueryAndDocumentRoles(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newTinyRoleEmbeddingTrainer(t, 0.1)
	trainer.config.Temperature = 0.2
	before := append([]float32(nil), trainer.roleEmbed.F32...)
	if _, err := trainer.TrainListwiseGeometryStepWithDiagnostics(tinyTokenizedListwiseGeometryBatches(false), false); err != nil {
		t.Fatalf("train listwise role step: %v", err)
	}
	queryDelta := roleRowDeltaNorm(before, trainer.roleEmbed.F32, int(trainer.queryRoleIndex()), trainer.roleEmbed.Shape[1])
	docDelta := roleRowDeltaNorm(before, trainer.roleEmbed.F32, int(trainer.documentRoleIndex()), trainer.roleEmbed.Shape[1])
	if queryDelta <= 0 || docDelta <= 0 {
		t.Fatalf("role row movement query=%g document=%g, want both nonzero", queryDelta, docDelta)
	}
}

func TestListwiseGeometrySequenceInputsAssignRoles(t *testing.T) {
	batch := EmbeddingTokenizedListwiseGeometryBatch{
		QueryTokens:    [][]int32{{1, 2}},
		DocumentTokens: [][]int32{{2, 3}},
		TeacherSimilarity: [][]float32{
			{1},
		},
	}
	queries, docs, _, err := listwiseGeometrySequenceInputs([]EmbeddingTokenizedListwiseGeometryBatch{batch}, 7, 9)
	if err != nil {
		t.Fatalf("listwise inputs: %v", err)
	}
	if len(queries) != 1 || queries[0].role != 7 || len(docs) != 1 || docs[0].role != 9 {
		t.Fatalf("roles queries=%+v docs=%+v", queries, docs)
	}
}

func TestScoreSpectrumSequenceInputsAssignRoles(t *testing.T) {
	example := EmbeddingScoreSpectrumExample{
		QueryTokens:     []int32{1, 2},
		QueryMask:       []int32{1, 1},
		CandidateTokens: [][]int32{{2, 3}, {1, 3}},
		CandidateMasks:  [][]int32{{1, 1}, {1, 1}},
		PositiveIndexes: []int{0},
		HardNegativeEligible: []bool{
			false,
			true,
		},
		TargetProbabilities: []float32{1, 0},
	}
	queries, candidates, _, err := scoreSpectrumSequenceInputs([]EmbeddingScoreSpectrumExample{example}, 5, 6)
	if err != nil {
		t.Fatalf("score-spectrum inputs: %v", err)
	}
	if len(queries) != 1 || queries[0].role != 5 || len(candidates) != 2 || candidates[0].role != 6 || candidates[1].role != 6 {
		t.Fatalf("roles queries=%+v candidates=%+v", queries, candidates)
	}
}

func TestEmbeddingRetrievalRoleModeResolver(t *testing.T) {
	legacy := &EmbeddingModel{manifest: tinyRoleEmbeddingManifest().legacyWithoutRole()}
	if _, _, _, err := resolveEmbeddingRetrievalRoles(legacy, EmbeddingRoleModeQueryDocument); err == nil {
		t.Fatal("query-document role mode succeeded for legacy model")
	}
	roleModel := &EmbeddingModel{manifest: tinyRoleEmbeddingManifest()}
	docRole, queryRole, mode, err := resolveEmbeddingRetrievalRoles(roleModel, EmbeddingRoleModeAuto)
	if err != nil {
		t.Fatalf("resolve auto: %v", err)
	}
	if docRole != EmbeddingRoleDocument || queryRole != EmbeddingRoleQuery || mode != EmbeddingRoleModeQueryDocument {
		t.Fatalf("resolved roles doc=%q query=%q mode=%q", docRole, queryRole, mode)
	}
}

func newTinyRoleEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	bundle, err := compiler.Build([]byte(tinyRoleEmbeddingSource()), compiler.Options{ModuleName: "tiny_role_train_embed"})
	if err != nil {
		t.Fatalf("build role trainer source: %v", err)
	}
	manifest := tinyRoleEmbeddingManifest()
	manifest.Name = "tiny_role_train_embed"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{4, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
			0.5, 0.25,
		}),
		"role_embedding": backend.NewTensorF32([]int{3, 2}, make([]float32, 6)),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new role trainer: %v", err)
	}
	return trainer
}

func roleRowDeltaNorm(before, after []float32, row, width int) float32 {
	if width <= 0 {
		return 0
	}
	sum := float64(0)
	base := row * width
	for i := 0; i < width && base+i < len(before) && base+i < len(after); i++ {
		delta := after[base+i] - before[base+i]
		sum += float64(delta * delta)
	}
	return float32(math.Sqrt(sum))
}

func tinyRoleEmbeddingManifest() EmbeddingManifest {
	return EmbeddingManifest{
		Name:                "role_embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		MaskInput:           "attention_mask",
		OutputDType:         "f16",
		TokenEmbeddingParam: "token_embedding",
		RoleConditioning:    EmbeddingRoleConditioningAdditiveV1,
		RoleEmbeddingParam:  "role_embedding",
		RoleInput:           "role_ids",
		BatchRoleInput:      "role_ids",
		RawRoleIndex:        0,
		QueryRoleIndex:      1,
		DocumentRoleIndex:   2,
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   4,
			MaxSequence: 4,
		},
	}
}

func (m EmbeddingManifest) legacyWithoutRole() EmbeddingManifest {
	m.RoleConditioning = EmbeddingRoleConditioningNone
	m.RoleEmbeddingParam = ""
	m.RoleInput = ""
	m.BatchRoleInput = ""
	return m
}

func tinyRoleEmbeddingSource() string {
	return `
param token_embedding: f16[V, D] @weight("weights/token_embedding") @trainable
param role_embedding: f16[3, D] @weight("weights/role_embedding") @trainable
param projection: f16[D, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T], role_ids: i32[T]) -> f16[D] {
    let token_hidden = gather(token_embedding, tokens)
    let role_hidden = gather(role_embedding, role_ids)
    let hidden = token_hidden + role_hidden
    let projected = @matmul(hidden, projection)
    return mean_pool(projected, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T], role_ids: i32[B, T]) -> f16[B, D] {
    let token_hidden = gather(token_embedding, tokens)
    let role_hidden = gather(role_embedding, role_ids)
    let hidden = token_hidden + role_hidden
    let projected = @matmul(hidden, projection)
    return mean_pool(projected, attention_mask)
}
`
}

func tinyMasklessRoleEmbeddingSource() string {
	return `
param token_embedding: f16[V, D] @weight("weights/token_embedding") @trainable
param role_embedding: f16[3, D] @weight("weights/role_embedding") @trainable
param projection: f16[D, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], role_ids: i32[T]) -> f16[D] {
    let token_hidden = gather(token_embedding, tokens)
    let role_hidden = gather(role_embedding, role_ids)
    let hidden = token_hidden + role_hidden
    let projected = @matmul(hidden, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T], role_ids: i32[B, T]) -> f16[B, D] {
    let token_hidden = gather(token_embedding, tokens)
    let role_hidden = gather(role_embedding, role_ids)
    let hidden = token_hidden + role_hidden
    let projected = @matmul(hidden, projection)
    return mean_pool(projected)
}
`
}
