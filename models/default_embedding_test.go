package models

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backends/vulkan"
	mll "m31labs.dev/mll"
)

func TestInitDefaultEmbeddingPackageCreatesTrainablePackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	for _, candidate := range []string{
		paths.ArtifactPath,
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
		paths.PackageManifestPath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.Name != DefaultEmbeddingModelName {
		t.Fatalf("manifest name = %q, want %q", manifest.Name, DefaultEmbeddingModelName)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.AttentionMaskMode != eosruntime.EmbeddingAttentionMaskModeKey {
		t.Fatalf("attention mask mode = %q, want %q", manifest.AttentionMaskMode, eosruntime.EmbeddingAttentionMaskModeKey)
	}
	if manifest.AttentionScoreScale != eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ {
		t.Fatalf("attention score scale = %q, want %q", manifest.AttentionScoreScale, eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ)
	}
	if manifest.PositionEncoding != eosruntime.EmbeddingPositionEncodingRoPE {
		t.Fatalf("position encoding = %q, want %q", manifest.PositionEncoding, eosruntime.EmbeddingPositionEncodingRoPE)
	}
	if manifest.ArchitectureVersion != eosruntime.EmbeddingArchitectureLegacyV1 ||
		manifest.ModelDim != 4 ||
		manifest.OutputDim != 4 ||
		manifest.AttentionHeads != 1 ||
		manifest.HeadDim != 4 ||
		manifest.FFNDim != 8 ||
		manifest.ParameterTying != eosruntime.EmbeddingParameterTyingLegacyTied {
		t.Fatalf("unexpected architecture metadata: %+v", manifest)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	if manifest.Tokenizer.PadID != 0 || manifest.Tokenizer.BOSID != 1 || manifest.Tokenizer.EOSID != 2 || manifest.Tokenizer.UnknownID != 3 {
		t.Fatalf("unexpected tokenizer ids: %+v", manifest.Tokenizer)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature != 0.05 {
		t.Fatalf("temperature = %f, want 0.05", checkpoint.Config.Temperature)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("load training package: %v", err)
	}
}

func TestInitDefaultEmbeddingPackageDefaultsOutputDimToModelDim(t *testing.T) {
	manifest := DefaultEmbeddingManifest(DefaultEmbeddingPackageConfig{
		ModelDim:  6,
		HiddenDim: 12,
	})
	if manifest.ModelDim != 6 || manifest.OutputDim != 6 {
		t.Fatalf("manifest dims = model:%d output:%d, want 6/6", manifest.ModelDim, manifest.OutputDim)
	}
}

func TestInitDefaultEmbeddingPackageRejectsUnsupportedCompactArchitecture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Architecture:   eosruntime.EmbeddingArchitectureCompactTransformerV1,
		ModelDim:       8,
		OutputDim:      4,
		HiddenDim:      16,
		AttentionHeads: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "compact_transformer_v1 is not supported by trainable package initialization yet") {
		t.Fatalf("InitDefaultEmbeddingPackage error = %v, want unsupported compact error", err)
	}
}

func TestInitDefaultEmbeddingPackageRejectsInvalidHeadConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad-heads.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Architecture:   eosruntime.EmbeddingArchitectureCompactTransformerV1,
		ModelDim:       7,
		OutputDim:      7,
		HiddenDim:      14,
		AttentionHeads: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "model_dim 7 must be divisible by attention_heads 2") {
		t.Fatalf("InitDefaultEmbeddingPackage error = %v, want head divisibility error", err)
	}
}

func TestInitDefaultEmbeddingPackageQ4DeclaresQ4Params(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
		WeightDType:  "q4",
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	wantParams := map[string]bool{
		"token_embedding": true,
		"role_embedding":  true,
		"attn_q":          true,
		"attn_k":          true,
		"attn_v":          true,
		"attn_o":          true,
		"ffn_up":          true,
		"projection":      true,
	}
	if got := len(mod.Params); got != len(wantParams) {
		t.Fatalf("param count = %d, want %d", got, len(wantParams))
	}
	for _, param := range mod.Params {
		if !wantParams[param.Name] {
			t.Fatalf("unexpected param %q", param.Name)
		}
		if param.Type.Tensor == nil || param.Type.Tensor.DType != "q4" {
			t.Fatalf("param %q dtype = %+v, want q4 tensor", param.Name, param.Type)
		}
		if !param.Trainable {
			t.Fatalf("param %q is not trainable", param.Name)
		}
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.WeightBits != 4 {
		t.Fatalf("weight bits = %d, want 4", checkpoint.Config.WeightBits)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("load training package: %v", err)
	}
}

func TestDefaultEmbeddingPackageV2PackagedInferenceParity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v2-generated.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Name:         "eos-embed-v2-generated",
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}

	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.AttentionMaskMode != eosruntime.EmbeddingAttentionMaskModeKey {
		t.Fatalf("attention mask mode = %q, want %q", manifest.AttentionMaskMode, eosruntime.EmbeddingAttentionMaskModeKey)
	}
	if manifest.AttentionScoreScale != eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ {
		t.Fatalf("attention score scale = %q, want %q", manifest.AttentionScoreScale, eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ)
	}
	if manifest.PositionEncoding != eosruntime.EmbeddingPositionEncodingRoPE {
		t.Fatalf("position encoding = %q, want %q", manifest.PositionEncoding, eosruntime.EmbeddingPositionEncodingRoPE)
	}

	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if !moduleHasKernelOpForTest(mod, "masked_softmax") {
		t.Fatal("generated v2 artifact is missing masked_softmax")
	}
	if !moduleHasKernelOpForTest(mod, "rope") {
		t.Fatal("generated v2 artifact is missing rope")
	}
	if moduleRequiresCapabilityForTest(mod, eosartifact.CapabilityDeviceExecution) {
		t.Fatalf("generated v2 artifact unexpectedly declares device-native serving capability: %v", mod.Requirements.Capabilities)
	}

	rt := eosruntime.New(vulkan.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		t.Fatalf("load generated v2 embedding package: %v", err)
	}
	if got := model.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}

	base, err := model.Embed(context.Background(), []int32{4, 5})
	if err != nil {
		t.Fatalf("embed base: %v", err)
	}
	padded, err := model.Embed(context.Background(), []int32{4, 5, 0, 0})
	if err != nil {
		t.Fatalf("embed padded: %v", err)
	}
	assertEmbeddingCloseForTest(t, "right-padding", padded.Embeddings.F32, base.Embeddings.F32, 1e-5)

	ordered, err := model.Embed(context.Background(), []int32{4, 5, 6})
	if err != nil {
		t.Fatalf("embed ordered: %v", err)
	}
	reordered, err := model.Embed(context.Background(), []int32{6, 5, 4})
	if err != nil {
		t.Fatalf("embed reordered: %v", err)
	}
	if maxAbs, l2 := embeddingDiffStatsForTest(ordered.Embeddings.F32, reordered.Embeddings.F32); maxAbs <= 1e-6 && l2 <= 1e-6 {
		t.Fatalf("v2 packaged embed is not order-sensitive: max_abs=%.9g l2=%.9g ordered=%v reordered=%v", maxAbs, l2, ordered.Embeddings.F32, reordered.Embeddings.F32)
	}

	batches := [][]int32{
		{4, 5, 0, 0},
		{6, 7, 8, 0},
		{9, 10, 11, 12},
	}
	batchResult, err := model.EmbedBatch(context.Background(), batches)
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	if got, want := batchResult.Embeddings.Shape, []int{len(batches), len(base.Embeddings.F32)}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("batch embedding shape = %v, want %v", got, want)
	}
	rowWidth := batchResult.Embeddings.Shape[1]
	for i, tokens := range batches {
		perExample, err := model.Embed(context.Background(), tokens)
		if err != nil {
			t.Fatalf("embed batch item %d: %v", i, err)
		}
		row := batchResult.Embeddings.F32[i*rowWidth : (i+1)*rowWidth]
		assertEmbeddingCloseForTest(t, "batch row", row, perExample.Embeddings.F32, 1e-5)
	}
}

func TestInitDefaultEmbeddingPackageRejectsUnknownWeightDType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if _, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		WeightDType:  "int4",
	}); err == nil {
		t.Fatal("expected weight dtype error")
	}
}

func moduleHasKernelOpForTest(mod *eosartifact.Module, op string) bool {
	if mod == nil {
		return false
	}
	for _, kernel := range mod.Kernels {
		for _, bodyOp := range kernel.Body {
			if bodyOp.Op == op {
				return true
			}
		}
	}
	return false
}

func moduleRequiresCapabilityForTest(mod *eosartifact.Module, capability string) bool {
	if mod == nil {
		return false
	}
	for _, got := range mod.Requirements.Capabilities {
		if got == capability {
			return true
		}
	}
	return false
}

func assertEmbeddingCloseForTest(t *testing.T, label string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len = %d, want %d", label, len(got), len(want))
	}
	if maxAbs, l2 := embeddingDiffStatsForTest(got, want); maxAbs > tolerance || l2 > tolerance {
		t.Fatalf("%s mismatch: max_abs=%.9g l2=%.9g got=%v want=%v", label, maxAbs, l2, got, want)
	}
}

func embeddingDiffStatsForTest(a, b []float32) (float32, float32) {
	maxAbs := float32(0)
	sumSquares := float64(0)
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSquares += float64(diff * diff)
	}
	return maxAbs, float32(math.Sqrt(sumSquares))
}

func TestDefaultEmbeddingPackageQ4TrainsContrastive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if _, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
		WeightDType:  "q4",
	}); err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	trainer, err := eosruntime.LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.Config.WeightBits != 4 {
		t.Fatalf("trainer weight bits = %d, want 4", checkpoint.Config.WeightBits)
	}
	trainSet := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 4}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{6, 7}, PositiveTokens: []int32{6, 6}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, eosruntime.EmbeddingTrainRunConfig{
		Epochs:    2,
		BatchSize: 2,
		Seed:      7,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
}

func TestExportDefaultEmbeddingPackageQ4SealsPackedTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if _, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
		WeightDType:  "q4",
	}); err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}

	outPath, err := eosruntime.ExportPackageToMLL(path, "")
	if err != nil {
		t.Fatalf("ExportPackageToMLL: %v", err)
	}
	reader, err := mll.ReadFile(outPath, mll.WithDigestVerification())
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if reader.Profile() != mll.ProfileSealed {
		t.Fatalf("profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}

	xmtaBody, ok := reader.Section(eosartifact.MLLTagXMTA)
	if !ok {
		t.Fatal("missing XMTA section")
	}
	meta, err := eosartifact.DecodeMLLMetadata(xmtaBody)
	if err != nil {
		t.Fatalf("decode XMTA: %v", err)
	}
	if got := meta.LogicalTensorDType["token_embedding"]; got != "q4" {
		t.Fatalf("logical dtype for token_embedding = %q, want q4", got)
	}

	strgBody, _ := reader.Section(mll.TagSTRG)
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		t.Fatalf("ReadStringTable: %v", err)
	}
	tnsrBody, _ := reader.Section(mll.TagTNSR)
	tnsrSection, err := mll.ReadTnsrSection(tnsrBody)
	if err != nil {
		t.Fatalf("ReadTnsrSection: %v", err)
	}
	if got := len(tnsrSection.Tensors); got != 8 {
		t.Fatalf("tensor count = %d, want 8", got)
	}

	// q4 tensors are sealed as real packed payloads: storage dtype Q4, two
	// offset-binary nibbles per byte, and a per-tensor scale in XMTA.
	for _, entry := range tnsrSection.Tensors {
		name := strg.At(entry.NameIdx)
		if meta.LogicalTensorDType[name] != "q4" {
			t.Fatalf("tensor %q logical dtype = %q, want q4", name, meta.LogicalTensorDType[name])
		}
		if entry.DType != mll.DTypeQ4 {
			t.Fatalf("tensor %q stored dtype = %d, want packed q4 (%d)", name, entry.DType, mll.DTypeQ4)
		}
		elements := uint64(1)
		for _, dim := range entry.Shape {
			elements *= dim
		}
		if want := (elements + 1) / 2; uint64(len(entry.Data)) != want {
			t.Fatalf("tensor %q packed bytes = %d, want %d", name, len(entry.Data), want)
		}
		scale, ok := meta.TensorScales[name]
		if !ok {
			t.Fatalf("tensor %q missing packed scale (scales=%v)", name, meta.TensorScales)
		}
		if name == "role_embedding" {
			if scale != 0 {
				t.Fatalf("tensor %q scale = %g, want zero for zero-initialized role embeddings", name, scale)
			}
		} else if scale <= 0 {
			t.Fatalf("tensor %q scale = %g, want positive", name, scale)
		}
	}
}

func TestInitDefaultEmbeddingPackageHonorsEncoderRepeats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:      16,
		MaxSequence:    8,
		EmbeddingDim:   4,
		HiddenDim:      8,
		EncoderRepeats: 3,
		Seed:           7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.EncoderRepeats != 3 {
		t.Fatalf("encoder repeats = %d, want 3", manifest.EncoderRepeats)
	}
}

func TestDefaultEmbedderAssetInfoAndVerify(t *testing.T) {
	info, err := DefaultEmbedderAssetInfo("")
	if err != nil {
		t.Fatalf("default embedder asset info: %v", err)
	}
	if info.AssetID != DefaultEmbedderAssetID {
		t.Fatalf("asset id = %q, want %q", info.AssetID, DefaultEmbedderAssetID)
	}
	if info.ModelName != DefaultEmbeddingModelName {
		t.Fatalf("model name = %q, want %q", info.ModelName, DefaultEmbeddingModelName)
	}
	if filepath.Base(info.ArtifactPath) != DefaultEmbedderArtifactFilename {
		t.Fatalf("artifact path = %q", info.ArtifactPath)
	}
	if filepath.Base(info.TokenizerPath) != DefaultEmbedderTokenizerFilename {
		t.Fatalf("tokenizer path = %q", info.TokenizerPath)
	}
	report, err := VerifyDefaultEmbedderAsset("")
	if err != nil {
		t.Fatalf("verify default embedder asset: %v", err)
	}
	if !report.OK || len(report.Files) != 2 {
		t.Fatalf("unexpected verification report: %+v", report)
	}
	for _, check := range report.Files {
		if !check.OK || check.SHA256 != check.ExpectedSHA256 || check.Bytes <= 0 {
			t.Fatalf("bad file check: %+v", check)
		}
	}
}
