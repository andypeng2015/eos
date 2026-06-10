package models

import (
	"os"
	"path/filepath"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	eosruntime "m31labs.dev/eos/runtime"
	mll "m31labs.dev/mll"
)

func TestInitDefaultEmbeddingPackageCreatesTrainablePackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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

func TestInitDefaultEmbeddingPackageQ4DeclaresQ4Params(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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
	if got := len(mod.Params); got != 7 {
		t.Fatalf("param count = %d, want 7", got)
	}
	for _, param := range mod.Params {
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

func TestInitDefaultEmbeddingPackageRejectsUnknownWeightDType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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

func TestDefaultEmbeddingPackageQ4TrainsContrastive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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
	if got := len(tnsrSection.Tensors); got != 7 {
		t.Fatalf("tensor count = %d, want 7", got)
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
		if scale, ok := meta.TensorScales[name]; !ok || scale <= 0 {
			t.Fatalf("tensor %q missing packed scale (scales=%v)", name, meta.TensorScales)
		}
	}
}

func TestInitDefaultEmbeddingPackageHonorsEncoderRepeats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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
