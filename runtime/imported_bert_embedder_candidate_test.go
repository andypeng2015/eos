package eosruntime

import (
	"context"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backends/cuda"
)

func TestLoadImportedBERTEmbedderCandidateLoadsPackage(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "masked_mean", 4)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	embedder, err := LoadImportedBERTEmbedderCandidate(context.Background(), packagePath, New(cuda.New()))
	if err != nil {
		t.Fatalf("load imported BERT embedder candidate: %v", err)
	}
	if embedder == nil {
		t.Fatal("embedder is nil")
	}
}

func TestLoadImportedBERTEmbedderCandidateRequiresPackagePath(t *testing.T) {
	_, err := LoadImportedBERTEmbedderCandidate(context.Background(), "", New())
	if err == nil || !strings.Contains(err.Error(), "package path is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}
