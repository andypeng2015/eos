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
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read fixture package: %v", err)
	}
	packageSHA, err := sha256FileHex(packagePath)
	if err != nil {
		t.Fatalf("hash fixture package: %v", err)
	}
	embedder, err := LoadVerifiedImportedBERTEmbedder(context.Background(), ImportedBERTEmbedderCandidateConfig{
		PackagePath:            packagePath,
		ExpectedSHA256:         packageSHA,
		ExpectedIdentitySHA256: pkg.IdentityHash(),
		ExpectedModelName:      pkg.ModelName,
		Runtime:                New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("load verified imported BERT embedder: %v", err)
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

func TestLoadImportedBERTEmbedderCandidateRejectsNonCandidateFixture(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "masked_mean", 4)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	_, err := LoadImportedBERTEmbedderCandidate(context.Background(), packagePath, New(cuda.New()))
	if err == nil || !strings.Contains(err.Error(), "package sha256 mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLoadVerifiedImportedBERTEmbedderRejectsIdentityMismatch(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "masked_mean", 4)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	packageSHA, err := sha256FileHex(packagePath)
	if err != nil {
		t.Fatalf("hash fixture package: %v", err)
	}
	_, err = LoadVerifiedImportedBERTEmbedder(context.Background(), ImportedBERTEmbedderCandidateConfig{
		PackagePath:            packagePath,
		ExpectedSHA256:         packageSHA,
		ExpectedIdentitySHA256: strings.Repeat("0", 64),
		Runtime:                New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "package identity mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLoadVerifiedImportedBERTEmbedderRejectsModelMismatch(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "masked_mean", 4)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read fixture package: %v", err)
	}
	packageSHA, err := sha256FileHex(packagePath)
	if err != nil {
		t.Fatalf("hash fixture package: %v", err)
	}
	_, err = LoadVerifiedImportedBERTEmbedder(context.Background(), ImportedBERTEmbedderCandidateConfig{
		PackagePath:            packagePath,
		ExpectedSHA256:         packageSHA,
		ExpectedIdentitySHA256: pkg.IdentityHash(),
		ExpectedModelName:      "different/model",
		Runtime:                New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "package model_name mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}
