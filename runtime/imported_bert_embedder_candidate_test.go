package eosruntime

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backends/cuda"
)

func TestLoadImportedBERTEmbedderCandidateLoadsPackage(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "masked_mean", 4)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	packagePath = writeValidPretrainedBERTPackageWithModelName(t, packagePath, ImportedBERTEmbedderCandidateSourceModel)
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read fixture package: %v", err)
	}
	packageSHA, err := sha256FileHex(packagePath)
	if err != nil {
		t.Fatalf("hash fixture package: %v", err)
	}
	embedder, err := LoadVerifiedImportedBERTEmbedder(context.Background(), ImportedBERTEmbedderCandidateConfig{
		PackagePath:              packagePath,
		ExpectedSHA256:           packageSHA,
		ExpectedIdentitySHA256:   pkg.IdentityHash(),
		ExpectedPackageModelName: ImportedBERTEmbedderCandidateSourceModel,
		Runtime:                  New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("load verified imported BERT embedder: %v", err)
	}
	if embedder == nil {
		t.Fatal("embedder is nil")
	}
	if pkg.ModelName == ImportedBERTEmbedderCandidateModelName {
		t.Fatalf("fixture package should use source package model name, got public candidate name %q", pkg.ModelName)
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
	packagePath = writeValidPretrainedBERTPackageWithModelName(t, packagePath, ImportedBERTEmbedderCandidateSourceModel)
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read fixture package: %v", err)
	}
	packageSHA, err := sha256FileHex(packagePath)
	if err != nil {
		t.Fatalf("hash fixture package: %v", err)
	}
	_, err = LoadVerifiedImportedBERTEmbedder(context.Background(), ImportedBERTEmbedderCandidateConfig{
		PackagePath:              packagePath,
		ExpectedSHA256:           packageSHA,
		ExpectedIdentitySHA256:   pkg.IdentityHash(),
		ExpectedPackageModelName: "different/model",
		Runtime:                  New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "package model_name mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func writeValidPretrainedBERTPackageWithModelName(t *testing.T, sourcePath, modelName string) string {
	t.Helper()
	pkg, err := ReadPretrainedBERTPackageFile(sourcePath)
	if err != nil {
		t.Fatalf("read source package: %v", err)
	}
	pkg.ModelName = modelName
	pkg.Files = pretrainedBERTPackageFiles(pkg)
	pkg.IdentitySHA256 = pkg.IdentityHash()
	data, err := encodePretrainedBERTPackageMLL(pkg)
	if err != nil {
		t.Fatalf("encode package: %v", err)
	}
	path := filepath.Join(t.TempDir(), "valid-"+filepath.Base(sourcePath))
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write package: %v", err)
	}
	return path
}
