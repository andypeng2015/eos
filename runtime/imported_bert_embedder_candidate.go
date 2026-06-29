package eosruntime

import (
	"context"
	"fmt"
)

const (
	ImportedBERTEmbedderCandidateModelName               = "eos-embed-v1"
	ImportedBERTEmbedderCandidateSourceModel             = "BAAI/bge-small-en-v1.5"
	ImportedBERTEmbedderCandidatePackageSHA256           = "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763"
	ImportedBERTEmbedderCandidatePackageIdentitySHA256   = "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637"
	ImportedBERTEmbedderCandidatePackageRelativePathHint = "runs/pretrained-bert-current-hf-parity-v1-20260629T090818Z/bge/bge-small-en-v1.5.imported.mll"
	ImportedBERTEmbedderCandidateSourceSnapshotCommit    = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
	ImportedBERTEmbedderCandidateUpstreamModelURL        = "https://huggingface.co/BAAI/bge-small-en-v1.5"
	ImportedBERTEmbedderCandidateLicenseID               = "MIT"
	ImportedBERTEmbedderCandidateAttribution             = "FlagEmbedding/BAAI"
	ImportedBERTEmbedderCandidatePooling                 = "cls"
	ImportedBERTEmbedderCandidateNormalization           = "l2"
	ImportedBERTEmbedderCandidateMaxLength               = 512
	ImportedBERTEmbedderCandidateNativeDim               = 384
)

type ImportedBERTEmbedderCandidateConfig struct {
	PackagePath              string
	ExpectedSHA256           string
	ExpectedIdentitySHA256   string
	ExpectedPackageModelName string
	Runtime                  *Runtime
}

func LoadImportedBERTEmbedderCandidate(ctx context.Context, packagePath string, rt *Runtime) (*PretrainedBERTTextEmbedder, error) {
	return LoadVerifiedImportedBERTEmbedder(ctx, ImportedBERTEmbedderCandidateConfig{
		PackagePath:              packagePath,
		ExpectedSHA256:           ImportedBERTEmbedderCandidatePackageSHA256,
		ExpectedIdentitySHA256:   ImportedBERTEmbedderCandidatePackageIdentitySHA256,
		ExpectedPackageModelName: ImportedBERTEmbedderCandidateSourceModel,
		Runtime:                  rt,
	})
}

func LoadVerifiedImportedBERTEmbedder(ctx context.Context, cfg ImportedBERTEmbedderCandidateConfig) (*PretrainedBERTTextEmbedder, error) {
	if cfg.PackagePath == "" {
		return nil, fmt.Errorf("package path is required")
	}
	if cfg.Runtime == nil {
		return nil, fmt.Errorf("runtime is required")
	}
	packageSHA, err := sha256FileHex(cfg.PackagePath)
	if err != nil {
		return nil, fmt.Errorf("hash package: %w", err)
	}
	if cfg.ExpectedSHA256 != "" && packageSHA != cfg.ExpectedSHA256 {
		return nil, fmt.Errorf("package sha256 mismatch: got %s want %s", packageSHA, cfg.ExpectedSHA256)
	}
	pkg, err := ReadPretrainedBERTPackageFile(cfg.PackagePath)
	if err != nil {
		return nil, err
	}
	identity := pkg.IdentityHash()
	if cfg.ExpectedIdentitySHA256 != "" && identity != cfg.ExpectedIdentitySHA256 {
		return nil, fmt.Errorf("package identity mismatch: got %s want %s", identity, cfg.ExpectedIdentitySHA256)
	}
	if cfg.ExpectedPackageModelName != "" && pkg.ModelName != cfg.ExpectedPackageModelName {
		return nil, fmt.Errorf("package model_name mismatch: got %q want %q", pkg.ModelName, cfg.ExpectedPackageModelName)
	}
	return LoadPretrainedBERTTextEmbedder(ctx, PretrainedBERTTextEmbedderConfig{
		PackagePath: cfg.PackagePath,
		Runtime:     cfg.Runtime,
	})
}
