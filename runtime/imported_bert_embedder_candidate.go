package eosruntime

import (
	"context"
	"fmt"
)

func LoadImportedBERTEmbedderCandidate(ctx context.Context, packagePath string, rt *Runtime) (*PretrainedBERTTextEmbedder, error) {
	if packagePath == "" {
		return nil, fmt.Errorf("package path is required")
	}
	return LoadPretrainedBERTTextEmbedder(ctx, PretrainedBERTTextEmbedderConfig{
		PackagePath: packagePath,
		Runtime:     rt,
	})
}
