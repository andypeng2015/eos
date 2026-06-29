package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/directml"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/eos/runtime/backends/vulkan"
	"m31labs.dev/eos/runtime/backends/webgpu"
)

const pretrainedBERTPackageEmbeddingSchema = "manta.pretrained_bert_package_embedding.v1"

type pretrainedBERTPackageEmbeddingJSON struct {
	Schema                string                                          `json:"schema"`
	PackagePath           string                                          `json:"package_path"`
	PackageSHA256         string                                          `json:"package_sha256,omitempty"`
	PackageIdentitySHA256 string                                          `json:"package_identity_sha256,omitempty"`
	ModelName             string                                          `json:"model_name,omitempty"`
	Role                  string                                          `json:"role"`
	PrefixApplied         string                                          `json:"prefix_applied"`
	QualityClaim          bool                                            `json:"quality_claim"`
	Dimensions            int                                             `json:"dimensions"`
	MaxLength             int                                             `json:"max_length"`
	MaxLengthSource       string                                          `json:"max_length_source,omitempty"`
	Pooling               string                                          `json:"pooling,omitempty"`
	Normalization         string                                          `json:"normalization,omitempty"`
	Embeddings            [][]float32                                     `json:"embeddings"`
	Texts                 []string                                        `json:"texts"`
	RetrievalRoleContract *eosruntime.PretrainedBERTRetrievalRoleContract `json:"retrieval_role_contract,omitempty"`
}

func runEmbedPretrainedBERTPackage(args []string) error {
	fs := flag.NewFlagSet("embed-pretrained-bert-package", flag.ContinueOnError)
	fs.SetOutput(flag.CommandLine.Output())
	role := fs.String("role", eosruntime.EmbeddingRoleRaw, "embedding role: raw, query, or document; default raw applies no prefix")
	jsonOut := fs.Bool("json", false, "write embeddings and metadata as JSON")
	maxLength := fs.Int("max-length", 0, "WordPiece max sequence length; default uses package metadata/config")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos embed-pretrained-bert-package [--role raw|query|document] [--json] [--max-length N] <package.mll> <text...>")
	}
	normalizedRole := strings.TrimSpace(strings.ToLower(*role))
	if normalizedRole == "" {
		normalizedRole = eosruntime.EmbeddingRoleRaw
	}
	if *maxLength < 0 {
		return fmt.Errorf("--max-length must be non-negative")
	}
	packagePath := fs.Arg(0)
	text := strings.Join(fs.Args()[1:], " ")
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	embedder, err := eosruntime.LoadPretrainedBERTTextEmbedder(context.Background(), eosruntime.PretrainedBERTTextEmbedderConfig{
		PackagePath: packagePath,
		MaxLength:   *maxLength,
		Runtime:     rt,
	})
	if err != nil {
		return err
	}
	embeddings, prefix, err := embedder.EmbedTextBatchWithRole(context.Background(), []string{text}, normalizedRole)
	if err != nil {
		return err
	}
	dim := 0
	if len(embeddings) > 0 {
		dim = len(embeddings[0])
	}
	out := pretrainedBERTPackageEmbeddingJSON{
		Schema:                pretrainedBERTPackageEmbeddingSchema,
		PackagePath:           packagePath,
		PackageSHA256:         embedder.PackageSHA256(),
		PackageIdentitySHA256: embedder.PackageIdentitySHA256(),
		ModelName:             embedder.ModelName(),
		Role:                  normalizedRole,
		PrefixApplied:         prefix,
		QualityClaim:          false,
		Dimensions:            dim,
		MaxLength:             embedder.MaxLength(),
		MaxLengthSource:       embedder.MaxLengthSource(),
		Pooling:               embedder.Pooling(),
		Normalization:         embedder.Normalization(),
		Embeddings:            embeddings,
		Texts:                 []string{text},
		RetrievalRoleContract: embedder.RetrievalRoleContract(),
	}
	if *jsonOut {
		data, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		_, err = os.Stdout.Write(data)
		return err
	}
	fmt.Printf("loaded pretrained BERT package: model=%s package_identity=%s quality_claim=false\n", displayOptionalString(out.ModelName), displayOptionalString(out.PackageIdentitySHA256))
	fmt.Printf("role: %s prefix_applied=%q\n", out.Role, out.PrefixApplied)
	fmt.Printf("embedding: f32[%d,%d] norm=%.6f sample=%s\n", len(out.Embeddings), out.Dimensions, embeddingNorm(out.Embeddings), embeddingSample(out.Embeddings, 6))
	return nil
}

func embeddingNorm(rows [][]float32) float64 {
	if len(rows) == 0 {
		return 0
	}
	var sum float64
	for _, value := range rows[0] {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}

func embeddingSample(rows [][]float32, n int) string {
	if len(rows) == 0 || len(rows[0]) == 0 || n <= 0 {
		return "[]"
	}
	if n > len(rows[0]) {
		n = len(rows[0])
	}
	parts := make([]string, 0, n)
	for _, value := range rows[0][:n] {
		parts = append(parts, fmt.Sprintf("%.6g", value))
	}
	if n < len(rows[0]) {
		parts = append(parts, "...")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
