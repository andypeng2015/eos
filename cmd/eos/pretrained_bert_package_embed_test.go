package main

import (
	"encoding/json"
	"strings"
	"testing"

	eosruntime "m31labs.dev/eos/runtime"
)

func TestRunEmbedPretrainedBERTPackageJSONUsesRoleContract(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "intfloat/e5-small-v2", true)

	output := captureRunOutput(t, []string{"embed-pretrained-bert-package", "--role", "query", "--json", packagePath, "hello", "world"})

	var got struct {
		Schema                string                                          `json:"schema"`
		PackagePath           string                                          `json:"package_path"`
		PackageSHA256         string                                          `json:"package_sha256"`
		PackageIdentitySHA256 string                                          `json:"package_identity_sha256"`
		ModelName             string                                          `json:"model_name"`
		Role                  string                                          `json:"role"`
		PrefixApplied         string                                          `json:"prefix_applied"`
		QualityClaim          bool                                            `json:"quality_claim"`
		Dimensions            int                                             `json:"dimensions"`
		MaxLength             int                                             `json:"max_length"`
		Embeddings            [][]float32                                     `json:"embeddings"`
		Texts                 []string                                        `json:"texts"`
		RetrievalRoleContract *eosruntime.PretrainedBERTRetrievalRoleContract `json:"retrieval_role_contract"`
	}
	if err := json.Unmarshal([]byte(output), &got); err != nil {
		t.Fatalf("parse embed json: %v\n%s", err, output)
	}
	if got.Schema != pretrainedBERTPackageEmbeddingSchema {
		t.Fatalf("schema = %q", got.Schema)
	}
	if got.PackagePath != packagePath || got.PackageSHA256 == "" || got.PackageIdentitySHA256 == "" {
		t.Fatalf("package identity fields = %+v", got)
	}
	if got.ModelName != "intfloat/e5-small-v2" || got.Role != eosruntime.EmbeddingRoleQuery || got.PrefixApplied != "query: " {
		t.Fatalf("role/model fields = %+v", got)
	}
	if got.RetrievalRoleContract == nil || got.RetrievalRoleContract.QueryPrefix != "query: " || got.RetrievalRoleContract.DocumentPrefix != "passage: " {
		t.Fatalf("retrieval role contract = %+v", got.RetrievalRoleContract)
	}
	if got.QualityClaim {
		t.Fatalf("quality_claim = true, want false")
	}
	if got.Dimensions != 2 || got.MaxLength != 4 || len(got.Embeddings) != 1 || len(got.Embeddings[0]) != got.Dimensions {
		t.Fatalf("shape fields = %+v", got)
	}
	if len(got.Texts) != 1 || got.Texts[0] != "hello world" {
		t.Fatalf("texts = %+v", got.Texts)
	}
}

func TestRunEmbedPretrainedBERTPackageJSONUsesDocumentRoleAndMaxLength(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "intfloat/e5-small-v2", true)

	output := captureRunOutput(t, []string{"embed-pretrained-bert-package", "--role", "document", "--max-length", "3", "--json", packagePath, "hello"})

	var got struct {
		Role            string      `json:"role"`
		PrefixApplied   string      `json:"prefix_applied"`
		MaxLength       int         `json:"max_length"`
		MaxLengthSource string      `json:"max_length_source"`
		Embeddings      [][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal([]byte(output), &got); err != nil {
		t.Fatalf("parse embed json: %v\n%s", err, output)
	}
	if got.Role != eosruntime.EmbeddingRoleDocument || got.PrefixApplied != "passage: " {
		t.Fatalf("role fields = %+v", got)
	}
	if got.MaxLength != 3 || got.MaxLengthSource != "explicit" {
		t.Fatalf("max length fields = %+v", got)
	}
	if len(got.Embeddings) != 1 || len(got.Embeddings[0]) != 2 {
		t.Fatalf("embedding shape = %+v", got.Embeddings)
	}
}

func TestRunEmbedPretrainedBERTPackageTextDefaultsToRaw(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "intfloat/e5-small-v2", true)

	output := captureRunOutput(t, []string{"embed-pretrained-bert-package", packagePath, "hello"})

	for _, want := range []string{
		"quality_claim=false",
		"role: raw prefix_applied=\"\"",
		"embedding: f32[1,2]",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output missing %q:\n%s", want, output)
		}
	}
}

func TestRunEmbedPretrainedBERTPackageJSONOmitsMissingRoleContract(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "fixture/bert", true)

	output := captureRunOutput(t, []string{"embed-pretrained-bert-package", "--json", packagePath, "hello"})

	var got map[string]json.RawMessage
	if err := json.Unmarshal([]byte(output), &got); err != nil {
		t.Fatalf("parse embed json: %v\n%s", err, output)
	}
	if value, ok := got["retrieval_role_contract"]; ok {
		t.Fatalf("retrieval_role_contract encoded despite missing package contract: %s", string(value))
	}
}

func TestRunEmbedPretrainedBERTPackageRejectsRoleWithoutContract(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "fixture/bert", true)

	_, err := captureRunOutputAndError(t, []string{"embed-pretrained-bert-package", "--role", "query", packagePath, "hello"})
	if err == nil || !strings.Contains(err.Error(), "does not declare retrieval_role_contract") {
		t.Fatalf("err = %v, want retrieval_role_contract error", err)
	}
}
