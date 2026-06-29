package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

type pretrainedBERTPackageInspectJSON struct {
	ModelName             string                                          `json:"model_name,omitempty"`
	Architecture          string                                          `json:"architecture,omitempty"`
	ModelType             string                                          `json:"model_type,omitempty"`
	Pooling               string                                          `json:"pooling,omitempty"`
	Normalization         string                                          `json:"normalization,omitempty"`
	MaxLength             int                                             `json:"max_length"`
	NativeDim             int                                             `json:"native_dim"`
	IdentitySHA256        string                                          `json:"identity_sha256,omitempty"`
	ModuleSHA256          string                                          `json:"module_sha256,omitempty"`
	WeightsSHA256         string                                          `json:"weights_sha256,omitempty"`
	FileCount             int                                             `json:"file_count"`
	FileRoles             []string                                        `json:"file_roles"`
	RetrievalRoleContract *eosruntime.PretrainedBERTRetrievalRoleContract `json:"retrieval_role_contract,omitempty"`
}

func runInspectPretrainedBERTPackage(args []string) error {
	fs := flag.NewFlagSet("inspect-pretrained-bert-package", flag.ContinueOnError)
	jsonOut := fs.Bool("json", false, "write package metadata as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos inspect-pretrained-bert-package [--json] <package.mll>")
	}
	pkg, err := eosruntime.ReadPretrainedBERTPackageFile(fs.Arg(0))
	if err != nil {
		return err
	}
	out := pretrainedBERTPackageInspectJSON{
		ModelName:             pkg.ModelName,
		Architecture:          pkg.Architecture,
		ModelType:             pkg.Config.ModelType,
		Pooling:               pkg.Pooling,
		Normalization:         pkg.Normalization,
		MaxLength:             pkg.MaxLength,
		NativeDim:             pkg.NativeDim,
		IdentitySHA256:        pkg.IdentitySHA256,
		ModuleSHA256:          pkg.ModuleSHA256,
		WeightsSHA256:         pkg.WeightsSHA256,
		FileCount:             len(pkg.Files),
		FileRoles:             pretrainedBERTPackageFileRoles(pkg.Files),
		RetrievalRoleContract: pkg.RetrievalRoleContract,
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
	fmt.Printf("model_name: %s\n", displayOptionalString(pkg.ModelName))
	fmt.Printf("architecture: %s\n", displayOptionalString(pkg.Architecture))
	fmt.Printf("model_type: %s\n", displayOptionalString(pkg.Config.ModelType))
	fmt.Printf("pooling: %s\n", displayOptionalString(pkg.Pooling))
	fmt.Printf("normalization: %s\n", displayOptionalString(pkg.Normalization))
	fmt.Printf("max_length: %d\n", pkg.MaxLength)
	fmt.Printf("native_dim: %d\n", pkg.NativeDim)
	fmt.Printf("identity_sha256: %s\n", pkg.IdentitySHA256)
	fmt.Printf("module_sha256: %s\n", pkg.ModuleSHA256)
	fmt.Printf("weights_sha256: %s\n", pkg.WeightsSHA256)
	fmt.Printf("files: count=%d roles=%s\n", len(pkg.Files), strings.Join(out.FileRoles, ", "))
	if pkg.RetrievalRoleContract != nil {
		fmt.Printf("retrieval_role_contract: schema=%s preset=%s query_role=%s document_role=%s query_prefix=%q document_prefix=%q pooling=%s max_length=%d\n",
			pkg.RetrievalRoleContract.Schema,
			pkg.RetrievalRoleContract.Preset,
			pkg.RetrievalRoleContract.QueryRole,
			pkg.RetrievalRoleContract.DocumentRole,
			pkg.RetrievalRoleContract.QueryPrefix,
			pkg.RetrievalRoleContract.DocumentPrefix,
			pkg.RetrievalRoleContract.Pooling,
			pkg.RetrievalRoleContract.MaxLength,
		)
	}
	return nil
}

func pretrainedBERTPackageFileRoles(files []eosruntime.PretrainedBERTPackageFile) []string {
	roles := make([]string, 0, len(files))
	for _, file := range files {
		roles = append(roles, file.Role)
	}
	sort.Strings(roles)
	return roles
}

func displayOptionalString(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return value
}
