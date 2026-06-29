package models

import (
	"fmt"
	"os"
	"path/filepath"

	eosruntime "m31labs.dev/eos/runtime"
)

const (
	ImportedEmbedderCandidateAssetID             = "corkscrewdb-imported-bge-eos-embed-v1-candidate"
	ImportedEmbedderCandidateModelName           = "eos-embed-v1"
	ImportedEmbedderCandidateDisplayName         = "Eos Embedder 1"
	ImportedEmbedderCandidateSourceModel         = "BAAI/bge-small-en-v1.5"
	ImportedEmbedderCandidateStatus              = "non_default_reference_candidate"
	ImportedEmbedderCandidatePackageRelativePath = "runs/pretrained-bert-current-hf-parity-v1-20260629T090818Z/bge/bge-small-en-v1.5.imported.mll"
	ImportedEmbedderCandidatePackageSHA256       = "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763"
	ImportedEmbedderCandidatePackageIdentity     = "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637"
)

type ImportedEmbedderCandidateAsset struct {
	AssetID             string `json:"asset_id"`
	ModelName           string `json:"model_name"`
	DisplayName         string `json:"display_name"`
	SourceModel         string `json:"source_model"`
	Status              string `json:"status"`
	PackagePath         string `json:"package_path"`
	PackageRelativePath string `json:"package_relative_path,omitempty"`
	PackageSHA256       string `json:"package_sha256"`
	PackageIdentity     string `json:"package_identity"`
	QualityClaim        bool   `json:"quality_claim"`
	DefaultAliasChanged bool   `json:"default_alias_changed"`
	LoadPath            string `json:"load_path"`
}

type ImportedEmbedderCandidateVerification struct {
	Asset    ImportedEmbedderCandidateAsset `json:"asset"`
	File     DefaultEmbedderFileCheck       `json:"file"`
	Identity ImportedEmbedderIdentityCheck  `json:"identity"`
	OK       bool                           `json:"ok"`
}

type ImportedEmbedderIdentityCheck struct {
	Path             string `json:"path"`
	Identity         string `json:"identity"`
	ExpectedIdentity string `json:"expected_identity"`
	OK               bool   `json:"ok"`
}

func ImportedEmbedderCandidateAssetInfo(root, packagePath string) (ImportedEmbedderCandidateAsset, error) {
	path := packagePath
	rel := ImportedEmbedderCandidatePackageRelativePath
	if path == "" {
		resolvedRoot, err := ResolveImportedEmbedderCandidateRoot(root)
		if err != nil {
			return ImportedEmbedderCandidateAsset{}, err
		}
		path = filepath.Join(resolvedRoot, ImportedEmbedderCandidatePackageRelativePath)
	} else {
		abs, err := filepath.Abs(path)
		if err != nil {
			return ImportedEmbedderCandidateAsset{}, err
		}
		path = abs
		rel = ""
	}
	return ImportedEmbedderCandidateAsset{
		AssetID:             ImportedEmbedderCandidateAssetID,
		ModelName:           ImportedEmbedderCandidateModelName,
		DisplayName:         ImportedEmbedderCandidateDisplayName,
		SourceModel:         ImportedEmbedderCandidateSourceModel,
		Status:              ImportedEmbedderCandidateStatus,
		PackagePath:         path,
		PackageRelativePath: rel,
		PackageSHA256:       ImportedEmbedderCandidatePackageSHA256,
		PackageIdentity:     ImportedEmbedderCandidatePackageIdentity,
		QualityClaim:        false,
		DefaultAliasChanged: false,
		LoadPath:            "runtime.LoadPretrainedBERTTextEmbedder",
	}, nil
}

func ResolveImportedEmbedderCandidateRoot(root string) (string, error) {
	if root != "" {
		return filepath.Abs(root)
	}
	if cwd, err := os.Getwd(); err == nil {
		for dir := cwd; ; dir = filepath.Dir(dir) {
			if importedEmbedderCandidatePackageExists(dir) {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
		}
	}
	return "", fmt.Errorf("imported embedder candidate package %q not found; pass --root or --package", ImportedEmbedderCandidatePackageRelativePath)
}

func VerifyImportedEmbedderCandidate(root, packagePath string) (ImportedEmbedderCandidateVerification, error) {
	info, err := ImportedEmbedderCandidateAssetInfo(root, packagePath)
	if err != nil {
		return ImportedEmbedderCandidateVerification{}, err
	}
	report := ImportedEmbedderCandidateVerification{
		Asset: info,
		OK:    true,
	}
	file, err := verifyDefaultEmbedderFile("package", info.PackagePath, info.PackageSHA256)
	if err != nil {
		report.OK = false
		return report, err
	}
	report.File = file
	if !file.OK {
		report.OK = false
	}
	pkg, err := eosruntime.ReadPretrainedBERTPackageFile(info.PackagePath)
	if err != nil {
		report.OK = false
		return report, err
	}
	identity := pkg.IdentityHash()
	report.Identity = ImportedEmbedderIdentityCheck{
		Path:             info.PackagePath,
		Identity:         identity,
		ExpectedIdentity: info.PackageIdentity,
		OK:               identity == info.PackageIdentity,
	}
	if !report.Identity.OK {
		report.OK = false
	}
	if !report.OK {
		return report, fmt.Errorf("imported embedder candidate verification failed")
	}
	return report, nil
}

func importedEmbedderCandidatePackageExists(root string) bool {
	path := filepath.Join(root, ImportedEmbedderCandidatePackageRelativePath)
	if st, err := os.Stat(path); err == nil && !st.IsDir() {
		return true
	}
	return false
}
