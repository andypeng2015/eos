package models

import (
	"fmt"
	"os"
	"path/filepath"

	eosruntime "m31labs.dev/eos/runtime"
)

const (
	ImportedEmbedderCandidateID                  = "corkscrewdb-imported-bge-eos-embed-v1-candidate"
	ImportedEmbedderCandidateModelName           = eosruntime.ImportedBERTEmbedderCandidateModelName
	ImportedEmbedderCandidateDisplayName         = "Eos Embedder 1"
	ImportedEmbedderCandidateSourceModel         = eosruntime.ImportedBERTEmbedderCandidateSourceModel
	ImportedEmbedderCandidateStatus              = "non_default_reference_candidate"
	ImportedEmbedderCandidatePackageRelativePath = eosruntime.ImportedBERTEmbedderCandidatePackageRelativePathHint
	ImportedEmbedderCandidatePackageSHA256       = eosruntime.ImportedBERTEmbedderCandidatePackageSHA256
	ImportedEmbedderCandidatePackageIdentity     = eosruntime.ImportedBERTEmbedderCandidatePackageIdentitySHA256
	ImportedEmbedderCandidatePublicIdentityNote  = "Public identity is model_name/display_name/source_model; candidate_id is an internal review slug."
	ImportedEmbedderCandidateAssetID             = ImportedEmbedderCandidateID
	ImportedEmbedderCandidateSourceSnapshot      = eosruntime.ImportedBERTEmbedderCandidateSourceSnapshotCommit
	ImportedEmbedderCandidateUpstreamModelURL    = eosruntime.ImportedBERTEmbedderCandidateUpstreamModelURL
	ImportedEmbedderCandidateLicenseID           = eosruntime.ImportedBERTEmbedderCandidateLicenseID
	ImportedEmbedderCandidateAttribution         = eosruntime.ImportedBERTEmbedderCandidateAttribution
	ImportedEmbedderCandidateRoleContractSchema  = eosruntime.PretrainedBERTRetrievalRoleContractSchema
	ImportedEmbedderCandidateQueryRole           = "query"
	ImportedEmbedderCandidateQueryPrefix         = "Represent this sentence for searching relevant passages: "
	ImportedEmbedderCandidateDocumentRole        = "document"
	ImportedEmbedderCandidateDocumentPrefix      = ""
	ImportedEmbedderCandidatePooling             = eosruntime.ImportedBERTEmbedderCandidatePooling
	ImportedEmbedderCandidateNormalization       = eosruntime.ImportedBERTEmbedderCandidateNormalization
	ImportedEmbedderCandidateMaxLength           = eosruntime.ImportedBERTEmbedderCandidateMaxLength
	ImportedEmbedderCandidateNativeDim           = eosruntime.ImportedBERTEmbedderCandidateNativeDim
)

type ImportedEmbedderCandidateAsset struct {
	CandidateID              string `json:"candidate_id"`
	ModelName                string `json:"model_name"`
	DisplayName              string `json:"display_name"`
	SourceModel              string `json:"source_model"`
	Status                   string `json:"status"`
	PublicIdentityNote       string `json:"public_identity_note"`
	PackagePath              string `json:"package_path"`
	PackageSHA256            string `json:"package_sha256"`
	PackageIdentity          string `json:"package_identity"`
	SourceSnapshotCommit     string `json:"source_snapshot_commit"`
	UpstreamModelURL         string `json:"upstream_model_url"`
	LicenseID                string `json:"license_id"`
	LicenseNoticeRequired    bool   `json:"license_notice_required"`
	ProvenanceNoticeRequired bool   `json:"provenance_notice_required"`
	Attribution              string `json:"attribution"`
	RoleContractSchema       string `json:"role_contract_schema"`
	QueryRole                string `json:"query_role"`
	QueryPrefix              string `json:"query_prefix"`
	DocumentRole             string `json:"document_role"`
	DocumentPrefix           string `json:"document_prefix"`
	Pooling                  string `json:"pooling"`
	Normalization            string `json:"normalization"`
	MaxLength                int    `json:"max_length"`
	NativeDim                int    `json:"native_dim"`
	QualityClaim             bool   `json:"quality_claim"`
	DefaultAliasChanged      bool   `json:"default_alias_changed"`
	LoadPath                 string `json:"load_path"`
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

type ImportedEmbedderCandidateVerifyConfig struct {
	Root                   string
	PackagePath            string
	ExpectedSHA256         string
	ExpectedIdentitySHA256 string
}

func ImportedEmbedderCandidateAssetInfo(root, packagePath string) (ImportedEmbedderCandidateAsset, error) {
	path := packagePath
	if path == "" {
		if root == "" {
			return ImportedEmbedderCandidateAsset{}, fmt.Errorf("imported embedder candidate package path is required; pass --package <path> or --root <repo-root>")
		}
		resolvedRoot, err := filepath.Abs(root)
		if err != nil {
			return ImportedEmbedderCandidateAsset{}, err
		}
		path = filepath.Join(resolvedRoot, ImportedEmbedderCandidatePackageRelativePath)
		if !importedEmbedderCandidatePackagePathExists(path) {
			return ImportedEmbedderCandidateAsset{}, fmt.Errorf("imported embedder candidate package %q not found under root %q; pass --package <path>", ImportedEmbedderCandidatePackageRelativePath, resolvedRoot)
		}
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return ImportedEmbedderCandidateAsset{}, err
	}
	path = abs
	return ImportedEmbedderCandidateAsset{
		CandidateID:              ImportedEmbedderCandidateID,
		ModelName:                ImportedEmbedderCandidateModelName,
		DisplayName:              ImportedEmbedderCandidateDisplayName,
		SourceModel:              ImportedEmbedderCandidateSourceModel,
		Status:                   ImportedEmbedderCandidateStatus,
		PublicIdentityNote:       ImportedEmbedderCandidatePublicIdentityNote,
		PackagePath:              path,
		PackageSHA256:            ImportedEmbedderCandidatePackageSHA256,
		PackageIdentity:          ImportedEmbedderCandidatePackageIdentity,
		SourceSnapshotCommit:     ImportedEmbedderCandidateSourceSnapshot,
		UpstreamModelURL:         ImportedEmbedderCandidateUpstreamModelURL,
		LicenseID:                ImportedEmbedderCandidateLicenseID,
		LicenseNoticeRequired:    true,
		ProvenanceNoticeRequired: true,
		Attribution:              ImportedEmbedderCandidateAttribution,
		RoleContractSchema:       ImportedEmbedderCandidateRoleContractSchema,
		QueryRole:                ImportedEmbedderCandidateQueryRole,
		QueryPrefix:              ImportedEmbedderCandidateQueryPrefix,
		DocumentRole:             ImportedEmbedderCandidateDocumentRole,
		DocumentPrefix:           ImportedEmbedderCandidateDocumentPrefix,
		Pooling:                  ImportedEmbedderCandidatePooling,
		Normalization:            ImportedEmbedderCandidateNormalization,
		MaxLength:                ImportedEmbedderCandidateMaxLength,
		NativeDim:                ImportedEmbedderCandidateNativeDim,
		QualityClaim:             false,
		DefaultAliasChanged:      false,
		LoadPath:                 "runtime.LoadImportedBERTEmbedderCandidate",
	}, nil
}

func ResolveImportedEmbedderCandidateRoot(root string) (string, error) {
	if root == "" {
		return "", fmt.Errorf("imported embedder candidate root is required; pass --root or --package")
	}
	resolved, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	if !importedEmbedderCandidatePackageExists(resolved) {
		return "", fmt.Errorf("imported embedder candidate package %q not found under root %q; pass --package", ImportedEmbedderCandidatePackageRelativePath, resolved)
	}
	return resolved, nil
}

func VerifyImportedEmbedderCandidate(root, packagePath string) (ImportedEmbedderCandidateVerification, error) {
	return VerifyImportedEmbedderCandidateWithConfig(ImportedEmbedderCandidateVerifyConfig{
		Root:                   root,
		PackagePath:            packagePath,
		ExpectedSHA256:         ImportedEmbedderCandidatePackageSHA256,
		ExpectedIdentitySHA256: ImportedEmbedderCandidatePackageIdentity,
	})
}

func VerifyImportedEmbedderCandidateWithConfig(cfg ImportedEmbedderCandidateVerifyConfig) (ImportedEmbedderCandidateVerification, error) {
	expectedSHA := cfg.ExpectedSHA256
	if expectedSHA == "" {
		expectedSHA = ImportedEmbedderCandidatePackageSHA256
	}
	expectedIdentity := cfg.ExpectedIdentitySHA256
	if expectedIdentity == "" {
		expectedIdentity = ImportedEmbedderCandidatePackageIdentity
	}
	info, err := ImportedEmbedderCandidateAssetInfo(cfg.Root, cfg.PackagePath)
	if err != nil {
		return ImportedEmbedderCandidateVerification{}, err
	}
	info.PackageSHA256 = expectedSHA
	info.PackageIdentity = expectedIdentity
	report := ImportedEmbedderCandidateVerification{
		Asset: info,
		OK:    true,
	}
	file, err := verifyDefaultEmbedderFile("package", info.PackagePath, expectedSHA)
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
		ExpectedIdentity: expectedIdentity,
		OK:               identity == expectedIdentity,
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
	return importedEmbedderCandidatePackagePathExists(path)
}

func importedEmbedderCandidatePackagePathExists(path string) bool {
	if st, err := os.Stat(path); err == nil && !st.IsDir() {
		return true
	}
	return false
}
