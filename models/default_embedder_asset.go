package models

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	goruntime "runtime"
)

const (
	DefaultEmbedderAssetID           = "corkscrewdb-default-embedder"
	DefaultEmbedderAssetRelativeDir  = "assets/corkscrewdb-default-embedder"
	DefaultEmbedderArtifactFilename  = "corkscrewdb-default-embedder.mll"
	DefaultEmbedderTokenizerFilename = "corkscrewdb-default-embedder.tokenizer.mll"
	DefaultEmbedderManifestFilename  = "manifest.json"

	DefaultEmbedderArtifactSHA256       = "f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a"
	DefaultEmbedderTokenizerSHA256      = "64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5"
	DefaultEmbedderReleasePackageSHA256 = "188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf"
)

// DefaultEmbedderAsset describes the durable in-repo CorkScrewDB default
// embedder package.
type DefaultEmbedderAsset struct {
	AssetID                 string `json:"asset_id"`
	ModelName               string `json:"model_name"`
	RelativeDir             string `json:"relative_dir"`
	ArtifactFilename        string `json:"artifact_filename"`
	TokenizerFilename       string `json:"tokenizer_filename"`
	ManifestFilename        string `json:"manifest_filename"`
	ArtifactPath            string `json:"artifact_path"`
	TokenizerPath           string `json:"tokenizer_path"`
	ManifestPath            string `json:"manifest_path"`
	ArtifactSHA256          string `json:"artifact_sha256"`
	TokenizerSHA256         string `json:"tokenizer_sha256"`
	ReleasePackageSHA256    string `json:"release_package_sha256"`
	CompactRecallTolerance  string `json:"compact_recall_tolerance"`
	CompactStrictRegression bool   `json:"compact_strict_zero_regression"`
}

type DefaultEmbedderAssetVerification struct {
	Asset DefaultEmbedderAsset       `json:"asset"`
	Files []DefaultEmbedderFileCheck `json:"files"`
	OK    bool                       `json:"ok"`
}

type DefaultEmbedderFileCheck struct {
	Role           string `json:"role"`
	Path           string `json:"path"`
	SHA256         string `json:"sha256"`
	ExpectedSHA256 string `json:"expected_sha256"`
	Bytes          int64  `json:"bytes"`
	OK             bool   `json:"ok"`
}

func DefaultEmbedderAssetInfo(root string) (DefaultEmbedderAsset, error) {
	root, err := ResolveDefaultEmbedderAssetRoot(root)
	if err != nil {
		return DefaultEmbedderAsset{}, err
	}
	dir := filepath.Join(root, DefaultEmbedderAssetRelativeDir)
	return DefaultEmbedderAsset{
		AssetID:                 DefaultEmbedderAssetID,
		ModelName:               DefaultEmbeddingModelName,
		RelativeDir:             DefaultEmbedderAssetRelativeDir,
		ArtifactFilename:        DefaultEmbedderArtifactFilename,
		TokenizerFilename:       DefaultEmbedderTokenizerFilename,
		ManifestFilename:        DefaultEmbedderManifestFilename,
		ArtifactPath:            filepath.Join(dir, DefaultEmbedderArtifactFilename),
		TokenizerPath:           filepath.Join(dir, DefaultEmbedderTokenizerFilename),
		ManifestPath:            filepath.Join(dir, DefaultEmbedderManifestFilename),
		ArtifactSHA256:          DefaultEmbedderArtifactSHA256,
		TokenizerSHA256:         DefaultEmbedderTokenizerSHA256,
		ReleasePackageSHA256:    DefaultEmbedderReleasePackageSHA256,
		CompactRecallTolerance:  "0",
		CompactStrictRegression: true,
	}, nil
}

func ResolveDefaultEmbedderAssetPath(root string) (string, error) {
	info, err := DefaultEmbedderAssetInfo(root)
	if err != nil {
		return "", err
	}
	return info.ArtifactPath, nil
}

func ResolveDefaultEmbedderAssetRoot(root string) (string, error) {
	if root != "" {
		return filepath.Abs(root)
	}
	if _, file, _, ok := goruntime.Caller(0); ok {
		candidate := filepath.Dir(filepath.Dir(file))
		if defaultEmbedderAssetExists(candidate) {
			return candidate, nil
		}
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for dir := wd; ; dir = filepath.Dir(dir) {
		if defaultEmbedderAssetExists(dir) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return "", fmt.Errorf("default embedder asset %q not found; pass --root", DefaultEmbedderAssetRelativeDir)
}

func VerifyDefaultEmbedderAsset(root string) (DefaultEmbedderAssetVerification, error) {
	info, err := DefaultEmbedderAssetInfo(root)
	if err != nil {
		return DefaultEmbedderAssetVerification{}, err
	}
	report := DefaultEmbedderAssetVerification{
		Asset: info,
		OK:    true,
	}
	for _, item := range []struct {
		role string
		path string
		want string
	}{
		{role: "artifact", path: info.ArtifactPath, want: info.ArtifactSHA256},
		{role: "tokenizer", path: info.TokenizerPath, want: info.TokenizerSHA256},
	} {
		check, err := verifyDefaultEmbedderFile(item.role, item.path, item.want)
		if err != nil {
			report.OK = false
			return report, err
		}
		if !check.OK {
			report.OK = false
		}
		report.Files = append(report.Files, check)
	}
	if !report.OK {
		return report, fmt.Errorf("default embedder asset hash verification failed")
	}
	return report, nil
}

func defaultEmbedderAssetExists(root string) bool {
	path := filepath.Join(root, DefaultEmbedderAssetRelativeDir, DefaultEmbedderArtifactFilename)
	if st, err := os.Stat(path); err == nil && !st.IsDir() {
		return true
	}
	return false
}

func verifyDefaultEmbedderFile(role, path, want string) (DefaultEmbedderFileCheck, error) {
	file, err := os.Open(path)
	if err != nil {
		return DefaultEmbedderFileCheck{}, err
	}
	defer file.Close()
	h := sha256.New()
	size, err := io.Copy(h, file)
	if err != nil {
		return DefaultEmbedderFileCheck{}, err
	}
	got := hex.EncodeToString(h.Sum(nil))
	return DefaultEmbedderFileCheck{
		Role:           role,
		Path:           path,
		SHA256:         got,
		ExpectedSHA256: want,
		Bytes:          size,
		OK:             got == want,
	}, nil
}
