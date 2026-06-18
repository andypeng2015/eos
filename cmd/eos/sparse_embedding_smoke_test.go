package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSparseEmbeddingSmokeConfigAdds32KPreflight(t *testing.T) {
	cfg, err := parseSparseEmbeddingSmokeConfig([]string{
		"-run-root", t.TempDir(),
		"-seq-len", "16",
		"-query-len", "2",
		"-dim", "4",
		"-top-k", "2",
		"-route-block-size", "4",
		"-route-top-blocks", "1",
		"-preflight-key-lens", "64,128",
	})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if !containsInt(cfg.PreflightKeyLens, 32768) {
		t.Fatalf("preflight key lens = %v, want 32768 included", cfg.PreflightKeyLens)
	}
	if cfg.ValueDim != cfg.Dim {
		t.Fatalf("value_dim = %d, want dim %d", cfg.ValueDim, cfg.Dim)
	}
	if cfg.Backend != "auto" {
		t.Fatalf("backend = %q, want auto", cfg.Backend)
	}
}

func TestSparseEmbeddingSmokeConfigBackendEnvAndFlag(t *testing.T) {
	t.Setenv("EOS_SPARSE_EMBED_SMOKE_BACKEND", "cuda")
	cfg, err := parseSparseEmbeddingSmokeConfig([]string{
		"-run-root", t.TempDir(),
		"-seq-len", "16",
		"-query-len", "2",
		"-dim", "4",
		"-top-k", "2",
		"-route-block-size", "4",
		"-route-top-blocks", "1",
		"-preflight-key-lens", "64",
		"-backend", "host",
	})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.Backend != "host" {
		t.Fatalf("backend = %q, want flag override host", cfg.Backend)
	}
	if _, err := parseSparseEmbeddingSmokeConfig([]string{
		"-run-root", t.TempDir(),
		"-backend", "bogus",
	}); err == nil {
		t.Fatal("parse invalid backend succeeded")
	}
}

func TestRunSmokeSparseEmbeddingEncoderWritesArtifacts(t *testing.T) {
	runRoot := t.TempDir()
	if err := runSmokeSparseEmbeddingEncoder([]string{
		"-run-root", runRoot,
		"-backend", "host",
		"-seq-len", "32",
		"-query-len", "2",
		"-dim", "8",
		"-top-k", "2",
		"-route-top-blocks", "2",
		"-preflight-key-lens", "4096,32768",
		"-max-score-fraction", "0.2",
	}); err != nil {
		t.Fatalf("run smoke: %v", err)
	}
	matches, err := filepath.Glob(filepath.Join(runRoot, "eos-sparse-embedding-encoder-smoke-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("run dirs = %v, want one", matches)
	}
	manifestPath := filepath.Join(matches[0], "manifest.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest sparseEmbeddingSmokeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Embedding.Dimension != 8 || len(manifest.Embedding.Vector) != 8 {
		t.Fatalf("embedding dimension/vector = %d/%d, want 8/8", manifest.Embedding.Dimension, len(manifest.Embedding.Vector))
	}
	if manifest.Runtime.OutputShape[0] != 2 || manifest.Runtime.OutputShape[1] != 8 {
		t.Fatalf("output shape = %v, want [2 8]", manifest.Runtime.OutputShape)
	}
	if manifest.Config.Backend != "host" || manifest.Runtime.RequestedBackend != "host" || manifest.Runtime.ActualBackend != "host_reference" {
		t.Fatalf("backend metadata config=%q requested=%q actual=%q", manifest.Config.Backend, manifest.Runtime.RequestedBackend, manifest.Runtime.ActualBackend)
	}
	if manifest.Runtime.CUDAAvailable || manifest.Runtime.CUDAEvidenceStatus != "not_requested" {
		t.Fatalf("cuda metadata available=%v evidence=%q", manifest.Runtime.CUDAAvailable, manifest.Runtime.CUDAEvidenceStatus)
	}
	if manifest.Runtime.DenseKVMaterialized != true || manifest.Runtime.KVDecode != "host_reference_decode" || manifest.Runtime.DeviceExecution {
		t.Fatalf("host runtime metadata = %+v", manifest.Runtime)
	}
	if !manifest.ThirtyTwoKPreflight.Present || !manifest.ThirtyTwoKPreflight.Passed {
		t.Fatalf("32k preflight = %+v, want present pass", manifest.ThirtyTwoKPreflight)
	}
	if !manifest.ThirtyTwoKPreflightOnly {
		t.Fatal("32k_preflight_only = false, want true for runtime seq_len 32")
	}
	if manifest.Runtime.AttentionMetadata["routing"] != "block_anchor" {
		t.Fatalf("routing metadata = %v, want block_anchor", manifest.Runtime.AttentionMetadata["routing"])
	}
	if manifest.Parity.Status != "pass" || !manifest.Parity.StrictGate {
		t.Fatalf("parity status=%q strict_gate=%v", manifest.Parity.Status, manifest.Parity.StrictGate)
	}
	if !manifest.Parity.BackendVsHostTurboQuant.Passed {
		t.Fatalf("backend parity = %+v, want pass", manifest.Parity.BackendVsHostTurboQuant)
	}
	if manifest.Parity.BackendVsHostTurboQuant.MaxAbsError != 0 || manifest.Parity.BackendVsHostTurboQuant.MSE != 0 {
		t.Fatalf("host backend parity = %+v, want exact self-match", manifest.Parity.BackendVsHostTurboQuant)
	}
	if manifest.Parity.BackendVsHostTurboQuant.CosineSimilarity < 0.999999 {
		t.Fatalf("host backend parity cosine = %.9g, want near 1", manifest.Parity.BackendVsHostTurboQuant.CosineSimilarity)
	}
	if manifest.Parity.BackendVsHostTurboQuant.ActualSHA256 == "" || manifest.Parity.BackendVsHostTurboQuant.ExpectedSHA256 == "" {
		t.Fatalf("backend parity hashes = %+v, want populated", manifest.Parity.BackendVsHostTurboQuant)
	}
	if manifest.Parity.Diagnostics.Status != "computed" {
		t.Fatalf("parity diagnostics status = %q, want computed", manifest.Parity.Diagnostics.Status)
	}
	if manifest.Parity.Diagnostics.DenseFullSHA256 == "" || manifest.Parity.Diagnostics.ExactSparseSHA256 == "" || manifest.Parity.Diagnostics.RoutedSparseDenseSHA256 == "" || manifest.Parity.Diagnostics.TurboQuantRoutedHostSHA256 == "" {
		t.Fatalf("parity diagnostic hashes missing: %+v", manifest.Parity.Diagnostics)
	}
	summary, err := os.ReadFile(filepath.Join(matches[0], "summary.tsv"))
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(summary)), "\n")
	if len(lines) != 2 {
		t.Fatalf("summary lines = %d, want 2", len(lines))
	}
	if got, want := len(strings.Split(lines[0], "\t")), len(strings.Split(lines[1], "\t")); got != want {
		t.Fatalf("summary columns header=%d row=%d", got, want)
	}
	header := strings.Split(lines[0], "\t")
	row := strings.Split(lines[1], "\t")
	summaryValues := map[string]string{}
	for i, name := range header {
		summaryValues[name] = row[i]
	}
	if summaryValues["parity_status"] != "pass" || summaryValues["parity_backend_vs_host_passed"] != "true" {
		t.Fatalf("summary parity status=%q pass=%q", summaryValues["parity_status"], summaryValues["parity_backend_vs_host_passed"])
	}
	if summaryValues["parity_backend_vs_host_max_abs_error"] != "0" || summaryValues["parity_backend_vs_host_mse"] != "0" {
		t.Fatalf("summary backend parity errors max=%q mse=%q", summaryValues["parity_backend_vs_host_max_abs_error"], summaryValues["parity_backend_vs_host_mse"])
	}
	if summaryValues["parity_diagnostics_status"] != "computed" {
		t.Fatalf("summary diagnostics status=%q, want computed", summaryValues["parity_diagnostics_status"])
	}
	scorecardData, err := os.ReadFile(filepath.Join(matches[0], "scorecard.json"))
	if err != nil {
		t.Fatalf("read scorecard: %v", err)
	}
	var scorecard sparseEmbeddingSmokeScorecard
	if err := json.Unmarshal(scorecardData, &scorecard); err != nil {
		t.Fatalf("decode scorecard: %v", err)
	}
	if scorecard.Schema != "eos.sparse_embedding_encoder_smoke_scorecard.v1" {
		t.Fatalf("scorecard schema = %q", scorecard.Schema)
	}
	if len(scorecard.Rows) != 1 {
		t.Fatalf("scorecard rows = %d, want 1", len(scorecard.Rows))
	}
	scoreRow := scorecard.Rows[0]
	if scoreRow.Category != "long_context_sparse_smoke" || scoreRow.Dataset != "synthetic_sparse_embedding_encoder_smoke" {
		t.Fatalf("scorecard identity category=%q dataset=%q", scoreRow.Category, scoreRow.Dataset)
	}
	if scoreRow.EvidenceLevel != "smoke_synthetic_kernel_evidence" || scoreRow.QualityClaim {
		t.Fatalf("scorecard evidence_level=%q quality_claim=%v", scoreRow.EvidenceLevel, scoreRow.QualityClaim)
	}
	if scoreRow.RuntimeSeqLen != 32 || scoreRow.ThirtyTwoKPreflightStatus != "pass" || !scoreRow.ThirtyTwoKPreflightOnly {
		t.Fatalf("scorecard 32k/runtime fields: %+v", scoreRow)
	}
	if scoreRow.Preflight32768ScoreFraction <= 0 || scoreRow.Preflight32768ScoreFraction > 0.2 || !scoreRow.Preflight32768Subquadratic {
		t.Fatalf("scorecard 32768 preflight fraction=%.9g subq=%v", scoreRow.Preflight32768ScoreFraction, scoreRow.Preflight32768Subquadratic)
	}
	if scoreRow.PreflightGateStatus != "pass" {
		t.Fatalf("scorecard preflight gate=%q, want pass", scoreRow.PreflightGateStatus)
	}
	if scoreRow.RequestedBackend != "host" || scoreRow.ActualBackend != "host_reference" || scoreRow.RuntimeBackend != "host_reference" {
		t.Fatalf("scorecard backend metadata requested=%q actual=%q runtime=%q", scoreRow.RequestedBackend, scoreRow.ActualBackend, scoreRow.RuntimeBackend)
	}
	if scoreRow.CUDAAvailable || scoreRow.CUDAEvidenceStatus != "not_requested" || scoreRow.DeviceExecution {
		t.Fatalf("scorecard cuda/device metadata: %+v", scoreRow)
	}
	if !scoreRow.DenseKVMaterialized || scoreRow.KVDecode != "host_reference_decode" {
		t.Fatalf("scorecard kv metadata dense=%v decode=%q", scoreRow.DenseKVMaterialized, scoreRow.KVDecode)
	}
	if scoreRow.Bits != 4 || scoreRow.QuantizerSeed != 5581486560434873699 || scoreRow.EmbeddingDim != 8 || scoreRow.EmbeddingSHA256 == "" {
		t.Fatalf("scorecard embedding/quant metadata: %+v", scoreRow)
	}
	if scoreRow.ParityStatus != "pass" || !scoreRow.ParityBackendVsHostPassed || scoreRow.ParityBackendVsHostMaxAbsError != 0 || scoreRow.ParityBackendVsHostMSE != 0 || scoreRow.ParityBackendVsHostCosineSimilarity < 0.999999 {
		t.Fatalf("scorecard parity metadata: %+v", scoreRow)
	}
	if scoreRow.ParityDiagnosticsStatus != "computed" {
		t.Fatalf("scorecard diagnostics status=%q, want computed", scoreRow.ParityDiagnosticsStatus)
	}
	if scoreRow.SourceManifest != manifestPath || !strings.Contains(scoreRow.ClaimBoundary, "not retrieval quality") {
		t.Fatalf("scorecard source/claim boundary source=%q claim=%q", scoreRow.SourceManifest, scoreRow.ClaimBoundary)
	}
	scorecardTSV, err := os.ReadFile(filepath.Join(matches[0], "scorecard.tsv"))
	if err != nil {
		t.Fatalf("read scorecard TSV: %v", err)
	}
	scorecardLines := strings.Split(strings.TrimSpace(string(scorecardTSV)), "\n")
	if len(scorecardLines) != 2 {
		t.Fatalf("scorecard TSV lines = %d, want 2", len(scorecardLines))
	}
	if got, want := len(strings.Split(scorecardLines[0], "\t")), len(strings.Split(scorecardLines[1], "\t")); got != want {
		t.Fatalf("scorecard TSV columns header=%d row=%d", got, want)
	}
	if !strings.Contains(scorecardLines[1], "\tsmoke_synthetic_kernel_evidence\tfalse\t") {
		t.Fatalf("scorecard TSV row missing evidence boundary:\n%s", scorecardLines[1])
	}
}
