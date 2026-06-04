package main

import (
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	eosruntime "m31labs.dev/eos/runtime"
	mll "m31labs.dev/mll"
)

func TestRunGraphPrintsSourceJSON(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	output := captureRunOutput(t, []string{"graph", "--format", "json", srcPath})
	var payload struct {
		GraphVersion int    `json:"graph_version"`
		InputKind    string `json:"input_kind"`
		Module       string `json:"module"`
		Counts       struct {
			SourceDecls      int `json:"source_decls"`
			ArtifactKernels  int `json:"artifact_kernels"`
			KernelSourceVars int `json:"kernel_source_variants"`
		} `json:"counts"`
		Artifact struct {
			Name    string `json:"name"`
			Kernels []struct {
				Name     string `json:"name"`
				Variants []struct {
					Backend     string `json:"backend"`
					SourceBytes int    `json:"source_bytes"`
				} `json:"variants"`
			} `json:"kernels"`
		} `json:"artifact"`
	}
	if err := json.Unmarshal([]byte(output), &payload); err != nil {
		t.Fatalf("unmarshal graph output: %v\n%s", err, output)
	}
	if payload.GraphVersion != 1 || payload.InputKind != "source" || payload.Module != "embed" {
		t.Fatalf("unexpected graph identity: %+v", payload)
	}
	if payload.Counts.SourceDecls == 0 || payload.Counts.ArtifactKernels == 0 || payload.Counts.KernelSourceVars == 0 {
		t.Fatalf("graph counts missing compiler structure: %+v", payload.Counts)
	}
	if len(payload.Artifact.Kernels) == 0 || len(payload.Artifact.Kernels[0].Variants) == 0 {
		t.Fatalf("graph output missing kernel variant summary: %+v", payload.Artifact)
	}
	if payload.Artifact.Kernels[0].Variants[0].SourceBytes == 0 {
		t.Fatalf("variant source byte count was not recorded: %+v", payload.Artifact.Kernels[0].Variants[0])
	}
}

func TestRunKernelsExtractsBackendSources(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	outDir := filepath.Join(dir, "kernels")
	output := captureRunOutput(t, []string{"kernels", "--backend", "webgpu", "--out", outDir, srcPath})
	if !strings.Contains(output, "wrote ") || !strings.Contains(output, outDir) {
		t.Fatalf("unexpected kernels output:\n%s", output)
	}
	data, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read kernel manifest: %v", err)
	}
	var manifest struct {
		Module            string `json:"module"`
		KernelSourceCount int    `json:"kernel_source_count"`
		Kernels           []struct {
			Backend     string `json:"backend"`
			SourceFile  string `json:"source_file"`
			SourceBytes int    `json:"source_bytes"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal kernel manifest: %v\n%s", err, data)
	}
	if manifest.Module != "embed" || manifest.KernelSourceCount == 0 || len(manifest.Kernels) == 0 {
		t.Fatalf("unexpected kernel manifest: %+v", manifest)
	}
	for _, kernel := range manifest.Kernels {
		if kernel.Backend != "webgpu" {
			t.Fatalf("backend filter leaked variant: %+v", kernel)
		}
		sourcePath := filepath.Join(outDir, kernel.SourceFile)
		source, err := os.ReadFile(sourcePath)
		if err != nil {
			t.Fatalf("read extracted source %q: %v", sourcePath, err)
		}
		if !strings.Contains(string(source), "@compute") || kernel.SourceBytes != len(source) {
			t.Fatalf("unexpected extracted WGSL source %q", sourcePath)
		}
	}
}

func TestRunKernelsValidateRecordsPrismChecks(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	outDir := filepath.Join(dir, "kernels")
	captureRunOutput(t, []string{"kernels", "--backend", "webgpu", "--validate", "--out", outDir, srcPath})
	data, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read kernel manifest: %v", err)
	}
	var manifest struct {
		Kernels []struct {
			Validation *struct {
				EntryChecked bool   `json:"entry_checked"`
				ToolSkipped  bool   `json:"tool_skipped"`
				ToolError    string `json:"tool_error"`
			} `json:"validation"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal kernel manifest: %v\n%s", err, data)
	}
	if len(manifest.Kernels) == 0 || manifest.Kernels[0].Validation == nil {
		t.Fatalf("validation metadata missing: %+v", manifest)
	}
	if !manifest.Kernels[0].Validation.EntryChecked {
		t.Fatalf("Prism entry check was not recorded: %+v", manifest.Kernels[0].Validation)
	}
}

func TestRunCompileBundleWritesInspectionArtifacts(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	outPath := filepath.Join(dir, "embed.mll")
	bundleDir := filepath.Join(dir, "bundle")
	output := captureRunOutput(t, []string{"compile", "--bundle", bundleDir, srcPath, outPath})
	for _, want := range []string{"bundle: " + bundleDir, "compiled "} {
		if !strings.Contains(output, want) {
			t.Fatalf("compile bundle output missing %q\noutput:\n%s", want, output)
		}
	}
	for _, path := range []string{
		filepath.Join(bundleDir, "manifest.json"),
		filepath.Join(bundleDir, "source.eos"),
		filepath.Join(bundleDir, "artifact.mll"),
		filepath.Join(bundleDir, "graph.json"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected bundle file %q: %v", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(bundleDir, "kernels")); err != nil {
		t.Fatalf("expected kernels dir: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(bundleDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read bundle manifest: %v", err)
	}
	var manifest struct {
		BundleVersion     int    `json:"bundle_version"`
		Module            string `json:"module"`
		ArtifactPath      string `json:"artifact_path"`
		KernelSourceCount int    `json:"kernel_source_count"`
		KernelSources     []struct {
			SourceFile string `json:"source_file"`
		} `json:"kernel_sources"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal bundle manifest: %v\n%s", err, data)
	}
	if manifest.BundleVersion != 1 || manifest.Module != "embed" || manifest.ArtifactPath != outPath || manifest.KernelSourceCount == 0 {
		t.Fatalf("unexpected bundle manifest: %+v", manifest)
	}
	if len(manifest.KernelSources) == 0 || !strings.HasPrefix(manifest.KernelSources[0].SourceFile, "kernels/") {
		t.Fatalf("bundle kernel sources should be manifest-relative: %+v", manifest.KernelSources)
	}
}

func TestRunDoctorReportsRuntimeFacts(t *testing.T) {
	output := captureRunOutput(t, []string{"doctor"})
	for _, want := range []string{
		"artifact schema:",
		"go: ",
		"backends:",
		"cuda",
		"webgpu",
		"tools:",
		"env:",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("doctor output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestFormatTrainThroughputIncludesExamplePairAndStepRates(t *testing.T) {
	summary := eosruntime.EmbeddingTrainRunSummary{
		Workload: eosruntime.EmbeddingTrainWorkload{
			ActualTotalExamples: 100,
			ActualTotalPairs:    10000,
			ActualTrainExamples: 80,
			ActualTrainPairs:    8000,
			ActualEvalExamples:  20,
			ActualEvalPairs:     2000,
		},
		Elapsed:       10 * time.Second,
		TrainDuration: 4 * time.Second,
		EvalDuration:  2 * time.Second,
		StepsRun:      8,
	}

	output := formatTrainThroughput(summary)
	for _, want := range []string{
		"elapsed=10s",
		"examples/s=10.00",
		"pairs/s=1000.00",
		"train_examples/s=20.00",
		"train_pairs/s=2000.00",
		"eval_examples/s=10.00",
		"eval_pairs/s=1000.00",
		"optimizer_steps/s=2.00",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("throughput output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunInitTrainCreatesTrainingPackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	for _, candidate := range []string{
		eosruntime.DefaultWeightFilePath(path),
		eosruntime.DefaultMemoryPlanPath(path),
		eosruntime.DefaultEmbeddingTrainManifestPath(path),
		eosruntime.DefaultEmbeddingCheckpointPath(path),
		eosruntime.DefaultEmbeddingTrainProfilePath(path),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
}

func TestRunInitTrainAppliesTrainingConfigWithDefaultManifest(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", "--lr", "0.0125", "--weight-decay", "0.001", "--contrastive-loss", "infonce", "--temperature", "0.05", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.LearningRate != 0.0125 {
		t.Fatalf("learning rate = %f, want 0.0125", checkpoint.Config.LearningRate)
	}
	if checkpoint.Config.WeightDecay != 0.001 {
		t.Fatalf("weight decay = %f, want 0.001", checkpoint.Config.WeightDecay)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature != 0.05 {
		t.Fatalf("temperature = %f, want 0.05", checkpoint.Config.Temperature)
	}
}

func TestRunInitModelCreatesDefaultEmbeddingTrainingPackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--seed", "7",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v1" {
		t.Fatalf("model name = %q, want manta-embed-v1", manifest.Name)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload initialized model package: %v", err)
	}
}

func TestRunInitModelHonorsEncoderRepeats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--encoder-repeats", "3",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.EncoderRepeats != 3 {
		t.Fatalf("encoder repeats = %d, want 3", manifest.EncoderRepeats)
	}
}

func TestRunInitMirageCreatesArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "mirage-v1.mll")
	output := captureRunOutput(t, []string{
		"init-mirage",
		"--height", "16",
		"--width", "16",
		"--latent-channels", "8",
		"--bits", "2",
		path,
	})
	for _, want := range []string{
		"initialized Mirage Image v1 module",
		"capabilities: image_ops, turboquant, training_losses, host_fallback",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("init-mirage output missing %q\noutput:\n%s", want, output)
		}
	}
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if mod.Name != "mirage_image_v1" || len(mod.EntryPoints) != 4 {
		t.Fatalf("unexpected Mirage artifact: %+v", mod)
	}
}

func TestRunInitModelTrainCorpusExportFlow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--seed", "7",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	corpus := "" +
		"ab ab cd. cd ab cd.\n" +
		"cd cd ab. ab cd ab.\n" +
		"ab cd ef. ef cd ab.\n" +
		"ef ef ab. ab ef ef.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "16", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "2", "--eval-pairs", "2", path, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained default package: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	sealedPath := eosruntime.DefaultMLLPath(path)
	if sealedPath == path {
		t.Fatalf("sealed export path reused artifact path %q", path)
	}
	if _, err := mll.ReadFile(sealedPath, mll.WithDigestVerification()); err != nil {
		t.Fatalf("read sealed default model MLL: %v", err)
	}
	sealedInspect := captureRunOutput(t, []string{"inspect", sealedPath})
	for _, want := range []string{
		"embedding manifest: embedded",
		"package: embedded sealed MLL",
		"package verify: OK",
		"embedding model: manta-embed-v1",
	} {
		if !strings.Contains(sealedInspect, want) {
			t.Fatalf("sealed inspect output missing %q\noutput:\n%s", want, sealedInspect)
		}
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("inspect trained default package after export: %v", err)
	}
}

func TestRunEmbedTextLoadsSealedMLLTokenizer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "manta-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}

	output := captureRunOutput(t, []string{"embed-text", sealedPath, "a"})
	for _, want := range []string{
		"loaded embedding \"manta-embed-v1\"",
		"tokens: 1",
		"output: result",
		"embedding: f16[4]",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("embed-text output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunEvalRetrievalWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a", "b"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "manta-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"a"}`+"\n"+
			`{"_id":"d2","text":"b"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"a"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	metricsPath := filepath.Join(dir, "retrieval.metrics.json")

	output := captureRunOutput(t, []string{"eval-retrieval", "--dataset", "tiny", "--batch-size", "2", "--metrics-json", metricsPath, sealedPath, datasetDir})
	for _, want := range []string{
		"retrieval eval: dataset=tiny",
		"quality: ndcg@10=",
		"recall@100=",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics struct {
		Schema  string `json:"schema"`
		Dataset string `json:"dataset"`
		Inputs  struct {
			Documents int `json:"documents"`
			Queries   int `json:"queries"`
		} `json:"inputs"`
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Inputs.Documents != 2 || metrics.Inputs.Queries != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestRunEvalRetrievalBM25WritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	metricsPath := filepath.Join(dir, "bm25.retrieval.metrics.json")

	output := captureRunOutput(t, []string{"eval-retrieval-bm25", "--dataset", "tiny", "--metrics-json", metricsPath, datasetDir})
	for _, want := range []string{
		"retrieval bm25: dataset=tiny backend=bm25",
		"quality: ndcg@10=1.000000",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval-bm25 output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Backend != "bm25" || metrics.Quality.NDCGAt10 != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestRunMineRetrievalHardNegativesWritesTextJSONL(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega unrelated"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")

	output := captureRunOutput(t, []string{"mine-retrieval-hard-negatives", "--dataset", "tiny", "--negatives", "1", datasetDir, outputPath})
	for _, want := range []string{
		"mined retrieval hard negatives: dataset=tiny examples=1 positives=1 negatives=1 queries=1",
		"output: " + outputPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("mine-retrieval-hard-negatives output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read hard negatives: %v", err)
	}
	if len(examples) != 1 || examples[0].Query != "alpha" || examples[0].Positive != "alpha target" || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] != "alpha distractor" {
		t.Fatalf("examples = %+v", examples)
	}
	if len(examples[0].TeacherScores) != 2 {
		t.Fatalf("teacher scores = %+v, want positive plus one negative", examples[0].TeacherScores)
	}
}

func TestRunMineRetrievalModelHardNegativesWritesTextJSONL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "alpha", "target", "distractor", "omega"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "manta-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath := filepath.Join(dir, "model-hard-negatives.jsonl")

	output := captureRunOutput(t, []string{"mine-retrieval-model-hard-negatives", "--dataset", "tiny", "--negatives", "1", "--candidate-top-k", "2", "--batch-size", "2", sealedPath, datasetDir, outputPath})
	for _, want := range []string{
		"mined model retrieval hard negatives: dataset=tiny",
		"examples=1 positives=1 negatives=1 queries=1",
		"output: " + outputPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("mine-retrieval-model-hard-negatives output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read model hard negatives: %v", err)
	}
	if len(examples) != 1 || examples[0].Query != "alpha" || examples[0].Positive != "alpha target" || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] == "alpha target" {
		t.Fatalf("examples = %+v", examples)
	}
	if len(examples[0].TeacherScores) != 2 {
		t.Fatalf("teacher scores = %+v, want positive plus one negative", examples[0].TeacherScores)
	}
}

func TestRunImportTeacherScoresWritesVectorsAndManifest(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "alpha", Positive: "target", Negatives: []string{"distractor"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	scoresPath := filepath.Join(dir, "scores.jsonl")
	if err := os.WriteFile(scoresPath, []byte(`{"source":"scifact","query":"alpha","scores":[0.9,0.1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write scores: %v", err)
	}
	outputPath := filepath.Join(dir, "with-teacher.jsonl")
	manifestPath := filepath.Join(dir, "teacher.manifest.json")

	output := captureRunOutput(t, []string{
		"import-teacher-scores",
		"--teacher-model-id", "teacher-a",
		"--teacher-revision", "rev1",
		"--score-scale", "cosine",
		"--manifest", manifestPath,
		inputPath,
		scoresPath,
		outputPath,
	})
	for _, want := range []string{
		"imported teacher scores: examples=1 updated=1",
		"teacher: model_id=teacher-a revision=rev1 score_scale=cosine",
		"output: " + outputPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("import-teacher-scores output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 2 || examples[0].TeacherScores[0] != 0.9 || examples[0].TeacherScores[1] != 0.1 {
		t.Fatalf("teacher scores = %+v", examples)
	}
	var manifest teacherScoreImportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != "manta.teacher_score_import.v1" || manifest.TeacherModelID != "teacher-a" || manifest.Updated != 1 || manifest.ExampleRows != 1 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestRunExportTeacherScoreRequestsRoundTripsThroughImport(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "nfcorpus", Query: "vitamin c", Positive: "ascorbic acid", Negatives: []string{"calcium", "zinc"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	requestPath := filepath.Join(dir, "teacher-requests.jsonl")
	manifestPath := filepath.Join(dir, "requests.manifest.json")

	output := captureRunOutput(t, []string{
		"export-teacher-score-requests",
		"--manifest", manifestPath,
		inputPath,
		requestPath,
	})
	for _, want := range []string{
		"exported teacher score requests: examples=1 exported=1 skipped_existing=0 rows=3 positive_rows=1 negative_rows=2",
		"output: " + requestPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("export-teacher-score-requests output missing %q\noutput:\n%s", want, output)
		}
	}
	data, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatalf("read requests: %v", err)
	}
	var requests []teacherScoreRequestRecord
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var record teacherScoreRequestRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("decode request %q: %v", line, err)
		}
		requests = append(requests, record)
	}
	if len(requests) != 3 || requests[0].Role != "positive" || requests[0].CandidateIndex != 0 || requests[1].Role != "negative" || requests[1].Candidate != "calcium" {
		t.Fatalf("requests = %+v", requests)
	}
	var manifest teacherScoreRequestSummary
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != "manta.teacher_score_requests.v1" || manifest.Rows != 3 || manifest.PositiveRows != 1 || manifest.NegativeRows != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}

	scorePath := filepath.Join(dir, "scores.jsonl")
	var scoreRows []string
	wantScores := []float32{0.8, 0.2, 0.1}
	for i, request := range requests {
		score := float64(wantScores[i])
		row, err := json.Marshal(teacherScoreImportRecord{
			Source:    request.Source,
			Query:     request.Query,
			Candidate: request.Candidate,
			Score:     &score,
		})
		if err != nil {
			t.Fatalf("encode score row: %v", err)
		}
		scoreRows = append(scoreRows, string(row))
	}
	if err := os.WriteFile(scorePath, []byte(strings.Join(scoreRows, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write scores: %v", err)
	}
	outputPath := filepath.Join(dir, "with-teacher.jsonl")
	_ = captureRunOutput(t, []string{"import-teacher-scores", inputPath, scorePath, outputPath})
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != len(wantScores) {
		t.Fatalf("examples = %+v", examples)
	}
	for i, want := range wantScores {
		if examples[0].TeacherScores[i] != want {
			t.Fatalf("teacher score %d = %f, want %f", i, examples[0].TeacherScores[i], want)
		}
	}
}

func TestRunExportTeacherScoreRequestsMissingOnly(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "q1", Positive: "p1", Negatives: []string{"n1"}, TeacherScores: []float32{0.8, 0.1}},
		{Source: "fiqa", Query: "q2", Positive: "p2", Negatives: []string{"n2"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	requestPath := filepath.Join(dir, "missing-requests.jsonl")

	output := captureRunOutput(t, []string{
		"export-teacher-score-requests",
		"--missing-only",
		inputPath,
		requestPath,
	})
	if !strings.Contains(output, "exported teacher score requests: examples=2 exported=1 skipped_existing=1 rows=2") {
		t.Fatalf("unexpected output:\n%s", output)
	}
	data, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatalf("read requests: %v", err)
	}
	if got := strings.Count(strings.TrimSpace(string(data)), "\n") + 1; got != 2 {
		t.Fatalf("request rows = %d, want 2\n%s", got, data)
	}
}

func TestRunImportTeacherScoresMatchesCandidateRows(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "nfcorpus", Query: "vitamin c", Positive: "ascorbic acid", Negatives: []string{"calcium", "zinc"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	scoresPath := filepath.Join(dir, "scores.jsonl")
	if err := os.WriteFile(scoresPath, []byte(
		`{"query":"vitamin c","candidate":"ascorbic acid","score":0.8}`+"\n"+
			`{"query":"vitamin c","candidate":"calcium","score":0.2}`+"\n"+
			`{"query":"vitamin c","candidate":"zinc","score":0.1}`+"\n"), 0o644); err != nil {
		t.Fatalf("write scores: %v", err)
	}
	outputPath := filepath.Join(dir, "with-teacher.jsonl")

	_ = captureRunOutput(t, []string{"import-teacher-scores", inputPath, scoresPath, outputPath})
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 3 {
		t.Fatalf("examples = %+v", examples)
	}
	want := []float32{0.8, 0.2, 0.1}
	for i, score := range want {
		if examples[0].TeacherScores[i] != score {
			t.Fatalf("teacher score %d = %f, want %f", i, examples[0].TeacherScores[i], score)
		}
	}
}

func TestRunScoreTeacherHardNegativesWritesScoresAndManifest(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "teacher.mll")
	if err := run([]string{
		"init-model",
		"--name", "tiny-teacher",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		artifactPath,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a", "b", "c", "d", "e", "f"},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(artifactPath); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "abc", Positive: "abc", Negatives: []string{"def"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	outputPath := filepath.Join(dir, "scored.jsonl")
	manifestPath := filepath.Join(dir, "teacher-score.manifest.json")

	output := captureRunOutput(t, []string{
		"score-teacher-hard-negatives",
		"--batch-size", "2",
		"--manifest", manifestPath,
		"--teacher-revision", "local",
		artifactPath,
		inputPath,
		outputPath,
	})
	for _, want := range []string{
		"scored teacher hard negatives: examples=1 updated=1",
		"teacher: model_id=tiny-teacher revision=local score_scale=cosine",
		"output: " + outputPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("score-teacher-hard-negatives output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 2 {
		t.Fatalf("teacher scores = %+v", examples)
	}
	for i, score := range examples[0].TeacherScores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			t.Fatalf("teacher score %d is not finite: %f", i, score)
		}
	}
	var manifest teacherHardNegativeScoreSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != "manta.teacher_hard_negative_score.v1" || manifest.TeacherModelID != "tiny-teacher" || manifest.Updated != 1 || manifest.BatchSize != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestRunAuditTeacherScoresWritesSummary(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "q1", Positive: "p1", Negatives: []string{"n1", "n2"}, TeacherScores: []float32{0.9, 0.1, 0.2}},
		{Source: "fiqa", Query: "q2", Positive: "p2", Negatives: []string{"n3"}, TeacherScores: []float32{0.1, 0.8}},
		{Source: "fiqa", Query: "q3", Positive: "p3", Negatives: []string{"n4"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	summaryPath := filepath.Join(dir, "teacher-audit.json")

	output := captureRunOutput(t, []string{
		"audit-teacher-scores",
		"--temperature", "1.5",
		inputPath,
		summaryPath,
	})
	for _, want := range []string{
		"audited teacher scores: examples=3 scored=2 missing=1",
		"positive_top1_rate=0.500000",
		"summary: " + summaryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("audit-teacher-scores output missing %q\noutput:\n%s", want, output)
		}
	}
	var summary teacherScoreAuditSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.Schema != "manta.teacher_score_audit.v1" || summary.Mode != "text" || summary.Examples != 3 || summary.ScoredExamples != 2 || summary.MissingExamples != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if summary.Candidates != 7 || summary.ScoredCandidates != 5 || summary.PositiveTop1 != 1 {
		t.Fatalf("summary counts = %+v", summary)
	}
	if math.Abs(summary.PositiveTop1Rate-0.5) > 0.000001 || math.Abs(summary.PositiveMeanRank-1.5) > 0.000001 {
		t.Fatalf("summary ranks = %+v", summary)
	}
	if summary.MeanNormalizedEntropy <= 0 || summary.MeanNormalizedEntropy > 1 {
		t.Fatalf("summary normalized entropy = %f", summary.MeanNormalizedEntropy)
	}
	fiqa := summary.Sources["fiqa"]
	if fiqa.Examples != 2 || fiqa.ScoredExamples != 1 || fiqa.MissingExamples != 1 || fiqa.PositiveTop1 != 0 {
		t.Fatalf("fiqa source summary = %+v", fiqa)
	}
}

func TestRunPlanSparseAttentionWritesBudgetReport(t *testing.T) {
	dir := t.TempDir()
	reportPath := filepath.Join(dir, "sparse-plan.json")
	output := captureRunOutput(t, []string{
		"plan-sparse-attention",
		"--key-lens", "64,256",
		"--query-dim", "16",
		"--value-dim", "32",
		"--top-k", "4",
		"--route-top-blocks", "2",
		"--bits", "4",
		"--require-subquadratic",
		"--max-score-fraction", "0.5",
		"--json", reportPath,
	})
	for _, want := range []string{
		"key_len\trouting",
		"64\tblock_anchor\t8\t2\t4\t16\t24\t0.375000",
		"256\tblock_anchor\t16\t2\t4\t32\t48\t0.187500",
		"gate=pass",
		"json: " + reportPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-sparse-attention output missing %q\noutput:\n%s", want, output)
		}
	}
	var report sparseAttentionPlanReport
	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if report.Schema != "manta.sparse_attention_plan.v1" || !report.Gate.Passed || report.Gate.SubquadraticRows != 2 {
		t.Fatalf("report gate = %+v schema=%q", report.Gate, report.Schema)
	}
	if len(report.Rows) != 2 {
		t.Fatalf("rows = %d", len(report.Rows))
	}
	first := report.Rows[0]
	if first.KeyLen != 64 || first.CandidateKeyBudget != 16 || first.EstimatedScoreCountPerQuery != 24 {
		t.Fatalf("first row = %+v", first)
	}
	if first.TurboQuantKVBytes != 2048 || math.Abs(first.TurboQuantCompressionRatio-3) > 0.000001 {
		t.Fatalf("first row TurboQuant memory = %+v", first)
	}
	if report.Gate.ScoreAlpha <= 0 || report.Gate.ScoreAlpha >= 1 {
		t.Fatalf("score alpha = %f, want sublinear", report.Gate.ScoreAlpha)
	}
}

func TestRunPlanSparseAttentionCanFailGate(t *testing.T) {
	output, err := captureRunOutputAndError(t, []string{
		"plan-sparse-attention",
		"--key-lens", "64",
		"--exact",
		"--require-subquadratic",
	})
	if err == nil {
		t.Fatalf("expected gate failure\noutput:\n%s", output)
	}
	if !strings.Contains(err.Error(), "not subquadratic") || !strings.Contains(output, "gate=fail") {
		t.Fatalf("unexpected failure err=%v output:\n%s", err, output)
	}
}

func TestRunCompareRetrievalMetricsCanRequireBaselineWin(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.retrieval.metrics.json")
	baselinePath := filepath.Join(dir, "baseline.retrieval.metrics.json")
	current := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "tiny",
		Backend: "cuda",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10: 0.30,
		},
	}
	baseline := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "tiny",
		Backend: "bm25",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10: 0.25,
		},
	}
	currentData, err := json.Marshal(current)
	if err != nil {
		t.Fatalf("marshal current: %v", err)
	}
	baselineData, err := json.Marshal(baseline)
	if err != nil {
		t.Fatalf("marshal baseline: %v", err)
	}
	if err := os.WriteFile(currentPath, currentData, 0o644); err != nil {
		t.Fatalf("write current: %v", err)
	}
	if err := os.WriteFile(baselinePath, baselineData, 0o644); err != nil {
		t.Fatalf("write baseline: %v", err)
	}

	output := captureRunOutput(t, []string{"compare-retrieval-metrics", "--require-win", currentPath, baselinePath})
	for _, want := range []string{
		"current: " + currentPath + " backend=cuda dataset=tiny",
		"baseline: " + baselinePath + " backend=bm25 dataset=tiny",
		"target: ndcg_at_10=0.3 baseline=0.25 required=0.25 ratio=1.2",
		"retrieval baseline gate: PASS",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("compare-retrieval-metrics output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedFitsContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", "--lr", "0.003", "--contrastive-loss", "infonce", "--temperature", "0.07", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.LearningRate < 0.00299 || checkpoint.Config.LearningRate > 0.00301 {
		t.Fatalf("learning rate = %f, want 0.003", checkpoint.Config.LearningRate)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature < 0.06999 || checkpoint.Config.Temperature > 0.07001 {
		t.Fatalf("temperature = %f, want 0.07", checkpoint.Config.Temperature)
	}
}

func TestRunRenameEmbedRewritesPackageIdentity(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	renamedPath := filepath.Join(t.TempDir(), "manta-embed-v1.mll")

	if err := run([]string{"rename-embed", "--name", "manta-embed-v1", path, renamedPath}); err != nil {
		t.Fatalf("run rename-embed: %v", err)
	}

	mod, err := eosartifact.ReadFile(renamedPath)
	if err != nil {
		t.Fatalf("read renamed artifact: %v", err)
	}
	if mod.Name != "manta-embed-v1" {
		t.Fatalf("module name = %q, want manta-embed-v1", mod.Name)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v1" {
		t.Fatalf("manifest name = %q, want manta-embed-v1", manifest.Name)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed checkpoint: %v", err)
	}
	if checkpoint.Manifest.Name != "manta-embed-v1" {
		t.Fatalf("checkpoint manifest name = %q, want manta-embed-v1", checkpoint.Manifest.Name)
	}
	if _, err := os.Stat(eosruntime.DefaultTokenizerPath(renamedPath)); err != nil {
		t.Fatalf("renamed tokenizer sidecar missing: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(renamedPath); err != nil {
		t.Fatalf("reload renamed package: %v", err)
	}
}

func TestRunTrainEmbedPlanOnlyShowsWorkload(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	output := captureRunOutput(t, []string{"train-embed", "--plan-only", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"planned workload:",
		"train=4 contrastive examples",
		"steps/epoch=2",
		"train_pairs/epoch=8",
		"eval=4 contrastive examples",
		"eval_pairs/pass=16",
		"pairs(planned=80 actual=0)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyUsesSingleContrastiveDataset(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", path, evalPath})
	for _, want := range []string{
		"evaluated package",
		"epochs: 0",
		"run_steps: 0",
		"final eval:",
		"train=0 contrastive examples",
		"eval=2 contrastive examples",
		"pairs(planned=4 actual=4)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyUsesSingleTextPairDataset(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", path, evalPath})
	for _, want := range []string{
		"evaluated package",
		"tokenizer:",
		"epochs: 0",
		"run_steps: 0",
		"final eval:",
		"pairs=2",
		"train=0 pairwise examples",
		"eval=2 pairwise examples",
		"pairs(planned=2 actual=2)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only text output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyWritesMetricsJSON(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	metricsPath := filepath.Join(t.TempDir(), "metrics.json")

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", "--metrics-json", metricsPath, path, evalPath})
	if !strings.Contains(output, "metrics: "+metricsPath) {
		t.Fatalf("eval-only output missing metrics path %q\noutput:\n%s", metricsPath, output)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics json: %v", err)
	}
	var got struct {
		Schema  string `json:"schema"`
		Command string `json:"command"`
		Mode    string `json:"mode"`
		Summary struct {
			StepsRun int `json:"steps_run"`
		} `json:"summary"`
		FinalEval *struct {
			PairCount int     `json:"pair_count"`
			Top1      float32 `json:"top1_accuracy"`
			MRR       float32 `json:"mean_reciprocal_rank"`
		} `json:"final_eval"`
		Workload struct {
			ActualEvalPairs int64 `json:"actual_eval_pairs"`
		} `json:"workload"`
		Throughput struct {
			EvalPairsPerSecond float64 `json:"eval_pairs_per_second"`
		} `json:"throughput"`
		ProfileDelta struct {
			OptimizerUpdates int64 `json:"optimizer_updates"`
		} `json:"profile_delta"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode metrics json: %v\n%s", err, string(data))
	}
	if got.Schema != "manta.embedding_train_metrics.v1" || got.Command != "train-embed" || got.Mode != "eval" {
		t.Fatalf("unexpected metrics identity: %+v", got)
	}
	if got.Summary.StepsRun != 0 {
		t.Fatalf("steps_run = %d, want 0", got.Summary.StepsRun)
	}
	if got.FinalEval == nil || got.FinalEval.PairCount != 4 {
		t.Fatalf("final_eval = %+v, want pair_count=4", got.FinalEval)
	}
	if got.FinalEval.Top1 <= 0 || got.FinalEval.MRR <= 0 {
		t.Fatalf("expected ranking metrics in JSON, got final_eval %+v", *got.FinalEval)
	}
	if got.Workload.ActualEvalPairs != 4 {
		t.Fatalf("actual_eval_pairs = %d, want 4", got.Workload.ActualEvalPairs)
	}
	if got.Throughput.EvalPairsPerSecond <= 0 {
		t.Fatalf("eval_pairs_per_second = %f, want positive", got.Throughput.EvalPairsPerSecond)
	}
	if got.ProfileDelta.OptimizerUpdates != 0 {
		t.Fatalf("optimizer_updates = %d, want 0", got.ProfileDelta.OptimizerUpdates)
	}
}

func TestRunCompareTrainMetricsReportsCurrentAndBaselineDeltas(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.metrics.json")
	baselinePath := filepath.Join(dir, "baseline.metrics.json")
	current := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			Top1Accuracy:       0.9,
			Top5Accuracy:       0.98,
			Top10Accuracy:      1,
			MeanReciprocalRank: 0.95,
			ROCAUC:             0.73,
			ScoreMargin:        0.12,
			Loss:               0.11,
			MeanPositiveRank:   1.1,
			PairCount:          128,
		},
		Throughput: trainThroughputJSON{
			TrainPairsPerSecond:     120000,
			EvalPairsPerSecond:      300,
			OptimizerStepsPerSecond: 0.15,
			PairsPerSecond:          150000,
			ElapsedSeconds:          10,
		},
		Accelerators: trainAcceleratorsJSON{Forward: "cuda", Optimizer: "cuda", Activation: "cuda", Contrastive: "cuda"},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   100,
			MatMulRunDownloadMB: 80,
			OptimizerUpdates:    4,
			ActivationCalls:     3,
			ContrastiveCalls:    2,
		},
	}
	baseline := current
	baseline.Artifact = "baseline.mll"
	baseline.FinalEval = &evalMetricsJSON{
		Top1Accuracy:       0.8,
		Top5Accuracy:       0.95,
		Top10Accuracy:      0.99,
		MeanReciprocalRank: 0.9,
		ROCAUC:             0.7,
		ScoreMargin:        0.10,
		Loss:               0.13,
		MeanPositiveRank:   1.3,
		PairCount:          128,
	}
	baseline.Throughput.TrainPairsPerSecond = 100000
	baseline.ProfileDelta.MatMulRuns = 1500
	writeMetricsJSONForTest(t, currentPath, current)
	writeMetricsJSONForTest(t, baselinePath, baseline)

	output := captureRunOutput(t, []string{"compare-train-metrics", currentPath, baselinePath})
	for _, want := range []string{
		"identity: schema=manta.embedding_train_metrics.v1 command=train-embed mode=eval artifact=current.mll",
		"quality: top1=0.900000",
		"throughput: train_pairs/s=120000.00",
		"accelerators: forward=cuda optimizer=cuda activation=cuda contrastive=cuda",
		"profile_delta: matmul_runs=1000",
		"baseline: " + baselinePath,
		"quality_delta: top1=+0.100000",
		"throughput_delta: train_pairs/s=+20000.00",
		"profile_delta_delta: matmul_runs=-500",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("compare output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsReportsDeviceBackedEfficiency(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Workload: trainWorkloadJSON{
			ActualTrainPairs: 100000,
		},
		Throughput: trainThroughputJSON{
			ElapsedSeconds:          90,
			TrainSeconds:            80,
			EvalSeconds:             10,
			TrainPairsPerSecond:     120000,
			EvalPairsPerSecond:      50000,
			OptimizerStepsPerSecond: 0.5,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "cuda",
			Optimizer:   "cuda",
			Activation:  "host",
			Contrastive: "cuda",
		},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   500,
			MatMulRunDownloadMB: 250,
			OptimizerUpdates:    10,
			OptimizerSyncs:      20,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"metrics: " + metricsPath,
		"backend: forward=cuda optimizer=cuda activation=host contrastive=cuda",
		"efficiency: matmul_runs/update=100.00 pairs/matmul_run=100.00 optimizer_syncs/update=2.00",
		"transfer: total_mb=750.00 mb/matmul_run=0.7500 kb/pair=7.6800",
		"finding: ok production-critical accelerators are device-backed",
		"finding: note activation accelerator is host",
		"diagnosis: OK warnings=0 notes=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsWarnsOnHostFallbacks(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Workload: trainWorkloadJSON{
			ActualTrainPairs: 100,
		},
		Throughput: trainThroughputJSON{
			ElapsedSeconds:      10,
			TrainSeconds:        10,
			TrainPairsPerSecond: 0,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "host",
			Optimizer:   "host",
			Activation:  "host",
			Contrastive: "host",
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"finding: warn production-critical accelerators include host fallback: forward=host optimizer=host contrastive=host",
		"finding: warn training run recorded zero optimizer updates",
		"finding: warn training pairs were processed but train_pairs/s is zero",
		"diagnosis: WARN warnings=3 notes=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("warning diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsWarnsOnMissingOptimizerStepRate(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Throughput: trainThroughputJSON{
			TrainSeconds:            2,
			TrainPairsPerSecond:     500,
			OptimizerStepsPerSecond: 0,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "cuda",
			Optimizer:   "cuda",
			Activation:  "cuda",
			Contrastive: "cuda",
		},
		ProfileDelta: trainProfileDeltaJSON{
			OptimizerUpdates: 2,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"finding: warn optimizer updates were recorded but optimizer_steps/s is zero",
		"diagnosis: WARN warnings=1 notes=0",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("optimizer-rate diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsChecksThresholdFile(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			Top1Accuracy:       0.9,
			Top5Accuracy:       0.98,
			Top10Accuracy:      1,
			MeanReciprocalRank: 0.95,
			ROCAUC:             0.73,
			ScoreMargin:        0.12,
			Loss:               0.11,
			MeanPositiveRank:   1.1,
		},
		Throughput: trainThroughputJSON{
			TrainPairsPerSecond:     120000,
			OptimizerStepsPerSecond: 0.15,
		},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   100,
			MatMulRunDownloadMB: 80,
			OptimizerUpdates:    0,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)
	thresholds := "" +
		"EOS_MIN_MRR=0.90\n" +
		"EOS_MIN_TOP1=0.80\n" +
		"EOS_MAX_MEAN_RANK=1.20\n" +
		"EOS_MIN_TRAIN_PAIRS_PER_SEC=100000\n" +
		"EOS_MAX_MATMUL_RUNS=2000\n"
	if err := os.WriteFile(thresholdsPath, []byte(thresholds), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-train-metrics", "--thresholds", thresholdsPath, metricsPath})
	for _, want := range []string{
		"metrics: " + metricsPath,
		"thresholds: " + thresholdsPath,
		"scope: all",
		"pass: mrr=0.95 >= 0.9",
		"pass: train_pairs/s=120000 >= 100000",
		"pass: matmul_runs=1000 <= 2000",
		"gate: PASS checks=5",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsChecksEvalOnlyOptimizerUpdates(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "eval.metrics.json")
	metrics := trainMetricsJSON{
		Schema:       "manta.embedding_train_metrics.v1",
		Command:      "train-embed",
		Mode:         "eval",
		Artifact:     "current.mll",
		ProfileDelta: trainProfileDeltaJSON{OptimizerUpdates: 0},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"gate-train-metrics", "--scope", "eval-only", metricsPath})
	for _, want := range []string{
		"scope: eval-only",
		"pass: optimizer_updates=0 == 0 (eval-only)",
		"gate: PASS checks=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsFailsMissedThreshold(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			MeanReciprocalRank: 0.5,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)
	if err := os.WriteFile(thresholdsPath, []byte("EOS_MIN_MRR=0.90\n"), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output, err := captureRunOutputAndError(t, []string{"gate-train-metrics", "--thresholds", thresholdsPath, metricsPath})
	if err == nil {
		t.Fatalf("gate unexpectedly passed\noutput:\n%s", output)
	}
	for _, want := range []string{
		"fail: mrr=0.5 >= 0.9",
		"gate: FAIL checks=1 failed=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("failed gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateRetrievalMetricsChecksDatasetThresholds(t *testing.T) {
	clearRetrievalMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "scifact.retrieval.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "scifact",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10:    0.23,
			MRRAt10:     0.22,
			RecallAt10:  0.32,
			RecallAt100: 0.60,
		},
		Throughput: eosruntime.RetrievalEvalThroughput{
			ScoresPerSecond: 8000000,
		},
	}
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal retrieval metrics: %v", err)
	}
	if err := os.WriteFile(metricsPath, data, 0o644); err != nil {
		t.Fatalf("write retrieval metrics: %v", err)
	}
	thresholds := "" +
		"EOS_MIN_RETRIEVAL_NDCG10_SCIFACT=0.22843\n" +
		"EOS_MIN_RETRIEVAL_MRR10_SCIFACT=0.213567\n" +
		"EOS_MIN_RETRIEVAL_SCORES_PER_SEC=7000000\n"
	if err := os.WriteFile(thresholdsPath, []byte(thresholds), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-retrieval-metrics", "--thresholds", thresholdsPath, metricsPath})
	for _, want := range []string{
		"dataset: scifact",
		"pass: ndcg_at_10=0.23 >= 0.22843",
		"pass: mrr_at_10=0.22 >= 0.213567",
		"pass: scores/s=8e+06 >= 7e+06",
		"retrieval gate: PASS checks=3",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("retrieval gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateRetrievalMetricsAllowsRoundedEquality(t *testing.T) {
	clearRetrievalMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "scifact.retrieval.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "scifact",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10: 0.22842998825189667,
		},
	}
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal retrieval metrics: %v", err)
	}
	if err := os.WriteFile(metricsPath, data, 0o644); err != nil {
		t.Fatalf("write retrieval metrics: %v", err)
	}
	if err := os.WriteFile(thresholdsPath, []byte("EOS_MIN_RETRIEVAL_NDCG10_SCIFACT=0.228430\n"), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-retrieval-metrics", "--thresholds", thresholdsPath, metricsPath})
	if !strings.Contains(output, "retrieval gate: PASS checks=1") {
		t.Fatalf("rounded equality gate did not pass\noutput:\n%s", output)
	}
}

func clearTrainMetricGateEnv(t *testing.T) {
	t.Helper()
	for _, threshold := range trainMetricThresholds {
		t.Setenv(threshold.Env, "")
	}
}

func clearRetrievalMetricGateEnv(t *testing.T) {
	t.Helper()
	for _, threshold := range retrievalMetricThresholds {
		t.Setenv(threshold.Env, "")
		t.Setenv(threshold.Env+"_SCIFACT", "")
	}
}

func writeMetricsJSONForTest(t *testing.T, path string, metrics trainMetricsJSON) {
	t.Helper()
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal metrics json: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write metrics json: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	examples := []eosruntime.EmbeddingTextContrastiveExample{
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
	}
	if err := eosruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingTextContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackageWithLabeledEvalPairs(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	trainData := "" +
		"{\"query\":\"ab\",\"positive\":\"ab\"}\n" +
		"{\"query\":\"cd\",\"positive\":\"cd\"}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text with labeled eval: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextPairwisePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-pairs.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-pairs.jsonl")
	trainData := "" +
		"{\"source\":\"scifact\",\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"source\":\"scifact\",\"query\":\"ab\",\"document\":\"cd\",\"label\":-1}\n" +
		"{\"source\":\"nfcorpus\",\"query\":\"cd\",\"document\":\"cd\",\"label\":1}\n" +
		"{\"source\":\"nfcorpus\",\"query\":\"cd\",\"document\":\"ab\",\"label\":-1}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text pairs: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text pairs: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--pairwise-train", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"trained package",
		"train=4 pairwise examples",
		"eval=2 pairwise examples",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("pairwise train output missing %q\noutput:\n%s", want, output)
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextHardNegativePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-pairs.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-pairs.jsonl")
	trainData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"query\":\"ab\",\"document\":\"cd\",\"label\":-1}\n" +
		"{\"query\":\"cd\",\"document\":\"cd\",\"label\":1}\n" +
		"{\"query\":\"cd\",\"document\":\"ab\",\"label\":-1}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text pairs: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text pairs: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--hard-negative-train", "--hard-negatives-per-query", "1", "--hard-negative-source-weights", "scifact=1,nfcorpus=2", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"trained package",
		"train=2 hard_negative_contrastive examples",
		"eval=2 pairwise examples",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("hard-negative train output missing %q\noutput:\n%s", want, output)
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTokenizeEmbedHardNegativeMode(t *testing.T) {
	path := writeTrainableArtifact(t)
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	inputPath := filepath.Join(t.TempDir(), "pairs.jsonl")
	outputPath := filepath.Join(t.TempDir(), "hard.tokens.jsonl")
	inputData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"query\":\"ab\",\"document\":\"cd\",\"label\":-1}\n"
	if err := os.WriteFile(inputPath, []byte(inputData), 0o644); err != nil {
		t.Fatalf("write input pairs: %v", err)
	}

	output := captureRunOutput(t, []string{"tokenize-embed", "--mode", "hard-negative", path, inputPath, outputPath})
	if !strings.Contains(output, "tokenized hard-negative examples: 1") {
		t.Fatalf("tokenize hard-negative output unexpected:\n%s", output)
	}
	examples, err := eosruntime.ReadEmbeddingHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read tokenized hard-negative output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].NegativeTokens) != 1 {
		t.Fatalf("tokenized examples = %+v", examples)
	}
}

func TestRunInspectReadsPackageManifest(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("run inspect: %v", err)
	}
}

func TestRunInspectShowsRepeatedEncoderEmbeddingDetails(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.eos")
	artifactPath := filepath.Join(dir, "encoder_trainable_q8x2.mll")
	copyExampleFile(t, dir, "encoder_trainable_q8x2.embedding.mll")
	if err := run([]string{"compile", srcPath, artifactPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	output := captureRunOutput(t, []string{"inspect", artifactPath})
	for _, want := range []string{
		"embedding manifest:",
		"embedding model: encoder-trainable-q8x2 pooled=embed_pooled batch=embed_pooled_batch output=result/f16",
		"encoder repeats: 2",
		"tokenizer: vocab=32768 max_sequence=256",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("inspect output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunExportMLLWritesContainer(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	outPath := eosruntime.DefaultMLLPath(path)
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("expected MLL file %q: %v", outPath, err)
	}
	if _, err := mll.ReadFile(outPath, mll.WithDigestVerification()); err != nil {
		t.Fatalf("read exported MLL: %v", err)
	}
}

func TestRunTrainTokenizerWritesSiblingTokenizer(t *testing.T) {
	path := writeTrainableArtifact(t)
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(corpusPath, []byte("ab ab cd ab cd\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "12", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	tokenizerPath := eosruntime.DefaultTokenizerPath(path)
	if _, err := os.Stat(tokenizerPath); err != nil {
		t.Fatalf("expected tokenizer file %q: %v", tokenizerPath, err)
	}
	if _, err := eosruntime.ReadTokenizerFile(tokenizerPath); err != nil {
		t.Fatalf("read tokenizer file: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Tokenizer.VocabSize != 11 {
		t.Fatalf("expected manifest vocab size to track tokenizer, got %d", manifest.Tokenizer.VocabSize)
	}
}

func TestRunTokenizeEmbedWritesTokenDatasets(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}

	dir := t.TempDir()
	trainTextPath := filepath.Join(dir, "train-text.jsonl")
	evalTextPath := filepath.Join(dir, "eval-text.jsonl")
	trainTokenPath := filepath.Join(dir, "train-token.jsonl")
	evalTokenPath := filepath.Join(dir, "eval-token.jsonl")
	if err := os.WriteFile(trainTextPath, []byte("{\"query\":\"ab\",\"positive\":\"ab\"}\n"), 0o644); err != nil {
		t.Fatalf("write train text: %v", err)
	}
	evalText := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(evalTextPath, []byte(evalText), 0o644); err != nil {
		t.Fatalf("write eval text: %v", err)
	}

	if err := run([]string{"tokenize-embed", "--mode", "contrastive", path, trainTextPath, trainTokenPath}); err != nil {
		t.Fatalf("run tokenize contrastive: %v", err)
	}
	if err := run([]string{"tokenize-embed", "--mode", "pair", path, evalTextPath, evalTokenPath}); err != nil {
		t.Fatalf("run tokenize pair: %v", err)
	}
	if _, err := eosruntime.ReadEmbeddingContrastiveExamplesFile(trainTokenPath); err != nil {
		t.Fatalf("read tokenized train: %v", err)
	}
	pairs, err := eosruntime.ReadEmbeddingPairExamplesFile(evalTokenPath)
	if err != nil {
		t.Fatalf("read tokenized eval: %v", err)
	}
	if len(pairs) != 2 || pairs[0].Target != 1 || pairs[1].Target != 0 {
		t.Fatalf("tokenized eval targets = %+v", pairs)
	}
	output := captureRunOutput(t, []string{"train-embed", "--eval-only", "--no-tokenizer", path, evalTokenPath})
	if !strings.Contains(output, "evaluated package") || !strings.Contains(output, "eval=2 pairwise examples") {
		t.Fatalf("eval-only token output missing expected summary:\n%s", output)
	}
}

func TestRunMineTextPairsWritesTrainAndEvalFiles(t *testing.T) {
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	if err := run([]string{"mine-text-pairs", "--min-chars", "5", "--eval-pairs", "2", corpusPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run mine-text-pairs: %v", err)
	}
	if _, err := eosruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath); err != nil {
		t.Fatalf("read mined train pairs: %v", err)
	}
	evalSet, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
	if err != nil {
		t.Fatalf("read mined eval pairs: %v", err)
	}
	var positives, negatives int
	for _, example := range evalSet {
		if example.Target > 0 {
			positives++
		} else if example.Target == 0 {
			negatives++
		}
	}
	if positives == 0 || negatives == 0 {
		t.Fatalf("expected mined eval set to include both classes, got positives=%d negatives=%d", positives, negatives)
	}
}

func TestRunMineTextPairsThenTrainEmbedFlow(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"ab ab cd. cd ab cd.\n" +
		"cd cd ab. ab cd ab.\n" +
		"ab cd ef. ef cd ab.\n" +
		"ef ef ab. ab ef ef.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "16", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	if err := run([]string{"mine-text-pairs", "--min-chars", "2", "--eval-pairs", "2", corpusPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run mine-text-pairs: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed from mined text: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusFlow(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "20", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "5", "--eval-pairs", "2", path, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("expected tokenizer file: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultMinedTrainPairsPath(path)); err != nil {
		t.Fatalf("expected mined train pairs: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultMinedEvalPairsPath(path)); err != nil {
		t.Fatalf("expected mined eval pairs: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusRepeatedEncoderExampleFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.eos")
	artifactPath := filepath.Join(dir, "encoder_trainable_q8x2.mll")
	copyExampleFile(t, dir, "encoder_trainable_q8x2.embedding.mll")

	if err := run([]string{"compile", srcPath, artifactPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=16", "--dim", "H=32", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	corpus := "" +
		"Eos trains and serves compact transformer encoders.\n" +
		"CorkScrewDB needs a small default model with strong retrieval quality.\n" +
		"Quantized embeddings should be fast, portable, and cheap to ship.\n" +
		"Native CUDA training should reuse weights, activations, and optimizer state.\n" +
		"Metal parity matters later, but the package path must already be clean.\n" +
		"Attention, residuals, and layernorm make the encoder more realistic.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "48", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "12", "--eval-pairs", "3", artifactPath, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if err := run([]string{"inspect", artifactPath}); err != nil {
		t.Fatalf("run inspect: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	profile, err := eosruntime.ReadEmbeddingTrainProfileFile(eosruntime.DefaultEmbeddingTrainProfilePath(artifactPath))
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if profile.Step == 0 {
		t.Fatal("expected non-zero training profile step")
	}
	if profile.ForwardBackend != "" && profile.ForwardResidency.MatMul.BindCalls == 0 {
		t.Fatal("expected matmul bind activity in repeated encoder train profile")
	}
	for _, candidate := range []string{
		eosruntime.DefaultTokenizerPath(artifactPath),
		eosruntime.DefaultMinedTrainPairsPath(artifactPath),
		eosruntime.DefaultMinedEvalPairsPath(artifactPath),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected generated training artifact %q: %v", candidate, err)
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained repeated-encoder package: %v", err)
	}
}

func TestRunCompileDefaultMLLThenTrainFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "tiny_train_embed.eos")
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	if err := os.WriteFile(srcPath, source, 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}
	if err := run([]string{"compile", srcPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	artifactPath := filepath.Join(dir, "tiny_train_embed.mll")
	if _, err := os.Stat(artifactPath); err != nil {
		t.Fatalf("expected compiled .mll artifact %q: %v", artifactPath, err)
	}
	manifest := eosruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: eosruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(eosruntime.DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(dir, "train.jsonl")
	evalPath := filepath.Join(dir, "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", artifactPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func writeTrainableArtifact(t *testing.T) string {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := eosruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: eosruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(eosruntime.DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

func copyExampleFile(t *testing.T, dir, name string) string {
	t.Helper()
	srcPath := filepath.Join("..", "..", "examples", name)
	data, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatalf("read example file %q: %v", srcPath, err)
	}
	dstPath := filepath.Join(dir, name)
	if err := os.WriteFile(dstPath, data, 0o644); err != nil {
		t.Fatalf("write example file %q: %v", dstPath, err)
	}
	return dstPath
}

func captureRunOutput(t *testing.T, args []string) string {
	t.Helper()
	output, runErr := captureRunOutputAndError(t, args)
	if runErr != nil {
		t.Fatalf("run %v: %v\noutput:\n%s", args, runErr, output)
	}
	return output
}

func captureRunOutputAndError(t *testing.T, args []string) (string, error) {
	t.Helper()
	origStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}
	os.Stdout = writer
	defer func() {
		os.Stdout = origStdout
	}()
	runErr := run(args)
	if err := writer.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stdout capture: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stdout reader: %v", err)
	}
	return string(data), runErr
}
