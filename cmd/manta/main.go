package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"time"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/models"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/directml"
	"github.com/odvcencio/manta/runtime/backends/metal"
	"github.com/odvcencio/manta/runtime/backends/vulkan"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
)

func main() {
	stopProfile, err := startOptionalProfiles()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer stopProfile()
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func startOptionalProfiles() (func(), error) {
	cpuPath := mantaEnv("MANTA_CPU_PROFILE")
	memPath := mantaEnv("MANTA_MEM_PROFILE")
	var cpuFile *os.File
	if cpuPath != "" {
		file, err := os.Create(cpuPath)
		if err != nil {
			return nil, fmt.Errorf("create CPU profile %q: %w", cpuPath, err)
		}
		if err := pprof.StartCPUProfile(file); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("start CPU profile %q: %w", cpuPath, err)
		}
		cpuFile = file
	}
	return func() {
		if cpuFile != nil {
			pprof.StopCPUProfile()
			_ = cpuFile.Close()
			fmt.Fprintf(os.Stderr, "cpu profile: %s\n", cpuPath)
		}
		if memPath != "" {
			runtime.GC()
			file, err := os.Create(memPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "write memory profile %q: %v\n", memPath, err)
				return
			}
			if err := pprof.WriteHeapProfile(file); err != nil {
				fmt.Fprintf(os.Stderr, "write memory profile %q: %v\n", memPath, err)
			}
			_ = file.Close()
			fmt.Fprintf(os.Stderr, "memory profile: %s\n", memPath)
		}
	}, nil
}

func mantaEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	switch args[0] {
	case "version":
		fmt.Println("manta dev")
		return nil
	case "compile":
		return runCompile(args[1:])
	case "run":
		return runArtifact(args[1:])
	case "embed-text":
		return runEmbedText(args[1:])
	case "eval-retrieval":
		return runEvalRetrieval(args[1:])
	case "eval-retrieval-bm25":
		return runEvalRetrievalBM25(args[1:])
	case "mine-retrieval-hard-negatives":
		return runMineRetrievalHardNegatives(args[1:])
	case "mine-retrieval-model-hard-negatives":
		return runMineRetrievalModelHardNegatives(args[1:])
	case "demo":
		return runDemo(args[1:])
	case "inspect":
		return runInspect(args[1:])
	case "export-mll":
		return runExportMLL(args[1:])
	case "init-model":
		return runInitModel(args[1:])
	case "init-mirage":
		return runInitMirage(args[1:])
	case "init-train":
		return runInitTrain(args[1:])
	case "rename-embed":
		return runRenameEmbed(args[1:])
	case "train-tokenizer":
		return runTrainTokenizer(args[1:])
	case "tokenize-embed":
		return runTokenizeEmbed(args[1:])
	case "mine-text-pairs":
		return runMineTextPairs(args[1:])
	case "import-teacher-scores":
		return runImportTeacherScores(args[1:])
	case "score-teacher-hard-negatives":
		return runScoreTeacherHardNegatives(args[1:])
	case "audit-teacher-scores":
		return runAuditTeacherScores(args[1:])
	case "plan-sparse-attention":
		return runPlanSparseAttention(args[1:])
	case "train-embed":
		return runTrainEmbed(args[1:])
	case "train-corpus":
		return runTrainCorpus(args[1:])
	case "compare-train-metrics":
		return runCompareTrainMetrics(args[1:])
	case "compare-retrieval-metrics":
		return runCompareRetrievalMetrics(args[1:])
	case "diagnose-train-metrics":
		return runDiagnoseTrainMetrics(args[1:])
	case "gate-train-metrics":
		return runGateTrainMetrics(args[1:])
	case "gate-retrieval-metrics":
		return runGateRetrievalMetrics(args[1:])
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func runCompile(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta compile <source.manta> [output.mll]")
	}
	srcPath := args[0]
	outPath := defaultArtifactPath(srcPath)
	if len(args) > 1 && args[1] != "" {
		outPath = args[1]
	}

	src, err := os.ReadFile(srcPath)
	if err != nil {
		return err
	}
	moduleName := strings.TrimSuffix(filepath.Base(srcPath), filepath.Ext(srcPath))
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: moduleName})
	if err != nil {
		return err
	}
	if err := mantaartifact.WriteFile(outPath, bundle.Artifact); err != nil {
		return err
	}

	fmt.Printf("compiled %q -> %q\n", srcPath, outPath)
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n",
		len(bundle.Artifact.EntryPoints), len(bundle.Artifact.Steps), len(bundle.Artifact.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(bundle.Artifact.Kernels))
	return nil
}

func runDemo(args []string) error {
	name := "tiny_embed"
	if len(args) > 0 && args[0] != "" {
		name = args[0]
	}
	preset := compiler.PresetTinyEmbed
	switch name {
	case "mirage_v1":
		mod, err := models.DefaultMirageV1Module(models.MirageV1Config{})
		if err != nil {
			return err
		}
		return runDemoModule(mod)
	case "tiny_embed_pooled":
		preset = compiler.PresetTinyEmbedPooled
	case "tiny_embed_masked_pooled":
		preset = compiler.PresetTinyEmbedMaskedPooled
	case "tiny_decode":
		preset = compiler.PresetTinyDecode
	case "tiny_score":
		preset = compiler.PresetTinyScore
	case "tiny_rerank":
		preset = compiler.PresetTinyRerank
	case "tiny_select":
		preset = compiler.PresetTinySelect
	case "tiny_retrieve":
		preset = compiler.PresetTinyRetrieve
	case "tiny_candidates":
		preset = compiler.PresetTinyCandidates
	case "tiny_batch_candidates":
		preset = compiler.PresetTinyBatchCandidates
	case "tiny_packed_candidates":
		preset = compiler.PresetTinyPackedCandidates
	}

	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: name, Preset: preset})
	if err != nil {
		return err
	}

	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, stubLoadOptions(bundle.Artifact)...)
	if err != nil {
		return err
	}

	entryName := defaultEntryName(bundle.Artifact)
	entry, err := entryPointByName(bundle.Artifact, entryName)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}

	fmt.Printf("loaded module %q for backend %q\n", bundle.Artifact.Name, prog.Backend())
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n",
		len(bundle.Artifact.EntryPoints), len(bundle.Artifact.Steps), len(bundle.Artifact.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(bundle.Artifact.Kernels))
	fmt.Printf("params: %d\n", len(bundle.Artifact.Params))
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(bundle.Artifact.Version))
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runDemoModule(mod *mantaartifact.Module) error {
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), mod, stubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	entryName := defaultEntryName(mod)
	entry, err := entryPointByName(mod, entryName)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}
	fmt.Printf("loaded module %q for backend %q\n", mod.Name, prog.Backend())
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(mod.Kernels))
	fmt.Printf("params: %d\n", len(mod.Params))
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(mod.Version))
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runArtifact(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta run <artifact.mll> [entry]")
	}
	path := args[0]
	mod, err := mantaartifact.ReadFile(path)
	if err != nil {
		return err
	}
	entryName := defaultEntryName(mod)
	if len(args) > 1 && args[1] != "" {
		entryName = args[1]
	}
	entry, err := entryPointByName(mod, entryName)
	if err != nil {
		return err
	}
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), mod, stubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}

	fmt.Printf("loaded artifact %q for backend %q\n", mod.Name, prog.Backend())
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runEmbedText(args []string) error {
	if len(args) < 2 || args[0] == "" {
		return fmt.Errorf("usage: manta embed-text <artifact.mll> <text...>")
	}
	path := args[0]
	text := strings.Join(args[1:], " ")
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		return err
	}
	tokens, _, err := model.TokenizeText(text)
	if err != nil {
		return err
	}
	result, err := model.EmbedText(context.Background(), text)
	if err != nil {
		return err
	}
	if result.Embeddings == nil {
		return fmt.Errorf("embedding output tensor is nil")
	}
	manifest := model.Manifest()
	fmt.Printf("loaded embedding %q for backend %q\n", displayManifestName(manifest.Name), model.Backend())
	fmt.Printf("tokens: %d\n", len(tokens))
	fmt.Printf("output: %s\n", displayManifestName(result.OutputName))
	fmt.Printf("embedding: %s%v\n", result.Embeddings.DType, result.Embeddings.Shape)
	return nil
}

func runEvalRetrieval(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval", flag.ContinueOnError)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta eval-retrieval [flags] <artifact.mll> <beir-dataset-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := mantaruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	metrics, err := mantaruntime.EvaluateEmbeddingRetrieval(context.Background(), model, mantaruntime.RetrievalEvalConfig{
		DatasetName:  *datasetName,
		ArtifactPath: artifactPath,
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    *qrelsPath,
		BatchSize:    *batchSize,
		TopK:         *topK,
		MaxDocs:      *maxDocs,
		MaxQueries:   *maxQueries,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval eval: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("quality: ndcg@10=%.6f mrr@10=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.MRRAt10, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	return nil
}

func runEvalRetrievalBM25(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-bm25", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta eval-retrieval-bm25 [flags] <beir-dataset-dir>")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := mantaruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := mantaruntime.EvaluateBM25Retrieval(context.Background(), mantaruntime.RetrievalEvalConfig{
		DatasetName: *datasetName,
		CorpusPath:  corpusPath,
		QueriesPath: queriesPath,
		QrelsPath:   *qrelsPath,
		TopK:        *topK,
		MaxDocs:     *maxDocs,
		MaxQueries:  *maxQueries,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval bm25: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("quality: ndcg@10=%.6f mrr@10=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.MRRAt10, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	return nil
}

func runMineRetrievalHardNegatives(args []string) error {
	fs := flag.NewFlagSet("mine-retrieval-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for status output")
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	negatives := fs.Int("negatives", 1, "BM25 hard negatives per positive qrel")
	candidateTopK := fs.Int("candidate-top-k", 100, "BM25 candidate depth to mine negatives from")
	maxExamples := fs.Int("max-examples", 0, "limit mined hard-negative examples")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta mine-retrieval-hard-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	}
	if *negatives <= 0 {
		return fmt.Errorf("negatives must be positive")
	}
	if *candidateTopK <= 0 {
		return fmt.Errorf("candidate-top-k must be positive")
	}
	if *maxExamples < 0 {
		return fmt.Errorf("max-examples must be non-negative")
	}
	datasetDir := fs.Arg(0)
	outputPath := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := mantaruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	examples, summary, err := mantaruntime.MineBM25TextHardNegatives(context.Background(), mantaruntime.RetrievalHardNegativeMiningConfig{
		DatasetName:          *datasetName,
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            *qrelsPath,
		NegativesPerPositive: *negatives,
		CandidateTopK:        *candidateTopK,
		MaxExamples:          *maxExamples,
		MaxDocs:              *maxDocs,
		MaxQueries:           *maxQueries,
	})
	if err != nil {
		return err
	}
	if err := mantaruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	fmt.Printf("mined retrieval hard negatives: dataset=%s examples=%d positives=%d negatives=%d queries=%d\n",
		summary.DatasetName, summary.Examples, summary.PositivePairs, summary.Negatives, summary.Queries)
	fmt.Printf("skipped: queries_without_text=%d positives_without_text=%d queries_without_negatives=%d\n",
		summary.SkippedQueriesNoText, summary.SkippedPositiveDocs, summary.SkippedQueriesNoNegative)
	fmt.Printf("output: %s\n", outputPath)
	return nil
}

func runMineRetrievalModelHardNegatives(args []string) error {
	fs := flag.NewFlagSet("mine-retrieval-model-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for status output")
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	negatives := fs.Int("negatives", 1, "model-ranked hard negatives per positive qrel")
	candidateTopK := fs.Int("candidate-top-k", 100, "model-ranked candidate depth to mine negatives from")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	maxExamples := fs.Int("max-examples", 0, "limit mined hard-negative examples")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: manta mine-retrieval-model-hard-negatives [flags] <artifact.mll> <beir-dataset-dir> <output.jsonl>")
	}
	if *negatives <= 0 {
		return fmt.Errorf("negatives must be positive")
	}
	if *candidateTopK <= 0 {
		return fmt.Errorf("candidate-top-k must be positive")
	}
	if *batchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	if *maxExamples < 0 {
		return fmt.Errorf("max-examples must be non-negative")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	outputPath := fs.Arg(2)
	corpusPath, queriesPath, defaultQrelsPath := mantaruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	examples, summary, err := mantaruntime.MineModelTextHardNegatives(context.Background(), model, mantaruntime.RetrievalHardNegativeMiningConfig{
		DatasetName:          *datasetName,
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            *qrelsPath,
		NegativesPerPositive: *negatives,
		CandidateTopK:        *candidateTopK,
		BatchSize:            *batchSize,
		MaxExamples:          *maxExamples,
		MaxDocs:              *maxDocs,
		MaxQueries:           *maxQueries,
	})
	if err != nil {
		return err
	}
	if err := mantaruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	fmt.Printf("mined model retrieval hard negatives: dataset=%s backend=%s examples=%d positives=%d negatives=%d queries=%d\n",
		summary.DatasetName, model.Backend(), summary.Examples, summary.PositivePairs, summary.Negatives, summary.Queries)
	fmt.Printf("skipped: queries_without_text=%d positives_without_text=%d queries_without_negatives=%d\n",
		summary.SkippedQueriesNoText, summary.SkippedPositiveDocs, summary.SkippedQueriesNoNegative)
	fmt.Printf("output: %s\n", outputPath)
	return nil
}

type teacherScoreImportRecord struct {
	Source         string    `json:"source,omitempty"`
	Query          string    `json:"query,omitempty"`
	Positive       string    `json:"positive,omitempty"`
	Document       string    `json:"document,omitempty"`
	Candidate      string    `json:"candidate,omitempty"`
	Text           string    `json:"text,omitempty"`
	Score          *float64  `json:"score,omitempty"`
	Scores         []float64 `json:"scores,omitempty"`
	TeacherScores  []float64 `json:"teacher_scores,omitempty"`
	PositiveScore  *float64  `json:"positive_score,omitempty"`
	NegativeScores []float64 `json:"negative_scores,omitempty"`
}

type teacherScoreImportTable struct {
	ExampleScores   map[string][]float32
	CandidateScores map[string]float32
	ExampleRows     int
	CandidateRows   int
}

type teacherScoreImportSummary struct {
	Schema          string `json:"schema"`
	CreatedUTC      string `json:"created_utc"`
	InputJSONL      string `json:"input_jsonl"`
	ScoresJSONL     string `json:"scores_jsonl"`
	OutputJSONL     string `json:"output_jsonl"`
	TeacherModelID  string `json:"teacher_model_id,omitempty"`
	TeacherRevision string `json:"teacher_revision,omitempty"`
	Prompt          string `json:"prompt,omitempty"`
	ScoreScale      string `json:"score_scale,omitempty"`
	Examples        int    `json:"examples"`
	Updated         int    `json:"updated"`
	SkippedExisting int    `json:"skipped_existing"`
	SkippedMissing  int    `json:"skipped_missing"`
	ExampleRows     int    `json:"example_rows"`
	CandidateRows   int    `json:"candidate_rows"`
}

type teacherHardNegativeScoreSummary struct {
	Schema          string `json:"schema"`
	CreatedUTC      string `json:"created_utc"`
	TeacherArtifact string `json:"teacher_artifact"`
	InputJSONL      string `json:"input_jsonl"`
	OutputJSONL     string `json:"output_jsonl"`
	TeacherModelID  string `json:"teacher_model_id,omitempty"`
	TeacherRevision string `json:"teacher_revision,omitempty"`
	Prompt          string `json:"prompt,omitempty"`
	ScoreScale      string `json:"score_scale"`
	Backend         string `json:"backend,omitempty"`
	BatchSize       int    `json:"batch_size"`
	Examples        int    `json:"examples"`
	Updated         int    `json:"updated"`
	SkippedExisting int    `json:"skipped_existing"`
}

type teacherScoreAuditSummary struct {
	Schema      string  `json:"schema"`
	CreatedUTC  string  `json:"created_utc"`
	InputJSONL  string  `json:"input_jsonl"`
	Mode        string  `json:"mode"`
	Temperature float64 `json:"temperature"`
	teacherScoreAuditStats
	Sources map[string]teacherScoreAuditStats `json:"sources,omitempty"`
}

type teacherScoreAuditStats struct {
	Examples              int     `json:"examples"`
	ScoredExamples        int     `json:"scored_examples"`
	MissingExamples       int     `json:"missing_examples"`
	Candidates            int     `json:"candidates"`
	ScoredCandidates      int     `json:"scored_candidates"`
	PositiveTop1          int     `json:"positive_top1"`
	PositiveTop1Rate      float64 `json:"positive_top1_rate"`
	PositiveMeanRank      float64 `json:"positive_mean_rank"`
	PositiveMeanMargin    float64 `json:"positive_mean_margin"`
	MeanScore             float64 `json:"mean_score"`
	MeanScoreRange        float64 `json:"mean_score_range"`
	MeanEntropy           float64 `json:"mean_entropy"`
	MeanNormalizedEntropy float64 `json:"mean_normalized_entropy"`
}

type teacherScoreAuditCounters struct {
	Examples             int
	ScoredExamples       int
	MissingExamples      int
	Candidates           int
	ScoredCandidates     int
	PositiveTop1         int
	PositiveRankSum      float64
	PositiveMarginSum    float64
	ScoreSum             float64
	ScoreRangeSum        float64
	EntropySum           float64
	NormalizedEntropySum float64
}

type sparseAttentionPlanReport struct {
	Schema     string                    `json:"schema"`
	CreatedUTC string                    `json:"created_utc"`
	Config     sparseAttentionPlanConfig `json:"config"`
	Gate       sparseAttentionPlanGate   `json:"gate"`
	Rows       []sparseAttentionPlanRow  `json:"rows"`
}

type sparseAttentionPlanConfig struct {
	KeyLens          []int   `json:"key_lens"`
	QueryLen         int     `json:"query_len"`
	QueryDim         int     `json:"query_dim"`
	ValueDim         int     `json:"value_dim"`
	TopK             int     `json:"top_k"`
	RouteBlockSize   int     `json:"route_block_size"`
	RouteTopBlocks   int     `json:"route_top_blocks"`
	Exact            bool    `json:"exact"`
	Bits             int     `json:"bits"`
	Batches          int     `json:"batches"`
	MaxScoreFraction float64 `json:"max_score_fraction,omitempty"`
	MaxTurboKVMiB    float64 `json:"max_turbo_kv_mib,omitempty"`
	RequireSubq      bool    `json:"require_subquadratic,omitempty"`
}

type sparseAttentionPlanGate struct {
	Passed                   bool     `json:"passed"`
	FailureReasons           []string `json:"failure_reasons,omitempty"`
	Rows                     int      `json:"rows"`
	SubquadraticRows         int      `json:"subquadratic_rows"`
	MaxScoreFractionObserved float64  `json:"max_score_fraction_observed"`
	MaxTurboKVMiBObserved    float64  `json:"max_turbo_kv_mib_observed"`
	ScoreAlpha               float64  `json:"score_alpha"`
}

type sparseAttentionPlanRow struct {
	KeyLen                      int     `json:"key_len"`
	QueryLen                    int     `json:"query_len"`
	QueryDim                    int     `json:"query_dim"`
	ValueDim                    int     `json:"value_dim"`
	TopK                        int     `json:"top_k"`
	Routing                     string  `json:"routing"`
	RouteBlockSize              int     `json:"route_block_size"`
	RouteTopBlocks              int     `json:"route_top_blocks"`
	RouteBlockCount             int     `json:"route_block_count"`
	SelectedRouteBlocks         int     `json:"selected_route_blocks"`
	SelectedKeyCount            int     `json:"selected_key_count"`
	CandidateKeyBudget          int     `json:"candidate_key_budget"`
	DenseScoreCountPerQuery     int     `json:"dense_score_count_per_query"`
	EstimatedScoreCountPerQuery int     `json:"estimated_score_count_per_query"`
	ScoreCountFraction          float64 `json:"score_count_fraction"`
	CandidateKeyFraction        float64 `json:"candidate_key_fraction"`
	SubquadraticScorePlan       bool    `json:"subquadratic_score_plan"`
	DenseTotalScoreCount        int64   `json:"dense_total_score_count"`
	EstimatedTotalScoreCount    int64   `json:"estimated_total_score_count"`
	Bits                        int     `json:"bits"`
	DenseKVBytes                int64   `json:"dense_kv_bytes"`
	TurboQuantKVBytes           int64   `json:"turboquant_kv_bytes"`
	TurboQuantKVMiB             float64 `json:"turboquant_kv_mib"`
	TurboQuantCompressionRatio  float64 `json:"turboquant_compression_ratio"`
}

func runImportTeacherScores(args []string) error {
	fs := flag.NewFlagSet("import-teacher-scores", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	overwrite := fs.Bool("overwrite", false, "replace existing teacher_scores")
	allowMissing := fs.Bool("allow-missing", false, "keep examples without complete teacher scores instead of failing")
	manifestPath := fs.String("manifest", "", "provenance manifest path; default is <output>.teacher-scores.manifest.json")
	teacherModelID := fs.String("teacher-model-id", "", "external teacher model id")
	teacherRevision := fs.String("teacher-revision", "", "external teacher model revision")
	prompt := fs.String("prompt", "", "teacher prompt or instruction provenance")
	scoreScale := fs.String("score-scale", "", "teacher score scale, such as cosine, logit, or probability")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: manta import-teacher-scores [flags] <hard-negatives.jsonl> <scores.jsonl> <output.jsonl>")
	}
	inputPath := fs.Arg(0)
	scoresPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".teacher-scores.manifest.json"
	}
	examples, err := mantaruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
	if err != nil {
		return err
	}
	table, err := readTeacherScoreImportTable(scoresPath)
	if err != nil {
		return err
	}
	summary := teacherScoreImportSummary{
		Schema:          "manta.teacher_score_import.v1",
		CreatedUTC:      time.Now().UTC().Format(time.RFC3339),
		InputJSONL:      inputPath,
		ScoresJSONL:     scoresPath,
		OutputJSONL:     outputPath,
		TeacherModelID:  *teacherModelID,
		TeacherRevision: *teacherRevision,
		Prompt:          *prompt,
		ScoreScale:      *scoreScale,
		Examples:        len(examples),
		ExampleRows:     table.ExampleRows,
		CandidateRows:   table.CandidateRows,
	}
	for i := range examples {
		example := &examples[i]
		if len(example.TeacherScores) > 0 && !*overwrite {
			summary.SkippedExisting++
			continue
		}
		scores, ok := teacherScoresForExample(*example, table)
		if !ok {
			summary.SkippedMissing++
			if *allowMissing {
				continue
			}
			return fmt.Errorf("missing complete teacher scores for example %d source=%q query=%q", i, example.Source, example.Query)
		}
		example.TeacherScores = scores
		summary.Updated++
	}
	if err := mantaruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	if err := writeTeacherScoreImportManifest(*manifestPath, summary); err != nil {
		return err
	}
	fmt.Printf("imported teacher scores: examples=%d updated=%d skipped_existing=%d skipped_missing=%d example_rows=%d candidate_rows=%d\n",
		summary.Examples, summary.Updated, summary.SkippedExisting, summary.SkippedMissing, summary.ExampleRows, summary.CandidateRows)
	if summary.TeacherModelID != "" || summary.TeacherRevision != "" || summary.ScoreScale != "" {
		fmt.Printf("teacher: model_id=%s revision=%s score_scale=%s\n", summary.TeacherModelID, summary.TeacherRevision, summary.ScoreScale)
	}
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}

func runScoreTeacherHardNegatives(args []string) error {
	fs := flag.NewFlagSet("score-teacher-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	batchSize := fs.Int("batch-size", 64, "embedding batch size for candidate texts")
	overwrite := fs.Bool("overwrite", false, "replace existing teacher_scores")
	manifestPath := fs.String("manifest", "", "provenance manifest path; default is <output>.teacher-scores.manifest.json")
	teacherModelID := fs.String("teacher-model-id", "", "teacher model id; defaults to embedding manifest name")
	teacherRevision := fs.String("teacher-revision", "", "teacher model revision")
	prompt := fs.String("prompt", "", "teacher prompt or instruction provenance")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: manta score-teacher-hard-negatives [flags] <teacher.mll> <hard-negatives.jsonl> <output.jsonl>")
	}
	if *batchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	artifactPath := fs.Arg(0)
	inputPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".teacher-scores.manifest.json"
	}
	examples, err := mantaruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
	if err != nil {
		return err
	}
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	manifest := model.Manifest()
	if *teacherModelID == "" {
		*teacherModelID = manifest.Name
	}
	summary := teacherHardNegativeScoreSummary{
		Schema:          "manta.teacher_hard_negative_score.v1",
		CreatedUTC:      time.Now().UTC().Format(time.RFC3339),
		TeacherArtifact: artifactPath,
		InputJSONL:      inputPath,
		OutputJSONL:     outputPath,
		TeacherModelID:  *teacherModelID,
		TeacherRevision: *teacherRevision,
		Prompt:          *prompt,
		ScoreScale:      "cosine",
		Backend:         string(model.Backend()),
		BatchSize:       *batchSize,
		Examples:        len(examples),
	}
	for i := range examples {
		example := &examples[i]
		if len(example.TeacherScores) > 0 && !*overwrite {
			summary.SkippedExisting++
			continue
		}
		scores, err := scoreTeacherHardNegativeExample(context.Background(), model, *example, *batchSize)
		if err != nil {
			return fmt.Errorf("example %d source=%q query=%q: %w", i, example.Source, example.Query, err)
		}
		example.TeacherScores = scores
		summary.Updated++
	}
	if err := mantaruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	if err := writeTeacherHardNegativeScoreManifest(*manifestPath, summary); err != nil {
		return err
	}
	fmt.Printf("scored teacher hard negatives: examples=%d updated=%d skipped_existing=%d backend=%s batch_size=%d\n",
		summary.Examples, summary.Updated, summary.SkippedExisting, summary.Backend, summary.BatchSize)
	fmt.Printf("teacher: model_id=%s revision=%s score_scale=%s\n", summary.TeacherModelID, summary.TeacherRevision, summary.ScoreScale)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}

func runAuditTeacherScores(args []string) error {
	fs := flag.NewFlagSet("audit-teacher-scores", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	mode := fs.String("mode", "text", "input mode: text or tokenized")
	temperature := fs.Float64("temperature", 1, "softmax temperature used for entropy diagnostics")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta audit-teacher-scores [flags] <hard-negatives.jsonl> [summary.json]")
	}
	if *temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	inputPath := fs.Arg(0)
	summaryPath := inputPath + ".teacher-score-audit.json"
	if fs.NArg() > 1 && fs.Arg(1) != "" {
		summaryPath = fs.Arg(1)
	}
	normalizedMode := strings.ToLower(strings.TrimSpace(*mode))
	total := teacherScoreAuditCounters{}
	sourceTotals := map[string]*teacherScoreAuditCounters{}
	add := func(source string, candidateCount int, scores []float32) {
		total.add(candidateCount, scores, *temperature)
		key := strings.TrimSpace(source)
		if key == "" {
			key = "unknown"
		}
		sourceTotal := sourceTotals[key]
		if sourceTotal == nil {
			sourceTotal = &teacherScoreAuditCounters{}
			sourceTotals[key] = sourceTotal
		}
		sourceTotal.add(candidateCount, scores, *temperature)
	}
	switch normalizedMode {
	case "text":
		examples, err := mantaruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
		if err != nil {
			return err
		}
		for _, example := range examples {
			add(example.Source, 1+len(example.Negatives), example.TeacherScores)
		}
	case "tokenized", "tokens":
		normalizedMode = "tokenized"
		examples, err := mantaruntime.ReadEmbeddingHardNegativeExamplesFile(inputPath)
		if err != nil {
			return err
		}
		for _, example := range examples {
			add(example.Source, 1+len(example.NegativeTokens), example.TeacherScores)
		}
	default:
		return fmt.Errorf("unsupported mode %q: want text or tokenized", *mode)
	}
	summary := teacherScoreAuditSummary{
		Schema:                 "manta.teacher_score_audit.v1",
		CreatedUTC:             time.Now().UTC().Format(time.RFC3339),
		InputJSONL:             inputPath,
		Mode:                   normalizedMode,
		Temperature:            *temperature,
		teacherScoreAuditStats: total.summary(),
	}
	if len(sourceTotals) > 0 {
		summary.Sources = make(map[string]teacherScoreAuditStats, len(sourceTotals))
		for source, counters := range sourceTotals {
			summary.Sources[source] = counters.summary()
		}
	}
	if err := writeTeacherScoreAuditSummary(summaryPath, summary); err != nil {
		return err
	}
	fmt.Printf("audited teacher scores: examples=%d scored=%d missing=%d positive_top1_rate=%.6f mean_margin=%.6f mean_normalized_entropy=%.6f\n",
		summary.Examples, summary.ScoredExamples, summary.MissingExamples, summary.PositiveTop1Rate, summary.PositiveMeanMargin, summary.MeanNormalizedEntropy)
	fmt.Printf("summary: %s\n", summaryPath)
	return nil
}

func runPlanSparseAttention(args []string) error {
	fs := flag.NewFlagSet("plan-sparse-attention", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	keyLensCSV := fs.String("key-lens", "4096,8192,16384,32768,65536", "comma-separated key/context lengths to sweep")
	queryLen := fs.Int("query-len", 1, "query rows per batch item")
	queryDim := fs.Int("query-dim", 128, "query/key head dimension")
	valueDim := fs.Int("value-dim", 128, "value head dimension")
	topK := fs.Int("top-k", 64, "selected sparse keys per query; 0 uses ceil(sqrt(key_len))")
	routeBlockSize := fs.Int("route-block-size", 0, "route block size; 0 uses ceil(sqrt(key_len)) when routing is enabled")
	routeTopBlocks := fs.Int("route-top-blocks", 2, "route blocks selected per query; 0 disables routing")
	exact := fs.Bool("exact", false, "disable routing and report exact sparse top-k scoring")
	bits := fs.Int("bits", 4, "TurboQuant K/V bits: 2, 4, or 8")
	batches := fs.Int("batches", 1, "batch items for total score and KV memory estimates")
	denseBytesPerValue := fs.Int("dense-bytes-per-value", 2, "dense K/V bytes per value, usually 2 for f16")
	jsonPath := fs.String("json", "", "write machine-readable plan JSON")
	maxScoreFraction := fs.Float64("max-score-fraction", 1, "fail if any row exceeds this estimated score-work fraction versus dense")
	maxTurboKVMiB := fs.Float64("max-turbo-kv-mib", 0, "fail if any row exceeds this TurboQuant K/V MiB budget; 0 disables")
	requireSubq := fs.Bool("require-subquadratic", false, "fail if any row does not reduce score work versus dense scoring")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: manta plan-sparse-attention [flags]")
	}
	keyLens, err := parsePositiveIntCSV(*keyLensCSV, "key-lens")
	if err != nil {
		return err
	}
	if *queryLen <= 0 {
		return fmt.Errorf("query-len must be positive")
	}
	if *queryDim <= 0 {
		return fmt.Errorf("query-dim must be positive")
	}
	if *valueDim <= 0 {
		return fmt.Errorf("value-dim must be positive")
	}
	if *topK < 0 {
		return fmt.Errorf("top-k must be non-negative")
	}
	if *routeBlockSize < 0 {
		return fmt.Errorf("route-block-size must be non-negative")
	}
	if *routeTopBlocks < 0 {
		return fmt.Errorf("route-top-blocks must be non-negative")
	}
	if *bits != 2 && *bits != 4 && *bits != 8 {
		return fmt.Errorf("bits must be 2, 4, or 8")
	}
	if *batches <= 0 {
		return fmt.Errorf("batches must be positive")
	}
	if *denseBytesPerValue <= 0 {
		return fmt.Errorf("dense-bytes-per-value must be positive")
	}
	if *maxScoreFraction <= 0 {
		return fmt.Errorf("max-score-fraction must be positive")
	}
	if *maxTurboKVMiB < 0 {
		return fmt.Errorf("max-turbo-kv-mib must be non-negative")
	}
	report := sparseAttentionPlanReport{
		Schema:     "manta.sparse_attention_plan.v1",
		CreatedUTC: time.Now().UTC().Format(time.RFC3339),
		Config: sparseAttentionPlanConfig{
			KeyLens:          keyLens,
			QueryLen:         *queryLen,
			QueryDim:         *queryDim,
			ValueDim:         *valueDim,
			TopK:             *topK,
			RouteBlockSize:   *routeBlockSize,
			RouteTopBlocks:   *routeTopBlocks,
			Exact:            *exact,
			Bits:             *bits,
			Batches:          *batches,
			MaxScoreFraction: *maxScoreFraction,
			MaxTurboKVMiB:    *maxTurboKVMiB,
			RequireSubq:      *requireSubq,
		},
	}
	report.Rows = make([]sparseAttentionPlanRow, 0, len(keyLens))
	for _, keyLen := range keyLens {
		blockSize := *routeBlockSize
		topBlocks := *routeTopBlocks
		if *exact {
			blockSize = 0
			topBlocks = 0
		} else if topBlocks > 0 && blockSize == 0 {
			blockSize = int(math.Ceil(math.Sqrt(float64(keyLen))))
		}
		plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
			QueryLen:       *queryLen,
			KeyLen:         keyLen,
			QueryDim:       *queryDim,
			ValueDim:       *valueDim,
			TopK:           *topK,
			RouteBlockSize: blockSize,
			RouteTopBlocks: topBlocks,
		})
		kv := backend.PlanTurboQuantKVMemory(backend.TurboQuantKVMemoryPlanInput{
			Batches:            *batches,
			KeyLen:             keyLen,
			KeyDim:             *queryDim,
			ValueDim:           *valueDim,
			Bits:               *bits,
			DenseBytesPerValue: *denseBytesPerValue,
		})
		row := sparseAttentionPlanRow{
			KeyLen:                      keyLen,
			QueryLen:                    plan.QueryLen,
			QueryDim:                    plan.QueryDim,
			ValueDim:                    plan.ValueDim,
			TopK:                        plan.TopK,
			Routing:                     plan.Routing,
			RouteBlockSize:              plan.RouteBlockSize,
			RouteTopBlocks:              plan.RouteTopBlocks,
			RouteBlockCount:             plan.RouteBlockCount,
			SelectedRouteBlocks:         plan.SelectedRouteBlocks,
			SelectedKeyCount:            plan.SelectedKeyCount,
			CandidateKeyBudget:          plan.CandidateKeyBudget,
			DenseScoreCountPerQuery:     plan.DenseScoreCountPerQuery,
			EstimatedScoreCountPerQuery: plan.EstimatedScoreCountPerQuery,
			ScoreCountFraction:          plan.ScoreCountFraction,
			CandidateKeyFraction:        plan.CandidateKeyFraction,
			SubquadraticScorePlan:       plan.SubquadraticScorePlan,
			DenseTotalScoreCount:        int64(*batches) * int64(plan.QueryLen) * int64(plan.DenseScoreCountPerQuery),
			EstimatedTotalScoreCount:    int64(*batches) * int64(plan.QueryLen) * int64(plan.EstimatedScoreCountPerQuery),
			Bits:                        kv.Bits,
			DenseKVBytes:                kv.DenseKVBytes,
			TurboQuantKVBytes:           kv.TurboQuantKVBytes,
			TurboQuantKVMiB:             float64(kv.TurboQuantKVBytes) / (1024 * 1024),
			TurboQuantCompressionRatio:  kv.CompressionRatio,
		}
		report.Rows = append(report.Rows, row)
	}
	report.Gate = evaluateSparseAttentionPlanGate(report.Rows, *maxScoreFraction, *maxTurboKVMiB, *requireSubq)
	printSparseAttentionPlanTSV(report)
	if *jsonPath != "" {
		if err := writeSparseAttentionPlanReport(*jsonPath, report); err != nil {
			return err
		}
		fmt.Printf("json: %s\n", *jsonPath)
	}
	fmt.Printf("summary: rows=%d subq_rows=%d score_alpha=%.3f max_score_fraction=%.6f max_turbo_kv_mib=%.3f gate=%s\n",
		report.Gate.Rows, report.Gate.SubquadraticRows, report.Gate.ScoreAlpha, report.Gate.MaxScoreFractionObserved, report.Gate.MaxTurboKVMiBObserved, passFail(report.Gate.Passed))
	if !report.Gate.Passed {
		return fmt.Errorf("sparse attention plan gate failed: %s", strings.Join(report.Gate.FailureReasons, "; "))
	}
	return nil
}

func parsePositiveIntCSV(raw, label string) ([]int, error) {
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for i, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("%s[%d] %q is not an integer", label, i, part)
		}
		if value <= 0 {
			return nil, fmt.Errorf("%s[%d] must be positive", label, i)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s must contain at least one positive integer", label)
	}
	return out, nil
}

func evaluateSparseAttentionPlanGate(rows []sparseAttentionPlanRow, maxScoreFraction, maxTurboKVMiB float64, requireSubq bool) sparseAttentionPlanGate {
	gate := sparseAttentionPlanGate{
		Passed:     true,
		Rows:       len(rows),
		ScoreAlpha: fitSparseAttentionPlanAlpha(rows),
	}
	for _, row := range rows {
		if row.SubquadraticScorePlan {
			gate.SubquadraticRows++
		} else if requireSubq {
			gate.Passed = false
			gate.FailureReasons = append(gate.FailureReasons, fmt.Sprintf("key_len=%d is not subquadratic", row.KeyLen))
		}
		if row.ScoreCountFraction > gate.MaxScoreFractionObserved {
			gate.MaxScoreFractionObserved = row.ScoreCountFraction
		}
		if row.TurboQuantKVMiB > gate.MaxTurboKVMiBObserved {
			gate.MaxTurboKVMiBObserved = row.TurboQuantKVMiB
		}
		if maxScoreFraction > 0 && row.ScoreCountFraction > maxScoreFraction {
			gate.Passed = false
			gate.FailureReasons = append(gate.FailureReasons, fmt.Sprintf("key_len=%d score_fraction %.6f exceeds %.6f", row.KeyLen, row.ScoreCountFraction, maxScoreFraction))
		}
		if maxTurboKVMiB > 0 && row.TurboQuantKVMiB > maxTurboKVMiB {
			gate.Passed = false
			gate.FailureReasons = append(gate.FailureReasons, fmt.Sprintf("key_len=%d turbo_kv_mib %.3f exceeds %.3f", row.KeyLen, row.TurboQuantKVMiB, maxTurboKVMiB))
		}
	}
	return gate
}

func fitSparseAttentionPlanAlpha(rows []sparseAttentionPlanRow) float64 {
	if len(rows) < 2 {
		return 0
	}
	var sumX, sumY, sumXX, sumXY float64
	n := 0
	for _, row := range rows {
		if row.KeyLen <= 0 || row.EstimatedScoreCountPerQuery <= 0 {
			continue
		}
		x := math.Log(float64(row.KeyLen))
		y := math.Log(float64(row.EstimatedScoreCountPerQuery))
		sumX += x
		sumY += y
		sumXX += x * x
		sumXY += x * y
		n++
	}
	if n < 2 {
		return 0
	}
	denom := float64(n)*sumXX - sumX*sumX
	if denom == 0 {
		return 0
	}
	return (float64(n)*sumXY - sumX*sumY) / denom
}

func printSparseAttentionPlanTSV(report sparseAttentionPlanReport) {
	fmt.Println("key_len\trouting\troute_block_size\troute_top_blocks\ttop_k\tcandidate_key_budget\testimated_scores_per_query\tscore_fraction\tturbo_kv_mib\tcompression_ratio\tsubquadratic")
	for _, row := range report.Rows {
		fmt.Printf("%d\t%s\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.3f\t%.3f\t%t\n",
			row.KeyLen,
			row.Routing,
			row.RouteBlockSize,
			row.RouteTopBlocks,
			row.TopK,
			row.CandidateKeyBudget,
			row.EstimatedScoreCountPerQuery,
			row.ScoreCountFraction,
			row.TurboQuantKVMiB,
			row.TurboQuantCompressionRatio,
			row.SubquadraticScorePlan,
		)
	}
}

func writeSparseAttentionPlanReport(path string, report sparseAttentionPlanReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func passFail(pass bool) string {
	if pass {
		return "pass"
	}
	return "fail"
}

func scoreTeacherHardNegativeExample(ctx context.Context, model *mantaruntime.EmbeddingModel, example mantaruntime.EmbeddingTextHardNegativeExample, batchSize int) ([]float32, error) {
	queryVector, err := embedTeacherText(ctx, model, example.Query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	candidates := append([]string{example.Positive}, example.Negatives...)
	scores := make([]float32, 0, len(candidates))
	for start := 0; start < len(candidates); start += batchSize {
		end := min(start+batchSize, len(candidates))
		result, err := model.EmbedTextBatch(ctx, candidates[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed candidates %d-%d: %w", start, end, err)
		}
		rows, err := teacherEmbeddingRows(result.Embeddings, end-start)
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			scores = append(scores, dotTeacherVectors(queryVector, normalizeTeacherVector(row)))
		}
	}
	return scores, nil
}

func embedTeacherText(ctx context.Context, model *mantaruntime.EmbeddingModel, text string) ([]float32, error) {
	result, err := model.EmbedText(ctx, text)
	if err != nil {
		return nil, err
	}
	rows, err := teacherEmbeddingRows(result.Embeddings, 1)
	if err != nil {
		return nil, err
	}
	return normalizeTeacherVector(rows[0]), nil
}

func teacherEmbeddingRows(t *backend.Tensor, wantRows int) ([][]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("embedding tensor is nil")
	}
	if len(t.F32) == 0 {
		return nil, fmt.Errorf("embedding tensor has no float data")
	}
	switch len(t.Shape) {
	case 1:
		if wantRows != 1 {
			return nil, fmt.Errorf("embedding tensor shape %v cannot provide %d rows", t.Shape, wantRows)
		}
		return [][]float32{t.F32}, nil
	case 2:
		rows, cols := t.Shape[0], t.Shape[1]
		if rows != wantRows {
			return nil, fmt.Errorf("embedding tensor rows = %d, want %d", rows, wantRows)
		}
		if len(t.F32) < rows*cols {
			return nil, fmt.Errorf("embedding tensor has %d values, want at least %d", len(t.F32), rows*cols)
		}
		out := make([][]float32, rows)
		for i := 0; i < rows; i++ {
			out[i] = t.F32[i*cols : (i+1)*cols]
		}
		return out, nil
	default:
		return nil, fmt.Errorf("embedding tensor shape %v is not rank 1 or 2", t.Shape)
	}
}

func normalizeTeacherVector(in []float32) []float32 {
	out := append([]float32(nil), in...)
	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return out
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range out {
		out[i] *= scale
	}
	return out
}

func dotTeacherVectors(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func (c *teacherScoreAuditCounters) add(candidateCount int, scores []float32, temperature float64) {
	c.Examples++
	if candidateCount < 0 {
		candidateCount = 0
	}
	c.Candidates += candidateCount
	if len(scores) == 0 {
		c.MissingExamples++
		return
	}
	c.ScoredExamples++
	c.ScoredCandidates += len(scores)
	positive := float64(scores[0])
	rank := 1
	minScore := positive
	maxScore := positive
	bestNegative := math.Inf(-1)
	for i, raw := range scores {
		score := float64(raw)
		c.ScoreSum += score
		if score < minScore {
			minScore = score
		}
		if score > maxScore {
			maxScore = score
		}
		if i > 0 {
			if score > positive {
				rank++
			}
			if score > bestNegative {
				bestNegative = score
			}
		}
	}
	if math.IsInf(bestNegative, -1) {
		bestNegative = positive
	}
	if rank == 1 {
		c.PositiveTop1++
	}
	entropy, normalizedEntropy := teacherScoreEntropy(scores, temperature)
	c.PositiveRankSum += float64(rank)
	c.PositiveMarginSum += positive - bestNegative
	c.ScoreRangeSum += maxScore - minScore
	c.EntropySum += entropy
	c.NormalizedEntropySum += normalizedEntropy
}

func (c teacherScoreAuditCounters) summary() teacherScoreAuditStats {
	summary := teacherScoreAuditStats{
		Examples:         c.Examples,
		ScoredExamples:   c.ScoredExamples,
		MissingExamples:  c.MissingExamples,
		Candidates:       c.Candidates,
		ScoredCandidates: c.ScoredCandidates,
		PositiveTop1:     c.PositiveTop1,
	}
	if c.ScoredExamples > 0 {
		inv := 1 / float64(c.ScoredExamples)
		summary.PositiveTop1Rate = float64(c.PositiveTop1) * inv
		summary.PositiveMeanRank = c.PositiveRankSum * inv
		summary.PositiveMeanMargin = c.PositiveMarginSum * inv
		summary.MeanScoreRange = c.ScoreRangeSum * inv
		summary.MeanEntropy = c.EntropySum * inv
		summary.MeanNormalizedEntropy = c.NormalizedEntropySum * inv
	}
	if c.ScoredCandidates > 0 {
		summary.MeanScore = c.ScoreSum / float64(c.ScoredCandidates)
	}
	return summary
}

func teacherScoreEntropy(scores []float32, temperature float64) (float64, float64) {
	if len(scores) == 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 1
	}
	maxScore := float64(scores[0])
	for _, raw := range scores[1:] {
		score := float64(raw)
		if score > maxScore {
			maxScore = score
		}
	}
	probs := make([]float64, len(scores))
	denom := 0.0
	for i, raw := range scores {
		v := math.Exp((float64(raw) - maxScore) / temperature)
		probs[i] = v
		denom += v
	}
	if denom == 0 {
		return 0, 0
	}
	entropy := 0.0
	for _, p := range probs {
		p /= denom
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	normalized := 0.0
	if len(scores) > 1 {
		normalized = entropy / math.Log(float64(len(scores)))
	}
	return entropy, normalized
}

func readTeacherScoreImportTable(path string) (teacherScoreImportTable, error) {
	f, err := os.Open(path)
	if err != nil {
		return teacherScoreImportTable{}, err
	}
	defer f.Close()
	table := teacherScoreImportTable{
		ExampleScores:   map[string][]float32{},
		CandidateScores: map[string]float32{},
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record teacherScoreImportRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if scores, ok, err := record.teacherScoreVector(); err != nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		} else if ok {
			if strings.TrimSpace(record.Query) == "" {
				return teacherScoreImportTable{}, fmt.Errorf("%s:%d: query is required for score vectors", path, lineNo)
			}
			key := teacherScoreExampleKey(record.Source, record.Query)
			if _, exists := table.ExampleScores[key]; exists {
				return teacherScoreImportTable{}, fmt.Errorf("%s:%d: duplicate score vector for source=%q query=%q", path, lineNo, record.Source, record.Query)
			}
			table.ExampleScores[key] = scores
			table.ExampleRows++
			continue
		}
		if record.Score == nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: expected score, scores, teacher_scores, or positive_score/negative_scores", path, lineNo)
		}
		score, err := finiteFloat32(*record.Score, "score")
		if err != nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		candidate := firstNonEmptyString(record.Candidate, record.Document, record.Text, record.Positive)
		if strings.TrimSpace(record.Query) == "" || strings.TrimSpace(candidate) == "" {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: query and candidate/document/text are required for candidate score rows", path, lineNo)
		}
		key := teacherScoreCandidateKey(record.Source, record.Query, candidate)
		if _, exists := table.CandidateScores[key]; exists {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: duplicate candidate score for source=%q query=%q candidate=%q", path, lineNo, record.Source, record.Query, candidate)
		}
		table.CandidateScores[key] = score
		table.CandidateRows++
	}
	if err := scanner.Err(); err != nil {
		return teacherScoreImportTable{}, err
	}
	if table.ExampleRows == 0 && table.CandidateRows == 0 {
		return teacherScoreImportTable{}, fmt.Errorf("teacher score file is empty: %s", path)
	}
	return table, nil
}

func (r teacherScoreImportRecord) teacherScoreVector() ([]float32, bool, error) {
	values := r.TeacherScores
	if len(values) == 0 {
		values = r.Scores
	}
	if len(values) == 0 && r.PositiveScore != nil {
		values = append([]float64{*r.PositiveScore}, r.NegativeScores...)
	}
	if len(values) == 0 {
		return nil, false, nil
	}
	out := make([]float32, len(values))
	for i, value := range values {
		score, err := finiteFloat32(value, fmt.Sprintf("score[%d]", i))
		if err != nil {
			return nil, true, err
		}
		out[i] = score
	}
	return out, true, nil
}

func teacherScoresForExample(example mantaruntime.EmbeddingTextHardNegativeExample, table teacherScoreImportTable) ([]float32, bool) {
	if scores, ok := lookupTeacherScoreVector(table, example.Source, example.Query); ok {
		if len(scores) != 1+len(example.Negatives) {
			return nil, false
		}
		return append([]float32(nil), scores...), true
	}
	candidates := append([]string{example.Positive}, example.Negatives...)
	out := make([]float32, len(candidates))
	for i, candidate := range candidates {
		score, ok := lookupTeacherCandidateScore(table, example.Source, example.Query, candidate)
		if !ok {
			return nil, false
		}
		out[i] = score
	}
	return out, true
}

func lookupTeacherScoreVector(table teacherScoreImportTable, source, query string) ([]float32, bool) {
	if scores, ok := table.ExampleScores[teacherScoreExampleKey(source, query)]; ok {
		return scores, true
	}
	if strings.TrimSpace(source) != "" {
		if scores, ok := table.ExampleScores[teacherScoreExampleKey("", query)]; ok {
			return scores, true
		}
	}
	return nil, false
}

func lookupTeacherCandidateScore(table teacherScoreImportTable, source, query, candidate string) (float32, bool) {
	if score, ok := table.CandidateScores[teacherScoreCandidateKey(source, query, candidate)]; ok {
		return score, true
	}
	if strings.TrimSpace(source) != "" {
		if score, ok := table.CandidateScores[teacherScoreCandidateKey("", query, candidate)]; ok {
			return score, true
		}
	}
	return 0, false
}

func teacherScoreExampleKey(source, query string) string {
	return strings.TrimSpace(source) + "\x00" + strings.TrimSpace(query)
}

func teacherScoreCandidateKey(source, query, candidate string) string {
	return teacherScoreExampleKey(source, query) + "\x00" + strings.TrimSpace(candidate)
}

func finiteFloat32(value float64, label string) (float32, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("%s must be finite", label)
	}
	return float32(value), nil
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func writeTeacherScoreImportManifest(path string, summary teacherScoreImportSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeTeacherHardNegativeScoreManifest(path string, summary teacherHardNegativeScoreSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeTeacherScoreAuditSummary(path string, summary teacherScoreAuditSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func runInspect(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta inspect <artifact.mll>")
	}
	path := args[0]
	mod, err := mantaartifact.ReadFile(path)
	if err != nil {
		return err
	}
	fmt.Printf("module: %s\n", mod.Name)
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(mod.Version))
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("backends: %s\n", joinBackendKinds(mod.Requirements.SupportedBackends))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
	embeddedPackage := false
	embeddingManifestPath := mantaruntime.ResolveEmbeddingManifestPath(path)
	if _, err := os.Stat(embeddingManifestPath); err == nil {
		manifest, err := mantaruntime.ReadEmbeddingManifestFile(embeddingManifestPath)
		if err != nil {
			return err
		}
		fmt.Printf("embedding manifest: %s\n", embeddingManifestPath)
		printEmbeddingManifestSummary(manifest)
	} else if sealed, err := mantaruntime.ReadSealedEmbeddingPackage(path); err == nil {
		fmt.Println("embedding manifest: embedded")
		printEmbeddingManifestSummary(sealed.Manifest)
		fmt.Println("package: embedded sealed MLL")
		fmt.Println("package verify: OK")
		embeddedPackage = true
	}
	packagePath := mantaruntime.ResolvePackageManifestPath(path)
	if !embeddedPackage {
		if _, err := os.Stat(packagePath); err == nil {
			pkg, err := mantaruntime.ReadPackageManifestFile(packagePath)
			if err != nil {
				return err
			}
			verifyPaths := map[string]string{
				"artifact":           path,
				"embedding_manifest": mantaruntime.DefaultEmbeddingManifestPath(path),
				"tokenizer":          mantaruntime.DefaultTokenizerPath(path),
				"weights":            mantaruntime.DefaultWeightFilePath(path),
				"memory_plan":        mantaruntime.DefaultMemoryPlanPath(path),
				"train_manifest":     mantaruntime.DefaultEmbeddingTrainManifestPath(path),
				"checkpoint":         mantaruntime.DefaultEmbeddingCheckpointPath(path),
				"train_profile":      mantaruntime.DefaultEmbeddingTrainProfilePath(path),
			}
			if pkg.Kind == mantaruntime.PackageEmbedding {
				delete(verifyPaths, "train_manifest")
				delete(verifyPaths, "checkpoint")
				delete(verifyPaths, "train_profile")
			}
			verifyErr := pkg.VerifyFiles(verifyPaths)
			fmt.Printf("package: %s (%s)\n", packagePath, pkg.Kind)
			if verifyErr != nil {
				fmt.Printf("package verify: FAIL (%v)\n", verifyErr)
			} else {
				fmt.Println("package verify: OK")
			}
		}
	}
	profilePath := mantaruntime.DefaultEmbeddingTrainProfilePath(path)
	if _, err := os.Stat(profilePath); err == nil {
		profile, err := mantaruntime.ReadEmbeddingTrainProfileFile(profilePath)
		if err != nil {
			return err
		}
		fmt.Printf("train profile: step=%d forward=%s optimizer=%s activation=%s contrastive=%s\n",
			profile.Step,
			displayTrainBackend(profile.ForwardBackend),
			displayTrainBackend(profile.OptimizerBackend),
			displayTrainBackend(profile.ActivationBackend),
			displayTrainBackend(profile.ContrastiveBackend),
		)
	}
	return nil
}

func printEmbeddingManifestSummary(manifest mantaruntime.EmbeddingManifest) {
	fmt.Printf("embedding model: %s pooled=%s batch=%s output=%s/%s\n",
		displayManifestName(manifest.Name),
		manifest.PooledEntry,
		displayManifestName(manifest.BatchEntry),
		manifest.OutputName,
		manifest.OutputDType,
	)
	fmt.Printf("encoder repeats: %d\n", manifest.EncoderRepeats)
	if manifest.Tokenizer.VocabSize > 0 || manifest.Tokenizer.MaxSequence > 0 {
		fmt.Printf("tokenizer: vocab=%d max_sequence=%d\n", manifest.Tokenizer.VocabSize, manifest.Tokenizer.MaxSequence)
	}
}

func runExportMLL(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta export-mll <artifact.mll> [output.mll]")
	}
	artifactPath := args[0]
	outputPath := ""
	if len(args) > 1 {
		outputPath = args[1]
	}
	writtenPath, err := mantaruntime.ExportPackageToMLL(artifactPath, outputPath)
	if err != nil {
		return err
	}
	fmt.Printf("exported %q -> %q\n", artifactPath, writtenPath)
	return nil
}

func runInitModel(args []string) error {
	fs := flag.NewFlagSet("init-model", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var vocabSize int
	var maxSequence int
	var embeddingDim int
	var hiddenDim int
	var encoderRepeats int
	var seed int64
	var learningRate float64
	var weightDecay float64
	var weightBits int
	var optimizer string
	var contrastiveLoss string
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	fs.StringVar(&name, "name", "", "model name")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size")
	fs.IntVar(&maxSequence, "max-seq", 0, "maximum token sequence length")
	fs.IntVar(&embeddingDim, "embedding-dim", 0, "embedding/model dimension")
	fs.IntVar(&hiddenDim, "hidden-dim", 0, "FFN hidden dimension")
	fs.IntVar(&encoderRepeats, "encoder-repeats", 0, "number of encoder layer repeats (weights are tied; default 2)")
	fs.Int64Var(&seed, "seed", 0, "initialization seed")
	fs.Float64Var(&learningRate, "lr", 0, "trainer learning rate")
	fs.Float64Var(&weightDecay, "weight-decay", 0, "trainer weight decay")
	fs.IntVar(&weightBits, "weight-bits", 0, "forward fake-quant bits")
	fs.StringVar(&optimizer, "optimizer", "", "optimizer name")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&temperature, "temperature", 0, "contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta init-model [flags] <artifact.mll>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if weightDecay < 0 {
		return fmt.Errorf("weight-decay must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if groupedLossWeight < 0 {
		return fmt.Errorf("grouped-loss-weight must be non-negative")
	}
	if teacherLossWeight < 0 {
		return fmt.Errorf("teacher-loss-weight must be non-negative")
	}
	if teacherTemperature < 0 {
		return fmt.Errorf("teacher-temperature must be non-negative")
	}
	path := fs.Arg(0)
	paths, err := models.InitDefaultEmbeddingPackage(path, models.DefaultEmbeddingPackageConfig{
		Name:               name,
		VocabSize:          vocabSize,
		MaxSequence:        maxSequence,
		EmbeddingDim:       embeddingDim,
		HiddenDim:          hiddenDim,
		EncoderRepeats:     encoderRepeats,
		Seed:               seed,
		LearningRate:       float32(learningRate),
		WeightDecay:        float32(weightDecay),
		WeightBits:         weightBits,
		Optimizer:          optimizer,
		ContrastiveLoss:    contrastiveLoss,
		Temperature:        float32(temperature),
		GroupedLossWeight:  float32(groupedLossWeight),
		TeacherLossWeight:  float32(teacherLossWeight),
		TeacherTemperature: float32(teacherTemperature),
	})
	if err != nil {
		return err
	}
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		return err
	}
	checkpoint, err := mantaruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		return err
	}
	fmt.Printf("initialized default embedding model %q\n", manifest.Name)
	fmt.Printf("artifact: %s\n", paths.ArtifactPath)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("tokenizer contract: vocab=%d max_sequence=%d\n", manifest.Tokenizer.VocabSize, manifest.Tokenizer.MaxSequence)
	fmt.Printf("encoder repeats: %d\n", manifest.EncoderRepeats)
	fmt.Printf("training: optimizer=%s loss=%s lr=%.6f temperature=%.6f weight_bits=%d\n",
		checkpoint.Config.Optimizer,
		checkpoint.Config.ContrastiveLoss,
		checkpoint.Config.LearningRate,
		checkpoint.Config.Temperature,
		checkpoint.Config.WeightBits,
	)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func runInitMirage(args []string) error {
	fs := flag.NewFlagSet("init-mirage", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var imageHeight int
	var imageWidth int
	var latentChannels int
	var hyperChannels int
	var bits int
	var factorization string
	var lambda float64
	fs.StringVar(&name, "name", "", "model name")
	fs.IntVar(&imageHeight, "height", 0, "image height")
	fs.IntVar(&imageWidth, "width", 0, "image width")
	fs.IntVar(&latentChannels, "latent-channels", 0, "latent channels")
	fs.IntVar(&hyperChannels, "hyper-channels", 0, "hyperprior channels")
	fs.IntVar(&bits, "bits", 0, "TurboQuant bits: 2, 4, or 8")
	fs.StringVar(&factorization, "factorization", "", "coordinate entropy factorization: categorical or bit-plane")
	fs.Float64Var(&lambda, "lambda", 0, "rate-distortion lambda")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta init-mirage [flags] <artifact.mll>")
	}
	cfg := models.MirageV1Config{
		Name:           name,
		ImageHeight:    imageHeight,
		ImageWidth:     imageWidth,
		LatentChannels: latentChannels,
		HyperChannels:  hyperChannels,
		BitWidth:       bits,
		Factorization:  factorization,
		Lambda:         lambda,
	}
	if err := models.InitMirageV1Artifact(fs.Arg(0), cfg); err != nil {
		return err
	}
	mod, err := mantaartifact.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	fmt.Printf("initialized Mirage Image v1 module %q\n", mod.Name)
	fmt.Printf("artifact: %s\n", fs.Arg(0))
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
	return nil
}

func runInitTrain(args []string) error {
	fs := flag.NewFlagSet("init-train", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var manifestPath string
	var seed int64
	var learningRate float64
	var weightDecay float64
	var weightBits int
	var optimizer string
	var contrastiveLoss string
	var beta1 float64
	var beta2 float64
	var epsilon float64
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	var dims dimFlag
	fs.StringVar(&manifestPath, "manifest", "", "path to embedding manifest (defaults to sibling .embedding.mll)")
	fs.Int64Var(&seed, "seed", 1, "initialization seed")
	fs.Float64Var(&learningRate, "lr", 0, "trainer learning rate")
	fs.Float64Var(&weightDecay, "weight-decay", 0, "trainer weight decay")
	fs.IntVar(&weightBits, "weight-bits", 0, "forward fake-quant bits")
	fs.StringVar(&optimizer, "optimizer", "", "optimizer name")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&beta1, "beta1", 0, "optimizer beta1")
	fs.Float64Var(&beta2, "beta2", 0, "optimizer beta2")
	fs.Float64Var(&epsilon, "epsilon", 0, "optimizer epsilon")
	fs.Float64Var(&temperature, "temperature", 0, "contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	fs.Var(&dims, "dim", "symbolic dimension binding in NAME=VALUE form; repeatable")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta init-train [flags] <artifact.mll>")
	}
	path := fs.Arg(0)
	cfg := mantaruntime.EmbeddingTrainConfig{
		LearningRate:       float32(learningRate),
		WeightDecay:        float32(weightDecay),
		WeightBits:         weightBits,
		Optimizer:          optimizer,
		Beta1:              float32(beta1),
		Beta2:              float32(beta2),
		Epsilon:            float32(epsilon),
		ContrastiveLoss:    contrastiveLoss,
		Temperature:        float32(temperature),
		GroupedLossWeight:  float32(groupedLossWeight),
		TeacherLossWeight:  float32(teacherLossWeight),
		TeacherTemperature: float32(teacherTemperature),
	}
	opts := mantaruntime.EmbeddingTrainInitOptions{
		Seed:       seed,
		ShapeSizes: dims.values(),
	}
	var (
		paths mantaruntime.EmbeddingTrainPackagePaths
		err   error
	)
	if manifestPath == "" {
		manifestPath = mantaruntime.ResolveEmbeddingManifestPath(path)
	}
	manifest, readErr := mantaruntime.ReadEmbeddingManifestFile(manifestPath)
	if readErr != nil {
		return readErr
	}
	paths, err = mantaruntime.InitializeEmbeddingTrainerPackageWithManifest(path, manifest, cfg, opts)
	if err != nil {
		return err
	}
	fmt.Printf("initialized training package %q\n", path)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("train manifest: %s\n", paths.TrainManifestPath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func runRenameEmbed(args []string) error {
	fs := flag.NewFlagSet("rename-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	fs.StringVar(&name, "name", "", "new embedding model name")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta rename-embed --name <model-name> <input.mll> <output.mll>")
	}
	inputPath := fs.Arg(0)
	outputPath := fs.Arg(1)
	trainer, err := mantaruntime.LoadEmbeddingTrainerPackage(inputPath)
	if err != nil {
		return err
	}
	if err := trainer.RenameEmbeddingModel(name); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	if err := copyTokenizerIfPresent(inputPath, outputPath); err != nil {
		return err
	}
	paths, err := trainer.WriteTrainingPackage(outputPath)
	if err != nil {
		return err
	}
	fmt.Printf("renamed embedding package %q -> %q\n", inputPath, outputPath)
	fmt.Printf("model: %s\n", name)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func copyTokenizerIfPresent(inputPath, outputPath string) error {
	sourcePath := mantaruntime.DefaultTokenizerPath(inputPath)
	if _, err := os.Stat(sourcePath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		return err
	}
	return os.WriteFile(mantaruntime.DefaultTokenizerPath(outputPath), data, 0o644)
}

func runTrainEmbed(args []string) error {
	fs := flag.NewFlagSet("train-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var evalEverySteps int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var tokenizerPath string
	var noTokenizer bool
	var planOnly bool
	var evalOnly bool
	var pairwiseTrain bool
	var hardNegativeTrain bool
	var hardNegativesPerQuery int
	var hardNegativeSourceWeights string
	var metricsJSONPath string
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	var teacherSourceTemperatures string
	var teacherScoreNormalization string
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&evalEverySteps, "eval-every-steps", 0, "evaluate every N optimizer steps within an epoch (0 disables)")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, score_margin, pair_accuracy, threshold_accuracy, auc, or loss")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", true, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer JSON for text-pair datasets")
	fs.BoolVar(&noTokenizer, "no-tokenizer", false, "disable sibling tokenizer discovery and treat JSONL as tokenized")
	fs.BoolVar(&planOnly, "plan-only", false, "print planned workload and exit without training")
	fs.BoolVar(&evalOnly, "eval-only", false, "evaluate the package without running optimizer steps")
	fs.BoolVar(&pairwiseTrain, "pairwise-train", false, "treat the training JSONL as labeled pair examples instead of contrastive positives")
	fs.BoolVar(&hardNegativeTrain, "hard-negative-train", false, "group labeled pair JSONL into query-positive-hard-negative contrastive batches")
	fs.IntVar(&hardNegativesPerQuery, "hard-negatives-per-query", 1, "maximum explicit negatives to attach to each query-positive example")
	fs.StringVar(&hardNegativeSourceWeights, "hard-negative-source-weights", "", "comma-separated source=weight hard-negative batch mix, for example scifact=2,nfcorpus=1,fiqa=2")
	fs.StringVar(&metricsJSONPath, "metrics-json", "", "write machine-readable run metrics JSON to this path")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	fs.StringVar(&teacherSourceTemperatures, "teacher-source-temperatures", "", "comma-separated source=temperature overrides for teacher distillation, for example scifact=10,nfcorpus:model=1.5")
	fs.StringVar(&teacherScoreNormalization, "teacher-score-normalization", "", "normalize hard-negative teacher_scores before distillation: none, source_zscore, family_zscore, or example_zscore")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]\n       manta train-embed --eval-only [flags] <artifact.mll> <eval.jsonl>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if groupedLossWeight < 0 {
		return fmt.Errorf("grouped-loss-weight must be non-negative")
	}
	if teacherLossWeight < 0 {
		return fmt.Errorf("teacher-loss-weight must be non-negative")
	}
	if teacherTemperature < 0 {
		return fmt.Errorf("teacher-temperature must be non-negative")
	}
	if progressEvery < 0 {
		return fmt.Errorf("progress-every must be non-negative")
	}
	if evalEverySteps < 0 {
		return fmt.Errorf("eval-every-steps must be non-negative")
	}
	if hardNegativesPerQuery < 0 {
		return fmt.Errorf("hard-negatives-per-query must be non-negative")
	}
	if pairwiseTrain && hardNegativeTrain {
		return fmt.Errorf("set either --pairwise-train or --hard-negative-train, not both")
	}
	parsedSourceWeights, parseErr := parsePositiveIntWeightMap(hardNegativeSourceWeights)
	if parseErr != nil {
		return fmt.Errorf("hard-negative-source-weights: %w", parseErr)
	}
	parsedTeacherSourceTemperatures, parseErr := parsePositiveFloatMap(teacherSourceTemperatures)
	if parseErr != nil {
		return fmt.Errorf("teacher-source-temperatures: %w", parseErr)
	}
	if len(parsedSourceWeights) > 0 && !hardNegativeTrain {
		return fmt.Errorf("--hard-negative-source-weights requires --hard-negative-train")
	}
	if len(parsedTeacherSourceTemperatures) > 0 && !hardNegativeTrain {
		return fmt.Errorf("--teacher-source-temperatures requires --hard-negative-train")
	}
	if strings.TrimSpace(teacherScoreNormalization) != "" && !hardNegativeTrain {
		return fmt.Errorf("--teacher-score-normalization requires --hard-negative-train")
	}
	path := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	if evalOnly && fs.NArg() == 2 {
		evalPath = trainPath
		trainPath = ""
	}
	if noTokenizer && tokenizerPath != "" {
		return fmt.Errorf("set either --tokenizer or --no-tokenizer, not both")
	}
	if tokenizerPath == "" && !noTokenizer {
		defaultTokenizerPath := mantaruntime.DefaultTokenizerPath(path)
		if _, err := os.Stat(defaultTokenizerPath); err == nil {
			tokenizerPath = defaultTokenizerPath
		}
	}
	runConfig := mantaruntime.EmbeddingTrainRunConfig{
		Epochs:                    epochs,
		BatchSize:                 batchSize,
		Shuffle:                   shuffle,
		Seed:                      seed,
		EvalEveryEpoch:            evalEvery,
		EvalEverySteps:            evalEverySteps,
		EarlyStoppingPatience:     patience,
		SelectMetric:              selectMetric,
		MinDelta:                  float32(minDelta),
		RestoreBest:               restoreBest,
		LengthBucketBatches:       lengthBucketBatches,
		LearningRate:              float32(learningRate),
		ContrastiveLoss:           contrastiveLoss,
		Temperature:               float32(temperature),
		GroupedLossWeight:         float32(groupedLossWeight),
		TeacherLossWeight:         float32(teacherLossWeight),
		TeacherTemperature:        float32(teacherTemperature),
		TeacherSourceTemperatures: parsedTeacherSourceTemperatures,
		TeacherScoreNormalization: teacherScoreNormalization,
		ProgressEverySteps:        progressEvery,
		EvalOnly:                  evalOnly,
		PairwiseTrain:             pairwiseTrain,
		HardNegativeTrain:         hardNegativeTrain,
		HardNegativesPerQuery:     hardNegativesPerQuery,
		HardNegativeSourceWeights: parsedSourceWeights,
	}
	if progressEvery > 0 {
		runConfig.Progress = printTrainProgress
	}
	workload, workloadErr := estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath, runConfig)
	if workloadErr == nil {
		fmt.Printf("planned workload: %s\n", formatTrainWorkload(workload))
	}
	if planOnly {
		if workloadErr != nil {
			return workloadErr
		}
		return nil
	}
	var (
		summary mantaruntime.EmbeddingTrainRunSummary
		paths   mantaruntime.EmbeddingTrainPackagePaths
		err     error
	)
	if tokenizerPath != "" {
		summary, paths, err = mantaruntime.TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, runConfig)
	} else {
		summary, paths, err = mantaruntime.TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, runConfig)
	}
	if err != nil {
		return err
	}
	if evalOnly {
		fmt.Printf("evaluated package %q\n", path)
	} else {
		fmt.Printf("trained package %q\n", path)
	}
	if tokenizerPath != "" {
		fmt.Printf("tokenizer: %s\n", tokenizerPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f threshold_accuracy=%.6f threshold=%.6f auc=%.6f top1=%.6f top5=%.6f top10=%.6f mrr=%.6f mean_rank=%.3f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.ThresholdAccuracy, summary.FinalEval.ScoreThreshold, summary.FinalEval.ROCAUC, summary.FinalEval.Top1Accuracy, summary.FinalEval.Top5Accuracy, summary.FinalEval.Top10Accuracy, summary.FinalEval.MeanReciprocalRank, summary.FinalEval.MeanPositiveRank, summary.FinalEval.PairCount)
	}
	fmt.Printf("workload: %s\n", formatTrainWorkload(summary.Workload))
	fmt.Printf("throughput: %s\n", formatTrainThroughput(summary))
	fmt.Printf("accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		displayTrainBackend(summary.EndProfile.ForwardBackend),
		displayTrainBackend(summary.EndProfile.OptimizerBackend),
		displayTrainBackend(summary.EndProfile.ActivationBackend),
		displayTrainBackend(summary.EndProfile.ContrastiveBackend),
	)
	fmt.Printf("profile delta: matmul_bind_calls=%d matmul_runs=%d matmul_run_upload_mb=%.2f matmul_run_download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		summary.DeltaProfile.ForwardResidency.MatMul.BindCalls,
		summary.DeltaProfile.ForwardResidency.MatMul.RunCalls,
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunUploadedBytes)/(1024*1024),
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunDownloadedBytes)/(1024*1024),
		summary.DeltaProfile.Optimizer.UpdateCalls,
		summary.DeltaProfile.Activation.GELUBackwardCalls+summary.DeltaProfile.Activation.SoftmaxBackwardCalls+summary.DeltaProfile.Activation.LayerNormBackwardCalls,
		summary.DeltaProfile.Contrastive.RunCalls,
	)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	if metricsJSONPath != "" {
		mode := "train"
		if evalOnly {
			mode = "eval"
		}
		if err := writeTrainMetricsJSON(metricsJSONPath, "train-embed", mode, path, tokenizerPath, summary, paths, nil); err != nil {
			return err
		}
		fmt.Printf("metrics: %s\n", metricsJSONPath)
	}
	return nil
}

func parsePositiveIntWeightMap(raw string) (map[string]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := map[string]int{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		key, value, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("entry %q must be source=weight", item)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		if key == "" {
			return nil, fmt.Errorf("entry %q has an empty source", item)
		}
		weight, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil || weight <= 0 {
			return nil, fmt.Errorf("entry %q must use a positive integer weight", item)
		}
		out[key] = weight
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func parsePositiveFloatMap(raw string) (map[string]float32, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := map[string]float32{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		key, value, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("entry %q must be source=value", item)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		if key == "" {
			return nil, fmt.Errorf("entry %q has an empty source", item)
		}
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 32)
		if err != nil || parsed <= 0 {
			return nil, fmt.Errorf("entry %q must use a positive float value", item)
		}
		out[key] = float32(parsed)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath string, cfg mantaruntime.EmbeddingTrainRunConfig) (mantaruntime.EmbeddingTrainWorkload, error) {
	if cfg.EvalOnly && evalPath == "" {
		evalPath = trainPath
		trainPath = ""
	}
	if tokenizerPath != "" {
		if cfg.EvalOnly {
			evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			allPositive := true
			positiveCount := 0
			for _, example := range evalPairs {
				if example.Target > 0 {
					positiveCount++
				} else {
					allPositive = false
				}
			}
			if allPositive {
				return mantaruntime.EstimateContrastiveTrainWorkload(0, positiveCount, cfg), nil
			}
			return mantaruntime.EstimatePairwiseTrainWorkload(0, len(evalPairs), cfg), nil
		}
		if cfg.PairwiseTrain {
			trainPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(trainPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount := 0
			if evalPath != "" {
				evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
				if err != nil {
					return mantaruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount = len(evalPairs)
			}
			return mantaruntime.EstimatePairwiseTrainWorkload(len(trainPairs), evalCount, cfg), nil
		}
		if cfg.HardNegativeTrain {
			trainSet, err := mantaruntime.ReadEmbeddingTextHardNegativeExamplesFile(trainPath)
			if err != nil {
				trainPairs, pairErr := mantaruntime.ReadEmbeddingTextPairExamplesFile(trainPath)
				if pairErr != nil {
					return mantaruntime.EmbeddingTrainWorkload{}, err
				}
				trainSet, err = mantaruntime.BuildEmbeddingTextHardNegativeExamplesFromPairs(trainPairs, cfg.HardNegativesPerQuery)
				if err != nil {
					return mantaruntime.EmbeddingTrainWorkload{}, err
				}
			}
			evalCount := 0
			if evalPath != "" {
				evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
				if err != nil {
					return mantaruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount = len(evalPairs)
			}
			return mantaruntime.EstimateHardNegativeTrainWorkload(len(trainSet), cfg.HardNegativesPerQuery, evalCount, cfg), nil
		}
		trainSet, err := mantaruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath)
		if err != nil {
			return mantaruntime.EmbeddingTrainWorkload{}, err
		}
		if evalPath == "" {
			return mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg), nil
		}
		evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
		if err != nil {
			return mantaruntime.EmbeddingTrainWorkload{}, err
		}
		allPositive := true
		positiveCount := 0
		for _, example := range evalPairs {
			if example.Target > 0 {
				positiveCount++
			} else {
				allPositive = false
			}
		}
		if allPositive {
			return mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), positiveCount, cfg), nil
		}
		workload := mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
		workload.EvalMode = "pairwise"
		workload.EvalExamples = len(evalPairs)
		workload.EvalPairsPerPass = int64(len(evalPairs))
		workload.PlannedEvalPasses = 1
		workload.PlannedEvalPairs = int64(len(evalPairs))
		workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
		return workload, nil
	}

	if cfg.EvalOnly {
		evalSet, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, pairErr := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if pairErr != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			return mantaruntime.EstimatePairwiseTrainWorkload(0, len(evalPairs), cfg), nil
		}
		return mantaruntime.EstimateContrastiveTrainWorkload(0, len(evalSet), cfg), nil
	}
	if cfg.PairwiseTrain {
		trainPairs, err := mantaruntime.ReadEmbeddingPairExamplesFile(trainPath)
		if err != nil {
			return mantaruntime.EmbeddingTrainWorkload{}, err
		}
		evalCount := 0
		if evalPath != "" {
			evalPairs, err := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount = len(evalPairs)
		}
		return mantaruntime.EstimatePairwiseTrainWorkload(len(trainPairs), evalCount, cfg), nil
	}
	if cfg.HardNegativeTrain {
		trainSet, err := mantaruntime.ReadEmbeddingHardNegativeExamplesFile(trainPath)
		if err != nil {
			trainPairs, pairErr := mantaruntime.ReadEmbeddingPairExamplesFile(trainPath)
			if pairErr != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			trainSet, err = mantaruntime.BuildEmbeddingHardNegativeExamplesFromPairs(trainPairs, cfg.HardNegativesPerQuery)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
		}
		evalCount := 0
		if evalPath != "" {
			evalPairs, err := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount = len(evalPairs)
		}
		return mantaruntime.EstimateHardNegativeTrainWorkload(len(trainSet), cfg.HardNegativesPerQuery, evalCount, cfg), nil
	}
	trainSet, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(trainPath)
	if err != nil {
		return mantaruntime.EmbeddingTrainWorkload{}, err
	}
	evalCount := 0
	if evalPath != "" {
		evalSet, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, pairErr := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if pairErr != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			workload := mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
			workload.EvalMode = "pairwise"
			workload.EvalExamples = len(evalPairs)
			workload.EvalPairsPerPass = int64(len(evalPairs))
			workload.PlannedEvalPasses = 1
			workload.PlannedEvalPairs = int64(len(evalPairs))
			workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
			return workload, nil
		}
		evalCount = len(evalSet)
	}
	return mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), evalCount, cfg), nil
}

func formatTrainWorkload(workload mantaruntime.EmbeddingTrainWorkload) string {
	parts := []string{
		fmt.Sprintf("train=%d %s examples", workload.TrainExamples, workload.TrainMode),
		fmt.Sprintf("batch=%d", workload.BatchSize),
		fmt.Sprintf("steps/epoch=%d", workload.TrainBatchesPerEpoch),
		fmt.Sprintf("train_pairs/epoch=%d", workload.TrainPairsPerEpoch),
	}
	if workload.EvalMode != "" {
		parts = append(parts,
			fmt.Sprintf("eval=%d %s examples", workload.EvalExamples, workload.EvalMode),
			fmt.Sprintf("eval_pairs/pass=%d", workload.EvalPairsPerPass),
			fmt.Sprintf("eval_passes(planned=%d actual=%d)", workload.PlannedEvalPasses, workload.ActualEvalPasses),
		)
	}
	parts = append(parts,
		fmt.Sprintf("pairs(planned=%d actual=%d)", workload.PlannedTotalPairs, workload.ActualTotalPairs),
	)
	return strings.Join(parts, " ")
}

func formatTrainThroughput(summary mantaruntime.EmbeddingTrainRunSummary) string {
	parts := []string{fmt.Sprintf("elapsed=%s", summary.Elapsed.Round(time.Millisecond))}
	if rate := itemsPerSecond(summary.Workload.ActualTotalExamples, summary.Elapsed); rate > 0 {
		parts = append(parts, fmt.Sprintf("examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualTotalPairs, summary.Elapsed); rate > 0 {
		parts = append(parts, fmt.Sprintf("pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(summary.Workload.ActualTrainExamples, summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("train_examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualTrainPairs, summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("train_pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(summary.Workload.ActualEvalExamples, summary.EvalDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("eval_examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualEvalPairs, summary.EvalDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("eval_pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(int64(summary.StepsRun), summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("optimizer_steps/s=%.2f", rate))
	}
	return strings.Join(parts, " ")
}

func printTrainProgress(progress mantaruntime.EmbeddingTrainProgress) {
	epochPairs := fmt.Sprintf("%d", progress.EpochTrainPairs)
	if progress.PlannedEpochPairs > 0 {
		epochPairs = fmt.Sprintf("%d/%d", progress.EpochTrainPairs, progress.PlannedEpochPairs)
	}
	fmt.Printf(
		"progress: epoch=%d batch=%d/%d step=%d loss=%.6f avg_score=%.6f batch_examples=%d batch_pairs=%d epoch_examples=%d epoch_pairs=%s elapsed=%s\n",
		progress.Epoch,
		progress.Batch,
		progress.Batches,
		progress.Step,
		progress.Loss,
		progress.AverageScore,
		progress.BatchExamples,
		progress.BatchPairs,
		progress.EpochTrainExamples,
		epochPairs,
		progress.Elapsed.Round(time.Millisecond),
	)
}

func itemsPerSecond(items int64, elapsed time.Duration) float64 {
	if items <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(items) / elapsed.Seconds()
}

func pairsPerSecond(pairs int64, elapsed time.Duration) float64 {
	if pairs <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(pairs) / elapsed.Seconds()
}

func runTrainCorpus(args []string) error {
	fs := flag.NewFlagSet("train-corpus", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var evalEverySteps int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var tokenizerPath string
	var vocabSize int
	var minFreq int
	var trainPairsPath string
	var evalPairsPath string
	var minChars int
	var maxPairs int
	var evalPairs int
	var metricsJSONPath string
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle/mining seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&evalEverySteps, "eval-every-steps", 0, "evaluate every N optimizer steps within an epoch (0 disables)")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, score_margin, pair_accuracy, threshold_accuracy, auc, or loss")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", true, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "output tokenizer path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for tokenizer merges")
	fs.StringVar(&trainPairsPath, "train-pairs", "", "output mined train pair dataset path")
	fs.StringVar(&evalPairsPath, "eval-pairs-path", "", "output mined eval pair dataset path")
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.StringVar(&metricsJSONPath, "metrics-json", "", "write machine-readable run metrics JSON to this path")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta train-corpus [flags] <artifact.mll> <corpus.txt>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if groupedLossWeight < 0 {
		return fmt.Errorf("grouped-loss-weight must be non-negative")
	}
	if teacherLossWeight < 0 {
		return fmt.Errorf("teacher-loss-weight must be non-negative")
	}
	if teacherTemperature < 0 {
		return fmt.Errorf("teacher-temperature must be non-negative")
	}
	if progressEvery < 0 {
		return fmt.Errorf("progress-every must be non-negative")
	}
	if evalEverySteps < 0 {
		return fmt.Errorf("eval-every-steps must be non-negative")
	}
	path := fs.Arg(0)
	corpusPath := fs.Arg(1)
	runConfig := mantaruntime.EmbeddingTrainRunConfig{
		Epochs:                epochs,
		BatchSize:             batchSize,
		Shuffle:               shuffle,
		Seed:                  seed,
		EvalEveryEpoch:        evalEvery,
		EvalEverySteps:        evalEverySteps,
		EarlyStoppingPatience: patience,
		SelectMetric:          selectMetric,
		MinDelta:              float32(minDelta),
		RestoreBest:           restoreBest,
		LengthBucketBatches:   lengthBucketBatches,
		LearningRate:          float32(learningRate),
		ContrastiveLoss:       contrastiveLoss,
		Temperature:           float32(temperature),
		GroupedLossWeight:     float32(groupedLossWeight),
		TeacherLossWeight:     float32(teacherLossWeight),
		TeacherTemperature:    float32(teacherTemperature),
		ProgressEverySteps:    progressEvery,
	}
	if progressEvery > 0 {
		runConfig.Progress = printTrainProgress
	}
	summary, paths, err := mantaruntime.TrainEmbeddingPackageFromCorpusFile(path, corpusPath, mantaruntime.EmbeddingCorpusTrainConfig{
		TokenizerPath:      tokenizerPath,
		TokenizerVocabSize: vocabSize,
		TokenizerMinFreq:   minFreq,
		TrainPairsPath:     trainPairsPath,
		EvalPairsPath:      evalPairsPath,
		Mining: mantaruntime.EmbeddingTextMiningConfig{
			MinChars:  minChars,
			MaxPairs:  maxPairs,
			EvalPairs: evalPairs,
			Seed:      seed,
		},
		Run: runConfig,
	})
	if err != nil {
		return err
	}
	fmt.Printf("trained package %q from corpus\n", path)
	fmt.Printf("tokenizer: %s\n", paths.TokenizerPath)
	fmt.Printf("train pairs: %s\n", paths.TrainPairsPath)
	if paths.EvalPairsPath != "" {
		fmt.Printf("eval pairs: %s\n", paths.EvalPairsPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f threshold_accuracy=%.6f threshold=%.6f auc=%.6f top1=%.6f top5=%.6f top10=%.6f mrr=%.6f mean_rank=%.3f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.ThresholdAccuracy, summary.FinalEval.ScoreThreshold, summary.FinalEval.ROCAUC, summary.FinalEval.Top1Accuracy, summary.FinalEval.Top5Accuracy, summary.FinalEval.Top10Accuracy, summary.FinalEval.MeanReciprocalRank, summary.FinalEval.MeanPositiveRank, summary.FinalEval.PairCount)
	}
	fmt.Printf("workload: %s\n", formatTrainWorkload(summary.Workload))
	fmt.Printf("throughput: %s\n", formatTrainThroughput(summary))
	fmt.Printf("accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		displayTrainBackend(summary.EndProfile.ForwardBackend),
		displayTrainBackend(summary.EndProfile.OptimizerBackend),
		displayTrainBackend(summary.EndProfile.ActivationBackend),
		displayTrainBackend(summary.EndProfile.ContrastiveBackend),
	)
	fmt.Printf("profile delta: matmul_bind_calls=%d matmul_runs=%d matmul_run_upload_mb=%.2f matmul_run_download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		summary.DeltaProfile.ForwardResidency.MatMul.BindCalls,
		summary.DeltaProfile.ForwardResidency.MatMul.RunCalls,
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunUploadedBytes)/(1024*1024),
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunDownloadedBytes)/(1024*1024),
		summary.DeltaProfile.Optimizer.UpdateCalls,
		summary.DeltaProfile.Activation.GELUBackwardCalls+summary.DeltaProfile.Activation.SoftmaxBackwardCalls+summary.DeltaProfile.Activation.LayerNormBackwardCalls,
		summary.DeltaProfile.Contrastive.RunCalls,
	)
	fmt.Printf("checkpoint: %s\n", paths.Package.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.Package.TrainProfilePath)
	if metricsJSONPath != "" {
		extra := map[string]string{
			"tokenizer":   paths.TokenizerPath,
			"train_pairs": paths.TrainPairsPath,
		}
		if paths.EvalPairsPath != "" {
			extra["eval_pairs"] = paths.EvalPairsPath
		}
		if err := writeTrainMetricsJSON(metricsJSONPath, "train-corpus", "train", path, paths.TokenizerPath, summary, paths.Package, extra); err != nil {
			return err
		}
		fmt.Printf("metrics: %s\n", metricsJSONPath)
	}
	return nil
}

type trainMetricsJSON struct {
	Schema       string                `json:"schema"`
	Command      string                `json:"command"`
	Mode         string                `json:"mode"`
	Artifact     string                `json:"artifact"`
	Tokenizer    string                `json:"tokenizer,omitempty"`
	Summary      trainRunSummaryJSON   `json:"summary"`
	Config       trainRunConfigJSON    `json:"config"`
	FinalTrain   trainBatchMetricsJSON `json:"final_train"`
	LastEval     *evalMetricsJSON      `json:"last_eval,omitempty"`
	BestEval     *evalMetricsJSON      `json:"best_eval,omitempty"`
	FinalEval    *evalMetricsJSON      `json:"final_eval,omitempty"`
	Workload     trainWorkloadJSON     `json:"workload"`
	Throughput   trainThroughputJSON   `json:"throughput"`
	Accelerators trainAcceleratorsJSON `json:"accelerators"`
	ProfileDelta trainProfileDeltaJSON `json:"profile_delta"`
	Package      trainPackagePathsJSON `json:"package"`
	Artifacts    map[string]string     `json:"artifacts,omitempty"`
}

type trainRunSummaryJSON struct {
	EpochsCompleted int  `json:"epochs_completed"`
	StepsCompleted  int  `json:"steps_completed"`
	StepsRun        int  `json:"steps_run"`
	BestEpoch       int  `json:"best_epoch"`
	BestStep        int  `json:"best_step"`
	RestoredBest    bool `json:"restored_best"`
	StoppedEarly    bool `json:"stopped_early"`
}

type trainRunConfigJSON struct {
	Epochs                    int                `json:"epochs"`
	BatchSize                 int                `json:"batch_size"`
	Shuffle                   bool               `json:"shuffle"`
	Seed                      int64              `json:"seed"`
	EvalEveryEpoch            int                `json:"eval_every_epoch"`
	EvalEverySteps            int                `json:"eval_every_steps"`
	Patience                  int                `json:"patience"`
	SelectMetric              string             `json:"select_metric"`
	MinDelta                  float32            `json:"min_delta"`
	RestoreBest               bool               `json:"restore_best"`
	LengthBucketBatches       bool               `json:"length_bucket_batches"`
	LearningRate              float32            `json:"learning_rate"`
	ContrastiveLoss           string             `json:"contrastive_loss,omitempty"`
	Temperature               float32            `json:"temperature"`
	GroupedLossWeight         float32            `json:"grouped_loss_weight,omitempty"`
	TeacherLossWeight         float32            `json:"teacher_loss_weight,omitempty"`
	TeacherTemperature        float32            `json:"teacher_temperature,omitempty"`
	TeacherSourceTemperatures map[string]float32 `json:"teacher_source_temperatures,omitempty"`
	TeacherScoreNormalization string             `json:"teacher_score_normalization,omitempty"`
	ProgressEverySteps        int                `json:"progress_every_steps"`
	EvalOnly                  bool               `json:"eval_only"`
	PairwiseTrain             bool               `json:"pairwise_train"`
	HardNegativeTrain         bool               `json:"hard_negative_train"`
	HardNegativesPerQuery     int                `json:"hard_negatives_per_query"`
	HardNegativeSourceWeights map[string]int     `json:"hard_negative_source_weights,omitempty"`
}

type trainBatchMetricsJSON struct {
	Loss         float32 `json:"loss"`
	AverageScore float32 `json:"average_score"`
	BatchSize    int     `json:"batch_size"`
}

type evalMetricsJSON struct {
	Loss               float32 `json:"loss"`
	AverageScore       float32 `json:"average_score"`
	PositiveMeanScore  float32 `json:"positive_mean_score"`
	NegativeMeanScore  float32 `json:"negative_mean_score"`
	PairAccuracy       float32 `json:"pair_accuracy"`
	ThresholdAccuracy  float32 `json:"threshold_accuracy"`
	ScoreThreshold     float32 `json:"score_threshold"`
	ROCAUC             float32 `json:"roc_auc"`
	ScoreMargin        float32 `json:"score_margin"`
	Top1Accuracy       float32 `json:"top1_accuracy"`
	Top5Accuracy       float32 `json:"top5_accuracy"`
	Top10Accuracy      float32 `json:"top10_accuracy"`
	MeanReciprocalRank float32 `json:"mean_reciprocal_rank"`
	MeanPositiveRank   float32 `json:"mean_positive_rank"`
	PairCount          int     `json:"pair_count"`
	PositiveCount      int     `json:"positive_count"`
	NegativeCount      int     `json:"negative_count"`
}

type trainWorkloadJSON struct {
	TrainMode            string `json:"train_mode"`
	EvalMode             string `json:"eval_mode,omitempty"`
	TrainExamples        int    `json:"train_examples"`
	EvalExamples         int    `json:"eval_examples"`
	BatchSize            int    `json:"batch_size"`
	PlannedEpochs        int    `json:"planned_epochs"`
	CompletedEpochs      int    `json:"completed_epochs"`
	TrainBatchesPerEpoch int    `json:"train_batches_per_epoch"`
	TrainPairsPerEpoch   int64  `json:"train_pairs_per_epoch"`
	EvalPairsPerPass     int64  `json:"eval_pairs_per_pass"`
	PlannedEvalPasses    int    `json:"planned_eval_passes"`
	ActualEvalPasses     int    `json:"actual_eval_passes"`
	PlannedTrainPairs    int64  `json:"planned_train_pairs"`
	ActualTrainPairs     int64  `json:"actual_train_pairs"`
	ActualTrainExamples  int64  `json:"actual_train_examples"`
	PlannedEvalPairs     int64  `json:"planned_eval_pairs"`
	ActualEvalPairs      int64  `json:"actual_eval_pairs"`
	ActualEvalExamples   int64  `json:"actual_eval_examples"`
	PlannedTotalPairs    int64  `json:"planned_total_pairs"`
	ActualTotalPairs     int64  `json:"actual_total_pairs"`
	ActualTotalExamples  int64  `json:"actual_total_examples"`
}

type trainThroughputJSON struct {
	ElapsedSeconds          float64 `json:"elapsed_seconds"`
	TrainSeconds            float64 `json:"train_seconds"`
	EvalSeconds             float64 `json:"eval_seconds"`
	ExamplesPerSecond       float64 `json:"examples_per_second"`
	PairsPerSecond          float64 `json:"pairs_per_second"`
	TrainExamplesPerSecond  float64 `json:"train_examples_per_second"`
	TrainPairsPerSecond     float64 `json:"train_pairs_per_second"`
	EvalExamplesPerSecond   float64 `json:"eval_examples_per_second"`
	EvalPairsPerSecond      float64 `json:"eval_pairs_per_second"`
	OptimizerStepsPerSecond float64 `json:"optimizer_steps_per_second"`
}

type trainAcceleratorsJSON struct {
	Forward     string `json:"forward"`
	Optimizer   string `json:"optimizer"`
	Activation  string `json:"activation"`
	Contrastive string `json:"contrastive"`
}

type trainProfileDeltaJSON struct {
	MatMulBindCalls     int64   `json:"matmul_bind_calls"`
	MatMulRuns          int64   `json:"matmul_runs"`
	MatMulRunUploadMB   float64 `json:"matmul_run_upload_mb"`
	MatMulRunDownloadMB float64 `json:"matmul_run_download_mb"`
	OptimizerUpdates    int64   `json:"optimizer_updates"`
	OptimizerSyncs      int64   `json:"optimizer_syncs"`
	ActivationCalls     int64   `json:"activation_calls"`
	ContrastiveCalls    int64   `json:"contrastive_calls"`
}

type trainPackagePathsJSON struct {
	Artifact     string `json:"artifact"`
	Checkpoint   string `json:"checkpoint"`
	TrainProfile string `json:"train_profile"`
	Manifest     string `json:"manifest,omitempty"`
	Weights      string `json:"weights,omitempty"`
	MemoryPlan   string `json:"memory_plan,omitempty"`
	Package      string `json:"package,omitempty"`
}

func writeTrainMetricsJSON(outputPath, command, mode, artifactPath, tokenizerPath string, summary mantaruntime.EmbeddingTrainRunSummary, paths mantaruntime.EmbeddingTrainPackagePaths, extraArtifacts map[string]string) error {
	payload := trainMetricsPayload(command, mode, artifactPath, tokenizerPath, summary, paths, extraArtifacts)
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("encode metrics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(outputPath, data, 0o644); err != nil {
		return fmt.Errorf("write metrics JSON %q: %w", outputPath, err)
	}
	return nil
}

func trainMetricsPayload(command, mode, artifactPath, tokenizerPath string, summary mantaruntime.EmbeddingTrainRunSummary, paths mantaruntime.EmbeddingTrainPackagePaths, extraArtifacts map[string]string) trainMetricsJSON {
	return trainMetricsJSON{
		Schema:       "manta.embedding_train_metrics.v1",
		Command:      command,
		Mode:         mode,
		Artifact:     artifactPath,
		Tokenizer:    tokenizerPath,
		Summary:      trainRunSummaryPayload(summary),
		Config:       trainRunConfigPayload(summary.Config),
		FinalTrain:   trainBatchMetricsPayload(summary.FinalTrain),
		LastEval:     evalMetricsPayload(summary.LastEval),
		BestEval:     evalMetricsPayload(summary.BestEval),
		FinalEval:    evalMetricsPayload(summary.FinalEval),
		Workload:     trainWorkloadPayload(summary.Workload),
		Throughput:   trainThroughputPayload(summary),
		Accelerators: trainAcceleratorsPayload(summary.EndProfile),
		ProfileDelta: trainProfileDeltaPayload(summary.DeltaProfile),
		Package:      trainPackagePathsPayload(paths),
		Artifacts:    extraArtifacts,
	}
}

func trainRunSummaryPayload(summary mantaruntime.EmbeddingTrainRunSummary) trainRunSummaryJSON {
	return trainRunSummaryJSON{
		EpochsCompleted: summary.EpochsCompleted,
		StepsCompleted:  summary.StepsCompleted,
		StepsRun:        summary.StepsRun,
		BestEpoch:       summary.BestEpoch,
		BestStep:        summary.BestStep,
		RestoredBest:    summary.RestoredBest,
		StoppedEarly:    summary.StoppedEarly,
	}
}

func trainRunConfigPayload(cfg mantaruntime.EmbeddingTrainRunConfig) trainRunConfigJSON {
	return trainRunConfigJSON{
		Epochs:                    cfg.Epochs,
		BatchSize:                 cfg.BatchSize,
		Shuffle:                   cfg.Shuffle,
		Seed:                      cfg.Seed,
		EvalEveryEpoch:            cfg.EvalEveryEpoch,
		EvalEverySteps:            cfg.EvalEverySteps,
		Patience:                  cfg.EarlyStoppingPatience,
		SelectMetric:              cfg.SelectMetric,
		MinDelta:                  cfg.MinDelta,
		RestoreBest:               cfg.RestoreBest,
		LengthBucketBatches:       cfg.LengthBucketBatches,
		LearningRate:              cfg.LearningRate,
		ContrastiveLoss:           cfg.ContrastiveLoss,
		Temperature:               cfg.Temperature,
		GroupedLossWeight:         cfg.GroupedLossWeight,
		TeacherLossWeight:         cfg.TeacherLossWeight,
		TeacherTemperature:        cfg.TeacherTemperature,
		TeacherSourceTemperatures: cfg.TeacherSourceTemperatures,
		TeacherScoreNormalization: cfg.TeacherScoreNormalization,
		ProgressEverySteps:        cfg.ProgressEverySteps,
		EvalOnly:                  cfg.EvalOnly,
		PairwiseTrain:             cfg.PairwiseTrain,
		HardNegativeTrain:         cfg.HardNegativeTrain,
		HardNegativesPerQuery:     cfg.HardNegativesPerQuery,
		HardNegativeSourceWeights: cfg.HardNegativeSourceWeights,
	}
}

func trainBatchMetricsPayload(metrics mantaruntime.EmbeddingTrainMetrics) trainBatchMetricsJSON {
	return trainBatchMetricsJSON{
		Loss:         metrics.Loss,
		AverageScore: metrics.AverageScore,
		BatchSize:    metrics.BatchSize,
	}
}

func evalMetricsPayload(metrics *mantaruntime.EmbeddingEvalMetrics) *evalMetricsJSON {
	if metrics == nil {
		return nil
	}
	return &evalMetricsJSON{
		Loss:               metrics.Loss,
		AverageScore:       metrics.AverageScore,
		PositiveMeanScore:  metrics.PositiveMeanScore,
		NegativeMeanScore:  metrics.NegativeMeanScore,
		PairAccuracy:       metrics.PairAccuracy,
		ThresholdAccuracy:  metrics.ThresholdAccuracy,
		ScoreThreshold:     metrics.ScoreThreshold,
		ROCAUC:             metrics.ROCAUC,
		ScoreMargin:        metrics.ScoreMargin,
		Top1Accuracy:       metrics.Top1Accuracy,
		Top5Accuracy:       metrics.Top5Accuracy,
		Top10Accuracy:      metrics.Top10Accuracy,
		MeanReciprocalRank: metrics.MeanReciprocalRank,
		MeanPositiveRank:   metrics.MeanPositiveRank,
		PairCount:          metrics.PairCount,
		PositiveCount:      metrics.PositiveCount,
		NegativeCount:      metrics.NegativeCount,
	}
}

func trainWorkloadPayload(workload mantaruntime.EmbeddingTrainWorkload) trainWorkloadJSON {
	return trainWorkloadJSON{
		TrainMode:            workload.TrainMode,
		EvalMode:             workload.EvalMode,
		TrainExamples:        workload.TrainExamples,
		EvalExamples:         workload.EvalExamples,
		BatchSize:            workload.BatchSize,
		PlannedEpochs:        workload.PlannedEpochs,
		CompletedEpochs:      workload.CompletedEpochs,
		TrainBatchesPerEpoch: workload.TrainBatchesPerEpoch,
		TrainPairsPerEpoch:   workload.TrainPairsPerEpoch,
		EvalPairsPerPass:     workload.EvalPairsPerPass,
		PlannedEvalPasses:    workload.PlannedEvalPasses,
		ActualEvalPasses:     workload.ActualEvalPasses,
		PlannedTrainPairs:    workload.PlannedTrainPairs,
		ActualTrainPairs:     workload.ActualTrainPairs,
		ActualTrainExamples:  workload.ActualTrainExamples,
		PlannedEvalPairs:     workload.PlannedEvalPairs,
		ActualEvalPairs:      workload.ActualEvalPairs,
		ActualEvalExamples:   workload.ActualEvalExamples,
		PlannedTotalPairs:    workload.PlannedTotalPairs,
		ActualTotalPairs:     workload.ActualTotalPairs,
		ActualTotalExamples:  workload.ActualTotalExamples,
	}
}

func trainThroughputPayload(summary mantaruntime.EmbeddingTrainRunSummary) trainThroughputJSON {
	return trainThroughputJSON{
		ElapsedSeconds:          summary.Elapsed.Seconds(),
		TrainSeconds:            summary.TrainDuration.Seconds(),
		EvalSeconds:             summary.EvalDuration.Seconds(),
		ExamplesPerSecond:       itemsPerSecond(summary.Workload.ActualTotalExamples, summary.Elapsed),
		PairsPerSecond:          pairsPerSecond(summary.Workload.ActualTotalPairs, summary.Elapsed),
		TrainExamplesPerSecond:  itemsPerSecond(summary.Workload.ActualTrainExamples, summary.TrainDuration),
		TrainPairsPerSecond:     pairsPerSecond(summary.Workload.ActualTrainPairs, summary.TrainDuration),
		EvalExamplesPerSecond:   itemsPerSecond(summary.Workload.ActualEvalExamples, summary.EvalDuration),
		EvalPairsPerSecond:      pairsPerSecond(summary.Workload.ActualEvalPairs, summary.EvalDuration),
		OptimizerStepsPerSecond: itemsPerSecond(int64(summary.StepsRun), summary.TrainDuration),
	}
}

func trainAcceleratorsPayload(profile mantaruntime.EmbeddingTrainProfile) trainAcceleratorsJSON {
	return trainAcceleratorsJSON{
		Forward:     displayTrainBackend(profile.ForwardBackend),
		Optimizer:   displayTrainBackend(profile.OptimizerBackend),
		Activation:  displayTrainBackend(profile.ActivationBackend),
		Contrastive: displayTrainBackend(profile.ContrastiveBackend),
	}
}

func trainProfileDeltaPayload(profile mantaruntime.EmbeddingTrainProfile) trainProfileDeltaJSON {
	return trainProfileDeltaJSON{
		MatMulBindCalls:     profile.ForwardResidency.MatMul.BindCalls,
		MatMulRuns:          profile.ForwardResidency.MatMul.RunCalls,
		MatMulRunUploadMB:   bytesToMiB(profile.ForwardResidency.MatMul.RunUploadedBytes),
		MatMulRunDownloadMB: bytesToMiB(profile.ForwardResidency.MatMul.RunDownloadedBytes),
		OptimizerUpdates:    profile.Optimizer.UpdateCalls,
		OptimizerSyncs:      profile.Optimizer.SyncCalls,
		ActivationCalls:     profile.Activation.GELUBackwardCalls + profile.Activation.SoftmaxBackwardCalls + profile.Activation.LayerNormBackwardCalls,
		ContrastiveCalls:    profile.Contrastive.RunCalls,
	}
}

func trainPackagePathsPayload(paths mantaruntime.EmbeddingTrainPackagePaths) trainPackagePathsJSON {
	return trainPackagePathsJSON{
		Artifact:     paths.ArtifactPath,
		Checkpoint:   paths.CheckpointPath,
		TrainProfile: paths.TrainProfilePath,
		Manifest:     paths.EmbeddingManifestPath,
		Weights:      paths.WeightFilePath,
		MemoryPlan:   paths.MemoryPlanPath,
		Package:      paths.PackageManifestPath,
	}
}

func bytesToMiB(bytes int64) float64 {
	return float64(bytes) / (1024 * 1024)
}

func runCompareTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("compare-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta compare-train-metrics <current.metrics.json> [baseline.metrics.json]")
	}
	currentPath := fs.Arg(0)
	current, err := readTrainMetricsJSON(currentPath)
	if err != nil {
		return err
	}
	var baseline *trainMetricsJSON
	baselinePath := ""
	if fs.NArg() > 1 && fs.Arg(1) != "" {
		baselinePath = fs.Arg(1)
		loaded, err := readTrainMetricsJSON(baselinePath)
		if err != nil {
			return err
		}
		baseline = &loaded
	}
	fmt.Print(formatTrainMetricsReport(currentPath, current, baselinePath, baseline))
	return nil
}

func runCompareRetrievalMetrics(args []string) error {
	fs := flag.NewFlagSet("compare-retrieval-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	metric := fs.String("metric", "ndcg_at_10", "metric to compare for gate mode")
	minRatio := fs.Float64("min-ratio", 1.0, "required current/baseline ratio when --require-win is set")
	requireWin := fs.Bool("require-win", false, "return non-zero unless current metric meets or beats baseline ratio")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta compare-retrieval-metrics [--metric ndcg_at_10] [--min-ratio 1.0] [--require-win] <current.retrieval.metrics.json> <baseline.retrieval.metrics.json>")
	}
	if *minRatio < 0 {
		return fmt.Errorf("min-ratio must be non-negative")
	}
	currentPath := fs.Arg(0)
	baselinePath := fs.Arg(1)
	current, err := readRetrievalMetricsJSON(currentPath)
	if err != nil {
		return err
	}
	baseline, err := readRetrievalMetricsJSON(baselinePath)
	if err != nil {
		return err
	}
	if current.Dataset != baseline.Dataset {
		return fmt.Errorf("dataset mismatch: current=%q baseline=%q", current.Dataset, baseline.Dataset)
	}
	fmt.Printf("current: %s backend=%s dataset=%s\n", currentPath, current.Backend, current.Dataset)
	fmt.Printf("baseline: %s backend=%s dataset=%s\n", baselinePath, baseline.Backend, baseline.Dataset)
	printRetrievalMetricDelta("ndcg_at_10", current.Quality.NDCGAt10, baseline.Quality.NDCGAt10)
	printRetrievalMetricDelta("mrr_at_10", current.Quality.MRRAt10, baseline.Quality.MRRAt10)
	printRetrievalMetricDelta("recall_at_10", current.Quality.RecallAt10, baseline.Quality.RecallAt10)
	printRetrievalMetricDelta("recall_at_100", current.Quality.RecallAt100, baseline.Quality.RecallAt100)
	currentValue, ok := retrievalMetricValue(current, *metric)
	if !ok {
		return fmt.Errorf("metric %q is unavailable", *metric)
	}
	baselineValue, ok := retrievalMetricValue(baseline, *metric)
	if !ok {
		return fmt.Errorf("metric %q is unavailable", *metric)
	}
	required := baselineValue * *minRatio
	ratio := 0.0
	if baselineValue != 0 {
		ratio = currentValue / baselineValue
	}
	fmt.Printf("target: %s=%.6g baseline=%.6g required=%.6g ratio=%.6g\n", *metric, currentValue, baselineValue, required, ratio)
	if *requireWin && !trainMetricThresholdPassed(currentValue, required, ">=") {
		fmt.Printf("retrieval baseline gate: FAIL metric=%s current=%.6g baseline=%.6g min_ratio=%.6g\n", *metric, currentValue, baselineValue, *minRatio)
		return fmt.Errorf("retrieval baseline gate failed")
	}
	if *requireWin {
		fmt.Printf("retrieval baseline gate: PASS metric=%s current=%.6g baseline=%.6g min_ratio=%.6g\n", *metric, currentValue, baselineValue, *minRatio)
	}
	return nil
}

func printRetrievalMetricDelta(name string, current, baseline float64) {
	ratio := 0.0
	if baseline != 0 {
		ratio = current / baseline
	}
	fmt.Printf("%s: current=%.6f baseline=%.6f delta=%+.6f ratio=%.6f\n", name, current, baseline, current-baseline, ratio)
}

func readTrainMetricsJSON(path string) (trainMetricsJSON, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return trainMetricsJSON{}, err
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		return trainMetricsJSON{}, fmt.Errorf("parse metrics JSON %q: %w", path, err)
	}
	if metrics.Schema == "" {
		return trainMetricsJSON{}, fmt.Errorf("metrics JSON %q is missing schema", path)
	}
	return metrics, nil
}

func formatTrainMetricsReport(currentPath string, current trainMetricsJSON, baselinePath string, baseline *trainMetricsJSON) string {
	var b strings.Builder
	fmt.Fprintf(&b, "metrics: %s\n", currentPath)
	fmt.Fprintf(&b, "identity: schema=%s command=%s mode=%s artifact=%s\n", current.Schema, current.Command, current.Mode, current.Artifact)
	if current.FinalEval != nil {
		fmt.Fprintf(&b, "quality: top1=%.6f top5=%.6f top10=%.6f mrr=%.6f auc=%.6f margin=%.6f loss=%.6f mean_rank=%.3f pairs=%d\n",
			current.FinalEval.Top1Accuracy,
			current.FinalEval.Top5Accuracy,
			current.FinalEval.Top10Accuracy,
			current.FinalEval.MeanReciprocalRank,
			current.FinalEval.ROCAUC,
			current.FinalEval.ScoreMargin,
			current.FinalEval.Loss,
			current.FinalEval.MeanPositiveRank,
			current.FinalEval.PairCount,
		)
	}
	fmt.Fprintf(&b, "throughput: train_pairs/s=%.2f eval_pairs/s=%.2f optimizer_steps/s=%.2f pairs/s=%.2f elapsed_s=%.3f\n",
		current.Throughput.TrainPairsPerSecond,
		current.Throughput.EvalPairsPerSecond,
		current.Throughput.OptimizerStepsPerSecond,
		current.Throughput.PairsPerSecond,
		current.Throughput.ElapsedSeconds,
	)
	fmt.Fprintf(&b, "accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		current.Accelerators.Forward,
		current.Accelerators.Optimizer,
		current.Accelerators.Activation,
		current.Accelerators.Contrastive,
	)
	fmt.Fprintf(&b, "profile_delta: matmul_runs=%d upload_mb=%.2f download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		current.ProfileDelta.MatMulRuns,
		current.ProfileDelta.MatMulRunUploadMB,
		current.ProfileDelta.MatMulRunDownloadMB,
		current.ProfileDelta.OptimizerUpdates,
		current.ProfileDelta.ActivationCalls,
		current.ProfileDelta.ContrastiveCalls,
	)
	if baseline == nil {
		return b.String()
	}
	fmt.Fprintf(&b, "baseline: %s\n", baselinePath)
	if current.FinalEval != nil && baseline.FinalEval != nil {
		fmt.Fprintf(&b, "quality_delta: top1=%+.6f top5=%+.6f top10=%+.6f mrr=%+.6f auc=%+.6f margin=%+.6f loss=%+.6f mean_rank=%+.3f\n",
			current.FinalEval.Top1Accuracy-baseline.FinalEval.Top1Accuracy,
			current.FinalEval.Top5Accuracy-baseline.FinalEval.Top5Accuracy,
			current.FinalEval.Top10Accuracy-baseline.FinalEval.Top10Accuracy,
			current.FinalEval.MeanReciprocalRank-baseline.FinalEval.MeanReciprocalRank,
			current.FinalEval.ROCAUC-baseline.FinalEval.ROCAUC,
			current.FinalEval.ScoreMargin-baseline.FinalEval.ScoreMargin,
			current.FinalEval.Loss-baseline.FinalEval.Loss,
			current.FinalEval.MeanPositiveRank-baseline.FinalEval.MeanPositiveRank,
		)
	}
	fmt.Fprintf(&b, "throughput_delta: train_pairs/s=%+.2f eval_pairs/s=%+.2f optimizer_steps/s=%+.2f pairs/s=%+.2f elapsed_s=%+.3f\n",
		current.Throughput.TrainPairsPerSecond-baseline.Throughput.TrainPairsPerSecond,
		current.Throughput.EvalPairsPerSecond-baseline.Throughput.EvalPairsPerSecond,
		current.Throughput.OptimizerStepsPerSecond-baseline.Throughput.OptimizerStepsPerSecond,
		current.Throughput.PairsPerSecond-baseline.Throughput.PairsPerSecond,
		current.Throughput.ElapsedSeconds-baseline.Throughput.ElapsedSeconds,
	)
	fmt.Fprintf(&b, "profile_delta_delta: matmul_runs=%+d upload_mb=%+.2f download_mb=%+.2f optimizer_updates=%+d activation_calls=%+d contrastive_calls=%+d\n",
		current.ProfileDelta.MatMulRuns-baseline.ProfileDelta.MatMulRuns,
		current.ProfileDelta.MatMulRunUploadMB-baseline.ProfileDelta.MatMulRunUploadMB,
		current.ProfileDelta.MatMulRunDownloadMB-baseline.ProfileDelta.MatMulRunDownloadMB,
		current.ProfileDelta.OptimizerUpdates-baseline.ProfileDelta.OptimizerUpdates,
		current.ProfileDelta.ActivationCalls-baseline.ProfileDelta.ActivationCalls,
		current.ProfileDelta.ContrastiveCalls-baseline.ProfileDelta.ContrastiveCalls,
	)
	return b.String()
}

func runDiagnoseTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("diagnose-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta diagnose-train-metrics <metrics.json>")
	}
	metricsPath := fs.Arg(0)
	metrics, err := readTrainMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	fmt.Print(formatTrainMetricsDiagnosis(metricsPath, metrics))
	return nil
}

func formatTrainMetricsDiagnosis(metricsPath string, metrics trainMetricsJSON) string {
	var b strings.Builder
	warnings := 0
	notes := 0
	fmt.Fprintf(&b, "metrics: %s\n", metricsPath)
	fmt.Fprintf(&b, "identity: schema=%s command=%s mode=%s artifact=%s\n", metrics.Schema, metrics.Command, metrics.Mode, metrics.Artifact)
	fmt.Fprintf(&b, "backend: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		metrics.Accelerators.Forward,
		metrics.Accelerators.Optimizer,
		metrics.Accelerators.Activation,
		metrics.Accelerators.Contrastive,
	)
	fmt.Fprintf(&b, "throughput: train_pairs/s=%.2f eval_pairs/s=%.2f optimizer_steps/s=%.2f elapsed_s=%.3f\n",
		metrics.Throughput.TrainPairsPerSecond,
		metrics.Throughput.EvalPairsPerSecond,
		metrics.Throughput.OptimizerStepsPerSecond,
		metrics.Throughput.ElapsedSeconds,
	)
	pairCount := trainMetricsDiagnosisPairCount(metrics)
	if metrics.ProfileDelta.MatMulRuns > 0 {
		parts := []string{}
		if metrics.ProfileDelta.OptimizerUpdates > 0 {
			parts = append(parts, fmt.Sprintf("matmul_runs/update=%.2f", float64(metrics.ProfileDelta.MatMulRuns)/float64(metrics.ProfileDelta.OptimizerUpdates)))
		} else {
			parts = append(parts, fmt.Sprintf("matmul_runs=%d", metrics.ProfileDelta.MatMulRuns))
		}
		if pairCount > 0 {
			parts = append(parts, fmt.Sprintf("pairs/matmul_run=%.2f", float64(pairCount)/float64(metrics.ProfileDelta.MatMulRuns)))
		}
		if metrics.ProfileDelta.OptimizerUpdates > 0 {
			parts = append(parts, fmt.Sprintf("optimizer_syncs/update=%.2f", float64(metrics.ProfileDelta.OptimizerSyncs)/float64(metrics.ProfileDelta.OptimizerUpdates)))
		}
		fmt.Fprintf(&b, "efficiency: %s\n", strings.Join(parts, " "))
	}
	totalTransferMB := metrics.ProfileDelta.MatMulRunUploadMB + metrics.ProfileDelta.MatMulRunDownloadMB
	if totalTransferMB > 0 {
		parts := []string{fmt.Sprintf("total_mb=%.2f", totalTransferMB)}
		if metrics.ProfileDelta.MatMulRuns > 0 {
			parts = append(parts, fmt.Sprintf("mb/matmul_run=%.4f", totalTransferMB/float64(metrics.ProfileDelta.MatMulRuns)))
		}
		if pairCount > 0 {
			parts = append(parts, fmt.Sprintf("kb/pair=%.4f", totalTransferMB*1024/float64(pairCount)))
		}
		fmt.Fprintf(&b, "transfer: %s\n", strings.Join(parts, " "))
	}
	hostBackends := trainMetricsHostCriticalBackends(metrics)
	if len(hostBackends) == 0 {
		fmt.Fprintf(&b, "finding: ok production-critical accelerators are device-backed\n")
	} else {
		warnings++
		fmt.Fprintf(&b, "finding: warn production-critical accelerators include host fallback: %s\n", strings.Join(hostBackends, " "))
	}
	if trainMetricsEvalOnly(metrics) {
		if metrics.ProfileDelta.OptimizerUpdates == 0 {
			fmt.Fprintf(&b, "finding: ok eval-only run recorded zero optimizer updates\n")
		} else {
			warnings++
			fmt.Fprintf(&b, "finding: warn eval-only run recorded optimizer_updates=%d\n", metrics.ProfileDelta.OptimizerUpdates)
		}
	} else {
		if metrics.ProfileDelta.OptimizerUpdates == 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn training run recorded zero optimizer updates\n")
		} else if metrics.Throughput.OptimizerStepsPerSecond == 0 && metrics.Throughput.TrainSeconds > 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn optimizer updates were recorded but optimizer_steps/s is zero\n")
		}
		if metrics.Throughput.TrainPairsPerSecond == 0 && metrics.Workload.ActualTrainPairs > 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn training pairs were processed but train_pairs/s is zero\n")
		}
		if metrics.Throughput.EvalSeconds > metrics.Throughput.TrainSeconds && metrics.Throughput.TrainSeconds > 0 {
			notes++
			fmt.Fprintf(&b, "finding: note eval time exceeds train time; keep production gates in final eval-only passes unless debugging convergence\n")
		}
	}
	if metrics.Accelerators.Activation == "" || metrics.Accelerators.Activation == "host" {
		notes++
		fmt.Fprintf(&b, "finding: note activation accelerator is host; this can be intentional until activation residency avoids extra transfers\n")
	}
	status := "OK"
	if warnings > 0 {
		status = "WARN"
	}
	fmt.Fprintf(&b, "diagnosis: %s warnings=%d notes=%d\n", status, warnings, notes)
	return b.String()
}

func trainMetricsDiagnosisPairCount(metrics trainMetricsJSON) int64 {
	if metrics.Workload.ActualTrainPairs > 0 {
		return metrics.Workload.ActualTrainPairs
	}
	if metrics.Workload.ActualEvalPairs > 0 {
		return metrics.Workload.ActualEvalPairs
	}
	return metrics.Workload.ActualTotalPairs
}

func trainMetricsEvalOnly(metrics trainMetricsJSON) bool {
	return strings.EqualFold(metrics.Mode, "eval") || metrics.Config.EvalOnly
}

func trainMetricsHostCriticalBackends(metrics trainMetricsJSON) []string {
	backends := []string{}
	if trainMetricsHostBackend(metrics.Accelerators.Forward) {
		backends = append(backends, "forward="+displayBackendLabel(metrics.Accelerators.Forward))
	}
	if !trainMetricsEvalOnly(metrics) && trainMetricsHostBackend(metrics.Accelerators.Optimizer) {
		backends = append(backends, "optimizer="+displayBackendLabel(metrics.Accelerators.Optimizer))
	}
	if trainMetricsHostBackend(metrics.Accelerators.Contrastive) {
		backends = append(backends, "contrastive="+displayBackendLabel(metrics.Accelerators.Contrastive))
	}
	return backends
}

func trainMetricsHostBackend(backend string) bool {
	return backend == "" || backend == "host"
}

func displayBackendLabel(backend string) string {
	if backend == "" {
		return "unknown"
	}
	return backend
}

type trainMetricThreshold struct {
	Env    string
	Metric string
	Op     string
	Scope  string
}

var trainMetricThresholds = []trainMetricThreshold{
	{Env: "MANTA_MIN_MRR", Metric: "mrr", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_TOP1", Metric: "top1", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_TOP5", Metric: "top5", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_TOP10", Metric: "top10", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MAX_MEAN_RANK", Metric: "mean_rank", Op: "<=", Scope: "quality"},
	{Env: "MANTA_MIN_AUC", Metric: "auc", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_THRESHOLD_ACCURACY", Metric: "threshold_accuracy", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_SCORE_MARGIN", Metric: "margin", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_PAIR_ACCURACY", Metric: "accuracy", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MAX_LOSS", Metric: "loss", Op: "<=", Scope: "quality"},
	{Env: "MANTA_MIN_TRAIN_PAIRS_PER_SEC", Metric: "train_pairs/s", Op: ">=", Scope: "efficiency"},
	{Env: "MANTA_MIN_OPTIMIZER_STEPS_PER_SEC", Metric: "optimizer_steps/s", Op: ">=", Scope: "efficiency"},
	{Env: "MANTA_MAX_MATMUL_RUNS", Metric: "matmul_runs", Op: "<=", Scope: "efficiency"},
	{Env: "MANTA_MAX_MATMUL_RUN_UPLOAD_MB", Metric: "matmul_run_upload_mb", Op: "<=", Scope: "efficiency"},
	{Env: "MANTA_MAX_MATMUL_RUN_DOWNLOAD_MB", Metric: "matmul_run_download_mb", Op: "<=", Scope: "efficiency"},
	{Env: "MANTA_MAX_OPTIMIZER_UPDATES", Metric: "optimizer_updates", Op: "<=", Scope: "eval-only"},
}

func runGateTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("gate-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var thresholdsPath string
	var scope string
	fs.StringVar(&thresholdsPath, "thresholds", "", "optional KEY=VALUE threshold env file")
	fs.StringVar(&scope, "scope", "all", "gate scope: all, quality, efficiency, or eval-only")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta gate-train-metrics [--thresholds thresholds.env] [--scope all|quality|efficiency|eval-only] <metrics.json>")
	}
	scope = strings.ToLower(scope)
	if !validTrainMetricsGateScope(scope) {
		return fmt.Errorf("unsupported gate scope %q", scope)
	}
	metricsPath := fs.Arg(0)
	metrics, err := readTrainMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	thresholds, err := trainMetricThresholdValues(thresholdsPath)
	if err != nil {
		return err
	}
	fmt.Printf("metrics: %s\n", metricsPath)
	if thresholdsPath != "" {
		fmt.Printf("thresholds: %s\n", thresholdsPath)
	}
	fmt.Printf("scope: %s\n", scope)
	checked := 0
	failed := 0
	for _, threshold := range trainMetricThresholds {
		if !trainMetricThresholdInScope(threshold.Scope, scope) {
			continue
		}
		limitText := thresholds[threshold.Env]
		if limitText == "" {
			continue
		}
		limit, err := strconv.ParseFloat(limitText, 64)
		if err != nil {
			return fmt.Errorf("%s=%q is not numeric: %w", threshold.Env, limitText, err)
		}
		got, ok := trainMetricValue(metrics, threshold.Metric)
		if !ok {
			return fmt.Errorf("metric %s is unavailable in %s", threshold.Metric, metricsPath)
		}
		checked++
		passed := trainMetricThresholdPassed(got, limit, threshold.Op)
		status := "pass"
		if !passed {
			status = "fail"
			failed++
		}
		fmt.Printf("%s: %s=%.6g %s %.6g (%s)\n", status, threshold.Metric, got, threshold.Op, limit, threshold.Env)
	}
	if scope == "eval-only" {
		got, ok := trainMetricValue(metrics, "optimizer_updates")
		if !ok {
			return fmt.Errorf("metric optimizer_updates is unavailable in %s", metricsPath)
		}
		checked++
		if got == 0 {
			fmt.Printf("pass: optimizer_updates=%.6g == 0 (eval-only)\n", got)
		} else {
			fmt.Printf("fail: optimizer_updates=%.6g == 0 (eval-only)\n", got)
			failed++
		}
	}
	if checked == 0 {
		return fmt.Errorf("no thresholds selected for scope %q", scope)
	}
	if failed > 0 {
		fmt.Printf("gate: FAIL checks=%d failed=%d\n", checked, failed)
		return fmt.Errorf("metrics gate failed")
	}
	fmt.Printf("gate: PASS checks=%d\n", checked)
	return nil
}

func validTrainMetricsGateScope(scope string) bool {
	switch scope {
	case "all", "quality", "efficiency", "eval-only":
		return true
	default:
		return false
	}
}

func trainMetricThresholdInScope(thresholdScope, requestedScope string) bool {
	switch requestedScope {
	case "all":
		return thresholdScope == "quality" || thresholdScope == "efficiency"
	default:
		return thresholdScope == requestedScope
	}
}

func trainMetricThresholdValues(path string) (map[string]string, error) {
	values := map[string]string{}
	for _, env := range os.Environ() {
		key, value, ok := strings.Cut(env, "=")
		if ok && strings.HasPrefix(key, "MANTA_") {
			values[key] = value
		}
	}
	if path == "" {
		return values, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected KEY=VALUE", path, i+1)
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if !strings.HasPrefix(key, "MANTA_") {
			return nil, fmt.Errorf("%s:%d: threshold key must start with MANTA_: %s", path, i+1, key)
		}
		if values[key] == "" {
			values[key] = value
		}
	}
	return values, nil
}

func trainMetricValue(metrics trainMetricsJSON, metric string) (float64, bool) {
	switch metric {
	case "train_pairs/s":
		return metrics.Throughput.TrainPairsPerSecond, true
	case "optimizer_steps/s":
		return metrics.Throughput.OptimizerStepsPerSecond, true
	case "matmul_runs":
		return float64(metrics.ProfileDelta.MatMulRuns), true
	case "matmul_run_upload_mb":
		return metrics.ProfileDelta.MatMulRunUploadMB, true
	case "matmul_run_download_mb":
		return metrics.ProfileDelta.MatMulRunDownloadMB, true
	case "optimizer_updates":
		return float64(metrics.ProfileDelta.OptimizerUpdates), true
	}
	if metrics.FinalEval == nil {
		return 0, false
	}
	switch metric {
	case "mrr":
		return float64(metrics.FinalEval.MeanReciprocalRank), true
	case "top1":
		return float64(metrics.FinalEval.Top1Accuracy), true
	case "top5":
		return float64(metrics.FinalEval.Top5Accuracy), true
	case "top10":
		return float64(metrics.FinalEval.Top10Accuracy), true
	case "mean_rank":
		return float64(metrics.FinalEval.MeanPositiveRank), true
	case "auc":
		return float64(metrics.FinalEval.ROCAUC), true
	case "threshold_accuracy":
		return float64(metrics.FinalEval.ThresholdAccuracy), true
	case "margin":
		return float64(metrics.FinalEval.ScoreMargin), true
	case "accuracy":
		return float64(metrics.FinalEval.PairAccuracy), true
	case "loss":
		return float64(metrics.FinalEval.Loss), true
	default:
		return 0, false
	}
}

func trainMetricThresholdPassed(got, limit float64, op string) bool {
	const epsilon = 1e-6
	switch op {
	case ">=":
		return got >= limit-epsilon
	case "<=":
		return got <= limit+epsilon
	default:
		return false
	}
}

type retrievalMetricThreshold struct {
	Env    string
	Metric string
	Op     string
	Scope  string
}

var retrievalMetricThresholds = []retrievalMetricThreshold{
	{Env: "MANTA_MIN_RETRIEVAL_NDCG10", Metric: "ndcg_at_10", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_RETRIEVAL_MRR10", Metric: "mrr_at_10", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_RETRIEVAL_RECALL10", Metric: "recall_at_10", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_RETRIEVAL_RECALL100", Metric: "recall_at_100", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_RETRIEVAL_DOCUMENTS_PER_SEC", Metric: "documents/s", Op: ">=", Scope: "efficiency"},
	{Env: "MANTA_MIN_RETRIEVAL_QUERIES_PER_SEC", Metric: "queries/s", Op: ">=", Scope: "efficiency"},
	{Env: "MANTA_MIN_RETRIEVAL_SCORES_PER_SEC", Metric: "scores/s", Op: ">=", Scope: "efficiency"},
}

func runGateRetrievalMetrics(args []string) error {
	fs := flag.NewFlagSet("gate-retrieval-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var thresholdsPath string
	var scope string
	fs.StringVar(&thresholdsPath, "thresholds", "", "optional KEY=VALUE threshold env file")
	fs.StringVar(&scope, "scope", "all", "gate scope: all, quality, or efficiency")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta gate-retrieval-metrics [--thresholds thresholds.env] [--scope all|quality|efficiency] <retrieval.metrics.json>")
	}
	scope = strings.ToLower(scope)
	if !validRetrievalMetricsGateScope(scope) {
		return fmt.Errorf("unsupported gate scope %q", scope)
	}
	metricsPath := fs.Arg(0)
	metrics, err := readRetrievalMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	thresholds, err := trainMetricThresholdValues(thresholdsPath)
	if err != nil {
		return err
	}
	fmt.Printf("metrics: %s\n", metricsPath)
	if thresholdsPath != "" {
		fmt.Printf("thresholds: %s\n", thresholdsPath)
	}
	fmt.Printf("dataset: %s\n", metrics.Dataset)
	fmt.Printf("scope: %s\n", scope)
	checked := 0
	failed := 0
	for _, threshold := range retrievalMetricThresholds {
		if !trainMetricThresholdInScope(threshold.Scope, scope) {
			continue
		}
		envName, limitText := retrievalThresholdValue(thresholds, threshold.Env, metrics.Dataset)
		if limitText == "" {
			continue
		}
		limit, err := strconv.ParseFloat(limitText, 64)
		if err != nil {
			return fmt.Errorf("%s=%q is not numeric: %w", envName, limitText, err)
		}
		got, ok := retrievalMetricValue(metrics, threshold.Metric)
		if !ok {
			return fmt.Errorf("metric %s is unavailable in %s", threshold.Metric, metricsPath)
		}
		checked++
		passed := trainMetricThresholdPassed(got, limit, threshold.Op)
		status := "pass"
		if !passed {
			status = "fail"
			failed++
		}
		fmt.Printf("%s: %s=%.6g %s %.6g (%s)\n", status, threshold.Metric, got, threshold.Op, limit, envName)
	}
	if checked == 0 {
		return fmt.Errorf("no retrieval thresholds selected for scope %q", scope)
	}
	if failed > 0 {
		fmt.Printf("retrieval gate: FAIL checks=%d failed=%d\n", checked, failed)
		return fmt.Errorf("retrieval metrics gate failed")
	}
	fmt.Printf("retrieval gate: PASS checks=%d\n", checked)
	return nil
}

func readRetrievalMetricsJSON(path string) (mantaruntime.RetrievalEvalMetrics, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return mantaruntime.RetrievalEvalMetrics{}, err
	}
	var metrics mantaruntime.RetrievalEvalMetrics
	if err := json.Unmarshal(data, &metrics); err != nil {
		return mantaruntime.RetrievalEvalMetrics{}, fmt.Errorf("parse retrieval metrics JSON %q: %w", path, err)
	}
	if metrics.Schema != mantaruntime.RetrievalEvalMetricsSchema {
		return mantaruntime.RetrievalEvalMetrics{}, fmt.Errorf("unsupported retrieval metrics schema %q", metrics.Schema)
	}
	return metrics, nil
}

func validRetrievalMetricsGateScope(scope string) bool {
	switch scope {
	case "all", "quality", "efficiency":
		return true
	default:
		return false
	}
}

func retrievalThresholdValue(values map[string]string, baseEnv, dataset string) (string, string) {
	if suffix := retrievalDatasetEnvSuffix(dataset); suffix != "" {
		datasetEnv := baseEnv + "_" + suffix
		if value := values[datasetEnv]; value != "" {
			return datasetEnv, value
		}
	}
	return baseEnv, values[baseEnv]
}

func retrievalDatasetEnvSuffix(dataset string) string {
	dataset = strings.ToUpper(strings.TrimSpace(dataset))
	if dataset == "" {
		return ""
	}
	var b strings.Builder
	lastUnderscore := false
	for _, r := range dataset {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	return strings.Trim(b.String(), "_")
}

func retrievalMetricValue(metrics mantaruntime.RetrievalEvalMetrics, metric string) (float64, bool) {
	switch metric {
	case "ndcg_at_10":
		return metrics.Quality.NDCGAt10, true
	case "mrr_at_10":
		return metrics.Quality.MRRAt10, true
	case "recall_at_10":
		return metrics.Quality.RecallAt10, true
	case "recall_at_100":
		return metrics.Quality.RecallAt100, true
	case "documents/s":
		return metrics.Throughput.DocumentsPerSecond, true
	case "queries/s":
		return metrics.Throughput.QueriesPerSecond, true
	case "scores/s":
		return metrics.Throughput.ScoresPerSecond, true
	default:
		return 0, false
	}
}

func runTrainTokenizer(args []string) error {
	fs := flag.NewFlagSet("train-tokenizer", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var outputPath string
	var manifestPath string
	var vocabSize int
	var minFreq int
	fs.StringVar(&outputPath, "output", "", "output tokenizer path")
	fs.StringVar(&manifestPath, "manifest", "", "embedding manifest path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for merges")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	}
	artifactPath := fs.Arg(0)
	corpusPath := fs.Arg(1)
	if outputPath == "" {
		outputPath = mantaruntime.DefaultTokenizerPath(artifactPath)
	}
	if manifestPath == "" {
		manifestPath = mantaruntime.DefaultEmbeddingManifestPath(artifactPath)
	}
	if vocabSize == 0 {
		manifest, err := mantaruntime.ReadEmbeddingManifestFile(manifestPath)
		if err != nil {
			return fmt.Errorf("read embedding manifest for vocab size: %w", err)
		}
		vocabSize = manifest.Tokenizer.VocabSize
	}
	if vocabSize <= 0 {
		return fmt.Errorf("tokenizer vocab size must be set via --vocab-size or embedding manifest")
	}
	tokenizer, err := mantaruntime.TrainTokenizerFromCorpus(mantaruntime.TokenizerTrainConfig{
		CorpusPath: corpusPath,
		VocabSize:  vocabSize,
		MinFreq:    minFreq,
	})
	if err != nil {
		return err
	}
	if err := tokenizer.WriteFile(outputPath); err != nil {
		return err
	}
	if err := mantaruntime.SyncEmbeddingTokenizerVocab(artifactPath, len(tokenizer.Tokens)); err != nil {
		return fmt.Errorf("sync tokenizer vocab through Manta package: %w", err)
	}
	fmt.Printf("trained tokenizer %q\n", outputPath)
	fmt.Printf("vocab: %d tokens, merges: %d\n", len(tokenizer.Tokens), len(tokenizer.Merges))
	return nil
}

func runTokenizeEmbed(args []string) error {
	fs := flag.NewFlagSet("tokenize-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var tokenizerPath string
	var mode string
	var hardNegativesPerQuery int
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer .mll (default: sibling tokenizer)")
	fs.StringVar(&mode, "mode", "contrastive", "output mode: contrastive, pair, or hard-negative")
	fs.IntVar(&hardNegativesPerQuery, "hard-negatives-per-query", 1, "maximum explicit negatives per query-positive example in hard-negative mode")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: manta tokenize-embed [--mode contrastive|pair|hard-negative] [--tokenizer tokenizer.mll] <artifact.mll> <input-text.jsonl> <output-token.jsonl>")
	}
	if hardNegativesPerQuery < 0 {
		return fmt.Errorf("hard-negatives-per-query must be non-negative")
	}
	artifactPath := fs.Arg(0)
	inputPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if tokenizerPath == "" {
		tokenizerPath = mantaruntime.DefaultTokenizerPath(artifactPath)
	}
	tokenizerFile, err := mantaruntime.ReadTokenizerFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("read tokenizer: %w", err)
	}
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(mantaruntime.ResolveEmbeddingManifestPath(artifactPath))
	if err != nil {
		return fmt.Errorf("read embedding manifest: %w", err)
	}
	tokenizer, err := mantaruntime.NewBPETokenizer(tokenizerFile, manifest.Tokenizer)
	if err != nil {
		return fmt.Errorf("build tokenizer: %w", err)
	}
	switch strings.ToLower(mode) {
	case "contrastive":
		examples, err := mantaruntime.ReadEmbeddingTextContrastiveExamplesFile(inputPath)
		if err != nil {
			return fmt.Errorf("read text contrastive dataset: %w", err)
		}
		tokenized, err := mantaruntime.TokenizeEmbeddingTextContrastiveExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize contrastive dataset: %w", err)
		}
		if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized contrastive examples: %d\n", len(tokenized))
	case "pair":
		examples, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(inputPath)
		if err != nil {
			return fmt.Errorf("read text pair dataset: %w", err)
		}
		tokenized, err := mantaruntime.TokenizeEmbeddingTextPairExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize pair dataset: %w", err)
		}
		if err := mantaruntime.WriteEmbeddingPairExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized pair examples: %d\n", len(tokenized))
	case "hard-negative", "hard_negative":
		examples, err := mantaruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
		if err != nil {
			pairs, pairErr := mantaruntime.ReadEmbeddingTextPairExamplesFile(inputPath)
			if pairErr != nil {
				return fmt.Errorf("read text hard-negative dataset: %w", err)
			}
			examples, err = mantaruntime.BuildEmbeddingTextHardNegativeExamplesFromPairs(pairs, hardNegativesPerQuery)
			if err != nil {
				return fmt.Errorf("build text hard-negative dataset: %w", err)
			}
		}
		tokenized, err := mantaruntime.TokenizeEmbeddingTextHardNegativeExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize hard-negative dataset: %w", err)
		}
		if err := mantaruntime.WriteEmbeddingHardNegativeExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized hard-negative examples: %d\n", len(tokenized))
	default:
		return fmt.Errorf("unsupported tokenize mode %q: want contrastive, pair, or hard-negative", mode)
	}
	fmt.Printf("tokenizer: %s\n", tokenizerPath)
	fmt.Printf("input: %s\n", inputPath)
	fmt.Printf("output: %s\n", outputPath)
	return nil
}

func runMineTextPairs(args []string) error {
	fs := flag.NewFlagSet("mine-text-pairs", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var minChars int
	var maxPairs int
	var evalPairs int
	var seed int64
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed used when truncating mined pairs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta mine-text-pairs [flags] <corpus.txt> <train.jsonl> [eval.jsonl]")
	}
	corpusPath := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	trainSet, evalSet, err := mantaruntime.MineEmbeddingTextDatasetsFromCorpusFile(corpusPath, mantaruntime.EmbeddingTextMiningConfig{
		MinChars:  minChars,
		MaxPairs:  maxPairs,
		EvalPairs: evalPairs,
		Seed:      seed,
	})
	if err != nil {
		return err
	}
	if err := mantaruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, trainSet); err != nil {
		return err
	}
	if evalPath != "" && len(evalSet) > 0 {
		if err := mantaruntime.WriteEmbeddingTextPairExamplesFile(evalPath, evalSet); err != nil {
			return err
		}
	}
	fmt.Printf("mined train pairs: %d\n", len(trainSet))
	if evalPath != "" {
		fmt.Printf("mined eval pairs: %d\n", len(evalSet))
	}
	fmt.Printf("train: %s\n", trainPath)
	if evalPath != "" {
		fmt.Printf("eval: %s\n", evalPath)
	}
	return nil
}

func printUsage() {
	fmt.Println("usage:")
	fmt.Println("  manta version")
	fmt.Println("  manta compile <source.manta> [output.mll]")
	fmt.Println("  manta inspect <artifact.mll>")
	fmt.Println("  manta export-mll <artifact.mll> [output.mll]")
	fmt.Println("  manta embed-text <artifact.mll> <text...>")
	fmt.Println("  manta eval-retrieval [flags] <artifact.mll> <beir-dataset-dir>")
	fmt.Println("  manta eval-retrieval-bm25 [flags] <beir-dataset-dir>")
	fmt.Println("  manta mine-retrieval-hard-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	fmt.Println("  manta mine-retrieval-model-hard-negatives [flags] <artifact.mll> <beir-dataset-dir> <output.jsonl>")
	fmt.Println("  manta import-teacher-scores [flags] <hard-negatives.jsonl> <scores.jsonl> <output.jsonl>")
	fmt.Println("  manta score-teacher-hard-negatives [flags] <teacher.mll> <hard-negatives.jsonl> <output.jsonl>")
	fmt.Println("  manta audit-teacher-scores [flags] <hard-negatives.jsonl> [summary.json]")
	fmt.Println("  manta plan-sparse-attention [flags]")
	fmt.Println("  manta init-model [flags] <artifact.mll>")
	fmt.Println("  manta init-mirage [flags] <artifact.mll>")
	fmt.Println("  manta init-train [flags] <artifact.mll>")
	fmt.Println("  manta rename-embed --name <model-name> <input.mll> <output.mll>")
	fmt.Println("  manta train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  manta tokenize-embed [flags] <artifact.mll> <input-text.jsonl> <output-token.jsonl>")
	fmt.Println("  manta train-corpus [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  manta train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]")
	fmt.Println("  manta compare-train-metrics <current.metrics.json> [baseline.metrics.json]")
	fmt.Println("  manta compare-retrieval-metrics <current.retrieval.metrics.json> <baseline.retrieval.metrics.json>")
	fmt.Println("  manta diagnose-train-metrics <metrics.json>")
	fmt.Println("  manta gate-train-metrics [flags] <metrics.json>")
	fmt.Println("  manta gate-retrieval-metrics [flags] <retrieval.metrics.json>")
	fmt.Println("  manta run <artifact.mll> [entry]")
	fmt.Println("  manta demo [module-name]")
	fmt.Println()
	fmt.Println("compile lowers a Manta source file into an .mll artifact.")
	fmt.Println("inspect summarizes an artifact and verifies its sibling package manifest when present.")
	fmt.Println("export-mll seals an artifact package into a weight-carrying .mll container while preserving Manta metadata in XMTA.")
	fmt.Println("embed-text loads a packaged or sealed embedding .mll and embeds text with its tokenizer.")
	fmt.Println("eval-retrieval scores a sealed embedding .mll on BEIR-style corpus/query/qrels files with nDCG/MRR/Recall metrics.")
	fmt.Println("eval-retrieval-bm25 scores the same BEIR files with an in-repo BM25 lexical baseline.")
	fmt.Println("mine-retrieval-hard-negatives creates text hard-negative training JSONL from BEIR qrels using the BM25 baseline.")
	fmt.Println("mine-retrieval-model-hard-negatives creates text hard-negative training JSONL from BEIR qrels using a Manta embedding model's own misses.")
	fmt.Println("import-teacher-scores merges external teacher score JSONL into text hard-negative JSONL and writes a provenance manifest.")
	fmt.Println("score-teacher-hard-negatives uses a Manta embedding teacher to score existing text hard-negative JSONL into teacher_scores.")
	fmt.Println("audit-teacher-scores summarizes teacher score coverage, positive rank, margins, and entropy before distillation runs.")
	fmt.Println("plan-sparse-attention preflights routed sparse attention plus logical TurboQuant K/V memory budgets before GPU runs.")
	fmt.Println("init-model creates the Manta-owned default quantized embedding training package.")
	fmt.Println("init-mirage creates the Manta-owned Mirage Image v1 host-reference artifact.")
	fmt.Println("init-train creates a native training package next to an artifact.")
	fmt.Println("rename-embed rewrites a training package under a new embedding model identity.")
	fmt.Println("train-tokenizer builds a sibling .tokenizer.mll from a raw text corpus, using embedding-manifest vocab_size by default.")
	fmt.Println("tokenize-embed converts text JSONL into reusable token JSONL for contrastive, pair, or hard-negative training and eval.")
	fmt.Println("train-corpus trains tokenizer + mined text pairs + embedder in one Manta job from a raw text corpus.")
	fmt.Println("train-embed reloads a training package, fits or --eval-only evaluates token JSONL or text JSONL (with --tokenizer or a sibling .tokenizer.mll; use --no-tokenizer for token JSONL beside a tokenizer), and writes it back.")
	fmt.Println("compare-train-metrics summarizes metrics JSON and prints deltas against a baseline metrics JSON when provided.")
	fmt.Println("compare-retrieval-metrics summarizes retrieval quality deltas and can gate a candidate against a baseline.")
	fmt.Println("diagnose-train-metrics explains backend use, transfer pressure, and suspicious training/eval counters from metrics JSON.")
	fmt.Println("gate-train-metrics checks metrics JSON against MANTA_* thresholds from the environment or a thresholds env file.")
	fmt.Println("gate-retrieval-metrics checks BEIR retrieval metrics against dataset-specific MANTA_* thresholds.")
	fmt.Println("run loads an artifact, binds stub weights and inputs, and executes one entrypoint.")
	fmt.Println("demo creates a tiny inference-style module and loads it through the runtime.")
}

func totalKernelOps(kernels []mantaartifact.Kernel) int {
	total := 0
	for _, kernel := range kernels {
		total += len(kernel.Body)
	}
	return total
}

func defaultArtifactPath(srcPath string) string {
	ext := filepath.Ext(srcPath)
	if ext == "" {
		return srcPath + ".mll"
	}
	return strings.TrimSuffix(srcPath, ext) + ".mll"
}

func stubLoadOptions(mod *mantaartifact.Module) []mantaruntime.LoadOption {
	sizes := defaultSymbolSizes(mod)
	opts := make([]mantaruntime.LoadOption, 0, len(mod.Params))
	for _, param := range mod.Params {
		opts = append(opts, mantaruntime.WithWeight(param.Name, stubTensorForParam(param.Name, param.Type, sizes)))
	}
	return opts
}

func defaultEntryName(mod *mantaartifact.Module) string {
	if mod != nil && len(mod.EntryPoints) > 0 {
		return mod.EntryPoints[0].Name
	}
	return ""
}

func entryPointByName(mod *mantaartifact.Module, name string) (mantaartifact.EntryPoint, error) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return mantaartifact.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func stubInputs(entry mantaartifact.EntryPoint) map[string]any {
	sizes := defaultShapeSizes(entry)
	out := make(map[string]any, len(entry.Inputs))
	for _, input := range entry.Inputs {
		if input.Type.Kind == mantaartifact.ValueKVCache {
			out[input.Name] = backend.NewKVCache(backend.NewTensorF16([]int{sizes["T"], sizes["D"]}, make([]float32, sizes["T"]*sizes["D"])))
			continue
		}
		out[input.Name] = stubTensorForInput(input.Name, input.Type, sizes)
	}
	return out
}

func displayTrainBackend(kind mantaartifact.BackendKind) string {
	if kind == "" {
		return "host"
	}
	return string(kind)
}

func displayArtifactVersion(version string) string {
	return version
}

func displayManifestName(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func joinBackendKinds(kinds []mantaartifact.BackendKind) string {
	if len(kinds) == 0 {
		return ""
	}
	out := make([]string, 0, len(kinds))
	for _, kind := range kinds {
		out = append(out, string(kind))
	}
	return strings.Join(out, ", ")
}

func sortedValueKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func outputSummaries(outputs map[string]backend.Value) []string {
	keys := sortedValueKeys(outputs)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		value := outputs[key]
		tensor, ok := value.Data.(*backend.Tensor)
		if ok && tensor != nil {
			out = append(out, fmt.Sprintf("%s=%s%v", key, tensor.DType, tensor.Shape))
			continue
		}
		if pack, ok := value.Data.(*backend.CandidatePack); ok && pack != nil {
			out = append(out, fmt.Sprintf("%s=candidate_pack(ids=i64%v,scores=f32%v,docs=%s%v)", key, pack.IDs.Shape, pack.Scores.Shape, pack.Docs.DType, pack.Docs.Shape))
			continue
		}
		out = append(out, key+"=<non-tensor>")
	}
	return out
}

type dimFlag struct {
	pairs []string
}

func (f *dimFlag) String() string {
	return strings.Join(f.pairs, ",")
}

func (f *dimFlag) Set(value string) error {
	if value == "" {
		return fmt.Errorf("empty dimension binding")
	}
	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("invalid dimension binding %q, want NAME=VALUE", value)
	}
	if _, err := strconv.Atoi(parts[1]); err != nil {
		return fmt.Errorf("invalid dimension binding %q: %w", value, err)
	}
	f.pairs = append(f.pairs, value)
	return nil
}

func (f *dimFlag) values() map[string]int {
	if len(f.pairs) == 0 {
		return nil
	}
	out := make(map[string]int, len(f.pairs))
	for _, pair := range f.pairs {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			continue
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		out[parts[0]] = n
	}
	return out
}

func defaultSymbolSizes(mod *mantaartifact.Module) map[string]int {
	sizes := map[string]int{
		"V":  3,
		"D":  2,
		"E":  2,
		"T":  2,
		"N":  2,
		"K":  2,
		"H":  2,
		"HD": 2,
	}
	if mod == nil {
		return sizes
	}
	for _, param := range mod.Params {
		if param.Type.Tensor == nil {
			continue
		}
		for _, dim := range param.Type.Tensor.Shape {
			if _, ok := sizes[dim]; !ok {
				sizes[dim] = 2
			}
		}
	}
	for _, entry := range mod.EntryPoints {
		for _, input := range entry.Inputs {
			if input.Type.Tensor == nil {
				continue
			}
			for _, dim := range input.Type.Tensor.Shape {
				if _, ok := sizes[dim]; !ok {
					sizes[dim] = 2
				}
			}
		}
	}
	return sizes
}

func defaultShapeSizes(entry mantaartifact.EntryPoint) map[string]int {
	sizes := map[string]int{
		"V":  3,
		"D":  2,
		"E":  2,
		"T":  2,
		"N":  2,
		"K":  2,
		"H":  2,
		"HD": 2,
	}
	for _, input := range entry.Inputs {
		if input.Type.Tensor == nil {
			continue
		}
		for _, dim := range input.Type.Tensor.Shape {
			if _, ok := sizes[dim]; !ok {
				sizes[dim] = 2
			}
		}
	}
	return sizes
}

func stubTensorForParam(name string, typ mantaartifact.ValueType, sizes map[string]int) *backend.Tensor {
	shape := concreteShape(typ, sizes)
	switch name {
	case "token_embedding":
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
			1, 1,
		})
	case "projection", "wq":
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	case "docs":
		return backend.NewTensorQ4(shape, []float32{
			1, 0,
			0, 1,
		})
	default:
		return fillTensor(typ, shape, 1)
	}
}

func stubTensorForInput(name string, typ mantaartifact.ValueType, sizes map[string]int) *backend.Tensor {
	shape := concreteShape(typ, sizes)
	if typ.Tensor != nil && typ.Tensor.DType == "i32" {
		values := make([]int32, product(shape))
		if name == "attention_mask" {
			for i := range values {
				values[i] = 1
			}
			return backend.NewTensorI32(shape, values)
		}
		limit := sizes["V"]
		if limit <= 0 {
			limit = len(values)
		}
		for i := range values {
			values[i] = int32(i % limit)
		}
		return backend.NewTensorI32(shape, values)
	}
	if name == "x" {
		if product(shape) != 4 {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	}
	if name == "query" {
		values := make([]float32, product(shape))
		if len(values) > 0 {
			values[0] = 1
		}
		return backend.NewTensorF16(shape, values)
	}
	if name == "queries" && typ.Tensor != nil && typ.Tensor.DType == "f16" && len(shape) == 2 && shape[0] == 2 && shape[1] == 2 {
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	}
	if name == "docs" && typ.Tensor != nil && typ.Tensor.DType == "q4" && len(shape) == 2 && shape[1] == 2 {
		values := []float32{1, 0, 0, 1}
		if shape[0] >= 3 {
			values = []float32{1, 0, 0, 1, 1, 1}
		}
		if product(shape) > len(values) {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorQ4(shape, values[:product(shape)])
	}
	if name == "docs" && typ.Tensor != nil && typ.Tensor.DType == "q4" && len(shape) == 3 && shape[0] == 2 && shape[2] == 2 {
		values := []float32{
			1, 0,
			0, 1,
			1, 1,
			0, 1,
			1, 0,
			1, 1,
		}
		if product(shape) > len(values) {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorQ4(shape, values[:product(shape)])
	}
	if name == "candidate_ids" && typ.Tensor != nil && typ.Tensor.DType == "i64" {
		values := make([]int64, product(shape))
		for i := range values {
			values[i] = int64(1001 + i*1001)
		}
		return backend.NewTensorI64(shape, values)
	}
	if typ.Tensor != nil && typ.Tensor.DType == "i64" {
		values := make([]int64, product(shape))
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	}
	return fillTensor(typ, shape, 0)
}

func fillTensor(typ mantaartifact.ValueType, shape []int, offset float32) *backend.Tensor {
	n := product(shape)
	switch typ.Tensor.DType {
	case "i32":
		values := make([]int32, n)
		for i := range values {
			values[i] = int32(i)
		}
		return backend.NewTensorI32(shape, values)
	case "i64":
		values := make([]int64, n)
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	case "f16":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorF16(shape, values)
	case "q2", "q4", "q_norm":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		if typ.Tensor.DType == "q2" {
			return backend.NewTensorQ2(shape, values)
		}
		if typ.Tensor.DType == "q_norm" {
			return backend.NewTensorQNorm(shape, values)
		}
		return backend.NewTensorQ4(shape, values)
	case "q8":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorQ8(shape, values)
	default:
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorF32(shape, values)
	}
}

func concreteShape(typ mantaartifact.ValueType, sizes map[string]int) []int {
	if typ.Tensor == nil {
		return []int{1}
	}
	shape := make([]int, len(typ.Tensor.Shape))
	for i, dim := range typ.Tensor.Shape {
		if literal, err := strconv.Atoi(dim); err == nil {
			shape[i] = literal
			continue
		}
		n, ok := sizes[dim]
		if !ok {
			n = 2
		}
		shape[i] = n
	}
	return shape
}

func product(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}
