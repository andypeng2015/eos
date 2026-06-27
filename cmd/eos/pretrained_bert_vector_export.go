package main

import (
	"context"
	"flag"
	"fmt"
	"path/filepath"

	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/directml"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/eos/runtime/backends/vulkan"
	"m31labs.dev/eos/runtime/backends/webgpu"
)

func runExportPretrainedBERTRetrievalVectors(args []string) error {
	fs := flag.NewFlagSet("export-pretrained-bert-retrieval-vectors", flag.ContinueOnError)
	fs.SetOutput(flag.CommandLine.Output())
	sourceDir := fs.String("source", "", "local Hugging Face BERT-family snapshot directory containing config.json and vocab.txt")
	modulePath := fs.String("module", "", "Eos MLL BERT embedder module artifact produced by import-pretrained-bert --module-out")
	weightsPath := fs.String("weights", "", "Eos MLL role-named BERT weights file produced by import-pretrained-bert --weights-out")
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	split := fs.String("split", "test", "qrels split under <beir-dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path; when present, export keeps qrels-relevant docs/queries under caps")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before WordPiece tokenization")
	docPrefix := fs.String("doc-prefix", "", "prefix prepended to document text before WordPiece tokenization")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit queries for smoke exports")
	maxLength := fs.Int("max-length", 0, "WordPiece max sequence length; default uses config max_position_embeddings")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest; default is <output-dir>/manifest.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos export-pretrained-bert-retrieval-vectors --source <hf-snapshot> --module <bert_embed.mll> --weights <weights.mll> [flags] <beir-dataset-dir> <output-dir>")
	}
	if *sourceDir == "" || *modulePath == "" || *weightsPath == "" {
		return fmt.Errorf("--source, --module, and --weights are required")
	}
	datasetDir := fs.Arg(0)
	outputDir := fs.Arg(1)
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	_, _, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	summary, err := eosruntime.ExportPretrainedBERTRetrievalVectors(context.Background(), eosruntime.PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:      *datasetName,
		DatasetDir:       datasetDir,
		QrelsPath:        *qrelsPath,
		OutputDir:        outputDir,
		SourceDir:        *sourceDir,
		ModulePath:       *modulePath,
		WeightsPath:      *weightsPath,
		QueryPrefix:      *queryPrefix,
		DocumentPrefix:   *docPrefix,
		BatchSize:        *batchSize,
		MaxDocs:          *maxDocs,
		MaxQueries:       *maxQueries,
		MaxLength:        *maxLength,
		Split:            *split,
		Runtime:          rt,
		ManifestJSONPath: *manifestPath,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported pretrained BERT retrieval vectors: dataset=%s mode=%s docs=%d queries=%d dim=%d\n", summary.Dataset, summary.ExecutionMode, summary.Documents, summary.Queries, summary.OutputDim)
	fmt.Printf("doc_vectors: %s\n", summary.DocVectorPath)
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	} else {
		fmt.Printf("manifest: %s\n", filepath.Join(outputDir, "manifest.json"))
	}
	return nil
}
