package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	eosruntime "m31labs.dev/eos/runtime"
)

type pretrainedBERTProjectionHeadWeightsJSON struct {
	InputDim       int               `json:"input_dim"`
	OutputDim      int               `json:"output_dim"`
	Weights        []float32         `json:"weights"`
	SourceModel    string            `json:"source_model,omitempty"`
	Seed           int64             `json:"seed,omitempty"`
	Loss           string            `json:"loss,omitempty"`
	DataProvenance string            `json:"data_provenance,omitempty"`
	Initialization string            `json:"initialization,omitempty"`
	Provenance     map[string]string `json:"provenance,omitempty"`
}

func runFitPretrainedBERTProjectionHead(args []string) error {
	fs := flag.NewFlagSet("fit-pretrained-bert-projection-head", flag.ContinueOnError)
	fs.SetOutput(flag.CommandLine.Output())
	weightsJSONPath := fs.String("weights-json", "", "JSON file containing learned projection weights in row-major [input_dim,output_dim] order")
	outPath := fs.String("out", "", "write projection head MLL sidecar")
	inputDim := fs.Int("input-dim", 0, "projection input dimension; defaults to weights JSON input_dim")
	outputDim := fs.Int("output-dim", 0, "projection output dimension; defaults to weights JSON output_dim")
	sourceModel := fs.String("source-model", "", "source model name or embedding-space label")
	seed := fs.Int64("seed", 0, "training seed")
	loss := fs.String("loss", "", "training loss provenance label")
	dataProvenance := fs.String("data-provenance", "", "training data provenance summary")
	initialization := fs.String("initialization", "", "initialization provenance summary")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *weightsJSONPath == "" || *outPath == "" {
		return fmt.Errorf("usage: eos fit-pretrained-bert-projection-head --weights-json weights.json --out head.mll [metadata flags]")
	}
	data, err := os.ReadFile(*weightsJSONPath)
	if err != nil {
		return err
	}
	var payload pretrainedBERTProjectionHeadWeightsJSON
	if err := json.Unmarshal(data, &payload); err != nil {
		return fmt.Errorf("parse weights json: %w", err)
	}
	if *inputDim > 0 {
		payload.InputDim = *inputDim
	}
	if *outputDim > 0 {
		payload.OutputDim = *outputDim
	}
	if *sourceModel != "" {
		payload.SourceModel = *sourceModel
	}
	if *seed != 0 {
		payload.Seed = *seed
	}
	if flagWasProvided(fs, "loss") {
		payload.Loss = *loss
	}
	if *dataProvenance != "" {
		payload.DataProvenance = *dataProvenance
	}
	if *initialization != "" {
		payload.Initialization = *initialization
	}
	head, err := eosruntime.NewPretrainedBERTProjectionHead(payload.InputDim, payload.OutputDim, payload.Weights)
	if err != nil {
		return err
	}
	head.SourceModel = payload.SourceModel
	head.Seed = payload.Seed
	head.Loss = payload.Loss
	head.DataProvenance = payload.DataProvenance
	head.Initialization = payload.Initialization
	head.Provenance = payload.Provenance
	if err := eosruntime.WritePretrainedBERTProjectionHeadFile(*outPath, head); err != nil {
		return err
	}
	loaded, err := eosruntime.ReadPretrainedBERTProjectionHeadFile(*outPath)
	if err != nil {
		return err
	}
	fmt.Printf("projection head: input_dim=%d output_dim=%d identity=%s weights_sha256=%s out=%s\n",
		loaded.InputDim, loaded.OutputDim, loaded.IdentityHash(), loaded.WeightsSHA256, *outPath)
	return nil
}
