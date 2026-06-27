package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	eosruntime "m31labs.dev/eos/runtime"
)

func runImportPretrainedBERT(args []string) error {
	fs := flag.NewFlagSet("import-pretrained-bert", flag.ContinueOnError)
	source := fs.String("source", "", "local Hugging Face snapshot directory containing config.json and optionally vocab.txt")
	modelName := fs.String("model-name", "", "model identifier to record in the import plan")
	planJSON := fs.String("plan-json", "", "write the plan JSON to this path instead of stdout")
	tokenizerSmoke := fs.String("tokenizer-smoke", "", "optional text to tokenize with vocab.txt as a local smoke check")
	verifyWeights := fs.Bool("verify-weights", false, "metadata-only verification of single-file or sharded safetensors tensor names, shapes, and dtypes")
	loadWeightsSmoke := fs.Bool("load-weights-smoke", false, "load planned BERT safetensors bytes into an intermediate in-memory weight set and report byte-ingest stats")
	decodeWeightsSmoke := fs.Bool("decode-weights-smoke", false, "decode planned BERT safetensors F32/F16/BF16 bytes into float32 values and report decode stats")
	weightsOut := fs.String("weights-out", "", "write decoded planned BERT weights to an Eos MLL weights-only file; this is not an executable BERT artifact")
	moduleOut := fs.String("module-out", "", "write the host-reference executable BERT embedder module to an Eos MLL artifact")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *source == "" {
		return fmt.Errorf("usage: eos import-pretrained-bert --source <hf-snapshot-dir> [--model-name name] [--plan-json plan.json] [--tokenizer-smoke text] [--verify-weights] [--load-weights-smoke] [--decode-weights-smoke] [--weights-out weights.mll] [--module-out artifact.mll]")
	}
	plan, err := eosruntime.PlanPretrainedBERTImportFromDir(*source, *modelName)
	if err != nil {
		return err
	}
	if *verifyWeights {
		report, err := eosruntime.VerifyPretrainedBERTWeightsFromDir(*source, plan)
		if err != nil {
			return fmt.Errorf("verify weights metadata: %w", err)
		}
		plan.WeightVerification = &report
	}
	if *loadWeightsSmoke {
		_, report, err := eosruntime.LoadPretrainedBERTWeightsFromDir(*source, plan)
		if err != nil {
			return fmt.Errorf("load weights smoke: %w", err)
		}
		plan.WeightLoadSmoke = &report
	}
	if *decodeWeightsSmoke {
		_, report, err := eosruntime.LoadPretrainedBERTDecodedWeightsFromDir(*source, plan)
		if err != nil {
			return fmt.Errorf("decode weights smoke: %w", err)
		}
		plan.WeightDecodeSmoke = &report
	}
	if *weightsOut != "" {
		report, err := eosruntime.ExportPretrainedBERTWeightFileFromDir(*source, plan, *weightsOut)
		if err != nil {
			return fmt.Errorf("export weights-only file: %w", err)
		}
		plan.WeightFileExport = &report
	}
	if *moduleOut != "" {
		report, err := eosruntime.ExportPretrainedBERTEmbedderModuleFromPlan(plan, *moduleOut)
		if err != nil {
			return fmt.Errorf("export embedder module: %w", err)
		}
		plan.ModuleExport = &report
	}
	if *tokenizerSmoke != "" {
		tokenizer, err := eosruntime.LoadHFWordPieceTokenizerFromDir(*source)
		if err != nil {
			return fmt.Errorf("tokenizer smoke: %w", err)
		}
		encoding, err := tokenizer.Encode(*tokenizerSmoke, eosruntime.HFWordPieceEncodeOptions{MaxLength: plan.Config.MaxPositionEmbeddings})
		if err != nil {
			return fmt.Errorf("tokenizer smoke encode: %w", err)
		}
		fmt.Fprintf(os.Stderr, "tokenizer smoke: vocab=%d tokens=%d ids=%v\n", tokenizer.VocabSize(), len(encoding.IDs), encoding.IDs)
	} else if _, err := os.Stat(filepath.Join(*source, "vocab.txt")); err == nil {
		tokenizer, err := eosruntime.LoadHFWordPieceTokenizerFromDir(*source)
		if err != nil {
			return fmt.Errorf("load tokenizer: %w", err)
		}
		fmt.Fprintf(os.Stderr, "tokenizer: vocab=%d\n", tokenizer.VocabSize())
	}
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if *planJSON != "" {
		return os.WriteFile(*planJSON, data, 0o644)
	}
	_, err = os.Stdout.Write(data)
	return err
}
