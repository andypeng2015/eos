package main

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosruntime "m31labs.dev/eos/runtime"
)

func TestRunExportPretrainedBERTRetrievalVectorsRequiresArtifacts(t *testing.T) {
	err := runExportPretrainedBERTRetrievalVectors([]string{"dataset", "out"})
	if err == nil || !strings.Contains(err.Error(), "--source, --module, and --weights are required") {
		t.Fatalf("err = %v, want required artifact flags", err)
	}
}

func TestRunExportPretrainedBERTRetrievalVectorsAcceptsResumeProgressFlags(t *testing.T) {
	err := runExportPretrainedBERTRetrievalVectors([]string{"--resume", "--progress-every", "7", "--projection-head", "head.mll", "--use-package-role-contract"})
	if err == nil {
		t.Fatal("runExportPretrainedBERTRetrievalVectors succeeded without positional args, want usage error")
	}
	if strings.Contains(err.Error(), "flag provided but not defined") {
		t.Fatalf("pretrained BERT vector export flags were not registered: %v", err)
	}
	if !strings.Contains(err.Error(), "usage: eos export-pretrained-bert-retrieval-vectors") {
		t.Fatalf("err = %v, want usage error after parsing flags", err)
	}
}

func TestRunFitPretrainedBERTProjectionHeadWritesMLLSidecar(t *testing.T) {
	dir := t.TempDir()
	weightsJSON := filepath.Join(dir, "weights.json")
	outPath := filepath.Join(dir, "head.mll")
	data := []byte(`{"input_dim":2,"output_dim":1,"weights":[1,0],"source_model":"fixture","loss":"json_loss","provenance":{"train_qrels":"train.tsv"}}` + "\n")
	if err := os.WriteFile(weightsJSON, data, 0o644); err != nil {
		t.Fatalf("write weights json: %v", err)
	}
	if err := runFitPretrainedBERTProjectionHead([]string{
		"--weights-json", weightsJSON,
		"--out", outPath,
		"--seed", "17",
		"--data-provenance", "unit train split",
		"--initialization", "svd",
	}); err != nil {
		t.Fatalf("fit projection head: %v", err)
	}
	head, err := eosruntime.ReadPretrainedBERTProjectionHeadFile(outPath)
	if err != nil {
		t.Fatalf("read projection head: %v", err)
	}
	if head.InputDim != 2 || head.OutputDim != 1 || head.SourceModel != "fixture" || head.Seed != 17 || head.Initialization != "svd" {
		t.Fatalf("head = %+v", head)
	}
	if head.Loss != "json_loss" {
		t.Fatalf("loss = %q, want JSON provenance loss", head.Loss)
	}
}

func TestRunFitPretrainedBERTProjectionHeadLossFlagOverridesJSON(t *testing.T) {
	dir := t.TempDir()
	weightsJSON := filepath.Join(dir, "weights.json")
	outPath := filepath.Join(dir, "head.mll")
	data := []byte(`{"input_dim":2,"output_dim":1,"weights":[1,0],"loss":"json_loss"}` + "\n")
	if err := os.WriteFile(weightsJSON, data, 0o644); err != nil {
		t.Fatalf("write weights json: %v", err)
	}
	if err := runFitPretrainedBERTProjectionHead([]string{
		"--weights-json", weightsJSON,
		"--out", outPath,
		"--loss", "flag_loss",
	}); err != nil {
		t.Fatalf("fit projection head: %v", err)
	}
	head, err := eosruntime.ReadPretrainedBERTProjectionHeadFile(outPath)
	if err != nil {
		t.Fatalf("read projection head: %v", err)
	}
	if head.Loss != "flag_loss" {
		t.Fatalf("loss = %q, want flag override", head.Loss)
	}
}

func TestRunExportPretrainedBERTRetrievalVectorsDocumentPrefixAlias(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "canonical",
			args: []string{"--document-prefix", "doc: "},
			want: "doc: ",
		},
		{
			name: "legacy",
			args: []string{"--doc-prefix", "doc: "},
			want: "doc: ",
		},
		{
			name: "both same",
			args: []string{"--document-prefix", "doc: ", "--doc-prefix", "doc: "},
			want: "doc: ",
		},
		{
			name: "absent",
			args: nil,
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fs := flag.NewFlagSet("test", flag.ContinueOnError)
			documentPrefix := fs.String("document-prefix", "", "")
			docPrefix := fs.String("doc-prefix", "", "")
			if err := fs.Parse(tt.args); err != nil {
				t.Fatalf("parse flags: %v", err)
			}
			got, err := resolvePretrainedBERTDocumentPrefix(fs, *documentPrefix, *docPrefix)
			if err != nil {
				t.Fatalf("resolve prefix: %v", err)
			}
			if got != tt.want {
				t.Fatalf("prefix = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRunExportPretrainedBERTRetrievalVectorsRejectsMismatchedDocumentPrefixAlias(t *testing.T) {
	err := runExportPretrainedBERTRetrievalVectors([]string{
		"--document-prefix", "document: ",
		"--doc-prefix", "doc: ",
		"dataset",
		"out",
	})
	if err == nil || !strings.Contains(err.Error(), "--document-prefix and --doc-prefix differ") {
		t.Fatalf("err = %v, want mismatch error", err)
	}
}
