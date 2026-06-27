package main

import (
	"flag"
	"strings"
	"testing"
)

func TestRunExportPretrainedBERTRetrievalVectorsRequiresArtifacts(t *testing.T) {
	err := runExportPretrainedBERTRetrievalVectors([]string{"dataset", "out"})
	if err == nil || !strings.Contains(err.Error(), "--source, --module, and --weights are required") {
		t.Fatalf("err = %v, want required artifact flags", err)
	}
}

func TestRunExportPretrainedBERTRetrievalVectorsAcceptsResumeProgressFlags(t *testing.T) {
	err := runExportPretrainedBERTRetrievalVectors([]string{"--resume", "--progress-every", "7"})
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
