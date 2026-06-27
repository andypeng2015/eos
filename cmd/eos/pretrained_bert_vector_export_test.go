package main

import (
	"strings"
	"testing"
)

func TestRunExportPretrainedBERTRetrievalVectorsRequiresArtifacts(t *testing.T) {
	err := runExportPretrainedBERTRetrievalVectors([]string{"dataset", "out"})
	if err == nil || !strings.Contains(err.Error(), "--source, --module, and --weights are required") {
		t.Fatalf("err = %v, want required artifact flags", err)
	}
}
