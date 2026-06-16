package eosruntime

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEventTraceVectorExportWritesEventCachesAndManifest(t *testing.T) {
	model := loadTinyRetrievalExportModel(t)
	dir := t.TempDir()
	tracesPath := filepath.Join(dir, "traces.jsonl")
	queriesPath := filepath.Join(dir, "source-queries.jsonl")
	outputDir := filepath.Join(dir, "vectors")
	manifestPath := filepath.Join(dir, "event-trace.manifest.json")
	if err := os.WriteFile(tracesPath, []byte(
		`{"id":"trace-1","label":"incident","text":"checkout incident","events":["gateway latency spike",{"id":"e2","type":"alert","timestamp":"2026-06-01T00:01:00Z","role":"monitor","actor":"prometheus","message":"payment retries rising","summary":"retry storm","metadata":{"severity":"high","service":"payments"}}]}`+"\n"+
			`{"_id":"trace-2","events":[{"type":"auth","time":"2026-06-01T00:02:00Z","text":"login failure burst"}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write traces: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"id":"q1","text":"payment retries rising"}`+"\n"+
			`{"_id":"q2","text":"login failures"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}

	summary, err := ExportEventTraceVectors(context.Background(), model, EventTraceVectorExportConfig{
		DatasetName:      "tiny-traces",
		ArtifactPath:     "tiny.mll",
		TracesPath:       tracesPath,
		QueriesPath:      queriesPath,
		OutputDir:        outputDir,
		BatchSize:        1,
		OutputDim:        1,
		TracePrefix:      "trace event: ",
		QueryPrefix:      "query: ",
		ManifestJSONPath: manifestPath,
	})
	if err != nil {
		t.Fatalf("export event trace vectors: %v", err)
	}
	if summary.Schema != EventTraceVectorExportManifestSchema || summary.Dataset != "tiny-traces" || summary.Traces != 2 || summary.Queries != 2 || summary.ChildVectors != 3 || summary.Dimension != 1 || summary.ModelDimension != 2 || summary.OutputDimension != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if summary.CorpusPath != filepath.Join(outputDir, "corpus.jsonl") ||
		summary.BEIRQueriesPath != filepath.Join(outputDir, "queries.jsonl") ||
		summary.ChildDocVectorPath != filepath.Join(outputDir, "child-doc-vectors.jsonl") ||
		summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary paths = %+v", summary)
	}

	corpusRows := readJSONLRows(t, summary.CorpusPath)
	if len(corpusRows) != 2 || corpusRows[0]["_id"] != "trace-1" || corpusRows[1]["_id"] != "trace-2" {
		t.Fatalf("corpus rows = %+v", corpusRows)
	}
	corpusText := corpusRows[0]["text"].(string)
	for _, want := range []string{"text: checkout incident", "label: incident", "trace_id: trace-1", "events: count=2"} {
		if !strings.Contains(corpusText, want) {
			t.Fatalf("corpus text missing %q:\n%s", want, corpusText)
		}
	}

	childRows := readJSONLRows(t, summary.ChildDocVectorPath)
	if len(childRows) != 3 {
		t.Fatalf("child row count = %d, want 3", len(childRows))
	}
	wantChildIDs := []string{"trace-1#event-0000", "trace-1#event-0001", "trace-2#event-0000"}
	for i, row := range childRows {
		if row["child_id"] != wantChildIDs[i] {
			t.Fatalf("child row %d id = %v, want %s", i, row["child_id"], wantChildIDs[i])
		}
		if got := len(row["embedding"].([]any)); got != 1 {
			t.Fatalf("child row %d embedding dim = %d, want 1", i, got)
		}
	}
	if childRows[0]["parent_id"] != "trace-1" || childRows[2]["parent_id"] != "trace-2" {
		t.Fatalf("parent ids = %v / %v", childRows[0]["parent_id"], childRows[2]["parent_id"])
	}
	queryRows := readJSONLRows(t, summary.QueryVectorPath)
	if len(queryRows) != 2 || queryRows[0]["id"] != "q1" || queryRows[1]["id"] != "q2" {
		t.Fatalf("query rows = %+v", queryRows)
	}

	var manifest EventTraceVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.ChildVectors != summary.ChildVectors || manifest.CorpusPath != summary.CorpusPath || manifest.BEIRQueriesPath != summary.BEIRQueriesPath || manifest.TracesPath != tracesPath || manifest.QueriesPath != queriesPath || manifest.OutputDir != outputDir {
		t.Fatalf("manifest = %+v, summary = %+v", manifest, summary)
	}
}

func TestEventTraceVectorExportOutputRunsMultiVectorEval(t *testing.T) {
	model := loadTinyRetrievalExportModel(t)
	dir := t.TempDir()
	tracesPath := filepath.Join(dir, "traces.jsonl")
	queriesPath := filepath.Join(dir, "source-queries.jsonl")
	outputDir := filepath.Join(dir, "vectors")
	qrelsPath := filepath.Join(dir, "qrels.tsv")
	if err := os.WriteFile(tracesPath, []byte(
		`{"id":"trace-1","events":["alpha incident",{"message":"payment alpha"}]}`+"\n"+
			`{"id":"trace-2","events":["beta incident"]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write traces: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"id":"q1","text":"alpha payment"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\ttrace-1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	summary, err := ExportEventTraceVectors(context.Background(), model, EventTraceVectorExportConfig{
		DatasetName:  "tiny-event-eval",
		ArtifactPath: "tiny.mll",
		TracesPath:   tracesPath,
		QueriesPath:  queriesPath,
		OutputDir:    outputDir,
		BatchSize:    1,
		OutputDim:    2,
		QueryPrefix:  "query: ",
	})
	if err != nil {
		t.Fatalf("export event trace vectors: %v", err)
	}
	metrics, err := EvaluateTurboQuantMultiVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny-event-eval",
		ArtifactPath:    "tiny.mll",
		CorpusPath:      summary.CorpusPath,
		QueriesPath:     summary.BEIRQueriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   summary.ChildDocVectorPath,
		QueryVectorPath: summary.QueryVectorPath,
		BackendName:     "tiny",
		TopK:            10,
	}, []int{8})
	if err != nil {
		t.Fatalf("eval exported event trace vectors: %v", err)
	}
	if metrics.Inputs.Parents != 2 || metrics.Inputs.ChildVectors != 3 || metrics.Inputs.Queries != 1 || metrics.Inputs.RelevantPairs != 1 || len(metrics.Rows) != 1 {
		t.Fatalf("metrics = %+v rows=%+v", metrics.Inputs, metrics.Rows)
	}
}

func TestEventTraceRenderingIsStableAndSortsMetadata(t *testing.T) {
	trace := eventTraceRecord{
		ID:    "trace-1",
		Label: "incident",
		Text:  "checkout incident",
		Events: []eventTraceEvent{{
			ID:       "e1",
			Type:     "alert",
			Time:     "2026-06-01T00:01:00Z",
			Role:     "monitor",
			Actor:    "prometheus",
			Text:     "gateway latency spike",
			Message:  "payment retries rising",
			Summary:  "retry storm",
			Metadata: map[string]string{"service": "payments", "severity": "high"},
		}},
	}
	chunks := renderEventTraceChunks([]eventTraceRecord{trace})
	if len(chunks) != 1 || chunks[0].ChildID != "trace-1#event-0000" {
		t.Fatalf("chunks = %+v", chunks)
	}
	wantText := "trace_text: checkout incident\ntrace_label: incident\ntrace_id: trace-1\nevent_index: 0\nevent_id: e1\nevent_type: alert\nevent_time: 2026-06-01T00:01:00Z\nrole: monitor\nactor: prometheus\ntext: gateway latency spike\nmessage: payment retries rising\nsummary: retry storm\nmetadata: service=payments severity=high"
	if chunks[0].Text != wantText {
		t.Fatalf("rendered text:\n%s\nwant:\n%s", chunks[0].Text, wantText)
	}
}

func TestEventTraceVectorExportRejectsInvalidInput(t *testing.T) {
	_, err := ExportEventTraceVectors(context.Background(), loadTinyRetrievalExportModel(t), EventTraceVectorExportConfig{
		TracesPath:  "traces.jsonl",
		QueriesPath: "queries.jsonl",
		OutputDir:   t.TempDir(),
		OutputDim:   -1,
	})
	if err == nil || !strings.Contains(err.Error(), "output-dim must be non-negative") {
		t.Fatalf("error = %v", err)
	}
}

func TestEventTraceVectorExportRejectsEmptyObjectEvent(t *testing.T) {
	dir := t.TempDir()
	tracesPath := filepath.Join(dir, "traces.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(tracesPath, []byte(`{"id":"trace-empty","events":[{"id":" ","type":"","text":" ","metadata":{" ":"ignored","empty":" "}}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write traces: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"id":"q1","text":"empty event"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}

	_, err := ExportEventTraceVectors(context.Background(), loadTinyRetrievalExportModel(t), EventTraceVectorExportConfig{
		TracesPath:  tracesPath,
		QueriesPath: queriesPath,
		OutputDir:   filepath.Join(dir, "vectors"),
		BatchSize:   1,
	})
	if err == nil || !strings.Contains(err.Error(), "object event has no supported non-empty fields") {
		t.Fatalf("error = %v", err)
	}
}
