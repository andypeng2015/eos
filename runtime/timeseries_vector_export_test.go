package eosruntime

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestTimeSeriesVectorExportWritesWindowCachesAndManifest(t *testing.T) {
	model := loadTinyRetrievalExportModel(t)
	dir := t.TempDir()
	seriesPath := filepath.Join(dir, "series.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	outputDir := filepath.Join(dir, "vectors")
	manifestPath := filepath.Join(dir, "timeseries.manifest.json")
	if err := os.WriteFile(seriesPath, []byte(
		`{"id":"s1","label":"temperature","values":[1,2,3,4,5]}`+"\n"+
			`{"_id":"s2","text":"short sensor","values":[10,12]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write series: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"id":"q1","text":"rising temperature"}`+"\n"+
			`{"_id":"q2","text":"short sensor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}

	summary, err := ExportTimeSeriesWindowVectors(context.Background(), model, TimeSeriesVectorExportConfig{
		DatasetName:      "tiny-series",
		ArtifactPath:     "tiny.mll",
		SeriesPath:       seriesPath,
		QueriesPath:      queriesPath,
		OutputDir:        outputDir,
		BatchSize:        1,
		OutputDim:        1,
		WindowSize:       3,
		WindowStride:     2,
		SeriesPrefix:     "series window: ",
		QueryPrefix:      "query: ",
		ManifestJSONPath: manifestPath,
	})
	if err != nil {
		t.Fatalf("export time-series vectors: %v", err)
	}
	if summary.Schema != TimeSeriesVectorExportManifestSchema || summary.Dataset != "tiny-series" || summary.Series != 2 || summary.Queries != 2 || summary.ChildVectors != 3 || summary.Dimension != 1 || summary.ModelDimension != 2 || summary.OutputDimension != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if summary.WindowSize != 3 || summary.WindowStride != 2 ||
		summary.CorpusPath != filepath.Join(outputDir, "corpus.jsonl") ||
		summary.BEIRQueriesPath != filepath.Join(outputDir, "queries.jsonl") ||
		summary.ChildDocVectorPath != filepath.Join(outputDir, "child-doc-vectors.jsonl") ||
		summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary paths/windows = %+v", summary)
	}

	corpusRows := readJSONLRows(t, summary.CorpusPath)
	if len(corpusRows) != 2 || corpusRows[0]["_id"] != "s1" || corpusRows[1]["_id"] != "s2" {
		t.Fatalf("corpus rows = %+v", corpusRows)
	}
	corpusText := corpusRows[0]["text"].(string)
	for _, want := range []string{"label: temperature", "series_id: s1", "points: count=5", "stats: min=1 max=5 mean=3 first=1 last=5 delta=4"} {
		if !strings.Contains(corpusText, want) {
			t.Fatalf("corpus text missing %q:\n%s", want, corpusText)
		}
	}
	beirQueryRows := readJSONLRows(t, summary.BEIRQueriesPath)
	if len(beirQueryRows) != 2 || beirQueryRows[0]["_id"] != "q1" || beirQueryRows[0]["text"] != "rising temperature" || beirQueryRows[1]["_id"] != "q2" {
		t.Fatalf("BEIR query rows = %+v", beirQueryRows)
	}

	childRows := readJSONLRows(t, summary.ChildDocVectorPath)
	if len(childRows) != 3 {
		t.Fatalf("child row count = %d, want 3", len(childRows))
	}
	wantChildIDs := []string{"s1#window-0000", "s1#window-0001", "s2#window-0000"}
	for i, row := range childRows {
		if row["child_id"] != wantChildIDs[i] {
			t.Fatalf("child row %d id = %v, want %s", i, row["child_id"], wantChildIDs[i])
		}
		embedding := row["embedding"].([]any)
		if len(embedding) != 1 {
			t.Fatalf("child row %d embedding dim = %d, want 1", i, len(embedding))
		}
	}
	if childRows[0]["parent_id"] != "s1" || childRows[2]["parent_id"] != "s2" {
		t.Fatalf("parent ids = %v / %v", childRows[0]["parent_id"], childRows[2]["parent_id"])
	}
	queryRows := readJSONLRows(t, summary.QueryVectorPath)
	if len(queryRows) != 2 || queryRows[0]["id"] != "q1" || queryRows[1]["id"] != "q2" {
		t.Fatalf("query rows = %+v", queryRows)
	}

	var manifest TimeSeriesVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.ChildVectors != summary.ChildVectors || manifest.CorpusPath != summary.CorpusPath || manifest.BEIRQueriesPath != summary.BEIRQueriesPath || manifest.SeriesPath != seriesPath || manifest.QueriesPath != queriesPath || manifest.OutputDir != outputDir {
		t.Fatalf("manifest = %+v, summary = %+v", manifest, summary)
	}
}

func TestTimeSeriesVectorExportOutputRunsMultiVectorEval(t *testing.T) {
	model := loadTinyRetrievalExportModel(t)
	dir := t.TempDir()
	seriesPath := filepath.Join(dir, "series.jsonl")
	queriesPath := filepath.Join(dir, "source-queries.jsonl")
	outputDir := filepath.Join(dir, "vectors")
	qrelsPath := filepath.Join(dir, "qrels.tsv")
	if err := os.WriteFile(seriesPath, []byte(
		`{"id":"s1","text":"alpha series","values":[1,2,3]}`+"\n"+
			`{"id":"s2","text":"beta series","values":[9,8,7]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write series: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\ts1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	summary, err := ExportTimeSeriesWindowVectors(context.Background(), model, TimeSeriesVectorExportConfig{
		DatasetName:  "tiny-series-eval",
		ArtifactPath: "tiny.mll",
		SeriesPath:   seriesPath,
		QueriesPath:  queriesPath,
		OutputDir:    outputDir,
		BatchSize:    1,
		OutputDim:    2,
		WindowSize:   3,
		QueryPrefix:  "query: ",
	})
	if err != nil {
		t.Fatalf("export time-series vectors: %v", err)
	}
	metrics, err := EvaluateTurboQuantMultiVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny-series-eval",
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
		t.Fatalf("eval exported time-series vectors: %v", err)
	}
	if metrics.Inputs.Parents != 2 || metrics.Inputs.ChildVectors != 2 || metrics.Inputs.Queries != 1 || metrics.Inputs.RelevantPairs != 1 || len(metrics.Rows) != 1 {
		t.Fatalf("metrics = %+v rows=%+v", metrics.Inputs, metrics.Rows)
	}
}

func TestTimeSeriesWindowRenderingIsStableAndCoversTail(t *testing.T) {
	record := timeSeriesRecord{ID: "s1", Label: "load", Text: "hourly", Values: []float64{1, 2, 3, 4, 5}}
	starts := timeSeriesWindowStarts(len(record.Values), 3, 2)
	if got, want := strings.Trim(strings.Join(intsToStrings(starts), ","), ","), "0,2"; got != want {
		t.Fatalf("starts = %s, want %s", got, want)
	}
	chunks := renderTimeSeriesWindowChunks([]timeSeriesRecord{record}, 3, 2)
	if len(chunks) != 2 || chunks[1].ChildID != "s1#window-0001" {
		t.Fatalf("chunks = %+v", chunks)
	}
	wantText := "text: hourly\nlabel: load\nseries_id: s1\nwindow: start=2 end=5 count=3\nvalues: 3 4 5\nstats: min=3 max=5 mean=4 first=3 last=5 delta=2"
	if chunks[1].Text != wantText {
		t.Fatalf("rendered text:\n%s\nwant:\n%s", chunks[1].Text, wantText)
	}
}

func TestTimeSeriesVectorExportRejectsInvalidInput(t *testing.T) {
	_, err := ExportTimeSeriesWindowVectors(context.Background(), loadTinyRetrievalExportModel(t), TimeSeriesVectorExportConfig{
		SeriesPath:  "series.jsonl",
		QueriesPath: "queries.jsonl",
		OutputDir:   t.TempDir(),
		WindowSize:  0,
	})
	if err == nil || !strings.Contains(err.Error(), "window-size must be positive") {
		t.Fatalf("error = %v", err)
	}
}

func intsToStrings(values []int) []string {
	out := make([]string, len(values))
	for i, value := range values {
		out[i] = strconv.Itoa(value)
	}
	return out
}
