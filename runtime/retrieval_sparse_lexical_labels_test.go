package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestSparseLexicalLabelsReconstructBM25Score(t *testing.T) {
	corpus := []retrievalTextRecord{
		{ID: "d1", Text: "alpha alpha finance"},
		{ID: "d2", Text: "beta medicine"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	queryTokens := tokenizeBM25Text("alpha finance alpha")
	queryTerms := sparseLexicalQueryTerms(queryTokens, 128, 0)
	for _, doc := range index.Documents {
		docTerms := sparseLexicalDocumentTerms(doc, index, 128, 0)
		got := SparseLexicalDot(queryTerms, docTerms)
		want := scoreBM25Document(queryTokens, doc, index)
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("sparse dot for %s = %.12f, want BM25 %.12f", doc.ID, got, want)
		}
	}
}

func TestExportSparseLexicalLabelsDeterministicAndBounded(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	firstLabels := filepath.Join(dir, "labels.first.jsonl")
	firstManifest := filepath.Join(dir, "manifest.first.json")
	secondLabels := filepath.Join(dir, "labels.second.jsonl")
	secondManifest := filepath.Join(dir, "manifest.second.json")
	cfg := SparseLexicalLabelExportConfig{
		DatasetName:  "tiny",
		Split:        "train",
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    qrelsPath,
		OutputPath:   firstLabels,
		ManifestPath: firstManifest,
		TopTerms:     2,
		HashBins:     8,
		OracleTopK:   100,
	}
	first, err := ExportSparseLexicalLabels(context.Background(), cfg)
	if err != nil {
		t.Fatalf("export first: %v", err)
	}
	cfg.OutputPath = secondLabels
	cfg.ManifestPath = secondManifest
	second, err := ExportSparseLexicalLabels(context.Background(), cfg)
	if err != nil {
		t.Fatalf("export second: %v", err)
	}
	firstData, err := os.ReadFile(firstLabels)
	if err != nil {
		t.Fatalf("read first labels: %v", err)
	}
	secondData, err := os.ReadFile(secondLabels)
	if err != nil {
		t.Fatalf("read second labels: %v", err)
	}
	if string(firstData) != string(secondData) {
		t.Fatalf("label exports are not deterministic\nfirst:\n%s\nsecond:\n%s", firstData, secondData)
	}
	if first.Stats.Documents != 3 || first.Stats.Queries != 2 || first.Stats.DocumentMaxNNZ > 2 || first.Stats.QueryMaxNNZ > 2 {
		t.Fatalf("summary stats = %+v", first.Stats)
	}
	if !first.Oracle.ExactScoreReconstruction || first.Oracle.Queries != 2 {
		t.Fatalf("oracle summary = %+v", first.Oracle)
	}
	if first.Oracle.ReconstructionTerms != "unbounded_internal" {
		t.Fatalf("oracle reconstruction scope = %q", first.Oracle.ReconstructionTerms)
	}
	if second.OutputPath == first.OutputPath || reflect.DeepEqual(first, second) {
		t.Fatalf("summaries should differ only by output/manifest paths")
	}
	records := readSparseLexicalRecords(t, firstLabels)
	if len(records) != 5 {
		t.Fatalf("records len = %d, want 5", len(records))
	}
	for _, record := range records {
		if record.NonZeros > 2 {
			t.Fatalf("record exceeded top terms bound: %+v", record)
		}
		for _, term := range record.Terms {
			if term.HashBin == nil || *term.HashBin >= 8 {
				t.Fatalf("invalid hash bin in record: %+v", record)
			}
		}
	}
}

func TestExportSparseLexicalLabelsRecordsExportTruncation(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	summary, err := ExportSparseLexicalLabels(context.Background(), SparseLexicalLabelExportConfig{
		DatasetName:  "tiny",
		Split:        "train",
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    qrelsPath,
		OutputPath:   filepath.Join(dir, "labels.jsonl"),
		ManifestPath: filepath.Join(dir, "manifest.json"),
		TopTerms:     1,
		OracleTopK:   100,
	})
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if !summary.Oracle.ExactScoreReconstruction {
		t.Fatalf("oracle should reconstruct BM25 exactly from unbounded internal terms: %+v", summary.Oracle)
	}
	if summary.Oracle.ReconstructionTerms != "unbounded_internal" || summary.Oracle.ExportedTermsExact {
		t.Fatalf("oracle export exactness scope = %+v", summary.Oracle)
	}
	if summary.Stats.DocumentTruncated == 0 || summary.Stats.QueryTruncated == 0 || summary.Stats.DocumentOmitted == 0 || summary.Stats.QueryOmitted == 0 {
		t.Fatalf("truncation stats were not recorded: %+v", summary.Stats)
	}
}

func TestExportSparseLexicalLabelsRejectsInvalidConfig(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	base := SparseLexicalLabelExportConfig{
		DatasetName:  "tiny",
		Split:        "train",
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    qrelsPath,
		OutputPath:   filepath.Join(dir, "labels.jsonl"),
		ManifestPath: filepath.Join(dir, "manifest.json"),
		TopTerms:     2,
		OracleTopK:   100,
	}
	tests := []struct {
		name string
		edit func(*SparseLexicalLabelExportConfig)
		want string
	}{
		{
			name: "oracle top k below recall depth",
			edit: func(cfg *SparseLexicalLabelExportConfig) { cfg.OracleTopK = 10 },
			want: "oracle top-k must be at least 100",
		},
		{
			name: "same labels and manifest path",
			edit: func(cfg *SparseLexicalLabelExportConfig) { cfg.ManifestPath = filepath.Join(dir, ".", "labels.jsonl") },
			want: "labels output path and manifest path must differ",
		},
		{
			name: "negative hash bins",
			edit: func(cfg *SparseLexicalLabelExportConfig) { cfg.HashBins = -1 },
			want: "hash bins must be non-negative",
		},
	}
	if math.MaxInt > math.MaxUint32 {
		tests = append(tests, struct {
			name string
			edit func(*SparseLexicalLabelExportConfig)
			want string
		}{
			name: "oversized hash bins",
			edit: func(cfg *SparseLexicalLabelExportConfig) { cfg.HashBins = int(math.MaxUint32) + 1 },
			want: "hash bins must be <=",
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base
			tt.edit(&cfg)
			_, err := ExportSparseLexicalLabels(context.Background(), cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func writeSparseLexicalDataset(t *testing.T) (dir, corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir = t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath = filepath.Join(dir, "corpus.jsonl")
	queriesPath = filepath.Join(dir, "queries.jsonl")
	qrelsPath = filepath.Join(dir, "qrels", "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine alpha"}`+"\n"+
			`{"_id":"d3","text":"gamma finance"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"alpha finance alpha"}`+"\n"+
			`{"_id":"q2","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir, corpusPath, queriesPath, qrelsPath
}

func readSparseLexicalRecords(t *testing.T, path string) []SparseLexicalLabelRecord {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open labels: %v", err)
	}
	defer f.Close()
	var records []SparseLexicalLabelRecord
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var record SparseLexicalLabelRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			t.Fatalf("decode record: %v", err)
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan labels: %v", err)
	}
	return records
}
