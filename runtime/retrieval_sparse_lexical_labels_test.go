package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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

func TestExportSparseLexicalLabelsIncludesRelevantEmptyDocument(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeSparseLexicalDatasetWithEmptyRelevantDocument(t)
	labelsPath := filepath.Join(dir, "labels.jsonl")
	summary, err := ExportSparseLexicalLabels(context.Background(), SparseLexicalLabelExportConfig{
		DatasetName:  "tiny",
		Split:        "train",
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    qrelsPath,
		OutputPath:   labelsPath,
		ManifestPath: filepath.Join(dir, "manifest.json"),
		TopTerms:     8,
		OracleTopK:   100,
	})
	if err != nil {
		t.Fatalf("export: %v", err)
	}
	if summary.Stats.Documents != 3 || summary.BM25.Documents != 3 {
		t.Fatalf("summary document counts = stats:%d bm25:%d, want 3", summary.Stats.Documents, summary.BM25.Documents)
	}

	records := readSparseLexicalRecords(t, labelsPath)
	var emptyDoc *SparseLexicalLabelRecord
	for i := range records {
		record := &records[i]
		if record.RecordType == "document" && record.ID == "d-empty" {
			emptyDoc = record
			break
		}
	}
	if emptyDoc == nil {
		t.Fatalf("missing sparse lexical label for qrels-relevant empty document")
	}
	if emptyDoc.NonZeros != 0 || len(emptyDoc.Terms) != 0 {
		t.Fatalf("empty document label = %+v, want zero terms", *emptyDoc)
	}

	metrics, err := EvaluateSparseLexicalLabels(context.Background(), SparseLexicalLabelEvalConfig{
		DatasetName: "tiny",
		Split:       "train",
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		LabelsPath:  labelsPath,
		TopK:        100,
	})
	if err != nil {
		t.Fatalf("eval labels: %v", err)
	}
	if metrics.SparseLexical == nil || metrics.SparseLexical.MissingDocLabels != 0 || metrics.SparseLexical.DocumentLabels != 3 {
		t.Fatalf("sparse eval stats = %+v", metrics.SparseLexical)
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

func TestEvaluateSparseLexicalLabelsRanksCappedLabels(t *testing.T) {
	dir, _, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "eval-labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":2,"terms":[{"term":"alpha","weight":2},{"term":"finance","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"alpha","weight":0.1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":1,"terms":[{"term":"gamma","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q2","nonzeros":1,"terms":[{"term":"gamma","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	metrics, err := EvaluateSparseLexicalLabels(context.Background(), SparseLexicalLabelEvalConfig{
		DatasetName: "tiny",
		Split:       "train",
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		LabelsPath:  labelsPath,
		TopK:        100,
	})
	if err != nil {
		t.Fatalf("eval labels: %v", err)
	}
	if metrics.Backend != "sparse_lexical_labels_capped" || metrics.Inputs.LabelPath != labelsPath || metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 2 {
		t.Fatalf("metrics identity/inputs = %+v", metrics)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.RecallAt100 != 1 {
		t.Fatalf("quality = %+v", metrics.Quality)
	}
	if metrics.SparseLexical == nil || metrics.SparseLexical.DocumentLabels != 3 || metrics.SparseLexical.QueryLabels != 2 || metrics.SparseLexical.Representation != "capped_exported_sparse_lexical_labels" {
		t.Fatalf("sparse stats = %+v", metrics.SparseLexical)
	}
}

func TestEvaluateSparseLexicalLabelsRejectsMissingRequiredLabels(t *testing.T) {
	dir, _, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "missing-labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"beta","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	_, err := EvaluateSparseLexicalLabels(context.Background(), SparseLexicalLabelEvalConfig{
		DatasetName: "tiny",
		Split:       "train",
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		LabelsPath:  labelsPath,
		TopK:        100,
	})
	if err == nil || !strings.Contains(err.Error(), "missing required qrels coverage") {
		t.Fatalf("err = %v, want missing required qrels coverage", err)
	}
}

func TestEvaluateSparseLexicalLabelsRejectsInvalidSchema(t *testing.T) {
	dir, _, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "bad-schema.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.other.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":0,"terms":[]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	_, err := EvaluateSparseLexicalLabels(context.Background(), SparseLexicalLabelEvalConfig{
		DatasetName: "tiny",
		Split:       "train",
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		LabelsPath:  labelsPath,
		TopK:        100,
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported schema") {
		t.Fatalf("err = %v, want unsupported schema", err)
	}
}

func TestFitSparseLexicalHashHeadWritesExperimentalArtifact(t *testing.T) {
	dir, _, _, _ := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "labels.jsonl")
	headPath := filepath.Join(dir, "head.json")
	writeHashHeadEvalLabels(t, labelsPath, false, false)
	head, err := FitSparseLexicalHashHead(SparseLexicalHashHeadFitConfig{
		DatasetName: "tiny",
		Split:       "train",
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		HashBins:    1,
	})
	if err != nil {
		t.Fatalf("fit hash head: %v", err)
	}
	if head.Schema != SparseLexicalHashHeadSchema || !head.Experimental || head.Hashing.Bins != 1 {
		t.Fatalf("head identity = %+v", head)
	}
	if head.Stats.DocumentLabels != 3 || head.Stats.QueryLabels != 2 || head.Stats.DocumentMaxHashNNZ != 1 || head.Stats.DocumentMergedBins == 0 {
		t.Fatalf("head stats = %+v", head.Stats)
	}
	loaded, err := ReadSparseLexicalHashHead(headPath)
	if err != nil {
		t.Fatalf("read head: %v", err)
	}
	if loaded.Schema != SparseLexicalHashHeadSchema || loaded.Hashing.Bins != 1 {
		t.Fatalf("loaded head = %+v", loaded)
	}
}

func TestEvaluateSparseLexicalHashHeadRanksHashedLabels(t *testing.T) {
	dir, _, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "labels.jsonl")
	headPath := filepath.Join(dir, "head.json")
	writeHashHeadEvalLabels(t, labelsPath, true, false)
	if _, err := FitSparseLexicalHashHead(SparseLexicalHashHeadFitConfig{
		DatasetName: "tiny",
		Split:       "train",
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		HashBins:    65536,
	}); err != nil {
		t.Fatalf("fit hash head: %v", err)
	}
	metrics, err := EvaluateSparseLexicalHashHead(context.Background(), SparseLexicalHashHeadEvalConfig{
		DatasetName: "tiny",
		Split:       "train",
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		TopK:        100,
	})
	if err != nil {
		t.Fatalf("eval hash head: %v", err)
	}
	if metrics.Backend != "sparse_lexical_hash_head" || metrics.Inputs.LabelPath != labelsPath || metrics.Inputs.HeadPath != headPath || metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 2 {
		t.Fatalf("metrics identity/inputs = %+v", metrics)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.RecallAt100 != 1 {
		t.Fatalf("quality = %+v", metrics.Quality)
	}
	if metrics.SparseLexical == nil || metrics.SparseLexical.HashBins != 65536 || metrics.SparseLexical.DocumentMaxHashNNZ != 2 || metrics.SparseLexical.Representation != "experimental_hashed_sparse_lexical_head" {
		t.Fatalf("sparse stats = %+v", metrics.SparseLexical)
	}
}

func TestSparseLexicalHashHeadRejectsInvalidHeadAndHashBins(t *testing.T) {
	dir, _, queriesPath, qrelsPath := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "labels.jsonl")
	headPath := filepath.Join(dir, "head.json")
	writeHashHeadEvalLabels(t, labelsPath, true, false)
	if _, err := FitSparseLexicalHashHead(SparseLexicalHashHeadFitConfig{
		DatasetName: "tiny",
		Split:       "train",
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		HashBins:    0,
	}); err == nil || !strings.Contains(err.Error(), "hash bins must be positive") {
		t.Fatalf("fit err = %v, want positive hash bins", err)
	}
	if err := os.WriteFile(headPath, []byte(`{"schema":"manta.other.v1","hashing":{"algorithm":"fnv1a32","bins":16}}`+"\n"), 0o644); err != nil {
		t.Fatalf("write bad head: %v", err)
	}
	_, err := EvaluateSparseLexicalHashHead(context.Background(), SparseLexicalHashHeadEvalConfig{
		DatasetName: "tiny",
		Split:       "train",
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		TopK:        100,
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported schema") {
		t.Fatalf("eval err = %v, want unsupported schema", err)
	}
}

func TestSparseLexicalHashHeadRejectsIncompatibleLabelHashBins(t *testing.T) {
	dir, _, _, _ := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "bad-hash-labels.jsonl")
	headPath := filepath.Join(dir, "head.json")
	writeHashHeadEvalLabels(t, labelsPath, true, true)
	_, err := FitSparseLexicalHashHead(SparseLexicalHashHeadFitConfig{
		DatasetName: "tiny",
		Split:       "train",
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		HashBins:    65536,
	})
	if err == nil || !strings.Contains(err.Error(), "incompatible") {
		t.Fatalf("err = %v, want incompatible hash_bin", err)
	}
}

func TestFitSparseLexicalProjectionHeadRejectsInvalidConfig(t *testing.T) {
	dir, _, _, _ := writeSparseLexicalDataset(t)
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if _, err := ExportSparseLexicalLabels(context.Background(), SparseLexicalLabelExportConfig{
		DatasetName: "tiny",
		Split:       "train",
		CorpusPath:  filepath.Join(dir, "corpus.jsonl"),
		QueriesPath: filepath.Join(dir, "queries.jsonl"),
		QrelsPath:   filepath.Join(dir, "qrels", "train.tsv"),
		OutputPath:  labelsPath,
		TopTerms:    2,
		OracleTopK:  100,
	}); err != nil {
		t.Fatalf("export labels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[1,0]}`+"\n"+
			`{"_id":"d2","embedding":[0,1]}`+"\n"+
			`{"_id":"d3","embedding":[1,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"_id":"q1","embedding":[1,0]}`+"\n"+
			`{"_id":"q2","embedding":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	valid := SparseLexicalProjectionHeadFitConfig{
		DatasetName:       "tiny",
		Split:             "train",
		LabelsPath:        labelsPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   queryVectorsPath,
		HeadPath:          filepath.Join(dir, "projection-head.json"),
		HashBins:          16,
		MaxPrototypes:     4,
		MaxPredictedTerms: 2,
	}
	head, err := FitSparseLexicalProjectionHead(valid)
	if err != nil {
		t.Fatalf("fit valid projection head: %v", err)
	}
	if head.Schema != SparseLexicalProjectionHeadSchema || head.Config.Dimension != 2 || head.Config.HashBins != 16 || head.Config.MaxPrototypes != 4 || head.Config.MaxPredictedTerms != 2 || head.Stats.StoredPrototypes == 0 {
		t.Fatalf("projection head = %+v", head)
	}
	if head.Config.PrototypeRank != SparseLexicalProjectionPrototypeRankSupport {
		t.Fatalf("prototype rank = %q, want support", head.Config.PrototypeRank)
	}
	loaded, err := ReadSparseLexicalProjectionHead(valid.HeadPath)
	if err != nil {
		t.Fatalf("read projection head: %v", err)
	}
	if loaded.Schema != SparseLexicalProjectionHeadSchema || loaded.Config.Normalization != "input_l2_and_prototype_l2" || loaded.Config.PrototypeRank != SparseLexicalProjectionPrototypeRankSupport {
		t.Fatalf("loaded projection head = %+v", loaded)
	}

	for _, tt := range []struct {
		name string
		edit func(*SparseLexicalProjectionHeadFitConfig)
		want string
	}{
		{
			name: "bad hash bins",
			edit: func(cfg *SparseLexicalProjectionHeadFitConfig) { cfg.HashBins = 0 },
			want: "hash bins must be positive",
		},
		{
			name: "bad max prototypes",
			edit: func(cfg *SparseLexicalProjectionHeadFitConfig) { cfg.MaxPrototypes = 0 },
			want: "max prototypes must be positive",
		},
		{
			name: "dimension mismatch",
			edit: func(cfg *SparseLexicalProjectionHeadFitConfig) {
				path := filepath.Join(dir, "bad-query-vectors.jsonl")
				if err := os.WriteFile(path, []byte(`{"_id":"q1","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
					t.Fatalf("write bad query vectors: %v", err)
				}
				cfg.QueryVectorPath = path
			},
			want: "dimension",
		},
		{
			name: "bad prototype rank",
			edit: func(cfg *SparseLexicalProjectionHeadFitConfig) { cfg.PrototypeRank = "score_magic" },
			want: "prototype rank must be one of support, total_weight, avg_weight",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := valid
			cfg.HeadPath = filepath.Join(dir, tt.name+".json")
			tt.edit(&cfg)
			_, err := FitSparseLexicalProjectionHead(cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestFitSparseLexicalProjectionHeadPrototypeRankPolicy(t *testing.T) {
	dir := t.TempDir()
	labelsPath := filepath.Join(dir, "labels.jsonl")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"","hash_bin":1,"weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"","hash_bin":1,"weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":1,"terms":[{"term":"","hash_bin":2,"weight":5}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":0,"terms":[]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[1,0]}`+"\n"+
			`{"_id":"d2","embedding":[1,0]}`+"\n"+
			`{"_id":"d3","embedding":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"_id":"q1","embedding":[0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	base := SparseLexicalProjectionHeadFitConfig{
		DatasetName:       "tiny",
		Split:             "train",
		LabelsPath:        labelsPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   queryVectorsPath,
		HashBins:          16,
		MaxPrototypes:     1,
		MaxPredictedTerms: 2,
	}
	for _, tt := range []struct {
		name string
		rank string
		bin  uint32
	}{
		{name: "default support", rank: "", bin: 1},
		{name: "explicit support", rank: SparseLexicalProjectionPrototypeRankSupport, bin: 1},
		{name: "total weight", rank: SparseLexicalProjectionPrototypeRankTotalWeight, bin: 2},
		{name: "avg weight", rank: SparseLexicalProjectionPrototypeRankAvgWeight, bin: 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base
			cfg.PrototypeRank = tt.rank
			cfg.HeadPath = filepath.Join(dir, strings.ReplaceAll(tt.name, " ", "-")+".json")
			head, err := FitSparseLexicalProjectionHead(cfg)
			if err != nil {
				t.Fatalf("fit projection head: %v", err)
			}
			if len(head.Prototypes) != 1 || head.Prototypes[0].Bin != tt.bin {
				t.Fatalf("retained prototypes = %+v, want only bin %d", head.Prototypes, tt.bin)
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

func writeSparseLexicalDatasetWithEmptyRelevantDocument(t *testing.T) (dir, corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir = t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath = filepath.Join(dir, "corpus.jsonl")
	queriesPath = filepath.Join(dir, "queries.jsonl")
	qrelsPath = filepath.Join(dir, "qrels", "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha finance"}`+"\n"+
			`{"_id":"d-empty","title":"","text":""}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td-empty\t1\n"), 0o644); err != nil {
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

func writeHashHeadEvalLabels(t *testing.T, labelsPath string, includeHash, incompatibleHash bool) {
	t.Helper()
	alphaTerm := `{"term":"alpha","weight":2}`
	if includeHash {
		alphaHash := *sparseLexicalHashBin("alpha", 65536)
		if incompatibleHash {
			alphaHash++
		}
		alphaTerm = fmt.Sprintf(`{"term":"alpha","weight":2,"hash_bin":%d}`, alphaHash)
	}
	if err := os.WriteFile(labelsPath, []byte(
		fmt.Sprintf(`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":2,"terms":[%s,{"term":"finance","weight":1}]}`+"\n", alphaTerm)+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"alpha","weight":0.1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":1,"terms":[{"term":"gamma","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q2","nonzeros":1,"terms":[{"term":"gamma","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
}
