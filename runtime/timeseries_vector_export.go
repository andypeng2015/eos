package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const TimeSeriesVectorExportManifestSchema = "manta.embedding_timeseries_window_vector_export.v1"

// TimeSeriesVectorExportConfig describes a bridge export from numeric series
// windows to text-rendered child-vector caches.
type TimeSeriesVectorExportConfig struct {
	DatasetName      string
	ArtifactPath     string
	SeriesPath       string
	QueriesPath      string
	OutputDir        string
	BatchSize        int
	MaxSeries        int
	MaxQueries       int
	OutputDim        int
	WindowSize       int
	WindowStride     int
	SeriesPrefix     string
	QueryPrefix      string
	ManifestJSONPath string
}

// TimeSeriesVectorExportSummary is the JSON manifest for generated time-series
// window vector caches.
type TimeSeriesVectorExportSummary struct {
	Schema             string    `json:"schema"`
	Dataset            string    `json:"dataset"`
	Artifact           string    `json:"artifact,omitempty"`
	Backend            string    `json:"backend,omitempty"`
	Series             int       `json:"series_count"`
	Queries            int       `json:"query_count"`
	ChildVectors       int       `json:"child_window_vectors"`
	Dimension          int       `json:"dimension"`
	ModelDimension     int       `json:"model_dimension,omitempty"`
	OutputDimension    int       `json:"output_dimension,omitempty"`
	WindowSize         int       `json:"window_size"`
	WindowStride       int       `json:"window_stride"`
	CorpusPath         string    `json:"corpus_path"`
	BEIRQueriesPath    string    `json:"beir_queries_path"`
	ChildDocVectorPath string    `json:"child_doc_vector_path"`
	QueryVectorPath    string    `json:"query_vector_path"`
	BatchSize          int       `json:"batch_size"`
	MaxSeries          int       `json:"max_series,omitempty"`
	MaxQueries         int       `json:"max_queries,omitempty"`
	SeriesPath         string    `json:"series_path,omitempty"`
	QueriesPath        string    `json:"queries_path,omitempty"`
	OutputDir          string    `json:"output_dir,omitempty"`
	ElapsedSeconds     float64   `json:"elapsed_seconds"`
	CreatedAt          time.Time `json:"created_at"`
}

type timeSeriesRecord struct {
	ID     string
	Label  string
	Text   string
	Values []float64
}

// ExportTimeSeriesWindowVectors renders numeric series windows as deterministic
// text and exports parent-child vector caches consumable by the existing
// multivector TurboQuant evaluator.
func ExportTimeSeriesWindowVectors(ctx context.Context, model *EmbeddingModel, cfg TimeSeriesVectorExportConfig) (TimeSeriesVectorExportSummary, error) {
	if model == nil {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeTimeSeriesVectorExportConfig(cfg)
	if cfg.SeriesPath == "" || cfg.QueriesPath == "" || cfg.OutputDir == "" {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("series path, queries path, and output dir are required")
	}
	if err := validateTimeSeriesVectorExportConfig(cfg); err != nil {
		return TimeSeriesVectorExportSummary{}, err
	}

	start := time.Now()
	series, err := readTimeSeriesExportRecords(cfg.SeriesPath, cfg.MaxSeries)
	if err != nil {
		return TimeSeriesVectorExportSummary{}, err
	}
	queries, err := readTimeSeriesExportQueries(cfg.QueriesPath, cfg.MaxQueries)
	if err != nil {
		return TimeSeriesVectorExportSummary{}, err
	}
	if len(series) == 0 {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("series file is empty")
	}
	if len(queries) == 0 {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("queries are empty")
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return TimeSeriesVectorExportSummary{}, err
	}

	windows := renderTimeSeriesWindowChunks(series, cfg.WindowSize, cfg.WindowStride)
	if len(windows) == 0 {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("time-series windowing selected no windows")
	}
	childPath := filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
	queryPath := filepath.Join(cfg.OutputDir, "query-vectors.jsonl")
	corpusPath := filepath.Join(cfg.OutputDir, "corpus.jsonl")
	beirQueriesPath := filepath.Join(cfg.OutputDir, "queries.jsonl")
	if err := writeTimeSeriesBEIRCorpus(series, corpusPath); err != nil {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("write BEIR corpus helper: %w", err)
	}
	if err := writeTimeSeriesBEIRQueries(queries, beirQueriesPath); err != nil {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("write BEIR queries helper: %w", err)
	}
	dim, modelDim, err := writeRetrievalChildVectorCache(ctx, model, windows, childPath, cfg.BatchSize, cfg.SeriesPrefix, cfg.OutputDim)
	if err != nil {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("write child window vectors: %w", err)
	}
	queryDim, queryModelDim, err := writeRetrievalVectorCache(ctx, model, queries, queryPath, cfg.BatchSize, cfg.QueryPrefix, cfg.OutputDim)
	if err != nil {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if dim != queryDim {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("window vectors have dimension %d but query vectors have dimension %d", dim, queryDim)
	}
	if modelDim != queryModelDim {
		return TimeSeriesVectorExportSummary{}, fmt.Errorf("window vectors have encoded dimension %d but query vectors have encoded dimension %d", modelDim, queryModelDim)
	}

	summary := TimeSeriesVectorExportSummary{
		Schema:             TimeSeriesVectorExportManifestSchema,
		Dataset:            cfg.DatasetName,
		Artifact:           cfg.ArtifactPath,
		Backend:            string(model.Backend()),
		Series:             len(series),
		Queries:            len(queries),
		ChildVectors:       len(windows),
		Dimension:          dim,
		ModelDimension:     modelDim,
		OutputDimension:    dim,
		WindowSize:         cfg.WindowSize,
		WindowStride:       cfg.WindowStride,
		CorpusPath:         corpusPath,
		BEIRQueriesPath:    beirQueriesPath,
		ChildDocVectorPath: childPath,
		QueryVectorPath:    queryPath,
		BatchSize:          cfg.BatchSize,
		MaxSeries:          cfg.MaxSeries,
		MaxQueries:         cfg.MaxQueries,
		SeriesPath:         cfg.SeriesPath,
		QueriesPath:        cfg.QueriesPath,
		OutputDir:          cfg.OutputDir,
		ElapsedSeconds:     time.Since(start).Seconds(),
		CreatedAt:          time.Now().UTC(),
	}
	if cfg.ManifestJSONPath != "" {
		if err := WriteTimeSeriesVectorExportSummaryFile(cfg.ManifestJSONPath, summary); err != nil {
			return TimeSeriesVectorExportSummary{}, err
		}
	}
	return summary, nil
}

// WriteTimeSeriesVectorExportSummaryFile writes a JSON manifest for a time-series export run.
func WriteTimeSeriesVectorExportSummaryFile(path string, summary TimeSeriesVectorExportSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func normalizeTimeSeriesVectorExportConfig(cfg TimeSeriesVectorExportConfig) TimeSeriesVectorExportConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "timeseries"
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.WindowStride == 0 {
		cfg.WindowStride = cfg.WindowSize
	}
	return cfg
}

func validateTimeSeriesVectorExportConfig(cfg TimeSeriesVectorExportConfig) error {
	if cfg.BatchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	if cfg.MaxSeries < 0 || cfg.MaxQueries < 0 {
		return fmt.Errorf("max-series and max-queries must be non-negative")
	}
	if cfg.OutputDim < 0 {
		return fmt.Errorf("output-dim must be non-negative")
	}
	if cfg.WindowSize <= 0 {
		return fmt.Errorf("window-size must be positive")
	}
	if cfg.WindowStride <= 0 {
		return fmt.Errorf("window-stride must be positive or zero to use window-size")
	}
	return nil
}

func readTimeSeriesExportRecords(path string, limit int) ([]timeSeriesRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	out := []timeSeriesRecord{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		if limit > 0 && len(out) >= limit {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record struct {
			ID     string    `json:"id"`
			BEIRID string    `json:"_id"`
			Label  string    `json:"label"`
			Text   string    `json:"text"`
			Values []float64 `json:"values"`
		}
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		id := firstNonEmptyString(record.ID, record.BEIRID)
		if id == "" {
			return nil, fmt.Errorf("%s:%d: series id is required", path, lineNo)
		}
		if len(record.Values) == 0 {
			return nil, fmt.Errorf("%s:%d: values array is required for series %q", path, lineNo, id)
		}
		for i, value := range record.Values {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("%s:%d: values[%d] for series %q is not finite", path, lineNo, i, id)
			}
		}
		out = append(out, timeSeriesRecord{
			ID:     id,
			Label:  strings.TrimSpace(record.Label),
			Text:   strings.TrimSpace(record.Text),
			Values: record.Values,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func readTimeSeriesExportQueries(path string, limit int) ([]retrievalTextRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	out := []retrievalTextRecord{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		if limit > 0 && len(out) >= limit {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record struct {
			ID     string `json:"id"`
			BEIRID string `json:"_id"`
			Text   string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		id := firstNonEmptyString(record.ID, record.BEIRID)
		text := strings.TrimSpace(record.Text)
		if id == "" || text == "" {
			continue
		}
		out = append(out, retrievalTextRecord{ID: id, Text: text})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func writeTimeSeriesBEIRCorpus(series []timeSeriesRecord, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	for _, record := range series {
		row := beirJSONRecord{
			ID:   record.ID,
			Text: renderTimeSeriesCorpusText(record),
		}
		data, err := json.Marshal(row)
		if err != nil {
			return err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return err
		}
	}
	return writer.Flush()
}

func writeTimeSeriesBEIRQueries(queries []retrievalTextRecord, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	for _, query := range queries {
		row := beirJSONRecord{
			ID:   query.ID,
			Text: query.Text,
		}
		data, err := json.Marshal(row)
		if err != nil {
			return err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return err
		}
	}
	return writer.Flush()
}

func renderTimeSeriesWindowChunks(series []timeSeriesRecord, windowSize, stride int) []retrievalDocumentChunk {
	out := []retrievalDocumentChunk{}
	for _, record := range series {
		starts := timeSeriesWindowStarts(len(record.Values), windowSize, stride)
		for i, start := range starts {
			end := start + windowSize
			if end > len(record.Values) {
				end = len(record.Values)
			}
			out = append(out, retrievalDocumentChunk{
				ParentID: record.ID,
				ChildID:  fmt.Sprintf("%s#window-%04d", record.ID, i),
				Text:     renderTimeSeriesWindowText(record, start, end),
			})
		}
	}
	return out
}

func timeSeriesWindowStarts(points, windowSize, stride int) []int {
	if points <= 0 {
		return nil
	}
	if points <= windowSize {
		return []int{0}
	}
	starts := []int{}
	for start := 0; start+windowSize <= points; start += stride {
		starts = append(starts, start)
		if start+windowSize == points {
			return starts
		}
	}
	tailStart := points - windowSize
	if len(starts) == 0 || starts[len(starts)-1] != tailStart {
		starts = append(starts, tailStart)
	}
	return starts
}

func renderTimeSeriesCorpusText(record timeSeriesRecord) string {
	minValue, maxValue, sum := record.Values[0], record.Values[0], 0.0
	for _, value := range record.Values {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
		sum += value
	}
	first := record.Values[0]
	last := record.Values[len(record.Values)-1]
	parts := []string{
		"series_id: " + record.ID,
		fmt.Sprintf("points: count=%d", len(record.Values)),
		fmt.Sprintf("stats: min=%s max=%s mean=%s first=%s last=%s delta=%s",
			formatTimeSeriesFloat(minValue),
			formatTimeSeriesFloat(maxValue),
			formatTimeSeriesFloat(sum/float64(len(record.Values))),
			formatTimeSeriesFloat(first),
			formatTimeSeriesFloat(last),
			formatTimeSeriesFloat(last-first),
		),
	}
	if record.Label != "" {
		parts = append([]string{"label: " + record.Label}, parts...)
	}
	if record.Text != "" {
		parts = append([]string{"text: " + record.Text}, parts...)
	}
	return strings.Join(parts, "\n")
}

func renderTimeSeriesWindowText(record timeSeriesRecord, start, end int) string {
	values := record.Values[start:end]
	minValue, maxValue, sum := values[0], values[0], 0.0
	for _, value := range values {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
		sum += value
	}
	first := values[0]
	last := values[len(values)-1]
	parts := []string{
		"series_id: " + record.ID,
		fmt.Sprintf("window: start=%d end=%d count=%d", start, end, len(values)),
		"values: " + formatTimeSeriesValues(values),
		fmt.Sprintf("stats: min=%s max=%s mean=%s first=%s last=%s delta=%s",
			formatTimeSeriesFloat(minValue),
			formatTimeSeriesFloat(maxValue),
			formatTimeSeriesFloat(sum/float64(len(values))),
			formatTimeSeriesFloat(first),
			formatTimeSeriesFloat(last),
			formatTimeSeriesFloat(last-first),
		),
	}
	if record.Label != "" {
		parts = append([]string{"label: " + record.Label}, parts...)
	}
	if record.Text != "" {
		parts = append([]string{"text: " + record.Text}, parts...)
	}
	return strings.Join(parts, "\n")
}

func formatTimeSeriesValues(values []float64) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = formatTimeSeriesFloat(value)
	}
	return strings.Join(parts, " ")
}

func formatTimeSeriesFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}
