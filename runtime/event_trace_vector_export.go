package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const EventTraceVectorExportManifestSchema = "manta.embedding_event_trace_vector_export.v1"

// EventTraceVectorExportConfig describes a bridge export from parent
// trace/session JSONL rows to text-rendered event child-vector caches.
type EventTraceVectorExportConfig struct {
	DatasetName      string
	ArtifactPath     string
	TracesPath       string
	QueriesPath      string
	OutputDir        string
	BatchSize        int
	MaxTraces        int
	MaxQueries       int
	OutputDim        int
	TracePrefix      string
	QueryPrefix      string
	ManifestJSONPath string
}

// EventTraceVectorExportSummary is the JSON manifest for generated event trace
// child-vector caches.
type EventTraceVectorExportSummary struct {
	Schema             string    `json:"schema"`
	Dataset            string    `json:"dataset"`
	Artifact           string    `json:"artifact,omitempty"`
	Backend            string    `json:"backend,omitempty"`
	Traces             int       `json:"trace_count"`
	Queries            int       `json:"query_count"`
	ChildVectors       int       `json:"child_event_vectors"`
	Dimension          int       `json:"dimension"`
	ModelDimension     int       `json:"model_dimension,omitempty"`
	OutputDimension    int       `json:"output_dimension,omitempty"`
	CorpusPath         string    `json:"corpus_path"`
	BEIRQueriesPath    string    `json:"beir_queries_path"`
	ChildDocVectorPath string    `json:"child_doc_vector_path"`
	QueryVectorPath    string    `json:"query_vector_path"`
	BatchSize          int       `json:"batch_size"`
	MaxTraces          int       `json:"max_traces,omitempty"`
	MaxQueries         int       `json:"max_queries,omitempty"`
	TracesPath         string    `json:"traces_path,omitempty"`
	QueriesPath        string    `json:"queries_path,omitempty"`
	OutputDir          string    `json:"output_dir,omitempty"`
	ElapsedSeconds     float64   `json:"elapsed_seconds"`
	CreatedAt          time.Time `json:"created_at"`
}

type eventTraceRecord struct {
	ID     string
	Label  string
	Text   string
	Events []eventTraceEvent
}

type eventTraceEvent struct {
	ID       string
	Type     string
	Time     string
	Role     string
	Actor    string
	Text     string
	Message  string
	Summary  string
	Metadata map[string]string
}

// ExportEventTraceVectors renders trace events as deterministic text and
// exports parent-child vector caches consumable by the existing multivector
// TurboQuant evaluator.
func ExportEventTraceVectors(ctx context.Context, model *EmbeddingModel, cfg EventTraceVectorExportConfig) (EventTraceVectorExportSummary, error) {
	if model == nil {
		return EventTraceVectorExportSummary{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeEventTraceVectorExportConfig(cfg)
	if cfg.TracesPath == "" || cfg.QueriesPath == "" || cfg.OutputDir == "" {
		return EventTraceVectorExportSummary{}, fmt.Errorf("traces path, queries path, and output dir are required")
	}
	if err := validateEventTraceVectorExportConfig(cfg); err != nil {
		return EventTraceVectorExportSummary{}, err
	}

	start := time.Now()
	traces, err := readEventTraceExportRecords(cfg.TracesPath, cfg.MaxTraces)
	if err != nil {
		return EventTraceVectorExportSummary{}, err
	}
	queries, err := readEventTraceExportQueries(cfg.QueriesPath, cfg.MaxQueries)
	if err != nil {
		return EventTraceVectorExportSummary{}, err
	}
	if len(traces) == 0 {
		return EventTraceVectorExportSummary{}, fmt.Errorf("traces file is empty")
	}
	if len(queries) == 0 {
		return EventTraceVectorExportSummary{}, fmt.Errorf("queries are empty")
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return EventTraceVectorExportSummary{}, err
	}

	events := renderEventTraceChunks(traces)
	if len(events) == 0 {
		return EventTraceVectorExportSummary{}, fmt.Errorf("event trace export selected no events")
	}
	childPath := filepath.Join(cfg.OutputDir, "child-doc-vectors.jsonl")
	queryPath := filepath.Join(cfg.OutputDir, "query-vectors.jsonl")
	corpusPath := filepath.Join(cfg.OutputDir, "corpus.jsonl")
	beirQueriesPath := filepath.Join(cfg.OutputDir, "queries.jsonl")
	if err := writeEventTraceBEIRCorpus(traces, corpusPath); err != nil {
		return EventTraceVectorExportSummary{}, fmt.Errorf("write BEIR corpus helper: %w", err)
	}
	if err := writeEventTraceBEIRQueries(queries, beirQueriesPath); err != nil {
		return EventTraceVectorExportSummary{}, fmt.Errorf("write BEIR queries helper: %w", err)
	}
	dim, modelDim, err := writeRetrievalChildVectorCache(ctx, model, events, childPath, cfg.BatchSize, cfg.TracePrefix, cfg.OutputDim)
	if err != nil {
		return EventTraceVectorExportSummary{}, fmt.Errorf("write child event vectors: %w", err)
	}
	queryDim, queryModelDim, err := writeRetrievalVectorCache(ctx, model, queries, queryPath, cfg.BatchSize, cfg.QueryPrefix, cfg.OutputDim)
	if err != nil {
		return EventTraceVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if dim != queryDim {
		return EventTraceVectorExportSummary{}, fmt.Errorf("event vectors have dimension %d but query vectors have dimension %d", dim, queryDim)
	}
	if modelDim != queryModelDim {
		return EventTraceVectorExportSummary{}, fmt.Errorf("event vectors have encoded dimension %d but query vectors have encoded dimension %d", modelDim, queryModelDim)
	}

	summary := EventTraceVectorExportSummary{
		Schema:             EventTraceVectorExportManifestSchema,
		Dataset:            cfg.DatasetName,
		Artifact:           cfg.ArtifactPath,
		Backend:            string(model.Backend()),
		Traces:             len(traces),
		Queries:            len(queries),
		ChildVectors:       len(events),
		Dimension:          dim,
		ModelDimension:     modelDim,
		OutputDimension:    dim,
		CorpusPath:         corpusPath,
		BEIRQueriesPath:    beirQueriesPath,
		ChildDocVectorPath: childPath,
		QueryVectorPath:    queryPath,
		BatchSize:          cfg.BatchSize,
		MaxTraces:          cfg.MaxTraces,
		MaxQueries:         cfg.MaxQueries,
		TracesPath:         cfg.TracesPath,
		QueriesPath:        cfg.QueriesPath,
		OutputDir:          cfg.OutputDir,
		ElapsedSeconds:     time.Since(start).Seconds(),
		CreatedAt:          time.Now().UTC(),
	}
	if cfg.ManifestJSONPath != "" {
		if err := WriteEventTraceVectorExportSummaryFile(cfg.ManifestJSONPath, summary); err != nil {
			return EventTraceVectorExportSummary{}, err
		}
	}
	return summary, nil
}

// WriteEventTraceVectorExportSummaryFile writes a JSON manifest for an event trace export run.
func WriteEventTraceVectorExportSummaryFile(path string, summary EventTraceVectorExportSummary) error {
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

func normalizeEventTraceVectorExportConfig(cfg EventTraceVectorExportConfig) EventTraceVectorExportConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "event-traces"
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	return cfg
}

func validateEventTraceVectorExportConfig(cfg EventTraceVectorExportConfig) error {
	if cfg.BatchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	if cfg.MaxTraces < 0 || cfg.MaxQueries < 0 {
		return fmt.Errorf("max-traces and max-queries must be non-negative")
	}
	if cfg.OutputDim < 0 {
		return fmt.Errorf("output-dim must be non-negative")
	}
	return nil
}

func readEventTraceExportRecords(path string, limit int) ([]eventTraceRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	out := []eventTraceRecord{}
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
			ID     string            `json:"id"`
			BEIRID string            `json:"_id"`
			Label  string            `json:"label"`
			Text   string            `json:"text"`
			Events []json.RawMessage `json:"events"`
		}
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		id := firstNonEmptyString(record.ID, record.BEIRID)
		if id == "" {
			return nil, fmt.Errorf("%s:%d: trace id is required", path, lineNo)
		}
		if len(record.Events) == 0 {
			return nil, fmt.Errorf("%s:%d: events array is required for trace %q", path, lineNo, id)
		}
		events := make([]eventTraceEvent, 0, len(record.Events))
		for i, raw := range record.Events {
			event, err := parseEventTraceEvent(raw)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: events[%d] for trace %q: %w", path, lineNo, i, id, err)
			}
			events = append(events, event)
		}
		out = append(out, eventTraceRecord{
			ID:     id,
			Label:  strings.TrimSpace(record.Label),
			Text:   strings.TrimSpace(record.Text),
			Events: events,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func parseEventTraceEvent(raw json.RawMessage) (eventTraceEvent, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		text = strings.TrimSpace(text)
		if text == "" {
			return eventTraceEvent{}, fmt.Errorf("string event is empty")
		}
		return eventTraceEvent{Text: text}, nil
	}

	var record struct {
		ID        string            `json:"id"`
		Type      string            `json:"type"`
		Time      string            `json:"time"`
		Timestamp string            `json:"timestamp"`
		Role      string            `json:"role"`
		Actor     string            `json:"actor"`
		Text      string            `json:"text"`
		Message   string            `json:"message"`
		Summary   string            `json:"summary"`
		Metadata  map[string]string `json:"metadata"`
	}
	if err := json.Unmarshal(raw, &record); err != nil {
		return eventTraceEvent{}, err
	}
	event := eventTraceEvent{
		ID:       strings.TrimSpace(record.ID),
		Type:     strings.TrimSpace(record.Type),
		Time:     strings.TrimSpace(firstNonEmptyString(record.Time, record.Timestamp)),
		Role:     strings.TrimSpace(record.Role),
		Actor:    strings.TrimSpace(record.Actor),
		Text:     strings.TrimSpace(record.Text),
		Message:  strings.TrimSpace(record.Message),
		Summary:  strings.TrimSpace(record.Summary),
		Metadata: trimStringMap(record.Metadata),
	}
	if event.isEmpty() {
		return eventTraceEvent{}, fmt.Errorf("object event has no supported non-empty fields")
	}
	return event, nil
}

func (event eventTraceEvent) isEmpty() bool {
	return event.ID == "" &&
		event.Type == "" &&
		event.Time == "" &&
		event.Role == "" &&
		event.Actor == "" &&
		event.Text == "" &&
		event.Message == "" &&
		event.Summary == "" &&
		len(event.Metadata) == 0
}

func readEventTraceExportQueries(path string, limit int) ([]retrievalTextRecord, error) {
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

func writeEventTraceBEIRCorpus(traces []eventTraceRecord, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	for _, record := range traces {
		row := beirJSONRecord{
			ID:   record.ID,
			Text: renderEventTraceCorpusText(record),
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

func writeEventTraceBEIRQueries(queries []retrievalTextRecord, path string) error {
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

func renderEventTraceChunks(traces []eventTraceRecord) []retrievalDocumentChunk {
	out := []retrievalDocumentChunk{}
	for _, trace := range traces {
		for i, event := range trace.Events {
			out = append(out, retrievalDocumentChunk{
				ParentID: trace.ID,
				ChildID:  fmt.Sprintf("%s#event-%04d", trace.ID, i),
				Text:     renderEventTraceEventText(trace, i, event),
			})
		}
	}
	return out
}

func renderEventTraceCorpusText(record eventTraceRecord) string {
	parts := []string{
		"trace_id: " + record.ID,
		fmt.Sprintf("events: count=%d", len(record.Events)),
	}
	if record.Label != "" {
		parts = append([]string{"label: " + record.Label}, parts...)
	}
	if record.Text != "" {
		parts = append([]string{"text: " + record.Text}, parts...)
	}
	return strings.Join(parts, "\n")
}

func renderEventTraceEventText(trace eventTraceRecord, index int, event eventTraceEvent) string {
	parts := []string{}
	if trace.Text != "" {
		parts = append(parts, "trace_text: "+trace.Text)
	}
	if trace.Label != "" {
		parts = append(parts, "trace_label: "+trace.Label)
	}
	parts = append(parts,
		"trace_id: "+trace.ID,
		fmt.Sprintf("event_index: %d", index),
	)
	appendField := func(name, value string) {
		value = strings.TrimSpace(value)
		if value != "" {
			parts = append(parts, name+": "+value)
		}
	}
	appendField("event_id", event.ID)
	appendField("event_type", event.Type)
	appendField("event_time", event.Time)
	appendField("role", event.Role)
	appendField("actor", event.Actor)
	appendField("text", event.Text)
	appendField("message", event.Message)
	appendField("summary", event.Summary)
	if len(event.Metadata) > 0 {
		keys := make([]string, 0, len(event.Metadata))
		for key := range event.Metadata {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		values := make([]string, 0, len(keys))
		for _, key := range keys {
			values = append(values, key+"="+event.Metadata[key])
		}
		parts = append(parts, "metadata: "+strings.Join(values, " "))
	}
	return strings.Join(parts, "\n")
}

func trimStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	out := map[string]string{}
	for key, value := range values {
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" || value == "" {
			continue
		}
		out[key] = value
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
