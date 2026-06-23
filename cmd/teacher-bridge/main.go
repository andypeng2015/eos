// Command teacher-bridge scores eos teacher-score-request rows with a local
// teacher scorer, for offline distillation-signal generation.
// It is OFFLINE TOOLING ONLY: the shipped eos model never depends on ollama or
// any external service — the teacher is used purely to label training data.
//
// Pipeline:
//
//	eos export-teacher-score-requests <hard-negatives.jsonl> <requests.jsonl>
//	teacher-bridge <model> <requests.jsonl> <scored.jsonl>      # legacy embedding mode
//	teacher-bridge --mode http-rerank --endpoint http://127.0.0.1:8080/score --model <model> <requests.jsonl> <scored.jsonl>
//	eos import-teacher-scores <hard-negatives.jsonl> <scored.jsonl> <out.jsonl>
//	eos audit-teacher-scores <out.jsonl>
//
// For each {query, candidate} request row it embeds both texts with the teacher
// model and writes the same row plus a cosine-similarity "score". Output is
// deduplicated by (query, candidate) so eos import-teacher-scores accepts it.
//
// The ollama endpoint defaults to http://localhost:11434/api/embed and can be
// overridden with OLLAMA_EMBED_URL.
//
// usage:
//
//	teacher-bridge <model> <requests.jsonl> <scored.jsonl>
//	teacher-bridge --mode http-rerank --endpoint <url> --model <model> [flags] <requests.jsonl> <scored.jsonl>
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func ollamaURL() string {
	if u := os.Getenv("OLLAMA_EMBED_URL"); u != "" {
		return u
	}
	return "http://localhost:11434/api/embed"
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 3 && !strings.HasPrefix(args[0], "-") {
		return runLegacyEmbedding(args[0], args[1], args[2])
	}
	fs := flag.NewFlagSet("teacher-bridge", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	mode := fs.String("mode", "", "scoring mode: http-rerank")
	endpoint := fs.String("endpoint", "", "HTTP reranker scoring endpoint")
	model := fs.String("model", "", "teacher model identifier to pass to the scorer")
	batchSize := fs.Int("batch-size", 16, "HTTP pair batch size")
	scoreScale := fs.String("score-scale", "logit", "score scale provenance, such as logit, probability, or raw")
	manifestPath := fs.String("manifest", "", "bridge provenance manifest path; default is <output>.teacher-bridge.manifest.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *mode == "" {
		return fmt.Errorf("usage: teacher-bridge <model> <requests.jsonl> <scored.jsonl>\n   or: teacher-bridge --mode http-rerank --endpoint <url> --model <model> [flags] <requests.jsonl> <scored.jsonl>")
	}
	if *mode != "http-rerank" {
		return fmt.Errorf("unsupported mode %q", *mode)
	}
	if fs.NArg() != 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: teacher-bridge --mode http-rerank --endpoint <url> --model <model> [flags] <requests.jsonl> <scored.jsonl>")
	}
	if strings.TrimSpace(*endpoint) == "" {
		return fmt.Errorf("--endpoint is required for http-rerank mode")
	}
	if strings.TrimSpace(*model) == "" {
		return fmt.Errorf("--model is required for http-rerank mode")
	}
	if *batchSize <= 0 {
		return fmt.Errorf("--batch-size must be positive")
	}
	inPath, outPath := fs.Arg(0), fs.Arg(1)
	if *manifestPath == "" {
		*manifestPath = outPath + ".teacher-bridge.manifest.json"
	}
	return runHTTPRerank(httpRerankConfig{
		Endpoint:     *endpoint,
		Model:        *model,
		BatchSize:    *batchSize,
		ScoreScale:   *scoreScale,
		InputPath:    inPath,
		OutputPath:   outPath,
		ManifestPath: *manifestPath,
	})
}

func runLegacyEmbedding(model, inPath, outPath string) error {
	rows, err := readRows(inPath)
	if err != nil {
		return fmt.Errorf("read: %w", err)
	}

	uniq := map[string]struct{}{}
	for _, r := range rows {
		uniq[clip(r.text("query"))] = struct{}{}
		uniq[clip(r.text("candidate"))] = struct{}{}
	}
	texts := make([]string, 0, len(uniq))
	for t := range uniq {
		texts = append(texts, t)
	}
	fmt.Fprintf(os.Stderr, "rows=%d unique_texts=%d model=%s\n", len(rows), len(texts), model)

	cache := map[string][]float64{}
	const batch = 8
	for i := 0; i < len(texts); i += batch {
		end := i + batch
		if end > len(texts) {
			end = len(texts)
		}
		embs, err := embed(model, texts[i:end])
		if err != nil {
			return fmt.Errorf("embed batch %d: %w", i, err)
		}
		for j, e := range embs {
			cache[texts[i+j]] = e
		}
		if (i/batch)%50 == 0 {
			fmt.Fprintf(os.Stderr, "embedded %d/%d\n", end, len(texts))
		}
	}

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()

	// Deduplicate by (query, candidate), keeping the highest score, so the
	// output is directly consumable by eos import-teacher-scores.
	seen := map[string]float64{}
	written := 0
	for _, r := range rows {
		q, c := clip(r.text("query")), clip(r.text("candidate"))
		score := cosine(cache[q], cache[c])
		key := q + "\x00" + c
		if prev, ok := seen[key]; ok {
			if score <= prev {
				continue
			}
		}
		seen[key] = score
		r.m["score"] = score
		b, _ := json.Marshal(r.m)
		w.Write(b)
		w.WriteByte('\n')
		written++
	}
	fmt.Fprintf(os.Stderr, "wrote %d deduped scored rows to %s\n", written, outPath)
	return nil
}

type row struct{ m map[string]any }

func (r row) text(k string) string {
	if s, ok := r.m[k].(string); ok {
		return s
	}
	return ""
}

func readRows(path string) ([]row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		m := map[string]any{}
		if err := json.Unmarshal(line, &m); err != nil {
			return nil, err
		}
		rows = append(rows, row{m: m})
	}
	return rows, sc.Err()
}

type httpRerankConfig struct {
	Endpoint     string
	Model        string
	BatchSize    int
	ScoreScale   string
	InputPath    string
	OutputPath   string
	ManifestPath string
}

type httpPair struct {
	ID             string `json:"id"`
	Source         string `json:"source,omitempty"`
	Query          string `json:"query"`
	Candidate      string `json:"candidate"`
	Role           string `json:"role,omitempty"`
	ExampleIndex   int    `json:"example_index,omitempty"`
	CandidateIndex int    `json:"candidate_index,omitempty"`
}

type httpScoreRequest struct {
	Model string     `json:"model"`
	Pairs []httpPair `json:"pairs"`
}

type httpScoreResponse struct {
	Scores []httpScore `json:"scores"`
}

type httpScore struct {
	ID    string          `json:"id"`
	Score json.RawMessage `json:"score"`
}

type bridgeManifest struct {
	Schema            string `json:"schema"`
	CreatedUTC        string `json:"created_utc"`
	Mode              string `json:"mode"`
	Endpoint          string `json:"endpoint"`
	Model             string `json:"model"`
	ScoreScale        string `json:"score_scale,omitempty"`
	InputJSONL        string `json:"input_jsonl"`
	OutputJSONL       string `json:"output_jsonl"`
	RowsRead          int    `json:"rows_read"`
	RowsWritten       int    `json:"rows_written"`
	DuplicatesSkipped int    `json:"duplicates_skipped"`
}

func runHTTPRerank(cfg httpRerankConfig) error {
	rows, err := readRows(cfg.InputPath)
	if err != nil {
		return fmt.Errorf("read: %w", err)
	}
	pairs, scoreRows, skipped := uniqueHTTPPairs(rows)
	if len(pairs) == 0 {
		return fmt.Errorf("no scoreable rows in %s", cfg.InputPath)
	}

	out, err := os.Create(cfg.OutputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	w := bufio.NewWriter(out)
	written := 0
	closeOutput := func() error {
		if err := w.Flush(); err != nil {
			_ = out.Close()
			return err
		}
		return out.Close()
	}

	for start := 0; start < len(pairs); start += cfg.BatchSize {
		end := start + cfg.BatchSize
		if end > len(pairs) {
			end = len(pairs)
		}
		scores, err := scoreHTTPBatch(cfg.Endpoint, cfg.Model, pairs[start:end])
		if err != nil {
			_ = out.Close()
			return fmt.Errorf("score batch %d: %w", start, err)
		}
		for _, pair := range pairs[start:end] {
			score, ok := scores[pair.ID]
			if !ok {
				_ = out.Close()
				return fmt.Errorf("score batch %d: missing score for id %q", start, pair.ID)
			}
			record := scoreRows[pair.ID]
			record.m["score"] = score
			b, err := json.Marshal(record.m)
			if err != nil {
				_ = out.Close()
				return fmt.Errorf("encode score row: %w", err)
			}
			if _, err := w.Write(b); err != nil {
				_ = out.Close()
				return err
			}
			if err := w.WriteByte('\n'); err != nil {
				_ = out.Close()
				return err
			}
			written++
		}
		fmt.Fprintf(os.Stderr, "scored %d/%d\n", end, len(pairs))
	}
	if err := closeOutput(); err != nil {
		return fmt.Errorf("close output: %w", err)
	}

	manifest := bridgeManifest{
		Schema:            "manta.teacher_bridge_http_rerank.v1",
		CreatedUTC:        time.Now().UTC().Format(time.RFC3339),
		Mode:              "http-rerank",
		Endpoint:          cfg.Endpoint,
		Model:             cfg.Model,
		ScoreScale:        cfg.ScoreScale,
		InputJSONL:        cfg.InputPath,
		OutputJSONL:       cfg.OutputPath,
		RowsRead:          len(rows),
		RowsWritten:       written,
		DuplicatesSkipped: skipped,
	}
	if err := writeBridgeManifest(cfg.ManifestPath, manifest); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %d scored rows to %s (duplicates_skipped=%d)\n", written, cfg.OutputPath, skipped)
	fmt.Fprintf(os.Stderr, "manifest: %s\n", cfg.ManifestPath)
	return nil
}

func uniqueHTTPPairs(rows []row) ([]httpPair, map[string]row, int) {
	seen := map[string]string{}
	scoreRows := map[string]row{}
	pairs := make([]httpPair, 0, len(rows))
	skipped := 0
	for _, r := range rows {
		source, query, candidate := r.text("source"), r.text("query"), r.text("candidate")
		key := source + "\x00" + query + "\x00" + candidate
		if _, ok := seen[key]; ok {
			skipped++
			continue
		}
		id := strconv.Itoa(len(pairs))
		seen[key] = id
		pair := httpPair{
			ID:             id,
			Source:         source,
			Query:          query,
			Candidate:      candidate,
			Role:           r.text("role"),
			ExampleIndex:   intFromRow(r, "example_index"),
			CandidateIndex: intFromRow(r, "candidate_index"),
		}
		pairs = append(pairs, pair)
		scoreRows[id] = r
	}
	return pairs, scoreRows, skipped
}

func intFromRow(r row, key string) int {
	switch v := r.m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case json.Number:
		i, _ := v.Int64()
		return int(i)
	default:
		return 0
	}
}

func scoreHTTPBatch(endpoint, model string, pairs []httpPair) (map[string]float64, error) {
	body, err := json.Marshal(httpScoreRequest{Model: model, Pairs: pairs})
	if err != nil {
		return nil, err
	}
	resp, err := client.Post(endpoint, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("http status %s", resp.Status)
	}
	var sr httpScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, err
	}
	out := make(map[string]float64, len(sr.Scores))
	for _, score := range sr.Scores {
		if score.ID == "" {
			return nil, fmt.Errorf("score response contains empty id")
		}
		if _, exists := out[score.ID]; exists {
			return nil, fmt.Errorf("duplicate score response for id %q", score.ID)
		}
		value, err := parseFiniteScore(score.Score)
		if err != nil {
			return nil, fmt.Errorf("id %q: %w", score.ID, err)
		}
		out[score.ID] = value
	}
	return out, nil
}

func parseFiniteScore(raw json.RawMessage) (float64, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, fmt.Errorf("missing score")
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err != nil {
		return 0, fmt.Errorf("score must be numeric: %w", err)
	}
	score, err := strconv.ParseFloat(number.String(), 64)
	if err != nil {
		return 0, fmt.Errorf("score must be finite: %w", err)
	}
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0, fmt.Errorf("score must be finite")
	}
	return score, nil
}

func writeBridgeManifest(path string, manifest bridgeManifest) error {
	if path == "" {
		return nil
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(manifest); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

type embedReq struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}
type embedResp struct {
	Embeddings [][]float64 `json:"embeddings"`
}

var client = &http.Client{Timeout: 120 * time.Second}

// clip caps text length on a rune boundary so an over-long document can't blow
// past the teacher's context window and fail the whole batch.
func clip(s string) string {
	const maxRunes = 8000
	r := []rune(s)
	if len(r) > maxRunes {
		return string(r[:maxRunes])
	}
	return s
}

// embed is robust: on a batch failure or count mismatch it binary-splits down to
// singletons, hard-truncates a stubborn single text, and falls back to a zero
// vector (cosine 0) so one bad document can never abort the whole run.
func embed(model string, texts []string) ([][]float64, error) {
	out, err := embedRaw(model, texts)
	if err == nil && len(out) == len(texts) {
		return out, nil
	}
	if len(texts) == 1 {
		hard := texts[0]
		if r := []rune(hard); len(r) > 2000 {
			hard = string(r[:2000])
		}
		if out, err2 := embedRaw(model, []string{hard}); err2 == nil && len(out) == 1 {
			return out, nil
		}
		fmt.Fprintf(os.Stderr, "WARN: zero-vector for unembeddable text (len=%d)\n", len(texts[0]))
		return [][]float64{nil}, nil
	}
	mid := len(texts) / 2
	left, err := embed(model, texts[:mid])
	if err != nil {
		return nil, err
	}
	right, err := embed(model, texts[mid:])
	if err != nil {
		return nil, err
	}
	return append(left, right...), nil
}

func embedRaw(model string, texts []string) ([][]float64, error) {
	body, _ := json.Marshal(embedReq{Model: model, Input: texts})
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		resp, err := client.Post(ollamaURL(), "application/json", bytes.NewReader(body))
		if err != nil {
			lastErr = err
			time.Sleep(time.Second)
			continue
		}
		var er embedResp
		err = json.NewDecoder(resp.Body).Decode(&er)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			time.Sleep(time.Second)
			continue
		}
		if len(er.Embeddings) != len(texts) {
			return nil, fmt.Errorf("got %d embeddings for %d texts", len(er.Embeddings), len(texts))
		}
		return er.Embeddings, nil
	}
	return nil, lastErr
}

func cosine(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
