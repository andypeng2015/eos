// Command teacher-bridge scores eos teacher-score-request rows with a local
// embedding model served by ollama, for offline distillation-signal generation.
// It is OFFLINE TOOLING ONLY: the shipped eos model never depends on ollama or
// any external service — the teacher is used purely to label training data.
//
// Pipeline:
//
//	eos export-teacher-score-requests <hard-negatives.jsonl> <requests.jsonl>
//	teacher-bridge <model> <requests.jsonl> <scored.jsonl>      # <- this tool
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
// usage: teacher-bridge <model> <requests.jsonl> <scored.jsonl>
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"time"
)

func ollamaURL() string {
	if u := os.Getenv("OLLAMA_EMBED_URL"); u != "" {
		return u
	}
	return "http://localhost:11434/api/embed"
}

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: teacher-bridge <model> <requests.jsonl> <scored.jsonl>")
		os.Exit(1)
	}
	model, inPath, outPath := os.Args[1], os.Args[2], os.Args[3]

	rows, err := readRows(inPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
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
			fmt.Fprintf(os.Stderr, "embed batch %d: %v\n", i, err)
			os.Exit(1)
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
		fmt.Fprintf(os.Stderr, "create: %v\n", err)
		os.Exit(1)
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
