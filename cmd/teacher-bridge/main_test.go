package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestHTTPRerankSendsJointPairsAndDedupesBySource(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "requests.jsonl")
	outputPath := filepath.Join(dir, "scores.jsonl")
	manifestPath := filepath.Join(dir, "scores.manifest.json")
	writeFile(t, inputPath, strings.Join([]string{
		`{"source":"s1","query":"q","candidate":"doc","role":"positive","example_index":0,"candidate_index":0}`,
		`{"source":"s1","query":"q","candidate":"doc","role":"positive","example_index":0,"candidate_index":0}`,
		`{"source":"s2","query":"q","candidate":"doc","role":"negative","example_index":1,"candidate_index":1}`,
	}, "\n")+"\n")

	var requests []httpScoreRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req httpScoreRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		requests = append(requests, req)
		resp := httpScoreResponse{}
		for _, pair := range req.Pairs {
			if pair.Query == "" || pair.Candidate == "" {
				t.Fatalf("pair missing joint text: %+v", pair)
			}
			resp.Scores = append(resp.Scores, httpScore{ID: pair.ID, Score: json.RawMessage(fmt.Sprintf("%g", 10+float64(len(resp.Scores))))})
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	if err := run([]string{
		"--mode", "http-rerank",
		"--endpoint", server.URL,
		"--model", "fake-reranker",
		"--batch-size", "8",
		"--score-scale", "logit",
		"--manifest", manifestPath,
		inputPath,
		outputPath,
	}); err != nil {
		t.Fatalf("run http rerank: %v", err)
	}

	if len(requests) != 1 || requests[0].Model != "fake-reranker" || len(requests[0].Pairs) != 2 {
		t.Fatalf("requests = %+v", requests)
	}
	if requests[0].Pairs[0].Source != "s1" || requests[0].Pairs[1].Source != "s2" {
		t.Fatalf("sources were not preserved across duplicate query/candidate: %+v", requests[0].Pairs)
	}
	rows := readJSONLines(t, outputPath)
	if len(rows) != 2 {
		t.Fatalf("output rows = %d, want 2", len(rows))
	}
	if rows[0]["source"] != "s1" || rows[1]["source"] != "s2" || rows[0]["query"] != "q" || rows[0]["candidate"] != "doc" {
		t.Fatalf("output rows did not preserve source/query/candidate: %+v", rows)
	}
	var manifest bridgeManifest
	decodeFile(t, manifestPath, &manifest)
	if manifest.Schema != "manta.teacher_bridge_http_rerank.v1" || manifest.RowsRead != 3 || manifest.RowsWritten != 2 || manifest.DuplicatesSkipped != 1 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestHTTPRerankFailsOnMissingScore(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "requests.jsonl")
	outputPath := filepath.Join(dir, "scores.jsonl")
	writeFile(t, inputPath, `{"source":"s","query":"q","candidate":"doc"}`+"\n")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"scores":[]}`))
	}))
	defer server.Close()

	err := run([]string{"--mode", "http-rerank", "--endpoint", server.URL, "--model", "fake", inputPath, outputPath})
	if err == nil || !strings.Contains(err.Error(), "missing score") {
		t.Fatalf("err = %v, want missing score failure", err)
	}
}

func TestHTTPRerankFailsOnNonFiniteScore(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "requests.jsonl")
	outputPath := filepath.Join(dir, "scores.jsonl")
	writeFile(t, inputPath, `{"source":"s","query":"q","candidate":"doc"}`+"\n")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"scores":[{"id":"0","score":1e999}]}`))
	}))
	defer server.Close()

	err := run([]string{"--mode", "http-rerank", "--endpoint", server.URL, "--model", "fake", inputPath, outputPath})
	if err == nil || !strings.Contains(err.Error(), "finite") {
		t.Fatalf("err = %v, want finite-score failure", err)
	}
}

func TestTEIRerankSendsNativeGroupedRequestsAndWritesImportRows(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "requests.jsonl")
	outputPath := filepath.Join(dir, "scores.jsonl")
	manifestPath := filepath.Join(dir, "scores.manifest.json")
	writeFile(t, inputPath, strings.Join([]string{
		`{"source":"s1","query":"q1","candidate":"doc-a","role":"positive","example_index":0,"candidate_index":0}`,
		`{"source":"s1","query":"q2","candidate":"doc-b","role":"negative","example_index":1,"candidate_index":1}`,
		`{"source":"s1","query":"q1","candidate":"doc-c","role":"negative","example_index":0,"candidate_index":2}`,
		`{"source":"s1","query":"q1","candidate":"doc-a","role":"positive","example_index":0,"candidate_index":0}`,
	}, "\n")+"\n")

	var requests []teiRerankRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req teiRerankRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		requests = append(requests, req)
		switch req.Query {
		case "q1":
			if fmt.Sprint(req.Texts) != "[doc-a doc-c]" {
				t.Fatalf("q1 texts = %+v", req.Texts)
			}
			_, _ = w.Write([]byte(`[{"index":0,"score":2.5},{"index":1,"score":1.5}]`))
		case "q2":
			if fmt.Sprint(req.Texts) != "[doc-b]" {
				t.Fatalf("q2 texts = %+v", req.Texts)
			}
			_, _ = w.Write([]byte(`{"results":[{"index":0,"score":0.5}]}`))
		default:
			t.Fatalf("unexpected query %q", req.Query)
		}
	}))
	defer server.Close()

	if err := run([]string{
		"--mode", "tei-rerank",
		"--endpoint", server.URL,
		"--model", "BAAI/bge-reranker-v2-m3",
		"--batch-size", "16",
		"--score-scale", "logit",
		"--manifest", manifestPath,
		inputPath,
		outputPath,
	}); err != nil {
		t.Fatalf("run tei rerank: %v", err)
	}

	if len(requests) != 2 || requests[0].Query != "q1" || requests[1].Query != "q2" {
		t.Fatalf("requests = %+v", requests)
	}
	rows := readJSONLines(t, outputPath)
	if len(rows) != 3 {
		t.Fatalf("output rows = %d, want 3", len(rows))
	}
	for _, row := range rows {
		if row["source"] == "" || row["query"] == "" || row["candidate"] == "" {
			t.Fatalf("output row is not import-compatible: %+v", row)
		}
		if _, ok := row["score"].(float64); !ok {
			t.Fatalf("output row missing numeric score: %+v", row)
		}
	}
	var manifest bridgeManifest
	decodeFile(t, manifestPath, &manifest)
	if manifest.Schema != "manta.teacher_bridge_tei_rerank.v1" || manifest.Mode != "tei-rerank" || manifest.Model != "BAAI/bge-reranker-v2-m3" || manifest.RowsRead != 4 || manifest.RowsWritten != 3 || manifest.DuplicatesSkipped != 1 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestParseTEIRerankResultsAcceptsArrayAndWrapper(t *testing.T) {
	got, err := parseTEIRerankResults(strings.NewReader(`[{"index":1,"score":0.25},{"index":0,"score":0.75}]`), 2)
	if err != nil {
		t.Fatalf("parse array: %v", err)
	}
	if got[0] != 0.75 || got[1] != 0.25 {
		t.Fatalf("array scores = %+v", got)
	}
	got, err = parseTEIRerankResults(strings.NewReader(`{"results":[{"index":0,"score":3.5}]}`), 1)
	if err != nil {
		t.Fatalf("parse wrapper: %v", err)
	}
	if got[0] != 3.5 {
		t.Fatalf("wrapper scores = %+v", got)
	}
}

func TestTEIRerankRejectsBadResponses(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{name: "missing score", body: `[{"index":0}]`, want: "missing score"},
		{name: "non finite score", body: `[{"index":0,"score":1e999}]`, want: "finite"},
		{name: "invalid index", body: `[{"index":2,"score":1}]`, want: "invalid TEI result index"},
		{name: "duplicate index", body: `[{"index":0,"score":1},{"index":0,"score":2}]`, want: "duplicate TEI result index"},
		{name: "missing index", body: `[{"score":1}]`, want: "missing TEI result index"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseTEIRerankResults(strings.NewReader(tc.body), 1)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestTEIRerankFailsOnMissingIndexResponse(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "requests.jsonl")
	outputPath := filepath.Join(dir, "scores.jsonl")
	writeFile(t, inputPath, strings.Join([]string{
		`{"source":"s","query":"q","candidate":"doc-a"}`,
		`{"source":"s","query":"q","candidate":"doc-b"}`,
	}, "\n")+"\n")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"index":0,"score":1}]`))
	}))
	defer server.Close()

	err := run([]string{"--mode", "tei-rerank", "--endpoint", server.URL, "--model", "fake", inputPath, outputPath})
	if err == nil || !strings.Contains(err.Error(), "missing score") {
		t.Fatalf("err = %v, want missing score failure", err)
	}
}

func TestHTTPRerankOutputImportsIntoEOS(t *testing.T) {
	dir := t.TempDir()
	hardNegativesPath := filepath.Join(dir, "hard-negatives.jsonl")
	requestPath := filepath.Join(dir, "requests.jsonl")
	scorePath := filepath.Join(dir, "scores.jsonl")
	outputPath := filepath.Join(dir, "with-teacher.jsonl")
	writeFile(t, hardNegativesPath, `{"source":"nfcorpus","query":"vitamin c","positive":"ascorbic acid","negatives":["calcium"]}`+"\n")
	writeFile(t, requestPath, strings.Join([]string{
		`{"source":"nfcorpus","query":"vitamin c","candidate":"ascorbic acid","role":"positive","example_index":0,"candidate_index":0}`,
		`{"source":"nfcorpus","query":"vitamin c","candidate":"calcium","role":"negative","example_index":0,"candidate_index":1}`,
	}, "\n")+"\n")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req httpScoreRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		resp := httpScoreResponse{}
		for _, pair := range req.Pairs {
			score := 0.1
			if pair.Candidate == "ascorbic acid" {
				score = 0.9
			}
			resp.Scores = append(resp.Scores, httpScore{ID: pair.ID, Score: json.RawMessage(fmt.Sprintf("%g", score))})
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()
	if err := run([]string{"--mode", "http-rerank", "--endpoint", server.URL, "--model", "fake", requestPath, scorePath}); err != nil {
		t.Fatalf("run http rerank: %v", err)
	}

	cmd := exec.Command("go", "run", "../eos", "import-teacher-scores",
		"--teacher-model-id", "fake",
		"--score-scale", "logit",
		hardNegativesPath,
		scorePath,
		outputPath,
	)
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("import-teacher-scores failed: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "updated=1") {
		t.Fatalf("import output = %s", out)
	}
	rows := readJSONLines(t, outputPath)
	scores, ok := rows[0]["teacher_scores"].([]any)
	if !ok || len(scores) != 2 {
		t.Fatalf("teacher_scores = %+v", rows[0]["teacher_scores"])
	}
	if math.Abs(scores[0].(float64)-0.9) > 0.000001 || math.Abs(scores[1].(float64)-0.1) > 0.000001 {
		t.Fatalf("teacher_scores = %+v", scores)
	}
}

func writeFile(t *testing.T, path, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readJSONLines(t *testing.T, path string) []map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var rows []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("decode %s: %v", line, err)
		}
		rows = append(rows, row)
	}
	return rows
}

func decodeFile(t *testing.T, path string, out any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, out); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}
