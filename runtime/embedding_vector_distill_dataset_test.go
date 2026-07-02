package eosruntime

import (
	"os"
	"path/filepath"
	"testing"
)

// TestReadEmbeddingVectorDistillExamplesFileParsesExplicitRole verifies that
// "role":"query"/"document"/"raw" rows are read back with their Role field set.
func TestReadEmbeddingVectorDistillExamplesFileParsesExplicitRole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vector-distill.jsonl")
	data := `` +
		`{"schema":"eos.vector_distill_example.v1","id":"q1","text":"a query","teacher_vector":[0.1,0.2],"role":"query"}` + "\n" +
		`{"schema":"eos.vector_distill_example.v1","id":"d1","text":"a document","teacher_vector":[0.3,0.4],"role":"document"}` + "\n" +
		`{"schema":"eos.vector_distill_example.v1","id":"r1","text":"raw text","teacher_vector":[0.5,0.6],"role":"raw"}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write vector-distill dataset: %v", err)
	}
	got, err := ReadEmbeddingVectorDistillExamplesFile(path)
	if err != nil {
		t.Fatalf("read vector-distill dataset: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3", len(got))
	}
	want := map[string]string{"q1": EmbeddingRoleQuery, "d1": EmbeddingRoleDocument, "r1": EmbeddingRoleRaw}
	for _, ex := range got {
		if ex.Role != want[ex.ID] {
			t.Errorf("example %q role = %q, want %q", ex.ID, ex.Role, want[ex.ID])
		}
	}
}

// TestReadEmbeddingVectorDistillExamplesFileAllowsMissingRole verifies legacy
// rows without a "role" field still parse, leaving Role empty (fallback to a
// trainer-configured default applies at training time, not at read time).
func TestReadEmbeddingVectorDistillExamplesFileAllowsMissingRole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vector-distill.jsonl")
	data := `{"schema":"eos.vector_distill_example.v1","id":"legacy1","text":"legacy row","teacher_vector":[0.1,0.2]}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write vector-distill dataset: %v", err)
	}
	got, err := ReadEmbeddingVectorDistillExamplesFile(path)
	if err != nil {
		t.Fatalf("read vector-distill dataset: %v", err)
	}
	if len(got) != 1 || got[0].Role != "" {
		t.Fatalf("got = %+v, want one example with empty Role", got)
	}
}

// TestReadEmbeddingVectorDistillExamplesFileRejectsUnknownRole verifies role
// values outside {query, document, raw, ""} are rejected at read time.
func TestReadEmbeddingVectorDistillExamplesFileRejectsUnknownRole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vector-distill.jsonl")
	data := `{"schema":"eos.vector_distill_example.v1","id":"bad1","text":"bad role","teacher_vector":[0.1,0.2],"role":"passage"}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write vector-distill dataset: %v", err)
	}
	if _, err := ReadEmbeddingVectorDistillExamplesFile(path); err == nil {
		t.Fatal("read vector-distill dataset succeeded with unknown role value")
	}
}

// TestTokenizeEmbeddingVectorDistillExamplesPreservesRole verifies the
// tokenized form carries the Role field through verbatim.
func TestTokenizeEmbeddingVectorDistillExamplesPreservesRole(t *testing.T) {
	tokenizer := newVectorDistillDatasetTestTokenizer(t)
	examples := []EmbeddingVectorDistillExample{
		{Schema: embeddingVectorDistillExampleSchemaV1, ID: "q1", Text: "ab", TeacherVector: []float32{0.1, 0.2}, Role: EmbeddingRoleQuery},
		{Schema: embeddingVectorDistillExampleSchemaV1, ID: "d1", Text: "cd", TeacherVector: []float32{0.3, 0.4}, Role: EmbeddingRoleDocument},
		{Schema: embeddingVectorDistillExampleSchemaV1, ID: "legacy1", Text: "ab", TeacherVector: []float32{0.5, 0.6}},
	}
	tokenized, err := TokenizeEmbeddingVectorDistillExamples(examples, tokenizer)
	if err != nil {
		t.Fatalf("tokenize vector-distill dataset: %v", err)
	}
	if len(tokenized) != 3 {
		t.Fatalf("len(tokenized) = %d, want 3", len(tokenized))
	}
	want := map[string]string{"q1": EmbeddingRoleQuery, "d1": EmbeddingRoleDocument, "legacy1": ""}
	for _, ex := range tokenized {
		if ex.Role != want[ex.ID] {
			t.Errorf("tokenized example %q role = %q, want %q", ex.ID, ex.Role, want[ex.ID])
		}
	}
}

func newVectorDistillDatasetTestTokenizer(t *testing.T) *BPETokenizer {
	t.Helper()
	file := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	tokenizer, err := NewBPETokenizer(file, TokenizerManifest{VocabSize: 8, MaxSequence: 8})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	return tokenizer
}
