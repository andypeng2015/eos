package eosruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadEmbeddingListwiseGeometryBatchesFileAcceptsTinyValidJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "listwise-geometry.jsonl")
	row := `{"schema":"eos.listwise_geometry_batch.v1","batch_id":"b1","source_counts":{"toy":2},"examples":[{"row_id":"r1","source":"toy","query_id":"q1","positive_doc_id":"d1","negative_doc_ids":["d2"]},{"row_id":"r2","source":"toy","query_id":"q2","positive_doc_id":"d3","negative_doc_ids":["d1"]}],"queries":[{"id":"q1","text":"first query"},{"id":"q2","text":"second query"}],"documents":[{"id":"d1","text":"first doc","role":"positive"},{"id":"d2","text":"second doc","role":"negative"},{"id":"d3","text":"third doc","role":"positive"}],"teacher_similarity":[[0.9,0.1,0.3],[0.2,0.4,0.8]],"teacher_model_id":"teacher-v1","score":"cosine","normalized":true}`
	if err := os.WriteFile(path, []byte(row+"\n"), 0o644); err != nil {
		t.Fatalf("write listwise geometry dataset: %v", err)
	}

	got, err := ReadEmbeddingListwiseGeometryBatchesFile(path)
	if err != nil {
		t.Fatalf("read listwise geometry dataset: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("batch count = %d, want 1", len(got))
	}
	batch := got[0]
	if batch.Schema != embeddingListwiseGeometryBatchSchemaV1 || batch.BatchID != "b1" || batch.TeacherModelID != "teacher-v1" || !batch.Normalized {
		t.Fatalf("batch metadata = %+v, want schema/batch/model/normalized preserved", batch)
	}
	if batch.Score != "cosine" {
		t.Fatalf("score = %q, want cosine", batch.Score)
	}
	if batch.SourceCounts["toy"] != 2 {
		t.Fatalf("source_counts = %+v, want toy=2", batch.SourceCounts)
	}
	if len(batch.Examples) != 2 || batch.Examples[0].RowID != "r1" || batch.Examples[1].PositiveDocID != "d3" {
		t.Fatalf("examples = %+v, want provenance preserved", batch.Examples)
	}
	if len(batch.Queries) != 2 || len(batch.Documents) != 3 || len(batch.TeacherSimilarity) != 2 || len(batch.TeacherSimilarity[0]) != 3 {
		t.Fatalf("shape = queries %d docs %d matrix %dx%d, want 2/3/2x3", len(batch.Queries), len(batch.Documents), len(batch.TeacherSimilarity), len(batch.TeacherSimilarity[0]))
	}
}

func TestReadEmbeddingListwiseGeometryBatchesFileRejectsInvalidRows(t *testing.T) {
	validPrefix := `"batch_id":"b1","source_counts":{"toy":1},"examples":[{"row_id":"r1","source":"toy","query_id":"q1","positive_doc_id":"d1","negative_doc_ids":["d2"]}],"queries":[{"id":"q1","text":"query"}],"documents":[{"id":"d1","text":"doc one"},{"id":"d2","text":"doc two"}],"teacher_model_id":"teacher-v1","normalized":true`
	twoQueryPrefix := `"batch_id":"b1","source_counts":{"toy":1},"examples":[{"row_id":"r1","source":"toy","query_id":"q1","positive_doc_id":"d1","negative_doc_ids":["d2"]}],"queries":[{"id":"q1","text":"query"},{"id":"q2","text":"second query"}],"documents":[{"id":"d1","text":"doc one"},{"id":"d2","text":"doc two"}],"teacher_model_id":"teacher-v1","normalized":true`
	tests := []struct {
		name string
		row  string
		want string
	}{
		{
			name: "bad schema",
			row:  `{` + validPrefix + `,"schema":"wrong","teacher_similarity":[[1,0]]}`,
			want: "schema",
		},
		{
			name: "ragged matrix",
			row:  `{` + twoQueryPrefix + `,"schema":"eos.listwise_geometry_batch.v1","teacher_similarity":[[1,0],[0]]}`,
			want: "document count",
		},
		{
			name: "non finite matrix",
			row:  `{` + validPrefix + `,"schema":"eos.listwise_geometry_batch.v1","teacher_similarity":[[1e999,0]]}`,
			want: "cannot unmarshal",
		},
		{
			name: "mismatched dimensions",
			row:  `{` + validPrefix + `,"schema":"eos.listwise_geometry_batch.v1","teacher_similarity":[[1,0],[0,1]]}`,
			want: "row count",
		},
		{
			name: "unsupported score",
			row:  `{` + validPrefix + `,"schema":"eos.listwise_geometry_batch.v1","teacher_similarity":[[1,0]],"score":"euclidean"}`,
			want: "unsupported",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "listwise-geometry.jsonl")
			if err := os.WriteFile(path, []byte(tc.row+"\n"), 0o644); err != nil {
				t.Fatalf("write listwise geometry dataset: %v", err)
			}
			_, err := ReadEmbeddingListwiseGeometryBatchesFile(path)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want containing %q", err, tc.want)
			}
		})
	}
}
