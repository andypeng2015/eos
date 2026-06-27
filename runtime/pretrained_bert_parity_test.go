package eosruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestPretrainedBERTHFParitySnapshot(t *testing.T) {
	snapshot := os.Getenv("EOS_HF_BERT_PARITY_SNAPSHOT")
	python := os.Getenv("EOS_HF_BERT_PARITY_PYTHON")
	if snapshot == "" || python == "" {
		t.Skip("set EOS_HF_BERT_PARITY_SNAPSHOT and EOS_HF_BERT_PARITY_PYTHON to run real HF parity")
	}

	texts := []string{
		"A neural retriever ranks scientific abstracts.",
		"The database stores compact vector embeddings.",
	}
	const maxLength = 32

	py, err := runPythonBERTParityFixture(t, python, snapshot, texts, maxLength)
	if err != nil {
		t.Fatalf("python parity fixture: %v", err)
	}

	tokenizer, err := LoadHFWordPieceTokenizerFromDir(snapshot)
	if err != nil {
		t.Fatalf("load Go tokenizer: %v", err)
	}
	goInputIDs, goAttentionMask, goTokenTypeIDs := encodeBERTParityTexts(t, tokenizer, texts, maxLength)
	assertInt32SliceEqual(t, goInputIDs, py.InputIDs)
	assertInt32SliceEqual(t, goAttentionMask, py.AttentionMask)
	assertInt32SliceEqual(t, goTokenTypeIDs, py.TokenTypeIDs)

	plan, err := PlanPretrainedBERTImportFromDir(snapshot, parityModelLabel())
	if err != nil {
		t.Fatalf("plan import: %v", err)
	}
	decoded, decodeReport, err := LoadPretrainedBERTDecodedWeightsFromDir(snapshot, plan)
	if err != nil {
		t.Fatalf("decode safetensors: %v", err)
	}
	if decodeReport.Status != "ok" {
		t.Fatalf("decode report status = %q", decodeReport.Status)
	}
	weights, weightReport, err := BuildPretrainedBERTWeightFileFromDecoded(decoded)
	if err != nil {
		t.Fatalf("build weight file: %v", err)
	}
	if weightReport.Status != "ok" {
		t.Fatalf("weight report status = %q", weightReport.Status)
	}
	mod, err := BuildPretrainedBERTEmbedderModule(plan)
	if err != nil {
		t.Fatalf("build embedder module: %v", err)
	}
	rt := New(bertEmbeddingHostBackend{})
	prog, err := rt.Load(context.Background(), mod, weights.LoadOptions()...)
	if err != nil {
		t.Fatalf("load embedder module: %v", err)
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "bert_embed",
		Inputs: map[string]any{
			"input_ids":      backend.NewTensorI32([]int{len(texts), maxLength}, goInputIDs),
			"attention_mask": backend.NewTensorI32([]int{len(texts), maxLength}, goAttentionMask),
			"token_type_ids": backend.NewTensorI32([]int{len(texts), maxLength}, goTokenTypeIDs),
		},
	})
	if err != nil {
		t.Fatalf("run Go embedder: %v", err)
	}
	value, ok := result.Outputs["embeddings"]
	if !ok {
		t.Fatalf("missing embeddings output: %+v", result.Outputs)
	}
	got, ok := value.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("embedding output data type = %T, want *backend.Tensor", value.Data)
	}
	want := flattenPythonEmbeddings(py.Embeddings)
	if gotLen, wantLen := len(got.F32), len(want); gotLen != wantLen {
		t.Fatalf("embedding value count = %d, want %d", gotLen, wantLen)
	}
	minCos, maxAbs := embeddingParityStats(got.F32, want, len(texts))
	t.Logf("BERT HF parity: min_cosine=%.8f max_abs=%.8g tensors=%d", minCos, maxAbs, decodeReport.TensorCount)
	if minCos < 0.9999 || maxAbs > 2e-3 {
		t.Fatalf("embedding parity mismatch: min_cosine=%.8f want >= 0.9999; max_abs=%.8g want <= 0.002", minCos, maxAbs)
	}
}

type pythonBERTParityFixture struct {
	InputIDs      []int32     `json:"input_ids"`
	AttentionMask []int32     `json:"attention_mask"`
	TokenTypeIDs  []int32     `json:"token_type_ids"`
	Embeddings    [][]float32 `json:"embeddings"`
}

func runPythonBERTParityFixture(t *testing.T, python, snapshot string, texts []string, maxLength int) (pythonBERTParityFixture, error) {
	t.Helper()
	textsJSON, err := json.Marshal(texts)
	if err != nil {
		return pythonBERTParityFixture{}, err
	}
	script := `
import json
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

snapshot = sys.argv[1]
max_length = int(sys.argv[2])
texts = json.loads(sys.argv[3])
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
model = AutoModel.from_pretrained(snapshot, local_files_only=True)
model.eval()
encoded = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
    return_token_type_ids=True,
)
if "token_type_ids" not in encoded:
    encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
with torch.no_grad():
    output = model(**encoded)
mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
summed = (output.last_hidden_state * mask).sum(dim=1)
counts = mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(summed / counts, p=2, dim=1)
payload = {
    "input_ids": encoded["input_ids"].reshape(-1).tolist(),
    "attention_mask": encoded["attention_mask"].reshape(-1).tolist(),
    "token_type_ids": encoded["token_type_ids"].reshape(-1).tolist(),
    "embeddings": embeddings.tolist(),
}
print(json.dumps(payload, separators=(",", ":")))
`
	cmd := exec.Command(python, "-c", script, snapshot, strconv.Itoa(maxLength), string(textsJSON))
	cmd.Env = append(os.Environ(),
		"HF_HUB_OFFLINE=1",
		"TRANSFORMERS_OFFLINE=1",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return pythonBERTParityFixture{}, fmt.Errorf("%w: %s", err, stderr.String())
	}
	var fixture pythonBERTParityFixture
	if err := json.Unmarshal(stdout.Bytes(), &fixture); err != nil {
		return pythonBERTParityFixture{}, fmt.Errorf("parse python JSON: %w: stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	return fixture, nil
}

func encodeBERTParityTexts(t *testing.T, tokenizer *HFWordPieceTokenizer, texts []string, maxLength int) ([]int32, []int32, []int32) {
	t.Helper()
	inputIDs := make([]int32, 0, len(texts)*maxLength)
	attentionMask := make([]int32, 0, len(texts)*maxLength)
	tokenTypeIDs := make([]int32, 0, len(texts)*maxLength)
	for _, text := range texts {
		encoded, err := tokenizer.Encode(text, HFWordPieceEncodeOptions{MaxLength: maxLength, PadToMaxLength: true})
		if err != nil {
			t.Fatalf("encode %q: %v", text, err)
		}
		inputIDs = append(inputIDs, encoded.IDs...)
		attentionMask = append(attentionMask, encoded.AttentionMask...)
		tokenTypeIDs = append(tokenTypeIDs, encoded.TokenTypeIDs...)
	}
	return inputIDs, attentionMask, tokenTypeIDs
}

func flattenPythonEmbeddings(rows [][]float32) []float32 {
	var total int
	for _, row := range rows {
		total += len(row)
	}
	out := make([]float32, 0, total)
	for _, row := range rows {
		out = append(out, row...)
	}
	return out
}

func embeddingParityStats(got, want []float32, rows int) (float64, float64) {
	rowWidth := len(got)
	if rows > 0 {
		rowWidth /= rows
	}
	minCos := 1.0
	maxAbs := 0.0
	for row := 0; row < rows; row++ {
		start := row * rowWidth
		end := start + rowWidth
		var dot, gotNorm, wantNorm float64
		for i := start; i < end; i++ {
			g := float64(got[i])
			w := float64(want[i])
			dot += g * w
			gotNorm += g * g
			wantNorm += w * w
			if diff := math.Abs(g - w); diff > maxAbs {
				maxAbs = diff
			}
		}
		cos := 0.0
		if gotNorm > 0 && wantNorm > 0 {
			cos = dot / (math.Sqrt(gotNorm) * math.Sqrt(wantNorm))
		}
		if cos < minCos {
			minCos = cos
		}
	}
	return minCos, maxAbs
}

func parityModelLabel() string {
	if label := os.Getenv("EOS_HF_BERT_PARITY_MODEL"); label != "" {
		return label
	}
	return "BAAI/bge-small-en-v1.5"
}
