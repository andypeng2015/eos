package eosruntime

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestHFWordPieceTokenizerFixtureParity(t *testing.T) {
	tokenizer := newTestHFWordPieceTokenizer(t)
	got, err := tokenizer.Encode("Unaffable Café, world!", HFWordPieceEncodeOptions{MaxLength: 12, PadToMaxLength: true})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	wantTokens := []string{"[CLS]", "una", "##ffa", "##ble", "cafe", ",", "world", "!", "[SEP]", "[PAD]", "[PAD]", "[PAD]"}
	wantIDs := []int32{2, 5, 6, 7, 8, 9, 10, 11, 3, 0, 0, 0}
	wantMask := []int32{1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}
	wantTypes := []int32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	assertStringSliceEqual(t, got.Tokens, wantTokens)
	assertInt32SliceEqual(t, got.IDs, wantIDs)
	assertInt32SliceEqual(t, got.AttentionMask, wantMask)
	assertInt32SliceEqual(t, got.TokenTypeIDs, wantTypes)
}

func TestHFWordPieceTokenizerPairTruncationTokenTypes(t *testing.T) {
	tokenizer := newTestHFWordPieceTokenizer(t)
	got, err := tokenizer.EncodePair("hello world unaffable", "cafe world", HFWordPieceEncodeOptions{MaxLength: 9})
	if err != nil {
		t.Fatalf("encode pair: %v", err)
	}
	assertStringSliceEqual(t, got.Tokens, []string{"[CLS]", "hello", "world", "una", "##ffa", "[SEP]", "cafe", "world", "[SEP]"})
	assertInt32SliceEqual(t, got.IDs, []int32{2, 12, 10, 5, 6, 3, 8, 10, 3})
	assertInt32SliceEqual(t, got.AttentionMask, []int32{1, 1, 1, 1, 1, 1, 1, 1, 1})
	assertInt32SliceEqual(t, got.TokenTypeIDs, []int32{0, 0, 0, 0, 0, 0, 1, 1, 1})
}

func TestHFWordPieceTokenizerUnknownAndChinesePunctuation(t *testing.T) {
	tokenizer := newTestHFWordPieceTokenizer(t)
	got, err := tokenizer.Encode("我爱?", HFWordPieceEncodeOptions{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	assertStringSliceEqual(t, got.Tokens, []string{"[CLS]", "我", "爱", "[UNK]", "[SEP]"})
	assertInt32SliceEqual(t, got.IDs, []int32{2, 13, 14, 1, 3})
}

func TestLoadHFWordPieceTokenizerFromDirReadsHFConfig(t *testing.T) {
	dir := t.TempDir()
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nHello\nworld\n"
	if err := os.WriteFile(filepath.Join(dir, "vocab.txt"), []byte(vocab), 0o644); err != nil {
		t.Fatalf("write vocab: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"do_lower_case": false, "model_max_length": 6}`), 0o644); err != nil {
		t.Fatalf("write tokenizer config: %v", err)
	}
	tokenizer, err := LoadHFWordPieceTokenizerFromDir(dir)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	got, err := tokenizer.Encode("Hello world", HFWordPieceEncodeOptions{PadToMaxLength: true})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	assertStringSliceEqual(t, got.Tokens, []string{"[CLS]", "Hello", "world", "[SEP]", "[PAD]", "[PAD]"})
	assertInt32SliceEqual(t, got.IDs, []int32{2, 5, 6, 3, 0, 0})
}

func newTestHFWordPieceTokenizer(t *testing.T) *HFWordPieceTokenizer {
	t.Helper()
	vocab := []string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"una", "##ffa", "##ble", "cafe", ",", "world", "!", "hello", "我", "爱",
	}
	tokenizer, err := NewHFWordPieceTokenizer(vocab, DefaultHFWordPieceTokenizerConfig())
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	return tokenizer
}

func assertStringSliceEqual(t *testing.T, got, want []string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("slice mismatch:\n got: %#v\nwant: %#v", got, want)
	}
}
