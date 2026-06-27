package eosruntime

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

const defaultWordPieceMaxInputCharsPerWord = 100

// HFWordPieceTokenizerConfig captures the BERT tokenizer behavior needed by
// common BGE/E5 Hugging Face snapshots.
type HFWordPieceTokenizerConfig struct {
	DoLowerCase          bool
	StripAccents         bool
	TokenizeChineseChar  bool
	ModelMaxLength       int
	MaxInputCharsPerWord int
	ContinuationPrefix   string
	PadToken             string
	UnknownToken         string
	CLSToken             string
	SEPToken             string
	MaskToken            string
}

// HFWordPieceTokenizer is a small BERT BasicTokenizer + WordPiece tokenizer.
type HFWordPieceTokenizer struct {
	vocab     []string
	tokenToID map[string]int32
	config    HFWordPieceTokenizerConfig
}

// HFWordPieceEncodeOptions controls truncation and padding for Encode calls.
type HFWordPieceEncodeOptions struct {
	MaxLength      int
	PadToMaxLength bool
}

// HFWordPieceEncoding mirrors the token ids and masks emitted by BERT-family
// Hugging Face tokenizers for a single sequence or sequence pair.
type HFWordPieceEncoding struct {
	Tokens        []string
	IDs           []int32
	AttentionMask []int32
	TokenTypeIDs  []int32
}

func DefaultHFWordPieceTokenizerConfig() HFWordPieceTokenizerConfig {
	return HFWordPieceTokenizerConfig{
		DoLowerCase:          true,
		StripAccents:         true,
		TokenizeChineseChar:  true,
		MaxInputCharsPerWord: defaultWordPieceMaxInputCharsPerWord,
		ContinuationPrefix:   "##",
		PadToken:             "[PAD]",
		UnknownToken:         "[UNK]",
		CLSToken:             "[CLS]",
		SEPToken:             "[SEP]",
		MaskToken:            "[MASK]",
	}
}

func LoadHFWordPieceTokenizerFromDir(dir string) (*HFWordPieceTokenizer, error) {
	cfg := DefaultHFWordPieceTokenizerConfig()
	if err := applyHFTokenizerConfig(filepath.Join(dir, "tokenizer_config.json"), &cfg); err != nil {
		return nil, err
	}
	if err := applyHFSpecialTokensMap(filepath.Join(dir, "special_tokens_map.json"), &cfg); err != nil {
		return nil, err
	}
	return NewHFWordPieceTokenizerFromVocabFile(filepath.Join(dir, "vocab.txt"), cfg)
}

func LoadHFWordPieceTokenizerFromBytes(vocab, tokenizerConfig, specialTokensMap []byte) (*HFWordPieceTokenizer, error) {
	cfg := DefaultHFWordPieceTokenizerConfig()
	if len(tokenizerConfig) > 0 {
		if err := applyHFTokenizerConfigBytes("tokenizer_config.json", tokenizerConfig, &cfg); err != nil {
			return nil, err
		}
	}
	if len(specialTokensMap) > 0 {
		if err := applyHFSpecialTokensMapBytes("special_tokens_map.json", specialTokensMap, &cfg); err != nil {
			return nil, err
		}
	}
	return NewHFWordPieceTokenizerFromVocabBytes(vocab, cfg)
}

func NewHFWordPieceTokenizerFromVocabFile(path string, cfg HFWordPieceTokenizerConfig) (*HFWordPieceTokenizer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return newHFWordPieceTokenizerFromScanner(path, bufio.NewScanner(file), cfg)
}

func NewHFWordPieceTokenizerFromVocabBytes(data []byte, cfg HFWordPieceTokenizerConfig) (*HFWordPieceTokenizer, error) {
	return newHFWordPieceTokenizerFromScanner("vocab.txt", bufio.NewScanner(bytes.NewReader(data)), cfg)
}

func newHFWordPieceTokenizerFromScanner(path string, scanner *bufio.Scanner, cfg HFWordPieceTokenizerConfig) (*HFWordPieceTokenizer, error) {
	var vocab []string
	for scanner.Scan() {
		token := strings.TrimRight(scanner.Text(), "\r")
		if token == "" {
			return nil, fmt.Errorf("vocab %q contains an empty token at line %d", path, len(vocab)+1)
		}
		vocab = append(vocab, token)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return NewHFWordPieceTokenizer(vocab, cfg)
}

func NewHFWordPieceTokenizer(vocab []string, cfg HFWordPieceTokenizerConfig) (*HFWordPieceTokenizer, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("wordpiece vocab is empty")
	}
	cfg = normalizeHFWordPieceConfig(cfg)
	tokenToID := make(map[string]int32, len(vocab))
	for i, token := range vocab {
		if token == "" {
			return nil, fmt.Errorf("wordpiece vocab token %d is empty", i)
		}
		if _, exists := tokenToID[token]; exists {
			return nil, fmt.Errorf("duplicate wordpiece token %q", token)
		}
		tokenToID[token] = int32(i)
	}
	for _, token := range []string{cfg.PadToken, cfg.UnknownToken, cfg.CLSToken, cfg.SEPToken} {
		if _, ok := tokenToID[token]; !ok {
			return nil, fmt.Errorf("wordpiece vocab missing required special token %q", token)
		}
	}
	return &HFWordPieceTokenizer{
		vocab:     append([]string(nil), vocab...),
		tokenToID: tokenToID,
		config:    cfg,
	}, nil
}

func (t *HFWordPieceTokenizer) VocabSize() int {
	if t == nil {
		return 0
	}
	return len(t.vocab)
}

func (t *HFWordPieceTokenizer) TokenID(token string) (int32, bool) {
	if t == nil {
		return 0, false
	}
	id, ok := t.tokenToID[token]
	return id, ok
}

func (t *HFWordPieceTokenizer) Config() HFWordPieceTokenizerConfig {
	if t == nil {
		return HFWordPieceTokenizerConfig{}
	}
	return t.config
}

func (t *HFWordPieceTokenizer) Encode(text string, opts HFWordPieceEncodeOptions) (HFWordPieceEncoding, error) {
	return t.EncodePair(text, "", opts)
}

func (t *HFWordPieceTokenizer) EncodePair(text, pair string, opts HFWordPieceEncodeOptions) (HFWordPieceEncoding, error) {
	if t == nil {
		return HFWordPieceEncoding{}, fmt.Errorf("nil wordpiece tokenizer")
	}
	first, err := t.tokenize(text)
	if err != nil {
		return HFWordPieceEncoding{}, err
	}
	var second []string
	if pair != "" {
		second, err = t.tokenize(pair)
		if err != nil {
			return HFWordPieceEncoding{}, err
		}
	}
	maxLen := opts.MaxLength
	if maxLen <= 0 {
		maxLen = t.config.ModelMaxLength
	}
	if maxLen > 0 {
		numSpecial := 2
		if pair != "" {
			numSpecial = 3
		}
		if maxLen < numSpecial {
			return HFWordPieceEncoding{}, fmt.Errorf("max length %d is too small for %d special tokens", maxLen, numSpecial)
		}
		truncateWordPiecePair(&first, &second, maxLen-numSpecial)
	}

	tokens := make([]string, 0, len(first)+len(second)+3)
	tokenTypes := make([]int32, 0, cap(tokens))
	tokens = append(tokens, t.config.CLSToken)
	tokenTypes = append(tokenTypes, 0)
	for _, token := range first {
		tokens = append(tokens, token)
		tokenTypes = append(tokenTypes, 0)
	}
	tokens = append(tokens, t.config.SEPToken)
	tokenTypes = append(tokenTypes, 0)
	if pair != "" {
		for _, token := range second {
			tokens = append(tokens, token)
			tokenTypes = append(tokenTypes, 1)
		}
		tokens = append(tokens, t.config.SEPToken)
		tokenTypes = append(tokenTypes, 1)
	}

	ids := make([]int32, len(tokens))
	for i, token := range tokens {
		id, ok := t.tokenToID[token]
		if !ok {
			return HFWordPieceEncoding{}, fmt.Errorf("token %q is not in wordpiece vocabulary", token)
		}
		ids[i] = id
	}
	mask := make([]int32, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	if opts.PadToMaxLength {
		if maxLen <= 0 {
			return HFWordPieceEncoding{}, fmt.Errorf("padding requires a positive max length")
		}
		padID := t.tokenToID[t.config.PadToken]
		for len(ids) < maxLen {
			tokens = append(tokens, t.config.PadToken)
			ids = append(ids, padID)
			mask = append(mask, 0)
			tokenTypes = append(tokenTypes, 0)
		}
	}
	return HFWordPieceEncoding{
		Tokens:        tokens,
		IDs:           ids,
		AttentionMask: mask,
		TokenTypeIDs:  tokenTypes,
	}, nil
}

func (t *HFWordPieceTokenizer) tokenize(text string) ([]string, error) {
	basic := basicWordPieceTokens(text, t.config)
	out := make([]string, 0, len(basic))
	for _, token := range basic {
		pieces := t.wordPiece(token)
		out = append(out, pieces...)
	}
	return out, nil
}

func (t *HFWordPieceTokenizer) wordPiece(token string) []string {
	if token == "" {
		return nil
	}
	if len([]rune(token)) > t.config.MaxInputCharsPerWord {
		return []string{t.config.UnknownToken}
	}
	runes := []rune(token)
	pieces := make([]string, 0, len(runes))
	for start := 0; start < len(runes); {
		end := len(runes)
		var current string
		for start < end {
			sub := string(runes[start:end])
			if start > 0 {
				sub = t.config.ContinuationPrefix + sub
			}
			if _, ok := t.tokenToID[sub]; ok {
				current = sub
				break
			}
			end--
		}
		if current == "" {
			return []string{t.config.UnknownToken}
		}
		pieces = append(pieces, current)
		start = end
	}
	return pieces
}

func basicWordPieceTokens(text string, cfg HFWordPieceTokenizerConfig) []string {
	text = cleanHFTokenizerText(text)
	if cfg.TokenizeChineseChar {
		text = spaceChineseChars(text)
	}
	words := strings.Fields(text)
	out := make([]string, 0, len(words))
	for _, word := range words {
		if cfg.DoLowerCase {
			word = strings.ToLower(word)
			if cfg.StripAccents {
				word = stripUnicodeAccents(word)
			}
		} else if cfg.StripAccents {
			word = stripUnicodeAccents(word)
		}
		out = append(out, splitOnPunctuation(word)...)
	}
	return out
}

func normalizeHFWordPieceConfig(cfg HFWordPieceTokenizerConfig) HFWordPieceTokenizerConfig {
	defaults := DefaultHFWordPieceTokenizerConfig()
	if cfg.ContinuationPrefix == "" {
		cfg.ContinuationPrefix = defaults.ContinuationPrefix
	}
	if cfg.MaxInputCharsPerWord <= 0 {
		cfg.MaxInputCharsPerWord = defaults.MaxInputCharsPerWord
	}
	if cfg.PadToken == "" {
		cfg.PadToken = defaults.PadToken
	}
	if cfg.UnknownToken == "" {
		cfg.UnknownToken = defaults.UnknownToken
	}
	if cfg.CLSToken == "" {
		cfg.CLSToken = defaults.CLSToken
	}
	if cfg.SEPToken == "" {
		cfg.SEPToken = defaults.SEPToken
	}
	if cfg.MaskToken == "" {
		cfg.MaskToken = defaults.MaskToken
	}
	return cfg
}

func truncateWordPiecePair(first, second *[]string, budget int) {
	for len(*first)+len(*second) > budget {
		if len(*second) > len(*first) {
			*second = (*second)[:len(*second)-1]
		} else {
			*first = (*first)[:len(*first)-1]
		}
	}
}

func cleanHFTokenizerText(text string) string {
	var b strings.Builder
	for _, r := range text {
		if r == 0 || r == unicode.ReplacementChar || unicode.IsControl(r) && !isHFWhitespace(r) {
			continue
		}
		if isHFWhitespace(r) {
			b.WriteByte(' ')
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func spaceChineseChars(text string) string {
	var b strings.Builder
	for _, r := range text {
		if isChineseChar(r) {
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func stripUnicodeAccents(text string) string {
	text = stripLatin1Accents(text)
	var b strings.Builder
	for _, r := range text {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func stripLatin1Accents(text string) string {
	replacer := strings.NewReplacer(
		"à", "a", "á", "a", "â", "a", "ã", "a", "ä", "a", "å", "a", "ā", "a",
		"è", "e", "é", "e", "ê", "e", "ë", "e", "ē", "e",
		"ì", "i", "í", "i", "î", "i", "ï", "i", "ī", "i",
		"ò", "o", "ó", "o", "ô", "o", "õ", "o", "ö", "o", "ō", "o",
		"ù", "u", "ú", "u", "û", "u", "ü", "u", "ū", "u",
		"ç", "c", "ñ", "n",
		"À", "A", "Á", "A", "Â", "A", "Ã", "A", "Ä", "A", "Å", "A", "Ā", "A",
		"È", "E", "É", "E", "Ê", "E", "Ë", "E", "Ē", "E",
		"Ì", "I", "Í", "I", "Î", "I", "Ï", "I", "Ī", "I",
		"Ò", "O", "Ó", "O", "Ô", "O", "Õ", "O", "Ö", "O", "Ō", "O",
		"Ù", "U", "Ú", "U", "Û", "U", "Ü", "U", "Ū", "U",
		"Ç", "C", "Ñ", "N",
	)
	return replacer.Replace(text)
}

func splitOnPunctuation(text string) []string {
	var out []string
	var current strings.Builder
	flush := func() {
		if current.Len() > 0 {
			out = append(out, current.String())
			current.Reset()
		}
	}
	for _, r := range text {
		if isHFPunctuation(r) {
			flush()
			out = append(out, string(r))
			continue
		}
		current.WriteRune(r)
	}
	flush()
	return out
}

func isHFWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || unicode.IsSpace(r)
}

func isHFPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

func isChineseChar(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}

func applyHFTokenizerConfig(path string, cfg *HFWordPieceTokenizerConfig) error {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	return applyHFTokenizerConfigBytes(path, data, cfg)
}

func applyHFTokenizerConfigBytes(source string, data []byte, cfg *HFWordPieceTokenizerConfig) error {
	var raw struct {
		DoLowerCase          *bool `json:"do_lower_case"`
		StripAccents         *bool `json:"strip_accents"`
		TokenizeChineseChars *bool `json:"tokenize_chinese_chars"`
		ModelMaxLength       int   `json:"model_max_length"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("parse %s: %w", source, err)
	}
	if raw.DoLowerCase != nil {
		cfg.DoLowerCase = *raw.DoLowerCase
	}
	if raw.StripAccents != nil {
		cfg.StripAccents = *raw.StripAccents
	}
	if raw.TokenizeChineseChars != nil {
		cfg.TokenizeChineseChar = *raw.TokenizeChineseChars
	}
	if raw.ModelMaxLength > 0 && raw.ModelMaxLength < 1_000_000_000 {
		cfg.ModelMaxLength = raw.ModelMaxLength
	}
	return nil
}

func applyHFSpecialTokensMap(path string, cfg *HFWordPieceTokenizerConfig) error {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	return applyHFSpecialTokensMapBytes(path, data, cfg)
}

func applyHFSpecialTokensMapBytes(source string, data []byte, cfg *HFWordPieceTokenizerConfig) error {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("parse %s: %w", source, err)
	}
	if value := specialTokenString(raw["pad_token"]); value != "" {
		cfg.PadToken = value
	}
	if value := specialTokenString(raw["unk_token"]); value != "" {
		cfg.UnknownToken = value
	}
	if value := specialTokenString(raw["cls_token"]); value != "" {
		cfg.CLSToken = value
	}
	if value := specialTokenString(raw["sep_token"]); value != "" {
		cfg.SEPToken = value
	}
	if value := specialTokenString(raw["mask_token"]); value != "" {
		cfg.MaskToken = value
	}
	return nil
}

func specialTokenString(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case map[string]any:
		if content, ok := typed["content"].(string); ok {
			return content
		}
	}
	return ""
}
