package eosruntime

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
)

const CompactHardNegativeManifestSchema = "manta.embedding_compact_hard_negative_manifest.v1"

type CompactHardNegativeMiningConfig struct {
	DatasetName       string
	Split             string
	CorpusPath        string
	QueriesPath       string
	QrelsPath         string
	PerQueryJSONLPath string
	OutputPath        string
	ManifestPath      string
	Method            string
	BitWidth          int
	Overfetch         int
	RerankStorage     string
	QuantizerSeed     int64
	TrainSelection    bool
	AllowTestSmoke    bool
	NegativesPerRow   int
	MaxExamples       int
	MaxDocs           int
	MaxQueries        int
	ArtifactSHA256    string
}

type CompactHardNegativeMiningManifest struct {
	Schema                   string         `json:"schema"`
	Dataset                  string         `json:"dataset"`
	Split                    string         `json:"split"`
	TrainSelection           bool           `json:"train_selection"`
	TrainAllowed             bool           `json:"train_allowed"`
	LeakGuardStatus          string         `json:"leak_guard_status"`
	CorpusPath               string         `json:"corpus_path"`
	QueriesPath              string         `json:"queries_path"`
	QrelsSource              string         `json:"qrels_source"`
	PerQueryJSONLPath        string         `json:"per_query_jsonl_path"`
	OutputPath               string         `json:"output_path"`
	Method                   string         `json:"method"`
	BitWidth                 int            `json:"bit_width"`
	Overfetch                int            `json:"overfetch,omitempty"`
	RerankStorage            string         `json:"rerank_storage,omitempty"`
	QuantizerSeed            int64          `json:"quantizer_seed"`
	ArtifactSHA256           string         `json:"artifact_sha256,omitempty"`
	PerQuerySHA256           string         `json:"per_query_sha256"`
	HardNegativesSHA256      string         `json:"hard_negatives_sha256"`
	RowsRead                 int            `json:"rows_read"`
	RowsMatched              int            `json:"rows_matched"`
	RowsEmitted              int            `json:"rows_emitted"`
	Queries                  int            `json:"queries"`
	PositivePairs            int            `json:"positive_pairs"`
	Negatives                int            `json:"negatives"`
	SkippedNoQueryText       int            `json:"skipped_no_query_text"`
	SkippedNoPositive        int            `json:"skipped_no_positive"`
	SkippedNoNegative        int            `json:"skipped_no_negative"`
	QrelsRelevanceMismatches int            `json:"qrels_relevance_mismatches,omitempty"`
	ReasonCounts             map[string]int `json:"reason_counts"`
}

// MineCompactTextHardNegatives consumes compact per-query diagnostics and emits
// existing text hard-negative rows with a manifest that records leak-guard state.
func MineCompactTextHardNegatives(ctx context.Context, cfg CompactHardNegativeMiningConfig) (CompactHardNegativeMiningManifest, error) {
	cfg = normalizeCompactHardNegativeMiningConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" || cfg.PerQueryJSONLPath == "" || cfg.OutputPath == "" {
		return CompactHardNegativeMiningManifest{}, fmt.Errorf("corpus, queries, qrels, per-query JSONL, and output path are required")
	}
	trainAllowed, guardStatus, err := compactMiningLeakGuard(cfg)
	if err != nil {
		return CompactHardNegativeMiningManifest{}, err
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return CompactHardNegativeMiningManifest{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return CompactHardNegativeMiningManifest{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return CompactHardNegativeMiningManifest{}, err
	}
	queryText := make(map[string]string, len(queries))
	for _, query := range queries {
		queryText[query.ID] = query.Text
	}
	docText := make(map[string]string, len(corpus))
	for _, doc := range corpus {
		docText[doc.ID] = doc.Text
	}
	rows, rowsRead, err := readCompactMiningPerQueryRows(cfg.PerQueryJSONLPath)
	if err != nil {
		return CompactHardNegativeMiningManifest{}, err
	}
	out := make([]EmbeddingTextHardNegativeExample, 0)
	manifest := CompactHardNegativeMiningManifest{
		Schema:             CompactHardNegativeManifestSchema,
		Dataset:            cfg.DatasetName,
		Split:              cfg.Split,
		TrainSelection:     cfg.TrainSelection,
		TrainAllowed:       trainAllowed,
		LeakGuardStatus:    guardStatus,
		CorpusPath:         cfg.CorpusPath,
		QueriesPath:        cfg.QueriesPath,
		QrelsSource:        cfg.QrelsPath,
		PerQueryJSONLPath:  cfg.PerQueryJSONLPath,
		OutputPath:         cfg.OutputPath,
		Method:             cfg.Method,
		BitWidth:           cfg.BitWidth,
		Overfetch:          cfg.Overfetch,
		RerankStorage:      cfg.RerankStorage,
		QuantizerSeed:      cfg.QuantizerSeed,
		ArtifactSHA256:     cfg.ArtifactSHA256,
		RowsRead:           rowsRead,
		ReasonCounts:       map[string]int{},
		SkippedNoQueryText: skippedQueries,
	}
	for _, row := range rows {
		if err := ctx.Err(); err != nil {
			return CompactHardNegativeMiningManifest{}, err
		}
		matched, err := compactMiningRowMatches(row, cfg)
		if err != nil {
			return manifest, err
		}
		if !matched {
			continue
		}
		manifest.RowsMatched++
		manifest.QrelsRelevanceMismatches += compactMiningQrelsRelevanceMismatches(row, qrels[row.QueryID])
		query := strings.TrimSpace(queryText[row.QueryID])
		if query == "" {
			manifest.SkippedNoQueryText++
			continue
		}
		positives, skippedPositive := bm25MiningPositiveDocs(qrels[row.QueryID], docText)
		manifest.SkippedNoPositive += skippedPositive
		if len(positives) == 0 {
			manifest.SkippedNoPositive++
			continue
		}
		negativeIDs, reason := compactMiningNegativeIDs(row, qrels[row.QueryID], cfg.NegativesPerRow)
		if len(negativeIDs) == 0 {
			manifest.SkippedNoNegative++
			continue
		}
		negatives := make([]string, 0, len(negativeIDs))
		seenText := map[string]bool{}
		for _, id := range negativeIDs {
			text := strings.TrimSpace(docText[id])
			if text == "" || seenText[text] {
				continue
			}
			seenText[text] = true
			negatives = append(negatives, text)
		}
		if len(negatives) == 0 {
			manifest.SkippedNoNegative++
			continue
		}
		for _, positive := range positives {
			if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
				break
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Source:    fmt.Sprintf("%s:%s:%s:%s:%s", cfg.DatasetName, cfg.Split, row.QueryID, positive.ID, reason),
				Query:     query,
				Positive:  positive.Text,
				Negatives: append([]string(nil), negatives...),
			})
			manifest.ReasonCounts[reason]++
			manifest.PositivePairs++
			manifest.Negatives += len(negatives)
		}
		if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
			break
		}
	}
	manifest.RowsEmitted = len(out)
	manifest.Queries = countCompactMiningQueries(out)
	if len(out) == 0 {
		return manifest, fmt.Errorf("compact hard-negative mining produced no examples")
	}
	if err := WriteEmbeddingTextHardNegativeExamplesFile(cfg.OutputPath, out); err != nil {
		return manifest, err
	}
	perQueryHash, err := fileSHA256Hex(cfg.PerQueryJSONLPath)
	if err != nil {
		return manifest, err
	}
	outputHash, err := fileSHA256Hex(cfg.OutputPath)
	if err != nil {
		return manifest, err
	}
	manifest.PerQuerySHA256 = perQueryHash
	manifest.HardNegativesSHA256 = outputHash
	if cfg.ManifestPath != "" {
		data, err := json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			return manifest, err
		}
		data = append(data, '\n')
		if err := os.WriteFile(cfg.ManifestPath, data, 0o644); err != nil {
			return manifest, err
		}
	}
	return manifest, nil
}

func normalizeCompactHardNegativeMiningConfig(cfg CompactHardNegativeMiningConfig) CompactHardNegativeMiningConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.Split == "" {
		cfg.Split = "test"
	}
	if cfg.BitWidth <= 0 {
		cfg.BitWidth = 4
	}
	if cfg.RerankStorage == "" && cfg.Overfetch > 0 {
		cfg.RerankStorage = TurboQuantRerankStorageFP16
	}
	if cfg.Method == "" {
		cfg.Method = turboQuantRetrievalMethodName(cfg.BitWidth, cfg.Overfetch, cfg.RerankStorage)
	}
	if cfg.QuantizerSeed == 0 {
		cfg.QuantizerSeed = DefaultTurboQuantMultiVectorQuantizerSeed
	}
	if cfg.NegativesPerRow <= 0 {
		cfg.NegativesPerRow = 4
	}
	return cfg
}

func compactMiningLeakGuard(cfg CompactHardNegativeMiningConfig) (bool, string, error) {
	split := strings.ToLower(strings.TrimSpace(cfg.Split))
	if cfg.TrainSelection && split == "test" {
		return false, "blocked_test_split_train_selection", fmt.Errorf("refusing to mine train-selection rows from test split; rerun with --train-selection=false for validation/no-train smoke")
	}
	if !cfg.TrainSelection {
		if split == "test" {
			return false, "validation_smoke_no_train_test_split", nil
		}
		return false, "validation_smoke_no_train", nil
	}
	return true, "train_selection_non_test_split", nil
}

func readCompactMiningPerQueryRows(path string) ([]TurboQuantRetrievalPerQueryRow, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	rows := []TurboQuantRetrievalPerQueryRow{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row TurboQuantRetrievalPerQueryRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, lineNo, fmt.Errorf("per-query line %d: %w", lineNo, err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, lineNo, err
	}
	return rows, lineNo, nil
}

func compactMiningRowMatches(row TurboQuantRetrievalPerQueryRow, cfg CompactHardNegativeMiningConfig) (bool, error) {
	if row.Schema != TurboQuantRetrievalPerQuerySchema {
		return false, nil
	}
	if cfg.Method != "" && row.Method != cfg.Method {
		return false, nil
	}
	if cfg.BitWidth > 0 && row.Bits != cfg.BitWidth {
		return false, nil
	}
	if cfg.Overfetch > 0 && row.RerankOverfetch != cfg.Overfetch {
		return false, nil
	}
	if cfg.RerankStorage != "" && row.RerankStorage != "" && row.RerankStorage != cfg.RerankStorage {
		return false, nil
	}
	if cfg.QuantizerSeed != 0 && row.QuantizerSeed != cfg.QuantizerSeed {
		return false, fmt.Errorf("per-query row quantizer seed mismatch for query %q method %q: row=%d configured=%d", row.QueryID, row.Method, row.QuantizerSeed, cfg.QuantizerSeed)
	}
	return true, nil
}

func compactMiningNegativeIDs(row TurboQuantRetrievalPerQueryRow, positiveRels map[string]float64, limit int) ([]string, string) {
	type candidate struct {
		id     string
		rank   int
		reason string
	}
	candidates := []candidate{}
	for _, doc := range row.TopK {
		if doc.Relevance > 0 || positiveRels[doc.DocID] > 0 {
			continue
		}
		reason := "compact_candidate"
		if doc.Rank <= 10 {
			reason = "top10_competitor"
		} else if row.RerankOverfetch > 0 && doc.CompactRank != nil && *doc.CompactRank >= row.RerankOverfetch-10 {
			reason = "rank_boundary"
		}
		candidates = append(candidates, candidate{id: doc.DocID, rank: doc.Rank, reason: reason})
	}
	slices.SortFunc(candidates, func(a, b candidate) int {
		if a.rank != b.rank {
			return a.rank - b.rank
		}
		return strings.Compare(a.id, b.id)
	})
	if limit <= 0 || limit > len(candidates) {
		limit = len(candidates)
	}
	out := make([]string, 0, limit)
	reason := "compact_candidate"
	for i := 0; i < limit; i++ {
		if i == 0 {
			reason = candidates[i].reason
		}
		out = append(out, candidates[i].id)
	}
	return out, reason
}

func compactMiningQrelsRelevanceMismatches(row TurboQuantRetrievalPerQueryRow, positiveRels map[string]float64) int {
	mismatches := 0
	for _, doc := range row.TopK {
		if positiveRels[doc.DocID] > 0 && doc.Relevance <= 0 {
			mismatches++
		}
	}
	return mismatches
}

func countCompactMiningQueries(rows []EmbeddingTextHardNegativeExample) int {
	seen := map[string]bool{}
	for _, row := range rows {
		seen[row.Query] = true
	}
	return len(seen)
}

func fileSHA256Hex(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
