package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	eosruntime "m31labs.dev/eos/runtime"
)

type relabelTeacherNegativesManifest struct {
	Schema     string                                       `json:"schema"`
	CreatedUTC string                                       `json:"created_utc"`
	InputJSONL string                                       `json:"input_jsonl"`
	PoolJSONL  string                                       `json:"pool_jsonl,omitempty"`
	OutputJSONL string                                      `json:"output_jsonl"`
	PromoteMin float64                                      `json:"promote_min"`
	PromoteSlack float64                                    `json:"promote_slack"`
	PromoteCap int                                          `json:"promote_cap"`
	NegativeMax float64                                     `json:"negative_max"`
	NegativesPerRow int                                     `json:"negatives_per_row"`
	EmitPairs  bool                                         `json:"emit_pairs"`
	Summary    eosruntime.RelabelTeacherNegativesSummary    `json:"summary"`
}

func runRelabelTeacherNegatives(args []string) error {
	fs := flag.NewFlagSet("relabel-teacher-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	promoteMin := fs.Float64("promote-min", 0.95, "absolute teacher-score floor for promoting a mined negative to a positive row")
	promoteSlack := fs.Float64("promote-slack", 0.05, "promoted candidate must also score at least positive_score-slack")
	promoteCap := fs.Int("promote-cap", 2, "max promoted positives per input example; 0 is unlimited")
	negativeMax := fs.Float64("negative-max", 0.80, "teacher-score ceiling for keeping a candidate as a true negative")
	negativesPerRow := fs.Int("negatives-per-row", 4, "true negatives attached per output row from surviving and pool negatives; 0 attaches none")
	poolPath := fs.String("negatives-file", "", "teacher-scored JSONL whose low-scoring candidates form the per-query true-negative pool")
	sourceSuffix := fs.String("promoted-source-suffix", "promoted", "suffix tag for promoted rows' source")
	defaultSource := fs.String("default-source", "", "source tag applied to input rows that lack one, for example fiqa")
	emit := fs.String("emit", "hard-negatives", "output shape: hard-negatives or pairs")
	stats := fs.Bool("stats", false, "print teacher-score quantiles and exit without writing output")
	manifestPath := fs.String("manifest", "", "manifest path; default is <output>.relabel.manifest.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *stats {
		if fs.NArg() < 1 || fs.Arg(0) == "" {
			return fmt.Errorf("usage: eos relabel-teacher-negatives -stats <scored-hard-negatives.jsonl>")
		}
		examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(fs.Arg(0))
		if err != nil {
			return err
		}
		fmt.Print(eosruntime.FormatTeacherScoreQuantiles(eosruntime.SummarizeTeacherScores(examples)))
		return nil
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos relabel-teacher-negatives [flags] <scored-hard-negatives.jsonl> <output.jsonl>")
	}
	inputPath := fs.Arg(0)
	outputPath := fs.Arg(1)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".relabel.manifest.json"
	}
	var emitPairs bool
	switch *emit {
	case "hard-negatives":
	case "pairs":
		emitPairs = true
	default:
		return fmt.Errorf("emit must be hard-negatives or pairs, got %q", *emit)
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
	if err != nil {
		return err
	}
	if *defaultSource != "" {
		for i := range examples {
			if examples[i].Source == "" {
				examples[i].Source = *defaultSource
			}
		}
	}
	var pool map[string][]eosruntime.RelabelScoredText
	if *poolPath != "" {
		poolExamples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(*poolPath)
		if err != nil {
			return fmt.Errorf("negatives-file: %w", err)
		}
		pool = eosruntime.BuildTeacherNegativePool(poolExamples, float32(*negativeMax))
	}
	cfg := eosruntime.RelabelTeacherNegativesConfig{
		PromoteMin:           float32(*promoteMin),
		PromoteSlack:         float32(*promoteSlack),
		PromoteCap:           *promoteCap,
		NegativeMax:          float32(*negativeMax),
		NegativesPerRow:      *negativesPerRow,
		PromotedSourceSuffix: *sourceSuffix,
		EmitPairs:            emitPairs,
	}
	out, summary, err := eosruntime.RelabelTeacherNegatives(examples, pool, cfg)
	if err != nil {
		return err
	}
	if emitPairs {
		if err := writeRelabeledPairRows(outputPath, out); err != nil {
			return err
		}
	} else {
		kept := out[:0]
		dropped := 0
		for _, row := range out {
			if len(row.Negatives) == 0 {
				dropped++
				continue
			}
			kept = append(kept, row)
		}
		if dropped > 0 {
			fmt.Fprintf(os.Stderr, "warning: dropped %d rows without surviving negatives (hard-negative rows require at least one negative; provide -negatives-file or emit pairs)\n", dropped)
			summary.OutputRows -= dropped
		}
		if len(kept) == 0 {
			return fmt.Errorf("no rows with negatives to write; use -emit pairs or provide -negatives-file")
		}
		if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, kept); err != nil {
			return err
		}
	}
	manifest := relabelTeacherNegativesManifest{
		Schema:          "eos.relabel_teacher_negatives.v1",
		CreatedUTC:      time.Now().UTC().Format(time.RFC3339),
		InputJSONL:      inputPath,
		PoolJSONL:       *poolPath,
		OutputJSONL:     outputPath,
		PromoteMin:      *promoteMin,
		PromoteSlack:    *promoteSlack,
		PromoteCap:      *promoteCap,
		NegativeMax:     *negativeMax,
		NegativesPerRow: *negativesPerRow,
		EmitPairs:       emitPairs,
		Summary:         summary,
	}
	payload, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*manifestPath, append(payload, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Printf("relabeled: input=%d scored=%d promoted=%d true_negatives=%d dropped_ambiguous=%d output_rows=%d (with_negatives=%d, promoted_rows=%d, duplicates_skipped=%d)\n",
		summary.InputExamples, summary.ScoredExamples, summary.Promoted, summary.TrueNegativesKept, summary.DroppedAmbiguous,
		summary.OutputRows, summary.OutputRowsWithNegs, summary.OutputPromotedRows, summary.DuplicateRowsSkipped)
	fmt.Printf("scores: positive_mean=%.3f promoted_mean=%.3f kept_negative_mean=%.3f pool_queries=%d pool_negatives=%d\n",
		summary.MeanPositiveScore, summary.MeanPromotedScore, summary.MeanKeptNegativeScore, summary.PoolQueries, summary.PoolNegatives)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}

// writeRelabeledPairRows writes train-pair JSONL ({source, query, positive})
// matching the shape of processed train files, preserving the source tag the
// blend and weighting machinery key on.
func writeRelabeledPairRows(path string, rows []eosruntime.EmbeddingTextHardNegativeExample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	type pairRecord struct {
		Source   string `json:"source,omitempty"`
		Query    string `json:"query"`
		Positive string `json:"positive"`
	}
	enc := json.NewEncoder(f)
	for i, row := range rows {
		if err := enc.Encode(pairRecord{Source: row.Source, Query: row.Query, Positive: row.Positive}); err != nil {
			return fmt.Errorf("row %d: %w", i, err)
		}
	}
	return nil
}

type sampleCorpusNegativesManifest struct {
	Schema     string                                    `json:"schema"`
	CreatedUTC string                                    `json:"created_utc"`
	DatasetDir string                                    `json:"dataset_dir"`
	Split      string                                    `json:"split"`
	PerQuery   int                                       `json:"per_query"`
	Seed       int64                                     `json:"seed"`
	Source     string                                    `json:"source"`
	OutputJSONL string                                   `json:"output_jsonl"`
	Summary    eosruntime.SampleCorpusNegativesSummary   `json:"summary"`
}

func runSampleCorpusNegatives(args []string) error {
	fs := flag.NewFlagSet("sample-corpus-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	perQuery := fs.Int("per-query", 8, "random non-qrel corpus documents sampled per query")
	seed := fs.Int64("seed", 1, "sampling seed for reproducibility")
	maxQueries := fs.Int("max-queries", 0, "cap sampled queries for smoke runs; 0 samples all")
	source := fs.String("source", "", "source tag for emitted rows, for example fiqa:random")
	manifestPath := fs.String("manifest", "", "manifest path; default is <output>.sample.manifest.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos sample-corpus-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	}
	datasetDir := fs.Arg(0)
	outputPath := fs.Arg(1)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".sample.manifest.json"
	}
	cfg := eosruntime.SampleCorpusNegativesConfig{
		DatasetDir: datasetDir,
		Split:      *split,
		PerQuery:   *perQuery,
		Seed:       *seed,
		MaxQueries: *maxQueries,
		Source:     *source,
	}
	rows, summary, err := eosruntime.SampleCorpusNegatives(cfg)
	if err != nil {
		return err
	}
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, rows); err != nil {
		return err
	}
	manifest := sampleCorpusNegativesManifest{
		Schema:      "eos.sample_corpus_negatives.v1",
		CreatedUTC:  time.Now().UTC().Format(time.RFC3339),
		DatasetDir:  datasetDir,
		Split:       *split,
		PerQuery:    *perQuery,
		Seed:        *seed,
		Source:      *source,
		OutputJSONL: outputPath,
		Summary:     summary,
	}
	payload, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*manifestPath, append(payload, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Printf("sampled corpus negatives: corpus_docs=%d qrels_queries=%d sampled_queries=%d emitted_negatives=%d skipped_no_positive=%d\n",
		summary.CorpusDocuments, summary.QrelsQueries, summary.SampledQueries, summary.EmittedNegatives, summary.SkippedNoPositive)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}
