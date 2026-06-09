package eosruntime

import (
	"fmt"
	"sort"
	"strings"
)

// RelabelTeacherNegativesConfig controls how teacher-scored hard-negative
// examples are relabeled into clean training rows. On sparse-label corpora,
// model-mined "negatives" are dominated by unlabeled relevant documents;
// relabeling promotes confirmed-relevant candidates to positives and keeps
// only teacher-confirmed-irrelevant candidates as negatives.
type RelabelTeacherNegativesConfig struct {
	// PromoteMin is the absolute teacher-score floor for promoting a mined
	// negative to a new positive row.
	PromoteMin float32
	// PromoteSlack additionally requires a promoted candidate to score at
	// least positive_score-PromoteSlack, so weak candidates are not promoted
	// past a strong labeled positive.
	PromoteSlack float32
	// PromoteCap bounds promoted positives per input example; 0 is unlimited.
	PromoteCap int
	// NegativeMax is the teacher-score ceiling for keeping a candidate as a
	// true negative. Candidates between the promote and negative bands are
	// dropped as ambiguous.
	NegativeMax float32
	// NegativesPerRow is how many true negatives to attach to each output
	// row, drawn from the example's own surviving negatives first and then
	// from the per-query pool. 0 attaches none.
	NegativesPerRow int
	// PromotedSourceSuffix tags promoted rows as "<source>:<suffix>" so
	// source-weighted training can distinguish them. Empty keeps the source.
	PromotedSourceSuffix string
	// EmitPairs flattens output rows to query-positive pairs without
	// negatives or teacher scores.
	EmitPairs bool
}

// RelabelTeacherNegativesSummary reports what the relabel pass did.
type RelabelTeacherNegativesSummary struct {
	InputExamples         int     `json:"input_examples"`
	ScoredExamples        int     `json:"scored_examples"`
	UnscoredExamples      int     `json:"unscored_examples"`
	CandidatesSeen        int     `json:"candidates_seen"`
	Promoted              int     `json:"promoted"`
	PromotedCapSkipped    int     `json:"promoted_cap_skipped"`
	TrueNegativesKept     int     `json:"true_negatives_kept"`
	DroppedAmbiguous      int     `json:"dropped_ambiguous"`
	PoolQueries           int     `json:"pool_queries"`
	PoolNegatives         int     `json:"pool_negatives"`
	OutputRows            int     `json:"output_rows"`
	OutputRowsWithNegs    int     `json:"output_rows_with_negatives"`
	OutputPromotedRows    int     `json:"output_promoted_rows"`
	DuplicateRowsSkipped  int     `json:"duplicate_rows_skipped"`
	MeanPositiveScore     float64 `json:"mean_positive_score"`
	MeanPromotedScore     float64 `json:"mean_promoted_score"`
	MeanKeptNegativeScore float64 `json:"mean_kept_negative_score"`
}

// RelabelScoredText is a candidate text with its teacher score, used for
// per-query true-negative pools.
type RelabelScoredText struct {
	Text  string
	Score float32
}

// BuildTeacherNegativePool extracts per-query true-negative candidates from
// teacher-scored examples: every negative scoring at or below negativeMax
// joins the query's pool. Texts are deduplicated per query.
func BuildTeacherNegativePool(examples []EmbeddingTextHardNegativeExample, negativeMax float32) map[string][]RelabelScoredText {
	pool := make(map[string][]RelabelScoredText)
	seen := make(map[string]map[string]bool)
	for _, example := range examples {
		if len(example.TeacherScores) != 1+len(example.Negatives) {
			continue
		}
		for i, negative := range example.Negatives {
			score := example.TeacherScores[1+i]
			if score > negativeMax {
				continue
			}
			if seen[example.Query] == nil {
				seen[example.Query] = make(map[string]bool)
			}
			if seen[example.Query][negative] {
				continue
			}
			seen[example.Query][negative] = true
			pool[example.Query] = append(pool[example.Query], RelabelScoredText{Text: negative, Score: score})
		}
	}
	for query := range pool {
		entries := pool[query]
		sort.SliceStable(entries, func(i, j int) bool { return entries[i].Score < entries[j].Score })
		pool[query] = entries
	}
	return pool
}

// RelabelTeacherNegatives converts teacher-scored mined hard negatives into
// clean training rows: confirmed-relevant candidates become new positive rows,
// confirmed-irrelevant candidates stay as negatives, and the ambiguous middle
// band is dropped. The optional pool supplies extra per-query true negatives
// (for example random corpus documents the teacher scored low).
func RelabelTeacherNegatives(examples []EmbeddingTextHardNegativeExample, pool map[string][]RelabelScoredText, cfg RelabelTeacherNegativesConfig) ([]EmbeddingTextHardNegativeExample, RelabelTeacherNegativesSummary, error) {
	if len(examples) == 0 {
		return nil, RelabelTeacherNegativesSummary{}, fmt.Errorf("relabel input is empty")
	}
	if cfg.PromoteMin <= 0 {
		return nil, RelabelTeacherNegativesSummary{}, fmt.Errorf("promote-min must be positive")
	}
	if cfg.NegativeMax <= 0 {
		return nil, RelabelTeacherNegativesSummary{}, fmt.Errorf("negative-max must be positive")
	}
	if cfg.NegativeMax >= cfg.PromoteMin {
		return nil, RelabelTeacherNegativesSummary{}, fmt.Errorf("negative-max %.3f must be below promote-min %.3f", cfg.NegativeMax, cfg.PromoteMin)
	}

	summary := RelabelTeacherNegativesSummary{InputExamples: len(examples), PoolQueries: len(pool)}
	for _, entries := range pool {
		summary.PoolNegatives += len(entries)
	}

	var out []EmbeddingTextHardNegativeExample
	emitted := make(map[string]bool)
	var positiveScoreSum, promotedScoreSum, keptNegativeScoreSum float64

	emit := func(source, query, positive string, positiveScore float32, negatives []RelabelScoredText, scored bool) {
		key := query + "\x00" + positive
		if emitted[key] {
			summary.DuplicateRowsSkipped++
			return
		}
		emitted[key] = true
		row := EmbeddingTextHardNegativeExample{Source: source, Query: query, Positive: positive}
		if !cfg.EmitPairs && len(negatives) > 0 {
			row.Negatives = make([]string, 0, len(negatives))
			if scored {
				row.TeacherScores = make([]float32, 0, 1+len(negatives))
				row.TeacherScores = append(row.TeacherScores, positiveScore)
			}
			for _, negative := range negatives {
				row.Negatives = append(row.Negatives, negative.Text)
				if scored {
					row.TeacherScores = append(row.TeacherScores, negative.Score)
				}
			}
			summary.OutputRowsWithNegs++
		}
		out = append(out, row)
		summary.OutputRows++
	}

	attach := func(query string, ownNegatives []RelabelScoredText, exclude map[string]bool) []RelabelScoredText {
		if cfg.NegativesPerRow <= 0 {
			return nil
		}
		attached := make([]RelabelScoredText, 0, cfg.NegativesPerRow)
		for _, negative := range ownNegatives {
			if len(attached) >= cfg.NegativesPerRow {
				break
			}
			if exclude[negative.Text] {
				continue
			}
			attached = append(attached, negative)
		}
		for _, negative := range pool[query] {
			if len(attached) >= cfg.NegativesPerRow {
				break
			}
			if exclude[negative.Text] {
				continue
			}
			duplicate := false
			for _, existing := range attached {
				if existing.Text == negative.Text {
					duplicate = true
					break
				}
			}
			if !duplicate {
				attached = append(attached, negative)
			}
		}
		return attached
	}

	type classifiedExample struct {
		example           EmbeddingTextHardNegativeExample
		promoteCandidates []RelabelScoredText
		ownNegatives      []RelabelScoredText
		exclude           map[string]bool
	}

	// Pass 1: classify candidates and emit every original labeled row first,
	// so qrel provenance wins (query, positive) dedup collisions against
	// promoted copies of the same document.
	classified := make([]classifiedExample, 0, len(examples))
	for _, example := range examples {
		scored := len(example.TeacherScores) == 1+len(example.Negatives) && len(example.TeacherScores) > 0
		if !scored {
			summary.UnscoredExamples++
			emit(example.Source, example.Query, example.Positive, 0, nil, false)
			continue
		}
		summary.ScoredExamples++
		positiveScore := example.TeacherScores[0]
		positiveScoreSum += float64(positiveScore)

		var promoteCandidates []RelabelScoredText
		var ownNegatives []RelabelScoredText
		for i, negative := range example.Negatives {
			summary.CandidatesSeen++
			score := example.TeacherScores[1+i]
			switch {
			case score >= cfg.PromoteMin && score >= positiveScore-cfg.PromoteSlack:
				promoteCandidates = append(promoteCandidates, RelabelScoredText{Text: negative, Score: score})
			case score <= cfg.NegativeMax:
				ownNegatives = append(ownNegatives, RelabelScoredText{Text: negative, Score: score})
				summary.TrueNegativesKept++
				keptNegativeScoreSum += float64(score)
			default:
				summary.DroppedAmbiguous++
			}
		}
		sort.SliceStable(promoteCandidates, func(i, j int) bool { return promoteCandidates[i].Score > promoteCandidates[j].Score })
		if cfg.PromoteCap > 0 && len(promoteCandidates) > cfg.PromoteCap {
			summary.PromotedCapSkipped += len(promoteCandidates) - cfg.PromoteCap
			promoteCandidates = promoteCandidates[:cfg.PromoteCap]
		}

		exclude := map[string]bool{example.Positive: true}
		for _, candidate := range promoteCandidates {
			exclude[candidate.Text] = true
		}
		emit(example.Source, example.Query, example.Positive, positiveScore, attach(example.Query, ownNegatives, exclude), true)
		classified = append(classified, classifiedExample{
			example:           example,
			promoteCandidates: promoteCandidates,
			ownNegatives:      ownNegatives,
			exclude:           exclude,
		})
	}

	// Pass 2: emit promoted rows.
	for _, entry := range classified {
		for _, candidate := range entry.promoteCandidates {
			summary.Promoted++
			promotedScoreSum += float64(candidate.Score)
			source := entry.example.Source
			if cfg.PromotedSourceSuffix != "" {
				if source == "" {
					source = cfg.PromotedSourceSuffix
				} else {
					source = source + ":" + cfg.PromotedSourceSuffix
				}
			}
			before := summary.OutputRows
			emit(source, entry.example.Query, candidate.Text, candidate.Score, attach(entry.example.Query, entry.ownNegatives, entry.exclude), true)
			if summary.OutputRows > before {
				summary.OutputPromotedRows++
			}
		}
	}

	if summary.ScoredExamples > 0 {
		summary.MeanPositiveScore = positiveScoreSum / float64(summary.ScoredExamples)
	}
	if summary.Promoted > 0 {
		summary.MeanPromotedScore = promotedScoreSum / float64(summary.Promoted)
	}
	if summary.TrueNegativesKept > 0 {
		summary.MeanKeptNegativeScore = keptNegativeScoreSum / float64(summary.TrueNegativesKept)
	}
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("relabel produced no output rows")
	}
	return out, summary, nil
}

// TeacherScoreQuantiles summarizes a teacher-scored dataset for threshold
// selection: per-band counts plus positive/negative score quantiles.
type TeacherScoreQuantiles struct {
	ScoredExamples   int       `json:"scored_examples"`
	UnscoredExamples int       `json:"unscored_examples"`
	Positives        int       `json:"positives"`
	Negatives        int       `json:"negatives"`
	PositiveQuantile []float32 `json:"positive_quantiles"`
	NegativeQuantile []float32 `json:"negative_quantiles"`
}

// SummarizeTeacherScores computes quantiles (q10..q90 by decile) over
// positive and negative teacher scores for threshold picking.
func SummarizeTeacherScores(examples []EmbeddingTextHardNegativeExample) TeacherScoreQuantiles {
	var positives, negatives []float32
	stats := TeacherScoreQuantiles{}
	for _, example := range examples {
		if len(example.TeacherScores) != 1+len(example.Negatives) || len(example.TeacherScores) == 0 {
			stats.UnscoredExamples++
			continue
		}
		stats.ScoredExamples++
		positives = append(positives, example.TeacherScores[0])
		negatives = append(negatives, example.TeacherScores[1:]...)
	}
	stats.Positives = len(positives)
	stats.Negatives = len(negatives)
	stats.PositiveQuantile = scoreDeciles(positives)
	stats.NegativeQuantile = scoreDeciles(negatives)
	return stats
}

func scoreDeciles(scores []float32) []float32 {
	if len(scores) == 0 {
		return nil
	}
	sorted := append([]float32(nil), scores...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	out := make([]float32, 0, 9)
	for decile := 1; decile <= 9; decile++ {
		idx := decile * len(sorted) / 10
		if idx >= len(sorted) {
			idx = len(sorted) - 1
		}
		out = append(out, sorted[idx])
	}
	return out
}

// FormatTeacherScoreQuantiles renders quantiles for terminal output.
func FormatTeacherScoreQuantiles(stats TeacherScoreQuantiles) string {
	var b strings.Builder
	fmt.Fprintf(&b, "scored_examples=%d unscored=%d positives=%d negatives=%d\n", stats.ScoredExamples, stats.UnscoredExamples, stats.Positives, stats.Negatives)
	render := func(label string, deciles []float32) {
		if len(deciles) == 0 {
			fmt.Fprintf(&b, "%s: none\n", label)
			return
		}
		fmt.Fprintf(&b, "%s:", label)
		for i, value := range deciles {
			fmt.Fprintf(&b, " q%d0=%.3f", i+1, value)
		}
		b.WriteString("\n")
	}
	render("positive_scores", stats.PositiveQuantile)
	render("negative_scores", stats.NegativeQuantile)
	return b.String()
}
