package eos_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestNoTrackedSpecOrPlanDocs(t *testing.T) {
	cmd := exec.Command("git", "ls-files")
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("git ls-files unavailable: %v", err)
	}
	var bad []string
	for _, raw := range strings.Split(string(out), "\n") {
		path := strings.TrimSpace(raw)
		if path == "" {
			continue
		}
		if isSpecOrPlanDoc(path) {
			bad = append(bad, path)
		}
	}
	if len(bad) > 0 {
		t.Fatalf("spec/plan docs must not be tracked:\n%s", strings.Join(bad, "\n"))
	}
}

func isSpecOrPlanDoc(path string) bool {
	path = strings.ToLower(filepath.ToSlash(path))
	if strings.HasPrefix(path, "specs/") ||
		strings.HasPrefix(path, "plans/") ||
		strings.HasPrefix(path, "docs/specs/") ||
		strings.HasPrefix(path, "docs/plans/") {
		return true
	}
	base := filepath.Base(path)
	switch base {
	case "spec.md", "specs.md", "specification.md", "specifications.md", "plan.md", "plans.md":
		return true
	}
	return strings.HasSuffix(path, ".spec.md") || strings.HasSuffix(path, ".plan.md")
}

func TestTrainMantaCandidateEvalOnlyUsesPairEvalAndExportsBeforeGates(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	if strings.Contains(src, "appendEvalModeArgs") {
		t.Fatalf("candidate wrapper must not append training-mode flags to final eval-only commands")
	}
	exportIdx := strings.Index(src, `"Export sealed MLL"`)
	finalEvalIdx := strings.Index(src, `"Final validation eval-only"`)
	if exportIdx < 0 || finalEvalIdx < 0 {
		t.Fatalf("candidate wrapper missing expected export or eval-only step")
	}
	if exportIdx > finalEvalIdx {
		t.Fatalf("sealed export must run before final eval-only gates so a trained artifact is not left stale")
	}
	finalEvalBlock := sliceBetween(t, src, `evalArgs := []string{"train-embed", "--eval-only"`, `err = runCmd(cfg, "Final validation eval-only"`)
	hardEvalBlock := sliceBetween(t, src, `hardArgs := []string{"train-embed", "--eval-only"`, `err = runCmd(cfg, "Hard holdout eval-only"`)
	for name, block := range map[string]string{"final": finalEvalBlock, "hard": hardEvalBlock} {
		if strings.Contains(block, "--hard-negative-train") || strings.Contains(block, "--pairwise-train") {
			t.Fatalf("%s eval-only command must not inherit training-mode flags:\n%s", name, block)
		}
		if !strings.Contains(block, "--no-tokenizer") {
			t.Fatalf("%s eval-only command must preserve token JSONL no-tokenizer handling:\n%s", name, block)
		}
	}
}

func sliceBetween(t *testing.T, src string, start string, end string) string {
	t.Helper()
	startIdx := strings.Index(src, start)
	if startIdx < 0 {
		t.Fatalf("missing start marker %q", start)
	}
	endIdx := strings.Index(src[startIdx:], end)
	if endIdx < 0 {
		t.Fatalf("missing end marker %q after %q", end, start)
	}
	return src[startIdx : startIdx+endIdx]
}
