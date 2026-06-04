// Command eos-grammar regenerates the embedded parser blob used by syntax.
package main

import (
	"fmt"
	"os"

	"github.com/odvcencio/gotreesitter/grammargen"
	"m31labs.dev/eos/syntax"
)

func main() {
	_, blob, err := grammargen.GenerateLanguageAndBlob(syntax.EosGrammar())
	if err != nil {
		fmt.Fprintln(os.Stderr, "eos-grammar: generate parse table:", err)
		os.Exit(1)
	}
	if err := os.WriteFile("grammar.bin", blob, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "eos-grammar: write grammar.bin:", err)
		os.Exit(1)
	}
	fmt.Printf("regenerated grammar.bin (%d bytes)\n", len(blob))
}
