package syntax

import _ "embed"

//go:generate go run ../cmd/eos-grammar

// grammarBlob is the pre-generated parse table for EosGrammar().
// TestGrammarBinIsCurrent guards it against drift.
//
//go:embed grammar.bin
var grammarBlob []byte
