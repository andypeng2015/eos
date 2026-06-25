package syntax

import (
	"fmt"

	gotreesitter "github.com/odvcencio/gotreesitter"
	walk "github.com/odvcencio/gotreesitter/taproot/walk"
)

// Language returns the Eos tree-sitter language, loaded (and cached) from the
// embedded pre-generated parse table. Loading the blob via taproot/walk keeps
// this package grammar-free (no grammargen / grammars registry); the grammar DSL
// lives in syntax/dsl and is used only to regenerate the blob.
func Language() (*gotreesitter.Language, error) {
	return walk.LanguageFromBlob("eos", grammarBlob)
}

// ParseTree parses Eos source and returns the tree-sitter tree and language.
func ParseTree(src []byte) (*gotreesitter.Tree, *gotreesitter.Language, error) {
	lang, err := Language()
	if err != nil {
		return nil, nil, fmt.Errorf("generate Eos language: %w", err)
	}
	parser := gotreesitter.NewParser(lang)
	tree, err := parser.Parse(src)
	if err != nil {
		return nil, nil, fmt.Errorf("parse Eos source: %w", err)
	}
	return tree, lang, nil
}
