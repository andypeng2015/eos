package main

import (
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/odvcencio/gotreesitter/taproot/diag"

	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/syntax"
)

type sourceError struct {
	file   string
	source []byte
	err    error
}

func (e *sourceError) Error() string { return e.err.Error() }
func (e *sourceError) Unwrap() error { return e.err }

func attachSource(file string, source []byte, err error) error {
	if err == nil {
		return nil
	}
	if len(diagnosticsFromError(err)) == 0 {
		return err
	}
	return &sourceError{file: file, source: source, err: err}
}

func printCommandError(w io.Writer, err error) {
	fmt.Fprintln(w, err)
	var se *sourceError
	if !errors.As(err, &se) {
		return
	}
	renderDiagnostics(w, se.file, se.source, se.err)
}

func renderDiagnostics(w io.Writer, file string, source []byte, err error) {
	for _, d := range diagnosticsFromError(err) {
		if d.Line > 0 {
			fmt.Fprintf(w, "  --> %s:%d:%d\n", file, d.Line, max(d.Col, 1))
		}
		fmt.Fprint(w, diag.Render(source, d))
	}
}

func diagnosticsFromError(err error) []diag.Diagnostic {
	var ce *compiler.DiagnosticsError
	if !errors.As(err, &ce) {
		return nil
	}
	out := make([]diag.Diagnostic, 0, len(ce.Diagnostics))
	for _, d := range ce.Diagnostics {
		out = append(out, diag.Diagnostic{
			Code:     eosDiagnosticCode(d),
			Severity: eosDiagnosticSeverity(d.Severity),
			Line:     d.Span.Line,
			Col:      d.Span.Column,
			Message:  d.Message,
			Hint:     eosDiagnosticHint(d),
		})
	}
	return out
}

func eosDiagnosticCode(d syntax.Diagnostic) string {
	if d.Severity == syntax.SeverityWarning {
		return "EOS0002"
	}
	if d.Message == "" || strings.HasPrefix(d.Message, "invalid Eos") {
		return "EOS0001"
	}
	return "EOS1001"
}

func eosDiagnosticSeverity(sev syntax.Severity) diag.Severity {
	switch sev {
	case syntax.SeverityWarning:
		return diag.SeverityWarning
	default:
		return diag.SeverityError
	}
}

func eosDiagnosticHint(d syntax.Diagnostic) string {
	if d.Message == "" || strings.HasPrefix(d.Message, "invalid Eos") {
		return "Check declarations, braces, tensor type syntax, and return expressions."
	}
	return "Check names, tensor shapes, callable signatures, and intrinsic arguments."
}
