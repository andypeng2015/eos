package eosruntime

import (
	"bufio"
	"io"
	"strings"
)

const (
	embeddingJSONLScannerInitialBuffer = 64 * 1024
	embeddingJSONLMaxRecordBytes       = 16 * 1024 * 1024
)

func scanEmbeddingJSONLLines(r io.Reader, visit func(lineNo int, line string) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, embeddingJSONLScannerInitialBuffer), embeddingJSONLMaxRecordBytes)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if err := visit(lineNo, line); err != nil {
			return err
		}
	}
	return scanner.Err()
}
