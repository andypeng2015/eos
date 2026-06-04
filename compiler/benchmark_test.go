package compiler

import (
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

var benchmarkBundle *Bundle
var benchmarkArtifactBytes []byte

func BenchmarkBuildTinyEmbed(b *testing.B) {
	src := []byte(sourceForPreset(PresetTinyEmbed))

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bundle, err := Build(src, Options{ModuleName: "tiny_embed"})
		if err != nil {
			b.Fatal(err)
		}
		benchmarkBundle = bundle
	}
}

func BenchmarkBuildTinyRerankArtifactSize(b *testing.B) {
	src := []byte(sourceForPreset(PresetTinyRerank))

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bundle, err := Build(src, Options{ModuleName: "tiny_rerank"})
		if err != nil {
			b.Fatal(err)
		}
		data, err := eosartifact.EncodeMLL(bundle.Artifact)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkBundle = bundle
		benchmarkArtifactBytes = data
	}
	b.ReportMetric(float64(len(benchmarkArtifactBytes)), "artifact_bytes")
}
