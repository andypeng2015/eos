# native-role-mining-and-space-identity-v1 report

## Outcome

Implemented the native Eos role/identity hardening slice.

- `MineModelTextHardNegatives` now defaults `RoleMode` to `auto`, resolves roles through `resolveEmbeddingRetrievalRoles`, embeds corpus records with the effective document role, and embeds queries with the effective query role.
- Explicit `raw` mining remains raw.
- Explicit `query-document` mining rejects legacy non-role-conditioned models through the same resolver used by eval/export.
- `mine-retrieval-model-hard-negatives` now accepts `--role-mode auto|raw|query-document`, passes it through, and prints the effective `role_mode`.
- Native retrieval vector export manifests now include `artifact_sha256`, package sidecar hashes/cache identity, and `embedding_space_id`.

## Follow-Up Fix

Root review found that native package behavior can depend on sibling sidecars while artifact bytes remain unchanged. Fixed by extending native vector export identity and manifest fields with:

- `weights_sha256,omitempty`
- `tokenizer_sha256,omitempty`
- `package_manifest_sha256,omitempty`
- `package_cache_key,omitempty`

These are hashed from conventional sibling paths derived from `ArtifactPath`: `DefaultWeightFilePath`, `DefaultTokenizerPath`, and `ResolvePackageManifestPath`. Missing sidecars are tolerated so sealed packages without siblings can still rely on `artifact_sha256` plus manifest/config identity. When a package manifest is present, its file hash and `PackageManifest.CacheKey()` are both included in `embedding_space_id`.

## Files Changed

- `runtime/retrieval_hard_negative_mining.go`
- `runtime/retrieval_vector_export.go`
- `runtime/retrieval_eval_test.go`
- `runtime/retrieval_vector_export_test.go`
- `cmd/eos/main.go`
- `cmd/eos/main_test.go`

## Exact Behavior Changes

- `RetrievalHardNegativeMiningConfig` has `RoleMode`.
- `RetrievalHardNegativeMiningSummary` has JSON field `role_mode`.
- Mining config normalization sets empty role mode to `auto`.
- Model hard-negative mining now follows default eval/export role behavior for role-conditioned native models.
- BM25 mining summary records the normalized config role mode but BM25 scoring is otherwise unchanged.
- Native vector export summary records:
  - `artifact_sha256,omitempty`
  - `weights_sha256,omitempty`
  - `tokenizer_sha256,omitempty`
  - `package_manifest_sha256,omitempty`
  - `package_cache_key,omitempty`
  - `embedding_space_id,omitempty`

## Identity Fields / Components

`embedding_space_id` is SHA256 hex over canonical Go JSON for a stable struct containing:

- manifest schema
- execution mode: `native_eos_embedding`
- artifact SHA256 when the artifact path points to a readable file
- weights SHA256 when the conventional sibling weights file exists
- tokenizer SHA256 when the conventional sibling tokenizer file exists
- package manifest SHA256 and package cache key when the conventional sibling package manifest exists
- model/manifest identity fields: model name, architecture version, manifest model/output dims, tokenizer vocab size/max sequence
- native encoded dimension and effective output dimension
- requested output dim
- role conditioning and raw/query/document role indices
- effective role mode
- query/document role-applied booleans
- query/document prefixes
- document chunking words/overlap/min words

Excluded from identity: dataset name, corpus/query/qrels paths, output paths, counts, batch size, timestamps, elapsed time, and backend name.

## Verification Commands and Results

- `gofmt -w runtime/retrieval_hard_negative_mining.go runtime/retrieval_vector_export.go runtime/retrieval_eval_test.go runtime/retrieval_vector_export_test.go cmd/eos/main.go cmd/eos/main_test.go`: passed.
- `go test ./runtime -run 'Role|RetrievalVectorExport|HardNegatives|Mine' -count=1`: passed (`ok m31labs.dev/eos/runtime 62.041s`).
- `go test ./cmd/eos -run 'MineRetrievalModelHardNegatives|ExportRetrievalVectors' -count=1`: passed (`ok m31labs.dev/eos/cmd/eos 1.205s`).
- `git diff --check`: passed.

Follow-up verification:

- `go test ./runtime -run 'RetrievalVectorExport|Role|HardNegatives|Mine' -count=1`: passed (`ok m31labs.dev/eos/runtime 59.948s`).
- `go test ./cmd/eos -run 'MineRetrievalModelHardNegatives|ExportRetrievalVectors' -count=1`: passed (`ok m31labs.dev/eos/cmd/eos 0.810s`).
- `git diff --check`: passed.

Follow-up test coverage:

- Native export identity test now asserts `weights_sha256`, `tokenizer_sha256`, `package_manifest_sha256`, and `package_cache_key` are present for the sibling package fixture.
- Native export identity test mutates only the sibling weights file while preserving artifact bytes and proves `weights_sha256` and `embedding_space_id` change.

## Caveats / Residual Risk

- For backward compatibility with existing callers/tests that pass descriptive non-file artifact names, native export hashes package files only when the conventional paths exist/readable; missing or empty artifact paths produce an identity without file-hash fields.
- No resume/cache rejection behavior was added for native vector exports; this patch only adds manifest identity/provenance fields.
- Identity is scoped to embedding-space behavior and does not hash dataset contents.

## Checkpoint Candidate

Yes. Coherent source/test slice with focused verification passing.

## Arbiter Next Action

Root should review the six-file diff, decide whether missing non-empty `ArtifactPath` should remain compatibility-tolerant, then checkpoint if accepted.
