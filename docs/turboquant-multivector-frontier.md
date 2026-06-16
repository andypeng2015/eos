# TurboQuant Multi-Vector Frontier

## Thesis

TurboQuant changes the CorkScrewDB/Eos vector-database design space from "one dense vector per object" toward "many compact child vectors per parent object." The differentiator is not just compressing a single embedding row. It is that a parent object can carry many semantic views or state slices while staying near the storage budget of one conventional large dense vector, especially when CorkScrewDB stores child vectors in a packed parent layout that pays parent/object overhead once.

This is a default-embedder product lane for CorkScrewDB: Eos can emit compact child vectors for document spans, event or trace timelines, time-series windows, agent memory snapshots, and later multimodal or object-state views. TurboQuant then keeps those child views cheap enough to index and search locally.

## Exact Claim Boundaries

- The high-child-count claim is for compact `128d` child vectors compared against larger `1024d` to `3072d` dense parent-vector budgets. Same-dimension `128d` accounting does not fit hundreds of children: with `32` bytes of packed parent-object overhead, same-dim packed `128d` children fit only `14` q2, `7` q4, or `3` q8 children.
- Packed-parent accounting is a storage/accounting and local flat API claim: parent overhead is paid once, while each child remains a compact TurboQuant payload. It is distinct from separate child-entry accounting, where per-vector overhead is paid per child.
- Quality evidence, storage evidence, and local CorkScrewDB API evidence are separate. Cache-only quality rows do not measure DB directory size or API latency; planner rows do not measure ranking quality; local flat CorkScrewDB rows do not prove remote mode, federation, HNSW, WAL/compaction behavior, or production service latency.
- The current `128d` Eos child cache is prefix-truncated and L2-renormalized. Treat it as a measured bridge, not native Matryoshka training or a dedicated numeric time-series encoder.
- q4/fp16 sidecar rerank is a different product surface. It preserves quality for two-stage retrieval, but a per-child fp16 sidecar erases much of the hundred-child storage advantage and should not be used as the direct child-vector storage claim.

## Evidence Table

| Evidence | Layer | Exact result | Boundary |
| --- | --- | --- | --- |
| Packed planner frontier | Storage/accounting | Against one `3072d` fp32 dense vector budget, packed `128d` children fit q2/q4/q8 counts `341`/`180`/`93`. | Planner math only; same-dim `128d` does not fit hundreds. |
| Event trace use case | Storage/accounting | q4 `180` children fits at `0.996104x` of the `3072d` dense budget. | Edge-fit storage row; no retrieval-quality claim. |
| q2 frontier use case | Storage/accounting | q2 `341` children fits at `0.999026x` of the `3072d` dense budget. | Storage frontier only. |
| Eos 128d SciFact child cache | Cache-only quality plus planner | q4 quality drop is about `-0.002630` nDCG@10 and `-0.001667` recall@100, while overhead-aware per-child accounting fits `123` q4 children per `3072d` dense budget. | Prefix-truncated cache; no CorkScrewDB API or DB-size measurement. |
| Packed q4 SciFact local DB | Local flat CorkScrewDB API | q4 packed parent DB multiple `0.025970x`, p95 `9.505725ms`, nDCG@10 `0.407586`, recall@100 `0.741889`. | Local flat exact parent search only; not remote, HNSW, or federation. |
| Scaled packed time-series smoke | Local flat CorkScrewDB API | `100` parents and `10,000` child windows, packed minimal DB bytes `1,037,918`, DB multiple `0.844660x`, p95 `5.916649ms`, recall@100 `1.000000`. | Synthetic text-rendered windows; not production time-series quality or a trained numeric encoder. |

## Novel Usage Buckets

- Document spans: store many compact vectors for passages, sections, tables, or extracted claims under one document parent, then search and roll up by parent.
- Event and trace timelines: represent a parent run, user session, service request, or incident as compact event-state vectors instead of one averaged embedding.
- Time-series windows: attach vectors for sliding windows or detected regimes under a parent series, keeping planner accounting separate from measured CorkScrewDB DB bytes.
- Agent memory snapshots: store compact state, observation, decision, and outcome views under one task or conversation parent so retrieval can hit the right memory facet.
- Multimodal and object state later: keep the same parent/child shape for image regions, UI states, sensor packets, or object lifecycle slices once native encoders and quality gates exist.

## Promotion Gates And Next Experiments

- Keep `plan-multivector-storage` and `scripts/smoke_eos_multivector_usecase_frontier.fw` as the byte-accounting gate. Report both same-dim controls and large-baseline rows.
- Use `scripts/smoke_eos_multivector_budget_quality.fw` when cache-only ranking quality must be cited beside overhead-aware fit counts.
- Use `scripts/smoke_eos_corkscrewdb_budget_quality.fw` or `scripts/smoke_corkscrewdb_child_vectors.fw` with candidate-specific child/query/qrels inputs when the actual local flat CorkScrewDB API path must pass.
- Promote packed-parent evidence only when layout, metadata mode, child-ID mode, DB bytes, p95 latency, and parent-search mode are recorded.
- Before broader product claims, run larger real-document and real-event workloads, native/trained compact heads or Matryoshka experiments, and remote/HNSW/federation-specific CorkScrewDB smokes rather than extrapolating from local flat rows.
