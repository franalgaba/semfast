# Comparison Benchmark Plan

Semfast currently documents a Moss-target comparison in `docs/production-benchmark.md`. A broader public benchmark against other local retrieval libraries is still planned.

## Candidate Baselines

- FAISS
- HNSWLib
- USearch
- LanceDB local
- Qdrant local

## Required Controls

Every baseline must use:

- same documents,
- same chunking,
- same embeddings,
- same query set,
- same topK,
- same recall target,
- same hardware,
- warm query path,
- cold load measured separately.

## Required Metrics

- build time,
- artifact size,
- cold load time,
- memory after load,
- search-only P50/P95/P99,
- end-to-end P50/P95/P99,
- recall@3 and recall@5,
- MRR,
- concurrent throughput,
- concurrent P99.

## Public Claim Policy

Only claim Semfast is faster than another library after the full matrix is reproducible from checked-in scripts and documented hardware. Until then, the public claim should stay limited to the documented Moss-target harness.
