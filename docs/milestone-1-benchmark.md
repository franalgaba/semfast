# Milestone 1 Benchmark

This benchmark is the current reproducible acceptance run for the Rust core.

## Setup

Generate a deterministic 100K document corpus and 1,000 benchmark queries:

```bash
cargo run --release -p semfast-cli -- fixture generate \
  --docs-out /private/tmp/semfast-100k/docs.jsonl \
  --queries-out /private/tmp/semfast-100k/queries.jsonl \
  --documents 100000 \
  --queries 1000
```

Build the local TurboVec-backed artifact:

```bash
cargo run --release -p semfast-cli -- index build \
  /private/tmp/semfast-100k/docs.jsonl \
  --out /private/tmp/semfast-100k/index-current
```

Inspect the artifact:

```bash
cargo run --release -p semfast-cli -- inspect /private/tmp/semfast-100k/index-current
```

Run the benchmark:

```bash
cargo run --release -p semfast-cli -- bench \
  /private/tmp/semfast-100k/index-current \
  --queries /private/tmp/semfast-100k/queries.jsonl
```

## Current Result

Result from the current implementation:

```json
{
  "query_count": 1000,
  "harness": {
    "embedding_model": "semfast-hash-embedding-v1",
    "vector_backend": "turbovec",
    "warmed": true
  },
  "search_only": {
    "p50_ms": 0.6513340000000001,
    "p95_ms": 0.722125,
    "p99_ms": 0.789833
  },
  "end_to_end": {
    "p50_ms": 0.659125,
    "p95_ms": 0.714916,
    "p99_ms": 0.755459
  },
  "quality": {
    "recall_at_3": 0.996,
    "recall_at_5": 0.996,
    "mrr": 0.996
  },
  "artifact": {
    "chunk_count": 100000,
    "artifact_size_bytes": 247596098,
    "load_time_ms": 692.171416
  }
}
```

The full CLI output also includes component timing for embedding, TurboVec search, BM25, filtering, fusion, and hydration.

## Interpretation

This run beats the Milestone 1 latency targets on the generated 100K corpus:

| Metric | Target | Current |
| --- | ---: | ---: |
| TurboVec search-only P50 | <= 3ms | 0.651ms |
| TurboVec search-only P99 | <= 6ms | 0.790ms |
| End-to-end local retrieval P50 | <= 5ms | 0.659ms |
| End-to-end local retrieval P99 | <= 10ms | 0.755ms |
| Golden-query recall@5 | >= 80% | 99.6% |

The benchmark is synthetic and deterministic. It proves the local runtime, artifact format, TurboVec hot path, benchmark harness, and 100K acceptance flow. For production semantic quality, build with `--features real-embeddings` and `--embedding-model minilm`; that uses `sentence-transformers/all-MiniLM-L6-v2` through FastEmbed.

## Production Moss Comparison

The real-model 100K production harness now uses MiniLM, TurboVec, ONNX Runtime 1.24.2, `SEMFAST_ONNX_INTRA_THREADS=4`, `--embedding-batch-size 256`, and `--alpha 0.3`.

Latest production result:

| Metric | Moss | Semfast |
| --- | ---: | ---: |
| End-to-end P50 | 3.1ms | 2.165ms |
| End-to-end P99 | 5.4ms | 2.408ms |
| Search-only P50 | n/a | 0.638ms |
| Search-only P99 | n/a | 0.762ms |
| Recall@5 | same-query-set baseline required | 1.000 |

See `docs/production-benchmark.md` for commands and the full component breakdown.
