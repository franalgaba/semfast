# Production Benchmark Harness

This harness is for comparing Semfast against Moss-style local retrieval with a real embedding model.

## Production Setup

The production path uses:

- `sentence-transformers/all-MiniLM-L6-v2` through FastEmbed
- TurboVec as the vector artifact backend
- local BM25 plus vector hybrid retrieval
- explicit benchmark warmup before latency samples
- JSONL documents and JSONL benchmark queries

This matches the important local runtime shape: build vectors once, package the artifact, load it locally, embed each query locally, and retrieve without a network hop.

It is not yet the full hosted Moss replacement. The missing hosted pieces are project API keys, ingestion endpoints, artifact storage, signed runtime downloads, index versioning, sync, and hot-swap support.

## Commands

Prepare the local runtime:

```bash
scripts/setup-production-runtime.sh
source /private/tmp/semfast-runtime/semfast-production.env
cargo run --release --features real-embeddings -p semfast-cli -- embedding doctor \
  --embedding-model minilm
```

Generate a deterministic corpus:

```bash
cargo run --release --features real-embeddings -p semfast-cli -- fixture generate \
  --docs-out /private/tmp/semfast-production/docs.jsonl \
  --queries-out /private/tmp/semfast-production/queries.jsonl \
  --documents 100000 \
  --queries 1000
```

Build the real-model artifact:

```bash
cargo run --release --features real-embeddings -p semfast-cli -- index build \
  /private/tmp/semfast-production/docs.jsonl \
  --out /private/tmp/semfast-production/index-minilm \
  --embedding-model minilm \
  --embedding-batch-size 256
```

Run the warmed benchmark:

```bash
cargo run --release --features real-embeddings -p semfast-cli -- bench \
  /private/tmp/semfast-production/index-minilm \
  --queries /private/tmp/semfast-production/queries.jsonl \
  --alpha 0.3
```

The report must include:

```json
{
  "harness": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_backend": "turbovec",
    "warmed": true
  }
}
```

## Success Bar

To claim parity or better than Moss, use the real-model harness above, not the hash fixture.

| Metric | Required |
| --- | ---: |
| Search-only P50 | <= Moss reported/local baseline |
| Search-only P99 | <= Moss reported/local baseline |
| End-to-end P50 | <= 10ms |
| End-to-end P99 | <= Moss reported/local baseline |
| Recall@5 | >= Moss baseline on the same query set |
| Harness model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector backend | `turbovec` |
| Warmed runtime | `true` |

## Current Environment Status

The current production harness passes on 100K chunks with real MiniLM embeddings:

```text
setup: ONNX Runtime 1.24.2 macOS arm64
embedding model: sentence-transformers/all-MiniLM-L6-v2
vector backend: turbovec
documents/chunks: 100000
queries: 1000
embedding batch size: 256
SEMFAST_ONNX_INTRA_THREADS: 4
alpha: 0.3

search_only p50/p99: 0.638ms / 0.762ms
end_to_end p50/p99: 2.165ms / 2.408ms
embedding p50/p99: 1.467ms / 1.677ms
vector_search p50/p99: 0.707ms / 0.751ms
recall@5: 1.000
mrr: 1.000
artifact bytes: 247596111
load time: 712.129ms
```

Compared with the documented Moss benchmark target:

| Metric | Moss | Semfast |
| --- | ---: | ---: |
| End-to-end P50 | 3.1ms | 2.165ms |
| End-to-end P99 | 5.4ms | 2.408ms |
| Search-only P50 | n/a | 0.638ms |
| Search-only P99 | n/a | 0.762ms |
| Recall@5 | same-query-set baseline required | 1.000 |

Semfast now beats the published Moss P50 and P99 bars on the 100K production harness. Embedding inference remains the dominant component, with TurboVec search under 1ms p99.

## TypeScript Package Benchmark

The Milestone 2 TypeScript package benchmark runs through the N-API binding, not through `semfast-cli`.

```bash
cd packages/semfast-node
source /private/tmp/semfast-runtime/semfast-production.env
bun run bench -- \
  --index /private/tmp/semfast-100k/index-minilm \
  --queries /private/tmp/semfast-100k/queries.jsonl \
  --alpha 0.3
```

Latest TypeScript result:

```text
bun: v1.3.11
platform: darwin arm64
package version: 0.1.0
native version: 0.1.0

search_only p50/p99: 0.630ms / 0.742ms
end_to_end p50/p99: 2.283ms / 2.526ms
embedding p50/p99: 1.575ms / 1.813ms
vector_search p50/p99: 0.709ms / 0.761ms
recall@5: 1.000
mrr: 1.000
wrapper overhead: 0.141ms
concurrent throughput: 440.239 qps
concurrent p99: 18.944ms
```

The TypeScript report also includes `typescript.wrapperOverheadMs` and a `concurrent` section with 8-caller throughput and p99 latency.

This is within the Milestone 2 target of `<= 3.5ms` P50 and `<= 6.0ms` P99 end-to-end.

## Resolved Setup Failure

The initial MiniLM benchmark failed because the environment was incomplete, not because TurboVec or query execution was slow:

- `.fastembed_cache` contained a partial MiniLM cache with `model.onnx` but missing tokenizer/config files.
- A stale FastEmbed `.lock` file from the interrupted download caused later initialization attempts to wait indefinitely.
- `ORT_DYLIB_PATH` was unset, and the dynamic ONNX Runtime build could not locate `libonnxruntime`.

`scripts/setup-production-runtime.sh` fixes those conditions by preparing ONNX Runtime, completing the MiniLM cache, removing stale cache locks, and writing a sourceable environment file.

The script writes `FASTEMBED_CACHE_DIR` as an absolute repository path so package commands can run from `packages/semfast-node` without looking for a separate relative `.fastembed_cache`.
