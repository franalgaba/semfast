# Embedding Models

Semfast now has two embedding backends:

| CLI value | Model | Use |
| --- | --- | --- |
| `hash` | `semfast-hash-embedding-v1` | Deterministic tests, fixtures, synthetic benchmarks |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` through FastEmbed | Real local semantic retrieval |

The hash backend is useful for repeatable development runs, but it is not a production semantic model.

## Real Local Model

The `real-embeddings` feature uses ONNX Runtime dynamic loading. Prepare ONNX Runtime and the FastEmbed MiniLM cache before initializing the backend:

```bash
scripts/setup-production-runtime.sh
source /private/tmp/semfast-runtime/semfast-production.env
```

The setup script downloads ONNX Runtime 1.24.2 for the current platform, removes stale FastEmbed `.lock` files, ensures the MiniLM ONNX/tokenizer/config files exist, and writes the environment file with `ORT_DYLIB_PATH`, `FASTEMBED_CACHE_DIR`, and `SEMFAST_ONNX_INTRA_THREADS=4`.

Run the preflight before building a production artifact:

```bash
cargo run --release --features real-embeddings -p semfast-cli -- embedding doctor \
  --embedding-model minilm
```

Build the CLI with the real embedding feature:

```bash
cargo run --release --features real-embeddings -p semfast-cli -- index build ./docs \
  --out ./index \
  --embedding-model minilm \
  --embedding-batch-size 256
```

Query the artifact with the same feature enabled:

```bash
cargo run --release --features real-embeddings -p semfast-cli -- query ./index \
  "what is the refund policy?"
```

The artifact manifest records the embedding model:

```json
{
  "vector_backend": "turbovec",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimensions": 384
}
```

This keeps build-time document vectors and query-time vectors in the same embedding space.

## Moss-Style Setup

This makes the local runtime closer to Moss's setup:

```text
documents
  -> real local embedding model
  -> TurboVec compressed vector artifact
  -> local runtime loads artifact
  -> query text embedded locally
  -> local TurboVec + BM25 retrieval
```

It still does not implement the hosted Moss replacement pieces: project keys, document ingestion API, artifact storage, signed downloads, sync, and hot-swapping across deployed runtimes.
