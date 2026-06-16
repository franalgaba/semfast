# Semfast

Local-first semantic retrieval for AI agents, voice agents, and realtime copilots.

Semfast is an open-source retrieval runtime for applications that need semantic search to happen inside the running agent process, not behind a network call. It builds a portable index once, loads that artifact locally, and serves low-latency hybrid retrieval from Rust or Bun/TypeScript.

The project is designed for realtime systems where retrieval latency is part of the user experience: voice agents, browser copilots, customer-support copilots, local-first AI tools, and embedded agent runtimes. Instead of sending every query to a hosted vector database, Semfast lets an application package the retrieval artifact with the runtime and query it directly.

## Capabilities

- Build local retrieval artifacts from files or JSONL documents.
- Query artifacts from Rust, CLI, or Bun/TypeScript through a native N-API binding.
- Run semantic, lexical, or hybrid search with `vector`, `bm25`, and `hybrid` query modes.
- Use TurboVec for local vector search.
- Use BM25 for keyword-sensitive retrieval.
- Use MiniLM embeddings through FastEmbed/ONNX Runtime for real semantic retrieval.
- Use deterministic hash embeddings for fixtures, smoke tests, and reproducible synthetic benchmarks.
- Inspect artifacts and verify backend metadata before loading them in production.
- Measure component timings for embedding, vector search, BM25, filtering, fusion, and hydration.
- Run warmed latency benchmarks and TypeScript wrapper-overhead benchmarks.
- Serve a long-lived local index from Hono for Bun applications.
- Validate retrieval behavior interactively with the BEIR/SciFact example app.

## Why Semfast Exists

Most retrieval stacks assume a service boundary: documents live in a hosted vector database, queries cross the network, and the application waits for a remote retrieval result before the model can respond. That architecture is flexible, but it is not always the right fit for realtime agents.

Voice agents and copilots often need retrieval in the same latency budget as transcription, planning, and response generation. A network hop can dominate that budget, introduce tail latency, and make offline or embedded use cases harder. Semfast explores a different shape: build the index ahead of time, ship the artifact, load it once, and keep retrieval local.

Semfast is not trying to replace every vector database. It is for cases where the working set can be packaged, synced, or downloaded as an artifact, and where predictable local query latency matters more than remote multi-tenant database features.

## How It Works

Semfast artifacts combine:

- document chunks and metadata,
- MiniLM or deterministic test embeddings,
- TurboVec vector index files,
- BM25 lexical index files,
- a manifest describing the embedding model, vector backend, dimensions, and artifact version.

At query time, Semfast embeds the query locally, searches the vector index, optionally runs BM25, fuses the scores, hydrates chunk metadata, and returns ranked results without calling a hosted retrieval service.

## Current Benchmark

Latest documented production run:

- Hardware/runtime: macOS arm64, ONNX Runtime 1.24.2
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector backend: `turbovec`
- Corpus: 100,000 chunks
- Queries: 1,000 warmed benchmark queries
- Hybrid alpha: `0.3`
- Embedding batch size: `256`
- ONNX intra threads: `4`

| Metric | Moss Published Target | Semfast Rust Core | Semfast TypeScript Package |
| --- | ---: | ---: | ---: |
| End-to-end P50 | 3.1ms | 2.165ms | 2.283ms |
| End-to-end P99 | 5.4ms | 2.408ms | 2.526ms |
| Search-only P50 | n/a | 0.638ms | 0.630ms |
| Search-only P99 | n/a | 0.762ms | 0.742ms |
| Recall@5 | same-query-set baseline required | 1.000 | 1.000 |
| Concurrent throughput | n/a | n/a | 440.239 qps |
| Concurrent P99 | n/a | n/a | 18.944ms |

These numbers are from the real-model 100K MiniLM/TurboVec harness, not the smaller SciFact demo app. Semfast currently beats the captured Moss P50/P99 latency bars in this documented setup. A broader apples-to-apples matrix against FAISS, HNSWLib, USearch, LanceDB, and Qdrant is not included yet.

See [docs/production-benchmark.md](docs/production-benchmark.md) for the full commands, component timings, load time, artifact size, TypeScript wrapper overhead, and root-cause notes from the MiniLM setup work.

## Repository Layout

- `crates/semfast-core`: Rust artifact, embedding, retrieval, and benchmark core.
- `crates/semfast-cli`: CLI for building, inspecting, querying, and benchmarking artifacts.
- `packages/semfast-node`: Bun/TypeScript package backed by the native Rust runtime.
- `examples/retrieval-latency-bench`: React + Vite retrieval latency bench using BEIR/SciFact.
- `docs/production-benchmark.md`: production benchmark report and Moss comparison.
- `docs/milestone-1-benchmark.md`: Rust milestone acceptance benchmark notes.

## TypeScript Package

```ts
import { SemfastIndex } from "@semfast/semfast";

const index = await SemfastIndex.load("/private/tmp/semfast-100k/index-minilm");

const results = await index.query("what is the refund policy?", {
  topK: 5,
  alpha: 0.3,
  mode: "hybrid",
});

await index.close();
```

See [packages/semfast-node/README.md](packages/semfast-node/README.md) for package setup, Hono integration, and Bun benchmark instructions.

## Local SciFact Demo

The example app downloads the public BEIR/SciFact benchmark, builds a local Semfast index, and shows query latency plus qrels hit/miss metadata in a browser UI.

```bash
cd examples/retrieval-latency-bench
bun install
bun run verify
```

See [examples/retrieval-latency-bench/README.md](examples/retrieval-latency-bench/README.md).

## Not Included Yet

Semfast is not yet a hosted Moss replacement service. Missing hosted-service pieces include project API keys, ingestion endpoints, artifact storage, signed runtime downloads, index versioning, sync, dashboard, and hot-swap support.

## License

Apache-2.0. See [LICENSE](LICENSE).
