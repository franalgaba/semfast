# Semfast

Local-first semantic retrieval for AI agents, voice agents, and realtime copilots.

Semfast builds a portable local index once, loads it into a Rust or Bun runtime, and answers retrieval queries without a hosted vector database in the hot path. The current artifact format combines MiniLM embeddings, TurboVec vector search, BM25 lexical search, and hybrid fusion.

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
