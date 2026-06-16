# Contributing

Semfast is early-stage infrastructure. Keep changes narrow, benchmarkable, and easy to reproduce.

## Development Setup

Requirements:

- Rust stable with Cargo
- Bun 1.3+
- macOS arm64 or Linux x64 for native-package development

Build and test the Rust workspace:

```bash
cargo test
```

Build and test the Bun package:

```bash
cd packages/semfast-node
bun install
bun run build
bun test
```

Run the retrieval demo:

```bash
cd examples/retrieval-latency-bench
bun install
bun run verify
```

## Benchmark Changes

Benchmark-related changes should state:

- corpus size,
- embedding model,
- vector backend,
- query count,
- warmup behavior,
- search-only and end-to-end P50/P99,
- recall@k or the reason recall is not measured.

Do not compare against another library unless the benchmark uses the same corpus, embeddings, topK, recall target, and hardware.

## Generated Artifacts

Do not commit:

- `target/`
- `node_modules/`
- package `dist/`
- generated example `data/`
- downloaded models or ONNX runtime files
- native `.node` binaries unless release packaging explicitly requires them.
