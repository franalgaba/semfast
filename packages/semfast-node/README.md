# @semfast/semfast

Local-first Semfast retrieval runtime for Bun and Hono.

This package loads the same TurboVec artifact produced by the Rust core and queries it through a native N-API binding from Bun. It does not call the Semfast CLI for load/query/benchmark, and it does not use a hosted retrieval service in the hot path.

## Setup

From the repository root:

```bash
scripts/setup-production-runtime.sh
source /private/tmp/semfast-runtime/semfast-production.env
```

The setup script prepares ONNX Runtime, FastEmbed MiniLM cache files, and the environment variables required by the real embedding backend.

## Build

```bash
cd packages/semfast-node
bun install
bun run build
```

## Query

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

## Hono App

```ts
import { createSemfastApp } from "@semfast/semfast";

const app = createSemfastApp({
  indexPath: "/private/tmp/semfast-100k/index-minilm",
});

Bun.serve({
  port: 3000,
  fetch: app.fetch,
});
```

The Hono app keeps the index loaded across requests and exposes `/query`, `/query-measured`, `/manifest`, `/doctor`, `/health`, and `/close`.

## Voice-Agent Integration

Keep one `SemfastIndex` or one Hono app alive for the lifetime of the voice-agent worker. Load the index during worker startup, reuse it for every transcript turn, and call `close()` only during shutdown.

```ts
import { SemfastIndex } from "@semfast/semfast";

const index = await SemfastIndex.load("/private/tmp/semfast-100k/index-minilm");

export async function retrieveForTurn(transcript: string) {
  return index.query(transcript, {
    topK: 5,
    alpha: 0.3,
    mode: "hybrid",
  });
}
```

Concurrent callers are supported. The embedding model may serialize internally, so p99 under concurrent load should be tracked with `bun run bench`.

## Doctor

```ts
import { doctor } from "@semfast/semfast";

const report = await doctor("minilm");
console.log(report);
```

## Benchmark

```bash
bun run bench -- \
  --index /private/tmp/semfast-100k/index-minilm \
  --queries /private/tmp/semfast-100k/queries.jsonl \
  --alpha 0.3
```

The benchmark loads the index once, warms the query path through the Rust core benchmark runner, and reports Bun/package/native metadata, wrapper overhead, component timings, and an 8-caller concurrent query report.

Latest 100K MiniLM/TurboVec result on macOS arm64:

```text
end_to_end p50/p99: 2.283ms / 2.526ms
search_only p50/p99: 0.630ms / 0.742ms
recall@5: 1.000
concurrent throughput: 440.239 qps
concurrent p99: 18.944ms
```

Compared with the captured Moss benchmark target:

| Metric | Moss Published Target | Semfast TypeScript Package |
| --- | ---: | ---: |
| End-to-end P50 | 3.1ms | 2.283ms |
| End-to-end P99 | 5.4ms | 2.526ms |
| Search-only P50 | n/a | 0.630ms |
| Search-only P99 | n/a | 0.742ms |
| Recall@5 | same-query-set baseline required | 1.000 |

The Rust core run is slightly faster at `2.165ms` P50 and `2.408ms` P99 end-to-end. See `../../docs/production-benchmark.md` for the complete component breakdown and acceptance context.

## Supported Platforms

The package targets Bun 1.3+. A native module is produced at `native/semfast_node_native.node` after `bun run build`.

Linux x64 is supported through the documented local build fallback:

```bash
bun run build:native
```

The build requires a Rust toolchain only when a matching prebuilt native module is not already present.

## Troubleshooting

- Missing native binary: run `bun run build:native` from `packages/semfast-node`, or install a package that includes a matching prebuilt `native/semfast_node_native.node`.
- Missing MiniLM runtime: run `scripts/setup-production-runtime.sh` from the repository root and source `/private/tmp/semfast-runtime/semfast-production.env`.
- Stale MiniLM cache lock: remove stale `.lock` files in `FASTEMBED_CACHE_DIR` or rerun the setup script.
- Incomplete MiniLM cache: rerun the setup script so `model.onnx`, `tokenizer.json`, `config.json`, `special_tokens_map.json`, and `tokenizer_config.json` are present.

## Not Included Yet

This package is not the hosted replacement service. It does not include project keys, cloud sync, signed artifact downloads, a dashboard, or remote ingestion.
