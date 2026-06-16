# Semfast Retrieval Latency Bench

This example is a React + Vite UI backed by a Bun + Hono API for measuring local Semfast retrieval latency. It uses the public BEIR SciFact retrieval benchmark instead of a hand-written demo knowledge base.

SciFact is distributed through BEIR, a benchmark suite for heterogeneous information retrieval tasks. The prep script downloads the public SciFact archive, converts `corpus.jsonl` into Semfast JSONL documents, preserves the official document IDs, and exposes test queries with qrels so the UI can show whether top-5 retrieval found an official relevant document.

Sources:
- [BEIR paper](https://arxiv.org/abs/2104.08663)
- [BEIR project repository](https://github.com/beir-cellar/beir)
- [SciFact dataset archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip)

## Stack

- React + Vite for the browser UI.
- Bun + Hono for the local API.
- `@semfast/semfast` for the TypeScript package.
- Rust core with Turbovec-backed local vector search.
- BEIR/SciFact for public benchmark corpus, queries, and qrels.

## Setup

```sh
cd examples/retrieval-latency-bench
bun install
bun run prepare:semfast
bun run prepare:benchmark
bun run build:index
```

`prepare:benchmark` downloads and extracts SciFact into `data/beir/`, then writes:

- `data/scifact/documents.jsonl`
- `data/scifact/queries.json`

`build:index` writes the Semfast artifact to `data/index/`.

## Run Locally

Start the API:

```sh
cd examples/retrieval-latency-bench
SEMFAST_INDEX_PATH=data/index bun run dev:api
```

Start the UI in another terminal:

```sh
cd examples/retrieval-latency-bench
bun run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

## API

- `GET /api/health` reports the loaded dataset and query count.
- `GET /api/queries` returns predefined SciFact test queries and relevant document IDs.
- `POST /api/retrieve` accepts `{ "text": "...", "queryId": "optional", "topK": 5 }`.

The retrieval response includes total browser round-trip time in the UI, engine component timings from Semfast, top-5 results, and qrels hit metadata for predefined queries.

## Verification

```sh
cd examples/retrieval-latency-bench
bun run verify
```

This builds the native package, prepares SciFact, builds the index, typechecks the app, builds the Vite bundle, and runs a Hono smoke test against the generated index.
