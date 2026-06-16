import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { expect, test } from "bun:test";

import {
  benchmark,
  createSemfastApp,
  doctor,
  inspect,
  nativeVersion,
  readBenchmarkQueries,
  SemfastIndex,
} from "../src/index.js";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
let fixturePromise: Promise<FixtureArtifact> | undefined;

test("loads, queries, benchmarks, and closes a Semfast artifact", async () => {
  const fixture = await getFixtureArtifact();
  const manifest = await inspect(fixture.indexPath);
  expect(manifest.vectorBackend).toBe("turbovec");
  expect(nativeVersion()).toBe("0.1.0");

  const index = await SemfastIndex.load(fixture.indexPath);
  try {
    const results = await index.query("sfdoc000001 account workflow 000001", {
      alpha: 0.3,
      topK: 5,
    });
    expect(results.length).toBeGreaterThan(0);
    expect(typeof results[0]?.id).toBe("string");
    expect(typeof results[0]?.text).toBe("string");
    expect(typeof results[0]?.score).toBe("number");

    const filteredResults = await index.query("account workflow", {
      filter: { tenant: "tenant-1" },
      topK: 5,
    });
    expect(filteredResults.length).toBeGreaterThan(0);
    expect(filteredResults.every((result) => result.metadata.tenant === "tenant-1")).toBe(true);

    const vectorResults = await index.query("sfdoc000001 account workflow 000001", {
      mode: "vector",
      topK: 5,
    });
    expect(vectorResults.length).toBeGreaterThan(0);

    const bm25Results = await index.query("sfdoc000001 account workflow 000001", {
      mode: "bm25",
      topK: 5,
    });
    expect(bm25Results.length).toBeGreaterThan(0);

    const measured = await index.queryMeasured("sfdoc000001 account workflow 000001", {
      alpha: 0.3,
      topK: 5,
    });
    expect(measured.timings.embeddingMs).toBeGreaterThanOrEqual(0);

    const embedding = await index.embedQuery("sfdoc000001 account workflow 000001");
    expect(embedding.length).toBe(manifest.dimensions);

    const vectorHits = await index.searchVector(embedding, 5);
    expect(vectorHits.length).toBeGreaterThan(0);
    expect(typeof vectorHits[0]?.id).toBe("string");
    expect(typeof vectorHits[0]?.score).toBe("number");

    const queries = await readBenchmarkQueries(fixture.queriesPath);
    const report = await benchmark(index, queries, { alpha: 0.3, topK: 5 });
    expect(report.harness.vectorBackend).toBe("turbovec");
    expect(report.typescript.bunVersion).toBe(Bun.version);
    expect(report.typescript.nativeVersion).toBe("0.1.0");
    expect(report.typescript.wrapperOverheadMs).toBeGreaterThanOrEqual(0);
    expect(report.concurrent.concurrency).toBe(8);
    expect(report.concurrent.queryCount).toBe(queries.length);
    expect(report.concurrent.throughputQps).toBeGreaterThan(0);
    expect(report.concurrent.p99Ms).toBeGreaterThanOrEqual(0);

    const concurrentResults = await Promise.all(
      Array.from({ length: 8 }, () =>
        index.query("sfdoc000002 account workflow 000002", {
          alpha: 0.3,
          topK: 5,
        }),
      ),
    );
    expect(concurrentResults.length).toBe(8);
  } finally {
    await index.close();
  }

  await expect(index.query("after close")).rejects.toThrow(/closed/);
}, 30_000);

test("runs the hash embedding doctor without production runtime setup", async () => {
  const report = await doctor("hash");
  expect(report.dimensions).toBe(384);
});

test("reports missing MiniLM runtime setup without hanging", async () => {
  const previousOrtDylibPath = process.env.ORT_DYLIB_PATH;
  delete process.env.ORT_DYLIB_PATH;

  try {
    await expect(doctor("minilm")).rejects.toThrow(/ORT_DYLIB_PATH/);
  } finally {
    if (previousOrtDylibPath === undefined) {
      delete process.env.ORT_DYLIB_PATH;
    } else {
      process.env.ORT_DYLIB_PATH = previousOrtDylibPath;
    }
  }
});

test("reports stale MiniLM cache locks before embedding initialization", async () => {
  const directory = await mkdtemp(join(tmpdir(), "semfast-node-lock-test-"));
  const fakeOrtPath = join(directory, "libonnxruntime.dylib");
  await writeFile(fakeOrtPath, "");
  await writeFakeMiniLmCache(directory);

  const result = spawnSync(
    "bun",
    [
      "-e",
      [
        "import { doctor } from './src/index.ts';",
        "try {",
        "  await doctor('minilm');",
        "  console.error('doctor unexpectedly succeeded');",
        "  process.exit(1);",
        "} catch (error) {",
        "  const message = error instanceof Error ? error.message : String(error);",
        "  console.error(message);",
        "  process.exit(message.includes('lock files') ? 0 : 1);",
        "}",
      ].join("\n"),
    ],
    {
      cwd: join(repositoryRoot, "packages", "semfast-node"),
      encoding: "utf8",
      env: {
        ...process.env,
        ORT_DYLIB_PATH: fakeOrtPath,
        FASTEMBED_CACHE_DIR: directory,
      },
    },
  );

  expect(result.status).toBe(0);
}, 30_000);

test("rejects invalid artifacts with actionable errors", async () => {
  const fixture = await getFixtureArtifact();
  const directory = await mkdtemp(join(tmpdir(), "semfast-node-invalid-artifact-"));

  const missingMetadataPath = join(directory, "missing-metadata");
  await cp(fixture.indexPath, missingMetadataPath, { recursive: true });
  await rm(join(missingMetadataPath, "metadata.jsonl"));
  await expect(SemfastIndex.load(missingMetadataPath)).rejects.toThrow(/metadata\.jsonl/);

  const unsupportedBackendPath = join(directory, "unsupported-backend");
  await cp(fixture.indexPath, unsupportedBackendPath, { recursive: true });
  const manifestPath = join(unsupportedBackendPath, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
    vector_backend: string;
  };
  manifest.vector_backend = "faiss";
  await writeFile(manifestPath, JSON.stringify(manifest));
  await expect(SemfastIndex.load(unsupportedBackendPath)).rejects.toThrow(
    /unsupported vector backend faiss/,
  );
}, 30_000);

test("serves queries through a Hono app backed by a long-lived index", async () => {
  const fixture = await getFixtureArtifact();
  const index = await SemfastIndex.load(fixture.indexPath);
  const app = createSemfastApp({ index });

  const response = await app.request("/query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      text: "sfdoc000001 account workflow 000001",
      topK: 5,
      mode: "hybrid",
    }),
  });

  expect(response.status).toBe(200);
  const results = (await response.json()) as unknown[];
  expect(results.length).toBeGreaterThan(0);

  const closeResponse = await app.request("/close", { method: "POST" });
  expect(closeResponse.status).toBe(200);
}, 30_000);

interface FixtureArtifact {
  readonly indexPath: string;
  readonly queriesPath: string;
}

async function getFixtureArtifact(): Promise<FixtureArtifact> {
  fixturePromise ??= buildFixtureArtifact();
  return fixturePromise;
}

async function buildFixtureArtifact(): Promise<FixtureArtifact> {
  const directory = await mkdtemp(join(tmpdir(), "semfast-node-test-"));
  const docsPath = join(directory, "docs.jsonl");
  const queriesPath = join(directory, "queries.jsonl");
  const indexPath = join(directory, "index");

  runCargo([
    "run",
    "-p",
    "semfast-cli",
    "--",
    "fixture",
    "generate",
    "--docs-out",
    docsPath,
    "--queries-out",
    queriesPath,
    "--documents",
    "100",
    "--queries",
    "10",
  ]);
  runCargo([
    "run",
    "-p",
    "semfast-cli",
    "--",
    "index",
    "build",
    docsPath,
    "--out",
    indexPath,
  ]);

  return { indexPath, queriesPath };
}

function runCargo(args: readonly string[]): void {
  const result = spawnSync("cargo", args, {
    cwd: repositoryRoot,
    encoding: "utf8",
  });

  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout);
  }
}

async function writeFakeMiniLmCache(cacheRoot: string): Promise<void> {
  const repoPath = join(cacheRoot, "models--Qdrant--all-MiniLM-L6-v2-onnx");
  const snapshotPath = join(repoPath, "snapshots", "fake-commit");
  await mkdir(join(repoPath, "refs"), { recursive: true });
  await mkdir(snapshotPath, { recursive: true });
  await writeFile(join(repoPath, "refs", "main"), "fake-commit");

  for (const fileName of [
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
  ]) {
    await writeFile(join(snapshotPath, fileName), "");
  }

  await writeFile(join(snapshotPath, "download.lock"), "");
}
