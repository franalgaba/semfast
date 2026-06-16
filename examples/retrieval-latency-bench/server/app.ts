import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { SemfastIndex } from "@semfast/semfast";
import type { QueryTimings, SearchResult } from "@semfast/semfast";

const queriesPath = "data/scifact/queries.json";
const datasetName = "BEIR/SciFact";
const defaultTopK = 5;
const maxTopK = 20;

interface BenchmarkQuery {
  readonly id: string;
  readonly text: string;
  readonly relevantDocumentIds: readonly string[];
}

interface RetrieveBody {
  readonly text?: string;
  readonly queryId?: string;
  readonly topK?: number;
}

export async function createExampleApp(indexPath: string): Promise<Hono> {
  const app = new Hono();
  const index = await SemfastIndex.load(indexPath);
  const benchmarkQueries = await loadBenchmarkQueries();
  const benchmarkQueryById = new Map(benchmarkQueries.map((query) => [query.id, query]));

  app.get("/api/queries", (context) =>
    context.json({
      dataset: datasetName,
      queries: benchmarkQueries,
    }),
  );

  app.post("/api/retrieve", async (context) => {
    const body = (await context.req.json()) as RetrieveBody;
    if (typeof body.text !== "string" || body.text.trim().length === 0) {
      return context.json({ error: "text is required" }, 400);
    }

    const query = body.queryId !== undefined ? benchmarkQueryById.get(body.queryId) : undefined;
    const relevantDocumentIds = query?.relevantDocumentIds ?? [];
    const topK = normalizeTopK(body.topK);
    const startedAt = performance.now();

    const measured = await index.queryMeasured(body.text.trim(), {
      topK,
      alpha: 0.3,
      mode: "hybrid",
    });
    const totalMs = performance.now() - startedAt;
    const retrieved = formatRetrievedChunks(measured.results, relevantDocumentIds);
    const firstRelevantRank = retrieved.find((chunk) => chunk.relevant)?.rank ?? null;

    return context.json({
      dataset: datasetName,
      query: {
        id: query?.id ?? null,
        text: body.text.trim(),
        relevantDocumentIds,
      },
      metrics: {
        totalMs,
        retrievalMs: sumTimings(measured.timings),
        embeddingMs: measured.timings.embeddingMs,
        vectorSearchMs: measured.timings.vectorSearchMs,
        bm25Ms: measured.timings.bm25Ms,
        filteringMs: measured.timings.filteringMs,
        fusionMs: measured.timings.fusionMs,
        hydrationMs: measured.timings.hydrationMs,
      },
      quality: {
        topK,
        officialQrelsAvailable: relevantDocumentIds.length > 0,
        hitRelevant: firstRelevantRank !== null,
        firstRelevantRank,
      },
      retrieved,
    });
  });

  app.get("/api/health", (context) =>
    context.json({ ok: true, dataset: datasetName, queryCount: benchmarkQueries.length }),
  );
  app.onError((error, context) => {
    return context.json({ error: error.message }, 500);
  });
  app.use("*", serveStatic({ root: "./dist" }));

  return app;
}

async function loadBenchmarkQueries(): Promise<readonly BenchmarkQuery[]> {
  const file = Bun.file(queriesPath);
  if (!(await file.exists())) {
    throw new Error(`missing ${queriesPath}; run bun run prepare:benchmark`);
  }

  return (await file.json()) as readonly BenchmarkQuery[];
}

function normalizeTopK(topK: number | undefined): number {
  if (topK === undefined || !Number.isFinite(topK)) {
    return defaultTopK;
  }

  return Math.min(Math.max(Math.trunc(topK), 1), maxTopK);
}

function formatRetrievedChunks(
  chunks: readonly SearchResult[],
  relevantDocumentIds: readonly string[],
) {
  const relevantDocumentIdSet = new Set(relevantDocumentIds);

  return chunks.map((chunk, index) => {
    const corpusId = String(chunk.metadata.corpus_id ?? chunk.id);
    const retrievedChunk = {
      rank: index + 1,
      id: chunk.id,
      corpusId,
      title: String(chunk.metadata.title ?? ""),
      text: chunk.text,
      score: chunk.score,
      relevant: relevantDocumentIdSet.has(corpusId),
    };

    return {
      ...retrievedChunk,
      ...optionalScore("vectorScore", chunk.vectorScore),
      ...optionalScore("lexicalScore", chunk.lexicalScore),
    };
  });
}

function optionalScore(key: "vectorScore" | "lexicalScore", score: number | null | undefined) {
  return typeof score === "number" && Number.isFinite(score) ? { [key]: score } : {};
}

function sumTimings(timings: QueryTimings): number {
  return (
    timings.embeddingMs +
    timings.vectorSearchMs +
    timings.bm25Ms +
    timings.filteringMs +
    timings.fusionMs +
    timings.hydrationMs
  );
}
