import { SemfastError, wrapNativeError } from "./errors.js";
import { nativeBinding, type NativeSemfastIndex } from "./native.js";
import {
  type ArtifactManifest,
  type BenchmarkQuery,
  type BenchmarkReport,
  type ConcurrentBenchmarkReport,
  type DoctorReport,
  type MeasuredQueryResult,
  type QueryMode,
  type QueryOptions,
  type SearchResult,
  type VectorHit,
} from "./types.js";

const PACKAGE_VERSION = "0.1.0";
const DEFAULT_TOP_K = 5;
const DEFAULT_ALPHA = 0.7;
const DEFAULT_QUERY_MODE: QueryMode = "hybrid";

export class SemfastIndex {
  private constructor(private readonly nativeIndex: NativeSemfastIndex) {}

  static async load(path: string): Promise<SemfastIndex> {
    try {
      return new SemfastIndex(nativeBinding.NativeSemfastIndex.load(path));
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async query(text: string, options: QueryOptions = {}): Promise<readonly SearchResult[]> {
    const queryOptions = normalizeQueryOptions(options);
    const filterJson = stringifyFilter(queryOptions.filter);

    try {
      return parseSearchResults(
        this.nativeIndex.queryJson(
          text,
          queryOptions.topK,
          queryOptions.alpha,
          queryOptions.mode,
          filterJson,
        ),
      );
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async queryMeasured(
    text: string,
    options: QueryOptions = {},
  ): Promise<MeasuredQueryResult> {
    const queryOptions = normalizeQueryOptions(options);
    const filterJson = stringifyFilter(queryOptions.filter);

    try {
      return parseMeasuredQueryResult(
        this.nativeIndex.queryMeasuredJson(
          text,
          queryOptions.topK,
          queryOptions.alpha,
          queryOptions.mode,
          filterJson,
        ),
      );
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async embedQuery(text: string): Promise<Float32Array> {
    try {
      return Float32Array.from(parseJson<number[]>(this.nativeIndex.embedQueryJson(text)));
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async searchVector(vector: Float32Array, topK: number = DEFAULT_TOP_K): Promise<readonly VectorHit[]> {
    try {
      return parseVectorHits(
        this.nativeIndex.searchVectorJson(Array.from(vector), normalizeTopK(topK)),
      );
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async manifest(): Promise<ArtifactManifest> {
    try {
      return parseManifest(this.nativeIndex.manifestJson());
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async benchmark(
    queries: readonly BenchmarkQuery[],
    options: QueryOptions = {},
  ): Promise<BenchmarkReport> {
    const queryOptions = normalizeQueryOptions(options);

    try {
      const nativeCallStartedAt = performance.now();
      const nativeReportJson = this.nativeIndex.benchmarkJson(
        JSON.stringify(queries),
        queryOptions.topK,
        queryOptions.alpha,
        queryOptions.mode,
      );
      const nativeCallWallTimeMs = performance.now() - nativeCallStartedAt;

      const jsonParseStartedAt = performance.now();
      const nativeReport = parseBenchmarkReport(nativeReportJson);
      const jsonParseMs = performance.now() - jsonParseStartedAt;

      return withTypeScriptReport(
        nativeReport,
        { nativeCallWallTimeMs, jsonParseMs },
        this,
        queries,
        queryOptions,
      );
    } catch (error) {
      throw wrapNativeError(error);
    }
  }

  async close(): Promise<void> {
    this.nativeIndex.close();
  }
}

export async function inspect(path: string): Promise<ArtifactManifest> {
  try {
    return parseManifest(nativeBinding.inspectArtifact(path));
  } catch (error) {
    throw wrapNativeError(error);
  }
}

export async function doctor(embeddingModel: string = "minilm"): Promise<DoctorReport> {
  try {
    return parseDoctorReport(nativeBinding.doctorEmbedding(embeddingModel));
  } catch (error) {
    throw wrapNativeError(error);
  }
}

export function nativeVersion(): string {
  return nativeBinding.nativeVersion();
}

export { benchmark, readBenchmarkQueries } from "./benchmark.js";
export { createSemfastApp, type SemfastHonoOptions } from "./server.js";
export * from "./errors.js";
export type * from "./types.js";

interface NativeSearchResult {
  readonly id: string;
  readonly text: string;
  readonly score: number;
  readonly vector_score?: number;
  readonly lexical_score?: number;
  readonly metadata: Record<string, string | number | boolean>;
}

interface NativeMeasuredQueryResult {
  readonly results: readonly NativeSearchResult[];
  readonly timings: {
    readonly embedding_ms: number;
    readonly vector_search_ms: number;
    readonly bm25_ms: number;
    readonly filtering_ms: number;
    readonly fusion_ms: number;
    readonly hydration_ms: number;
  };
}

interface NativeManifest {
  readonly version: number;
  readonly vector_backend: string;
  readonly embedding_model: string;
  readonly dimensions: number;
  readonly chunk_count: number;
  readonly created_at: string;
  readonly created_at_unix_seconds: number;
}

interface NativeBenchmarkReport {
  readonly query_count: number;
  readonly harness: {
    readonly embedding_model: string;
    readonly vector_backend: string;
    readonly warmed: boolean;
  };
  readonly search_only: NativeLatencyReport;
  readonly end_to_end: NativeLatencyReport;
  readonly components: {
    readonly embedding_ms: NativeLatencyReport;
    readonly vector_search_ms: NativeLatencyReport;
    readonly bm25_ms: NativeLatencyReport;
    readonly filtering_ms: NativeLatencyReport;
    readonly fusion_ms: NativeLatencyReport;
    readonly hydration_ms: NativeLatencyReport;
  };
  readonly quality: {
    readonly recall_at_3: number;
    readonly recall_at_5: number;
    readonly mrr: number;
  };
  readonly artifact: {
    readonly chunk_count: number;
    readonly artifact_size_bytes: number;
    readonly load_time_ms: number;
  };
}

interface NativeLatencyReport {
  readonly p50_ms: number;
  readonly p95_ms: number;
  readonly p99_ms: number;
}

interface BenchmarkTiming {
  readonly nativeCallWallTimeMs: number;
  readonly jsonParseMs: number;
}

function normalizeQueryOptions(options: QueryOptions): Required<QueryOptions> {
  return {
    topK: normalizeTopK(options.topK ?? DEFAULT_TOP_K),
    alpha: normalizeAlpha(options.alpha ?? DEFAULT_ALPHA),
    mode: normalizeMode(options.mode ?? DEFAULT_QUERY_MODE),
    filter: options.filter ?? {},
  };
}

function normalizeTopK(topK: number): number {
  if (!Number.isFinite(topK)) {
    throw new SemfastError(`topK must be finite, received ${topK}`);
  }

  return Math.max(1, Math.trunc(topK));
}

function normalizeAlpha(alpha: number): number {
  if (!Number.isFinite(alpha)) {
    throw new SemfastError(`alpha must be finite, received ${alpha}`);
  }

  return Math.min(1, Math.max(0, alpha));
}

function normalizeMode(mode: QueryMode): QueryMode {
  if (mode === "vector" || mode === "bm25" || mode === "hybrid") {
    return mode;
  }

  throw new SemfastError(`unsupported query mode: ${mode}`);
}

function stringifyFilter(filter: QueryOptions["filter"]): string | undefined {
  if (filter === undefined || Object.keys(filter).length === 0) {
    return undefined;
  }

  return JSON.stringify(filter);
}

function parseSearchResults(json: string): readonly SearchResult[] {
  return parseJson<NativeSearchResult[]>(json).map(toSearchResult);
}

function parseMeasuredQueryResult(json: string): MeasuredQueryResult {
  const measured = parseJson<NativeMeasuredQueryResult>(json);

  return {
    results: measured.results.map(toSearchResult),
    timings: {
      embeddingMs: measured.timings.embedding_ms,
      vectorSearchMs: measured.timings.vector_search_ms,
      bm25Ms: measured.timings.bm25_ms,
      filteringMs: measured.timings.filtering_ms,
      fusionMs: measured.timings.fusion_ms,
      hydrationMs: measured.timings.hydration_ms,
    },
  };
}

function parseVectorHits(json: string): readonly VectorHit[] {
  return parseJson<VectorHit[]>(json);
}

function parseManifest(json: string): ArtifactManifest {
  const manifest = parseJson<NativeManifest>(json);

  return {
    version: manifest.version,
    vectorBackend: manifest.vector_backend,
    embeddingModel: manifest.embedding_model,
    dimensions: manifest.dimensions,
    chunkCount: manifest.chunk_count,
    createdAt: manifest.created_at,
    createdAtUnixSeconds: manifest.created_at_unix_seconds,
  };
}

function parseDoctorReport(json: string): DoctorReport {
  const report = parseJson<{
    readonly embedding_model: string;
    readonly dimensions: number;
    readonly sample_norm: number;
  }>(json);

  return {
    embeddingModel: report.embedding_model,
    dimensions: report.dimensions,
    sampleNorm: report.sample_norm,
  };
}

function parseBenchmarkReport(json: string): Omit<BenchmarkReport, "typescript" | "concurrent"> {
  const report = parseJson<NativeBenchmarkReport>(json);

  return {
    queryCount: report.query_count,
    harness: {
      embeddingModel: report.harness.embedding_model,
      vectorBackend: report.harness.vector_backend,
      warmed: report.harness.warmed,
    },
    searchOnly: toLatencyReport(report.search_only),
    endToEnd: toLatencyReport(report.end_to_end),
    components: {
      embeddingMs: toLatencyReport(report.components.embedding_ms),
      vectorSearchMs: toLatencyReport(report.components.vector_search_ms),
      bm25Ms: toLatencyReport(report.components.bm25_ms),
      filteringMs: toLatencyReport(report.components.filtering_ms),
      fusionMs: toLatencyReport(report.components.fusion_ms),
      hydrationMs: toLatencyReport(report.components.hydration_ms),
    },
    quality: {
      recallAt3: report.quality.recall_at_3,
      recallAt5: report.quality.recall_at_5,
      mrr: report.quality.mrr,
    },
    artifact: {
      chunkCount: report.artifact.chunk_count,
      artifactSizeBytes: report.artifact.artifact_size_bytes,
      loadTimeMs: report.artifact.load_time_ms,
    },
  };
}

async function withTypeScriptReport(
  report: Omit<BenchmarkReport, "typescript" | "concurrent">,
  benchmarkTiming: BenchmarkTiming,
  index: SemfastIndex,
  queries: readonly BenchmarkQuery[],
  options: Required<QueryOptions>,
): Promise<BenchmarkReport> {
  return {
    ...report,
    typescript: {
      bunVersion: Bun.version,
      packageVersion: PACKAGE_VERSION,
      nativeVersion: nativeVersion(),
      platform: process.platform,
      arch: process.arch,
      nativeCallWallTimeMs: benchmarkTiming.nativeCallWallTimeMs,
      jsonParseMs: benchmarkTiming.jsonParseMs,
      wrapperOverheadMs: benchmarkTiming.jsonParseMs,
    },
    concurrent: await runConcurrentBenchmark(index, queries, options),
  };
}

async function runConcurrentBenchmark(
  index: SemfastIndex,
  queries: readonly BenchmarkQuery[],
  options: Required<QueryOptions>,
): Promise<ConcurrentBenchmarkReport> {
  const concurrency = 8;
  const latencies: number[] = [];
  const startedAt = performance.now();

  for (let offset = 0; offset < queries.length; offset += concurrency) {
    const batch = queries.slice(offset, offset + concurrency);
    const batchLatencies = await Promise.all(
      batch.map(async (query) => {
        const queryStartedAt = performance.now();
        await index.query(query.text, options);
        return performance.now() - queryStartedAt;
      }),
    );
    latencies.push(...batchLatencies);
  }

  const elapsedSeconds = Math.max((performance.now() - startedAt) / 1000, 0.001);
  const sortedLatencies = [...latencies].sort((left, right) => left - right);

  return {
    concurrency,
    queryCount: latencies.length,
    throughputQps: latencies.length / elapsedSeconds,
    p50Ms: percentile(sortedLatencies, 0.5),
    p95Ms: percentile(sortedLatencies, 0.95),
    p99Ms: percentile(sortedLatencies, 0.99),
  };
}

function toSearchResult(result: NativeSearchResult): SearchResult {
  return {
    id: result.id,
    text: result.text,
    score: result.score,
    vectorScore: result.vector_score,
    lexicalScore: result.lexical_score,
    metadata: result.metadata,
  };
}

function toLatencyReport(report: NativeLatencyReport) {
  return {
    p50Ms: report.p50_ms,
    p95Ms: report.p95_ms,
    p99Ms: report.p99_ms,
  };
}

function percentile(sortedValues: readonly number[], percentileValue: number): number {
  if (sortedValues.length === 0) {
    return 0;
  }

  const index = Math.min(
    sortedValues.length - 1,
    Math.floor(sortedValues.length * percentileValue),
  );
  return sortedValues[index] ?? 0;
}

function parseJson<T>(json: string): T {
  return JSON.parse(json) as T;
}
