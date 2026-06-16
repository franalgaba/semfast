export type QueryMode = "vector" | "bm25" | "hybrid";

export type MetadataValue = string | number | boolean;

export type Metadata = Record<string, MetadataValue>;

export type MetadataFilter = Metadata;

export interface QueryOptions {
  readonly topK?: number;
  readonly alpha?: number;
  readonly mode?: QueryMode;
  readonly filter?: MetadataFilter;
}

export interface SearchResult {
  readonly id: string;
  readonly text: string;
  readonly score: number;
  readonly vectorScore?: number;
  readonly lexicalScore?: number;
  readonly metadata: Metadata;
}

export interface VectorHit {
  readonly id: string;
  readonly score: number;
}

export interface QueryTimings {
  readonly embeddingMs: number;
  readonly vectorSearchMs: number;
  readonly bm25Ms: number;
  readonly filteringMs: number;
  readonly fusionMs: number;
  readonly hydrationMs: number;
}

export interface MeasuredQueryResult {
  readonly results: readonly SearchResult[];
  readonly timings: QueryTimings;
}

export interface ArtifactManifest {
  readonly version: number;
  readonly vectorBackend: string;
  readonly embeddingModel: string;
  readonly dimensions: number;
  readonly chunkCount: number;
  readonly createdAt: string;
  readonly createdAtUnixSeconds: number;
}

export interface DoctorReport {
  readonly embeddingModel: string;
  readonly dimensions: number;
  readonly sampleNorm: number;
}

export interface BenchmarkQuery {
  readonly text: string;
  readonly expected_chunk_id?: number | null;
}

export interface LatencyReport {
  readonly p50Ms: number;
  readonly p95Ms: number;
  readonly p99Ms: number;
}

export interface ComponentLatencyReport {
  readonly embeddingMs: LatencyReport;
  readonly vectorSearchMs: LatencyReport;
  readonly bm25Ms: LatencyReport;
  readonly filteringMs: LatencyReport;
  readonly fusionMs: LatencyReport;
  readonly hydrationMs: LatencyReport;
}

export interface QualityReport {
  readonly recallAt3: number;
  readonly recallAt5: number;
  readonly mrr: number;
}

export interface ArtifactReport {
  readonly chunkCount: number;
  readonly artifactSizeBytes: number;
  readonly loadTimeMs: number;
}

export interface HarnessReport {
  readonly embeddingModel: string;
  readonly vectorBackend: string;
  readonly warmed: boolean;
}

export interface TypeScriptBenchmarkReport {
  readonly bunVersion: string;
  readonly packageVersion: string;
  readonly nativeVersion: string;
  readonly platform: string;
  readonly arch: string;
  readonly nativeCallWallTimeMs: number;
  readonly jsonParseMs: number;
  readonly wrapperOverheadMs: number;
}

export interface ConcurrentBenchmarkReport {
  readonly concurrency: number;
  readonly queryCount: number;
  readonly throughputQps: number;
  readonly p50Ms: number;
  readonly p95Ms: number;
  readonly p99Ms: number;
}

export interface BenchmarkReport {
  readonly queryCount: number;
  readonly harness: HarnessReport;
  readonly searchOnly: LatencyReport;
  readonly endToEnd: LatencyReport;
  readonly components: ComponentLatencyReport;
  readonly quality: QualityReport;
  readonly artifact: ArtifactReport;
  readonly typescript: TypeScriptBenchmarkReport;
  readonly concurrent: ConcurrentBenchmarkReport;
}
