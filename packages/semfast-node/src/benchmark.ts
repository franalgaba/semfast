import type { SemfastIndex } from "./index.js";
import type { BenchmarkQuery, BenchmarkReport, QueryOptions } from "./types.js";

export async function benchmark(
  index: SemfastIndex,
  queries: readonly BenchmarkQuery[],
  options: QueryOptions = {},
): Promise<BenchmarkReport> {
  return index.benchmark(queries, options);
}

export async function readBenchmarkQueries(path: string): Promise<readonly BenchmarkQuery[]> {
  const contents = await Bun.file(path).text();
  return contents
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as BenchmarkQuery);
}
