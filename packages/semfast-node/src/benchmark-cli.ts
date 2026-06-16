#!/usr/bin/env bun
import { benchmark, readBenchmarkQueries, SemfastIndex } from "./index.js";
import type { QueryMode } from "./types.js";

interface BenchmarkCliOptions {
  readonly indexPath: string;
  readonly queriesPath: string;
  readonly topK: number;
  readonly alpha: number;
  readonly mode: QueryMode;
}

const options = parseArguments(process.argv.slice(2));
const index = await SemfastIndex.load(options.indexPath);

try {
  const queries = await readBenchmarkQueries(options.queriesPath);
  const report = await benchmark(index, queries, {
    topK: options.topK,
    alpha: options.alpha,
    mode: options.mode,
  });
  console.log(JSON.stringify(report, null, 2));
} finally {
  await index.close();
}

function parseArguments(args: readonly string[]): BenchmarkCliOptions {
  const values = new Map<string, string>();

  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (key === undefined || !key.startsWith("--") || value === undefined) {
      throw new Error(`invalid argument sequence near ${key ?? "<end>"}`);
    }
    values.set(key.slice(2), value);
  }

  const indexPath = requiredValue(values, "index");
  const queriesPath = requiredValue(values, "queries");

  return {
    indexPath,
    queriesPath,
    topK: optionalInteger(values, "top-k", 5),
    alpha: optionalNumber(values, "alpha", 0.7),
    mode: optionalMode(values, "mode", "hybrid"),
  };
}

function requiredValue(values: ReadonlyMap<string, string>, key: string): string {
  const value = values.get(key);
  if (value === undefined || value.length === 0) {
    throw new Error(`missing required --${key}`);
  }

  return value;
}

function optionalInteger(
  values: ReadonlyMap<string, string>,
  key: string,
  defaultValue: number,
): number {
  const value = values.get(key);
  if (value === undefined) {
    return defaultValue;
  }

  return Number.parseInt(value, 10);
}

function optionalNumber(
  values: ReadonlyMap<string, string>,
  key: string,
  defaultValue: number,
): number {
  const value = values.get(key);
  if (value === undefined) {
    return defaultValue;
  }

  return Number.parseFloat(value);
}

function optionalMode(
  values: ReadonlyMap<string, string>,
  key: string,
  defaultValue: QueryMode,
): QueryMode {
  const value = values.get(key);
  if (value === undefined) {
    return defaultValue;
  }
  if (value === "vector" || value === "bm25" || value === "hybrid") {
    return value;
  }

  throw new Error(`unsupported --${key}: ${value}`);
}
