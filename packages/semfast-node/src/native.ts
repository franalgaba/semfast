const NATIVE_MODULE_PATHS = [
  new URL("../native/semfast_node_native.node", import.meta.url).pathname,
  new URL("../../native/semfast_node_native.node", import.meta.url).pathname,
] as const;

export interface NativeSemfastIndex {
  queryJson(
    text: string,
    topK?: number,
    alpha?: number,
    mode?: string,
    filterJson?: string,
  ): string;
  queryMeasuredJson(
    text: string,
    topK?: number,
    alpha?: number,
    mode?: string,
    filterJson?: string,
  ): string;
  embedQueryJson(text: string): string;
  searchVectorJson(vector: readonly number[], topK?: number): string;
  benchmarkJson(
    queriesJson: string,
    topK?: number,
    alpha?: number,
    mode?: string,
  ): string;
  manifestJson(): string;
  close(): void;
}

interface NativeBinding {
  readonly NativeSemfastIndex: {
    load(path: string): NativeSemfastIndex;
  };
  readonly doctorEmbedding: (embeddingModel?: string) => string;
  readonly inspectArtifact: (path: string) => string;
  readonly nativeVersion: () => string;
}

export const nativeBinding = loadNativeBinding();

function loadNativeBinding(): NativeBinding {
  const missingPaths: string[] = [];

  for (const modulePath of NATIVE_MODULE_PATHS) {
    try {
      return require(modulePath) as NativeBinding;
    } catch (error) {
      if (!isMissingNativeModuleError(error)) {
        throw error;
      }
      missingPaths.push(modulePath);
    }
  }

  throw new Error(
    `Semfast native module was not found at any expected path: ${missingPaths.join(", ")}. ` +
      "Install a package with a matching prebuilt binary or run `bun run build:native` from packages/semfast-node.",
  );
}

function isMissingNativeModuleError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("Cannot find module") ||
    (typeof error === "object" &&
      error !== null &&
      "code" in error &&
      (error as { readonly code: unknown }).code === "MODULE_NOT_FOUND")
  );
}
