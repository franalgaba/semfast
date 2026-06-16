export class SemfastError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "SemfastError";
  }
}

export class SemfastNativeError extends SemfastError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "SemfastNativeError";
  }
}

export function wrapNativeError(error: unknown): SemfastNativeError {
  if (error instanceof SemfastNativeError) {
    return error;
  }

  if (error instanceof Error) {
    return new SemfastNativeError(error.message, { cause: error });
  }

  return new SemfastNativeError(String(error));
}
