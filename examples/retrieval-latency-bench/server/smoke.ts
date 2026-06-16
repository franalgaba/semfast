import { createExampleApp } from "./app.ts";

const indexPath = process.env.SEMFAST_INDEX_PATH;
if (indexPath === undefined || indexPath.trim().length === 0) {
  throw new Error("SEMFAST_INDEX_PATH is required for the smoke test");
}

const app = await createExampleApp(indexPath);

const queriesResponse = await app.request("/api/queries");
if (!queriesResponse.ok) {
  throw new Error(await queriesResponse.text());
}

const queriesBody = (await queriesResponse.json()) as {
  readonly queries?: readonly { readonly id: string; readonly text: string }[];
};
const firstQuery = queriesBody.queries?.[0];
if (firstQuery === undefined) {
  throw new Error("smoke response did not include benchmark queries");
}

const response = await app.request("/api/retrieve", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    text: firstQuery.text,
    queryId: firstQuery.id,
  }),
});

if (!response.ok) {
  throw new Error(await response.text());
}

const body = (await response.json()) as {
  readonly metrics?: {
    readonly totalMs?: number;
    readonly retrievalMs?: number;
    readonly embeddingMs?: number;
  };
  readonly quality?: {
    readonly officialQrelsAvailable?: boolean;
  };
  readonly retrieved?: readonly unknown[];
};

if (!Array.isArray(body.retrieved) || body.retrieved.length === 0) {
  throw new Error("smoke response did not include retrieved Semfast chunks");
}

if (
  typeof body.metrics?.totalMs !== "number" ||
  typeof body.metrics.retrievalMs !== "number" ||
  typeof body.metrics.embeddingMs !== "number"
) {
  throw new Error("smoke response did not include latency metrics");
}

if (body.quality?.officialQrelsAvailable !== true) {
  throw new Error("smoke response did not include official qrels metadata");
}

console.log(
  JSON.stringify({
    ok: true,
    retrieved: body.retrieved.length,
    retrievalMs: body.metrics.retrievalMs,
  }),
);
