import { Hono } from "hono";

import { doctor, SemfastIndex } from "./index.js";
import { SemfastError } from "./errors.js";
import type { QueryOptions } from "./types.js";

export interface SemfastHonoOptions {
  readonly index?: SemfastIndex;
  readonly indexPath?: string;
}

interface QueryRequestBody extends QueryOptions {
  readonly text?: string;
}

export function createSemfastApp(options: SemfastHonoOptions): Hono {
  const app = new Hono();
  const state = createIndexState(options);

  app.get("/health", (context) => context.json({ ok: true }));

  app.get("/manifest", async (context) => {
    const index = await state.index();
    return context.json(await index.manifest());
  });

  app.post("/query", async (context) => {
    const index = await state.index();
    const body = await readQueryBody(context.req);
    return context.json(await index.query(body.text, body.options));
  });

  app.post("/query-measured", async (context) => {
    const index = await state.index();
    const body = await readQueryBody(context.req);
    return context.json(await index.queryMeasured(body.text, body.options));
  });

  app.get("/doctor", async (context) => {
    const embeddingModel = context.req.query("embeddingModel") ?? "minilm";
    return context.json(await doctor(embeddingModel));
  });

  app.post("/close", async (context) => {
    await state.close();
    return context.json({ closed: true });
  });

  app.onError((error, context) => {
    const status = error instanceof SemfastError ? 400 : 500;
    return context.json({ error: error.message }, status);
  });

  return app;
}

function createIndexState(options: SemfastHonoOptions) {
  let loadedIndex = options.index;
  let loadingIndex: Promise<SemfastIndex> | undefined;

  return {
    async index(): Promise<SemfastIndex> {
      if (loadedIndex !== undefined) {
        return loadedIndex;
      }

      if (options.indexPath === undefined) {
        throw new SemfastError("createSemfastApp requires either index or indexPath");
      }

      loadingIndex ??= SemfastIndex.load(options.indexPath);
      loadedIndex = await loadingIndex;
      return loadedIndex;
    },

    async close(): Promise<void> {
      if (loadedIndex !== undefined) {
        await loadedIndex.close();
        loadedIndex = undefined;
      }
      loadingIndex = undefined;
    },
  };
}

async function readQueryBody(request: { json(): Promise<unknown> }): Promise<{
  readonly text: string;
  readonly options: QueryOptions;
}> {
  const body = (await request.json()) as QueryRequestBody;
  if (typeof body.text !== "string" || body.text.length === 0) {
    throw new SemfastError("request body must include a non-empty text field");
  }

  return {
    text: body.text,
    options: {
      topK: body.topK,
      alpha: body.alpha,
      mode: body.mode,
      filter: body.filter,
    },
  };
}
