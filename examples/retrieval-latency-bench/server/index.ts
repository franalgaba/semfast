import { createExampleApp } from "./app.ts";

const indexPath = requiredEnv("SEMFAST_INDEX_PATH");
const app = await createExampleApp(indexPath);

Bun.serve({
  hostname: "127.0.0.1",
  port: Number.parseInt(process.env.PORT ?? "8787", 10),
  fetch: app.fetch,
});

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim().length === 0) {
    throw new Error(`${name} is required`);
  }

  return value;
}
