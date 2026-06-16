const datasetUrl =
  "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip";
const zipPath = "data/beir/scifact.zip";
const extractedRoot = "data/beir/scifact";
const outputRoot = "data/scifact";
const documentsPath = `${outputRoot}/documents.jsonl`;
const queriesPath = `${outputRoot}/queries.json`;

interface BeirCorpusRecord {
  readonly _id: string;
  readonly title?: string;
  readonly text: string;
  readonly metadata?: Record<string, unknown>;
}

interface BeirQueryRecord {
  readonly _id: string;
  readonly text: string;
}

interface UiQuery {
  readonly id: string;
  readonly text: string;
  readonly relevantDocumentIds: readonly string[];
}

await ensureDir("data/beir");
await ensureDir(outputRoot);
await ensureDataset();

const corpus = await readJsonl<BeirCorpusRecord>(`${extractedRoot}/corpus.jsonl`);
const queries = await readJsonl<BeirQueryRecord>(`${extractedRoot}/queries.jsonl`);
const qrels = await readQrels(`${extractedRoot}/qrels/test.tsv`);

const corpusById = new Map(corpus.map((record) => [record._id, record]));
const queryById = new Map(queries.map((record) => [record._id, record]));
const testQueries = [...qrels.entries()]
  .map(([queryId, relevantDocumentIds]) => {
    const query = queryById.get(queryId);
    if (query === undefined) {
      return undefined;
    }

    return {
      id: queryId,
      text: query.text,
      relevantDocumentIds,
    };
  })
  .filter((query): query is UiQuery => query !== undefined)
  .slice(0, 24);

await writeDocuments(corpus, corpusById);
await Bun.write(queriesPath, `${JSON.stringify(testQueries, null, 2)}\n`);

console.log(
  JSON.stringify({
    dataset: "BEIR/SciFact",
    documents: corpus.length,
    testQueries: testQueries.length,
    documentsPath,
    queriesPath,
  }),
);

async function ensureDataset(): Promise<void> {
  if (!(await exists(zipPath))) {
    const response = await fetch(datasetUrl);
    if (!response.ok) {
      throw new Error(`failed to download SciFact: ${response.status} ${response.statusText}`);
    }

    await Bun.write(zipPath, response);
  }

  if (await exists(`${extractedRoot}/corpus.jsonl`)) {
    return;
  }

  const result = Bun.spawnSync({
    cmd: ["unzip", "-o", zipPath, "-d", "data/beir"],
    stdout: "inherit",
    stderr: "inherit",
  });

  if (!result.success) {
    throw new Error(`failed to extract SciFact zip with exit code ${result.exitCode}`);
  }
}

async function readJsonl<T>(path: string): Promise<readonly T[]> {
  const text = await Bun.file(path).text();
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as T);
}

async function readQrels(path: string): Promise<Map<string, string[]>> {
  const qrels = new Map<string, string[]>();
  const lines = (await Bun.file(path).text()).split("\n").slice(1);

  for (const line of lines) {
    const [queryId, corpusId, score] = line.trim().split(/\s+/);
    if (queryId === undefined || corpusId === undefined || Number(score) <= 0) {
      continue;
    }

    const relevant = qrels.get(queryId) ?? [];
    relevant.push(corpusId);
    qrels.set(queryId, relevant);
  }

  return qrels;
}

async function writeDocuments(
  corpus: readonly BeirCorpusRecord[],
  corpusById: ReadonlyMap<string, BeirCorpusRecord>,
): Promise<void> {
  const lines = corpus.map((record) => {
    const title = record.title?.trim() ?? "";
    const text = title.length > 0 ? `${title}\n\n${record.text}` : record.text;
    const metadataValues = {
      dataset: "BEIR/SciFact",
      corpus_id: record._id,
      title,
      source: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    };

    return JSON.stringify({
      id: record._id,
      text,
      metadata: {
        values: metadataValues,
      },
    });
  });

  if (corpus.some((record) => corpusById.get(record._id) !== record)) {
    throw new Error("duplicate SciFact corpus id detected");
  }

  await Bun.write(documentsPath, `${lines.join("\n")}\n`);
}

async function ensureDir(path: string): Promise<void> {
  await Bun.$`mkdir -p ${path}`.quiet();
}

async function exists(path: string): Promise<boolean> {
  return await Bun.file(path).exists();
}
