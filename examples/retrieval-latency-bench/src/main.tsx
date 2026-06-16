import { StrictMode, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const corpusDocumentCount = "5,183 docs";
const retrievalModeLabel = "hybrid top-5";
const defaultQueryPlaceholder = "Type a retrieval query against SciFact...";

interface BenchmarkQuery {
  readonly id: string;
  readonly text: string;
  readonly relevantDocumentIds: readonly string[];
}

interface RetrievedChunk {
  readonly rank: number;
  readonly id: string;
  readonly corpusId: string;
  readonly title: string;
  readonly text: string;
  readonly score: number;
  readonly vectorScore?: number | null;
  readonly lexicalScore?: number | null;
  readonly relevant: boolean;
}

interface RetrieveResponse {
  readonly dataset: string;
  readonly query: {
    readonly id: string | null;
    readonly text: string;
    readonly relevantDocumentIds: readonly string[];
  };
  readonly metrics: RetrievalMetrics;
  readonly quality: RetrievalQuality;
  readonly retrieved: readonly RetrievedChunk[];
}

interface RetrievalMetrics {
  readonly totalMs: number;
  readonly retrievalMs: number;
  readonly embeddingMs: number;
  readonly vectorSearchMs: number;
  readonly bm25Ms: number;
  readonly filteringMs: number;
  readonly fusionMs: number;
  readonly hydrationMs: number;
}

interface RetrievalQuality {
  readonly topK: number;
  readonly officialQrelsAvailable: boolean;
  readonly hitRelevant: boolean;
  readonly firstRelevantRank: number | null;
}

type RequestState = "idle" | "loading" | "error";

function App() {
  const [queries, setQueries] = useState<readonly BenchmarkQuery[]>([]);
  const [selectedQueryId, setSelectedQueryId] = useState<string | undefined>();
  const [freeText, setFreeText] = useState("");
  const [response, setResponse] = useState<RetrieveResponse | undefined>();
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [error, setError] = useState<string | undefined>();

  const selectedQuery = useMemo(
    () => queries.find((query) => query.id === selectedQueryId),
    [queries, selectedQueryId],
  );

  useEffect(() => {
    void initializeQueries();
  }, []);

  async function initializeQueries() {
    try {
      const benchmarkQueries = await fetchBenchmarkQueries();
      setQueries(benchmarkQueries);
      setSelectedQueryId(benchmarkQueries[0]?.id);
      setFreeText(benchmarkQueries[0]?.text ?? "");
    } catch (caughtError) {
      showError(caughtError);
    }
  }

  async function runPredefinedQuery(query: BenchmarkQuery) {
    setSelectedQueryId(query.id);
    setFreeText(query.text);
    await runRetrieval({ text: query.text, queryId: query.id });
  }

  async function runFreeTextQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runRetrieval({ text: freeText });
  }

  async function runRetrieval(request: RetrievalRequest) {
    const text = request.text.trim();
    if (text.length === 0) {
      return;
    }

    setRequestState("loading");
    setError(undefined);

    try {
      const retrievalResponse = await fetchRetrieval({ ...request, text });
      setResponse(retrievalResponse);
      setRequestState("idle");
    } catch (caughtError) {
      showError(caughtError);
    }
  }

  function updateFreeText(nextFreeText: string) {
    setFreeText(nextFreeText);
    setSelectedQueryId(undefined);
  }

  function showError(caughtError: unknown) {
    setRequestState("error");
    setError(readableError(caughtError));
  }

  return (
    <main className="app-shell">
      <Header requestState={requestState} />
      <section className="workspace">
        <QueryList
          disabled={requestState === "loading"}
          queries={queries}
          selectedQueryId={selectedQueryId}
          onSelectQuery={(query) => void runPredefinedQuery(query)}
        />

        <BenchmarkConsole
          error={error}
          freeText={freeText}
          requestState={requestState}
          response={response}
          selectedQuery={selectedQuery}
          onFreeTextChange={updateFreeText}
          onRunFreeText={runFreeTextQuery}
          onRunSelected={selectedQuery === undefined ? undefined : () => void runPredefinedQuery(selectedQuery)}
        />
      </section>
    </main>
  );
}

function Header({ requestState }: { readonly requestState: RequestState }) {
  return (
    <section className="toolbar">
      <div>
        <p className="eyebrow">BEIR / SciFact</p>
        <h1>Semfast Retrieval Latency Bench</h1>
      </div>
      <div className="status-strip" aria-live="polite">
        <Status label="Corpus" value={corpusDocumentCount} />
        <Status label="Mode" value={retrievalModeLabel} />
        <Status label="State" value={requestState === "loading" ? "running" : "ready"} />
      </div>
    </section>
  );
}

interface QueryListProps {
  readonly disabled: boolean;
  readonly queries: readonly BenchmarkQuery[];
  readonly selectedQueryId?: string;
  readonly onSelectQuery: (query: BenchmarkQuery) => void;
}

function QueryList({ disabled, queries, selectedQueryId, onSelectQuery }: QueryListProps) {
  return (
    <aside className="query-list" aria-label="Predefined benchmark questions">
      <div className="section-heading">
        <h2>Predefined Queries</h2>
        <span>{queries.length}</span>
      </div>
      <div className="query-buttons">
        {queries.map((query) => (
          <button
            className={query.id === selectedQueryId ? "query active" : "query"}
            key={query.id}
            type="button"
            onClick={() => onSelectQuery(query)}
            disabled={disabled}
          >
            <span>#{query.id}</span>
            {query.text}
          </button>
        ))}
      </div>
    </aside>
  );
}

interface BenchmarkConsoleProps {
  readonly error?: string;
  readonly freeText: string;
  readonly requestState: RequestState;
  readonly response?: RetrieveResponse;
  readonly selectedQuery?: BenchmarkQuery;
  readonly onFreeTextChange: (nextFreeText: string) => void;
  readonly onRunFreeText: (event: FormEvent<HTMLFormElement>) => void;
  readonly onRunSelected?: () => void;
}

function BenchmarkConsole({
  error,
  freeText,
  requestState,
  response,
  selectedQuery,
  onFreeTextChange,
  onRunFreeText,
  onRunSelected,
}: BenchmarkConsoleProps) {
  const isLoading = requestState === "loading";

  return (
    <section className="console">
      <SearchBox
        disabled={isLoading}
        freeText={freeText}
        placeholder={selectedQuery?.text ?? defaultQueryPlaceholder}
        onFreeTextChange={onFreeTextChange}
        onRunFreeText={onRunFreeText}
        onRunSelected={onRunSelected}
      />

      {error !== undefined ? <p className="error">{error}</p> : null}

      <MetricsPanel metrics={response?.metrics} />
      <QualityPanel response={response} />
      <ResultsPanel response={response} />
    </section>
  );
}

interface SearchBoxProps {
  readonly disabled: boolean;
  readonly freeText: string;
  readonly placeholder: string;
  readonly onFreeTextChange: (nextFreeText: string) => void;
  readonly onRunFreeText: (event: FormEvent<HTMLFormElement>) => void;
  readonly onRunSelected?: () => void;
}

function SearchBox({
  disabled,
  freeText,
  placeholder,
  onFreeTextChange,
  onRunFreeText,
  onRunSelected,
}: SearchBoxProps) {
  return (
    <form className="search-box" onSubmit={onRunFreeText}>
      <label htmlFor="free-text">Free Text Retrieval</label>
      <textarea
        id="free-text"
        value={freeText}
        onChange={(event) => onFreeTextChange(event.target.value)}
        placeholder={placeholder}
      />
      <div className="actions">
        <button type="submit" disabled={disabled || freeText.trim().length === 0}>
          {disabled ? "Retrieving..." : "Retrieve"}
        </button>
        {onRunSelected !== undefined ? (
          <button type="button" className="secondary" onClick={onRunSelected} disabled={disabled}>
            Run Selected
          </button>
        ) : null}
      </div>
    </form>
  );
}

function MetricsPanel({ metrics }: { readonly metrics?: RetrievalMetrics }) {
  return (
    <section className="metrics" aria-label="Latency metrics">
      <Metric label="Round Trip" value={metrics?.totalMs} />
      <Metric label="Engine Time" value={metrics?.retrievalMs} />
      <Metric label="Embedding" value={metrics?.embeddingMs} />
      <Metric label="Turbovec" value={metrics?.vectorSearchMs} />
      <Metric label="BM25" value={metrics?.bm25Ms} />
      <Metric label="Fusion" value={metrics?.fusionMs} />
    </section>
  );
}

function QualityPanel({ response }: { readonly response?: RetrieveResponse }) {
  return (
    <section className="quality">
      <QualityMetric
        label="Official qrels"
        value={response?.quality.officialQrelsAvailable ? "available" : "free text"}
      />
      <QualityMetric label="Relevant hit" value={formatRelevantHit(response)} />
      <QualityMetric label="Relevant IDs" value={formatRelevantDocumentIds(response)} />
    </section>
  );
}

function ResultsPanel({ response }: { readonly response?: RetrieveResponse }) {
  return (
    <section className="results" aria-label="Retrieved documents">
      <div className="section-heading">
        <h2>Retrieved Documents</h2>
        <span>{response?.retrieved.length ?? 0}</span>
      </div>
      {response === undefined ? (
        <p className="empty">Run a predefined SciFact query or submit free text.</p>
      ) : (
        <ol className="documents">
          {response.retrieved.map((chunk) => (
            <DocumentResult chunk={chunk} key={chunk.id} />
          ))}
        </ol>
      )}
    </section>
  );
}

function DocumentResult({ chunk }: { readonly chunk: RetrievedChunk }) {
  return (
    <li className={chunk.relevant ? "document relevant" : "document"}>
      <div className="document-head">
        <span>Rank {chunk.rank}</span>
        <strong>{chunk.score.toFixed(3)}</strong>
        {chunk.relevant ? <em>qrels hit</em> : null}
      </div>
      <h3>{chunk.title || `SciFact document ${chunk.corpusId}`}</h3>
      <p>{chunk.text}</p>
      <div className="document-meta">
        <span>Corpus ID {chunk.corpusId}</span>
        <span>Vector {formatScore(chunk.vectorScore)}</span>
        <span>BM25 {formatScore(chunk.lexicalScore)}</span>
      </div>
    </li>
  );
}

function Status({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <div className="status">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value }: { readonly label: string; readonly value?: number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{formatMilliseconds(value)}</strong>
    </div>
  );
}

function QualityMetric({ label, value }: { readonly label: string; readonly value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

interface RetrievalRequest {
  readonly text: string;
  readonly queryId?: string;
}

async function fetchBenchmarkQueries(): Promise<readonly BenchmarkQuery[]> {
  const result = await fetch("/api/queries");
  if (!result.ok) {
    throw new Error(await result.text());
  }

  const body = (await result.json()) as { readonly queries: readonly BenchmarkQuery[] };
  return body.queries;
}

async function fetchRetrieval(request: RetrievalRequest): Promise<RetrieveResponse> {
  const startedAt = performance.now();
  const result = await fetch("/api/retrieve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...request, topK: 5 }),
  });
  const browserRoundTripMs = performance.now() - startedAt;

  if (!result.ok) {
    throw new Error(await result.text());
  }

  const response = (await result.json()) as RetrieveResponse;
  return {
    ...response,
    metrics: {
      ...response.metrics,
      totalMs: browserRoundTripMs,
    },
  };
}

function formatMilliseconds(value: number | undefined): string {
  return value === undefined ? "--" : `${value.toFixed(2)} ms`;
}

function formatScore(score: number | null | undefined): string {
  return typeof score === "number" && Number.isFinite(score) ? score.toFixed(3) : "--";
}

function formatRelevantHit(response: RetrieveResponse | undefined): string {
  if (response === undefined) {
    return "--";
  }

  if (!response.quality.officialQrelsAvailable) {
    return "not scored";
  }

  return response.quality.firstRelevantRank === null
    ? "miss"
    : `rank ${response.quality.firstRelevantRank}`;
}

function formatRelevantDocumentIds(response: RetrieveResponse | undefined): string {
  return response?.query.relevantDocumentIds.slice(0, 3).join(", ") || "none";
}

function readableError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
