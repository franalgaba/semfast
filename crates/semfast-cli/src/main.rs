use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use semfast_core::benchmark::{BenchmarkQuery, BenchmarkRunner};
use semfast_core::embedding::EmbeddingBackendKind;
use semfast_core::metadata::{Metadata, MetadataValue};
use semfast_core::{Index, IndexBuilder, IndexDocument, Query, QueryMode};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(name = "semfast")]
#[command(about = "Local-first TurboVec-backed retrieval runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Embedding {
        #[command(subcommand)]
        command: EmbeddingCommand,
    },
    Index {
        #[command(subcommand)]
        command: IndexCommand,
    },
    Query {
        index_path: PathBuf,
        text: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = 0.7)]
        alpha: f32,
        #[arg(long, value_enum, default_value_t = QueryModeArg::Hybrid)]
        mode: QueryModeArg,
    },
    Inspect {
        index_path: PathBuf,
    },
    Bench {
        index_path: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = 0.7)]
        alpha: f32,
        #[arg(long, value_enum, default_value_t = QueryModeArg::Hybrid)]
        mode: QueryModeArg,
    },
    Fixture {
        #[command(subcommand)]
        command: FixtureCommand,
    },
}

#[derive(Debug, Subcommand)]
enum EmbeddingCommand {
    Doctor {
        #[arg(long, default_value = "minilm")]
        embedding_model: String,
        #[arg(long, default_value = "semfast production embedding smoke test")]
        text: String,
    },
}

#[derive(Debug, Subcommand)]
enum IndexCommand {
    Build {
        docs_path: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "hash")]
        embedding_model: String,
        #[arg(long, default_value_t = 256)]
        embedding_batch_size: usize,
    },
}

#[derive(Debug, Subcommand)]
enum FixtureCommand {
    Generate {
        #[arg(long)]
        docs_out: PathBuf,
        #[arg(long)]
        queries_out: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        documents: usize,
        #[arg(long, default_value_t = 1_000)]
        queries: usize,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QueryModeArg {
    Vector,
    Bm25,
    Hybrid,
}

impl From<QueryModeArg> for QueryMode {
    fn from(value: QueryModeArg) -> Self {
        match value {
            QueryModeArg::Vector => QueryMode::VectorOnly,
            QueryModeArg::Bm25 => QueryMode::Bm25Only,
            QueryModeArg::Hybrid => QueryMode::Hybrid,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Embedding { command } => match command {
            EmbeddingCommand::Doctor {
                embedding_model,
                text,
            } => doctor_embedding(&embedding_model, &text),
        },
        Command::Index { command } => match command {
            IndexCommand::Build {
                docs_path,
                out,
                embedding_model,
                embedding_batch_size,
            } => build_index(&docs_path, &out, &embedding_model, embedding_batch_size),
        },
        Command::Query {
            index_path,
            text,
            top_k,
            alpha,
            mode,
        } => query_index(&index_path, text, top_k, alpha, mode.into()),
        Command::Inspect { index_path } => inspect_index(&index_path),
        Command::Bench {
            index_path,
            queries,
            top_k,
            alpha,
            mode,
        } => bench_index(&index_path, &queries, top_k, alpha, mode.into()),
        Command::Fixture { command } => match command {
            FixtureCommand::Generate {
                docs_out,
                queries_out,
                documents,
                queries,
            } => generate_fixture(&docs_out, &queries_out, documents, queries),
        },
    }
}

fn doctor_embedding(embedding_model: &str, text: &str) -> Result<()> {
    let embedding_kind = embedding_model
        .parse::<EmbeddingBackendKind>()
        .map_err(anyhow::Error::from)?;
    let cache_dir = PathBuf::from(
        std::env::var("FASTEMBED_CACHE_DIR").unwrap_or_else(|_| ".fastembed_cache".to_string()),
    );

    eprintln!("embedding_model={}", embedding_kind.model_name());
    eprintln!("fastembed_cache_dir={}", cache_dir.display());
    eprintln!(
        "hf_home={}",
        std::env::var("HF_HOME").unwrap_or_else(|_| "<unset>".to_string())
    );
    eprintln!(
        "ort_lib_location={}",
        std::env::var("ORT_DYLIB_PATH")
            .or_else(|_| std::env::var("ORT_LIB_LOCATION"))
            .unwrap_or_else(|_| "<dynamic lookup>".to_string())
    );
    preflight_embedding_cache(embedding_kind, &cache_dir)?;
    preflight_onnx_runtime(embedding_kind)?;
    eprintln!("initializing embedding backend");
    let embedder = semfast_core::embedding::EmbeddingBackend::new(embedding_kind)
        .context("failed to initialize embedding backend")?;

    eprintln!("embedding sample text");
    let embedding = embedder
        .embed(text)
        .context("failed to embed sample text")?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "embedding_model": embedder.model_name(),
            "dimensions": embedding.len(),
            "sample_norm": embedding.iter().map(|value| value * value).sum::<f32>().sqrt(),
        }))?
    );
    Ok(())
}

fn preflight_embedding_cache(embedding_kind: EmbeddingBackendKind, cache_dir: &Path) -> Result<()> {
    if embedding_kind != EmbeddingBackendKind::FastEmbedMiniLm {
        return Ok(());
    }

    let repo_dir = cache_dir.join("models--Qdrant--all-MiniLM-L6-v2-onnx");
    if !repo_dir.exists() {
        eprintln!("cache_preflight=missing_repo_cache");
        return Ok(());
    }

    let stale_locks = find_files_with_extension(&repo_dir, "lock")?;
    if !stale_locks.is_empty() {
        anyhow::bail!(
            "MiniLM cache contains lock files; remove them after confirming no FastEmbed process is running: {}",
            stale_locks
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let snapshot = current_snapshot_dir(&repo_dir)?;
    let required_files = [
        "model.onnx",
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ];
    let missing_files = required_files
        .iter()
        .filter(|file_name| !snapshot.join(file_name).exists())
        .copied()
        .collect::<Vec<_>>();

    if !missing_files.is_empty() {
        anyhow::bail!(
            "MiniLM cache is incomplete at {}; missing files: {}",
            snapshot.display(),
            missing_files.join(", ")
        );
    }

    eprintln!("cache_preflight=ready snapshot={}", snapshot.display());
    Ok(())
}

fn preflight_onnx_runtime(embedding_kind: EmbeddingBackendKind) -> Result<()> {
    if embedding_kind != EmbeddingBackendKind::FastEmbedMiniLm {
        return Ok(());
    }

    let Ok(dylib_path) = std::env::var("ORT_DYLIB_PATH") else {
        anyhow::bail!(
            "ORT_DYLIB_PATH is required for MiniLM in the current dynamic ONNX Runtime build"
        );
    };
    if dylib_path.trim().is_empty() {
        anyhow::bail!("ORT_DYLIB_PATH is set but empty");
    }

    let path = PathBuf::from(&dylib_path);
    if !path.is_file() {
        anyhow::bail!(
            "ORT_DYLIB_PATH does not point to a file: {}",
            path.display()
        );
    }

    eprintln!("onnxruntime_preflight=ready dylib={}", path.display());
    Ok(())
}

fn current_snapshot_dir(repo_dir: &Path) -> Result<PathBuf> {
    let ref_path = repo_dir.join("refs").join("main");
    if ref_path.is_file() {
        let commit = std::fs::read_to_string(&ref_path)
            .with_context(|| format!("failed to read {}", ref_path.display()))?;
        let snapshot = repo_dir.join("snapshots").join(commit.trim());
        if snapshot.is_dir() {
            return Ok(snapshot);
        }
    }

    let mut snapshots = std::fs::read_dir(repo_dir.join("snapshots"))
        .with_context(|| format!("failed to read snapshots for {}", repo_dir.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    snapshots.sort_by_key(|entry| entry.path());

    snapshots
        .into_iter()
        .rev()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .ok_or_else(|| anyhow::anyhow!("MiniLM cache has no snapshots at {}", repo_dir.display()))
}

fn find_files_with_extension(root: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut matches = Vec::new();
    for entry in WalkDir::new(root) {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|value| value == extension)
        {
            matches.push(entry.path().to_path_buf());
        }
    }
    matches.sort();
    Ok(matches)
}

fn build_index(
    docs_path: &Path,
    out: &Path,
    embedding_model: &str,
    embedding_batch_size: usize,
) -> Result<()> {
    let documents = read_documents(docs_path)?;
    let embedding_kind = embedding_model
        .parse::<EmbeddingBackendKind>()
        .map_err(anyhow::Error::from)?;
    IndexBuilder::new()
        .with_embedding_backend(embedding_kind)
        .with_embedding_batch_size(embedding_batch_size)
        .build_to_path(&documents, out)
        .with_context(|| format!("failed to build index at {}", out.display()))?;

    println!(
        "built index at {} from {} documents using {} embeddings with batch size {}",
        out.display(),
        documents.len(),
        embedding_kind.model_name(),
        embedding_batch_size.max(1)
    );
    Ok(())
}

fn query_index(
    index_path: &Path,
    text: String,
    top_k: usize,
    alpha: f32,
    mode: QueryMode,
) -> Result<()> {
    let index = Index::load(index_path)
        .with_context(|| format!("failed to load index {}", index_path.display()))?;
    let results = index.query(Query {
        text,
        top_k,
        alpha,
        filter: None,
        mode,
    })?;

    for result in results {
        println!(
            "{}\t{:.4}\tvector={:?}\tbm25={:?}\t{}",
            result.id.0,
            result.score,
            result.vector_score,
            result.lexical_score,
            result.text.replace('\n', " ")
        );
    }

    Ok(())
}

fn inspect_index(index_path: &Path) -> Result<()> {
    let index = Index::load(index_path)
        .with_context(|| format!("failed to load index {}", index_path.display()))?;
    println!("{}", serde_json::to_string_pretty(index.manifest())?);
    Ok(())
}

fn bench_index(
    index_path: &Path,
    queries_path: &Path,
    top_k: usize,
    alpha: f32,
    mode: QueryMode,
) -> Result<()> {
    let load_started_at = Instant::now();
    let index = Index::load(index_path)
        .with_context(|| format!("failed to load index {}", index_path.display()))?;
    let load_time_ms = load_started_at.elapsed().as_secs_f64() * 1_000.0;
    let queries = read_benchmark_queries(queries_path)?;
    let report = BenchmarkRunner::with_artifact_stats(&index, index_path, load_time_ms)?
        .with_top_k(top_k)
        .with_alpha(alpha)
        .with_mode(mode)
        .run(&queries)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn read_documents(path: &Path) -> Result<Vec<IndexDocument>> {
    if path.is_file() {
        if path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            return read_jsonl_documents(path);
        }
        return Ok(vec![read_document(path)?]);
    }

    let mut documents = Vec::new();
    let mut entries = WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| !is_hidden(entry.path()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by(|left, right| left.path().cmp(right.path()));

    for entry in entries {
        if !entry.file_type().is_file() {
            continue;
        }
        documents.push(read_document(entry.path())?);
    }

    Ok(documents)
}

#[derive(Debug, Deserialize, Serialize)]
struct JsonlDocument {
    id: String,
    text: String,
    #[serde(default)]
    metadata: Metadata,
}

fn read_jsonl_documents(path: &Path) -> Result<Vec<IndexDocument>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open JSONL documents {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let document: JsonlDocument = serde_json::from_str(&line)?;
        documents.push(IndexDocument::new(
            document.id,
            document.text,
            document.metadata,
        ));
    }

    Ok(documents)
}

fn read_document(path: &Path) -> Result<IndexDocument> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read document {}", path.display()))?;
    let mut values = BTreeMap::new();
    values.insert(
        "source_path".to_string(),
        MetadataValue::String(path.display().to_string()),
    );

    Ok(IndexDocument::new(
        path.display().to_string(),
        text,
        Metadata::new(values),
    ))
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

fn read_benchmark_queries(path: &Path) -> Result<Vec<BenchmarkQuery>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open benchmark queries {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut queries = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        queries.push(serde_json::from_str(&line)?);
    }

    Ok(queries)
}

fn generate_fixture(
    docs_out: &Path,
    queries_out: &Path,
    document_count: usize,
    query_count: usize,
) -> Result<()> {
    if let Some(parent) = docs_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = queries_out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut docs_writer = std::io::BufWriter::new(File::create(docs_out)?);
    for index in 0..document_count {
        let mut metadata_values = BTreeMap::new();
        metadata_values.insert(
            "tenant".to_string(),
            MetadataValue::String(format!("tenant-{}", index % 16)),
        );
        metadata_values.insert(
            "category".to_string(),
            MetadataValue::String(format!("category-{}", index % 128)),
        );

        let document = JsonlDocument {
            id: format!("doc-{index:06}"),
            text: synthetic_document_text(index),
            metadata: Metadata::new(metadata_values),
        };
        serde_json::to_writer(&mut docs_writer, &document)?;
        use std::io::Write;
        docs_writer.write_all(b"\n")?;
    }

    let mut query_writer = std::io::BufWriter::new(File::create(queries_out)?);
    for index in 0..query_count {
        let document_index = index % document_count;
        let query = BenchmarkQuery {
            text: synthetic_query_text(document_index),
            expected_chunk_id: Some(document_index as u64 + 1),
        };
        serde_json::to_writer(&mut query_writer, &query)?;
        use std::io::Write;
        query_writer.write_all(b"\n")?;
    }

    println!(
        "generated {} documents at {} and {} queries at {}",
        document_count,
        docs_out.display(),
        query_count,
        queries_out.display()
    );
    Ok(())
}

fn synthetic_document_text(index: usize) -> String {
    format!(
        "semfast unique topic {index:06} handles account workflow {index:06}. \
         The answer token sfdoc{index:06} belongs to generated document {index:06}. \
         Refund policy shipping billing support latency retrieval voice agent benchmark."
    )
}

fn synthetic_query_text(index: usize) -> String {
    format!("sfdoc{index:06} account workflow {index:06}")
}
