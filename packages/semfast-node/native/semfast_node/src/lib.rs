use napi::bindgen_prelude::*;
use napi_derive::napi;
use semfast_core::benchmark::{BenchmarkQuery, BenchmarkRunner};
use semfast_core::embedding::{EmbeddingBackend, EmbeddingBackendKind};
use semfast_core::metadata::{Filter, MetadataValue};
use semfast_core::{ArtifactManifest, Index, Query, QueryMode, SearchResult};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_TOP_K: u32 = 5;
const DEFAULT_ALPHA: f64 = 0.7;
const DEFAULT_MODE: &str = "hybrid";

#[napi]
pub fn native_version() -> String {
    VERSION.to_string()
}

#[napi]
pub fn inspect_artifact(path: String) -> Result<String> {
    let manifest = read_manifest(Path::new(&path))?;
    serialize_json(&manifest)
}

#[napi]
pub fn doctor_embedding(embedding_model: Option<String>) -> Result<String> {
    let embedding_kind = parse_embedding_kind(embedding_model.as_deref().unwrap_or("minilm"))?;
    preflight_embedding_runtime(embedding_kind)?;

    let embedder = EmbeddingBackend::new(embedding_kind).map_err(to_napi_error)?;
    let embedding = embedder
        .embed("semfast production embedding smoke test")
        .map_err(to_napi_error)?;

    serialize_json(&DoctorReport {
        embedding_model: embedder.model_name().to_string(),
        dimensions: embedding.len(),
        sample_norm: l2_norm(&embedding),
    })
}

#[napi]
pub struct NativeSemfastIndex {
    index: Option<Index>,
    artifact_path: PathBuf,
    load_time_ms: f64,
}

#[napi]
impl NativeSemfastIndex {
    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let started_at = Instant::now();
        let index = Index::load(&path).map_err(to_napi_error)?;
        Ok(Self {
            index: Some(index),
            artifact_path: PathBuf::from(path),
            load_time_ms: elapsed_ms(started_at),
        })
    }

    #[napi]
    pub fn query_json(
        &self,
        text: String,
        top_k: Option<u32>,
        alpha: Option<f64>,
        mode: Option<String>,
        filter_json: Option<String>,
    ) -> Result<String> {
        let measured = self.loaded_index()?.query_measured(Query {
            text,
            top_k: normalized_top_k(top_k),
            alpha: normalized_alpha(alpha),
            filter: parse_filter(filter_json)?,
            mode: parse_query_mode(mode.as_deref().unwrap_or(DEFAULT_MODE))?,
        });
        let measured = measured.map_err(to_napi_error)?;
        serialize_json(
            &measured
                .results
                .iter()
                .map(SerializableSearchResult::from)
                .collect::<Vec<_>>(),
        )
    }

    #[napi]
    pub fn query_measured_json(
        &self,
        text: String,
        top_k: Option<u32>,
        alpha: Option<f64>,
        mode: Option<String>,
        filter_json: Option<String>,
    ) -> Result<String> {
        let measured = self.loaded_index()?.query_measured(Query {
            text,
            top_k: normalized_top_k(top_k),
            alpha: normalized_alpha(alpha),
            filter: parse_filter(filter_json)?,
            mode: parse_query_mode(mode.as_deref().unwrap_or(DEFAULT_MODE))?,
        });
        let measured = measured.map_err(to_napi_error)?;
        serialize_json(&SerializableMeasuredQueryResult {
            results: measured
                .results
                .iter()
                .map(SerializableSearchResult::from)
                .collect(),
            timings: SerializableQueryTimings {
                embedding_ms: measured.timings.embedding_ms,
                vector_search_ms: measured.timings.vector_search_ms,
                bm25_ms: measured.timings.bm25_ms,
                filtering_ms: measured.timings.filtering_ms,
                fusion_ms: measured.timings.fusion_ms,
                hydration_ms: measured.timings.hydration_ms,
            },
        })
    }

    #[napi]
    pub fn embed_query_json(&self, text: String) -> Result<String> {
        let embedding = self
            .loaded_index()?
            .embed_query(&text)
            .map_err(to_napi_error)?;
        serialize_json(&embedding)
    }

    #[napi]
    pub fn search_vector_json(&self, vector: Vec<f64>, top_k: Option<u32>) -> Result<String> {
        let query_embedding = vector
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let hits = self
            .loaded_index()?
            .search_vector(&query_embedding, normalized_top_k(top_k))
            .map_err(to_napi_error)?;
        serialize_json(
            &hits
                .iter()
                .map(|hit| SerializableVectorHit {
                    id: hit.chunk_id.0.to_string(),
                    score: hit.score,
                })
                .collect::<Vec<_>>(),
        )
    }

    #[napi]
    pub fn benchmark_json(
        &self,
        queries_json: String,
        top_k: Option<u32>,
        alpha: Option<f64>,
        mode: Option<String>,
    ) -> Result<String> {
        let queries = serde_json::from_str::<Vec<BenchmarkQuery>>(&queries_json)
            .map_err(|error| Error::from_reason(error.to_string()))?;
        let report = BenchmarkRunner::with_artifact_stats(
            self.loaded_index()?,
            &self.artifact_path,
            self.load_time_ms,
        )
        .map_err(to_napi_error)?
        .with_top_k(normalized_top_k(top_k))
        .with_alpha(normalized_alpha(alpha))
        .with_mode(parse_query_mode(mode.as_deref().unwrap_or(DEFAULT_MODE))?)
        .run(&queries)
        .map_err(to_napi_error)?;

        serialize_json(&report)
    }

    #[napi]
    pub fn manifest_json(&self) -> Result<String> {
        serialize_json(self.loaded_index()?.manifest())
    }

    #[napi]
    pub fn close(&mut self) {
        self.index = None;
    }
}

impl NativeSemfastIndex {
    fn loaded_index(&self) -> Result<&Index> {
        self.index
            .as_ref()
            .ok_or_else(|| Error::from_reason("Semfast index is closed".to_string()))
    }
}

#[derive(Serialize)]
struct DoctorReport {
    embedding_model: String,
    dimensions: usize,
    sample_norm: f32,
}

#[derive(Serialize)]
struct SerializableMeasuredQueryResult {
    results: Vec<SerializableSearchResult>,
    timings: SerializableQueryTimings,
}

#[derive(Serialize)]
struct SerializableQueryTimings {
    embedding_ms: f64,
    vector_search_ms: f64,
    bm25_ms: f64,
    filtering_ms: f64,
    fusion_ms: f64,
    hydration_ms: f64,
}

#[derive(Serialize)]
struct SerializableSearchResult {
    id: String,
    text: String,
    score: f32,
    vector_score: Option<f32>,
    lexical_score: Option<f32>,
    metadata: BTreeMap<String, MetadataValue>,
}

impl From<&SearchResult> for SerializableSearchResult {
    fn from(result: &SearchResult) -> Self {
        Self {
            id: result.id.0.to_string(),
            text: result.text.clone(),
            score: result.score,
            vector_score: result.vector_score,
            lexical_score: result.lexical_score,
            metadata: result.metadata.values().clone(),
        }
    }
}

#[derive(Serialize)]
struct SerializableVectorHit {
    id: String,
    score: f32,
}

fn read_manifest(path: &Path) -> Result<ArtifactManifest> {
    let manifest_path = path.join("manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path).map_err(|error| {
        Error::from_reason(format!(
            "failed to read {}: {error}",
            manifest_path.display()
        ))
    })?;
    serde_json::from_str(&manifest).map_err(|error| Error::from_reason(error.to_string()))
}

fn preflight_embedding_runtime(embedding_kind: EmbeddingBackendKind) -> Result<()> {
    if embedding_kind != EmbeddingBackendKind::FastEmbedMiniLm {
        return Ok(());
    }

    preflight_onnx_runtime()?;
    preflight_minilm_cache()
}

fn preflight_onnx_runtime() -> Result<()> {
    let dylib_path = std::env::var("ORT_DYLIB_PATH")
        .map_err(|_| Error::from_reason("ORT_DYLIB_PATH is required for MiniLM".to_string()))?;
    if dylib_path.trim().is_empty() {
        return Err(Error::from_reason(
            "ORT_DYLIB_PATH is set but empty".to_string(),
        ));
    }

    let path = PathBuf::from(dylib_path);
    if !path.is_file() {
        return Err(Error::from_reason(format!(
            "ORT_DYLIB_PATH does not point to a file: {}",
            path.display()
        )));
    }

    Ok(())
}

fn preflight_minilm_cache() -> Result<()> {
    let cache_dir =
        std::env::var("FASTEMBED_CACHE_DIR").unwrap_or_else(|_| ".fastembed_cache".to_string());
    let repo_dir = PathBuf::from(cache_dir).join("models--Qdrant--all-MiniLM-L6-v2-onnx");
    if !repo_dir.exists() {
        return Ok(());
    }

    let lock_files = find_files_with_extension(&repo_dir, "lock")?;
    if !lock_files.is_empty() {
        let lock_list = lock_files
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(Error::from_reason(format!(
            "MiniLM cache contains lock files: {lock_list}"
        )));
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
        return Err(Error::from_reason(format!(
            "MiniLM cache is incomplete at {}; missing files: {}",
            snapshot.display(),
            missing_files.join(", ")
        )));
    }

    Ok(())
}

fn current_snapshot_dir(repo_dir: &Path) -> Result<PathBuf> {
    let ref_path = repo_dir.join("refs").join("main");
    if ref_path.is_file() {
        let commit = std::fs::read_to_string(&ref_path).map_err(|error| {
            Error::from_reason(format!("failed to read {}: {error}", ref_path.display()))
        })?;
        let snapshot = repo_dir.join("snapshots").join(commit.trim());
        if snapshot.is_dir() {
            return Ok(snapshot);
        }
    }

    let snapshots_dir = repo_dir.join("snapshots");
    let mut snapshots = std::fs::read_dir(&snapshots_dir)
        .map_err(|error| {
            Error::from_reason(format!(
                "failed to read {}: {error}",
                snapshots_dir.display()
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::from_reason(error.to_string()))?;
    snapshots.sort_by_key(|entry| entry.path());

    snapshots
        .into_iter()
        .rev()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .ok_or_else(|| {
            Error::from_reason(format!(
                "MiniLM cache has no snapshots at {}",
                repo_dir.display()
            ))
        })
}

fn find_files_with_extension(root: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut matches = Vec::new();
    for entry in WalkDir::new(root) {
        let entry = entry.map_err(|error| Error::from_reason(error.to_string()))?;
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

fn parse_filter(filter_json: Option<String>) -> Result<Option<Filter>> {
    let Some(filter_json) = filter_json else {
        return Ok(None);
    };
    if filter_json.trim().is_empty() {
        return Ok(None);
    }

    let values = serde_json::from_str::<BTreeMap<String, MetadataValue>>(&filter_json)
        .map_err(|error| Error::from_reason(error.to_string()))?;
    let mut filter = Filter::empty();
    for (key, value) in values {
        filter = filter.and_equals(key, value);
    }

    if filter.is_empty() {
        Ok(None)
    } else {
        Ok(Some(filter))
    }
}

fn parse_query_mode(mode: &str) -> Result<QueryMode> {
    match mode {
        "vector" => Ok(QueryMode::VectorOnly),
        "bm25" => Ok(QueryMode::Bm25Only),
        "hybrid" => Ok(QueryMode::Hybrid),
        _ => Err(Error::from_reason(format!(
            "unsupported query mode: {mode}"
        ))),
    }
}

fn parse_embedding_kind(value: &str) -> Result<EmbeddingBackendKind> {
    value
        .parse()
        .map_err(|error: semfast_core::SemfastError| Error::from_reason(error.to_string()))
}

fn normalized_top_k(top_k: Option<u32>) -> usize {
    top_k.unwrap_or(DEFAULT_TOP_K).max(1) as usize
}

fn normalized_alpha(alpha: Option<f64>) -> f32 {
    alpha.unwrap_or(DEFAULT_ALPHA).clamp(0.0, 1.0) as f32
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1_000.0
}

fn serialize_json(value: &impl Serialize) -> Result<String> {
    serde_json::to_string(value).map_err(|error| Error::from_reason(error.to_string()))
}

fn to_napi_error(error: semfast_core::SemfastError) -> Error {
    Error::from_reason(error.to_string())
}
