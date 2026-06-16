use crate::bm25::Bm25Index;
use crate::chunking::{Chunk, ChunkId, ChunkingConfig, chunk_text};
use crate::embedding::{EmbeddingBackend, EmbeddingBackendKind};
use crate::metadata::Metadata;
use crate::query::{MeasuredQueryResult, Query, QueryMode, QueryTimings, SearchResult};
use crate::ranking::fuse_hits;
use crate::vector::VectorIndex;
use crate::vector::turbovec::TurboVecIndex;
use crate::{Result, SemfastError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MANIFEST_FILE: &str = "manifest.json";
const CHUNKS_FILE: &str = "chunks.jsonl";
const METADATA_FILE: &str = "metadata.jsonl";
const VECTORS_FILE: &str = "vectors.turbovec";
const BM25_FILE: &str = "bm25.index";
const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 256;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArtifactManifest {
    pub version: u32,
    pub vector_backend: String,
    pub embedding_model: String,
    pub dimensions: usize,
    pub chunk_count: usize,
    pub created_at: String,
    pub created_at_unix_seconds: u64,
}

#[derive(Clone, Debug)]
pub struct IndexDocument {
    pub id: String,
    pub text: String,
    pub metadata: Metadata,
}

impl IndexDocument {
    pub fn new(id: impl Into<String>, text: impl Into<String>, metadata: Metadata) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata,
        }
    }
}

pub struct IndexBuilder {
    chunking: ChunkingConfig,
    embedding_kind: EmbeddingBackendKind,
    embedding_batch_size: usize,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chunking(mut self, chunking: ChunkingConfig) -> Self {
        self.chunking = chunking;
        self
    }

    pub fn with_embedding_backend(mut self, embedding_kind: EmbeddingBackendKind) -> Self {
        self.embedding_kind = embedding_kind;
        self
    }

    pub fn with_embedding_batch_size(mut self, embedding_batch_size: usize) -> Self {
        self.embedding_batch_size = embedding_batch_size.max(1);
        self
    }

    pub fn build_to_path(&self, documents: &[IndexDocument], output_path: &Path) -> Result<()> {
        std::fs::create_dir_all(output_path)?;

        let embedder = EmbeddingBackend::new(self.embedding_kind)?;
        let chunks = self.chunk_documents(documents);
        if chunks.is_empty() {
            return Err(SemfastError::InvalidInput(
                "cannot build an index with no chunks".to_string(),
            ));
        }

        let created_at_unix_seconds = unix_timestamp();
        let manifest = ArtifactManifest {
            version: 1,
            vector_backend: "turbovec".to_string(),
            embedding_model: embedder.model_name().to_string(),
            dimensions: embedder.dimensions(),
            chunk_count: chunks.len(),
            created_at: format!("unix:{created_at_unix_seconds}"),
            created_at_unix_seconds,
        };

        let chunk_ids = chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>();
        let mut vectors = Vec::with_capacity(chunks.len() * embedder.dimensions());
        self.embed_chunks(&embedder, &chunks, &mut vectors)?;

        let vector_index = TurboVecIndex::build(&vectors, &chunk_ids, embedder.dimensions())?;
        let bm25_index =
            Bm25Index::build(chunks.iter().map(|chunk| (chunk.id, chunk.text.clone())));

        write_json(output_path.join(MANIFEST_FILE), &manifest)?;
        write_jsonl(output_path.join(CHUNKS_FILE), &chunks)?;
        write_jsonl(
            output_path.join(METADATA_FILE),
            &chunks
                .iter()
                .map(|chunk| ChunkMetadataRecord {
                    id: chunk.id,
                    metadata: chunk.metadata.clone(),
                })
                .collect::<Vec<_>>(),
        )?;
        vector_index.save(&output_path.join(VECTORS_FILE))?;
        write_json(output_path.join(BM25_FILE), &bm25_index)?;

        Ok(())
    }

    fn chunk_documents(&self, documents: &[IndexDocument]) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut next_base_id = 1;

        for document in documents {
            let document_chunks = chunk_text(
                next_base_id,
                &document.id,
                &document.text,
                &document.metadata,
                self.chunking,
            );
            next_base_id += document_chunks.len() as u64;
            chunks.extend(document_chunks);
        }

        chunks
    }

    fn embed_chunks(
        &self,
        embedder: &EmbeddingBackend,
        chunks: &[Chunk],
        vectors: &mut Vec<f32>,
    ) -> Result<()> {
        for chunk_batch in chunks.chunks(self.embedding_batch_size) {
            let chunk_texts = chunk_batch
                .iter()
                .map(|chunk| chunk.text.clone())
                .collect::<Vec<_>>();
            let embeddings = embedder.embed_batch(&chunk_texts)?;
            if embeddings.len() != chunk_batch.len() {
                return Err(SemfastError::EmbeddingModel(format!(
                    "embedding backend returned {} vectors for {} chunks",
                    embeddings.len(),
                    chunk_batch.len()
                )));
            }

            for embedding in embeddings {
                if embedding.len() != embedder.dimensions() {
                    return Err(SemfastError::EmbeddingModel(format!(
                        "embedding backend returned {} dimensions but {} were expected",
                        embedding.len(),
                        embedder.dimensions()
                    )));
                }
                vectors.extend(embedding);
            }
        }

        Ok(())
    }
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self {
            chunking: ChunkingConfig::default(),
            embedding_kind: EmbeddingBackendKind::default(),
            embedding_batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
        }
    }
}

pub struct Index {
    manifest: ArtifactManifest,
    chunks: BTreeMap<ChunkId, Chunk>,
    vector_index: TurboVecIndex,
    bm25_index: Bm25Index,
    embedder: EmbeddingBackend,
}

impl Index {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure_file(path, MANIFEST_FILE)?;
        ensure_file(path, CHUNKS_FILE)?;
        ensure_file(path, METADATA_FILE)?;
        ensure_file(path, VECTORS_FILE)?;
        ensure_file(path, BM25_FILE)?;

        let manifest: ArtifactManifest = read_json(path.join(MANIFEST_FILE))?;
        if manifest.vector_backend != "turbovec" {
            return Err(SemfastError::InvalidInput(format!(
                "unsupported vector backend {}",
                manifest.vector_backend
            )));
        }

        let chunks_vec: Vec<Chunk> = read_jsonl(path.join(CHUNKS_FILE))?;
        let chunks = chunks_vec
            .into_iter()
            .map(|chunk| (chunk.id, chunk))
            .collect::<BTreeMap<_, _>>();

        let embedder = EmbeddingBackend::from_model_name(&manifest.embedding_model)?;
        if embedder.dimensions() != manifest.dimensions {
            return Err(SemfastError::InvalidInput(format!(
                "embedding model {} has {} dimensions but artifact expects {}",
                manifest.embedding_model,
                embedder.dimensions(),
                manifest.dimensions
            )));
        }

        let vector_index = TurboVecIndex::load(&path.join(VECTORS_FILE))?;
        let bm25_index = read_json(path.join(BM25_FILE))?;

        Ok(Self {
            embedder,
            manifest,
            chunks,
            vector_index,
            bm25_index,
        })
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn query(&self, query: Query) -> Result<Vec<SearchResult>> {
        Ok(self.query_measured(query)?.results)
    }

    pub fn embed_query(&self, query_text: &str) -> Result<Vec<f32>> {
        self.embedder.embed(query_text)
    }

    pub fn search_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<crate::vector::VectorHit>> {
        self.vector_index.search(query_embedding, top_k)
    }

    pub fn query_measured(&self, query: Query) -> Result<MeasuredQueryResult> {
        if query.top_k == 0 {
            return Ok(MeasuredQueryResult {
                results: Vec::new(),
                timings: QueryTimings::default(),
            });
        }

        let candidate_count = (query.top_k * 4).max(32);
        let embedding_started_at = Instant::now();
        let query_embedding = self.embedder.embed(&query.text)?;
        let embedding_ms = elapsed_ms(embedding_started_at);

        let vector_started_at = Instant::now();
        let vector_hits = match query.mode {
            QueryMode::Bm25Only => Vec::new(),
            QueryMode::VectorOnly | QueryMode::Hybrid => self
                .vector_index
                .search(&query_embedding, candidate_count)?,
        };
        let vector_search_ms = elapsed_ms(vector_started_at);

        let bm25_started_at = Instant::now();
        let lexical_hits = match query.mode {
            QueryMode::VectorOnly => Vec::new(),
            QueryMode::Bm25Only | QueryMode::Hybrid => {
                self.bm25_index.search(&query.text, candidate_count)
            }
        };
        let bm25_ms = elapsed_ms(bm25_started_at);

        let fusion_started_at = Instant::now();
        let ranked_hits = fuse_hits(
            &vector_hits,
            &lexical_hits,
            candidate_count,
            query.alpha,
            query.mode,
        );
        let fusion_ms = elapsed_ms(fusion_started_at);

        let mut results = Vec::with_capacity(query.top_k);
        let mut filtering_ms = 0.0;
        let mut hydration_ms = 0.0;

        for hit in ranked_hits {
            let Some(chunk) = self.chunks.get(&hit.chunk_id) else {
                continue;
            };

            let filtering_started_at = Instant::now();
            if let Some(filter) = &query.filter
                && !chunk.metadata.matches_filter(filter)
            {
                filtering_ms += elapsed_ms(filtering_started_at);
                continue;
            }
            filtering_ms += elapsed_ms(filtering_started_at);

            let hydration_started_at = Instant::now();
            results.push(SearchResult {
                id: chunk.id,
                text: chunk.text.clone(),
                score: hit.score,
                vector_score: hit.vector_score,
                lexical_score: hit.lexical_score,
                metadata: chunk.metadata.clone(),
            });
            hydration_ms += elapsed_ms(hydration_started_at);

            if results.len() == query.top_k {
                break;
            }
        }

        Ok(MeasuredQueryResult {
            results,
            timings: QueryTimings {
                embedding_ms,
                vector_search_ms,
                bm25_ms,
                filtering_ms,
                fusion_ms,
                hydration_ms,
            },
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChunkMetadataRecord {
    id: ChunkId,
    metadata: Metadata,
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1_000.0
}

fn ensure_file(path: &Path, file_name: &'static str) -> Result<()> {
    let file_path = path.join(file_name);
    if file_path.is_file() {
        Ok(())
    } else {
        Err(SemfastError::MissingArtifact(file_name))
    }
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), value)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(BufReader::new(file))?)
}

fn write_jsonl<T: Serialize>(path: PathBuf, values: &[T]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
    }

    Ok(())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<Vec<T>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(serde_json::from_str(&line)?);
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Filter, MetadataValue};

    #[test]
    fn builds_loads_and_queries_artifact() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut metadata = Metadata::empty();
        metadata.insert("source", MetadataValue::String("faq".to_string()));
        let documents = vec![IndexDocument::new(
            "doc-1",
            "Refunds are available for damaged shipments. Shipping labels are emailed.",
            metadata,
        )];

        IndexBuilder::new()
            .with_chunking(ChunkingConfig {
                max_words: 16,
                overlap_words: 0,
            })
            .build_to_path(&documents, temp_dir.path())
            .unwrap();

        let index = Index::load(temp_dir.path()).unwrap();
        let mut query = Query::hybrid("damaged shipment refund", 3);
        query.filter = Some(Filter::equals(
            "source",
            MetadataValue::String("faq".to_string()),
        ));

        let results = index.query(query).unwrap();

        assert!(!results.is_empty());
        assert!(results[0].text.contains("Refunds"));
    }
}
