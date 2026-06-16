use crate::{Result, SemfastError};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(feature = "real-embeddings")]
use std::sync::Mutex;

pub const HASH_EMBEDDING_DIMENSIONS: usize = 384;
pub const HASH_EMBEDDING_MODEL: &str = "semfast-hash-embedding-v1";
pub const MINILM_EMBEDDING_DIMENSIONS: usize = 384;
pub const MINILM_EMBEDDING_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingBackendKind {
    #[default]
    Hash,
    FastEmbedMiniLm,
}

impl EmbeddingBackendKind {
    pub fn model_name(self) -> &'static str {
        match self {
            Self::Hash => HASH_EMBEDDING_MODEL,
            Self::FastEmbedMiniLm => MINILM_EMBEDDING_MODEL,
        }
    }

    pub fn dimensions(self) -> usize {
        match self {
            Self::Hash => HASH_EMBEDDING_DIMENSIONS,
            Self::FastEmbedMiniLm => MINILM_EMBEDDING_DIMENSIONS,
        }
    }
}

impl std::str::FromStr for EmbeddingBackendKind {
    type Err = SemfastError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "hash" | HASH_EMBEDDING_MODEL => Ok(Self::Hash),
            "minilm" | "fastembed-minilm" | MINILM_EMBEDDING_MODEL => Ok(Self::FastEmbedMiniLm),
            _ => Err(SemfastError::InvalidInput(format!(
                "unsupported embedding backend {value}"
            ))),
        }
    }
}

pub enum EmbeddingBackend {
    Hash(HashEmbedder),
    #[cfg(feature = "real-embeddings")]
    FastEmbedMiniLm(Box<FastEmbedMiniLmEmbedder>),
}

impl EmbeddingBackend {
    pub fn new(kind: EmbeddingBackendKind) -> Result<Self> {
        match kind {
            EmbeddingBackendKind::Hash => Ok(Self::Hash(HashEmbedder::default())),
            EmbeddingBackendKind::FastEmbedMiniLm => Self::new_fastembed_minilm(),
        }
    }

    pub fn from_model_name(model_name: &str) -> Result<Self> {
        Self::new(model_name.parse()?)
    }

    pub fn kind(&self) -> EmbeddingBackendKind {
        match self {
            Self::Hash(_) => EmbeddingBackendKind::Hash,
            #[cfg(feature = "real-embeddings")]
            Self::FastEmbedMiniLm(_) => EmbeddingBackendKind::FastEmbedMiniLm,
        }
    }

    pub fn dimensions(&self) -> usize {
        self.kind().dimensions()
    }

    pub fn model_name(&self) -> &'static str {
        self.kind().model_name()
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Hash(embedder) => Ok(embedder.embed(text)),
            #[cfg(feature = "real-embeddings")]
            Self::FastEmbedMiniLm(embedder) => embedder.embed(text),
        }
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Hash(embedder) => Ok(texts.iter().map(|text| embedder.embed(text)).collect()),
            #[cfg(feature = "real-embeddings")]
            Self::FastEmbedMiniLm(embedder) => embedder.embed_batch(texts),
        }
    }

    pub fn warm_up(&self) -> Result<()> {
        let _ = self.embed("semfast warmup query")?;
        Ok(())
    }

    #[cfg(feature = "real-embeddings")]
    fn new_fastembed_minilm() -> Result<Self> {
        Ok(Self::FastEmbedMiniLm(Box::new(
            FastEmbedMiniLmEmbedder::new()?,
        )))
    }

    #[cfg(not(feature = "real-embeddings"))]
    fn new_fastembed_minilm() -> Result<Self> {
        Err(SemfastError::InvalidInput(
            "real embedding models require building with --features real-embeddings".to_string(),
        ))
    }
}

impl Default for EmbeddingBackend {
    fn default() -> Self {
        Self::Hash(HashEmbedder::default())
    }
}

#[derive(Clone, Debug)]
pub struct HashEmbedder {
    dimensions: usize,
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(HASH_EMBEDDING_DIMENSIONS)
    }
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimensions];

        for token in text.split_whitespace().map(normalize_token) {
            if token.is_empty() {
                continue;
            }

            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            let hash = hasher.finish();
            let index = (hash as usize) % self.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }

        normalize(&mut vector);
        vector
    }
}

#[cfg(feature = "real-embeddings")]
pub struct FastEmbedMiniLmEmbedder {
    model: Mutex<fastembed::TextEmbedding>,
}

#[cfg(feature = "real-embeddings")]
impl FastEmbedMiniLmEmbedder {
    pub fn new() -> Result<Self> {
        let mut options = fastembed::TextInitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false);
        if let Some(intra_threads) = configured_intra_threads()? {
            options = options.with_intra_threads(intra_threads);
        }
        let model = fastembed::TextEmbedding::try_new(options)
            .map_err(|error| SemfastError::EmbeddingModel(error.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text.to_string()];
        self.embed_batch(&texts)?
            .into_iter()
            .next()
            .ok_or_else(|| SemfastError::EmbeddingModel("model returned no embedding".to_string()))
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.model.lock().map_err(|_| {
            SemfastError::EmbeddingModel("embedding model lock poisoned".to_string())
        })?;
        let embeddings = model
            .embed(texts, None)
            .map_err(|error| SemfastError::EmbeddingModel(error.to_string()))?;
        if embeddings.len() != texts.len() {
            return Err(SemfastError::EmbeddingModel(format!(
                "model returned {} embeddings for {} texts",
                embeddings.len(),
                texts.len()
            )));
        }
        Ok(embeddings)
    }
}

#[cfg(feature = "real-embeddings")]
fn configured_intra_threads() -> Result<Option<usize>> {
    let Some(value) = std::env::var_os("SEMFAST_ONNX_INTRA_THREADS") else {
        return Ok(None);
    };
    let value = value.to_string_lossy();
    if value.trim().is_empty() {
        return Ok(None);
    }

    let threads = value.parse::<usize>().map_err(|error| {
        SemfastError::InvalidInput(format!(
            "invalid SEMFAST_ONNX_INTRA_THREADS={value}: {error}"
        ))
    })?;
    if threads == 0 {
        return Err(SemfastError::InvalidInput(
            "SEMFAST_ONNX_INTRA_THREADS must be greater than 0".to_string(),
        ));
    }

    Ok(Some(threads))
}

fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();

    if norm == 0.0 {
        return;
    }

    for value in vector {
        *value /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embeddings_are_deterministic() {
        let embedder = HashEmbedder::new(16);

        assert_eq!(
            embedder.embed("Refund policy"),
            embedder.embed("Refund policy")
        );
    }

    #[test]
    fn model_names_round_trip_to_backend_kinds() {
        assert_eq!(
            MINILM_EMBEDDING_MODEL
                .parse::<EmbeddingBackendKind>()
                .unwrap(),
            EmbeddingBackendKind::FastEmbedMiniLm
        );
    }
}
