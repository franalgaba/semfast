pub mod artifact;
pub mod benchmark;
pub mod bm25;
pub mod chunking;
pub mod embedding;
pub mod metadata;
pub mod query;
pub mod ranking;
pub mod vector;

pub use artifact::{ArtifactManifest, Index, IndexBuilder, IndexDocument};
pub use benchmark::{BenchReport, BenchmarkQuery, BenchmarkRunner};
pub use metadata::{Filter, Metadata, MetadataValue};
pub use query::{Query, QueryMode, SearchResult};

pub type Result<T> = std::result::Result<T, SemfastError>;

#[derive(Debug, thiserror::Error)]
pub enum SemfastError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("index artifact is missing {0}")]
    MissingArtifact(&'static str),
    #[error("turbovec error: {0}")]
    TurboVec(String),
    #[error("embedding model error: {0}")]
    EmbeddingModel(String),
}
