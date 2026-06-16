use crate::chunking::ChunkId;
use crate::metadata::{Filter, Metadata};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryMode {
    VectorOnly,
    Bm25Only,
    Hybrid,
}

#[derive(Clone, Debug)]
pub struct Query {
    pub text: String,
    pub top_k: usize,
    pub alpha: f32,
    pub filter: Option<Filter>,
    pub mode: QueryMode,
}

impl Query {
    pub fn hybrid(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            alpha: 0.7,
            filter: None,
            mode: QueryMode::Hybrid,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: ChunkId,
    pub text: String,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub lexical_score: Option<f32>,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Default)]
pub struct QueryTimings {
    pub embedding_ms: f64,
    pub vector_search_ms: f64,
    pub bm25_ms: f64,
    pub filtering_ms: f64,
    pub fusion_ms: f64,
    pub hydration_ms: f64,
}

#[derive(Clone, Debug)]
pub struct MeasuredQueryResult {
    pub results: Vec<SearchResult>,
    pub timings: QueryTimings,
}
