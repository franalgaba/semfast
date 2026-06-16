pub mod exact;
pub mod turbovec;

use crate::Result;
use crate::chunking::ChunkId;

#[derive(Clone, Copy, Debug)]
pub struct VectorHit {
    pub chunk_id: ChunkId,
    pub score: f32,
}

pub trait VectorIndex {
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>>;
    fn save(&self, path: &std::path::Path) -> Result<()>;
}
